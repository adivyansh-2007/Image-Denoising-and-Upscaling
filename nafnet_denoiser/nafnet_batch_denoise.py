import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class SimplifiedChannelAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(c, c, 1)

    def forward(self, x):
        return x * self.conv(self.pool(x))

class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(1, c, eps=1e-6)
        self.conv1 = nn.Conv2d(c, c * 2, 1)
        self.dwconv = nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2)
        self.sg = SimpleGate()
        self.sca = SimplifiedChannelAttention(c)
        self.conv2 = nn.Conv2d(c, c, 1)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        identity = x
        out = self.norm(x)
        out = self.conv1(out)
        out = self.dwconv(out)
        out = self.sg(out)
        out = self.sca(out)
        out = self.conv2(out)
        return identity + out * self.beta

class HeavyNAFNetDenoiser(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super().__init__()

        enc_blks = [2, 2, 4, 8]
        mid_blks = 12
        dec_blks = [2, 2, 2, 2]

        self.intro = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.enc1 = nn.Sequential(*[NAFBlock(base_channels) for _ in range(enc_blks[0])])
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 2, stride=2)

        self.enc2 = nn.Sequential(*[NAFBlock(base_channels * 2) for _ in range(enc_blks[1])])
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 2, stride=2)

        self.enc3 = nn.Sequential(*[NAFBlock(base_channels * 4) for _ in range(enc_blks[2])])
        self.down3 = nn.Conv2d(base_channels * 4, base_channels * 8, 2, stride=2)

        self.enc4 = nn.Sequential(*[NAFBlock(base_channels * 8) for _ in range(enc_blks[3])])
        self.down4 = nn.Conv2d(base_channels * 8, base_channels * 16, 2, stride=2)

        self.bottleneck = nn.Sequential(*[NAFBlock(base_channels * 16) for _ in range(mid_blks)])

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(base_channels * 16, base_channels * 8, 1))
        self.dec4 = nn.Sequential(
            nn.Conv2d(base_channels * 16, base_channels * 8, 1),
            *[NAFBlock(base_channels * 8) for _ in range(dec_blks[0])]
        )

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(base_channels * 8, base_channels * 4, 1))
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 4, 1),
            *[NAFBlock(base_channels * 4) for _ in range(dec_blks[1])]
        )

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(base_channels * 4, base_channels * 2, 1))
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 1),
            *[NAFBlock(base_channels * 2) for _ in range(dec_blks[2])]
        )

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(base_channels * 2, base_channels, 1))
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 1),
            *[NAFBlock(base_channels) for _ in range(dec_blks[3])]
        )

        self.outro = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, x):
        out = self.intro(x)

        e1 = self.enc1(out)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))

        b = self.bottleneck(self.down4(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.clamp(x + self.outro(d1), 0.0, 1.0)


CONFIG = {
    "model_path": "checkpoints/nafnet_denoiser.pth",

    "input_dir": "data/noisy/",

    "output_dir": "data/denoised/",

    "batch_size": 32,
    "num_workers": 8
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

class SimpleInferenceDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
        self.folder = folder

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.folder, self.files[idx])
        img = np.load(path).astype(np.float32)

        return torch.from_numpy(img).unsqueeze(0), self.files[idx]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running batch inference on: {device}")

    model = HeavyNAFNetDenoiser(in_channels=1, out_channels=1, base_channels=64).to(device)
    state_dict = torch.load(CONFIG["model_path"], map_location=device)

    if "module.intro.weight" in state_dict:
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    model.eval()

    dataset = SimpleInferenceDataset(CONFIG["input_dir"])
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], num_workers=CONFIG["num_workers"], shuffle=False)

    print(f"Processing {len(dataset)} images into {CONFIG['output_dir']}...")

    with torch.no_grad():
        for batch_imgs, filenames in tqdm(loader):
            batch_imgs = batch_imgs.to(device)

            outputs = model(batch_imgs)

            outputs_np = outputs.cpu().numpy()

            for i in range(len(filenames)):
                save_path = os.path.join(CONFIG["output_dir"], filenames[i])
                np.save(save_path, outputs_np[i].squeeze())

    print(f"\nSUCCESS: Stage 1 NAFNet outputs generated at {CONFIG['output_dir']}")

if __name__ == '__main__':
    main()
