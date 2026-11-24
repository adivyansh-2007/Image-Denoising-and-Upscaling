import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

CONFIG = {
    "noisy_dir": "data/noisy/",
    "clean_dir": "data/clean_hr/",
    "save_dir": "./checkpoints_denoiser_sobel/",
    "batch_size": 32,
    "epochs": 100,
    "lr": 1e-4,
    "num_workers": 8,
    "val_patch": "data/noisy/000001.npy"
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)

class DenoiseDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir):
        self.noisy_files = sorted([os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir) if f.endswith('.npy')])
        self.clean_files = sorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith('.npy')])

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_patch = np.load(self.noisy_files[idx]).astype(np.float32)
        clean_patch = np.load(self.clean_files[idx]).astype(np.float32)

        if np.random.random() > 0.5:
            noisy_patch, clean_patch = np.flip(noisy_patch, axis=1).copy(), np.flip(clean_patch, axis=1).copy()
        if np.random.random() > 0.5:
            noisy_patch, clean_patch = np.flip(noisy_patch, axis=0).copy(), np.flip(clean_patch, axis=0).copy()
        k = np.random.randint(0, 4)
        if k > 0:
            noisy_patch, clean_patch = np.rot90(noisy_patch, k).copy(), np.rot90(clean_patch, k).copy()

        noisy_tensor = torch.from_numpy(noisy_patch).unsqueeze(0)
        clean_tensor = F.interpolate(torch.from_numpy(clean_patch).unsqueeze(0).unsqueeze(0),
                                     size=(128, 128), mode='bicubic', align_corners=False).squeeze(0)
        return noisy_tensor, clean_tensor

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.shortcut = nn.Sequential() if in_c == out_c else nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
        )
        self.final_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.final_relu(self.net(x) + self.shortcut(x))

class Downsample(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.down = nn.Sequential(nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True))
    def forward(self, x): return self.down(x)

class SEResUNet1x(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_f=64):
        super().__init__()
        self.enc1 = ResBlock(in_channels, base_f)
        self.down1 = Downsample(base_f, base_f * 2)
        self.enc2 = ResBlock(base_f * 2, base_f * 2)
        self.down2 = Downsample(base_f * 2, base_f * 4)
        self.enc3 = ResBlock(base_f * 4, base_f * 4)
        self.down3 = Downsample(base_f * 4, base_f * 8)

        self.bottleneck = ResBlock(base_f * 8, base_f * 8)

        self.up3 = nn.ConvTranspose2d(base_f * 8, base_f * 4, kernel_size=2, stride=2)
        self.se3 = SEBlock(base_f * 8)
        self.dec3 = ResBlock(base_f * 8, base_f * 4)

        self.up2 = nn.ConvTranspose2d(base_f * 4, base_f * 2, kernel_size=2, stride=2)
        self.se2 = SEBlock(base_f * 4)
        self.dec2 = ResBlock(base_f * 4, base_f * 2)

        self.up1 = nn.ConvTranspose2d(base_f * 2, base_f, kernel_size=2, stride=2)
        self.se1 = SEBlock(base_f * 2)
        self.dec1 = ResBlock(base_f * 2, base_f)

        self.final_conv = nn.Conv2d(base_f, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))

        u3 = self.dec3(self.se3(torch.cat([self.up3(b), e3], dim=1)))
        u2 = self.dec2(self.se2(torch.cat([self.up2(u3), e2], dim=1)))
        u1 = self.dec1(self.se1(torch.cat([self.up1(u2), e1], dim=1)))

        return self.final_conv(u1)

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.Tensor([[-1., 0., 1.],
                          [-2., 0., 2.],
                          [-1., 0., 1.]])

        self.kx = k.view(1, 1, 3, 3)
        self.ky = k.t().view(1, 1, 3, 3)

    def forward(self, pred, target):
        kx = self.kx.to(device=pred.device, dtype=pred.dtype)
        ky = self.ky.to(device=pred.device, dtype=pred.dtype)

        pred_dx = F.conv2d(pred, kx, padding=1)
        pred_dy = F.conv2d(pred, ky, padding=1)
        tgt_dx = F.conv2d(target, kx, padding=1)
        tgt_dy = F.conv2d(target, ky, padding=1)

        return torch.mean(torch.abs(pred_dx - tgt_dx)) + torch.mean(torch.abs(pred_dy - tgt_dy))

class SSIMLoss(nn.Module):
    def __init__(self, window_size=7):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window.expand(channel, 1, window_size, window_size).contiguous()

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type(): window = self.window
        else:
            window = self.create_window(self.window_size, channel).to(img1.device).type_as(img1)
            self.window, self.channel = window, channel

        mu1, mu2 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel), F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1, C2 = 0.05**2, 0.15**2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()

class FinalDenoiseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = SSIMLoss(window_size=7)
        self.l1 = nn.L1Loss()
        self.edge = EdgeLoss()

    def forward(self, pred, target):
        return (0.85 * self.ssim(pred, target)) + (0.10 * self.l1(pred, target)) + (0.05 * self.edge(pred, target))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Targeting compute device: {device}")

    train_loader = DataLoader(DenoiseDataset(CONFIG["noisy_dir"], CONFIG["clean_dir"]),
                              batch_size=CONFIG["batch_size"], shuffle=True,
                              num_workers=CONFIG["num_workers"], pin_memory=True,
                              persistent_workers=True, worker_init_fn=worker_init_fn)

    model = SEResUNet1x(base_f=64).to(device)
    criterion = FinalDenoiseLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    scaler = GradScaler()

    fixed_val_np = np.load(CONFIG["val_patch"]).astype(np.float32)
    fixed_val_tensor = torch.from_numpy(fixed_val_np).unsqueeze(0).unsqueeze(0).to(device)

    for epoch in range(CONFIG["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

        for noisy, clean in pbar:
            noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                restored = model(noisy)
                loss = criterion(restored, clean)

            if torch.isnan(loss) or torch.isinf(loss): continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], f"unet_stage1_ep{epoch+1}.pth"))
            model.eval()
            with torch.no_grad():
                sample_out = model(fixed_val_tensor).cpu().numpy().squeeze()
                np.save(os.path.join(CONFIG["save_dir"], f"sample_ep{epoch+1}.npy"), sample_out)

if __name__ == '__main__':
    main()
