import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import vgg19, VGG19_Weights
from tqdm import tqdm

from models.swinir_arch import SwinIR

CONFIG = {
    "clean_lr_dir": "data/denoised/",
    "gt_hr_dir": "data/clean_hr/",
    "pretrained_path": "checkpoints/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth",
    "save_dir": "./checkpoints_swin/",
    "batch_size": 4,
    "accum_steps": 8,
    "epochs": 30,
    "lr": 2e-5,
    "num_workers": 4
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)

class Stage2Dataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_files = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir) if f.endswith('.npy')])
        self.hr_files = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.endswith('.npy')])

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr = np.load(self.lr_files[idx]).astype(np.float32)
        hr = np.load(self.hr_files[idx]).astype(np.float32)

        lr_tensor = torch.from_numpy(lr).unsqueeze(0).repeat(3, 1, 1)
        hr_tensor = torch.from_numpy(hr).unsqueeze(0)

        return lr_tensor, hr_tensor

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
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
        window = self.create_window(self.window_size, channel).to(img1.device).type_as(img1)
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2
        C1, C2 = 0.05**2, 0.15**2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:16].to(device).eval()
        for param in vgg.parameters(): param.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        p_rgb, t_rgb = pred.repeat(1, 3, 1, 1), target.repeat(1, 3, 1, 1)
        return self.criterion(self.vgg(p_rgb), self.vgg(t_rgb))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SwinIR(upscale=2, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv').to(device)

    if os.path.exists(CONFIG["pretrained_path"]):
        checkpoint = torch.load(CONFIG["pretrained_path"], map_location=device)
        weights = checkpoint['params'] if 'params' in checkpoint else checkpoint
        model.load_state_dict(weights, strict=False)
        print(f"Loaded Pretrained weights: {CONFIG['pretrained_path']}")

    loader = DataLoader(Stage2Dataset(CONFIG["clean_lr_dir"], CONFIG["gt_hr_dir"]),
                        batch_size=CONFIG["batch_size"], shuffle=True,
                        num_workers=CONFIG["num_workers"], pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion_l1 = nn.L1Loss()
    criterion_ssim = SSIMLoss().to(device)
    criterion_perc = PerceptualLoss(device)
    scaler = GradScaler()

    for epoch in range(CONFIG["epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(loader, desc=f"Stage 2 Epoch {epoch+1}")

        for i, (lr, hr) in enumerate(pbar):
            lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)

            with autocast():
                out_rgb = model(lr)
                out_gray = torch.mean(out_rgb, dim=1, keepdim=True)

                loss = (0.5 * criterion_l1(out_gray, hr)) + \
                       (1.0 * criterion_ssim(out_gray, hr)) + \
                       (0.02 * criterion_perc(out_gray, hr))

                loss = loss / CONFIG["accum_steps"]

            scaler.scale(loss).backward()

            if (i + 1) % CONFIG["accum_steps"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            pbar.set_postfix({"Loss": f"{loss.item() * CONFIG['accum_steps']:.5f}"})

        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(CONFIG["save_dir"], f"swinir_stage2_ep{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved: {save_path}")

if __name__ == '__main__':
    main()
