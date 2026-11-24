import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import random

from models.hat_arch import HAT

torch.backends.cudnn.benchmark = True

CONFIG = {
    "clean_lr_dir": "data/denoised/",
    "gt_hr_dir": "data/clean_hr/",
    "pretrained_path": "checkpoints/HAT-S_SRx2.pth",
    "save_dir": "./checkpoints_hat_s_fast/",
    "batch_size": 2,
    "accum_steps": 16,
    "epochs": 20,
    "lr": 2e-4,
    "warmup_epochs": 3,
    "val_size": 50,
    "num_workers": 0
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)

class PairedNpyDataset(Dataset):
    def __init__(self, lr_files, hr_files, phase='train'):
        print(f"Preloading {phase} data into RAM...")
        self.phase = phase
        self.lr_data = [np.load(f).astype(np.float32) for f in tqdm(lr_files, desc="Loading LR")]
        self.hr_data = [np.load(f).astype(np.float32) for f in tqdm(hr_files, desc="Loading HR")]

    def __len__(self):
        return len(self.lr_data)

    def __getitem__(self, idx):
        lr = self.lr_data[idx]
        hr = self.hr_data[idx]

        if self.phase == 'train':
            x = random.randint(0, 64)
            y = random.randint(0, 64)
        else:
            x = 32
            y = 32

        lr_crop = lr[y:y+64, x:x+64]

        hr_crop = hr[y*2:y*2+128, x*2:x*2+128]

        lr_tensor = torch.from_numpy(lr_crop).unsqueeze(0).repeat(3, 1, 1)

        hr_tensor = torch.from_numpy(hr_crop).unsqueeze(0)

        return lr_tensor, hr_tensor

def split_dataset(lr_dir, hr_dir, val_size):
    lr_files = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir) if f.endswith('.npy')])
    hr_files = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.endswith('.npy')])

    train_lr, val_lr = lr_files[:-val_size], lr_files[-val_size:]
    train_hr, val_hr = hr_files[:-val_size], hr_files[-val_size:]

    return train_lr, train_hr, val_lr, val_hr

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target)**2 + self.eps**2))

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)

def fast_gpu_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
    return psnr.mean().item()

def apply_gpu_augmentations(lr, hr):
    if torch.rand(1) > 0.5:
        lr, hr = torch.flip(lr, dims=[3]), torch.flip(hr, dims=[3])
    if torch.rand(1) > 0.5:
        lr, hr = torch.flip(lr, dims=[2]), torch.flip(hr, dims=[2])

    k = torch.randint(0, 4, (1,)).item()
    if k > 0:
        lr, hr = torch.rot90(lr, k, [2, 3]), torch.rot90(hr, k, [2, 3])

    return lr, hr

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Targeting compute device: {device}")

    model = HAT(
        upscale=2, in_chans=3, img_size=64, window_size=16,
        compress_ratio=24, squeeze_factor=24, conv_scale=0.01, overlap_ratio=0.5,
        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=144,
        num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle',
        resi_connection='1conv'
    ).to(device)

    if os.path.exists(CONFIG["pretrained_path"]):
        checkpoint = torch.load(CONFIG["pretrained_path"], map_location=device)
        weights = checkpoint['params_ema'] if 'params_ema' in checkpoint else checkpoint.get('params', checkpoint)
        model.load_state_dict(weights, strict=True)
        print(f"Loaded Pretrained weights: {CONFIG['pretrained_path']}")
    else:
        print("CRITICAL: HAT-S_SRx2.pth missing. Aborting.")
        return

    t_lr, t_hr, v_lr, v_hr = split_dataset(CONFIG["clean_lr_dir"], CONFIG["gt_hr_dir"], CONFIG["val_size"])
    print(f"Dataset Split -> Train: {len(t_lr)} | Val: {len(v_lr)}")

    train_loader = DataLoader(PairedNpyDataset(t_lr, t_hr, phase='train'),
                              batch_size=CONFIG["batch_size"], shuffle=True,
                              num_workers=CONFIG["num_workers"], pin_memory=True)

    val_loader = DataLoader(PairedNpyDataset(v_lr, v_hr, phase = 'val'),
                            batch_size=1, shuffle=False,
                            num_workers=CONFIG["num_workers"], pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)

    total_iters_per_epoch = math.ceil(len(train_loader) / CONFIG["accum_steps"])
    total_warmup_iters = CONFIG["warmup_epochs"] * total_iters_per_epoch
    total_cosine_iters = (CONFIG["epochs"] - CONFIG["warmup_epochs"]) * total_iters_per_epoch

    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=total_warmup_iters)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_cosine_iters)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[total_warmup_iters])

    criterion_charb = CharbonnierLoss().to(device)
    criterion_ssim = SSIMLoss().to(device)
    scaler = GradScaler()

    for epoch in range(CONFIG["epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_loss = 0.0
        epoch_psnr = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]")

        for i, (lr, hr) in enumerate(pbar):
            lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)

            lr, hr = apply_gpu_augmentations(lr, hr)

            with autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                out_rgb = model(lr)

            out_rgb32 = out_rgb.float()
            hr32 = hr.float()

            out_gray = (0.299 * out_rgb32[:, 0:1, :, :]) + \
                       (0.587 * out_rgb32[:, 1:2, :, :]) + \
                       (0.114 * out_rgb32[:, 2:3, :, :])

            loss = criterion_charb(out_gray, hr32) + criterion_ssim(out_gray, hr32)

            loss = loss / CONFIG["accum_steps"]

            scaler.scale(loss).backward()

            current_loss = loss.item() * CONFIG["accum_steps"]
            current_psnr = fast_gpu_psnr(out_gray.detach(), hr)

            epoch_loss += current_loss
            epoch_psnr += current_psnr

            is_accum_step = (i + 1) % CONFIG["accum_steps"] == 0 or (i + 1) == len(train_loader)

            if is_accum_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                scheduler.step()

            pbar.set_postfix({"Loss": f"{current_loss:.4f}", "PSNR": f"{current_psnr:.2f}dB"})

        model.eval()
        val_loss = 0.0
        val_psnr = 0.0

        with torch.no_grad():
            for lr, hr in val_loader:
                lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)

                out_rgb = model(lr)

                out_gray = (0.2989 * out_rgb[:, 0:1, :, :] + \
                            0.5870 * out_rgb[:, 1:2, :, :] + \
                            0.1140 * out_rgb[:, 2:3, :, :]).clip(0, 1)

                val_loss += (criterion_charb(out_gray, hr) + criterion_ssim(out_gray, hr)).item()
                val_psnr += fast_gpu_psnr(out_gray, hr)

        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_psnr = epoch_psnr / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = val_psnr / len(val_loader)

        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"LR:   {scheduler.get_last_lr()[0]:.6f}")
        print(f"Train - Avg Loss: {avg_train_loss:.5f} | Avg PSNR: {avg_train_psnr:.2f} dB")
        print(f"Val   - Avg Loss: {avg_val_loss:.5f} | Avg PSNR: {avg_val_psnr:.2f} dB\n")

        if (epoch + 1) % 5 == 0 or (epoch + 1) == CONFIG["epochs"]:
            torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], f"hat_s_ep{epoch+1}.pth"))

if __name__ == '__main__':
    main()
