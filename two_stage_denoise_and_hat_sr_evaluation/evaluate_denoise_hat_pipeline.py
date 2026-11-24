import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

from models.hat_arch import HAT

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

def calculate_metrics(target, reference):
    target = target.squeeze()
    reference = reference.squeeze()
    dr = 1.0 if reference.max() <= 1.0 else 255.0
    psnr_val = psnr_metric(reference, target, data_range=dr)

    win_size = min(7, target.shape[0], target.shape[1])
    win_size = win_size if win_size % 2 != 0 else win_size - 1

    ssim_val = ssim_metric(reference, target, data_range=dr, win_size=win_size)
    return psnr_val, ssim_val

def run_validation_scoring(unet_path, hat_path, noisy_dir, clean_dir, val_size=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting validation inference on: {device}")

    unet = SEResUNet1x(base_f=64).to(device)
    unet.load_state_dict(torch.load(unet_path, map_location=device))
    unet.eval()

    hat = HAT(upscale=2, in_chans=3, img_size=64, window_size=16, compress_ratio=24,
              squeeze_factor=24, conv_scale=0.01, overlap_ratio=0.5, img_range=1.,
              depths=[6, 6, 6, 6, 6, 6], embed_dim=144, num_heads=[6, 6, 6, 6, 6, 6],
              mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv').to(device)
    hat.load_state_dict(torch.load(hat_path, map_location=device))
    hat.eval()

    all_noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.npy')])
    val_noisy_files = all_noisy_files[-val_size:]

    results = {
        "unet_psnr": [], "unet_ssim": [],
        "hat_psnr": [], "hat_ssim": []
    }

    print(f"Total dataset size: {len(all_noisy_files)}")
    print(f"Evaluating strictly on the {len(val_noisy_files)} validation images...")

    with torch.no_grad():
        for filename in tqdm(val_noisy_files):
            noisy_np = np.load(os.path.join(noisy_dir, filename)).astype(np.float32).squeeze()
            clean_hr_np = np.load(os.path.join(clean_dir, filename)).astype(np.float32).squeeze()

            clean_lr_tensor = F.interpolate(torch.from_numpy(clean_hr_np).unsqueeze(0).unsqueeze(0),
                                            size=(128, 128), mode='bicubic', align_corners=False)
            clean_lr_np = clean_lr_tensor.squeeze().numpy()

            input_t = torch.from_numpy(noisy_np).unsqueeze(0).unsqueeze(0).to(device)

            denoised_t = unet(input_t)
            denoised_np = denoised_t.cpu().numpy().squeeze()

            out_rgb = hat(denoised_t.repeat(1, 3, 1, 1))
            final_sr_t = torch.mean(out_rgb, dim=1, keepdim=True)
            final_sr_np = final_sr_t.cpu().numpy().squeeze().clip(0, 1)

            p1, s1 = calculate_metrics(denoised_np, clean_lr_np)
            p2, s2 = calculate_metrics(final_sr_np, clean_hr_np)

            results["unet_psnr"].append(p1)
            results["unet_ssim"].append(s1)
            results["hat_psnr"].append(p2)
            results["hat_ssim"].append(s2)

    print("\n" + "="*30)
    print("VALIDATION SET EVALUATION RESULTS")
    print("="*30)
    print(f"STAGE 1 (UNet Denoising):")
    print(f"  Avg PSNR: {np.mean(results['unet_psnr']):.4f} dB")
    print(f"  Avg SSIM: {np.mean(results['unet_ssim']):.4f}")
    print("-"*30)
    print(f"STAGE 2 (HAT Super-Res):")
    print(f"  Avg PSNR: {np.mean(results['hat_psnr']):.4f} dB")
    print(f"  Avg SSIM: {np.mean(results['hat_ssim']):.4f}")
    print("="*30)

if __name__ == '__main__':
    UNET_W = 'checkpoints_denoiser_sobel/unet_stage1_ep30.pth'
    HAT_W = 'checkpoints_hat_s_fast/hat_s_ep20.pth'

    N_DIR = 'data/noisy/'
    C_DIR = 'data/clean_hr/'

    run_validation_scoring(UNET_W, HAT_W, N_DIR, C_DIR)
