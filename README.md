# Image Restoration: Denoising and 2x Super-Resolution

Pipelines for restoring noisy 128x128 grayscale `.npy` images and producing 256x256 clean outputs, scored with PSNR and SSIM.

The approach has two stages: denoise first, then upscale with a pretrained transformer fine-tuned on the task data.

## Pipelines

| Folder | What it does |
|---|---|
| `fft_notch_filter_denoising/` | Classical baseline. Removes periodic noise by detecting spikes in the 2D FFT spectrum (z-score threshold, dilated notch mask) and inverting the transform. |
| `se_resunet_edge_aware_denoiser/` | Stage 1. Squeeze-and-Excitation residual U-Net trained with a combined SSIM, L1 and edge loss (85/10/5) to reduce blur. Mixed precision, cosine LR, gradient clipping. |
| `nafnet_denoiser/` | Stage 1 alternative. Deep NAFNet (36 blocks, U-shaped encoder/decoder with simplified channel attention and SimpleGate) running batched inference to produce denoised images. |
| `hat_super_resolution_finetuning/` | Stage 2. Fine-tunes a pretrained HAT-S x2 model on denoised inputs with Charbonnier and SSIM loss. Data is preloaded into RAM, with GPU augmentations, gradient accumulation and warmup. |
| `swinir_perceptual_super_resolution/` | Alternative Stage 2. Fine-tunes lightweight SwinIR x2 with L1, SSIM and VGG19 perceptual loss. |
| `two_stage_denoise_and_hat_sr_evaluation/` | End-to-end evaluation of denoiser followed by HAT on a held-out split, reporting PSNR and SSIM after each stage. |

Shared model definitions are in `models/`.

## Attribution

`models/hat_arch.py` is adapted from [HAT](https://github.com/XPixelGroup/HAT) (Apache-2.0). `models/swinir_arch.py` is adapted from [SwinIR](https://github.com/JingyunLiang/SwinIR) (Apache-2.0). Pretrained weights are from those projects and are not included here.
