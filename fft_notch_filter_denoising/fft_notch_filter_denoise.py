import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

def automated_fft_denoise(image_path, save_path, center_radius=40, z_score_thresh=5.0):
    try:
        img = np.load(image_path)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return

    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((X - ccol)**2 + (Y - crow)**2)

    high_freq_mag = magnitude.copy()
    high_freq_mag[dist_from_center <= center_radius] = 0

    valid_pixels = high_freq_mag[high_freq_mag > 0]
    if len(valid_pixels) == 0:
        np.save(save_path, img)
        return

    mean_val = np.mean(valid_pixels)
    std_val = np.std(valid_pixels)
    threshold = mean_val + (z_score_thresh * std_val)

    anomalies = (high_freq_mag > threshold).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    dilated_anomalies = cv2.dilate(anomalies, kernel, iterations=1)

    notch_mask = 1 - dilated_anomalies
    f_shift_filtered = f_shift * notch_mask

    f_ishift = np.fft.ifftshift(f_shift_filtered)
    img_restored = np.fft.ifft2(f_ishift)
    img_restored = np.abs(img_restored)

    np.save(save_path, img_restored)

def process_dataset(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    images = list(input_path.glob("*.npy"))

    if not images:
        print(f"No .npy files found in {input_path.absolute()}")
        return

    print(f"Found {len(images)} .npy files to process.")

    for img_path in tqdm(images, desc="Filtering FFT Spikes"):
        save_path = output_path / img_path.name
        automated_fft_denoise(img_path, save_path)

if __name__ == "__main__":
    process_dataset("data/noisy", "output/fft_filtered")
