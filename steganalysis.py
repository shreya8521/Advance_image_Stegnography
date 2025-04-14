# import sys
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2


# def calculate_histogram_difference(original_path: str, stego_path: str):
#     original_img = Image.open(original_path).convert("RGB")
#     stego_img = Image.open(stego_path).convert("RGB")

#     original_hist = original_img.histogram()
#     stego_hist = stego_img.histogram()

#     # Normalize the histograms
#     original_hist_np = np.array(original_hist).astype(float)
#     stego_hist_np = np.array(stego_hist).astype(float)

#     # Apply Chi-square distance to the histograms
#     chi_square_dist = 0.5 * np.sum(((original_hist_np - stego_hist_np) ** 2) / (original_hist_np + stego_hist_np + 1e-6))

#     print(f"Chi-Square Histogram Difference: {chi_square_dist:.2f}")


#     plt.figure(figsize=(12, 6))
#     plt.plot(original_hist_np, label='Original')
#     plt.plot(stego_hist_np, label='Stego', alpha=0.7)
#     plt.title('Histogram Comparison')
#     plt.legend()
#     plt.xlabel('Pixel Value')
#     plt.ylabel('Frequency')
#     plt.grid(True)
#     plt.show()


# def calculate_noise_variance(image_path):
#     # Ensure the image is loaded properly
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if image is None:
#         raise FileNotFoundError(f"Image not found: {image_path}")

#     # Apply Laplacian to detect noise
#     laplacian = cv2.Laplacian(image, cv2.CV_64F)
#     variance = laplacian.var()
#     return variance



# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage: python steganalysis.py <original_image> <stego_image>")
#         sys.exit(1)

#     original_image_path = sys.argv[1]
#     stego_image_path = sys.argv[2]

#     calculate_histogram_difference(original_image_path, stego_image_path)

#     # ✅ Use dynamic paths, not hardcoded
#     original_noise = calculate_noise_variance(original_image_path)
#     stego_noise = calculate_noise_variance(stego_image_path)

#     print(f"Original Noise Variance: {original_noise:.2f}")
#     print(f"Stego Noise Variance: {stego_noise:.2f}")
#     print(f"Noise Difference: {abs(original_noise - stego_noise):.2f}")

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import wave
import contextlib

def is_image(file_path):
    return file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))

def is_audio(file_path):
    return file_path.lower().endswith(('.wav',))


# ---------- Image Analysis ----------
def calculate_histogram_difference(original_path, stego_path):
    original_img = Image.open(original_path).convert("RGB")
    stego_img = Image.open(stego_path).convert("RGB")

    original_hist = np.array(original_img.histogram()).astype(float)
    stego_hist = np.array(stego_img.histogram()).astype(float)

    chi_square_dist = 0.5 * np.sum(((original_hist - stego_hist) ** 2) / (original_hist + stego_hist + 1e-6))
    print(f"[Image] Chi-Square Histogram Difference: {chi_square_dist:.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(original_hist, label='Original')
    plt.plot(stego_hist, label='Stego', alpha=0.7)
    plt.title('Histogram Comparison (Image)')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_noise_variance(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return laplacian.var()

# ---------- Audio Analysis ----------
def analyze_audio_difference(original_path, stego_path):
    import scipy.io.wavfile as wav
    rate1, data1 = wav.read(original_path)
    rate2, data2 = wav.read(stego_path)

    if data1.shape != data2.shape:
        print("[Audio] Warning: Audio dimensions do not match")
        return

    mse = np.mean((data1.astype(np.float32) - data2.astype(np.float32)) ** 2)
    print(f"[Audio] MSE between original and stego audio: {mse:.2f}")

    plt.figure(figsize=(12, 4))
    plt.plot(data1[:1000], label='Original')
    plt.plot(data2[:1000], label='Stego', alpha=0.7)
    plt.title('Waveform Comparison (First 1000 samples)')
    plt.legend()
    plt.grid(True)
    plt.show()


# ---------- Main ----------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python steganalysis.py <original_file> <stego_file>")
        sys.exit(1)

    original_path = sys.argv[1]
    stego_path = sys.argv[2]

    if is_image(original_path) and is_image(stego_path):
        calculate_histogram_difference(original_path, stego_path)
        orig_noise = calculate_noise_variance(original_path)
        stego_noise = calculate_noise_variance(stego_path)
        print(f"[Image] Original Noise Variance: {orig_noise:.2f}")
        print(f"[Image] Stego Noise Variance: {stego_noise:.2f}")
        print(f"[Image] Noise Difference: {abs(orig_noise - stego_noise):.2f}")

    elif is_audio(original_path) and is_audio(stego_path):
        analyze_audio_difference(original_path, stego_path)

    else:
        print("Unsupported file type or mismatched types. Please provide valid image/audio/video pairs.")
