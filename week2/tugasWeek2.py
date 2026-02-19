# ==========================================================
# PROYEK MINI
# Implementasi & Analisis Konversi Model Warna
# ==========================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import warnings
warnings.filterwarnings('ignore')

# ==========================================================
# 1. LOAD 3 CITRA DENGAN KONDISI BERBEDA
# ==========================================================

def load_images():

    paths = [
        "C:/Users/manta/Pictures/IMG_20241127_142238_127.jpg",
        "C:/Users/manta/Pictures/IMG_20241024_135051_815.jpg",
        "C:/Users/manta/Pictures/IMG_20240819_202454_668.jpg"
    ]

    images = []

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            img = np.zeros((300, 400, 3), dtype=np.uint8)
            cv2.rectangle(img, (100, 50), (300, 250), (255,255,255), -1)
        images.append(img)

    return images


# ==========================================================
# 2. KONVERSI RUANG WARNA
# ==========================================================

def convert_color_spaces(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    return gray, hsv, lab


# ==========================================================
# 3. KUANTISASI
# ==========================================================

def uniform_quantization(img, levels):

    step = 256 // levels
    quant = (img // step) * step
    return quant


def nonuniform_quantization(img, levels):

    hist, bins = np.histogram(img.flatten(), 256, [0,256])
    cdf = hist.cumsum()
    cdf_norm = cdf / cdf.max()

    mapping = np.floor(cdf_norm * (levels-1))
    quant = mapping[img].astype(np.uint8)
    quant = (quant * (255 // (levels-1))).astype(np.uint8)

    return quant


# ==========================================================
# 4. ANALISIS HISTOGRAM
# ==========================================================

def plot_histogram(img, title):

    plt.hist(img.ravel(), 256, [0,256])
    plt.title(title)
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")


# ==========================================================
# 5. PARAMETER TEKNIS
# ==========================================================

def compute_metrics(original, quantized):

    mse = np.mean((original.astype(float) - quantized.astype(float))**2)
    psnr = 10*np.log10(255**2/mse) if mse > 0 else float('inf')

    mem_before = original.nbytes
    mem_after  = quantized.nbytes
    ratio = mem_before / mem_after

    return mse, psnr, mem_before, mem_after, ratio


# ==========================================================
# MAIN PROGRAM
# ==========================================================

if __name__ == "__main__":

    print("=== PROYEK MINI MODEL WARNA ===")

    images = load_images()
    levels = 16

    for idx, img in enumerate(images):

        print(f"\n--- Citra {idx+1} ---")

        gray, hsv, lab = convert_color_spaces(img)

        # Gunakan channel intensitas untuk kuantisasi
        gray_u = uniform_quantization(gray, levels)
        gray_nu = nonuniform_quantization(gray, levels)

        start = time.time()
        mse, psnr, mem_before, mem_after, ratio = compute_metrics(gray, gray_u)
        end = time.time()

        print("MSE:", mse)
        print("PSNR:", psnr)
        print("Memory Before:", mem_before)
        print("Memory After :", mem_after)
        print("Compression Ratio:", ratio)
        print("Processing Time:", end-start)

        # ======================
        # VISUALISASI
        # ======================

        plt.figure(figsize=(12,6))

        plt.subplot(2,3,1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original")
        plt.axis("off")

        plt.subplot(2,3,2)
        plt.imshow(gray, cmap='gray')
        plt.title("Grayscale")
        plt.axis("off")

        plt.subplot(2,3,3)
        plt.imshow(gray_u, cmap='gray')
        plt.title("Uniform Quantization")
        plt.axis("off")

        plt.subplot(2,3,4)
        plot_histogram(gray, "Histogram Original")

        plt.subplot(2,3,5)
        plot_histogram(gray_u, "Histogram Uniform")

        plt.subplot(2,3,6)
        plt.imshow(gray_nu, cmap='gray')
        plt.title("Non-Uniform Quantization")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
