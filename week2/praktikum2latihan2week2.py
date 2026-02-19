# ===============================================
# PRAKTIKUM 2 - SIMULASI EFEK ALIASING PADA CITRA
# ===============================================

import cv2
import numpy as np
import matplotlib.pyplot as plt


def simulate_image_aliasing(image, downsampling_factors):

    if image is None:
        raise ValueError("Input image tidak boleh None")

    results = {}
    h, w = image.shape[:2]

    for factor in downsampling_factors:

        # Downsampling langsung (tanpa anti-aliasing filter)
        small = image[::factor, ::factor]

        # Upscale kembali agar ukuran sama (nearest interpolation)
        restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        # Hitung error (MSE)
        mse = np.mean((image.astype(float) - restored.astype(float)) ** 2)

        results[factor] = {
            "downsampled": small,
            "restored": restored,
            "mse": float(mse)
        }

    return results


# ===============================
# MAIN PROGRAM
# ===============================

if __name__ == "__main__":

    print("=== SIMULASI IMAGE ALIASING ===")

    img = cv2.imread("C:/Users/manta/Pictures/Camera Roll/WIN_20260212_11_52_15_Pro.jpg")

    if img is None:
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        for i in range(0, 400, 10):
            cv2.line(img, (0, i), (400, i), (255, 255, 255), 1)
        print("Menggunakan citra sintetik garis frekuensi tinggi")

    factors = [2, 4, 8]

    results = simulate_image_aliasing(img, factors)

    fig, axes = plt.subplots(1, len(factors) + 1, figsize=(15, 5))

    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i, factor in enumerate(factors):
        restored = results[factor]["restored"]
        mse = results[factor]["mse"]

        axes[i+1].imshow(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))
        axes[i+1].set_title(f"Factor {factor}\nMSE={mse:.1f}")
        axes[i+1].axis("off")

    plt.suptitle("Simulasi Efek Aliasing akibat Downsampling", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    for f in factors:
        print(f"Factor {f} → MSE:", results[f]["mse"])
