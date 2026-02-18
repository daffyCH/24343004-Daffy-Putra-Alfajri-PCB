# ============================================
# PRAKTIKUM 1
# Eksplorasi Digitalisasi Citra
# ============================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

print("=== PRAKTIKUM DIGITALISASI CITRA (RGB) ===\n")

# =============================
# 1. AKUISISI DAN PEMBACAAN
# =============================

image_path = "C:/Users/manta/Pictures/" \
"Gambar WhatsApp 2024-10-04 pukul 10.13.06_f98fb544.jpg"

img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError("File citra tidak ditemukan.")

height, width, channels = img.shape

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Citra Digital (RGB)")
plt.axis("off")
plt.show()


# =============================
# 2. REPRESENTASI MATRIKS & VEKTOR
# =============================

print("Matriks 5x5 piksel pertama (Format BGR):")
print(img[0:5, 0:5])

vector = img.flatten()
print("\n25 elemen pertama representasi vektor:")
print(vector[:25])


# =============================
# 3. ANALISIS PARAMETER
# =============================

bit_depth_per_channel = img.dtype.itemsize * 8
total_bit_depth = bit_depth_per_channel * channels

resolution = width * height
aspect_ratio = width / height
intensity_per_channel = 2 ** bit_depth_per_channel
total_possible_colors = intensity_per_channel ** channels

memory_bytes = img.size * img.dtype.itemsize
memory_mb = memory_bytes / (1024*1024)

print("\n=== PARAMETER CITRA ===")
print(f"Resolusi: {width} x {height}")
print(f"Total Piksel: {resolution}")
print(f"Channel: {channels}")
print(f"Bit Depth per Channel: {bit_depth_per_channel}-bit")
print(f"Total Bit Depth: {total_bit_depth}-bit")
print(f"Tingkat Intensitas per Channel: {intensity_per_channel}")
print(f"Total Kemungkinan Warna: {total_possible_colors:,}")
print(f"Aspect Ratio: {aspect_ratio:.2f}")
print(f"Ukuran Memori: {memory_mb:.4f} MB")

# Simulasi perubahan
new_resolution = (width*2) * (height*2)
new_bit_depth = bit_depth_per_channel // 2
new_memory = new_resolution * (new_bit_depth/8) * channels
new_memory_mb = new_memory / (1024*1024)

print("\nSimulasi:")
print(f"Resolusi 2x, Bit depth 1/2 → Memori ≈ {new_memory_mb:.4f} MB")


# =============================
# 4. MANIPULASI DASAR
# =============================

# Cropping
crop = img[100:300, 100:300]

# Resizing
resize = cv2.resize(img, (width//2, height//2))

# Rotasi 90 derajat
rotate = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

# Visualisasi
fig, axes = plt.subplots(2,2, figsize=(10,8))

axes[0,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0,0].set_title("Original")

axes[0,1].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
axes[0,1].set_title("Cropping")

axes[1,0].imshow(cv2.cvtColor(resize, cv2.COLOR_BGR2RGB))
axes[1,0].set_title("Resizing 0.5x")

axes[1,1].imshow(cv2.cvtColor(rotate, cv2.COLOR_BGR2RGB))
axes[1,1].set_title("Rotasi 90°")

for ax in axes.ravel():
    ax.axis("off")

plt.tight_layout()
plt.show()

print("\n=== PRAKTIKUM SELESAI ===")

