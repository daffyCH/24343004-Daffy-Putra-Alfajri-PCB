# ============================================
# LATIHAN 1: ANALISIS CITRA PRIBADI
# ============================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image


# =================================================
# LOAD GAMBAR (Template Lama)
# =================================================
def load_image(path):
    try:
        if path.startswith("http"):
            response = requests.get(path)
            img = Image.open(BytesIO(response.content))
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        else:
            return cv2.imread(path)
    except:
        return None


# =================================================
# DISPLAY GRID (Template Lama)
# =================================================
def display_image_grid(images, titles, rows, cols):

    fig, axes = plt.subplots(rows, cols, figsize=(12,8))
    axes = axes.ravel()

    for i,(img,title) in enumerate(zip(images,titles)):
        if len(img.shape)==2:
            axes[i].imshow(img, cmap='gray')
        else:
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        axes[i].set_title(title)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


# =================================================
# ANALISIS CITRA SESUAI PERMINTAAN SOAL
# =================================================
def analyze_my_image(my_img, sample_img):

    # ============================================
    # 1. DIMENSI DAN RESOLUSI
    # ============================================
    h, w, c = my_img.shape
    resolution = w * h

    print("\n1. DIMENSI DAN RESOLUSI")
    print(f"Dimensi : {w} x {h}")
    print(f"Resolusi: {resolution:,} pixel")

    # ============================================
    # 2. ASPECT RATIO
    # ============================================
    aspect_ratio = w / h

    print("\n2. ASPECT RATIO")
    print(f"Aspect Ratio : {aspect_ratio:.2f}")

    # ============================================
    # 3. KONVERSI GRAYSCALE & PERBANDINGAN UKURAN
    # ============================================
    my_gray = cv2.cvtColor(my_img, cv2.COLOR_BGR2GRAY)

    original_size = my_img.size * my_img.dtype.itemsize
    gray_size = my_gray.size * my_gray.dtype.itemsize

    print("\n3. PERBANDINGAN UKURAN MEMORI")
    print(f"Original : {original_size/1024:.2f} KB")
    print(f"Grayscale: {gray_size/1024:.2f} KB")

    display_image_grid(
        [my_img, my_gray],
        ["Original", "Grayscale"],
        1,2
    )

    # ============================================
    # 4. STATISTIK CITRA
    # ============================================
    print("\n4. STATISTIK CITRA (Grayscale)")

    print(f"Mean : {np.mean(my_gray):.2f}")
    print(f"Std  : {np.std(my_gray):.2f}")
    print(f"Min  : {np.min(my_gray)}")
    print(f"Max  : {np.max(my_gray)}")

    # ============================================
    # 5. HISTOGRAM SEMUA CHANNEL
    # ============================================
    print("\n5. HISTOGRAM CHANNEL WARNA")

    plt.figure(figsize=(10,4))

    colors = ('b','g','r')
    for i,color in enumerate(colors):
        hist = cv2.calcHist([my_img],[i],None,[256],[0,256])
        plt.plot(hist, label=color)

    plt.title("Histogram RGB")
    plt.legend()
    plt.show()

    # ============================================
    # 6. PERBANDINGAN DENGAN SAMPLE
    # ============================================
    print("\n6. PERBANDINGAN DENGAN SAMPLE")

    sample_gray = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)

    display_image_grid(
        [my_img, sample_img],
        ["My Image", "Sample Image"],
        1,2
    )

    print(f"My Image Mean     : {np.mean(my_gray):.2f}")
    print(f"Sample Image Mean : {np.mean(sample_gray):.2f}")


# =================================================
# INPUT GAMBAR
# =================================================
my_image_path = "C:/Users/manta/Pictures/Camera Roll/WIN_20260212_11_52_15_Pro.jpg"
sample_image_path = "C:/Users/manta/Pictures/Camera Roll/WIN_20260212_11_52_33_Pro.jpg"

my_img = load_image(my_image_path)
sample_img = load_image(sample_image_path)

if my_img is None or sample_img is None:
    print("Gagal memuat gambar")
    exit()


# =================================================
# JALANKAN ANALISIS
# =================================================
analyze_my_image(my_img, sample_img)
