# ============================================
# PRAKTIKUM 1: DASAR-DASAR CITRA DIGITAL
# ============================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image

print("=== PRAKTIKUM 1: DASAR-DASAR CITRA DIGITAL ===")
print("Materi: Representasi Citra, Resolusi, Depth, Aspect Ratio\n")

# =============== FUNGSI BANTU ===============
def download_sample_image():
    """Download sample image from internet"""
    try:
        url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except:
        print("Gagal download gambar. Menggunakan gambar dummy.")
        return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)


def analyze_image_properties(img, name="Image"):
    """Analyze and display image properties"""
    if len(img.shape) == 2:
        height, width = img.shape
        channels = 1
    else:
        height, width, channels = img.shape
    
    resolution = width * height
    aspect_ratio = width / height
    depth = img.dtype.itemsize * 8
    
    print(f"\n{'='*40}")
    print(f"ANALYSIS: {name}")
    print(f"{'='*40}")
    print(f"Dimensions: {width} x {height}")
    print(f"Channels: {channels}")
    print(f"Resolution: {resolution:,} pixels")
    print(f"Aspect Ratio: {aspect_ratio:.2f} ({width}:{height})")
    print(f"Bit Depth: {depth}-bit ({img.dtype})")
    
    memory_bytes = img.size * img.dtype.itemsize
    memory_kb = memory_bytes / 1024
    memory_mb = memory_kb / 1024
    
    print(f"Memory Size: {memory_bytes:,} bytes")
    print(f"             {memory_kb:.2f} KB")
    print(f"             {memory_mb:.2f} MB")
    
    if channels == 1:
        print(f"Intensity Range: [{img.min()}, {img.max()}]")
        print(f"Mean Intensity: {img.mean():.2f}")
        print(f"Std Deviation: {img.std():.2f}")
    
    return {
        'width': width,
        'height': height,
        'channels': channels,
        'resolution': resolution,
        'aspect_ratio': aspect_ratio,
        'depth': depth,
        'memory_bytes': memory_bytes
    }


def display_image_grid(images, titles, rows, cols, figsize=(15, 10)):
    """Display multiple images in a grid"""
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.ravel() if rows > 1 or cols > 1 else [axes]
    
    for idx, (img, title) in enumerate(zip(images, titles)):
        if len(img.shape) == 2:
            axes[idx].imshow(img, cmap='gray')
        else:
            axes[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(title)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

# =============== MAIN PRAKTIKUM ===============

# 1. LOAD IMAGE
print("\n1. LOADING SAMPLE IMAGE")
original_img = download_sample_image()
props_original = analyze_image_properties(original_img, "Original Color Image")

# 2. REPRESENTASI MATRIKS
print("\n\n2. REPRESENTASI SEBAGAI MATRIKS")

x, y = 100, 100
pixel_value = original_img[x, y]
print(f"Pixel pada posisi ({x}, {y}): BGR = {pixel_value}")

print("\nArea 5x5 pixel dari posisi (100,100):")
print(original_img[100:105, 100:105])

# 3. KONVERSI GRAYSCALE
print("\n\n3. KONVERSI KE GRAYSCALE")
gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
props_gray = analyze_image_properties(gray_img, "Grayscale Image")

# 4. ANALISIS BIT DEPTH
print("\n\n4. PENGARUH BIT DEPTH")

img_8bit = gray_img.astype(np.uint8)
img_4bit = (gray_img // 16).astype(np.uint8) * 16
img_2bit = (gray_img // 64).astype(np.uint8) * 64
img_1bit = (gray_img // 128).astype(np.uint8) * 255

images = [img_8bit, img_4bit, img_2bit, img_1bit]
titles = ['8-bit', '4-bit', '2-bit', '1-bit']
display_image_grid(images, titles, 1, 4, figsize=(16, 4))

# 5. ANALISIS ASPECT RATIO
print("\n\n5. PENGARUH ASPECT RATIO")

img_4_3 = cv2.resize(gray_img, (800, 600))
img_16_9 = cv2.resize(gray_img, (800, 450))
img_1_1 = cv2.resize(gray_img, (600, 600))
img_21_9 = cv2.resize(gray_img, (840, 360))

images = [img_4_3, img_16_9, img_1_1, img_21_9]
titles = ['4:3', '16:9', '1:1', '21:9']
display_image_grid(images, titles, 2, 2, figsize=(12, 8))

# 6. SEPARASI CHANNEL
print("\n\n6. SEPARASI CHANNEL WARNA")

b, g, r = cv2.split(original_img)

zeros = np.zeros_like(b)
blue_channel = cv2.merge([b, zeros, zeros])
green_channel = cv2.merge([zeros, g, zeros])
red_channel = cv2.merge([zeros, zeros, r])

images = [original_img, blue_channel, green_channel, red_channel]
titles = ['Original', 'Blue', 'Green', 'Red']
display_image_grid(images, titles, 2, 2, figsize=(12, 8))

# 7. HISTOGRAM
print("\n\n7. ANALISIS HISTOGRAM")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(gray_img.ravel(), bins=256, range=(0, 256))
axes[0].set_title('Grayscale Histogram')

colors = ('b', 'g', 'r')
for i, color in enumerate(colors):
    hist = cv2.calcHist([original_img], [i], None, [256], [0, 256])
    axes[1].plot(hist)
axes[1].set_title('Color Histogram')

cumulative_hist = np.cumsum(np.histogram(gray_img.ravel(), 256, [0, 256])[0])
axes[2].plot(cumulative_hist)
axes[2].set_title('Cumulative Histogram')

plt.tight_layout()
plt.show()

# 8. MEMORY ANALYSIS
print("\n\n8. ANALISIS MEMORI")

sizes = [(640, 480), (1920, 1080), (3840, 2160)]
formats = ['Grayscale', 'RGB', 'RGBA']

for w, h in sizes:
    for fmt in formats:
        if fmt == 'Grayscale':
            depth = 1
        elif fmt == 'RGB':
            depth = 3
        else:
            depth = 4
        
        memory = w * h * depth
        memory_mb = memory / (1024 * 1024)
        
        print(f"{w}x{h} {fmt} -> {memory_mb:.2f} MB")

print("\n=== PRAKTIKUM SELESAI ===")
