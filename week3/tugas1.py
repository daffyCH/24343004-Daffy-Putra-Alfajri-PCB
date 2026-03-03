# ============================================
# PRAKTIKUM: TRANSFORMASI GEOMETRIK & INTERPOLASI
# ============================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

print("=== PIPELINE TRANSFORMASI GEOMETRIK ===")


# =====================================================
# KONFIGURASI PATH GAMBAR (EDIT DI SINI)
# =====================================================

PATH_REFERENCE = "C:/Users/manta/Pictures/lurus.jpg"   # gambar lurus
PATH_MOVING    = "C:/Users/manta/Pictures/miring.jpg"      # gambar miring


# =====================================================
# FUNGSI METRIK
# =====================================================

def compute_metrics(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 0, float("inf")

    psnr = 10 * np.log10((255 ** 2) / mse)
    return mse, psnr


def show_rgb(title, img):
    plt.figure(figsize=(6,5))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


# =====================================================
# LOAD CITRA
# =====================================================

ref_img = cv2.imread(PATH_REFERENCE)
moving_img = cv2.imread(PATH_MOVING)

if ref_img is None:
    raise ValueError("Reference image tidak ditemukan.")
if moving_img is None:
    raise ValueError("Moving image tidak ditemukan.")

h, w, _ = ref_img.shape
moving_img = cv2.resize(moving_img, (w, h))

show_rgb("Reference Image", ref_img)
show_rgb("Moving Image", moving_img)


# =====================================================
# ================== TAHAP 1 ==========================
# TRANSFORMASI DASAR (MATRIS HOMOGEN)
# =====================================================

print("\n=== TAHAP 1: Translasi, Rotasi, Scaling ===")

# Translasi
M_trans = np.array([[1,0,40],
                    [0,1,30],
                    [0,0,1]], dtype=np.float32)
trans_img = cv2.warpPerspective(ref_img, M_trans, (w,h))

# Rotasi
theta = np.deg2rad(20)
cx, cy = w/2, h/2
M_rot = np.array([
    [np.cos(theta), -np.sin(theta), cx - cx*np.cos(theta) + cy*np.sin(theta)],
    [np.sin(theta),  np.cos(theta), cy - cx*np.sin(theta) - cy*np.cos(theta)],
    [0,0,1]
], dtype=np.float32)
rot_img = cv2.warpPerspective(ref_img, M_rot, (w,h))

# Scaling
M_scale = np.array([[1.2,0,0],
                    [0,1.2,0],
                    [0,0,1]], dtype=np.float32)
scale_img = cv2.warpPerspective(ref_img, M_scale, (w,h))

plt.figure(figsize=(12,4))
imgs = [trans_img, rot_img, scale_img]
titles = ["Translasi", "Rotasi", "Scaling"]

for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()


# =====================================================
# ================== TAHAP 2 ==========================
# TRANSFORMASI AFFINE (3 TITIK)
# =====================================================

print("\n=== TAHAP 2: Transformasi Affine ===")

pts_ref_aff = np.float32([[100,100],
                          [w-100,100],
                          [100,h-100]])

pts_mov_aff = np.float32([[120,150],
                          [w-80,120],
                          [150,h-80]])

start = time.time()
M_aff = cv2.getAffineTransform(pts_mov_aff, pts_ref_aff)
reg_aff = cv2.warpAffine(moving_img, M_aff, (w, h))
time_aff = time.time() - start

mse_aff, psnr_aff = compute_metrics(ref_img, reg_aff)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
plt.title("Reference")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(reg_aff, cv2.COLOR_BGR2RGB))
plt.title(f"Affine\nPSNR={psnr_aff:.2f}")
plt.axis("off")

plt.tight_layout()
plt.show()

print(f"MSE  : {mse_aff:.2f}")
print(f"PSNR : {psnr_aff:.2f} dB")
print(f"Waktu: {time_aff:.6f} detik")


# =====================================================
# ================== TAHAP 3 ==========================
# TRANSFORMASI PERSPEKTIF (4 TITIK) + INTERPOLASI
# =====================================================

print("\n=== TAHAP 3: Transformasi Perspektif ===")

pts_ref = np.float32([[100,100],
                      [w-100,100],
                      [w-100,h-100],
                      [100,h-100]])

pts_mov = np.float32([[120,150],
                      [w-80,120],
                      [w-150,h-80],
                      [150,h-50]])

start = time.time()
M_persp = cv2.getPerspectiveTransform(pts_mov, pts_ref)
reg_persp = cv2.warpPerspective(moving_img, M_persp, (w, h))
time_persp = time.time() - start

mse_persp, psnr_persp = compute_metrics(ref_img, reg_persp)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
plt.title("Reference")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(reg_persp, cv2.COLOR_BGR2RGB))
plt.title(f"Perspective\nPSNR={psnr_persp:.2f}")
plt.axis("off")

plt.tight_layout()
plt.show()

print(f"MSE  : {mse_persp:.2f}")
print(f"PSNR : {psnr_persp:.2f} dB")
print(f"Waktu: {time_persp:.6f} detik")


# ================= INTERPOLASI =================

print("\n--- Perbandingan Interpolasi ---")

methods = [
    ("Nearest", cv2.INTER_NEAREST),
    ("Bilinear", cv2.INTER_LINEAR),
    ("Bicubic", cv2.INTER_CUBIC)
]

plt.figure(figsize=(12,4))

for i,(name, flag) in enumerate(methods):
    start = time.time()
    reg = cv2.warpPerspective(moving_img, M_persp, (w,h), flags=flag)
    t = time.time() - start
    mse, psnr = compute_metrics(ref_img, reg)

    plt.subplot(1,3,i+1)
    plt.imshow(cv2.cvtColor(reg, cv2.COLOR_BGR2RGB))
    plt.title(f"{name}\nPSNR={psnr:.2f}")
    plt.axis("off")

    print(f"{name}: MSE={mse:.2f}, PSNR={psnr:.2f}, Waktu={t:.6f}")

plt.tight_layout()
plt.show()

print("\n=== SELESAI ===")
