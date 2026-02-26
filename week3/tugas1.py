import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

print("=== PIPELINE TRANSFORMASI GEOMETRIK ===")

# =====================================================
# FUNGSI BANTU
# =====================================================

def create_test_image(size=300):
    img = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(img, (50,50), (size-50,size-50), 255, 2)
    cv2.circle(img, (size//2, size//2), 50, 200, 2)
    cv2.putText(img, 'DOC', (size//2-40, size//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    return img

def compute_metrics(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float))**2)
    psnr = 10*np.log10(255**2/mse) if mse != 0 else float("inf")
    return mse, psnr


# =====================================================
# 1. DUA CITRA OBJEK SERUPA (PERSPEKTIF BERBEDA)
# =====================================================

print("\n1. Membuat dua citra perspektif berbeda")

ref_img = create_test_image(300)
h, w = ref_img.shape

pts1 = np.float32([[50,50], [w-50,50], [w-50,h-50], [50,h-50]])
pts2 = np.float32([[20,80], [w-30,30], [w-80,h-20], [60,h-10]])

M_true = cv2.getPerspectiveTransform(pts1, pts2)
moving_img = cv2.warpPerspective(ref_img, M_true, (w, h))


# =====================================================
# 2. TRANSFORMASI GEOMETRIK DASAR (MATRIS & HOMOGEN)
# =====================================================

print("\n2. Transformasi Dasar (Translasi, Rotasi, Scaling)")

# Translasi
# Translasi (homogen)
M_trans = np.array([[1,0,30],
                    [0,1,20],
                    [0,0,1]], dtype=np.float32)
trans_img = cv2.warpPerspective(ref_img, M_trans, (w,h))

# Rotasi (homogen)
theta = np.deg2rad(20)
M_rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta),  np.cos(theta), 0],
                  [0,0,1]], dtype=np.float32)
rot_img = cv2.warpPerspective(ref_img, M_rot, (w,h))

# Scaling (homogen)
M_scale = np.array([[1.2,0,0],
                    [0,1.2,0],
                    [0,0,1]], dtype=np.float32)
scale_img = cv2.warpPerspective(ref_img, M_scale, (w,h))


# =====================================================
# 3. ESTIMASI AFFINE (3 TITIK)
# =====================================================

print("\n3. Registrasi Affine (3 titik)")

pts1_aff = np.float32([[50,50], [250,50], [50,250]])
pts2_aff = np.float32([[20,80], [260,40], [70,260]])

start = time.time()
M_aff = cv2.getAffineTransform(pts2_aff, pts1_aff)
reg_aff = cv2.warpAffine(moving_img, M_aff, (w, h))
time_aff = time.time() - start

mse_aff, psnr_aff = compute_metrics(ref_img, reg_aff)


# =====================================================
# 4. ESTIMASI PERSPEKTIF (4 TITIK)
# =====================================================

print("\n4. Registrasi Perspective (4 titik)")

start = time.time()
M_est = cv2.getPerspectiveTransform(pts2, pts1)
reg_persp = cv2.warpPerspective(moving_img, M_est, (w, h))
time_persp = time.time() - start

mse_persp, psnr_persp = compute_metrics(ref_img, reg_persp)


# =====================================================
# 5. INTERPOLASI (Nearest, Bilinear, Bicubic)
# =====================================================

print("\n5. Perbandingan Interpolasi")

methods = [
    ("Nearest", cv2.INTER_NEAREST),
    ("Bilinear", cv2.INTER_LINEAR),
    ("Bicubic", cv2.INTER_CUBIC)
]

interp_results = []

for name, flag in methods:
    start = time.time()
    reg = cv2.warpPerspective(moving_img, M_est, (w,h), flags=flag)
    t = time.time() - start
    mse, psnr = compute_metrics(ref_img, reg)
    interp_results.append((name, mse, psnr, t, reg))


# =====================================================
# 6. VISUALISASI
# =====================================================

fig, ax = plt.subplots(3,4, figsize=(16,10))

# Baris 1
ax[0,0].imshow(ref_img, cmap='gray'); ax[0,0].set_title("Reference"); ax[0,0].axis("off")
ax[0,1].imshow(moving_img, cmap='gray'); ax[0,1].set_title("Moving"); ax[0,1].axis("off")
ax[0,2].imshow(trans_img, cmap='gray'); ax[0,2].set_title("Translasi"); ax[0,2].axis("off")
ax[0,3].imshow(rot_img, cmap='gray'); ax[0,3].set_title("Rotasi"); ax[0,3].axis("off")

# Baris 2
ax[1,0].imshow(scale_img, cmap='gray'); ax[1,0].set_title("Scaling"); ax[1,0].axis("off")
ax[1,1].imshow(reg_aff, cmap='gray'); ax[1,1].set_title("Affine"); ax[1,1].axis("off")
ax[1,2].imshow(reg_persp, cmap='gray'); ax[1,2].set_title("Perspective"); ax[1,2].axis("off")
ax[1,3].axis("off")

# Baris 3 (Interpolasi)
for i,(name,mse,psnr,t,img) in enumerate(interp_results):
    ax[2,i].imshow(img, cmap='gray')
    ax[2,i].set_title(f"{name}\nPSNR={psnr:.2f}")
    ax[2,i].axis("off")

ax[2,3].axis("off")

plt.tight_layout()
plt.show()


# =====================================================
# 7. EVALUASI NUMERIK
# =====================================================

print("\n=== HASIL EVALUASI ===")

print("\nAffine:")
print(f"MSE  : {mse_aff:.2f}")
print(f"PSNR : {psnr_aff:.2f} dB")
print(f"Waktu: {time_aff:.6f} detik")

print("\nPerspective:")
print(f"MSE  : {mse_persp:.2f}")
print(f"PSNR : {psnr_persp:.2f} dB")
print(f"Waktu: {time_persp:.6f} detik")

print("\nInterpolasi:")
for name,mse,psnr,t,_ in interp_results:
    print(f"{name} → MSE={mse:.2f}, PSNR={psnr:.2f} dB, Waktu={t:.6f} detik")