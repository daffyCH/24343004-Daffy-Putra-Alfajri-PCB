# ============================================================
# EVALUASI SPATIAL FILTERING UNTUK RESTORASI CITRA BISING
# ============================================================

# -----------------------------
# Import Library
# -----------------------------
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

np.random.seed(42)


# ------------------------------------------------------------
# Fungsi Menghitung Metrik Evaluasi
# ------------------------------------------------------------
def calculate_metrics(img_original, img_restored):

    # Mean Squared Error
    mse = np.mean((img_original.astype(np.float32) -
                   img_restored.astype(np.float32)) ** 2)

    # PSNR
    if mse == 0:
        psnr = 100
    else:
        psnr = 10 * np.log10(255**2 / mse)

    # Simplified SSIM
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img_original.astype(np.float64)
    img2 = img_restored.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1, (11,11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11,11), 1.5)

    sigma1_sq = cv2.GaussianBlur(img1**2, (11,11), 1.5) - mu1**2
    sigma2_sq = cv2.GaussianBlur(img2**2, (11,11), 1.5) - mu2**2
    sigma12 = cv2.GaussianBlur(img1*img2, (11,11), 1.5) - mu1*mu2

    ssim = np.mean(
        ((2*mu1*mu2 + C1)*(2*sigma12 + C2)) /
        ((mu1**2 + mu2**2 + C1)*(sigma1_sq + sigma2_sq + C2))
    )

    return mse, psnr, ssim


# ------------------------------------------------------------
# 1. Load Citra Asli
# ------------------------------------------------------------
img_path = 'C:/Users/Izukiyama/Pictures/Camera Roll/WIN_20260316_23_23_22_Pro.jpg'
original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Jika gambar tidak ditemukan
if original is None:
    print("File tidak ditemukan, menggunakan citra contoh")
    original = np.zeros((400,400), dtype=np.uint8)

    cv2.rectangle(original,(50,50),(150,150),200,-1)
    cv2.circle(original,(300,100),60,255,-1)
    cv2.putText(original,"TES",(100,350),
                cv2.FONT_HERSHEY_SIMPLEX,3,255,5)


# ------------------------------------------------------------
# 2. Membuat Noise
# ------------------------------------------------------------

# Gaussian Noise
gauss = np.random.normal(0,25, original.shape)
noisy_gauss = np.clip(original + gauss,0,255).astype(np.uint8)

# Salt Pepper Noise
noisy_sp = original.copy()
prob = 0.05

mask = np.random.rand(*original.shape)

noisy_sp[mask < prob/2] = 0
noisy_sp[mask > 1-prob/2] = 255

# Speckle Noise
speckle = np.random.randn(*original.shape)

noisy_speckle = np.clip(
    original + original * speckle * 0.2,
    0,255
).astype(np.uint8)


noises = [
    ("Gaussian", noisy_gauss),
    ("SaltPepper", noisy_sp),
    ("Speckle", noisy_speckle)
]


# ------------------------------------------------------------
# 3. Definisi Filter
# ------------------------------------------------------------
filters = [

    # Linear filter
    ("Mean 3x3", lambda img: cv2.blur(img,(3,3))),
    ("Mean 7x7", lambda img: cv2.blur(img,(7,7))),

    ("Gaussian s1", lambda img: cv2.GaussianBlur(img,(5,5),1)),
    ("Gaussian s2", lambda img: cv2.GaussianBlur(img,(7,7),2)),

    # Non linear
    ("Median 3x3", lambda img: cv2.medianBlur(img,3)),
    ("Median 5x5", lambda img: cv2.medianBlur(img,5)),

    ("Min Filter", lambda img: cv2.erode(img,np.ones((3,3),np.uint8)))
]


# ------------------------------------------------------------
# 4. Evaluasi Filter
# ------------------------------------------------------------

print(f"{'Noise':<12} {'Filter':<15} {'MSE':<10} {'PSNR':<10} {'SSIM':<10} {'Time(ms)'}")
print("-"*70)

results = []

for noise_name, noisy_img in noises:

    for filter_name, filter_func in filters:

        start = time.time()

        restored = filter_func(noisy_img)

        elapsed = (time.time() - start) * 1000

        mse, psnr, ssim = calculate_metrics(original, restored)

        results.append((noise_name, filter_name, mse, psnr, ssim, elapsed))

        print(f"{noise_name:<12} {filter_name:<15} {mse:<10.2f} {psnr:<10.2f} {ssim:<10.4f} {elapsed:.2f}")


# ------------------------------------------------------------
# 5. Visualisasi
# ------------------------------------------------------------

fig, axes = plt.subplots(3,5, figsize=(18,12))

for i,(name, noisy_img) in enumerate(noises):

    axes[i,0].imshow(original, cmap='gray')
    axes[i,0].set_title("Original")
    axes[i,0].axis("off")

    axes[i,1].imshow(noisy_img, cmap='gray')
    axes[i,1].set_title("Noisy "+name)
    axes[i,1].axis("off")

    # contoh 3 filter untuk visual
    demo_filters = [filters[0], filters[2], filters[4]]

    for j,(f_name,f_func) in enumerate(demo_filters):

        restored = f_func(noisy_img)

        mse, psnr, ssim = calculate_metrics(original, restored)

        axes[i,j+2].imshow(restored, cmap='gray')

        axes[i,j+2].set_title(
            f"{f_name}\nPSNR:{psnr:.2f}\nSSIM:{ssim:.3f}"
        )

        axes[i,j+2].axis("off")

plt.tight_layout()
plt.show()