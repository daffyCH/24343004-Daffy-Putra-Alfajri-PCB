# =========================
# IMPORT
# =========================
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

# =========================
# LOAD IMAGE
# =========================
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Gambar tidak ditemukan")
    return img


# =========================
# PSF & DEGRADASI
# =========================
def motion_psf(length=15, angle=30):
    psf = np.zeros((length, length))
    center = length // 2
    angle_rad = np.deg2rad(angle)

    x1 = int(center - (length/2)*np.cos(angle_rad))
    y1 = int(center - (length/2)*np.sin(angle_rad))
    x2 = int(center + (length/2)*np.cos(angle_rad))
    y2 = int(center + (length/2)*np.sin(angle_rad))

    cv2.line(psf, (x1, y1), (x2, y2), 1, 1)
    psf /= np.sum(psf)

    return psf


def add_motion_blur(img, psf):
    return cv2.filter2D(img.astype(float), -1, psf)


def add_gaussian_noise(img, sigma=20):
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img + noise, 0, 255)


def add_salt_pepper(img, prob=0.05):
    noisy = img.copy()
    num = int(prob * img.size)

    coords = [np.random.randint(0, i, num) for i in img.shape]
    noisy[coords[0], coords[1]] = 255

    coords = [np.random.randint(0, i, num) for i in img.shape]
    noisy[coords[0], coords[1]] = 0

    return noisy

# =========================
# ESTIMASI PSF
# =========================
def estimate_psf(image):
    # Edge detection
    edges = cv2.Canny(image.astype(np.uint8), 50, 150)

    # Hough transform untuk deteksi arah
    lines = cv2.HoughLines(edges, 1, np.pi/180, 80)

    angles = []
    if lines is not None:
        for line in lines[:10]:
            rho, theta = line[0]
            angles.append(np.degrees(theta))

    # fallback kalau gagal
    angle = np.mean(angles) if angles else 30

    # estimasi panjang blur (sederhana, cepat)
    profile = np.mean(image, axis=0)
    length = np.sum(profile > 0.5*np.max(profile)) // 10

    return int(max(length, 5)), angle

# =========================
# RESTORASI
# =========================
def inverse_filter(img, psf, eps=1e-3):
    G = np.fft.fft2(img)
    H = np.fft.fft2(psf, s=img.shape)
    F = G / (H + eps)
    return np.abs(np.fft.ifft2(F))


def wiener_filter(img, psf, K=0.01):
    G = np.fft.fft2(img)
    H = np.fft.fft2(psf, s=img.shape)

    Hc = np.conj(H)
    F = (Hc / (np.abs(H)**2 + K)) * G

    return np.abs(np.fft.ifft2(F))


def richardson_lucy(img, psf, iters=5):  # ↓ iterasi dikurangi
    img = img.astype(float)
    est = img.copy()
    psf_flip = np.flip(psf)

    for i in range(iters):
        conv = cv2.filter2D(est, -1, psf)
        conv[conv == 0] = 1e-8
        ratio = img / conv
        est *= cv2.filter2D(ratio, -1, psf_flip)
        print(f"   RL iter {i+1}/{iters}")

    return np.clip(est, 0, 255)

# =========================
# SPEKTRUM
# =========================
def show_spectrum(img, title="Spectrum"):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(1 + np.abs(fshift))

    plt.imshow(magnitude, cmap='gray')
    plt.title(title)
    plt.axis('off')


# =========================
# METRIK
# =========================
def mse(a, b):
    return np.mean((a - b)**2)


def psnr(a, b):
    return 10 * np.log10(255**2 / mse(a, b))


def ssim(img1, img2):
    C1 = (0.01*255)**2
    C2 = (0.03*255)**2

    mu1 = cv2.GaussianBlur(img1, (11,11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11,11), 1.5)

    sigma1 = cv2.GaussianBlur(img1**2, (11,11), 1.5) - mu1**2
    sigma2 = cv2.GaussianBlur(img2**2, (11,11), 1.5) - mu2**2
    sigma12 = cv2.GaussianBlur(img1*img2, (11,11), 1.5) - mu1*mu2

    return np.mean((2*mu1*mu2 + C1)*(2*sigma12 + C2) /
                   ((mu1**2 + mu2**2 + C1)*(sigma1 + sigma2 + C2)))


# =========================
# MAIN
# =========================
def main():
    print("Loading image...")
    original = load_image("C:/Users/Izukiyama/Pictures/Camera Roll/WIN_20260316_23_23_22_Pro.jpg")

    # ↓ RESIZE (WAJIB untuk performa)
    original = cv2.resize(original, (512, 512))

    true_psf = motion_psf(15, 30)

    # === DEGRADASI ===
    print("Applying degradation...")
    blur = add_motion_blur(original, true_psf)
    gauss_blur = add_gaussian_noise(blur, 20)
    sp_blur = add_salt_pepper(blur, 0.05)

    datasets = {
        "Motion Blur": blur,
        "Gaussian + Blur": gauss_blur,
        "S&P + Blur": sp_blur
    }

    # === PROSES ===
    for name, degraded in datasets.items():
        print(f"\n=== {name} ===")

        print("Estimating PSF...")
        length, angle = estimate_psf(degraded)
        psf_est = motion_psf(length, angle)
        print(f"PSF: length={length}, angle={angle:.2f}")

        print("Start Inverse...")
        start = time.time()
        inv = inverse_filter(degraded, psf_est)
        t_inv = time.time() - start
        print("Done Inverse")

        print("Start Wiener...")
        noise_var = np.var(degraded - blur)
        signal_var = np.var(original)
        K = noise_var / signal_var

        start = time.time()
        wien = wiener_filter(degraded, psf_est)
        t_wien = time.time() - start
        print("Done Wiener")

        print("Start RL...")
        start = time.time()
        rl = richardson_lucy(degraded, psf_est, iters=5)
        t_rl = time.time() - start
        print("Done RL")

        methods = {
            "Inverse": (inv, t_inv),
            "Wiener": (wien, t_wien),
            "RL": (rl, t_rl)
        }

        for m, (img, t) in methods.items():
            print(f"{m} | MSE={mse(original,img):.2f} | "
                  f"PSNR={psnr(original,img):.2f} | "
                  f"SSIM={ssim(original,img):.3f} | "
                  f"Time={t:.4f}s")

        plt.figure(figsize=(14,8))

        # ======================
        # BARIS 1: CITRA
        # ======================
        plt.subplot(2,4,1); plt.imshow(original, cmap='gray'); plt.title("Original"); plt.axis('off')
        plt.subplot(2,4,2); plt.imshow(degraded, cmap='gray'); plt.title("Degraded"); plt.axis('off')
        plt.subplot(2,4,3); plt.imshow(wien, cmap='gray'); plt.title("Wiener"); plt.axis('off')
        plt.subplot(2,4,4); plt.imshow(rl, cmap='gray'); plt.title("RL"); plt.axis('off')

        # ======================
        # BARIS 2: SPEKTRUM
        # ======================
        plt.subplot(2,4,5); show_spectrum(original, "Spec Original")
        plt.subplot(2,4,6); show_spectrum(degraded, "Spec Degraded")
        plt.subplot(2,4,7); show_spectrum(wien, "Spec Wiener")
        plt.subplot(2,4,8); show_spectrum(rl, "Spec RL")

        plt.tight_layout()
        plt.show()
        plt.close()


if __name__ == "__main__":
    main()