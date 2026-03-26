import numpy as np
import cv2
import matplotlib.pyplot as plt
import pywt
import time
from scipy.fft import fft2, fftshift, ifft2, ifftshift

# =========================
# METRIC
# =========================
def calculate_metrics(img_original, img_restored):
    mse = np.mean((img_original.astype(np.float32) - img_restored.astype(np.float32))**2)
    if mse == 0:
        return 0, 100
    psnr = 10 * np.log10((255**2) / mse)
    return mse, psnr

# =========================
# LOAD IMAGE
# =========================
def load_images(path=None):
    if path is None:
        path = 'C:/Users/Izukiyama/Pictures/Camera Roll/WIN_20260316_23_23_22_Pro.jpg'
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (256, 256))

    rows, cols = img.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

    # noise periodik
    noise = 30 * np.sin(2*np.pi*X/20)
    img_noise = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img, img_noise

# =========================
# FFT
# =========================
def compute_fft(image):
    f = fft2(image)
    fshift = fftshift(f)
    magnitude = np.abs(fshift)
    phase = np.angle(fshift)
    log_magnitude = np.log(1 + magnitude)
    return fshift, magnitude, phase, log_magnitude

def reconstruct_full(magnitude, phase):
    complex_img = magnitude * np.exp(1j * phase)
    img_back = np.abs(ifft2(ifftshift(complex_img)))
    return np.clip(img_back, 0, 255).astype(np.uint8)

def reconstruct_magnitude_only(magnitude):
    phase = np.zeros_like(magnitude)
    return reconstruct_full(magnitude, phase)

def reconstruct_phase_only(phase):
    magnitude = np.ones_like(phase)
    return reconstruct_full(magnitude, phase)

def find_dominant_freq(magnitude, threshold_ratio=0.6):
    thresh = magnitude.max() * threshold_ratio
    coords = np.where(magnitude > thresh)
    return list(zip(coords[0], coords[1]))

# =========================
# FILTERING
# =========================
def gaussian_lowpass(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows//2, cols//2

    y, x = np.ogrid[:rows, :cols]
    distance = (y - crow)**2 + (x - ccol)**2
    mask = np.exp(-distance/(2*(cutoff**2)))

    return mask

def gaussian_highpass(shape, cutoff):
    return 1 - gaussian_lowpass(shape, cutoff)

def ideal_lowpass(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows//2, cols//2

    y, x = np.ogrid[:rows, :cols]
    mask = ((y-crow)**2 + (x-ccol)**2 <= cutoff**2).astype(np.float32)
    return mask

def notch_filter(shape, centers, radius=5):
    rows, cols = shape
    mask = np.ones((rows, cols), np.float32)

    for (u, v) in centers:
        cv2.circle(mask, (v, u), radius, 0, -1)
        cv2.circle(mask, (cols-v, rows-u), radius, 0, -1)

    return mask

def apply_filter(image, mask):
    fshift, _, _, _ = compute_fft(image)
    filtered = fshift * mask
    img_back = np.abs(ifft2(ifftshift(filtered)))
    return np.clip(img_back, 0, 255).astype(np.uint8)

# =========================
# SPATIAL FILTER
# =========================
def spatial_filter(image):
    gauss = cv2.GaussianBlur(image, (5,5), 0)
    median = cv2.medianBlur(image, 5)
    return gauss, median

# =========================
# WAVELET
# =========================
def wavelet_decomposition(image):
    coeffs = pywt.wavedec2(image, 'db4', level=2)
    return coeffs

def visualize_wavelet(coeffs):
    cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs

    plt.figure(figsize=(10,6))
    plt.subplot(2,4,1); plt.imshow(cA2, cmap='gray'); plt.title("cA2")
    plt.subplot(2,4,2); plt.imshow(cH2, cmap='gray'); plt.title("cH2")
    plt.subplot(2,4,3); plt.imshow(cV2, cmap='gray'); plt.title("cV2")
    plt.subplot(2,4,4); plt.imshow(cD2, cmap='gray'); plt.title("cD2")

    plt.subplot(2,4,5); plt.imshow(cH1, cmap='gray'); plt.title("cH1")
    plt.subplot(2,4,6); plt.imshow(cV1, cmap='gray'); plt.title("cV1")
    plt.subplot(2,4,7); plt.imshow(cD1, cmap='gray'); plt.title("cD1")

    plt.tight_layout()
    plt.show()

def wavelet_reconstruction(coeffs):
    return np.clip(pywt.waverec2(coeffs, 'db4'), 0, 255).astype(np.uint8)

# =========================
# MAIN
# =========================
def main():
    img, img_noise = load_images()

    # FFT
    fshift, mag, phase, log_mag = compute_fft(img_noise)

    # Rekonstruksi
    img_full = reconstruct_full(mag, phase)
    img_mag_only = reconstruct_magnitude_only(mag)
    img_phase_only = reconstruct_phase_only(phase)

    # Dominant frequency
    dom = find_dominant_freq(mag)
    print("Dominant freq:", dom[:5])

    # Filtering
    cutoffs = [10, 30, 60]
    print("\n=== Evaluasi Cutoff ===")
    for c in cutoffs:
        start = time.time()
        filtered = apply_filter(img, gaussian_lowpass(img.shape, c))
        t = time.time() - start
        mse, psnr = calculate_metrics(img, filtered)
        print(f"Cutoff={c} | PSNR={psnr:.2f} | Time={t:.4f}s")

    # Spatial vs Frequency
    gauss_s, med_s = spatial_filter(img)
    gauss_f = apply_filter(img, gaussian_lowpass(img.shape, 30))

    print("\n=== Perbandingan ===")
    for name, im in [("Spatial Gaussian", gauss_s),
                     ("Median", med_s),
                     ("Freq Gaussian", gauss_f)]:
        _, psnr = calculate_metrics(img, im)
        print(f"{name}: PSNR={psnr:.2f}")

    # Notch filter
    notch = notch_filter(img_noise.shape, [(128,100),(128,156)])
    img_notch = apply_filter(img_noise, notch)

    # Wavelet
    coeffs = wavelet_decomposition(img)

    cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs

    # Hilangkan detail (denoising)
    coeffs_smooth = [
        cA2,
        (np.zeros_like(cH2), np.zeros_like(cV2), np.zeros_like(cD2)),
        (np.zeros_like(cH1), np.zeros_like(cV1), np.zeros_like(cD1))
    ]

    img_wavelet = wavelet_reconstruction(coeffs_smooth)

    # ================= VISUAL =================
    plt.figure(figsize=(16,10))

    plt.subplot(2,4,1); plt.imshow(img, cmap='gray'); plt.title("Original")
    plt.subplot(2,4,2); plt.imshow(img_noise, cmap='gray'); plt.title("Noise")
    plt.subplot(2,4,3); plt.imshow(log_mag, cmap='gray'); plt.title("Spectrum")
    plt.subplot(2,4,4); plt.imshow(img_notch, cmap='gray'); plt.title("Notch")

    plt.subplot(2,4,5); plt.imshow(img_full, cmap='gray'); plt.title("Reconstruct Full")
    plt.subplot(2,4,6); plt.imshow(img_mag_only, cmap='gray'); plt.title("Magnitude Only")
    plt.subplot(2,4,7); plt.imshow(img_phase_only, cmap='gray'); plt.title("Phase Only")
    plt.subplot(2,4,8); plt.imshow(img_wavelet, cmap='gray'); plt.title("Wavelet")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()