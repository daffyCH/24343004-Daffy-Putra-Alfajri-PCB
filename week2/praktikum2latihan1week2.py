# ==========================================================
# Praktikum 2 - Analisis Model Warna untuk Aplikasi Spesifik
# ==========================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def analyze_color_model_suitability(image, application):

    if image is None:
        raise ValueError("Input image tidak boleh None")

    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    H, S, V = cv2.split(hsv_img)
    L, A, B = cv2.split(lab_img)

    results = {}

    if application == 'skin_detection':

        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        hsv_mask = cv2.inRange(hsv_img, lower_skin, upper_skin)
        hsv_ratio = np.sum(hsv_mask > 0) / hsv_mask.size

        b, g, r = cv2.split(image)
        rgb_mask = (r > 95) & (g > 40) & (b > 20)
        rgb_ratio = np.sum(rgb_mask) / rgb_mask.size

        results['HSV_skin_ratio'] = float(hsv_ratio)
        results['RGB_skin_ratio'] = float(rgb_ratio)

        best_model = 'HSV' if hsv_ratio > rgb_ratio else 'RGB'

    elif application == 'shadow_removal':

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        L_enhanced = clahe.apply(L)

        contrast_lab = np.std(L_enhanced)
        gray_eq = cv2.equalizeHist(gray_img)
        contrast_gray = np.std(gray_eq)

        results['LAB_contrast'] = float(contrast_lab)
        results['GRAY_contrast'] = float(contrast_gray)

        best_model = 'LAB' if contrast_lab > contrast_gray else 'GRAY'

    elif application == 'text_extraction':

        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        gray_edge = np.mean(np.sqrt(sobelx**2 + sobely**2))

        sobel_v = cv2.Sobel(V, cv2.CV_64F, 1, 1, ksize=3)
        hsv_edge = np.mean(np.abs(sobel_v))

        results['GRAY_edge'] = float(gray_edge)
        results['HSV_edge'] = float(hsv_edge)

        best_model = 'GRAY' if gray_edge > hsv_edge else 'HSV'

    elif application == 'object_detection':

        rgb_var = np.mean([np.var(ch) for ch in cv2.split(image)])
        hsv_var = np.mean([np.var(ch) for ch in [H, S, V]])
        lab_var = np.mean([np.var(ch) for ch in [L, A, B]])

        results['RGB_variance'] = float(rgb_var)
        results['HSV_variance'] = float(hsv_var)
        results['LAB_variance'] = float(lab_var)

        variances = {'RGB': rgb_var, 'HSV': hsv_var, 'LAB': lab_var}
        best_model = max(variances, key=variances.get)

    else:
        raise ValueError("Application tidak dikenali")

    return {
        "application": application,
        "best_model": best_model,
        "analysis": results
    }


# ===============================
# MAIN PROGRAM
# ===============================

if __name__ == "__main__":

    print("=== ANALISIS MODEL WARNA ===")

    img = cv2.imread("C:/Users/manta/Pictures/Camera Roll/WIN_20260212_11_52_15_Pro.jpg")

    if img is None:
        # buat citra sintetik jika file tidak ada
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)
        cv2.circle(img, (250, 100), 50, (0, 255, 0), -1)
        cv2.ellipse(img, (300, 200), (80, 40), 30, 0, 360, (0, 0, 255), -1)
        print("Menggunakan citra sintetik")

    applications = [
        'skin_detection',
        'shadow_removal',
        'text_extraction',
        'object_detection'
    ]

    for app in applications:
        result = analyze_color_model_suitability(img, app)
        print("\nAplikasi:", app)
        print("Best Model:", result["best_model"])
        print("Detail:", result["analysis"])
