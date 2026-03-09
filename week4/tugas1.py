import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Load Images
# ===============================

under = cv2.imread(r"C:/Users/manta/Downloads/Documents/Document 2026/Pengolahan Citra Digital - PCD/Foto/underexposed.jpg", 0)
over = cv2.imread(r"C:/Users/manta/Downloads/Documents/Document 2026/Pengolahan Citra Digital - PCD/Foto/overexposed.png", 0)
uneven = cv2.imread(r"C:/Users/manta/Downloads/Documents/Document 2026/Pengolahan Citra Digital - PCD/Foto/silau.jpg", 0)
images = {
    "Underexposed": under,
    "Overexposed": over,
    "Uneven": uneven
}

# ===============================
# Point Processing
# ===============================

def negative(img):
    return 255 - img


def log_transform(img):
    c = 255 / np.log(1 + np.max(img))
    log_img = c * np.log(1 + img)
    return np.array(log_img, dtype=np.uint8)


def gamma_transform(img, gamma):
    table = np.array([(i / 255.0) ** gamma * 255
        for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

# ===============================
# Contrast Stretching
# ===============================

def contrast_stretch_manual(img):
    rmin = np.min(img)
    rmax = np.max(img)

    stretched = (img - rmin) / (rmax - rmin) * 255
    return stretched.astype(np.uint8)


def contrast_stretch_auto(img):
    p2, p98 = np.percentile(img, (2, 98))
    stretched = np.clip((img - p2) * 255 / (p98 - p2), 0, 255)
    return stretched.astype(np.uint8)

# ===============================
# Histogram Equalization
# ===============================

def hist_equalization(img):
    return cv2.equalizeHist(img)

# ===============================
# CLAHE
# ===============================

def clahe_enhancement(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

# ===============================
# Metrics
# ===============================

def contrast_ratio(img):
    Imax = np.max(img)
    Imin = np.min(img)
    return (Imax - Imin) / (Imax + Imin + 1e-5)


def entropy(img):
    hist = cv2.calcHist([img], [0], None, [256], [0,256])
    hist = hist / hist.sum()

    hist = hist.flatten()   # penting

    ent = 0
    for p in hist:
        if p > 0:
            ent += -p * np.log2(p)

    return ent

# ===============================
# Histogram Function
# ===============================

def show_histogram(img, title):
    plt.figure()
    plt.hist(img.ravel(), bins=256, range=[0,256])
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.show()

# ===============================
# Processing Pipeline
# ===============================

for name, img in images.items():

    if img is None:
        print(name, "image not found!")
        continue

    print("\n==========================")
    print("Processing:", name)
    print("==========================")

    neg = negative(img)
    log_img = log_transform(img)

    gamma05 = gamma_transform(img, 0.5)
    gamma1 = gamma_transform(img, 1)
    gamma2 = gamma_transform(img, 2)

    stretch_manual = contrast_stretch_manual(img)
    stretch_auto = contrast_stretch_auto(img)

    hist_eq = hist_equalization(img)
    clahe = clahe_enhancement(img)

    results = {
        "Original": img,
        "Negative": neg,
        "Log": log_img,
        "Gamma 0.5": gamma05,
        "Gamma 1": gamma1,
        "Gamma 2": gamma2,
        "Stretch Manual": stretch_manual,
        "Stretch Auto": stretch_auto,
        "Histogram EQ": hist_eq,
        "CLAHE": clahe
    }

    # ===============================
    # Show Images
    # ===============================

    plt.figure(figsize=(14,8))

    for i,(k,v) in enumerate(results.items()):
        plt.subplot(3,4,i+1)
        plt.imshow(v, cmap='gray')
        plt.title(k)
        plt.axis("off")

    plt.suptitle(name)
    plt.show()

    # ===============================
    # Histogram Before / After
    # ===============================

    show_histogram(img, name + " Histogram Before")
    show_histogram(hist_eq, name + " Histogram After Histogram Equalization")

    # ===============================
    # Metrics
    # ===============================   

    print("\nMetrics:")

    for k,v in results.items():
        print(
            k,
            "| Contrast:", round(contrast_ratio(v),3),
            "| Entropy:", round(entropy(v),3)
        )
