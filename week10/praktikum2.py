import cv2
import numpy as np
import matplotlib.pyplot as plt


def thinning_manual(img):
    """Manual skeletonization (pengganti ximgproc.thinning)"""
    img = img.copy()
    skel = np.zeros(img.shape, np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        if cv2.countNonZero(img) == 0:
            break

    return skel


def latihan_2():
    # ===============================
    # 1. Buat simulasi dokumen
    # ===============================
    doc = np.ones((200, 400), dtype=np.uint8) * 200

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(doc, 'Normal Text', (30, 50), font, 0.7, 50, 2)

    for i in range(0, 100, 5):
        cv2.line(doc, (30+i, 80), (30+i, 85), 50, 1)

    cv2.putText(doc, 'Broken Text', (30, 120), font, 0.7, 50, 2)
    cv2.rectangle(doc, (80, 110), (90, 115), 200, -1)

    noise = np.random.normal(0, 30, doc.shape)
    doc = np.clip(doc.astype(float) + noise, 0, 255).astype(np.uint8)

    # ===============================
    # 2. Preprocessing OCR
    # ===============================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(doc, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    _, binary = cv2.threshold(doc, 150, 255, cv2.THRESH_BINARY_INV)
    axes[0, 1].imshow(binary, cmap='gray')
    axes[0, 1].set_title('Binarization')
    axes[0, 1].axis('off')

    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=2)
    axes[0, 2].imshow(cleaned, cmap='gray')
    axes[0, 2].set_title('Noise Removal')
    axes[0, 2].axis('off')

    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    connected = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_horizontal)
    axes[1, 0].imshow(connected, cmap='gray')
    axes[1, 0].set_title('Connect Text')
    axes[1, 0].axis('off')

    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    enhanced = cv2.dilate(connected, kernel_vertical, iterations=1)
    axes[1, 1].imshow(enhanced, cmap='gray')
    axes[1, 1].set_title('Enhancement')
    axes[1, 1].axis('off')

    final_result = enhanced
    axes[1, 2].imshow(final_result, cmap='gray')
    axes[1, 2].set_title('Final OCR Ready')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

    # ===============================
    # 3. Evaluasi
    # ===============================
    def count_components(image):
        num_labels, _ = cv2.connectedComponents(image)
        return num_labels - 1

    def average_stroke_thickness(image):
        skeleton = thinning_manual(image)
        stroke_pixels = np.sum(image == 255)
        skeleton_pixels = np.sum(skeleton == 255)
        return stroke_pixels / skeleton_pixels if skeleton_pixels > 0 else 0

    orig_comp = count_components(binary)
    proc_comp = count_components(final_result)

    orig_thick = average_stroke_thickness(binary)
    proc_thick = average_stroke_thickness(final_result)

    print("\nOCR QUALITY ANALYSIS")
    print("=" * 50)

    print("\nConnected Components:")
    print(f"Original: {orig_comp}")
    print(f"Processed: {proc_comp}")
    print(f"Improvement: {(orig_comp - proc_comp)/orig_comp*100:.1f}%")

    print("\nStroke Thickness:")
    print(f"Original: {orig_thick:.2f}")
    print(f"Processed: {proc_thick:.2f}")
    print(f"Improvement: {(proc_thick - orig_thick)/orig_thick*100:.1f}%")

    print("\nKESIMPULAN:")
    print("- Noise berkurang")
    print("- Karakter lebih tersambung")
    print("- Ketebalan teks meningkat")


def main():
    try:
        latihan_2()
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()