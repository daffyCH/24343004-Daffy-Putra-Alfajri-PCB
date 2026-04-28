import cv2
import numpy as np
import matplotlib.pyplot as plt


def buat_citra():
    """Membuat citra biner dengan beberapa bentuk"""
    img = np.zeros((200, 300), dtype=np.uint8)

    # Bentuk-bentuk
    cv2.rectangle(img, (30, 30), (80, 80), 255, -1)      # Kotak
    cv2.circle(img, (150, 50), 20, 255, -1)              # Lingkaran
    cv2.rectangle(img, (200, 30), (220, 70), 255, -1)    # Garis vertikal
    cv2.rectangle(img, (250, 40), (270, 60), 255, -1)    # Garis horizontal

    return img


def tambah_noise(img):
    """Menambahkan noise salt & pepper"""
    noise = np.random.random(img.shape) < 0.05
    img_noisy = img.copy()
    img_noisy[noise] = 255 - img_noisy[noise]
    return img_noisy


def get_kernels():
    """Structuring elements"""
    return {
        '3x3 Rectangle': cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        '5x5 Rectangle': cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        '3x3 Ellipse': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        '3x3 Cross': cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    }


def proses_morfologi(img_noisy, kernels):
    """Melakukan operasi morfologi"""
    operations = ['Erosion', 'Dilation', 'Opening', 'Closing']
    hasil = {}

    for op in operations:
        hasil[op] = {}
        for kernel_name, kernel in kernels.items():
            if op == 'Erosion':
                result = cv2.erode(img_noisy, kernel)
            elif op == 'Dilation':
                result = cv2.dilate(img_noisy, kernel)
            elif op == 'Opening':
                result = cv2.morphologyEx(img_noisy, cv2.MORPH_OPEN, kernel)
            elif op == 'Closing':
                result = cv2.morphologyEx(img_noisy, cv2.MORPH_CLOSE, kernel)

            hasil[op][kernel_name] = result

    return hasil


def tampilkan(img, img_noisy, hasil):
    """Menampilkan semua hasil"""
    fig, axes = plt.subplots(5, 4, figsize=(12, 10))

    # Baris 1: Original
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_noisy, cmap='gray')
    axes[0, 1].set_title('Noisy')
    axes[0, 1].axis('off')

    for i in range(2, 4):
        axes[0, i].axis('off')

    operations = list(hasil.keys())

    # Baris berikutnya
    for row, op in enumerate(operations, 1):
        for col, (kernel_name, result) in enumerate(hasil[op].items()):
            axes[row, col].imshow(result, cmap='gray')
            axes[row, col].set_title(f'{op}\n{kernel_name}')
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


def analisis():
    """Menampilkan analisis"""
    print("\nANALISIS OPERASI MORFOLOGI")
    print("=" * 50)

    print("\n1. Erosion:")
    print("- Menghilangkan noise kecil")
    print("- Mengecilkan objek")

    print("\n2. Dilation:")
    print("- Membesarkan objek")
    print("- Menutup lubang kecil")

    print("\n3. Opening:")
    print("- Kombinasi Erosion + Dilation")
    print("- Efektif menghilangkan noise")

    print("\n4. Closing:")
    print("- Kombinasi Dilation + Erosion")
    print("- Mengisi lubang pada objek")


def main():
    try:
        img = buat_citra()
        img_noisy = tambah_noise(img)
        kernels = get_kernels()
        hasil = proses_morfologi(img_noisy, kernels)

        tampilkan(img, img_noisy, hasil)
        analisis()

    except Exception as e:
        print("Terjadi error:", e)


# Entry point
if __name__ == "__main__":
    main()
