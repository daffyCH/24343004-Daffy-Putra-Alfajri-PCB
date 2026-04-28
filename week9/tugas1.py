import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time

def tugas_segmentasi_citra():

    print("=" * 70)
    print("EVALUASI KOMPREHENSIF SEGMENTASI CITRA")
    print("=" * 70)

    # Load 3 jenis citra
    def load_images():
        base_path = os.path.dirname(__file__)

        path_bimodal = os.path.join(base_path, "bimodal.png")
        path_uneven  = os.path.join(base_path, "uneven.png")
        path_overlap = os.path.join(base_path, "overlapping.jpg")

        img_bimodal = cv2.imread(path_bimodal, 0)
        img_uneven  = cv2.imread(path_uneven, 0)
        img_overlap = cv2.imread(path_overlap, 0)

        if img_bimodal is None or img_uneven is None or img_overlap is None:
            raise ValueError("Pastikan path gambar benar!")

        return img_bimodal, img_uneven, img_overlap

    # Ground truth (pseudo, Otsu)
    def create_ground_truth(image):
        _, gt = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        gt = cv2.morphologyEx(gt, cv2.MORPH_CLOSE, kernel)
        gt = cv2.medianBlur(gt, 5)
        return gt

    def watershed_segmentation(img):
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.5*dist.max(), 255, 0)

        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(thresh, sure_fg)

        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        markers = cv2.watershed(img_color, markers)

        result = np.zeros_like(img)
        result[markers > 1] = 255
        return result

    def connected_components(img):
        _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        _, labels = cv2.connectedComponents(bw)
        return np.uint8(255 * labels / np.max(labels))

    # Thresholding
    def thresholding(img):
        res = {}
        t = {}

        start = time.time()
        _, res['Global'] = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        t['Global'] = time.time() - start

        start = time.time()
        _, res['Otsu'] = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t['Otsu'] = time.time() - start

        start = time.time()
        res['Mean'] = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY,11,2)
        t['Mean'] = time.time() - start

        start = time.time()
        res['Gaussian'] = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY,11,2)
        t['Gaussian'] = time.time() - start

        return res, t

    # Edge detection (magnitude + orientation)
    def edge_detection(img):
        res = {}
        t = {}

        start = time.time()
        sx = cv2.Sobel(img, cv2.CV_64F,1,0,ksize=3)
        sy = cv2.Sobel(img, cv2.CV_64F,0,1,ksize=3)

        mag = np.sqrt(sx**2 + sy**2)
        mag = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)

        ori = np.arctan2(sy, sx)

        res['Sobel Mag'] = mag
        res['Sobel Ori'] = ori
        t['Sobel'] = time.time() - start

        start = time.time()
        res['Prewitt'] = cv2.filter2D(img,-1,np.array([[1,0,-1],[1,0,-1],[1,0,-1]]))
        t['Prewitt'] = time.time() - start

        start = time.time()
        res['Canny Wide'] = cv2.Canny(img,10,150)
        res['Canny Tight'] = cv2.Canny(img,150,250)
        t['Canny'] = time.time() - start

        return res, t

    # Region growing (untuk overlapping)
    def region_growing(img):
        h, w = img.shape

        _, base = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        ys, xs = np.where(base == 255)
        if len(xs) == 0:
            return np.zeros_like(img)

        seed = (xs[len(xs)//2], ys[len(ys)//2])

        visited = np.zeros_like(img, dtype=bool)
        result = np.zeros_like(img)
        stack = [seed]

        threshold = 25

        while stack:
            x, y = stack.pop()

            if visited[y, x]:
                continue

            visited[y, x] = True

            if base[y, x] == 0:
                continue

            result[y, x] = 255

            for nx, ny in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                    if abs(int(img[ny, nx]) - int(img[y, x])) < threshold:
                        stack.append((nx, ny))

        return result

    # Metrics evaluasi
    def metrics(pred, gt):
        pred = (pred>0).astype(int)
        gt   = (gt>0).astype(int)

        tp = np.sum((pred==1)&(gt==1))
        fp = np.sum((pred==1)&(gt==0))
        fn = np.sum((pred==0)&(gt==1))
        tn = np.sum((pred==0)&(gt==0))

        acc = (tp+tn)/(tp+tn+fp+fn)
        prec = tp/(tp+fp+1e-6)
        rec = tp/(tp+fn+1e-6)
        dice = 2*tp/(2*tp+fp+fn+1e-6)
        iou = tp/(tp+fp+fn+1e-6)

        return acc, prec, rec, dice, iou

    # ================= MAIN =================
    bimodal, uneven, overlap = load_images()

    uneven_res, _ = thresholding(uneven)
    gt = create_ground_truth(bimodal)

    th_res, th_time = thresholding(bimodal)
    ed_res, ed_time = edge_detection(bimodal)

    rg = region_growing(overlap)
    ws = watershed_segmentation(overlap)
    cc = connected_components(overlap)

    print("\n--- TABEL METRIK ---")

    results_table = []

    # Thresholding methods
    for method in ['Global', 'Otsu', 'Mean', 'Gaussian']:
        acc, prec, rec, dice, iou = metrics(th_res[method], gt)
        results_table.append([
            method,
            acc, prec, rec, dice, iou,
            th_time[method]
        ])

    # Edge detection (tidak cocok → isi None / "-")
    for method in ['Sobel', 'Prewitt', 'Canny']:
        results_table.append([
            method,
            None, None, None, None, None,
            ed_time[method]
        ])

    # Print tabel rapi
    print(f"{'Method':<15} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'Dice':<8} {'IoU':<8} {'Time':<8}")
    for row in results_table:
        print(f"{row[0]:<15} "
            f"{'-' if row[1] is None else f'{row[1]:.3f}':<8} "
            f"{'-' if row[2] is None else f'{row[2]:.3f}':<8} "
            f"{'-' if row[3] is None else f'{row[3]:.3f}':<8} "
            f"{'-' if row[4] is None else f'{row[4]:.3f}':<8} "
            f"{'-' if row[5] is None else f'{row[5]:.3f}':<8} "
            f"{row[6]:.4f}")


    # Visualisasi utama
    fig, ax = plt.subplots(3,5, figsize=(20,12))

    ax[0,0].imshow(bimodal,cmap='gray'); ax[0,0].set_title("Original")
    ax[0,1].imshow(gt,cmap='gray'); ax[0,1].set_title("Ground Truth")
    ax[0,2].imshow(th_res['Otsu'],cmap='gray'); ax[0,2].set_title("Otsu")

    overlay = cv2.cvtColor(bimodal, cv2.COLOR_GRAY2BGR)
    contours,_ = cv2.findContours(th_res['Otsu'],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay,contours,-1,(0,255,0),2)
    ax[0,3].imshow(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB))
    ax[0,3].set_title("Overlay")

    ax[1,0].imshow(ed_res['Sobel Mag'], cmap='gray'); ax[1,0].set_title("Sobel Mag")
    ax[1,1].imshow(ed_res['Sobel Ori'], cmap='jet'); ax[1,1].set_title("Sobel Ori")
    ax[1,2].imshow(ed_res['Prewitt'], cmap='gray'); ax[1,2].set_title("Prewitt")
    ax[1,3].imshow(ed_res['Canny Wide'], cmap='gray'); ax[1,3].set_title("Canny")
    ax[1,4].imshow(rg, cmap='gray'); ax[1,4].set_title("Region Growing")

    ax[2,0].imshow(ws, cmap='gray'); ax[2,0].set_title("Watershed")
    ax[2,1].imshow(cc, cmap='jet'); ax[2,1].set_title("Connected Components")

    for a in ax.flat:
        a.axis('off')

    plt.show()

    # Robustness
    noise = np.random.normal(0,25,bimodal.shape)
    noisy = np.clip(bimodal + noise,0,255).astype(np.uint8)

    dark = (bimodal * 0.5).astype(np.uint8)
    bright = np.clip(bimodal * 1.5,0,255).astype(np.uint8)

    fig2, ax2 = plt.subplots(1,3, figsize=(12,4))
    ax2[0].imshow(noisy,cmap='gray'); ax2[0].set_title("Noise")
    ax2[1].imshow(dark,cmap='gray'); ax2[1].set_title("Dark")
    ax2[2].imshow(bright,cmap='gray'); ax2[2].set_title("Bright")

    for a in ax2:
        a.axis('off')

    plt.show()

    # Uneven image (adaptive threshold)
    fig3, ax3 = plt.subplots(1,4, figsize=(16,4))

    ax3[0].imshow(uneven, cmap='gray'); ax3[0].set_title("Original Uneven")
    ax3[1].imshow(uneven_res['Global'], cmap='gray'); ax3[1].set_title("Global")
    ax3[2].imshow(uneven_res['Mean'], cmap='gray'); ax3[2].set_title("Adaptive Mean")
    ax3[3].imshow(uneven_res['Gaussian'], cmap='gray'); ax3[3].set_title("Adaptive Gaussian")

    for a in ax3:
        a.axis('off')

    plt.show()

    print("\n--- WAKTU KOMPUTASI ---")
    print(th_time)
    print(ed_time)


if __name__ == "__main__":
    tugas_segmentasi_citra()