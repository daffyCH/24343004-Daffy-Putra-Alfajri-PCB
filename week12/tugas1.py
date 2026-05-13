import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(BASE_DIR, "dataset")

kelas_objek = [
    "mouse",
    "mainan",
    "kipas",
    "gelas",
    "gamepad"
]

metode_list = ["SIFT", "ORB"]

def load_dataset():

    data = []

    for label in kelas_objek:

        folder = os.path.join(dataset_path, label)

        for file in os.listdir(folder):

            if file.endswith((".jpg", ".png", ".jpeg")):

                path = os.path.join(folder, file)

                img = cv2.imread(path)

                if img is not None:

                    if "Referensi" in file:
                        tipe = "referensi"
                    else:
                        tipe = "uji"

                    data.append({
                        "label": label,
                        "filename": file,
                        "image": img,
                        "path": path,
                        "type": tipe
                    })

    return data

def get_detector(method):

    if method == "SIFT":
        return cv2.SIFT_create()

    elif method == "ORB":
        return cv2.ORB_create(nfeatures=500)

def extract_features(detector, image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    start = time.time()

    kp, desc = detector.detectAndCompute(gray, None)

    end = time.time()

    waktu = end - start

    jumlah_kp = 0 if kp is None else len(kp)

    dimensi = 0 if desc is None else desc.shape[1]

    return kp, desc, waktu, jumlah_kp, dimensi

def brute_force_matching(desc1, desc2, method):

    if desc1 is None or desc2 is None:
        return [], []

    if method == "SIFT":
        bf = cv2.BFMatcher(cv2.NORM_L2)

    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.knnMatch(desc1, desc2, k=2)

    good = []

    for pair in matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good.append(m)

    return matches, good

def flann_matching(desc1, desc2, method):

    if desc1 is None or desc2 is None:
        return [], []

    if method == "SIFT":

        index_params = dict(
            algorithm=1,
            trees=5
        )

        search_params = dict(
            checks=50
        )

        flann = cv2.FlannBasedMatcher(
            index_params,
            search_params
        )

    else:

        index_params = dict(
            algorithm=6,
            table_number=6,
            key_size=12,
            multi_probe_level=1
        )

        search_params = dict()

        flann = cv2.FlannBasedMatcher(
            index_params,
            search_params
        )

        desc1 = np.uint8(desc1)
        desc2 = np.uint8(desc2)

    matches = flann.knnMatch(desc1, desc2, k=2)

    good = []

    for pair in matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good.append(m)

    return matches, good

def ransac_filter(kp1, kp2, good_matches):

    if len(good_matches) < 4:
        return None, [], []

    src_pts = np.float32([
        kp1[m.queryIdx].pt for m in good_matches
    ]).reshape(-1,1,2)

    dst_pts = np.float32([
        kp2[m.trainIdx].pt for m in good_matches
    ]).reshape(-1,1,2)

    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        5.0
    )

    if mask is None:
        return None, [], []

    matches_mask = mask.ravel().tolist()

    inlier = []
    outlier = []

    for i, m in enumerate(good_matches):

        if matches_mask[i] == 1:
            inlier.append(m)
        else:
            outlier.append(m)

    return H, inlier, outlier

def visualisasi_keypoints(data):

    fig, axes = plt.subplots(
        len(metode_list),
        len(kelas_objek),
        figsize=(24,10)
    )

    fig.suptitle(
        "Visualisasi Keypoints",
        fontsize=22,
        fontweight='bold'
    )

    for row, metode in enumerate(metode_list):

        detector = get_detector(metode)

        for col, label in enumerate(kelas_objek):

            sample = None

            for item in data:

                if (
                    item["label"] == label and
                    item["type"] == "referensi"
                ):
                    sample = item
                    break

            kp, desc, _, _, _ = extract_features(
                detector,
                sample["image"]
            )

            visual = cv2.drawKeypoints(
                sample["image"],
                kp,
                None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )

            axes[row, col].imshow(
                cv2.cvtColor(
                    visual,
                    cv2.COLOR_BGR2RGB
                )
            )

            axes[row, col].set_title(
                f"{metode} - {label}"
            )

            axes[row, col].axis("off")

    plt.tight_layout()

    plt.show()

def evaluasi_matching(data):

    hasil_precision = []

    hasil_recall = []

    tabel_hasil = []

    fig_inlier, axes_inlier = plt.subplots(
        len(metode_list),
        len(kelas_objek),
        figsize=(28,12)
    )

    fig_outlier, axes_outlier = plt.subplots(
        len(metode_list),
        len(kelas_objek),
        figsize=(28,12)
    )

    fig_inlier.suptitle(
        "Feature Matching Inlier",
        fontsize=22,
        fontweight='bold'
    )

    fig_outlier.suptitle(
        "Feature Matching Outlier",
        fontsize=22,
        fontweight='bold'
    )

    for row, metode in enumerate(metode_list):

        detector = get_detector(metode)

        for col, label in enumerate(kelas_objek):

            referensi = None

            data_uji = []

            for item in data:

                if item["label"] == label:

                    if item["type"] == "referensi":
                        referensi = item
                    else:
                        data_uji.append(item)

            img_ref = referensi["image"]

            kp1, desc1, _, _, _ = extract_features(
                detector,
                img_ref
            )

            best_inlier = []

            best_outlier = []

            best_kp2 = None

            best_img_uji = None

            for uji in data_uji:

                img_uji = uji["image"]

                kp2, desc2, _, _, _ = extract_features(
                    detector,
                    img_uji
                )

                start_bf = time.time()

                matches_bf, good_bf = brute_force_matching(
                    desc1,
                    desc2,
                    metode
                )

                end_bf = time.time()

                waktu_bf = end_bf - start_bf

                H_bf, inlier_bf, outlier_bf = ransac_filter(
                    kp1,
                    kp2,
                    good_bf
                )

                start_flann = time.time()

                matches_flann, good_flann = flann_matching(
                    desc1,
                    desc2,
                    metode
                )

                end_flann = time.time()

                waktu_flann = end_flann - start_flann

                H_flann, inlier_flann, outlier_flann = ransac_filter(
                    kp1,
                    kp2,
                    good_flann
                )

                precision = (
                    len(inlier_flann) /
                    len(good_flann)
                    if len(good_flann) > 0
                    else 0
                )

                recall = (
                    len(inlier_flann) /
                    (
                        len(inlier_flann) +
                        len(outlier_flann)
                    )
                    if (
                        len(inlier_flann) +
                        len(outlier_flann)
                    ) > 0
                    else 0
                )

                hasil_precision.append(
                    precision
                )

                hasil_recall.append(
                    recall
                )

                tabel_hasil.append([
                    metode,
                    label,
                    uji["filename"],
                    len(inlier_bf),
                    len(outlier_bf),
                    waktu_bf,
                    len(inlier_flann),
                    len(outlier_flann),
                    waktu_flann,
                    precision,
                    recall
                ])

                print(f"""
============================================================
[{metode}] {label}
File Uji : {uji['filename']}
============================================================

[BRUTE FORCE]
Total Match     : {len(matches_bf)}
Good Match      : {len(good_bf)}
Inlier          : {len(inlier_bf)}
Outlier         : {len(outlier_bf)}
Waktu Matching  : {waktu_bf:.6f} detik

[FLANN]
Total Match     : {len(matches_flann)}
Good Match      : {len(good_flann)}
Inlier          : {len(inlier_flann)}
Outlier         : {len(outlier_flann)}
Waktu Matching  : {waktu_flann:.6f} detik

Precision       : {precision:.4f}
Recall          : {recall:.4f}
                """)

                if len(inlier_flann) > len(best_inlier):

                    best_inlier = inlier_flann

                    best_outlier = outlier_flann

                    best_kp2 = kp2

                    best_img_uji = img_uji

            if len(best_inlier) > 0:

                visual_inlier = cv2.drawMatches(
                    img_ref,
                    kp1,
                    best_img_uji,
                    best_kp2,
                    best_inlier,
                    None,
                    matchColor=(0,255,0),
                    flags=2
                )

                axes_inlier[row, col].imshow(
                    cv2.cvtColor(
                        visual_inlier,
                        cv2.COLOR_BGR2RGB
                    )
                )

                axes_inlier[row, col].set_title(
                    f"{metode} - {label}"
                )

                axes_inlier[row, col].axis("off")

            if len(best_outlier) > 0:

                visual_outlier = cv2.drawMatches(
                    img_ref,
                    kp1,
                    best_img_uji,
                    best_kp2,
                    best_outlier,
                    None,
                    matchColor=(255,0,0),
                    flags=2
                )

                axes_outlier[row, col].imshow(
                    cv2.cvtColor(
                        visual_outlier,
                        cv2.COLOR_BGR2RGB
                    )
                )

                axes_outlier[row, col].set_title(
                    f"{metode} - {label}"
                )

                axes_outlier[row, col].axis("off")

    plt.tight_layout()

    plt.show()

    return (
        hasil_precision,
        hasil_recall,
        tabel_hasil
    )

def precision_recall_visual(
    precision_data,
    recall_data
):

    precision_data = np.array(
        precision_data
    )

    recall_data = np.array(
        recall_data
    )

    sorted_index = np.argsort(
        recall_data
    )

    recall_sorted = recall_data[
        sorted_index
    ]

    precision_sorted = precision_data[
        sorted_index
    ]

    plt.figure(figsize=(8,6))

    plt.plot(
        recall_sorted,
        precision_sorted,
        marker='o'
    )

    plt.fill_between(
        recall_sorted,
        precision_sorted,
        alpha=0.3
    )

    plt.title(
        "Precision Recall Curve"
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.xlim([0,1])

    plt.ylim([0,1])

    plt.grid(True)

    plt.show()

def build_vocabulary(
    descriptor_list,
    k
):

    all_descriptors = np.vstack(
        descriptor_list
    )

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        batch_size=k*20
    )

    kmeans.fit(
        all_descriptors
    )

    return kmeans

def build_histogram(
    descriptors,
    kmeans
):

    histogram = np.zeros(
        len(kmeans.cluster_centers_)
    )

    if descriptors is not None:

        words = kmeans.predict(
            descriptors
        )

        for w in words:
            histogram[w] += 1

    return histogram

def evaluasi_bovw(data):

    detector = cv2.SIFT_create()

    descriptor_list = []

    for item in data:

        gray = cv2.cvtColor(
            item["image"],
            cv2.COLOR_BGR2GRAY
        )

        kp, desc = detector.detectAndCompute(
            gray,
            None
        )

        if desc is not None:
            descriptor_list.append(desc)

    hasil_akurasi = []

    for k in [10,20,50,100]:

        print("\n")
        print("="*60)
        print(f"VOCABULARY SIZE : {k}")
        print("="*60)

        kmeans = build_vocabulary(
            descriptor_list,
            k
        )

        histograms = []

        labels = []

        for item in data:

            gray = cv2.cvtColor(
                item["image"],
                cv2.COLOR_BGR2GRAY
            )

            kp, desc = detector.detectAndCompute(
                gray,
                None
            )

            hist = build_histogram(
                desc,
                kmeans
            )

            histograms.append(
                hist
            )

            labels.append(
                item["label"]
            )

        X = np.array(histograms)

        y = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42
        )

        svm = SVC(
            kernel='linear'
        )

        svm.fit(
            X_train,
            y_train
        )

        pred_svm = svm.predict(
            X_test
        )

        acc_svm = accuracy_score(
            y_test,
            pred_svm
        )

        knn = KNeighborsClassifier(
            n_neighbors=3
        )

        knn.fit(
            X_train,
            y_train
        )

        pred_knn = knn.predict(
            X_test
        )

        acc_knn = accuracy_score(
            y_test,
            pred_knn
        )

        hasil_akurasi.append(
            acc_svm
        )

        print(f"""
Akurasi SVM : {acc_svm:.4f}
Akurasi kNN : {acc_knn:.4f}
        """)

        cm = confusion_matrix(
            y_test,
            pred_svm
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=svm.classes_
        )

        disp.plot()

        plt.title(
            f"Confusion Matrix k={k}"
        )

        plt.show()

    plt.figure(figsize=(8,6))

    plt.plot(
        [10,20,50,100],
        hasil_akurasi,
        marker='o'
    )

    plt.title(
        "Pengaruh Vocabulary Size"
    )

    plt.xlabel(
        "Jumlah Vocabulary"
    )

    plt.ylabel(
        "Akurasi"
    )

    plt.grid(True)

    plt.show()

def evaluasi_pca(data):

    detector = cv2.SIFT_create()

    all_desc = []

    for item in data:

        gray = cv2.cvtColor(
            item["image"],
            cv2.COLOR_BGR2GRAY
        )

        kp, desc = detector.detectAndCompute(
            gray,
            None
        )

        if desc is not None:

            all_desc.append(desc)

    all_desc = np.vstack(
        all_desc
    )

    hasil_variance = []

    komponen_list = [
        16,
        32,
        64,
        128
    ]

    print("\n")
    print("="*60)
    print("PCA")
    print("="*60)

    for komponen in komponen_list:

        if komponen > all_desc.shape[1]:
            continue

        pca = PCA(
            n_components=komponen
        )

        reduced = pca.fit_transform(
            all_desc
        )

        variance = np.sum(
            pca.explained_variance_ratio_
        )

        hasil_variance.append(
            variance
        )

        print(f"""
Komponen PCA      : {komponen}
Dimensi Baru      : {reduced.shape}
ExplainedVariance : {variance:.4f}
        """)

    plt.figure(figsize=(8,6))

    plt.plot(
        komponen_list,
        hasil_variance,
        marker='o'
    )

    plt.title(
        "PCA Components vs Explained Variance"
    )

    plt.xlabel(
        "Jumlah Komponen"
    )

    plt.ylabel(
        "Explained Variance"
    )

    plt.grid(True)

    plt.show()

def tabel_perbandingan(
    tabel_hasil
):

    print("\n")
    print("="*140)
    print("TABEL KOMPARASI AKHIR")
    print("="*140)

    header = (
        f"{'Metode':10}"
        f"{'Objek':12}"
        f"{'BF_In':10}"
        f"{'BF_Out':10}"
        f"{'BF_Time':14}"
        f"{'FLANN_In':12}"
        f"{'FLANN_Out':12}"
        f"{'FLANN_Time':16}"
        f"{'Precision':12}"
        f"{'Recall':10}"
    )

    print(header)

    print("-"*140)

    for row in tabel_hasil:

        print(
            f"{row[0]:10}"
            f"{row[1]:12}"
            f"{row[3]:<10}"
            f"{row[4]:<10}"
            f"{row[5]:<14.6f}"
            f"{row[6]:<12}"
            f"{row[7]:<12}"
            f"{row[8]:<16.6f}"
            f"{row[9]:<12.4f}"
            f"{row[10]:<10.4f}"
        )

print("="*60)
print("SISTEM PENCOCOKAN OBJEK BERBASIS FITUR LOKAL")
print("="*60)

data = load_dataset()

print(f"Jumlah Data : {len(data)}")

print("\n")
print("="*60)
print("EKSTRAKSI FITUR")
print("="*60)

for metode in metode_list:

    detector = get_detector(
        metode
    )

    print("\n")
    print(f"METODE : {metode}")

    for item in data:

        kp, desc, waktu, jumlah_kp, dimensi = extract_features(
            detector,
            item["image"]
        )

        print(f"""
Objek              : {item['label']}
File               : {item['filename']}
Jumlah Keypoints   : {jumlah_kp}
Dimensi Descriptor : {dimensi}
Waktu Ekstraksi    : {waktu:.4f} detik
        """)

visualisasi_keypoints(
    data
)

precision_data, recall_data, tabel_hasil = evaluasi_matching(
    data
)

precision_recall_visual(
    precision_data,
    recall_data
)

tabel_perbandingan(
    tabel_hasil
)

evaluasi_bovw(
    data
)

evaluasi_pca(
    data
)

print("\nPROGRAM SELESAI")