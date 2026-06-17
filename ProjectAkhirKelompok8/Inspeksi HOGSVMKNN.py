import os
import cv2
import joblib
import random
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ======================================================
# CONFIG
# ======================================================

IMAGE_SIZE = (128, 128)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset_hog"
)

# ======================================================
# LOAD DATASET
# ======================================================

print("=" * 50)
print("MEMBACA DATASET")
print("=" * 50)

images = []
labels = []

for class_name in os.listdir(DATASET_PATH):

    class_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    if not os.path.isdir(class_path):
        continue

    count = 0

    for file in os.listdir(class_path):

        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):

            path = os.path.join(
                class_path,
                file
            )

            img = cv2.imread(path)

            if img is None:
                continue

            images.append(img)
            labels.append(class_name)

            count += 1

    print(f"{class_name:<15}: {count}")

print("\nTotal Data :", len(images))

# ======================================================
# FEATURE EXTRACTION
# ======================================================

print("\nMengekstraksi fitur HOG...")

features = []

for img in images:

    img_resize = cv2.resize(
        img,
        IMAGE_SIZE
    )

    gray = cv2.cvtColor(
        img_resize,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blur,
        50,
        150
    )

    feature = hog(
        edges,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    features.append(feature)

X = np.array(features)
y = np.array(labels)

print("Shape Feature :", X.shape)

# ======================================================
# SPLIT DATA
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain :", len(X_train))
print("Test  :", len(X_test))

# ======================================================
# TRAIN SVM
# ======================================================

print("\nTraining SVM...")

svm_model = make_pipeline(

    StandardScaler(),

    SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced"
    )
)

svm_model.fit(
    X_train,
    y_train
)

svm_pred = svm_model.predict(
    X_test
)

svm_acc = accuracy_score(
    y_test,
    svm_pred
)

print("\nAccuracy SVM :", round(
    svm_acc * 100,
    2
), "%")

print("\nClassification Report SVM\n")

print(
    classification_report(
        y_test,
        svm_pred
    )
)

# ======================================================
# TRAIN KNN
# ======================================================

print("\nTraining KNN...")

knn_model = make_pipeline(

    StandardScaler(),

    KNeighborsClassifier(
        n_neighbors=5
    )
)

knn_model.fit(
    X_train,
    y_train
)

knn_pred = knn_model.predict(
    X_test
)

knn_acc = accuracy_score(
    y_test,
    knn_pred
)

print("\nAccuracy KNN :", round(
    knn_acc * 100,
    2
), "%")

print("\nClassification Report KNN\n")

print(
    classification_report(
        y_test,
        knn_pred
    )
)

# ======================================================
# CONFUSION MATRIX SVM
# ======================================================

cm_svm = confusion_matrix(
    y_test,
    svm_pred
)

disp_svm = ConfusionMatrixDisplay(
    confusion_matrix=cm_svm,
    display_labels=svm_model.classes_
)

fig, ax = plt.subplots(
    figsize=(7,7)
)

disp_svm.plot(ax=ax)

plt.title(
    "Confusion Matrix HOG + SVM"
)

plt.savefig(
    "confusion_matrix_svm.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ======================================================
# CONFUSION MATRIX KNN
# ======================================================

cm_knn = confusion_matrix(
    y_test,
    knn_pred
)

disp_knn = ConfusionMatrixDisplay(
    confusion_matrix=cm_knn,
    display_labels=knn_model.classes_
)

fig, ax = plt.subplots(
    figsize=(7,7)
)

disp_knn.plot(ax=ax)

plt.title(
    "Confusion Matrix HOG + KNN"
)

plt.savefig(
    "confusion_matrix_knn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ======================================================
# PERBANDINGAN AKURASI
# ======================================================

plt.figure(figsize=(7,5))

metode = [
    "HOG + SVM",
    "HOG + KNN"
]

akurasi = [
    svm_acc * 100,
    knn_acc * 100
]

bars = plt.bar(
    metode,
    akurasi
)

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "Perbandingan Akurasi Metode"
)

for bar in bars:

    h = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        h + 0.5,
        f"{h:.2f}%",
        ha="center"
    )

plt.savefig(
    "perbandingan_akurasi.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ======================================================
# SIMPAN MODEL TERBAIK
# ======================================================

if svm_acc >= knn_acc:

    best_model = svm_model

    joblib.dump(
        best_model,
        "best_model_svm.pkl"
    )

    print(
        "\nModel terbaik : SVM"
    )

else:

    best_model = knn_model

    joblib.dump(
        best_model,
        "best_model_knn.pkl"
    )

    print(
        "\nModel terbaik : KNN"
    )

# ======================================================
# VISUALISASI TAHAPAN
# ======================================================

print("\nMenampilkan visualisasi...")

for class_name in os.listdir(DATASET_PATH):

    class_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    if not os.path.isdir(class_path):
        continue

    files = [
        f for f in os.listdir(class_path)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        )
    ]

    if len(files) < 2:
        continue

    samples = random.sample(
        files,
        2
    )

    for file in samples:

        path = os.path.join(
            class_path,
            file
        )

        img = cv2.imread(path)

        img_resize = cv2.resize(
            img,
            IMAGE_SIZE
        )

        gray = cv2.cvtColor(
            img_resize,
            cv2.COLOR_BGR2GRAY
        )

        blur = cv2.GaussianBlur(
            gray,
            (5,5),
            0
        )

        edges = cv2.Canny(
            blur,
            50,
            150
        )

        feature = hog(
            edges,
            orientations=9,
            pixels_per_cell=(8,8),
            cells_per_block=(2,2),
            block_norm="L2-Hys"
        )

        prediction = best_model.predict(
            [feature]
        )[0]

        plt.figure(
            figsize=(16,4)
        )

        plt.subplot(1,4,1)

        plt.imshow(
            cv2.cvtColor(
                img_resize,
                cv2.COLOR_BGR2RGB
            )
        )

        plt.title("Original")
        plt.axis("off")

        plt.subplot(1,4,2)

        plt.imshow(
            gray,
            cmap="gray"
        )

        plt.title("Grayscale")
        plt.axis("off")

        plt.subplot(1,4,3)

        plt.imshow(
            blur,
            cmap="gray"
        )

        plt.title("Gaussian Blur")
        plt.axis("off")

        plt.subplot(1,4,4)

        plt.imshow(
            edges,
            cmap="gray"
        )

        plt.title("Canny Edge")
        plt.axis("off")

        plt.suptitle(
            f"Actual : {class_name} | Prediksi : {prediction}"
        )

        plt.tight_layout()

        plt.savefig(
            f"sample_{class_name}_{file}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

# ======================================================
# HASIL AKHIR
# ======================================================

print("\n" + "="*50)
print("HASIL AKHIR")
print("="*50)

print(
    f"SVM Accuracy : {svm_acc*100:.2f}%"
)

print(
    f"KNN Accuracy : {knn_acc*100:.2f}%"
)

if svm_acc > knn_acc:

    print(
        "\nMetode terbaik: HOG + SVM"
    )

else:

    print(
        "\nMetode terbaik: HOG + KNN"
    )
