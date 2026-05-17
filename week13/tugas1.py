import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt

from skimage.feature import hog

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    learning_curve
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

from sklearn.preprocessing import (
    label_binarize,
    StandardScaler
)

from sklearn.decomposition import PCA

# =====================================================
# LOAD DATASET
# =====================================================

def load_dataset():

    base_dir = os.path.dirname(os.path.abspath(__file__))

    path = os.path.join(base_dir, "dataset")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")

    images = []
    labels = []

    class_names = sorted(os.listdir(path))

    for label, cls in enumerate(class_names):

        cls_path = os.path.join(path, cls)

        if not os.path.isdir(cls_path):
            continue

        for file in os.listdir(cls_path):

            img_path = os.path.join(cls_path, file)

            img = cv2.imread(img_path, 0)

            if img is None:
                continue

            img = cv2.resize(img, (128,128))

            img = cv2.GaussianBlur(img, (3,3), 0)

            _, img = cv2.threshold(
                img,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            images.append(img)
            labels.append(label)

    return images, np.array(labels), class_names

# =====================================================
# FEATURE EXTRACTION
# Shape + Hu Moments + HOG
# =====================================================

def extract_features(images):

    feats = []

    for img in images:

        contours, _ = cv2.findContours(
            img,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:

            feature = np.zeros(161)

            feats.append(feature)

            continue

        contours = sorted(
            contours,
            key=cv2.contourArea,
            reverse=True
        )

        c = contours[0]

        area = cv2.contourArea(c)

        perimeter = cv2.arcLength(c, True)

        x, y, w, h = cv2.boundingRect(c)

        aspect_ratio = w / h if h != 0 else 0

        compactness = (
            (4 * np.pi * area) /
            (perimeter * perimeter + 1e-6)
        )

        M = cv2.moments(c)

        hu = cv2.HuMoments(M).flatten()

        hog_feat = hog(
            img,
            orientations=9,
            pixels_per_cell=(8,8),
            cells_per_block=(2,2),
            visualize=False
        )

        feature = [
            area,
            perimeter,
            aspect_ratio,
            compactness
        ]

        feature.extend(list(hu))

        # ambil sebagian HOG biar tidak berat
        feature.extend(hog_feat[:300])

        feats.append(feature)

    return np.array(feats)

# =====================================================
# EVALUASI
# =====================================================

def evaluate(y_test, y_pred):

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test,
            y_pred,
            average='macro',
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            y_pred,
            average='macro',
            zero_division=0
        ),
        "f1": f1_score(
            y_test,
            y_pred,
            average='macro',
            zero_division=0
        ),
        "cm": confusion_matrix(y_test, y_pred)
    }

# =====================================================
# CONFUSION MATRIX
# =====================================================

def plot_confusion_subplot(cm, title, index, total):

    plt.subplot(total[0], total[1], index)

    plt.imshow(cm, cmap='Blues')

    plt.title(title)

    plt.xlabel("Predicted")

    plt.ylabel("True")

# =====================================================
# KNN
# =====================================================

def run_knn(X_train, X_test, y_train, y_test):

    print("\n========== KNN ==========")

    k_values = [1,3,5,7,9,11]

    metrics = [
        'euclidean',
        'manhattan',
        'minkowski'
    ]

    plt.figure(figsize=(18,12))

    idx = 1

    for metric in metrics:

        for k in k_values:

            t0 = time.time()

            model = KNeighborsClassifier(
                n_neighbors=k,
                metric=metric
            )

            model.fit(X_train, y_train)

            train_time = time.time() - t0

            t1 = time.time()

            y_pred = model.predict(X_test)

            test_time = time.time() - t1

            res = evaluate(y_test, y_pred)

            print(f"\nK={k}, metric={metric}")

            print(
                f"Accuracy : {res['accuracy']:.4f}"
            )

            print(
                f"Precision: {res['precision']:.4f}"
            )

            print(
                f"Recall   : {res['recall']:.4f}"
            )

            print(
                f"F1 Score : {res['f1']:.4f}"
            )

            print(
                f"Train Time : {train_time:.4f}"
            )

            print(
                f"Test Time  : {test_time:.4f}"
            )

            plot_confusion_subplot(
                res["cm"],
                f"K={k}\n{metric}",
                idx,
                (3,6)
            )

            idx += 1

    plt.tight_layout()

    plt.show()

# =====================================================
# SVM
# =====================================================

def run_svm(
    X_train,
    X_test,
    y_train,
    y_test,
    n_classes
):

    print("\n========== SVM ==========")

    param_grid = {

        'C': [0.1,1,10,100],

        'gamma': [0.001,0.01,0.1,1],

        'kernel': [
            'linear',
            'rbf',
            'poly'
        ]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    grid = GridSearchCV(

        SVC(),

        param_grid,

        cv=cv,

        verbose=0,

        n_jobs=-1
    )

    t0 = time.time()

    grid.fit(X_train, y_train)

    train_time = time.time() - t0

    print("\nBest Parameter:")
    print(grid.best_params_)

    model = grid.best_estimator_

    t1 = time.time()

    y_pred = model.predict(X_test)

    test_time = time.time() - t1

    res = evaluate(y_test, y_pred)

    print(
        f"\nAccuracy : {res['accuracy']:.4f}"
    )

    print(
        f"Precision: {res['precision']:.4f}"
    )

    print(
        f"Recall   : {res['recall']:.4f}"
    )

    print(
        f"F1 Score : {res['f1']:.4f}"
    )

    print(
        f"Train Time : {train_time:.4f}"
    )

    print(
        f"Test Time  : {test_time:.4f}"
    )

    # =========================================
    # VISUALISASI
    # =========================================

    plt.figure(figsize=(12,5))

    # confusion matrix
    plt.subplot(1,2,1)

    plt.imshow(res["cm"], cmap='Blues')

    plt.title("SVM Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("True")

    # ROC
    plt.subplot(1,2,2)

    y_bin = label_binarize(
        y_test,
        classes=range(n_classes)
    )

    y_score = model.decision_function(X_test)

    for i in range(n_classes):

        fpr, tpr, _ = roc_curve(
            y_bin[:,i],
            y_score[:,i]
        )

        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"Class {i} AUC={roc_auc:.2f}"
        )

    plt.title("ROC Curve")

    plt.legend()

    plt.tight_layout()

    plt.show()

# =====================================================
# PCA VISUAL
# =====================================================

def plot_pca(X, y):

    pca = PCA(n_components=2)

    Xp = pca.fit_transform(X)

    plt.figure(figsize=(8,6))

    scatter = plt.scatter(
        Xp[:,0],
        Xp[:,1],
        c=y
    )

    plt.title("PCA 2D")

    plt.xlabel("PC1")

    plt.ylabel("PC2")

    plt.colorbar(scatter)

    plt.show()

# =====================================================
# LEARNING CURVE
# =====================================================

def plot_learning_curve(X, y):

    model = SVC(kernel='linear')

    train_sizes, train_scores, test_scores = learning_curve(

        model,

        X,

        y,

        cv=3,

        train_sizes=np.linspace(0.1,1.0,5),

        shuffle=True,
        
        random_state=42,

        n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)

    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8,6))

    plt.plot(
        train_sizes,
        train_mean,
        label="Training Score"
    )

    plt.plot(
        train_sizes,
        test_mean,
        label="Validation Score"
    )

    plt.title("Learning Curve")

    plt.xlabel("Training Size")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.show()

# =====================================================
# MAIN
# =====================================================

def run():

    images, y, classes = load_dataset()

    X = extract_features(images)

    # scaling penting untuk SVM
    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.3,

        stratify=y,

        random_state=42
    )

    run_knn(
        X_train,
        X_test,
        y_train,
        y_test
    )

    run_svm(
        X_train,
        X_test,
        y_train,
        y_test,
        len(classes)
    )

    plot_pca(X, y)

    plot_learning_curve(X, y)

# =====================================================

if __name__ == "__main__":

    run()