import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler


def load_dataset(folder_name="dataset"):

    base_dir = os.path.dirname(os.path.abspath(__file__))

    dataset_path = os.path.join(
        base_dir,
        folder_name
    )

    if not os.path.exists(dataset_path):

        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan:\n{dataset_path}"
        )

    images = []
    labels = []
    class_names = []

    classes = [

        cls for cls in os.listdir(dataset_path)

        if os.path.isdir(
            os.path.join(dataset_path, cls)
        )
    ]

    classes.sort()

    for label, cls in enumerate(classes):

        class_names.append(cls)

        cls_path = os.path.join(
            dataset_path,
            cls
        )

        for file in os.listdir(cls_path):

            file_path = os.path.join(
                cls_path,
                file
            )

            if not os.path.isfile(file_path):
                continue

            img = cv2.imread(file_path)

            if img is None:
                continue

            gray = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

            gray = cv2.resize(
                gray,
                (256, 256)
            )

            blur = cv2.GaussianBlur(
                gray,
                (5, 5),
                0
            )

            _, thresh = cv2.threshold(
                blur,
                0,
                255,
                cv2.THRESH_BINARY +
                cv2.THRESH_OTSU
            )

            images.append(thresh)

            labels.append(label)

    return images, labels, class_names


def show_dataset(
    images,
    labels,
    class_names,
    max_images=9
):

    plt.figure(figsize=(10, 10))

    for i in range(
        min(max_images, len(images))
    ):

        plt.subplot(3, 3, i + 1)

        plt.imshow(
            images[i],
            cmap="gray"
        )

        plt.title(
            class_names[labels[i]]
        )

        plt.axis("off")

    plt.tight_layout()

    plt.show()


def region_features(contour):

    area = cv2.contourArea(contour)

    perimeter = cv2.arcLength(
        contour,
        True
    )

    M = cv2.moments(contour)

    cx = (
        M["m10"] / M["m00"]
        if M["m00"] != 0 else 0
    )

    cy = (
        M["m01"] / M["m00"]
        if M["m00"] != 0 else 0
    )

    x, y, w, h = cv2.boundingRect(
        contour
    )

    aspect_ratio = (
        w / h
        if h != 0 else 0
    )

    rect_area = w * h

    extent = (
        area / rect_area
        if rect_area != 0 else 0
    )

    hull = cv2.convexHull(contour)

    hull_area = cv2.contourArea(hull)

    solidity = (
        area / hull_area
        if hull_area != 0 else 0
    )

    return {

        "area": area,

        "perimeter": perimeter,

        "centroid_x": cx,

        "centroid_y": cy,

        "aspect_ratio": aspect_ratio,

        "extent": extent,

        "solidity": solidity
    }


def moment_features(contour):

    M = cv2.moments(contour)

    hu = cv2.HuMoments(M).flatten()

    return {

        "m00": M["m00"],

        "m10": M["m10"],

        "m01": M["m01"],

        "mu20": M["mu20"],

        "mu02": M["mu02"],

        "mu11": M["mu11"],

        "hu1": hu[0],

        "hu2": hu[1],

        "hu3": hu[2],

        "hu4": hu[3],

        "hu5": hu[4],

        "hu6": hu[5],

        "hu7": hu[6]
    }


def chain_code_8(contour):

    pts = contour.reshape(-1, 2)

    directions = [

        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),

        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1)
    ]

    code = []

    for i in range(len(pts) - 1):

        dx = (
            pts[i + 1][0] -
            pts[i][0]
        )

        dy = (
            pts[i + 1][1] -
            pts[i][1]
        )

        dx = int(np.sign(dx))
        dy = int(np.sign(dy))

        for idx, (x, y) in enumerate(directions):

            if dx == x and dy == y:

                code.append(idx)

                break

    return code


def chain_code_4(contour):

    pts = contour.reshape(-1, 2)

    directions = [

        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1)
    ]

    code = []

    for i in range(len(pts) - 1):

        dx = (
            pts[i + 1][0] -
            pts[i][0]
        )

        dy = (
            pts[i + 1][1] -
            pts[i][1]
        )

        dx = int(np.sign(dx))

        dy = int(np.sign(dy))

        for idx, (x, y) in enumerate(directions):

            if dx == x and dy == y:

                code.append(idx)

                break

    return code


def normalize_chain(code):

    if len(code) == 0:
        return code

    min_index = np.argmin(code)

    normalized = (

        code[min_index:] +

        code[:min_index]
    )

    return normalized


def polygon_approx(contour):

    epsilon = (

        0.01 *

        cv2.arcLength(contour, True)
    )

    approx = cv2.approxPolyDP(

        contour,

        epsilon,

        True
    )

    return len(approx)


def fourier_features(contour, n=20):

    pts = contour.reshape(-1, 2)

    complex_points = (

        pts[:, 0] +

        1j * pts[:, 1]
    )

    fd = np.fft.fft(complex_points)

    fd = np.abs(fd)

    if fd[0] != 0:

        fd = fd / fd[0]

    fd = list(fd[:n])

    if len(fd) < n:

        fd += [0] * (n - len(fd))

    return fd


def reconstruct_fourier(
    contour,
    descriptors=10
):

    pts = contour.reshape(-1, 2)

    complex_points = (

        pts[:, 0] +

        1j * pts[:, 1]
    )

    fft_result = np.fft.fft(
        complex_points
    )

    truncated = np.zeros_like(
        fft_result
    )

    truncated[:descriptors] = (
        fft_result[:descriptors]
    )

    truncated[-descriptors:] = (
        fft_result[-descriptors:]
    )

    reconstructed = np.fft.ifft(
        truncated
    )

    return reconstructed


def show_fourier_reconstruction(contour):

    pts = contour.reshape(-1, 2)

    descriptor_values = [5, 10, 20]

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)

    plt.plot(
        pts[:, 0],
        pts[:, 1]
    )

    plt.gca().invert_yaxis()

    plt.title("Original")

    for i, d in enumerate(descriptor_values):

        recon = reconstruct_fourier(
            contour,
            d
        )

        plt.subplot(1, 4, i + 2)

        plt.plot(
            recon.real,
            recon.imag
        )

        plt.gca().invert_yaxis()

        plt.title(
            f"Descriptor {d}"
        )

    plt.tight_layout()

    plt.show()


def show_contour(img, contour):

    canvas = cv2.cvtColor(
        img,
        cv2.COLOR_GRAY2BGR
    )

    cv2.drawContours(
        canvas,
        [contour],
        -1,
        (0, 255, 0),
        2
    )

    plt.figure(figsize=(5, 5))

    plt.imshow(
        cv2.cvtColor(
            canvas,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.axis("off")

    plt.show()


def extract_features(images):

    feature_vectors = []

    for img in images:

        contours, _ = cv2.findContours(

            img,

            cv2.RETR_EXTERNAL,

            cv2.CHAIN_APPROX_NONE
        )

        if len(contours) == 0:
            continue

        contour = max(
            contours,
            key=cv2.contourArea
        )

        region = region_features(
            contour
        )

        moment = moment_features(
            contour
        )

        cc8 = normalize_chain(
            chain_code_8(contour)
        )[:20]

        cc4 = normalize_chain(
            chain_code_4(contour)
        )[:20]

        cc8 += [0] * (
            20 - len(cc8)
        )

        cc4 += [0] * (
            20 - len(cc4)
        )

        polygon = [
            polygon_approx(contour)
        ]

        fourier = fourier_features(
            contour,
            20
        )

        vector = []

        vector.extend(
            list(region.values())
        )

        vector.extend(
            list(moment.values())
        )

        vector.extend(cc8)

        vector.extend(cc4)

        vector.extend(polygon)

        vector.extend(fourier)

        feature_vectors.append(vector)

    return np.array(feature_vectors)


def evaluate_knn(
    X,
    y,
    class_names
):

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.3,

        random_state=42,

        stratify=y
    )

    best_accuracy = 0

    best_prediction = None

    best_k = 0

    for k in [1, 3, 5]:

        model = KNeighborsClassifier(
            n_neighbors=k
        )

        model.fit(
            X_train,
            y_train
        )

        prediction = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            prediction
        )

        print(
            f"k = {k} | Accuracy = {accuracy:.4f}"
        )

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            best_prediction = prediction

            best_k = k

    print("\nBest k:", best_k)

    print(
        "Best Accuracy:",
        best_accuracy
    )

    print(
        classification_report(
            y_test,
            best_prediction,
            target_names=class_names
        )
    )

    cm = confusion_matrix(
        y_test,
        best_prediction
    )

    disp = ConfusionMatrixDisplay(

        confusion_matrix=cm,

        display_labels=class_names
    )

    disp.plot()

    plt.show()


def compare_descriptors(
    X,
    y
):

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.3,

        random_state=42,

        stratify=y
    )

    descriptor_sets = {

        "Region":
        X[:, :7],

        "Moments":
        X[:, 7:20],

        "Fourier":
        X[:, -20:]
    }

    print("\n===== DESCRIPTOR COMPARISON =====")

    for name, feature in descriptor_sets.items():

        X_train, X_test, y_train, y_test = train_test_split(

            feature,
            y,

            test_size=0.3,

            random_state=42,

            stratify=y
        )

        model = KNeighborsClassifier(
            n_neighbors=3
        )

        model.fit(
            X_train,
            y_train
        )

        prediction = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            prediction
        )

        print(
            f"{name} Accuracy = {accuracy:.4f}"
        )


def run_pipeline():

    images, labels, class_names = load_dataset()

    print("\n===== DATASET =====")

    print(class_names)

    print(
        "\nJumlah gambar:",
        len(images)
    )

    show_dataset(
        images,
        labels,
        class_names
    )

    X = extract_features(images)

    y = np.array(
        labels[:len(X)]
    )

    print(
        "\nFeature Matrix:",
        X.shape
    )

    sample_img = images[0]

    contours, _ = cv2.findContours(

        sample_img,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_NONE
    )

    contour = max(
        contours,
        key=cv2.contourArea
    )

    print("\n===== REGION =====")

    region = region_features(
        contour
    )

    for k, v in region.items():

        print(f"{k}: {v}")

    print("\n===== MOMENTS =====")

    moments = moment_features(
        contour
    )

    for k, v in moments.items():

        print(f"{k}: {v}")

    print("\n===== CHAIN CODE =====")

    cc4 = normalize_chain(
        chain_code_4(contour)
    )[:20]

    cc8 = normalize_chain(
        chain_code_8(contour)
    )[:20]

    print("Chain Code 4:")

    print(cc4)

    print("\nChain Code 8:")

    print(cc8)

    print("\n===== POLYGON =====")

    poly = polygon_approx(
        contour
    )

    print(
        "Polygon Vertex:",
        poly
    )

    print("\n===== FOURIER =====")

    fourier = fourier_features(
        contour,
        20
    )

    print(fourier)

    show_contour(
        sample_img,
        contour
    )

    show_fourier_reconstruction(
        contour
    )

    print("\n===== KNN =====")

    evaluate_knn(
        X,
        y,
        class_names
    )

    compare_descriptors(
        X,
        y
    )


if __name__ == "__main__":

    run_pipeline()