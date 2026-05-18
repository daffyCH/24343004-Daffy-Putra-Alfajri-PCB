import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import time

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import (
    VGG16,
    MobileNetV2,
    ResNet50
)

from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobile_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

# =========================================================
# KONFIGURASI
# =========================================================

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 30

DATASET_PATH = "dataset"

CLASS_NAMES = ["circle", "triangle", "square"]
NUM_CLASSES = len(CLASS_NAMES)

# =========================================================
# LOAD DATASET
# =========================================================

print("=" * 50)
print("LOAD DATASET")
print("=" * 50)

X = []
y = []

for class_idx, class_name in enumerate(CLASS_NAMES):

    class_folder = os.path.join(DATASET_PATH, class_name)

    for filename in os.listdir(class_folder):

        img_path = os.path.join(class_folder, filename)

        image = cv2.imread(img_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        image = image.astype("float32") / 255.0

        X.append(image)
        y.append(class_idx)

X = np.array(X)
y = np.array(y)

print(f"Total Dataset : {len(X)}")
print(f"Shape Dataset : {X.shape}")

# =========================================================
# SPLIT DATASET
# =========================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print(f"Train : {len(X_train)}")
print(f"Val   : {len(X_val)}")
print(f"Test  : {len(X_test)}")

# =========================================================
# VISUALISASI DATASET
# =========================================================

plt.figure(figsize=(12, 6))

for i in range(12):

    plt.subplot(3, 4, i + 1)

    plt.imshow(X_train[i])

    plt.title(CLASS_NAMES[y_train[i]])

    plt.axis("off")

plt.tight_layout()
plt.show()

# =========================================================
# DATA AUGMENTATION
# =========================================================

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

# =========================================================
# VISUALISASI AUGMENTATION
# =========================================================

sample = X_train[0]

sample = np.expand_dims(sample, axis=0)

plt.figure(figsize=(12, 6))

i = 0

for batch in datagen.flow(sample, batch_size=1):

    plt.subplot(2, 3, i + 1)

    img = batch[0]

    img = np.clip(img, 0, 1)

    plt.imshow(img)

    plt.axis("off")

    i += 1

    if i >= 6:
        break

plt.tight_layout()
plt.show()

# =========================================================
# CNN FROM SCRATCH
# =========================================================

print("=" * 50)
print("CNN FROM SCRATCH")
print("=" * 50)

cnn_model = keras.Sequential([

    layers.Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    ),

    layers.MaxPooling2D((2,2)),

    layers.Conv2D(
        64,
        (3,3),
        activation='relu'
    ),

    layers.MaxPooling2D((2,2)),

    layers.Conv2D(
        128,
        (3,3),
        activation='relu'
    ),

    layers.Flatten(),

    layers.Dense(
        256,
        activation='relu'
    ),

    layers.Dropout(0.5),

    layers.Dense(
        NUM_CLASSES,
        activation='softmax'
    )
])

cnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

cnn_model.summary()

# =========================================================
# TRAIN CNN
# =========================================================

history_cnn = cnn_model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    verbose=1
)

# =========================================================
# EVALUASI CNN
# =========================================================

print("=" * 50)
print("EVALUASI CNN")
print("=" * 50)

test_loss, test_acc = cnn_model.evaluate(X_test, y_test)

print(f"Test Accuracy : {test_acc:.4f}")

# =========================================================
# PREDIKSI
# =========================================================

y_pred_probs = cnn_model.predict(X_test)

y_pred = np.argmax(y_pred_probs, axis=1)

# =========================================================
# METRICS
# =========================================================

precision = precision_score(
    y_test,
    y_pred,
    average='macro'
)

recall = recall_score(
    y_test,
    y_pred,
    average='macro'
)

f1 = f1_score(
    y_test,
    y_pred,
    average='macro'
)

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\nCLASSIFICATION REPORT")
print("=" * 50)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES
    )
)

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()

# =========================================================
# LEARNING CURVE
# =========================================================

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)

plt.plot(
    history_cnn.history['accuracy'],
    label='Train Accuracy'
)

plt.plot(
    history_cnn.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title("Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

# Loss
plt.subplot(1, 2, 2)

plt.plot(
    history_cnn.history['loss'],
    label='Train Loss'
)

plt.plot(
    history_cnn.history['val_loss'],
    label='Validation Loss'
)

plt.title("Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.tight_layout()
plt.show()

# =========================================================
# ROC CURVE
# =========================================================

y_test_bin = label_binarize(
    y_test,
    classes=[0,1,2]
)

plt.figure(figsize=(7,6))

for i in range(NUM_CLASSES):

    fpr, tpr, _ = roc_curve(
        y_test_bin[:, i],
        y_pred_probs[:, i]
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{CLASS_NAMES[i]} (AUC={roc_auc:.2f})"
    )

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

# =========================================================
# FEATURE MAP VISUALIZATION
# =========================================================

print("=" * 50)
print("FEATURE MAP VISUALIZATION")
print("=" * 50)

layer_outputs = [
    layer.output
    for layer in cnn_model.layers[:3]
]

activation_model = keras.Model(
    inputs=cnn_model.inputs,
    outputs=layer_outputs
)

sample_image = np.expand_dims(X_test[0], axis=0)

activations = activation_model(sample_image, training=False)

first_layer_activation = activations[0]

plt.figure(figsize=(12, 8))

for i in range(16):

    plt.subplot(4, 4, i + 1)

    plt.imshow(
        first_layer_activation[0, :, :, i],
        cmap='viridis'
    )

    plt.axis("off")

plt.tight_layout()
plt.show()

# =========================================================
# t-SNE VISUALIZATION
# =========================================================

print("=" * 50)
print("t-SNE VISUALIZATION")
print("=" * 50)

feature_extractor = keras.Model(
    inputs=cnn_model.inputs,
    outputs=cnn_model.layers[-3].output
)

features = feature_extractor(X_test, training=False)
features = features.numpy()

tsne = TSNE(
    n_components=2,
    random_state=42
)

features_2d = tsne.fit_transform(features)

plt.figure(figsize=(8,6))

for i, class_name in enumerate(CLASS_NAMES):

    idx = y_test == i

    plt.scatter(
        features_2d[idx, 0],
        features_2d[idx, 1],
        label=class_name
    )

plt.legend()

plt.title("t-SNE Feature Embedding")

plt.show()

# =========================================================
# SAVE MODEL
# =========================================================

cnn_model.save("cnn_geometri_model.keras")

print("=" * 50)
print("MODEL SAVED")
print("=" * 50)

# =========================================================
# TRANSFER LEARNING COMPARISON
# VGG16 vs MobileNetV2 vs ResNet50
# =========================================================


print("=" * 60)
print("TRANSFER LEARNING COMPARISON")
print("=" * 60)

transfer_results = {}

transfer_models = {

    "VGG16": (
        VGG16,
        vgg_preprocess
    ),

    "MobileNetV2": (
        MobileNetV2,
        mobile_preprocess
    ),

    "ResNet50": (
        ResNet50,
        resnet_preprocess
    )
}

for model_name, (model_fn, preprocess_fn) in transfer_models.items():

    print("\n" + "=" * 60)
    print(f"MODEL : {model_name}")
    print("=" * 60)

    # =====================================================
    # PREPROCESS
    # =====================================================

    X_train_tf = preprocess_fn(
        X_train.copy() * 255
    )

    X_val_tf = preprocess_fn(
        X_val.copy() * 255
    )

    X_test_tf = preprocess_fn(
        X_test.copy() * 255
    )

    # =====================================================
    # BASE MODEL
    # =====================================================

    base_model = model_fn(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    base_model.trainable = False

    # =====================================================
    # BUILD MODEL
    # =====================================================

    transfer_model = keras.Sequential([

        base_model,

        layers.GlobalAveragePooling2D(),

        layers.Dense(
            128,
            activation='relu'
        ),

        layers.Dropout(0.5),

        layers.Dense(
            len(CLASS_NAMES),
            activation='softmax'
        )
    ])

    transfer_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    transfer_model.summary()

    # =====================================================
    # TRAINING
    # =====================================================

    start_time = time.time()

    history_transfer = transfer_model.fit(
        X_train_tf,
        y_train,
        validation_data=(X_val_tf, y_val),
        epochs=10,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    training_time = time.time() - start_time

    # =====================================================
    # FINE TUNING
    # =====================================================

    print("\nFINE TUNING...")

    base_model.trainable = True

    for layer in base_model.layers[:-4]:
        layer.trainable = False

    transfer_model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history_fine = transfer_model.fit(
        X_train_tf,
        y_train,
        validation_data=(X_val_tf, y_val),
        epochs=5,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # =====================================================
    # EVALUASI
    # =====================================================

    test_loss, test_acc = transfer_model.evaluate(
        X_test_tf,
        y_test,
        verbose=0
    )

    y_pred_probs = transfer_model.predict(
        X_test_tf,
        verbose=0
    )

    y_pred = np.argmax(
        y_pred_probs,
        axis=1
    )

    precision = precision_score(
        y_test,
        y_pred,
        average='macro'
    )

    recall = recall_score(
        y_test,
        y_pred,
        average='macro'
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average='macro'
    )

    print(f"\nAccuracy  : {test_acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"Training Time : {training_time:.2f} detik")

    # =====================================================
    # SIMPAN HASIL
    # =====================================================

    transfer_results[model_name] = {
        "accuracy": test_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "time": training_time
    }

    # =====================================================
    # SAVE MODEL
    # =====================================================

    transfer_model.save(
        f"{model_name}_model.keras"
    )

# =========================================================
# PERBANDINGAN MODEL
# =========================================================

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

for model_name, result in transfer_results.items():

    print(f"""
{model_name}
-------------------------
Accuracy  : {result['accuracy']:.4f}
Precision : {result['precision']:.4f}
Recall    : {result['recall']:.4f}
F1-Score  : {result['f1']:.4f}
Train Time: {result['time']:.2f} detik
""")

# =========================================================
# VISUALISASI PERBANDINGAN MODEL
# =========================================================

model_names = list(transfer_results.keys())

accuracies = [
    transfer_results[m]['accuracy']
    for m in model_names
]

f1_scores = [
    transfer_results[m]['f1']
    for m in model_names
]

plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)

plt.bar(
    model_names,
    accuracies
)

plt.title("Transfer Learning Accuracy")

plt.ylabel("Accuracy")

# F1
plt.subplot(1,2,2)

plt.bar(
    model_names,
    f1_scores
)

plt.title("Transfer Learning F1-Score")

plt.ylabel("F1-Score")

plt.tight_layout()
plt.show()