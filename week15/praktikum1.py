import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import torch
import torchvision
from PIL import Image

def praktikum_7_1():
    print("COMPUTER VISION: OBJECT DETECTION WITH YOLO")
    print("=" * 50)

    # Download sample image
    def download_sample_image():
        url = ""
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Load YOLO model (using OpenCV's DNN module)
    def load_yolo_model():
        # Download YOLO weights and config
        weights_url = "https://github.com/pjreddie/darknet/raw/master/cfg/yolov3.weights"
        config_url = "https://github.com/pjreddie/darknet/raw/master/cfg/yolov3.cfg"
        names_url = "https://github.com/pjreddie/darknet/raw/master/data/coco.names"

        # Load class names
        response = requests.get(names_url)
        classes = response.text.strip().split('\n')

        # Load model
        net = cv2.dnn.readNetFromDarknet(config_url, weights_url)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        return net, classes

    # Object detection function
    def detect_objects_yolo(net, image, classes, conf_threshold=0.5, nms_threshold=0.4):
        height, width = image.shape[:2]

        # Create blob from image
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        # Get output layer names
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # Forward pass
        outputs = net.forward(output_layers)

        # Process detections
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > conf_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # Draw detections
        result_image = image.copy()
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"

                # Draw bounding box
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Draw label
                cv2.putText(result_image, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return result_image, len(indices)

    try:
        # Load model and image
        net, classes = load_yolo_model()
        image = download_sample_image()

        # Perform detection
        result_image, num_detections = detect_objects_yolo(net, image, classes)

        # Display results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image')
        ax1.axis('off')

        ax2.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        ax2.set_title(f'YOLO Detection Results: {num_detections} objects found')
        ax2.axis('off')

        plt.tight_layout()
        plt.show()

        print(f"Detected {num_detections} objects in the image")

        # Simple object counting by class
        print("\nObject Count by Class:")
        print("-" * 30)

        return result_image, num_detections

    except Exception as e:
        print(f"YOLO demonstration skipped: {e}")
        # Fallback: Simulate object detection with simple methods
        print("Using simulated object detection...")
        return simulate_object_detection()

def simulate_object_detection():
    # Create synthetic image with objects
    image = np.zeros((400, 600, 3), dtype=np.uint8)

    # Add different colored rectangles as "objects"
    cv2.rectangle(image, (50, 50), (150, 150), (255, 0, 0), -1)  # Blue object
    cv2.rectangle(image, (200, 80), (300, 180), (0, 255, 0), -1)  # Green object
    cv2.rectangle(image, (350, 120), (450, 220), (0, 0, 255), -1)  # Red object
    cv2.rectangle(image, (100, 250), (200, 350), (255, 255, 0), -1)  # Cyan object

    # Simple color-based detection
    result_image = image.copy()
    colors = {
        'Blue Object': [255, 0, 0],
        'Green Object': [0, 255, 0],
        'Red Object': [0, 0, 255],
        'Cyan Object': [255, 255, 0]
    }

    detection_count = 0
    for label, color in colors.items():
        # Create mask for each color
        lower = np.array(color) - 10
        upper = np.array(color) + 10
        mask = cv2.inRange(image, lower, upper)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Filter small detections
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(result_image, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                detection_count += 1

    # Display results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title('Synthetic Image with Objects')
    ax1.axis('off')

    ax2.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    ax2.set_title(f'Simulated Detection: {detection_count} objects found')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

    return result_image, detection_count

result_image, detections = praktikum_7_1()
