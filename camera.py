import os
# Set environment variables for Qt and display
os.environ["QT_QPA_PLATFORM"] = "eglfs"
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/aarch64-linux-gnu/qt5/plugins"
os.environ["DISPLAY"] = ":0"
os.environ["XDG_RUNTIME_DIR"] = "/run/user/1000"

import cv2
import time
import numpy as np
import requests
from picamera2 import Picamera2

# YOLOv4-tiny model files
MODEL_URL = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
CONFIG_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
CLASSES_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"

MODEL_PATH = "yolov4-tiny.weights"
CONFIG_PATH = "yolov4-tiny.cfg"
CLASSES_PATH = "coco.names"

def download_file(url, path):
    """Download a file from URL if it doesn't exist"""
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {path} successfully")
            return True
        except Exception as e:
            print(f"Failed to download {path}: {e}")
            return False
    return True

# Download model files
config_ok = download_file(CONFIG_URL, CONFIG_PATH)
model_ok = download_file(MODEL_URL, MODEL_PATH)
classes_ok = download_file(CLASSES_URL, CLASSES_PATH)

# Initialize the model only if files exist
net = None
classes = []
if config_ok and model_ok and classes_ok:
    try:
        # Load class names
        with open(CLASSES_PATH, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        
        # Load model
        net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, MODEL_PATH)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("Model loaded successfully")
        print(f"Detectable objects: {len(classes)} classes")
    except Exception as e:
        print(f"Failed to load model: {e}")
        net = None
else:
    print("Skipping model initialization due to missing files")

# Generate random colors for each class
COLORS = np.random.uniform(0, 255, size=(len(classes) if classes else 80, 3))

def setup_cam():
    picam2 = Picamera2()
    # Use 480x360 resolution for better performance with YOLO
    config = picam2.create_preview_configuration(
        main={"size": (480, 360), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # Camera warm-up time
    return picam2

def detect_objects(frame):
    if net is None:
        return frame
    
    # Store original frame for display
    display_frame = frame.copy()
    
    # Create blob from image (YOLO expects 416x416)
    blob = cv2.dnn.blobFromImage(
        frame, 1/255.0, (416, 416), 
        swapRB=True,  # Swap red and blue channels
        crop=False
    )
    
    # Pass the blob through the network
    net.setInput(blob)
    
    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Forward pass
    detections = net.forward(output_layers)
    
    # Post-processing
    boxes = []
    confidences = []
    class_ids = []
    
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Filter out weak detections
            if confidence > 0.5:
                # Scale bounding box coordinates
                (h, w) = frame.shape[:2]
                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                
                # Calculate top-left corner
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                # Add to lists
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-max suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Draw final detections on the display frame
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, w, h) = boxes[i]
            
            # Ensure coordinates stay within frame boundaries
            x = max(0, x)
            y = max(0, y)
            w = min(w, w - x)
            h = min(h, h - y)
            
            # Draw rectangle and label
            color = COLORS[class_ids[i]]
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            y_text = y - 15 if y - 15 > 15 else y + 15
            cv2.putText(display_frame, label, (x, y_text), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return display_frame

def object_detection(cam):
    try:
        # Create window with correct size
        cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Object Detection', 480, 360)
        
        while True:
            frame = cam.capture_array("main")
            
            # Camera gives RGB format, but we need to convert to BGR for OpenCV
            # Only convert if we have 3 channels
            if frame.shape[2] == 3:
                # Convert from RGB to BGR
                processing_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif frame.shape[2] == 4:
                # Convert from RGBA to BGR
                processing_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else:
                processing_frame = frame.copy()
            
            # Perform object detection
            detected_frame = detect_objects(processing_frame)
            
            # Convert back to RGB for display (since we'll show with cv2.imshow)
            display_frame = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            cv2.imshow('Object Detection', display_frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting object detection...")
    print("Press 'q' to exit")
    camera = setup_cam()
    object_detection(camera)