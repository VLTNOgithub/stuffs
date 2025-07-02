import os
os.environ["QT_QPA_PLATFORM"] = "eglfs"
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/aarch64-linux-gnu/qt5/plugins"
os.environ["DISPLAY"] = ":0"
os.environ["XDG_RUNTIME_DIR"] = "/run/user/1000"

import cv2
import time
import numpy as np
import requests
from picamera2 import Picamera2

# ===== CONFIGURATION =====
MODEL_URL = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
CONFIG_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
CLASSES_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"

MODEL_PATH = "yolov4-tiny.weights"
CONFIG_PATH = "yolov4-tiny.cfg"
CLASSES_PATH = "coco.names"

# Detection parameters (adjust these as needed)
DRAW_BOXES = True  # Set to False to disable bounding boxes
CONFIDENCE_THRESHOLD = 0.4  # Higher = fewer but more accurate detections
NMS_THRESHOLD = 0.3  # Higher = more overlapping boxes allowed
RESOLUTION = (480, 360)  # Lower for better performance

# ===== INITIALIZATION =====
def download_file(url, path):
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk: f.write(chunk)
            print(f"Downloaded {path} successfully")
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False
    return True

# Download model files
config_ok = download_file(CONFIG_URL, CONFIG_PATH)
model_ok = download_file(MODEL_URL, MODEL_PATH)
classes_ok = download_file(CLASSES_URL, CLASSES_PATH)

# Initialize model
net = None
classes = []
if config_ok and model_ok and classes_ok:
    try:
        with open(CLASSES_PATH, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        
        net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, MODEL_PATH)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print(f"Model loaded - {len(classes)} detectable objects")
    except Exception as e:
        print(f"Model loading failed: {e}")
        net = None
else:
    print("Skipping model initialization")

# ===== CAMERA SETUP =====
def setup_cam():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": RESOLUTION, "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # Warm-up
    return picam2

# ===== OBJECT DETECTION =====
def detect_objects(frame):
    if net is None:
        return frame, []
    
    # Convert to BGR for processing
    if frame.shape[2] == 3:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    
    # Create blob and run detection
    blob = cv2.dnn.blobFromImage(
        frame_bgr, 1/255.0, (416, 416), 
        swapRB=True, crop=False
    )
    net.setInput(blob)
    
    # Get output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    detections = net.forward(output_layers)
    
    # Process detections
    boxes = []
    confidences = []
    class_ids = []
    class_names = []
    h, w = frame.shape[:2]
    
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > CONFIDENCE_THRESHOLD:
                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                class_names.append(classes[class_id])
    
    # Apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    
    # Draw boxes if enabled
    if DRAW_BOXES and len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            x, y = max(0, x), max(0, y)
            
            color = (0, 255, 0)  # Green boxes
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.putText(frame, label, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame, class_names

# ===== MAIN LOOP =====
def object_detection(cam):
    try:
        cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Object Detection', *RESOLUTION)
        
        while True:
            frame = cam.capture_array("main")
            detected_frame, objects = detect_objects(frame)
            
            # Display detected objects in console
            if objects:
                print(f"Detected: {', '.join(set(objects))}")
            
            # Show frame
            cv2.imshow('Object Detection', detected_frame)
            
            # Exit on 'q'
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