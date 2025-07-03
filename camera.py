import os
os.environ["QT_QPA_PLATFORM"] = "eglfs"
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/aarch64-linux-gnu/qt5/plugins"
os.environ["DISPLAY"] = ":0"
os.environ["XDG_RUNTIME_DIR"] = "/run/user/1000"

import cv2
import time
import numpy as np
import requests
import threading
from picamera2 import Picamera2
from sense_hat import SenseHat
import yapper as ypr

# ===== CONFIGURATION =====
MODEL_URL = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
CONFIG_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
CLASSES_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"

MODEL_PATH = "yolov4-tiny.weights"
CONFIG_PATH = "yolov4-tiny.cfg"
CLASSES_PATH = "coco.names"

# Detection parameters
DRAW_BOXES = True
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3
RESOLUTION = (480, 360)
DEADZONE = 6
MIDDLE_X = RESOLUTION[0] // 2
BLOB_SIZE = 320  # Reduced network input size

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
output_layers = []
if config_ok and model_ok and classes_ok:
    try:
        with open(CLASSES_PATH, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        
        net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, MODEL_PATH)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Precompute output layers once
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        
        print(f"Model loaded - {len(classes)} detectable objects")
    except Exception as e:
        print(f"Model loading failed: {e}")
        net = None
else:
    print("Skipping model initialization")

# Initialise Sense HAT
sense = SenseHat()
sense.clear()

# Initialise Yapper
yapper = ypr.Yapper()

# Precompute deadzone lines
deadzone_lines = np.zeros((RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8)
cv2.line(deadzone_lines, (MIDDLE_X - DEADZONE, 0), 
        (MIDDLE_X - DEADZONE, RESOLUTION[1]), (255, 0, 0), 1)
cv2.line(deadzone_lines, (MIDDLE_X + DEADZONE, 0), 
        (MIDDLE_X + DEADZONE, RESOLUTION[1]), (255, 0, 0), 1)

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
def detect_objects(frame_bgr):
    if net is None:
        return frame_bgr, []
    
    # Create blob and run detection
    blob = cv2.dnn.blobFromImage(
        frame_bgr, 1/255.0, (BLOB_SIZE, BLOB_SIZE), 
        swapRB=True, crop=False
    )
    net.setInput(blob)
    detections = net.forward(output_layers)
    
    # Process detections
    boxes = []
    confidences = []
    class_ids = []
    class_names = []
    centers = []
    h, w = frame_bgr.shape[:2]
    
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > CONFIDENCE_THRESHOLD:
                # Scale bounding box to frame dimensions
                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                
                # Calculate top-left corner
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                # Store center point
                center = (int(centerX), int(centerY))
                centers.append(center)
                
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                class_names.append(classes[class_id])
    
    # Apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    
    # Process detected objects
    detected_objects = {}
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            center_x, center_y = centers[i]
            class_name = classes[class_ids[i]]

            size = w * h
            state = ""

            # Determine object state
            if size >= 60000:
                state = "CLOSE"
            elif center_x >= (MIDDLE_X - DEADZONE) and center_x <= (MIDDLE_X + DEADZONE):
                state = "MIDDLE"
            elif center_x < (MIDDLE_X - DEADZONE):
                state = "LEFT"
            elif center_x > (MIDDLE_X + DEADZONE):
                state = "RIGHT"
            
            # Draw bounding box if enabled
            if DRAW_BOXES:
                color = (0, 255, 0)  # Green boxes
                cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
                label = f"{class_name}: {confidences[i]:.2f} | {state}"
                cv2.putText(frame_bgr, label, (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw circle at center
            cv2.circle(frame_bgr, (center_x, center_y), 2, (0, 0, 255), -1)
            detected_objects.__setitem__(class_name, state)
    
    return frame_bgr, detected_objects

# ===== ASYNC DETECTION CLASS =====
class AsyncDetector:
    def __init__(self):
        self.active = False
        self.result_frame = None
        self.result_objects = {}
        self.thread = None
        self.lock = threading.Lock()
        
    def start_detection(self, frame_bgr):
        with self.lock:
            if self.active:
                return False
                
            self.active = True
            self.thread = threading.Thread(target=self._detect_thread, args=(frame_bgr.copy(),))
            self.thread.daemon = True
            self.thread.start()
            return True
            
    def _detect_thread(self, frame_bgr):
        processed_frame, objects = detect_objects(frame_bgr)
        with self.lock:
            self.result_frame = processed_frame
            self.result_objects = objects
            self.active = False
            
    def get_results(self):
        with self.lock:
            if self.result_frame is None:
                return None, []
            return self.result_frame, self.result_objects
            
    def is_active(self):
        with self.lock:
            return self.active

# ===== MAIN LOOP WITH ASYNC DETECTION =====
def object_detection(cam):
    try:
        cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Object Detection', *RESOLUTION)
        
        # Initialize async detector
        detector = AsyncDetector()
        last_detection_time = 0
        debounce_time = 1.0  # Minimum time between detections
        
        # Store the last processed frame
        last_processed_frame = None
        
        while True:
            # Capture frame
            frame = cam.capture_array("main")
            frame_rgb = frame.copy()  # Preserve RGB version
            
            # Convert to BGR for potential detection
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Check for joystick press
            for event in sense.stick.get_events():
                if event.action == 'pressed' and not detector.is_active():
                    current_time = time.time()
                    if current_time - last_detection_time > debounce_time:
                        last_detection_time = current_time
                        detector.start_detection(frame_bgr)
                        sense.clear(0, 255, 0)  # Green light when detecting
                        time.sleep(0.1)
                        sense.clear(0, 0, 0)
            
            # Check if detection is complete
            if not detector.is_active():
                processed_frame, objects = detector.get_results()
                if processed_frame is not None:
                    last_processed_frame = processed_frame
                    
                    if objects:
                        print(f"Detected: {', '.join(dict(objects))}")
                        for obj in objects.items():
                            obj_name = obj[0]
                            obj_state = obj[1]
                            print(obj_name, obj_state)
                            if (obj_state == "CLOSE"):
                                yapper.yap(f"There is a {obj_name} close.")
                                break

                            yapper.yap(f"There is a {obj_name} to the {obj_state}")
            
            # Prepare display frame
            if last_processed_frame is not None:
                display_frame = cv2.cvtColor(last_processed_frame, cv2.COLOR_BGR2RGB)
            else:
                display_frame = frame_rgb.copy()
            
            # Add deadzone lines
            cv2.addWeighted(display_frame, 1.0, deadzone_lines, 1.0, 0, display_frame)
            
            # Show detection status
            status = "DETECTING" if detector.is_active() else "READY"
            cv2.putText(display_frame, f"Status: {status}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Object Detection', display_frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        sense.clear()

if __name__ == "__main__":
    print("Starting object detection...")
    print("Press Sense HAT joystick to detect objects")
    print("Press 'q' to exit")
    camera = setup_cam()
    object_detection(camera)