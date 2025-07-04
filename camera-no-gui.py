import os
import time
import numpy as np
import requests
import threading
import cv2
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
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3
RESOLUTION = (480, 360)
DEADZONE = 8
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
        return []
    
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
                
                # Store center point
                center = (int(centerX), int(centerY))
                centers.append(center)
                confidences.append(float(confidence))
                class_ids.append(class_id)
                boxes.append([int(centerX - width / 2), int(centerY - height / 2), int(width), int(height)])
    
    # Apply non-max suppression only if there are valid boxes
    detected_objects = []
    if len(boxes) > 0 and len(confidences) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
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
                
                detected_objects.append({"name": class_name, "state": state})
    
    return detected_objects

# ===== ASYNC DETECTION CLASS =====
class AsyncDetector:
    def __init__(self):
        self.active = False
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
        objects = detect_objects(frame_bgr)
        with self.lock:
            self.result_objects = objects
            self.active = False
            
    def get_results(self):
        with self.lock:
            return self.result_objects
            
    def is_active(self):
        with self.lock:
            return self.active

class AsyncTTS:
    def __init__(self):
        self.active = False
        self.thread = None
        self.lock = threading.Lock()
        self.yapper = ypr.Yapper(use_stdout=True)
        
    def start_tts(self, text):
        with self.lock:
            self.active = True
            self.thread = threading.Thread(target=self._async_tts, args=(text,))
            self.thread.daemon = True
            self.thread.start()
            return True
    
    def _async_tts(self, text):
        with self.lock:
            self.yapper.yap(text)


def process_tts(objects):
    # Initialise async tts
    async_tts = AsyncTTS()

    # Group by name
    from collections import defaultdict
    grouped_objects = defaultdict(list)
    for obj in objects:
        grouped_objects[obj["name"]].append(obj["state"])  # {"person": ["LEFT"]}
    
    # Priority: CLOSE > MIDDLE > LEFT/RIGHT
    close_obj = None
    for obj_name, states in grouped_objects.items():
        if "CLOSE" in states:
            close_obj = obj_name
            break

    if close_obj:
        async_tts.start_tts(f"There is a {close_obj} in front of you.")
        return
    
    # Count all objects by state
    state_counts = defaultdict(lambda: defaultdict(int))
    for obj_name, states in grouped_objects.items():
        for state in states:
            state_counts[state][obj_name] += 1  # {'RIGHT': {'person': 2, 'clock': 1, 'tvmonitor': 1}), 'LEFT': {'person': 1})}
    
    # Build phrases for LEFT, RIGHT, and MIDDLE
    phrases = []
    for state in ["LEFT", "RIGHT", "MIDDLE"]:  # Include "MIDDLE" in the loop
        objs = state_counts[state]  # {'person': 2, 'clock': 1, 'tvmonitor': 1}
        if objs:
            obj_phrases = []
            for obj_name, count in objs.items():
                # Correctly pluralize "person" to "people" for counts > 1
                if obj_name == "person":
                    obj_name = "people" if count > 1 else "person"
                obj_phrases.append(f"{count} {obj_name}{'s' if count > 1 and obj_name != 'people' else ''}")  # '2 people', '1 car'
            if state == "LEFT":
                direction = "to your left"
            elif state == "RIGHT":
                direction = "to your right"
            else:  # MIDDLE
                direction = "in front of you"
            phrases.append(f"{' and '.join(obj_phrases)} {direction}")  # Combine phrases for each state
    
    if phrases:
        async_tts.start_tts("There " + ("is" if phrases[0].split(" ")[0] == 1 else "are") + " " + " and ".join(phrases) + ".")

# ===== MAIN LOOP WITH ASYNC DETECTION =====
def object_detection(cam):
    try:
        # Initialize async detector
        detector = AsyncDetector()
        last_press_time = 0
        debounce_time = 0.5  # Half-second debounce
        
        # Flag to prevent repeated detection and TTS for the same joystick press
        detection_triggered = False
        
        while True:
            # Check for joystick press with proper debounce
            current_time = time.time()
            events = sense.stick.get_events()
            
            if events and (current_time - last_press_time) > debounce_time:
                for event in events:
                    if event.action == 'pressed' and not detector.is_active():
                        last_press_time = current_time
                        
                        # Capture a NEW frame specifically for this detection
                        detection_frame = cam.capture_array("main")
                        detection_frame_bgr = cv2.cvtColor(detection_frame, cv2.COLOR_RGB2BGR)
                        
                        # Start detection on this new frame
                        detector.start_detection(detection_frame_bgr)
                        
                        # Reset detection trigger flag
                        detection_triggered = False
                        
                        # Visual feedback
                        sense.clear(0, 255, 0)  # Green light when detecting
                        time.sleep(0.1)
                        sense.clear(0, 0, 0)
                        break
            
            # Check if detection is complete
            if not detector.is_active():
                objects = detector.get_results()
                
                # Perform detection and TTS only once per joystick press
                if objects and not detection_triggered:
                    process_tts(objects)
                    detection_triggered = True
    finally:
        cam.stop()
        sense.clear()

if __name__ == "__main__":
    print("Starting object detection...")
    print("Press Sense HAT joystick to detect objects")
    camera = setup_cam()
    object_detection(camera)