PREVIEW = True # Enable / disable the camera preview

import os

if PREVIEW:
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

MODEL_PATH = MODEL_URL.split("/")[-1]
CONFIG_PATH = CONFIG_URL.split("/")[-1]
CLASSES_PATH = CLASSES_URL.split("/")[-1]

# Detection parameters
DRAW_BOXES = True
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3
RESOLUTION = (480, 360)
MIDDLE_DEADZONE = 25
CLOSE_THRESHOLD = 50000
MIDDLE_X = RESOLUTION[0] // 2
BLOB_SIZE = 320  # Reduced network input size

# ===== INITIALIZATION =====
def download_file(url):
    path = url.split("/")[-1]
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

# Precompute deadzone lines
deadzone_lines = np.zeros((RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8)
cv2.line(deadzone_lines, (MIDDLE_X - MIDDLE_DEADZONE, 0), 
        (MIDDLE_X - MIDDLE_DEADZONE, RESOLUTION[1]), (255, 0, 0), 1)
cv2.line(deadzone_lines, (MIDDLE_X + MIDDLE_DEADZONE, 0), 
        (MIDDLE_X + MIDDLE_DEADZONE, RESOLUTION[1]), (255, 0, 0), 1)

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
                
                boxes.append([x, y, int(width), int(height)])
                
                # Store center point
                center = (int(centerX), int(centerY))
                centers.append(center)
                confidences.append(float(confidence))
                class_ids.append(class_id)
                class_names.append(classes[class_id])
    
    # Process detected objects only if there are valid boxes
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
                if size >= CLOSE_THRESHOLD:
                    state = "CLOSE"
                elif center_x >= (MIDDLE_X - MIDDLE_DEADZONE) and center_x <= (MIDDLE_X + MIDDLE_DEADZONE):
                    state = "MIDDLE"
                elif center_x < (MIDDLE_X - MIDDLE_DEADZONE):
                    state = "LEFT"
                elif center_x > (MIDDLE_X + MIDDLE_DEADZONE):
                    state = "RIGHT"

                detected_objects.append({"name": class_name, "state": state})

                if PREVIEW:
                    # Draw bounding box if enabled
                    if DRAW_BOXES:
                        color = (0, 255, 0)  # Green boxes
                        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
                        label = f"{class_name}: {confidences[i]:.2f} | {state}"
                        cv2.putText(frame_bgr, label, (x, y-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Draw circle at center
                    cv2.circle(frame_bgr, (center_x, center_y), 2, color, -1)
    
    if PREVIEW: return frame_bgr, detected_objects
    return detected_objects

# ===== ASYNC DETECTION CLASS =====
class AsyncDetector:
    def __init__(self):
        self.active = False
        if PREVIEW: self.result_frame = None
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
        if PREVIEW:
            processed_frame, objects = detect_objects(frame_bgr)
        else:
            objects = detect_objects(frame_bgr)
        
        with self.lock:
            if PREVIEW: self.result_frame = processed_frame
            self.result_objects = objects
            self.active = False
            
    def get_results(self):
        with self.lock:
            if PREVIEW: return self.result_frame, self.result_objects
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
        grouped_objects[obj["name"]].append(obj["state"]) # {"person": ["LEFT"]}
    
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
            state_counts[state][obj_name] += 1 # {'RIGHT': {'person': 2, 'clock': 1, 'tvmonitor': 1}), 'LEFT': {'person': 1})}
    
    # Build phrases for LEFT and RIGHT
    phrases = []
    for state in ["MIDDLE", "LEFT", "RIGHT"]: # ["MIDDLE", "LEFT", "RIGHT"] => "MIDDLE" then "LEFT" then "RIGHT"
        # Gets all objects bases on state (left or right)
        objs = state_counts[state] # {'person': 2, 'clock': 1, 'tvmonitor': 1}
        if objs:
            obj_phrases = []
            for obj_name, count in objs.items():
                if obj_name == "person":
                    obj_name = "people" if count > 1 else "person"
                elif obj_name == "tvmonitor":
                    obj_name = "screen"
                obj_phrases.append(f"{count if count > 1 else 'a'} {obj_name}{'s' if count > 1 and obj_name != 'people' else ''}") # '2 people', '1 car'
            if state == "LEFT":
                direction = "to your left"
            elif state == "RIGHT":
                direction = "to your right"
            else: # Not left or right so has to be middle
                direction = "in front of you"
            phrases.append(f"{' and '.join(obj_phrases)} {direction}") # ' and '.join(obj_phrases) => '2 people and 1 car' => '2 people and 1 car to your left'
    if phrases:
        async_tts.start_tts("There " + ("is" if len(phrases) == 1 or phrases[0].__contains__("person") else "are") + " " + " and ".join(phrases) + ".")
    else:
        # If only MIDDLE objects, list them
        middle_objs = state_counts["MIDDLE"]
        if middle_objs:
            obj_phrases = []
            for obj_name, count in middle_objs.items():
                obj_phrases.append(f"{count} {obj_name}{'s' if count > 1 else ''}")
            async_tts.start_tts("There " + ("is" if sum(middle_objs.values()) == 1 else "are") + " " + " and ".join(obj_phrases) + " in front of you.")

# ===== MAIN LOOP WITH ASYNC DETECTION =====
def object_detection(cam):
    try:
        if PREVIEW:
            cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Object Detection', *RESOLUTION)

        # Initialize async detector
        detector = AsyncDetector()
        last_press_time = 0
        debounce_time = 0.5  # Half-second debounce
        
        if PREVIEW:
            # Store the last processed frame
            last_processed_frame = None
        
        # Flag to prevent repeated detection and TTS for the same joystick press
        detection_triggered = False
        
        while True:
            if PREVIEW:
                # Capture current frame
                current_frame = cam.capture_array("main")
                current_frame_rgb = current_frame.copy()
                
                # Prepare display frame
                if last_processed_frame is not None:
                    display_frame = cv2.cvtColor(last_processed_frame, cv2.COLOR_BGR2RGB)
                else:
                    display_frame = current_frame_rgb.copy()
            
                # Add deadzone lines
                cv2.addWeighted(display_frame, 1.0, deadzone_lines, 1.0, 0, display_frame)
            
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
                if PREVIEW:
                    processed_frame, objects = detector.get_results()
                    last_processed_frame = processed_frame
                else: 
                    objects = detector.get_results()

                # Perform detection and TTS only once per joystick press
                if objects and not detection_triggered:
                    process_tts(objects)
                    detection_triggered = True

            if PREVIEW:
                # Show frame
                cv2.imshow('Object Detection', display_frame)
                
                # Exit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cam.stop()
        if PREVIEW: cv2.destroyAllWindows()
        sense.clear()

if __name__ == "__main__":
    print("Starting object detection...")
    print("Press Sense HAT joystick to detect objects")
    if PREVIEW: print("Press 'q' to exit")
    camera = setup_cam()
    object_detection(camera)
