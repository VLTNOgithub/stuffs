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

from yolov5 import detect  # Import YOLOv5 detection module
import torch

# ===== CONFIGURATION =====
MODEL_PATH = "yolov5s.pt"  # Use YOLOv5 small model (pretrained)

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
# Initialize model
try:
    model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH)
    print(f"YOLOv5 model loaded successfully")
except Exception as e:
    print(f"Failed to load YOLOv5 model: {e}")
    model = None

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
    if model is None:
        return frame_bgr, []

    # Convert frame to RGB for YOLOv5
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Perform detection
    results = model(frame_rgb)

    # Get detection results
    detected_objects = []
    for *box, conf, cls in results.xyxy[0]:  # xyxy format: [x1, y1, x2, y2, confidence, class]
        x1, y1, x2, y2 = map(int, box)
        class_name = model.names[int(cls)]
        confidence = float(conf)

        # Determine object state
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        size = (x2 - x1) * (y2 - y1)
        state = ""

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
            # Draw boxes if enabled
            if DRAW_BOXES:
                color = (0, 255, 0)
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name}: {confidence:.2f} | {state}"
                cv2.putText(frame_bgr, label, (x1, y1 - 5),
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
        grouped_objects[obj["name"]].append(obj["state"]) # e.g. {"person": ["LEFT"]}
    
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
            state_counts[state][obj_name] += 1 # e.g. {"RIGHT": {"person": 2, "clock": 1, "tvmonitor": 1}), "LEFT": {"person": 1})}
    
    # Build phrases
    phrases = []
    for state in ["MIDDLE", "LEFT", "RIGHT"]: # ["MIDDLE", "LEFT", "RIGHT"] => "MIDDLE" then "LEFT" then "RIGHT"
        objs = state_counts[state] # e.g. {"person": 2, "clock": 1, "tvmonitor": 1}
        if objs:
            obj_phrases = []
            for obj_name, count in objs.items():
                if obj_name == "person":
                    obj_name = "people" if count > 1 else "person"
                elif obj_name == "tvmonitor":
                    obj_name = "screen"
                obj_phrases.append(f"{count if count > 1 else 'a'} {obj_name}{'s' if count > 1 and obj_name != 'people' else ''}") # "2 people", "1 car"
            if state == "LEFT":
                direction = "to your left"
            elif state == "RIGHT":
                direction = "to your right"
            else: # Not left or right so has to be middle
                direction = "in front of you"
            phrases.append(f"{' and '.join(obj_phrases)} {direction}") # e.g. " and ".join(obj_phrases) => "2 people and 1 car to your left"
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
            cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Object Detection", *RESOLUTION)

        # Initialize async detector
        detector = AsyncDetector()
        last_press_time = 0
        debounce_time = 0.5  # Half-second debounce

        cv2.imshow("Object Detection", cam.capture_array("main"))
        
        while True:
            # Check for joystick press debounce
            current_time = time.time()
            events = sense.stick.get_events()
            
            if events and (current_time - last_press_time) > debounce_time:
                for event in events:
                    if event.action == "pressed" and not detector.is_active():
                        last_press_time = current_time
                        
                        # Capture a new frame specifically for this detection
                        detection_frame = cam.capture_array("main")
                        detection_frame_bgr = cv2.cvtColor(detection_frame, cv2.COLOR_RGB2BGR)

                        detector.start_detection(detection_frame_bgr)
                        
                        sense.clear(0, 255, 0)
                        time.sleep(0.1)
                        sense.clear(0, 0, 0)

                        while not detector.is_active():
                            if PREVIEW:
                                frame, objects = detector.get_results()

                                print(frame)

                                if objects:
                                    process_tts(objects)

                                # Add deadzone lines
                                cv2.addWeighted(detection_frame, 1.0, deadzone_lines, 1.0, 0, detection_frame)

                                cv2.imshow("Object Detection", frame)
                        
                        break


            if PREVIEW:
                # Exit on q
                if cv2.waitKey(1) & 0xFF == ord("q"):
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