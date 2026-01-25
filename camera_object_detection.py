import cv2
import numpy as np
import time
import RPi.GPIO as GPIO
import math
from picamera2 import Picamera2

# =========================
# GPIO Setup
# =========================
BUTTON_PIN = 17
SERVO_PINS = [5, 6, 12, 13]
RELAY_A_PIN = 23
RELAY_B_PIN = 24

# Relay configuration
RELAY_A_ACTIVE_LOW = True
RELAY_B_ACTIVE_LOW = False

# =========================
# CONFIGURABLE SETTINGS
# =========================
# Enable/Disable Relay B
RELAY_B_ENABLED = False  # Set to False to disable Relay B completely
# Enable/Disable System Busy blocking
SYSTEM_BUSY_ENABLED = True  # Set to False to allow measurements while servo/relay is active

# Durations (in seconds)
BUTTON_ON_DURATION_18 = 1.0  # Relay A duration when button pressed
BUTTON_ON_DURATION_22 = 1.0  # Relay B duration when button pressed

# Unified activation times (in seconds) - same for both servo AND Relay B
ACTIVATION_DURATIONS = {
    5: 10.0,   # When object goes to servo 5: servo AND Relay B stay active for 2.0s
    6: 10.0,   # When object goes to servo 6: servo AND Relay B stay active for 2.0s  
    12: 10.0,  # When object goes to servo 12: servo AND Relay B stay active for 2.0s
    13: 10.0   # When object goes to servo 13: servo AND Relay B stay active for 2.0s
}

# Servo Positions
SERVO_POSITIONS = {
    5:  {"start": 150,  "end": 100},
    6:  {"start": 150,  "end": 100},
    12: {"start": 30,   "end": 80},
    13: {"start": 30,   "end": 80},
}

BUTTON_DEBOUNCE_SEC = 0.05

# Object Detection Parameters
BLUR_KSIZE = (7, 7)
MORPH_KSIZE = (5, 5)
OPEN_ITER = 1
CLOSE_ITER = 1
ADAPTIVE_BLOCK_SIZE = 19
ADAPTIVE_C = 5
AREA_MIN = 2000
DOWNSCALE_FACTOR = 0.5

# ROI Configuration
USE_ROI = True
ROI_RECT = (100, 100, 1050, 520)  # (x, y, width, height)

# 2-Frame Confirmation
_CONFIRMATION_FRAMES = 2
_LENGTH_TOLERANCE_CM = 0.3
_CONFIRMATION_DURATION_SEC = 0.5  # Minimum time span for confirmation

# ArUco Parameters
MARKER_SIZE_CM = 5.00
CALIB_K = 1.00
SCALE_EMA_ALPHA = 0.3

# Stability tuning
LENGTH_EMA_ALPHA = 0.2
HYSTERESIS_CM = 0.6
LOCK_MS = 400
MAX_CENTER_JUMP = 120

# =========================
# Initialize GPIO
# =========================
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Setup pins
for pin in SERVO_PINS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Setup relays with proper initial states
if RELAY_A_ACTIVE_LOW:
    GPIO.setup(RELAY_A_PIN, GPIO.OUT, initial=GPIO.HIGH)
else:
    GPIO.setup(RELAY_A_PIN, GPIO.OUT, initial=GPIO.LOW)

if RELAY_B_ACTIVE_LOW:
    GPIO.setup(RELAY_B_PIN, GPIO.OUT, initial=GPIO.HIGH)
else:
    GPIO.setup(RELAY_B_PIN, GPIO.OUT, initial=GPIO.LOW)

# =========================
# Camera State and Switching
# =========================
_current_camera = "webcam"  # Start with webcam
_last_button_state = GPIO.input(BUTTON_PIN)
_last_stable_button_state = _last_button_state
_last_state_change_t = time.monotonic()

# Camera objects
webcam = None
picam2 = None
current_capture = None

# =========================
# Object Detection State Variables
# =========================
# Timing/state variables
_last_raw_button_state_obj = GPIO.input(BUTTON_PIN)
_last_stable_button_state_obj = _last_raw_button_state_obj
_last_state_change_t_obj = time.monotonic()

_relay_a_expiry_t = 0.0
_relay_b_button_expiry_t = 0.0
_relay_b_detect_expiry_t = 0.0
_servo_expiry_t = 0.0  # Timer for servo activation

# Object detection variables
pixel_cm_ratio = None
_last_scale = None
_stable_length_cm = None
_locked_until_t = 0.0
_prev_center = None
_frame_history = []
_active_servo_pin = None  # Initialize at module level
_relay_b_active = False  # Track if Relay B is currently active
_system_busy = False  # NEW: Track if system is busy (servo/relay active)

# =========================
# Utility Functions
# =========================
def _now():
    return time.monotonic()

def _ema(prev, new, alpha):
    return new if prev is None else (alpha * new + (1 - alpha) * prev)

def _order_box_points(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.int32)

def _draw_long_side(img, box, color=(0, 255, 255), thickness=3):
    tl, tr, br, bl = box
    len_top = np.linalg.norm(tr - tl)
    len_right = np.linalg.norm(br - tr)
    if len_top >= len_right:
        p1, p2 = tl, tr
        p3, p4 = bl, br
    else:
        p1, p2 = tr, br
        p3, p4 = tl, bl
    cv2.line(img, tuple(p1), tuple(p2), color, thickness)
    cv2.line(img, tuple(p3), tuple(p4), color, thickness)

# =========================
# Camera Switching Functions
# =========================
def poll_camera_switch_button():
    """Poll button with debouncing for camera switching"""
    global _last_button_state, _last_stable_button_state, _last_state_change_t
    global _current_camera
    
    now = _now()
    raw = GPIO.input(BUTTON_PIN)
    
    # Detect raw state change
    if raw != _last_button_state:
        _last_button_state = raw
        _last_state_change_t = now
    
    # Only update stable state after debounce period
    if now - _last_state_change_t >= BUTTON_DEBOUNCE_SEC:
        if raw != _last_stable_button_state:
            prev = _last_stable_button_state
            _last_stable_button_state = raw
            
            # Detect button PRESS (HIGH to LOW transition)
            if prev == GPIO.HIGH and raw == GPIO.LOW:
                # Toggle camera
                if _current_camera == "webcam":
                    _current_camera = "picam"
                    print("Switching to Raspberry Pi Camera")
                else:
                    _current_camera = "webcam"
                    print("Switching to USB Webcam")
                
                # Close current camera and open new one
                switch_camera(_current_camera)
                return True
    
    return False

def init_webcam():
    """Initialize USB webcam"""
    global webcam, current_capture
    
    try:
        if webcam is None or not webcam.isOpened():
            webcam = cv2.VideoCapture(0)
            if webcam.isOpened():
                webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                print("USB Webcam initialized")
                current_capture = webcam
                return True
            else:
                print("Failed to open USB webcam")
                return False
        current_capture = webcam
        return True
    except Exception as e:
        print(f"Failed to initialize USB webcam: {e}")
        return False

def init_picam():
    """Initialize Raspberry Pi Camera"""
    global picam2, current_capture
    
    try:
        if picam2 is None:
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(
                main={"size": (1280, 720), "format": "RGB888"}
            )
            picam2.configure(config)
            picam2.start()
            time.sleep(1)  # Warm up
            print("Raspberry Pi Camera initialized")
        
        current_capture = picam2
        return True
    except Exception as e:
        print(f"Failed to initialize Raspberry Pi Camera: {e}")
        return False

def switch_camera(camera_type):
    """Switch between cameras"""
    global current_capture
    
    # Initialize new camera
    if camera_type == "webcam":
        if init_webcam():
            return True
        else:
            print("Failed to switch to USB webcam")
            return False
    elif camera_type == "picam":
        if init_picam():
            return True
        else:
            print("Failed to switch to Raspberry Pi Camera")
            return False
    
    return False

def get_frame():
    """Get frame from current camera"""
    global current_capture, _current_camera
    
    if current_capture is None:
        return None
    
    try:
        if _current_camera == "webcam":
            ret, frame = current_capture.read()
            if not ret:
                print("Failed to grab frame from USB webcam")
                return None
            return frame
        else:  # picam
            frame = current_capture.capture_array()
            # Convert from RGB to BGR for OpenCV
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
    except Exception as e:
        print(f"Error getting frame from {_current_camera}: {e}")
        return None

# =========================
# Relay Control Functions (Object Detection Only)
# =========================
def _start_or_extend(expiry_ref_name, duration):
    """Only extends timers, doesn't activate relays directly"""
    global _relay_a_expiry_t, _relay_b_button_expiry_t, _relay_b_detect_expiry_t, _servo_expiry_t
    
    now = _now()
    if expiry_ref_name == "A":
        _relay_a_expiry_t = max(_relay_a_expiry_t, now + duration)
    elif expiry_ref_name == "B_BUTTON":
        _relay_b_button_expiry_t = max(_relay_b_button_expiry_t, now + duration)
    elif expiry_ref_name == "B_DETECT":
        _relay_b_detect_expiry_t = max(_relay_b_detect_expiry_t, now + duration)
    elif expiry_ref_name == "SERVO":
        _servo_expiry_t = max(_servo_expiry_t, now + duration)

def _drive_relay(pin, on, active_low):
    """Control relay output"""
    if active_low:
        # Active LOW relay: ON when pin is LOW
        GPIO.output(pin, GPIO.LOW if on else GPIO.HIGH)
    else:
        # Active HIGH relay: ON when pin is HIGH  
        GPIO.output(pin, GPIO.HIGH if on else GPIO.LOW)

def _apply_output_levels():
    """Actually applies the relay states based on timers"""
    global _relay_b_active, _active_servo_pin, _system_busy  # Add _system_busy here
    
    now = _now()
    
    # Relay A: Only ON if button timer is active
    a_on = now < _relay_a_expiry_t
    
    # Relay B: ON if either button timer OR detection timer is active (if enabled)
    if RELAY_B_ENABLED:
        b_on = now < max(_relay_b_button_expiry_t, _relay_b_detect_expiry_t)
    else:
        b_on = False  # Relay B disabled
    
    # Track if Relay B is currently active
    _relay_b_active = b_on
    
    # Apply relays
    _drive_relay(RELAY_A_PIN, a_on, RELAY_A_ACTIVE_LOW)
    _drive_relay(RELAY_B_PIN, b_on, RELAY_B_ACTIVE_LOW)
    
    # Check if servo timer has expired
    servo_on = now < _servo_expiry_t
    
    # Update system busy state - system is busy if servo OR relay B is active
    _system_busy = servo_on or (b_on and RELAY_B_ENABLED)
    
    if not servo_on and _active_servo_pin is not None:
        # Servo timer expired, return servo to start position
        _move_servo_to_start(_active_servo_pin)
        _active_servo_pin = None

    return a_on, b_on, servo_on

def poll_object_detection_button():
    """Polls button with debouncing and triggers relay timers on press"""
    global _last_raw_button_state_obj, _last_stable_button_state_obj, _last_state_change_t_obj

    now = _now()
    raw = GPIO.input(BUTTON_PIN)

    # Detect raw state change
    if raw != _last_raw_button_state_obj:
        _last_raw_button_state_obj = raw
        _last_state_change_t_obj = now

    # Only update stable state after debounce period
    if now - _last_state_change_t_obj >= BUTTON_DEBOUNCE_SEC:
        if raw != _last_stable_button_state_obj:
            prev = _last_stable_button_state_obj
            _last_stable_button_state_obj = raw
            
            # Detect button PRESS (HIGH to LOW transition, active-LOW button)
            if prev == GPIO.HIGH and raw == GPIO.LOW:
                # Button pressed: Start timers for BOTH relays
                _start_or_extend("A", BUTTON_ON_DURATION_18)
                _start_or_extend("B_BUTTON", BUTTON_ON_DURATION_22)
                return True
    
    return False

def pulse_relay_b_for_detection(duration):
    """Triggers Relay B for object detection with specified duration"""
    if RELAY_B_ENABLED:  # Only trigger if Relay B is enabled
        _start_or_extend("B_DETECT", duration)

# =========================
# Servo PWM Setup with New Position System
# =========================
servo_pwm = {}

def setup_servos():
    for pin in SERVO_PINS:
        pwm = GPIO.PWM(pin, 50)  # 50Hz
        # Start all servos at their "start" position
        start_pos = SERVO_POSITIONS[pin]["start"]
        duty = _angle_to_duty(start_pos)
        pwm.start(duty)
        servo_pwm[pin] = pwm
        time.sleep(0.1)  # Small delay between servo initialization

def stop_servos():
    # Return all servos to their start position before stopping
    for pin in SERVO_PINS:
        if pin in servo_pwm:
            _move_servo_to_start(pin)
            time.sleep(0.1)
    for pwm in servo_pwm.values():
        pwm.stop()

def _angle_to_duty(angle):
    # Map 0-180Â° -> 2.5-12.5 duty cycle
    angle = max(0, min(180, angle))  # Clamp angle to valid range
    return 2.5 + (angle / 180.0) * 10.0

def _move_servo(pin, angle):
    if pin in servo_pwm:
        duty = _angle_to_duty(angle)
        servo_pwm[pin].ChangeDutyCycle(duty)
        time.sleep(0.05)  # Small delay for servo movement

def _move_servo_to_start(pin):
    """Move servo to its start position"""
    if pin in SERVO_POSITIONS:
        start_pos = SERVO_POSITIONS[pin]["start"]
        _move_servo(pin, start_pos)

def _move_servo_to_end(pin):
    """Move servo to its end position"""
    if pin in SERVO_POSITIONS:
        end_pos = SERVO_POSITIONS[pin]["end"]
        _move_servo(pin, end_pos)

def _park_others(active_pin):
    """Move all other servos to their start position"""
    for p in SERVO_PINS:
        if p != active_pin and p in servo_pwm:
            _move_servo_to_start(p)

def park_all_servos():
    """Move all servos to their start position"""
    for p in SERVO_PINS:
        if p in servo_pwm:
            _move_servo_to_start(p)

# =========================
# 2-Frame Confirmation System - WITH BUSY CHECK
# =========================
def _check_length_confirmation(current_length_cm):
    """Check if we have 2 consecutive frames with the same length over 0.5 seconds"""
    global _frame_history
    
    # If system is busy (servo or relay active), don't process new confirmations (if enabled)
    if SYSTEM_BUSY_ENABLED and _system_busy:
        return None
    
    now = _now()
    
    # Clean old entries (older than 1 second)
    _frame_history = [(length, ts) for length, ts in _frame_history if now - ts < 1.0]
    
    # Add current measurement
    _frame_history.append((current_length_cm, now))
    
    # Keep only last N frames (N = CONFIRMATION_FRAMES + 1 for buffer)
    if len(_frame_history) > _CONFIRMATION_FRAMES + 1:
        _frame_history = _frame_history[-(_CONFIRMATION_FRAMES + 1):]
    
    # Check if we have enough frames
    if len(_frame_history) >= _CONFIRMATION_FRAMES:
        # Get the last N frames
        recent_frames = _frame_history[-_CONFIRMATION_FRAMES:]
        
        # Check if all recent frames have approximately the same length
        lengths = [frame[0] for frame in recent_frames]
        min_length = min(lengths)
        max_length = max(lengths)
        
        # Check time span between first and last frame
        time_span = recent_frames[-1][1] - recent_frames[0][1]
        
        # If all lengths are within tolerance AND time span is at least 0.5 seconds, confirmation achieved
        if (max_length - min_length) <= _LENGTH_TOLERANCE_CM and time_span >= _CONFIRMATION_DURATION_SEC:
            # Return the average length
            return np.mean(lengths)
    
    return None  # No confirmation yet

def control_servos_with_confirmation(length_cm):
    """Control servos only after 2-frame confirmation"""
    global _active_servo_pin, _frame_history, _system_busy
    
    # If system is already busy (servo or relay active), don't start new actions (if enabled)
    if SYSTEM_BUSY_ENABLED and _system_busy:
        return length_cm, False, 0.0
    
    # Check for 2-frame confirmation
    confirmed_length = _check_length_confirmation(length_cm)
    
    if confirmed_length is not None:
        # Confirmation achieved! Now control servos
        if 8.7 <= confirmed_length <= 10.0:
            sel = 5
        elif 10.0 < confirmed_length <= 11.5:
            sel = 6
        elif 11.5 < confirmed_length <= 14.3:
            sel = 12
        else:
            sel = 13
        
        # Get unified activation duration from config
        unified_duration = ACTIVATION_DURATIONS.get(sel, 2.0)
        
        # Move servo to END position and start timer
        _move_servo_to_end(sel)
        _park_others(sel)
        _active_servo_pin = sel
        _start_or_extend("SERVO", unified_duration)
        
        # Activate Relay B with SAME duration (if enabled)
        if RELAY_B_ENABLED:
            pulse_relay_b_for_detection(unified_duration)
        
        # Clear history after successful confirmation to avoid repeated triggers
        _frame_history = []
        
        return confirmed_length, True, unified_duration  # Return confirmed length, success flag, and duration
    else:
        # No confirmation yet, check if servo timer has expired
        if _active_servo_pin is not None and _now() >= _servo_expiry_t:
            # Timer expired, return servo to start position
            _move_servo_to_start(_active_servo_pin)
            _active_servo_pin = None
        
        return length_cm, False, 0.0  # Return current length and failure flag

# =========================
# Object Detection Classes
# =========================
class DetectorObj:
    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KSIZE)

    def detect_objects(self, frame_small, area_min_small):
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, BLUR_KSIZE, 0)
        mask = cv2.adaptiveThreshold(
            gray_blur, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
            ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=OPEN_ITER)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=CLOSE_ITER)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cnt for cnt in contours if cv2.contourArea(cnt) > area_min_small]

# Initialize detectors
detector = DetectorObj()
parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# =========================
# Object Detection Processing (Webcam Only) - WITH BUSY CHECK
# =========================
def process_object_detection(frame):
    """Process object detection only on webcam frames"""
    global pixel_cm_ratio, _last_scale
    global _stable_length_cm, _locked_until_t, _prev_center
    global _frame_history, _active_servo_pin, _relay_b_active, _system_busy

    img_full = frame.copy()

    # Apply relay outputs and get current states
    a_on, b_on, servo_on = _apply_output_levels()
    
    # Check if system is busy (servo active OR relay B active)
    system_busy = servo_on or (b_on and RELAY_B_ENABLED)
    
    # Store busy state globally
    _system_busy = system_busy
    
    # If system is busy AND system busy blocking is enabled, skip ALL object detection processing
    if SYSTEM_BUSY_ENABLED and system_busy:
        # Display system busy message
        cv2.putText(img_full, "SYSTEM BUSY - Waiting for timer...", 
                    (img_full.shape[1]//2 - 250, img_full.shape[0]//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        # Show what's active
        if servo_on and _active_servo_pin is not None:
            remaining = max(0, _servo_expiry_t - _now())
            servo_text = f"Servo {_active_servo_pin} active: {remaining:.1f}s remaining"
            cv2.putText(img_full, servo_text, 
                        (img_full.shape[1]//2 - 200, img_full.shape[0]//2 + 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        if b_on and RELAY_B_ENABLED:
            relay_remaining = max(0, max(_relay_b_button_expiry_t, _relay_b_detect_expiry_t) - _now())
            relay_text = f"Relay B active: {relay_remaining:.1f}s remaining"
            y_offset = 100 if servo_on and _active_servo_pin is not None else 50
            cv2.putText(img_full, relay_text, 
                        (img_full.shape[1]//2 - 200, img_full.shape[0]//2 + y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Display minimal status
        display_minimal_status(img_full, a_on, b_on, servo_on)
        
        # Add camera indicator
        cv2.putText(img_full, "USB Webcam - System BUSY", 
                    (img_full.shape[1] - 450, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 0, 255), 2)
        
        return img_full

    # ---------- ArUco detection for scale ----------
    corners, ids, _ = aruco_detector.detectMarkers(img_full)
    aruco_mask = None
    
    if corners:
        # Create a mask to exclude ArUco marker areas from object detection
        aruco_mask = np.zeros(img_full.shape[:2], dtype=np.uint8)
        int_corners = np.intp(corners)
        
        # Draw filled polygons on the mask where ArUco markers are detected
        for corner in int_corners:
            cv2.fillPoly(aruco_mask, corner, 255)
            
        # Also draw the outlines for visualization
        cv2.polylines(img_full, int_corners, True, (0, 255, 0), 5)
        
        # Calculate pixel_cm_ratio from ArUco marker
        pts = corners[0][0]
        d01 = np.linalg.norm(pts[1] - pts[0])
        d12 = np.linalg.norm(pts[2] - pts[1])
        d23 = np.linalg.norm(pts[3] - pts[2])
        d30 = np.linalg.norm(pts[0] - pts[3])
        mean_side_px = (d01 + d12 + d23 + d30) / 4.0

        measured_px_per_cm = mean_side_px / float(MARKER_SIZE_CM)
        _last_scale = _ema(_last_scale, measured_px_per_cm, SCALE_EMA_ALPHA)
        pixel_cm_ratio = _last_scale
        
        cv2.putText(img_full, f"ArUco: px/cm = {pixel_cm_ratio:.2f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        if pixel_cm_ratio:
            cv2.putText(img_full, f"No ArUco: using px/cm = {pixel_cm_ratio:.2f}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        else:
            cv2.putText(img_full, "No ArUco: waiting for calibration",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

    # ---------- ROI & downscale ----------
    if USE_ROI:
        x0, y0, rw, rh = ROI_RECT
        x1, y1 = x0 + rw, y0 + rh
        offx, offy = x0, y0
        proc_img_full = img_full[y0:y1, x0:x1]
        # Draw ROI rectangle
        cv2.rectangle(img_full, (x0, y0), (x1, y1), (0, 200, 255), 2)
        cv2.putText(img_full, "ROI Active", (x0 + 10, y0 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    else:
        offx = offy = 0
        proc_img_full = img_full

    fx = fy = float(DOWNSCALE_FACTOR)
    if not (0 < fx < 1):
        fx = fy = 1.0
    
    proc_small = cv2.resize(proc_img_full, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    area_min_small = AREA_MIN * (fx * fy)

    # ---------- Object detection with ArUco exclusion ----------
    contours_small = detector.detect_objects(proc_small, area_min_small)

    rects_full = []
    if contours_small:
        scale_back = 1.0 / fx
        for cnt_s in contours_small:
            cnt_full = (cnt_s.astype(np.float32) * scale_back) + np.array([[[offx, offy]]], dtype=np.float32)
            rect = cv2.minAreaRect(cnt_full)
            (cx, cy), (w, h), angle = rect
            
            # Check if this contour is inside an ArUco marker area
            is_inside_aruco = False
            if aruco_mask is not None:
                # Convert center point to integer
                center_int = (int(cx), int(cy))
                # Check if the center is within the ArUco mask
                if (0 <= center_int[0] < aruco_mask.shape[1] and 
                    0 <= center_int[1] < aruco_mask.shape[0]):
                    if aruco_mask[center_int[1], center_int[0]] > 0:
                        is_inside_aruco = True
            
            # Only add to rects_full if NOT inside ArUco marker
            if not is_inside_aruco:
                rects_full.append((cx, cy, w, h, angle))

    chosen = None
    now = _now()

    if rects_full:
        if _prev_center is not None:
            cx_prev, cy_prev = _prev_center
            rects_sorted = sorted(
                rects_full,
                key=lambda r: (np.hypot(r[0] - cx_prev, r[1] - cy_prev), -max(r[2], r[3]))
            )
            candidate = rects_sorted[0]
            dist = np.hypot(candidate[0] - cx_prev, candidate[1] - cy_prev)
            if dist <= MAX_CENTER_JUMP or now < _locked_until_t:
                chosen = candidate
            else:
                if now >= _locked_until_t:
                    chosen = candidate
                    _locked_until_t = now + (LOCK_MS / 1000.0)
        else:
            chosen = max(rects_full, key=lambda r: max(r[2], r[3]))
            _locked_until_t = now + (LOCK_MS / 1000.0)

    confirmed = False
    confirmed_length = None
    activation_duration = 0.0
    
    if chosen and pixel_cm_ratio:
        cx, cy, w, h, angle = chosen
        rect_s = ((cx, cy), (w, h), angle)
        box = cv2.boxPoints(rect_s).astype(np.int32)
        box = _order_box_points(box)

        longer_side_px = max(w, h)
        raw_length_cm = (float(longer_side_px) / float(pixel_cm_ratio)) * float(CALIB_K)

        if _stable_length_cm is None:
            _stable_length_cm = raw_length_cm
        else:
            if abs(raw_length_cm - _stable_length_cm) >= HYSTERESIS_CM:
                _stable_length_cm = _ema(_stable_length_cm, raw_length_cm, LENGTH_EMA_ALPHA)

        cv2.polylines(img_full, [box], True, (255, 0, 0), 2)
        _draw_long_side(img_full, box, (0, 255, 255), 3)

        cv2.circle(img_full, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        
        # Check for 2-frame confirmation and control servos
        confirmed_length, confirmed, activation_duration = control_servos_with_confirmation(_stable_length_cm)
        
        # Display confirmation status with duration info
        if confirmed:
            label_len = f"Length {round(confirmed_length, 1)} cm"
            color = (0, 255, 0)  # Green for confirmed
            # Add duration info
            duration_text = f"Active for {activation_duration}s"
            cv2.putText(img_full, duration_text, (int(cx - 100), int(cy + 45)),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 200, 255), 2)
        else:
            # Show confirmation progress (frames and time)
            progress = min(len(_frame_history), _CONFIRMATION_FRAMES)
            time_elapsed = 0.0
            if len(_frame_history) >= 2:
                time_elapsed = _frame_history[-1][1] - _frame_history[0][1]
            label_len = f"Length {round(_stable_length_cm, 1)} cm [{progress}/{_CONFIRMATION_FRAMES}, {time_elapsed:.1f}s/{_CONFIRMATION_DURATION_SEC}s]"
            color = (100, 200, 0)  # Yellow for in-progress
        
        cv2.putText(img_full, label_len, (int(cx - 100), int(cy + 15)),
                    cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

        if _prev_center is None or np.hypot(cx - _prev_center[0], cy - _prev_center[1]) > 3:
            _prev_center = (cx, cy)
            _locked_until_t = now + (LOCK_MS / 1000.0)

    else:
        _stable_length_cm = None
        _prev_center = None
        _frame_history = []  # Reset confirmation history when no object detected
        # Check if servo timer has expired
        if _active_servo_pin is not None and _now() >= _servo_expiry_t:
            # Timer expired, return servo to start position
            _move_servo_to_start(_active_servo_pin)
            _active_servo_pin = None

    # Display status information
    display_full_status(img_full, a_on, b_on, servo_on, confirmed, pixel_cm_ratio, activation_duration)
    
    # Add camera indicator
    status_text = "USB Webcam - Object Detection READY"
    if not RELAY_B_ENABLED:
        status_text += " (Relay B DISABLED)"
    cv2.putText(img_full, status_text, 
                (img_full.shape[1] - 500, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (0, 255, 0), 2)
    
    cv2.putText(img_full, "Press button to switch to Pi Camera", 
                (img_full.shape[1] - 500, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (100, 255, 100), 2)

    return img_full

def display_minimal_status(img, a_on, b_on, servo_on):
    """Display minimal status when system is busy"""
    global _active_servo_pin, _servo_expiry_t, _relay_b_button_expiry_t, _relay_b_detect_expiry_t  # Add these globals
    
    y = 60
    
    # Display system busy warning
    cv2.putText(img, "SYSTEM BUSY - Waiting for timer...", 
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    y += 40
    
    # Show relay states
    cv2.putText(img, f"Relay A: {'ON' if a_on else 'OFF'}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if a_on else (0, 0, 255), 2)
    y += 25
    cv2.putText(img, f"Relay B: {'ON' if b_on else 'OFF'}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if b_on else (0, 0, 255), 2)
    
    y += 30
    # Show servo state with remaining time
    if _active_servo_pin is not None and servo_on:
        remaining = max(0, _servo_expiry_t - _now())
        start_pos = SERVO_POSITIONS[_active_servo_pin]["start"]
        end_pos = SERVO_POSITIONS[_active_servo_pin]["end"]
        cv2.putText(img, f"Servo {_active_servo_pin}: {end_pos}Â° ({remaining:.1f}s)", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 25
        cv2.putText(img, f"Will return to {start_pos}Â°", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
    elif _active_servo_pin is not None:
        cv2.putText(img, f"Servo {_active_servo_pin}: Returning to start...", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
    else:
        cv2.putText(img, "All servos: START position", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def display_full_status(img, a_on, b_on, servo_on, confirmed, pixel_cm_ratio, activation_duration=0.0):
    """Display full status information"""
    global _active_servo_pin, _servo_expiry_t, _relay_b_button_expiry_t, _relay_b_detect_expiry_t  # Add these globals
    
    y = 60
    
    # Display calibration status
    if pixel_cm_ratio:
        calibration_status = f"Calibrated: {pixel_cm_ratio:.2f} px/cm"
        calibration_color = (200, 255, 200)
    else:
        calibration_status = "NOT CALIBRATED - Show ArUco marker"
        calibration_color = (100, 100, 255)
    
    cv2.putText(img, calibration_status, 
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, calibration_color, 2)
    y += 25
    
    # Display Relay B enabled status
    if RELAY_B_ENABLED:
        relay_b_status = "Relay B: ENABLED"
        relay_b_color = (100, 255, 100)
    else:
        relay_b_status = "Relay B: DISABLED"
        relay_b_color = (255, 100, 100)
    
    cv2.putText(img, relay_b_status, 
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, relay_b_color, 2)
    y += 25
    
    # Display system status
    if _system_busy:
        system_status = "System Status: BUSY âš ï¸"
        system_color = (0, 0, 255)
    else:
        system_status = "System Status: READY âœ“"
        system_color = (0, 255, 0)
    
    cv2.putText(img, system_status, 
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, system_color, 2)
    y += 25
    
    # Display confirmation requirements
    cv2.putText(img, f"Confirmation: {_CONFIRMATION_FRAMES} frames + {_CONFIRMATION_DURATION_SEC}s", 
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 100), 2)
    y += 25
    
    if pixel_cm_ratio:
        cv2.putText(img, f"CALIB_K: {CALIB_K:.3f}",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
        y += 25
    
    cv2.putText(img, f"Relay A (BTN only): {'ON' if a_on else 'OFF'}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if a_on else (0, 0, 255), 2)
    y += 25
    
    relay_b_text = f"Relay B (BTN+detect): {'ON' if b_on else 'OFF'}"
    if not RELAY_B_ENABLED:
        relay_b_text += " (DISABLED)"
    cv2.putText(img, relay_b_text, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if b_on else (0, 0, 255), 2)
    y += 25

    # Show servo positions table
    y += 10
    cv2.putText(img, "Servo Positions:", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 100), 2)
    y += 25
    
    for pin, positions in SERVO_POSITIONS.items():
        current_pin = (_active_servo_pin == pin)
        start_pos = positions["start"]
        end_pos = positions["end"]
        
        if current_pin and confirmed:
            pin_text = f"Servo {pin}: {start_pos}Â°â†’{end_pos}Â° âš¡"
            color = (0, 255, 0)
        elif current_pin:
            pin_text = f"Servo {pin}: {start_pos}Â°â†’{end_pos}Â° â†’"
            color = (255, 255, 0)
        else:
            pin_text = f"Servo {pin}: {start_pos}Â°â†’{end_pos}Â°"
            color = (200, 200, 200)
        
        cv2.putText(img, pin_text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 20

    y += 10
    # Show current servo state with timer
    if _active_servo_pin is not None and servo_on:
        remaining = max(0, _servo_expiry_t - _now())
        start_pos = SERVO_POSITIONS[_active_servo_pin]["start"]
        end_pos = SERVO_POSITIONS[_active_servo_pin]["end"]
        status_text = f"Servo {_active_servo_pin}: {end_pos}Â° ({remaining:.1f}s)"
        cv2.putText(img, status_text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 25
        
        # Show return position
        cv2.putText(img, f"Will return to {start_pos}Â°", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
        y += 25
        
        # If Relay B is also active, show that too
        if b_on and RELAY_B_ENABLED:
            relay_b_remaining = max(0, max(_relay_b_button_expiry_t, _relay_b_detect_expiry_t) - _now())
            relay_text = f"Relay B: ACTIVE ({relay_b_remaining:.1f}s remaining)"
            cv2.putText(img, relay_text, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    elif _active_servo_pin is not None:
        cv2.putText(img, f"Servo {_active_servo_pin}: Returning to start...", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
    else:
        cv2.putText(img, "All servos at START position", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def display_picam_view(frame):
    """Display Raspberry Pi Camera view (no object detection)"""
    img_full = frame.copy()
    
    # Add simple info overlay
    h, w = img_full.shape[:2]
    
    # Create a semi-transparent overlay for info
    overlay = img_full.copy()
    cv2.rectangle(overlay, (10, 10), (500, 240), (0, 0, 0), -1)
    img_full = cv2.addWeighted(overlay, 0.7, img_full, 0.3, 0)
    
    # Display camera info
    cv2.putText(img_full, "Raspberry Pi Camera - View Only", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    cv2.putText(img_full, f"Resolution: {w}x{h}", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    cv2.putText(img_full, "Press button to switch to Webcam (Object Detection)", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
    
    # Show Relay B status
    if RELAY_B_ENABLED:
        relay_status = "Relay B: ENABLED in Webcam mode"
        relay_color = (100, 255, 100)
    else:
        relay_status = "Relay B: DISABLED in Webcam mode"
        relay_color = (255, 100, 100)
    
    cv2.putText(img_full, relay_status, (20, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, relay_color, 2)
    
    # Show system behavior info
    cv2.putText(img_full, "Webcam Mode Behavior:", (20, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 100), 2)
    y = 185
    cv2.putText(img_full, "- System waits for timer completion", (30, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y += 20
    cv2.putText(img_full, "- No new detection during active period", (30, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return img_full

# =========================
# Main Function
# =========================
def main():
    global webcam, picam2
    
    print("=" * 60)
    print("Camera Switcher with Object Detection")
    print("=" * 60)
    print(f"Relay B: {'ENABLED' if RELAY_B_ENABLED else 'DISABLED'}")
    print(f"ROI: {'ENABLED' if USE_ROI else 'DISABLED'}")
    if USE_ROI:
        print(f"ROI Rect: {ROI_RECT}")
    print("\nSYSTEM BEHAVIOR:")
    print("- Waits for timer completion before new detection")
    print("- No object detection during servo/relay activation")
    print("\nSERVO POSITIONS:")
    for pin, positions in SERVO_POSITIONS.items():
        print(f"  Servo {pin}: {positions['start']}Â° â†’ {positions['end']}Â°")
    print("\nUNIFIED ACTIVATION DURATIONS (Servo & Relay B):")
    for pin, duration in ACTIVATION_DURATIONS.items():
        print(f"  Servo {pin}: {duration}s")
    print("\nButton Durations:")
    print(f"  Relay A (button): {BUTTON_ON_DURATION_18}s")
    print(f"  Relay B (button): {BUTTON_ON_DURATION_22}s")
    print("-" * 60)
    print("Starting with USB Webcam (Object Detection Active)")
    print("Press button on GPIO 17 to switch cameras")
    print("Press ESC to exit")
    print("=" * 60)
    
    # Setup servos (starts at their "start" position)
    setup_servos()
    
    # Initialize with webcam first
    if not init_webcam():
        print("Failed to initialize USB webcam!")
        print("Trying Raspberry Pi Camera instead...")
        if not init_picam():
            print("No cameras available!")
            return
    
    try:
        while True:
            # Check for camera switch button
            poll_camera_switch_button()
            
            # Get frame from current camera
            frame = get_frame()
            
            if frame is not None:
                if _current_camera == "webcam":
                    # Process object detection on webcam
                    poll_object_detection_button()
                    processed_frame = process_object_detection(frame)
                else:
                    # Just display Pi Camera view
                    processed_frame = display_picam_view(frame)
                
                # Resize and display
                processed_frame = cv2.resize(processed_frame, (1024, 768))
                cv2.imshow("Camera View - Press ESC to exit", processed_frame)
            
            # Check for ESC key
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                break
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        
        stop_servos()
        
        if webcam is not None:
            webcam.release()
            print("USB Webcam released")
        
        if picam2 is not None:
            picam2.stop()
            print("Raspberry Pi Camera stopped")
        
        GPIO.cleanup()
        cv2.destroyAllWindows()
        print("Program exited")

if __name__ == "__main__":
    main()
