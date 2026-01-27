import cv2
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from ultralytics import YOLO
import time
import numpy as np
from collections import deque

# =========================
# GPIO SETUP
# =========================
BUTTON_PIN = 17
RELAY_1_PIN = 23  # First relay for complete detection
RELAY_2_PIN = 24  # Second relay for complete detection

# Button logic configuration
USE_ACTIVE_HIGH = False  # Set to True for active HIGH, False for active LOW

GPIO.setmode(GPIO.BCM)

# Configure button pin based on active high/low preference
if USE_ACTIVE_HIGH:
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)  # Active HIGH logic
    print(f"Button configured as ACTIVE HIGH (press = HIGH)")
else:
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)    # Active LOW logic (default)
    print(f"Button configured as ACTIVE LOW (press = LOW)")

# Configure relay pins
GPIO.setup(RELAY_1_PIN, GPIO.OUT)
GPIO.setup(RELAY_2_PIN, GPIO.OUT)

# Turn off relays initially (assuming LOW = off, HIGH = on)
GPIO.output(RELAY_1_PIN, GPIO.LOW)
GPIO.output(RELAY_2_PIN, GPIO.LOW)
print(f"Relays initialized on pins {RELAY_1_PIN} and {RELAY_2_PIN} (LOW = OFF)")

# =========================
# RELAY CONFIGURATION
# =========================
RELAY_ACTIVATION_DURATION = 5.0  # How long relays stay active for complete detection (seconds)
_relay_expiry_t = 0.0  # Timer for relay activation
_relay_active = False  # Track if relays are currently active

# =========================
# SERVO CONFIGURATION
# =========================
# Servo Pins
SERVO_PINS = [5, 6, 12, 13]

# Servo Positions - Start and End angles for each servo
SERVO_POSITIONS = {
    5:  {"start": 150,  "end": 100},  # Servo 5: Start at 150Â°, move to 100Â° when activated
    6:  {"start": 150,  "end": 100},  # Servo 6: Start at 150Â°, move to 100Â° when activated
    12: {"start": 30,   "end": 80},   # Servo 12: Start at 30Â°, move to 80Â° when activated
    13: {"start": 30,   "end": 80},   # Servo 13: Start at 30Â°, move to 80Â° when activated
}

# Activation durations (in seconds) for each servo
ACTIVATION_DURATIONS = {
    5: 10.0,   # Servo 5 stays active for 10.0 seconds
    6: 10.0,   # Servo 6 stays active for 10.0 seconds  
    12: 10.0,  # Servo 12 stays active for 10.0 seconds
    13: 10.0   # Servo 13 stays active for 10.0 seconds
}

# =========================
# MEASUREMENT CONFIGURATION
# =========================
# TIME-BASED Confirmation
_CONFIRMATION_WINDOW_SEC = 1.0  # Check stability over 1 second window
_LENGTH_TOLERANCE_CM = 0.3      # Measurements must be within this tolerance

# Manual Pixel-per-cm Calibration
MANUAL_PX_PER_CM = 25.0  # Default value, adjust based on your setup
CALIB_K = 1.00  # Calibration multiplier (adjust if measurements are off)

# Stability tuning
LENGTH_EMA_ALPHA = 0.3  # Increased for Roboflow stability

# =========================
# SYSTEM STATE VARIABLES
# =========================
# Measurement variables
pixel_cm_ratio = MANUAL_PX_PER_CM
_stable_length_cm = None
_measurement_history = deque(maxlen=30)
_active_servo_pin = None
_system_busy = False
SYSTEM_BUSY_ENABLED = False

# Servo variables
servo_pwm = {}  # Dictionary to store PWM objects for each servo
_servo_expiry_t = 0.0  # Timer for servo activation

# Detection tracking
_last_detection_time = 0.0
_no_detection_threshold = 2.0  # 2 seconds without detection resets measurement

# Current camera mode
camera_mode = 0  # 0 = Webcam (samaral detection + measurement), 1 = Pi Camera (complete/incomplete + relays)
button_pressed = False

# Complete detection tracking (for Pi Camera mode)
_complete_detected = False
_complete_confirmation_history = deque(maxlen=10)  # Track complete detections for confirmation
_COMPLETE_CONFIRMATION_FRAMES = 5  # Need 5 consecutive frames of complete detection

# =========================
# TIME AND UTILITY FUNCTIONS
# =========================
def _now():
    """Get current time in seconds"""
    return time.time()

def _ema(prev, new, alpha):
    """Exponential Moving Average for smoothing measurements"""
    return new if prev is None else (alpha * new + (1 - alpha) * prev)

# =========================
# RELAY CONTROL FUNCTIONS
# =========================
def activate_relays(duration=RELAY_ACTIVATION_DURATION):
    """Activate both relays for specified duration"""
    global _relay_expiry_t, _relay_active
    
    print(f"âœ“ Activating relays 23 & 24 for {duration} seconds")
    
    # Turn ON both relays
    GPIO.output(RELAY_1_PIN, GPIO.HIGH)
    GPIO.output(RELAY_2_PIN, GPIO.HIGH)
    
    # Set expiry time
    _relay_expiry_t = _now() + duration
    _relay_active = True
    
    return True

def deactivate_relays():
    """Deactivate both relays"""
    global _relay_active
    
    if _relay_active:
        print("âœ“ Deactivating relays 23 & 24")
        GPIO.output(RELAY_1_PIN, GPIO.LOW)
        GPIO.output(RELAY_2_PIN, GPIO.LOW)
        _relay_active = False
        return True
    return False

def check_relay_timer():
    """Check if relay timer has expired and deactivate if needed"""
    global _relay_active
    
    if _relay_active and _now() >= _relay_expiry_t:
        deactivate_relays()
        return False
    return _relay_active

# =========================
# TIME-BASED CONFIRMATION SYSTEM
# =========================
def _check_length_confirmation(current_length_cm):
    """Check if measurements are stable within 1-second window"""
    global _measurement_history, _last_detection_time
    
    if SYSTEM_BUSY_ENABLED and _system_busy:
        return None
    
    # ONLY check confirmation in webcam mode (samaral mode)
    if camera_mode != 0:
        return None
    
    now = _now()
    
    # Check if too much time has passed since last detection
    if now - _last_detection_time > _no_detection_threshold:
        _measurement_history.clear()
        return None
    
    # Add current measurement with timestamp
    _measurement_history.append((current_length_cm, now))
    _last_detection_time = now
    
    # Remove measurements older than confirmation window
    while _measurement_history and now - _measurement_history[0][1] > _CONFIRMATION_WINDOW_SEC:
        _measurement_history.popleft()
    
    if len(_measurement_history) >= 2:
        # Get all measurements in the time window
        measurements = [length for length, ts in _measurement_history]
        
        min_length = min(measurements)
        max_length = max(measurements)
        
        # Check if measurements are stable within tolerance
        if (max_length - min_length) <= _LENGTH_TOLERANCE_CM:
            time_span = now - _measurement_history[0][1]
            if time_span >= _CONFIRMATION_WINDOW_SEC:
                confirmed_length = np.mean(measurements)
                print(f"? Measurement confirmed: {confirmed_length:.1f}cm (from {len(measurements)} samples)")
                return confirmed_length
    
    return None

# =========================
# SERVO PWM SETUP AND CONTROL (IMPROVED - NO DRIFT)
# =========================
def setup_servos():
    """Initialize all servos to their start positions"""
    print("Setting up servos...")
    for pin in SERVO_PINS:
        GPIO.setup(pin, GPIO.OUT)
        pwm = GPIO.PWM(pin, 50)  # 50Hz frequency for servos
        pwm.start(0)  # Start with 0 duty cycle
        servo_pwm[pin] = pwm
        time.sleep(0.1)
    
    # Move all servos to start positions
    for pin in SERVO_PINS:
        _move_servo_to_position(pin, SERVO_POSITIONS[pin]["start"])
    
    print(f"Servos initialized: {SERVO_PINS}")

def stop_servos():
    """Cleanup servo PWM"""
    print("Stopping servos...")
    for pwm in servo_pwm.values():
        pwm.stop()
    print("Servos stopped and cleaned up")

def _angle_to_duty(angle):
    """Convert angle (0-180) to duty cycle (2.5-12.5) - STABLE VERSION"""
    angle = max(0, min(180, angle))  # Clamp angle to valid range
    return 2.5 + (angle / 18.0)  # More stable calculation

def _move_servo_to_position(pin, angle):
    """Move specific servo to given angle and STOP signal to prevent drift"""
    if pin in servo_pwm:
        duty = _angle_to_duty(angle)
        servo_pwm[pin].ChangeDutyCycle(duty)
        time.sleep(0.2)  # Give time to move to position (0.2s is enough)
        servo_pwm[pin].ChangeDutyCycle(0)  # CRITICAL: Stop sending signal to hold position
        print(f"Servo {pin} moved to {angle}Â° and signal stopped")

def _move_servo_to_start(pin):
    """Move servo to its start position"""
    if pin in SERVO_POSITIONS:
        start_pos = SERVO_POSITIONS[pin]["start"]
        _move_servo_to_position(pin, start_pos)

def _move_servo_to_end(pin):
    """Move servo to its end position (activated state)"""
    if pin in SERVO_POSITIONS:
        end_pos = SERVO_POSITIONS[pin]["end"]
        _move_servo_to_position(pin, end_pos)

def _park_others(active_pin):
    """Move all other servos to their start positions"""
    for p in SERVO_PINS:
        if p != active_pin and p in servo_pwm:
            _move_servo_to_start(p)

# =========================
# SERVO ACTIVATION CONTROL - UPDATED (NO DRIFT)
# =========================
def control_servos_with_confirmation(length_cm, current_class):
    """Control servos based on measured length - ONLY for samaral class in webcam mode"""
    global _active_servo_pin, _measurement_history, _system_busy, _servo_expiry_t
    
    if SYSTEM_BUSY_ENABLED and _system_busy:
        return length_cm, False, 0.0
    
    # ONLY activate servos in webcam mode for samaral class
    if camera_mode != 0 or current_class.lower() != "samaral":
        return length_cm, False, 0.0
    
    confirmed_length = _check_length_confirmation(length_cm)
    
    if confirmed_length is not None:
        # Determine which servo to activate based on length
        if 8.7 <= confirmed_length <= 10.0:
            sel = 5
        elif 10.0 < confirmed_length <= 11.5:
            sel = 6
        elif 11.5 < confirmed_length <= 14.3:
            sel = 12
        else:
            sel = 13
        
        unified_duration = ACTIVATION_DURATIONS.get(sel, 2.0)
        
        print(f"âœ“ Activating Servo {sel} for {unified_duration}s (Length: {confirmed_length:.1f}cm)")
        
        # Activate the selected servo
        _move_servo_to_end(sel)      # Move to end position
        _park_others(sel)            # Move others to start positions
        _active_servo_pin = sel      # Set as active servo
        _servo_expiry_t = _now() + unified_duration  # Set expiry time
        
        # Clear measurement history after activation
        _measurement_history.clear()
        
        return confirmed_length, True, unified_duration
    else:
        # Check if servo timer has expired
        now = _now()
        if _active_servo_pin is not None and now >= _servo_expiry_t:
            print(f"âœ“ Returning Servo {_active_servo_pin} to start position")
            _move_servo_to_start(_active_servo_pin)  # Return to start position
            _active_servo_pin = None
        
        return length_cm, False, 0.0

def _apply_output_levels():
    """Check and apply servo states based on timers"""
    global _active_servo_pin, _system_busy
    
    now = _now()
    
    # Check if servo timer has expired
    servo_on = now < _servo_expiry_t
    
    # Update system busy state
    if SYSTEM_BUSY_ENABLED:
        _system_busy = servo_on
    
    # If servo timer expired and there's an active servo, return it to start
    if not servo_on and _active_servo_pin is not None:
        _move_servo_to_start(_active_servo_pin)
        _active_servo_pin = None

    return servo_on

# =========================
# MEASUREMENT CALCULATION FROM YOLO DETECTION
# =========================
def calculate_measurement_from_yolo(box_data):
    """Calculate physical length from YOLO bounding box"""
    global pixel_cm_ratio, _stable_length_cm
    
    if box_data is None or len(box_data) < 4:
        return None
    
    # Extract bounding box coordinates
    x1, y1, x2, y2 = box_data
    
    # Calculate width and height in pixels
    w_px = abs(x2 - x1)
    h_px = abs(y2 - y1)
    
    # Use the longer side for measurement
    longer_side_px = max(w_px, h_px)
    
    # Convert pixels to centimeters
    raw_length_cm = (float(longer_side_px) / float(pixel_cm_ratio)) * float(CALIB_K)
    
    # Apply EMA smoothing for stability
    if _stable_length_cm is None:
        _stable_length_cm = raw_length_cm
    else:
        _stable_length_cm = _ema(_stable_length_cm, raw_length_cm, LENGTH_EMA_ALPHA)
    
    return _stable_length_cm

# =========================
# DETECTION AND DISPLAY FUNCTIONS
# =========================
def process_detections(results, frame, confidence_threshold=0.5):
    """Process YOLO detections based on current camera mode"""
    global _stable_length_cm, _measurement_history, _last_detection_time
    global _relay_active, _relay_expiry_t
    
    annotated_frame = frame.copy()
    current_measurement = None
    confirmed_detection = False
    confirmed_length = None
    detected_class = None
    
    now = _now()
    
    # Reset measurement history if switching modes or no detection
    if now - _last_detection_time > _no_detection_threshold:
        _measurement_history.clear()
        if camera_mode != 0:  # Don't reset stable length for Pi Camera mode
            _stable_length_cm = None
    
    if hasattr(results[0], 'boxes') and results[0].boxes is not None:
        # Get bounding boxes, classes, and confidences
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        # Get class names from model
        class_names = results[0].names
        
        # Process each detection
        for i, (box, cls_idx, conf) in enumerate(zip(boxes, classes, confidences)):
            if conf < confidence_threshold:
                continue
                
            class_name = class_names[int(cls_idx)]
            detected_class = class_name.lower()
            
            # Extract box coordinates
            x1, y1, x2, y2 = box.astype(int)
            
            # Calculate center of bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Different processing based on camera mode
            if camera_mode == 0:  # WEBCAM MODE - Samaral detection with measurement
                if detected_class == "samaral":
                    # Calculate measurement only for samaral in webcam mode
                    current_measurement = calculate_measurement_from_yolo(box)
                    
                    if current_measurement is not None:
                        # Check for confirmation and control servos
                        confirmed_length, confirmed, activation_duration = control_servos_with_confirmation(
                            current_measurement, detected_class
                        )
                        confirmed_detection = confirmed
                        
                        # Display measurement info
                        display_measurement_info(
                            annotated_frame, 
                            center_x, center_y, 
                            current_measurement, 
                            confirmed, 
                            confirmed_length if confirmed else None
                        )
                    
                    # Update last detection time
                    _last_detection_time = now
                    
            else:  # PI CAMERA MODE - Complete/Incomplete detection with relay control
                if detected_class in ["complete", "incomplete"]:
                    # ACTIVATE RELAYS IMMEDIATELY WHEN "COMPLETE" IS DETECTED
                    if detected_class == "complete" and not _relay_active:
                        print(f"âœ“ 'Complete' detected! Activating relays for {RELAY_ACTIVATION_DURATION}s")
                        activate_relays()
                    
                    # Display Pi Camera info
                    display_pi_camera_info(
                        annotated_frame,
                        center_x, center_y,
                        detected_class,
                        conf,
                        _relay_active
                    )
                    
                    # Update last detection time
                    _last_detection_time = now
            
            # Draw bounding box with different colors based on mode and class
            if camera_mode == 0:  # Webcam mode
                if detected_class == "samaral":
                    box_color = (0, 255, 0) if confirmed_detection else (0, 165, 255)  # Green if confirmed, orange if measuring
                else:
                    box_color = (255, 255, 0)  # Yellow for other classes in webcam mode
            else:  # Pi Camera mode
                if detected_class == "complete":
                    box_color = (0, 255, 0)  # Green for complete
                elif detected_class == "incomplete":
                    box_color = (0, 0, 255)  # Red for incomplete
                else:
                    box_color = (255, 255, 255)  # White for other classes
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            if camera_mode == 0 and detected_class == "samaral" and current_measurement is not None:
                label += f" ({current_measurement:.1f}cm)"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 5), 
                          (x1 + label_size[0], y1), box_color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Add confirmation indicator for samaral in webcam mode
            if camera_mode == 0 and detected_class == "samaral" and confirmed_detection:
                cv2.circle(annotated_frame, (center_x, center_y), 15, (0, 255, 0), 3)
                cv2.putText(annotated_frame, "CONFIRMED!", 
                           (center_x - 60, center_y - 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return annotated_frame, current_measurement, confirmed_detection, confirmed_length, detected_class

def display_measurement_info(frame, x, y, stable_length_cm, confirmed=False, confirmed_length=None):
    """Display measurement information on the frame (webcam mode only)"""
    
    # Choose position for text (above bounding box)
    text_x = int(x - 200 if x > 250 else x + 10)
    text_y = int(y - 40 if y > 50 else y + 40)
    
    if confirmed and confirmed_length is not None:
        text = f"CONFIRMED: {confirmed_length:.1f}cm"
        color = (0, 255, 0)  # Green
        thickness = 3
    else:
        # Show measurement progress
        if _measurement_history:
            time_span = _now() - _measurement_history[0][1] if _measurement_history else 0
            progress = min(100, (time_span / _CONFIRMATION_WINDOW_SEC) * 100)
            text = f"Measuring: {stable_length_cm:.1f}cm [{progress:.0f}%]"
            color = (100, 200, 0)  # Light green
        else:
            text = f"Measuring: {stable_length_cm:.1f}cm"
            color = (100, 100, 255)  # Light blue
        thickness = 2
    
    # Draw text with background for better visibility
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, thickness)[0]
    cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), 
                  (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
    
    cv2.putText(frame, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)

def display_pi_camera_info(frame, x, y, detected_class, confidence, relay_active=False):
    """Display information for Pi Camera mode (complete/incomplete)"""
    
    text_x = int(x - 100 if x > 150 else x + 10)
    text_y = int(y - 40 if y > 50 else y + 40)
    
    if detected_class == "complete":
        text = f"âœ“ COMPLETE"
        color = (0, 255, 0)  # Green
    else:  # incomplete
        text = f"âœ— INCOMPLETE"
        color = (0, 0, 255)  # Red
    
    text += f" {confidence:.2f}"
    
    # Add relay status if active
    if relay_active and detected_class == "complete":
        text += " [RELAYS ON]"
        color = (0, 255, 255)  # Cyan when relays are active
    
    # Draw text with background
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), 
                  (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
    
    cv2.putText(frame, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def display_servo_status(img, servo_on):
    """Display servo status on the video feed"""
    global _active_servo_pin, _servo_expiry_t, _relay_active
    
    y = 180  # Starting Y position for status
    
    if camera_mode == 0:  # Webcam mode - show servo status
        if _active_servo_pin is not None and servo_on:
            remaining = max(0, _servo_expiry_t - _now())
            servo_text = f"SERVO {_active_servo_pin}: ACTIVE ({remaining:.1f}s)"
            cv2.putText(img, servo_text, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif _active_servo_pin is not None:
            cv2.putText(img, f"SERVO {_active_servo_pin}: RETURNING", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        else:
            cv2.putText(img, "SERVOS: READY", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:  # Pi Camera mode - show relay status
        if _relay_active:
            remaining = max(0, _relay_expiry_t - _now())
            relay_text = f"RELAYS 23&24: ACTIVE ({remaining:.1f}s)"
            cv2.putText(img, relay_text, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(img, "RELAYS: READY", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
    
    return y

# =========================
# CAMERA INITIALIZATION
# =========================
def init_webcam():
    """Initialize webcam (USB camera)"""
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return webcam

def init_picam():
    """Initialize Pi Camera"""
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (1280, 720)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()
    return picam2

# =========================
# MAIN PROGRAM
# =========================

# Setup servos first
print("Initializing GPIO, servos, and relays...")
setup_servos()
print(f"Relays initialized on pins {RELAY_1_PIN} and {RELAY_2_PIN}")
print(f"Relay activation duration: {RELAY_ACTIVATION_DURATION} seconds")

# Initialize cameras
print("\nInitializing cameras...")
webcam = init_webcam()
picam2 = init_picam()
print("Cameras initialized successfully!")

# Load YOLOv8 model
model = YOLO("/home/jelo/inheat/samaral.pt")

# Check webcam initialization
if not webcam.isOpened():
    print("Warning: Could not open webcam!")
    webcam = None

# Display class names from model
print("\n=== MODEL CLASSES ===")
for idx, name in model.names.items():
    print(f"  Class {idx}: {name}")
print("=====================")

# System configuration
print("\n=== SYSTEM CONFIGURATION ===")
print(f"Webcam Mode: Samaral detection with measurement & servo control")
print(f"Pi Camera Mode: Complete/Incomplete detection with relay control")
print(f"Relay activation: {RELAY_ACTIVATION_DURATION}s for 'complete' detection")
print(f"Complete confirmation: {_COMPLETE_CONFIRMATION_FRAMES} consecutive frames")
print(f"Pixel-to-cm ratio: {pixel_cm_ratio}")
print(f"Confirmation window: {_CONFIRMATION_WINDOW_SEC} seconds")
print(f"Length tolerance: {_LENGTH_TOLERANCE_CM} cm")
print("\n=== SERVO CONFIGURATION ===")
print(f"Servo pins: {SERVO_PINS}")
for pin in SERVO_PINS:
    print(f"  Servo {pin}: {SERVO_POSITIONS[pin]['start']}Â° â†’ {SERVO_POSITIONS[pin]['end']}Â° for {ACTIVATION_DURATIONS[pin]}s")
print("\n=== ACTIVATION RANGES ===")
print("8.7-10.0cm: Servo 5")
print("10.0-11.5cm: Servo 6") 
print("11.5-14.3cm: Servo 12")
print("Other: Servo 13")
print("==========================\n")

print("\nCamera Control System Started")
print("==============================")
print("Current mode: Webcam (Samaral detection + measurement)")
print("Press button on GPIO 17 to switch to Pi Camera (Complete/Incomplete + relays)")
print("Press 'q' to quit")
print("Press '+' to increase confidence threshold")
print("Press '-' to decrease confidence threshold")
print("Press 'c' to calibrate pixel ratio")
print("Press 's' to show/hide servo ranges")
print("Press 'r' to reset measurement")
print("Press 'd' to change relay duration")
print("\n")

# Variables for main loop
confidence_threshold = 0.5
show_servo_ranges = True
button_pressed = False

try:
    while True:
        # Check button state
        button_state = GPIO.input(BUTTON_PIN)
        
        # Detect button press based on active high/low configuration
        if USE_ACTIVE_HIGH:
            # Active HIGH logic: button pressed when HIGH
            if button_state == GPIO.HIGH and not button_pressed:
                button_pressed = True
                camera_mode = 1 - camera_mode  # Toggle between 0 and 1
                
                # Clear measurement history when switching modes
                _measurement_history.clear()
                _stable_length_cm = None
                _last_detection_time = _now()
                reset_complete_detection()
                
                # Deactivate relays if switching away from Pi Camera mode
                if camera_mode == 0 and _relay_active:
                    deactivate_relays()
                
                if camera_mode == 0:
                    print("\n" + "="*50)
                    print("SWITCHED TO: WEBCAM MODE")
                    print("- Samaral detection with measurement")
                    print("- Servo activation based on length")
                    print("="*50 + "\n")
                else:
                    print("\n" + "="*50)
                    print("SWITCHED TO: PI CAMERA MODE")
                    print("- Complete/Incomplete detection")
                    print("- Relays activate on 'complete' detection")
                    print(f"- Relay duration: {RELAY_ACTIVATION_DURATION}s")
                    print("="*50 + "\n")
                
                time.sleep(0.3)  # Debounce delay
                
            # Reset button state when released (active HIGH)
            elif button_state == GPIO.LOW and button_pressed:
                button_pressed = False
                
        else:
            # Active LOW logic: button pressed when LOW (default)
            if button_state == GPIO.LOW and not button_pressed:
                button_pressed = True
                camera_mode = 1 - camera_mode  # Toggle between 0 and 1
                
                # Clear measurement history when switching modes
                _measurement_history.clear()
                _stable_length_cm = None
                _last_detection_time = _now()
                reset_complete_detection()
                
                # Deactivate relays if switching away from Pi Camera mode
                if camera_mode == 0 and _relay_active:
                    deactivate_relays()
                
                if camera_mode == 0:
                    print("\n" + "="*50)
                    print("SWITCHED TO: WEBCAM MODE")
                    print("- Samaral detection with measurement")
                    print("- Servo activation based on length")
                    print("="*50 + "\n")
                else:
                    print("\n" + "="*50)
                    print("SWITCHED TO: PI CAMERA MODE")
                    print("- Complete/Incomplete detection")
                    print("- Relays activate on 'complete' detection")
                    print(f"- Relay duration: {RELAY_ACTIVATION_DURATION}s")
                    print("="*50 + "\n")
                
                time.sleep(0.3)  # Debounce delay
                
            # Reset button state when released (active LOW)
            elif button_state == GPIO.HIGH and button_pressed:
                button_pressed = False
        
        # Capture frame based on current mode
        if camera_mode == 0:  # Webcam mode
            if webcam is not None:
                ret, frame = webcam.read()
                if not ret:
                    print("Error: Could not read from webcam!")
                    # Create a blank frame with error message
                    height, width = 720, 1280
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    cv2.putText(frame, "Webcam Error", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("Dual Camera System", frame)
                    if cv2.waitKey(1) == ord("q"):
                        break
                    continue
            else:
                # Create a blank frame with error message
                height, width = 720, 1280
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(frame, "Webcam Not Available", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Dual Camera System", frame)
                if cv2.waitKey(1) == ord("q"):
                    break
                continue
        else:  # Pi Camera mode
            frame = picam2.capture_array()
            # Convert from RGB to BGR for OpenCV compatibility
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Run YOLO model with confidence threshold
        results = model(frame, conf=confidence_threshold)
        
        # Check servo states (only matters in webcam mode)
        servo_on = _apply_output_levels()
        
        # Check relay timer (only matters in Pi Camera mode)
        check_relay_timer()
        
        # Process detections based on current mode
        annotated_frame, current_measurement, confirmed_detection, confirmed_length, detected_class = process_detections(
            results, frame, confidence_threshold
        )
        
        # Get inference time and calculate FPS
        inference_time = results[0].speed['inference']
        fps = 1000 / inference_time if inference_time > 0 else 0
        
        # Prepare status text
        mode_name = "Webcam (Samaral)" if camera_mode == 0 else "PiCam (Complete/Incomplete)"
        status_text = f"FPS: {fps:.1f} | Mode: {mode_name} | Conf: {confidence_threshold:.2f}"
        if camera_mode == 0 and current_measurement is not None:
            status_text += f" | Length: {current_measurement:.1f}cm"
            if confirmed_detection:
                status_text += f" âœ“"
        elif camera_mode == 1 and _relay_active:
            remaining = max(0, _relay_expiry_t - _now())
            status_text += f" | Relays ON ({remaining:.1f}s)"  # <<< RELAY DURATION DISPLAY HERE
        
        # Display status
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(status_text, font, 0.6, 1)[0]
        text_x = annotated_frame.shape[1] - text_size[0] - 10
        text_y = text_size[1] + 10
        
        # Draw status background
        cv2.rectangle(annotated_frame, (text_x - 5, text_y - text_size[1] - 5), 
                      (annotated_frame.shape[1] - 5, text_y + 5), (0, 0, 0), -1)
        
        # Draw status text
        cv2.putText(annotated_frame, status_text, (text_x, text_y), 
                    font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Draw camera mode indicator
        if camera_mode == 0:
            mode_text = "WEBCAM: Samaral Detection & Servo Control"
            mode_color = (0, 255, 0)
        else:
            mode_text = f"PI CAMERA: Complete/Incomplete Detection & Relays ({RELAY_ACTIVATION_DURATION}s)"
            mode_color = (255, 165, 0)
        
        cv2.putText(annotated_frame, mode_text, (10, 30), 
                    font, 0.8, mode_color, 2, cv2.LINE_AA)
        
        # Display servo/relay status
        status_y = display_servo_status(annotated_frame, servo_on)
        
        # Display relay duration info for Pi Camera mode
        if camera_mode == 1:
            y = status_y + 40
            if _relay_active:
                remaining = max(0, _relay_expiry_t - _now())
                cv2.putText(annotated_frame, f"Relays active: {remaining:.1f}s remaining", 
                           (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y += 30
            else:
                cv2.putText(annotated_frame, f"Relays ready - will run {RELAY_ACTIVATION_DURATION}s on 'complete'", 
                           (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
                y += 30
        
        # Display measurement progress for webcam mode
        elif camera_mode == 0 and _measurement_history:
            y = status_y + 40
            progress = min(100, ((_now() - _measurement_history[0][1]) / _CONFIRMATION_WINDOW_SEC) * 100)
            cv2.putText(annotated_frame, f"Measurement Stability: {len(_measurement_history)} samples", 
                       (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
            cv2.putText(annotated_frame, f"Progress: {progress:.0f}%", 
                       (20, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
            
            # Display servo ranges if enabled
            if show_servo_ranges:
                y += 50
                cv2.putText(annotated_frame, "Servo Activation Ranges:", (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
                y += 25
                
                ranges_text = [
                    "8.7-10.0cm: Servo 5",
                    "10.0-11.5cm: Servo 6", 
                    "11.5-14.3cm: Servo 12",
                    "Other: Servo 13"
                ]
                
                for text in ranges_text:
                    cv2.putText(annotated_frame, text, (30, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    y += 20
        
        # Draw instructions
        instructions = "GPIO17/S: Switch | +/-: Conf | c: Calib | s: Ranges | r: Reset | d: Relay Dur | q: Quit"
        cv2.putText(annotated_frame, instructions, (10, annotated_frame.shape[0] - 10), 
                    font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Display the frame
        cv2.imshow("Dual Camera System", annotated_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s') or key == ord('S'):  # <<< ADDED KEYBOARD SWITCHING HERE
            # Keyboard switch cameras
            camera_mode = 1 - camera_mode  # Toggle between 0 and 1
            
            # Clear measurement history when switching modes
            _measurement_history.clear()
            _stable_length_cm = None
            _last_detection_time = _now()
            reset_complete_detection()
            
            # Deactivate relays if switching away from Pi Camera mode
            if camera_mode == 0 and _relay_active:
                deactivate_relays()
            
            if camera_mode == 0:
                print("\n" + "="*50)
                print("KEYBOARD SWITCHED TO: WEBCAM MODE")
                print("- Samaral detection with measurement")
                print("- Servo activation based on length")
                print("="*50 + "\n")
            else:
                print("\n" + "="*50)
                print("KEYBOARD SWITCHED TO: PI CAMERA MODE")
                print("- Complete/Incomplete detection")
                print("- Relays activate on 'complete' detection")
                print(f"- Relay duration: {RELAY_ACTIVATION_DURATION}s")
                print("="*50 + "\n")
        elif key == ord('+'):
            confidence_threshold = min(0.95, confidence_threshold + 0.05)
            print(f"Confidence threshold increased to: {confidence_threshold:.2f}")
        elif key == ord('-'):
            confidence_threshold = max(0.05, confidence_threshold - 0.05)
            print(f"Confidence threshold decreased to: {confidence_threshold:.2f}")
        elif key == ord('c'):
            # Calibration mode - only for webcam mode
            if camera_mode == 0:
                print("\n=== CALIBRATION MODE ===")
                print(f"Current pixel-to-cm ratio: {pixel_cm_ratio}")
                try:
                    new_ratio = float(input("Enter new pixel-to-cm ratio (or press Enter to keep current): "))
                    if new_ratio > 0:
                        pixel_cm_ratio = new_ratio
                        print(f"Pixel-to-cm ratio updated to: {pixel_cm_ratio}")
                except:
                    print("Keeping current value")
                print("=========================\n")
            else:
                print("Calibration only available in Webcam mode")
        elif key == ord('s'):
            show_servo_ranges = not show_servo_ranges
            print(f"Servo ranges display: {'ON' if show_servo_ranges else 'OFF'}")
        elif key == ord('r'):
            if camera_mode == 0:
                _measurement_history.clear()
                _stable_length_cm = None
                print("Measurement reset")
            else:
                reset_complete_detection()
                print("Complete detection reset")
        elif key == ord('d'):
            if camera_mode == 1:
                print("\n=== RELAY DURATION SETTING ===")
                print(f"Current relay duration: {RELAY_ACTIVATION_DURATION} seconds")
                try:
                    new_duration = float(input("Enter new relay duration in seconds: "))
                    if new_duration > 0:
                        RELAY_ACTIVATION_DURATION = new_duration
                        print(f"Relay duration updated to: {RELAY_ACTIVATION_DURATION} seconds")
                except:
                    print("Keeping current duration")
                print("==============================\n")
            else:
                print("Relay duration setting only available in Pi Camera mode")

except KeyboardInterrupt:
    print("\nProgram interrupted by user")

except Exception as e:
    print(f"\nError occurred: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Cleanup
    print("\nCleaning up resources...")
    
    # Return all servos to start position
    if _active_servo_pin is not None:
        print(f"Returning servo {_active_servo_pin} to start position...")
        _move_servo_to_start(_active_servo_pin)
    
    # Deactivate relays
    deactivate_relays()
    
    # Stop all servos
    stop_servos()
    
    # Release cameras
    if webcam is not None:
        webcam.release()
    picam2.stop()
    
    # Close windows
    cv2.destroyAllWindows()
    
    # Cleanup GPIO
    GPIO.cleanup()
    
    print("Resources cleaned up successfully!")
    print("Goodbye!")


