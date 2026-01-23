import cv2
import numpy as np
import time
import RPi.GPIO as GPIO
import math
from picamera2 import Picamera2

# =========================
# GPIO setup (using second code's pin layout)
# =========================
SERVO_PINS = [5, 6, 12, 13]
BUTTON_PIN = 17
RELAY_A_PIN = 23  # Changed from 18 to 23
RELAY_B_PIN = 24  # Changed from 22 to 24

# Relay configuration from second code
RELAY_A_ACTIVE_LOW = True
RELAY_B_ACTIVE_LOW = False

# Durations from first code
BUTTON_ON_DURATION_18 = 1.0
BUTTON_ON_DURATION_22 = 1.0
DETECT_ON_DURATION_22 = 2.0
BUTTON_DEBOUNCE_SEC = 0.05

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Setup pins
for pin in SERVO_PINS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Setup relays with proper initial states
if RELAY_A_ACTIVE_LOW:
    GPIO.setup(RELAY_A_PIN, GPIO.OUT, initial=GPIO.HIGH)  # Start HIGH (OFF for active-low)
else:
    GPIO.setup(RELAY_A_PIN, GPIO.OUT, initial=GPIO.LOW)   # Start LOW (OFF for active-high)

if RELAY_B_ACTIVE_LOW:
    GPIO.setup(RELAY_B_PIN, GPIO.OUT, initial=GPIO.HIGH)  # Start HIGH (OFF for active-low)
else:
    GPIO.setup(RELAY_B_PIN, GPIO.OUT, initial=GPIO.LOW)   # Start LOW (OFF for active-high)

# =========================
# Timing/state variables (from first code)
# =========================
_last_raw_button_state = GPIO.input(BUTTON_PIN)
_last_stable_button_state = _last_raw_button_state
_last_state_change_t = time.monotonic()

_relay_a_expiry_t = 0.0
_relay_b_button_expiry_t = 0.0
_relay_b_detect_expiry_t = 0.0

def _now():
    return time.monotonic()

def _start_or_extend(expiry_ref_name, duration):
    """Only extends timers, doesn't activate relays directly (from first code)"""
    global _relay_a_expiry_t, _relay_b_button_expiry_t, _relay_b_detect_expiry_t
    now = _now()
    if expiry_ref_name == "A":
        _relay_a_expiry_t = max(_relay_a_expiry_t, now + duration)
    elif expiry_ref_name == "B_BUTTON":
        _relay_b_button_expiry_t = max(_relay_b_button_expiry_t, now + duration)
    elif expiry_ref_name == "B_DETECT":
        _relay_b_detect_expiry_t = max(_relay_b_detect_expiry_t, now + duration)

def _drive_relay(pin, on, active_low):
    """Control relay output (from second code)"""
    if active_low:
        # Active LOW relay: ON when pin is LOW
        GPIO.output(pin, GPIO.LOW if on else GPIO.HIGH)
    else:
        # Active HIGH relay: ON when pin is HIGH  
        GPIO.output(pin, GPIO.HIGH if on else GPIO.LOW)

def _apply_output_levels():
    """Actually applies the relay states based on timers (from first code)"""
    now = _now()
    
    # Relay A: Only ON if button timer is active
    a_on = now < _relay_a_expiry_t
    
    # Relay B: ON if either button timer OR detection timer is active
    b_on = now < max(_relay_b_button_expiry_t, _relay_b_detect_expiry_t)

    # Apply relays using second code's driver function
    _drive_relay(RELAY_A_PIN, a_on, RELAY_A_ACTIVE_LOW)
    _drive_relay(RELAY_B_PIN, b_on, RELAY_B_ACTIVE_LOW)

    return a_on, b_on  # return states for HUD

def poll_button_and_handle_edge():
    """Polls button with debouncing and triggers relay timers on press (from first code)"""
    global _last_raw_button_state, _last_stable_button_state, _last_state_change_t

    now = _now()
    raw = GPIO.input(BUTTON_PIN)

    # Detect raw state change
    if raw != _last_raw_button_state:
        _last_raw_button_state = raw
        _last_state_change_t = now

    # Only update stable state after debounce period
    if now - _last_state_change_t >= BUTTON_DEBOUNCE_SEC:
        if raw != _last_stable_button_state:
            prev = _last_stable_button_state
            _last_stable_button_state = raw
            
            # Detect button PRESS (HIGH to LOW transition, active-LOW button)
            if prev == GPIO.HIGH and raw == GPIO.LOW:
                # Button pressed: Start timers for BOTH relays
                _start_or_extend("A", BUTTON_ON_DURATION_18)
                _start_or_extend("B_BUTTON", BUTTON_ON_DURATION_22)

def pulse_relay_b_for_detection():
    """Triggers Relay B for object detection - does NOT affect Relay A (from first code)"""
    _start_or_extend("B_DETECT", DETECT_ON_DURATION_22)

# =========================
# Servo PWM setup (RPi.GPIO from second code)
# =========================
servo_pwm = {}

def setup_servos():
    for pin in SERVO_PINS:
        pwm = GPIO.PWM(pin, 50)  # 50Hz
        pwm.start(7.5)            # Neutral 90Â°
        servo_pwm[pin] = pwm

def stop_servos():
    for pwm in servo_pwm.values():
        pwm.stop()

def _angle_to_duty(angle):
    # Map 0-180Â° -> 2.5-12.5 duty cycle
    return 2.5 + (angle / 180.0) * 10.0

BIN_ANGLES = {5: 60, 6: 90, 12: 120, 13: 30}
REVERSE = {5: False, 6: False, 12: False, 13: True}  # From second code
_active_servo_pin = None

def _move_servo(pin, angle):
    if REVERSE.get(pin, False):
        angle = 180 - angle
    servo_pwm[pin].ChangeDutyCycle(_angle_to_duty(angle))

def _park_others(active_pin):
    for p in SERVO_PINS:
        if p != active_pin:
            _move_servo(p, 90)

def control_servos(length_cm):
    """Controls servos based on detected length. Also pulses Relay B (not A!) (from first code)"""
    global _active_servo_pin

    if 8.7 <= length_cm <= 10.0:
        sel = 5
    elif 10.0 < length_cm <= 11.5:
        sel = 6
    elif 11.5 < length_cm <= 14.3:
        sel = 12
    else:
        sel = 13

    _move_servo(sel, BIN_ANGLES[sel])
    _park_others(sel)
    _active_servo_pin = sel

    # Only pulse Relay B for detection, NOT Relay A (from first code)
    pulse_relay_b_for_detection()

# =========================
# Vision setup (from first code, ROI removed)
# =========================
BLUR_KSIZE = (5, 5)
MORPH_KSIZE = (3, 3)
OPEN_ITER = 1
CLOSE_ITER = 1
ADAPTIVE_BLOCK_SIZE = 19
ADAPTIVE_C = 5
AREA_MIN = 2000

DOWNSCALE_FACTOR = 0.5
# REMOVED: USE_ROI and ROI_RECT

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

# ArUco detector
parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# -------------------------
# Accurate metric scaling
# -------------------------
MARKER_SIZE_CM = 5.00
CALIB_K = 1.00
SCALE_EMA_ALPHA  = 0.3
pixel_cm_ratio = None
_last_scale = None

detector = DetectorObj()

# =========================
# Stability tuning & state (from first code)
# =========================
LENGTH_EMA_ALPHA = 0.2
HYSTERESIS_CM    = 0.6
LOCK_MS          = 400
MAX_CENTER_JUMP  = 120

_stable_length_cm = None
_locked_until_t = 0.0
_prev_center = None

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
# Frame processing (from first code, ROI removed)
# =========================
def process_frame(frame):
    global pixel_cm_ratio, _last_scale
    global _stable_length_cm, _locked_until_t, _prev_center

    img_full = frame.copy()

    # CRITICAL: Poll button first, then apply outputs
    poll_button_and_handle_edge()

    # ---------- ArUco: compute accurate px/cm from side length ----------
    corners, ids, _ = aruco_detector.detectMarkers(img_full)
    if corners:
        int_corners = np.intp(corners)
        cv2.polylines(img_full, int_corners, True, (0, 255, 0), 5)

        pts = corners[0][0]
        d01 = np.linalg.norm(pts[1] - pts[0])
        d12 = np.linalg.norm(pts[2] - pts[1])
        d23 = np.linalg.norm(pts[3] - pts[2])
        d30 = np.linalg.norm(pts[0] - pts[3])
        mean_side_px = (d01 + d12 + d23 + d30) / 4.0

        measured_px_per_cm = mean_side_px / float(MARKER_SIZE_CM)
        _last_scale = _ema(_last_scale, measured_px_per_cm, SCALE_EMA_ALPHA)
        pixel_cm_ratio = _last_scale
    else:
        cv2.putText(img_full, "No ArUco: using last scale",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    # ---------- Downscale entire frame (no ROI) ----------
    fx = fy = float(DOWNSCALE_FACTOR)
    if not (0 < fx < 1):
        fx = fy = 1.0
    
    # Process entire frame
    offx = offy = 0
    proc_img_full = img_full
    proc_small = cv2.resize(proc_img_full, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    area_min_small = AREA_MIN * (fx * fy)

    contours_small = detector.detect_objects(proc_small, area_min_small)

    rects_full = []
    if contours_small:
        scale_back = 1.0 / fx
        for cnt_s in contours_small:
            cnt_full = (cnt_s.astype(np.float32) * scale_back) + np.array([[[offx, offy]]], dtype=np.float32)
            rect = cv2.minAreaRect(cnt_full)
            (cx, cy), (w, h), angle = rect
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
        label_len = f"Length {round(_stable_length_cm, 1)} cm"
        cv2.putText(img_full, label_len, (int(cx - 100), int(cy + 15)),
                    cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

        if _prev_center is None or np.hypot(cx - _prev_center[0], cy - _prev_center[1]) > 3:
            _prev_center = (cx, cy)
            _locked_until_t = now + (LOCK_MS / 1000.0)

        # This only affects servos and Relay B, NOT Relay A
        control_servos(_stable_length_cm)

    else:
        _stable_length_cm = None
        _prev_center = None

    # Apply relay outputs and get current states
    a_on, b_on = _apply_output_levels()

    if pixel_cm_ratio:
        cv2.putText(img_full, f"px/cm: {pixel_cm_ratio:.2f}  CALIB_K: {CALIB_K:.3f}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

    y = 60
    cv2.putText(img_full, f"Relay A (BTN only): {'ON' if a_on else 'OFF'}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if a_on else (0, 0, 255), 2)
    y += 25
    cv2.putText(img_full, f"Relay B (BTN+detect): {'ON' if b_on else 'OFF'}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if b_on else (0, 0, 255), 2)

    y += 25
    for pin in SERVO_PINS:
        on = (pin == _active_servo_pin)
        cv2.putText(img_full, f"Servo {pin}: {'ACTIVE' if on else 'PARK'}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if on else (0, 0, 255), 2)
        y += 25

    # Resize for display (from second code)
    img_full = cv2.resize(img_full, (1024, 768))
    cv2.imshow("Image", img_full)

# =========================
# Main (from second code with Raspberry Pi Camera)
# =========================
def main():
    setup_servos()
    picam2 = Picamera2()
    
    # Configure for RGB format
    config = picam2.create_preview_configuration(
        main={"size": (1920, 1080), "format": "RGB888"}
    )
    picam2.configure(config)
    
    picam2.start()
    
    # Warm up
    time.sleep(2)
    
    try:
        while True:
            frame = picam2.capture_array()
            
            # Convert from RGB to BGR for OpenCV
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            process_frame(frame)
            
            if cv2.waitKey(1) == 27:
                break
    finally:
        stop_servos()
        picam2.stop()
        GPIO.cleanup()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()
