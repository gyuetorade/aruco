import cv2
import numpy as np
import time
import RPi.GPIO as GPIO
from picamera2 import Picamera2

# =========================
# GPIO setup
# =========================
SERVO_PINS = [5, 6, 12, 13]
BUTTON_PIN = 17
RELAY_A_PIN = 23
RELAY_B_PIN = 24

RELAY_A_ACTIVE_LOW = True
RELAY_B_ACTIVE_LOW = False

BUTTON_ON_DURATION = 5.0
DETECT_ON_DURATION_22 = 5.0
BUTTON_DEBOUNCE_SEC = 0.05

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)  # Add this to suppress warnings

# Setup pins
for pin in SERVO_PINS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# For active HIGH relay (RELAY_A_ACTIVE_LOW = False):
# - Initial HIGH means relay is ON initially
# - Initial LOW means relay is OFF initially
GPIO.setup(RELAY_A_PIN, GPIO.OUT, initial=GPIO.HIGH)  # Starts HIGH (ON)

GPIO.setup(RELAY_B_PIN, GPIO.OUT, initial=GPIO.HIGH if RELAY_B_ACTIVE_LOW else GPIO.LOW)

# If you have LED_PIN, define it first:
# LED_PIN = 18  # Uncomment and set your LED pin
# GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)OW))

# =========================
# Timing/state variables
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
    global _relay_a_expiry_t, _relay_b_button_expiry_t, _relay_b_detect_expiry_t
    now = _now()
    if expiry_ref_name == "A":
        _relay_a_expiry_t = max(_relay_a_expiry_t, now + duration)
    elif expiry_ref_name == "B_BUTTON":
        _relay_b_button_expiry_t = max(_relay_b_button_expiry_t, now + duration)
    elif expiry_ref_name == "B_DETECT":
        _relay_b_detect_expiry_t = max(_relay_b_detect_expiry_t, now + duration)

def _drive_relay(pin, on, active_low):
    """Control relay output
    on: True to turn relay ON, False to turn OFF
    active_low: True if relay activates with LOW signal, False if activates with HIGH
    """
    if active_low:
        # Active LOW relay: ON when pin is LOW
        GPIO.output(pin, GPIO.LOW if on else GPIO.HIGH)
    else:
        # Active HIGH relay: ON when pin is HIGH  
        GPIO.output(pin, GPIO.HIGH if on else GPIO.LOW)
        
def _apply_output_levels():
    now = _now()
    a_on = now < _relay_a_expiry_t
    b_on = now < max(_relay_b_button_expiry_t, _relay_b_detect_expiry_t)
    _drive_relay(RELAY_A_PIN, a_on, RELAY_A_ACTIVE_LOW)
    _drive_relay(RELAY_B_PIN, b_on, RELAY_B_ACTIVE_LOW)
    return a_on, b_on

def relay_b_is_on():
    return _now() < max(_relay_b_button_expiry_t, _relay_b_detect_expiry_t)

def poll_button_and_handle_edge():
    global _last_raw_button_state, _last_stable_button_state, _last_state_change_t
    now = _now()
    raw = GPIO.input(BUTTON_PIN)
    if raw != _last_raw_button_state:
        _last_raw_button_state = raw
        _last_state_change_t = now
    if now - _last_state_change_t >= BUTTON_DEBOUNCE_SEC:
        if raw != _last_stable_button_state:
            prev = _last_stable_button_state
            _last_stable_button_state = raw
            if prev == GPIO.HIGH and raw == GPIO.LOW:
                _start_or_extend("A", BUTTON_ON_DURATION)
                _start_or_extend("B_BUTTON", BUTTON_ON_DURATION)
                _apply_output_levels()

def pulse_relay_b_for_detection():
    _start_or_extend("B_DETECT", DETECT_ON_DURATION_22)

# =========================
# Servo PWM setup (RPi.GPIO)
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
REVERSE = {5: False, 6: False, 12: False, 13: True}
_active_servo_pin = None

def _move_servo(pin, angle):
    if REVERSE.get(pin, False):
        angle = 180 - angle
    servo_pwm[pin].ChangeDutyCycle(_angle_to_duty(angle))

def _park_others(active_pin):
    for p in SERVO_PINS:
        if p != active_pin:
            _move_servo(p, 90)

def park_all_servos():
    for p in SERVO_PINS:
        _move_servo(p, 90)

def control_servos(length_cm):
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
    pulse_relay_b_for_detection()

# =========================
# ArUco / detection setup
# =========================
BLUR_KSIZE = (5, 5)
MORPH_KSIZE = (3, 3)
OPEN_ITER = 1
CLOSE_ITER = 1
ADAPTIVE_BLOCK_SIZE = 19
ADAPTIVE_C = 5
AREA_MIN = 2000

DOWNSCALE_FACTOR = 0.5
USE_ROI = True
ROI_RECT = (200, 200, 850, 420)

detector_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KSIZE)

parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

MARKER_SIZE_CM = 5.0
CALIB_K = 1.0
SCALE_EMA_ALPHA = 0.3
pixel_cm_ratio = None
_last_scale = None
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
# Frame processing
# =========================
def process_frame(frame):
    global pixel_cm_ratio, _last_scale, _stable_length_cm, _locked_until_t, _prev_center, _active_servo_pin
    img_full = frame.copy()

    poll_button_and_handle_edge()

    if relay_b_is_on():
        _stable_length_cm = None
        _prev_center = None
        _active_servo_pin = None
        park_all_servos()
        a_on, b_on = _apply_output_levels()
        cv2.putText(img_full, "Measuring paused: Relay B active", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255),2)
        cv2.imshow("Image", img_full)
        return

    # --- ArUco detection ---
    corners, ids, _ = aruco_detector.detectMarkers(img_full)
    if corners:
        int_corners = np.intp(corners)
        cv2.polylines(img_full, int_corners, True, (0,255,0),5)
        pts = corners[0][0]
        d01 = np.linalg.norm(pts[1]-pts[0])
        d12 = np.linalg.norm(pts[2]-pts[1])
        d23 = np.linalg.norm(pts[3]-pts[2])
        d30 = np.linalg.norm(pts[0]-pts[3])
        mean_side_px = (d01 + d12 + d23 + d30)/4.0
        measured_px_per_cm = mean_side_px / float(MARKER_SIZE_CM)
        _last_scale = _ema(_last_scale, measured_px_per_cm, SCALE_EMA_ALPHA)
        pixel_cm_ratio = _last_scale
    else:
        cv2.putText(img_full, "No ArUco: using last scale", (20,40), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,165,255),2)

    # --- ROI & downscale ---
    if USE_ROI:
        x0,y0,rw,rh = ROI_RECT
        x1,y1 = x0+rw, y0+rh
        offx, offy = x0, y0
        proc_img_full = img_full[y0:y1, x0:x1]
        cv2.rectangle(img_full,(x0,y0),(x1,y1),(0,200,255),2)
    else:
        offx = offy = 0
        proc_img_full = img_full

    fx = fy = float(DOWNSCALE_FACTOR)
    proc_small = cv2.resize(proc_img_full,None,fx=fx,fy=fy,interpolation=cv2.INTER_AREA)
    area_min_small = AREA_MIN*(fx*fy)
    gray = cv2.cvtColor(proc_small, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5,5),0)
    mask = cv2.adaptiveThreshold(gray_blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,19,5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, detector_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, detector_kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects_full = []
    for cnt in contours:
        if cv2.contourArea(cnt) > area_min_small:
            rect = cv2.minAreaRect(cnt)
            (cx,cy),(w,h),angle = rect
            rects_full.append((cx,cy,w,h,angle))

    chosen = None
    now = _now()
    if rects_full:
        if _prev_center is not None:
            cx_prev, cy_prev = _prev_center
            rects_sorted = sorted(rects_full, key=lambda r: (np.hypot(r[0]-cx_prev,r[1]-cy_prev), -max(r[2],r[3])))
            candidate = rects_sorted[0]
            dist = np.hypot(candidate[0]-cx_prev,candidate[1]-cy_prev)
            if dist <= 120 or now < _locked_until_t:
                chosen = candidate
            else:
                if now >= _locked_until_t:
                    chosen = candidate
                    _locked_until_t = now + 0.4
        else:
            chosen = max(rects_full, key=lambda r: max(r[2],r[3]))
            _locked_until_t = now + 0.4

    if chosen and pixel_cm_ratio:
        cx,cy,w,h,angle = chosen
        box = cv2.boxPoints(((cx,cy),(w,h),angle)).astype(np.int32)
        box = _order_box_points(box)
        longer_side_px = max(w,h)
        raw_length_cm = longer_side_px/pixel_cm_ratio*CALIB_K
        if _stable_length_cm is None:
            _stable_length_cm = raw_length_cm
        else:
            if abs(raw_length_cm-_stable_length_cm)>=0.6:
                _stable_length_cm = _ema(_stable_length_cm, raw_length_cm, 0.2)
        cv2.polylines(img_full,[box],True,(255,0,0),2)
        _draw_long_side(img_full, box,(0,255,255),3)
        cv2.circle(img_full,(int(cx),int(cy)),5,(0,0,255),-1)
        label_len = f"Length {round(_stable_length_cm,1)} cm"
        cv2.putText(img_full,label_len,(int(cx-100),int(cy+15)),cv2.FONT_HERSHEY_PLAIN,2,(100,200,0),2)
        if _prev_center is None or np.hypot(cx-_prev_center[0],cy-_prev_center[1])>3:
            _prev_center = (cx,cy)
            _locked_until_t = now + 0.4
        control_servos(_stable_length_cm)
    else:
        _stable_length_cm = None
        _prev_center = None
        _active_servo_pin = None
        park_all_servos()

    _apply_output_levels()


    img_full = cv2.resize(img_full, (1024, 768))  # Change these numbers to fit your screen
    cv2.imshow("Image", img_full)
    

# =========================
# Main
# =========================
def main():
    setup_servos()
    picam2 = Picamera2()
    
    # Configure for RGB format (not RGBA)
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
            
            # If frame has 4 channels (RGBA), convert to 3 (RGB)
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            elif len(frame.shape) == 3 and frame.shape[2] == 3:
                # Already RGB, ensure it's the right order for OpenCV
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

