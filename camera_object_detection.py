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
CONFIRMATION_DELAY_SEC = 0.5  # Delay for confirmation (in seconds)
_LENGTH_TOLERANCE_CM = 0.3

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

# _check_length_confirmation updated to include confirmation delay
def _check_length_confirmation(current_length_cm):
    """Ensure at least CONFIRMATION_DELAY_SEC has passed before confirming size."""
    global _frame_history

    # If system is busy (servo or relay active), don't process new confirmations
    if _system_busy:
        return None

    now = _now()

    # Clean old entries (older than 1 second)
    _frame_history = [(length, ts) for length, ts in _frame_history if now - ts < 1.0]

    # Add the current measurement to history
    _frame_history.append((current_length_cm, now))

    # Keep only last N frames (N = CONFIRMATION_FRAMES + 1 for buffer)
    if len(_frame_history) > _CONFIRMATION_FRAMES + 1:
        _frame_history = _frame_history[-(_CONFIRMATION_FRAMES + 1):]

    # Check if we have enough frames
    if len(_frame_history) >= _CONFIRMATION_FRAMES:
        # Get the timestamps of the first and last frames in the confirmation window
        first_time = _frame_history[-_CONFIRMATION_FRAMES][1]
        last_time = _frame_history[-1][1]

        # Ensure that the time span between first and last frame is at least CONFIRMATION_DELAY_SEC
        if (last_time - first_time) >= CONFIRMATION_DELAY_SEC:
            # Check if all recent frames have approximately the same length
            lengths = [frame[0] for frame in _frame_history[-_CONFIRMATION_FRAMES:]]
            min_length = min(lengths)
            max_length = max(lengths)

            # If all lengths are within tolerance, confirmation achieved
            if (max_length - min_length) <= _LENGTH_TOLERANCE_CM:
                # Return the average length
                return np.mean(lengths)

    return None  # No confirmation yet