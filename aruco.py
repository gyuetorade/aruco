import cv2
import numpy as np
import math

# =========================
# Tunable parameters
# =========================
# --- Preprocessing ---
BLUR_KSIZE = (5, 5)         # Mild blur to reduce high-frequency noise
MORPH_KSIZE = (3, 3)        # Kernel size for morphological ops
OPEN_ITER = 1               # Opening iterations (remove small specks)
CLOSE_ITER = 1              # Closing iterations (fill small holes)
ADAPTIVE_BLOCK_SIZE = 19    # Must be odd; local window for adaptive threshold
ADAPTIVE_C = 5              # Constant subtracted from mean in adaptive threshold
AREA_MIN = 2000             # Minimum contour area (pixels) to keep (FULL-RES basis)

# --- Smoothing / tracking ---
EMA_ALPHA = 0.3             # 0<alpha<=1; lower = smoother but slower to react
MAX_MATCH_DIST = 60.0       # Max center distance (px) to match a prior track
TRACK_TTL = 15              # Frames to keep tracks that werenâ€™t matched

# --- Performance: downscale & ROI ---
DOWNSCALE_FACTOR = 0.5      # 0.25â€“0.75 commonly good; <1 means smaller processing frame
USE_ROI = True              # Process only a region (e.g., table area)
ROI_RECT = (0, 400, 1280, 520)  # <--- YOUR ROI (x, y, w, h) in FULL-RES pixels
# =========================


class DetectorObj:
    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KSIZE)

    def detect_objects(self, frame_small, area_min_small):
        """
        Run detection on the *downscaled or ROI image* provided.
        Returns a list of contours in the same coordinate space as frame_small.
        The area filter threshold should be given in that same scale.
        """
        # 1) Grayscale
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        # 2) Mild blur to reduce noise / texture before thresholding
        gray_blur = cv2.GaussianBlur(gray, BLUR_KSIZE, 0)

        # 3) Adaptive threshold to get a binary mask (foreground = white)
        mask = cv2.adaptiveThreshold(
            gray_blur, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
            ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
        )

        # 4) Morphological OPEN then CLOSE to clean noise and fill small gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=OPEN_ITER)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=CLOSE_ITER)

        # 5) Find contours on the cleaned binary mask
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 6) Keep only sufficiently large contours (threshold is already scaled)
        objects_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > area_min_small:
                objects_contours.append(cnt)

        return objects_contours


class EmaSmoother:
    """
    Simple nearest-neighbor tracker + EMA smoother.
    Keeps per-object smoothed (x, y, w, h) across frames to reduce flicker.
    Operates in FULL-RES coordinates.
    """
    def __init__(self, alpha=EMA_ALPHA, max_match_dist=MAX_MATCH_DIST, ttl=TRACK_TTL):
        self.alpha = float(alpha)
        self.max_match_dist = float(max_match_dist)
        self.ttl = int(ttl)
        self.tracks = {}       # id -> dict(x,y,w,h,age,last_seen)
        self.next_id = 1

    def _dist(self, p, q):
        return math.hypot(p[0] - q[0], p[1] - q[1])

    def _create_track(self, x, y, w, h):
        tid = self.next_id
        self.next_id += 1
        self.tracks[tid] = {
            "x": float(x), "y": float(y),
            "w": float(w), "h": float(h),
            "age": 0, "last_seen": 0
        }
        return tid

    def update(self, rects_full):
        """
        rects_full: list of (x, y, w, h, angle) in FULL-RES coordinates.
        Returns list of dicts with smoothed values and id.
        """
        # Age/TTL tick
        for tid, tr in self.tracks.items():
            tr["age"] += 1
            tr["last_seen"] += 1

        unmatched_track_ids = set(self.tracks.keys())
        results = []

        for rect in rects_full:
            (x, y, w, h, angle) = rect

            # Find best (nearest) track within threshold
            best_tid = None
            best_dist = float("inf")
            for tid in unmatched_track_ids:
                tr = self.tracks[tid]
                d = self._dist((x, y), (tr["x"], tr["y"]))
                if d < best_dist and d <= self.max_match_dist:
                    best_dist = d
                    best_tid = tid

            if best_tid is None:
                # New track
                tid = self._create_track(x, y, w, h)
                self.tracks[tid]["last_seen"] = 0
                x_s, y_s, w_s, h_s = x, y, w, h
            else:
                # Match -> EMA
                tid = best_tid
                tr = self.tracks[tid]
                a = self.alpha
                x_s = a * x + (1 - a) * tr["x"]
                y_s = a * y + (1 - a) * tr["y"]
                w_s = a * w + (1 - a) * tr["w"]
                h_s = a * h + (1 - a) * tr["h"]
                tr["x"], tr["y"], tr["w"], tr["h"] = x_s, y_s, w_s, h_s
                tr["last_seen"] = 0
                unmatched_track_ids.remove(tid)

            results.append({"id": tid, "x": x_s, "y": y_s, "w": w_s, "h": h_s, "angle": angle})

        # Drop stale tracks
        to_delete = [tid for tid, tr in self.tracks.items() if tr["last_seen"] > self.ttl]
        for tid in to_delete:
            del self.tracks[tid]

        return results


# ---- ArUco detector setup ----
parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

pixel_cm_ratio = None
detector = DetectorObj()
smoother = EmaSmoother(alpha=EMA_ALPHA, max_match_dist=MAX_MATCH_DIST, ttl=TRACK_TTL)


def process_frame(frame):
    """
    Full-resolution frame in, all drawing happens on full-res copy.
    Detection runs on (optional ROI) and downscaled sub-image for speed, then
    contours are mapped back to full-res for measurement + drawing.
    """
    global pixel_cm_ratio
    img_full = frame.copy()
    H, W = img_full.shape[:2]

    # ---------------------------------
    # 1) Detect ArUco (FULL-RES) for scale
    # ---------------------------------
    corners, ids, _ = aruco_detector.detectMarkers(img_full)
    if corners:
        int_corners = np.intp(corners)
        cv2.polylines(img_full, int_corners, True, (0, 255, 0), 5)
        aruco_perimeter = cv2.arcLength(corners[0], True)
        # Assumes marker perimeter is 20 cm (i.e., 5 cm side)
        pixel_cm_ratio = aruco_perimeter / 20.0
    else:
        # Optional hint when no scale is available
        cv2.putText(
            img_full, "No ArUco marker detected: scale not set",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
        )

    # ---------------------------------
    # 2) Select processing region (ROI or full image)
    # ---------------------------------
    if USE_ROI:
        x0, y0, rw, rh = ROI_RECT
        x1, y1 = x0 + rw, y0 + rh
        # Clamp ROI to frame
        x0 = max(0, min(x0, W - 1)); y0 = max(0, min(y0, H - 1))
        x1 = max(1, min(x1, W));     y1 = max(1, min(y1, H))
        rw = x1 - x0; rh = y1 - y0
        proc_origin = (x0, y0)
        proc_img_fullres = img_full[y0:y1, x0:x1]
        # Show ROI rectangle for visualization
        cv2.rectangle(img_full, (x0, y0), (x1, y1), (0, 200, 255), 2)
    else:
        proc_origin = (0, 0)
        proc_img_fullres = img_full

    # ---------------------------------
    # 3) Downscale processing region for fast detection
    # ---------------------------------
    fx = fy = float(DOWNSCALE_FACTOR)
    if fx <= 0 or fx >= 1:
        # If mis-set, fall back to no downscale
        fx = fy = 1.0
    proc_img_small = cv2.resize(
        proc_img_fullres, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA
    )

    # Area threshold must be scaled to the downscaled area (factor^2)
    area_min_small = AREA_MIN * (fx * fy)

    # ---------------------------------
    # 4) Detect contours on SMALL image, then map back to FULL-RES
    # ---------------------------------
    contours_small = detector.detect_objects(proc_img_small, area_min_small)

    rects_full = []
    if contours_small:
        scale_back = 1.0 / fx
        offx, offy = proc_origin

        for cnt_s in contours_small:
            # Scale contour points back to processing-region full-res
            cnt_full_region = (cnt_s.astype(np.float32) * scale_back)
            # Shift from ROI space (if any) to full-frame space
            cnt_full_frame = cnt_full_region + np.array([[[offx, offy]]], dtype=np.float32)

            # Compute rotated rect in FULL-RES coordinates
            rect = cv2.minAreaRect(cnt_full_frame)
            (cx, cy), (w, h), angle = rect
            rects_full.append((cx, cy, w, h, angle))

    # ---------------------------------
    # 5) Smooth & draw (FULL-RES)
    # ---------------------------------
    if pixel_cm_ratio and rects_full:
        smoothed = smoother.update(rects_full)
        for item in smoothed:
            cx, cy, w_s, h_s, angle = item["x"], item["y"], item["w"], item["h"], item["angle"]
            rect_s = ((cx, cy), (w_s, h_s), angle)
            box = cv2.boxPoints(rect_s)
            box = np.intp(box)

            # Measurement in cm: display ONLY "Length" using the previous "height"
            object_length = h_s / pixel_cm_ratio

            # Draw geometry
            cv2.circle(img_full, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            cv2.polylines(img_full, [box], True, (255, 0, 0), 2)

            # Draw ONLY Length label
            label_len = f"Length {round(object_length, 1)} cm"
            cv2.putText(
                img_full, label_len, (int(cx - 100), int(cy + 15)),
                cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2
            )
    elif rects_full:
        # Contours found but no scale â†’ draw only shapes
        for (cx, cy, w, h, angle) in rects_full:
            rect_s = ((cx, cy), (w, h), angle)
            box = cv2.boxPoints(rect_s)
            box = np.intp(box)
            cv2.circle(img_full, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            cv2.polylines(img_full, [box], True, (255, 0, 0), 2)

    cv2.imshow("Image", img_full)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        process_frame(frame)
        key = cv2.waitKey(1)
        if key == 27:  # Esc to exit
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
