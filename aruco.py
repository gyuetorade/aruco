import cv2 as cv
import numpy as np
from time import time

# --------- PHYSICAL PAPER (Legal) ----------
PHYS_W_MM = 330.2   # width (left->right)
PHYS_H_MM = 215.9   # height (top->bottom)

# Choose canvas resolution ~2.5 px/mm (fits Pi, good precision)
PX_PER_MM = 2.5
RECTIFIED_W = int(round(PHYS_W_MM * PX_PER_MM))   # ~825-990 works too
RECTIFIED_H = int(round(PHYS_H_MM * PX_PER_MM))

# Ruler calibration: distance between major ticks you want to read
RULER_MM_BETWEEN_TICKS = 10.0   # 1cm major ticks

WEBCAM_INDEX = 0

# --------- ArUco Setup ----------
aru_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
aru_params = cv.aruco.DetectorParameters()
aru = cv.aruco.ArucoDetector(aru_dict, aru_params)

# We assume IDs are placed at the PAPER corners:
# 1=TopLeft, 2=TopRight, 3=BottomRight, 4=BottomLeft
# We must use the OUTERMOST CORNER of each tag (the one touching the paper corner).
# OpenCV returns marker corners in this order: [TL, TR, BR, BL]
MARKER_CORNER_INDEX = {
    1: 0,  # TL tag -> take its TL corner
    2: 1,  # TR tag -> take its TR corner
    3: 2,  # BR tag -> take its BR corner
    4: 3   # BL tag -> take its BL corner
}

DST_CORNERS = np.float32([
    [0, 0],                                # paper TL
    [RECTIFIED_W - 1, 0],                  # paper TR
    [RECTIFIED_W - 1, RECTIFIED_H - 1],    # paper BR
    [0, RECTIFIED_H - 1],                  # paper BL
])

def compute_homography_using_outer_corners(frame_bgr):
    gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    corners, ids, _ = aru.detectMarkers(gray)
    if ids is None:
        return None, None

    ids = ids.flatten().tolist()
    # build list in TL, TR, BR, BL order of the PAPER
    src_pts = [None, None, None, None]  # TL, TR, BR, BL
    for c, i in zip(corners, ids):
        if i in MARKER_CORNER_INDEX:
            idx = MARKER_CORNER_INDEX[i]               # which point of this tag
            paper_slot = [1,2,3,4].index(i)            # 0..3 matching TL,TR,BR,BL by ID
            pt = c.reshape(-1, 2)[idx]                 # the OUTER corner of the tag
            src_pts[paper_slot] = pt

    if any(p is None for p in src_pts):
        return None, corners

    src = np.float32(src_pts)
    H = cv.getPerspectiveTransform(src, DST_CORNERS)
    return H, corners

def rectify(img_bgr, H):
    return cv.warpPerspective(img_bgr, H, (RECTIFIED_W, RECTIFIED_H))

def choose_sharpest(frames):
    scores = [(cv.Laplacian(cv.cvtColor(f, cv.COLOR_BGR2GRAY), cv.CV_64F).var()) for f in frames]
    return frames[int(np.argmax(scores))]

def estimate_mm_per_px_from_ruler(rect_bgr):
    """Look for vertical tick spacing in a left strip."""
    gray = cv.cvtColor(rect_bgr, cv.COLOR_BGR2GRAY)
    strip_w = min(140, rect_bgr.shape[1] // 5)
    roi = gray[:, :strip_w]
    roi = cv.GaussianBlur(roi, (5,5), 0)

    # Emphasize vertical transitions (ticks)
    sobel = cv.Sobel(roi, cv.CV_32F, 0, 1, ksize=3)
    sobel = cv.convertScaleAbs(sobel)
    prof = sobel.mean(axis=1)

    rng = prof.max() - prof.min()
    if rng < 1e-6:
        return None
    prof = (prof - prof.min()) / rng

    # Simple peak detection
    peaks = []
    thresh, min_sep = 0.35, 4
    last = -999
    for i in range(1, len(prof)-1):
        if prof[i] > thresh and prof[i] > prof[i-1] and prof[i] > prof[i+1] and (i - last) > min_sep:
            peaks.append(i); last = i
    if len(peaks) < 6:
        return None

    diffs = np.diff(peaks)
    px_per_tick = float(np.median(diffs))
    if px_per_tick <= 0:
        return None
    return RULER_MM_BETWEEN_TICKS / px_per_tick

def segment_object(rect_bgr):
    V = cv.cvtColor(rect_bgr, cv.COLOR_BGR2HSV)[:,:,2]
    blur = cv.GaussianBlur(V, (5,5), 0)
    th = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 5)
    # Make object = white
    if np.sum(th==255) < np.sum(th==0):
        th = cv.bitwise_not(th)
    th = cv.morphologyEx(th, cv.MORPH_OPEN, np.ones((5,5), np.uint8))
    th = cv.morphologyEx(th, cv.MORPH_CLOSE, np.ones((7,7), np.uint8))
    cnts,_ = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return (max(cnts, key=cv.contourArea), th) if cnts else (None, th)

def measure_major_axis_mm(cnt, mm_per_px):
    rect = cv.minAreaRect(cnt)
    (_, _), (w, h), _ = rect
    major_px = max(w, h)
    return float(major_px * mm_per_px), rect

def main():
    cap = cv.VideoCapture(WEBCAM_INDEX)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    print(">>> Show legal paper with ArUco IDs 1(TL),2(TR),3(BR),4(BL) at the paper corners.")
    print(">>> Press 'h' to lock homography (uses the OUTERMOST tag corners).")
    H = None
    while True:
        ok, fr = cap.read()
        if not ok: continue
        cv.putText(fr, "Press 'h' when 4 tags visible. ESC to quit.",
                   (20,40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv.imshow("raw", fr)
        k = cv.waitKey(1) & 0xFF
        if k == ord('h'):
            H, corners = compute_homography_using_outer_corners(fr)
            if H is not None:
                print("Homography locked.")
                break
            print("Homography failed. Check tag visibility and ID placement.")
        if k == 27:
            cap.release(); cv.destroyAllWindows(); return

    print(">>> Place the ruler along LEFT edge of the paper. Press 's' to read mm/px.")
    mm_per_px = None
    while True:
        ok, f = cap.read()
        if not ok: continue
        rect = rectify(f, H)
        show = rect.copy()
        cv.rectangle(show, (0,0), (min(140, rect.shape[1]//5), rect.shape[0]), (0,255,0), 2)
        cv.putText(show, "Align ruler on LEFT; press 's' to calibrate.", (20,40),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv.imshow("rectified", show)
        k = cv.waitKey(1) & 0xFF
        if k == ord('s'):
            mm_per_px = estimate_mm_per_px_from_ruler(rect)
            if mm_per_px:
                print(f"Estimated mm/px: {mm_per_px:.4f}  (~{1/mm_per_px:.2f} px/mm)")
                break
            else:
                print("Could not read ticks. Increase contrast / move ruler closer to the left.")
        if k == 27:
            cap.release(); cv.destroyAllWindows(); return

    print(">>> Measuring. Keys: ESC quit | r recalibrate | p show paper frame | s save snapshot")
    while True:
        # short burst -> pick sharpest
        burst = []
        t0 = time()
        while time()-t0 < 0.2:
            ok, f = cap.read()
            if ok: burst.append(f)
        if not burst: continue
        frame = choose_sharpest(burst)
        rect = rectify(frame, H)

        cnt, th = segment_object(rect)
        vis = rect.copy()

        # draw paper border for sanity
        cv.rectangle(vis, (0,0), (RECTIFIED_W-1, RECTIFIED_H-1), (0,255,0), 2)
        cv.putText(vis, f"mm/px: {mm_per_px:.4f}  (~{1/mm_per_px:.2f} px/mm)",
                   (20,80), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        if cnt is not None:
            length_mm, r = measure_major_axis_mm(cnt, mm_per_px)
            box = cv.boxPoints(r).astype(int)
            cv.drawContours(vis, [box], 0, (0,255,0), 2)
            cv.putText(vis, f"{length_mm:.1f} mm", (20,120), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        else:
            cv.putText(vis, "No object detected", (20,120), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

        cv.imshow("rectified", vis)
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == ord('r'):
            print("Recalibrating homography: show 4 tags and press 'h'...")
            while True:
                ok, f2 = cap.read()
                if not ok: continue
                cv.imshow("raw", f2)
                kk = cv.waitKey(1) & 0xFF
                if kk == ord('h'):
                    H, corners = compute_homography_using_outer_corners(f2)
                    if H is not None:
                        print("Homography locked.")
                        break
                    else:
                        print("Failed; ensure 4 tags visible at paper corners.")
                if kk == 27:
                    break
        elif k == ord('s'):
            cv.imwrite("measurement.jpg", vis)
            print("Saved measurement.jpg")

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
