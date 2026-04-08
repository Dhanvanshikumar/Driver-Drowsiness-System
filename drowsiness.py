"""
Driver Drowsiness Detection System
===================================
Detects drowsiness using three indicators:
  1. PERCLOS  - Percentage of Eye Closure over time
  2. Yawning  - Detected with a trained CNN model
  3. Head Pose - Nodding / tilting detected via 3D pose estimation

Press 'q' to quit the camera window.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import mediapipe as mp
import numpy as np
import os
import time
from collections import deque

# =====================================================================
# CONFIG
# =====================================================================
EAR_THRESHOLD = 0.22          # Below this -> eyes considered closed
PERCLOS_WINDOW_SEC = 6        # Seconds of history for PERCLOS calc
PERCLOS_DROWSY = 0.35         # 35% eye closure -> drowsy
EYE_CLOSED_MIN_SEC = 3.0      # Eyes must be closed 3s before flagging drowsy

YAWN_CONF_THRESHOLD = 0.60    # Min confidence to count a yawn
YAWN_COOLDOWN_SEC = 1         # Seconds between separate yawn events
YAWN_FRAMES_NEEDED = 25        # Consecutive yawn frames to trigger

HEAD_PITCH_THRESH = 25        # Degrees – head nodding forward
HEAD_NOD_DURATION = 2.0       # Head must be down 3s before flagging drowsy

# Drowsiness scoring weights
W_PERCLOS = 0.45
W_YAWN    = 0.30
W_HEAD    = 0.25

DROWSY_SCORE_THRESH = 0.60    # Overall score threshold for alert

# =====================================================================
# SETUP
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
base_dir = os.path.dirname(os.path.abspath(__file__))

# ---------- Yawn model --------------------------------------------------
yawn_model = models.mobilenet_v2(weights=None)
yawn_model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(yawn_model.last_channel, 2)
)
model_path = os.path.join(base_dir, "yawn_model.pth")
yawn_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
yawn_model.to(device).eval()
print("Yawn model loaded.")

yawn_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------- MediaPipe Face Mesh ------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.3,
)
print("MediaPipe Face Mesh loaded.")

# =====================================================================
# LANDMARK INDICES  (MediaPipe 478-point mesh)
# =====================================================================
# Left eye
LEFT_EYE = [362, 385, 387, 263, 373, 380]
# Right eye
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Mouth outer – used to crop the mouth region for the yawn model
MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
               409, 270, 269, 267, 0, 37, 39, 40, 185]

# Head pose reference points (nose tip, chin, left eye corner,
# right eye corner, left mouth corner, right mouth corner)
POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]

# 3-D model points (generic face model, arbitrary units)
MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),      # Nose tip
    (0.0,   -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),    # Left eye left corner
    (225.0,  170.0, -135.0),    # Right eye right corner
    (-150.0, -150.0, -125.0),   # Left mouth corner
    (150.0,  -150.0, -125.0),   # Right mouth corner
], dtype=np.float64)


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    """Compute EAR for one eye."""
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices])
    # Vertical distances
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    # Horizontal distance
    horiz = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * horiz + 1e-6)


def get_head_pose(landmarks, w, h):
    """Return (pitch, yaw, roll) in degrees using solvePnP."""
    image_points = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in POSE_LANDMARKS],
        dtype=np.float64
    )
    focal_length = w
    center = (w / 2, h / 2)
    cam_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS, image_points, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0, 0, 0

    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = angles[0], angles[1], angles[2]
    return pitch, yaw, roll


def get_mouth_crop(frame, landmarks, w, h):
    """Extract mouth region from landmarks to feed yawn model."""
    xs = [int(landmarks[i].x * w) for i in MOUTH_OUTER]
    ys = [int(landmarks[i].y * h) for i in MOUTH_OUTER]
    x1 = max(0, min(xs) - 10)
    y1 = max(0, min(ys) - 10)
    x2 = min(w, max(xs) + 10)
    y2 = min(h, max(ys) + 10)
    crop = frame[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)


def predict_yawn(crop_bgr):
    """Return (is_yawn: bool, confidence: float, probs)."""
    if crop_bgr.size == 0 or crop_bgr.shape[0] < 10 or crop_bgr.shape[1] < 10:
        return False, 0.0, np.array([1.0, 0.0])
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = yawn_transform(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        out = yawn_model(tensor)
        probs = torch.softmax(out, 1)[0].cpu().numpy()
    is_yawn = probs[1] > YAWN_CONF_THRESHOLD
    return is_yawn, float(probs[1]), probs


# =====================================================================
# STATE VARIABLES
# =====================================================================
# PERCLOS tracking
ear_history = deque()          # (timestamp, ear_value)
eyes_closed_start = None       # Timestamp when eyes first closed
eyes_sustained_closed = False  # True only after 3s continuous closure

# Yawn tracking
yawn_frame_counter = 0
yawn_active = False
last_yawn_time = 0.0
yawn_count = 0
yawn_start_time = 0.0
recent_yawns = deque()         # timestamps of yawn events (last 60s)

# Head nod tracking
head_down_start = None
head_nod_detected = False
head_nod_time = 0.0
recent_nods = deque()          # timestamps of nod events (last 60s)

# Alert
alert_active = False
alert_start_time = 0.0

# =====================================================================
# DRAWING HELPERS
# =====================================================================

def draw_meter(img, x, y, width, height, value, label,
               color_low=(0, 200, 0), color_high=(0, 0, 255), threshold=0.5):
    """Draw a horizontal bar meter with label."""
    cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (60, 60, 60), -1)
    fill = int(width * min(value, 1.0))
    color = color_high if value >= threshold else color_low
    cv2.rectangle(img, (x, y), (x + fill, y + height), color, -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (120, 120, 120), 1)
    pct_text = f"{value:.0%}"
    cv2.putText(img, pct_text, (x + width + 5, y + height - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def draw_status_icon(img, x, y, active, label):
    """Draw a colored circle + label."""
    color = (0, 0, 255) if active else (0, 180, 0)
    cv2.circle(img, (x, y), 8, color, -1)
    cv2.circle(img, (x, y), 8, (200, 200, 200), 1)
    cv2.putText(img, label, (x + 15, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)


# =====================================================================
# MAIN LOOP
# =====================================================================
print("\nStarting camera … Press 'q' to quit.\n")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera.")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fps = 0
fps_timer = time.time()
frame_cnt = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    now = time.time()

    display = frame.copy()

    # Semi-transparent dark panel on the right for stats
    overlay = display.copy()
    panel_x = w - 220
    cv2.rectangle(overlay, (panel_x, 0), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)

    # --- Face Mesh ---
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    perclos = 0.0
    ear_avg = 0.0
    pitch, yaw, roll = 0, 0, 0
    is_yawn_frame = False
    yawn_conf = 0.0
    yawn_probs = np.array([1.0, 0.0])
    face_detected = False

    if results.multi_face_landmarks:
        face_detected = True
        lm = results.multi_face_landmarks[0].landmark

        # ---- 1) EYE CLOSURE (EAR + PERCLOS) ----
        ear_l = eye_aspect_ratio(lm, LEFT_EYE, w, h)
        ear_r = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
        ear_avg = (ear_l + ear_r) / 2.0
        eyes_closed = ear_avg < EAR_THRESHOLD

        # Track sustained eye closure (must be closed >= 3s continuously)
        if eyes_closed:
            if eyes_closed_start is None:
                eyes_closed_start = now
            elif (now - eyes_closed_start) >= EYE_CLOSED_MIN_SEC:
                eyes_sustained_closed = True
        else:
            eyes_closed_start = None
            eyes_sustained_closed = False

        # Store in history
        ear_history.append((now, eyes_closed))
        # Trim old entries
        while ear_history and (now - ear_history[0][0]) > PERCLOS_WINDOW_SEC:
            ear_history.popleft()

        # Calculate PERCLOS
        if len(ear_history) > 0:
            closed_count = sum(1 for _, closed in ear_history if closed)
            perclos = closed_count / len(ear_history)

        # Draw eye landmarks
        for idx in LEFT_EYE + RIGHT_EYE:
            px = int(lm[idx].x * w)
            py = int(lm[idx].y * h)
            color = (0, 0, 255) if eyes_sustained_closed else (0, 255, 0)
            cv2.circle(display, (px, py), 2, color, -1)

        # Debug: show live EAR on frame so user can verify
        ear_debug_color = (0, 0, 255) if eyes_closed else (0, 220, 0)
        cv2.putText(display, f"EAR: {ear_avg:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_debug_color, 2)
        if eyes_closed and eyes_closed_start is not None:
            elapsed = now - eyes_closed_start
            cv2.putText(display, f"Closed: {elapsed:.1f}s / {EYE_CLOSED_MIN_SEC:.0f}s", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 1)

        # ---- 2) YAWN DETECTION ----
        mouth_crop, (mx1, my1, mx2, my2) = get_mouth_crop(frame, lm, w, h)
        is_yawn_frame, yawn_conf, yawn_probs = predict_yawn(mouth_crop)

        if is_yawn_frame:
            yawn_frame_counter += 1
        else:
            yawn_frame_counter = max(0, yawn_frame_counter - 1)

        # Trigger yawn event
        if yawn_frame_counter >= YAWN_FRAMES_NEEDED and not yawn_active:
            if (now - last_yawn_time) > YAWN_COOLDOWN_SEC:
                yawn_active = True
                yawn_count += 1
                last_yawn_time = now
                yawn_start_time = now
                recent_yawns.append(now)

        if yawn_active and yawn_frame_counter < 2:
            yawn_active = False

        # Trim old yawn events
        while recent_yawns and (now - recent_yawns[0]) > 15:
            recent_yawns.popleft()

        # Draw mouth region
        mouth_color = (0, 0, 255) if yawn_active else (0, 180, 0)
        cv2.rectangle(display, (mx1, my1), (mx2, my2), mouth_color, 2)

        # ---- 3) HEAD POSE ----
        pitch, yaw, roll = get_head_pose(lm, w, h)

        # Head nod detection (pitch going very negative = looking down)
        if pitch < -HEAD_PITCH_THRESH:
            if head_down_start is None:
                head_down_start = now
            elif (now - head_down_start) > HEAD_NOD_DURATION:
                if not head_nod_detected:
                    head_nod_detected = True
                    head_nod_time = now
                    recent_nods.append(now)
        else:
            head_down_start = None
            if head_nod_detected and (now - head_nod_time) > 2.0:
                head_nod_detected = False

        # Trim old nods
        while recent_nods and (now - recent_nods[0]) > 60:
            recent_nods.popleft()

        # Draw nose direction line
        nose = (int(lm[1].x * w), int(lm[1].y * h))
        end_x = int(nose[0] + yaw * 2)
        end_y = int(nose[1] - pitch * 2)
        cv2.arrowedLine(display, nose, (end_x, end_y), (255, 200, 0), 2, tipLength=0.3)

    # ---- DROWSINESS SCORING ----
    # Eyes: sustained closure (3s+) is a direct drowsiness trigger
    perclos_score = min(perclos / PERCLOS_DROWSY, 1.0)
    # Yawn: instant boost when actively yawning, plus accumulated
    yawn_instant = 0.8 if yawn_active else 0.0
    yawn_accum = min(len(recent_yawns) / 2.0, 1.0)
    yawn_score = max(yawn_instant, yawn_accum)
    # Head: only counts after 3s sustained nod
    head_instant = 0.8 if head_nod_detected else 0.0
    head_accum = min(len(recent_nods) / 2.0, 1.0)
    head_score = max(head_instant, head_accum)

    drowsy_score = (W_PERCLOS * perclos_score +
                    W_YAWN * yawn_score +
                    W_HEAD * head_score)

    # Override: sustained eye closure or head nod = immediate drowsiness
    if eyes_sustained_closed or head_nod_detected:
        drowsy_score = max(drowsy_score, 0.85)

    is_drowsy = drowsy_score >= DROWSY_SCORE_THRESH

    if is_drowsy and not alert_active:
        alert_active = True
        alert_start_time = now
    elif not is_drowsy:
        alert_active = False

    # ====================== DRAW UI ==========================

    # --- Right Panel Stats ---
    px = panel_x + 12
    py_start = 20

    cv2.putText(display, "DROWSINESS MONITOR", (px, py_start),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    cv2.line(display, (px, py_start + 8), (w - 10, py_start + 8), (80, 80, 80), 1)

    # EAR
    py = py_start + 35
    ear_text = f"EAR: {ear_avg:.2f}"
    ear_color = (0, 0, 255) if ear_avg < EAR_THRESHOLD else (0, 220, 0)
    cv2.putText(display, ear_text, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ear_color, 1)

    # PERCLOS meter
    py += 25
    draw_meter(display, px, py, 150, 14, perclos, "PERCLOS", threshold=PERCLOS_DROWSY)

    # Head pose
    py += 40
    cv2.putText(display, f"Pitch: {pitch:.1f}", (px, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(display, f"Yaw: {yaw:.1f}", (px + 90, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    py += 16
    cv2.putText(display, f"Roll: {roll:.1f}", (px, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Yawn info
    py += 30
    yawn_color = (0, 0, 255) if yawn_active else (0, 220, 0)
    cv2.putText(display, f"Yawn: {'YES' if yawn_active else 'NO'}", (px, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, yawn_color, 1)
    py += 20
    cv2.putText(display, f"Yawns (60s): {len(recent_yawns)}", (px, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Status icons
    py += 35
    cv2.line(display, (px, py - 12), (w - 10, py - 12), (80, 80, 80), 1)
    cv2.putText(display, "STATUS", (px, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

    py += 25
    draw_status_icon(display, px + 8, py, eyes_sustained_closed, "Eyes Drowsy")
    py += 25
    draw_status_icon(display, px + 8, py, yawn_active, "Yawning")
    py += 25
    draw_status_icon(display, px + 8, py, head_nod_detected, "Head Nod")

    # Overall drowsiness score meter
    py += 35
    cv2.line(display, (px, py - 12), (w - 10, py - 12), (80, 80, 80), 1)
    draw_meter(display, px, py, 150, 18, drowsy_score, "DROWSY SCORE",
               color_low=(0, 180, 0), color_high=(0, 0, 255),
               threshold=DROWSY_SCORE_THRESH)

    # Face detection status
    py += 45
    if not face_detected:
        cv2.putText(display, "NO FACE", (px, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)

    # --- Alert Banner ---
    if alert_active:
        # Flashing red banner
        flash = int((now - alert_start_time) * 4) % 2 == 0
        if flash:
            overlay2 = display.copy()
            cv2.rectangle(overlay2, (0, 0), (panel_x, 50), (0, 0, 200), -1)
            cv2.addWeighted(overlay2, 0.6, display, 0.4, 0, display)
            cv2.putText(display, "DROWSINESS ALERT!", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        else:
            overlay2 = display.copy()
            cv2.rectangle(overlay2, (0, 0), (panel_x, 50), (0, 0, 150), -1)
            cv2.addWeighted(overlay2, 0.4, display, 0.6, 0, display)
            cv2.putText(display, "DROWSINESS ALERT!", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 255), 2)

        # Warning border
        cv2.rectangle(display, (0, 0), (panel_x - 1, h - 1), (0, 0, 255), 3)

    # --- FPS ---
    frame_cnt += 1
    if frame_cnt % 10 == 0:
        fps = 10 / (time.time() - fps_timer + 1e-6)
        fps_timer = time.time()
    if frame_cnt >= 10:
        cv2.putText(display, f"FPS: {fps:.0f}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1)

    cv2.putText(display, "'q' to quit", (panel_x + 12, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 130, 130), 1)

    cv2.imshow("Driver Drowsiness Detection", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera released. Goodbye!")
