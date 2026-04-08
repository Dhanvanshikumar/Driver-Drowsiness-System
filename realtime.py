import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import os
import time

# ---- Setup ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

base_dir = os.path.dirname(os.path.abspath(__file__))

# ---- Load Model ----
model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.last_channel, 2)
)

model_path = os.path.join(base_dir, "yawn_model.pth")
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model = model.to(device)
model.eval()
print("Model loaded successfully.")

# ---- Image Transform (same as validation) ----
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---- Face Detector ----
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
print("Face detector loaded.")

classes = ["No Yawn", "Yawn"]
colors = {
    "No Yawn": (0, 200, 0),    # Green
    "Yawn": (0, 0, 255),       # Red
}

# ---- Smoothing ----
prediction_history = []
HISTORY_SIZE = 7
YAWN_CONFIDENCE_THRESHOLD = 0.65  # Only count as yawn if confidence > this


def extract_mouth_region(frame, face_rect):
    """
    Extract the mouth/lower-face region from a detected face.
    The training data contains crops of the mouth area (lower ~50% of face),
    so we need to crop similarly from the webcam face detection.
    """
    x, y, w, h = face_rect
    frame_h, frame_w = frame.shape[:2]

    # The mouth region is roughly the bottom 45-50% of the face
    # with a little extra width for cheeks
    mouth_y_start = y + int(h * 0.45)    # Start from ~45% down the face
    mouth_y_end = y + int(h * 1.05)      # Go slightly below face box
    mouth_x_start = x - int(w * 0.05)    # Slight horizontal padding
    mouth_x_end = x + w + int(w * 0.05)

    # Clamp to frame boundaries
    mouth_y_start = max(0, mouth_y_start)
    mouth_y_end = min(frame_h, mouth_y_end)
    mouth_x_start = max(0, mouth_x_start)
    mouth_x_end = min(frame_w, mouth_x_end)

    return (mouth_x_start, mouth_y_start, mouth_x_end, mouth_y_end)


def predict_yawn(roi_bgr):
    """Run yawn prediction on a mouth ROI (BGR numpy array)."""
    face_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(face_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)
    return classes[pred.item()], confidence.item(), probs[0].cpu().numpy()


def get_smoothed_prediction(label, confidence):
    """Smooth predictions using rolling window majority vote with threshold."""
    # Only add "Yawn" if confidence is above threshold
    effective_label = label if (label == "Yawn" and confidence >= YAWN_CONFIDENCE_THRESHOLD) else "No Yawn"
    if label == "No Yawn":
        effective_label = "No Yawn"

    prediction_history.append(effective_label)
    if len(prediction_history) > HISTORY_SIZE:
        prediction_history.pop(0)

    yawn_count = prediction_history.count("Yawn")
    # Need majority (more than half) of recent frames to say "Yawn"
    if yawn_count > HISTORY_SIZE // 2:
        return "Yawn"
    return "No Yawn"


# ---- Main Loop ----
print("\nStarting camera... Press 'q' to quit.\n")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fps = 0
fps_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    display = frame.copy()
    h, w = frame.shape[:2]

    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
    )

    if len(faces) > 0:
        # Use the largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        fx, fy, fw, fh = faces[0]

        # Extract mouth region (matching training data crop style)
        mx1, my1, mx2, my2 = extract_mouth_region(frame, (fx, fy, fw, fh))
        mouth_roi = frame[my1:my2, mx1:mx2]

        if mouth_roi.size > 0 and mouth_roi.shape[0] > 10 and mouth_roi.shape[1] > 10:
            # Predict
            label, confidence, probs = predict_yawn(mouth_roi)
            smoothed_label = get_smoothed_prediction(label, confidence)
            color = colors[smoothed_label]

            # Draw face box (thin, gray)
            cv2.rectangle(display, (fx, fy), (fx + fw, fy + fh), (150, 150, 150), 1)

            # Draw mouth region box (colored)
            cv2.rectangle(display, (mx1, my1), (mx2, my2), color, 2)

            # Label on mouth box
            label_text = f"{smoothed_label} ({confidence:.0%})"
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(display, (mx1, my1 - 28), (mx1 + text_size[0] + 10, my1), color, -1)
            cv2.putText(display, label_text, (mx1 + 5, my1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Probability bars at bottom
            bar_y = h - 60
            bar_w = 200
            bar_h = 18

            cv2.putText(display, "No Yawn:", (10, bar_y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(display, (110, bar_y - 15), (110 + bar_w, bar_y - 15 + bar_h), (50, 50, 50), -1)
            cv2.rectangle(display, (110, bar_y - 15),
                          (110 + int(bar_w * probs[0]), bar_y - 15 + bar_h), (0, 200, 0), -1)

            cv2.putText(display, "Yawn:", (10, bar_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(display, (110, bar_y + 15), (110 + bar_w, bar_y + 15 + bar_h), (50, 50, 50), -1)
            cv2.rectangle(display, (110, bar_y + 15),
                          (110 + int(bar_w * probs[1]), bar_y + 15 + bar_h), (0, 0, 255), -1)

            # Yawn alert banner
            if smoothed_label == "Yawn":
                overlay = display.copy()
                cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 200), -1)
                cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
                cv2.putText(display, "!! YAWN DETECTED !!", (w // 2 - 140, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        cv2.putText(display, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    # FPS counter
    frame_count += 1
    if frame_count % 10 == 0:
        fps = 10 / (time.time() - fps_time + 1e-6)
        fps_time = time.time()
    if frame_count >= 10:
        cv2.putText(display, f"FPS: {fps:.0f}", (w - 100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(display, "Press 'q' to quit", (w - 170, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow("Yawn Detection - Real Time", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera released. Goodbye!")
