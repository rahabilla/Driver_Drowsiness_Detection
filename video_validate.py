import cv2
import numpy as np
import tensorflow as tf
import shutil
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from eye_crop_best import detect_best_eye


# ======================================================
# SETTINGS
# ======================================================
VIDEO_PATH = r"D:\miniproject\demo_videos\WhatsApp Video 2025-11-17 at 3.14.47 PM.mp4"
MODEL_PATH = "final_model.h5"
INPUT_SIZE = (224, 224)
CLASS_NAMES = ["Closed", "Open", "no_yawn", "yawn"]


# ======================================================
# LOAD MODEL
# ======================================================
print("🔄 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded:", MODEL_PATH)


# ======================================================
# PREPROCESSING
# ======================================================
def preprocess_frame(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, INPUT_SIZE)
    img = np.expand_dims(img, 0).astype(np.float32)
    return preprocess_input(img)


# ======================================================
# SCREEN-FIT RESIZER
# ======================================================
def resize_to_fit(frame, max_width=800, max_height=500):
    """Resize while keeping aspect ratio so it fits inside max box."""
    h, w = frame.shape[:2]

    # Compute scaling factors for width and height
    scale_w = max_width / w
    scale_h = max_height / h

    # Pick the SMALLER scale so the frame fits BOTH limits
    scale = min(scale_w, scale_h, 1.0)  # never upscale

    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ======================================================
# PREDICTION PIPELINE
# ======================================================
def predict_state(frame):

    # -------------------------------
    # 1️⃣ FACE prediction (4 classes)
    # -------------------------------
    face_input = preprocess_frame(frame)
    pred_face = model.predict(face_input, verbose=0)
    face_label = CLASS_NAMES[int(np.argmax(pred_face))]
    face_conf = float(np.max(pred_face))

    # -------------------------------
    # 2️⃣ EYE prediction (force 2 classes)
    # -------------------------------
    eye_crop, eye_score, _, _ = detect_best_eye(frame)

    if eye_crop is not None and eye_score > 0.001:
        eye_input = preprocess_frame(eye_crop)
        pred_eye = model.predict(eye_input, verbose=0)

        # === OPTION 1 FIX: ONLY USE FIRST 2 LOGITS ===
        eye_logits = pred_eye[0][:2]     # [Closed, Open]
        eye_label = ["Closed", "Open"][int(np.argmax(eye_logits))]
        eye_conf = float(np.max(eye_logits))

    else:
        # fallback if eye not detected
        eye_label = face_label
        eye_conf = face_conf

    # -------------------------------
    # 3️⃣ FINAL STATE LOGIC
    # -------------------------------
    if eye_label == "Closed" or face_label == "yawn":
        state = "Drowsy"
    else:
        state = "Non-Drowsy"

    return face_label, eye_label, state, eye_conf



# ======================================================
# VIDEO LOOP
# ======================================================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ ERROR: Cannot open video:", VIDEO_PATH)
    exit()

print("🎥 Processing video... Press Q to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_label, eye_label, state, conf = predict_state(frame)

    # -----------------------------------
    # Minimalistic Bright Colors
    # -----------------------------------
    WHITE  = (255, 255, 255)
    GREEN  = (0, 255, 0)      # bright green
    RED    = (0, 0, 255)      # bright red (fixed)
    YELLOW = (0, 255, 255)

    # -------------------
    # DRAW CLEAN TEXT
    # -------------------
    cv2.putText(frame, f"Face: {face_label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, WHITE, 2)

    cv2.putText(frame, f"Eye : {eye_label}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, WHITE, 2)

    state_color = RED if state == "Drowsy" else GREEN

    cv2.putText(frame, f"State: {state}", (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, state_color, 3)

    # FIT TO SCREEN
    display_frame = resize_to_fit(frame, max_width=1280, max_height=720)
    cv2.imshow("Driver Monitoring", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
