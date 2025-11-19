import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from eye_crop_best import detect_best_eye

# ======================================================
# SETTINGS
# ======================================================
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
def resize_to_fit(frame, max_width=800, max_height=600):
    h, w = frame.shape[:2]
    scale_w = max_width / w
    scale_h = max_height / h

    scale = min(scale_w, scale_h, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ======================================================
# PREDICTION PIPELINE
# ======================================================
def predict_state(frame):

    # 1️⃣ FACE prediction
    face_input = preprocess_frame(frame)
    pred_face = model.predict(face_input, verbose=0)

    face_idx = int(np.argmax(pred_face))
    face_label = CLASS_NAMES[face_idx]
    face_conf = float(np.max(pred_face))

    # 2️⃣ EYE prediction
    eye_crop, eye_score, _, _ = detect_best_eye(frame)

    if eye_crop is not None and eye_score > 0.001:
        eye_input = preprocess_frame(eye_crop)
        pred_eye = model.predict(eye_input, verbose=0)

        # Use only first 2 logits: [Closed, Open]
        eye_logits = pred_eye[0][:2]
        eye_idx = int(np.argmax(eye_logits))
        eye_label = ["Closed", "Open"][eye_idx]
        eye_conf = float(np.max(eye_logits))
    else:
        eye_label = face_label
        eye_conf = face_conf

    # 3️⃣ Drowsiness Logic
    if eye_label == "Closed" or face_label == "yawn":
        state = "Drowsy"
    else:
        state = "Non-Drowsy"

    return face_label, eye_label, state, eye_conf


# ======================================================
# WEBCAM LOOP
# ======================================================
cap = cv2.VideoCapture(0)   # webcam index

if not cap.isOpened():
    print("❌ ERROR: Unable to access webcam!")
    exit()

print("🎥 Webcam running... Press Q to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame.")
        break

    face_label, eye_label, state, conf = predict_state(frame)

    # Colors
    WHITE  = (255, 255, 255)
    GREEN  = (0, 255, 0)
    RED    = (0, 0, 255)

    # Draw text
    cv2.putText(frame, f"Face: {face_label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)

    cv2.putText(frame, f"Eye : {eye_label}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)

    color = RED if state == "Drowsy" else GREEN
    cv2.putText(frame, f"State: {state}", (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Resize for display
    display_frame = resize_to_fit(frame, max_width=1280, max_height=720)
    cv2.imshow("Driver Monitoring - Webcam", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
