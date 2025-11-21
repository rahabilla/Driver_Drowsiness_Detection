
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from eye_crop_best import detect_best_eye  # your Haar-based eye detector

# Load trained model
model = load_model('best_model.h5')

# Class names
class_names = ['Closed', 'Open', 'no_yawn', 'yawn']

# ---- CHANGE THIS TO ANY REAL EXISTING IMAGE ----
image_path = r"D:\miniproject\splitted_Data\test\no_yawn\2093.jpg"
# -------------------------------------------------

# Load the image
image = cv2.imread(image_path)
if image is None:
    print("ERROR: Could not load image:", image_path)
    exit()

# Convert to RGB for plotting
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Preprocess face image for model
face_img = cv2.resize(image_rgb, (224, 224))
face_input = np.expand_dims(face_img, axis=0)
face_input = preprocess_input(face_input)

# ---- Eye detection & preprocessing ----
eye, best_score, all_eyes, all_scores = detect_best_eye(image_path)

if eye is None:
    print("WARNING: Eye detection failed. Using full face image as fallback for eye input.")
    eye_input_img = face_img.copy()
else:
    eye_rgb = cv2.cvtColor(eye, cv2.COLOR_BGR2RGB)
    eye_input_img = cv2.resize(eye_rgb, (224, 224))

# Prepare model input
eye_input = np.expand_dims(eye_input_img, axis=0)
eye_input = preprocess_input(eye_input)

# ---- Predictions ----
pred_face = model.predict(face_input)
pred_eye = model.predict(eye_input)

face_class = class_names[np.argmax(pred_face)]
eye_class = class_names[np.argmax(pred_eye)]

face_conf = float(np.max(pred_face))
eye_conf = float(np.max(pred_eye))

print(f"Face: {face_class} ({face_conf:.2f})")
print(f"Eye: {eye_class} ({eye_conf:.2f})")

# ---- Drowsiness decision ----
if (face_class == "yawn") or (eye_class == "Closed"):
    state = "Drowsy"
else:
    state = "Non-Drowsy"

avg_conf = (face_conf + eye_conf) / 2
print("State:", state)

# ---- Plot result ----
plt.imshow(image_rgb)
plt.title(f"{face_class} + {eye_class} → {state} ({avg_conf:.2f})")
plt.axis('off')
plt.show()

