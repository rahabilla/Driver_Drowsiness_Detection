"""
eye_crop_best.py
Detect best eye region from a BGR image (numpy array) or from an image path.
Returns: (best_eye_bgr (224x224), best_score, all_crops, all_scores)
If no eye found returns (None, 0, [], [])
This version avoids writing to disk and is tuned for speed on CPU.
"""

import cv2
import numpy as np
import os

# Use OpenCV Haar cascade (bundled)
_EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"
_eye_cascade = cv2.CascadeClassifier(_EYE_CASCADE_PATH)

# Target size for the classifier input
TARGET_SIZE = (224, 224)

def sharpness_score(gray):
    # Laplacian variance - fast and effective
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def contrast_score(gray):
    # Combined std + dynamic range
    nonzero = gray[gray > 10]
    if len(nonzero) < 10:
        return 0.0
    p10 = np.percentile(nonzero, 10)
    p90 = np.percentile(nonzero, 90)
    return gray.std() * 0.6 + (p90 - p10) * 0.4

def completeness_score(gray):
    # Simple heuristic: center intensity vs edges
    h, w = gray.shape
    cx1, cx2 = w // 4, 3 * w // 4
    cy1, cy2 = h // 4, 3 * h // 4
    center = gray[cy1:cy2, cx1:cx2]
    if center.size == 0:
        return 0.0
    return abs(float(center.mean()) - float(gray.mean()))

def combined_quality(gray):
    # Quick reject for very dark/bright or too small
    if gray.size < 25 * 25:
        return 0.0
    mean = gray.mean()
    if mean < 20 or mean > 235:
        return 0.0
    s = sharpness_score(gray)
    c = contrast_score(gray)
    comp = completeness_score(gray)
    s_norm = min(s, 300) / 300.0 * 100.0
    return 0.45 * s_norm + 0.35 * c + 0.2 * comp

def _detect_eyes_in_gray(gray):
    # Faster detection options: increase scaleFactor slightly and minNeighbors
    eyes = _eye_cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return eyes

def detect_best_eye(input_image):
    """
    Accepts either:
      - a numpy BGR image (preferred), or
      - a string path to an image
    Returns:
      best_eye_bgr_resized (224x224) or None, best_score, list_of_color_crops, list_of_scores
    """
    if isinstance(input_image, str):
        if not os.path.exists(input_image):
            raise ValueError(f"Image path not found: {input_image}")
        img = cv2.imread(input_image)
        if img is None:
            raise ValueError(f"Could not read image: {input_image}")
    else:
        img = input_image

    if img is None:
        return None, 0, [], []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    eyes = _detect_eyes_in_gray(gray)
    if len(eyes) == 0:
        return None, 0, [], []

    crops = []
    scores = []
    rects = []

    for (x, y, w, h) in eyes:
        # expand a little for context
        ex = max(2, int(w * 0.12))
        ey = max(2, int(h * 0.12))
        x1 = max(0, x - ex)
        y1 = max(0, y - ey)
        x2 = min(img.shape[1], x + w + ex)
        y2 = min(img.shape[0], y + h + ey)
        crop_color = img[y1:y2, x1:x2]
        crop_gray = gray[y1:y2, x1:x2]
        score = combined_quality(crop_gray)
        crops.append(crop_color)
        scores.append(score)
        rects.append((x1, y1, x2 - x1, y2 - y1))

    if len(scores) == 0:
        return None, 0, [], []

    best_idx = int(np.argmax(scores))
    best_crop = crops[best_idx]

    # resize best crop to target model input size
    best_resized = cv2.resize(best_crop, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

    return best_resized, float(scores[best_idx]), crops, scores

# If run as script, small demo
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        crop, score, crops, scores = detect_best_eye(path)
        if crop is not None:
            print("Best score:", score)
            cv2.imshow("best eye", crop)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No eye found....")

