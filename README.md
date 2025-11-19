#  Driver Drowsiness Detection Using Deep Learning

### EfficientNetB0-Based Eye + Yawn Detection with Video & Webcam Validation

This project implements a **real-time driver drowsiness monitoring system** using **EfficientNetB0**, combining:

* Eye state classification (Open / Closed)
* Yawn classification (yawn / no_yawn)
* Eye-region–only detection using a custom Haar-based cropper
* Live prediction from **video files** and **webcam**

The system uses a **4-class classifier**, then post-processes predictions to decide whether the driver is **Drowsy** or **Non-Drowsy**.

---

## 📌 Features

### ✔ Training (CPU-Optimized)

* EfficientNetB0 with partial layer unfreezing
* tf.data pipeline (cache + prefetch)
* On-the-fly augmentation
* Classification report + confusion matrix
* Automatic **best model saving**
* Optional **INT8 TFLite conversion**

### ✔ Eye Detection (Fast Haar Cascade)

* Extracts best eye crop based on:

  * Sharpness
  * Contrast
  * Completeness score
* Works directly on frames without writing to disk

### ✔ Real-time Inference

* `video_validate.py` → Runs prediction on video file
* `webcam_validate.py` → Runs prediction from webcam
* Auto screen-fit (no stretching)
* Minimal, readable overlay UI

---

## 📂 Project Structure

```
Driver-Drowsiness-Detection/
│
├── train_effnet_model.py        # EfficientNetB0 training pipeline
├── eye_crop_best.py             # High-quality eye-region detector
├── video_validate.py            # Process video file
├── webcam_validate.py           # Webcam inference
│
├── splitted_Data/               # dataset root
│   ├── train/
│   ├── val/
│   └── test/
│
├── final_model.h5               # final trained model (generated)
├── best_model.h5                # best epoch model (generated)
├── best_model_quant.tflite      # optional quantized model
│
├── confusion_matrix.png         # generated during training
├── requirements.txt
└── README.md
```

---

## 🧪 Dataset Format

Place the dataset inside:

```
splitted_Data/
    ├── train/
    │   ├── Closed/
    │   ├── Open/
    │   ├── no_yawn/
    │   └── yawn/
    ├── val/
    └── test/
```

Each folder contains images belonging to that class.

---

## 🔧 Installation & Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Recommended core packages:

```
tensorflow
opencv-python
numpy
matplotlib
scikit-learn
```

---

## 🏋️‍♂️ Training the Model

Run:

```bash
python train_effnet_model.py
```

This will automatically:

* Load dataset
* Train EfficientNetB0
* Save:

  * **best_model.h5**
  * **final_model.h5**
  * **best_model_quant.tflite** (optional INT8)
* Generate confusion matrix

---

## 🎥 Testing with a Video File

Edit `VIDEO_PATH` inside `video_validate.py`
Then run:

```bash
python video_validate.py
```

A window will display:

* Face classification
* Eye classification
* Final state (Drowsy / Non-Drowsy)

Press **Q** to quit.

---

## 📸 Testing with Webcam

Simply run:

```bash
python webcam_validate.py
```

Works with default webcam index `0`.
Press **Q** to exit.

---

## 🧠 Drowsiness Logic

Based on predictions:

```
If eye == Closed  OR face == yawn:
        state = Drowsy
Else:
        state = Non-Drowsy
```

Eye classifier only uses the first **2 classes** of the model
→ `[Closed, Open]`

---

## 🛠 How Eye Detection Works

`eye_crop_best.py` uses Haar cascade → produces multiple eye candidates.
Each crop is scored using:

* Laplacian variance (sharpness)
* Contrast
* Completeness (center vs edges brightness)

Best crop is passed to EfficientNet for eye-only prediction.

---
