# Driver Drowsiness Detection

This project implements a **driver drowsiness detection system** using deep learning. It combines **face and eye analysis** to detect signs of drowsiness (closed eyes or yawning) from images and videos. The model is based on **EfficientNetB0** and optimized for CPU usage.

---

## **Project Structure**

```

Driver_Drowsiness_Detection/
│
├── train_effnet_model.py    # Train EfficientNetB0 model on eye/face dataset
├── eye_crop_best.py         # Eye detection helper using Haar cascades
├── video_validate.py        # Real-time drowsiness detection on video
├── image_validate.py        # Drowsiness prediction on single image
├── splitted_Data/           # Train/val/test dataset folders
├── requirements.txt         # Python dependencies
└── README.md                # Project overview and instructions

````

---

## **Setup & Installation**

1. Clone the repository:

```bash
git clone https://github.com/rahabilla/Driver_Drowsiness_Detection.git
cd Driver_Drowsiness_Detection
````

2. Create a virtual environment:

```bash
python -m venv venv
# Activate the virtual environment
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

**`requirements.txt` contents:**

```
tensorflow==2.15.0
numpy
opencv-python
matplotlib
scikit-learn
```

---

## **1️⃣ Train the Model**

```bash
python train_effnet_model.py
```

* Trains **EfficientNetB0** on the dataset in `splitted_Data/train`, `val`, and `test`.
* Uses light augmentation for training and CPU-friendly batch sizes.
* Saves the following models:

  * `best_model.h5` → best performing model during training
  * `final_model.h5` → final trained model
  * `best_model_quant.tflite` → optional quantized model for faster CPU inference
* Generates a **confusion matrix** (`confusion_matrix.png`) for evaluation.

---

## **2️⃣ Test on a Single Image**

```bash
python image_validate.py
```

* Uses `best_model.h5` for predictions.
* Detects eyes using `eye_crop_best.py`.
* Combines face and eye predictions to determine **Drowsy** or **Non-Drowsy**.
* Displays the image with predicted labels and driver state.

---

## **3️⃣ Test on Video**

```bash
python video_validate.py
```

* Uses `final_model.h5` for real-time predictions on a video.
* Displays frame-by-frame:

  * Face state
  * Eye state
  * Drowsiness status (Drowsy / Non-Drowsy)
* Press **`Q`** to quit the video display.

**Logic:**

* `Drowsy` if `eye == Closed` or `face == yawn`
* Otherwise `Non-Drowsy`

---

## **Dataset Structure**

```
splitted_Data/
├── train/
│   ├── Closed/
│   ├── Open/
│   ├── no_yawn/
│   └── yawn/
├── val/
│   ├── Closed/
│   ├── Open/
│   ├── no_yawn/
│   └── yawn/
└── test/
    ├── Closed/
    ├── Open/
    ├── no_yawn/
    └── yawn/
```

* Each folder contains images for the corresponding class.

---

## **Notes**

* Eye detection uses Haar cascades and is fast on CPU.
* Model input size is **224x224** for both face and eye crops.
* TFLite conversion provides faster CPU inference.
* For better video performance, consider frame skipping or temporal smoothing.

---

## **Acknowledgements**

* [TensorFlow EfficientNet](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0)
* OpenCV Haar cascades for eye detection

```
