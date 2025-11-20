import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import sys

# ---------------------------
# CPU-only: hide GPUs (important for non-GPU machines)
# ---------------------------
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

print("TensorFlow version:", tf.__version__)
print("Running on devices:", tf.config.get_visible_devices())

# ---------------------------
# Settings (tune these)
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "splitted_Data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 12
AUTO = tf.data.AUTOTUNE
NUM_CLASSES = 4
CLASS_NAMES = ["Closed", "Open", "no_yawn", "yawn"]
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "best_model.h5")
FINAL_SAVE_PATH = os.path.join(BASE_DIR, "final_model.h5")
TFLITE_PATH = os.path.join(BASE_DIR, "best_model_quant.tflite")

# ---------------------------
# Helpers: plotting confusion matrix
# ---------------------------
def plot_confusion_matrix(cm, classes, out_path=None):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], "d"),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    plt.show()

# ---------------------------
# Dataset creation with tf.data
# ---------------------------
def make_datasets(train_dir, val_dir, test_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False
    )

    from tensorflow.keras.applications.efficientnet import preprocess_input
    normalization = lambda x, y: (preprocess_input(x), y)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.Resizing(img_size, img_size),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.12),
        tf.keras.layers.RandomZoom(0.12),
        tf.keras.layers.RandomTranslation(0.06, 0.06)
    ], name="data_augmentation")

    def prepare_train(ds):
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTO)
        ds = ds.map(normalization, num_parallel_calls=AUTO)
        ds = ds.cache()
        ds = ds.prefetch(AUTO)
        return ds

    def prepare_eval(ds):
        ds = ds.map(normalization, num_parallel_calls=AUTO)
        ds = ds.cache()
        ds = ds.prefetch(AUTO)
        return ds

    return prepare_train(train_ds), prepare_eval(val_ds), prepare_eval(test_ds)

# ---------------------------
# Build model
# ---------------------------
def build_model(img_size=IMG_SIZE, num_classes=NUM_CLASSES):
    base = tf.keras.applications.EfficientNetB0(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet"
    )

    for layer in base.layers[:-20]:
        layer.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = tf.keras.layers.Dense(512, activation="relu", name="dense_512")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout")(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return tf.keras.Model(inputs=base.input, outputs=out, name="EffNetB0_head")

# ---------------------------
# Train + evaluate
# ---------------------------
def train_and_evaluate():
    train_ds, val_ds, test_ds = make_datasets(TRAIN_DIR, VAL_DIR, TEST_DIR)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="val_loss")
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # ==========================
    # ACCURACY & LOSS GRAPHS
    # ==========================
    plt.figure(figsize=(14, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()
    # ==========================

    model.save(FINAL_SAVE_PATH)
    print("Saved final model to", FINAL_SAVE_PATH)

    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc*100:.2f}%, loss: {test_loss:.4f}")

    y_true, y_pred = [], []
    for batch_x, batch_y in test_ds:
        pred = model.predict(batch_x, verbose=0)
        y_pred.extend(np.argmax(pred, axis=1).tolist())
        y_true.extend(np.argmax(batch_y.numpy(), axis=1).tolist())

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, CLASS_NAMES, out_path="confusion_matrix.png")

    print("Best model path:", MODEL_SAVE_PATH)

    try:
        convert_to_tflite = True
        if convert_to_tflite:
            print("Converting best_model.h5 to quantized TFLite...")
            loaded = tf.keras.models.load_model(MODEL_SAVE_PATH)

            def representative_gen():
                for batch_x, _ in test_ds.take(10):
                    yield [tf.cast(batch_x, tf.float32)]

            converter = tf.lite.TFLiteConverter.from_keras_model(loaded)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8

            tflite_model = converter.convert()
            with open(TFLITE_PATH, "wb") as f:
                f.write(tflite_model)

            print("Saved quantized TFLite to", TFLITE_PATH)

    except Exception as e:
        print("TFLite conversion skipped:", str(e))

    return model, history


if __name__ == "__main__":
    train_and_evaluate()
