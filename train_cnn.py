"""
CNN Eye State Classifier — Training Script
==========================================
Trains a lightweight CNN to classify eye images as Open or Closed.

Dataset structure expected:
    dataset/
        open/      <- images of open eyes
        closed/    <- images of closed eyes

Recommended dataset: MRL Eye Dataset
    http://mrl.cs.vsb.cz/eyedataset

Usage:
    python train_cnn.py
    python train_cnn.py --data dataset --epochs 25 --batch 32
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless-safe backend
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser(description="Train CNN eye classifier")
    ap.add_argument("--data",   default="dataset",
                    help="Root folder with open/ and closed/ subfolders")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch",  type=int, default=32)
    ap.add_argument("--output", default="model/eye_model.h5",
                    help="Where to save the trained model")
    ap.add_argument("--plot",   default="training_plot.png",
                    help="Where to save the training accuracy/loss plot")
    return vars(ap.parse_args())


def build_model():
    """Lightweight CNN suitable for 24x24 grayscale eye images."""
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu",
                      padding="same", input_shape=(24, 24, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"],     label="Train acc")
    ax1.plot(history.history["val_accuracy"], label="Val acc")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history.history["loss"],     label="Train loss")
    ax2.plot(history.history["val_loss"], label="Val loss")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[INFO] Training plot saved to {save_path}")


def main():
    args = parse_args()

    for split in ["open", "closed"]:
        path = os.path.join(args["data"], split)
        if not os.path.isdir(path):
            print(f"[ERROR] Missing folder: {path}")
            print("  Create dataset/open/ and dataset/closed/ with eye images.")
            sys.exit(1)

    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except ImportError:
        print("[ERROR] TensorFlow not installed. Run: pip install tensorflow")
        sys.exit(1)

    print(f"[INFO] TensorFlow version: {tf.__version__}")
    print(f"[INFO] Loading data from: {args['data']}")

    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.20,
        rotation_range=12,
        zoom_range=0.12,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        width_shift_range=0.1,
        height_shift_range=0.1,
    )

    train_gen = datagen.flow_from_directory(
        args["data"], target_size=(24, 24), color_mode="grayscale",
        batch_size=args["batch"], class_mode="binary",
        subset="training", shuffle=True,
    )
    val_gen = datagen.flow_from_directory(
        args["data"], target_size=(24, 24), color_mode="grayscale",
        batch_size=args["batch"], class_mode="binary",
        subset="validation", shuffle=False,
    )

    print(f"[INFO] Classes: {train_gen.class_indices}")
    print(f"[INFO] Training samples : {train_gen.samples}")
    print(f"[INFO] Validation samples: {val_gen.samples}")

    model = build_model()
    model.summary()

    os.makedirs(os.path.dirname(args["output"]) or ".", exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            args["output"], monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-6, verbose=1
        ),
    ]

    print(f"\n[INFO] Training for up to {args['epochs']} epochs...")
    history = model.fit(
        train_gen, validation_data=val_gen,
        epochs=args["epochs"], callbacks=callbacks,
    )

    best_acc = max(history.history["val_accuracy"])
    print(f"\n[DONE] Best validation accuracy: {best_acc * 100:.2f}%")
    print(f"       Model saved to: {args['output']}")

    plot_history(history, args["plot"])


if __name__ == "__main__":
    main()
