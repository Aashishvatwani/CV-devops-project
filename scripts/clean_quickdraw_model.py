import argparse
import os

import tensorflow as tf


def build_model(num_classes: int) -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )


def load_labels(labels_path: str) -> list:
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    if not labels:
        raise ValueError("Labels file is empty.")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and resave a Quick Draw model.")
    parser.add_argument("--model-in", default="model/quickdraw_cnn.h5")
    parser.add_argument("--labels", default="model/labels.txt")
    parser.add_argument("--model-out", default="model/quickdraw_cnn.keras")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    num_classes = len(labels)

    if not args.force_rebuild:
        # Try direct load first (may fail on quantization_config).
        try:
            model = tf.keras.models.load_model(args.model_in, compile=False)
            model.save(args.model_out)
            print(f"Saved cleaned model to {args.model_out}")
            return
        except Exception:
            pass

    # Rebuild architecture and load weights.
    model = build_model(num_classes)
    model.load_weights(args.model_in)
    model.save(args.model_out)
    print(f"Saved cleaned model to {args.model_out}")


if __name__ == "__main__":
    main()
