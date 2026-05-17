import argparse
import os
from urllib.parse import quote

import numpy as np
import requests
import tensorflow as tf


ALIASES = {
    "phone": "cell phone",
    "mobile": "cell phone",
    "cellphone": "cell phone",
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


def normalize_label(label: str) -> str:
    raw = label.strip()
    key = raw.lower()
    if len(raw) == 1 and raw.isalpha():
        return f"The letter {raw.upper()}"
    if key.startswith("the letter ") and raw[-1].isalpha():
        return f"The letter {raw[-1].upper()}"
    return ALIASES.get(key, raw)


def label_variants(label: str) -> list:
    normalized = normalize_label(label)
    variants = [normalized, normalized.lower()]
    if normalized.lower().startswith("the letter "):
        letter = normalized[-1]
        variants.append(f"The letter {letter.upper()}")
    if normalized in ALIASES:
        variants.append(ALIASES[normalized])
    return list(dict.fromkeys(variants))


def download_class(label: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for candidate in label_variants(label):
        filename = f"{candidate}.npy"
        path = os.path.join(out_dir, filename)
        if os.path.exists(path):
            return path

        url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{quote(candidate)}.npy"
        response = requests.get(url, timeout=30)
        if response.status_code == 404:
            continue
        response.raise_for_status()

        with open(path, "wb") as f:
            f.write(response.content)
        return path

    return None


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Quick Draw CNN.")
    parser.add_argument(
        "--classes",
        type=str,
        default="apple,cat,car,tree,phone",
        help="Comma-separated class labels.",
    )
    parser.add_argument("--labels-file", type=str, default="")
    parser.add_argument("--preset", type=str, default="")
    parser.add_argument("--samples-per-class", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--model-out", type=str, default="model/quickdraw_cnn.h5")
    parser.add_argument("--labels-out", type=str, default="model/labels.txt")
    parser.add_argument("--data-dir", type=str, default="dataset")
    args = parser.parse_args()

    if args.labels_file:
        with open(args.labels_file, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
    elif args.preset.lower() == "digits":
        labels = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "plus", "minus", "multiplication", "divide"]
        args.model_out = "model/digit_cnn.h5"
        args.labels_out = "model/digit_labels.txt"
    else:
        labels = [label.strip() for label in args.classes.split(",") if label.strip()]
    if not labels:
        raise ValueError("Provide at least one label.")

    data = []
    target = []

    labels_used = []
    print(f"Loading labels: {len(labels)}")
    for idx, label in enumerate(labels):
        print(f"Downloading {label}...")
        npy_path = download_class(label, args.data_dir)
        if npy_path is None:
            print(f"Skipping label (not found): {label}")
            continue
        raw = np.load(npy_path)
        if args.samples_per_class < len(raw):
            raw = raw[: args.samples_per_class]
        data.append(raw)
        target.append(np.full(len(raw), len(labels_used), dtype=np.int32))
        labels_used.append(label)

    if not data:
        raise ValueError("No valid labels were downloaded. Check label names.")

    x = np.concatenate(data, axis=0)
    y = np.concatenate(target, axis=0)

    x = x.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    indices = np.random.permutation(len(x))
    x = x[indices]
    y = y[indices]

    split_idx = int(len(x) * 0.9)
    x_train, x_val = x[:split_idx], x[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    model = build_model(len(labels_used))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    print(f"Training samples: {len(x_train)}, validation samples: {len(x_val)}")
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=args.epochs, batch_size=128)

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    model.save(args.model_out)
    if args.model_out.endswith(".h5"):
        keras_out = args.model_out[:-3] + ".keras"
        model.save(keras_out)

    with open(args.labels_out, "w", encoding="utf-8") as f:
        for label in labels_used:
            f.write(label + "\n")

    print(f"Saved model to {args.model_out}")
    print(f"Saved labels to {args.labels_out}")


if __name__ == "__main__":
    main()
