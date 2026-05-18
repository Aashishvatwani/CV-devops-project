from typing import Tuple
import os

import cv2
import numpy as np


class SketchPredictor:
    def __init__(
        self,
        model_path: str = "model/quickdraw_cnn.h5",
        labels_path: str = "model/labels.txt"
    ) -> None:
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.labels = None

    def _resolve_model_path(self) -> str:
        base, ext = os.path.splitext(self.model_path)
        candidates = [self.model_path]
        if ext == ".keras":
            candidates.append(base + ".h5")
        elif ext == ".h5":
            candidates.append(base + ".keras")

        for path in candidates:
            if os.path.exists(path):
                return path

        return self.model_path

    def _load_model(self) -> None:
        if self.model is not None:
            return

        model_path = self._resolve_model_path()

        if os.path.exists(self.labels_path):
            try:
                with open(self.labels_path, "r", encoding="utf-8") as f:
                    self.labels = [line.strip() for line in f if line.strip()]
            except Exception as e:
                print(f"Labels loading failed: {e}")

        if not os.path.exists(model_path):
            print(f"Model weights not found: {model_path}")
            return

        num_classes = len(self.labels) if self.labels else 10

        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
        except Exception as e:
            print(f"TensorFlow import error: {e}")
            return

        try:
            model = Sequential(
                [
                    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
                    MaxPooling2D(pool_size=(2, 2)),
                    Conv2D(64, (3, 3), activation="relu"),
                    MaxPooling2D(pool_size=(2, 2)),
                    Flatten(),
                    Dense(128, activation="relu"),
                    Dropout(0.3),
                    Dense(num_classes, activation="softmax"),
                ]
            )
            model.load_weights(model_path)
            self.model = model
            print("Model loaded successfully")
        except Exception as e:
            print(f"Model loading failed: {e}")
            self.model = None

    def predict(self, canvas_bgr: np.ndarray) -> Tuple[str, float]:
        model_path = self._resolve_model_path()

        if not os.path.exists(model_path):
            return "model_missing", 0.0

        self._load_model()

        if self.model is None:
            return "model_load_failed", 0.0

        try:
            # Handle already resized grayscale image
            if (
                canvas_bgr.ndim == 2
                and canvas_bgr.shape[:2] == (28, 28)
            ):
                normalized = canvas_bgr.astype("float32") / 255.0
                input_tensor = normalized.reshape(1, 28, 28, 1)

            # Handle 3-channel image already 28x28
            elif (
                canvas_bgr.ndim == 3
                and canvas_bgr.shape[:2] == (28, 28)
            ):
                gray = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY)
                normalized = gray.astype("float32") / 255.0
                input_tensor = normalized.reshape(1, 28, 28, 1)

            # Handle any other input image
            else:
                gray = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(
                    gray,
                    (28, 28),
                    interpolation=cv2.INTER_AREA
                )

                normalized = resized.astype("float32") / 255.0
                input_tensor = normalized.reshape(1, 28, 28, 1)

            preds = self.model.predict(input_tensor, verbose=0)

            idx = int(np.argmax(preds))
            conf = float(preds[0][idx])

            if self.labels and idx < len(self.labels):
                return self.labels[idx], conf

            return f"class_{idx}", conf

        except Exception as e:
            print(f"Prediction failed: {e}")
            return "prediction_failed", 0.0