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
        return self.model_path

    def _load_model(self) -> None:
        if self.model is not None:
            return

        model_path = self._resolve_model_path()

        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return

        try:
            from tensorflow.keras.models import load_model
        except Exception as e:
            print(f"TensorFlow import error: {e}")
            return

        try:
            self.model = load_model(model_path, compile=False)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Model loading failed: {e}")
            self.model = None
            return

        if os.path.exists(self.labels_path):
            try:
                with open(self.labels_path, "r", encoding="utf-8") as f:
                    self.labels = [
                        line.strip()
                        for line in f
                        if line.strip()
                    ]
            except Exception as e:
                print(f"Labels loading failed: {e}")

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