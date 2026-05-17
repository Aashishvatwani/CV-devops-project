from typing import Tuple
import os

import cv2
import numpy as np


class SketchPredictor:
    def __init__(self, model_path: str = "model/quickdraw_cnn.h5", labels_path: str = "model/labels.txt") -> None:
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.labels = None
        self._resolved_model_path = None

    def _resolve_model_path(self) -> str:
        if self._resolved_model_path:
            return self._resolved_model_path

        if os.path.exists(self.model_path):
            self._resolved_model_path = self.model_path
            return self._resolved_model_path

        base, ext = os.path.splitext(self.model_path)
        if ext == ".h5" and os.path.exists(base + ".keras"):
            self._resolved_model_path = base + ".keras"
            return self._resolved_model_path
        if ext == ".keras" and os.path.exists(base + ".h5"):
            self._resolved_model_path = base + ".h5"
            return self._resolved_model_path

        self._resolved_model_path = self.model_path
        return self._resolved_model_path

    def _load_model(self) -> None:
        if self.model is not None:
            return
        model_path = self._resolve_model_path()
        if not os.path.exists(model_path):
            return

        try:
            from tensorflow.keras.models import load_model
        except Exception:
            return

        try:
            self.model = load_model(model_path, compile=False, safe_mode=False)
        except TypeError:
            self.model = load_model(model_path, compile=False)
        except Exception:
            return

        if os.path.exists(self.labels_path):
            with open(self.labels_path, "r", encoding="utf-8") as f:
                self.labels = [line.strip() for line in f if line.strip()]

    def predict(self, canvas_bgr: np.ndarray) -> Tuple[str, float]:
        model_path = self._resolve_model_path()
        if not os.path.exists(model_path):
            return "model_missing", 0.0

        self._load_model()
        if self.model is None:
            return "model_load_failed", 0.0

        if canvas_bgr.ndim == 3 and canvas_bgr.shape[:2] == (28, 28):
            normalized = canvas_bgr.astype("float32")
            if canvas_bgr.shape[-1] != 1:
                normalized = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY).reshape(28, 28, 1)
            input_tensor = normalized.reshape(1, 28, 28, 1)
        elif canvas_bgr.ndim == 2 and canvas_bgr.shape[:2] == (28, 28):
            normalized = canvas_bgr.astype("float32")
            input_tensor = normalized.reshape(1, 28, 28, 1)
        else:
            gray = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
            normalized = resized.astype("float32") / 255.0
            input_tensor = normalized.reshape(1, 28, 28, 1)

        preds = self.model.predict(input_tensor, verbose=0)
        idx = int(np.argmax(preds))
        conf = float(preds[0][idx])

        if self.labels and idx < len(self.labels):
            return self.labels[idx], conf
        return f"class_{idx}", conf
