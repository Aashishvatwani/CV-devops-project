from typing import List, Tuple
import ast

import cv2
import numpy as np

from ai.predictor import SketchPredictor


DIGIT_MAP = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "0": "0",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
}

OP_MAP = {
    "plus": "+",
    "minus": "-",
    "multiplication": "*",
    "divide": "/",
    "+": "+",
    "-": "-",
    "x": "*",
    "*": "*",
    "/": "/",
}


class CalculatorEngine:
    def __init__(self, model_path: str = "model/digit_cnn.h5", labels_path: str = "model/digit_labels.txt") -> None:
        self.predictor = SketchPredictor(model_path=model_path, labels_path=labels_path)

    def ocr_expression(self, canvas_bgr: np.ndarray) -> str:
        symbols = self._segment_symbols(canvas_bgr)
        if not symbols:
            return ""

        tokens: List[str] = []
        for symbol in symbols:
            label, _ = self.predictor.predict(symbol)
            label = label.lower().strip()
            if label in DIGIT_MAP:
                tokens.append(DIGIT_MAP[label])
            elif label in OP_MAP:
                tokens.append(OP_MAP[label])

        return "".join(tokens)

    def evaluate(self, expression: str) -> str:
        if not expression:
            return ""
        try:
            node = ast.parse(expression, mode="eval")
            if not self._is_safe(node):
                return ""
            result = eval(compile(node, "<expr>", "eval"), {"__builtins__": {}})
            return str(result)
        except Exception:
            return ""

    def _segment_symbols(self, canvas_bgr: np.ndarray) -> List[np.ndarray]:
        gray = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[Tuple[int, int, int, int]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 80:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))

        boxes.sort(key=lambda b: b[0])
        symbols = []
        for x, y, w, h in boxes:
            pad = 6
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(canvas_bgr.shape[1], x + w + pad)
            y1 = min(canvas_bgr.shape[0], y + h + pad)
            crop = gray[y0:y1, x0:x1]
            resized = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA)
            normalized = resized.astype("float32") / 255.0
            symbols.append(normalized.reshape(28, 28, 1))

        return symbols

    def _is_safe(self, node: ast.AST) -> bool:
        allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div)
        for child in ast.walk(node):
            if not isinstance(child, allowed):
                return False
        return True
