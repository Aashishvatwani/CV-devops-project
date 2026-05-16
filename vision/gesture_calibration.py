from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]
TIP_IDXS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP_IDXS = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}


@dataclass
class GestureCalibration:
    thresholds: Dict[str, float]
    handedness: str

    def to_dict(self) -> Dict[str, object]:
        return {"thresholds": self.thresholds, "handedness": self.handedness}

    @staticmethod
    def from_dict(data: Dict[str, object]) -> "GestureCalibration":
        thresholds = {k: float(v) for k, v in data.get("thresholds", {}).items()}
        handedness = str(data.get("handedness", "Right"))
        return GestureCalibration(thresholds=thresholds, handedness=handedness)


def metrics_from_landmarks(landmarks, handedness: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    # For fingers, use vertical distance: pip_y - tip_y (positive means finger up).
    for finger in ["index", "middle", "ring", "pinky"]:
        pip = landmarks[PIP_IDXS[finger]]
        tip = landmarks[TIP_IDXS[finger]]
        metrics[finger] = pip[1] - tip[1]

    # For thumb, use sideways distance along x based on handedness.
    tip = landmarks[TIP_IDXS["thumb"]]
    pip = landmarks[PIP_IDXS["thumb"]]
    if handedness == "Right":
        metrics["thumb"] = tip[0] - pip[0]
    else:
        metrics["thumb"] = pip[0] - tip[0]

    return metrics


def compute_calibration(
    open_samples: Iterable[Dict[str, float]],
    fist_samples: Iterable[Dict[str, float]],
    handedness: str,
) -> GestureCalibration:
    def average(samples: Iterable[Dict[str, float]]) -> Dict[str, float]:
        totals = {f: 0.0 for f in FINGER_ORDER}
        count = 0
        for sample in samples:
            count += 1
            for finger in FINGER_ORDER:
                totals[finger] += float(sample[finger])
        if count == 0:
            return totals
        return {finger: totals[finger] / count for finger in FINGER_ORDER}

    open_avg = average(open_samples)
    fist_avg = average(fist_samples)

    thresholds = {}
    for finger in FINGER_ORDER:
        thresholds[finger] = (open_avg[finger] + fist_avg[finger]) / 2.0

    return GestureCalibration(thresholds=thresholds, handedness=handedness)


def save_calibration(path: str, calibration: GestureCalibration) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(calibration.to_dict(), f, indent=2)


def load_calibration(path: str) -> Optional[GestureCalibration]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return GestureCalibration.from_dict(data)
    except Exception:
        return None
