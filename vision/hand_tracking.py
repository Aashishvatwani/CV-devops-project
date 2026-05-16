from typing import Dict, List, Optional

import cv2
import mediapipe as mp

from vision.gesture_calibration import GestureCalibration, load_calibration, metrics_from_landmarks


class HandTracker:
    def __init__(
        self,
        calibration_path: Optional[str] = "config/gesture_calibration.json",
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        model_complexity: int = 1,
    ) -> None:
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.calibration: Optional[GestureCalibration] = None
        if calibration_path:
            self.calibration = load_calibration(calibration_path)

    def process(self, frame_bgr) -> List[Dict]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        output = []

        if not results.multi_hand_landmarks:
            return output

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = "Right"
            if results.multi_handedness and idx < len(results.multi_handedness):
                handedness = results.multi_handedness[idx].classification[0].label
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            fingers_up = self._fingers_up(landmarks, handedness)
            output.append({"landmarks": landmarks, "fingers_up": fingers_up, "handedness": handedness})

        return output

    def _fingers_up(self, landmarks, handedness: str) -> Dict[str, bool]:
        # Finger tips: 4, 8, 12, 16, 20
        # Finger PIP joints: 3, 6, 10, 14, 18
        fingers = {}

        if self.calibration:
            metrics = metrics_from_landmarks(landmarks, handedness)
            for finger, threshold in self.calibration.thresholds.items():
                fingers[finger] = metrics[finger] > threshold
            return fingers

        # Thumb uses x-axis comparison because it bends sideways
        if handedness == "Right":
            fingers["thumb"] = landmarks[4][0] > landmarks[3][0]
        else:
            fingers["thumb"] = landmarks[4][0] < landmarks[3][0]

        fingers["index"] = landmarks[8][1] < landmarks[6][1]
        fingers["middle"] = landmarks[12][1] < landmarks[10][1]
        fingers["ring"] = landmarks[16][1] < landmarks[14][1]
        fingers["pinky"] = landmarks[20][1] < landmarks[18][1]

        return fingers
