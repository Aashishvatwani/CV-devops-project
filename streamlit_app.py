import os
import threading
import time
import subprocess
from typing import Dict, Optional, Tuple
from collections import deque

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from ai.predictor import SketchPredictor
from ai.calculator import CalculatorEngine
from api.unsplash import fetch_image_url
from vision.drawing_canvas import DrawingCanvas
from vision.gesture_calibration import (
    compute_calibration,
    metrics_from_landmarks,
    save_calibration,
)
from vision.hand_tracking import HandTracker


CALIBRATION_PATH = "config/gesture_calibration.json"


def _create_kalman() -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    return kf


class DrawProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.tracker = HandTracker(
            calibration_path=CALIBRATION_PATH,
            min_detection_confidence=0.9,
            min_tracking_confidence=0.9,
            model_complexity=1,
        )
        self.predictor = SketchPredictor()
        self.ocr_predictor = SketchPredictor(model_path="model/letter_cnn.h5", labels_path="model/letter_labels.txt")
        self.calculator = CalculatorEngine()
        self.canvas: Optional[DrawingCanvas] = None
        self.last_prediction = ""
        self.last_confidence = 0.0
        self.last_image_url = ""
        self.last_shape_log = ""
        self.last_ocr_text = ""
        self.last_calc_expr = ""
        self.last_calc_result = ""
        self._smoothed_point: Optional[Tuple[int, int]] = None
        self._point_history = deque(maxlen=7)
        self._min_move_px = 1
        self._last_draw_time = 0.0
        self._draw_grace_s = 0.5
        self._draw_lock = False
        self._draw_lock_timeout = 1.5
        self._notepad_mode = False
        self._kf = _create_kalman()
        self._tool = "pen"
        self._color = (0, 255, 0)
        self._thickness = 5
        self._mode = "auto"
        self._use_gestures = True
        self._last_gesture_time = 0.0
        self._color_cycle = [(0, 255, 0), (0, 255, 255), (255, 0, 0), (0, 128, 255), (255, 0, 255), (255, 255, 255)]
        self._color_index = 0

        self._lock = threading.Lock()
        self._actions: Dict[str, bool] = {
            "clear": False,
            "predict": False,
            "undo": False,
            "redo": False,
            "ocr": False,
            "calc": False,
            "save": False,
        }

    def set_action(self, name: str) -> None:
        with self._lock:
            self._actions[name] = True

    def get_status(self) -> Tuple[str, float, str, str]:
        with self._lock:
            return (
                self.last_prediction,
                self.last_confidence,
                self.last_image_url,
                self.last_shape_log,
                self.last_ocr_text,
                self.last_calc_expr,
                self.last_calc_result,
            )

    def set_controls(
        self,
        tool: str,
        color: Tuple[int, int, int],
        thickness: int,
        mode: str,
        use_gestures: bool,
        draw_lock: bool,
        notepad_mode: bool,
    ) -> None:
        with self._lock:
            self._tool = tool
            self._color = color
            self._thickness = thickness
            self._mode = mode
            self._use_gestures = use_gestures
            self._draw_lock = draw_lock
            self._notepad_mode = notepad_mode

    def _consume_actions(self) -> Dict[str, bool]:
        with self._lock:
            actions = dict(self._actions)
            for key in self._actions:
                self._actions[key] = False
        return actions

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        if self.canvas is None:
            h, w = img.shape[:2]
            self.canvas = DrawingCanvas(w, h)
        self.canvas.set_tool(self._tool)
        self.canvas.set_color(self._color)
        self.canvas.set_thickness(self._thickness)

        hands = self.tracker.process(img)
        draw_mode = False
        erase_mode = False
        clear_mode = False
        cursor_point = None

        if hands:
            hand = hands[0]
            fingers = hand["fingers_up"]
            index_tip = hand["landmarks"][8]
            index_pip = hand["landmarks"][6]
            index_mcp = hand["landmarks"][5]
            middle_tip = hand["landmarks"][12]
            middle_pip = hand["landmarks"][10]
                        fist = not fingers["thumb"] and not fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]
                        index_rotated = abs(index_tip[0] - index_mcp[0]) > 0.08
            raw_x = int(index_tip[0] * self.canvas.width)
            raw_y = int(index_tip[1] * self.canvas.height)
            measurement = np.array([[np.float32(raw_x)], [np.float32(raw_y)]])
            self._kf.correct(measurement)
            pred = self._kf.predict()
            index_point = (int(pred[0]), int(pred[1]))
            cursor_point = index_point

            self._point_history.append(index_point)
            avg_x = int(sum(p[0] for p in self._point_history) / len(self._point_history))
            avg_y = int(sum(p[1] for p in self._point_history) / len(self._point_history))
            if self._smoothed_point is None:
                self._smoothed_point = (avg_x, avg_y)
            else:
                if abs(avg_x - self._smoothed_point[0]) < self._min_move_px and abs(avg_y - self._smoothed_point[1]) < self._min_move_px:
                    pass
                else:
                    alpha = 0.4
                    sx, sy = self._smoothed_point
                    nx = int(sx * (1 - alpha) + avg_x * alpha)
                    ny = int(sy * (1 - alpha) + avg_y * alpha)
                    self._smoothed_point = (nx, ny)

            index_dist = ((index_tip[0] - index_pip[0]) ** 2 + (index_tip[1] - index_pip[1]) ** 2) ** 0.5
            middle_dist = ((middle_tip[0] - middle_pip[0]) ** 2 + (middle_tip[1] - middle_pip[1]) ** 2) ** 0.5
            index_extended = index_dist > 0.04
            middle_extended = middle_dist > 0.04
            draw_mode = index_extended and not middle_extended and not fingers["ring"] and not fingers["pinky"]
            erase_mode = index_extended and middle_extended and not fingers["ring"] and not fingers["pinky"]
            clear_mode = all(fingers.values())

            if self._use_gestures:
                now = time.time()
                if now - self._last_gesture_time > 0.5:
                    thumb_tip = hand["landmarks"][4]
                    thumb_ip = hand["landmarks"][3]
                    thumb_up = thumb_tip[1] < thumb_ip[1] - 0.02
                    thumb_down = thumb_tip[1] > thumb_ip[1] + 0.02
                    if thumb_up and not fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
                        self.canvas.increase_thickness()
                        self._thickness = self.canvas.thickness
                        self._last_gesture_time = now
                    elif thumb_down and not fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
                        self.canvas.decrease_thickness()
                        self._thickness = self.canvas.thickness
                        self._last_gesture_time = now
                    elif fingers["index"] and fingers["middle"] and fingers["ring"] and not fingers["pinky"]:
                        self._color_index = (self._color_index + 1) % len(self._color_cycle)
                        self._color = self._color_cycle[self._color_index]
                        self._last_gesture_time = now

            if index_rotated:
                self.canvas.clear()
            elif clear_mode:
                self.canvas.clear()
            else:
                if fist:
                    self.canvas.reset_last_point()
                    self._last_draw_time = 0.0
                    return av.VideoFrame.from_ndarray(img, format="bgr24")
                if self._mode == "draw":
                    self.canvas.draw(self._smoothed_point)
                    self._last_draw_time = time.time()
                elif self._mode == "erase":
                    self.canvas.erase(self._smoothed_point)
                else:
                    if draw_mode:
                        self.canvas.draw(self._smoothed_point)
                        self._last_draw_time = time.time()
                    elif erase_mode:
                        self.canvas.erase(self._smoothed_point)
                    else:
                        grace = self._draw_lock_timeout if self._draw_lock else self._draw_grace_s
                        if time.time() - self._last_draw_time > grace:
                            self.canvas.reset_last_point()
        else:
            grace = self._draw_lock_timeout if self._draw_lock else self._draw_grace_s
            if time.time() - self._last_draw_time > grace:
                self.canvas.reset_last_point()
            self._smoothed_point = None
            self._point_history.clear()

        actions = self._consume_actions()
        if actions["clear"]:
            self.canvas.clear()
        if actions["undo"]:
            self.canvas.undo()
        if actions["redo"]:
            self.canvas.redo()
        if actions["save"]:
            os.makedirs("outputs", exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join("outputs", f"drawing_{ts}.png")
            cv2.imwrite(out_path, self.canvas.get_canvas())
        if actions["predict"]:
            self._update_shape_log()
            label, conf = self.predictor.predict(self.canvas.get_canvas())
            access_key = os.getenv("UNSPLASH_ACCESS_KEY", "")
            image_url = ""
            if access_key and label and label != "unknown":
                image_url = fetch_image_url(label, access_key) or ""
            with self._lock:
                self.last_prediction = label
                self.last_confidence = conf
                self.last_image_url = image_url
                self.last_ocr_text = label
                self.last_calc_expr = ""
                self.last_calc_result = ""
        if actions["ocr"]:
            label, conf = self._ocr_predict(self.canvas.get_canvas())
            with self._lock:
                self.last_ocr_text = label
                self.last_confidence = conf
        if actions["calc"]:
            expr = self.calculator.ocr_expression(self.canvas.get_canvas())
            result = self.calculator.evaluate(expr)
            with self._lock:
                self.last_calc_expr = expr
                self.last_calc_result = result

        overlay = self.canvas.get_canvas()
        mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        if self._notepad_mode:
            blank = np.full_like(img, 255)
            bg = cv2.bitwise_and(blank, blank, mask=mask_inv)
        else:
            bg = cv2.bitwise_and(img, img, mask=mask_inv)
        fg = cv2.bitwise_and(overlay, overlay, mask=mask)
        combined = cv2.add(bg, fg)

        status = "DRAW" if draw_mode else "ERASE" if erase_mode else "IDLE"
        cv2.putText(combined, f"Mode: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if cursor_point:
            cv2.circle(combined, cursor_point, 6, (0, 0, 255), -1)

        return av.VideoFrame.from_ndarray(combined, format="bgr24")

    def _update_shape_log(self) -> None:
        if self.canvas is None:
            return
        gray = cv2.cvtColor(self.canvas.get_canvas(), cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            log = "Shape: empty"
        else:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            x, y, w, h = cv2.boundingRect(largest)
            log = f"Shape: area={area:.0f}, bbox=({w}x{h})"
        with self._lock:
            self.last_shape_log = log
        print(log)

    def _ocr_predict(self, canvas_bgr) -> Tuple[str, float]:
        label, conf = self.ocr_predictor.predict(canvas_bgr)
        if label == "unknown":
            return self.predictor.predict(canvas_bgr)
        return label, conf


class CalibrationProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.tracker = HandTracker(calibration_path=None)
        self._lock = threading.Lock()
        self.last_sample: Optional[Tuple[str, list]] = None

    def get_last_sample(self) -> Optional[Tuple[str, list]]:
        with self._lock:
            return self.last_sample

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        hands = self.tracker.process(img)

        if hands:
            hand = hands[0]
            with self._lock:
                self.last_sample = (hand["handedness"], hand["landmarks"])
        else:
            with self._lock:
                self.last_sample = None

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.set_page_config(page_title="Air Sketch AI", layout="wide")

st.title("Air Sketch AI")

page = st.sidebar.radio("Mode", ["Draw", "Calibration", "About"], index=0)

if page == "Draw":
    st.subheader("Live Air Drawing")
    tool = st.sidebar.selectbox("Tool", ["pen", "pencil", "marker"], index=0)
    mode = st.sidebar.selectbox("Mode override", ["auto", "draw", "erase"], index=0)
    thickness = st.sidebar.slider("Brush thickness", 1, 30, 5)
    color_hex = st.sidebar.color_picker("Color", "#00ff00")
    use_gestures = st.sidebar.checkbox("Enable gesture controls", value=True)
    draw_lock = st.sidebar.checkbox("Draw lock (keep stroke) ", value=True)
    notepad_mode = st.sidebar.checkbox("Notepad mode (white background)", value=False)

    if st.sidebar.button("Open On-Screen Keyboard"):
        try:
            subprocess.Popen(["cmd", "/c", "start", "osk"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            st.sidebar.write("Could not open on-screen keyboard.")

    webrtc_ctx = webrtc_streamer(
        key="draw",
        video_processor_factory=DrawProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
        desired_playing_state=True,
    )

    col1, col2, col3 = st.columns(3)
    if webrtc_ctx.video_processor:
        rgb = color_hex.lstrip("#")
        color = (int(rgb[4:6], 16), int(rgb[2:4], 16), int(rgb[0:2], 16))
        webrtc_ctx.video_processor.set_controls(tool, color, thickness, mode, use_gestures, draw_lock, notepad_mode)

        if col1.button("Clear"):
            webrtc_ctx.video_processor.set_action("clear")
        if col2.button("Undo"):
            webrtc_ctx.video_processor.set_action("undo")
        if col3.button("Redo"):
            webrtc_ctx.video_processor.set_action("redo")

        col4, col5, col6, col7 = st.columns(4)
        if col4.button("Save Canvas"):
            webrtc_ctx.video_processor.set_action("save")
        if col5.button("Predict"):
            webrtc_ctx.video_processor.set_action("predict")
        if col6.button("OCR"):
            webrtc_ctx.video_processor.set_action("ocr")
        if col7.button("Calc"):
            webrtc_ctx.video_processor.set_action("calc")

        pred, conf, img_url, shape_log, ocr_text, calc_expr, calc_result = webrtc_ctx.video_processor.get_status()
        st.write(f"Prediction: {pred or 'n/a'}")
        st.write(f"Confidence: {conf:.2f}")
        st.write(shape_log or "Shape: n/a")
        st.write(f"OCR: {ocr_text or 'n/a'}")
        if calc_expr:
            st.write(f"Calc expr: {calc_expr}")
        if calc_result:
            st.write(f"Calc result: {calc_result}")
        if img_url:
            st.image(img_url, caption=pred)

elif page == "Calibration":
    st.subheader("Gesture Calibration")
    st.write("Capture a few samples for open palm and fist to calibrate finger detection.")

    if "open_samples" not in st.session_state:
        st.session_state.open_samples = []
    if "fist_samples" not in st.session_state:
        st.session_state.fist_samples = []
    if "last_handedness" not in st.session_state:
        st.session_state.last_handedness = "Right"

    webrtc_ctx = webrtc_streamer(
        key="calib",
        video_processor_factory=CalibrationProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
        desired_playing_state=True,
    )

    col1, col2, col3 = st.columns(3)

    if webrtc_ctx.video_processor:
        sample = webrtc_ctx.video_processor.get_last_sample()
        if sample:
            handedness, landmarks = sample
            st.session_state.last_handedness = handedness
            metrics = metrics_from_landmarks(landmarks, handedness)
        else:
            metrics = None

        if col1.button("Capture Open Palm") and metrics:
            st.session_state.open_samples.append(metrics)
        if col2.button("Capture Fist") and metrics:
            st.session_state.fist_samples.append(metrics)
        if col3.button("Reset Samples"):
            st.session_state.open_samples = []
            st.session_state.fist_samples = []

    st.write(f"Open palm samples: {len(st.session_state.open_samples)}")
    st.write(f"Fist samples: {len(st.session_state.fist_samples)}")

    if len(st.session_state.open_samples) >= 5 and len(st.session_state.fist_samples) >= 5:
        if st.button("Save Calibration"):
            calibration = compute_calibration(
                st.session_state.open_samples,
                st.session_state.fist_samples,
                st.session_state.last_handedness,
            )
            save_calibration(CALIBRATION_PATH, calibration)
            st.success("Calibration saved to config/gesture_calibration.json")

else:
    st.subheader("About")
    st.write(
        "This UI wraps the Air Sketch AI MVP. Use the Draw tab to sketch in the air, and the Calibration tab to "
        "improve gesture detection."
    )
