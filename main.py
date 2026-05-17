import os
import time
from collections import deque
import cv2
import numpy as np

from vision.hand_tracking import HandTracker
from vision.drawing_canvas import DrawingCanvas
from ai.predictor import SketchPredictor
from api.unsplash import fetch_image_url
from ai.calculator import CalculatorEngine


def _create_kalman() -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    return kf


def _shape_log(canvas_bgr: np.ndarray) -> str:
    gray = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "Shape: empty"
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    x, y, w, h = cv2.boundingRect(largest)
    return f"Shape: area={area:.0f}, bbox=({w}x{h})"


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    tracker = HandTracker(
        calibration_path="config/gesture_calibration.json",
        min_detection_confidence=0.9,
        min_tracking_confidence=0.9,
        model_complexity=1,
    )
    predictor = SketchPredictor()
    ocr_predictor = SketchPredictor(model_path="model/letter_cnn.h5", labels_path="model/letter_labels.txt")
    calculator = CalculatorEngine()

    canvas = None
    last_prediction = ""
    last_confidence = 0.0
    last_ocr = ""
    last_calc = ""
    last_calc_result = ""
    last_shape_log = ""
    tool = "pen"
    color = (0, 255, 0)
    thickness = 5
    mode_override = "auto"
    point_history = deque(maxlen=7)
    smoothed_point = None
    last_gesture_time = 0.0
    last_draw_time = 0.0
    draw_grace_s = 1.0
    draw_lock = True
    draw_lock_timeout = 5.0
    notepad_mode = False
    kf = _create_kalman()
    color_cycle = [(0, 255, 0), (0, 255, 255), (255, 0, 0), (0, 128, 255), (255, 0, 255), (255, 255, 255)]
    color_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)

        if canvas is None:
            h, w = frame.shape[:2]
            canvas = DrawingCanvas(w, h)

        canvas.set_tool(tool)
        canvas.set_color(color)
        canvas.set_thickness(thickness)

        hands = tracker.process(frame)
        draw_mode = False
        erase_mode = False
        clear_mode = False

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
            raw_x = int(index_tip[0] * canvas.width)
            raw_y = int(index_tip[1] * canvas.height)
            measurement = np.array([[np.float32(raw_x)], [np.float32(raw_y)]])
            kf.correct(measurement)
            pred = kf.predict()
            index_point = (int(pred[0, 0]), int(pred[1, 0]))
            point_history.append(index_point)
            avg_x = int(sum(p[0] for p in point_history) / len(point_history))
            avg_y = int(sum(p[1] for p in point_history) / len(point_history))
            if smoothed_point is None:
                smoothed_point = (avg_x, avg_y)
            else:
                alpha = 0.4
                sx, sy = smoothed_point
                nx = int(sx * (1 - alpha) + avg_x * alpha)
                ny = int(sy * (1 - alpha) + avg_y * alpha)
                smoothed_point = (nx, ny)

            index_dist = ((index_tip[0] - index_pip[0]) ** 2 + (index_tip[1] - index_pip[1]) ** 2) ** 0.5
            middle_dist = ((middle_tip[0] - middle_pip[0]) ** 2 + (middle_tip[1] - middle_pip[1]) ** 2) ** 0.5
            index_extended = index_dist > 0.04
            middle_extended = middle_dist > 0.04
            draw_mode = index_extended and not middle_extended and not fingers["ring"] and not fingers["pinky"]
            erase_mode = index_extended and middle_extended and not fingers["ring"] and not fingers["pinky"]
            clear_mode = all(fingers.values())

            now = time.time()
            if now - last_gesture_time > 0.5:
                thumb_tip = hand["landmarks"][4]
                thumb_ip = hand["landmarks"][3]
                thumb_up = thumb_tip[1] < thumb_ip[1] - 0.02
                thumb_down = thumb_tip[1] > thumb_ip[1] + 0.02
                if thumb_up and not fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
                    canvas.increase_thickness()
                    thickness = canvas.thickness
                    last_gesture_time = now
                elif thumb_down and not fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
                    canvas.decrease_thickness()
                    thickness = canvas.thickness
                    last_gesture_time = now
                elif fingers["index"] and fingers["middle"] and fingers["ring"] and not fingers["pinky"]:
                    color_index = (color_index + 1) % len(color_cycle)
                    color = color_cycle[color_index]
                    last_gesture_time = now

            if index_rotated:
                canvas.clear()
            elif clear_mode:
                canvas.clear()
            else:
                if fist:
                    canvas.reset_last_point()
                    last_draw_time = 0.0
                    continue
                if mode_override == "draw":
                    canvas.draw(smoothed_point)
                    last_draw_time = time.time()
                elif mode_override == "erase":
                    canvas.erase(smoothed_point)
                else:
                    if draw_mode:
                        canvas.draw(smoothed_point)
                        last_draw_time = time.time()
                    elif erase_mode:
                        canvas.erase(smoothed_point)
                    else:
                        grace = draw_lock_timeout if draw_lock else draw_grace_s
                        if time.time() - last_draw_time > grace:
                            canvas.reset_last_point()
        else:
            grace = draw_lock_timeout if draw_lock else draw_grace_s
            if time.time() - last_draw_time > grace:
                canvas.reset_last_point()
            smoothed_point = None
            point_history.clear()

        overlay = canvas.get_canvas()
        mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        if notepad_mode:
            blank = np.full_like(frame, 255)
            bg = cv2.bitwise_and(blank, blank, mask=mask_inv)
        else:
            bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        fg = cv2.bitwise_and(overlay, overlay, mask=mask)
        combined = cv2.add(bg, fg)

        status = "DRAW" if draw_mode else "ERASE" if erase_mode else "IDLE"
        cv2.putText(combined, f"Mode: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if last_prediction:
            cv2.putText(
                combined,
                f"Pred: {last_prediction} ({last_confidence:.2f})",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
        if last_ocr:
            cv2.putText(combined, f"OCR: {last_ocr}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if last_calc:
            cv2.putText(
                combined,
                f"Calc: {last_calc} = {last_calc_result}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
        if last_shape_log:
            cv2.putText(
                combined,
                last_shape_log,
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                2,
            )

        cv2.imshow("Air Sketch AI MVP", combined)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("c"):
            canvas.clear()
        if key == ord("p"):
            last_shape_log = _shape_log(canvas.get_canvas())
            label, conf = predictor.predict(canvas.get_canvas())
            last_prediction = label
            last_confidence = conf
            print(f"Prediction: {label} ({conf:.2f})")
            print(last_shape_log)

            access_key = os.getenv("UNSPLASH_ACCESS_KEY", "")
            if not access_key:
                print("Unsplash key missing. Set UNSPLASH_ACCESS_KEY.")
            elif label in ("model_missing", "model_load_failed"):
                print("Prediction model not available. Train or place model/quickdraw_cnn.h5.")
            elif label and label != "unknown":
                url = fetch_image_url(label, access_key)
                if url:
                    print(f"Unsplash image: {url}")
                else:
                    print("Unsplash image not found.")
        if key == ord("o"):
            last_shape_log = _shape_log(canvas.get_canvas())
            label, conf = ocr_predictor.predict(canvas.get_canvas())
            if label == "unknown":
                label, conf = predictor.predict(canvas.get_canvas())
            last_ocr = label
            last_confidence = conf
            print(f"OCR: {label} ({conf:.2f})")
            print(last_shape_log)
        if key == ord("s"):
            os.makedirs("outputs", exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join("outputs", f"drawing_{ts}.png")
            cv2.imwrite(out_path, canvas.get_canvas())
            print(f"Saved canvas to {out_path}")
        if key == ord("m"):
            expr = calculator.ocr_expression(canvas.get_canvas())
            result = calculator.evaluate(expr)
            last_calc = expr
            last_calc_result = result
            print(f"Calc: {expr} = {result}")
        if key == ord("l"):
            draw_lock = not draw_lock
            print(f"Draw lock: {draw_lock}")
        if key == ord("n"):
            notepad_mode = not notepad_mode
            print(f"Notepad mode: {notepad_mode}")
        if key == ord("u"):
            canvas.undo()
        if key == ord("z"):
            canvas.redo()
        if key == ord("1"):
            tool = "pen"
        if key == ord("2"):
            tool = "pencil"
        if key == ord("3"):
            tool = "marker"
        if key == ord("e"):
            mode_override = "erase"
        if key == ord("d"):
            mode_override = "draw"
        if key == ord("a"):
            mode_override = "auto"
        if key == ord("]"):
            canvas.increase_thickness()
            thickness = canvas.thickness
        if key == ord("["):
            canvas.decrease_thickness()
            thickness = canvas.thickness
        if key == ord("r"):
            color = (0, 0, 255)
        if key == ord("g"):
            color = (0, 255, 0)
        if key == ord("b"):
            color = (255, 0, 0)
        if key == ord("k"):
            color = (0, 255, 255)
        if key == ord("w"):
            color = (255, 255, 255)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
