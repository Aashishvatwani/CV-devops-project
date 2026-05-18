# Air Sketch AI MVP

A minimum viable product for a hand-gesture drawing system that recognizes a sketch and can fetch a real image from the internet. The app uses a webcam, tracks your hand in real time, lets you draw in the air, and optionally predicts what you drew.

## What this MVP does

- Hand tracking with MediaPipe.
- Air drawing on a virtual canvas with OpenCV.
- Gesture control for draw, erase, and clear.
- Save drawings to disk.
- Optional sketch prediction via a CNN model file if you provide one.
- Optional Unsplash image lookup for the predicted label.
- Streamlit UI with live webcam, prediction panel, and calibration screen.
- Tool controls (pen, pencil, marker), thickness, colors, undo/redo.
- OCR label logging and a simple calculator mode.

## How it works (simple flow)

1. Webcam frames go into MediaPipe Hands.
2. We detect finger states (up/down).
3. If the index finger is up, we draw.
4. If index + middle are up, we erase.
5. If all fingers are up, we clear the canvas.
6. On key press, we save the drawing or run prediction.
7. If prediction works and an Unsplash key exists, we fetch an image URL.

## Project structure

```
CV-project/
├── ai/
│   └── predictor.py
├── api/
│   └── unsplash.py
├── model/
│   ├── quickdraw_cnn.h5        (optional)
│   └── labels.txt              (optional)
├── outputs/                    (auto-created)
├── scripts/
│   └── train_quickdraw.py
├── vision/
│   ├── drawing_canvas.py
│   ├── gesture_calibration.py
│   └── hand_tracking.py
├── main.py
├── requirements.txt
├── streamlit_app.py
└── README.md
```

## Setup (Windows)

1. Create a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Optional: if you have a trained model, place it at:

   - `model/quickdraw_cnn.h5`
   - `model/labels.txt` (one label per line, in the same order as the model output)

4. Optional: set an Unsplash key for image search:

   ```powershell
   $env:UNSPLASH_ACCESS_KEY="YOUR_KEY_HERE"
   ```

## Run the MVP

```powershell
python main.py
```

## Run the Streamlit UI

```powershell
streamlit run streamlit_app.py
```

## Controls and gestures

Gestures (based on finger states):

- Index finger up: draw
- Index + middle up: erase
- All fingers up: clear canvas

Keyboard shortcuts:

- `q`: quit
- `c`: clear canvas
- `s`: save canvas to `outputs/` (no video)
- `p`: predict current sketch (if model exists)
- `o`: OCR label (convert sketch to text)
- `m`: calculator OCR + evaluate
- `u`: undo
- `z`: redo
- `1/2/3`: pen / pencil / marker
- `d/e/a`: draw / erase / auto mode
- `[ / ]`: decrease / increase thickness
- `r/g/b/k/w`: change color
- `l`: toggle draw lock
- `n`: toggle notepad mode (white background)

Streamlit controls:

- Clear: clear the canvas
- Save Canvas: saves only the drawing (no video) to `outputs/`
- Predict: run the model on the current sketch
- Shape log: shows basic shape info after prediction
- OCR: convert the sketch to a label
- Calc: attempt to recognize digits and evaluate a math expression
- Tool selector: pen / pencil / marker
- Mode override: auto / draw / erase
- Thickness + color picker
- Undo / Redo
- Draw lock toggle
- Notepad mode toggle

## What each file does

- `main.py` is the entry point. It reads camera frames, runs hand tracking, manages drawing, and handles prediction and API calls.
- `vision/hand_tracking.py` wraps MediaPipe Hands and turns landmarks into finger states.
- `vision/gesture_calibration.py` saves and loads calibration thresholds for finger detection.
- `vision/drawing_canvas.py` keeps the drawing buffer, draws lines, and erases parts of the canvas.
- `ai/predictor.py` loads an optional CNN model to classify the sketch.
- `api/unsplash.py` calls the Unsplash API to fetch a real photo for the predicted label.
- `streamlit_app.py` provides a UI with live video, controls, prediction, and calibration screens.
- `scripts/train_quickdraw.py` trains a CNN on the Quick Draw dataset and outputs a model + labels file.

## How to explain it to someone (simple story)

This project turns your index finger into a virtual pen using a webcam and hand tracking. As you move your finger, OpenCV draws on a hidden canvas. You can erase with two fingers or clear with an open palm. When you are done, you can press a key to save the drawing or run a prediction. If you provide a model trained on sketches, the app recognizes your drawing and can even fetch a real image from the internet using Unsplash.

## Notes and limitations

- This MVP runs in an OpenCV window, not a full web UI.
- Prediction works only if a model file is provided.
- Gesture detection is intentionally simple and can be improved with calibration.

## Train a real Quick Draw model

This creates a small CNN trained on the Google Quick Draw dataset and writes:

- `model/quickdraw_cnn.h5`
- `model/labels.txt`

Example (8 classes, 5000 samples each):

```powershell
python scripts\train_quickdraw.py --classes "apple,cat,car,tree,cell phone,sun,moon,earth" --samples-per-class 5000
```

Train 10 classes using the provided list:

```powershell
python scripts\train_quickdraw.py --labels-file labels_80.txt --samples-per-class 5000
```

If any label is not in the Quick Draw dataset, it will be skipped with a warning.

If you want more accuracy, increase `--samples-per-class` and `--epochs`.
The training script downloads Quick Draw data from Google storage, so an internet connection is required.

## Fix quantization_config model load errors

If you see `quantization_config` errors when loading the `.h5` model, run:

```powershell
python scripts\clean_quickdraw_model.py --model-in model\quickdraw_cnn.h5 --labels model\labels.txt --model-out model\quickdraw_cnn.keras --force-rebuild
```

Then use the `.keras` model for prediction.

## Train the calculator model (digits + operators)

```powershell
python scripts\train_quickdraw.py --preset digits --samples-per-class 8000
```

## Train letter OCR model (A-Z, EMNIST)

```powershell
python scripts\train_emnist_letters.py
```

The OCR button uses the letter model if present; otherwise it falls back to the general sketch model.

## Gesture calibration (Streamlit)

Open the Streamlit app, go to the Calibration tab, and capture samples for:

1. Open palm
2. Fist

The app saves thresholds to `config/gesture_calibration.json`, which the tracker uses to detect fingers more reliably.

## Next steps to improve it

1. Train a CNN on Google Quick Draw and plug it in.
2. Build a React or Streamlit UI around the webcam feed.
3. Add confidence scores and voice feedback.
4. Use a larger model like CLIP for more flexible recognition.
