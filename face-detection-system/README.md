# Face Detection System

Detect human faces in **images** or in a **live webcam feed** using OpenCV's
Haar Cascade classifier.

* No deep learning, no face recognition / identity matching
* Runs on any normal CPU - no GPU needed
* Only two real dependencies: `opencv-python` and `numpy`

Typical uses: security systems, attendance/visitor counters, camera apps,
or as a building block for a larger CV project.

---

## 1. Project structure

```
face-detection-system/
├── models/
│   └── haarcascade_frontalface_default.xml   # the trained classifier (bundled)
├── src/
│   ├── __init__.py      # makes src a package
│   ├── detector.py      # ALL detection logic (no user interface)
│   └── utils.py         # load/save images, draw rectangles, parse sizes
├── tests/
│   ├── conftest.py      # makes the project importable in tests
│   └── test_detector.py # unit tests
├── input/               # put your photos here
├── output/              # processed images are saved here
├── screenshots/         # screenshots for documentation
├── app.py               # command line entry point (the user interface)
├── requirements.txt
├── README.md
└── .gitignore
```

## 2. Install

```bash
cd face-detection-system
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## 3. Run

### Detect faces in one image

Put a photo in `input/` (for example `input/photo.jpg`), then:

```bash
python app.py --image input/photo.jpg
```

* a green rectangle is drawn around every face
* the number of faces is printed **and** written on the image
* the result is saved to `output/photo_detected.jpg`
* the annotated image is shown in a window (press any key to close it)

### Detect faces in the live webcam

```bash
python app.py --webcam
```

| Key      | Action              |
| -------- | ------------------- |
| `s`      | save the frame to `output/` |
| `q` / ESC | quit              |

Use `--camera-index 1` if you have more than one camera.

### Tuning the detection

| Option           | Default | Meaning                                                        |
| ---------------- | ------- | -------------------------------------------------------------- |
| `--scale-factor` | `1.1`   | shrink step between passes; smaller = slower but more accurate  |
| `--min-neighbors`| `5`     | neighbours that must agree; larger = fewer false positives      |
| `--min-size`     | `60x60` | smallest face to look for, in pixels                            |
| `--max-size`     | none    | largest face to look for, in pixels                             |
| `--output-dir`   | `output`| where results are written                                       |
| `--cascade`      | `models/haarcascade_frontalface_default.xml` | different classifier XML |
| `--no-show`      | off     | never open a window (servers, CI, remote machines)             |

Examples:

```bash
# stricter, better for crowded scenes
python app.py --image input/crowd.jpg --min-neighbors 8 --min-size 40x40

# more accurate but slower
python app.py --image input/crowd.jpg --scale-factor 1.05

# small faces, no preview window
python app.py --image input/crowd.jpg --min-size 20x20 --no-show
```

`python app.py --help` prints everything.

## 4. Tests

```bash
pytest -v
```

The tests check the parts that can be checked without a camera or a photo:
the classifier loads, bad parameters and broken files are rejected with clear
messages, boxes stay inside the image, drawings do not modify the original
picture, files are written to `output/`, and the command line returns the
right exit codes.

Two tests are **skipped unless you add your own photo to `input/`** (they are
marked with `pytest.skip`), because a synthetic drawing is not a real face and
the cascade must not be "taught" fake results:

* `test_stricter_parameters_do_not_report_more_faces`
* `test_real_photo_detection`

Drop a real picture into `input/` and run `pytest -v` again to exercise them.

## 5. How the code is organised (and why)

```
app.py  ──uses──▶  src/detector.py  ──uses──▶  src/utils.py
 (UI: print,           (logic: load the           (helpers: paths,
  windows, camera,      classifier, run           load/save image,
  command line)         detectMultiScale)          draw rectangles)
```

* **`src/detector.py`** - knows *what a face is*. Contains `DetectionParams`
  (the `minNeighbors` / `scaleFactor` / `minSize` knobs),
  `FaceDetector.detect()`, and the project's error classes. It imports no UI
  code, so it can be tested and reused from any other program.
* **`src/utils.py`** - knows *how to move and mark up pictures*.
* **`app.py`** - the user interface: argument parsing, printing, `cv2.imshow`
  windows, camera loop. It never implements detection itself.

Quick use as a library:

```python
from src.detector import FaceDetector, DetectionParams

detector = FaceDetector(params=DetectionParams(min_neighbors=8, min_size=40))
faces = detector.detect(image)        # -> [(x, y, w, h), ...]
print(len(faces))
```

## 6. Error handling

The program never crashes with a raw traceback on user mistakes:

| Situation                        | Behaviour                                        |
| -------------------------------- | ------------------------------------------------ |
| image file missing               | `[error] Image file does not exist: ...`         |
| file is not a real image         | `[error] File is not a readable image: ...`      |
| `input/` given instead of a file | `[error] Not a file (but a folder?): ...`        |
| cascade XML missing / corrupt    | `[error] Haar cascade file not found: ...`       |
| `min_size 60x70` (not square)    | argparse usage error with the accepted format    |
| webcam not available / busy      | `[error] Could not open camera 0. ...`           |
| camera stops delivering frames   | `[error] could not read a frame from the camera` |

Exit codes: `0` success, `1` usage, `2` bad image, `3` no camera.

## 7. Notes and limits

* The classifier file is the one shipped with OpenCV
  (`cv2.data/haarcascades/haarcascade_frontalface_default.xml`); it is copied
  into `models/` so the project works offline and does not depend on the
  OpenCV installation path.
* It detects **frontal** faces only. Sideways faces, small faces in a big
  crowd, and heavy darkness/occlusion are the usual weak spots - lower
  `--min-size`, raise `--min-neighbors`, or improve the lighting.
* It answers "how many faces and where", never "who is it" - there is no
  recognition, embedding or identity matching anywhere in this project.
* All paths are relative to the project folder (`input/`, `output/`,
  `models/`); nothing is hard-coded to a personal directory.