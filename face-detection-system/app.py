"""Command line front end for the face detection system.

Two modes:

    python app.py --image input/photo.jpg      detect faces in one picture
    python app.py --webcam                      detect faces in the live camera

This file only handles the user interface (printing, windows, key presses)
and camera access. All detection logic lives in ``src/detector.py``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import cv2

# Allow "python app.py" as well as "python -m app" from the project folder.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import (  # noqa: E402  (import after sys.path fix)
    DEFAULT_CASCADE_PATH,
    CascadeNotFoundError,
    DetectionParams,
    FaceDetectionError,
    FaceDetector,
    InvalidImageError,
)
from src.utils import (  # noqa: E402
    build_output_path,
    draw_banner,
    draw_faces,
    ensure_dir,
    load_image,
    parse_size,
    save_image,
)

WINDOW_NAME = "Face Detection System"

# Exit codes, so scripts and tests can tell what happened.
EXIT_OK = 0
EXIT_USAGE = 1
EXIT_BAD_INPUT = 2
EXIT_NO_CAMERA = 3


class CameraError(Exception):
    """The webcam could not be opened."""


# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------
def open_camera(index: int) -> cv2.VideoCapture:
    """Open a camera and return it, or raise :class:`CameraError`."""
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        capture.release()
        raise CameraError(
            "Could not open camera %d.\n"
            "Check that it is connected, not used by another program, and that "
            "a camera exists on this machine (a virtual camera can be created "
            "with '--webcam-source' in tools like OBS or Virtual Camera)." % index
        )
    return capture


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------
def detect_and_annotate(detector: FaceDetector, image, params: DetectionParams):
    """Detect + draw + label. Returns ``(annotated_image, face_count)``."""
    faces = detector.detect(image, params)
    labels = ["Face %d" % (index + 1) for index in range(len(faces))]
    annotated = draw_faces(image, faces, labels=labels)
    annotated = draw_banner(annotated, "Faces detected: %d" % len(faces))
    return annotated, len(faces)


def run_image_mode(detector: FaceDetector, args: argparse.Namespace) -> int:
    """Detect faces in a single image file, save the result, show it."""
    params = build_params(args)
    print("[info] loading image: %s" % args.image)
    image = load_image(args.image)

    annotated, count = detect_and_annotate(detector, image, params)
    print("[info] image size   : %dx%d" % (image.shape[1], image.shape[0]))
    print("[info] parameters   : %s" % params.as_dict())
    print("[result] faces found : %d" % count)

    output_dir = ensure_dir(Path(args.output_dir))
    destination = build_output_path(Path(args.image), output_dir)
    saved_path = save_image(annotated, destination)
    print("[saved] %s" % saved_path)

    if not args.no_show:
        show_image(annotated)
    return EXIT_OK


def run_webcam_mode(detector: FaceDetector, args: argparse.Namespace) -> int:
    """Detect faces in the live camera feed until the user presses Q."""
    params = build_params(args)
    capture = open_camera(args.camera_index)
    output_dir = ensure_dir(Path(args.output_dir))

    if args.camera_width and args.camera_height:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    print("[info] camera %d opened - press 'q' or ESC to quit, 's' to save a frame."
          % args.camera_index)

    frame = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("[error] could not read a frame from the camera.")
                break

            annotated, count = detect_and_annotate(detector, frame, params)
            if not args.no_show and not handle_window(annotated, output_dir):
                break
    except KeyboardInterrupt:
        print("\n[info] stopped by the user.")
    finally:
        capture.release()
        close_windows()

    print("[info] camera closed.")
    return EXIT_OK


# --------------------------------------------------------------------------
# Window helpers (all OpenCV GUI code lives here)
# --------------------------------------------------------------------------
_window_broken = False


def show_image(image, window: str = WINDOW_NAME) -> bool:
    """Show an image until a key is pressed. Returns False if impossible."""
    global _window_broken
    if _window_broken:
        return False
    try:
        cv2.imshow(window, image)
        cv2.waitKey(0)
    except cv2.error as exc:
        _window_broken = True
        print("[info] no window could be opened (%s)." % str(exc).splitlines()[0])
        print("[info] continuing without a preview; the file was still saved.")
    finally:
        close_windows()
    return not _window_broken


def handle_window(image, output_dir: Path) -> bool:
    """Show one camera frame. Returns False when the user wants to quit."""
    global _window_broken
    if _window_broken:
        return True  # no preview available: keep processing frames
    try:
        cv2.imshow(WINDOW_NAME, image)
        key = cv2.waitKey(1) & 0xFF
    except cv2.error as exc:
        _window_broken = True
        print("[info] no window could be opened (%s)." % str(exc).splitlines()[0])
        print("[info] continuing without a preview - press Ctrl+C to stop.")
        return True
    return key_action(key, image, output_dir)


def key_action(key: int, image, output_dir: Path) -> bool:
    """React to a key press. Returns False when the loop should stop."""
    if key in (ord("q"), 27):  # 'q' or ESC
        return False
    if key == ord("s"):
        name = "webcam_%s.png" % datetime.now().strftime("%Y%m%d_%H%M%S")
        saved = save_image(image, Path(output_dir) / name)
        print("[saved] %s" % saved)
    return True


def close_windows() -> None:
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------
def build_params(args: argparse.Namespace) -> DetectionParams:
    """Turn parsed command line values into :class:`DetectionParams`."""
    return DetectionParams(
        scale_factor=args.scale_factor,
        min_neighbors=args.min_neighbors,
        min_size=args.min_size,
        max_size=args.max_size,
    ).validate()


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive whole number")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="Detect human faces in an image or in the live webcam feed "
                    "using an OpenCV Haar Cascade classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="path of the image to process")
    source.add_argument("--webcam", action="store_true",
                        help="use the live camera instead of a file")

    parser.add_argument("--camera-index", type=non_negative_int, default=0,
                        help="which camera to open (0 is the usual built-in one)")
    parser.add_argument("--camera-width", type=positive_int,
                        help="requested camera width in pixels")
    parser.add_argument("--camera-height", type=positive_int,
                        help="requested camera height in pixels")

    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output",
                        help="folder for the processed image")
    parser.add_argument("--cascade", type=Path, default=DEFAULT_CASCADE_PATH,
                        help="Haar cascade XML file")

    parser.add_argument("--scale-factor", type=float, default=1.1,
                        help="image shrinking step, must be > 1.0 (smaller = slower, "
                             "more accurate)")
    parser.add_argument("--min-neighbors", type=int, default=5,
                        help="how many neighbours must agree (larger = stricter)")
    parser.add_argument("--min-size", type=parse_size, default="60x60",
                        help="smallest face to look for, in pixels, e.g. 60x60")
    parser.add_argument("--max-size", type=parse_size,
                        help="largest face to look for, in pixels, e.g. 300x300")

    parser.add_argument("--no-show", action="store_true",
                        help="do not open a window (useful on a server)")
    return parser


def non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be 0 or a positive whole number")
    return number


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        detector = FaceDetector(cascade_path=args.cascade, params=build_params(args))
    except CascadeNotFoundError as exc:
        print("[error] %s" % exc)
        return EXIT_BAD_INPUT
    except FaceDetectionError as exc:
        print("[error] invalid parameters: %s" % exc)
        return EXIT_USAGE

    try:
        if args.image is not None:
            return run_image_mode(detector, args)
        return run_webcam_mode(detector, args)
    except InvalidImageError as exc:
        print("[error] %s" % exc)
        return EXIT_BAD_INPUT
    except CameraError as exc:
        print("[error] %s" % exc)
        return EXIT_NO_CAMERA
    except FaceDetectionError as exc:
        print("[error] %s" % exc)
        return EXIT_BAD_INPUT


if __name__ == "__main__":
    sys.exit(main())