"""File and drawing helpers.

Nothing in here decides *whether* a face is a face - that is
``detector.py``'s job. These helpers only move images in and out of the
project and put rectangles/text on top of them.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from .detector import FaceBox, InvalidImageError, InvalidParameterError

# BGR colour used for the face rectangles (green).
DEFAULT_BOX_COLOR: Tuple[int, int, int] = (0, 255, 0)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Folders and paths
# --------------------------------------------------------------------------
def ensure_dir(path: Path) -> Path:
    """Create the folder (and parents) if it does not exist yet."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_output_path(
    input_path: Path,
    output_dir: Path,
    suffix: str = "_detected",
    extension: Optional[str] = None,
) -> Path:
    """Work out where the processed image should be written.

    ``input/photo.jpg`` + ``output/`` + suffix ``_detected``
    becomes ``output/photo_detected.jpg``.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    stem = input_path.stem or "image"
    ext = extension if extension is not None else input_path.suffix
    if not ext:
        ext = ".png"
    return output_dir / (stem + suffix + ext)


def unique_path(path: Path) -> Path:
    """Return ``path`` or the first free ``name_1``, ``name_2``... variant."""
    path = Path(path)
    if not path.exists():
        return path
    for number in range(1, 1000):
        candidate = path.with_name("%s_%d%s" % (path.stem, number, path.suffix))
        if not candidate.exists():
            return candidate
    raise RuntimeError("Could not find a free file name for %s" % path)


# --------------------------------------------------------------------------
# Reading and writing images
# --------------------------------------------------------------------------
def imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """``cv2.imread`` that also works with non-ASCII paths on Windows."""
    path = Path(path)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:  # unreadable file, permission problem, ...
        raise InvalidImageError("Could not read file %s (%s)" % (path, exc)) from exc
    image = cv2.imdecode(data, flags)
    if image is None:
        raise InvalidImageError(
            "File is not a readable image: %s\n"
            "It may be empty, corrupted or in an unsupported format." % path
        )
    return image


def imwrite_unicode(image: np.ndarray, path: Path) -> bool:
    """``cv2.imwrite`` that also works with non-ASCII paths on Windows."""
    path = Path(path)
    extension = path.suffix or ".png"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        return False
    encoded.tofile(str(path))
    return True


def load_image(path: Path) -> np.ndarray:
    """Load an image from disk.

    Raises :class:`InvalidImageError` when the file is missing, is not a
    file, or does not contain an image OpenCV understands.
    """
    path = Path(path)
    if not path.exists():
        raise InvalidImageError("Image file does not exist: %s" % path)
    if not path.is_file():
        raise InvalidImageError("Not a file (but a folder?): %s" % path)
    return imread_unicode(path)


def save_image(image: np.ndarray, path: Path, overwrite: bool = True) -> Path:
    """Write ``image`` to ``path``, creating the folder when needed."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise InvalidImageError("Cannot save an empty image to %s" % path)

    path = Path(path)
    ensure_dir(path.parent)
    if not overwrite:
        path = unique_path(path)
    if not imwrite_unicode(image, path) or not path.is_file():
        raise InvalidImageError("Failed to write image to %s" % path)
    return path


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def draw_faces(
    image: np.ndarray,
    faces: Sequence[FaceBox],
    labels: Optional[Iterable[str]] = None,
    color: Tuple[int, int, int] = DEFAULT_BOX_COLOR,
    thickness: int = 2,
) -> np.ndarray:
    """Return a **copy** of ``image`` with one rectangle per face.

    The original image is never modified, so the caller keeps a clean
    picture for saving.
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise InvalidImageError("Cannot draw on an empty image.")

    canvas = image.copy()
    label_list = list(labels) if labels is not None else []

    for index, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
        if index < len(label_list):
            text_position = (x, max(0, y - 8))
            cv2.putText(
                canvas,
                str(label_list[index]),
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
    return canvas


def draw_banner(
    image: np.ndarray,
    text: str,
    color: Tuple[int, int, int] = DEFAULT_BOX_COLOR,
) -> np.ndarray:
    """Return a **copy** of ``image`` with ``text`` written on top."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise InvalidImageError("Cannot draw on an empty image.")

    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
    return canvas


# --------------------------------------------------------------------------
# Command line values
# --------------------------------------------------------------------------
_SIZE_PATTERN = re.compile(r"^\s*(\d+)\s*[xX*]\s*(\d+)\s*$")


def parse_size(value: str) -> int:
    """Turn ``"60x60"`` into ``60``.

    Haar cascades use a square window, so both numbers must be equal.
    """
    if isinstance(value, int):  # already a number
        if value <= 0:
            raise InvalidParameterError("Size must be positive, got %r" % value)
        return value

    match = _SIZE_PATTERN.match(str(value))
    if not match:
        raise InvalidParameterError(
            "Invalid size %r - use the form 60x60 (pixels)." % value
        )
    width, height = int(match.group(1)), int(match.group(2))
    if width != height:
        raise InvalidParameterError(
            "Size %r is not square - use for example 60x60." % value
        )
    if width == 0:
        raise InvalidParameterError("Size must be greater than zero.")
    return width