"""Face detection logic (OpenCV Haar Cascade).

This module is the "brain" of the project. It has no printing, no window
handling and no command line code, which makes it easy to unit test.

A face is always represented as a box: ``(x, y, width, height)`` - the
same format that ``cv2.CascadeClassifier.detectMultiScale`` returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Type alias for one detected face box.
FaceBox = Tuple[int, int, int, int]

# The classifier XML lives inside the project (not in a personal folder),
# so the path is built from this file's location.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASCADE_PATH = PROJECT_ROOT / "models" / "haarcascade_frontalface_default.xml"


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
class FaceDetectionError(Exception):
    """Base class for all errors raised by this project."""


class CascadeNotFoundError(FaceDetectionError):
    """The Haar cascade XML file is missing or cannot be read."""


class InvalidImageError(FaceDetectionError):
    """The image is missing, empty or not an image at all."""


class InvalidParameterError(FaceDetectionError, ValueError):
    """A detection parameter is outside its allowed range."""


# --------------------------------------------------------------------------
# Detection parameters
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DetectionParams:
    """User tunable knobs of ``detectMultiScale``.

    scale_factor : how much the image is shrunk between passes (> 1.0).
                   Smaller  = more accurate but slower.
    min_neighbors : how many neighbouring candidates must agree (>= 0).
                   Larger   = fewer false positives.
    min_size     : smallest face, in pixels, that is allowed.
    max_size     : largest face, in pixels, that is allowed (None = no limit).
    """

    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_size: int = 60
    max_size: Optional[int] = None

    def validate(self) -> "DetectionParams":
        """Raise :class:`InvalidParameterError` if a value makes no sense."""
        if self.scale_factor <= 1.0:
            raise InvalidParameterError(
                "scale_factor must be greater than 1.0 (got %r)" % self.scale_factor
            )
        if self.min_neighbors < 0:
            raise InvalidParameterError(
                "min_neighbors cannot be negative (got %r)" % self.min_neighbors
            )
        if self.min_size <= 0:
            raise InvalidParameterError(
                "min_size must be a positive number of pixels (got %r)" % self.min_size
            )
        if self.max_size is not None and self.max_size < self.min_size:
            raise InvalidParameterError(
                "max_size (%r) must be >= min_size (%r)" % (self.max_size, self.min_size)
            )
        return self

    def as_dict(self) -> Dict[str, object]:
        """Plain dictionary, handy for printing or saving to JSON."""
        return {
            "scale_factor": self.scale_factor,
            "min_neighbors": self.min_neighbors,
            "min_size": self.min_size,
            "max_size": self.max_size,
        }


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert any OpenCV image to single channel grayscale."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def validate_image(image: Optional[np.ndarray]) -> np.ndarray:
    """Check that ``image`` really is a usable picture and return it."""
    if image is None:
        raise InvalidImageError("No image was given (received None).")
    if not isinstance(image, np.ndarray):
        raise InvalidImageError(
            "Expected an OpenCV image (numpy array), got %s." % type(image).__name__
        )
    if image.size == 0:
        raise InvalidImageError("The image is empty.")
    if image.ndim not in (2, 3):
        raise InvalidImageError(
            "Unsupported image shape %s (expected 2 or 3 dimensions)." % (image.shape,)
        )
    return image


def load_cascade(cascade_path: Optional[Path] = None) -> cv2.CascadeClassifier:
    """Load the Haar cascade classifier from an XML file.

    Raises :class:`CascadeNotFoundError` when the file is missing or when
    OpenCV cannot make sense of it.
    """
    path = Path(cascade_path) if cascade_path is not None else DEFAULT_CASCADE_PATH

    if not path.is_file():
        raise CascadeNotFoundError(
            "Haar cascade file not found: %s\n"
            "Expected it inside the project 'models' folder." % path
        )

    try:
        classifier = cv2.CascadeClassifier(str(path))
    except (cv2.error, SystemError) as exc:
        # Some OpenCV builds raise here for a damaged file instead of
        # returning an empty classifier.
        raise CascadeNotFoundError(
            "OpenCV could not read the Haar cascade file: %s (%s)" % (path, exc)
        ) from exc
    if classifier.empty():
        raise CascadeNotFoundError(
            "OpenCV could not read the Haar cascade file: %s" % path
        )
    return classifier


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
def detect_faces_in_gray(
    gray_image: np.ndarray,
    classifier: cv2.CascadeClassifier,
    params: Optional[DetectionParams] = None,
) -> List[FaceBox]:
    """Run detection on an already grayscale image.

    Returns a list of ``(x, y, w, h)`` tuples - an empty list means
    "no faces found", which is a normal result and not an error.
    """
    settings = (params or DetectionParams()).validate()
    gray_image = validate_image(gray_image)

    # OpenCV needs the optional limits only when they were given.
    options: Dict[str, object] = {
        "scaleFactor": float(settings.scale_factor),
        "minNeighbors": int(settings.min_neighbors),
        "minSize": (int(settings.min_size), int(settings.min_size)),
    }
    if settings.max_size is not None:
        options["maxSize"] = (int(settings.max_size), int(settings.max_size))

    boxes = classifier.detectMultiScale(gray_image, **options)
    return [tuple(int(value) for value in box) for box in boxes]


def detect_faces(
    image: np.ndarray,
    classifier: Optional[cv2.CascadeClassifier] = None,
    params: Optional[DetectionParams] = None,
    cascade_path: Optional[Path] = None,
) -> List[FaceBox]:
    """Detect every face in ``image`` and return the boxes.

    ``classifier`` is reused when given; otherwise it is loaded once from
    ``cascade_path`` (or from the bundled models folder).
    """
    validate_image(image)
    if classifier is None:
        classifier = load_cascade(cascade_path)
    return detect_faces_in_gray(to_gray(image), classifier, params)


class FaceDetector:
    """Object oriented wrapper around the Haar cascade detector.

    The classifier is loaded in ``__init__`` so a missing model file is
    reported immediately instead of failing later in the middle of a
    camera loop.
    """

    def __init__(
        self,
        cascade_path: Optional[Path] = None,
        params: Optional[DetectionParams] = None,
    ) -> None:
        self.cascade_path = Path(cascade_path) if cascade_path else DEFAULT_CASCADE_PATH
        self.params = (params or DetectionParams()).validate()
        self.classifier = load_cascade(self.cascade_path)

    def detect(
        self,
        image: np.ndarray,
        params: Optional[DetectionParams] = None,
    ) -> List[FaceBox]:
        """Return the list of ``(x, y, w, h)`` face boxes in ``image``."""
        return detect_faces(image, self.classifier, params or self.params)

    def count_faces(
        self,
        image: np.ndarray,
        params: Optional[DetectionParams] = None,
    ) -> int:
        """Return only the number of detected faces."""
        return len(self.detect(image, params))