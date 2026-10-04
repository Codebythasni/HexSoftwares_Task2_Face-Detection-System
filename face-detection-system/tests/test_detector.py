"""Unit tests for the face detection system.

Run them with:

    pytest -v

The tests never download anything and never need a camera. Tests that
would need a real photo or a real camera are skipped when those are not
available, so the suite is honest about what it actually checked.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app  # noqa: E402
from src.detector import (  # noqa: E402
    DEFAULT_CASCADE_PATH,
    CascadeNotFoundError,
    DetectionParams,
    FaceDetector,
    InvalidImageError,
    InvalidParameterError,
    detect_faces,
    load_cascade,
    to_gray,
    validate_image,
)
from src.utils import (  # noqa: E402
    build_output_path,
    draw_banner,
    draw_faces,
    ensure_dir,
    load_image,
    parse_size,
    save_image,
    unique_path,
)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


# --------------------------------------------------------------------------
# Fixtures / sample data
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def classifier():
    """The bundled Haar cascade classifier."""
    return load_cascade()


@pytest.fixture(scope="module")
def detector(classifier):
    return FaceDetector()


@pytest.fixture
def photo_without_faces():
    """A plain grey picture - valid image, but no faces in it."""
    return np.full((240, 320, 3), 128, dtype=np.uint8)


@pytest.fixture
def photo_path(tmp_path, photo_without_faces):
    """A real image file on disk (JPEG, so it gets compressed a little)."""
    path = tmp_path / "sample.jpg"
    assert cv2.imwrite(str(path), photo_without_faces)
    return path


@pytest.fixture
def real_photos():
    """Every picture the user put in ``input/`` (may be an empty list)."""
    input_dir = PROJECT_ROOT / "input"
    if not input_dir.is_dir():
        return []
    return sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


# --------------------------------------------------------------------------
# Model file
# --------------------------------------------------------------------------
def test_bundled_cascade_file_exists():
    assert DEFAULT_CASCADE_PATH.is_file(), "models/ is missing the cascade XML"
    assert DEFAULT_CASCADE_PATH.parent.name == "models"
    assert DEFAULT_CASCADE_PATH.parent.parent == PROJECT_ROOT


def test_load_cascade_returns_usable_classifier(classifier):
    assert isinstance(classifier, cv2.CascadeClassifier)
    assert not classifier.empty()


def test_load_cascade_missing_file_raises(tmp_path):
    with pytest.raises(CascadeNotFoundError):
        load_cascade(tmp_path / "not_here.xml")


def test_load_cascade_broken_file_raises(tmp_path):
    broken = tmp_path / "broken.xml"
    broken.write_text("this is not XML", encoding="utf-8")
    with pytest.raises(CascadeNotFoundError):
        load_cascade(broken)


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
def test_default_parameters_are_valid():
    params = DetectionParams()
    assert params.validate() is params
    assert params.scale_factor == 1.1
    assert params.min_neighbors == 5
    assert params.min_size == 60
    assert params.max_size is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale_factor": 1.0},
        {"scale_factor": 0.5},
        {"min_neighbors": -1},
        {"min_size": 0},
        {"min_size": -20},
        {"min_size": 100, "max_size": 50},
    ],
)
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(InvalidParameterError):
        DetectionParams(**kwargs).validate()


def test_as_dict_contains_all_parameters():
    keys = DetectionParams(min_size=30).as_dict().keys()
    assert keys == {"scale_factor", "min_neighbors", "min_size", "max_size"}


# --------------------------------------------------------------------------
# Image validation
# --------------------------------------------------------------------------
def test_validate_image_accepts_color_and_gray_images(photo_without_faces):
    assert validate_image(photo_without_faces) is photo_without_faces
    gray = to_gray(photo_without_faces)
    assert validate_image(gray) is gray


def test_validate_image_rejects_none():
    with pytest.raises(InvalidImageError):
        validate_image(None)


def test_validate_image_rejects_non_image_object():
    with pytest.raises(InvalidImageError):
        validate_image("input/photo.jpg")


def test_validate_image_rejects_empty_array():
    with pytest.raises(InvalidImageError):
        validate_image(np.zeros((0, 0, 3), dtype=np.uint8))


def test_to_gray_converts_bgr_to_single_channel(photo_without_faces):
    gray = to_gray(photo_without_faces)
    assert gray.shape == photo_without_faces.shape[:2]
    assert to_gray(gray) is gray  # already gray -> returned unchanged


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
def test_detect_returns_empty_list_for_image_without_faces(detector, photo_without_faces):
    faces = detector.detect(photo_without_faces)
    assert isinstance(faces, list)
    assert faces == []


def test_count_faces_matches_detect(detector, photo_without_faces):
    assert detector.count_faces(photo_without_faces) == len(
        detector.detect(photo_without_faces)
    )


def test_detection_gives_same_result_for_color_and_gray(detector, photo_without_faces):
    assert detector.detect(photo_without_faces) == detector.detect(
        to_gray(photo_without_faces)
    )


def test_every_detected_box_is_inside_the_image(detector, real_photos, photo_without_faces):
    """Boxes must be valid (x, y, w, h) integers inside the picture."""
    height, width = photo_without_faces.shape[:2]
    boxes = detector.detect(photo_without_faces) or []
    for x, y, w, h in boxes:
        assert isinstance(x, int) and isinstance(w, int)
        assert w > 0 and h > 0
        assert x >= 0 and y >= 0
        assert x + w <= width and y + h <= height


def test_detect_faces_function_works_without_a_detector_object(photo_without_faces):
    assert detect_faces(photo_without_faces) == []


def test_detect_rejects_none():
    with pytest.raises(InvalidImageError):
        detect_faces(None)


def test_detect_rejects_invalid_parameters(detector, photo_without_faces):
    with pytest.raises(InvalidParameterError):
        detector.detect(photo_without_faces, DetectionParams(scale_factor=1.0))


def test_stricter_parameters_do_not_report_more_faces(detector, real_photos):
    """min_neighbors/min_size may only remove candidates, never add them."""
    if not real_photos:
        pytest.skip("no photo in input/ - add one to compare parameter settings")

    strict = DetectionParams(min_neighbors=20, min_size=120)
    for photo in real_photos:
        image = load_image(photo)
        assert len(detector.detect(image, strict)) <= len(detector.detect(image))


@pytest.mark.parametrize("photo_name", ["photo_with_faces.jpg"])
def test_real_photo_detection(photo_name, detector, real_photos):
    """Detect faces in a real photo from input/ (skipped when absent)."""
    if not real_photos:
        pytest.skip("input/ has no photo - drop a group picture there to enable this")

    for photo in real_photos:
        faces = detector.detect(load_image(photo))
        assert isinstance(faces, list)
        for x, y, w, h in faces:
            assert w >= 0 and h >= 0


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def test_draw_faces_does_not_modify_the_original(photo_without_faces):
    before = photo_without_faces.copy()
    annotated = draw_faces(photo_without_faces, [(10, 10, 50, 50)])
    assert np.array_equal(photo_without_faces, before)
    assert not np.array_equal(annotated, photo_without_faces)


def test_draw_faces_draws_one_box_per_face(photo_without_faces):
    boxes = [(10, 10, 40, 40), (100, 60, 80, 80)]
    annotated = draw_faces(photo_without_faces, boxes)
    assert annotated.shape == photo_without_faces.shape
    # A rectangle was really drawn: some pixels are now green.
    green_pixels = np.all(annotated == (0, 255, 0), axis=2).sum()
    assert green_pixels > 0


def test_draw_faces_with_labels(photo_without_faces):
    annotated = draw_faces(photo_without_faces, [(10, 10, 40, 40)], labels=["Face 1"])
    assert annotated.shape == photo_without_faces.shape


def test_draw_faces_rejects_empty_image():
    with pytest.raises(InvalidImageError):
        draw_faces(np.zeros((0, 0, 3), dtype=np.uint8), [])


def test_draw_banner_changes_the_image(photo_without_faces):
    annotated = draw_banner(photo_without_faces, "Faces detected: 0")
    assert annotated.shape == photo_without_faces.shape
    assert not np.array_equal(annotated, photo_without_faces)


# --------------------------------------------------------------------------
# Loading and saving
# --------------------------------------------------------------------------
def test_load_image_reads_a_real_file(photo_path):
    image = load_image(photo_path)
    assert isinstance(image, np.ndarray)
    assert image.ndim == 3


def test_load_image_missing_file_raises(tmp_path):
    with pytest.raises(InvalidImageError) as info:
        load_image(tmp_path / "missing.jpg")
    assert "does not exist" in str(info.value)


def test_load_image_folder_raises(tmp_path):
    with pytest.raises(InvalidImageError):
        load_image(tmp_path)


def test_load_image_invalid_file_raises(tmp_path):
    """A .jpg that actually contains text must not crash the program."""
    fake = tmp_path / "fake.jpg"
    fake.write_text("definitely not an image", encoding="utf-8")
    with pytest.raises(InvalidImageError) as info:
        load_image(fake)
    assert "not a readable image" in str(info.value)


def test_save_image_creates_the_output_folder(tmp_path, photo_without_faces):
    target = tmp_path / "new_folder" / "result.png"
    saved = save_image(photo_without_faces, target)
    assert saved == target
    assert saved.is_file()
    assert saved.stat().st_size > 0


def test_save_image_result_can_be_read_back(tmp_path, photo_without_faces):
    saved = save_image(photo_without_faces, tmp_path / "roundtrip.jpg")
    reloaded = load_image(saved)
    assert reloaded.shape == photo_without_faces.shape


def test_save_image_rejects_empty_image(tmp_path):
    with pytest.raises(InvalidImageError):
        save_image(np.zeros((0, 0, 3), dtype=np.uint8), tmp_path / "empty.png")


def test_ensure_dir_is_idempotent(tmp_path):
    folder = tmp_path / "a" / "b"
    assert ensure_dir(folder).is_dir()
    assert ensure_dir(folder).is_dir()


def test_unique_path_adds_a_counter(tmp_path):
    first = tmp_path / "shot.png"
    assert unique_path(first) == first
    first.write_bytes(b"x")
    assert unique_path(first).name == "shot_1.png"


def test_build_output_path_keeps_name_and_folder():
    result = build_output_path(Path("input/photo.jpg"), Path("output"))
    assert result == Path("output/photo_detected.jpg")


def test_build_output_path_uses_custom_suffix():
    result = build_output_path(
        Path("input/photo.png"), Path("output"), suffix="_faces", extension=".png"
    )
    assert result == Path("output/photo_faces.png")


# --------------------------------------------------------------------------
# Command line size parsing
# --------------------------------------------------------------------------
@pytest.mark.parametrize("value,expected", [("60x60", 60), ("40X40", 40), ("100*100", 100)])
def test_parse_size_accepts_square_sizes(value, expected):
    assert parse_size(value) == expected


@pytest.mark.parametrize("value", ["60", "60x70", "abc", "-10x-10", "0x0", ""])
def test_parse_size_rejects_bad_sizes(value):
    with pytest.raises(InvalidParameterError):
        parse_size(value)


# --------------------------------------------------------------------------
# End to end through app.main()
# --------------------------------------------------------------------------
def test_cli_image_mode_saves_a_result_file(tmp_path, photo_path):
    output_dir = tmp_path / "output"
    exit_code = app.main(["--image", str(photo_path), "--output-dir", str(output_dir),
                          "--no-show"])
    assert exit_code == app.EXIT_OK
    result = output_dir / "sample_detected.jpg"
    assert result.is_file()
    assert load_image(result).ndim == 3


def test_cli_reports_missing_image_without_crashing(tmp_path, capsys):
    exit_code = app.main(["--image", str(tmp_path / "nope.jpg"), "--no-show"])
    captured = capsys.readouterr()
    assert exit_code == app.EXIT_BAD_INPUT
    assert "[error]" in captured.out
    assert "Traceback" not in captured.out


def test_cli_reports_invalid_image_without_crashing(tmp_path, capsys):
    fake = tmp_path / "fake.jpg"
    fake.write_text("not an image", encoding="utf-8")
    exit_code = app.main(["--image", str(fake), "--no-show"])
    assert exit_code == app.EXIT_BAD_INPUT
    assert "[error]" in capsys.readouterr().out


def test_cli_reports_missing_cascade_without_crashing(tmp_path, photo_path, capsys):
    exit_code = app.main(["--image", str(photo_path), "--cascade",
                          str(tmp_path / "missing.xml"), "--no-show"])
    assert exit_code == app.EXIT_BAD_INPUT
    assert "not found" in capsys.readouterr().out


def test_cli_reports_bad_parameters_before_doing_work(photo_path, capsys):
    with pytest.raises(SystemExit) as info:
        app.main(["--image", str(photo_path), "--min-size", "60x70", "--no-show"])
    assert info.value.code == 2  # argparse usage error
    assert "60x70" in capsys.readouterr().err


def test_cli_requires_image_or_webcam():
    with pytest.raises(SystemExit) as info:
        app.main([])
    assert info.value.code == 2


def test_cli_rejects_image_and_webcam_together(photo_path):
    with pytest.raises(SystemExit) as info:
        app.main(["--image", str(photo_path), "--webcam"])
    assert info.value.code == 2


def test_webcam_failure_is_handled_gracefully(capsys):
    """Camera index 99 does not exist, so the program must explain and stop."""
    exit_code = app.main(["--webcam", "--camera-index", "99", "--no-show"])
    assert exit_code == app.EXIT_NO_CAMERA
    assert "Traceback" not in capsys.readouterr().out


def test_open_camera_raises_camera_error():
    with pytest.raises(app.CameraError):
        app.open_camera(99)


def test_key_q_stops_the_webcam_loop(tmp_path):
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    assert app.key_action(ord("q"), image, tmp_path) is False


def test_key_s_saves_a_snapshot(tmp_path):
    image = np.full((10, 10, 3), 200, dtype=np.uint8)
    assert app.key_action(ord("s"), image, tmp_path) is True
    assert list(tmp_path.glob("webcam_*.png"))