import pytest

from src.dataset.geometry import (clip_to_tile, from_yolo, merge_detections,
                                  point_to_box, tile_windows, to_original, to_yolo)


def test_coordinate_round_trip():
    box = (10, 20, 50, 60)
    normalized = to_yolo(box, 100, 200)
    assert normalized == pytest.approx((.3, .2, .4, .2))
    assert from_yolo(normalized, 100, 200) == pytest.approx(box)


def test_point_to_box_and_edges():
    assert point_to_box(50, 40, 20, 100, 100) == (40, 30, 60, 50)
    assert point_to_box(0, 0, 20, 100, 100) == (0, 0, 10, 10)
    assert point_to_box(99, 99, 20, 100, 100) == (89, 89, 100, 100)
    with pytest.raises(ValueError):
        point_to_box(100, 0, 20, 100, 100)


@pytest.mark.parametrize("width,height,size,overlap", [(11, 9, 4, 1), (2, 3, 8, 2), (8, 8, 4, 0), (10, 10, 4, 3)])
def test_tiling_covers_every_pixel(width, height, size, overlap):
    windows = tile_windows(width, height, size, overlap)
    assert len(windows) == len(set(windows))
    covered = set()
    for x, y, w, h in windows:
        assert 0 < w <= size and 0 < h <= size
        assert x + w <= width and y + h <= height
        covered.update((a, b) for a in range(x, x + w) for b in range(y, y + h))
    assert len(covered) == width * height


def test_mapping_and_partial_intersection():
    window = (100, 200, 50, 50)
    assert to_original((1, 2, 10, 20), window) == (101, 202, 110, 220)
    assert clip_to_tile((90, 190, 110, 210), window) == (0, 0, 10, 10)
    assert clip_to_tile((0, 0, 100, 100), window) is None


def test_duplicate_merging_keeps_confident_and_separate_mites():
    detections = [{"box": (0, 0, 10, 10), "confidence": .8},
                  {"box": (1, 1, 11, 11), "confidence": .9},
                  {"box": (20, 20, 30, 30), "confidence": .7}]
    assert merge_detections(detections) == [detections[1], detections[2]]
    assert merge_detections([]) == []


def test_invalid_tiling():
    with pytest.raises(ValueError):
        tile_windows(20, 20, 10, 10)
