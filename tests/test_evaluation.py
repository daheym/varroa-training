import pytest

from src.evaluation.metrics import image_metrics, match_detections, summarize


def detection(x, y):
    return {"box": (x - 1, y - 1, x + 1, y + 1), "confidence": .9}


def test_one_to_one_and_count_error():
    metrics = image_metrics([(10, 10), (50, 50)], [detection(10, 10), detection(11, 10), detection(90, 90)], 3)
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (1, 2, 1)
    assert metrics["count_error"] == 1
    assert metrics["f1"] == pytest.approx(.4)


def test_matching_maximizes_valid_matches():
    metrics = image_metrics([(0, 0), (3, 0)], [detection(1, 0), detection(-2, 0)], 2)
    assert metrics["tp"] == 2
    matching = match_detections([(0, 0), (3, 0)], [detection(1, 0), detection(-2, 0)], 2)
    assert matching == {"matches": [(0, 1), (1, 0)], "missed": [], "extra": []}


def test_matching_exposes_correct_error_indices():
    matching = match_detections([(10, 10), (50, 50)], [detection(10, 10), detection(90, 90)], 3)
    assert matching == {"matches": [(0, 0)], "missed": [1], "extra": [1]}
    assert match_detections([], [], 3) == {"matches": [], "missed": [], "extra": []}


def test_empty_cases_and_aggregate():
    empty = image_metrics([], [], 1)
    assert empty["f1"] == 0 and empty["absolute_count_error"] == 0
    missed = image_metrics([(1, 1)], [], 1)
    extra = image_metrics([], [detection(5, 5)], 1)
    result = summarize([missed, extra])
    assert result["count_mae"] == 1 and result["count_bias"] == 0
