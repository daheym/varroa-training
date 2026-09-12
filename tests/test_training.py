import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image
import pytest
import yaml

from src.config import Config
from src.dataset.prepare import prepare


@pytest.mark.parametrize("held_out,mode", [(False, "holdout"), (True, "holdout"), (False, "overfit")])
def test_training_cpu_and_checkpoint_protocol(tmp_path, monkeypatch, held_out, mode):
    training = importlib.import_module("src.training.train")
    images, annotations = tmp_path / "images", tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()
    for i in range(1 if mode == "overfit" else 2):
        Image.new("RGB", (20, 20), color=(i, 0, 0)).save(images / f"{i}.png")
        (annotations / f"{i}.json").write_text('{"points": [{"x": 5, "y": 5}]}')
    config = Config(allow_legacy_original=True, mode=mode, images=images, annotations=annotations, processed=tmp_path / "data", splits=tmp_path / "split.json")
    prepare(config)
    calls = []

    class FakeYOLO:
        def __init__(self, weights):
            calls.append(weights)

        def train(self, **kwargs):
            assert kwargs["device"] == "cpu" and kwargs["amp"] is False
            assert kwargs["seed"] == config.seed and kwargs["deterministic"] is True
            assert kwargs["val"] == (not held_out and mode != "overfit") and kwargs["patience"] == 0
            if mode == "overfit":
                assert all(kwargs[key] == 0 for key in ("hsv_h", "hsv_s", "hsv_v", "translate", "scale", "fliplr", "flipud"))
            data = yaml.safe_load(Path(kwargs["data"]).read_text())
            assert (data["val"] == data["train"]) == (held_out or mode == "overfit")
            weights = Path(kwargs["project"]) / "yolo" / "weights"
            weights.mkdir(parents=True)
            (weights / "best.pt").write_bytes(b"best")
            (weights / "last.pt").write_bytes(b"last")

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.setattr(training, "evaluate", lambda *args: {"aggregate": {"f1": 0.0}})
    result = training.train(config, tmp_path / "run", held_out_evaluation=held_out)
    assert result["aggregate"]["f1"] == 0
    assert calls[0] == "yolo11n.pt"
    assert Path(calls[-1]).name == ("last.pt" if held_out or mode == "overfit" else "best.pt")
    metadata = json.loads((tmp_path / "run" / "metadata.json").read_text())
    assert metadata["status"] == "complete" and metadata["checkpoint_sha256"]
    assert "training/train.py" in metadata["source_sha256"]
    assert metadata["git_commit"] != "HEAD"
    assert metadata["evaluation_scope"] == ("training_image_reconstruction" if mode == "overfit" else "held_out_originals")
    assert (tmp_path / "run" / "metrics.json").exists()


def test_changed_source_during_prepare_is_rejected(tmp_path):
    from src.dataset.validation import validate_dataset

    images, annotations = tmp_path / "images", tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()
    for i in range(2):
        Image.new("RGB", (20, 20), color=(i, 0, 0)).save(images / f"{i}.png")
        (annotations / f"{i}.json").write_text('{"points": [{"x": 5, "y": 5}]}')
    config = Config(allow_legacy_original=True, images=images, annotations=annotations, processed=tmp_path / "data", splits=tmp_path / "split.json")
    records = validate_dataset(images, annotations, config=config)
    Image.new("RGB", (20, 20), color="red").save(images / "0.png")
    with pytest.raises(ValueError, match="changed during preparation"):
        prepare(config, records=records)
