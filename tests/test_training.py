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
            assert kwargs["val"] == (not held_out and mode != "overfit")
            assert kwargs["patience"] == (0 if held_out or mode == "overfit" else config.patience)
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
    assert metadata["evaluation_scope"] == ("training_image_reconstruction" if mode == "overfit" else "held_out_tiles")
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


def test_device_selection_and_yaml_config(monkeypatch):
    training = importlib.import_module("src.training.train")
    import torch

    assert training.resolve_device("cpu") == "cpu"
    assert training.resolve_device("auto") == ("0" if torch.cuda.is_available() else "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert training.resolve_device("cuda") == "0"
    assert training.resolve_device("cuda:1") == "1"
    with pytest.raises(ValueError, match="out of range"):
        training.resolve_device("2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        training.resolve_device("cuda:0")
    from src.config import read_config
    assert read_config("training/configs/varroa.yaml").device == "auto"


def test_normal_train_cli_overrides_config(tmp_path, monkeypatch, capsys):
    cli = importlib.import_module("src.cli")
    training = importlib.import_module("src.training.train")
    seen = []
    monkeypatch.setattr(training, "train", lambda config, output, data=None: seen.append((config, output, data)) or {
        "evaluation_scope": "ultralytics_validation", "aggregate": {},
    })
    dataset_config = tmp_path / "dataset.yaml"
    cli.main([
        "train", "--config", "training/configs/varroa.yaml", "--data", str(dataset_config),
        "--output", str(tmp_path / "run"), "--device", "cpu", "--seed", "7", "--image-size", "320",
        "--batch-size", "4", "--epochs", "3", "--patience", "5", "--workers", "2",
    ])
    config, output, data = seen[0]
    assert (config.device, config.seed, config.imgsz, config.batch, config.epochs, config.patience, config.workers) == ("cpu", 7, 320, 4, 3, 5, 2)
    assert output == tmp_path / "run" and data == dataset_config
    assert "ultralytics_validation" in capsys.readouterr().out


def test_external_yolo_dataset_uses_shared_training_path(tmp_path, monkeypatch):
    training = importlib.import_module("src.training.train")
    images = tmp_path / "dataset" / "images"
    labels = tmp_path / "dataset" / "labels"
    for split in ("train", "val"):
        (images / split).mkdir(parents=True)
        (labels / split).mkdir(parents=True)
        Image.new("RGB", (16, 16), color="white").save(images / split / "one.png")
        (labels / split / "one.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    data_yaml = tmp_path / "dataset" / "dataset.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": ".", "train": "images/train", "val": "images/val", "names": {0: "varroa"},
    }))
    config = Config(data=data_yaml, processed=tmp_path / "unused", device="cpu", epochs=1)
    calls = []

    class FakeYOLO:
        ckpt_path = None

        def __init__(self, weights):
            calls.append(("init", weights))

        def train(self, **kwargs):
            calls.append(("train", kwargs))
            assert kwargs["device"] == "cpu" and kwargs["amp"] is False
            resolved = yaml.safe_load(Path(kwargs["data"]).read_text())
            assert Path(resolved["train"]).is_absolute()
            assert Path(resolved["val"]).is_absolute()
            weights = Path(kwargs["project"]) / "yolo" / "weights"
            weights.mkdir(parents=True)
            (weights / "best.pt").write_bytes(b"best")
            (weights / "last.pt").write_bytes(b"last")
            results = Path(kwargs["project"]) / "yolo" / "results.csv"
            results.write_text("epoch,metrics/precision(B),metrics/recall(B)\n0,0.7,0.6\n")

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.setattr(training, "evaluate", lambda *args: pytest.fail("external YOLO data has no project manifest"))
    output = tmp_path / "run"
    report = training.train(config, output)
    assert report == {"evaluation_scope": "ultralytics_validation", "aggregate": {"epoch": "0", "metrics/precision(B)": "0.7", "metrics/recall(B)": "0.6"}}
    assert (output / "config.yaml").exists()
    assert (output / "data.yaml").exists()
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["status"] == "complete" and metrics["evaluation_scope"] == "ultralytics_validation"
    assert (output / "yolo" / "weights" / "best.preprocessing.json").exists()
