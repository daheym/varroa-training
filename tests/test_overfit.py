import importlib
import json

from PIL import Image
import pytest
import yaml

from src.config import Config
from src.dataset.prepare import check_splits, prepare
from src.evaluation.loo import leave_one_out


@pytest.fixture
def single_image(tmp_path):
    images, annotations = tmp_path / "images", tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()
    Image.new("RGB", (30, 20), color="gray").save(images / "single.png")
    (annotations / "single.json").write_text('{"points": [{"x": 5, "y": 5}]}')
    return Config(allow_legacy_original=True, mode="overfit", images=images, annotations=annotations, processed=tmp_path / "dataset",
                  splits=tmp_path / "split.json", tile_size=10, overlap=2, box_size=4, negative_fraction=1)


def test_overfit_has_no_validation_originals_or_tiles(single_image):
    manifest = prepare(single_image)
    assert manifest["mode"] == "overfit"
    assert manifest["splits"] == {"train": sorted(t["id"] for t in manifest["tiles"]), "val": []}
    assert all(t["split"] == "train" for t in manifest["tiles"])
    assert any(not t["positive"] for t in manifest["tiles"])
    assert not list((single_image.processed / "images" / "val").iterdir())
    data = yaml.safe_load((single_image.processed / "dataset.yaml").read_text())
    assert data["train"] == data["val"] == "images/train"
    with pytest.raises(ValueError, match="Empty split"):
        check_splits(manifest["splits"], manifest["sources"])


def test_single_image_can_split_tiles_but_not_leave_one_original_out(single_image):
    manifest = prepare(single_image.model_copy(update={"mode": "holdout"}))
    assert manifest["splits"]["train"] and manifest["splits"]["val"]
    assert {tile["source_id"] for tile in manifest["tiles"]} == {"single"}
    with pytest.raises(ValueError, match="holdout mode"):
        leave_one_out(single_image, single_image.processed)


def test_holdout_needs_at_least_two_generated_tiles(single_image):
    config = single_image.model_copy(update={"mode": "holdout", "tile_size": 100, "overlap": 0,
                                             "processed": single_image.processed.parent / "one-tile",
                                             "splits": single_image.splits.parent / "one-tile.json"})
    with pytest.raises(ValueError, match="at least two unique generated tiles"):
        prepare(config)


def test_overfit_rejects_multiple_originals_and_no_positive_labels(single_image):
    Image.new("RGB", (30, 20), color="red").save(single_image.images / "second.png")
    (single_image.annotations / "second.json").write_text('{"points": []}')
    with pytest.raises(ValueError, match="exactly one"):
        prepare(single_image)
    (single_image.images / "second.png").unlink()
    (single_image.annotations / "second.json").unlink()
    (single_image.annotations / "single.json").write_text('{"points": []}')
    with pytest.raises(ValueError, match="no positive annotations"):
        prepare(single_image)
    assert not single_image.splits.exists()


def test_evaluation_uses_training_original_and_labels_results(single_image, monkeypatch):
    evaluation = importlib.import_module("src.evaluation.evaluate")
    manifest = prepare(single_image)
    seen = []

    def fake_predict(model, image_path, config, perspective=None):
        seen.append(image_path)
        return {"image": image_path, "count": 1, "detections": [{"box": (3, 3, 7, 7), "confidence": .9}]}

    monkeypatch.setattr(evaluation, "predict_image", fake_predict)
    output = single_image.processed.parent / "experiment" / "evaluation.json"
    report = evaluation.evaluate(None, manifest, single_image, output)
    assert len(seen) == 1 and report["aggregate"]["f1"] == 1
    assert report["evaluation_scope"] == "training_image_reconstruction"
    assert "does not measure generalization" in report["interpretation"]
    assert (output.parent / "reconstruction" / "single.png").exists()
    prediction = json.loads((output.parent / "reconstruction" / "single.json").read_text())
    assert prediction["evaluation_scope"] == "training_image_reconstruction"
    with pytest.raises(ValueError, match="mode differs"):
        evaluation.evaluate(None, manifest, single_image.model_copy(update={"mode": "holdout"}), output)


def test_overfit_command_prepares_and_saves_resolved_config(single_image, monkeypatch):
    overfit = importlib.import_module("src.training.overfit")
    seen = []

    def fake_train(config, output):
        seen.append((config, output))
        assert config.mode == "overfit" and (config.processed / "manifest.json").exists()
        return {"aggregate": {}, "evaluation_scope": "training_image_reconstruction"}

    monkeypatch.setattr(overfit, "train", fake_train)
    output = single_image.processed.parent / "run"
    overfit.overfit_image(single_image.model_copy(update={"mode": "holdout"}), output)
    assert len(seen) == 1 and seen[0][1] == output / "experiment"
    config = json.loads((output / "config.json").read_text())
    assert config["mode"] == "overfit"
    assert config["processed"] == str(output / "dataset")
    with pytest.raises(ValueError, match="already exists"):
        overfit.overfit_image(single_image, output)


def test_explicit_selection_excludes_other_originals(single_image, monkeypatch):
    overfit = importlib.import_module("src.training.overfit")
    Image.new("RGB", (30, 20), color="red").save(single_image.images / "unfinished.png")
    # An unfinished annotation on another original must not prevent the selected experiment.
    (single_image.annotations / "unfinished.json").write_text('{"points":')
    seen = []
    monkeypatch.setattr(overfit, "train", lambda config, output: seen.append(config))
    output = single_image.processed.parent / "selected-run"
    overfit.overfit_image(single_image, output, image_id="single")
    manifest = json.loads((seen[0].processed / "manifest.json").read_text())
    assert [r["id"] for r in manifest["sources"]] == ["single"]
    with pytest.raises(ValueError, match="image_id=missing"):
        overfit.overfit_image(single_image, output.parent / "missing-run", image_id="missing")
