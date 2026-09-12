import json

from PIL import Image
import pytest

from src.config import Config
from src.dataset.prepare import check_splits, prepare, split_originals
from src.dataset.validation import validate_dataset


@pytest.fixture
def dataset(tmp_path):
    images, annotations = tmp_path / "images", tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()
    for i in range(3):
        Image.new("RGB", (30, 20), color=(i * 50, 20, 30)).save(images / f"{i}.png")
        (annotations / f"{i}.json").write_text(json.dumps({"width": 30, "height": 20, "points": [{"x": 5, "y": 5}]}))
    return Config(images=images, annotations=annotations, processed=tmp_path / "processed", splits=tmp_path / "splits.json",
                  tile_size=10, overlap=2, box_size=4, negative_fraction=0)


def test_prepare_has_no_original_leakage_and_preserves_mapping(dataset):
    manifest = prepare(dataset)
    train = {t["source_id"] for t in manifest["tiles"] if t["split"] == "train"}
    val = {t["source_id"] for t in manifest["tiles"] if t["split"] == "val"}
    assert train.isdisjoint(val)
    assert train | val == {"0", "1", "2"}
    assert all(t["positive"] for t in manifest["tiles"] if t["split"] == "train")
    assert any(not t["positive"] for t in manifest["tiles"] if t["split"] == "val")
    for tile in manifest["tiles"]:
        with Image.open(dataset.processed / tile["image"]) as image:
            assert image.size == (tile["width"], tile["height"])
        for row in (dataset.processed / tile["label"]).read_text().splitlines():
            cls, *coords = row.split()
            assert cls == "0" and len(coords) == 4
            assert all(0 <= float(c) <= 1 for c in coords)
    assert dataset.splits.exists()
    with pytest.raises(ValueError, match="not empty"):
        prepare(dataset)


def test_reproducible_splits_and_leave_one_out(dataset):
    records = validate_dataset(dataset.images, dataset.annotations)
    assert split_originals(records) == split_originals(list(reversed(records)))
    held_out_ids = []
    for record in records:
        split = split_originals(records, held_out=record["id"])
        check_splits(split, records)
        held_out_ids.extend(split["val"])
        assert record["id"] not in split["train"]
    assert sorted(held_out_ids) == ["0", "1", "2"]
    with pytest.raises(ValueError, match="leakage"):
        check_splits({"train": ["0", "1"], "val": ["1", "2"]}, records)


@pytest.mark.parametrize("change,match", [
    ({"points": [{"x": 30, "y": 1}]}, "outside"),
    ({"points": [{"x": -1, "y": 1}]}, "outside"),
    ({"points": [{"x": 1, "y": 1}, {"x": 1, "y": 1}]}, "duplicate"),
    ({"width": 31}, "dimensions"),
    ({"points": [{"x": float("nan"), "y": 1}]}, "finite"),
    ({"points": [{"x": "1", "y": 1}]}, "number"),
    ({"unexpected": 1}, "Extra inputs"),
])
def test_validation_rejects_bad_annotations(dataset, change, match):
    path = dataset.annotations / "0.json"
    annotation = json.loads(path.read_text())
    annotation.update(change)
    path.write_text(json.dumps(annotation))
    with pytest.raises(ValueError, match=match):
        validate_dataset(dataset.images, dataset.annotations)


def test_missing_annotation_and_corrupt_image(dataset):
    (dataset.annotations / "0.json").unlink()
    (dataset.images / "1.png").write_bytes(b"not an image")
    with pytest.raises(ValueError) as exc:
        validate_dataset(dataset.images, dataset.annotations)
    assert "0.png" in str(exc.value) and "1.png" in str(exc.value)


def test_duplicate_images_rejected(dataset):
    (dataset.images / "1.png").write_bytes((dataset.images / "0.png").read_bytes())
    with pytest.raises(ValueError, match="duplicate decoded"):
        validate_dataset(dataset.images, dataset.annotations)


def test_labeling_schema_and_review_gate(dataset):
    path = dataset.annotations / "0.json"
    doc = {"image_id": "0", "filename": "original.jpg", "original_width": 30, "original_height": 20,
           "coordinate_space": "original", "dataset_version": "1", "annotations": [
               {"x": 5, "y": 5, "id": "a", "className": "varroa", "source": "human", "reviewed": True}],
           "rejected_predictions": []}
    path.write_text(json.dumps(doc))
    assert len(validate_dataset(dataset.images, dataset.annotations)) == 3
    doc["annotations"][0]["reviewed"] = False
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="Review all"):
        validate_dataset(dataset.images, dataset.annotations)


def test_all_negative_tiles_can_be_included(dataset):
    config = dataset.model_copy(update={"negative_fraction": 1.0})
    manifest = prepare(config)
    assert any(not t["positive"] and t["split"] == "train" for t in manifest["tiles"])
