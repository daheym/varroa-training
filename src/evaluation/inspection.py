"""Dense, portable object crops with immutable run and matching provenance."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

from src.config import write_json
from src.dataset.geometry import point_to_box, clip_to_tile
from src.dataset.perspective import normalize_image
from src.dataset.quality import geometry_hash, review_key
from src.evaluation.metrics import match_detections


def _crop(image, x, y, size=96):
    left, top = round(x - size / 2), round(y - size / 2)
    return image.crop((left, top, left + size, top + size))


def build_inspection(run):
    run = Path(run).resolve()
    manifest = json.loads((run / "dataset_manifest.json").read_text(encoding="utf-8"))
    report = json.loads((run / "evaluation.json").read_text(encoding="utf-8"))
    output = run / "inspection"
    (output / "crops").mkdir(parents=True, exist_ok=True)
    root = Path(manifest["config"]["processed"])
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    results = {r.get("tile_id", r["source_id"]): r
               for r in [*report["per_image"], *report.get("training_diagnostics", [])]}
    config = manifest["config"]
    box_size = (config["box_size_mm"] * config["pixels_per_mm"]
                if config.get("box_size_mm") is not None else config["box_size"])
    items = []
    for record in manifest["sources"]:
        with Image.open(record["image"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Source image changed since training: {record['id']}")
        if record.get("perspective"):
            image = normalize_image(image, record["perspective"])
        annotated = []
        for index, (x, y) in enumerate(record["points"]):
            annotated.append(dict(id=f"{record['id']}-point-{index}", kind="annotation",
                                  source_id=record["id"], x=x, y=y,
                                  review_key=review_key(record, x, y),
                                  source_sha256=record["sha256"], geometry_sha256=geometry_hash(record),
                                  splits=[], outcomes=[]))
        extras = []
        for tile in manifest["tiles"]:
            if tile["source_id"] != record["id"]:
                continue
            window = (tile["x"], tile["y"], tile["width"], tile["height"])
            indices = [i for i, (x, y) in enumerate(record["points"])
                       if clip_to_tile(point_to_box(x, y, box_size, record["width"], record["height"]), window)]
            for i in indices:
                if tile["split"] not in annotated[i]["splits"]:
                    annotated[i]["splits"].append(tile["split"])
            result = results.get(tile["id"])
            if result is None:
                continue
            points = result.get("points")
            if points is None:
                points = []
                for line in (root / tile["label"]).read_text(encoding="utf-8").splitlines():
                    _, x, y, _, _ = map(float, line.split())
                    points.append((x * tile["width"], y * tile["height"]))
            if len(points) != len(indices):
                raise ValueError(f"Snapshot label count disagrees with tile: {tile['id']}")
            matching = match_detections(points, result["detections"], report["match_radius"])
            for label_index, detection_index in matching["matches"]:
                annotated[indices[label_index]]["outcomes"].append({
                    "tile_id": tile["id"], "split": tile["split"], "status": "matched",
                    "confidence": result["detections"][detection_index]["confidence"]})
            for label_index in matching["missed"]:
                annotated[indices[label_index]]["outcomes"].append({
                    "tile_id": tile["id"], "split": tile["split"], "status": "missed"})
            for detection_index in matching["extra"]:
                detection = result["detections"][detection_index]
                a, b, c, d = detection["box"]
                extras.append(dict(id=f"{tile['id']}-extra-{detection_index}", kind="prediction",
                                   source_id=record["id"], x=tile["x"] + (a + c) / 2,
                                   y=tile["y"] + (b + d) / 2, splits=[tile["split"]],
                                   outcomes=[dict(tile_id=tile["id"], split=tile["split"],
                                                  status="extra", confidence=detection["confidence"])]))
        # Original-level protocols (overfit and leave-one-original-out) use full-image matching.
        if record["id"] in results:
            result = results[record["id"]]
            matching = match_detections(record["points"], result["detections"], report["match_radius"])
            split = "train" if report["evaluation_scope"] == "training_image_reconstruction" else "val"
            for i, j in matching["matches"]:
                annotated[i]["outcomes"].append(dict(split=split, status="matched", confidence=result["detections"][j]["confidence"]))
            for i in matching["missed"]:
                annotated[i]["outcomes"].append(dict(split=split, status="missed"))
            for j in matching["extra"]:
                a, b, c, d = result["detections"][j]["box"]
                extras.append(dict(id=f"{record['id']}-extra-{j}", kind="prediction", source_id=record["id"],
                                   x=(a+c)/2, y=(b+d)/2, splits=[split],
                                   outcomes=[dict(split=split, status="extra", confidence=result["detections"][j]["confidence"])]))
        for item in [*annotated, *extras]:
            statuses = {r["status"] for r in item["outcomes"]}
            item["status"] = ("mixed" if {"missed", "matched"} <= statuses else
                              "missed" if "missed" in statuses else "extra" if "extra" in statuses else
                              "matched" if "matched" in statuses else "not_evaluated")
            item["crop"] = f"crops/{item['id']}.jpg"
            _crop(image, item["x"], item["y"]).save(output / item["crop"], quality=92)
            items.append(item)
    payload = dict(schema_version=1, run_name=run.name, crop_size=96,
                   evaluation_scope=report["evaluation_scope"], interpretation=report.get("interpretation", ""),
                   note="Each annotation appears once. Outcomes retain every overlapping tile; unmatched predictions may repeat across tiles. Training outcomes are reconstruction, not unseen-image accuracy.",
                   counts=dict(Counter(item["status"] for item in items)), items=items)
    write_json(output / "manifest.json", payload)
    data = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("inspection.html").read_text(encoding="utf-8")
    (output / "index.html").write_text(template.replace("__INSPECTION_DATA__", data), encoding="utf-8")
    return output / "index.html"
