import hashlib
from pathlib import Path

from PIL import Image, ImageOps

from src.config import write_json
from src.dataset.prepare import check_splits
from src.evaluation.metrics import image_metrics, summarize
from src.inference.predict import predict_image, save_prediction


def evaluate(model, manifest, config, output):
    if manifest.get("mode", "holdout") != config.mode:
        raise ValueError("Prepared dataset mode differs from evaluation config")
    check_splits(manifest["splits"], manifest["tiles"], config.mode)
    overfit = config.mode == "overfit"
    rows = []
    if not overfit and manifest.get("split_strategy") == "generated_tiles":
        root = Path(output).parent / "dataset"
        manifest_root = Path(manifest["config"]["processed"]).expanduser().resolve()
        root = manifest_root if (manifest_root / "manifest.json").is_file() else root
        validation_ids = set(manifest["splits"]["val"])
        for tile in manifest["tiles"]:
            if tile["id"] not in validation_ids:
                continue
            prediction = predict_image(model, root / tile["image"], config, normalized=True)
            points = []
            for line in (root / tile["label"]).read_text(encoding="utf-8").splitlines():
                _, x, y, _, _ = map(float, line.split())
                points.append((x * tile["width"], y * tile["height"]))
            rows.append({"tile_id": tile["id"], "source_id": tile["source_id"],
                         **image_metrics(points, prediction["detections"], config.match_radius),
                         "detections": prediction["detections"]})
        scope = "held_out_tiles"
        interpretation = "Metrics are measured on generated validation tiles; overlapping tiles may share image content."
    else:
        split = "train" if overfit else "val"
        held_out_sources = set(manifest.get("held_out_sources", []))
        for record in manifest["sources"]:
            if not overfit and record["id"] not in held_out_sources:
                continue
            if overfit and not any(tile["source_id"] == record["id"] and tile["id"] in manifest["splits"][split]
                                   for tile in manifest["tiles"]):
                continue
            with Image.open(record["image"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            if digest != record["sha256"]:
                raise ValueError(f"Source image changed since preparation: {record['id']}")
            prediction = predict_image(model, record["image"], config, perspective=record.get("perspective"))
            if overfit:
                prediction["evaluation_scope"] = "training_image_reconstruction"
                save_prediction(prediction, Path(output).parent / "reconstruction" / record["id"])
            rows.append({"source_id": record["id"], **image_metrics(record["points"], prediction["detections"], config.match_radius),
                         "detections": prediction["detections"]})
        scope = "training_image_reconstruction" if overfit else "held_out_originals"
        interpretation = ("Same image used for training; does not measure generalization." if overfit
                          else "Evaluation on explicitly held-out originals.")
    report = {"matching": "one-to-one center distance in prepared-image pixels", "match_radius": config.match_radius,
              "evaluation_scope": scope, "interpretation": interpretation,
              "config": config.model_dump(mode="json"), "per_image": rows, "aggregate": summarize(rows)}
    write_json(output, report)
    return report
