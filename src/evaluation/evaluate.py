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
    check_splits(manifest["splits"], manifest["sources"], config.mode)
    overfit = config.mode == "overfit"
    split = "train" if overfit else "val"
    rows = []
    for record in manifest["sources"]:
        if record["id"] not in manifest["splits"][split]:
            continue
        with Image.open(record["image"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Source image changed since preparation: {record['id']}")
        prediction = predict_image(model, record["image"], config)
        if overfit:
            prediction["evaluation_scope"] = "training_image_reconstruction"
            save_prediction(prediction, Path(output).parent / "reconstruction" / record["id"])
        rows.append({"source_id": record["id"], **image_metrics(record["points"], prediction["detections"], config.match_radius),
                     "detections": prediction["detections"]})
    report = {"matching": "one-to-one center distance in original pixels", "match_radius": config.match_radius,
              "evaluation_scope": "training_image_reconstruction" if overfit else "held_out_originals",
              "interpretation": "Same image used for training; does not measure generalization." if overfit else "Evaluation on held-out originals.",
              "config": config.model_dump(mode="json"), "per_image": rows, "aggregate": summarize(rows)}
    write_json(output, report)
    return report
