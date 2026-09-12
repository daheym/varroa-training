import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.config import write_json
from src.dataset.prepare import check_splits
from src.evaluation.evaluate import evaluate


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train(config, output, held_out_evaluation=False):
    """Each call constructs a fresh pretrained model, including each LOO fold."""
    from ultralytics import YOLO

    output = Path(output).resolve()
    manifest_path = config.processed / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("mode", "holdout") != config.mode:
        raise ValueError("Prepared dataset mode differs from training config")
    overfit = config.mode == "overfit"
    if overfit and held_out_evaluation:
        raise ValueError("Overfit mode cannot perform held-out evaluation")
    check_splits(manifest["splits"], manifest["sources"], config.mode)
    evaluation_scope = "training_image_reconstruction" if overfit else "held_out_originals"
    for key in ("box_size", "tile_size", "overlap", "negative_fraction", "seed"):
        if manifest["config"][key] != config.model_dump(mode="json")[key]:
            raise ValueError(f"Prepared dataset config differs: {key}")
    output.mkdir(parents=True, exist_ok=False)
    versions = {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "ultralytics", "opencv-python", "Pillow", "pydantic", "numpy", "scipy")}
    git_commit = None
    if shutil.which("git"):
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        git_commit = git.stdout.strip() if git.returncode == 0 else None
    source_hashes = {path.relative_to(Path(__file__).resolve().parents[1]).as_posix(): sha256(path)
                     for path in sorted(Path(__file__).resolve().parents[1].rglob("*.py"))}
    metadata = dict(status="running", started_at=datetime.now(timezone.utc).isoformat(),
                    config=config.model_dump(mode="json"), device="cpu", python=platform.python_version(),
                    platform=platform.platform(), packages=versions, git_commit=git_commit,
                    source_sha256=source_hashes, manifest_sha256=sha256(manifest_path),
                    held_out_evaluation=held_out_evaluation, mode=config.mode, evaluation_scope=evaluation_scope)
    write_json(output / "metadata.json", metadata)
    write_json(output / "dataset_manifest.json", manifest)
    write_json(output / "metrics.json", {"status": "running", "evaluation_scope": evaluation_scope})
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        (output / "requirements-freeze.txt").write_text(freeze.stdout, encoding="utf-8")
        data_path = (config.processed / "dataset.yaml").resolve()
        if held_out_evaluation:
            # Ultralytics also validates at finalization with val=False. Keep the
            # outer fold out of its data config entirely and ignore training-set metrics.
            data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
            data["val"] = data["train"]
            data_path = output / "training_only.yaml"
            data_path.write_text(yaml.safe_dump(data), encoding="utf-8")
        model = YOLO(config.model)
        initial_path = Path(getattr(model, "ckpt_path", config.model))
        if initial_path.is_file():
            metadata["initial_weights_sha256"] = sha256(initial_path)
        augmentation = dict(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, translate=0.0, scale=0.0,
                            fliplr=0.0, flipud=0.0, degrees=0.0, shear=0.0, perspective=0.0) if overfit else {}
        model.train(data=str(data_path), device="cpu",
                    imgsz=config.imgsz, batch=config.batch, epochs=config.epochs, workers=config.workers,
                    seed=config.seed, deterministic=True, pretrained=True, amp=False,
                    project=str(output), name="yolo", exist_ok=False,
                    # LOO holds the outer image out of epoch validation and checkpoint selection.
                    val=not (held_out_evaluation or overfit), patience=0, plots=False, save=True,
                    mosaic=0.0, mixup=0.0, erasing=0.0, max_det=config.max_det, **augmentation)
        checkpoint = output / "yolo" / "weights" / ("last.pt" if held_out_evaluation or overfit else "best.pt")
        report = evaluate(YOLO(str(checkpoint)), manifest, config, output / "evaluation.json")
        history_path = output / "yolo" / "results.csv"
        history = []
        if history_path.exists():
            with history_path.open(encoding="utf-8", newline="") as stream:
                history = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(stream)]
        write_json(output / "metrics.json", {"status": "complete", "evaluation_scope": evaluation_scope,
                                              "evaluation": report["aggregate"], "epochs": history})
        metadata.update(status="complete", checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
        return report
    except Exception as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "metrics.json", {"status": "failed", "evaluation_scope": evaluation_scope, "error": metadata["error"]})
        raise
    finally:
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "metadata.json", metadata)
