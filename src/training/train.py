import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.config import write_json
from src.dataset.prepare import check_splits
from src.evaluation.evaluate import evaluate
from src.inference.contract import preprocessing_contract


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(requested="auto"):
    """Resolve provider-neutral device values to the format Ultralytics accepts."""
    import torch

    requested = str(requested).strip().lower()
    available = torch.cuda.is_available()
    count = torch.cuda.device_count() if available else 0
    if requested == "auto":
        return "0" if available else "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        requested = "0"
    parts = requested.split(",")
    indices = []
    for part in parts:
        match = re.fullmatch(r"(?:cuda:)?(\d+)", part.strip())
        if not match:
            raise ValueError(f"Unsupported device '{requested}'. Use auto, cpu, cuda, cuda:N, or a CUDA index.")
        indices.append(int(match.group(1)))
    if not available:
        raise RuntimeError(f"CUDA device '{requested}' was requested, but CUDA is not available")
    if any(index >= count for index in indices):
        raise ValueError(f"CUDA device index is out of range: requested {indices}, available device count is {count}")
    return ",".join(str(index) for index in indices)


def _resolve_dataset_value(value, root, yaml_parent):
    if isinstance(value, list):
        return [_resolve_dataset_value(item, root, yaml_parent) for item in value]
    if not isinstance(value, str) or "://" in value:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path.resolve())
    candidate = (root / path).resolve()
    if not candidate.exists():
        yaml_candidate = (yaml_parent / path).resolve()
        if yaml_candidate.exists():
            candidate = yaml_candidate
    if not candidate.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {candidate}")
    return str(candidate)


def resolve_dataset_yaml(data_path):
    """Load a YOLO dataset config with portable paths made explicit."""
    data_path = Path(data_path).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset config does not exist: {data_path}")
    payload = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset config must contain a YAML mapping: {data_path}")
    configured_root = payload.get("path")
    if configured_root:
        root = Path(configured_root).expanduser()
        if not root.is_absolute():
            root = data_path.parent / root
    else:
        root = data_path.parent
    root = root.resolve()
    payload["path"] = str(root)
    for split in ("train", "val", "test", "minival"):
        if payload.get(split):
            payload[split] = _resolve_dataset_value(payload[split], root, data_path.parent)
    return payload


def _load_prepared_manifest(data_path):
    manifest_path = Path(data_path).parent / "manifest.json"
    if not manifest_path.is_file():
        return None, None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not all(key in manifest for key in ("mode", "config", "splits", "sources", "tiles")):
        return None, None
    return manifest_path, manifest


def _history(csv_path):
    if not csv_path.exists():
        return []
    with csv_path.open(encoding="utf-8", newline="") as stream:
        return [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(stream)]


def _write_resolved_config(path, config, data_path):
    resolved = config.model_dump(mode="json")
    resolved["data"] = str(Path(data_path).expanduser().resolve())
    path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")


def train(config, output, held_out_evaluation=False, data=None):
    """Train YOLO on prepared project data or an ordinary YOLO dataset config."""
    from ultralytics import YOLO

    output = Path(output).expanduser().resolve()
    requested_data = data or config.data or (config.processed / "dataset.yaml")
    manifest_path, manifest = _load_prepared_manifest(requested_data)
    if manifest is None and (held_out_evaluation or config.mode == "overfit"):
        raise ValueError("Overfit and held-out evaluation require the project's prepared dataset manifest")
    overfit = bool(manifest and manifest.get("mode") == "overfit")
    if manifest:
        if manifest.get("mode", "holdout") != config.mode:
            raise ValueError("Prepared dataset mode differs from training config")
        check_splits(manifest["splits"], manifest["tiles"], config.mode)
        for key in ("box_size", "tile_size", "overlap", "negative_fraction", "pixels_per_mm", "box_size_mm", "min_source_scale_ratio", "quality_policy", "allow_legacy_original"):
            if manifest["config"][key] != config.model_dump(mode="json")[key]:
                raise ValueError(f"Prepared dataset config differs: {key}")
    evaluation_scope = (
        "training_image_reconstruction" if overfit
        else "held_out_originals" if manifest and manifest.get("split_strategy") == "held_out_source"
        else "held_out_tiles" if manifest
        else "ultralytics_validation"
    )
    device = resolve_device(config.device)
    data_payload = resolve_dataset_yaml(requested_data)
    output.mkdir(parents=True, exist_ok=False)
    _write_resolved_config(output / "config.yaml", config, requested_data)
    data_path = output / "data.yaml"
    data_path.write_text(yaml.safe_dump(data_payload, sort_keys=False), encoding="utf-8")
    if held_out_evaluation:
        # Leave the outer fold out of training and checkpoint selection.
        data_payload["val"] = data_payload["train"]
        data_path.write_text(yaml.safe_dump(data_payload, sort_keys=False), encoding="utf-8")

    package_names = ("torch", "torchvision", "ultralytics", "opencv-python", "Pillow", "pydantic", "numpy", "scipy", "PyYAML")
    versions = {package: importlib.metadata.version(package) for package in package_names}
    repo_root = Path(__file__).resolve().parents[2]
    git_commit = None
    if shutil.which("git"):
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=False)
        git_commit = git.stdout.strip() if git.returncode == 0 else None
    source_hashes = {path.relative_to(repo_root).as_posix(): sha256(path)
                     for path in sorted(repo_root.rglob("*.py"))
                     if ".venv" not in path.parts and "__pycache__" not in path.parts}
    metadata = dict(status="running", started_at=datetime.now(timezone.utc).isoformat(),
                    config=config.model_dump(mode="json"), requested_device=config.device, device=device,
                    data=str(Path(requested_data).expanduser().resolve()), data_sha256=sha256(requested_data),
                    python=platform.python_version(), platform=platform.platform(), packages=versions,
                    git_commit=git_commit, source_sha256=source_hashes,
                    manifest_sha256=sha256(manifest_path) if manifest_path else None,
                    held_out_evaluation=held_out_evaluation, mode=config.mode,
                    evaluation_scope=evaluation_scope)
    write_json(output / "metadata.json", metadata)
    if manifest:
        write_json(output / "dataset_manifest.json", manifest)
    write_json(output / "metrics.json", {"status": "running", "evaluation_scope": evaluation_scope})
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], cwd=repo_root,
                                capture_output=True, text=True, check=True)
        (output / "requirements-freeze.txt").write_text(freeze.stdout, encoding="utf-8")
        initial_model = config.resume if config.resume else config.model
        model = YOLO(str(initial_model))
        initial_path = Path(getattr(model, "ckpt_path", None) or initial_model)
        if initial_path.is_file():
            metadata["initial_weights_sha256"] = sha256(initial_path)
        augmentation = dict(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, translate=0.0, scale=0.0,
                            fliplr=0.0, flipud=0.0, degrees=0.0, shear=0.0, perspective=0.0) if overfit else dict(scale=0.0, degrees=0.0, shear=0.0, perspective=0.0)
        options = dict(data=str(data_path), device=device, imgsz=config.imgsz, batch=config.batch,
                       epochs=config.epochs, workers=config.workers, seed=config.seed, deterministic=True,
                       pretrained=True, amp=device != "cpu", project=str(output), name="yolo",
                       exist_ok=False, val=not (held_out_evaluation or overfit),
                       patience=0 if held_out_evaluation or overfit else config.patience,
                       plots=not overfit, save=True, mosaic=0.0, mixup=0.0, erasing=0.0,
                       max_det=config.max_det, **augmentation)
        if config.resume:
            resume_path = Path(config.resume).expanduser().resolve()
            if not resume_path.is_file():
                raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
            options.update(resume=True, save_dir=str(output / "yolo"))
        model.train(**options)
        checkpoint = output / "yolo" / "weights" / ("last.pt" if held_out_evaluation or overfit else "best.pt")
        for weights in checkpoint.parent.glob("*.pt"):
            write_json(weights.with_suffix(".preprocessing.json"), preprocessing_contract(config))
        history = _history(output / "yolo" / "results.csv")
        if manifest:
            report = evaluate(YOLO(str(checkpoint)), manifest, config, output / "evaluation.json")
            write_json(output / "metrics.json", {"status": "complete", "evaluation_scope": evaluation_scope,
                                                  "evaluation": report["aggregate"], "epochs": history})
        else:
            report = {"evaluation_scope": evaluation_scope, "aggregate": history[-1] if history else {}}
            write_json(output / "metrics.json", {"status": "complete", "evaluation_scope": evaluation_scope,
                                                  "validation": report["aggregate"], "epochs": history})
        metadata.update(status="complete", checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
        return report
    except Exception as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "metrics.json", {"status": "failed", "evaluation_scope": evaluation_scope,
                                              "error": metadata["error"]})
        raise
    finally:
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "metadata.json", metadata)
