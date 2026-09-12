import hashlib
import json
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field


class Point(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)
    x: float
    y: float


class Annotation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    image: str | None = None
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    points: list[Point]


class LabelingPoint(Point):
    id: str
    className: Literal["varroa"]
    source: Literal["human", "model"]
    confidence: float | None = Field(default=None, ge=0, le=1)
    reviewed: bool


class LabelingDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    image_id: str
    filename: str
    original_width: int = Field(gt=0)
    original_height: int = Field(gt=0)
    coordinate_space: Literal["original"]
    perspective: dict | None = None
    dataset_version: str
    model_version: str | None = None
    annotations: list[LabelingPoint]
    rejected_predictions: list[LabelingPoint] = Field(default_factory=list)
    effort: dict = Field(default_factory=dict)


def load_annotation(path, image_path):
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    if isinstance(payload, dict) and "image_id" in payload:
        doc = LabelingDocument.model_validate_json(raw)
        if doc.image_id != path.stem or doc.image_id != image_path.stem:
            raise ValueError("image_id must match annotation and image stems")
        if any(not p.reviewed for p in doc.annotations):
            raise ValueError("Review all annotations before using them as ground truth")
        if len({p.id for p in doc.annotations}) != len(doc.annotations):
            raise ValueError("duplicate annotation IDs")
        return Annotation(width=doc.original_width, height=doc.original_height,
                          points=[Point(x=p.x, y=p.y) for p in doc.annotations])
    return Annotation.model_validate_json(raw)


def validate_dataset(images: Path, annotations: Path, image_id: str | None = None):
    """Collect all errors; never silently treat missing annotations as negatives."""
    records, errors = [], []
    files = sorted(p for p in images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) if images.is_dir() else []
    if image_id is not None:
        files = [path for path in files if path.stem == image_id]
    if not files:
        errors.append(f"No JPEG/PNG image found for image_id={image_id}" if image_id is not None else "No JPEG/PNG images found")
    stems, hashes = set(), set()
    for path in files:
        try:
            if path.stem.casefold() in stems:
                raise ValueError("duplicate image stem")
            stems.add(path.stem.casefold())
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                im.load()
                im = ImageOps.exif_transpose(im)
                width, height = im.size
                # Detect re-encoded copies too, so copies cannot cross splits.
                digest = hashlib.sha256(str(im.size).encode() + im.convert("RGB").tobytes()).hexdigest()
            if digest in hashes:
                raise ValueError("duplicate decoded image content")
            hashes.add(digest)
            ann_path = annotations / f"{path.stem}.json"
            ann = load_annotation(ann_path, path)
            if ann.image is not None and ann.image != path.name:
                raise ValueError("annotation image name differs")
            if (ann.width is not None and ann.width != width) or (ann.height is not None and ann.height != height):
                raise ValueError("annotation dimensions differ from image")
            points = [(p.x, p.y) for p in ann.points]
            if len(set(points)) != len(points):
                raise ValueError("duplicate annotation points")
            if any(not (0 <= x < width and 0 <= y < height) for x, y in points):
                raise ValueError("point outside image bounds")
            records.append(dict(id=path.stem, image=str(path.resolve()), annotation=str(ann_path.resolve()),
                                width=width, height=height, points=points, sha256=digest,
                                annotation_sha256=hashlib.sha256(ann_path.read_bytes()).hexdigest()))
        except (ValueError, OSError, SyntaxError) as exc:
            errors.append(f"{path.name}: {exc}")
    if annotations.is_dir() and image_id is None:
        for path in annotations.glob("*.json"):
            if path.stem.casefold() not in stems:
                errors.append(f"{path.name}: annotation has no source image")
    if errors:
        raise ValueError("Dataset validation failed:\n" + "\n".join(errors))
    return records
