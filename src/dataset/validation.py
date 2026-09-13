import hashlib
import json
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field
from src.config import Config
from src.dataset.perspective import resolve_metadata, transform_points


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
    coordinate_space: Literal['original', 'canonical'] = 'original'
    perspective: dict | None = None


class LabelingPoint(Point):
    id: str
    className: Literal["varroa"]
    source: Literal["human", "model"]
    confidence: float | None = Field(default=None, ge=0, le=1)
    reviewed: bool
    model_version: str | None = None
    bbox: dict | None = None


class LabelingDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["1.0"] = "1.0"
    review_complete: bool = False
    image_id: str
    filename: str
    original_width: int = Field(gt=0)
    original_height: int = Field(gt=0)
    coordinate_space: Literal["original", "canonical"]
    perspective: dict | None = None
    dataset_version: str
    model_version: str | None = None
    annotations: list[LabelingPoint]
    rejected_predictions: list[LabelingPoint] = Field(default_factory=list)
    effort: dict = Field(default_factory=dict)


def load_annotation(path, image_path):
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    if isinstance(payload, dict) and "schema_version" in payload and "image" in payload:
        from src.interchange import validate, AnnotationFile
        doc = validate(payload, AnnotationFile)
        if any(not a.reviewed for a in doc.annotations):
            raise ValueError("Unreviewed predictions cannot become ground truth")
        return Annotation(image=doc.image.filename, width=doc.image.width, height=doc.image.height,
                          perspective=doc.perspective, points=[Point(x=a.x, y=a.y) for a in doc.annotations])
    if isinstance(payload, dict) and "image_id" in payload:
        doc = LabelingDocument.model_validate_json(raw)
        if doc.image_id != path.stem or doc.image_id != image_path.stem:
            raise ValueError("image_id must match annotation and image stems")
        if any(not p.reviewed for p in doc.annotations):
            raise ValueError("Review all annotations before using them as ground truth")
        if len({p.id for p in doc.annotations}) != len(doc.annotations):
            raise ValueError("duplicate annotation IDs")
        return Annotation(width=doc.original_width, height=doc.original_height,
                          coordinate_space=doc.coordinate_space, perspective=doc.perspective,
                          points=[Point(x=p.x, y=p.y) for p in doc.annotations])
    return Annotation.model_validate_json(raw)


def validate_dataset(images: Path, annotations: Path, image_id: str | None = None, config=None):
    """Collect all errors; never silently treat missing annotations as negatives."""
    config = config or Config()
    records, errors = [], []
    entries = None
    if (images.parent / "manifest.json").exists():
        from src.interchange import load_bundle
        manifest, _ = load_bundle(images.parent, training=True)
        entries = {e.filename: e for e in manifest.images}
    files = sorted(p for p in images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) if images.is_dir() else []
    if entries is not None:
        files = [images / filename for filename in entries]
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
            ann_path = annotations / (entries[path.name].annotation_file if entries else f"{path.stem}.json")
            ann = load_annotation(ann_path, path)
            if ann.image is not None and ann.image != path.name:
                raise ValueError("annotation image name differs")
            if (ann.width is not None and ann.width != width) or (ann.height is not None and ann.height != height):
                raise ValueError("annotation dimensions differ from image")
            points = [(p.x, p.y) for p in ann.points]
            original_width, original_height = width, height
            perspective = None
            if ann.perspective is not None:
                perspective = resolve_metadata(ann.perspective, width, height, config.pixels_per_mm,
                    config.min_source_scale_ratio, config.quality_policy)
                if ann.coordinate_space == 'original':
                    points = [tuple(p) for p in transform_points(points, perspective['homography'])]
                else:
                    # Canonical points must refer to this exact geometry, never guessed.
                    import numpy as np
                    if (ann.perspective.get('version') != 1 or
                        ann.perspective.get('output_width') != perspective['output_width'] or
                        ann.perspective.get('output_height') != perspective['output_height'] or
                        not isinstance(ann.perspective.get('homography'), list) or
                        np.shape(ann.perspective['homography']) != (3,3) or
                        not np.allclose(ann.perspective['homography'], perspective['homography'], rtol=1e-9, atol=1e-9)):
                        raise ValueError('Canonical annotation transform differs; migrate explicitly')
                width, height = perspective['output_width'], perspective['output_height']
            elif ann.coordinate_space == 'canonical' or not config.allow_legacy_original:
                raise ValueError('Physical perspective region required; legacy original compatibility must be explicit')
            if len(set(points)) != len(points):
                raise ValueError("duplicate annotation points")
            if any(not (0 <= x < width and 0 <= y < height) for x, y in points):
                raise ValueError("point outside image bounds")
            records.append(dict(id=entries[path.name].id if entries else path.stem, image=str(path.resolve()), annotation=str(ann_path.resolve()),
                                original_width=original_width, original_height=original_height,
                                perspective=perspective, coordinate_space='canonical' if perspective else 'original',
                                width=width, height=height, points=points, sha256=digest,
                                annotation_sha256=hashlib.sha256(ann_path.read_bytes()).hexdigest()))
        except (ValueError, OSError, SyntaxError) as exc:
            errors.append(f"{path.name}: {exc}")
    if annotations.is_dir() and image_id is None and entries is None:
        for path in annotations.glob("*.json"):
            if path.stem.casefold() not in stems:
                errors.append(f"{path.name}: annotation has no source image")
    if errors:
        raise ValueError("Dataset validation failed:\n" + "\n".join(errors))
    return records
