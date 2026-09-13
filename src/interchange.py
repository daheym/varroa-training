"""Version 1 file contract. No dependency on the annotator implementation."""
import json
from pathlib import Path
from typing import Literal
from datetime import datetime
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, model_validator
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError as SchemaError

class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, populate_by_name=True)

class Box(Strict):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)

class Mite(Strict):
    id: str = Field(min_length=1)
    class_: Literal["varroa"] = Field(alias="class")
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    bbox: Box | None = None
    source: Literal["human", "model"]
    confidence: float | None = Field(ge=0, le=1)
    model_version: str | None
    reviewed: bool

    @model_validator(mode="after")
    def provenance(self):
        if self.source == "model" and (self.confidence is None or not self.model_version):
            raise ValueError("Model annotations require confidence and model_version")
        if self.source == "human" and not self.reviewed:
            raise ValueError("Human annotations must be reviewed")
        return self

class ImageInfo(Strict):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    filename: str = Field(pattern=r"^[^/\\:]+\.(jpg|jpeg|png|JPG|JPEG|PNG)$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)

class AnnotationFile(Strict):
    schema_version: Literal["1.0"]
    image: ImageInfo
    perspective: dict | None
    annotations: list[Mite]

    @model_validator(mode="after")
    def bounds(self):
        if len({a.id for a in self.annotations}) != len(self.annotations):
            raise ValueError("duplicate annotation IDs")
        for a in self.annotations:
            if not (a.x < self.image.width and a.y < self.image.height):
                raise ValueError("annotation coordinates outside original image")
            if a.bbox and (a.bbox.x + a.bbox.width > self.image.width or a.bbox.y + a.bbox.height > self.image.height):
                raise ValueError("bbox outside original image")
        return self

class Entry(Strict):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    filename: str = Field(pattern=r"^[^/\\:]+\.(jpg|jpeg|png|JPG|JPEG|PNG)$")
    annotation_file: str = Field(pattern=r"^[A-Za-z0-9_-]+\.json$")

class Manifest(Strict):
    schema_version: Literal["1.0"]
    dataset_version: str = Field(min_length=1)
    created_at: str
    purpose: Literal["ground_truth", "predictions", "unreviewed"]
    reviewed_image_ids: list[str]
    images: list[Entry]

    @model_validator(mode="after")
    def unique(self):
        for key in ("id", "filename", "annotation_file"):
            values = [getattr(i, key).casefold() for i in self.images]
            if len(values) != len(set(values)):
                raise ValueError("duplicate " + key)
        if len(set(self.reviewed_image_ids)) != len(self.reviewed_image_ids) or not set(self.reviewed_image_ids) <= {i.id for i in self.images}:
            raise ValueError("invalid reviewed_image_ids")
        return self

class ReviewFile(Strict):
    schema_version: Literal["1.0"]
    image_id: str
    model_predictions: int = Field(ge=0)
    accepted_unchanged: int = Field(ge=0)
    corrected: int = Field(ge=0)
    rejected: int = Field(ge=0)
    manually_added: int = Field(ge=0)
    final_reviewed_mite_count: int = Field(ge=0)

SCHEMA = Path(__file__).resolve().parents[1] / "schemas/varroa-annotation-v1.schema.json"

def validate(payload, kind):
    version = payload.get("schema_version")
    if version != "1.0":
        raise ValueError(f"Unsupported schema_version {version!r}; supported version is 1.0")
    schema = json.loads(SCHEMA.read_text("utf-8"))
    try:
        Draft202012Validator({"$ref": f"#/$defs/{kind.__name__}", "$defs": schema["$defs"]}, format_checker=FormatChecker()).validate(payload)
    except SchemaError as exc:
        raise ValueError(f"Invalid {kind.__name__}: {exc.message}") from exc
    return kind.model_validate(payload)

def dump(path, payload, kind):
    validate(payload, kind)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def load_bundle(root, training=False, images=True):
    root = Path(root).resolve()
    manifest = validate(json.loads((root / "manifest.json").read_text("utf-8")), Manifest)
    if training and (manifest.purpose != "ground_truth" or set(manifest.reviewed_image_ids) != {e.id for e in manifest.images}):
        raise ValueError("Training requires explicitly reviewed ground_truth images")
    docs, ids = [], set()
    for entry in manifest.images:
        ann_path = root / "annotations" / entry.annotation_file
        image_path = root / "images" / entry.filename
        for path in (ann_path, image_path):
            if not path.resolve().is_relative_to(root):
                raise ValueError("Bundle path escapes dataset")
        doc = validate(json.loads(ann_path.read_text("utf-8")), AnnotationFile)
        if (doc.image.id, doc.image.filename) != (entry.id, entry.filename):
            raise ValueError("Manifest and annotation image identity differ")
        if training and any(not a.reviewed for a in doc.annotations):
            raise ValueError("Unreviewed predictions cannot become ground truth")
        if manifest.purpose == "predictions" and any(a.source != "model" or a.reviewed for a in doc.annotations):
            raise ValueError("Prediction bundles require unreviewed model annotations")
        for a in doc.annotations:
            if a.id in ids:
                raise ValueError("duplicate annotation IDs across images")
            ids.add(a.id)
        if images:
            with Image.open(image_path) as im:
                oriented = ImageOps.exif_transpose(im)
                oriented.load()
                if oriented.size != (doc.image.width, doc.image.height):
                    raise ValueError("Image dimensions differ")
        docs.append(doc)
    return manifest, docs
