import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    mode: Literal["holdout", "overfit"] = "holdout"
    images: Path = Path("data/images")
    annotations: Path = Path("data/annotations")
    processed: Path = Path("data/processed/default")
    splits: Path = Path("data/splits/default.json")
    box_size: float = Field(default=32, gt=0)
    tile_size: int = Field(default=640, gt=0)
    overlap: int = Field(default=128, ge=0)
    negative_fraction: float = Field(default=0.25, ge=0, le=1)
    val_fraction: float = Field(default=0.25, gt=0, lt=1)
    seed: int = Field(default=42, ge=0)
    model: str = "yolo11n.pt"
    imgsz: int = Field(default=640, ge=32, multiple_of=32)
    batch: int = Field(default=2, gt=0)
    epochs: int = Field(default=20, gt=0)
    workers: int = Field(default=0, ge=0)
    confidence: float = Field(default=0.25, gt=0, le=1)
    merge_iou: float = Field(default=0.3, gt=0, le=1)
    match_radius: float = Field(default=16, gt=0)
    max_det: int = Field(default=3000, gt=0)

    @model_validator(mode="after")
    def geometry(self):
        if self.overlap >= self.tile_size:
            raise ValueError("overlap must be less than tile_size")
        return self


def read_config(path):
    return Config.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
