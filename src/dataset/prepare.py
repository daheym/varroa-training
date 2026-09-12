import math
import hashlib
import random
from pathlib import Path

from PIL import Image, ImageOps
import yaml

from src.config import write_json
from src.dataset.geometry import clip_to_tile, point_to_box, tile_windows, to_yolo
from src.dataset.validation import validate_dataset
from src.dataset.perspective import normalize_image, resolve_metadata


def split_originals(records, val_fraction=0.25, seed=42, held_out=None):
    ids = sorted(r["id"] for r in records)
    if len(ids) < 2 or len(ids) != len(set(ids)):
        raise ValueError("Need at least two unique original images")
    if held_out is not None:
        if held_out not in ids:
            raise ValueError("Unknown held-out image")
        val = [held_out]
    else:
        if not 0 < val_fraction < 1:
            raise ValueError("val_fraction must be in (0, 1)")
        random.Random(seed).shuffle(ids)
        val = ids[:min(len(ids) - 1, max(1, round(len(ids) * val_fraction)))]
    return {"train": sorted(set(ids) - set(val)), "val": sorted(val)}


def check_splits(splits, records, mode="holdout"):
    train, val = splits["train"], splits["val"]
    if mode == "overfit":
        if len(records) != 1 or train != [records[0]["id"]] or val:
            raise ValueError("Overfit mode requires exactly one training original and no validation originals")
        return
    if mode != "holdout":
        raise ValueError(f"Unknown dataset mode: {mode}")
    if not train or not val or set(train) & set(val):
        raise ValueError("Empty split or train/validation leakage")
    if len(train + val) != len(set(train + val)) or set(train + val) != {r["id"] for r in records}:
        raise ValueError("Split must contain every original exactly once")


def prepare(config, held_out=None, records=None):
    records = records if records is not None else validate_dataset(config.images, config.annotations, config=config)
    for record in records:
        if not record.get('perspective') and not config.allow_legacy_original:
            raise ValueError('Physical perspective region required')
        if record.get('perspective') and record['perspective']['pixels_per_mm'] != config.pixels_per_mm:
            raise ValueError('pixels_per_mm differs from prepared records')
        if config.box_size_mm is not None and not record.get('perspective'):
            raise ValueError('Physical box size requires physical normalization')
        if record.get('perspective'):
            p = record['perspective']
            if (p['quality_policy'], p['min_source_scale_ratio']) != (config.quality_policy, config.min_source_scale_ratio):
                record['perspective'] = resolve_metadata(p, record['original_width'], record['original_height'],
                    config.pixels_per_mm, config.min_source_scale_ratio, config.quality_policy)
    if config.mode == "overfit":
        if held_out is not None:
            raise ValueError("Overfit mode cannot hold out an image")
        splits = {"train": [r["id"] for r in records], "val": []}
    else:
        splits = split_originals(records, config.val_fraction, config.seed, held_out)
    check_splits(splits, records, config.mode)
    if not any(r["points"] for r in records if r["id"] in splits["train"]):
        raise ValueError("Training split has no positive annotations")
    root = config.processed.resolve()
    # Never mix old tiles with a new split or configuration.
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Output is not empty: {root}. Choose a fresh processed path.")
    if config.splits.exists():
        raise ValueError(f"Split file already exists: {config.splits}. Choose a fresh path.")
    write_json(config.splits, {"mode": config.mode, "seed": config.seed, "splits": splits, "sources": records})
    for split in splits:
        for kind in ("images", "labels"):
            (root / kind / split).mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    tiles = []
    for record in records:
        split = "val" if record["id"] in splits["val"] else "train"
        box_size = config.box_size_mm * config.pixels_per_mm if config.box_size_mm is not None else config.box_size
        boxes = [point_to_box(x, y, box_size, record["width"], record["height"]) for x, y in record["points"]]
        windows = tile_windows(record["width"], record["height"], config.tile_size, config.overlap)
        candidates = []
        for index, window in enumerate(windows):
            # Label all intersecting boxes, even if the point is just outside the tile.
            labels = [clipped for box in boxes if (clipped := clip_to_tile(box, window)) is not None]
            candidates.append((index, window, labels))
        negatives = [i for i, _, labels in candidates if not labels]
        fraction = config.negative_fraction if split == "train" else 1.0
        selected = set(rng.sample(negatives, math.ceil(len(negatives) * fraction)))
        with Image.open(record["image"]) as source:
            source = ImageOps.exif_transpose(source).convert("RGB")
            digest = hashlib.sha256(str(source.size).encode() + source.tobytes()).hexdigest()
            if digest != record["sha256"]:
                raise ValueError(f"Source image changed during preparation: {record['id']}")
            if record.get('perspective'):
                source = normalize_image(source, record['perspective'])
            for index, window, labels in candidates:
                if not labels and index not in selected:
                    continue
                x, y, w, h = window
                name = f"{record['id']}__{index:06d}"
                image_rel = f"images/{split}/{name}.png"
                label_rel = f"labels/{split}/{name}.txt"
                crop = source.crop((x, y, x + w, y + h))
                if record.get('perspective'):
                    padded = Image.new('RGB', (config.tile_size, config.tile_size))
                    padded.paste(crop, (0, 0))
                    crop = padded
                tile_w, tile_h = crop.size
                crop.save(root / image_rel)
                rows = ["0 " + " ".join(f"{v:.10f}" for v in to_yolo(box, tile_w, tile_h)) for box in labels]
                (root / label_rel).write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
                tiles.append(dict(image=image_rel, label=label_rel, source_id=record["id"],
                                  source_image=record["image"], split=split, x=x, y=y, width=tile_w, height=tile_h, valid_width=w, valid_height=h,
                                  coordinate_space=record.get('coordinate_space', 'original'),
                                  positive=bool(labels), box_count=len(labels)))
    if not any(t["split"] == "train" and t["positive"] for t in tiles):
        raise ValueError("Training split has no positive tiles; revise source split or annotations")
    manifest = dict(mode=config.mode, config=config.model_dump(mode="json"), splits=splits, sources=records, tiles=tiles)
    write_json(root / "manifest.json", manifest)
    (root / "dataset.yaml").write_text(yaml.safe_dump({"path": root.as_posix(), "train": "images/train",
                                                      # YOLO requires a val path even when val=False.
                                                      "val": "images/train" if config.mode == "overfit" else "images/val",
                                                      "names": {0: "varroa"}}), encoding="utf-8")
    return manifest
