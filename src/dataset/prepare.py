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


def split_tiles(tiles, val_fraction=0.25, seed=42):
    """Split generated tile candidates, not their source images."""
    ids = sorted(tile["id"] for tile in tiles)
    if len(ids) < 2 or len(ids) != len(set(ids)):
        raise ValueError("Need at least two unique generated tiles")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")
    random.Random(seed).shuffle(ids)
    val_count = min(len(ids) - 1, max(1, round(len(ids) * val_fraction)))
    val = set(ids[:val_count])

    # Training must retain positive supervision. Preserve the requested split
    # size by swapping with a training negative when possible.
    positive = {tile["id"] for tile in tiles if tile["positive"]}
    train = set(ids) - val
    if positive and not train & positive:
        positive_id = next(tile_id for tile_id in ids if tile_id in positive)
        replacement = next((tile_id for tile_id in ids if tile_id in train and tile_id not in positive), None)
        val.remove(positive_id)
        train.add(positive_id)
        if replacement is not None:
            train.remove(replacement)
            val.add(replacement)
    return {"train": sorted(train), "val": sorted(val)}


def check_splits(splits, items, mode="holdout"):
    train, val = splits["train"], splits["val"]
    if mode == "overfit":
        if val:
            raise ValueError("Overfit mode requires no validation tiles")
        expected = {item["id"] for item in items}
        if set(train) != expected or len(train) != len(expected):
            raise ValueError("Overfit split must contain every generated tile exactly once")
        return
    if mode != "holdout":
        raise ValueError(f"Unknown dataset mode: {mode}")
    if not train or not val or set(train) & set(val):
        raise ValueError("Empty split or train/validation leakage")
    if len(train + val) != len(set(train + val)) or set(train + val) != {item["id"] for item in items}:
        raise ValueError("Split must contain every generated tile exactly once")


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
        # Overview decisions edit labels directly. Legacy quality flags never
        # remove context tiles or neighboring valid annotations from a dataset.
        record.pop("quality_exclusions", None)
    root = config.processed.resolve()
    # Never mix old tiles with a new split or configuration.
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Output is not empty: {root}. Choose a fresh processed path.")
    if config.splits.exists():
        raise ValueError(f"Split file already exists: {config.splits}. Choose a fresh path.")
    rng = random.Random(config.seed)
    candidates = []
    for record in records:
        box_size = config.box_size_mm * config.pixels_per_mm if config.box_size_mm is not None else config.box_size
        boxes = [point_to_box(x, y, box_size, record["width"], record["height"]) for x, y in record["points"]]
        windows = tile_windows(record["width"], record["height"], config.tile_size, config.overlap)
        for index, window in enumerate(windows):
            # Label all intersecting boxes, even if the point is just outside the tile.
            labels = [clipped for box in boxes if (clipped := clip_to_tile(box, window)) is not None]
            candidates.append({"id": f"{record['id']}__{index:06d}", "index": index,
                               "record": record, "window": window, "labels": labels,
                               "positive": bool(labels)})
    if not any(tile["positive"] for tile in candidates):
        raise ValueError("Training split has no positive annotations")

    if config.mode == "overfit":
        if held_out is not None:
            raise ValueError("Overfit mode cannot hold out an image")
        if len(records) != 1:
            raise ValueError("Overfit mode requires exactly one original image")
        splits = {"train": sorted(tile["id"] for tile in candidates), "val": []}
        split_strategy = "all_tiles_train"
        held_out_sources = []
    elif held_out is not None:
        source_splits = split_originals(records, config.val_fraction, config.seed, held_out)
        held_out_sources = source_splits["val"]
        splits = {
            split: sorted(tile["id"] for tile in candidates if tile["record"]["id"] in source_splits[split])
            for split in ("train", "val")
        }
        split_strategy = "held_out_source"
    else:
        splits = split_tiles(candidates, config.val_fraction, config.seed)
        split_strategy = "generated_tiles"
        held_out_sources = []

    split_by_tile = {tile_id: split for split, tile_ids in splits.items() for tile_id in tile_ids}
    retained = []
    for record in records:
        training_negatives = [tile for tile in candidates
                              if tile["record"]["id"] == record["id"]
                              and split_by_tile[tile["id"]] == "train" and not tile["positive"]]
        keep_count = math.ceil(len(training_negatives) * config.negative_fraction)
        kept_negative_ids = {tile["id"] for tile in rng.sample(training_negatives, keep_count)}
        retained.extend(tile for tile in candidates if tile["record"]["id"] == record["id"]
                        and (split_by_tile[tile["id"]] == "val" or tile["positive"]
                             or tile["id"] in kept_negative_ids))
    retained_ids = {tile["id"] for tile in retained}
    splits = {split: [tile_id for tile_id in tile_ids if tile_id in retained_ids]
              for split, tile_ids in splits.items()}
    check_splits(splits, retained, config.mode)
    if not any(tile["positive"] and split_by_tile[tile["id"]] == "train" for tile in retained):
        raise ValueError("Training split has no positive tiles; revise tile split or annotations")

    for split in ("train", "val"):
        for kind in ("images", "labels"):
            (root / kind / split).mkdir(parents=True, exist_ok=True)
    split_manifest = {"mode": config.mode, "seed": config.seed, "split_unit": "tile",
                      "split_strategy": split_strategy, "held_out_sources": held_out_sources,
                      "splits": splits, "sources": records}
    write_json(config.splits, split_manifest)

    tiles = []
    by_source = {record["id"]: [] for record in records}
    for tile in retained:
        by_source[tile["record"]["id"]].append(tile)
    for record in records:
        with Image.open(record["image"]) as source:
            source = ImageOps.exif_transpose(source).convert("RGB")
            digest = hashlib.sha256(str(source.size).encode() + source.tobytes()).hexdigest()
            if digest != record["sha256"]:
                raise ValueError(f"Source image changed during preparation: {record['id']}")
            if record.get('perspective'):
                source = normalize_image(source, record['perspective'])
            for candidate in by_source[record["id"]]:
                index, window, labels = candidate["index"], candidate["window"], candidate["labels"]
                split = split_by_tile[candidate["id"]]
                x, y, w, h = window
                name = candidate["id"]
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
                tiles.append(dict(id=name, image=image_rel, label=label_rel, source_id=record["id"],
                                  source_image=record["image"], split=split, x=x, y=y, width=tile_w, height=tile_h, valid_width=w, valid_height=h,
                                  coordinate_space=record.get('coordinate_space', 'original'),
                                  positive=bool(labels), box_count=len(labels)))
    manifest = dict(mode=config.mode, config=config.model_dump(mode="json"), split_unit="tile",
                    split_strategy=split_strategy, held_out_sources=held_out_sources,
                    splits=splits, sources=records, tiles=tiles)
    write_json(root / "manifest.json", manifest)
    (root / "dataset.yaml").write_text(yaml.safe_dump({"path": root.as_posix(), "train": "images/train",
                                                      # YOLO requires a val path even when val=False.
                                                      "val": "images/train" if config.mode == "overfit" else "images/val",
                                                      "names": {0: "varroa"}}), encoding="utf-8")
    return manifest
