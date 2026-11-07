"""Non-destructive quality review shared with the local labeling application."""
import hashlib
import json


def geometry_hash(record):
    perspective = record.get("perspective") or {}
    geometry = [record["coordinate_space"], record["width"], record["height"], perspective.get("homography")]
    return hashlib.sha256(json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def review_key(record, x, y):
    identity = [record["id"], record["sha256"], geometry_hash(record), round(float(x), 6), round(float(y), 6)]
    return hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()


def load_exclusions(record, annotations):
    path = annotations.resolve().parent / "quality_reviews" / f"{record['id']}.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("regions"), dict):
        raise ValueError(f"Invalid quality review: {path}")
    current = {review_key(record, x, y) for x, y in record["points"]}
    regions = []
    for key, region in payload["regions"].items():
        if key not in current:
            continue  # Edits to the image, geometry or point invalidate old decisions.
        if region.get("reason") not in ("not_mite", "poor_quality") or region.get("crop_size") != 96:
            raise ValueError(f"Invalid exclusion region: {path}")
        x, y = region["x"], region["y"]
        if review_key(record, x, y) != key:
            raise ValueError(f"Invalid exclusion identity: {path}")
        regions.append(region)
    return regions


def intersects_exclusion(window, regions):
    x, y, width, height = window
    return any(x < r["x"] + 48 and x + width > r["x"] - 48
               and y < r["y"] + 48 and y + height > r["y"] - 48 for r in regions)
