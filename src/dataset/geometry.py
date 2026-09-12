def point_to_box(x, y, size, width, height):
    if size <= 0 or not (0 <= x < width and 0 <= y < height):
        raise ValueError("Invalid point or box size")
    half = size / 2
    return (max(0, x - half), max(0, y - half), min(width, x + half), min(height, y + half))


def to_yolo(box, width, height):
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Invalid box")
    return ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height)


def from_yolo(box, width, height):
    x, y, w, h = box
    return ((x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height)


def tile_windows(width, height, size, overlap):
    if min(width, height, size) <= 0 or not 0 <= overlap < size:
        raise ValueError("Invalid tiling dimensions")

    def starts(length):
        last = max(0, length - size)
        return sorted(set([*range(0, last + 1, size - overlap), last]))

    return [(x, y, min(size, width - x), min(size, height - y)) for y in starts(height) for x in starts(width)]


def clip_to_tile(box, window):
    x, y, w, h = window
    a, b, c, d = box
    result = (max(a, x) - x, max(b, y) - y, min(c, x + w) - x, min(d, y + h) - y)
    return result if result[2] > result[0] and result[3] > result[1] else None


def to_original(box, window):
    x, y, _, _ = window
    a, b, c, d = box
    return (a + x, b + y, c + x, d + y)


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def merge_detections(detections, threshold=0.3):
    """Class-agnostic greedy NMS; detections are {box: xyxy, confidence: float}."""
    if not 0 < threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    kept = []
    for detection in sorted(detections, key=lambda d: d["confidence"], reverse=True):
        if all(iou(detection["box"], other["box"]) < threshold for other in kept):
            kept.append(detection)
    return kept
