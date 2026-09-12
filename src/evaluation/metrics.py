import math

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_detections(points, detections, radius):
    """Maximum-cardinality, minimum-distance one-to-one point/box-center matching."""
    if radius <= 0:
        raise ValueError("Match radius must be positive")
    centers = [((d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2) for d in detections]
    matches = []
    if points and centers:
        distances = np.linalg.norm(np.asarray(points)[:, None, :] - np.asarray(centers)[None, :, :], axis=2)
        penalty = (min(len(points), len(centers)) + 1) * (radius + 1)
        cost = np.where(distances <= radius, distances, penalty)
        rows, cols = linear_sum_assignment(cost)
        matches = [(int(row), int(col)) for row, col in zip(rows, cols) if distances[row, col] <= radius]
    matched_points = {row for row, _ in matches}
    matched_detections = {col for _, col in matches}
    return {"matches": matches,
            "missed": [i for i in range(len(points)) if i not in matched_points],
            "extra": [i for i in range(len(detections)) if i not in matched_detections]}


def image_metrics(points, detections, radius):
    matching = match_detections(points, detections, radius)
    tp = len(matching["matches"])
    fp, fn = len(matching["extra"]), len(matching["missed"])
    return {**scores(tp, fp, fn), "true_count": len(points), "predicted_count": len(detections),
            "count_error": len(detections) - len(points), "absolute_count_error": abs(len(detections) - len(points))}


def scores(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall,
                f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0)


def summarize(rows):
    if not rows:
        raise ValueError("No images to evaluate")
    errors = [r["count_error"] for r in rows]
    return {**scores(*(sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))),
            "images": len(rows), "count_mae": sum(abs(e) for e in errors) / len(errors),
            "count_rmse": math.sqrt(sum(e * e for e in errors) / len(errors)),
            "count_bias": sum(errors) / len(errors)}
