"""Visual diagnostics from saved predictions; no model inference or retraining."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps

from src.config import write_json
from src.dataset.perspective import normalize_image
from src.evaluation.metrics import match_detections


GREEN, RED, ORANGE, BLUE = "#00c77b", "#ff3855", "#ffae00", "#26c6ff"


def draw_detections(ax, image, points, detections, matching, window=None):
    x, y, width, height = window or (0, 0, image.width, image.height)
    ax.imshow(image.crop((x, y, x + width, y + height)))
    extra = set(matching["extra"])
    for index, detection in enumerate(detections):
        a, b, c, d = detection["box"]
        if c < x or a > x + width or d < y or b > y + height:
            continue
        ax.add_patch(Rectangle((a - x, b - y), c - a, d - b, fill=False,
                               edgecolor=ORANGE if index in extra else GREEN, linewidth=1.3))
    for index, (a, b) in enumerate(points):
        if x <= a < x + width and y <= b < y + height:
            missed = index in matching["missed"]
            ax.plot(a - x, b - y, marker="x" if missed else "+", color=RED if missed else BLUE,
                    markersize=11 if missed else 4, markeredgewidth=2 if missed else .8)
            if missed:
                ax.add_patch(plt.Circle((a - x, b - y), 18, fill=False, edgecolor=RED, linewidth=1.5))
    ax.set_xlim(-.5, width - .5)
    ax.set_ylim(height - .5, -.5)
    ax.axis("off")


def learning_curves(csv_path, output, scope):
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    if not rows:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout="constrained")
    for ax, key, title, color in zip(axes, ("train/box_loss", "train/cls_loss", "train/dfl_loss"),
                                    ("Box localization loss", "Classification loss", "Box refinement loss (DFL)"),
                                    ("#176b52", "#2263b5", "#8552a1")):
        values = [r[key] for r in rows]
        ax.plot([r["epoch"] for r in rows], values, color=color, linewidth=2)
        ax.set(title=title, xlabel="Epoch", ylabel="Training loss (lower is better)")
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(.98, .95, f"{values[0]:.3f} -> {values[-1]:.3f}", transform=ax.transAxes, ha="right", va="top")
    fig.suptitle(f"{int(rows[-1]['epoch'])} epochs | {scope}\nTraining loss only; no unseen-image performance curve", fontsize=14)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)
    return output


def visualize_run(run):
    run = Path(run).resolve()
    report = json.loads((run / "evaluation.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "dataset_manifest.json").read_text(encoding="utf-8"))
    output = run / "visualizations"
    output.mkdir(exist_ok=True)
    scope = report.get("evaluation_scope", "held_out_originals")
    scope_label = "Training-image reconstruction" if scope == "training_image_reconstruction" else "Held-out originals"
    sources = {r["id"]: r for r in manifest["sources"]}
    artifacts, errors = [], []
    for result in report["per_image"]:
        record = sources[result["source_id"]]
        with Image.open(record["image"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Source image changed since training: {record['id']}")
        if record.get("perspective"):
            image = normalize_image(image, record["perspective"])
        points, detections = record["points"], result["detections"]
        matching = match_detections(points, detections, report["match_radius"])
        if (len(matching["matches"]), len(matching["extra"]), len(matching["missed"])) != (result["tp"], result["fp"], result["fn"]):
            raise ValueError("Saved metrics disagree with reconstructed matching")
        title = f"{scope_label} | {len(detections)} predicted / {len(points)} annotated\n{result['tp']} matched | {result['fn']} missed | {result['fp']} extra"
        legend = "Green box = matched prediction   |   Cyan + = annotation   |   Red x/circle = missed annotation   |   Orange box = extra prediction"
        fig, ax = plt.subplots(figsize=(12, 12 * image.height / image.width + 1.2))
        draw_detections(ax, image, points, detections, matching)
        fig.suptitle(title, fontsize=14, y=.98)
        fig.text(.5, .025, legend, ha="center", fontsize=8)
        fig.subplots_adjust(left=.01, right=.99, bottom=.055, top=.91)
        overlay = output / f"{record['id']}-overlay.png"
        fig.savefig(overlay, dpi=180, facecolor="white")
        plt.close(fig)
        artifacts.append(overlay)
        image_errors = []
        for index in matching["missed"]:
            image_errors.append({"type": "missed_annotation", "annotation_index": index,
                                 "x": points[index][0], "y": points[index][1]})
        for index in matching["extra"]:
            box = detections[index]["box"]
            image_errors.append({"type": "extra_prediction", "detection_index": index,
                                 "x": (box[0] + box[2]) / 2, "y": (box[1] + box[3]) / 2,
                                 "confidence": detections[index]["confidence"]})
        errors.extend({"source_id": record["id"], **error} for error in image_errors)
        # Paginate dense-error runs instead of constructing unbounded-size figures.
        for start in range(0, len(image_errors), 6):
            page = image_errors[start:start + 6]
            columns = min(3, len(page))
            fig, axes = plt.subplots((len(page) + columns - 1) // columns, columns,
                                     figsize=(columns * 4, 4.5 * ((len(page) + columns - 1) // columns)), squeeze=False)
            for ax in axes.flat:
                ax.axis("off")
            for ax, error in zip(axes.flat, page):
                w, h = min(160, image.width), min(160, image.height)
                x = int(max(0, min(image.width - w, error["x"] - w / 2)))
                y = int(max(0, min(image.height - h, error["y"] - h / 2)))
                draw_detections(ax, image, points, detections, matching, (x, y, w, h))
                label = "Missed annotation" if error["type"] == "missed_annotation" else "Extra prediction"
                detail = f"x={error['x']:.1f}, y={error['y']:.1f}"
                if "confidence" in error:
                    detail += f" | confidence={error['confidence']:.3f}"
                ax.set_title(f"{label}\n{detail}", fontsize=10, color=RED if label.startswith("Missed") else "#946000")
            fig.suptitle(f"{scope_label}: inspect discrepancies against saved annotations", fontsize=12, y=.99)
            fig.text(.5, .025, legend, ha="center", fontsize=7)
            fig.tight_layout(rect=(0, .06, 1, .9))
            path = output / f"{record['id']}-errors-{start // 6 + 1}.png"
            fig.savefig(path, dpi=180, facecolor="white")
            plt.close(fig)
            artifacts.append(path)
    history = run / "yolo" / "results.csv"
    if history.exists():
        curve = learning_curves(history, output / "learning-curves.png", scope_label)
        if curve is not None:
            artifacts.append(curve)
    write_json(output / "errors.json", {"evaluation_scope": scope, "match_radius": report["match_radius"],
                                        "errors": errors})
    return artifacts
