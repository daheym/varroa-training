from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.config import write_json
from src.dataset.geometry import merge_detections, tile_windows, to_original


def predict_image(model, image_path, config):
    detections = []
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        for window in tile_windows(width, height, config.tile_size, config.overlap):
            x, y, w, h = window
            tile = image.crop((x, y, x + w, y + h))
            result = model.predict(source=tile, device="cpu", imgsz=config.imgsz, conf=config.confidence,
                                   iou=config.merge_iou, max_det=config.max_det, classes=[0], verbose=False)[0]
            if result.boxes is None:
                continue
            for box, confidence in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                a, b, c, d = box
                clipped = (max(0, a), max(0, b), min(w, c), min(h, d))
                if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                    continue
                detections.append({"box": to_original(clipped, window), "confidence": confidence})
    merged = merge_detections(detections, config.merge_iou)
    return {"image": str(Path(image_path).resolve()), "width": width, "height": height,
            "count": len(merged), "detections": merged, "tile_detection_count": len(detections),
            "coordinate_space": "original_exif_oriented", "config": config.model_dump(mode="json")}


def save_prediction(result, output):
    output = Path(output)
    write_json(output.with_suffix(".json"), result)
    with Image.open(result["image"]) as source:
        canvas = cv2.cvtColor(np.asarray(ImageOps.exif_transpose(source).convert("RGB")), cv2.COLOR_RGB2BGR)
    for detection in result["detections"]:
        x1, y1, x2, y2 = [round(v) for v in detection["box"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(canvas, f"{detection['confidence']:.2f}", (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 180, 0), 1)
    cv2.putText(canvas, f"Varroa count: {result['count']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2)
    Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(output.with_suffix(".png"))
