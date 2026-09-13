"""Export merged inference results in original-image pixels."""
import uuid
from datetime import datetime, timezone
from pathlib import Path
from src.interchange import load_bundle, dump, AnnotationFile, Manifest
from src.inference.predict import predict_image
from src.dataset.perspective import transform_points

def prediction_document(doc, result, model_version):
    annotations = []
    for detection in result["detections"]:
        a, b, c, d = detection["box"]
        x, y = (a+c)/2, (b+d)/2
        if result.get("perspective"):
            x, y = transform_points([(x,y)], result["perspective"]["inverse_homography"])[0]
            polygon = detection["original_polygon"]
            a, b = min(p[0] for p in polygon), min(p[1] for p in polygon)
            c, d = max(p[0] for p in polygon), max(p[1] for p in polygon)
        a, b = max(0., a), max(0., b)
        c, d = min(float(doc.image.width), c), min(float(doc.image.height), d)
        if c <= a or d <= b or not (0 <= x < doc.image.width and 0 <= y < doc.image.height):
            raise ValueError("Prediction lies outside original image")
        annotations.append({"id": "pred_" + uuid.uuid4().hex, "class": "varroa", "x": x, "y": y,
            "bbox": {"x": a, "y": b, "width": c-a, "height": d-b}, "source": "model",
            "confidence": detection["confidence"], "model_version": model_version, "reviewed": False})
    return {"schema_version": "1.0", "image": doc.image.model_dump(), "perspective": doc.perspective, "annotations": annotations}

def predict_dataset(dataset, model, output, config, model_version):
    manifest, docs = load_bundle(dataset)
    output = Path(output)
    if output.exists():
        raise ValueError("Prediction output already exists; choose a new directory")
    selected = [(e,d) for e,d in zip(manifest.images, docs) if e.id not in manifest.reviewed_image_ids]
    if not selected:
        raise ValueError("No unreviewed images in dataset")
    # Finish inference and validation before publishing a manifest.
    results = []
    from src.interchange import validate
    for entry, doc in selected:
        result = predict_image(model, Path(dataset)/"images"/entry.filename, config, doc.perspective)
        payload = prediction_document(doc, result, model_version)
        validate(payload, AnnotationFile)
        results.append((entry,payload))
    for entry, payload in results:
        dump(output/"annotations"/entry.annotation_file, payload, AnnotationFile)
    payload = manifest.model_dump()
    payload.update(created_at=datetime.now(timezone.utc).isoformat(), purpose="predictions", reviewed_image_ids=[],
        images=[e.model_dump() for e,_ in results])
    dump(output/"manifest.json", payload, Manifest)
