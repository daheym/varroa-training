from pathlib import Path

from src.config import write_json
from src.training.train import sha256


def export_onnx(weights, imgsz=640):
    from ultralytics import YOLO
    import onnx

    artifact = Path(YOLO(str(weights)).export(format="onnx", device="cpu", imgsz=imgsz, batch=1,
                                            dynamic=False, simplify=False, opset=17, nms=False))
    onnx.checker.check_model(str(artifact))
    graph = onnx.load(str(artifact)).graph

    def tensor_info(value):
        return {"name": value.name, "shape": [dim.dim_value or dim.dim_param for dim in value.type.tensor_type.shape.dim]}

    write_json(artifact.with_suffix(".json"), {"weights": str(Path(weights).resolve()),
               "weights_sha256": sha256(weights), "onnx_sha256": sha256(artifact), "imgsz": imgsz,
               "input": "RGB float32 NCHW / 255, aspect-preserving letterbox to imgsz",
               "output": "Raw YOLO detections; decode and perform NMS in the client",
               "classes": {"0": "varroa"}, "opset": 17, "nms": False,
               "inputs": [tensor_info(value) for value in graph.input],
               "outputs": [tensor_info(value) for value in graph.output]})
    return artifact
