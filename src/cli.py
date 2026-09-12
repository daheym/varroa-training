import argparse
import json
from pathlib import Path

from src.config import read_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="CPU Varroa detection experiments")
    parser.add_argument("--config", default="configs/default.json", help="Paths in config are relative to the working directory")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    commands.add_parser("prepare")
    sub = commands.add_parser("visualize")
    sub.add_argument("--run", required=True, type=Path, help="Experiment directory containing evaluation.json")
    for command in ("train", "loo", "overfit"):
        sub = commands.add_parser(command)
        sub.add_argument("--output", required=True, type=Path, help="New experiment directory")
        if command == "overfit":
            sub.add_argument("--image-id", help="Source filename stem to use when the input folder contains multiple images")
    sub = commands.add_parser("evaluate")
    sub.add_argument("--weights", required=True, type=Path)
    sub.add_argument("--output", required=True, type=Path)
    sub = commands.add_parser("infer")
    sub.add_argument("--weights", required=True, type=Path)
    sub.add_argument("--image", required=True, type=Path)
    sub.add_argument("--output", required=True, type=Path, help="Output filename stem for JSON and PNG")
    sub = commands.add_parser("export")
    sub.add_argument("--weights", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "visualize":
        from src.evaluation.visualize import visualize_run
        for path in visualize_run(args.run):
            print(path)
        return
    config = read_config(args.config)
    if args.command == "validate":
        from src.dataset.validation import validate_dataset
        records = validate_dataset(config.images, config.annotations)
        print(json.dumps({"images": len(records), "points": sum(len(r["points"]) for r in records)}, indent=2))
    elif args.command == "prepare":
        from src.dataset.prepare import prepare
        manifest = prepare(config)
        print(json.dumps({"splits": manifest["splits"], "tiles": len(manifest["tiles"])}, indent=2))
    elif args.command == "train":
        from src.training.train import train
        report = train(config, args.output)
        print(json.dumps({"evaluation_scope": report["evaluation_scope"], **report["aggregate"]}, indent=2))
    elif args.command == "overfit":
        from src.training.overfit import overfit_image
        print("Single-image overfit experiment: reported metrics use the training image and do not measure generalization.")
        report = overfit_image(config, args.output, image_id=args.image_id)
        print(json.dumps({"evaluation_scope": report["evaluation_scope"], **report["aggregate"]}, indent=2))
    elif args.command == "loo":
        from src.evaluation.loo import leave_one_out
        print(json.dumps(leave_one_out(config, args.output)["aggregate"], indent=2))
    elif args.command == "export":
        from src.inference.export import export_onnx
        print(export_onnx(args.weights, config.imgsz))
    else:
        from ultralytics import YOLO
        model = YOLO(str(args.weights))
        if args.command == "evaluate":
            from src.evaluation.evaluate import evaluate
            manifest = json.loads((config.processed / "manifest.json").read_text(encoding="utf-8"))
            report = evaluate(model, manifest, config, args.output)
            print(json.dumps({"evaluation_scope": report["evaluation_scope"], **report["aggregate"]}, indent=2))
        else:
            from src.inference.predict import predict_image, save_prediction
            prediction = predict_image(model, args.image, config)
            prediction["weights"] = str(args.weights.resolve())
            save_prediction(prediction, args.output)
            print(f"Mite count: {prediction['count']}")


if __name__ == "__main__":
    main()
