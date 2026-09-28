import argparse
import json
from pathlib import Path

from src.config import Config, read_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Varroa detection training and evaluation")
    parser.add_argument("--config", default="configs/default.json", help="Paths in config are relative to the working directory")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    commands.add_parser("prepare")
    sub = commands.add_parser("visualize")
    sub.add_argument("--run", required=True, type=Path, help="Experiment directory containing evaluation.json")
    for command in ("train", "loo", "overfit"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", default=argparse.SUPPRESS, help="JSON or YAML configuration file")
        output_help = "New run directory" if command == "train" else "New experiment directory"
        sub.add_argument("--output", required=True, type=Path, help=output_help)
        if command == "train":
            sub.add_argument("--data", type=Path, help="YOLO dataset YAML; defaults to config data or processed/dataset.yaml")
            sub.add_argument("--device", help="auto, cpu, cuda, cuda:N, or CUDA index (default: config value)")
            sub.add_argument("--seed", type=int)
            sub.add_argument("--imgsz", "--image-size", dest="imgsz", type=int)
            sub.add_argument("--batch", "--batch-size", dest="batch", type=int)
            sub.add_argument("--epochs", type=int)
            sub.add_argument("--patience", type=int, help="Epochs without validation improvement before stopping; 0 disables")
            sub.add_argument("--workers", type=int)
            sub.add_argument("--resume", type=Path, help="Ultralytics checkpoint with optimizer state, usually last.pt")
        if command == "overfit":
            sub.add_argument("--image-id", help="Source filename stem to use when the input folder contains multiple images")
    sub = commands.add_parser("evaluate")
    sub.add_argument("--weights", required=True, type=Path)
    sub.add_argument("--output", required=True, type=Path)
    sub = commands.add_parser("infer")
    sub.add_argument("--weights", required=True, type=Path)
    sub.add_argument("--image", required=True, type=Path)
    sub.add_argument("--perspective", type=Path, help="Annotation document or physical perspective JSON")
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
        records = validate_dataset(config.images, config.annotations, config=config)
        print(json.dumps({"images": len(records), "points": sum(len(r["points"]) for r in records)}, indent=2))
    elif args.command == "prepare":
        from src.dataset.prepare import prepare
        manifest = prepare(config)
        print(json.dumps({"split_unit": manifest["split_unit"],
                          "splits": {name: len(ids) for name, ids in manifest["splits"].items()},
                          "tiles": len(manifest["tiles"])}, indent=2))
    elif args.command == "train":
        from src.training.train import train
        overrides = {key: value for key, value in {
            "data": args.data, "device": args.device, "seed": args.seed,
            "imgsz": args.imgsz, "batch": args.batch, "epochs": args.epochs,
            "patience": args.patience, "workers": args.workers, "resume": args.resume,
        }.items() if value is not None}
        if overrides:
            config = Config.model_validate({**config.model_dump(), **overrides})
        report = train(config, args.output, data=args.data)
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
        from src.inference.contract import check_model_contract
        check_model_contract(args.weights, config)
        model = YOLO(str(args.weights))
        if args.command == "evaluate":
            from src.evaluation.evaluate import evaluate
            manifest = json.loads((config.processed / "manifest.json").read_text(encoding="utf-8"))
            report = evaluate(model, manifest, config, args.output)
            print(json.dumps({"evaluation_scope": report["evaluation_scope"], **report["aggregate"]}, indent=2))
        else:
            from src.inference.predict import predict_image, save_prediction
            metadata = json.loads(args.perspective.read_text('utf-8')) if args.perspective else None
            if metadata and 'perspective' in metadata:
                metadata = metadata['perspective']
            prediction = predict_image(model, args.image, config, perspective=metadata)
            prediction["weights"] = str(args.weights.resolve())
            save_prediction(prediction, args.output)
            print(f"Mite count: {prediction['count']}")


if __name__ == "__main__":
    main()
