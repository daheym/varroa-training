"""Predict unreviewed images using the existing tiled inference pipeline."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Config, read_config
from src.inference.interchange import predict_dataset

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--model-version")
    p.add_argument("--config", type=Path)
    p.add_argument("--allow-legacy-original", action="store_true")
    args = p.parse_args()
    config = read_config(args.config) if args.config else read_config(Path(__file__).resolve().parents[1] / "configs/default.json")
    if args.allow_legacy_original:
        config.allow_legacy_original = True
    from ultralytics import YOLO
    from src.inference.contract import check_model_contract
    check_model_contract(args.model, config)
    predict_dataset(args.dataset, YOLO(str(args.model)), args.output, config, args.model_version or args.model.stem)

if __name__ == "__main__":
    main()
