"""Standalone model export CLI, separate from training."""
import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export a trained Varroa YOLO checkpoint")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--format", choices=("onnx",), default="onnx")
    parser.add_argument("--imgsz", "--image-size", type=int, help="Defaults to the checkpoint preprocessing contract")
    args = parser.parse_args(argv)
    from src.inference.export import export_onnx
    print(export_onnx(args.checkpoint, imgsz=args.imgsz))


if __name__ == "__main__":
    main()
