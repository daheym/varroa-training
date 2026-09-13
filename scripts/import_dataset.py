"""Validate a portable directory and derive tiled YOLO training artifacts."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Config, read_config, write_json
from src.interchange import load_bundle
from src.dataset.prepare import prepare

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("--output", type=Path, default=Path("data/processed/imported"))
    p.add_argument("--config", type=Path)
    p.add_argument("--allow-legacy-original", action="store_true")
    args = p.parse_args()
    root, output = args.dataset.resolve(), args.output.resolve()
    load_bundle(root, training=True)
    if output.is_relative_to(root) or root.is_relative_to(output):
        p.error("Derived output must be separate from the source dataset")
    config = read_config(args.config) if args.config else read_config(Path(__file__).resolve().parents[1] / "configs/default.json")
    config = Config.model_validate({**config.model_dump(), "images": root / "images", "annotations": root / "annotations",
        "processed": output, "splits": output.parent / (output.name + "-splits.json"),
        "allow_legacy_original": args.allow_legacy_original or config.allow_legacy_original})
    prepare(config)
    write_json(output / "training-config.json", config.model_dump(mode="json"))
    print(output / "training-config.json")

if __name__ == "__main__":
    main()
