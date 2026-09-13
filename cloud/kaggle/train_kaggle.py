"""Thin Kaggle launcher for the repository's standard training CLI."""
import argparse
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Launch normal Varroa training from a Kaggle notebook")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "training" / "configs" / "varroa.yaml")
    parser.add_argument("--data", type=Path, default=Path("/kaggle/input/varroa-dataset/dataset.yaml"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/runs"))
    parser.add_argument("--run-name", help="Optional run folder name under --output")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", action="store_true", help="Fail early unless Kaggle CUDA is available")
    args, training_args = parser.parse_known_args(argv)
    if args.require_cuda:
        import torch
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable; enable a GPU accelerator in the Kaggle notebook")
    command = [
        sys.executable, str(REPO_ROOT / "training" / "train.py"),
        "--config", str(args.config), "--data", str(args.data),
        "--output", str(args.output), "--device", args.device,
        *training_args,
    ]
    if args.run_name:
        command.extend(("--run-name", args.run_name))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
