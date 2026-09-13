"""Canonical normal training CLI; implementation lives in src.training.train."""
from pathlib import Path
from datetime import datetime
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.cli import main  # noqa: E402


if __name__ == "__main__":
    arguments = list(sys.argv[1:])
    run_name = None
    for position, argument in enumerate(arguments):
        if argument == "--run-name":
            if position + 1 >= len(arguments):
                raise SystemExit("--run-name requires a value")
            run_name = arguments[position + 1]
            del arguments[position:position + 2]
            break
        if argument.startswith("--run-name="):
            run_name = argument.split("=", 1)[1]
            del arguments[position]
            break
    output_index = next((i for i, value in enumerate(arguments) if value == "--output" or value.startswith("--output=")), None)
    if output_index is not None:
        if arguments[output_index] == "--output":
            if output_index + 1 >= len(arguments):
                raise SystemExit("--output requires a path")
            output_root = Path(arguments[output_index + 1]).expanduser()
            value_index = output_index + 1
        else:
            output_root = Path(arguments[output_index].split("=", 1)[1]).expanduser()
            value_index = output_index
        if not run_name:
            run_name = datetime.now().strftime("run-%Y%m%d-%H%M%S")
            candidate = output_root / run_name
            suffix = 1
            while candidate.exists():
                candidate = output_root / f"{run_name}-{suffix:02d}"
                suffix += 1
            run_name = candidate.name
        if Path(run_name).name != run_name or run_name in {".", ".."}:
            raise SystemExit("--run-name must be a single directory name")
        run_path = output_root / run_name
        if value_index == output_index:
            arguments[output_index] = f"--output={run_path}"
        else:
            arguments[value_index] = str(run_path)
        print(f"Run artifacts: {run_path}")
    main(["train", *arguments])
