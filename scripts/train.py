"""Run from repository root after editable installation."""
import sys
from src.cli import main

if __name__ == "__main__":
    main(["--config", sys.argv[1], "train", *sys.argv[2:]])
