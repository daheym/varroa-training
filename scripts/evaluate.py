import sys
from src.cli import main

if __name__ == "__main__":
    main(["--config", sys.argv[1], "evaluate", *sys.argv[2:]])
