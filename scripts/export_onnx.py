import sys
from src.cli import main

if __name__ == "__main__":
    main(["--config", sys.argv[1], "export", *sys.argv[2:]])
