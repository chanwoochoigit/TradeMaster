import warnings

warnings.filterwarnings("ignore")
import os
import sys
from pathlib import Path
from glob import glob

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT)


def main():

    paths = glob(os.path.join(ROOT, "configs", "*", "*.py"))

    # Create sequential pipeline
    with open(os.path.join(ROOT, "tools", "export_pipeline.sh"), "w") as f:
        f.write("#!/bin/bash\n")
        for path in paths:
            path = path.split("TradeMaster/")[1]
            path = path.replace(ROOT, ".")
            # Run sequentially with output
            f.write(f"echo 'Running {path}...'\n")
            f.write(f"python tools/export_allocations.py --config {path}\n")
            f.write(f"echo 'Finished {path}'\n\n")


if __name__ == "__main__":
    main()
