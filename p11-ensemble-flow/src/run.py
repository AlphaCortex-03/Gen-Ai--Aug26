"""Run this project end-to-end (times and figures appear on the console).

Each project's src/main.py reads its own configs/default.yaml, so the
configuration is controlled there. Use --full only as a reminder that the
default training length is already the full-quality setting.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from main import main  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run this Aug project")
    parser.add_argument("--full", action="store_true",
                        help="accepts for compatibility; config comes from "
                             "configs/default.yaml")
    parser.parse_args()

    t0 = time.perf_counter()
    main()
    print(f"[{Path(__file__).parent.name}] done in {time.perf_counter() - t0:.2f} s")
