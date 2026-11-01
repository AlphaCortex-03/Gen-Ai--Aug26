"""run_all.py -- run every project in this repo sequentially, timing each one.

Each project is discovered automatically and its entry point resolved as:
    src/main.py    (p5-p14, pipeline style)
    src/train.py   (p15-p20, self-contained style)

Usage (PowerShell):
    python run_all.py                 # run everything, full training
    python run_all.py --smoke         # fast smoke run (GENAI_ITERS=100 for 15-20)
    python run_all.py --only 6,15     # run only these project numbers
    python run_all.py --stop          # abort at the first failure
    python run_all.py --list          # show what would run, then exit

Exit code is 0 only when every project succeeds.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------- discovery
def project_num(name: str) -> int:
    """p5-calo-demo -> 5 (also accepts legacy project-5-... and 05-... formats)."""
    tokens = name.split("-")
    raw = tokens[0][1:] if tokens[0].startswith("p") else tokens[0]
    return int(raw)


def find_projects() -> list[Path]:
    """Top-level project folders p5- ... p20-, in numeric order."""
    projects = [d for d in ROOT.iterdir()
                if d.is_dir() and d.name != "common"
                and d.name.split("-")[0].startswith("p")
                and d.name.split("-")[0][1:].isdigit()]
    projects.sort(key=lambda p: project_num(p.name))
    return projects


def entry_point(pdir: Path) -> Path | None:
    for candidate in ("src/main.py", "src/train.py", "train.py"):
        cand = pdir / candidate
        if cand.is_file():
            return cand
    return None


# ---------------------------------------------------------------- runner
def run_project(pdir: Path, entry: Path, smoke: bool, timeout: int) -> dict:
    env = os.environ.copy()
    if smoke:
        env["GENAI_ITERS"] = "100"          # only honoured by 15-20 train.py
    cmd = [sys.executable, str(entry)]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=str(pdir), env=env, timeout=timeout)
        code = proc.returncode
        err = None
    except subprocess.TimeoutExpired:
        code, err = None, f"timeout after {timeout}s"
    elapsed = time.perf_counter() - t0
    return {
        "num": project_num(pdir.name),
        "dir": pdir.name,
        "entry": str(entry.relative_to(pdir)).replace("\\", "/"),
        "seconds": round(elapsed, 2),
        "exit_code": code,
        "error": err,
        "status": "PASS" if code == 0 and err is None else "FAIL",
    }


def print_table(results: list[dict], total_s: float) -> None:
    w = max(len(r["dir"]) for r in results)
    print()
    print("=" * 66)
    print("SUMMARY")
    print("=" * 66)
    print(f"{'#':<4}{'dir':<{w + 2}}{'entry':<16}{'sec':>8}  {'status':<6} {'exit'}")
    print("-" * 66)
    for r in results:
        exit_s = "-" if r["exit_code"] is None else str(r["exit_code"])
        print(f"{r['num']:<4}{r['dir']:<{w + 2}}{r['entry']:<16}{r['seconds']:>8.2f}  "
              f"{r['status']:<6}{exit_s}")
    print("-" * 66)
    passed = sum(1 for r in results if r["status"] == "PASS")
    print(f"{passed}/{len(results)} passed  |  total time {total_s:.2f} s "
          f"({total_s / 60:.1f} min)")
    print("=" * 66)


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="run all Aug projects sequentially")
    ap.add_argument("--smoke", action="store_true",
                    help="quick iterations for the train.py projects (15-20)")
    ap.add_argument("--only", default="",
                    help="comma-separated project numbers to run, e.g. 6,15")
    ap.add_argument("--skip", default="",
                    help="comma-separated project numbers to skip")
    ap.add_argument("--stop", action="store_true",
                    help="stop at the first failure")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="per-project timeout in seconds (default 3600)")
    ap.add_argument("--list", action="store_true",
                    help="list what would run and exit")
    args = ap.parse_args()

    only = {int(x) for x in args.only.split(",") if x.strip()}
    skip = {int(x) for x in args.skip.split(",") if x.strip()}

    jobs = []
    failed_restore = []
    for pdir in find_projects():
        entry = entry_point(pdir)
        num = project_num(pdir.name)
        if entry is None:
            failed_restore.append(pdir.name)
            continue
        if (only and num not in only) or num in skip:
            continue
        jobs.append((pdir, entry))

    if args.list:
        for pdir, entry in jobs:
            print(f"{pdir.name:<26} -> {entry.relative_to(pdir)}")
        return 0

    if failed_restore:
        print("WARNING: these project folders have no entry point (skipped):")
        for n in failed_restore:
            print(f"  {n}")

    print(f"run_all: {len(jobs)} projects, mode={'SMOKE' if args.smoke else 'FULL'}, "
          f"torch CPU/GPU as available")
    print(f"started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 66)

    results = []
    total_t0 = time.perf_counter()
    for pdir, entry in jobs:
        num = project_num(pdir.name)
        print(f"\n----- [{pdir.name}] running {entry.name} -----")
        r = run_project(pdir, entry, smoke=args.smoke, timeout=args.timeout)
        results.append(r)
        if r["error"]:
            print(f"  ERROR: {r['error']}")
        print(f"[TEST TIME {pdir.name}] {r['seconds']:.2f} sec  (exit code "
              f"{r['exit_code']})  {r['status']}")
        if args.stop and r["status"] == "FAIL":
            print("stop-on-failure: aborting remaining projects")
            break

    total_s = time.perf_counter() - total_t0
    print_table(results, total_s)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": "smoke" if args.smoke else "full",
        "total_seconds": round(total_s, 2),
        "results": results,
    }
    out = ROOT / "run_results.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report written to {out}")

    return 0 if all(r["status"] == "PASS" for r in results) and results else 1


if __name__ == "__main__":
    sys.exit(main())
