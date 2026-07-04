"""
benchmarks/run.py
==================
Master benchmark runner for VolSurface.

Usage
-----
# Run all benchmarks
python benchmarks/run.py

# Run one category
python benchmarks/run.py pricing
python benchmarks/run.py iv
python benchmarks/run.py greeks
python benchmarks/run.py surface
python benchmarks/run.py backtester

# Quick mode (fewer iterations, faster wall time)
python benchmarks/run.py --quick
"""
import sys
import time
import subprocess
from pathlib import Path
import os

BENCH_DIR = Path(__file__).parent

MODULES = {
    "pricing":   "benchmark_pricing.py",
    "iv":        "benchmark_iv.py",
    "greeks":    "benchmark_greeks.py",
    "surface":   "benchmark_surface.py",
    "backtester":"benchmark_backtester.py",
}


def _header(title: str) -> None:
    w = 62
    print(f"\n{'═' * w}")
    print(f"  {title}")
    print(f"{'═' * w}")


def _run_module(name: str, path: Path) -> float:
    """Run a benchmark module in a subprocess; return wall-clock seconds."""
    t0 = time.perf_counter()

    project_root = BENCH_DIR.parent

    env = os.environ.copy()

    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(project_root)
        if not existing
        else str(project_root) + os.pathsep + existing
    )

    res = subprocess.run(
        [sys.executable, str(path)],
        cwd=project_root,
        env=env,
        capture_output=False,
        text=True,
    )

    elapsed = time.perf_counter() - t0

    if res.returncode != 0:
        print(f"\n  [WARNING] {name} exited with code {res.returncode}")

    return elapsed


def main() -> None:
    args = sys.argv[1:]

    # Which modules to run
    quick   = "--quick" in args
    targets = [a for a in args if not a.startswith("--")]

    if targets:
        selected = {k: v for k, v in MODULES.items() if k in targets}
        if not selected:
            print(f"Unknown target(s): {targets}")
            print(f"Available: {list(MODULES)}")
            sys.exit(1)
    else:
        selected = MODULES

    # Environment info
    import platform, numpy as np
    _header("VolSurface Benchmark Suite")
    print(f"  Python   : {sys.version.split()[0]}")
    print(f"  Platform : {platform.platform()}")
    print(f"  NumPy    : {np.__version__}")
    try:
        import numba
        print(f"  Numba    : {numba.__version__}  (JIT enabled)")
    except ImportError:
        print(f"  Numba    : not installed  (NumPy fallback active)")

    if quick:
        print(f"\n  [QUICK MODE — reduced iteration counts]")

    # Run each module
    timings: dict[str, float] = {}
    for name, filename in selected.items():
        path = BENCH_DIR / filename
        if not path.exists():
            print(f"\n  [SKIP] {filename} not found")
            continue
        _header(f"{name.upper()} — {filename}")
        elapsed = _run_module(name, path)
        timings[name] = elapsed

    # Summary table
    if len(timings) > 1:
        _header("Summary")
        print(f"  {'Module':<14}  {'Wall time':>10}")
        print(f"  {'-'*14}  {'-'*10}")
        total = 0.0
        for name, t in timings.items():
            print(f"  {name:<14}  {t:>8.1f}s")
            total += t
        print(f"  {'─'*26}")
        print(f"  {'TOTAL':<14}  {total:>8.1f}s")

    print()


if __name__ == "__main__":
    main()
