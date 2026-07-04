"""
benchmarks/benchmark_iv.py
===========================
Benchmark the implied-volatility solver pipeline.

Measures
--------
- compute_iv scalar: ATM / OTM call / OTM put at multiple vol regimes
- compute_iv batch via compute_iv_surface on a realistic surface snapshot
- Halley-only vs Brent-only vs auto method comparison
- Solver iteration count proxy (measures wall-time convergence difference)
- IV surface recomputation throughput (options/second end-to-end)
"""
import sys
import time
import statistics
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, "..")

from src.iv_calculator import (
    bs_price,
    compute_iv,
    compute_iv_surface,
    _halley,
    _brent,
    _to_call_price,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _timer(fn, n_reps: int = 5):
    times: List[float] = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), times


def _fmt(seconds: float, n_ops: int) -> str:
    rate = n_ops / seconds
    if rate >= 1e6:
        return f"{seconds * 1000:.3f} ms  ({rate / 1e6:.2f} M ops/s)"
    if rate >= 1e3:
        return f"{seconds * 1000:.3f} ms  ({rate / 1e3:.1f} K ops/s)"
    return f"{seconds * 1000:.3f} ms  ({rate:.1f} ops/s)"


# ── Test cases covering the full regime space ────────────────────────────────

CASES = [
    # label,          F,      K,       T,     sigma,  option_type
    ("ATM 3M",       60_000, 60_000,  0.25,  0.80,   "call"),
    ("25d OTM call", 60_000, 68_000,  0.25,  0.72,   "call"),
    ("25d OTM put",  60_000, 52_000,  0.25,  0.88,   "put"),
    ("10d OTM put",  60_000, 44_000,  0.25,  1.05,   "put"),
    ("1W ATM",       60_000, 60_000,  7/365, 0.85,   "call"),
    ("6M ATM",       60_000, 60_000,  0.50,  0.70,   "call"),
    ("Crisis vol",   60_000, 60_000,  0.25,  2.50,   "call"),
    ("Low vol",      60_000, 60_000,  0.25,  0.15,   "call"),
    ("ETH-style",    3_000,  3_000,   0.25,  0.80,   "call"),
]


# ── 1. Scalar IV solver by regime ────────────────────────────────────────────

def bench_scalar_iv(n: int = 10_000):
    print(f"\n[1] compute_iv scalar (n={n:,} per case, method='auto')")
    print(f"    {'Case':<18} {'Median':>15}  {'True σ':>8}  {'Accuracy':>10}")
    print(f"    {'-'*55}")

    for label, F, K, T, true_sigma, ot in CASES:
        price = bs_price(F, K, T, true_sigma, option_type=ot)
        med, _ = _timer(
            lambda p=price, f=F, k=K, t=T, o=ot: [
                compute_iv(p, f, k, t, option_type=o) for _ in range(n)
            ],
            n_reps=3,
        )
        iv = compute_iv(price, F, K, T, option_type=ot)
        err = abs(iv - true_sigma) * 10_000 if np.isfinite(iv) else float("nan")
        rate = n / med
        print(
            f"    {label:<18} {med*1000:>8.3f} ms  "
            f"  {true_sigma:>6.2f}     {err:>6.2f} bps"
        )


# ── 2. Method comparison ─────────────────────────────────────────────────────

def bench_method_comparison(n: int = 5_000):
    print(f"\n[2] Halley vs Brent vs Auto (ATM, n={n:,})")
    F, K, T, sigma = 60_000.0, 60_000.0, 0.25, 0.80
    price   = bs_price(F, K, T, sigma)
    call_px = _to_call_price(price, F, K, T, 0.0, "call")

    for name, fn in [
        ("Halley", lambda: [_halley(F, K, T, call_px) for _ in range(n)]),
        ("Brent",  lambda: [_brent(F, K, T, call_px)  for _ in range(n)]),
        ("Auto",   lambda: [compute_iv(price, F, K, T) for _ in range(n)]),
    ]:
        med, _ = _timer(fn, n_reps=3)
        print(f"    {name:<8}: {_fmt(med, n)}")


# ── 3. Batch surface IV throughput ───────────────────────────────────────────

def bench_surface_iv(n_strikes: int = 50, n_expiries: int = 8):
    total = n_strikes * n_expiries
    print(f"\n[3] compute_iv_surface  ({n_expiries} expiries × {n_strikes} strikes = {total} options)")

    rng       = np.random.default_rng(42)
    F         = 60_000.0
    expiries  = np.linspace(7 / 365.25, 1.0, n_expiries)
    rows: List[dict] = []

    for T in expiries:
        strikes = np.linspace(F * 0.70, F * 1.35, n_strikes)
        for K in strikes:
            sigma = 0.70 + 0.15 * abs(np.log(K / F))
            ot    = "call" if rng.random() > 0.5 else "put"
            price = bs_price(F, K, T, sigma, option_type=ot)
            rows.append({
                "mid":     price,
                "spot":    F,
                "forward": F,
                "strike":  K,
                "tte":     T,
                "type":    ot,
                "expiry":  f"T{T*365:.0f}",
            })

    df = pd.DataFrame(rows)

    med, _ = _timer(lambda: compute_iv_surface(df.copy()), n_reps=3)
    rate   = total / med
    print(f"    Total: {med*1000:.1f} ms  ({rate:.0f} options/s)")
    print(f"    Per option: {med/total*1000:.4f} ms")


# ── 4. Accuracy stress: roundtrip error distribution ─────────────────────────

def bench_roundtrip_accuracy(n: int = 500):
    print(f"\n[4] Roundtrip accuracy distribution (n={n} random options)")

    rng    = np.random.default_rng(0)
    errors = []
    fails  = 0

    for _ in range(n):
        F     = rng.uniform(10_000, 100_000)
        km    = rng.choice([0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.25])
        K     = F * km
        T     = rng.uniform(3 / 365, 1.0)
        sigma = rng.uniform(0.20, 2.00)
        ot    = rng.choice(["call", "put"])

        price = bs_price(F, K, T, sigma, option_type=ot)
        iv    = compute_iv(price, F, K, T, option_type=ot)

        if not np.isfinite(iv):
            fails += 1
        else:
            errors.append(abs(iv - sigma) * 10_000)  # in bps

    errors_arr = np.array(errors)
    print(f"    NaN / fails : {fails} / {n}  ({fails/n:.1%})")
    print(f"    Median err  : {np.median(errors_arr):.3f} bps")
    print(f"    p95 err     : {np.percentile(errors_arr, 95):.3f} bps")
    print(f"    Max err     : {errors_arr.max():.3f} bps")
    print(f"    Within 1bp  : {(errors_arr < 1).mean():.1%}")
    print(f"    Within 10bp : {(errors_arr < 10).mean():.1%}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  VolSurface — IV Solver Benchmarks")
    print("=" * 60)

    bench_scalar_iv()
    bench_method_comparison()
    bench_surface_iv()
    bench_roundtrip_accuracy()

    print("\n" + "=" * 60)
    print("Done.")
