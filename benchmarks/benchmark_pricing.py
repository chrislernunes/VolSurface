"""
benchmarks/benchmark_pricing.py
================================
Benchmark Black-76 pricing: scalar bs_price vs Numba-JIT batch_bs_price.

Measures
--------
- Single-option bs_price call throughput (pure Python, no JIT)
- batch_bs_price Numba JIT warmup cost (first call)
- batch_bs_price Numba JIT steady-state throughput for N = 100, 1_000, 10_000, 100_000
- Speedup ratio: Numba / Python loop at each N
- bs_vega, bs_delta throughput (used in IV solvers and hedging)
"""
import sys
import time
import statistics
from typing import List

import numpy as np

sys.path.insert(0, "..")

from src.iv_calculator import bs_price, bs_vega, bs_delta, batch_bs_price, _NUMBA_AVAILABLE

# ── Benchmark helpers ────────────────────────────────────────────────────────

def _timer(fn, n_reps: int = 5):
    """Run fn() n_reps times; return (median_seconds, all_seconds)."""
    times: List[float] = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), times


def _fmt(seconds: float, n_ops: int) -> str:
    rate = n_ops / seconds
    if rate >= 1e6:
        return f"{seconds*1000:.3f} ms  ({rate/1e6:.1f} M ops/s)"
    return f"{seconds*1000:.3f} ms  ({rate/1e3:.1f} K ops/s)"


# ── Fixed test cases ─────────────────────────────────────────────────────────

F, K, T, S = 60_000.0, 60_000.0, 0.25, 0.80   # ATM BTC-style option

# ── 1. Single-option scalar throughput ──────────────────────────────────────

def bench_scalar_price(n: int = 100_000):
    print(f"\n[1] bs_price (scalar, n={n:,})")
    med, _ = _timer(lambda: [bs_price(F, K, T, S) for _ in range(n)], n_reps=3)
    print(f"    {_fmt(med, n)}")


def bench_scalar_vega(n: int = 100_000):
    print(f"\n[2] bs_vega (scalar, n={n:,})")
    med, _ = _timer(lambda: [bs_vega(F, K, T, S) for _ in range(n)], n_reps=3)
    print(f"    {_fmt(med, n)}")


def bench_scalar_delta(n: int = 100_000):
    print(f"\n[3] bs_delta (scalar, n={n:,})")
    med, _ = _timer(lambda: [bs_delta(F, K, T, S) for _ in range(n)], n_reps=3)
    print(f"    {_fmt(med, n)}")


# ── 2. Numba batch pricer ────────────────────────────────────────────────────

def _make_batch(n: int):
    rng = np.random.default_rng(42)
    return (
        np.full(n, F),
        np.full(F, K) if False else rng.uniform(40_000, 80_000, n).astype(np.float64),
        np.full(n, T),
        rng.uniform(0.40, 1.20, n).astype(np.float64),
        np.ones(n, dtype=np.int8),
    )


def bench_batch_price():
    if not _NUMBA_AVAILABLE:
        print("\n[4-8] batch_bs_price: Numba not available — using NumPy fallback.")

    print(f"\n[4] batch_bs_price — JIT warmup (first call, n=100)")
    F_a, K_a, T_a, S_a, ic = _make_batch(100)
    t0 = time.perf_counter()
    batch_bs_price(F_a, K_a, T_a, S_a, ic)
    warmup = time.perf_counter() - t0
    tag = "JIT compile" if _NUMBA_AVAILABLE else "NumPy (no Numba)"
    print(f"    {warmup*1000:.1f} ms  ({tag})")

    for n in [100, 1_000, 10_000, 100_000]:
        F_a, K_a, T_a, S_a, ic = _make_batch(n)
        label = "Numba" if _NUMBA_AVAILABLE else "NumPy"

        # batch
        med_batch, _ = _timer(lambda: batch_bs_price(F_a, K_a, T_a, S_a, ic), n_reps=5)

        # python loop baseline
        med_loop, _ = _timer(
            lambda: [bs_price(float(F_a[i]), float(K_a[i]), float(T_a[i]), float(S_a[i]))
                     for i in range(n)],
            n_reps=3
        )

        speedup = med_loop / med_batch
        print(f"\n[N={n:>7,}] {label} batch: {_fmt(med_batch, n)}")
        print(f"           Python loop: {_fmt(med_loop, n)}")
        print(f"           Speedup:     {speedup:.1f}×")


# ── 3. NumPy vectorised baseline ─────────────────────────────────────────────

def bench_numpy_vectorised(n: int = 100_000):
    print(f"\n[9] NumPy vectorised bs_price (n={n:,})")
    F_a = np.full(n, F)
    K_a = np.random.default_rng(0).uniform(40_000, 80_000, n)
    T_a = np.full(n, T)
    S_a = np.full(n, S)

    # Inline vectorised formula (what a numpy-first implementation would do)
    def numpy_price():
        from scipy.stats import norm
        sqT = np.sqrt(T_a)
        d1  = (np.log(F_a / K_a) + 0.5 * S_a**2 * T_a) / (S_a * sqT)
        d2  = d1 - S_a * sqT
        return F_a * norm.cdf(d1) - K_a * norm.cdf(d2)

    med, _ = _timer(numpy_price, n_reps=5)
    print(f"    {_fmt(med, n)}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  VolSurface — Pricing Benchmarks")
    print(f"  Numba available: {_NUMBA_AVAILABLE}")
    print("=" * 60)

    bench_scalar_price()
    bench_scalar_vega()
    bench_scalar_delta()
    bench_batch_price()
    bench_numpy_vectorised()

    print("\n" + "=" * 60)
    print("Done.")
