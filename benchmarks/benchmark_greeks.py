"""
benchmarks/benchmark_greeks.py
================================
Benchmark analytical and finite-difference Greek computation.

Measures
--------
- Single compute() call throughput (all Greeks at once)
- Per-Greek breakdown: Delta, Gamma, Vega, Theta, Vanna, Volga
- compute_surface_greeks() batch throughput on a realistic surface
- portfolio_greeks() aggregation throughput
- sticky_strike_delta vs sticky_delta_delta comparison
"""
import sys
import time
import statistics
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, "..")

from src.greeks import GreeksCalculator
from src.iv_calculator import bs_price

CALC = GreeksCalculator(r=0.0)

F, K, T, S = 60_000.0, 60_000.0, 0.25, 0.80   # ATM BTC-style


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
    return f"{seconds * 1000:.3f} ms  ({rate / 1e3:.1f} K ops/s)"


# ── 1. Full Greeks per call ───────────────────────────────────────────────────

def bench_full_greeks(n: int = 50_000):
    print(f"\n[1] compute() — all Greeks (n={n:,})")

    for ot in ("call", "put"):
        med, _ = _timer(
            lambda o=ot: [CALC.compute(F, K, T, S, o) for _ in range(n)],
            n_reps=3,
        )
        print(f"    {ot:<5}: {_fmt(med, n)}")


# ── 2. Individual Greek breakdown ─────────────────────────────────────────────

def bench_individual_greeks(n: int = 50_000):
    print(f"\n[2] Individual Greek access after compute() (n={n:,})")
    g = CALC.compute(F, K, T, S, "call")

    attrs = ["delta", "gamma", "vega", "theta", "rho",
             "vanna", "volga", "charm", "veta", "speed", "zomma", "color",
             "dollar_delta", "dollar_gamma", "dollar_vega"]

    for attr in attrs:
        med, _ = _timer(lambda a=attr: [getattr(g, a) for _ in range(n)], n_reps=3)
        rate = n / med
        print(f"    {attr:<16}: {rate/1e6:.0f} M/s")


# ── 3. Batch surface Greeks ───────────────────────────────────────────────────

def bench_surface_greeks(n_options: int = 500):
    print(f"\n[3] compute_surface_greeks() — batch (n={n_options} options)")

    rng     = np.random.default_rng(42)
    strikes = np.linspace(F * 0.75, F * 1.30, n_options)
    ivs     = 0.70 + 0.15 * np.abs(np.log(strikes / F))

    df = pd.DataFrame({
        "calc_iv": ivs,
        "spot":    np.full(n_options, F),
        "forward": np.full(n_options, F),
        "strike":  strikes,
        "tte":     np.full(n_options, T),
        "type":    ["call"] * n_options,
    })

    med, _ = _timer(lambda: CALC.compute_surface_greeks(df.copy()), n_reps=5)
    print(f"    {_fmt(med, n_options)}")

    # Single-row baseline
    med1, _ = _timer(
        lambda: CALC.compute_surface_greeks(df.iloc[:1].copy()), n_reps=10
    )
    print(f"    Single row : {med1 * 1000:.3f} ms")
    print(f"    Batch overhead vs {n_options}× single: {med / (med1 * n_options):.2f}×")


# ── 4. Portfolio aggregation ──────────────────────────────────────────────────

def bench_portfolio_greeks(n_legs: int = 200, n_reps: int = 1000):
    print(f"\n[4] portfolio_greeks() aggregation (n_legs={n_legs}, n_reps={n_reps})")

    rng     = np.random.default_rng(0)
    strikes = rng.uniform(F * 0.70, F * 1.30, n_legs)
    ivs     = 0.65 + 0.20 * np.abs(np.log(strikes / F))

    df = pd.DataFrame({
        "calc_iv": ivs,
        "spot":    np.full(n_legs, F),
        "forward": np.full(n_legs, F),
        "strike":  strikes,
        "tte":     np.full(n_legs, T),
        "type":    ["call"] * n_legs,
        "qty":     rng.choice([-1, 1], n_legs).astype(float),
    })
    greeks_df = CALC.compute_surface_greeks(df.copy())
    greeks_df["qty"] = df["qty"]

    med, _ = _timer(
        lambda: GreeksCalculator.portfolio_greeks(greeks_df),
        n_reps=n_reps,
    )
    print(f"    {_fmt(med, n_legs)}")

    net = GreeksCalculator.portfolio_greeks(greeks_df)
    print(f"    Net delta: {net.get('net_delta', 0):.4f}  "
          f"Net vega: ${net.get('net_dollar_vega', 0):,.0f}")


# ── 5. Sticky delta methods ───────────────────────────────────────────────────

def bench_sticky_delta(n: int = 10_000):
    print(f"\n[5] sticky_strike_delta vs sticky_delta_delta (n={n:,})")

    for name, fn in [
        ("sticky_strike", lambda: [CALC.sticky_strike_delta(F, K, T, S, "call") for _ in range(n)]),
        ("sticky_delta",  lambda: [CALC.sticky_delta_delta(F, K, T, S, "call", dsigma_dF=-1e-5) for _ in range(n)]),
    ]:
        med, _ = _timer(fn, n_reps=3)
        print(f"    {name:<16}: {_fmt(med, n)}")


# ── 6. Numerical greeks (FD cross-check) ─────────────────────────────────────

def bench_numerical_greeks(n: int = 5_000):
    print(f"\n[6] numerical_greeks() finite-difference (n={n:,})")
    med, _ = _timer(
        lambda: [CALC.numerical_greeks(F, K, T, S, "call") for _ in range(n)],
        n_reps=3,
    )
    print(f"    {_fmt(med, n)}")
    print(f"    (≈12 bs_price evaluations per call)")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  VolSurface — Greeks Benchmarks")
    print("=" * 60)

    bench_full_greeks()
    bench_individual_greeks()
    bench_surface_greeks()
    bench_portfolio_greeks()
    bench_sticky_delta()
    bench_numerical_greeks()

    print("\n" + "=" * 60)
    print("Done.")
