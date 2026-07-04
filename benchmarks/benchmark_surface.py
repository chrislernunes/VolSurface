"""
benchmarks/benchmark_surface.py
=================================
Benchmark SVI calibration and vol surface query throughput.

Measures
--------
- SVISliceFitter.fit() per-slice calibration time (DE + NM phases)
- VolSurface.fit() full multi-expiry calibration time
- VolSurface.get_iv() point query throughput
- VolSurface.vol_grid() 3D grid generation
- SVIParams.total_var() and implied_vol() vectorised evaluation
- SABRFitter.fit() calibration time
- SplineSlice.fit() and query throughput
"""
import sys
import time
import statistics
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, "..")

from src.surface_fit import (
    SVIParams, SVISliceFitter, SABRFitter, SplineSlice, VolSurface
)
from src.iv_calculator import bs_price

# ── Helpers ──────────────────────────────────────────────────────────────────

def _timer(fn, n_reps: int = 3):
    times: List[float] = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), times


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f} ms"


def _fmt_rate(seconds: float, n_ops: int) -> str:
    rate = n_ops / seconds
    if rate >= 1e6:
        return f"{seconds*1000:.3f} ms  ({rate/1e6:.2f} M ops/s)"
    if rate >= 1e3:
        return f"{seconds*1000:.3f} ms  ({rate/1e3:.1f} K ops/s)"
    return f"{seconds*1000:.3f} ms  ({rate:.1f} ops/s)"


# ── Synthetic data builder ────────────────────────────────────────────────────

def _make_slice(F: float = 60_000.0, T: float = 0.25, n: int = 20):
    """Return (k, w, T, F) for a realistic BTC smile."""
    true_p = SVIParams(a=0.04, b=0.10, rho=-0.50, m=0.00, sigma=0.30)
    k      = np.linspace(-0.40, 0.40, n)
    w      = true_p.total_var(k)
    noise  = np.random.default_rng(42).normal(0, w * 0.003, n)
    return k, np.maximum(w + noise, 1e-6), T, F


def _make_surface_df(
    F: float = 60_000.0,
    n_expiries: int = 6,
    n_strikes: int = 20,
) -> pd.DataFrame:
    true_p   = SVIParams(a=0.04, b=0.10, rho=-0.50, m=0.00, sigma=0.30)
    expiries = [(f"EXP{i}", (i * 30) / 365.25) for i in range(1, n_expiries + 1)]
    rows: List[dict] = []
    rng = np.random.default_rng(0)
    for exp_str, T in expiries:
        strikes = np.linspace(F * 0.78, F * 1.28, n_strikes)
        for K in strikes:
            k  = np.log(K / F)
            iv = float(true_p.implied_vol(np.array([k]), T)[0])
            iv = max(iv + rng.normal(0, 0.004), 0.01)
            rows.append({
                "expiry":  exp_str, "tte": T,
                "spot": float(F), "forward": float(F),
                "strike": float(K), "type": "call",
                "calc_iv": iv, "spread": 10.0,
            })
    return pd.DataFrame(rows)


# ── 1. SVIParams vectorised operations ───────────────────────────────────────

def bench_svi_params(n: int = 1_000_000):
    print(f"\n[1] SVIParams vectorised (n={n:,} strike grid)")
    p = SVIParams(a=0.04, b=0.10, rho=-0.50, m=0.00, sigma=0.30)
    k = np.linspace(-2.0, 2.0, n)

    for name, fn in [
        ("total_var",         lambda: p.total_var(k)),
        ("implied_vol(T=.25)",lambda: p.implied_vol(k, 0.25)),
        ("butterfly_density", lambda: p.butterfly_density(k)),
        ("dw_dk",             lambda: p.dw_dk(k)),
    ]:
        med, _ = _timer(fn, n_reps=5)
        print(f"    {name:<22}: {_fmt_rate(med, n)}")


# ── 2. SVISliceFitter calibration time ───────────────────────────────────────

def bench_svi_calibration():
    print("\n[2] SVISliceFitter.fit() calibration time")
    k, w, T, _ = _make_slice(n=20)
    fitter      = SVISliceFitter()

    # Warm up once
    fitter.fit(k, w, T)

    # Time 3 independent fits
    times: List[float] = []
    for _ in range(3):
        t0 = time.perf_counter()
        fitter.fit(k, w, T)
        times.append(time.perf_counter() - t0)

    med = statistics.median(times)
    print(f"    n_strikes=20: {_fmt_ms(med)}  (DE + Nelder-Mead)")

    # Vary n_strikes
    for n_strikes in [10, 20, 40]:
        k2, w2, T2, _ = _make_slice(n=n_strikes)
        t0 = time.perf_counter()
        fitter.fit(k2, w2, T2)
        elapsed = time.perf_counter() - t0
        print(f"    n_strikes={n_strikes:<3}: {_fmt_ms(elapsed)}")


# ── 3. Full VolSurface calibration ────────────────────────────────────────────

def bench_vol_surface_fit():
    print("\n[3] VolSurface.fit() — full surface calibration")
    for n_expiries in [3, 6, 10]:
        df      = _make_surface_df(n_expiries=n_expiries, n_strikes=20)
        surface = VolSurface(min_strikes=5)
        t0      = time.perf_counter()
        surface.fit(df, iv_col="calc_iv")
        elapsed = time.perf_counter() - t0
        n_slices = len(surface.slices)
        avg_rmse = sum(s.calibration_rmse for s in surface.slices) / n_slices
        print(
            f"    {n_expiries} expiries × 20 strikes: "
            f"{_fmt_ms(elapsed)}  "
            f"(avg RMSE {avg_rmse*100:.3f} vp)"
        )


# ── 4. VolSurface query throughput ────────────────────────────────────────────

def bench_vol_surface_query():
    print("\n[4] VolSurface.get_iv() point query throughput")
    df      = _make_surface_df(n_expiries=6, n_strikes=20)
    surface = VolSurface(min_strikes=5)
    surface.fit(df, iv_col="calc_iv")
    F       = surface.slices[0].F
    T       = surface.slices[1].T   # interpolated (between slices)

    for n in [100, 1_000, 10_000]:
        med, _ = _timer(
            lambda: [surface.get_iv(K=F * 0.95, T=T, F=F) for _ in range(n)],
            n_reps=5,
        )
        print(f"    n={n:>6,}: {_fmt_rate(med, n)}")


# ── 5. vol_grid 3D generation ─────────────────────────────────────────────────

def bench_vol_grid():
    print("\n[5] VolSurface.vol_grid() 3D grid generation")
    df      = _make_surface_df(n_expiries=8, n_strikes=20)
    surface = VolSurface(min_strikes=5)
    surface.fit(df, iv_col="calc_iv")

    for n_strikes in [50, 100, 200]:
        med, _ = _timer(
            lambda n=n_strikes: surface.vol_grid(n_strikes=n),
            n_reps=5,
        )
        total = len(surface.slices) * n_strikes
        print(f"    {len(surface.slices)} slices × {n_strikes} strikes = {total} pts: {_fmt_ms(med)}")


# ── 6. SABR calibration ───────────────────────────────────────────────────────

def bench_sabr_fit():
    print("\n[6] SABRFitter.fit() calibration time")
    F       = 60_000.0
    T       = 0.25
    strikes = np.linspace(F * 0.80, F * 1.25, 15)
    from src.surface_fit import sabr_implied_vol
    ivs     = np.array([sabr_implied_vol(F, K, T, 0.40, 0.5, -0.30, 0.50) for K in strikes])
    fitter  = SABRFitter(beta=0.5)

    times: List[float] = []
    for _ in range(5):
        t0 = time.perf_counter()
        fitter.fit(F, strikes, ivs, T)
        times.append(time.perf_counter() - t0)

    print(f"    n_strikes=15: {_fmt_ms(statistics.median(times))}")


# ── 7. SplineSlice ────────────────────────────────────────────────────────────

def bench_spline():
    print("\n[7] SplineSlice fit + query")
    k, w, T, _ = _make_slice(n=20)
    sp = SplineSlice()

    t0 = time.perf_counter()
    sp.fit(k, w, T)
    print(f"    fit (n=20)  : {_fmt_ms(time.perf_counter() - t0)}")

    k_query = np.linspace(-0.4, 0.4, 1000)
    med, _  = _timer(lambda: sp.total_var(k_query), n_reps=20)
    print(f"    query 1000k : {_fmt_ms(med)}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  VolSurface — Surface Calibration Benchmarks")
    print("=" * 60)

    bench_svi_params()
    bench_svi_calibration()
    bench_vol_surface_fit()
    bench_vol_surface_query()
    bench_vol_grid()
    bench_sabr_fit()
    bench_spline()

    print("\n" + "=" * 60)
    print("Done.")
