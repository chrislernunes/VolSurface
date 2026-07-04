"""
benchmarks/benchmark_backtester.py
====================================
Benchmark the delta-hedge backtester engine.

Measures
--------
- FeeModel arithmetic throughput (option_fee, perp_fee, settlement_fee)
- linear_slippage throughput
- DeltaHedgeBacktester.run() on price paths of varying length
- _settle_positions throughput (ITM and OTM)
- BacktestResult analytics: Sharpe, max_drawdown, pnl_attribution
- Realistic multi-cycle rolling straddle simulation (timing only)
"""
import sys
import time
import statistics
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, "..")

from src.backtester import (
    DeltaHedgeBacktester,
    FeeModel,
    BacktestResult,
    PnLRecord,
    linear_slippage,
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
        return f"{seconds*1000:.3f} ms  ({rate/1e6:.2f} M ops/s)"
    if rate >= 1e3:
        return f"{seconds*1000:.3f} ms  ({rate/1e3:.1f} K ops/s)"
    return f"{seconds*1000:.3f} ms  ({rate:.1f} ops/s)"


def _price_path(n: int, start: float = 60_000.0, sigma_annual: float = 0.80) -> pd.Series:
    """Deterministic GBM hourly price path."""
    rng = np.random.default_rng(42)
    dt  = 1.0 / (365.25 * 24)
    log_rets = rng.normal(
        -0.5 * sigma_annual**2 * dt,
        sigma_annual * np.sqrt(dt),
        n,
    )
    prices = start * np.exp(np.cumsum(log_rets))
    idx    = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.Series(prices, index=idx)


def _make_result(n_steps: int = 500) -> BacktestResult:
    """Build a BacktestResult with n_steps synthetic PnL records."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=n_steps, freq="1h", tz="UTC")
    recs = [
        PnLRecord(
            timestamp=ts, spot=60_000.0, portfolio_iv=0.80,
            delta_pnl=float(rng.normal(0, 50)),
            gamma_pnl=float(rng.normal(50, 20)),
            theta_pnl=float(rng.normal(30, 10)),
            fee_pnl=float(rng.normal(-5, 2)),
            total_pnl=float(rng.normal(75, 60)),
        )
        for ts in idx
    ]
    return BacktestResult(recs, positions=[])


# ── 1. FeeModel throughput ────────────────────────────────────────────────────

def bench_fee_model(n: int = 1_000_000):
    print(f"\n[1] FeeModel arithmetic (n={n:,})")
    fm = FeeModel()

    for name, fn in [
        ("option_fee (taker)", lambda: [fm.option_fee(60_000, 1.0, 5_000) for _ in range(n)]),
        ("option_fee (maker)", lambda: [fm.option_fee(60_000, 1.0, 5_000, is_maker=True) for _ in range(n)]),
        ("perp_fee",           lambda: [fm.perp_fee(60_000, 1.0) for _ in range(n)]),
        ("settlement_fee",     lambda: [fm.settlement_fee(60_000, 1.0) for _ in range(n)]),
    ]:
        med, _ = _timer(fn, n_reps=3)
        print(f"    {name:<22}: {_fmt(med, n)}")


# ── 2. linear_slippage throughput ────────────────────────────────────────────

def bench_slippage(n: int = 1_000_000):
    print(f"\n[2] linear_slippage (n={n:,})")
    med, _ = _timer(
        lambda: [linear_slippage(1.0, 60_000.0, 1.0) for _ in range(n)],
        n_reps=3,
    )
    print(f"    {_fmt(med, n)}")


# ── 3. Backtester run() on varying path lengths ───────────────────────────────

def bench_run_loop():
    print("\n[3] DeltaHedgeBacktester.run() — varying path length")

    for n_bars in [24, 168, 720, 8_760]:   # 1d, 1w, 1m, 1y in hours
        prices = _price_path(n_bars)

        def _run(prices=prices, n=n_bars):
            bt = DeltaHedgeBacktester(
                fee_model=FeeModel(),
                hedge_threshold=0.05,
                hedge_frequency_h=1.0,
                slippage_bps=1.0,
            )
            bt.add_short_straddle(
                strike=60_000.0,
                initial_T=max(n / (365.25 * 24), 7 / 365.25),
                n_lots=1.0,
                spot=60_000.0,
                iv=0.80,
                timestamp=prices.index[0],
            )
            return bt.run(prices)

        # Warm up
        _run()

        med, _ = _timer(_run, n_reps=3)
        result = _run()
        steps  = len(result.pnl_df)
        label  = {24: "1 day", 168: "1 week", 720: "1 month", 8_760: "1 year"}[n_bars]
        print(
            f"    {label:<10} ({n_bars:>5} bars, {steps} steps): "
            f"{med*1000:.1f} ms  ({steps/med:.0f} steps/s)"
        )


# ── 4. Settlement throughput ──────────────────────────────────────────────────

def bench_settlement():
    print("\n[4] _settle_positions throughput (ITM and OTM)")
    fm = FeeModel()

    for scenario, spot in [("OTM (spot=strike)", 60_000.0),
                             ("ITM (spot=70k)",   70_000.0)]:
        bt = DeltaHedgeBacktester(fee_model=fm)
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        bt.add_short_straddle(60_000.0, 0.25, 1.0, 60_000.0, 0.80, ts)

        n = 10_000
        med, _ = _timer(
            lambda s=spot: [bt._settle_positions(s) for _ in range(n)],
            n_reps=3,
        )
        print(f"    {scenario:<28}: {_fmt(med, n)}")


# ── 5. BacktestResult analytics throughput ────────────────────────────────────

def bench_result_analytics():
    print("\n[5] BacktestResult analytics throughput")
    result = _make_result(n_steps=8_760)   # 1 year of hourly steps

    for n in [1_000, 10_000]:
        for name, fn in [
            ("sharpe()",          lambda: result.sharpe()),
            ("max_drawdown()",    lambda: result.max_drawdown()),
            ("win_rate()",        lambda: result.win_rate()),
            ("calmar()",          lambda: result.calmar()),
            ("pnl_attribution()", lambda: result.pnl_attribution()),
            ("summary()",         lambda: result.summary()),
        ]:
            if n > 1_000 and "summary" not in name:
                continue
            med, _ = _timer(fn, n_reps=n)
            print(f"    {name:<24}: {med*1e6:.1f} µs/call")
        break  # only one n for analytics — they don't scale with n


# ── 6. Rolling straddle simulation (realistic multi-cycle) ────────────────────

def bench_rolling_straddles(n_cycles: int = 52):
    print(f"\n[6] Rolling weekly straddle simulation ({n_cycles} cycles)")

    prices  = _price_path(n_cycles * 168, start=60_000.0)  # n_cycles weeks of hourly data
    results: List[BacktestResult] = []

    t0 = time.perf_counter()
    for i in range(n_cycles):
        week = prices.iloc[i * 168 : (i + 1) * 168]
        if len(week) < 48:
            break
        S_entry = float(week.iloc[0])
        bt      = DeltaHedgeBacktester(fee_model=FeeModel(), slippage_bps=1.0)
        bt.add_short_straddle(S_entry, 7 / 365.25, 1.0, S_entry, 0.80, week.index[0])
        results.append(bt.run(week))

    elapsed = time.perf_counter() - t0
    pnls    = [r.summary()["total_pnl_usd"] for r in results]

    print(f"    {n_cycles} cycles in {elapsed*1000:.0f} ms  ({n_cycles/elapsed:.0f} cycles/s)")
    print(f"    Avg P&L / cycle: ${np.mean(pnls):+,.0f}")
    print(f"    Std P&L / cycle: ${np.std(pnls):,.0f}")
    print(f"    Win rate        : {(np.array(pnls) > 0).mean():.1%}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  VolSurface — Backtester Benchmarks")
    print("=" * 60)

    bench_fee_model()
    bench_slippage()
    bench_run_loop()
    bench_settlement()
    bench_result_analytics()
    bench_rolling_straddles()

    print("\n" + "=" * 60)
    print("Done.")
