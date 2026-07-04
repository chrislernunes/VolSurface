"""
benchmarks/_common.py
──────────────────────────────────────────────────────────────────────────
Shared timing harness + live-data loader for every benchmark_*.py script.

Design choices
--------------
* Every number a benchmark times comes from either a real Deribit snapshot
  (options chain, spot, mark_iv) or real Deribit OHLCV/DVOL history — never
  from hand-typed or randomly generated parameters. That's not just a data
  policy: it means the benchmarks measure performance on the same shape and
  distribution of inputs the library sees in production (real strike/expiry
  grids are not evenly spaced; real IV surfaces are not flat).
* Snapshots are cached to data/raw/ (max_age_hours, default 1h) so running
  the full benchmark suite repeatedly during development doesn't hammer
  Deribit's public API on every invocation. A stale/missing cache triggers
  exactly one live fetch, which is itself then cached for next time.
* If Deribit is unreachable and there is no usable cache, every loader
  raises RuntimeError with an actionable message. There is no synthetic
  fallback — a benchmark on fabricated numbers would be actively misleading.
"""
from __future__ import annotations

import logging
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd


def repo_root() -> Path:
    """Absolute path to the VolSurface repo root, regardless of cwd."""
    return Path(__file__).resolve().parent.parent


sys.path.insert(0, str(repo_root()))

from src.deribit_client import DeribitClient  # noqa: E402
from src.iv_calculator import compute_iv_surface  # noqa: E402

DATA_RAW = repo_root() / "data" / "raw"


# ──────────────────────────────────────────────────────────────────────────
# Timing harness
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkStats:
    label: str
    n_calls: int          # logical operations represented by ONE timed call
                           # (e.g. options priced per call) — used for ops/sec
    iterations: int        # how many times the timed call was repeated
    samples_us: List[float] = field(repr=False, default_factory=list)

    @property
    def mean_us(self) -> float:
        return statistics.mean(self.samples_us)

    @property
    def median_us(self) -> float:
        return statistics.median(self.samples_us)

    @property
    def std_us(self) -> float:
        return statistics.stdev(self.samples_us) if len(self.samples_us) > 1 else 0.0

    @property
    def min_us(self) -> float:
        return min(self.samples_us)

    @property
    def max_us(self) -> float:
        return max(self.samples_us)

    @property
    def p95_us(self) -> float:
        s = sorted(self.samples_us)
        idx = max(int(round(0.95 * len(s))) - 1, 0)
        return s[idx]

    @property
    def ops_per_sec(self) -> float:
        return self.n_calls / (self.mean_us / 1e6) if self.mean_us > 0 else float("inf")


def time_call(
    fn: Callable[[], object],
    *,
    iterations: int = 200,
    warmup: int = 20,
    n_calls: int = 1,
    label: str = "",
) -> BenchmarkStats:
    """
    Time a zero-argument callable.

    n_calls : how many "logical operations" one invocation of *fn* performs
              (e.g. len(df) if fn prices a whole DataFrame in one call).
              Purely for reporting ops/sec — does not affect timing itself.
    """
    for _ in range(max(warmup, 0)):
        fn()

    samples_us = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        samples_us.append((time.perf_counter() - t0) * 1e6)

    return BenchmarkStats(label=label, n_calls=n_calls, iterations=iterations,
                           samples_us=samples_us)


def _fmt_us(us: float) -> str:
    if us >= 1_000_000:
        return f"{us / 1_000_000:.2f}s"
    if us >= 1_000:
        return f"{us / 1_000:.2f}ms"
    return f"{us:.2f}µs"


def print_table(title: str, rows: List[BenchmarkStats], width: int = 88) -> None:
    print(f"\n{'─' * width}\n{title}\n{'─' * width}")
    print(f"{'operation':<34}{'mean':>10}{'median':>10}{'p95':>10}{'min':>10}{'ops/sec':>14}")
    print("─" * width)
    for r in rows:
        ops = f"{r.ops_per_sec:,.0f}" if r.ops_per_sec != float("inf") else "n/a"
        print(
            f"{r.label:<34}{_fmt_us(r.mean_us):>10}{_fmt_us(r.median_us):>10}"
            f"{_fmt_us(r.p95_us):>10}{_fmt_us(r.min_us):>10}{ops:>14}"
        )


@contextmanager
def quiet_logging(level: int = logging.WARNING):
    """
    Suppress src.* INFO logging for the duration of a benchmark loop.

    VolSurface.fit() and DeltaHedgeBacktester.run() log a line per call by
    design (useful in a notebook, not across hundreds of timed iterations).
    Each src.* module logger is given its own explicit level by
    utils.get_logger(), so a parent "src" logger's level would NOT
    propagate down — every matching logger is targeted directly instead.
    Restores each logger's original level on exit regardless of how the
    block exits.
    """
    targets = [
        lg for name, lg in logging.Logger.manager.loggerDict.items()
        if name.startswith("src.") and isinstance(lg, logging.Logger)
    ]
    prev_levels = {lg: lg.level for lg in targets}
    for lg in targets:
        lg.setLevel(level)
    try:
        yield
    finally:
        for lg, lvl in prev_levels.items():
            lg.setLevel(lvl)


def print_speedup(label: str, baseline: BenchmarkStats, faster: BenchmarkStats) -> None:
    mult = baseline.mean_us / faster.mean_us if faster.mean_us > 0 else float("inf")
    print(f"  → {label}: {mult:,.1f}× faster "
          f"({_fmt_us(baseline.mean_us)} → {_fmt_us(faster.mean_us)})")


# ──────────────────────────────────────────────────────────────────────────
# Live-data loaders (real Deribit API only — no synthetic fallback)
# ──────────────────────────────────────────────────────────────────────────

def load_live_snapshot(
    currency: str = "BTC",
    max_age_hours: float = 1.0,
    force_refresh: bool = False,
    max_spread_pct: float = 0.40,
) -> pd.DataFrame:
    """
    Return a real, IV-computed Deribit options snapshot.

    Reuses the most recent data/raw/<ccy>_surface_*.parquet cache if it's
    younger than *max_age_hours*; otherwise pulls a fresh one from the live
    API (same as notebooks/01_data_collection.ipynb) and caches it. Raises
    RuntimeError — never returns fabricated data — if Deribit is unreachable
    and no usable cache exists.
    """
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    pattern = f"{currency.lower()}_surface_*.parquet"

    if not force_refresh:
        cached = sorted(DATA_RAW.glob(pattern))
        if cached:
            latest = cached[-1]
            age_h = (time.time() - latest.stat().st_mtime) / 3600.0
            if age_h <= max_age_hours:
                snap = pd.read_parquet(latest)
                print(f"[benchmarks] using cached {latest.name} "
                      f"({age_h:.2f}h old, {len(snap)} rows)")
                return snap

    print(f"[benchmarks] fetching live {currency} snapshot from Deribit …")
    try:
        client = DeribitClient()
        raw = client.get_vol_surface_snapshot(currency, max_spread_pct=max_spread_pct)
    except Exception as exc:
        raise RuntimeError(
            f"Live fetch of the {currency} snapshot failed ({exc!r}) and no "
            f"fresh cache exists in {DATA_RAW}. Benchmarks never substitute "
            "fabricated data — check network connectivity / Deribit status, "
            "or run notebooks/01_data_collection.ipynb once to seed a cache."
        ) from exc

    if raw.empty:
        raise RuntimeError(
            f"Deribit returned zero {currency} options passing quality filters. "
            "Benchmarks never substitute fabricated data — check Deribit status."
        )

    snap = compute_iv_surface(raw)
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = DATA_RAW / f"{currency.lower()}_surface_{ts_str}.parquet"
    snap.to_parquet(out_path, index=False)
    print(f"[benchmarks] cached fresh snapshot → {out_path.name} ({len(snap)} rows)")
    return snap


def load_live_price_history(
    instrument_name: str = "BTC-PERPETUAL",
    days: float = 14.0,
    resolution: str = "60",
) -> pd.DataFrame:
    """Real OHLCV history for benchmark_backtester.py — see get_price_history()."""
    client = DeribitClient()
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=days)
    hist = client.get_price_history(instrument_name, start=start, end=end, resolution=resolution)
    if hist.empty:
        raise RuntimeError(
            f"Deribit returned no price history for {instrument_name} over the "
            f"last {days:g} days. Benchmarks never substitute a simulated path "
            "— check network connectivity / Deribit status."
        )
    return hist


def load_live_dvol(currency: str = "BTC") -> pd.Series:
    """Real DVOL history — used to source a real entry IV for backtester benchmarks."""
    client = DeribitClient()
    dvol = client.get_historical_volatility(currency)
    if dvol.empty:
        raise RuntimeError(
            f"Deribit returned no DVOL history for {currency}. Benchmarks never "
            "substitute a hand-picked IV — check network connectivity / Deribit status."
        )
    return dvol
