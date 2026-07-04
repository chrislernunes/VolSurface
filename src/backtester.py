"""
VolSurface · backtester.py
===========================
Event-driven delta-hedge backtest engine with full P&L attribution.

Strategy
--------
Short straddle / strangle delta-hedged continuously (or at fixed intervals).
The engine simulates selling options at implied vol and delta-hedging with
the underlying (perpetual swap or spot).

P&L Components
--------------
  Gamma P&L  : ½ · Γ · (ΔS)² — realised move benefit / cost
  Theta P&L  : Θ · Δt          — daily time decay earned
  Vega P&L   : ν · Δσ          — P&L from change in implied vol
  Delta P&L  : ~ 0 (hedged)
  Fees       : option premium + underlying hedge transaction costs

Transaction Cost Model (Deribit, 2024)
  Options:        0.03% of underlying notional per trade
  Perpetual swap: 0.05% per trade (taker), 0.02% (maker)
  Settlement:     0.015% of underlying notional
  Max option fee: min(fee, 12.5% of option premium)

Performance Metrics
-------------------
  Total P&L, Sharpe ratio, max drawdown, win rate,
  Calmar ratio, average daily P&L, daily vol of P&L

Author: VolSurface
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .greeks import GreeksCalculator, BSGreeks
from .iv_calculator import bs_price, compute_iv, batch_bs_price
from .surface_fit import VolSurface
from .utils import get_logger, get_cfg

LOG = get_logger(__name__)
__all__ = [
    "FeeModel",
    "Position",
    "PnLRecord",
    "DeltaHedgeBacktester",
    "BacktestResult",
    "linear_slippage",
]


_CALL = "call"
_PUT  = "put"


# ─────────────────────────────────────────────────────────────────────────────
# Fee model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FeeModel:
    """
    Deribit fee schedule.

    All fees as fraction of the **underlying** notional (not option premium),
    except where noted.
    """
    option_maker:   float = 0.0003    # 0.03% of underlying
    option_taker:   float = 0.0003    # 0.03% of underlying
    perp_maker:     float = 0.0002    # 0.02% of underlying
    perp_taker:     float = 0.0005    # 0.05% of underlying
    settlement:     float = 0.00015   # 0.015% of underlying at expiry
    max_pct_premium: float = 0.125    # cap: fee ≤ 12.5% of option premium

    @classmethod
    def from_config(cls) -> "FeeModel":
        """Load fee model from config.yaml."""
        return cls(
            option_taker =float(get_cfg("backtester.taker_fee",     0.0003)),
            option_maker =float(get_cfg("backtester.maker_fee",     0.0003)),
            perp_taker   =float(get_cfg("backtester.perp_fee",      0.0005)),
            settlement   =float(get_cfg("backtester.settlement_fee", 0.00015)),
            max_pct_premium=float(get_cfg("backtester.max_fee_pct_premium", 0.125)),
        )

    def option_fee(
        self, underlying: float, n_contracts: float, premium: float, is_maker: bool = False
    ) -> float:
        """
        Compute option transaction fee in USD.

        Parameters
        ----------
        underlying   : current index price
        n_contracts  : number of contracts (1 contract = 1 BTC/ETH)
        premium      : option premium in USD per contract
        is_maker     : True = post limit, False = take liquidity
        """
        rate = self.option_maker if is_maker else self.option_taker
        raw_fee = rate * underlying * abs(n_contracts)
        cap     = self.max_pct_premium * premium * abs(n_contracts)
        return float(min(raw_fee, cap))

    def perp_fee(
        self, underlying: float, n_contracts: float, is_maker: bool = False
    ) -> float:
        """Compute perpetual swap hedge fee in USD."""
        rate = self.perp_maker if is_maker else self.perp_taker
        return float(rate * underlying * abs(n_contracts))

    def settlement_fee(self, underlying: float, n_contracts: float) -> float:
        """Settlement fee paid at option expiry."""
        return float(self.settlement * underlying * abs(n_contracts))


# ─────────────────────────────────────────────────────────────────────────────
# Position
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Position:
    """A single option leg in the portfolio."""
    instrument:  str
    option_type: str         # 'call' | 'put'
    strike:      float
    expiry_T:    float       # initial time-to-expiry (years)
    qty:         float       # signed: negative = short
    entry_price: float       # USD premium paid/received per contract
    entry_iv:    float       # IV at trade entry
    entry_spot:  float       # spot at trade entry
    entry_ts:    pd.Timestamp
    current_T:   float = 0.0  # updated each step
    is_expired:  bool  = False


# ─────────────────────────────────────────────────────────────────────────────
# P&L record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PnLRecord:
    """Single time-step P&L record for the portfolio."""
    timestamp:   pd.Timestamp
    spot:        float
    portfolio_iv: float       # weighted-avg IV of portfolio
    delta_pnl:   float = 0.0  # P&L from delta hedge rebalancing
    gamma_pnl:   float = 0.0  # ½Γ(ΔS)² realised
    theta_pnl:   float = 0.0  # Θ·Δt time decay
    vega_pnl:    float = 0.0  # ν·Δσ vol change P&L
    fee_pnl:     float = 0.0  # negative = cost
    total_pnl:   float = 0.0  # sum of components

    @property
    def net_pnl(self) -> float:
        return self.total_pnl + self.fee_pnl


# ─────────────────────────────────────────────────────────────────────────────
# Slippage model
# ─────────────────────────────────────────────────────────────────────────────

def linear_slippage(
    n_contracts: float,
    underlying:  float,
    bps_per_contract: float = 1.0,
) -> float:
    """
    Linear-in-size slippage model.

    USD slippage = n × underlying × bps_per_contract × 1e-4
    """
    return float(abs(n_contracts) * underlying * bps_per_contract * 1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# DeltaHedgeBacktester
# ─────────────────────────────────────────────────────────────────────────────

class DeltaHedgeBacktester:
    """
    Event-driven delta-hedging backtest engine.

    Workflow
    --------
    1. Enter short straddle (or user-specified positions) at t=0.
    2. At each time step:
       a. Update Greeks with current spot and IV.
       b. If |net_delta| > threshold, rebalance hedge.
       c. Record P&L components.
    3. At expiry: settle options, apply settlement fee.
    4. Report: P&L attribution, Sharpe, max drawdown, etc.

    Parameters
    ----------
    fee_model         : FeeModel (Deribit defaults)
    hedge_threshold   : rebalance when |net_delta| > this value (per contract)
    hedge_frequency_h : also rebalance every N hours regardless of delta
    slippage_bps      : option slippage in BPS per contract
    rate              : risk-free rate (for BS pricing)

    Usage
    -----
    >>> bt = DeltaHedgeBacktester()
    >>> bt.add_position(instrument='BTC-27DEC24-50000-C', qty=-1, ...)
    >>> result = bt.run(price_data, iv_data)
    >>> result.summary()
    """

    def __init__(
        self,
        fee_model:         Optional[FeeModel] = None,
        hedge_threshold:   float = 0.05,
        hedge_frequency_h: float = 1.0,
        slippage_bps:      float = 1.0,
        rate:              float = 0.0,
    ) -> None:
        self.fees              = fee_model or FeeModel.from_config()
        self.hedge_threshold   = hedge_threshold
        self.hedge_freq_h      = hedge_frequency_h
        self.slippage_bps      = slippage_bps
        self.rate              = rate
        self.positions:        List[Position] = []
        self._calc             = GreeksCalculator(r=rate)
        self._hedge_units      = 0.0   # current delta-hedge position (in BTC/ETH)
        self._cash             = 0.0   # cumulative cash from option premia and hedges

    # ── Position management ──────────────────────────────────────────────────

    def add_position(
        self,
        instrument:  str,
        option_type: str,
        strike:      float,
        initial_T:   float,
        qty:         float,
        spot:        float,
        iv:          float,
        timestamp:   pd.Timestamp,
        is_maker:    bool = False,
    ) -> None:
        """
        Add an option leg to the portfolio and record entry P&L.

        Parameters
        ----------
        instrument  : Deribit instrument name
        option_type : 'call' | 'put'
        strike      : option strike
        initial_T   : time to expiry at entry (years)
        qty         : signed quantity (negative = short)
        spot        : spot price at entry
        iv          : implied vol at entry (fractional)
        timestamp   : entry timestamp
        is_maker    : whether entry was a maker order
        """
        price = bs_price(spot, strike, initial_T, iv, self.rate, option_type)

        pos = Position(
            instrument=instrument,
            option_type=option_type,
            strike=strike,
            expiry_T=initial_T,
            current_T=initial_T,
            qty=qty,
            entry_price=price,
            entry_iv=iv,
            entry_spot=spot,
            entry_ts=timestamp,
        )
        self.positions.append(pos)

        # Cash from option premium (short = receive premium)
        premium_cash = -qty * price  # short qty < 0 → receive premium > 0
        fee = self.fees.option_fee(spot, qty, price, is_maker)
        self._cash += premium_cash - fee

        LOG.info(
            "Entered %s %s  qty=%.2f  price=$%.2f  iv=%.1f%%  fee=$%.2f",
            "SHORT" if qty < 0 else "LONG",
            instrument, qty, price, iv * 100, fee,
        )

    def add_short_straddle(
        self,
        strike:    float,
        initial_T: float,
        n_lots:    float,
        spot:      float,
        iv:        float,
        timestamp: pd.Timestamp,
        expiry_str: str = "EXPIRY",
    ) -> None:
        """Convenience: add short call + short put (straddle)."""
        for ot in [_CALL, _PUT]:
            self.add_position(
                instrument=f"SYN-{expiry_str}-{int(strike)}-{ot[0].upper()}",
                option_type=ot,
                strike=strike,
                initial_T=initial_T,
                qty=-n_lots,
                spot=spot,
                iv=iv,
                timestamp=timestamp,
            )

    # ── Core backtest ────────────────────────────────────────────────────────

    def run(
        self,
        price_data: pd.Series,       # daily (or intraday) spot price, datetime index
        iv_data:    Optional[pd.DataFrame] = None,  # optional IV surface over time
        expiry_T_init: float = 0.25,  # initial time to expiry (years)
    ) -> "BacktestResult":
        """
        Run the delta-hedge backtest over a historical price series.

        Parameters
        ----------
        price_data     : pd.Series of spot prices (UTC-indexed, any frequency)
        iv_data        : pd.DataFrame of IV by (timestamp, expiry) — if None,
                         IV is held constant at entry levels
        expiry_T_init  : initial time to expiry for synthetic trades

        Returns
        -------
        BacktestResult with daily P&L and performance metrics.
        """
        if self.positions and price_data.empty:
            raise ValueError("price_data cannot be empty.")

        price_data = price_data.sort_index()
        pnl_records: List[PnLRecord] = []
        prev_spot   = float(price_data.iloc[0])
        prev_ts     = price_data.index[0]
        hours_since_hedge = 0.0

        for _ts, spot in price_data.items():
            ts: pd.Timestamp = pd.Timestamp(_ts)  # type: ignore[arg-type]
            spot = float(spot)
            dt_years = max(
                (ts - prev_ts).total_seconds() / (365.25 * 86400), 1e-8
            )
            dt_hours = dt_years * 365.25 * 24

            # ── Update T for each position ───────────────────────────────
            active = []
            for pos in self.positions:
                pos.current_T = max(pos.current_T - dt_years, 0.0)
                if pos.current_T <= 1e-6:
                    pos.is_expired = True
                active.append(pos)

            active = [p for p in active if not p.is_expired]

            if not active:
                break

            # ── Compute portfolio Greeks ─────────────────────────────────
            net_delta  = 0.0
            net_gamma  = 0.0
            net_vega   = 0.0
            net_theta  = 0.0
            total_iv   = 0.0
            n_legs     = len(active)

            for pos in active:
                # Use entry IV (constant-vol assumption for simplicity;
                # replace with iv_data lookup for vol-surface-aware simulation)
                iv = pos.entry_iv
                total_iv += iv

                g = self._calc.compute(
                    F=spot, K=pos.strike, T=pos.current_T,
                    sigma=iv, option_type=pos.option_type
                )
                net_delta += pos.qty * g.delta
                net_gamma += pos.qty * g.gamma
                net_vega  += pos.qty * g.vega
                net_theta += pos.qty * g.theta

            avg_iv = total_iv / n_legs if n_legs else 0.0
            dS     = spot - prev_spot

            # ── P&L attribution ─────────────────────────────────────────
            gamma_pnl = 0.5 * net_gamma * dS * dS
            theta_pnl = net_theta * dt_years * 365.0   # theta is per calendar day

            # Delta P&L from the *hedge* position: hedge is sized to offset net_delta
            # Hedge gain = -net_delta_before × dS  (hedge offsets option delta)
            delta_pnl = -self._hedge_units * dS

            # Rebalance the hedge if threshold breached or time-based
            hours_since_hedge += dt_hours
            rebalance = (
                abs(net_delta + self._hedge_units) > self.hedge_threshold
                or hours_since_hedge >= self.hedge_freq_h
            )
            hedge_fee = 0.0
            if rebalance:
                target_hedge  = -net_delta
                hedge_change  = target_hedge - self._hedge_units
                hedge_fee     = self.fees.perp_fee(spot, hedge_change)
                hedge_fee    += linear_slippage(hedge_change, spot, self.slippage_bps)
                self._hedge_units = target_hedge
                hours_since_hedge = 0.0

            total = delta_pnl + gamma_pnl + theta_pnl - hedge_fee
            self._cash += total

            rec = PnLRecord(
                timestamp    = ts,
                spot         = spot,
                portfolio_iv = avg_iv,
                delta_pnl    = delta_pnl,
                gamma_pnl    = gamma_pnl,
                theta_pnl    = theta_pnl,
                vega_pnl     = 0.0,   # held constant here; extend with iv_data
                fee_pnl      = -hedge_fee,
                total_pnl    = total,
            )
            pnl_records.append(rec)

            prev_spot = spot
            prev_ts   = ts

        # ── Expire remaining positions ────────────────────────────────────
        final_spot = float(price_data.iloc[-1])
        expiry_pnl = self._settle_positions(final_spot)

        return BacktestResult(
            records   = pnl_records,
            positions = self.positions,
            expiry_pnl= expiry_pnl,
        )

    def _settle_positions(self, spot: float) -> float:
        """Settle all active positions at expiry and apply settlement fees."""
        total = 0.0
        for pos in self.positions:
            # Intrinsic value at expiry
            if pos.option_type == _CALL:
                intrinsic = max(spot - pos.strike, 0.0)
            else:
                intrinsic = max(pos.strike - spot, 0.0)
            # Short position: we pay the intrinsic to the buyer
            pnl      = -pos.qty * intrinsic * (-1)  # short → qty is negative
            fee      = self.fees.settlement_fee(spot, abs(pos.qty))
            total   += pnl - fee
            pos.is_expired = True
        return total


# ─────────────────────────────────────────────────────────────────────────────
# BacktestResult
# ─────────────────────────────────────────────────────────────────────────────

class BacktestResult:
    """
    Container for backtest output with P&L time series and analytics.

    Attributes
    ----------
    pnl_df    : pd.DataFrame — daily/step P&L broken down by component
    positions : list of Position objects
    expiry_pnl: settlement P&L at expiry

    Methods
    -------
    summary()         → dict of performance metrics
    cumulative_pnl()  → pd.Series of cumulative P&L
    sharpe()          → annualised Sharpe ratio
    max_drawdown()    → maximum peak-to-trough P&L drawdown
    """

    def __init__(
        self,
        records:    List[PnLRecord],
        positions:  List[Position],
        expiry_pnl: float = 0.0,
    ) -> None:
        self.positions  = positions
        self.expiry_pnl = expiry_pnl
        self.pnl_df     = self._build_df(records, expiry_pnl)

    # ── Build DataFrame ───────────────────────────────────────────────────────

    @staticmethod
    def _build_df(records: List[PnLRecord], expiry_pnl: float) -> pd.DataFrame:
        if not records:
            return pd.DataFrame()
        data = [
            {
                "timestamp":   r.timestamp,
                "spot":        r.spot,
                "portfolio_iv": r.portfolio_iv,
                "delta_pnl":   r.delta_pnl,
                "gamma_pnl":   r.gamma_pnl,
                "theta_pnl":   r.theta_pnl,
                "vega_pnl":    r.vega_pnl,
                "fee_pnl":     r.fee_pnl,
                "total_pnl":   r.total_pnl,
            }
            for r in records
        ]
        df = pd.DataFrame(data).set_index("timestamp")
        df.loc[df.index[-1], "total_pnl"] += expiry_pnl
        df["cum_pnl"] = df["total_pnl"].cumsum()
        return df

    # ── Cumulative P&L ────────────────────────────────────────────────────────

    def cumulative_pnl(self) -> pd.Series:
        """Cumulative P&L over time."""
        return self.pnl_df["cum_pnl"] if not self.pnl_df.empty else pd.Series(dtype=float)

    # ── Risk metrics ─────────────────────────────────────────────────────────

    def sharpe(self, ann_factor: float = 365.25) -> float:
        """Annualised Sharpe ratio (daily P&L / std of daily P&L × √252)."""
        pnl = self.pnl_df["total_pnl"].dropna()
        if pnl.std() < 1e-8:
            return np.nan
        return float(pnl.mean() / pnl.std() * np.sqrt(ann_factor))

    def max_drawdown(self) -> float:
        """Maximum peak-to-trough cumulative P&L drawdown (absolute USD)."""
        cum = self.cumulative_pnl()
        if cum.empty:
            return 0.0
        roll_max = cum.cummax()
        drawdown = cum - roll_max
        return float(drawdown.min())

    def calmar(self, ann_factor: float = 365.25) -> float:
        """Calmar ratio: annualised return / |max drawdown|."""
        pnl   = self.pnl_df["total_pnl"]
        ann   = float(pnl.sum() * ann_factor / max(len(pnl), 1))
        dd    = abs(self.max_drawdown())
        return ann / dd if dd > 1e-8 else np.nan

    def win_rate(self) -> float:
        """Fraction of time steps with positive P&L."""
        pnl = self.pnl_df["total_pnl"]
        return float((pnl > 0).mean())

    def pnl_attribution(self) -> pd.Series:
        """Total P&L broken down by component (sum over all steps)."""
        cols = ["delta_pnl", "gamma_pnl", "theta_pnl", "vega_pnl", "fee_pnl"]
        return self.pnl_df[[c for c in cols if c in self.pnl_df.columns]].sum()

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Comprehensive performance summary."""
        pnl = self.pnl_df["total_pnl"]
        n   = len(pnl)
        return {
            "total_pnl_usd":  round(float(pnl.sum()), 2),
            "avg_daily_pnl":  round(float(pnl.mean()), 2),
            "daily_pnl_std":  round(float(pnl.std()), 2),
            "sharpe":         round(self.sharpe(), 3),
            "calmar":         round(self.calmar(), 3),
            "max_drawdown":   round(self.max_drawdown(), 2),
            "win_rate":       round(self.win_rate(), 4),
            "n_steps":        n,
            "expiry_pnl":     round(self.expiry_pnl, 2),
            "attribution":    self.pnl_attribution().round(2).to_dict(),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"BacktestResult | Total PnL=${s['total_pnl_usd']:,.0f} | "
            f"Sharpe={s['sharpe']:.2f} | MaxDD=${s['max_drawdown']:,.0f} | "
            f"WinRate={s['win_rate']:.1%} | n={s['n_steps']}"
        )
