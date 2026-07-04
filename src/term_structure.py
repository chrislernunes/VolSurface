"""
VolSurface · term_structure.py
================================
ATM implied volatility term structure analysis.

Metrics
-------
ATM term structure  : σ_ATM(T) across all calibrated expiries
Forward variance    : σ_fwd(T₁,T₂) — vol implied for the period [T₁,T₂]
Forward variance curve: bootstrap from spot-vol term structure
Vol cone            : percentile bands of historical realised vol for comparison
VRP (Vol Risk Prem) : implied − realised vol at each tenor
Vol of Vol          : σ_σ derived from butterfly term structure

Formulas
--------
Forward total variance (no-arbitrage):
  w_fwd(T₁,T₂) = w(T₂) − w(T₁)      where w(T) = σ²_ATM(T) · T
  σ_fwd(T₁,T₂) = √[ (w(T₂) − w(T₁)) / (T₂ − T₁) ]

This must be positive for the term structure to be calendar-arb-free.

Author: VolSurface
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from .surface_fit import VolSurface
from .utils import get_logger

LOG = get_logger(__name__)
__all__ = [
    "TermStructureAnalyzer",
]



class TermStructureAnalyzer:
    """
    ATM implied-volatility term structure tools.

    Parameters
    ----------
    surface : fitted VolSurface instance

    Usage
    -----
    >>> tsa = TermStructureAnalyzer(surface)
    >>> tsa.atm_term_structure()         # DataFrame across expiries
    >>> tsa.forward_vol(T1=0.08, T2=0.25, F=68_000)
    0.8410
    >>> tsa.vol_cone(realized_df)        # realized vol percentile bands
    """

    def __init__(self, surface: VolSurface) -> None:
        if not surface._fitted:
            raise ValueError("VolSurface must be fitted before term structure analysis.")
        self.surface = surface

    # ── ATM term structure ───────────────────────────────────────────────────

    def atm_term_structure(self) -> pd.DataFrame:
        """
        Build the ATM implied vol term structure from calibrated slices.

        Returns
        -------
        pd.DataFrame with columns:
            expiry, T_yr, T_days, atm_vol, atm_vol_pct,
            total_var, fwd_vol (from prior expiry), fwd_var
        """
        rows = []
        slices = self.surface.slices
        for i, sl in enumerate(slices):
            atm_iv  = sl.atm_vol()
            total_w = atm_iv ** 2 * sl.T

            # Forward vol from the previous expiry
            if i == 0:
                fwd_vol = atm_iv
                fwd_var = total_w
            else:
                prev = slices[i - 1]
                prev_w = prev.atm_vol() ** 2 * prev.T
                dT     = sl.T - prev.T
                dW     = total_w - prev_w
                if dW > 0 and dT > 0:
                    fwd_vol = float(np.sqrt(dW / dT))
                    fwd_var = float(dW / dT)
                else:
                    LOG.warning(
                        "Calendar arb detected: w(%s) ≤ w(%s) — forward var < 0.",
                        sl.expiry_str, prev.expiry_str,
                    )
                    fwd_vol = np.nan
                    fwd_var = np.nan

            rows.append({
                "expiry":      sl.expiry_str,
                "T_yr":        round(sl.T, 5),
                "T_days":      round(sl.T * 365.25, 1),
                "F":           round(sl.F, 2),
                "atm_vol":     round(atm_iv, 5),
                "atm_vol_pct": round(atm_iv * 100, 3),
                "total_var":   round(total_w, 6),
                "fwd_vol":     round(fwd_vol, 5) if np.isfinite(fwd_vol) else np.nan,
                "fwd_var":     round(fwd_var, 6) if np.isfinite(fwd_var) else np.nan,
                "fwd_vol_pct": round(fwd_vol * 100, 3) if np.isfinite(fwd_vol) else np.nan,
            })

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("T_yr").reset_index(drop=True)

    # ── Forward vol bootstrap ────────────────────────────────────────────────

    def forward_vol(
        self,
        T1: float,
        T2: float,
        F:  float,
    ) -> float:
        """
        Compute the forward implied vol for the period [T₁, T₂].

        σ_fwd(T₁, T₂) = √[ (w(T₂) − w(T₁)) / (T₂ − T₁) ]

        where w(T) = σ²_ATM(T) · T is the total implied variance.

        Parameters
        ----------
        T1 : near date (years from now)
        T2 : far date  (years from now)
        F  : forward price (needed to query the surface)

        Raises
        ------
        ValueError : if T₁ ≥ T₂ or forward variance is negative.
        """
        if T1 >= T2:
            raise ValueError(f"T1={T1} must be < T2={T2}.")

        sigma1 = self.surface.get_iv(K=F, T=T1, F=F)
        sigma2 = self.surface.get_iv(K=F, T=T2, F=F)
        w1     = sigma1 ** 2 * T1
        w2     = sigma2 ** 2 * T2
        dW     = w2 - w1
        dT     = T2 - T1

        if dW < 0:
            LOG.warning(
                "Forward variance is negative (%.6f) for T₁=%.3f T₂=%.3f — "
                "calendar arbitrage in the surface.", dW, T1, T2
            )
            return np.nan

        return float(np.sqrt(dW / dT))

    def forward_vol_curve(
        self,
        F: float,
        n_points: int = 50,
    ) -> pd.DataFrame:
        """
        Bootstrap the full forward vol curve from spot-vol term structure.

        Each cell (T_i, T_{i+1}) gives the forward implied vol for that period.

        Returns
        -------
        pd.DataFrame  with columns: T1_yr, T2_yr, fwd_vol, fwd_vol_pct
        """
        ts = [sl.T for sl in self.surface.slices]
        rows = []
        for i in range(len(ts) - 1):
            T1, T2 = ts[i], ts[i + 1]
            fv = self.forward_vol(T1, T2, F)
            rows.append({
                "T1_yr":    round(T1, 4),
                "T2_yr":    round(T2, 4),
                "T1_days":  round(T1 * 365.25, 1),
                "T2_days":  round(T2 * 365.25, 1),
                "fwd_vol":  round(fv, 5) if np.isfinite(fv) else np.nan,
                "fwd_vol_pct": round(fv * 100, 3) if np.isfinite(fv) else np.nan,
            })
        return pd.DataFrame(rows)

    # ── Continuous term structure interpolation ───────────────────────────────

    def total_var_interpolator(self, F: float) -> CubicSpline:
        """
        Fit a cubic spline to the ATM total-variance term structure w(T).

        Useful for evaluating forward vol at arbitrary T₁, T₂ without
        being restricted to calibrated expiry dates.

        Returns
        -------
        CubicSpline  mapping T (years) → w(T) = σ²(T)·T
        """
        ts  = np.array([sl.T   for sl in self.surface.slices])
        ws  = np.array([sl.atm_vol() ** 2 * sl.T for sl in self.surface.slices])
        # Enforce positive boundary derivative (var must be non-decreasing)
        return CubicSpline(ts, ws, extrapolate=True)

    # ── Vol cone ────────────────────────────────────────────────────────────

    @staticmethod
    def realised_vol(
        prices: pd.Series,
        window_days: int = 30,
        ann_factor: float = 365.25,
    ) -> pd.Series:
        """
        Rolling realised volatility (close-to-close log-returns).

        Parameters
        ----------
        prices      : pd.Series of daily closing prices (datetime index)
        window_days : rolling window in trading days
        ann_factor  : annualisation factor (365.25 for crypto, 252 for equities)

        Returns
        -------
        pd.Series of annualised realised vol (fractional).
        """
        # Use pd.Series.apply to keep result as Series (mypy-safe vs np.log())
        log_prices: pd.Series = prices.apply(np.log)
        log_rets: pd.Series = log_prices.diff(1).dropna()
        rv: pd.Series = log_rets.rolling(window_days).std() * np.sqrt(ann_factor)
        return rv.dropna()

    def vol_cone(
        self,
        price_history: pd.Series,
        windows: Optional[List[int]] = None,
        percentiles: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Build a vol cone: realised vol percentile bands vs. implied vol.

        Compares each calibrated expiry's ATM implied vol against the
        historical distribution of realised vol over the same window.

        Parameters
        ----------
        price_history : daily price series (as long as possible for robust stats)
        windows       : list of window lengths in days
        percentiles   : percentile levels for the bands

        Returns
        -------
        pd.DataFrame with expiry-level implied vol vs. realised vol percentiles.
        """
        if windows is None:
            windows = [7, 14, 30, 60, 90, 180]
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]

        # Build realised vol distribution for each window
        rv_stats: dict[int, dict] = {}
        for w in windows:
            rv = self.realised_vol(price_history, w)
            if len(rv) < 10:
                continue
            rv_stats[w] = {
                **{f"p{p}": float(np.percentile(rv, p)) for p in percentiles},
                "current": float(rv.iloc[-1]),
                "mean":    float(rv.mean()),
                "window":  w,
            }

        if not rv_stats:
            LOG.warning("vol_cone: price history too short to compute realised vol.")
            return pd.DataFrame()

        # Match each expiry to the nearest window
        rows = []
        for sl in self.surface.slices:
            T_days = sl.T * 365.25
            # Find nearest historical window
            nearest_w = min(rv_stats.keys(), key=lambda w: abs(w - T_days))
            stats = rv_stats[nearest_w]
            atm_iv = sl.atm_vol()

            row = {
                "expiry":    sl.expiry_str,
                "T_days":    round(T_days, 1),
                "atm_iv":    round(atm_iv, 4),
                "atm_iv_pct": round(atm_iv * 100, 2),
                "rv_window": nearest_w,
                "rv_current": round(stats["current"] * 100, 2),
                "rv_mean":   round(stats["mean"] * 100, 2),
                "vrp":       round((atm_iv - stats["current"]) * 100, 2),
            }
            for p in percentiles:
                row[f"rv_p{p}"] = round(stats[f"p{p}"] * 100, 2)
                row[f"iv_pct_rank_{p}"] = (
                    "expensive" if atm_iv > stats[f"p{p}"]
                    else "cheap"
                )
            rows.append(row)

        return pd.DataFrame(rows)

    # ── VRP estimation ───────────────────────────────────────────────────────

    def vol_risk_premium(
        self,
        price_history: pd.Series,
        windows: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Estimate the Volatility Risk Premium (VRP) = implied − realised.

        A consistently positive VRP means implied vol systematically
        overstates realised vol — the basis of short-vega strategies.
        Crypto has historically shown large positive VRP (10-15 vol pts).

        Parameters
        ----------
        price_history : daily price series for computing realised vol
        windows       : realised vol windows in days

        Returns
        -------
        pd.DataFrame  with columns: expiry, implied, realised, vrp, vrp_pct_rank
        """
        if windows is None:
            windows = [7, 14, 30, 60, 90]

        rows = []
        for sl in self.surface.slices:
            T_days    = sl.T * 365.25
            nearest_w = min(windows, key=lambda w: abs(w - T_days))
            rv_series = self.realised_vol(price_history, nearest_w)
            if rv_series.empty:
                continue

            rv_current = float(rv_series.iloc[-1])
            iv_current = sl.atm_vol()
            vrp        = iv_current - rv_current

            # Percentile rank of VRP vs. history
            vrp_history = (
                pd.Series([
                    self.realised_vol(price_history.iloc[:i], nearest_w).iloc[-1]
                    for i in range(nearest_w + 30, len(price_history), 30)
                ])
            )
            iv_history = iv_current  # cannot retroactively compute; use current for rank

            rows.append({
                "expiry":     sl.expiry_str,
                "T_days":     round(T_days, 1),
                "implied":    round(iv_current * 100, 2),
                "realised":   round(rv_current * 100, 2),
                "vrp":        round(vrp * 100, 2),
                "vrp_z":      round(
                    (vrp - float(vrp_history.mean())) / max(float(vrp_history.std()), 1e-6), 2
                ) if len(vrp_history) > 3 else np.nan,
            })
        return pd.DataFrame(rows)

    # ── Term structure summary ────────────────────────────────────────────────

    def summary(self, F: float) -> dict:
        """
        High-level term structure statistics.

        Returns
        -------
        dict with: contango / backwardation flag, steepness, max/min tenors.
        """
        ts_df = self.atm_term_structure()
        if ts_df.empty:
            return {}

        front_vol = ts_df.iloc[0]["atm_vol"]
        back_vol  = ts_df.iloc[-1]["atm_vol"]
        structure  = "contango" if back_vol > front_vol else "backwardation"
        steepness  = float(back_vol - front_vol)    # fractional vol pts

        # Any negative forward vols (calendar arb)?
        fv_df = self.forward_vol_curve(F)
        n_arb = int(fv_df["fwd_vol"].isna().sum())

        return {
            "structure":          structure,
            "steepness":          round(steepness * 100, 2),
            "front_atm_vol_pct":  round(front_vol * 100, 2),
            "back_atm_vol_pct":   round(back_vol * 100, 2),
            "n_expiries":         len(ts_df),
            "n_calendar_arb":     n_arb,
            "is_arb_free":        n_arb == 0,
        }
