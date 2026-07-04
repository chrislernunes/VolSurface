"""
VolSurface · skew_analyzer.py
==============================
Volatility smile analysis: skew metrics, risk reversals, butterfly spreads,
and cross-expiry dynamics.

Metrics
-------
Risk Reversal (RR)
  The vol spread between a call and a put at equal delta distance from ATM.
  RR_{25Δ} = σ(C25Δ) − σ(P25Δ)
  Positive: call wing bid up (bullish skew, common in crypto bull markets).
  Negative: put wing bid up (bearish skew, common in crypto fear).

Butterfly Spread (BF / Strangle)
  Curvature of the smile around ATM.
  BF_{25Δ} = ½·[σ(C25Δ) + σ(P25Δ)] − σ(ATM)
  Positive: wings are more expensive than ATM (kurtosis premium).

Skew Slope
  ∂σ_imp/∂k  evaluated at k = 0 (ATM).
  Read directly from the SVI first derivative w'(0)/(2σ_ATM·√T).

Skew Stickiness Ratio
  Measures how the smile translates vs. the underlying move.
  SSR = 1 implies sticky-delta, SSR = 0 implies sticky-strike.

Skew Term Structure
  How RR and BF evolve across expiries — useful for detecting macro events
  priced into specific calendar dates.

Author: VolSurface
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .surface_fit import VolSurface, ExpirySlice, SVIParams
from .utils import get_logger, delta_to_strike, N_inv

LOG = get_logger(__name__)
__all__ = [
    "SkewAnalyzer",
]


_CALL = "call"
_PUT  = "put"

# Standard delta pillars used across rates, FX, and crypto vol desks
_DELTA_PILLARS = [0.10, 0.25]   # 10Δ and 25Δ


# ─────────────────────────────────────────────────────────────────────────────
# SkewAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class SkewAnalyzer:
    """
    Smile metrics derived from a calibrated VolSurface.

    Computes risk reversals, butterflies, smile slopes, and term-structure
    dynamics for each expiry in the fitted surface.

    Parameters
    ----------
    surface : a fitted VolSurface instance

    Usage
    -----
    >>> sa = SkewAnalyzer(surface)
    >>> sa.risk_reversal(T=0.25, F=68_000, delta=0.25)
    -0.052   # 5.2 vol-point put skew
    >>> sa.summary_table()   # DataFrame across all expiries
    """

    def __init__(self, surface: VolSurface) -> None:
        if not surface._fitted:
            raise ValueError("VolSurface must be fitted before skew analysis.")
        self.surface = surface

    # ── Core metrics ─────────────────────────────────────────────────────────

    def risk_reversal(
        self,
        T: float,
        F: float,
        delta: float = 0.25,
        iv_atm: Optional[float] = None,
    ) -> float:
        """
        Compute the δ-risk-reversal: RR = σ(CδΔ) − σ(PδΔ).

        Uses the smile to find the strike for each delta pillar, then
        queries the vol surface at those strikes.

        Parameters
        ----------
        T     : time to expiry (years)
        F     : forward price
        delta : absolute delta (0.25 = 25Δ, 0.10 = 10Δ)
        iv_atm: ATM vol to use for delta-to-strike inversion (estimated
                from surface if not provided)

        Returns
        -------
        Risk reversal in vol units (fractional, e.g. -0.05 = -5 vol pts).
        Negative ↔ put skew (puts bid up vs. calls).
        """
        if iv_atm is None:
            iv_atm = self.surface.get_iv(F, T, F)
        K_call = delta_to_strike(delta, F, iv_atm, T, _CALL)
        K_put  = delta_to_strike(delta, F, iv_atm, T, _PUT)
        sigma_call = self.surface.get_iv(K_call, T, F)
        sigma_put  = self.surface.get_iv(K_put,  T, F)
        return float(sigma_call - sigma_put)

    def butterfly(
        self,
        T: float,
        F: float,
        delta: float = 0.25,
        iv_atm: Optional[float] = None,
    ) -> float:
        """
        Compute the δ-butterfly spread: BF = ½·[σ(CδΔ) + σ(PδΔ)] − σ(ATM).

        Positive BF ↔ wings more expensive than ATM (leptokurtic density).

        Parameters
        ----------
        T     : time to expiry (years)
        F     : forward price
        delta : absolute delta pillar
        iv_atm: ATM vol (estimated if not provided)
        """
        if iv_atm is None:
            iv_atm = self.surface.get_iv(F, T, F)
        K_call = delta_to_strike(delta, F, iv_atm, T, _CALL)
        K_put  = delta_to_strike(delta, F, iv_atm, T, _PUT)
        sigma_call = self.surface.get_iv(K_call, T, F)
        sigma_put  = self.surface.get_iv(K_put,  T, F)
        return float(0.5 * (sigma_call + sigma_put) - iv_atm)

    def atm_skew(
        self,
        T: float,
        F: float,
        dk: float = 0.01,
    ) -> float:
        """
        Finite-difference ATM skew: ∂σ_imp/∂k|_{k=0}.

        Positive: call wing steeper (unusual). Negative: put wing steeper (typical).

        Parameters
        ----------
        dk : log-moneyness bump for central difference (default 0.01 ≈ 1%)
        """
        k_up = np.log((F * np.exp(dk)) / F)
        k_dn = np.log((F * np.exp(-dk)) / F)
        K_up = F * np.exp(k_up)
        K_dn = F * np.exp(k_dn)
        iv_up = self.surface.get_iv(K_up, T, F)
        iv_dn = self.surface.get_iv(K_dn, T, F)
        return float((iv_up - iv_dn) / (2.0 * dk))

    def smile_curvature(
        self,
        T: float,
        F: float,
        dk: float = 0.05,
    ) -> float:
        """
        Second derivative of σ_imp w.r.t. log-moneyness at k=0.

        ∂²σ/∂k² = [σ(k+dk) − 2σ(k) + σ(k−dk)] / dk²

        Positive curvature ↔ convex smile (typical: wings expensive vs. ATM).
        """
        K_up  = F * np.exp(dk)
        K_dn  = F * np.exp(-dk)
        iv_0  = self.surface.get_iv(F, T, F)
        iv_up = self.surface.get_iv(K_up, T, F)
        iv_dn = self.surface.get_iv(K_dn, T, F)
        return float((iv_up - 2.0 * iv_0 + iv_dn) / dk ** 2)

    # ── Skew stickiness ───────────────────────────────────────────────────────

    def skew_stickiness_ratio(
        self,
        T: float,
        F: float,
        iv_atm: Optional[float] = None,
    ) -> float:
        """
        Skew Stickiness Ratio (SSR).

        Derman (1999): SSR measures how much the ATM vol moves per unit
        spot move, normalised by the skew slope.

        SSR = − (∂σ_ATM / ∂F) / (∂σ_strike / ∂F)

        In practice estimated as:
          SSR = − (∂σ_ATM / ∂k) × (∂k / ∂F) / skew_slope
              ≈ 1 − (∂σ_ATM / ∂F) / (σ / F)   [approximate]

        SSR = 0: sticky strike (ATM vol constant as spot moves)
        SSR = 1: sticky delta  (ATM vol moves with spot to keep delta = 0.5)
        SSR = 2: floating smile (smile moves 2× the moneyness change)

        This implementation uses a simple finite-difference bump.
        """
        if iv_atm is None:
            iv_atm = self.surface.get_iv(F, T, F)

        dF = F * 0.005   # 0.5% spot bump
        iv_atm_up = self.surface.get_iv(F + dF, T, F + dF)  # keep K=F (ATM) concept
        iv_atm_dn = self.surface.get_iv(F - dF, T, F - dF)

        d_iv_atm_dF = (iv_atm_up - iv_atm_dn) / (2.0 * dF)
        skew = self.atm_skew(T, F)

        if abs(skew) < 1e-6:
            return np.nan
        # SSR = (∂σ_ATM/∂F) / (skew × 1/F)
        return float(d_iv_atm_dF / (skew / F))

    # ── Per-slice metrics ────────────────────────────────────────────────────

    def slice_metrics(
        self,
        sl: ExpirySlice,
        deltas: Optional[List[float]] = None,
    ) -> dict:
        """
        Compute all smile metrics for a single ExpirySlice.

        Returns
        -------
        dict with keys: expiry, T, F, atm_vol, skew_slope, curvature,
                        rr_25d, bf_25d, rr_10d, bf_10d, ssr
        """
        if deltas is None:
            deltas = _DELTA_PILLARS

        T, F  = sl.T, sl.F
        iv_atm = sl.atm_vol()
        row: dict = {
            "expiry":       sl.expiry_str,
            "T_yr":         round(T, 4),
            "T_days":       round(T * 365.25, 1),
            "F":            round(F, 2),
            "atm_vol":      round(iv_atm, 4),
            "atm_vol_pct":  round(iv_atm * 100, 2),
            "skew_slope":   round(self.atm_skew(T, F), 4),
            "curvature":    round(self.smile_curvature(T, F), 4),
            "ssr":          round(self.skew_stickiness_ratio(T, F, iv_atm), 3),
        }
        for d in deltas:
            d_pct = int(d * 100)
            row[f"rr_{d_pct}d"]   = round(self.risk_reversal(T, F, d, iv_atm), 4)
            row[f"bf_{d_pct}d"]   = round(self.butterfly(T, F, d, iv_atm), 4)
            row[f"rr_{d_pct}d_pct"] = round(self.risk_reversal(T, F, d, iv_atm) * 100, 2)
            row[f"bf_{d_pct}d_pct"] = round(self.butterfly(T, F, d, iv_atm) * 100, 2)
        return row

    # ── Summary table across all expiries ────────────────────────────────────

    def summary_table(self, deltas: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Build a summary DataFrame of skew metrics for every calibrated expiry.

        Columns include: expiry, T_yr, T_days, F, atm_vol, skew_slope,
                         curvature, rr_25d, bf_25d, rr_10d, bf_10d, ssr

        This is the primary output for desk-level smile reporting.
        """
        rows = []
        for sl in self.surface.slices:
            try:
                row = self.slice_metrics(sl, deltas)
                rows.append(row)
            except Exception as exc:
                LOG.warning("Skew metrics failed for %s: %s", sl.expiry_str, exc)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("T_yr").reset_index(drop=True)

    # ── Vol smile query grid ─────────────────────────────────────────────────

    def smile_grid(
        self,
        expiry_str: str,
        n_points:   int   = 100,
        delta_lo:   float = 0.05,
        delta_hi:   float = 0.95,
    ) -> pd.DataFrame:
        """
        Compute the full smile for one expiry expressed in delta space.

        Returns a DataFrame with columns: delta, strike, iv, total_var,
        log_moneyness — useful for plotting and further analysis.
        """
        sl = next(
            (s for s in self.surface.slices if s.expiry_str == expiry_str), None
        )
        if sl is None:
            raise ValueError(f"Expiry {expiry_str!r} not found in calibrated slices.")

        T, F   = sl.T, sl.F
        iv_atm = sl.atm_vol()

        deltas = np.linspace(delta_lo, delta_hi, n_points)
        rows   = []
        for d in deltas:
            ot = _CALL if d >= 0.5 else _PUT
            K  = delta_to_strike(d if d >= 0.5 else 1.0 - d, F, iv_atm, T, ot)
            k  = float(np.log(K / F))
            iv = float(sl.params.implied_vol(np.array([k]), T)[0])
            rows.append({
                "delta":         round(d, 4),
                "strike":        round(K, 2),
                "log_moneyness": round(k, 5),
                "iv":            round(iv, 5),
                "iv_pct":        round(iv * 100, 3),
                "total_var":     round(iv * iv * T, 6),
            })
        return pd.DataFrame(rows)

    # ── Skew dynamics across time ─────────────────────────────────────────────

    def skew_term_structure(
        self,
        delta: float = 0.25,
    ) -> pd.DataFrame:
        """
        Evolution of RR and BF across all calibrated expiries.

        Useful for detecting:
          - Event risk priced into specific dates (hump in RR term structure)
          - Structural put skew in the crypto market
          - Mean-reversion of skew across the term structure

        Parameters
        ----------
        delta : delta pillar for RR / BF computation

        Returns
        -------
        pd.DataFrame  with T_yr, T_days, atm_vol, rr, bf for each expiry
        """
        rows = []
        for sl in self.surface.slices:
            T, F   = sl.T, sl.F
            iv_atm = sl.atm_vol()
            rr     = self.risk_reversal(T, F, delta, iv_atm)
            bf     = self.butterfly(T, F, delta, iv_atm)
            rows.append({
                "expiry":    sl.expiry_str,
                "T_yr":      round(T, 4),
                "T_days":    round(T * 365.25, 1),
                "atm_vol":   round(iv_atm * 100, 2),
                "rr":        round(rr * 100, 2),
                "bf":        round(bf * 100, 2),
                "skew":      round(self.atm_skew(T, F) * 100, 2),
            })
        return pd.DataFrame(rows).sort_values("T_yr").reset_index(drop=True)

    # ── Model-implied density ────────────────────────────────────────────────

    def risk_neutral_density(
        self,
        expiry_str: str,
        n_points:   int = 200,
        k_lo:       float = -1.5,
        k_hi:       float =  1.5,
    ) -> pd.DataFrame:
        """
        Compute the model-implied (Breeden-Litzenberger) risk-neutral density.

        q(K) = e^{rT} × ∂²C/∂K²

        In log-moneyness space, using the SVI butterfly density g(k) as a proxy:
        q(k) ∝ g(k) / (F × w(k))   (Gatheral 2004, eq. 2.2)

        Parameters
        ----------
        expiry_str : expiry code matching a calibrated slice

        Returns
        -------
        pd.DataFrame with columns: k, K, g (density proxy), iv
        """
        sl = next(
            (s for s in self.surface.slices if s.expiry_str == expiry_str), None
        )
        if sl is None:
            raise ValueError(f"Expiry {expiry_str!r} not in surface.")

        k  = np.linspace(k_lo, k_hi, n_points)
        g  = sl.params.butterfly_density(k)
        iv = sl.params.implied_vol(k, sl.T)
        K  = sl.F * np.exp(k)

        return pd.DataFrame({
            "k":       k,
            "K":       K,
            "density": np.maximum(g, 0.0),
            "iv":      iv,
            "iv_pct":  iv * 100,
        })
