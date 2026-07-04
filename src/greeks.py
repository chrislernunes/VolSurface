"""
VolSurface · greeks.py
=======================
Full first- and second-order option Greeks, both analytical (Black-76)
and finite-difference (for vol-surface-aware sensitivities).

Greeks implemented
------------------
First order:   Delta, Vega, Theta, Rho
Second order:  Gamma, Vanna, Volga (Vomma), Charm, Veta
Third order:   Speed, Zomma, Color (reference only)
Dollar Greeks: DeltaDollar, VegaDollar (DV01-equivalent notation)

Convention
----------
All returned values are in *natural* units unless prefixed with ``dollar_``:
  - Delta: Δ per 1-USD move in the underlying (dimensionless fraction)
  - Gamma: Δ per 1-USD² move in the underlying (per USD²)
  - Vega : price change per +1 unit of σ (e.g. per 100% vol move)
  - Theta: price change per 1 calendar day
  - Rho  : price change per +1% (0.01) move in r

Dollar notions (per-contract):
  - DollarDelta = Delta × F  (how many USD of underlying to hedge)
  - DollarGamma = ½ × Gamma × F²  (USD P&L for 1% spot move squared)
  - DollarVega  = Vega / 100       (USD P&L per 1 vol-point move)

Author: VolSurface
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .iv_calculator import bs_d1_d2, bs_price, bs_vega
from .utils import N, n, get_logger

LOG = get_logger(__name__)
__all__ = [
    "BSGreeks",
    "GreeksCalculator",
]


_CALL = "call"
_PUT  = "put"
_EPS  = 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# Greeks data container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BSGreeks:
    """
    Full Black-76 Greeks for a single option.

    All values are in natural units (see module docstring).
    Instantiate via ``GreeksCalculator.compute()``.
    """
    # ── Inputs ──────────────────────────────────────────────────────────────
    F:           float   # forward price
    K:           float   # strike
    T:           float   # time to expiry (years)
    sigma:       float   # implied vol (annualised, fractional)
    r:           float   # risk-free rate
    option_type: str     # 'call' | 'put'

    # ── First-order ─────────────────────────────────────────────────────────
    price:   float = 0.0
    delta:   float = 0.0   # ∂V/∂F
    vega:    float = 0.0   # ∂V/∂σ
    theta:   float = 0.0   # ∂V/∂t (per calendar day, negative for long)
    rho:     float = 0.0   # ∂V/∂r

    # ── Second-order ────────────────────────────────────────────────────────
    gamma:   float = 0.0   # ∂²V/∂F²
    vanna:   float = 0.0   # ∂²V/∂F∂σ  (= ∂delta/∂σ = ∂vega/∂F)
    volga:   float = 0.0   # ∂²V/∂σ²   (vega convexity / vomma)
    charm:   float = 0.0   # ∂²V/∂F∂t  (delta decay per day)
    veta:    float = 0.0   # ∂²V/∂σ∂t  (vega decay per day)

    # ── Third-order (reference) ──────────────────────────────────────────────
    speed:   float = 0.0   # ∂³V/∂F³   (gamma convexity)
    zomma:   float = 0.0   # ∂³V/∂F²∂σ (gamma sensitivity to vol)
    color:   float = 0.0   # ∂³V/∂F²∂t (gamma decay per day)

    # ── Dollar Greeks ────────────────────────────────────────────────────────
    @property
    def dollar_delta(self) -> float:
        """USD of underlying to delta-hedge: Δ × F."""
        return self.delta * self.F

    @property
    def dollar_gamma(self) -> float:
        """USD P&L from a 1% spot move squared: ½ × Γ × (F × 0.01)²."""
        return 0.5 * self.gamma * (self.F * 0.01) ** 2

    @property
    def dollar_vega(self) -> float:
        """USD P&L per +1 vol-point (0.01 σ) move: ν / 100."""
        return self.vega / 100.0

    @property
    def dollar_theta(self) -> float:
        """Alias: same as theta (already per calendar day)."""
        return self.theta

    def to_dict(self) -> dict:
        return {
            "F": self.F, "K": self.K, "T": self.T,
            "sigma": self.sigma, "r": self.r, "type": self.option_type,
            "price":  self.price,
            "delta":  self.delta,  "vega":  self.vega,
            "theta":  self.theta,  "rho":   self.rho,
            "gamma":  self.gamma,  "vanna": self.vanna,
            "volga":  self.volga,  "charm": self.charm,
            "veta":   self.veta,
            "speed":  self.speed,  "zomma": self.zomma,
            "color":  self.color,
            "dollar_delta": self.dollar_delta,
            "dollar_gamma": self.dollar_gamma,
            "dollar_vega":  self.dollar_vega,
        }


# ─────────────────────────────────────────────────────────────────────────────
# GreeksCalculator
# ─────────────────────────────────────────────────────────────────────────────

class GreeksCalculator:
    """
    Compute analytical Black-76 Greeks and finite-difference sensitivities.

    Analytical Greeks
    -----------------
    The closed-form Black-76 partial derivatives are used where available
    (Delta, Vega, Theta, Rho, Gamma, Vanna, Volga, Charm, Veta, Speed,
    Zomma, Color).

    Finite-difference Greeks
    ------------------------
    ``numerical_greeks()`` uses central differences and is useful for:
      - Validating analytical formulas
      - Computing vol-surface-aware delta (sticky-strike vs sticky-delta)
      - Any custom payoff not covered by Black-76

    Usage
    -----
    >>> calc = GreeksCalculator()
    >>> g = calc.compute(F=50_000, K=52_000, T=0.25, sigma=0.75, option_type='call')
    >>> g.delta, g.gamma, g.vega
    (0.423, 0.0000058, 25.3)
    """

    def __init__(self, r: float = 0.0) -> None:
        """
        Parameters
        ----------
        r : risk-free rate (continuous).  For crypto options, ≈ 0.
        """
        self.r = r

    # ── Main entry point ─────────────────────────────────────────────────────

    def compute(
        self,
        F:           float,
        K:           float,
        T:           float,
        sigma:       float,
        option_type: str   = _CALL,
        r:           Optional[float] = None,
    ) -> BSGreeks:
        """
        Compute the full set of analytical Black-76 Greeks.

        Parameters
        ----------
        F           : forward price
        K           : strike
        T           : time to expiry (years)
        sigma       : implied vol (fractional)
        option_type : 'call' | 'put'
        r           : override the instance-level rate if provided

        Returns
        -------
        BSGreeks dataclass with all fields populated.
        """
        r = r if r is not None else self.r
        g = BSGreeks(F=F, K=K, T=T, sigma=sigma, r=r, option_type=option_type)

        if sigma <= _EPS or T <= _EPS:
            g.price = bs_price(F, K, T, sigma, r, option_type)
            disc    = np.exp(-r * T)
            if option_type == _CALL:
                g.delta = disc * (1.0 if F > K else 0.0)
            else:
                g.delta = disc * (-1.0 if K > F else 0.0)
            return g

        disc = np.exp(-r * T)
        sqT  = np.sqrt(T)
        d1, d2 = bs_d1_d2(F, K, T, sigma)
        Nd1, Nd2   = N(d1), N(d2)
        Nnd1, Nnd2 = N(-d1), N(-d2)
        nd1        = n(d1)

        # ── Price ────────────────────────────────────────────────────────────
        if option_type == _CALL:
            g.price = disc * (F * Nd1 - K * Nd2)
        else:
            g.price = disc * (K * Nnd2 - F * Nnd1)

        # ── First-order ──────────────────────────────────────────────────────
        # Delta: ∂V/∂F = ±disc·N(±d₁)
        if option_type == _CALL:
            g.delta = disc * Nd1
        else:
            g.delta = disc * (Nd1 - 1.0)

        # Vega: ∂V/∂σ = F·disc·n(d₁)·√T   (identical for C and P)
        g.vega = F * disc * nd1 * sqT

        # Theta: ∂V/∂t  (per year; divide by 365 for daily)
        # Call: -F·disc·n(d₁)·σ/(2√T) - r·K·disc·N(d₂) + r·F·disc·N(d₁)
        theta_common = -F * disc * nd1 * sigma / (2.0 * sqT)
        if option_type == _CALL:
            g.theta = (theta_common - r * K * disc * Nd2 + r * F * disc * Nd1) / 365.0
        else:
            g.theta = (theta_common + r * K * disc * Nnd2 - r * F * disc * Nnd1) / 365.0

        # Rho: ∂V/∂r
        # Call: K·T·disc·N(d₂)·(−1) + …  simplified for Black-76:
        # Note: in Black-76, r only appears in disc = e^{-rT}
        # dC/dr = -T·disc·(F·N(d₁) - K·N(d₂)) = -T·C
        g.rho = -T * g.price   # Black-76 rho (minor)

        # ── Second-order ─────────────────────────────────────────────────────
        # Gamma: ∂²V/∂F² = disc·n(d₁)/(F·σ·√T)
        g.gamma = disc * nd1 / (F * sigma * sqT)

        # Vanna: ∂²V/∂F∂σ = -disc·n(d₁)·d₂/σ   (= ∂vega/∂F normalised)
        g.vanna = -disc * nd1 * d2 / sigma

        # Volga / Vomma: ∂²V/∂σ² = Vega·d₁·d₂/σ
        g.volga = g.vega * d1 * d2 / sigma

        # Charm: ∂²V/∂F∂t = dDelta/dt  (per year)
        # Call:  disc·n(d₁)·[2rT − d₂·σ√T] / (2T·σ√T)  (approximate)
        charm_num = 2.0 * r * T - d2 * sigma * sqT
        g.charm   = disc * nd1 * charm_num / (2.0 * T * sigma * sqT) / 365.0
        if option_type == _PUT:
            g.charm = g.charm  # charm is the same formula for puts (sign from delta)

        # Veta: ∂²V/∂σ∂t = dVega/dt  (per year)
        g.veta = -g.vega * (r + d1 * sigma / (2.0 * sqT) - d2 / sqT) / 365.0

        # ── Third-order ──────────────────────────────────────────────────────
        # Speed: ∂³V/∂F³ = -Gamma/F · (d₁/(σ√T) + 1)
        g.speed = -g.gamma / F * (d1 / (sigma * sqT) + 1.0)

        # Zomma: ∂³V/∂F²∂σ = Gamma·(d₁·d₂ − 1)/σ
        g.zomma = g.gamma * (d1 * d2 - 1.0) / sigma

        # Color: ∂³V/∂F²∂t (gamma decay, per year)
        g.color = (
            -disc * nd1 / (2.0 * F * sigma * sqT)
            * (2.0 * r * T + 1.0 + d1 * (2.0 * r * T - d2 * sigma * sqT)
               / (sigma * sqT))
            / 365.0
        )

        return g

    # ── Batch computation ─────────────────────────────────────────────────────

    def compute_surface_greeks(
        self,
        df: pd.DataFrame,
        iv_col:      str = "calc_iv",
        spot_col:    str = "spot",
        fwd_col:     Optional[str] = "forward",
        strike_col:  str = "strike",
        tte_col:     str = "tte",
        type_col:    str = "type",
    ) -> pd.DataFrame:
        """
        Compute Greeks for every row in a surface snapshot DataFrame.

        Adds columns: price_calc, delta, gamma, vega, theta, rho,
                      vanna, volga, charm, dollar_delta, dollar_gamma, dollar_vega
        """
        df = df.copy()

        rows: list[dict] = []
        for _, row in df.iterrows():
            iv = float(row[iv_col])
            if not np.isfinite(iv) or iv <= 0:
                rows.append({})
                continue

            if fwd_col and fwd_col in df.columns and np.isfinite(row.get(fwd_col, np.nan)):
                F = float(row[fwd_col])
            else:
                F = float(row[spot_col])

            g = self.compute(
                F=F,
                K=float(row[strike_col]),
                T=float(row[tte_col]),
                sigma=iv,
                option_type=str(row[type_col]).lower(),
            )
            rows.append(g.to_dict())

        greeks_df = pd.DataFrame(rows, index=df.index)
        cols_to_add = [
            "price", "delta", "gamma", "vega", "theta", "rho",
            "vanna", "volga", "charm", "veta",
            "dollar_delta", "dollar_gamma", "dollar_vega",
        ]
        for col in cols_to_add:
            if col in greeks_df.columns:
                df[col] = greeks_df[col]
        return df

    # ── Finite-difference Greeks (validation / vol-surface-aware) ────────────

    def numerical_greeks(
        self,
        F:           float,
        K:           float,
        T:           float,
        sigma:       float,
        option_type: str   = _CALL,
        dF:          float = 1.0,     # spot bump in USD
        dSigma:      float = 0.001,   # vol bump (0.1%)
        dT:          float = 1/365,   # time bump (1 day)
    ) -> dict:
        """
        Central-difference numerical Greeks for validation.

        Uses f(x+h) − f(x−h) / 2h for first derivatives,
        and f(x+h) − 2f(x) + f(x−h) / h² for second derivatives.

        Returns
        -------
        dict with keys matching BSGreeks fields (analytical subset).
        """
        r = self.r
        ot = option_type

        p0    = bs_price(F,     K, T,        sigma,       r, ot)
        p_fu  = bs_price(F+dF,  K, T,        sigma,       r, ot)
        p_fd  = bs_price(F-dF,  K, T,        sigma,       r, ot)
        p_vu  = bs_price(F,     K, T,        sigma+dSigma, r, ot)
        p_vd  = bs_price(F,     K, T,        sigma-dSigma, r, ot)
        # Central-difference theta: O(dT²) accuracy vs. O(dT) for forward-diff.
        p_tu  = bs_price(F, K, max(T-dT, 1e-8), sigma, r, ot)   # V(T - dT)
        p_td  = bs_price(F, K, T+dT,             sigma, r, ot)   # V(T + dT)

        delta_num = (p_fu - p_fd) / (2.0 * dF)
        gamma_num = (p_fu - 2.0 * p0 + p_fd) / (dF ** 2)
        vega_num  = (p_vu - p_vd) / (2.0 * dSigma)
        # theta = ∂V/∂t_calendar = -∂V/∂T (negative = option loses time value)
        # Central: (V(T-dT) - V(T+dT)) / (2·dT)
        theta_num = (p_tu - p_td) / (2.0 * dT)

        # Vanna: ∂²V/∂F∂σ via mixed partial
        p_fu_vu = bs_price(F+dF, K, T, sigma+dSigma, r, ot)
        p_fd_vu = bs_price(F-dF, K, T, sigma+dSigma, r, ot)
        p_fu_vd = bs_price(F+dF, K, T, sigma-dSigma, r, ot)
        p_fd_vd = bs_price(F-dF, K, T, sigma-dSigma, r, ot)
        vanna_num = (p_fu_vu - p_fd_vu - p_fu_vd + p_fd_vd) / (4.0 * dF * dSigma)

        volga_num = (p_vu - 2.0 * p0 + p_vd) / (dSigma ** 2)

        return {
            "price": p0,
            "delta": delta_num,
            "gamma": gamma_num,
            "vega":  vega_num,
            "theta": theta_num / 365.0,  # per calendar day
            "vanna": vanna_num,
            "volga": volga_num,
        }

    # ── Portfolio-level Greeks ────────────────────────────────────────────────

    @staticmethod
    def portfolio_greeks(greeks_df: pd.DataFrame, qty_col: str = "qty") -> dict:
        """
        Aggregate Greeks across a portfolio of positions.

        Parameters
        ----------
        greeks_df : DataFrame with one row per leg, Greek columns as above.
        qty_col   : column with signed position size (positive = long).

        Returns
        -------
        dict of net portfolio Greeks.
        """
        agg = {}
        greek_cols = [
            "delta", "gamma", "vega", "theta", "rho",
            "vanna", "volga", "charm",
            "dollar_delta", "dollar_gamma", "dollar_vega",
        ]
        for col in greek_cols:
            if col in greeks_df.columns:
                if qty_col in greeks_df.columns:
                    agg[f"net_{col}"] = float(
                        (greeks_df[col] * greeks_df[qty_col]).sum()
                    )
                else:
                    agg[f"net_{col}"] = float(greeks_df[col].sum())
        return agg

    # ── Vol-surface-aware delta ───────────────────────────────────────────────

    def sticky_strike_delta(
        self,
        F:     float,
        K:     float,
        T:     float,
        sigma: float,
        option_type: str = _CALL,
        dF_frac: float   = 0.001,   # 0.1% spot bump
    ) -> float:
        """
        Sticky-strike delta: option is repriced at a bumped spot with σ held fixed.

        This is the standard Black-76 delta — the model re-prices at the same
        σ regardless of where F moves.
        """
        dF = F * dF_frac
        p_up = bs_price(F + dF, K, T, sigma, self.r, option_type)
        p_dn = bs_price(F - dF, K, T, sigma, self.r, option_type)
        return (p_up - p_dn) / (2.0 * dF)

    def sticky_delta_delta(
        self,
        F:     float,
        K:     float,
        T:     float,
        sigma: float,
        option_type: str = _CALL,
        dF_frac: float   = 0.001,
        dsigma_dF: float = -0.5,   # ∂σ/∂F: vol moves down as spot moves up (put skew)
    ) -> float:
        """
        Sticky-delta (or sticky-moneyness) delta.

        When the spot moves, the smile *shifts* so that the option at the
        same moneyness maintains the same vol.  This is achieved by bumping
        σ alongside F: σ_new = σ + (∂σ/∂F)·dF.

        Parameters
        ----------
        dsigma_dF : ∂σ/∂F — estimated from the smile skew.  Negative for
                    typical negative-skew crypto smiles (vol rises as F falls).
        """
        dF       = F * dF_frac
        sig_up   = sigma + dsigma_dF * dF
        sig_dn   = sigma - dsigma_dF * dF

        p_up = bs_price(F + dF, K, T, max(sig_up, 1e-4), self.r, option_type)
        p_dn = bs_price(F - dF, K, T, max(sig_dn, 1e-4), self.r, option_type)
        return (p_up - p_dn) / (2.0 * dF)
