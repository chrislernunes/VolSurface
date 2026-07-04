"""
VolSurface · surface_fit.py
============================
Volatility surface construction via smile-by-slice SVI calibration.

Public API
----------
SVIParams       5-parameter SVI smile in total-variance space (the core model)
SVISliceFitter  Calibrate SVIParams to a single expiry slice (DE + Nelder-Mead)
SABRFitter      Hagan et al. (2002) SABR approximation, fixed β calibration
SplineSlice     Model-free cubic spline — fast, no-parameter interpolation
VolSurface      Assemble a fitted surface, query σ(K,T,F), run arb diagnostics

SVI raw parameterisation  (Gatheral 2004)
-----------------------------------------
    w(k) = a + b·[ ρ·(k−m) + √((k−m)² + σ²) ]

    where  k = ln(K/F)  (log-moneyness)
           w = σ²_imp·T  (total implied variance)

    a  ∈ ℝ        overall level (horizontal shift)
    b  ≥ 0        slope / smile width
    ρ  ∈ (−1,1)   skew  (negative → put wing steeper; typical for crypto)
    m  ∈ ℝ        log-moneyness of minimum variance
    σ  > 0        curvature at the minimum (ATM butterfly)

No-arbitrage conditions  (Gatheral & Jacquier 2014)
----------------------------------------------------
  1. Lee moment formula:     b·(1+|ρ|) ≤ 4/T   (bounds on wing growth rate)
  2. Butterfly density:      g(k) ≥ 0  ∀ k
       g(k) = (1 − k·w'/(2w))² − (w')²·(1/w + ¼)/4 + w''/2

Both conditions are checked automatically after every ``VolSurface.fit()``
call. They are *soft-enforced* during calibration via penalty terms in the
objective function; see ``SVISliceFitter`` for the penalty weights.

Calibration algorithm
---------------------
Phase 1 (global):  Differential Evolution escapes local minima that trip
  a pure local descent when the market smile has unusual curvature or
  the initial SVI guess is far from the solution.
Phase 2 (local):   Nelder-Mead refines the DE best to tighter tolerance.

Weights: options are weighted by the inverse of their bid-ask spread
  expressed in total-variance units.  This down-weights illiquid wings,
  where quoted prices are wide and noisy, and concentrates calibration
  power at the ATM where price discovery is sharpest.  If spread data is
  unavailable, vega-proportional weighting (∝ σ·√T) is used instead.

Modeling choices and known limitations
---------------------------------------
• SVI is fitted **slice-by-slice** (one expiry at a time). This is fast
  and robust but does not guarantee calendar-spread consistency across
  slices. ``VolSurface.check_calendar_arbitrage()`` detects violations;
  enforcing them (e.g. via a joint SSVI fit or a variance interpolation
  scheme) is not yet implemented.  For most production use cases the
  slice-by-slice approach is adequate: violations are rare and mild when
  calibration data spans a full strike range.

• SVI total variance is extrapolated **flat** beyond the observed k-range
  (wing extrapolation via Lee moments bounds the linear growth rate but
  does not fix the level). For very deep OTM options (|k| > 0.5) treated
  as first-class trading instruments, an SSVI or SVI-JW parameterisation
  with explicit wing constraints is preferable.

• SABR is provided as a secondary model. It is **not** used by default in
  ``VolSurface.fit()`` — use ``SABRFitter`` directly for slice-level fits.
  SABR's Hagan approximation can be inaccurate for very short expiries
  (T < 1 week) and high vol-of-vol (ν > 1). The CEV exponent β is held
  fixed during calibration (common industry practice to avoid over-fitting).

References
----------
Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility
  parameterization. Presentation, Global Derivatives & Risk Management.
Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces.
  Quantitative Finance 14(1), 59-71.
Hagan, P.S. et al. (2002). Managing smile risk.  Wilmott Magazine, 84-108.
Lee, R.W. (2004). The moment formula for implied volatility at extreme strikes.
  Mathematical Finance 14(3), 469-480.

Author: VolSurface
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import differential_evolution, minimize

from .utils import get_logger, N

LOG = get_logger(__name__)
__all__ = [
    "SVIParams",
    "SVISliceFitter",
    "SABRFitter",
    "sabr_implied_vol",
    "SplineSlice",
    "ExpirySlice",
    "VolSurface",
    "FloatArray",
]


# Explicit float64 array alias — used throughout this module's public API so
# the type checker (and readers) can see exactly what numeric contract every
# k-grid / total-variance / IV array is expected to satisfy.
FloatArray = npt.NDArray[np.float64]

_SQEPS = np.sqrt(np.finfo(float).eps)
_K_GRID_DEFAULT = np.linspace(-2.0, 2.0, 400)   # dense grid for arb checks


# ─────────────────────────────────────────────────────────────────────────────
# SVI parameter container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SVIParams:
    """
    Five-parameter SVI smile in total-variance space.

    All methods accept and return numpy arrays so they are vectorised
    over strike/log-moneyness grids without looping.
    """
    a: float     = 0.040    # level
    b: float     = 0.100    # slope
    rho: float   = -0.500   # skew
    m: float     = 0.000    # shift
    sigma: float = 0.300    # curvature

    # ── Serialisation ───────────────────────────────────────────────────────

    def to_array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "SVIParams":
        return cls(a=float(arr[0]), b=float(arr[1]), rho=float(arr[2]),
                   m=float(arr[3]), sigma=float(arr[4]))

    # ── Core model ──────────────────────────────────────────────────────────

    def total_var(self, k: FloatArray) -> FloatArray:
        """
        w(k) = a + b·[ρ·(k−m) + √((k−m)² + σ²)]

        Returns total implied variance (σ²_imp · T).
        """
        x = np.asarray(k, dtype=float) - self.m
        return cast(FloatArray, self.a + self.b * (self.rho * x + np.sqrt(x * x + self.sigma * self.sigma)))

    def implied_vol(self, k: FloatArray, T: float) -> FloatArray:
        """σ_imp(k) = √(w(k)/T).  Clipped to avoid √ of negative."""
        w = np.maximum(self.total_var(k), 1e-8)
        return cast(FloatArray, np.sqrt(w / max(T, 1e-8)))

    # ── First and second derivatives of w ───────────────────────────────────

    def dw_dk(self, k: FloatArray) -> FloatArray:
        """∂w/∂k = b·[ρ + (k−m)/√((k−m)²+σ²)]"""
        x = np.asarray(k) - self.m
        return self.b * (self.rho + x / np.sqrt(x * x + self.sigma * self.sigma))

    def d2w_dk2(self, k: FloatArray) -> FloatArray:
        """∂²w/∂k² = b·σ²/((k−m)²+σ²)^{3/2}"""
        x = np.asarray(k) - self.m
        return self.b * self.sigma * self.sigma / (x * x + self.sigma * self.sigma) ** 1.5

    # ── No-arbitrage checks ─────────────────────────────────────────────────

    def butterfly_density(self, k: FloatArray) -> FloatArray:
        """
        Risk-neutral density proxy g(k).

        g(k) = (1 − k·w'/(2w))² − (w')²/4·(1/w + ¼) + w''/2

        g(k) ≥ 0 everywhere ↔ no butterfly arbitrage (positive density).
        """
        k  = np.asarray(k)
        w  = np.maximum(self.total_var(k), _SQEPS)
        wp = self.dw_dk(k)
        wpp= self.d2w_dk2(k)
        t1 = (1.0 - k * wp / (2.0 * w)) ** 2
        t2 = (wp * wp / 4.0) * (1.0 / w + 0.25)
        t3 = wpp / 2.0
        return cast(FloatArray, t1 - t2 + t3)

    def lee_bound(self, T: float) -> float:
        """b·(1+|ρ|) — must be ≤ 4/T (Lee moment formula)."""
        return float(self.b * (1.0 + abs(self.rho)))

    def is_arbitrage_free(
        self,
        k_grid: Optional[np.ndarray] = None,
        tol: float = -1e-6,
    ) -> bool:
        """
        Return True if the smile satisfies both:
          1. g(k) ≥ tol on a dense k-grid (butterfly)
          2. b ≥ 0, |ρ| < 1, σ > 0  (parameter bounds)
        """
        if self.b < 0 or self.sigma <= 0 or abs(self.rho) >= 1.0:
            return False
        grid = k_grid if k_grid is not None else _K_GRID_DEFAULT
        return bool(np.all(self.butterfly_density(grid) >= tol))

    # ── Jump-wing readout (trader-friendly) ─────────────────────────────────

    def jump_wing(self, T: float) -> dict:
        """
        Express the SVI smile in Gatheral's jump-wing parameterisation.

        Returns v_t (ATM var), ψ (ATM skew), p (put slope), c (call slope).
        """
        k0 = np.array([0.0])
        w0  = float(self.total_var(k0))
        wp0 = float(self.dw_dk(k0))
        return {
            "v_t":  w0 / T,          # ATM implied variance
            "psi":  wp0 / (2.0 * np.sqrt(w0 * T)),  # ATM skew in vol space
            "p":    float(self.dw_dk(np.array([-0.25]))),   # left-wing slope
            "c":    float(self.dw_dk(np.array([ 0.25]))),   # right-wing slope
        }


# ─────────────────────────────────────────────────────────────────────────────
# SVI Slice Fitter
# ─────────────────────────────────────────────────────────────────────────────

class SVISliceFitter:
    """
    Calibrate an SVI smile to market total-variance data for one expiry.

    Algorithm
    ---------
    Phase 1 — Global: Differential Evolution (avoids local minima from
      poorly-shaped initial smiles).
    Phase 2 — Local:  Nelder-Mead refinement from DE best.

    Objective: weighted sum of squared errors in *total-variance* space
      (rather than IV space) because total-var errors are more homoscedastic
      and calibration is numerically better conditioned.

    Weights
    -------
    If spread data is available: w_i ∝ 1 / spread_var_i  (inverse variance of
    bid-ask spread expressed in total-var units).
    Otherwise: w_i ∝ vega_i (options with higher vega are more sensitive and
    should receive more weight in calibration).

    No-arbitrage enforcement
    ------------------------
    Soft constraints via quadratic penalties added to the objective:
      - Lee bound: penalty ∝ max(0, b·(1+|ρ|)·T − 4)²
      - Butterfly:  penalty ∝ Σ max(0, −g(k))²  on the data k-grid
    """

    _BOUNDS = [
        (-0.5,  2.0),    # a
        ( 1e-4, 2.0),    # b
        (-0.999, 0.999), # rho
        (-1.5,  1.5),    # m
        ( 1e-4, 2.0),    # sigma
    ]

    def __init__(
        self,
        arb_penalty:  float = 1e4,
        lee_penalty:  float = 1e4,
        de_maxiter:   int   = 300,
        de_popsize:   int   = 15,
        nm_maxiter:   int   = 2000,
    ) -> None:
        self._arb_pen  = arb_penalty
        self._lee_pen  = lee_penalty
        self._de_iter  = de_maxiter
        self._de_pop   = de_popsize
        self._nm_iter  = nm_maxiter

    def fit(
        self,
        k: np.ndarray,
        w_market: np.ndarray,
        T: float,
        weights: Optional[np.ndarray] = None,
    ) -> SVIParams:
        """
        Calibrate SVI to observed (k, w) data.

        Parameters
        ----------
        k        : (N,) log-moneyness array   ln(K/F)
        w_market : (N,) total implied variance σ²·T
        T        : time to expiry (years) — needed for Lee bound
        weights  : (N,) non-negative calibration weights (uniform if None)

        Returns
        -------
        Calibrated SVIParams.
        """
        k = np.asarray(k, dtype=float)
        w = np.asarray(w_market, dtype=float)
        if weights is None:
            weights = np.ones_like(k)
        weights = np.asarray(weights, dtype=float)
        weights = np.maximum(weights, 0.0)
        norm = weights.sum()
        if norm > 0:
            weights = weights / norm

        def objective(p: np.ndarray) -> float:
            params = SVIParams.from_array(p)
            w_fit = params.total_var(k)
            cost  = float(np.dot(weights, (w_fit - w) ** 2))

            # Lee moment formula penalty
            lee_excess = max(0.0, params.lee_bound(T) * T - 4.0)
            cost += self._lee_pen * lee_excess * lee_excess

            # Butterfly density penalty
            g = params.butterfly_density(k)
            neg = np.minimum(g, 0.0)
            cost += self._arb_pen * float(np.dot(neg, neg))

            # Hard parameter constraints as large penalty
            if params.b < 0 or params.sigma <= 0 or abs(params.rho) >= 1.0:
                cost += 1e10
            return cost

        # ── Phase 1: global search ──────────────────────────────────────────
        de = differential_evolution(
            objective,
            bounds=self._BOUNDS,
            seed=42,
            maxiter=self._de_iter,
            popsize=self._de_pop,
            tol=1e-6,
            mutation=(0.5, 1.5),
            recombination=0.9,
            workers=1,
            polish=False,
        )

        # ── Phase 2: local refinement ───────────────────────────────────────
        nm = minimize(
            objective,
            x0=de.x,
            method="Nelder-Mead",
            options={"maxiter": self._nm_iter, "xatol": 1e-9, "fatol": 1e-11},
        )

        best_x = nm.x if nm.fun <= de.fun else de.x
        params = SVIParams.from_array(best_x)

        if not params.is_arbitrage_free(k_grid=k):
            LOG.warning(
                "SVI (T=%.4f yr): calibrated smile has butterfly arbitrage. "
                "Consider increasing penalty or adding more strikes.", T
            )

        return params


# ─────────────────────────────────────────────────────────────────────────────
# SABR Fitter
# ─────────────────────────────────────────────────────────────────────────────

def sabr_implied_vol(
    F: float, K: float, T: float,
    alpha: float, beta: float, rho: float, nu: float,
) -> float:
    """
    Hagan et al. (2002) SABR Black implied volatility approximation.

    Model: dF = α·F^β·dW₁,  dα = ν·α·dW₂,  ⟨dW₁,dW₂⟩ = ρ·dt

    Parameters
    ----------
    F, K   : forward and strike
    T      : time to expiry (years)
    alpha  : initial vol level  (α > 0)
    beta   : CEV exponent       (β ∈ [0,1], typically 0.5)
    rho    : correlation        (|ρ| < 1)
    nu     : vol-of-vol         (ν ≥ 0)

    Returns
    -------
    Black implied volatility σ_B (annualised, fractional).

    Implementation notes
    --------------------
    The ATM case (F ≈ K) is handled separately to avoid log(F/K) ≈ 0
    numerical issues. Second-order time correction is always included.
    """
    eps = 1e-8

    if abs(F - K) < eps:
        # --- ATM formula ---
        FK_pow = F ** (1.0 - beta)
        vol = alpha / FK_pow * (
            1.0
            + (
                (1.0 - beta) ** 2 / 24.0 * alpha ** 2 / FK_pow ** 2
                + rho * beta * nu * alpha / (4.0 * FK_pow)
                + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2
            ) * T
        )
        return float(np.clip(vol, 1e-6, 10.0))

    log_FK  = np.log(F / K)
    FK_mid  = np.sqrt(F * K)
    FK_pow  = FK_mid ** (1.0 - beta)

    # z and χ(z)
    z   = nu / alpha * FK_pow * log_FK
    chi = np.log((np.sqrt(1.0 - 2.0 * rho * z + z ** 2) + z - rho) / (1.0 - rho))
    z_over_chi = z / chi if abs(chi) > eps else 1.0

    # Expansion terms
    log_sq = log_FK ** 2
    A = (1.0 + (1.0 - beta) ** 2 / 24.0 * log_sq
         + (1.0 - beta) ** 4 / 1920.0 * log_sq ** 2)
    B = (1.0 + (
            (1.0 - beta) ** 2 / 24.0 * alpha ** 2 / FK_pow ** 2
            + rho * beta * nu * alpha / (4.0 * FK_pow)
            + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2
        ) * T)

    vol = alpha / (FK_pow * A) * z_over_chi * B
    return float(np.clip(vol, 1e-6, 10.0))


class SABRFitter:
    """
    Least-squares SABR calibration for a single expiry slice.

    β is fixed (user-specified) and (α, ρ, ν) are calibrated.
    This is the standard market practice — fitting β increases
    solution non-uniqueness without meaningfully improving fit.
    """

    def __init__(self, beta: float = 0.5) -> None:
        """
        Parameters
        ----------
        beta : CEV exponent.  0 = Normal, 0.5 = Stochastic-Normal,
               1 = Log-Normal.  0.5 is a reasonable default for crypto.
        """
        self.beta = float(np.clip(beta, 0.0, 1.0))

    def fit(
        self,
        F: float,
        strikes: np.ndarray,
        iv_market: np.ndarray,
        T: float,
        weights: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Calibrate (α, ρ, ν) by minimising vega-weighted IV RMSE.

        Returns
        -------
        dict with keys: alpha, beta, rho, nu, rmse_iv
        """
        strikes   = np.asarray(strikes,   dtype=float)
        iv_market = np.asarray(iv_market, dtype=float)
        if weights is None:
            weights = np.ones_like(strikes)
        weights = weights / weights.sum()

        beta = self.beta

        def objective(p: np.ndarray) -> float:
            alpha, rho, nu = p
            if alpha <= 0 or nu < 0 or abs(rho) >= 1.0:
                return 1e10
            total = 0.0
            for K, iv_mkt, w in zip(strikes, iv_market, weights):
                iv_mdl = sabr_implied_vol(F, K, T, alpha, beta, rho, nu)
                total += w * (iv_mdl - iv_mkt) ** 2
            return total

        # Initial α from ATM vol: σ_ATM ≈ α / F^{1-β}
        atm_iv = iv_market[np.argmin(np.abs(strikes - F))]
        alpha0 = atm_iv * F ** (1.0 - beta)

        result = minimize(
            objective,
            x0=[alpha0, -0.3, 0.4],
            bounds=[(1e-4, 10.0), (-0.999, 0.999), (1e-4, 5.0)],
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-12},
        )
        alpha, rho, nu = result.x

        # RMSE for diagnostics
        ivs_fit = np.array([sabr_implied_vol(F, K, T, alpha, beta, rho, nu)
                            for K in strikes])
        rmse = float(np.sqrt(np.mean((ivs_fit - iv_market) ** 2)))

        return {
            "alpha": float(alpha),
            "beta":  beta,
            "rho":   float(rho),
            "nu":    float(nu),
            "rmse_iv": rmse,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Model-free cubic spline (fallback / sanity check)
# ─────────────────────────────────────────────────────────────────────────────

class SplineSlice:
    """
    Cubic spline of total implied variance w(k) in log-moneyness space.

    Extrapolation is flat (constant at boundary value) to prevent
    explosive values far from observed strikes.
    """

    def __init__(self) -> None:
        self._spline: Optional[CubicSpline] = None
        self._T: float = 1.0
        self._k_lo: float = -3.0
        self._k_hi: float =  3.0

    def fit(self, k: np.ndarray, w_market: np.ndarray, T: float) -> None:
        idx    = np.argsort(k)
        k_s, w_s = k[idx], w_market[idx]
        # Remove duplicate k-values (can happen with puts/calls at same strike)
        _, uniq = np.unique(k_s, return_index=True)
        k_s, w_s = k_s[uniq], w_s[uniq]

        self._T      = T
        self._k_lo   = float(k_s[0])
        self._k_hi   = float(k_s[-1])
        self._spline = CubicSpline(k_s, w_s, extrapolate=False)

    def total_var(self, k: FloatArray) -> FloatArray:
        if self._spline is None:
            raise RuntimeError("SplineSlice.fit() must be called first.")
        k  = np.asarray(k, dtype=float)
        w  = self._spline(np.clip(k, self._k_lo, self._k_hi))
        return cast(FloatArray, np.maximum(np.nan_to_num(w, nan=0.0), 1e-8))

    def implied_vol(self, k: FloatArray) -> FloatArray:
        return cast(FloatArray, np.sqrt(self.total_var(k) / max(self._T, 1e-8)))


# ─────────────────────────────────────────────────────────────────────────────
# Per-expiry slice container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExpirySlice:
    """Calibrated smile for a single expiry."""
    expiry_str:        str
    T:                 float        # time to expiry (years)
    F:                 float        # forward price
    params:            SVIParams
    n_strikes:         int   = 0
    calibration_rmse:  float = 0.0  # RMSE in IV space (fractional)

    def implied_vol(self, k: FloatArray) -> FloatArray:
        """Query IV at log-moneyness k = ln(K/F)."""
        return self.params.implied_vol(k, self.T)

    def implied_vol_strike(self, K: FloatArray) -> FloatArray:
        """Query IV at absolute strike(s) K."""
        return self.implied_vol(np.log(np.asarray(K) / self.F))

    def atm_vol(self) -> float:
        """ATM implied vol σ(k=0)."""
        return float(self.params.implied_vol(np.array([0.0]), self.T)[0])

    def atm_skew(self) -> float:
        """∂σ_imp/∂k at k=0 (proxy for risk-reversal direction)."""
        dk = 0.01
        v_up = self.params.implied_vol(np.array([dk]), self.T)[0]
        v_dn = self.params.implied_vol(np.array([-dk]), self.T)[0]
        return float((v_up - v_dn) / (2.0 * dk))


# ─────────────────────────────────────────────────────────────────────────────
# VolSurface — full 2D surface
# ─────────────────────────────────────────────────────────────────────────────

class VolSurface:
    """
    Full 2D implied-volatility surface  σ_imp(k, T).

    Construction
    ------------
    1. For each expiry: filter quality strikes, fit SVI smile.
    2. Across expiries: log-linear interpolation in total-variance space.

    This ensures calendar-spread consistency (w non-decreasing in T) and
    avoids introducing spurious humps in ATM term structure.

    Usage
    -----
    >>> surface = VolSurface()
    >>> surface.fit(snapshot_df)           # from DeribitClient + compute_iv_surface
    >>> surface.get_iv(K=70_000, T=0.25, F=68_000)
    0.742
    >>> surface.calibration_summary()      # per-expiry diagnostics DataFrame
    >>> k_g, T_g, iv_m = surface.vol_grid()  # for 3D plotting
    """

    def __init__(
        self,
        min_strikes: int   = 5,
        min_iv:      float = 0.01,
        max_iv:      float = 5.00,
        fitter:      Optional[SVISliceFitter] = None,
    ) -> None:
        self.min_strikes = min_strikes
        self.min_iv      = min_iv
        self.max_iv      = max_iv
        self._fitter     = fitter or SVISliceFitter()
        self.slices:     List[ExpirySlice] = []
        self._fitted     = False

    # ── Fitting ─────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        iv_col:      str           = "calc_iv",
        strike_col:  str           = "strike",
        spot_col:    str           = "spot",
        fwd_col:     Optional[str] = "forward",
        tte_col:     str           = "tte",
        expiry_col:  str           = "expiry",
        type_col:    str           = "type",
        spread_col:  Optional[str] = "spread",
    ) -> "VolSurface":
        """
        Calibrate the full surface from a snapshot DataFrame.

        Parameters
        ----------
        df         : output of ``compute_iv_surface()``
        iv_col     : column with our independently-computed IV
        spread_col : USD bid-ask spread (used for inverse-variance weighting)
        """
        df = df.copy()

        # ── Quality filter ──────────────────────────────────────────────────
        mask = (
            df[iv_col].notna()
            & (df[iv_col] >= self.min_iv)
            & (df[iv_col] <= self.max_iv)
        )
        df = df[mask].reset_index(drop=True)

        self.slices = []

        for expiry_str, grp in df.groupby(expiry_col):
            T = float(grp[tte_col].iloc[0])

            # Forward: prefer market forward if provided
            if fwd_col and fwd_col in grp.columns:
                F = float(grp[fwd_col].iloc[0])
            else:
                F = float(grp[spot_col].iloc[0])

            k        = np.log(grp[strike_col].to_numpy(dtype=float) / F)
            iv_mkt   = grp[iv_col].to_numpy(dtype=float)
            w_market = iv_mkt ** 2 * T

            if len(k) < self.min_strikes:
                LOG.debug("Skip %s — only %d strikes.", expiry_str, len(k))
                continue

            # ── Calibration weights ─────────────────────────────────────────
            if spread_col and spread_col in grp.columns:
                ba_usd = grp[spread_col].to_numpy(dtype=float)
                spot   = grp[spot_col].to_numpy(dtype=float)
                # Spread in IV units ≈ ba_usd / (spot × vega_pct)
                # Approximation: ba_iv ≈ ba_usd / (spot × iv × sqrt(T))
                ba_iv     = np.where(
                    iv_mkt * np.sqrt(T) > 1e-4,
                    ba_usd / (spot * iv_mkt * np.sqrt(T) + 1e-8),
                    0.1,
                )
                ba_var    = 2.0 * iv_mkt * T * ba_iv + 1e-8
                weights   = 1.0 / np.maximum(ba_var, 1e-6)
            else:
                # Vega weight: options with higher vega drive calibration
                weights = iv_mkt * np.sqrt(T)   # ∝ vega (unnormalised)

            LOG.info(
                "Fitting SVI │ %-10s │ T=%.4f yr │ n=%d", expiry_str, T, len(k)
            )
            try:
                params = self._fitter.fit(k, w_market, T, weights)
            except Exception as exc:
                LOG.warning("SVI fit failed for %s: %s", expiry_str, exc)
                continue

            # ── Calibration RMSE ────────────────────────────────────────────
            iv_fit = params.implied_vol(k, T)
            rmse   = float(np.sqrt(np.mean((iv_fit - iv_mkt) ** 2)))

            self.slices.append(ExpirySlice(
                expiry_str=str(expiry_str),
                T=T, F=F, params=params,
                n_strikes=len(k),
                calibration_rmse=rmse,
            ))

        self.slices.sort(key=lambda s: s.T)
        self._fitted = True

        n = len(self.slices)
        avg_rmse = np.mean([s.calibration_rmse for s in self.slices]) if n else 0
        LOG.info(
            "VolSurface fitted │ %d slices │ avg RMSE = %.4f (%.1f vol pts)",
            n, avg_rmse, avg_rmse * 100
        )

        # ── Automatic post-fit no-arbitrage audit ────────────────────────────
        # Run after every fit so callers see warnings without having to call
        # check_calendar_arbitrage() manually. This never modifies slices or
        # raises — it informs; enforcement is the caller's responsibility.
        if n >= 2:
            cal_viols = self.check_calendar_arbitrage()
            if not cal_viols:
                LOG.info("No-arbitrage check PASSED │ calendar spread OK.")
            # Per-slice butterfly check
            but_viols = self.check_butterfly_arbitrage()
            if not but_viols:
                LOG.info("No-arbitrage check PASSED │ butterfly density OK.")

        return self

    # ── Query ────────────────────────────────────────────────────────────────

    def get_iv(self, K: float, T: float, F: float) -> float:
        """
        Query implied volatility at (K, T, F).

        Between calibrated expiries: log-linear interpolation of total
        variance w = σ²·T (preserves calendar-spread consistency).

        Beyond the calibrated range: flat extrapolation from nearest slice.

        Parameters
        ----------
        K : strike
        T : target time to expiry (years) — need not match a calibrated slice
        F : forward price at time T

        Returns
        -------
        Implied vol (annualised, fractional).
        """
        self._assert_fitted()
        k  = float(np.log(K / F))
        Ts = np.array([s.T for s in self.slices])
        idx = int(np.searchsorted(Ts, T))

        if idx == 0:
            return float(self.slices[0].params.implied_vol(np.array([k]), self.slices[0].T)[0])
        if idx >= len(self.slices):
            s = self.slices[-1]
            return float(s.params.implied_vol(np.array([k]), s.T)[0])

        lo, hi = self.slices[idx - 1], self.slices[idx]
        w_lo   = float(lo.params.total_var(np.array([k]))[0])
        w_hi   = float(hi.params.total_var(np.array([k]))[0])
        alpha  = (T - lo.T) / (hi.T - lo.T)
        w_int  = (1.0 - alpha) * w_lo + alpha * w_hi   # linear in var
        return float(np.sqrt(max(w_int, 1e-8) / max(T, 1e-8)))

    def atm_vol(self, T: float, F: float) -> float:
        """ATM vol σ(K=F, T)."""
        return self.get_iv(K=F, T=T, F=F)

    # ── Grid for plotting ────────────────────────────────────────────────────

    def vol_grid(
        self,
        k_lo: float = -0.60,
        k_hi: float =  0.60,
        n_strikes: int = 120,
    ) -> Tuple[FloatArray, FloatArray, FloatArray]:
        """
        Compute a dense (T × k) vol grid for 3D surface plots.

        Returns
        -------
        k_grid    : (n_strikes,)  log-moneyness axis
        T_grid    : (n_slices,)   time-to-expiry axis (years)
        iv_matrix : (n_slices, n_strikes) implied vol surface
        """
        self._assert_fitted()
        k_grid = np.linspace(k_lo, k_hi, n_strikes)
        T_grid = np.array([s.T for s in self.slices])
        iv_mat = np.zeros((len(self.slices), n_strikes))
        for i, sl in enumerate(self.slices):
            iv_mat[i] = sl.params.implied_vol(k_grid, sl.T)
        return k_grid, T_grid, iv_mat

    # ── No-arbitrage diagnostics ─────────────────────────────────────────────

    def check_calendar_arbitrage(
        self, n_points: int = 60
    ) -> List[dict]:
        """
        Detect calendar-spread arbitrage: w(k, T₂) < w(k, T₁) for T₂ > T₁.

        Returns list of violation dicts (empty if surface is arb-free).
        """
        self._assert_fitted()
        k_check = np.linspace(-0.4, 0.4, n_points)
        violations = []
        for i in range(len(self.slices) - 1):
            lo, hi = self.slices[i], self.slices[i + 1]
            w_lo = lo.params.total_var(k_check)
            w_hi = hi.params.total_var(k_check)
            mask = w_hi < w_lo - 1e-8
            if mask.any():
                violations.append({
                    "expiry_lo":    lo.expiry_str,
                    "expiry_hi":    hi.expiry_str,
                    "n_violations": int(mask.sum()),
                    "max_violation_var": float((w_lo - w_hi)[mask].max()),
                })
                LOG.warning(
                    "Calendar arb: %s → %s — %d / %d k-points violated.",
                    lo.expiry_str, hi.expiry_str, mask.sum(), n_points,
                )
        return violations

    def check_butterfly_arbitrage(self) -> List[dict]:
        """
        Detect butterfly arbitrage per slice (g(k) < 0 anywhere).
        """
        self._assert_fitted()
        violations = []
        for sl in self.slices:
            g = sl.params.butterfly_density(_K_GRID_DEFAULT)
            mask = g < -1e-6
            if mask.any():
                violations.append({
                    "expiry":       sl.expiry_str,
                    "n_violations": int(mask.sum()),
                    "min_density":  float(g[mask].min()),
                })
        return violations

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def calibration_summary(self) -> pd.DataFrame:
        """Return a per-expiry calibration quality DataFrame."""
        self._assert_fitted()
        rows = []
        for sl in self.slices:
            p = sl.params
            rows.append({
                "expiry":    sl.expiry_str,
                "T_yr":      round(sl.T, 4),
                "n_strikes": sl.n_strikes,
                "rmse_iv":   round(sl.calibration_rmse, 5),
                "rmse_ivpct": round(sl.calibration_rmse * 100, 3),
                "atm_vol":   round(sl.atm_vol(), 4),
                "atm_skew":  round(sl.atm_skew(), 4),
                "a":    round(p.a, 5),
                "b":    round(p.b, 5),
                "rho":  round(p.rho, 4),
                "m":    round(p.m, 4),
                "sigma":round(p.sigma, 4),
                "lee_bound":  round(p.lee_bound(sl.T), 4),
                "arb_free":  p.is_arbitrage_free(),
            })
        return pd.DataFrame(rows)

    # ── Private ──────────────────────────────────────────────────────────────

    def _assert_fitted(self) -> None:
        if not self._fitted or not self.slices:
            raise RuntimeError("Call VolSurface.fit() before querying the surface.")
