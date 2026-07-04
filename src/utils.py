"""
VolSurface · utils.py
=====================
Shared utilities used across all modules:
  - Structured logging
  - YAML config management (cached singleton)
  - Deribit instrument / expiry parsing
  - Mathematical primitives (N, n, N_inv)
  - Finance helpers (forward price, log-moneyness, delta-to-strike)
  - Validation guards

Author: VolSurface
"""
from __future__ import annotations

import logging
import re
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union, cast

import numpy as np
import yaml
from scipy.stats import norm as _scipy_norm

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

_LOG_FMT = "%(asctime)s │ %(levelname)-8s │ %(name)-22s │ %(message)s"
_LOG_DATE = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a consistently-formatted module logger.

    Safe to call multiple times — handlers are only added once, avoiding
    duplicate log lines in long-running processes.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FMT, _LOG_DATE))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


ROOT_LOG = get_logger("volsurface.root")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_CFG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: Path = _CFG_PATH) -> dict:
    """
    Load and cache the YAML configuration file.

    The result is cached indefinitely for the process lifetime — call
    ``load_config.cache_clear()`` to force a reload (e.g., in tests).
    """
    resolved = Path(path)
    if not resolved.exists():
        ROOT_LOG.warning("Config not found at %s — using empty dict.", resolved)
        return {}
    with open(resolved) as fh:
        return yaml.safe_load(fh) or {}


def get_cfg(dotted_key: str, default=None):
    """
    Dot-notation accessor into the config, e.g. ``get_cfg('api.timeout_seconds')``.
    Returns *default* if any segment is missing.
    """
    node: Any = load_config()
    for part in dotted_key.split("."):
        if not isinstance(node, dict):
            return default
        node = node.get(part)
    return node if node is not None else default


# ─────────────────────────────────────────────────────────────────────────────
# Mathematical primitives
# ─────────────────────────────────────────────────────────────────────────────

def N(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Standard normal CDF  Φ(x)."""
    return cast(Union[float, np.ndarray], _scipy_norm.cdf(x))


def n(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Standard normal PDF  φ(x) = (1/√2π)·exp(−x²/2)."""
    return cast(Union[float, np.ndarray], _scipy_norm.pdf(x))


def N_inv(p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Inverse normal CDF  Φ⁻¹(p).  Used for delta → strike inversion."""
    return cast(Union[float, np.ndarray], _scipy_norm.ppf(p))


# ─────────────────────────────────────────────────────────────────────────────
# Deribit instrument / expiry parsing
# ─────────────────────────────────────────────────────────────────────────────

_MONTH_NUM: dict[str, int] = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
_EXPIRY_RE = re.compile(
    r"(?P<day>\d{1,2})(?P<mon>[A-Za-z]{3})(?P<year>\d{2})",
)


def parse_deribit_expiry(token: str) -> datetime:
    """
    Parse a Deribit expiry token, e.g. ``'27DEC24'`` or ``'5JAN25'``.

    Deribit options expire at **08:00 UTC** on the stated calendar date.

    Raises
    ------
    ValueError
        If the token cannot be matched to the expected pattern.
    """
    m = _EXPIRY_RE.search(token)
    if not m:
        raise ValueError(f"Cannot parse Deribit expiry: {token!r}")
    day = int(m.group("day"))
    mon_str = m.group("mon").upper()
    if mon_str not in _MONTH_NUM:
        raise ValueError(f"Unknown month abbreviation: {mon_str!r} in {token!r}")
    mon = _MONTH_NUM[mon_str]
    year = 2000 + int(m.group("year"))
    return datetime(year, mon, day, 8, 0, 0, tzinfo=timezone.utc)


def time_to_expiry(
    expiry_dt: datetime,
    now: Optional[datetime] = None,
    day_count: float = 365.25,
) -> float:
    """
    Compute time to expiry **T** in years (actual / 365.25 convention).

    Clipped to [1e-8, ∞) so downstream code never divides by zero at expiry.

    Parameters
    ----------
    expiry_dt : timezone-aware UTC datetime of option expiry
    now       : reference timestamp (defaults to ``datetime.now(UTC)``)
    day_count : trading-day convention denominator (365.25 matches Deribit)
    """
    if now is None:
        now = datetime.now(timezone.utc)
    seconds = (expiry_dt - now).total_seconds()
    return max(seconds / (day_count * 86_400.0), 1e-8)


def parse_instrument(name: str) -> dict:
    """
    Decompose a Deribit instrument name into its constituent fields.

    Parameters
    ----------
    name : e.g. ``'BTC-27DEC24-50000-C'`` or ``'ETH-5JAN25-3200-P'``

    Returns
    -------
    dict
        ``currency``, ``expiry_str``, ``expiry_dt``, ``strike``, ``option_type``

    Raises
    ------
    ValueError
        On malformed names.

    Examples
    --------
    >>> parse_instrument("BTC-27DEC24-50000-C")
    {'currency': 'BTC', 'expiry_str': '27DEC24', 'expiry_dt': datetime(...),
     'strike': 50000.0, 'option_type': 'call'}
    """
    parts = name.split("-")
    if len(parts) != 4:
        raise ValueError(
            f"Expected 4 dash-separated parts in instrument name, got: {name!r}"
        )
    currency, expiry_str, strike_str, cp = parts
    try:
        strike = float(strike_str)
    except ValueError:
        raise ValueError(f"Cannot parse strike {strike_str!r} in {name!r}")
    option_type = "call" if cp.upper() == "C" else "put"
    return {
        "currency": currency,
        "expiry_str": expiry_str,
        "expiry_dt": parse_deribit_expiry(expiry_str),
        "strike": strike,
        "option_type": option_type,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Finance helpers
# ─────────────────────────────────────────────────────────────────────────────

def forward_price(
    spot: float,
    rate: float,
    carry: float,
    tte: float,
) -> float:
    """
    Continuous compounding forward:  F = S · exp((r − q) · T).

    Parameters
    ----------
    spot  : spot / index price
    rate  : USD risk-free rate (continuous, annualised)
    carry : cost-of-carry or dividend yield (continuous)
    tte   : time to expiry (years)
    """
    return cast(float, spot * np.exp((rate - carry) * tte))


def log_moneyness(strike: float, fwd: float) -> float:
    """Return k = ln(K / F).  Negative for OTM puts, positive for OTM calls."""
    return cast(float, np.log(strike / fwd))


def delta_to_strike(
    delta_target: float,
    fwd: float,
    iv: float,
    tte: float,
    option_type: str = "call",
) -> float:
    """
    Invert the Black-76 Δ formula to find the strike for a target delta.

    For calls  : Δ = N(d₁)  →  d₁ = N⁻¹(Δ)
    For puts   : Δ = -N(-d₁) so |Δ_put| maps to the same formula via sign flip.

    d₁ = ln(F/K)/(σ√T) + ½σ√T   →   K = F · exp(−d₁·σ√T + ½σ²T)

    Parameters
    ----------
    delta_target : absolute delta (e.g. 0.25 for 25Δ, positive regardless of type)
    fwd          : forward price
    iv           : implied volatility (annualised)
    tte          : time to expiry (years)
    option_type  : 'call' | 'put'

    Returns
    -------
    Strike price corresponding to the target delta.
    """
    if option_type == "put":
        # Put delta is negative; work with magnitude
        delta_target = abs(delta_target)
        # For put: Δ_put = N(d₁) − 1  →  N(d₁) = 1 − |Δ_put|
        n_d1 = 1.0 - delta_target
    else:
        n_d1 = delta_target

    d1 = N_inv(np.clip(n_d1, 1e-6, 1.0 - 1e-6))
    sqT = np.sqrt(tte)
    return float(fwd * np.exp(-d1 * iv * sqT + 0.5 * iv**2 * tte))


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

_IV_LO: float = 1e-4   # 0.01% — below this is numerical noise
_IV_HI: float = 20.0   # 2000% — above this implies bad data


def validate_iv(iv: float, context: str = "") -> float:
    """
    Clip implied volatility to a sensible range.

    Returns ``np.nan`` for non-finite inputs and logs a warning.
    Clips (rather than errors) for out-of-range but finite values.
    """
    tag = f" [{context}]" if context else ""
    if not np.isfinite(iv):
        ROOT_LOG.debug("IV is non-finite%s — returning NaN.", tag)
        return np.nan
    if iv < _IV_LO:
        ROOT_LOG.debug("IV %.6f < floor%.s — clipping.", iv, tag)
        return _IV_LO
    if iv > _IV_HI:
        ROOT_LOG.debug("IV %.2f > ceiling%s — clipping.", iv, tag)
        return _IV_HI
    return float(iv)


def safe_div(num: float, den: float, fallback: float = np.nan) -> float:
    """Return num / den, or *fallback* if denominator is effectively zero."""
    return num / den if abs(den) > 1e-14 else fallback
