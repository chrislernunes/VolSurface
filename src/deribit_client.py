"""
VolSurface · deribit_client.py
================================
Production-grade Deribit REST API client for options market data.

Wraps the public Deribit v2 REST API (no credentials required for public
endpoints) with:
  - Typed dataclasses for tickers and order books
  - Automatic retry with exponential back-off (urllib3 Retry)
  - Polite rate-limiting (≤ 16 req/s, well below Deribit's 20/s ceiling)
  - Response validation and IV unit normalisation (Deribit returns IV in %)
  - Bulk surface snapshot with quality filters

Deribit API reference: https://docs.deribit.com/#public-get_instruments

Author: VolSurface
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .utils import (
    get_logger,
    get_cfg,
    parse_instrument,
    time_to_expiry,
)

LOG = get_logger(__name__)
__all__ = [
    "DeribitClient",
    "OptionTicker",
    "OrderBook",
    "get_index_price_history",
]



# ─────────────────────────────────────────────────────────────────────────────
# Typed data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OptionTicker:
    """
    Full snapshot of a single Deribit option instrument.

    All prices are in **USD** (converted from native Deribit units which
    are expressed as a fraction of the underlying index price).
    """
    instrument_name: str
    currency: str
    expiry_str: str
    strike: float
    option_type: str            # 'call' | 'put'
    tte: float                  # time to expiry (years, actual/365.25)
    underlying_price: float     # index price in USD
    forward_price: float        # Deribit's estimated delivery price
    mark_price_usd: float       # mark_price × underlying_price
    mark_iv: float              # Deribit IV estimate (annualised, fractional)
    bid_iv: float               # bid-side IV (annualised, fractional)
    ask_iv: float               # ask-side IV (annualised, fractional)
    best_bid_usd: float
    best_ask_usd: float
    open_interest: float        # in contracts (1 contract = 1 BTC/ETH)
    volume_usd: float           # 24h notional volume
    # BS greeks (as reported by Deribit)
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    timestamp: pd.Timestamp

    # ──────────────────────────────────────────────────────────────────────
    # Derived properties
    # ──────────────────────────────────────────────────────────────────────

    @property
    def mid_price_usd(self) -> float:
        """Bid-ask midpoint in USD.  Falls back to mark if market is one-sided."""
        if self.best_bid_usd > 0 and self.best_ask_usd > 0:
            return (self.best_bid_usd + self.best_ask_usd) / 2.0
        return self.mark_price_usd

    @property
    def bid_ask_spread_usd(self) -> float:
        return max(self.best_ask_usd - self.best_bid_usd, 0.0)

    @property
    def relative_spread(self) -> float:
        """Bid-ask spread as fraction of mid price.  Used as liquidity filter."""
        mid = self.mid_price_usd
        return self.bid_ask_spread_usd / mid if mid > 1e-8 else np.nan

    @property
    def log_moneyness(self) -> float:
        """k = ln(K / F) using Deribit's forward as F."""
        return float(np.log(self.strike / self.forward_price))

    @property
    def mid_iv(self) -> float:
        """Mid-market IV: average of bid_iv and ask_iv if both valid."""
        if self.bid_iv > 0 and self.ask_iv > 0:
            return (self.bid_iv + self.ask_iv) / 2.0
        return self.mark_iv

    def to_dict(self) -> dict:
        """Serialise to flat dict for DataFrame construction."""
        return {
            "instrument": self.instrument_name,
            "currency": self.currency,
            "expiry": self.expiry_str,
            "strike": self.strike,
            "type": self.option_type,
            "tte": self.tte,
            "spot": self.underlying_price,
            "forward": self.forward_price,
            "mark_price": self.mark_price_usd,
            "mark_iv": self.mark_iv,
            "bid_iv": self.bid_iv,
            "ask_iv": self.ask_iv,
            "mid_iv": self.mid_iv,
            "bid": self.best_bid_usd,
            "ask": self.best_ask_usd,
            "mid": self.mid_price_usd,
            "spread": self.bid_ask_spread_usd,
            "rel_spread": self.relative_spread,
            "open_interest": self.open_interest,
            "volume_usd": self.volume_usd,
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
            "log_moneyness": self.log_moneyness,
            "timestamp": self.timestamp,
        }


@dataclass
class OrderBook:
    """Level-2 order book snapshot for a single instrument."""
    instrument_name: str
    timestamp: pd.Timestamp
    bids: List[tuple[float, float]] = field(default_factory=list)  # [(price, qty), …]
    asks: List[tuple[float, float]] = field(default_factory=list)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None

    @property
    def mid(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2.0
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    def bids_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.bids, columns=["price", "qty"])

    def asks_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.asks, columns=["price", "qty"])


# ─────────────────────────────────────────────────────────────────────────────
# HTTP session factory
# ─────────────────────────────────────────────────────────────────────────────

def _build_session(max_retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    """
    Build a requests.Session with automatic retry on transient errors.

    Retries on:
      - 429 Too Many Requests (rate limit)
      - 500, 502, 503, 504 Server errors
    with exponential back-off: wait = backoff_factor × 2^(retry − 1)
    """
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ─────────────────────────────────────────────────────────────────────────────
# DeribitClient
# ─────────────────────────────────────────────────────────────────────────────

class DeribitClient:
    """
    Thin, type-safe wrapper around the Deribit public REST API v2.

    All market data endpoints are public (no API key required).

    Parameters
    ----------
    testnet          : Use Deribit testnet (test.deribit.com). Default False.
    timeout          : HTTP timeout in seconds.
    rate_limit_delay : Minimum gap between requests (seconds).
    """

    _PROD_BASE  = "https://www.deribit.com/api/v2/public"
    _TEST_BASE  = "https://test.deribit.com/api/v2/public"

    def __init__(
        self,
        testnet: bool = False,
        timeout: Optional[float] = None,
        rate_limit_delay: Optional[float] = None,
    ) -> None:
        self._base = self._TEST_BASE if testnet else self._PROD_BASE
        self._timeout = timeout or float(get_cfg("api.timeout_seconds", 10.0))
        self._delay  = rate_limit_delay or float(get_cfg("api.rate_limit_delay", 0.06))
        self._session = _build_session(
            max_retries=int(get_cfg("api.max_retries", 3)),
            backoff_factor=float(get_cfg("api.retry_backoff", 0.5)),
        )
        self._last_req: float = 0.0
        env = "TESTNET" if testnet else "PROD"
        LOG.info("DeribitClient ready  │  env=%s  base=%s", env, self._base)

    # ──────────────────────────────────────────────────────────────────────
    # Core HTTP
    # ──────────────────────────────────────────────────────────────────────

    def _get(self, endpoint: str, params: dict) -> Any:
        """Throttled GET → parse JSON → return ``result`` payload."""
        # Polite rate limiting
        wait = self._delay - (time.monotonic() - self._last_req)
        if wait > 0:
            time.sleep(wait)

        url = f"{self._base}/{endpoint}"
        LOG.debug("→ GET /%s  %s", endpoint, params)

        resp = self._session.get(url, params=params, timeout=self._timeout)
        resp.raise_for_status()
        self._last_req = time.monotonic()

        body = resp.json()
        if "error" in body:
            raise RuntimeError(
                f"Deribit API error [{endpoint}]: {body['error']}"
            )
        return body.get("result", body)

    # ──────────────────────────────────────────────────────────────────────
    # Instruments
    # ──────────────────────────────────────────────────────────────────────

    def get_instruments(
        self,
        currency: str = "BTC",
        kind: str = "option",
        expired: bool = False,
    ) -> pd.DataFrame:
        """
        Return all active instruments as a DataFrame.

        Columns: instrument_name, currency, expiry_str, expiry_dt, tte,
                 strike, option_type, tick_size, min_trade_amount, contract_size
        """
        raw = self._get("get_instruments", {
            "currency": currency.upper(),
            "kind": kind,
            "expired": str(expired).lower(),
        })
        rows = []
        for inst in raw:
            try:
                meta = parse_instrument(inst["instrument_name"])
            except ValueError:
                continue
            tte = time_to_expiry(meta["expiry_dt"])
            if tte <= 0:
                continue
            rows.append({
                "instrument_name": inst["instrument_name"],
                "currency": meta["currency"],
                "expiry_str": meta["expiry_str"],
                "expiry_dt": meta["expiry_dt"],
                "tte": tte,
                "strike": meta["strike"],
                "option_type": meta["option_type"],
                "tick_size": inst.get("tick_size", np.nan),
                "min_trade_amount": inst.get("min_trade_amount", np.nan),
                "contract_size": inst.get("contract_size", 1.0),
            })

        if not rows:
            LOG.warning("get_instruments: no results for %s %s.", currency, kind)
            return pd.DataFrame()

        df = (
            pd.DataFrame(rows)
            .sort_values(["expiry_dt", "strike"])
            .reset_index(drop=True)
        )
        LOG.info(
            "get_instruments: %d %s options across %d expiries.",
            len(df), currency, df["expiry_str"].nunique(),
        )
        return df

    # ──────────────────────────────────────────────────────────────────────
    # Ticker
    # ──────────────────────────────────────────────────────────────────────

    def get_ticker(self, instrument_name: str) -> OptionTicker:
        """
        Fetch a real-time ticker snapshot for one instrument.

        Prices are converted from Deribit's native fraction-of-underlying
        representation to USD by multiplying by ``index_price``.
        """
        raw = self._get("ticker", {"instrument_name": instrument_name})
        return self._parse_ticker(raw)

    def _parse_ticker(self, raw: dict) -> OptionTicker:
        name = raw["instrument_name"]
        meta = parse_instrument(name)
        tte  = time_to_expiry(meta["expiry_dt"])

        idx  = float(raw.get("index_price", raw.get("underlying_price", 1.0)))
        fwd  = float(raw.get("estimated_delivery_price", idx))

        # Deribit reports IV in percentage — convert to decimal fraction
        def _pct(key: str) -> float:
            v = raw.get(key, 0.0)
            return float(v) / 100.0 if v else 0.0

        mark_iv = _pct("mark_iv")
        bid_iv  = _pct("bid_iv")
        ask_iv  = _pct("ask_iv")

        # Prices: native (fraction of underlying) → USD
        def _usd(key: str) -> float:
            return float(raw.get(key, 0.0)) * idx

        greeks = raw.get("greeks", {})

        return OptionTicker(
            instrument_name=name,
            currency=meta["currency"],
            expiry_str=meta["expiry_str"],
            strike=meta["strike"],
            option_type=meta["option_type"],
            tte=tte,
            underlying_price=idx,
            forward_price=fwd,
            mark_price_usd=_usd("mark_price"),
            mark_iv=mark_iv,
            bid_iv=bid_iv,
            ask_iv=ask_iv,
            best_bid_usd=_usd("best_bid_price"),
            best_ask_usd=_usd("best_ask_price"),
            open_interest=float(raw.get("open_interest", 0.0)),
            volume_usd=float(raw.get("stats", {}).get("volume_usd", 0.0)),
            delta=float(greeks.get("delta", np.nan)),
            gamma=float(greeks.get("gamma", np.nan)),
            vega=float(greeks.get("vega", np.nan)),
            theta=float(greeks.get("theta", np.nan)),
            rho=float(greeks.get("rho", np.nan)),
            timestamp=pd.Timestamp.now("UTC"),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Bulk surface snapshot
    # ──────────────────────────────────────────────────────────────────────

    def get_vol_surface_snapshot(
        self,
        currency: str = "BTC",
        min_oi: float = 0.0,
        min_tte_days: float = 1.0,
        max_tte_years: float = 2.0,
        max_spread_pct: float = 0.40,
    ) -> pd.DataFrame:
        """
        Fetch the complete option surface for *currency* in one call.

        Applies quality filters:
          1. TTE filter: keeps options with ``min_tte_days`` ≤ expiry ≤ ``max_tte_years``
          2. Open-interest filter: removes illiquid instruments
          3. Spread filter: removes options where bid-ask / mid > ``max_spread_pct``
          4. Two-sided market filter: both bid and ask must be positive

        Returns
        -------
        pd.DataFrame — one row per option, ready for ``compute_iv_surface()``.
        """
        instruments = self.get_instruments(currency)
        if instruments.empty:
            return pd.DataFrame()

        min_tte = min_tte_days / 365.25
        mask = (instruments["tte"] >= min_tte) & (instruments["tte"] <= max_tte_years)
        instruments = instruments[mask].reset_index(drop=True)

        LOG.info("Fetching %d %s tickers …", len(instruments), currency)

        rows = []
        skipped_spread = 0
        skipped_market = 0
        failed = 0

        for _, row in instruments.iterrows():
            try:
                ticker = self.get_ticker(row["instrument_name"])
            except Exception as exc:
                LOG.debug("Ticker failed %s: %s", row["instrument_name"], exc)
                failed += 1
                continue

            if ticker.open_interest < min_oi:
                continue
            if ticker.best_bid_usd <= 0 or ticker.best_ask_usd <= 0:
                skipped_market += 1
                continue
            if np.isfinite(ticker.relative_spread) and ticker.relative_spread > max_spread_pct:
                skipped_spread += 1
                continue

            rows.append(ticker.to_dict())

        df = pd.DataFrame(rows)
        LOG.info(
            "Snapshot: %d options │ %d expiries │ skipped_spread=%d "
            "skipped_market=%d  fetch_errors=%d",
            len(df),
            df["expiry"].nunique() if not df.empty else 0,
            skipped_spread, skipped_market, failed,
        )
        if df.empty:
            return df
        return df.sort_values(["expiry", "strike"]).reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────────────
    # Order book
    # ──────────────────────────────────────────────────────────────────────

    def get_order_book(
        self,
        instrument_name: str,
        depth: int = 10,
    ) -> OrderBook:
        """
        Fetch the limit order book for a single instrument.

        Parameters
        ----------
        depth : price levels on each side (1–50)
        """
        raw = self._get("get_order_book", {
            "instrument_name": instrument_name,
            "depth": depth,
        })
        return OrderBook(
            instrument_name=instrument_name,
            timestamp=pd.Timestamp.now("UTC"),
            bids=[(float(b[0]), float(b[1])) for b in raw.get("bids", [])],
            asks=[(float(a[0]), float(a[1])) for a in raw.get("asks", [])],
        )

    # ──────────────────────────────────────────────────────────────────────
    # Index price & DVOL
    # ──────────────────────────────────────────────────────────────────────

    def get_index_price(self, index_name: str = "btc_usd") -> float:
        """Return current USD index price (``'btc_usd'`` or ``'eth_usd'``)."""
        raw = self._get("get_index_price", {"index_name": index_name.lower()})
        return float(raw["index_price"])

    def get_historical_volatility(self, currency: str = "BTC") -> pd.Series:
        """
        Fetch Deribit DVOL (30-day realised vol) history.

        Returns
        -------
        pd.Series  indexed by UTC timestamp, values in **percentage** (annualised).
        """
        raw = self._get("get_historical_volatility", {"currency": currency.upper()})
        if not raw:
            return pd.Series(dtype=float, name=f"{currency}_dvol_pct")
        index = [pd.Timestamp(r[0], unit="ms", tz="UTC") for r in raw]
        values = [float(r[1]) for r in raw]
        return pd.Series(values, index=index, name=f"{currency}_dvol_pct").sort_index()

    def get_futures_term_structure(self, currency: str = "BTC") -> pd.DataFrame:
        """
        Build a futures term structure table from perpetual + dated futures.

        Returns
        -------
        pd.DataFrame  columns: instrument, expiry, tte, mark_price, implied_yield
        """
        raw = self._get("get_instruments", {
            "currency": currency.upper(),
            "kind": "future",
        })
        spot = self.get_index_price(f"{currency.lower()}_usd")
        rows = []
        for inst in raw:
            name = inst["instrument_name"]
            if inst.get("settlement_period") == "perpetual":
                continue
            try:
                ticker_raw = self._get("ticker", {"instrument_name": name})
                mark = float(ticker_raw.get("mark_price", 0.0)) * spot
                meta = parse_instrument(name)
                tte  = time_to_expiry(meta["expiry_dt"])
                # Implied yield from cost-of-carry: F = S·e^{rT}
                implied_yield = np.log(mark / spot) / tte if tte > 0 else np.nan
                rows.append({
                    "instrument": name,
                    "expiry_str": meta["expiry_str"],
                    "tte": tte,
                    "mark_price": mark,
                    "implied_yield": implied_yield,
                })
            except Exception:
                continue
        if not rows:
            return pd.DataFrame()
        return (
            pd.DataFrame(rows)
            .sort_values("tte")
            .reset_index(drop=True)
        )

    def get_index_price_history(
        self,
        index_name:    str = "btc_usd",
        resolution:    str = "1D",
        start_ts:      Optional[int] = None,
        end_ts:        Optional[int] = None,
        days_back:     int = 365,
    ) -> pd.Series:
        """
        Fetch historical index price data via Deribit's TradingView endpoint.

        Parameters
        ----------
        index_name : ``'btc_usd'`` | ``'eth_usd'``
        resolution : candle resolution — ``'1D'`` (daily) | ``'60'`` (hourly) |
                     ``'30'`` | ``'15'`` | ``'5'`` | ``'1'``
        start_ts   : start Unix timestamp in milliseconds (default: days_back ago)
        end_ts     : end Unix timestamp in milliseconds (default: now)
        days_back  : convenience parameter when start_ts is None

        Returns
        -------
        pd.Series  of close prices, UTC-indexed, sorted ascending.
        """
        import time as _time
        now_ms   = int(_time.time() * 1000)
        end_ts   = end_ts   or now_ms
        start_ts = start_ts or (now_ms - days_back * 86_400 * 1000)

        raw = self._get("get_tradingview_chart_data", {
            "instrument_name": index_name.upper().replace("_", ""),
            "start_timestamp": start_ts,
            "end_timestamp":   end_ts,
            "resolution":      resolution,
        })

        if not raw or "ticks" not in raw or not raw["ticks"]:
            LOG.warning("get_index_price_history: empty response for %s.", index_name)
            return pd.Series(dtype=float, name=index_name)

        ticks  = raw["ticks"]           # ms timestamps
        closes = raw.get("close", [])

        if not closes:
            LOG.warning("get_index_price_history: no close prices in response.")
            return pd.Series(dtype=float, name=index_name)

        idx    = pd.to_datetime(ticks, unit="ms", utc=True)
        series = pd.Series(
            [float(c) for c in closes],
            index=idx,
            name=index_name,
        ).sort_index()

        LOG.info(
            "get_index_price_history: %d %s bars (%s → %s).",
            len(series), resolution,
            series.index[0].date(), series.index[-1].date(),
        )
        return series
