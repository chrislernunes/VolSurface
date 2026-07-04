# VolSurface

**Volatility surface construction and options analytics for Deribit BTC/ETH options.**

[![Tests](https://img.shields.io/badge/tests-1695%20passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-93%25-brightgreen)]()
[![mypy](https://img.shields.io/badge/mypy-0%20errors-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()

---

## What it does

VolSurface ingests live Deribit option chains, solves for implied volatilities using **Halley's cubic-convergence method** (verified against Brent's bracketed fallback), fits a no-arbitrage **SVI smile** to each expiry slice, and exposes a full analytics layer:

| Module | Capability |
|---|---|
| `iv_calculator` | Black-76 pricing · Halley + Brent IV solvers · Numba batch pricer |
| `surface_fit` | SVI (Gatheral 2004) · SABR · cubic spline · 2D surface + arb checks |
| `greeks` | Analytical Greeks Δ Γ ν Θ ρ · Vanna Volga Charm · sticky-Δ |
| `skew_analyzer` | 25Δ/10Δ RR & BF · ATM skew · curvature · SSR · risk-neutral density |
| `term_structure` | Forward vol bootstrap · vol cone · VRP · CubicSpline interpolation |
| `backtester` | Delta-hedge engine · Deribit fee model · P&L attribution |
| `deribit_client` | Live Deribit REST API · retries · rate-limiting · quality filters |

---

## Installation

```bash
git clone https://github.com/chrislernunes/VolSurface.git
cd VolSurface
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate.ps1
pip install -r requirements.txt
pytest tests/ -q        # 1,695 tests · ~75s
```

---

## Quick start

```python
from src import DeribitClient, VolSurface, GreeksCalculator
from src.iv_calculator import compute_iv_surface
from src.skew_analyzer import SkewAnalyzer
from src.term_structure import TermStructureAnalyzer

# 1. Fetch live surface
client  = DeribitClient()
snap    = client.get_vol_surface_snapshot("BTC", max_spread_pct=0.40)
snap    = compute_iv_surface(snap)          # adds calc_iv, log_moneyness, total_var

# 2. Fit SVI
surface = VolSurface().fit(snap)
print(surface.calibration_summary())
# expiry    T_yr   rmse_ivpct  atm_vol  arb_free
# 5JAN25   0.022       0.31    81.2%      True
# 31JAN25  0.107       0.19    78.4%      True
# 28MAR25  0.258       0.16    72.1%      True

# 3. Query vol at arbitrary (K, T, F)
iv = surface.get_iv(K=70_000, T=0.25, F=68_500)

# 4. Greeks
calc = GreeksCalculator()
g    = calc.compute(F=68_500, K=70_000, T=0.25, sigma=iv)
print(f"Δ={g.delta:.4f}  Γ={g.gamma:.6f}  ν=${g.vega:,.0f}  Θ=${g.theta:,.0f}/day")

# 5. Skew
sa  = SkewAnalyzer(surface)
print(sa.summary_table()[["expiry", "atm_vol_pct", "rr_25d_pct", "bf_25d_pct"]])

# 6. Term structure
tsa = TermStructureAnalyzer(surface)
print(tsa.atm_term_structure())
fv  = tsa.forward_vol(T1=0.083, T2=0.25, F=68_500)
```

---

## Mathematical foundations

### IV solver: Halley's method + Brent's fallback

```
σ_{n+1} = σ_n − f / [f' − f·f''/(2f')]
         where f(σ) = BS(σ) − price,  f' = vega,  f'' = volga
```

Cubic convergence: typically 3–5 iterations from a Corrado-Miller (1996) seed.
Brent's method (guaranteed bracketed convergence) runs as fallback whenever
Halley's output fails a re-pricing verification check.

**Known precision floor:** Direct price inversion computes `F·N(d1) − K·N(d2)`.
For deep ITM, short-dated options this subtraction can produce a result that
is bit-for-bit equal to intrinsic in float64. The solver detects this via
exact equality and returns `nan` rather than a fabricated number. Any non-zero
time value — even 1e-77 USD — remains resolvable by Brent's to 10+ decimal
places (verified empirically). See the module docstring in `iv_calculator.py`
and `JAECKEL_NOTE.md` for the full analysis.

### SVI surface parameterisation

```
w(k) = a + b·[ ρ·(k−m) + √((k−m)² + σ²) ]
```

Calibrated slice-by-slice using **Differential Evolution** (global) then
**Nelder-Mead** (local), with inverse-spread-variance weighting. Both
no-arbitrage conditions (Lee moment formula + butterfly density `g(k) ≥ 0`)
are soft-enforced via penalty terms and automatically verified after every
`VolSurface.fit()` call.

### Greeks

Full analytical Black-76 Greeks to third order, each cross-validated against
central finite differences in the test suite:

| First | Second | Third |
|---|---|---|
| Delta, Vega, Theta, Rho | Gamma, Vanna, Volga, Charm, Veta | Speed, Zomma, Color |

---

## Honest limitations

| Area | Current state | Fix / workaround |
|---|---|---|
| **Calendar arb enforcement** | Detected + logged; not enforced across slices | Use `check_calendar_arbitrage()` and re-fit with a tighter penalty on problem slices |
| **IV precision floor** | `nan` returned for prices bit-for-bit equal to intrinsic | Upstream prevention: use sensible strike/tenor coverage that avoids extreme ITM+near-expiry combos |
| **SVI wing extrapolation** | Flat beyond observed strike range | For deep OTM wings as first-class instruments, extend the strike grid or switch to SSVI |
| **SABR short-expiry accuracy** | Hagan approx degrades for T < 1 week at high ν | Use `SplineSlice` for very short tenors |
| **Backtester** | Delta-hedge only; no gamma-scalping, no portfolio optimisation | Extend via `DeltaHedgeBacktester.add_position()` with custom positions |
| **Deribit funding rate** | Rate treated as zero in Black-76 | Pass `carry=funding_rate` to `compute_iv_surface()` |

---

## Architecture

```
src/
├── utils.py             Shared utilities (logging, config, date parsing, math)
├── deribit_client.py    Deribit REST adapter (no auth needed for market data)
├── iv_calculator.py     Black-76 kernel + Halley/Brent IV solvers + Numba batch
├── surface_fit.py       SVI · SABR · spline · VolSurface 2D assembly
├── greeks.py            Analytical + FD Greeks · portfolio aggregation
├── skew_analyzer.py     RR · BF · SSR · risk-neutral density
├── term_structure.py    Forward vol · vol cone · VRP · total-var spline
└── backtester.py        Delta-hedge engine · Deribit fees · P&L attribution

tests/
├── conftest.py                   Shared fixtures (session-scoped VolSurface)
├── test_iv_calculator.py         71 tests  — Black-76 + IV solver + edge cases
├── test_greeks.py                69 tests  — analytical vs FD + properties
├── test_surface_fit.py           37 tests  — SVI arb-free · calibration roundtrip
├── test_backtester.py            61 tests  — fees · settlement · P&L components
├── test_deribit_client.py        69 tests  — fully mocked API coverage
├── test_utils.py                 47 tests  — all helpers + math primitives
├── test_skew_analyzer.py         54 tests  — RR · BF · density · grid
├── test_term_structure.py        61 tests  — forward vol · cone · VRP · summary
├── test_numerical_robustness.py  1222 tests — parametric stress across all regimes
└── test_backtester.py            61 tests
```

---

## Test and coverage

```bash
# Full suite (75s)
pytest tests/ -q

# Coverage
pytest tests/ --cov=src --cov-report=term-missing

# Stress tests only (fast: 17s, 1222 cases)
pytest tests/test_numerical_robustness.py -q

# Type checking
mypy src/ --ignore-missing-imports   # → Success: no issues found in 9 files
```

**Coverage by module:**

| Module | Coverage | Notes |
|---|---|---|
| `backtester.py` | 100% | |
| `utils.py` | 99% | |
| `skew_analyzer.py` | 98% | |
| `deribit_client.py` | 98% | |
| `surface_fit.py` | 88% | Calendar-arb enforcement path; joint SSVI not yet implemented |
| `greeks.py` | 95% | |
| `term_structure.py` | 95% | |
| `iv_calculator.py` | 79% | Numba JIT body not traceable by Python coverage |
| **Total** | **93%** | |

---

## Notebooks

```bash
jupyter lab
```

| Notebook | Contents |
|---|---|
| `01_data_collection.ipynb` | Fetch live Deribit surface · IV crosscheck vs mark_iv |
| `02_surface_construction.ipynb` | SVI fit · Plotly 3D surface · arb diagnostics |
| `03_skew_term_structure.ipynb` | RR / BF term structure · risk-neutral density |
| `04_delta_hedge_backtest.ipynb` | 52-week rolling straddle backtest · P&L attribution |

---

## References

1. Gatheral, J. (2004). *A parsimonious arbitrage-free implied volatility parameterization.* Global Derivatives & Risk Management.
2. Gatheral, J. & Jacquier, A. (2014). *Arbitrage-free SVI volatility surfaces.* Quantitative Finance 14(1), 59–71.
3. Hagan, P.S. et al. (2002). *Managing smile risk.* Wilmott Magazine, 84–108.
4. Corrado, C.J. & Miller, T.W. (1996). *A note on a simple, accurate formula to compute implied standard deviations.* Journal of Banking & Finance 20(3), 595–603.
5. Lee, R.W. (2004). *The moment formula for implied volatility at extreme strikes.* Mathematical Finance 14(3), 469–480.
6. Jaeckel, P. (2015). *Let's Be Rational.* Wilmott Magazine.
7. Derman, E. (1999). *Regimes of Volatility.* Risk Magazine.

---

*Author: Chrisler Nunes — Mumbai · [chrisler.xyz](https://chrisler.xyz)*
