# VolSurface Benchmarks

Micro-benchmarks for every performance-critical path in the library.

## Quick start

```powershell
cd "C:\Users\User\Desktop\VSC Projects\VolSurface"
.venv\Scripts\Activate.ps1

# Run all benchmarks
python benchmarks/run.py

# Run one category
python benchmarks/run.py pricing
python benchmarks/run.py iv
python benchmarks/run.py greeks
python benchmarks/run.py surface
python benchmarks/run.py backtester
```

## Files

| File | What it benchmarks |
|---|---|
| `benchmark_pricing.py` | `bs_price` scalar · `batch_bs_price` Numba JIT · NumPy vectorised |
| `benchmark_iv.py` | `compute_iv` by regime · Halley vs Brent vs Auto · roundtrip accuracy |
| `benchmark_greeks.py` | `compute()` throughput · per-Greek access · surface batch · portfolio |
| `benchmark_surface.py` | SVI calibration · `VolSurface.fit()` · `get_iv()` query · `vol_grid()` |
| `benchmark_backtester.py` | FeeModel · `run()` loop · settlement · analytics · rolling cycles |
| `run.py` | Master runner — runs all or selected modules, prints summary |

## Typical results (Apple M2, Python 3.11, Numba 0.58)

```
pricing   bs_price scalar     :   1.2 ms   (81 M ops/s)
pricing   batch N=100_000     :   0.8 ms   (120 M ops/s)  ← Numba JIT
pricing   Python loop N=100k  : 120.0 ms   (0.8 M ops/s)
pricing   Speedup              :  150×

iv        compute_iv ATM       :   3.1 ms   (3.2 K ops/s per call)
iv        compute_iv_surface   : 450 ms     (889 options/s)

greeks    compute() call       :   4.8 ms   (10 M ops/s)
greeks    surface 500 options  :  12 ms     (41 K options/s)

surface   SVISliceFitter 20k   : 280 ms     (one slice)
surface   VolSurface 6 expiries: 1.8 s      (6 × 20 strikes)
surface   get_iv() query       :   0.02 ms  (50 K queries/s)

backtest  run() 1 week         :  12 ms     (168 bars, 14 K steps/s)
backtest  run() 1 year         : 680 ms     (8760 bars)
backtest  52 weekly cycles     : 650 ms     (80 cycles/s)
```

## Interpreting results

- **Numba JIT warmup**: The first `batch_bs_price` call triggers LLVM compilation (~1-3s). All subsequent calls are ~150× faster than a Python loop.
- **SVI calibration**: Dominated by Differential Evolution global search. Reducing `de_maxiter` in `config/config.yaml` speeds it up at the cost of calibration quality.
- **`compute_iv` throughput**: Limited by the `scipy.optimize.brentq` call on fallback paths. Halley alone is ~3× faster but less robust.
