# data/

This directory is **populated exclusively by live Deribit API pulls** —
`notebooks/01_data_collection.ipynb`, `benchmarks/_common.py`, or your own
scripts calling `DeribitClient`. Nothing under here is checked in, hand-
written, or fabricated; both subfolders ship empty (`.gitkeep` only) and
that's by design.

If you're seeing this file and `raw/` is empty, that's expected on a fresh
clone — run notebook 01 (or any of the `benchmarks/`) once to pull real
snapshots before running `02_surface_construction.ipynb` /
`03_skew_term_structure.ipynb`, which read the latest cached snapshot
instead of re-fetching every time.

```
data/
├── raw/        btc_surface_<YYYYMMDD_HHMMSS>.parquet — one file per pull,
│               output of DeribitClient.get_vol_surface_snapshot() +
│               compute_iv_surface() (calc_iv, log_moneyness, total_var
│               already computed). Notebooks 02/03 load the most recent one.
└── processed/  reserved for derived artifacts (e.g. multi-day panels) —
                nothing writes here yet.
```

Paths are configurable in `config/config.yaml` under `paths.data_raw` /
`paths.data_processed` if you'd rather point them elsewhere.
