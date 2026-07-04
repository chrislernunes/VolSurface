"""Standalone performance benchmarks for VolSurface's numerical core.

Run individually (`python benchmarks/benchmark_pricing.py`) or all together
via `python benchmarks/run.py`. Every benchmark sources its inputs from a
real (cached-or-live) Deribit snapshot — see `_common.py` — never synthetic
parameters.
"""
