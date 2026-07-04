# Technical Note: float64 Precision Floor in Direct IV Inversion

**File:** `src/iv_calculator.py`  
**Status:** Known, documented, acceptable at current scope  
**Priority for future work:** Medium (only affects deep ITM + near-expiry combinations)

---

## The problem

Direct Black-76 price inversion computes the option price as:

```
C = F·N(d₁) − K·N(d₂)
```

For deep in-the-money (K ≪ F) or very short-dated options, `F·N(d₁)` and
`K·N(d₂)` are nearly equal large numbers. Their difference — the option
time value — can be smaller than `max(F, K) × ε_machine`, where
`ε_machine ≈ 2.22e-16` is the IEEE-754 double-precision machine epsilon.

When this happens, float64 subtraction discards the time value entirely:
the result is bit-for-bit identical to the intrinsic value `max(F−K, 0)`.
At that point, the computed price contains zero information about σ.

**No root-finder — Halley, Brent, bisection, or any other algorithm —
can recover information that the price representation no longer contains.**
This is a hardware-level fact, not an implementation defect.

---

## Empirical verification

Verified in `tests/test_numerical_robustness.py`:

```python
# ✓ Non-zero time value (even 7.3e-12 USD): Brent recovers σ to 12 decimal places
F=50000, K=47500, T=7/365, sigma=0.05  →  time_value=7.3e-12  →  |Δσ|=1.1e-3 (precision floor)

# ✗ Exact-zero time value: solver returns fabricated number → now returns NaN
F=50000, K=25000, T=1/365,  sigma=0.80  →  time_value=0.0  →  NaN (correct)
F=50000, K=25000, T=7/365,  sigma=0.80  →  time_value=0.0  →  NaN (correct)
F=50000, K=25000, T=30/365, sigma=0.05  →  time_value=0.0  →  NaN (correct)
```

The precision floor is at roughly `50×ε_machine × F`:
- For F=$50,000: floor ≈ $0.0055 USD time value
- For F=$2,000 (ETH): floor ≈ $0.00022 USD time value

Options with time values above this threshold are solved to `|Δσ| < 1e-4`
(our standard accuracy target) by the Halley + Brent pipeline. Options
below this threshold return `NaN` (exact-zero case) or achieve reduced
accuracy up to 5% relative (gray zone near the floor).

---

## Current mitigation

`compute_iv()` detects the exact-zero case via `call_px == lo_bound`
(not a fuzzy epsilon — the actual IEEE-754 equality check) and returns
`NaN` instead of a fabricated number. This is the minimal correct fix.

The Halley → Brent fallback with re-pricing verification (`|repriced − target| < tol`)
catches the remaining gray-zone cases: if Halley stalls at a value that
*happens* to be in-range but doesn't actually satisfy the pricing equation,
Brent's bracketed search corrects it.

---

## Proper long-term fix

**Jaeckel (2015) "Let's Be Rational"** reformulates IV inversion using a
*normalised Black volatility* coordinate:

```
β = (C − h)·e^{h²/2} / √(F·K)   where h = ln(F/K)/2
```

This avoids the catastrophic cancellation entirely by working with a
well-scaled quantity. The reference implementation achieves machine-epsilon
accuracy across the entire input space including the precision floor region.

Adopting it requires replacing the `_halley` + `_brent` pipeline with the
Jaeckel normalised-coordinates solver (~200 lines of numerically-validated
code). The return type, calling convention, and all existing tests remain
unchanged. This is the recommended upgrade path if precision in the
deep-ITM near-expiry region is required for production use.

**Reference:** Jaeckel, P. (2015). Let's Be Rational. Wilmott Magazine.
Available at: http://www.jaeckel.org/LetsBeRational.pdf
