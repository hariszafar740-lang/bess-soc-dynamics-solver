# Contributing to BESS SoC & Electro-Thermal Dynamics Solver

Contributions improving physical fidelity, numerical stability, solver speed, or
verification coverage are welcome.

## Engineering & Numerical Standards

1. **SI Units & Sign Convention**:
   - Maintain strict SI units internally (`A`, `V`, `Ω`, `F`, `s`, `J`, `W`, `K`); suffix Celsius helpers explicitly with `_c`.
   - Preserve the discharge-positive sign convention: `I > 0` is discharge, `I < 0` is charge.
2. **Input Validation & Bounds**:
   - Reject non-finite (`NaN`, `±inf`) or out-of-envelope physical inputs with explicit `ValueError` diagnostics rather than silent clipping.
   - Verify analytical conservation bounds, closed-form step limits, and time-step convergence in `tests/`.
3. **Pre-Submission Quality Gate**:
   - Run `ruff check .` (zero warnings/errors).
   - Run `python -m pytest -v` (all tests passing).
   - Run `python main.py` to confirm deterministic benchmark & figure generation.

## Pull Request Workflow

1. Open an issue describing the physical model, numerical method, or bug fix.
2. Create a focused feature branch from `main`.
3. Include unit/analytical regression tests in `tests/` and update `README.md` equations and limitations transparently.
