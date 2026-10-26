# Part B (Idea 2) baseline sweep: pilot

Produced by `python3 python/baselines.py` and `python3 python/figures.py` from the
datasets in `data/diffusion` and `data/helmholtz` (test split). Iterations to convergence for
standard solvers as the difficulty knob increases; empty `iters` (NaN) = did not converge
within the cap.

| File | Contents |
|---|---|
| `baselines_diffusion.csv` | CG, Jacobi-PCG, symmetric Gauss-Seidel-PCG, AMG-PCG (pyamg smoothed aggregation) to ‖r‖/‖b‖ ≤ 1e-8; channels, inclusions, barriers at contrast 10^0..10^8 and lognormal (exponential covariance) at σ² = 0.1..4 |
| `baselines_helmholtz.csv` | GMRES(200) without preconditioner and with the exact shifted-Laplacian preconditioner (β = 0.5, LU), to 1e-6, cap 1000; constant, layered, smooth, inclusions media at 2..32 wavelengths |
| `baselines.png` | iterations vs. contrast / frequency for both |
| `gallery.png` | example coefficient fields and media |

**Scope (pilot):** diffusion at N = 128 only, 1 test instance per family, no lognormal-Gaussian
family; Helmholtz 2 test instances per medium. The full sweep in `docs/idea2.md` (all families,
N = 64 and 128, all test instances) has not been run yet.

Headline numbers (inclusions): AMG-PCG 8 → 251 iterations from contrast 10^0 to 10^8, SGS-PCG
135 → 1755, Jacobi-PCG 378 → 4882, CG fails from 10^3. Helmholtz with the exact
shifted-Laplacian preconditioner, test instance 0: 12 → 60 (constant), 13 → 95 (layered), 15 → 91
(smooth), 15 → 260 (inclusions) from 2 to 32 wavelengths (instance 1: 12 → 63, 12 → 98, 15 → 96,
14 → 224); unpreconditioned GMRES fails beyond 4 wavelengths.
