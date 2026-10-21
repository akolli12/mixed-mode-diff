"""Prompts for the Idea 2 search loop.

The three modes see the same instructions and the same scores; they differ only in what
else they are told after each evaluation:
    llm_only     the LLM's own numbers are used as written; it sees the scores (AlphaEvolve-style)
    score_only   an optimizer tunes the numbers with exact gradients; the LLM also sees the
                 tuned values (Gallego / MadEvolve-style)
    ours         as score_only, plus the gradient report
"""

MODES = ("ours", "score_only", "llm_only")

FAMILIES = {
    "lognormal_exponential": "a lognormal random field, k = exp(σ φ) with φ a Gaussian random field "
                             "(exponential covariance); the contrast grows with σ²",
    "lognormal_gaussian": "a lognormal random field, k = exp(σ φ) with φ a smooth Gaussian random field; "
                          "the contrast grows with σ²",
    "channels": "thin meandering high-conductivity channels (k = 10^p) in a background of k = 1",
    "inclusions": "rectangular high-conductivity inclusions (k = 10^p) in a background of k = 1",
    "barriers": "thin low-conductivity barriers (k = 10^-p) in a background of k = 1",
}

INTERFACES = {
    "precond": """You write a PRECONDITIONER. A fixed flexible conjugate-gradient loop applies it.
Your code must define:
    setup(P, θ) -> S                 # anything precomputed from the problem (any type)
    precond!(z, r, P, S, θ)          # z = M⁻¹ r, written into z in place; return nothing
The preconditioner should approximate A⁻¹ and be (close to) symmetric positive definite.""",
    "solver": """You write an ITERATIVE SOLVER, one iteration at a time.
Your code must define:
    init(P, θ) -> S                  # state carried between iterations (any type)
    step!(x, r, P, S, θ)             # one iteration: update x in place, given the current
                                     # residual r = b - A x (read-only); return nothing
The harness recomputes r after every step and starts from x = 0.""",
}

SYSTEM = """You design iterative methods for linear systems from diffusion problems with highly variable coefficients.

The problem: -div(k grad u) = f on the unit square with u = 0 on the boundary, discretised with cell-centred finite volumes on an N x N grid (N = {N}). This gives A x = b with A sparse, symmetric positive definite and 5-point. The coefficient k is {family_description}. Standard methods (for example Jacobi- or incomplete-factorisation-preconditioned conjugate gradients) need many more iterations as the contrast grows.

The goal: a method that reaches |r|/|b| <= 1e-8 in as few iterations as possible, and whose iteration count stays low as the contrast grows, on instances it has not seen.

{interface}

Inside your code you can use:
    P.N, P.n               grid size N and number of unknowns n = N^2; unknown (i, j) has index i + N*(j-1)
    P.k                    N x N matrix of cell coefficients
    P.tw, P.te, P.ts, P.tn N x N face transmissibilities to the west/east/south/north neighbour
                           (A's off-diagonal entries are their negatives; boundary faces use 2k)
    P.A, P.diag, P.b       the sparse matrix (SparseMatrixCSC), its diagonal, and the right-hand side
    matvec!(y, P, x)       y = A x
    LinearAlgebra          dot, norm, and dense linear algebra on small matrices

Tunable numbers: put every number you want tuned into the vector θ, and define
    const THETA0 = [...]           # starting values (Vector{{Float64}}, at most 20 entries)
    const LOWER = [...]            # lower bounds, same length
    const UPPER = [...]            # upper bounds, same length
    const THETA_NAMES = [...]      # a short name for each entry
{tuning_note}

Rules:
- Plain Julia with only Base and LinearAlgebra: no other packages, no sparse factorisations (lu, cholesky, ldlt of sparse matrices), no try/catch, no global mutable state, no printing or files.
- The code must be differentiable with respect to θ by automatic differentiation (Enzyme): do not branch on values computed from θ, convert θ to integers, or sort by values that depend on θ.
- The cost of one iteration should grow roughly linearly with n: no dense n x n matrices. Small dense matrices (a coarse space of up to a few hundred unknowns) are fine, and setup may precompute anything from P.
- Methods are scored on several instances at each of these contrast levels: {levels}.

Reply with the complete new program in a single ```julia code block. You may explain your idea briefly before it."""

TUNING_NOTES = {
    "llm_only": "Your THETA0 values are used exactly as written; nothing tunes them.",
    "score_only": "After you write the program, an optimizer tunes θ within [LOWER, UPPER] using exact gradients "
                  "of the training loss, so choose sensible ranges rather than exact values.",
    "ours": "After you write the program, an optimizer tunes θ within [LOWER, UPPER] using exact gradients "
            "of the training loss, so choose sensible ranges rather than exact values.",
}


def system_prompt(mode, frame, family, N, levels):
    return SYSTEM.format(N=N, family_description=FAMILIES[family], interface=INTERFACES[frame],
                         tuning_note=TUNING_NOTES[mode], levels=", ".join(f"{lv:g}" for lv in levels))


def _scores_table(val, levels):
    rows = ["contrast level | median iterations to 1e-8 | converged | median |r|/|b| after the cap"]
    for lv in levels:
        k = f"{lv:g}"
        its = val["median_iterations_by_contrast"].get(k)
        conv = val["converged_by_contrast"].get(k, 0)
        tot = val["problems_per_contrast"].get(k, 0)
        res = val["median_final_residual_by_contrast"].get(k)
        mark = "" if conv == tot else " (estimated: not all converged)"
        rows.append(f"{k} | {its:.0f}{mark} | {conv} of {tot} | {res:.1e}" if res is not None
                    else f"{k} | {its:.0f}{mark} | {conv} of {tot} | broke down")
    return "\n".join(rows)


def _tuned_values(ev):
    rows = []
    for name, start, val, lo, hi in zip(ev["names"], ev["theta0"], ev["theta"], ev["lower"], ev["upper"]):
        rows.append(f"  {name} = {val:.6g}   (you set {start:.6g}; range [{lo:.6g}, {hi:.6g}])")
    return "\n".join(rows)


def _gradient_report(ev):
    rep = ev["report"]
    lines = [f"Training loss (mean log10 |r|/|b| after a fixed number of iterations on training instances): "
             f"{rep['train_loss_start']:.3f} at your starting values -> {rep['train_loss_tuned']:.3f} after tuning."]
    lines.append("Per tuned value, at the tuned point (relative sensitivity: |d loss / d value| times its range, "
                 "scaled so the largest is 1):")
    for p in sorted(rep["params"], key=lambda p: -p["relative_sensitivity"]):
        notes = []
        if p["at_bound"] in ("lower", "upper"):
            notes.append(f"stuck at its {p['at_bound']} bound: the loss wants it to go further")
        if p["no_effect"]:
            notes.append("has no effect on the loss")
        direction = "increasing it would lower the loss" if p["grad_per_range"] < 0 else \
                    "decreasing it would lower the loss" if p["grad_per_range"] > 0 else ""
        lines.append(f"  {p['name']}: sensitivity {p['relative_sensitivity']:.2f}"
                     + (f"; {direction}" if direction and not p["no_effect"] else "")
                     + (f"; {'; '.join(notes)}" if notes else ""))
    by_level = rep["train_loss_by_contrast"]
    lines.append("Training loss by contrast level (higher = worse): "
                 + ", ".join(f"{k}: {v:.2f}" for k, v in sorted(by_level.items(), key=lambda kv: float(kv[0]))))
    chk = rep.get("gradient_check")
    if chk and not chk["agrees"]:
        lines.append(f"Warning: the gradient for {chk['param']} disagrees with finite differences "
                     f"({chk['enzyme']:.3g} vs {chk['finite_difference']:.3g}), so the program may not be smooth in θ.")
    return "\n".join(lines)


def user_prompt(mode, parent, levels, last_failure=None, history=None):
    """parent: the best program so far (code plus its evaluation)."""
    ev = parent["eval"]
    parts = [f"Current best program (score {ev['val']['score']:.3f} = mean log10 iterations; lower is better):",
             f"```julia\n{parent['code'].strip()}\n```",
             "Validation results (instances not used for tuning):",
             _scores_table(ev["val"], levels)]
    if mode in ("score_only", "ours"):
        parts += ["Values of θ after tuning (a good starting point for THETA0):", _tuned_values(ev)]
    if mode == "ours":
        parts += ["Gradient report from tuning:", _gradient_report(ev)]
    if history:
        parts.append("Earlier attempts (score, or the stage where it failed): " + "; ".join(history[-8:]))
    if last_failure:
        parts.append(f"Your previous program failed while {last_failure['stage']}:\n{last_failure['error'][:1200]}")
    parts.append("Write an improved program.")
    return "\n\n".join(parts)
