# Idea 2 (Part B): gradient-guided LLM solver search — plan and status

A tentative plan for Part B, and a reference for how the pieces built so far fit together.
The plan will change as results come in. See also the "Idea 2" sections of the
[README](../README.md).

**The idea.** An LLM writes the *structure* of a solver (a preconditioner, or a whole
iterative method) as Julia code with a vector of tunable numbers θ. Enzyme differentiates
through the solver's iterations, so the numbers are tuned with exact gradients instead of
being guessed by the LLM, and the gradient information is reported back to the LLM to guide
its next program. Part A (Idea 1) showed LLMs are poor at setting numbers; Part B tests
whether giving that job to Enzyme, and feeding gradients back, gives better solvers.

## Plan

### Step 0: pilot
1. Choose thinking vs no thinking for Kimi (see "Notes": thinking ran out of output
   tokens on the first try). Run 5 iterations, `--frame precond --mode ours` on
   `channels`, N = 64.
2. Read `candidates.jsonl`: do programs compile, run, tune? What fails and why? Adjust
   the prompt (`python/idea2/prompts.py`) or the rules if the LLM repeatedly breaks one
   (e.g. branching on θ, slow setup). Repeat with `--frame solver`.

### Step 1: diffusion baselines (independent of the loop)
Run `python/baselines.py` over all families and contrasts at N = 64 and 128 (test split),
so we know where AMG-PCG struggles. These are the reference curves for Part 2.

### Step 2: Part 1 — does gradient feedback help? (Kimi)
1. **Runs.** All three feedback modes at equal LLM calls (e.g. 20), same family, frame
   and settings, different `--name`. Start with `channels` + `precond`, ~3 repeats per
   mode (the LLM is random), then `solver`, then the other families. Full grid: 3 modes ×
   2 frames × 5 families × 3 repeats = 90 runs; expand only if the first results justify
   it. Suggested naming: `<family><N>-<frame>-<mode>-r<repeat>`.
2. **Summary script** (`python/idea2/summarize.py`, to write), from `candidates.jsonl`
   and `test.json`:
   - test iterations vs contrast for each run's best program (final quality; does it stay
     fast as contrast grows?),
   - best validation score vs number of LLM calls (how fast each mode improves),
   - failure rate by stage,
   - does the LLM act on the report? (e.g. after "θ_i stuck at upper bound", does the next
     program widen UPPER for it; does it drop parameters marked "no effect").

### Step 3: Part 2 — beat the state of the art (strongest available model)
1. **Count work, not only iterations.** Add to `evaluate_iterations` (and to the
   baselines) the total cost of setup + solve in matvec equivalents, or Julia runtime,
   so a program cannot win by doing a lot of work per iteration.
2. **Strong seeds** in `julia/idea2/seeds/`: ILU(0) preconditioner and a two-level
   smoothed-aggregation AMG, written in plain Julia within the rules (θ = smooth knobs only,
   e.g. smoothing weights; discrete choices such as the coarsening fixed in `setup`).
   Check their iteration counts match `pyamg`/ILU from `baselines.py`. `search.py` picks the
   seed from `SEEDS`; add a `--seed` option to choose a file.
3. **Model**: the best model we can get access to (e.g. the newest Claude or GPT model via
   its API). Add an engine for it next to `KimiEngine` in `python/idea1/llm.py`, selected
   by model name, with its key in `.env`.
4. **Runs**: `ours` (or the best mode from Part 1) with more LLM calls, from the strong
   seeds and from the simple ones.
5. **Win** = against AMG-PCG on the test split (unseen instances), also at N = 128:
   fewer iterations to 1e-8, especially at high contrast; at equal or lower total work;
   slower growth of iterations with contrast. Report cost per iteration and runtime next
   to iteration counts.

### Later / if time: Helmholtz
Same loop on `data/helmholtz` (frequency instead of contrast as the difficulty). Needs a
GMRES outer loop instead of CG, complex arithmetic in the harness (check Enzyme), and a
matching training loss.

## Status

| Piece | Status | Where |
|---|---|---|
| Diffusion datasets (5 families, N = 64/128, train/test) | done | `data/diffusion/`, `python/diffusion.py` |
| Baseline code (CG, Jacobi/ILU(0)/AMG-PCG) | done; full sweep not yet run | `python/baselines.py` |
| Julia harness (run, tune, report, score a program) | done, gradients checked against finite differences | `julia/idea2/harness.jl` |
| Starting programs (one per search space) | done | `julia/idea2/seeds/` |
| Python ↔ Julia bridge | done | `julia/idea2/server.jl` |
| Search loop + prompts, 3 feedback modes | done, tested end to end with the mock model | `python/idea2/search.py`, `python/idea2/prompts.py` |
| Kimi pilot | first try (thinking on) returned no code; see "Notes" | not committed |
| Part 1 experiments, summary script | not started | |
| Part 2 (work counting, strong seeds, engine for the strongest model) | not started | |
| Helmholtz | data and baselines only | `data/helmholtz/`, `python/helmholtz.py` |

## The problems

−∇·(k∇u) = f on the unit square, u = 0 on the boundary, cell-centred finite volumes on an
N×N grid → A x = b, A sparse, SPD, 5-point (n = N² unknowns). The coefficient k is drawn
from a **family**; for the binary families one geometry per instance is reused at every
contrast level, so "iterations vs contrast" is measured on identical geometry.

| Family (file name part) | k | Contrast levels |
|---|---|---|
| `lognormal_exponential`, `lognormal_gaussian` | exp(σφ), φ a Gaussian random field (rough / smooth) | σ² ∈ {0.1, 0.5, 1, 2, 4} |
| `channels`, `inclusions` | 1 + (10^p − 1)·mask | p = 0..8 |
| `barriers` | 1 + (10^−p − 1)·mask | p = 0..8 |

Files: `data/diffusion/diffusion_<family>_N<N>_<split>.h5`, split = `train` (32 instances)
or `test` (16). They store the mask (or k) and f, not A; `harness.jl` (`make_problem`)
assembles A exactly as `python/diffusion.py` documents, via the `MMDData` package
(`julia/MMDData`, `load_diffusion`, `diffusion_instance`).

**Instance use** (defaults in `search.py`): train instances 1–4 → Enzyme tuning;
train instances 5–8 → validation (picks the best program); test instances 1–16 → final
numbers only. `--levels every_other` (default) uses every other contrast level in the
search; the test evaluation always uses all levels.

## How the loop works

```
search.py ──JSON lines──> server.jl ──include──> harness.jl
   │  (LLM calls, logging)        (setup / evaluate / test)
   └─ prompts.py (what the LLM is told)
```

One iteration of `search.py`:

1. Build the prompt from the **best program so far** (the parent) and its evaluation
   (`prompts.user_prompt`); the system prompt (`prompts.system_prompt`) explains the
   problem, the interface and the rules.
2. Call the LLM (`python/idea1/llm.py`, `get_engine`, reused from Part A); take the last
   ` ```julia ` block of the reply.
3. Send it to Julia (`{"cmd": "evaluate"}`), which:
   - **loads** it into a fresh module (`load_candidate`) and checks the interface,
   - **tunes θ** (`tune`): projected Adam, 25 steps, on the training loss = mean over
     training problems of log10(‖r_m‖/‖b‖) after m = 30 iterations, gradients from Enzyme
     (`loss_and_grad`),
   - writes the **gradient report** (`gradient_report`): per parameter its tuned value,
     gradient, relative sensitivity, whether it is stuck at a bound or has no effect; the
     training loss per contrast level; an Enzyme vs finite-difference check,
   - **scores** on validation (`evaluate_iterations`): iterations to ‖r‖/‖b‖ ≤ 1e-8 per
     problem, capped at `--maxit` (500); for runs that hit the cap the count is
     extrapolated from the convergence rate (`effective_iterations`). **Score = mean
     log10 iterations; lower is better.**
4. Log the row; if the score beats the best, the new program becomes the parent.
5. A failure (no code block, does not compile, crashes, times out after `--timeout` s,
   empty LLM answer) is logged with its stage and error, and the error is shown to the LLM
   in the next prompt. Julia is restarted after a timeout or crash.

After the last iteration the best program and the starting program are scored on the
test split (`{"cmd": "test"}`).

### Two search spaces (`--frame`)

| Frame | The LLM writes | Fixed by the harness | Seed (`julia/idea2/seeds/`) |
|---|---|---|---|
| `precond` | `setup(P, θ) -> S`, `precond!(z, r, P, S, θ)` | flexible CG (`run_precond`, `iterations_precond`) | `precond_jacobi_poly.jl`: z = Σ c_j (D⁻¹A)^j D⁻¹ r; θ0 = [1,0,0] is Jacobi-PCG |
| `solver` | `init(P, θ) -> S`, `step!(x, r, P, S, θ)` | residual update, x0 = 0 (`run_solver`, `iterations_solver`) | `solver_jacobi_momentum.jl`: damped Jacobi with momentum (ω, β) |

Every program also defines `THETA0`, `LOWER`, `UPPER`, `THETA_NAMES`. Inside, it can use
the fields of `DiffusionProblem` (`N, n, k, tw, te, ts, tn, A, diag, b`) and `matvec!`.
Rules given to the LLM (`prompts.SYSTEM`): Base + LinearAlgebra only, no sparse
factorizations, differentiable in θ (no branching on θ), cost per iteration ~linear in n.

### Three feedback modes (`--mode`)

| Mode | θ set by | The LLM additionally sees | Stands for |
|---|---|---|---|
| `llm_only` | the LLM's `THETA0`, untuned | validation results only | AlphaEvolve-style |
| `score_only` | Enzyme tuning | + the tuned θ | MadEvolve / Gallego-style |
| `ours` | Enzyme tuning | + the gradient report | our method |

All modes get the same instructions, the same scores and the same number of LLM calls;
only the feedback differs (`prompts.user_prompt`). Comparisons: `score_only` vs
`llm_only` = does exact tuning help; `ours` vs `score_only` = does gradient feedback help.

## Running it

Setup is the same as Part A (README "Setup"); the Kimi key goes in `.env`.

```
# free: Enzyme vs finite differences, tuning and scores for both seeds (~1 min)
julia --project=julia julia/idea2/check_harness.jl channels 64

# free: the whole loop with a fake model that perturbs THETA0 and sometimes breaks the code
python/.venv/bin/python python/idea2/search.py --model mock --iterations 4 \
    --train 1 2 --val 5 6 --test 1 2 --name mock-test

# real run
python/.venv/bin/python python/idea2/search.py --family channels --N 64 \
    --frame precond --mode ours --model kimi-k2.6:nothink --iterations 20 \
    --name channels64-precond-ours-r1
```

`search.py --help` lists every option (`--tune-steps`, `--m`, `--maxit`, `--train`,
`--val`, `--test`, `--levels`, `--timeout`, `--seed-only`). A `--name` that already has
results is refused.

**Outputs** in `results/idea2/<name>/`:
- `config.json`: all settings + git commit + whether there were uncommitted changes
- `candidates.jsonl`: one row per program (iteration 0 = seed): `reply` (full LLM text),
  `code`, `eval` (θ, report, validation results, or failure stage + error), `score`,
  `improved`, `parent`, running totals of `calls`, `prompt_tokens`, `completion_tokens`
- `test.json`: `best` and `start` on the test split: score, median iterations per
  contrast, converged counts, θ

**Timing** (channels, N = 64, default 4+4 instances, 5 levels): seed evaluation ~30 s,
a candidate roughly 15–60 s depending on its cost; plus the LLM call.

## Notes from what we've run so far

- **Kimi with thinking on ran out of output tokens.** First pilot attempt:
  `kimi-k2.6:think`, `--frame precond --mode ours`, `channels`, N = 64. The seed scored
  2.564, but all 3 LLM calls hit the default output limit of 32,768 tokens while still
  thinking and returned no code (`NoAnswerError: empty answer (finish_reason=length)`),
  ~6 min and ~$0.13 per call; we stopped the run after 3 calls (~$0.40). Writing a whole
  program needs more room than tuning a few numbers did in Part A. Options: use
  `kimi-k2.6:nothink` (fast, ~$0.02 per call), or pass a larger `max_tokens` to the
  `completion(...)` call in `python/idea1/llm.py` (`KimiEngine.lite_llm_generate`), if
  the API allows it (slower and pricier per call, but likely better code).
- **Loaded modules must be kept alive.** `load_candidate` pushes each module onto
  `LOADED_MODULES`; without that the GC frees the code and Enzyme segfaults.
- **c0 in the Jacobi-polynomial seed** has zero gradient at θ0 (CG is invariant to scaling
  the preconditioner). Once c1, c2 ≠ 0 it matters again, so the report gives it a nonzero
  sensitivity after tuning. Expected, not a bug.
- **The solver seed never converges in 500 iterations**; its scores are extrapolated.
  Fine for ranking, but expect large numbers.
- Debugging the bridge: `MMD_DEBUG=1` makes `server.jl` include Julia backtraces in errors.

## Conventions

- Keys only in `.env` (gitignored); never commit them.
- Commit each finished run folder under `results/idea2/` with the code version it ran on
  (`config.json` records the commit); failed or exploratory runs can stay local.
