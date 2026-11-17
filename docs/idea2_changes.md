# Changes to the Idea 2 code after the first Kimi pilots

Notes for Artemis on what was changed in the Idea 2 code (and in `python/idea1/llm.py`),
why, and how to check it. Nothing here is committed yet. All of the changes are backward
compatible: the seeds and any program without `prepare` behave exactly as before.

## Why: what the pilots showed

| Pilot | What happened | Fix |
|---|---|---|
| `pilot-channels64-precond-ours` (thinking on) | Kimi ran out of its 32k output tokens before writing code. | Decision 1: `kimi-k2.6:nothink`. |
| first `nothink` tries | Replies held 23–29 drafts of the program and were cut off at 32k tokens, so the **last** ```julia block (the one `extract_code` took) was a fragment. | Prompt asks for one complete block; `extract_code` takes the last *complete* program (1, 2 below). |
| same | Programs ran `lu` / `\` on matrices that depend on θ. Enzyme fails. | Prompt rule (1). |
| `pilot2-channels64-precond-ours` | Worked: most candidates evaluated and tuned. | – |
| `pilot2-channels64-solver-ours` | 4 of 5 failed during tuning with Enzyme errors on LAPACK `getrf` (from `inv`/`lu`) and `syevr` (from `eigen`) inside `init`, **even on matrices built only from P**. Enzyme has no rules for these calls, so they crash autodiff even when they don't depend on θ. (`cholesky` + `ldiv!` and BLAS `mul!` did work.) | Optional `prepare(P)` that runs outside autodiff (3–5 below). |
| solver pilot stalled for > 15 min at 0 % Julia CPU | The original `llm.py` waited on an API answer that never arrived (900 s timeout + 2 hidden OpenAI-client retries). | Streaming `llm.py` (6 below). |

## 1. `python/idea2/prompts.py`

- **Interfaces:** both frames now say that a program can optionally define
  `prepare(P) -> C` for work that does not depend on θ, in which case the first function
  becomes `setup(P, C, θ)` (precond) or `init(P, C, θ)` (solver).
- **New rule:** "Automatic differentiation goes through everything except prepare, and it
  cannot go through computing matrix factorisations, inverses or decompositions (lu,
  cholesky, qr, eigen, svd, inv, \\). Compute all of these in prepare(P), which runs once
  per problem without differentiation and must not use θ. The other functions can use the
  results freely: apply a Cholesky factor F from prepare with ldiv!(y, F, x), or an inverse
  with mul!. An exact coarse solve this way (a coarse matrix built from P, factorised in
  prepare) is usually much stronger than a few iterations of a simple method. Never
  factorise or invert a matrix that depends on θ."
  - The first version of this rule said "use their results only through mul!" and "use a
    few iterations of a simple iterative method instead". In `pilot3-channels64-precond-ours`
    Kimi then replaced the exact coarse solve (`cholesky` + `ldiv!`, used by all of
    pilot2's programs) with 3 damped-Jacobi sweeps, and the best test score got much worse
    (1.743 in pilot2 → 2.505). Pilot2 had shown that `cholesky` + `ldiv!` on a matrix built
    from P works with Enzyme (pilot2 even computed the factor in `setup`); what fails is
    `inv`/`lu`/`eigen`. The rule now says that applying a factor from `prepare` with `ldiv!`
    is fine, and that an exact coarse solve is usually the stronger choice.
- Package rule now adds "(so no SparseArrays: store small sparse operators as dense
  matrices or index/value vectors)", since some programs did `using SparseArrays`.
- "setup may precompute anything from P" → "prepare may precompute anything from P".
- **Reply format:** the system prompt ends "Reply with a short explanation of your idea
  (a few sentences), then the complete program in exactly one ```julia code block. Do not
  write drafts, alternatives or partial code blocks: only a complete program is used, and
  very long replies are cut off before they finish." The user prompt's last line matches.
- The `\` in the prompt text is escaped (`\\\\` in the source) so Python no longer prints a
  `SyntaxWarning: invalid escape sequence`.

## 2. `python/idea2/search.py`

- `extract_code(reply)` returns `(code, n_blocks)`: the **last block that is a complete
  program** (contains `const THETA0`, `const LOWER`, `const UPPER`, `const THETA_NAMES` and
  `precond!` or `step!`), instead of simply the last block.
- If no block is complete, the error shown to the LLM says how many blocks there were and
  that the reply may have been cut off, and asks for one complete block (before it always
  said "no ```julia code block in the reply").

## 3. `julia/idea2/harness.jl`

- Header documents the optional `prepare(P) -> C`.
- `run_precond`, `run_solver`, `problem_loss`, `iterations_precond`, `iterations_solver`
  take an extra argument `C` after `P`, and call `setup_f(P, C, θ)` / `init_f(P, C, θ)`.
- `Candidate` has two new fields: `prep` (the program's `prepare`, or `P -> nothing`) and
  `cache::IdDict` (C per problem, so `prepare` runs once per problem per candidate).
  `f1` is now always called as `f1(P, C, θ)`.
- `load_candidate`:
  - if the program defines `prepare`, its `setup`/`init` must have a 3-argument method,
    otherwise a clear error;
  - if not, `setup`/`init` must have a 2-argument method and is wrapped as
    `(P, C, θ) -> f1(P, θ)`, so **old programs and the seeds are unchanged**.
  - helper `takes_args(f, k)` checks the arity with `methods`.
- `prepared(c, P)` returns the cached C (computing it with `invokelatest` the first time).
- `loss_and_grad`: C is computed **outside** `autodiff` and passed as `Const(C)` (between
  `Const(P)` and `Duplicated(θ, dθ)`), so Enzyme never looks inside `prepare`.
- `mean_loss` and `evaluate_iterations` pass `prepared(c, P)` as well.

## 4. `julia/idea2/server.jl`

- `evaluate` has a new first stage `"prepare"`: it runs `prepared` for every train and
  validation problem and records `prepare_seconds` in the result (so an expensive prepare
  is visible, and a crash in prepare is reported as its own stage).
- `AD_HINT`: when an error message contains "Enzyme", the error shown to the LLM starts
  with a short explanation (factorisation/inverse/decomposition inside the differentiated
  functions → move it into `prepare`, use the result via `mul!` or `ldiv!`). The raw Enzyme error is
  long and gets truncated, so the useful part would otherwise be lost.

## 5. Tests: `julia/idea2/check_harness.jl` and new `julia/idea2/tests/`

- New `tests/precond_block_jacobi_prepare.jl`: 4×4-cell block Jacobi. `prepare(P)` builds
  the block index sets and their inverses with `inv` (exactly what failed in the pilot);
  `precond!` applies them with `mul!`, blended with θ[2]·Jacobi.
- New `tests/solver_block_jacobi_prepare.jl`: the same blocks as a stationary solver,
  x += ω·B⁻¹r + β·(momentum), θ = (ω, β), θ0 = [0.8, 0], bounds [0, 0]–[1.5, 0.99].
- `check_harness.jl` now runs the two seeds **and** these two tests (paths are relative to
  `julia/idea2/`).

## 6. `python/idea1/llm.py` (shared by Idea 1 and Idea 2)

Kimi calls are **streamed**: a call is abandoned only if no data arrives for 300 s
(`STREAM_IDLE_TIMEOUT`) or it is still streaming after 3600 s (`MAX_CALL_SECONDS`).
`max_retries=0` turns off the OpenAI client's hidden retries; only the visible
`QUICK_RETRIES` loop retries, and it prints why. Token counting is unchanged (usage comes in
the last chunk). This is the version used for the Idea 1 models-k3 run.

## 7. Docs

- `docs/idea2.md`: interface table and a paragraph on `prepare`, the rules and `AD_HINT`.
- `docs/idea2_runs.md`: steps 2–3 updated (history, check commands, `pilot3-…` names).
- Housekeeping: ILU(0) replaced by symmetric Gauss–Seidel in `README.md`, `docs/idea2.md`
  and the `python/baselines.py` docstring; `docs/idea1_runs.md` updated (models-k3 done,
  history = last 10, streaming `llm.py`).

## How to check

```
julia --project=julia julia/idea2/check_harness.jl channels 64
```
Expected: the two seeds give the same numbers as before (precond 2.807 → 2.657 …); the two
tests load, Enzyme's gradient agrees with finite differences, and tuning lowers the loss.

```
python/.venv/bin/python python/idea2/search.py --model mock --iterations 4 \
    --train 1 2 --val 5 6 --test 1 2 --name mock-test
rm -r results/idea2/mock-test
```
Expected: runs end to end as before (results now also have `prepare_seconds`).

Then rerun the Kimi pilots as `pilot3-channels64-precond-ours` and
`pilot3-channels64-solver-ours` (`docs/idea2_runs.md`, steps 2–3).
