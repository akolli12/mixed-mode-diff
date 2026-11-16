# Idea 2 runs: what is left and how to run it

This is the run plan for the rest of the project, in the same format as
[idea1_runs.md](idea1_runs.md). Most of it is Idea 2 (Part B): an LLM writes a solver's
structure, Enzyme tunes its numbers, and in our mode a gradient report goes back to the LLM.
How the loop works, its files and its options are in [idea2.md](idea2.md); this file says
what to do, in what order, and records the status and result of each step.

Fill in **Who** when we split the work, update **Status** as steps finish, add one row per
finished run to the **Run log**, and write each step's headline result in the **Results log**
at the bottom.

## What is being varied

| | Values | Option |
|---|---|---|
| Family | `channels`, `inclusions`, `barriers`, `lognormal_exponential`, `lognormal_gaussian` | `--family` |
| Search space (frame) | `precond` (a preconditioner inside flexible CG) or `solver` (a whole iterative method) | `--frame` |
| Feedback mode | `llm_only` (the LLM sets θ, sees the score), `score_only` (Enzyme tunes θ; the LLM sees the score and tuned θ), `ours` (also the gradient report) | `--mode` |
| Grid | N = 64 for the search; N = 128 only for the final generalization test | `--N` |
| Model | `kimi-k2.6:nothink` for Phase 1; GPT-6 Astra (Prime Intellect credits) for Phase 2 | `--model` |
| Repeats | 3 per configuration: the LLM is random, so one run per mode is not enough | in `--name` |
| Budget | 20 LLM calls per run, the same for every mode | `--iterations` |

Every run tunes on training instances 1–4, picks its best program on instances 5–8 (also
from the training file), and is scored at the end on all 16 test instances at every
contrast level. Name runs `<family><N>-<frame>-<mode>-r<repeat>`, e.g.
`channels64-precond-ours-r1`; add the model for non-Kimi runs, e.g.
`channels64-precond-ours-astra-r1`.

## Decisions to make first

| # | Decision | Options | Decided |
|---|---|---|---|
| 1 | Model and thinking for Phase 1 | `kimi-k2.6:nothink` (~$0.02 per call, fast). Thinking on ran out of its 32k output tokens before writing any code in the first pilot; it would need a larger `max_tokens` in `python/idea1/llm.py`, and costs ~$0.13+ and ~6 min per call. GPT-6 Astra is too expensive for all of Phase 1 (below). | `kimi-k2.6:nothink` |
| 2 | Who runs what | The steps form two tracks that can go in parallel: **runs** (pilots, Phase 1, summary script) and **code** (baselines, Prime Intellect engine, work counting, strong seeds). One track each is a natural split. | Alaina: Phase 1 and the Part A follow-up (pilots, Phase 1a–c, summary script, ablation, Astra pilot and Phase 1a with Astra, step-length correction). Artemis: the solver infrastructure and Phase 2 (baselines, Prime Intellect engine, work counting, strong seeds, Phase 2 runs, N = 128 test, Helmholtz, optional Part A runs). Together: the step 8 checkpoint and the final report. |
| 3 | How to spend the Prime Intellect credits | see "Budget and credits" below | Proposed: Alaina's $200 for steps 12–13 (~$60, rest in reserve); Artemis's $200 for step 16 and optional A3 (~$155). |

## Budget and credits

| Source | Amount | Used for | Spent so far |
|---|---|---|---|
| Kimi (Moonshot) API | prepaid balance | Part A (all stages), Phase 1 | Part A: ~$215–220 (sum of the stage table in `idea1_runs.md`) |
| Prime Intellect credits | $400 ($200 each, on our own accounts) | GPT-6 Astra: pilot, Phase 2, one Phase 1 comparison (steps 12, 13, 16) | $0 |

**Prime Intellect.** The credits cover their inference API as well as GPUs. The API is
OpenAI-compatible: base URL `https://api.pinference.ai/api/v1`, key in `.env` as
`PRIME_API_KEY` (Bearer token; add the `X-Prime-Team-ID` header to bill a team account).
List the available models and their prices with `GET /models` or `prime inference models`,
and use the model IDs exactly as listed. Its catalog includes **GPT-6 Astra**, OpenAI's
newest model. OpenAI's own price for it is $10 per million input tokens and $50 per million
output tokens (reasoning tokens count as output), with reasoning effort low to max; check
Prime Intellect's listed price, which may differ.

**What a Part B call costs** (input ≈ 3–6k tokens: the rules, the parent program and its
evaluation; output = the new program plus any reasoning):

| Model | Output per call (est.) | Cost per call (est.) | 20-call run | Phase 1a+1b (360 calls) | All of Phase 1 (1,800 calls) |
|---|---|---|---|---|---|
| `kimi-k2.6:nothink` | ~2–3k | ~$0.02 | ~$0.40 | ~$7 | ~$36 |
| GPT-6 Astra, low reasoning | ~5k | ~$0.30 | ~$6 | ~$110 | ~$540 |
| GPT-6 Astra, medium reasoning | ~15k | ~$0.80 | ~$16 | ~$290 | ~$1,400 |
| GPT-6 Astra, high reasoning | ~30k+ | ~$1.50+ | ~$30+ | ~$550+ | ~$2,700+ |

The Astra rows are estimates at OpenAI's list price; step 12 measures the real cost. Time
per call depends on how much it reasons: roughly 1–5 minutes at low to medium, on top of
the 15–60 s Julia evaluation, so a 20-call run takes about 30–120 minutes.

**Planned use of the $400 of Prime Intellect credits** (GPT-6 Astra at the estimates
above; about 1,300 calls at low reasoning or 500 at medium). Replace the estimates with
measured numbers after the pilot. If a medium-reasoning call costs much more than ~$1,
drop item 4 and use low reasoning for Phase 2 too.

| Use | Step | Setting | Calls | Est. cost | Spent |
|---|---|---|---|---|---|
| 1. Pilot: measure real cost and time per call | 12 | 5 calls at low + 5 at medium reasoning | 10 | ~$5 | |
| 2. Phase 2, the main use | 16 | best mode from Phase 1, medium reasoning, ~4 runs × 30–40 calls (strong and simple seeds) | ~140 | ~$110 | |
| 3. Phase 1a replicated with Astra | 13 | `channels`, `precond`, 3 modes × 3 repeats × 20 calls, low reasoning | 180 | ~$55 | |
| 4. Optional: Astra row in the Part A model table | A3 | 20 problems × 5 steps × 3 calls, low reasoning | 300 | ~$45 | |
| Reserve | | reruns, failed runs, price surprises | | ~$150–185 | |

The credits are on two separate accounts, so each of us runs with our own `PRIME_API_KEY`
(e.g. one person's $200 for Phase 2, the other's for the pilot and the Phase 1a
replication), or we put both in a team and send `X-Prime-Team-ID`.

## The steps, in order

**Track** says whether a step is runs (R) or code (C); the two tracks can go in parallel.
**Needs** lists the steps that must finish first. Kimi times and costs assume
`kimi-k2.6:nothink` and 15–60 s to evaluate a candidate in Julia, so roughly 15–30 minutes
per 20-call run. Several runs can go at once on different families (each starts its own
Julia process), CPU permitting.

| Step | What | Track | Needs | Who | Runs | LLM calls | Time | Cost | Status |
|---|---|---|---|---|---|---|---|---|---|
| 0 | Setup and free checks (below) | – | – | each of us | – | 0 | 15 min | free | **done** on both machines |
| 1 | Housekeeping | C | – | Alaina | – | 0 | 10 min | free | to do |
| 2 | Pilot, `precond` frame (Kimi) | R | 0 | Alaina | 1 | 5 | ~30 min | ~$0.10 | started (thinking-on try returned no code); redo with decision 1 |
| 3 | Pilot, `solver` frame (Kimi) | R | 0 | Alaina | 1 | 5 | ~30 min | ~$0.10 | to do |
| 4 | Full baseline sweep (small code change, then run) | C | – | Artemis | – | 0 | ~1–2 h | free | pilot done (N = 128, 1 instance) |
| 5 | Phase 1a: `channels`, `precond`, 3 modes × 3 repeats (Kimi) | R | 2 | Alaina | 9 | 180 | ~3–5 h | ~$4 | to do |
| 6 | Summary script `python/idea2/summarize.py` | C | write while 5 runs | Alaina | – | 0 | – | free | to do |
| 7 | Phase 1b: `channels`, `solver`, 3 modes × 3 repeats (Kimi) | R | 3 | Alaina | 9 | 180 | ~3–5 h | ~$4 | to do |
| 8 | Checkpoint: do the modes differ on `channels`? | both | 5, 6, 7 | both | – | 0 | 1 h | free | to do |
| 9 | Phase 1c: the other 4 families × 2 frames × 3 modes × 3 repeats (Kimi) | R | 8 (only if the modes differ) | Alaina | 72 | 1,440 | ~1–2 days | ~$29 | to do |
| 10 | Ablation of the report's parts (Kimi) | R + C | 5 | Alaina | 9–12 | 180–240 | ~4–6 h | ~$5 | to do |
| 11 | Prime Intellect engine in `python/idea1/llm.py` | C | – | Artemis | – | 0 | 1–2 h | free | to do |
| 12 | Astra pilot: cost and time per call | R | 11 | Alaina | 2 | 10 | ~1 h | ~$5 | to do |
| 13 | Phase 1a with Astra (low reasoning) | R | 8, 12 | Alaina | 9 | 180 | ~5–10 h | ~$55 | to do |
| 14 | Work counting in the harness and baselines | C | 4 | Artemis | – | 0 | – | free | to do |
| 15 | Strong seeds in Julia: ILU(0) and two-level smoothed-aggregation AMG | C | – | Artemis | – | 0 | – | free | to do |
| 16 | Phase 2 runs with Astra (medium reasoning) | R | 8, 12, 14, 15 | Artemis | ~4 | ~140 | ~5–8 h | ~$110 | to do |
| 17 | Generalization: best programs at N = 128 (small script) | C + R | 16 | Artemis | – | 0 | ~1 h | free | to do |
| 18 | Helmholtz (stretch): GMRES harness, complex arithmetic, training loss | C | 16 | Artemis | TBD | TBD | TBD | TBD | only if time |
| A1 | Part A: correcting the LLM's step length (Section 7.2 of the midterm) | C + R | – | Alaina | 20 problems | 0–600 | ~2 h | free–$10 | to do |
| A2 | Part A (optional): `blackbox` feedback, history without the code | C + R | – | Artemis | 20 | ~600 | ~1–2 h | ~$10 | optional |
| A3 | Part A (optional): GPT-6 Astra row in the model table | R | 11 | Artemis | 20 | 300 | ~3 h | ~$45 | optional |
| F | Final report and presentation | – | everything above | both | – | – | – | – | to do |

### Step 0. Setup and free checks (once per machine)

```
git pull
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
python3.11 -m venv python/.venv
python/.venv/bin/pip install -r python/requirements.txt
```

Keys go in `.env` at the repo root (git ignores it), one per line:

```
KIMI_API_KEY=sk-...
PRIME_API_KEY=...
```

Then, with no API calls:

```
# Enzyme vs finite differences, tuning and scores for both seeds (~1 min)
julia --project=julia julia/idea2/check_harness.jl channels 64

# the whole loop with a fake model (~5 min); delete its folder afterwards
python/.venv/bin/python python/idea2/search.py --model mock --iterations 4 \
    --train 1 2 --val 5 6 --test 1 2 --name mock-test
rm -r results/idea2/mock-test
```

### Step 1. Housekeeping

- `idea2.md` lists ILU(0) among the baselines, but the pilot sweep used symmetric
  Gauss–Seidel-PCG because SciPy's ILU never converged. Update the doc, or add ILU(0) in
  Julia with the strong seeds (step 15).
- `idea1_runs.md` says history feedback shows "every earlier θ and loss"; the code shows the
  last 10 (`HISTORY_LEN` in `python/idea1/textgrad_step.py`).

### Steps 2–3. Kimi pilots

```
python/.venv/bin/python python/idea2/search.py --family channels --N 64 \
    --frame precond --mode ours --model kimi-k2.6:nothink --iterations 5 \
    --name pilot-channels64-precond-ours
```

and the same with `--frame solver --name pilot-channels64-solver-ours`. Then read
`results/idea2/<name>/candidates.jsonl` (or use the quick check under "When a run
finishes"): do the programs compile, run and tune? Which stage fails, and with what error?
If the LLM keeps breaking the same rule (branching on θ, expensive setup, wrong interface),
fix the prompt in `python/idea2/prompts.py` before Phase 1. Pilot folders don't need to be
committed. **Done when** most candidates evaluate without failing.

### Step 4. Full baseline sweep

`python/baselines.py` currently runs diffusion at N = 128 with 1 test instance per family
(`run_diffusion(N=128, n_inst=1)`) and Helmholtz with 2 (`run_helmholtz(n_inst=2)`). Change
these to run every diffusion family at N = 64 and 128 on all 16 test instances, and
Helmholtz on all 8 (or add command-line options), then:

```
cd python && python3 baselines.py diffusion helmholtz && python3 figures.py && cd ..
git add results/baselines* && git commit -m "Idea 2: full baseline sweep"
```

This uses the dataset-generation Python (numpy, scipy, pyamg, h5py), not the TextGrad venv.
Before comparing with search results, note that the baselines use SciPy solvers with a cap of
5,000 iterations, while the search scores in Julia (flexible CG, cap 500, extrapolated past
the cap): compare at the same cap, or score the baselines in the Julia harness (step 15 does
this for ILU(0) and AMG).

### Steps 5, 7, 9. Phase 1 runs (Kimi)

Step 5 (9 runs, one after another):

```
for mode in llm_only score_only ours; do for r in 1 2 3; do
  caffeinate -i python/.venv/bin/python python/idea2/search.py --family channels --N 64 \
      --frame precond --mode $mode --model kimi-k2.6:nothink --iterations 20 \
      --name channels64-precond-$mode-r$r
done; done
```

Step 7: the same with `--frame solver` and `--name channels64-solver-$mode-r$r`.
Step 9: the same loop over `--family` (`inclusions barriers lognormal_exponential
lognormal_gaussian`) and both frames; run two or three families at once in separate
terminals.

- A `--name` that already has results is refused, so an interrupted run must be rerun under
  a new name, or its folder deleted first.
- Add a row to the **Run log** as each run finishes, and commit its folder (see "When a run
  finishes").

### Step 6. Summary script

`python/idea2/summarize.py`, reading `candidates.jsonl` and `test.json` from any number of run
folders, should produce:
- **final quality:** test iterations vs contrast for each run's best program, with the
  starting program and the baselines on the same plot;
- **search efficiency:** best validation score after each LLM call, by mode;
- **reliability:** fraction of candidates failing at each stage (no code, compile, crash,
  timeout, empty answer);
- **use of the report** (`ours` only): how often the next program acts on it, e.g. widens
  the bound of a parameter flagged as stuck, or drops one flagged as having no effect;
- **cost:** LLM calls, tokens and dollars per run.

### Step 8. Checkpoint

Run the summary script on the 18 `channels` runs. If the three modes clearly differ (test
iterations, or best validation score per call), expand to step 9; if not, discuss before
spending more. Either way, record the answer in the Results log.

### Step 10. Ablation

Rerun `ours` with one part of the gradient report removed at a time (sensitivities, bound
flags, per-contrast losses), 3 repeats each, on `channels` + `precond`. Needs a small option
in `prompts.py` (and `search.py`) to drop each part.

### Step 11. Prime Intellect engine

Add an engine next to `KimiEngine` in `python/idea1/llm.py` for Prime Intellect's
OpenAI-compatible API (base URL above, key `PRIME_API_KEY`), with the reasoning effort as an
option, and make `get_engine` pick it by model name (e.g. a `prime:` prefix). Reuse the same
streaming and retry code as `KimiEngine`, and record tokens the same way so costs can be
computed. Test it with a single call before step 12.

### Step 12. Astra pilot

Two 5-call runs on `channels` + `precond` in `ours` mode, one at low and one at medium
reasoning, with the step 11 model name. Record cost per call (from the Prime Intellect
billing page or the token counts) and seconds per call, and update the budget tables above.

### Step 13. Phase 1a with Astra

Step 5's loop with the Astra model at low reasoning, names ending `-astra-r$r`. Tells us
whether the gradient report helps more with a stronger model.

### Steps 14–16. Phase 2

- **14, work counting:** add the total work of setup and solve, in matrix–vector-product
  equivalents (and wall time), to `evaluate_iterations` in `julia/idea2/harness.jl` and to the
  baselines, so a program cannot win by doing more work per iteration.
- **15, strong seeds:** ILU(0) and a two-level smoothed-aggregation AMG preconditioner in
  `julia/idea2/seeds/`, written within the rules (θ = smooth knobs such as smoothing
  weights; discrete choices fixed in `setup`). Check that their iteration counts match the
  Python baselines. `search.py` chooses the seed from `SEEDS`; add a `--seed` option to pick
  a file.
- **16, runs:** the best mode from Phase 1 with Astra at medium reasoning, 30–40 calls per
  run, from the strong seeds and from the simple ones.

Success means fewer iterations than AMG-preconditioned CG on the test split at equal or lower
total work, especially at high contrast.

### Step 17. Generalization at N = 128

`search.py` has no test-only mode. Write a small script that starts `julia/idea2/server.jl`
with `N = 128` and sends the `test` command for each best program (code and θ from its
`test.json` and `candidates.jsonl`), and compare with AMG-PCG at N = 128.

### Step 18. Helmholtz (stretch)

Only if Phase 2 goes well: a GMRES outer loop instead of CG, complex arithmetic in the
harness (check that Enzyme handles it), and a matching training loss. See `idea2.md`.

### Step A1. Part A step-length correction

Keep the LLM's direction and replace its length, either (1) rescaling each step,
θ_{t+1} = θ_t + β_t d_t, with β_t estimated from how far earlier steps overshot, or (2) a
short line search along d_t with 2–3 extra loss evaluations. Test the line-search version
first on the saved proposals in `results/idea1/mainA-*` (no API calls); the rescaling
version needs new runs on the 20-problem subset with history feedback. Compare at equal loss
evaluations with uncorrected TextGrad and random search (`results/idea1/dfo`).

### Steps A2–A3. Optional Part A runs

- **A2:** a `blackbox` feedback mode (history shown, code hidden) does not exist yet: add it
  to `FEEDBACK_MODES` and the prompt in `python/idea1/textgrad_step.py`, then run the
  20-problem subset with `kimi-k2.6:nothink`. Tells us whether the LLM's edge at large n
  comes from the code or the history.
- **A3:** `python/idea1/run.py --stage models-k3`-style run with the step 11 Astra model
  (add a `models-astra` stage to `python/idea1/plan.py`).

## When a run finishes

Quick check of one run (until the summary script exists):

```
python3 - results/idea2/channels64-precond-ours-r1 <<'EOF'
import json, sys
d = sys.argv[1]
rows = [json.loads(l) for l in open(f"{d}/candidates.jsonl")]
test = json.load(open(f"{d}/test.json"))
failed = sum(1 for r in rows[1:] if not r["eval"].get("ok"))
print("calls", rows[-1].get("calls"), "| failed candidates", failed, "of", len(rows) - 1)
print("validation score: seed", round(rows[0]["score"], 3), "-> best", round(min(r["score"] for r in rows), 3))
print("test score: seed", test["start"].get("score"), "-> best", test["best"].get("score"))
print("tokens in/out", rows[-1].get("prompt_tokens"), rows[-1].get("completion_tokens"))
EOF
```

Add a row to the Run log, then commit the folder:

```
git add results/idea2/channels64-precond-ours-r1
git commit -m "Idea 2 results: channels64-precond-ours-r1"
git pull --rebase && git push
```

Commit every finished Phase 1 or Phase 2 run with the code it ran on (`config.json` records
the commit); pilots and failed runs can stay local.

## Run log

One row per finished run. Scores are mean log10 iterations (lower is better). Cost: Kimi at
$0.95 / $4 per million input / output tokens; Astra at Prime Intellect's listed price.

| Run name | Step | Who | Calls | Failed candidates | Validation score (seed → best) | Test score (seed → best) | Tokens in / out | Cost | Notes |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

## Results log

One entry per finished step: date, who, and the headline result in a sentence or two, with
the folder it came from.

| Step | Date | Who | Result | Folder |
|---|---|---|---|---|
| 0 | | Artemis | Enzyme's gradients through all solver iterations match finite differences | `julia/idea2/check_harness.jl` |
| 0 | | Artemis | Tuning the Jacobi-polynomial seed alone lowers its median iterations at contrast 10⁴ from 480 to 371 | |
| 4 (pilot) | 2026-10-17 | Alaina | Inclusions, N = 128: AMG-PCG 8 → 251 iterations from contrast 10⁰ to 10⁸; Helmholtz inclusions with exact shifted Laplacian 15 → 260 (2 → 32 wavelengths) | `results/baselines_*.csv` |

## What gets recorded

Each run writes `results/idea2/<name>/`: `config.json` (settings and git commit),
`candidates.jsonl` (one row per program: the LLM's reply, the code, its evaluation, score,
whether it improved, and running call and token counts), and `test.json` (best and starting
program on the test split: score, median iterations per contrast level, converged counts,
θ).
