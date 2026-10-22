# Idea 1 runs: how to run them

This is the run plan for Idea 1: TextGrad updates the parameters θ of each problem in
`data/`, and every update is scored against Enzyme's exact gradient. Artemis runs the
pilot, all the solver stages, the thinking-model comparison for `kimi-k2.6`, the
consistency and 30-step stages, and the Enzyme timing. Alaina runs the three largest
stages: the test-function sweeps and `kimi-k3`.

## What is being varied

| | Values | Comes from |
|---|---|---|
| Problem family | quadratic, logcosh, rosenbrock, logistic; Richardson, Chebyshev, SOR solvers | the datasets |
| Dimension n | 1 to 100 | the datasets |
| Conditioning | κ 1 to 10⁴ (test functions), contrast 1 to ~400 (solvers) | the datasets |
| Model | `kimi-k2.6:nothink`, `kimi-k2.6:think`, `kimi-k3` (thinking is its default) | the plan |
| Feedback to the LLM | `loss` (current loss only) or `history` (every earlier θ and loss) | the plan |

The LLM always sees the problem's code, never its constants or the answer key. Every run
takes 10 TextGrad steps (3 LLM calls each), except the `long` stage, which takes 30, and
the two thinking-model stages, which take 5. The
LLM is never told how many steps it has, so results after fewer steps are simply the first
steps of a run: budgets from 1 to 10 need no separate runs. Enzyme gradient descent from
the same start, for the same number of steps, is recorded for every problem.

Each dataset setting has 3 random instances. Stages "A" use instance 1 of every setting;
stages "B" add instances 2 and 3, which gives error bars.

## The stages

`python/.venv/bin/python python/idea1/plan.py` prints this table from the code, so it is
always current.

| Order | Stage | Who | Problems | Runs | LLM calls | Time (4 workers) | Cost | Status |
|---|---|---|---|---|---|---|---|---|
| 0 | Enzyme timing (below) | Artemis | 330 | – | 0 | 4 min | free | **done** |
| 1 | `pilot` | Artemis | 6 | 12 | 360 | 21 min | $1.98 | **done** |
| 2 | `mainA-solvers` | Artemis | 20 | 40 | 1,200 | 70 min | $6.81 | **done** |
| 3 | `mainA-testfns` | Alaina | 90 | 180 | 5,400 | ~5 h | $29.00 | **done** |
| 4 | `models-k26think` | Artemis | 20 | 20 | 300 | 3 h | $14.26 | **done** |
| 5 | `models-k3` | Alaina | 20 | 20 | 300 | ~10 h? time it first (below) | ~$70–90? | to do |
| 6 | `samples` | Artemis | 20 | 20 | 1,800 | 70 min | $9.69 | **done** |
| 7 | `long` | Artemis | 20 | 20 | 1,800 | 2 h | $10.04 | **done** |
| 8 | `mainB-solvers` | Artemis | 40 | 80 | 2,400 | 3.5 h | $13.71 | **done** |
| 9 | `mainB-testfns` | Alaina | 180 | 360 | 10,800 | 13 h | $61 | **done** |
| 10 | `direct` | Artemis | 20 | 40 | 400 | 1 h | $5.91 | **done** |
| 11 | `opro` | Artemis | 20 | 20 | 800 | 4 h | $9.12 | **done** |
| 12 | derivative-free baselines (below) | Artemis | 330 | – | 0 | 30 s | free | **done** |

The order matters only within each person's list. We can run our stages at the same time.

- **models-k26think and models-k3** run 20 problems spread across families, n and
  conditioning with history feedback, for **5 steps** (changed from 10). They are compared
  with the first 5 steps of the same 20 problems from `mainA` (which used
  `kimi-k2.6:nothink`); since the LLM is never told its budget, those are directly
  comparable. Why 5: one measured `kimi-k2.6:think` step took 699 s and 44.7k output
  tokens ($0.18), about 20× a `nothink` step, so 10 steps would cost ~$40 for k2.6 and
  likely ~$140+ for `kimi-k3`, whose output costs 3.75× more.
- **samples** draws 3 proposals at every state on those 20 problems, all scored, to
  measure how consistent the LLM's steps are.
- **direct** and **opro** run other LLM optimizers on the same 20 problems with the same
  scoring (`python/idea1/other_methods.py`). `direct`: one LLM call per step, given the
  same information as TextGrad's critique call; it tests whether TextGrad's critique and
  feedback calls add anything. `opro`: OPRO (Yang et al., 2024), 4 proposals per step,
  shown the 20 best evaluations so far, best so far kept. Neither sees gradients. Both
  allow the model to reason before answering, so each call writes more than a TextGrad
  call: `direct` averaged ~$0.015 per call, which puts `opro` (800 calls) at ~$12–15.
- **long** runs the same 20 problems for 30 steps instead of 10, to see whether TextGrad
  keeps improving or plateaus.
- Times and costs for `nothink` stages come from finished stages: about 30 s per step and
  $0.16–0.17 per 10-step run. The large-n problems and history feedback have longer
  prompts, so allow extra.
- `models-k26think` (done) averaged ~7 min, ~34k output tokens and ~$0.14 per step. For
  `models-k3` the numbers are a guess assuming it thinks about as much (about $0.55 per
  step at k3's prices, probably slower); its `--limit 1` run will tell.
- **Empty answers with thinking models.** In `models-k26think`, 6 of 100 steps (all on
  logcosh problems) ended with an empty answer: the model used the API's default output
  limit thinking during the update call and never wrote θ. These are recorded as the
  LLM failing the step (`NoAnswerError`, error_kind `format`), not retried, and the run
  continues. We kept the API defaults for every model so the comparison is fair, so
  expect the same with `kimi-k3`; no action needed.

## Enzyme timing (free, no API)

For the speed and cost comparison with TextGrad. TextGrad's time per step is already
recorded in every run (`seconds` in `steps.csv`); this measures Enzyme's side once for all
330 problems, and also runs Enzyme gradient descent to convergence (up to 500 steps):

```
julia --project=julia julia/idea1/time_enzyme.jl data/idea1_testfns data/idea1_solvers
```

It writes `results/idea1/enzyme-timing/timing.jsonl`, one row per problem: Enzyme's
one-time compile time, the time of one gradient, one loss evaluation and one
gradient-descent step, the full gradient-descent loss curve, and the machine it ran on.
Commit that folder like a stage. Rerun it if the datasets change.

## Derivative-free baselines (free, no API)

The fair comparison for the LLM optimizers: classical methods that, like the LLMs, see
only loss values and no gradients. For every problem, from `THETA0`, they record the best
loss after each of 100 loss evaluations, so each LLM method can be compared at an equal
number of evaluations (TextGrad and `direct`: 1 per step; `opro`: 4 per step):

```
julia --project=julia julia/idea1/dfo_baselines.jl data/idea1_solvers data/idea1_testfns
```

- **(1+1)-ES:** random Gaussian steps with the 1/5th success rule for the step size, 5 seeds.
- **Nelder–Mead:** standard coefficients; its first n evaluations build the initial
  simplex, so on problems with n ≥ 100 it never gets past that.

It writes `results/idea1/dfo/dfo.jsonl` in about 30 s. Commit that folder like a stage.

## Setup (once)

```
git pull
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
python3.11 -m venv python/.venv
python/.venv/bin/pip install -r python/requirements.txt
```

Put your own Kimi key in a file named `.env` at the repo root. Git ignores this file.

```
KIMI_API_KEY=sk-...
```

Check that every problem loads and that Enzyme agrees with the stored gradients and with
finite differences (about 1.5 min, no API calls). Both datasets should end with `OK` and
no `FAIL` lines:

```
julia --project=julia julia/idea1/check_dataset.jl data/idea1_testfns data/idea1_solvers
```

Then test a stage end to end for free with the mock model:

```
python/.venv/bin/python python/idea1/run.py --stage mainA-testfns --mock --limit 2
rm -r results/idea1/mainA-testfns-mock
```

## Running a stage

Always pull first, so we both run on the same code and data:

```
git pull
caffeinate -i python/.venv/bin/python python/idea1/run.py --stage mainA-testfns
```

`caffeinate -i` keeps a Mac awake while it runs. On other systems, leave it out and make
sure the machine won't sleep.

- **Progress:** one line per TextGrad step, and `=== problems finished: k/N` after each
  problem.
- **Stopping and resuming:** Ctrl-C at any time. Rerun the exact same command later and it
  skips finished runs. Interrupted runs are redone from the start, and their partial rows
  are ignored by the summary.
- **API errors:** a failed call is retried every 60 s, up to 5 times. If it still fails,
  that run is marked `RUN FAILED` and redone on the next launch. API errors are never
  counted as the LLM's mistakes. If you see many, Kimi is rate-limiting: rerun with
  `--workers 2`.
- **Slow models (stages 4 and 5):** time one problem first, then run the rest. The second
  command skips the problem already done.
  ```
  python/.venv/bin/python python/idea1/run.py --stage models-k3 --limit 1
  python/.venv/bin/python python/idea1/run.py --stage models-k3
  ```
  If one 5-step `kimi-k3` run costs much more than ~$5 or takes much more than ~3 hours,
  check with each other before running the rest.

## What gets recorded

Each stage writes only to its own folder, `results/idea1/<stage>/`, so our results never
conflict:

| File | One line per | Contents |
|---|---|---|
| `steps.jsonl` | TextGrad proposal | problem, family, n, knobs, model, feedback, step; θ before and after; Enzyme's scores (cosine to −∇f, descent, step-length ratio, gap to f*, distance to θ*, cosine to θ*); the LLM's critique, feedback and update text; tokens and seconds |
| `gd.jsonl` | problem | Enzyme gradient descent's loss after each step |
| `runs.jsonl` | finished run | final loss, total LLM calls and tokens |
| `config.jsonl` | launch | settings used and the git commit they ran on |

Every θ the LLM visited is saved at full precision. Any further metric can be computed
afterwards from the saved runs, with no new API calls.

## When a stage finishes

1. Make the summary spreadsheets and plots (free, a few seconds):
   ```
   python/.venv/bin/python python/idea1/summarize.py results/idea1/mainA-testfns
   python/.venv/bin/python python/idea1/plots.py results/idea1/mainA-testfns
   ```
   This adds to the stage folder:
   - `summary.csv`: one row per problem setting, model and feedback mode, with the
     direction, descent and step-size metrics and both final losses
   - `steps.csv`: one row per TextGrad step with all of Enzyme's scores (the LLM's text
     stays in `steps.jsonl`)
   - `plots/`: loss vs. Enzyme gradient descent per problem (stages with up to 24
     problems), the remaining gap to f* per family, and per-step direction, step size
     and descent
2. Commit and push the folder, CSVs and plots included:
   ```
   git add results/idea1/mainA-testfns
   git commit -m "Idea 1 results: mainA-testfns"
   git pull --rebase && git push
   ```

## Combining everything

`summarize.py` and `plots.py` take any number of folders and treat them as one dataset:

```
python/.venv/bin/python python/idea1/summarize.py results/idea1/*
python/.venv/bin/python python/idea1/plots.py results/idea1/*
```

Every row records its own problem, model and settings, so stages run by different people
on different machines combine directly. The combined CSVs and plots go to
`results/idea1/combined/`, which git ignores: either of us can regenerate them at any
time, so they are never committed and never conflict.
