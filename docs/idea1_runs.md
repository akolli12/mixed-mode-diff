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
takes 10 TextGrad steps (3 LLM calls each), except the `long` stage, which takes 30. The
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
| 0 | Enzyme timing (below) | Artemis | 330 | – | 0 | ~15 min | free | to do |
| 1 | `pilot` | Artemis | 6 | 12 | 360 | 21 min | $1.98 | **done** |
| 2 | `mainA-solvers` | Artemis | 20 | 40 | 1,200 | 70 min | $6.81 | **done** |
| 3 | `mainA-testfns` | Alaina | 90 | 180 | 5,400 | ~5 h | ~$30–40 | to do |
| 4 | `models-k26think` | Artemis | 20 | 20 | 600 | unknown: time it first (below) | ~$8–25 | to do |
| 5 | `models-k3` | Alaina | 20 | 20 | 600 | unknown, likely 5–10 h: time it first | ~$50–140 | to do |
| 6 | `samples` | Artemis | 20 | 20 | 1,800 | ~1.5 h | ~$10 | to do |
| 7 | `long` | Artemis | 20 | 20 | 1,800 | ~1.5 h | ~$10–15 | to do |
| 8 | `mainB-solvers` | Artemis | 40 | 80 | 2,400 | ~2.5 h | ~$14 | to do |
| 9 | `mainB-testfns` | Alaina | 180 | 360 | 10,800 | ~10 h | ~$60–75 | to do |

The order matters only within each person's list. We can run our stages at the same time.

- **models-k26think and models-k3** run 20 problems spread across families, n and
  conditioning with history feedback. They are compared with the same 20 problems from
  `mainA` (which used `kimi-k2.6:nothink`).
- **samples** draws 3 proposals at every state on those 20 problems, all scored, to
  measure how consistent the LLM's steps are.
- **long** runs the same 20 problems for 30 steps instead of 10, to see whether TextGrad
  keeps improving or plateaus.
- Times and costs come from the pilot and `mainA-solvers`: about 30 s per step and $0.17
  per 10-step run with `nothink`. The large-n problems and history feedback have longer
  prompts, so allow extra. The thinking models are the uncertain ones.

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
  If one `kimi-k3` run takes much more than 2 hours, check with each other before running
  the rest.

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
