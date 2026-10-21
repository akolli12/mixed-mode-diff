# mixed-mode-diff

LLMs and exact derivatives, divided by job: the LLM proposes program structure, Enzyme's
exact gradients set the numbers. Two parts:

- **Idea 1 (Part A):** are LLMs good at numerical updates? LLM optimizers (TextGrad,
  direct prompting, OPRO) change a frozen Julia program's parameters, and every step is
  scored against Enzyme's exact gradient.
- **Idea 2 (Part B):** gradient-guided LLM discovery of robust solvers for hard PDE
  problem families (high-contrast diffusion, high-frequency Helmholtz).

```
data/idea1_testfns, data/idea1_solvers   Idea 1 problems (see "Problem format" below)
data/diffusion, data/helmholtz           Idea 2 problem families (HDF5, train/test splits)
julia/idea1/                             Idea 1: loading, Enzyme scoring, dataset checks,
                                         timing, derivative-free baselines, Python bridge
julia/idea2/                             Idea 2: harness (solver runs, Enzyme tuning, gradient
                                         report, scoring), starting programs, Python bridge
julia/MMDData/                           Julia loaders and matrix assemblers for all datasets
python/idea1/                            Idea 1: TextGrad and other LLM optimizers, run plan
                                         and runner, summary and plots
python/idea2/                            Idea 2: search loop and the LLM's prompts
python/idea1.py, python/idea1_solvers.py Idea 1 dataset generators
python/diffusion.py, python/helmholtz.py Idea 2 dataset generators
python/audit.py                          independent check of every generated dataset file
python/baselines.py, python/figures.py   Idea 2 baseline solver sweeps and their figures
make_all.sh                              regenerates all datasets from fixed seeds, then audits
docs/idea1_runs.md                       Idea 1 run plan: stages, who runs what, results
docs/idea2.md                            Idea 2 plan and status: steps, what is built, how it works
results/idea1/<stage>/                   Idea 1 results, one folder per stage
results/idea2/<name>/                    Idea 2 search runs, one folder per run
```

## Setup

```
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
python3.11 -m venv python/.venv && python/.venv/bin/pip install -r python/requirements.txt
```

The dataset generators, audit and Idea 2 baselines also need
`pip install numpy scipy h5py pyamg matplotlib`.

For the LLM runs, put your Kimi key in a file `.env` at the repo root (git ignores it):

```
KIMI_API_KEY=sk-...
```

## Idea 1: LLM optimizers vs. exact gradients

An LLM optimizer updates a problem's parameters θ. Every update is scored against
Enzyme's exact gradient at that point: cosine to −∇f, whether the loss went down, and
the step length relative to the best length along the same direction. The LLM never sees
the gradient. Enzyme's own gradient descent runs from the same start for the same number
of steps, and classical derivative-free methods, which like the LLMs see only loss
values, run on the same problems.

### Problem format

```
data/<dataset>/problems/P<id>.jl    the code: shown to the LLM, differentiated by Enzyme
data/<dataset>/constants/P<id>.jl   the numbers the code needs and THETA0; never shown to the LLM
data/<dataset>/hidden/              answer key (θ*, f*, settings), used only for scoring
```

A problem file is a `module` defining `loss(θ::AbstractVector)`; it includes its constants
file, which defines `THETA0`. The LLM sees the code but not the constants, so it can't read
off the optimum and has to search for it. See `julia/idea1/load_program.jl`.

### Run

Check that the datasets are in the right format and that Enzyme matches the stored values
and finite differences:

```
julia --project=julia julia/idea1/check_dataset.jl data/idea1_testfns data/idea1_solvers
```

The full run plan (stages, who runs what, how results are recorded and combined) is in
**[docs/idea1_runs.md](docs/idea1_runs.md)**; the stages themselves are defined in
`python/idea1/plan.py`. Run a stage, then summarize (add `--mock` to test for free):

```
python/.venv/bin/python python/idea1/plan.py                  # list the stages
python/.venv/bin/python python/idea1/run.py --stage pilot
python/.venv/bin/python python/idea1/summarize.py results/idea1/pilot   # table + CSVs
python/.venv/bin/python python/idea1/plots.py results/idea1/pilot       # PNGs in pilot/plots/
```

`run.py` also takes problems, models and feedback modes by hand; `run.py --help` lists
every option, including `--method direct|opro` for LLM optimizers other than TextGrad.
`julia/idea1/time_enzyme.jl` times Enzyme on every problem for the speed comparison, and
`julia/idea1/dfo_baselines.jl` runs classical derivative-free baselines (both free; see
the guide).

## Idea 2: problem families and baselines

- **High-contrast diffusion** (`python/diffusion.py`): −∇·(k∇u) = f on the unit square,
  cell-centred finite volumes, four coefficient families (lognormal, channels, inclusions,
  barriers) at N = 64 and 128, with contrast swept from 10⁰ to 10⁸ on fixed geometry.
- **High-frequency Helmholtz** (`python/helmholtz.py`): with a PML, four media
  (constant, layered, smooth, inclusions), frequency swept from 2 to 32 wavelengths across
  the domain at 10 points per wavelength.
- **Baselines** (`python/baselines.py`): CG and Jacobi-, ILU(0)- and AMG-preconditioned CG
  on diffusion; GMRES with and without the shifted-Laplacian preconditioner on Helmholtz.
- **Julia access** (`julia/MMDData`): loaders and assemblers that rebuild each system
  exactly as the Python generators do, with `selfcheck` against stored reference values.

Regenerate every dataset and audit it with `./make_all.sh` (add `--baselines` to also run
the baseline sweeps, about an hour).

## Idea 2: gradient-guided solver search

Each iteration the LLM sees the best program so far and its validation results, and writes
a new one. The program is compiled, its tunable numbers θ are tuned with Enzyme gradients
of a training loss (mean log10 ‖r‖/‖b‖ after a fixed number of iterations), and it is scored
on validation instances by its iterations to ‖r‖/‖b‖ ≤ 1e-8 (score = mean log10 iterations;
for runs that hit the cap, the count is extrapolated from the convergence rate). The best
program so far is the parent of the next iteration. Two kinds of program (`--frame`):

- `precond`: the LLM writes a preconditioner `setup`/`precond!`; a fixed flexible CG applies it.
- `solver`: the LLM writes the whole iteration `init`/`step!`.

Three feedback modes (`--mode`), compared at the same number of LLM calls:

| mode | θ set by | the LLM sees |
|---|---|---|
| `llm_only` | the LLM | validation results |
| `score_only` | Enzyme tuning | + the tuned θ |
| `ours` | Enzyme tuning | + the gradient report (sensitivities, θ at bounds, loss by contrast) |

The starting programs are in `julia/idea2/seeds/` (Jacobi polynomial preconditioner;
damped Jacobi with momentum). Check the harness for free (Enzyme vs finite differences,
tuning, iteration counts), then run a search (`--model mock` tests the loop for free):

```
julia --project=julia julia/idea2/check_harness.jl channels 64
python/.venv/bin/python python/idea2/search.py --family channels --N 64 --frame precond \
    --mode ours --iterations 20 --name channels64-precond-ours
```

Training instances 1–4 and validation instances 5–8 come from the train split; at the end
the best program and the starting program are scored on all test-split instances.
`search.py --help` lists every option. Outputs in `results/idea2/<name>/`: `config.json`,
`candidates.jsonl` (every reply, program and evaluation), `test.json`.
