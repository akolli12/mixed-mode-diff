# mixed-mode-diff

## Idea 1: TextGrad vs. Enzyme on the same Julia program

TextGrad (an LLM) updates a problem's parameters θ. Every update is scored against
Enzyme's exact gradient at that point: cosine to −∇f, whether the loss went down, and
the step length relative to the best length along the same direction. Enzyme's own
gradient descent runs from the same start for the same number of steps.

```
data/<dataset>/problems/P<id>.jl    the code: shown to the LLM, differentiated by Enzyme
data/<dataset>/constants/P<id>.jl   the numbers the code needs and THETA0; never shown to the LLM
data/<dataset>/hidden/              answer key (θ*, f*, settings), used only for scoring
julia/idea1/                        loading, Enzyme scoring, dataset checks, Python bridge
python/idea1/                       TextGrad loop, run plan and runner, summary and plots
results/idea1/<stage>/              one folder per stage: raw results, CSVs and plots
```

A problem file is a `module` defining `loss(θ::AbstractVector)`; it includes its constants
file, which defines `THETA0`. The LLM sees the code but not the constants, so it can't read
off the optimum and has to search for it. See `julia/idea1/load_program.jl`.

### Setup

```
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
python3.11 -m venv python/.venv && python/.venv/bin/pip install -r python/requirements.txt
```

Put your Kimi key in a file `.env` at the repo root (git ignores it):

```
KIMI_API_KEY=sk-...
```

### Run

Check that a dataset is in the right format and that Enzyme matches the stored values
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
every option. `julia/idea1/time_enzyme.jl` times Enzyme on every problem for the speed
comparison (free; see the guide).
