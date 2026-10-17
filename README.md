# mixed-mode-diff

## Idea 1: TextGrad vs. Enzyme on the same Julia program

TextGrad (an LLM) updates a problem's parameters θ. Every update is scored against
Enzyme's exact gradient at that point: cosine to −∇f, whether the loss went down, and
the step length relative to the best length along the same direction. Enzyme's own
gradient descent runs from the same start for the same number of steps.

```
data/<dataset>/problems/P<id>.jl   problem shown to the LLM and differentiated by Enzyme
data/<dataset>/hidden/             answer key (θ*, f*, settings), used only for scoring
julia/idea1/                       loading, Enzyme scoring, dataset checks, Python bridge
python/idea1/                      TextGrad loop, sweep runner, summary
```

A problem file is a `module` defining `loss(θ::AbstractVector)` and `THETA0`; see
`julia/idea1/load_program.jl`.

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

Run TextGrad on some problems, then summarize (`--models mock` tests the pipeline for free):

```
python/.venv/bin/python python/idea1/run.py --programs data/idea1_testfns/problems/P7f72b5e797.jl \
    --models kimi-k2.6:nothink --feedback loss history --steps 10 --name my-run
python/.venv/bin/python python/idea1/summarize.py results/idea1/my-run
```

`python python/idea1/run.py --help` lists every setting that can be swept.
