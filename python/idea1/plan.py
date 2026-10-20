"""The Idea 1 run plan: every stage, which problems, models and feedback modes it uses,
and who runs it. Run a stage with `run.py --stage <name>`; list them with

    python/.venv/bin/python python/idea1/plan.py

Each dataset setting (family, n, knobs) has 3 random instances. Instances are ordered by
problem id, so "instance 1" is the same file on every machine.
"""

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["data/idea1_testfns", "data/idea1_solvers"]
TESTFNS = ("quadratic", "logcosh", "rosenbrock", "logistic")
SOLVERS = ("solver_richardson", "solver_chebyshev", "solver_sor")
NOTHINK = "kimi-k2.6:nothink"
STEPS = 10
THINKING_STEPS = 5
CALLS_PER_PROPOSAL = {"textgrad": 3, "direct": 1, "opro": 1}
OPRO_CANDIDATES = 4


def load_settings():
    """{(family, n, knobs as JSON): [problem paths sorted by id]} for every dataset."""
    settings = defaultdict(list)
    for ds in DATASETS:
        for entry in json.loads((REPO_ROOT / ds / "manifest.json").read_text()):
            path = Path(ds) / entry["file"]
            hidden = json.loads((REPO_ROOT / ds / "hidden" / (path.stem + ".json")).read_text())
            key = (hidden["family"], hidden["n"], json.dumps(hidden.get("knobs") or {}, sort_keys=True))
            settings[key].append(str(path))
    return {k: sorted(v) for k, v in sorted(settings.items())}


SETTINGS = load_settings()


def instances(families, which):
    """Instance numbers `which` (0-based) of every setting in `families`."""
    return [paths[i] for (family, _, _), paths in SETTINGS.items()
            if family in families for i in which]


def pick(family, n, **knobs):
    """Instance 0 of the one setting matching family, n and the given knobs."""
    for (fam, n_, kn), paths in SETTINGS.items():
        kn = json.loads(kn)
        if fam == family and n_ == n and all(abs(kn[k] - v) < 1e-9 for k, v in knobs.items()):
            return paths[0]
    raise KeyError(f"no problem with family={family} n={n} {knobs}")


PILOT = [
    pick("quadratic", 2, kappa=1000),
    pick("quadratic", 10, kappa=10),
    pick("logcosh", 20, kappa=100),
    pick("rosenbrock", 5, b=10),
    pick("logistic", 10, spread=10),
    pick("solver_richardson", 8, amp=2.0),
]

# 20 problems spread over families, n and conditioning, for comparing models.
MODEL_SUBSET = (
    [pick("quadratic", n, kappa=k) for n in (2, 10, 50) for k in (10, 1000)]
    + [pick("logcosh", n, kappa=k) for n in (5, 20) for k in (10, 1000)]
    + [pick("rosenbrock", n, b=b) for n in (2, 10) for b in (10, 100)]
    + [pick("logistic", n, spread=s) for n in (5, 20) for s in (1, 100)]
    + [pick("solver_richardson", 8, amp=a) for a in (0.0, 3.0)]
)


def stage(owner, programs, models, feedback, samples=1, steps=STEPS, method="textgrad", about=""):
    return dict(owner=owner, programs=programs, models=models, feedback=feedback,
                samples=samples, steps=steps, method=method, about=about)


# In the order they should be run.
STAGES = {
    "pilot": stage("Artemis", PILOT, [NOTHINK], ["loss", "history"],
                   about="6 problems; checks metrics and token cost before anything else"),
    "mainA-solvers": stage("Artemis", instances(SOLVERS, [0]), [NOTHINK], ["loss", "history"],
                           about="instance 1 of every solver setting"),
    "mainA-testfns": stage("Alaina", instances(TESTFNS, [0]), [NOTHINK], ["loss", "history"],
                           about="instance 1 of every test-function setting: the n and κ sweep"),
    # Thinking models get 5 steps: a thinking step costs ~20x a nothink step (measured: 699 s
    # and 44.7k output tokens for one kimi-k2.6:think step). They are compared with the first
    # 5 steps of the nothink runs on the same problems; the LLM is never told its budget.
    "models-k26think": stage("Artemis", MODEL_SUBSET, ["kimi-k2.6:think"], ["history"],
                             steps=THINKING_STEPS, about="same model with thinking on"),
    "models-k3": stage("Alaina", MODEL_SUBSET, ["kimi-k3"], ["history"], steps=THINKING_STEPS,
                       about="strongest model; slow, run --limit 1 first"),
    "samples": stage("Artemis", MODEL_SUBSET, [NOTHINK], ["history"], samples=3,
                     about="3 proposals per state: how consistent are the LLM's steps?"),
    "long": stage("Artemis", MODEL_SUBSET, [NOTHINK], ["history"], steps=30,
                  about="30 steps instead of 10: does TextGrad keep improving or plateau?"),
    # Other LLM optimizers on the same 20 problems; neither sees gradients (other_methods.py).
    "direct": stage("Artemis", MODEL_SUBSET, [NOTHINK], ["loss", "history"], method="direct",
                    about="one LLM call per step, no critique or feedback call"),
    "opro": stage("Artemis", MODEL_SUBSET, [NOTHINK], ["history"], method="opro",
                  samples=OPRO_CANDIDATES, about="OPRO: 4 proposals per step, best so far kept"),
    "mainB-solvers": stage("Artemis", instances(SOLVERS, [1, 2]), [NOTHINK], ["loss", "history"],
                           about="instances 2-3 of every solver setting, for error bars"),
    "mainB-testfns": stage("Alaina", instances(TESTFNS, [1, 2]), [NOTHINK], ["loss", "history"],
                           about="instances 2-3 of every test-function setting, for error bars"),
}


def size(s):
    runs = len(s["programs"]) * len(s["models"]) * len(s["feedback"])
    return runs, runs * s["steps"] * s["samples"] * CALLS_PER_PROPOSAL[s["method"]]


if __name__ == "__main__":
    print(f"{'stage':<16} {'owner':<8} {'problems':>8} {'runs':>5} {'LLM calls':>9}  about")
    for name, s in STAGES.items():
        runs, calls = size(s)
        print(f"{name:<16} {s['owner']:<8} {len(s['programs']):>8} {runs:>5} {calls:>9}  {s['about']}")
