"""Information-matched reference for TextGrad's history-feedback steps (no API calls).

At every saved step with history feedback, this rebuilds exactly the history the prompt showed
(the last HISTORY_LEN earlier (θ, loss) pairs plus the current θ and loss, rounded to 6
significant digits as in the prompt) and estimates the gradient from those numbers alone: a
least-squares plane through the current point (minimum norm when there are fewer than n + 1
points). Its downhill direction is scored against the exact gradient, exactly like the LLM's step.
A second reference moves from the current point toward the best point shown, when an earlier
point had a lower loss.

The question it answers: does the LLM get more out of the history than simple arithmetic on the
same numbers would? (The LLM also sees the code; the references see only the numbers.)

The exact gradient is computed by the complex-step method on the Python evaluators in
python/audit.py, which mirror the Julia programs line for line. At the LLM's steps it is checked
against the cosine Enzyme stored, and the script stops if they disagree.

    python3 python/idea1/history_reference.py mainA-testfns mainB-testfns

Needs numpy, scipy, h5py and pandas (python/audit.py imports the dataset generators); the
TextGrad venv does not have scipy or h5py, so use the Python you generate datasets with.
Writes results/idea1/history_reference/history_reference.csv (one row per scored proposal)
and prints a summary by n.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
from audit import eval_solver, eval_testfn_meta, parse_consts  # noqa: E402

HISTORY_LEN = 10   # as in textgrad_step.py


def r6(x):
    return float(f"{x:.6g}")   # the rounding used in the prompt


_problems = {}


def problem_loss(program):
    """The problem's loss as a Python function of θ (also accepts complex θ)."""
    if program not in _problems:
        path = REPO / program
        pid = path.stem[1:]
        suite = path.parent.parent
        meta = json.loads((suite / "hidden" / f"P{pid}.json").read_text())
        consts = parse_consts((suite / "constants" / f"P{pid}.jl").read_text())
        src = path.read_text()
        evaluate = eval_solver if meta["family"].startswith("solver_") else eval_testfn_meta
        _problems[program] = lambda th, m=meta, c=consts, s=src, e=evaluate: e(m, c, s, th)
    return _problems[program]


def exact_gradient(f, theta, h=1e-30):
    theta = np.asarray(theta, float)
    g = np.zeros(len(theta))
    for i in range(len(theta)):
        e = np.zeros(len(theta), complex)
        e[i] = h * 1j
        with np.errstate(all="ignore"):
            g[i] = np.imag(f(theta + e)) / h
    return g


def plane_gradient(points, cur_theta, cur_loss):
    """Least-squares slope of loss vs θ through the current point; minimum norm if underdetermined."""
    X = np.array([np.asarray(t) - cur_theta for t, _ in points])
    y = np.array([f - cur_loss for _, f in points])
    keep = np.linalg.norm(X, axis=1) > 0
    if not keep.any():
        return None
    g, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
    return g if np.linalg.norm(g) > 0 else None


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


def accepted(row):
    """Mirrors run.py: the trajectory follows sample 0; a diverged proposal leaves θ unchanged."""
    return isinstance(row.theta_new, str) and not bool(row.diverged)


def process_stage(stage, results):
    steps = pd.read_csv(results / stage / "steps.csv")
    steps = steps[steps["feedback"] == "history"]
    rows = []
    for run_id, run in steps.groupby("run_id"):
        run = run.sort_values(["step", "sample"])
        program = run["program"].iloc[0]
        f = problem_loss(program)
        first = run[(run["step"] == 0) & (run["sample"] == 0)].iloc[0]
        theta, loss = np.array(json.loads(first.theta)), first.loss_before
        past = []   # earlier (θ, loss) shown to the LLM, oldest first, as in run.py
        for step, proposals in run.groupby("step"):
            g = exact_gradient(f, theta)
            shown = [(np.array([r6(x) for x in t]), r6(fv)) for t, fv in past[-HISTORY_LEN:]
                     if math.isfinite(fv)]
            cur_theta, cur_loss = np.array([r6(x) for x in theta]), r6(loss)
            g_plane = plane_gradient(shown, cur_theta, cur_loss) if shown else None
            plane_cos = cosine(-g_plane, -g) if g_plane is not None else np.nan
            better = [p for p in shown if p[1] < cur_loss]
            best_cos = (cosine(min(shown, key=lambda p: p[1])[0] - cur_theta, -g)
                        if better else np.nan)
            for _, p in proposals.iterrows():
                new = json.loads(p.theta_new) if isinstance(p.theta_new, str) else None
                llm_cos = cosine(np.array(new) - theta, -g) if new is not None else np.nan
                if new is not None and abs(llm_cos - p.cos_to_neg_grad) > 1e-6:
                    raise RuntimeError(f"gradient check failed: {program} step {step}: "
                                       f"cosine {llm_cos} vs Enzyme's {p.cos_to_neg_grad}")
                rows.append(dict(stage=stage, program=program, family=p.family, n=int(p.n),
                                 knobs=p.knobs, model=p.model, run_id=run_id, step=int(step),
                                 sample=int(p["sample"]), points_shown=len(shown),
                                 llm_cos=llm_cos, plane_cos=plane_cos, toward_best_cos=best_cos))
            take = proposals[proposals["sample"] == 0].iloc[0]
            new = json.loads(take.theta_new) if isinstance(take.theta_new, str) else None
            if new is not None and bool(take.diverged):
                past.append((new, math.inf))
            elif new is not None:
                past.append((theta.tolist(), loss))
                theta, loss = np.array(new), take.loss_after
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+", help="results folders under --results, e.g. mainA-testfns")
    ap.add_argument("--results", default=str(REPO / "results" / "idea1"))
    ap.add_argument("--out", default=None, help="default: <results>/history_reference")
    args = ap.parse_args()

    results = Path(args.results)
    table = pd.concat([process_stage(s, results) for s in args.stages], ignore_index=True)
    out = Path(args.out) if args.out else results / "history_reference"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "history_reference.csv", index=False)

    both = table[table["points_shown"] > 0].dropna(subset=["llm_cos", "plane_cos"])
    by_n = both.groupby("n").agg(steps=("llm_cos", "size"), llm_cos=("llm_cos", "mean"),
                                 plane_cos=("plane_cos", "mean"),
                                 llm_downhill=("llm_cos", lambda c: (c > 0).mean()),
                                 plane_downhill=("plane_cos", lambda c: (c > 0).mean()))
    pd.set_option("display.width", 200)
    print(by_n.round(2).to_string())
    print(f"\nAll: {len(both)} steps; mean cosine LLM {both.llm_cos.mean():.2f}, "
          f"plane fit {both.plane_cos.mean():.2f}")
    tb = both.dropna(subset=["toward_best_cos"])
    print(f"Steps where an earlier point was better ({len(tb) / len(both):.0%}): "
          f"LLM {tb.llm_cos.mean():.2f}, toward the best point {tb.toward_best_cos.mean():.2f}")
    print(f"Wrote {out / 'history_reference.csv'}")


if __name__ == "__main__":
    main()
