"""Plots for one or more results folders written by run.py.

    python/.venv/bin/python python/idea1/plots.py results/idea1/pilot
    python/.venv/bin/python python/idea1/plots.py results/idea1/*        # everything

Writes PNGs to <folder>/plots/ (or results/idea1/combined/plots/ for several folders), one
set per optimizer and model (<model> is prefixed with "direct-" or "opro-" for those methods):
    loss_curves_<model>.png    per problem: loss − f* after each step, TextGrad (one line per
                               feedback mode) vs Enzyme gradient descent. Only for ≤ 24 problems.
    gap_by_family_<model>.png  per family: median fraction of the starting gap to f* that
                               remains after each step, over all problems in the family
    step_metrics_<model>.png   per step: cosine to −∇f, step-length ratio and descent rate,
                               median and middle 50% over all runs
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import default_out, load  # noqa: E402

SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
# Fixed series colors (validated categorical slots 1-3); Enzyme is also dashed.
FEEDBACK_COLOR = {"loss": "#2a78d6", "history": "#eb6834"}
ENZYME_COLOR = "#1baf7a"
GAP_FLOOR = 1e-8   # gaps at or below this (reached the best known value) are drawn here
MAX_PROBLEM_PANELS = 24
RATIO_CAP = 100.0  # step_ratio values above this (including uphill steps, ratio = inf) are drawn here

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
    "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "lines.linewidth": 1.8, "lines.markersize": 4, "legend.frameon": False,
})

KNOB_LABELS = {"kappa": "κ", "spread": "spread", "b": "b", "contrast": "contrast"}


def problem_label(family, n, knobs):
    knobs = json.loads(knobs) if isinstance(knobs, str) else (knobs or {})
    parts = [f"{KNOB_LABELS[k]}={v:.3g}" for k, v in knobs.items() if k in KNOB_LABELS]
    return f"{family.replace('solver_', '')}  n={n}" + ("  " + " ".join(parts) if parts else "")


def trajectories(steps, gd):
    """One row per run: settings plus the loss after each step (index 0 = THETA0).
    A failed or diverged step leaves θ, and so the loss, unchanged."""
    rows = []
    for run_id, g in steps[steps["sample"] == 0].sort_values("step").groupby("run_id"):
        first = g.iloc[0]
        loss = gd.loc[first["program"], "losses"][0]
        losses = [loss]
        for _, r in g.iterrows():
            if isinstance(r["theta_new"], list) and r.get("diverged") is not True:
                loss = r["loss_after"]
            losses.append(loss)
        rows.append(dict(run_id=run_id, program=first["program"], family=first["family"],
                         n=first["n"], knobs=first["knobs"], model=first["model"],
                         spec=first["spec"] if "spec" in first else first["model"],
                         feedback=first["feedback"], losses=losses))
    return pd.DataFrame(rows)


def gap(losses, f_star):
    g = np.asarray(losses, dtype=float) - f_star
    g[~np.isfinite(g)] = np.nan
    return np.maximum(g, GAP_FLOOR)


def optimizer_name(spec):
    method = spec.split("/")[0] if "/" in spec else "textgrad"
    return {"textgrad": "TextGrad", "direct": "Direct prompting", "opro": "OPRO"}[method]


def legend(fig, feedbacks, spec):
    handles = [plt.Line2D([], [], color=FEEDBACK_COLOR[f], label=f"{optimizer_name(spec)}, {f} feedback")
               for f in feedbacks]
    handles.append(plt.Line2D([], [], color=ENZYME_COLOR, linestyle="--", label="Enzyme gradient descent"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.0))


def grid(n_panels, width=3.2, height=2.4, max_cols=4):
    cols = min(max_cols, n_panels)
    rows = math.ceil(n_panels / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(width * cols, height * rows + 0.6), squeeze=False)
    for ax in axes.flat[n_panels:]:
        ax.set_visible(False)
    return fig, axes.flat


def plot_loss_curves(traj, gd, model, out):
    traj = traj[traj["spec"] == model]
    programs = traj.sort_values(["family", "n", "knobs"])["program"].unique()
    if len(programs) > MAX_PROBLEM_PANELS:
        return None
    feedbacks = [f for f in FEEDBACK_COLOR if f in set(traj["feedback"])]
    fig, axes = grid(len(programs))
    for ax, program in zip(axes, programs):
        runs = traj[traj["program"] == program]
        f_star = gd.loc[program, "f_star"]
        for _, r in runs.iterrows():
            ax.plot(gap(r["losses"], f_star), color=FEEDBACK_COLOR[r["feedback"]], marker="o")
        ax.plot(gap(gd.loc[program, "losses"], f_star), color=ENZYME_COLOR, linestyle="--")
        first = runs.iloc[0]
        ax.set_title(problem_label(first["family"], first["n"], first["knobs"]), loc="left")
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("loss − f*")
    legend(fig, feedbacks, model)
    fig.suptitle(f"Loss above the optimum after each step ({model})", y=1.04, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = out / f"loss_curves_{model.replace(':', '-').replace('/', '-')}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_gap_by_family(traj, gd, model, out):
    traj = traj[traj["spec"] == model]
    families = sorted(traj["family"].unique())
    feedbacks = [f for f in FEEDBACK_COLOR if f in set(traj["feedback"])]
    fig, axes = grid(len(families))
    for ax, family in zip(axes, families):
        fam = traj[traj["family"] == family]
        for feedback in feedbacks:
            rel = [gap(r["losses"], gd.loc[r["program"], "f_star"]) / gap(r["losses"], gd.loc[r["program"], "f_star"])[0]
                   for _, r in fam[fam["feedback"] == feedback].iterrows()]
            ax.plot(np.nanmedian(rel, axis=0), color=FEEDBACK_COLOR[feedback], marker="o")
        rel_gd = [gap(gd.loc[p, "losses"], gd.loc[p, "f_star"]) / gap(gd.loc[p, "losses"], gd.loc[p, "f_star"])[0]
                  for p in fam["program"].unique()]
        ax.plot(np.nanmedian(rel_gd, axis=0), color=ENZYME_COLOR, linestyle="--")
        ax.axhline(1.0, color=TEXT2, linewidth=0.8, linestyle=":")
        count = fam["program"].nunique()
        ax.set_title(f"{family.replace('solver_', '')}  ({count} problem{'s' if count != 1 else ''})", loc="left")
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("gap / starting gap")
    legend(fig, feedbacks, model)
    fig.suptitle(f"Fraction of the starting gap to f* that remains (median, {model})",
                 y=1.04, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = out / f"gap_by_family_{model.replace(':', '-').replace('/', '-')}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_step_metrics(steps, model, out):
    ok = steps[(steps["spec"] == model) & steps["theta_new"].map(lambda t: isinstance(t, list))].copy()
    ok["step1"] = ok["step"] + 1
    # A step with no useful length along its direction (uphill) is infinitely too long; cap for plotting.
    ok["ratio"] = ok["step_ratio"].astype(float).fillna(np.inf).clip(upper=RATIO_CAP)
    feedbacks = [f for f in FEEDBACK_COLOR if f in set(ok["feedback"])]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    panels = [("cos_to_neg_grad", "cosine to −∇f  (1 = steepest, 0 = random)", "Direction"),
              ("ratio", f"step length / best length  (1 = ideal, cap {RATIO_CAP:g})", "Step size"),
              ("descent", "fraction of steps that lowered the loss", "Descent")]
    for ax, (col, ylabel, title) in zip(axes, panels):
        for feedback in feedbacks:
            by_step = ok[ok["feedback"] == feedback].groupby("step1")[col]
            x = np.array(sorted(by_step.groups))
            if col == "descent":
                ax.plot(x, by_step.mean().values, color=FEEDBACK_COLOR[feedback], marker="o")
                continue
            q = by_step.quantile([0.25, 0.5, 0.75]).unstack()
            ax.plot(x, q[0.5].values, color=FEEDBACK_COLOR[feedback], marker="o")
            ax.fill_between(x, q[0.25].values, q[0.75].values, color=FEEDBACK_COLOR[feedback],
                            alpha=0.15, linewidth=0)
        if col == "ratio":
            ax.set_yscale("log")
            ax.set_ylim(1e-2, RATIO_CAP * 1.5)
            ax.axhline(1.0, color=TEXT2, linewidth=0.8, linestyle=":")
        if col == "cos_to_neg_grad":
            ax.axhline(0.0, color=TEXT2, linewidth=0.8, linestyle=":")
            ax.set_ylim(-1, 1)
        if col == "descent":
            ax.set_ylim(0, 1)
        ax.set_title(title, loc="left")
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
    handles = [plt.Line2D([], [], color=FEEDBACK_COLOR[f], label=f"{optimizer_name(model)}, {f} feedback") for f in feedbacks]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Every {optimizer_name(model)} step scored against Enzyme ({model}; median and middle 50%)",
                 y=1.1, fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = out / f"step_metrics_{model.replace(':', '-').replace('/', '-')}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--out", help="where to write the plots folder")
    args = ap.parse_args()

    runs, steps, gd = load(args.folders)
    out = (Path(args.out) if args.out else default_out(args.folders)) / "plots"
    out.mkdir(parents=True, exist_ok=True)
    # One set of figures per optimizer and model; TextGrad results are named by model alone.
    steps["spec"] = [m if meth == "textgrad" else f"{meth}/{m}" for meth, m in zip(steps["method"], steps["model"])]
    traj = trajectories(steps, gd)
    for model in sorted(traj["spec"].unique()):
        for path in (plot_loss_curves(traj, gd, model, out),
                     plot_gap_by_family(traj, gd, model, out),
                     plot_step_metrics(steps, model, out)):
            if path:
                print(f"Wrote {path}")


if __name__ == "__main__":
    main()
