"""Summarizes one or more results folders written by run.py, treated as one dataset.

    python/.venv/bin/python python/idea1/summarize.py results/idea1/pilot
    python/.venv/bin/python python/idea1/summarize.py results/idea1/*        # everything

Prints the table below and writes two spreadsheets to the folder (or to
results/idea1/combined/ when given several folders, which git ignores; override with --out):
    summary.csv   the table below
    steps.csv     one row per scored TextGrad proposal: settings, θ, Enzyme's scores
                  (the LLM's texts stay in steps.jsonl)

Only finished runs count; rows from attempts that were interrupted and redone are dropped.

Per problem family x n x knobs (from the dataset's answer key) x method x model x feedback:
    runs          finished runs
    failed        fraction of proposals the LLM got wrong (unparseable answer)
    cos_mean      mean cosine between the step and −∇f (1 = steepest descent, 0 = random)
    cos_pos       fraction of steps pointing downhill (cosine > 0)
    descent       fraction of steps that lowered the loss
    ratio_med     median step_ratio (step length / best length along its direction; 1 = ideal)
    overshoot     fraction of steps longer than the best length (step_ratio > 1)
    final_loss    mean loss after the last TextGrad step
    gd_final      mean loss after the same number of Enzyme gradient-descent steps
"""

import argparse
import json
from pathlib import Path

import pandas as pd

KEYS = ["family", "n", "knobs", "method", "model", "feedback"]
TEXT_COLUMNS = ["evaluation", "gradient", "update"]


def read_jsonl(folders, name):
    rows = []
    for folder in folders:
        path = Path(folder) / name
        if path.exists():
            rows += [json.loads(line) for line in path.open()]
    return pd.DataFrame(rows)


def load(folders):
    """(runs, steps, gd) for the given folders: finished runs only; gd indexed by program."""
    runs = read_jsonl(folders, "runs.jsonl")
    steps = read_jsonl(folders, "steps.jsonl")
    steps = steps[steps["run_id"].isin(runs["run_id"])].copy()
    for df in (runs, steps):   # results from before --method existed are all TextGrad
        df["method"] = df["method"].fillna("textgrad") if "method" in df else "textgrad"
    gd = read_jsonl(folders, "gd.jsonl").drop_duplicates("program").set_index("program")
    gd["f_star"] = gd["meta"].map(lambda m: (m or {}).get("f_star"))
    runs["gd_final"] = runs["program"].map(gd["losses"].str[-1])
    return runs, steps, gd


def summary_table(runs, steps):
    ok = steps[steps["theta_new"].notna()]
    g = ok.groupby(KEYS)
    table = runs.groupby(KEYS).size().rename("runs").to_frame()
    table["failed"] = steps.groupby(KEYS)["theta_new"].apply(lambda s: s.isna().mean())
    table["cos_mean"] = g["cos_to_neg_grad"].mean()
    table["cos_pos"] = g["cos_to_neg_grad"].apply(lambda s: (s > 0).mean())
    table["descent"] = g["descent"].mean()
    table["ratio_med"] = g["step_ratio"].median()
    table["overshoot"] = g["step_ratio"].apply(lambda s: (s.isna() | (s > 1)).mean())
    table["final_loss"] = runs.groupby(KEYS)["final_loss"].mean()
    table["gd_final"] = runs.groupby(KEYS)["gd_final"].mean()
    return table


def default_out(folders):
    return Path(folders[0]) if len(folders) == 1 else Path(folders[0]).parent / "combined"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--out", help="where to write the CSVs")
    args = ap.parse_args()

    runs, steps, _ = load(args.folders)
    table = summary_table(runs, steps)
    out = Path(args.out) if args.out else default_out(args.folders)
    out.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out / "summary.csv")
    steps.drop(columns=TEXT_COLUMNS).to_csv(out / "steps.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(table.round(3).to_string())
    print(f"\n{len(runs)} finished runs, {len(steps)} scored proposals. "
          f"LLM calls: {runs['llm_calls'].sum()}, tokens in/out: "
          f"{runs['prompt_tokens'].sum():,}/{runs['completion_tokens'].sum():,}")
    print(f"Wrote {out / 'summary.csv'} and {out / 'steps.csv'}")


if __name__ == "__main__":
    main()
