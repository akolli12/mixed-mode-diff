"""Summarizes one or more results folders written by run.py, treated as one dataset.

    python/.venv/bin/python python/idea1/summarize.py results/idea1/mainA-testfns results/idea1/mainA-solvers
    python/.venv/bin/python python/idea1/summarize.py results/idea1/*        # everything

Only finished runs count; rows from attempts that were interrupted and redone are dropped.

Per problem family x n x knobs (from the dataset's answer key) x model x feedback:
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

import json
import sys
from pathlib import Path

import pandas as pd


def read_jsonl(folders, name):
    rows = []
    for folder in folders:
        path = Path(folder) / name
        if path.exists():
            rows += [json.loads(line) for line in path.open()]
    return pd.DataFrame(rows)


def main(folders):
    runs = read_jsonl(folders, "runs.jsonl")
    steps = read_jsonl(folders, "steps.jsonl")
    steps = steps[steps["run_id"].isin(runs["run_id"])]        # finished runs only
    gd = read_jsonl(folders, "gd.jsonl").drop_duplicates("program").set_index("program")
    runs["gd_final"] = runs["program"].map(gd["losses"].str[-1])

    keys = ["family", "n", "knobs", "model", "feedback"]
    ok = steps[steps["theta_new"].notna()]
    g = ok.groupby(keys)
    table = runs.groupby(keys).size().rename("runs").to_frame()
    table["failed"] = steps.groupby(keys)["theta_new"].apply(lambda s: s.isna().mean())
    table["cos_mean"] = g["cos_to_neg_grad"].mean()
    table["cos_pos"] = g["cos_to_neg_grad"].apply(lambda s: (s > 0).mean())
    table["descent"] = g["descent"].mean()
    table["ratio_med"] = g["step_ratio"].median()
    table["overshoot"] = g["step_ratio"].apply(lambda s: (s.isna() | (s > 1)).mean())
    table["final_loss"] = runs.groupby(keys)["final_loss"].mean()
    table["gd_final"] = runs.groupby(keys)["gd_final"].mean()

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(table.round(3).to_string())
    print(f"\n{len(runs)} finished runs, {len(steps)} scored proposals. "
          f"LLM calls: {runs['llm_calls'].sum()}, tokens in/out: "
          f"{runs['prompt_tokens'].sum():,}/{runs['completion_tokens'].sum():,}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
