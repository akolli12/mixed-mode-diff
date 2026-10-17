"""Summarizes a results folder written by run.py.

    python/.venv/bin/python python/idea1/summarize.py results/idea1/<name>

Per problem family x n x knobs (from the dataset's answer key) x model x feedback:
    steps         proposals scored
    failed        fraction of proposals that could not be parsed or errored
    cos_mean      mean cosine between the step and −∇f (1 = steepest descent, 0 = random)
    cos_pos       fraction of steps pointing downhill (cosine > 0)
    descent       fraction of steps that lowered the loss
    ratio_med     median step_ratio (step length / best length along its direction; 1 = ideal)
    overshoot     fraction of steps longer than the best length (step_ratio > 1)
    final_loss    mean loss at the end of each run
and, per program, Enzyme gradient descent's final loss after the same number of steps.
"""

import json
import sys
from pathlib import Path

import pandas as pd


def main(folder: str):
    folder = Path(folder)
    steps = pd.read_json(folder / "steps.jsonl", lines=True)
    runs = pd.read_json(folder / "runs.jsonl", lines=True)
    keys = ["family", "n", "knobs", "model", "feedback"]
    for df in (steps, runs):
        df["family"] = df["family"].fillna(df["program"])

    ok = steps[steps["theta_new"].notna()]
    table = steps.groupby(keys).size().rename("steps").to_frame()
    table["failed"] = steps.groupby(keys)["theta_new"].apply(lambda s: s.isna().mean())
    g = ok.groupby(keys)
    table["cos_mean"] = g["cos_to_neg_grad"].mean()
    table["cos_pos"] = g["cos_to_neg_grad"].apply(lambda s: (s > 0).mean())
    table["descent"] = g["descent"].mean()
    table["ratio_med"] = g["step_ratio"].median()
    table["overshoot"] = g["step_ratio"].apply(lambda s: (s.isna() | (s > 1)).mean())
    table["final_loss"] = runs.groupby(keys)["final_loss"].mean()

    pd.set_option("display.width", 200)
    print(table.round(3).to_string())
    print("\nEnzyme gradient descent, same number of steps:")
    for line in (folder / "gd.jsonl").open():
        row = json.loads(line)
        print(f"  {row['program']}: {row['losses'][0]:.6g} -> {row['losses'][-1]:.6g}")
    print(f"\nLLM calls: {runs['llm_calls'].sum()}, tokens in/out: "
          f"{runs['prompt_tokens'].sum()}/{runs['completion_tokens'].sum()}")


if __name__ == "__main__":
    main(sys.argv[1])
