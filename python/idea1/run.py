"""Idea 1: run TextGrad on Julia programs and score every update against Enzyme.

For each combination of program x model x feedback x repeat, TextGrad takes --steps
updates from the program's P0. Every update is scored by Enzyme at the point it was
made. Enzyme's own gradient descent runs from the same P0 for the same number of steps.

Problems are files in the dataset format described in julia/idea1/load_program.jl, e.g.
data/idea1_testfns/problems/P<id>.jl. Run from the repo root, e.g.

    python/.venv/bin/python python/idea1/run.py \\
        --programs data/idea1_testfns/problems/P7f72b5e797.jl \\
        --models kimi-k3 kimi-k2.6 kimi-k2.6:nothink \\
        --feedback loss history --repeats 3 --steps 10 --name pilot

Outputs go to results/idea1/<name>/:
    steps.jsonl   one row per proposed update: settings, p before/after, Enzyme's scores,
                  and the LLM's critique, feedback and raw update text
    gd.jsonl      Enzyme gradient-descent losses for each program
    runs.jsonl    finished runs; rerunning with the same --name skips them
"""

import argparse
import itertools
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import get_engine  # noqa: E402
from oracle import REPO_ROOT, JuliaOracle  # noqa: E402
from textgrad_step import FEEDBACK_MODES, textgrad_step  # noqa: E402

write_lock = threading.Lock()


def append(path: Path, row: dict):
    with write_lock, path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def run_key(program, model, feedback, repeat):
    return f"{program}|{model}|{feedback}|{repeat}"


def run_one(oracle, program, model, feedback, repeat, args, out_dir):
    engine = get_engine(model)
    theta, loss = list(oracle.theta0), oracle.loss0
    history = [(theta, loss)]
    settings = dict(program=program, model=model, feedback=feedback, repeat=repeat,
                    family=oracle.meta.get("family"), n=len(oracle.theta0),
                    knobs=json.dumps(oracle.meta.get("knobs"), sort_keys=True))
    for step in range(args.steps):
        accepted = None
        for sample in range(args.samples):
            t0 = time.time()
            proposal = textgrad_step(engine, oracle.source, theta, loss, history, feedback)
            row = dict(settings, step=step, sample=sample, theta=theta,
                       seconds=round(time.time() - t0, 1), **proposal)
            if proposal["theta_new"] is not None:
                row.update(oracle.score(theta, proposal["theta_new"]))
            append(out_dir / "steps.jsonl", row)
            if sample == 0:
                accepted = row
        # Like TextGrad, keep the first proposal even if the loss went up.
        # An unparseable proposal leaves p unchanged and counts as a failed step.
        if accepted["theta_new"] is not None:
            theta, loss = accepted["theta_new"], accepted["loss_after"]
            history.append((theta, loss))
        print(f"[{model} | {feedback} | rep {repeat}] step {step + 1}/{args.steps}: "
              f"loss = {loss:.6g}" + (f"  ({accepted['error']})" if accepted["error"] else ""),
              flush=True)
    append(out_dir / "runs.jsonl", dict(settings, key=run_key(program, model, feedback, repeat),
                                         final_loss=loss, llm_calls=engine.calls,
                                         prompt_tokens=engine.prompt_tokens,
                                         completion_tokens=engine.completion_tokens))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--programs", nargs="+", required=True, help="Julia program files")
    ap.add_argument("--models", nargs="+", default=["kimi-k3"], help='e.g. kimi-k3 kimi-k2.6:nothink, or "mock"')
    ap.add_argument("--feedback", nargs="+", default=["loss"], choices=FEEDBACK_MODES)
    ap.add_argument("--repeats", type=int, default=1, help="independent runs per setting")
    ap.add_argument("--steps", type=int, default=10, help="TextGrad updates per run")
    ap.add_argument("--samples", type=int, default=1,
                    help="proposals drawn at each state; all are scored, the first is kept")
    ap.add_argument("--workers", type=int, default=4, help="runs in parallel")
    ap.add_argument("--name", default=time.strftime("%Y-%m-%d_%H-%M-%S"), help="results folder name")
    args = ap.parse_args()

    out_dir = REPO_ROOT / "results" / "idea1" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    done = set()
    if (out_dir / "runs.jsonl").exists():
        done = {json.loads(line)["key"] for line in (out_dir / "runs.jsonl").open()}

    for program in args.programs:
        program_rel = str(Path(program).resolve().relative_to(REPO_ROOT))
        with JuliaOracle(program) as oracle:
            if not (out_dir / "gd.jsonl").exists() or program_rel not in (out_dir / "gd.jsonl").read_text():
                append(out_dir / "gd.jsonl", dict(program=program_rel, meta=oracle.meta,
                                                  losses=oracle.gradient_descent(args.steps)))
            todo = [c for c in itertools.product(args.models, args.feedback, range(args.repeats))
                    if run_key(program_rel, *c) not in done]
            print(f"{program_rel}: {len(todo)} runs to do, start loss {oracle.loss0:.6g}", flush=True)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(run_one, oracle, program_rel, m, fb, r, args, out_dir)
                           for m, fb, r in todo]
                for f in futures:
                    f.result()
    print(f"Done. Results in {out_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
