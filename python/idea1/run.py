"""Idea 1: run TextGrad on Julia problems and score every update against Enzyme.

For each combination of problem x model x feedback x repeat, TextGrad takes --steps
updates from the problem's THETA0. Every update is scored by Enzyme at the point it was
made. Enzyme's own gradient descent runs from the same THETA0 for the same number of steps.

Run a stage of the plan (see plan.py) from the repo root:

    python/.venv/bin/python python/idea1/run.py --stage pilot

or choose everything by hand (problem files are described in julia/idea1/load_program.jl):

    python/.venv/bin/python python/idea1/run.py \\
        --programs data/idea1_testfns/problems/P7f72b5e797.jl \\
        --models kimi-k2.6:nothink --feedback loss history --steps 10 --name my-test

Outputs go to results/idea1/<name>/ (the stage name, for --stage):
    steps.jsonl   one row per proposed update: settings, θ before/after, Enzyme's scores,
                  the LLM's critique, feedback and update text, tokens used
    gd.jsonl      Enzyme gradient-descent losses for each problem
    runs.jsonl    finished runs; rerunning the same command skips them
    config.jsonl  one line per launch: the settings used and the git commit they ran on

A run that cannot finish (e.g. the API keeps failing) is left out of runs.jsonl and redone
from the start on the next launch; summarize.py ignores rows from unfinished attempts.
"""

import argparse
import itertools
import json
import math
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import get_engine  # noqa: E402
from oracle import REPO_ROOT, JuliaOracle  # noqa: E402
from plan import STAGES  # noqa: E402
from textgrad_step import FEEDBACK_MODES, textgrad_step  # noqa: E402

API_RETRIES = 5     # retries of one step after an API error before giving up on the run
API_WAIT = 60       # seconds between those retries

write_lock = threading.Lock()


def append(path: Path, row: dict):
    with write_lock, path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def run_key(program, model, feedback, repeat):
    return f"{program}|{model}|{feedback}|{repeat}"


def git_info():
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"),
            "uncommitted_changes": bool(git("status", "--porcelain", "--", "data", "julia", "python"))}


def propose(engine, oracle, theta, loss, past, feedback, label):
    """One TextGrad proposal, retrying API errors. Raises if the API keeps failing."""
    for attempt in range(API_RETRIES + 1):
        proposal = textgrad_step(engine, oracle.source, theta, loss, past, feedback)
        if proposal["error_kind"] != "api":
            return proposal
        if attempt < API_RETRIES:
            print(f"{label}: API error, retrying in {API_WAIT}s: {proposal['error'][:200]}", flush=True)
            time.sleep(API_WAIT)
    raise RuntimeError(f"API failed {API_RETRIES + 1} times: {proposal['error'][:300]}")


def run_one(oracle, program, model, feedback, repeat, args, out_dir):
    engine = get_engine(model)
    theta, loss = list(oracle.theta0), oracle.loss0
    past = []   # earlier (θ, loss) evaluations, shown to the LLM with history feedback
    settings = dict(program=program, model=model, feedback=feedback, repeat=repeat,
                    family=oracle.meta.get("family"), n=len(oracle.theta0),
                    knobs=json.dumps(oracle.meta.get("knobs"), sort_keys=True),
                    run_id=uuid.uuid4().hex)
    label = f"[{Path(program).stem} | {model} | {feedback} | rep {repeat}]"
    for step in range(args.steps):
        accepted = None
        for sample in range(args.samples):
            t0, tokens0 = time.time(), (engine.prompt_tokens, engine.completion_tokens)
            proposal = propose(engine, oracle, theta, loss, past, feedback, label)
            row = dict(settings, step=step, sample=sample, theta=theta,
                       seconds=round(time.time() - t0, 1),
                       prompt_tokens=engine.prompt_tokens - tokens0[0],
                       completion_tokens=engine.completion_tokens - tokens0[1], **proposal)
            if proposal["theta_new"] is not None:
                row.update(oracle.score(theta, proposal["theta_new"]))
            append(out_dir / "steps.jsonl", row)
            if sample == 0:
                accepted = row
        # Like TextGrad, keep the first proposal even if the loss went up. Exceptions:
        # an unusable answer leaves θ unchanged (a failed step), and so does a θ where the
        # program diverged, which the LLM then sees in its history as "not finite".
        note = accepted["error"] or ""
        if accepted.get("diverged"):
            past.append((accepted["theta_new"], math.inf))
            note = "proposal diverged; θ kept"
        elif accepted["theta_new"] is not None:
            past.append((theta, loss))
            theta, loss = accepted["theta_new"], accepted["loss_after"]
        print(f"{label} step {step + 1}/{args.steps}: loss = {loss:.6g}"
              + (f"  ({note})" if note else ""), flush=True)
    append(out_dir / "runs.jsonl", dict(settings, key=run_key(program, model, feedback, repeat),
                                         final_loss=loss, llm_calls=engine.calls,
                                         prompt_tokens=engine.prompt_tokens,
                                         completion_tokens=engine.completion_tokens))


def safe_run_one(*args):
    try:
        run_one(*args)
        return True
    except Exception as e:
        print(f"RUN FAILED, will be redone on the next launch: {args[1]} {args[2]} {args[3]}: {e}",
              flush=True)
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=list(STAGES), help="run a stage of the plan in plan.py")
    ap.add_argument("--programs", nargs="+", help="problem files (instead of --stage)")
    ap.add_argument("--models", nargs="+", default=["kimi-k2.6:nothink"],
                    help='e.g. kimi-k3 kimi-k2.6 kimi-k2.6:nothink, or "mock" for a free test')
    ap.add_argument("--feedback", nargs="+", default=["loss"], choices=FEEDBACK_MODES)
    ap.add_argument("--repeats", type=int, default=1, help="independent runs per setting")
    ap.add_argument("--steps", type=int, default=10, help="TextGrad updates per run")
    ap.add_argument("--samples", type=int, default=1,
                    help="proposals drawn at each state; all are scored, the first is kept")
    ap.add_argument("--limit", type=int, help="only the first N problems (e.g. to time a slow model)")
    ap.add_argument("--mock", action="store_true",
                    help="use the free mock model instead (results go to <name>-mock); tests the pipeline")
    ap.add_argument("--workers", type=int, default=4, help="problems run in parallel")
    ap.add_argument("--name", help="results folder name (default: the stage name)")
    args = ap.parse_args()

    if args.stage:
        s = STAGES[args.stage]
        args.programs, args.models, args.feedback = s["programs"], s["models"], s["feedback"]
        args.steps, args.samples = s["steps"], s["samples"]
        args.name = args.name or args.stage
    if not args.programs:
        ap.error("give --stage or --programs")
    args.name = args.name or time.strftime("%Y-%m-%d_%H-%M-%S")
    if args.mock:
        args.models, args.name = ["mock"], args.name + "-mock"
    programs = args.programs[:args.limit] if args.limit else args.programs

    out_dir = REPO_ROOT / "results" / "idea1" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    git = git_info()
    if git["uncommitted_changes"]:
        print("WARNING: data/, julia/ or python/ has uncommitted changes; results may not be reproducible.")
    config = dict(vars(args), programs=programs, git=git, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    append(out_dir / "config.jsonl", config)

    done = set()
    if (out_dir / "runs.jsonl").exists():
        done = {json.loads(line)["key"] for line in (out_dir / "runs.jsonl").open()}
    gd_done = set()
    if (out_dir / "gd.jsonl").exists():
        gd_done = {json.loads(line)["program"] for line in (out_dir / "gd.jsonl").open()}

    def do_problem(program):
        """All remaining runs on one problem, in its own Julia process. Returns failures."""
        program_rel = str(Path(program).resolve().relative_to(REPO_ROOT))
        todo = [c for c in itertools.product(args.models, args.feedback, range(args.repeats))
                if run_key(program_rel, *c) not in done]
        if not todo and program_rel in gd_done:
            return 0
        with JuliaOracle(REPO_ROOT / program_rel) as oracle:
            if program_rel not in gd_done:
                append(out_dir / "gd.jsonl", dict(program=program_rel, meta=oracle.meta,
                                                  losses=oracle.gradient_descent(args.steps)))
            failures = sum(not safe_run_one(oracle, program_rel, *c, args, out_dir) for c in todo)
        with write_lock:
            progress["done"] += 1
            print(f"=== problems finished: {progress['done']}/{len(programs)}", flush=True)
        return failures

    # Problems run in parallel, one worker (and one Julia process) per problem.
    progress = {"done": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        failed = sum(pool.map(do_problem, programs))
    print(f"Done. {failed} runs failed and will be redone if you rerun the same command. "
          f"Results in {out_dir.relative_to(REPO_ROOT)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
