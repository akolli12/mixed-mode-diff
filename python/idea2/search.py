"""Idea 2: an LLM writes solvers, Enzyme tunes their numbers, and (in "ours") the gradient
report goes back to the LLM.

Each iteration the LLM sees the best program so far with its validation results and writes
a new one. The new program is compiled, tuned (except in llm_only) and scored on the
validation instances; if it beats the best, it becomes the next parent. Modes differ only
in the feedback (see prompts.py), so at the same number of LLM calls they are comparable.

Run from the repo root, e.g.

    python/.venv/bin/python python/idea2/search.py --family channels --N 64 \\
        --frame precond --mode ours --iterations 20 --name channels64-precond-ours

    python/.venv/bin/python python/idea2/search.py --model mock --iterations 4 --name mock-test

"mock" is a fake model that perturbs the parent's THETA0 and sometimes returns broken code,
for testing the loop without calling any API.

Outputs go to results/idea2/<name>/:
    config.json        the settings and the git commit they ran on
    candidates.jsonl   one row per program: the LLM's reply, the code, its evaluation
    test.json          the best program and the starting program on the test instances
"""

import argparse
import json
import math
import queue
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "idea1"))
from prompts import MODES, system_prompt, user_prompt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS = {"precond": "precond_jacobi_poly.jl", "solver": "solver_jacobi_momentum.jl"}


class JuliaWorker:
    """The Julia bridge (julia/idea2/server.jl). A request that takes longer than its
    timeout (e.g. a candidate that loops forever) kills the process; a new one is started
    and set up again, and the request counts as a failure."""

    def __init__(self, setup: dict):
        self.setup = setup
        self._start()

    def _start(self):
        self._proc = subprocess.Popen(
            ["julia", "--project=julia", "julia/idea2/server.jl"],
            cwd=REPO_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self._lines = queue.Queue()
        threading.Thread(target=self._read, args=(self._proc, self._lines), daemon=True).start()
        reply = self._send(self.setup, timeout=1800)
        if not reply.get("ok"):
            raise RuntimeError(f"Julia setup failed: {reply}")
        self.info = reply

    @staticmethod
    def _read(proc, lines):
        for line in proc.stdout:
            if line.startswith("{"):          # anything else is stray printing
                lines.put(line)
        lines.put(None)

    def _send(self, request, timeout):
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            return None
        if line is None:
            raise RuntimeError("Julia bridge exited; see the Julia error above.")
        return json.loads(line)

    def call(self, request, timeout):
        try:
            reply = self._send(request, timeout)
        except (RuntimeError, BrokenPipeError) as err:
            reply = {"ok": False, "stage": "running", "error": f"Julia crashed: {err}"}
            self._restart()
            return reply
        if reply is None:
            self._restart()
            return {"ok": False, "stage": "running", "error": f"took longer than {timeout} s"}
        return reply

    def _restart(self):
        self._proc.kill()
        self._proc.wait()
        self._start()

    def close(self):
        self._proc.stdin.close()
        self._proc.wait()


class MockCoder:
    """Offline stand-in for the LLM: copies the parent program with THETA0 nudged, and
    every third call returns code that does not compile."""

    def __init__(self, spec="mock"):
        self.spec = spec
        self.calls = self.prompt_tokens = self.completion_tokens = 0

    def lite_llm_generate(self, content, system_prompt=None):
        self.calls += 1
        code = re.search(r"```julia\n(.*?)```", content, re.S).group(1)
        if self.calls % 3 == 0:
            return "Mock idea.\n```julia\n" + code.replace("return nothing", "return nothing)", 1) + "```"
        vec = lambda name: [float(x) for x in re.search(rf"const {name} = \[(.*?)\]", code).group(1).split(",")]
        lo, hi = vec("LOWER"), vec("UPPER")
        new = [min(max(x + random.gauss(0, 0.1) * (u - l), l), u) for x, l, u in zip(vec("THETA0"), lo, hi)]
        code = re.sub(r"const THETA0 = \[.*?\]", "const THETA0 = [" + ", ".join(f"{x:.4g}" for x in new) + "]", code)
        return f"Mock idea.\n```julia\n{code}```"


def get_coder(spec):
    if spec.startswith("mock"):
        return MockCoder(spec)
    from llm import get_engine
    return get_engine(spec)


def extract_code(reply):
    blocks = re.findall(r"```julia\s*\n(.*?)```", reply, re.S)
    return blocks[-1] if blocks else None


def score_of(ev):
    if not ev.get("ok") or ev["val"].get("score") is None:
        return math.inf
    return ev["val"]["score"]


def git_info():
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"),
            "uncommitted_changes": bool(git("status", "--porcelain", "--", "data", "julia", "python"))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="channels")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--frame", choices=SEEDS, default="precond")
    ap.add_argument("--mode", choices=MODES, default="ours")
    ap.add_argument("--iterations", type=int, default=20, help="LLM calls")
    ap.add_argument("--model", default="kimi-k2.6:think")
    ap.add_argument("--name", required=True)
    ap.add_argument("--tune-steps", type=int, default=25)
    ap.add_argument("--m", type=int, default=30, help="solver iterations in the training loss")
    ap.add_argument("--maxit", type=int, default=500, help="iteration cap when scoring")
    ap.add_argument("--train", type=int, nargs="+", default=[1, 2, 3, 4], help="training instances")
    ap.add_argument("--val", type=int, nargs="+", default=[5, 6, 7, 8], help="validation instances")
    ap.add_argument("--test", type=int, nargs="+", default=list(range(1, 17)), help="test instances")
    ap.add_argument("--levels", default="every_other", help='"every_other", "all", or comma-separated levels')
    ap.add_argument("--timeout", type=float, default=900, help="seconds allowed per evaluation")
    ap.add_argument("--seed-only", action="store_true", help="evaluate the starting program and stop")
    args = ap.parse_args()

    out = REPO_ROOT / "results" / "idea2" / args.name
    if (out / "candidates.jsonl").exists():
        sys.exit(f"{out} already has results; choose another --name.")
    out.mkdir(parents=True, exist_ok=True)
    levels = args.levels if args.levels in ("all", "every_other") else [float(x) for x in args.levels.split(",")]
    setup = {"cmd": "setup", "family": args.family, "N": args.N, "train": args.train, "val": args.val,
             "levels": levels, "m": args.m, "maxit": args.maxit}
    (out / "config.json").write_text(json.dumps({**vars(args), **git_info()}, indent=2))

    coder = get_coder(args.model)
    print(f"Starting Julia ({args.family}, N={args.N}) ...", flush=True)
    julia = JuliaWorker(setup)
    levels = julia.info["levels"]
    print(f"contrast levels {levels}; {julia.info['train_problems']} training and "
          f"{julia.info['val_problems']} validation problems", flush=True)
    tune = args.mode != "llm_only"

    def evaluate(code):
        return julia.call({"cmd": "evaluate", "code": code, "frame": args.frame, "tune": tune,
                           "tune_steps": args.tune_steps}, timeout=args.timeout)

    def log(row):
        with (out / "candidates.jsonl").open("a") as fh:
            fh.write(json.dumps(row) + "\n")

    seed_code = (REPO_ROOT / "julia" / "idea2" / "seeds" / SEEDS[args.frame]).read_text()
    ev = evaluate(seed_code)
    if not ev.get("ok"):
        sys.exit(f"The starting program failed: {ev}")
    seed = best = {"iteration": 0, "code": seed_code, "eval": ev}
    log({"iteration": 0, "parent": None, "reply": None, "code": seed_code, "eval": ev, "score": score_of(ev),
         "best_score": score_of(ev)})
    print(f"[0] starting program: score {score_of(ev):.3f} ({ev['seconds']:.0f} s)", flush=True)

    history, last_failure = [], None
    for it in range(1, 0 if args.seed_only else args.iterations + 1):
        t0 = time.time()
        sys_p = system_prompt(args.mode, args.frame, args.family, args.N, levels)
        user_p = user_prompt(args.mode, best, levels, last_failure, history)
        try:
            reply = coder.lite_llm_generate(user_p, system_prompt=sys_p)
        except Exception as err:          # API failure or empty answer: counts as a used call
            reply = None
            ev = {"ok": False, "stage": "asking the model", "error": f"{type(err).__name__}: {err}"}
        code = extract_code(reply) if reply else None
        if reply and code is None:
            ev = {"ok": False, "stage": "reading the reply", "error": "no ```julia code block in the reply"}
        if code is not None:
            ev = evaluate(code)
        s = score_of(ev)
        if ev.get("ok"):
            history.append(f"{s:.3f}" if math.isfinite(s) else "no finite score")
            last_failure = None
        else:
            history.append(f"failed while {ev['stage']}")
            last_failure = ev
        improved = s < score_of(best["eval"])
        log({"iteration": it, "parent": best["iteration"], "reply": reply, "code": code, "eval": ev, "score": s,
             "improved": improved, "seconds": time.time() - t0,
             "calls": coder.calls, "prompt_tokens": coder.prompt_tokens, "completion_tokens": coder.completion_tokens})
        if improved:
            best = {"iteration": it, "code": code, "eval": ev}
        status = f"score {s:.3f}" if ev.get("ok") else f"failed while {ev['stage']}"
        print(f"[{it}] {status}{'  NEW BEST' if improved else ''}   best {score_of(best['eval']):.3f} "
              f"(iteration {best['iteration']}); {time.time() - t0:.0f} s", flush=True)

    print("Scoring the best and the starting program on the test instances ...", flush=True)
    test = {}
    for label, prog in (("best", best), ("start", seed)):
        r = julia.call({"cmd": "test", "code": prog["code"], "frame": args.frame, "theta": prog["eval"]["theta"],
                        "instances": args.test, "maxit": args.maxit}, timeout=max(args.timeout, 3600))
        test[label] = {"iteration": prog["iteration"], "theta": prog["eval"]["theta"], **r}
        if r.get("ok"):
            print(f"  {label}: test score {r['score']}; median iterations by contrast "
                  f"{ {k: round(v) for k, v in r['median_iterations_by_contrast'].items()} }", flush=True)
        else:
            print(f"  {label}: test failed: {r}", flush=True)
    (out / "test.json").write_text(json.dumps(test, indent=2))
    julia.close()
    print(f"LLM calls {coder.calls}, tokens {coder.prompt_tokens} in / {coder.completion_tokens} out. "
          f"Results in {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
