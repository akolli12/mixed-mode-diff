"""Talks to the Julia bridge (julia/idea1/server.jl) for one problem file.

The problem's loss is evaluated, and differentiated with Enzyme, in Julia. Python only
sends parameter vectors and gets numbers back.
"""

import json
import subprocess
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class JuliaOracle:
    def __init__(self, program_path: str):
        self.program_path = str(Path(program_path).resolve())
        self._lock = threading.Lock()
        self._proc = subprocess.Popen(
            ["julia", "--project=julia", "julia/idea1/server.jl", self.program_path],
            cwd=REPO_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        info = self._call({"cmd": "info"})
        self.source = info["source"]       # exactly what the LLM is shown
        self.theta0 = info["theta0"]
        self.loss0 = info["loss0"]
        self.meta = info["meta"] or {}    # answer-key fields for grouping; never shown to the LLM

    def _call(self, request: dict) -> dict:
        with self._lock:
            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("Julia bridge exited; see the Julia error above.")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(f"Julia bridge: {reply['error']}")
        return reply

    def loss(self, theta) -> float:
        return self._call({"cmd": "loss", "theta": theta})["loss"]

    def score(self, theta, theta_new) -> dict:
        """Enzyme's scores for the update theta -> theta_new (never shown to the LLM)."""
        return self._call({"cmd": "score", "theta": theta, "theta_new": theta_new})

    def gradient_descent(self, steps: int) -> list:
        """Losses of Enzyme-driven gradient descent from the same start, one per step."""
        return self._call({"cmd": "gd", "steps": steps})["losses"]

    def close(self):
        self._proc.stdin.close()
        self._proc.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
