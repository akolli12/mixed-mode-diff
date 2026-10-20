"""LLM optimizers other than TextGrad, for comparison on the same problems and scoring.

    direct  one LLM call per step: given the same information as TextGrad's critique call
            (code, current θ, loss, and with history feedback the earlier evaluations), the
            LLM replies with the next θ. No critique, no feedback call. Tests whether
            TextGrad's "textual gradient" machinery adds anything.
    opro    OPRO (Yang et al., 2024): the LLM sees the best evaluated θ so far, sorted by
            loss, and proposes a new θ that should beat them all. The runner draws several
            proposals per step, evaluates them all and keeps the best so far.

Like TextGrad, neither ever sees a gradient or the hidden answer key. Each function returns
the same dict as textgrad_step, so the runner scores every method identically.
"""

import math

from llm import NoAnswerError
from textgrad_step import HISTORY_BLOCK, HISTORY_LEN, format_loss, format_theta, parse_theta

PROGRAM_BLOCK = """You are optimizing the parameters of a fixed numerical program written in Julia. \
The code cannot be changed; only the parameter vector θ passed to `loss` can. The goal is to make \
loss(θ) as small as possible.

Program source:
```julia
{source}
```
"""

DIRECT_INSTRUCTION = PROGRAM_BLOCK + """
At the current θ, loss(θ) = {loss}.
{history}
You will receive the current θ next. Propose the next value of θ, which should make loss(θ) \
smaller. You may reason first; end your answer with the new θ as a JSON list of exactly {n} \
numbers, such as [0.5,-1.2,3], on its own line."""

OPRO_INSTRUCTION = PROGRAM_BLOCK + """
Below are earlier values of θ and loss(θ) for each, arranged in descending order of loss \
(lower is better):
{rows}

Give a new θ that is different from all θ above and has a lower loss than all of them. You may \
reason first; end your answer with the new θ as a JSON list of exactly {n} numbers, such as \
[0.5,-1.2,3], on its own line."""

OPRO_TOP = 20   # evaluated θ shown to the LLM, as in the OPRO paper


def _call(engine, system_prompt: str, content: str, n: int) -> dict:
    out = {"theta_new": None, "evaluation": None, "gradient": None, "update": None,
           "error": None, "error_kind": None}
    try:
        reply = engine.generate(content, system_prompt=system_prompt)
        out["evaluation"] = out["update"] = reply
        out["theta_new"] = parse_theta(reply, n)
        if out["theta_new"] is None:
            out["error"], out["error_kind"] = "could not parse a list of the right length", "format"
    except NoAnswerError as e:   # the model spent its whole output budget thinking
        out["error"], out["error_kind"] = f"NoAnswerError: {e}", "format"
    except Exception as e:       # rate limits, timeouts, connection errors
        out["error"], out["error_kind"] = f"{type(e).__name__}: {e}", "api"
    return out


def direct_step(engine, source: str, theta: list, loss: float, past: list, feedback: str) -> dict:
    """past: earlier (θ, loss) evaluations, oldest first, not including the current θ."""
    history_text = ""
    if feedback == "history" and past:
        rows = [f"  {t}: θ = {format_theta(th)}, loss = {format_loss(f)}"
                for t, (th, f) in list(enumerate(past, 1))[-HISTORY_LEN:]]
        history_text = HISTORY_BLOCK.format(rows="\n".join(rows))
    system = DIRECT_INSTRUCTION.format(source=source.strip(), loss=format_loss(loss),
                                       history=history_text, n=len(theta))
    return _call(engine, system, format_theta(theta), len(theta))


def opro_step(engine, source: str, theta: list, loss: float, past: list, feedback: str) -> dict:
    """past: every (θ, loss) evaluated so far in this run, including the current best θ."""
    evaluated = sorted(past, key=lambda p: p[1] if math.isfinite(p[1]) else math.inf)[:OPRO_TOP]
    rows = "\n".join(f"  θ = {format_theta(th)}, loss = {format_loss(f)}" for th, f in reversed(evaluated))
    system = OPRO_INSTRUCTION.format(source=source.strip(), rows=rows, n=len(theta))
    content = f"The best θ so far is {format_theta(theta)} with loss {format_loss(loss)}. Give a new θ."
    return _call(engine, system, content, len(theta))
