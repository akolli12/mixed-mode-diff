"""One TextGrad update of a program's parameter vector θ, as in the hw1 notebook:

    evaluation = loss_fn(θ)      # E: the LLM critiques θ, given the code and loss(θ)
    evaluation.backward()        # the LLM turns that critique into feedback on θ
    optimizer.step()             # U: the LLM rewrites θ using the feedback

The notebook optimized a prompt; here the variable is the parameter vector θ, written as
a JSON list. The LLM sees the problem file's source, θ and loss(θ), never Enzyme or the
hidden answer key.

These prompt templates are fixed for every run, so runs differ only in the settings swept.
"""

import json
import math
import re

import textgrad as tg

from llm import NoAnswerError

# What the LLM is told between updates (the "feedback" sweep setting):
#   "loss":    the current output only
#   "history": also every earlier θ evaluated in this run and its loss (last HISTORY_LEN of
#              them), including proposals where the program diverged
FEEDBACK_MODES = ("loss", "history")
HISTORY_LEN = 10

EVAL_INSTRUCTION = """You are evaluating the parameters of a fixed numerical program written in Julia. \
The code cannot be changed; only the parameter vector θ passed to `loss` can. The goal is to make \
loss(θ) as small as possible.

Program source:
```julia
{source}
```

At the current θ, loss(θ) = {loss}.
{history}
Evaluate the current θ, which you will receive next: is it good, and how should each parameter change \
to make loss(θ) smaller?"""

HISTORY_BLOCK = "\nEarlier values of θ in this run and loss(θ) for each, oldest first:\n{rows}\n"

ROLE = "parameter vector θ passed to loss: a JSON list of {n} numbers"

CONSTRAINT = ("The improved variable must be a JSON list of exactly {n} numbers, "
              "such as [0.5,-1.2,3], with no spaces and no other text.")


def format_theta(theta) -> str:
    # No spaces: TextGrad shortens any variable longer than 20 space-separated words,
    # which would hide the middle of a long parameter list from the optimizer.
    return "[" + ",".join(f"{x:.6g}" for x in theta) + "]"


def format_loss(x: float) -> str:
    return f"{x:.6g}" if math.isfinite(x) else "not finite (the program diverged)"


def parse_theta(text: str, n: int):
    """The last JSON list of n finite numbers in text, or None."""
    for match in reversed(re.findall(r"\[[^\[\]]*\]", text or "")):
        try:
            values = [float(v) for v in json.loads(match)]
        except (ValueError, TypeError):
            continue
        if len(values) == n and all(math.isfinite(v) for v in values):
            return values
    return None


def eval_instruction(source: str, loss: float, past: list, feedback: str) -> str:
    """past: earlier (θ, loss) evaluations in this run, oldest first, not including the current θ."""
    history_text = ""
    if feedback == "history" and past:
        rows = [f"  {t}: θ = {format_theta(th)}, loss = {format_loss(f)}"
                for t, (th, f) in list(enumerate(past, 1))[-HISTORY_LEN:]]
        history_text = HISTORY_BLOCK.format(rows="\n".join(rows))
    return EVAL_INSTRUCTION.format(source=source.strip(), loss=format_loss(loss), history=history_text)


def textgrad_step(engine, source: str, theta: list, loss: float, past: list, feedback: str) -> dict:
    """Runs one TextGrad update. Returns the proposed parameters (None if unparseable)
    and every text the LLM produced along the way.

    error_kind is "format" when the LLM's answer could not be used (the LLM's failure,
    part of the results) and "api" when the API call itself failed (not the LLM's
    failure; the runner retries these)."""
    n = len(theta)
    p = tg.Variable(format_theta(theta), requires_grad=True, role_description=ROLE.format(n=n))
    loss_fn = tg.TextLoss(eval_instruction(source, loss, past, feedback), engine=engine)
    optimizer = tg.TextualGradientDescent(parameters=[p], engine=engine,
                                          constraints=[CONSTRAINT.format(n=n)])
    out = {"theta_new": None, "evaluation": None, "gradient": None, "update": None,
           "error": None, "error_kind": None}
    try:
        evaluation = loss_fn(p)
        out["evaluation"] = evaluation.value
        evaluation.backward(engine)
        out["gradient"] = p.get_gradient_text()
        optimizer.step()
        out["update"] = p.value
        out["theta_new"] = parse_theta(p.value, n)
        if out["theta_new"] is None:
            out["error"], out["error_kind"] = "could not parse a list of the right length", "format"
    except IndexError as e:  # TextGrad raises this when the answer lacks the <IMPROVED_VARIABLE> tags
        out["error"], out["error_kind"] = f"IndexError: {e}", "format"
    except NoAnswerError as e:  # the model spent its whole output budget thinking
        out["error"], out["error_kind"] = f"NoAnswerError: {e}", "format"
    except Exception as e:   # rate limits, timeouts, connection errors
        out["error"], out["error_kind"] = f"{type(e).__name__}: {e}", "api"
    return out
