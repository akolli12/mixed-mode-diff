"""Kimi models as TextGrad engines.

# ============================ KIMI API KEY ============================
# Put your key in the file `.env` at the repo root, as a single line:
#
#     KIMI_API_KEY=sk-...
#
# `.env` is in .gitignore, so the key is never committed or pushed.
# Setting KIMI_API_KEY in your shell (export KIMI_API_KEY=sk-...) also works.
# ======================================================================

A model is named by its Kimi model id, optionally with ":nothink" to turn thinking off
(e.g. "kimi-k3", "kimi-k2.6", "kimi-k2.6:nothink"). "mock" is a fake model that makes
small random updates without calling any API, for testing the pipeline.
"""

import json
import os
import random
import re
import threading
from pathlib import Path

from litellm import completion
from textgrad.engine_experimental.litellm import LiteLLMEngine

KIMI_BASE_URL = "https://api.moonshot.ai/v1"
REQUEST_TIMEOUT = 300  # seconds per API call; a stuck call fails the step instead of hanging
REPO_ROOT = Path(__file__).resolve().parents[2]


def kimi_api_key() -> str:
    key = os.environ.get("KIMI_API_KEY", "").strip()
    env_file = REPO_ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "KIMI_API_KEY":
                key = value.strip().strip("'\"")
    if not key:
        raise RuntimeError(f"No Kimi API key. Put KIMI_API_KEY=sk-... in {env_file}")
    return key


class KimiEngine(LiteLLMEngine):
    """TextGrad's LiteLLM engine pointed at Kimi's OpenAI-compatible API.

    TextGrad's own engine ignores temperature and similar settings, so this subclass
    passes them to the API itself. Responses are never cached: repeated calls on the
    same state must be independent samples.
    """

    def __init__(self, spec: str):
        name, _, option = spec.partition(":")
        super().__init__(model_string="openai/" + name, cache=False)
        self.spec = spec
        self.thinking_off = option == "nothink"
        self.api_key = kimi_api_key()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._lock = threading.Lock()

    def lite_llm_generate(self, content, system_prompt=None, **kwargs) -> str:
        extra = {}
        if self.thinking_off:
            # Same settings as the hw1 notebook's weak task model.
            extra = dict(temperature=0.6, extra_body={"thinking": {"type": "disabled"}})
        response = completion(
            model=self.model_string,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": content}],
            api_base=KIMI_BASE_URL,
            api_key=self.api_key,
            timeout=REQUEST_TIMEOUT,
            **extra,
        )
        with self._lock:
            self.calls += 1
            if response.usage:
                self.prompt_tokens += response.usage.prompt_tokens or 0
                self.completion_tokens += response.usage.completion_tokens or 0
        return response.choices[0].message.content


class MockEngine(LiteLLMEngine):
    """Offline stand-in: returns canned feedback and nudges the parameters at random."""

    def __init__(self, spec: str = "mock"):
        super().__init__(model_string="mock", cache=False)
        self.spec = spec
        self.calls = self.prompt_tokens = self.completion_tokens = 0

    def lite_llm_generate(self, content, system_prompt=None, **kwargs) -> str:
        self.calls += 1
        if "IMPROVED_VARIABLE" not in (system_prompt or ""):
            return "Mock feedback: some parameters look too large; try changing them slightly."
        current = re.search(r"<VARIABLE>\s*(\[.*?\])\s*</VARIABLE>", content, re.S)
        theta = json.loads(current.group(1)) if current else []
        new = [x + random.gauss(0.0, 0.05) for x in theta]
        return "<IMPROVED_VARIABLE>[" + ",".join(f"{x:.6g}" for x in new) + "]</IMPROVED_VARIABLE>"


def get_engine(spec: str):
    return MockEngine(spec) if spec == "mock" else KimiEngine(spec)
