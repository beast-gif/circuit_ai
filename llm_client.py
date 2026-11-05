"""
Thin wrapper around Velona's OpenAI-compatible API.

Every module asks for a *role* ("text_parser", "explainer", ...) instead of a model id,
so swapping models is a one-line change in config.yaml.

    client = VelonaClient()
    data   = client.chat_json("text_parser", system_prompt, question)
    data   = client.vision_json("image_parser", system_prompt, "Extract the netlist", "circuit.png")
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, BadRequestError, OpenAI, RateLimitError

ROOT = Path(__file__).resolve().parent


class LLMError(RuntimeError):
    pass


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_conventions(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    with open(ROOT / cfg.get("conventions_file", "conventions.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_json(text: str) -> Dict[str, Any]:
    """Pull a JSON object out of a model reply (handles ```json fences and extra prose)."""
    if not text:
        raise LLMError("Empty reply from model")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise LLMError(f"No JSON object in reply:\n{text[:500]}")
    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as e:
        raise LLMError(f"Invalid JSON in reply ({e}):\n{text[:500]}") from e


class VelonaClient:
    def __init__(self, config_path: Optional[str] = None):
        load_dotenv(ROOT / ".env")
        self.cfg = load_config(config_path)
        llm = self.cfg["llm"]

        api_key = os.getenv(llm["api_key_env"])
        if not api_key:
            raise LLMError(f"Set {llm['api_key_env']} in circuit_ai/.env (see .env.example)")

        self.client = OpenAI(base_url=llm["base_url"], api_key=api_key,
                             timeout=llm.get("timeout_s", 60))
        self.models: Dict[str, str] = llm["models"]
        self.fallback_parser: str = llm.get("fallback_parser", self.models["text_parser"])
        self.temperature: float = llm.get("temperature", 0.0)
        self.max_retries: int = llm.get("max_retries", 2)

        # token usage per role, so you can see where the rupees go
        self.usage: Dict[str, Dict[str, int]] = defaultdict(lambda: {"prompt": 0, "completion": 0, "calls": 0})

    # ------------------------------------------------------------------
    def model_for(self, role: str) -> str:
        if role not in self.models:
            raise LLMError(f"Unknown role '{role}'. Known: {list(self.models)}")
        return self.models[role]

    def chat(self, role: str, messages: List[Dict[str, Any]], json_mode: bool = False,
             max_tokens: Optional[int] = None, model: Optional[str] = None) -> str:
        """Send a chat request and return the reply text.

        Some models reject `temperature`, `response_format` or `max_tokens`
        (reasoning models like GPT-5 Nano). On a 400 error mentioning one of
        them, that parameter is dropped/renamed and the call is retried once.
        """
        kwargs: Dict[str, Any] = {"model": model or self.model_for(role), "messages": messages}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        adapted = False
        attempt = 0
        while True:
            try:
                resp = self.client.chat.completions.create(**kwargs)
                self._record_usage(role, resp)
                return resp.choices[0].message.content or ""

            except BadRequestError as e:
                msg = str(e).lower()
                if adapted:
                    raise LLMError(f"[{role}] request rejected: {e}") from e
                changed = False
                named = any(p in msg for p in ("temperature", "response_format", "max_tokens"))
                # Error names the parameter -> drop just that one.
                # Generic 400 (e.g. "could not process this request") -> drop all optional ones.
                if ("temperature" in msg or not named) and "temperature" in kwargs:
                    kwargs.pop("temperature"); changed = True
                if ("response_format" in msg or not named) and "response_format" in kwargs:
                    kwargs.pop("response_format"); changed = True
                if ("max_tokens" in msg or not named) and "max_tokens" in kwargs:
                    kwargs["max_completion_tokens"] = kwargs.pop("max_tokens"); changed = True
                if not changed:
                    raise LLMError(f"[{role}] request rejected: {e}") from e
                adapted = True

            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                attempt += 1
                if attempt > self.max_retries:
                    raise LLMError(f"[{role}] failed after {self.max_retries} retries: {e}") from e
                time.sleep(2 ** attempt)          # 2 s, 4 s, ...

    def chat_json(self, role: str, system: str, user: str,
                  model: Optional[str] = None, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        text = self.chat(role, messages, json_mode=True, model=model, max_tokens=max_tokens)
        return extract_json(text)

    def vision_json(self, role: str, system: str, prompt: str, image_path: str,
                    model: Optional[str] = None, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        path = Path(image_path)
        if not path.exists():
            raise LLMError(f"Image not found: {image_path}")
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]},
        ]
        text = self.chat(role, messages, json_mode=True, model=model, max_tokens=max_tokens)
        return extract_json(text)

    # ------------------------------------------------------------------
    def _record_usage(self, role: str, resp: Any) -> None:
        u = getattr(resp, "usage", None)
        rec = self.usage[role]
        rec["calls"] += 1
        if u:
            rec["prompt"] += getattr(u, "prompt_tokens", 0) or 0
            rec["completion"] += getattr(u, "completion_tokens", 0) or 0

    def usage_report(self) -> str:
        if not self.usage:
            return "No LLM calls yet."
        rows = [f"{r:<14} calls={u['calls']:<3} prompt={u['prompt']:<7} completion={u['completion']}"
                for r, u in self.usage.items()]
        return "\n".join(rows)