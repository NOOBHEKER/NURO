"""Optional local LLM backend for Nuro — PC only, offline, small.

Why this file exists
--------------------
The hand-written study cards in ``host/chatbot.py`` are accurate but limited to
a handful of topics. This module adds a real (small) language model so Nuro can
answer *general* and *detailed* questions across subjects.

Design rules
------------
* **No third-party Python packages.** Only the standard library (``urllib``),
  so it works on Python 3.14 where ``llama-cpp-python`` has no wheels.
* **Runtime agnostic.** It speaks to:
    - Ollama            ``POST /api/chat``            (default 127.0.0.1:11434)
    - OpenAI-compatible ``POST /v1/chat/completions`` (LM Studio, llama.cpp
      server, vLLM, ...) via ``NURO_LLM_URL``.
* **Never required.** Every function returns ``None``/``False`` when no runtime
  is available, so the chatbot silently falls back to study cards + TinyLLM.
* **PC only.** No ``serial``/GPIO imports. Nothing here is flashed to the
  ESP32-S3.

Environment variables
---------------------
``NURO_LLM_URL``    OpenAI-compatible base URL (e.g. http://127.0.0.1:1234)
``NURO_LLM_MODEL``  model name (default ``qwen3:0.6b``, ~522 MB on disk — ultra-fast on CPU & USB)
``NURO_LLM_TIMEOUT``seconds per request (default 180)
``OLLAMA_HOST``     Ollama base URL (default http://127.0.0.1:11434)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

DEFAULT_MODEL = os.environ.get("NURO_LLM_MODEL", "qwen3:0.6b")
FAST_MODEL = os.environ.get("NURO_LLM_FAST_MODEL", "qwen3:0.6b")
OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OPENAI_BASE = os.environ.get("NURO_LLM_URL", "").rstrip("/")
DEFAULT_TIMEOUT = float(os.environ.get("NURO_LLM_TIMEOUT", "180"))

# Ultra-fast CPU & USB defaults:
#   * compact context (512) -> minimal KV cache bandwidth per token
#   * all logical cores     -> maximum CPU tok/s
#   * keep_alive 30m        -> zero reload delay between follow-up questions
CTX = int(os.environ.get("NURO_LLM_CTX", "512"))
THREADS = int(os.environ.get("NURO_LLM_THREADS", str(os.cpu_count() or 4)))
KEEP_ALIVE = os.environ.get("NURO_LLM_KEEP_ALIVE", "30m")

# Ultra-concise system prompt (~22 tokens) so prompt-eval takes <0.3s on CPU!
SYSTEM_PROMPT = (
    "You are Nuro, a fast, accurate offline AI tutor. "
    "Answer clearly and concisely; say if unsure."
)


@dataclass
class Backend:
    kind: str          # "ollama" or "openai"
    base: str
    model: str

    def label(self) -> str:
        return f"{self.kind} · {self.model} · {self.base}"


def _request(url: str, payload: Optional[Dict[str, Any]] = None,
             timeout: float = DEFAULT_TIMEOUT, stream: bool = False):
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if data is not None else "GET")
    return urllib.request.urlopen(req, timeout=timeout)


def _get_json(url: str, timeout: float = 5.0) -> Optional[Any]:
    try:
        with _request(url, None, timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except Exception:
        return None


def ollama_models() -> List[str]:
    """Names of models present in the local Ollama store."""
    data = _get_json(f"{OLLAMA_BASE}/api/tags")
    if not isinstance(data, dict):
        return []
    return [m.get("name", "") for m in data.get("models", []) if m.get("name")]


_MODEL_OVERRIDE: Optional[str] = None


def set_model(model: str) -> None:
    """Force a specific model for this process (``nuro ask --model``)."""
    global _MODEL_OVERRIDE
    _MODEL_OVERRIDE = model.strip() or None


def detect() -> Optional[Backend]:
    """Return a usable backend, or ``None``. Cheap: two short HTTP probes."""
    wanted = _MODEL_OVERRIDE or DEFAULT_MODEL
    if OPENAI_BASE:
        # Trust an explicitly configured OpenAI-compatible server.
        return Backend("openai", OPENAI_BASE, wanted)

    models = ollama_models()
    if models:
        # Prefer the requested model; otherwise the first installed one.
        chosen = wanted if (wanted in models or _MODEL_OVERRIDE) else models[0]
        return Backend("ollama", OLLAMA_BASE, chosen)
    return None


def _portable_ollama_dirs() -> List[str]:
    """Candidate portable ollama folders next to nuro.exe or current working directory."""
    dirs: List[str] = []
    try:
        import sys

        if getattr(sys, "frozen", False):
            exe_dir = os.path.dirname(os.path.abspath(sys.executable))
            dirs.append(os.path.join(exe_dir, "ollama"))
            dirs.append(os.path.join(os.path.dirname(exe_dir), "ollama"))
    except Exception:
        pass
    try:
        cwd_ol = os.path.join(os.getcwd(), "ollama")
        if cwd_ol not in dirs:
            dirs.append(cwd_ol)
    except Exception:
        pass
    return dirs


def ollama_exe() -> Optional[str]:
    """Locate an ``ollama`` binary (portable pendrive next to nuro.exe, G:, user install, or PATH)."""
    candidates = [os.environ.get("NURO_OLLAMA_EXE", "")]
    for d in _portable_ollama_dirs():
        candidates.append(os.path.join(d, "ollama.exe"))
    candidates.extend([
        r"G:\ollama\ollama.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
        os.path.expandvars(r"%ProgramFiles%\Ollama\ollama.exe"),
    ])
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return shutil.which("ollama")


def _ollama_env() -> Dict[str, str]:
    env = os.environ.copy()
    if not env.get("OLLAMA_MODELS"):
        for d in _portable_ollama_dirs():
            m_dir = os.path.join(d, "models")
            if os.path.isdir(m_dir):
                env["OLLAMA_MODELS"] = m_dir
                return env
        if os.path.isdir(r"G:\ollama\models"):
            env["OLLAMA_MODELS"] = r"G:\ollama\models"
    return env


def ensure_server(wait_seconds: float = 25.0) -> bool:
    """Start a local Ollama server if it is installed but not running."""
    if detect() is not None:
        return True
    exe = ollama_exe()
    if not exe:
        return False
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        subprocess.Popen(
            [exe, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
            env=_ollama_env(),
        )
    except Exception:
        return False
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if detect() is not None:
            return True
        time.sleep(1.0)
    return False


def warmup(timeout: float = 60.0) -> bool:
    """Load the model into RAM so the first real question is not slow.

    Ollama unloads after ``KEEP_ALIVE``; a cold load costs several seconds.
    """
    backend = detect()
    if backend is None:
        return False
    payload = _payload(backend, [{"role": "user", "content": "hi"}], 1, 0.0, False)
    try:
        with _request(_endpoint(backend), payload, timeout):
            return True
    except Exception:
        return False


def fast_backend() -> Optional[Backend]:
    """The small fast model, if it is installed."""
    models = ollama_models()
    if FAST_MODEL in models:
        return Backend("ollama", OLLAMA_BASE, FAST_MODEL)
    return None


def pull_model(model: str) -> int:
    """Download a model through the local Ollama binary (0 = success)."""
    exe = ollama_exe()
    if not exe:
        return 127
    return subprocess.call([exe, "pull", model], env=_ollama_env())


def status_text() -> str:
    backend = detect()
    if backend is None:
        return (
            "No local LLM runtime found.\n"
            "  • Ollama: expected at " + OLLAMA_BASE + " (no models responded)\n"
            "  • OpenAI-compatible: set NURO_LLM_URL\n"
            "Nuro will keep using its offline study cards."
        )
    models = ollama_models() if backend.kind == "ollama" else []
    lines = [f"Local LLM ready: {backend.label()}"]
    if models:
        lines.append("Installed models: " + ", ".join(models))
    return "\n".join(lines)


def _messages(question: str, history: Optional[List[Dict[str, str]]],
              system: Optional[str]) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "system", "content": system or SYSTEM_PROMPT}]
    if history:
        # Keep the last few turns only; small models lose focus with long context.
        msgs.extend(history[-6:])
    msgs.append({"role": "user", "content": question})
    return msgs


def _payload(backend: Backend, messages: List[Dict[str, str]],
             max_tokens: int, temperature: float, stream: bool) -> Dict[str, Any]:
    if backend.kind == "ollama":
        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": CTX,
        }
        if THREADS > 0:
            options["num_thread"] = THREADS
        return {
            "model": backend.model,
            "messages": messages,
            "stream": stream,
            "think": False,
            "options": options,
            "keep_alive": KEEP_ALIVE,
        }
    return {
        "model": backend.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }


def _endpoint(backend: Backend) -> str:
    return f"{backend.base}/api/chat" if backend.kind == "ollama" else f"{backend.base}/v1/chat/completions"


def ask(question: str, *, history: Optional[List[Dict[str, str]]] = None,
        system: Optional[str] = None, max_tokens: int = 320,
        temperature: float = 0.3, timeout: float = DEFAULT_TIMEOUT) -> Optional[str]:
    """One-shot answer, or ``None`` if no runtime / request failed."""
    backend = detect()
    if backend is None:
        return None
    payload = _payload(backend, _messages(question, history, system), max_tokens, temperature, False)
    try:
        with _request(_endpoint(backend), payload, timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        if backend.kind == "ollama":
            return (data.get("message") or {}).get("content", "").strip() or None
        return (data["choices"][0]["message"]["content"] or "").strip() or None
    except Exception:
        return None


def ask_stream(question: str, *, history: Optional[List[Dict[str, str]]] = None,
               system: Optional[str] = None, max_tokens: int = 320,
               temperature: float = 0.3,
               timeout: float = DEFAULT_TIMEOUT) -> Iterator[str]:
    """Yield answer chunks as they arrive (falls back to one chunk)."""
    backend = detect()
    if backend is None:
        return
    payload = _payload(backend, _messages(question, history, system), max_tokens, temperature, True)
    try:
        with _request(_endpoint(backend), payload, timeout, stream=True) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                if backend.kind == "ollama":
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    piece = (obj.get("message") or {}).get("content", "")
                    if piece:
                        yield piece
                    if obj.get("done"):
                        return
                else:
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if line == "[DONE]":
                        return
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    piece = (obj["choices"][0].get("delta") or {}).get("content", "")
                    if piece:
                        yield piece
    except Exception:
        # Fall back to a single non-streaming answer.
        answer = ask(question, history=history, system=system,
                     max_tokens=max_tokens, temperature=temperature, timeout=timeout)
        if answer:
            yield answer
