"""Built-in Gemini LLM AI Engine for nuro.

Provides direct AI mathematical, physical, and chemical reasoning using the
connected Gemini API key. Handles complex word problems, conceptual explanations,
quadratic derivations, and natural-language queries.
"""
from __future__ import annotations

import os
import sys
import json
from typing import Optional, List, Dict, Any
import requests

from host.calc import (
    C_RESET, C_BOLD, C_CYAN, C_GREEN, C_AMBER, C_RED, C_GRAY, C_WHITE, C_PURPLE,
    clean_math_signs,
)

GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
]


def get_api_key() -> Optional[str]:
    """Retrieve Gemini API key from environment."""
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def query_gemini(prompt: str, system_instruction: Optional[str] = None) -> str:
    """Send a query to the Gemini API with automatic model fallback."""
    key = get_api_key()
    if not key:
        return (
            f"{C_RED}[ERROR]: GEMINI_API_KEY not found in environment.{C_RESET}\n"
            f"{C_GRAY}To enable LLM reasoning, set the environment variable:{C_RESET}\n"
            f"  set GEMINI_API_KEY=your_key_here"
        )

    sys_text = (
        system_instruction
        or "You are nuro-AI, an expert STEM tutor and engineer. "
        "Answer with extreme mathematical and scientific precision. "
        "Be concise and point-wise (minimum verbose). "
        "STRICT REQUIREMENT: NEVER output LaTeX math syntax or dollar delimiters. "
        "Never output strings like '$x_2 = 6 - \\sqrt{33}$', '\\frac{a}{b}', or '\\pm'. "
        "Always use plain Unicode mathematical signs: x₂ = 6 - √33, (a)/(b), ±, ², ³, ·, ≈, Δ, π, →. "
        "Provide step-by-step derivations for algebra, calculus, physics, and chemistry questions."
    )

    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "systemInstruction": {
            "parts": [{"text": sys_text}]
        },
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024,
        },
    }

    last_error = ""
    for model in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        raw_ai_text = parts[0].get("text", "").strip()
                        cleaned_ai_text = clean_math_signs(raw_ai_text)
                        out = [
                            f"{C_BOLD}{C_GREEN}🧠 nuro-AI ({model}):{C_RESET}",
                            f"{C_WHITE}{cleaned_ai_text}{C_RESET}",
                        ]
                        return "\n".join(out)
            else:
                last_error = f"HTTP {resp.status_code}: {resp.text[:120]}"
        except Exception as e:
            last_error = str(e)

    return f"{C_RED}[LLM ERROR]: Failed to query Gemini models ({last_error}){C_RESET}"


def run_ai_repl() -> None:
    """Interactive LLM Shell."""
    if os.name == "nt":
        os.system("")
    print(f"\n{C_BOLD}{C_CYAN}╔═══════════════════════════════════════════════════════════════╗{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}║     🧠  nuro-AI — Intelligent STEM & Engineering Reasoning     ║{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}╚═══════════════════════════════════════════════════════════════╝{C_RESET}")
    print(f"{C_GRAY}Powered by Gemini LLM. Ask any math, physics, or chemistry question ('q' to quit){C_RESET}")
    print(f"{C_GRAY}Examples:{C_RESET} {C_GREEN}explain quadratic formula derivation{C_RESET}")
    print(f"          {C_AMBER}why does pressure transmit equally in fluids?{C_RESET}\n")

    while True:
        try:
            raw = input(f"{C_BOLD}{C_CYAN}nuro-ai>{C_RESET} ").strip()
            if not raw:
                continue
            if raw.lower() in ("exit", "quit", "q"):
                print(f"{C_GRAY}Exiting nuro-ai.{C_RESET}")
                break
            print()
            print(query_gemini(raw))
            print()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{C_GRAY}Exiting nuro-ai.{C_RESET}")
            break
        except Exception as e:
            print(f"{C_RED}[AI ERROR]:{C_RESET} {e}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(query_gemini(" ".join(sys.argv[1:])))
    else:
        run_ai_repl()
