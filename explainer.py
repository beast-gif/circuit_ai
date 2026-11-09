"""
StepTrace -> readable worked solution.

Rule: the LLM only REWORDS. It never calculates.
  - It receives the solver's steps (key or full level), the question and the topic.
  - It writes one explained step per trace step: title, the idea behind it (KCL, Ohm's law,
    impedance ...), and the equation exactly as given.
  - The "Final answers" block and the verification line are added by code, not by the LLM,
    so the answers shown are always the solver's exact values.

Number guard: every number in the LLM's text must already appear in the question, the trace or
the answers (allowing unit rescaling: 3.333 mA = 0.003333 A, and rounding within 0.5 %).
If the LLM introduces a new number, it gets one retry with the offending numbers listed; if it
still does, the deterministic template (render_steps) is used instead. So a wrong number from the
LLM can never reach the student.

Levels:
  "key"  -> only steps marked key, concise          (default)
  "full" -> every step, with a bit more explanation  ("show complete")
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set

from llm_client import LLMError, VelonaClient
from schemas import Netlist, SolveResult, Step, parse_value

LEVELS = ("key", "full")

EXPLAINER_SYSTEM = """You turn a circuit solver's step trace into a clear worked solution for an electronics student.

STRICT RULES
1. Use ONLY the steps, equations and numbers you are given. Never calculate anything, never add a
   number that is not in the input, never change a number.
2. Copy every equation exactly. You may only tidy notation: * -> ·, ^2 -> ², sqrt -> √.
3. One numbered step per trace step, in the same order. Each step:
   **N. Short title**
   One sentence saying WHY this step is done (the idea: KCL, Ohm's law, voltage source fixes a node,
   impedance of C is 1/(sC), poles are roots of the denominator ...).
   The equation(s) in backticks on their own line, then the result in backticks if there is one.
4. Start with one line:  Approach: <one sentence naming the method>.
5. Do NOT write a final-answer section or a verification line; they are added automatically.
6. Markdown only. No LaTeX. No greeting, no closing remarks.
LEVEL "key": keep the why-sentence short. LEVEL "full": the why-sentence may be two sentences and may
mention the sign convention or the physical meaning."""

EXPLAINER_USER = """Question: {question}
Topic: {topic}
Level: {level}
Assumptions stated in the question: {assumptions}

Solver steps (in order):
{steps}

Write the worked solution now."""

RETRY_NUMBERS = ("These numbers in your answer do not appear in the solver steps: {nums}. "
                 "Rewrite the solution using only numbers from the steps. Return the full solution.")


@dataclass
class Explanation:
    text: str
    level: str
    source: str                       # "llm" | "template"
    has_more: bool = False            # True if level "full" would show more steps
    warnings: List[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# LaTeX cleanup: models sometimes write \(V_1\) or V_{\text{out}} despite "no LaTeX".
# A web page would show those backslashes literally, so convert them to plain Unicode.
# ----------------------------------------------------------------------
_LATEX_SYMBOLS = {
    r"\cdot": "·", r"\times": "×", r"\div": "÷", r"\approx": "≈", r"\neq": "≠", r"\leq": "≤", r"\geq": "≥",
    r"\le": "≤", r"\ge": "≥", r"\ll": "≪", r"\gg": "≫", r"\pm": "±", r"\infty": "∞", r"\rightarrow": "→",
    r"\to": "→", r"\Rightarrow": "⇒", r"\angle": "∠", r"\circ": "°", r"\degree": "°",
    r"\omega": "ω", r"\Omega": "Ω", r"\pi": "π", r"\beta": "β", r"\alpha": "α", r"\mu": "µ", r"\tau": "τ",
    r"\theta": "θ", r"\phi": "φ", r"\Delta": "Δ", r"\delta": "δ", r"\lambda": "λ", r"\gamma": "γ",
    r"\sigma": "σ", r"\zeta": "ζ", r"\partial": "∂",
}


def clean_latex(text: str) -> str:
    t = text
    for delim in (r"\(", r"\)", r"\[", r"\]", "$$"):
        t = t.replace(delim, "")
    t = re.sub(r"(?<!\\)\$", "", t)                                     # inline $...$
    t = re.sub(r"\\(?:left|right|displaystyle|,|;|!|quad|qquad)", " ", t)
    for _ in range(4):                                                   # nested braces, innermost first
        t = re.sub(r"\\(?:text|mathrm|mathbf|mathit|operatorname)\{([^{}]*)\}", r"\1", t)
        t = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", t)
        t = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", t)
        t = re.sub(r"_\{([^{}]*)\}", r"_\1", t)
        t = re.sub(r"\^\{([^{}]*)\}", r"^\1", t)
    for k in sorted(_LATEX_SYMBOLS, key=len, reverse=True):
        t = re.sub(re.escape(k) + r"(?![A-Za-z])", _LATEX_SYMBOLS[k], t)
    t = re.sub(r"\\([A-Za-z]+)", r"\1", t)                               # any leftover \command
    t = re.sub(r"\^2(?![\d.])", "²", t)
    t = re.sub(r"\^3(?![\d.])", "³", t)
    return t


# ----------------------------------------------------------------------
# Deterministic rendering (no LLM) -- also the fallback
# ----------------------------------------------------------------------
def _pretty_math(s: str) -> str:
    s = s.replace("**", "^").replace("*", "·")
    s = re.sub(r"\^2\b", "²", s)
    s = re.sub(r"\^3\b", "³", s)
    return s


def select_steps(res: SolveResult, level: str) -> List[Step]:
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    return res.trace.key_steps() if level == "key" else res.trace.all_steps()


def render_steps(steps: Iterable[Step]) -> str:
    out = []
    for i, st in enumerate(steps, 1):
        block = [f"**{i}. {st.action}**"]
        if st.equation:
            block.append(f"`{_pretty_math(st.equation)}`")
        if st.result:
            block.append(f"→ `{_pretty_math(st.result)}`")
        out.append("  \n".join(block))
    return "\n\n".join(out)


def render_answers(res: SolveResult) -> str:
    lines = ["**Final answers**"]
    for q, a in res.answers.items():
        lines.append(f"- {q} = {_pretty_math(a)}" if not a.startswith("Could not") else f"- {q}: {a}")
    return "\n".join(lines)


def render_verification(res: SolveResult) -> str:
    sat = [n.removeprefix("saturation: ") for n in res.verification_notes if n.startswith("saturation:")]
    info = [n.removeprefix("info: ") for n in res.verification_notes if n.startswith("info:")]
    other = [n for n in res.verification_notes if not n.startswith(("saturation:", "info:"))]
    if res.verified is True:
        line = "✅ Cross-checked with ngspice: all answers match."
    elif res.verified is False:
        line = "⚠️ ngspice disagrees with at least one answer, so treat it with caution: " + "; ".join(
            n for n in other if "vs ngspice" in n)
    else:
        line = "ℹ️ Not cross-checked with ngspice" + (f" ({other[-1]})" if other else "") + "."
    return line + "".join(f"\n\n⚠️ {s}" for s in sat) + "".join(f"\n\nℹ️ {s}" for s in info)


# ----------------------------------------------------------------------
# Number guard
# ----------------------------------------------------------------------
_NUM = re.compile(r"(?<![A-Za-z_\d.])[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_NUM_SI = re.compile(r"(?<![A-Za-z_\d.])(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?(?:Meg|meg|[TGMkKmuµnpf])?")
_CONSTANTS = [math.pi, 2 * math.pi, math.sqrt(2), 1 / math.sqrt(2), 70.71, 0.632, 63.2, 0.368, 36.8,
              math.e, 20.0, 3.0, 45.0, 90.0, 180.0, 360.0]


def allowed_numbers(texts: Iterable[str]) -> Set[float]:
    vals: Set[float] = set()
    for t in texts:
        for m in _NUM_SI.finditer(t or ""):
            tok = m.group(0)
            for v in (_safe_float(re.sub(r"[A-Za-zµ]+$", "", tok)), _safe_parse(tok)):
                if v is not None:
                    vals.add(abs(v))
    vals.update(_CONSTANTS)
    expanded = set()
    for v in vals:                                  # unit rescaling: 3.333 mA <-> 0.003333 A
        for k in range(-5, 6):
            expanded.add(v * 10 ** (3 * k))
    return expanded


def _safe_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except ValueError:
        return None


def _safe_parse(s: str) -> Optional[float]:
    try:
        return parse_value(s)
    except ValueError:
        return None


def unsupported_numbers(text: str, allowed: Set[float], rel: float = 5e-3) -> List[str]:
    """Numbers in `text` that match nothing in `allowed` (integers up to 20 are always fine:
    step numbers, exponents, the 2 in 2π)."""
    bad = []
    allowed_list = sorted(allowed)
    for m in _NUM.finditer(text):
        x = _safe_float(m.group(0))
        if x is None:
            continue
        x = abs(x)
        if x == int(x) and x <= 20:
            continue
        if not any(abs(x - a) <= rel * max(a, 1e-30) for a in _near(allowed_list, x)):
            bad.append(m.group(0))
    return sorted(set(bad))


def _near(sorted_vals: List[float], x: float) -> List[float]:
    import bisect
    i = bisect.bisect_left(sorted_vals, x)
    return sorted_vals[max(0, i - 2): i + 2]


# ----------------------------------------------------------------------
def explain(nl: Netlist, res: SolveResult, level: str = "key", topic: str = "",
            client: Optional[VelonaClient] = None, use_llm: bool = True) -> Explanation:
    steps = select_steps(res, level)
    has_more = level == "key" and len(res.trace.all_steps()) > len(steps)
    tail = "\n\n" + render_answers(res) + "\n\n" + render_verification(res)
    if has_more:
        tail += "\n\n_Ask for the complete solution to see every step._"

    if not use_llm:
        return Explanation(render_steps(steps) + tail, level, "template", has_more)

    step_lines = []
    for i, st in enumerate(steps, 1):
        line = f"{i}. {st.action}"
        if st.equation:
            line += f"\n   equation: {st.equation}"
        if st.result:
            line += f"\n   result: {st.result}"
        step_lines.append(line)

    allowed = allowed_numbers(
        [nl.question, nl.to_spice(), *nl.assumptions, *res.answers.values()]
        + [f"{s.action} {s.equation or ''} {s.result or ''}" for s in res.trace.all_steps()])

    warnings: List[str] = []
    try:
        client = client or VelonaClient()
        messages = [{"role": "system", "content": EXPLAINER_SYSTEM},
                    {"role": "user", "content": EXPLAINER_USER.format(
                        question=nl.question or "(not given)", topic=topic or "circuit analysis", level=level,
                        assumptions=", ".join(nl.assumptions) or "none", steps="\n".join(step_lines))}]
        for attempt in range(2):
            text = clean_latex(client.chat("explainer", messages).strip())
            bad = unsupported_numbers(text, allowed)
            if not bad and text:
                return Explanation(text + tail, level, "llm", has_more, warnings)
            warnings.append(f"attempt {attempt + 1}: LLM used numbers not in the solution: {bad}")
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": RETRY_NUMBERS.format(nums=", ".join(bad))}]
    except LLMError as e:
        warnings.append(f"LLM unavailable: {e}")

    warnings.append("used the plain step template instead of the LLM wording")
    return Explanation(render_steps(steps) + tail, level, "template", has_more, warnings)