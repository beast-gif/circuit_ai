"""
Theory questions ("why does negative feedback stabilize gain?") -> answer from DeepSeek V4 Pro.

There is no circuit to solve, so there is no solver and no ngspice check. To keep answers
consistent with the solved questions, the prompt includes conventions.yaml (VT = 25 mV,
VBE = 0.7 V, ideal op-amp ...), and the answer follows a fixed shape:
  intuition -> key formula(s) -> a small worked numeric example -> common mistakes.

Safety nets (the LLM does the arithmetic here, so it is checked):
  1. clean_latex()      : \\(V_1\\), V_{\\text{out}} -> plain Unicode
  2. arithmetic_errors(): every "numbers-only expression = number" in the answer is recomputed in
     Python, e.g. "500 / 51 ≈ 9.80" -> 9.804 OK;  "500 / 51 ≈ 9.70" -> flagged.
     On a slip the LLM gets one retry with the wrong lines listed; if it still slips, the answer is
     shown with a warning naming the line and the correct value.
  (Wrong REASONING with correct arithmetic cannot be caught this way; the prompt keeps the worked
   example to one scenario with exact formulas to make that less likely.)

Levels: "key" = short answer (~150 words), "full" = detailed with derivation.
"""
from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any, Dict, List, Optional

import yaml

from explainer import Explanation, clean_latex
from llm_client import LLMError, VelonaClient, load_conventions

CONCEPTUAL_SYSTEM = """You are an analog electronics tutor for engineering students.
Answer the theory question with this structure (Markdown, NO LaTeX, use Unicode like Ω, µ, √, ², β, ω, π):

**Idea** - the intuition in plain words (2-4 sentences).
**Key formula(s)** - the exact formula(s), each symbol defined with its unit.
**Worked example** - ONE scenario with round numbers (you may change ONE value once to show its effect).
  At most 5 arithmetic lines. Write each line as  symbol = formula = numbers = result, with the
  numbers-only part free of units, e.g.  "gm = IC/VT = 0.002 / 0.025 = 0.08 A/V".
  Use only exact formulas. Do not compare an approximate (small-change) formula against an exact
  calculation unless the question asks for it.
**Common mistakes** - 1-3 short bullets.

Rules:
- Use these textbook defaults unless the question gives its own values:
{conventions}
- Never invent datasheet figures for specific parts (e.g. an exact LM741 slew rate). If the question
  needs one, give the typical textbook value, say it is typical, and tell the student to check the datasheet.
- If the question is actually a numeric circuit problem, say so in one line and ask for the circuit values.
LEVEL "key": about 150 words in total. LEVEL "full": up to 450 words, and derive the key formula."""

RETRY_ARITHMETIC = ("Some arithmetic in your answer is wrong:\n{errors}\n"
                    "Recompute those lines and return the complete corrected answer.")


# ----------------------------------------------------------------------
# Arithmetic checker
# ----------------------------------------------------------------------
_EXPR_CHARS = set("0123456789.+-−×*/÷·()^√π eE")
_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


def _normalize_line(line: str) -> str:
    """10⁻³ -> 10^(-3),  2π -> 2*π  so they stay inside one arithmetic expression."""
    line = re.sub(r"[⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+", lambda m: "^(" + m.group(0).translate(_SUPERSCRIPT) + ")", line)
    return re.sub(r"(\d|\))\s*π", r"\1*π", line)
_LITERAL = re.compile(r"[-−+]?\d+(?:\.(\d+))?(?:[eE][-+]?\d+)?")
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow}


def _eval_node(n):
    if isinstance(n, ast.Expression):
        return _eval_node(n.body)
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
        return float(n.value)
    if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
        a, b = _eval_node(n.left), _eval_node(n.right)
        if isinstance(n.op, ast.Pow) and abs(b) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(n.op)](a, b)
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
        v = _eval_node(n.operand)
        return -v if isinstance(n.op, ast.USub) else v
    if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "sqrt"
            and len(n.args) == 1 and not n.keywords):
        return math.sqrt(_eval_node(n.args[0]))
    raise ValueError("not plain arithmetic")


def safe_eval(expr: str) -> Optional[float]:
    """Evaluate a numbers-only expression ('1000 / (1 + 1000 × 0.1)', '√(1e8/1)'); None if not arithmetic."""
    e = expr.strip().rstrip(".,;:").strip()
    if not e or not re.search(r"\d", e):
        return None
    e = (e.replace("×", "*").replace("·", "*").replace("÷", "/").replace("−", "-").replace("^", "**")
          .replace("π", f"({math.pi!r})"))
    e = re.sub(r"√\s*\(", "sqrt(", e)
    e = re.sub(r"√\s*(\d+(?:\.\d+)?)", r"sqrt(\1)", e)
    try:
        return _eval_node(ast.parse(e, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError):
        return None


def _tail(seg: str) -> str:
    """Arithmetic at the END of a segment ('the 5 + 3 ' -> '5 + 3 ')."""
    i = len(seg)
    while i > 0 and seg[i - 1] in _EXPR_CHARS:
        i -= 1
    return re.sub(r"^[eE\s]+", "", seg[i:])        # an 'e' here is the end of a word, not an exponent


def _head(seg: str) -> str:
    """Arithmetic at the START of a segment (' 9.80 each' -> ' 9.80')."""
    i = 0
    while i < len(seg) and seg[i] in _EXPR_CHARS:
        i += 1
    return re.sub(r"[eE\s]+$", "", seg[:i])


def _round_tol(side: str) -> float:
    """A plain literal like '9.90' was rounded: allow half a unit in its last digit."""
    s = side.strip().rstrip(".,;:").strip()
    m = _LITERAL.fullmatch(s)
    if not m:
        return 0.0
    decimals = len(m.group(1) or "")
    if "e" in s.lower():
        mant = s.lower().split("e")[0]
        decimals = len(mant.split(".")[1]) if "." in mant else 0
        return 0.5 * 10 ** (-decimals) * 10 ** int(s.lower().split("e")[1])
    return 0.5 * 10 ** (-decimals)


def arithmetic_errors(text: str) -> List[str]:
    """Check every 'arithmetic = arithmetic' pair in the text (units/letters make a side unchecked)."""
    errors = []
    for line in text.splitlines():
        parts = re.split(r"(=|≈)", _normalize_line(line))
        for k in range(1, len(parts) - 1, 2):
            left, sign, right = _tail(parts[k - 1]), parts[k], _head(parts[k + 1])
            a, b = safe_eval(left), safe_eval(right)
            if a is None or b is None:
                continue
            pct = parts[k + 1][len(right):].lstrip().startswith("%")   # "≈ 1%" means 0.01
            if pct:
                b /= 100
            if not (re.search(r"[+\-−×*/÷·^√]", left.strip().lstrip("-−+")) or
                    re.search(r"[+\-−×*/÷·^√]", right.strip().lstrip("-−+"))):
                continue                                       # number = number: nothing to compute
            lt, rt = _round_tol(left), _round_tol(right) / (100 if pct else 1)
            tol = max(lt, rt, 1e-9 * max(abs(a), abs(b)))
            if sign == "≈":
                tol = max(tol, 5e-3 * max(abs(a), abs(b)))
            if abs(a - b) > tol:
                exact = a if lt == 0 else b
                shown = f"{exact * 100:.4g}%" if pct else f"{exact:.6g}"
                errors.append(f"`{left.strip()} {sign} {right.strip().rstrip('.,;:')}{'%' if pct else ''}` "
                              f"(the arithmetic gives {shown})")
    return errors


# ----------------------------------------------------------------------
def _conventions_text(conv: Dict[str, Any]) -> str:
    keep = {k: conv[k] for k in ("semiconductor", "mosfet", "opamp") if k in conv}
    return "\n".join("  " + line for line in yaml.safe_dump(keep, sort_keys=False).splitlines())


def answer_conceptual(question: str, level: str = "key", client: Optional[VelonaClient] = None,
                      conventions: Optional[Dict[str, Any]] = None) -> Explanation:
    if level not in ("key", "full"):
        raise ValueError("level must be 'key' or 'full'")
    conventions = conventions or load_conventions()
    system = CONCEPTUAL_SYSTEM.format(conventions=_conventions_text(conventions))
    note = ("\n\nℹ️ Theory answer from the language model. Not computed or checked by the solver "
            "(only its arithmetic was re-checked).")
    if level == "key":
        note += "\n\n_Ask for the complete answer for the derivation._"

    warnings: List[str] = []
    try:
        client = client or VelonaClient()
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": f"Level: {level}\nQuestion: {question}"}]
        text = clean_latex(client.chat("conceptual", messages).strip())
        errs = arithmetic_errors(text)
        if errs:
            warnings.append(f"attempt 1 arithmetic slips: {errs}")
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": RETRY_ARITHMETIC.format(errors="\n".join(f"- {e}" for e in errs))}]
            text = clean_latex(client.chat("conceptual", messages).strip())
            errs = arithmetic_errors(text)
    except LLMError as e:
        return Explanation(f"Sorry, the theory model is unavailable right now ({e}).", level, "template",
                           warnings=[str(e)])

    if errs:
        warnings.append(f"still wrong after retry: {errs}")
        text += "\n\n⚠️ Possible arithmetic slip in this answer: " + "; ".join(errs)
    return Explanation(text + note, level, "llm", has_more=(level == "key"), warnings=warnings)