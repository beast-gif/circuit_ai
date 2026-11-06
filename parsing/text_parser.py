"""
Text question -> validated Netlist.

Loop:
  1. LLM (text_parser role) writes the netlist JSON
  2. normalise common LLM slips ("resistor" -> "R", "gnd" -> "0", numbers -> strings)
  3. schemas.Netlist checks each component, validator.validate checks the whole circuit
  4. consistency_errors() compares the netlist with phrases in the question
     ("output taken across R", "voltage across R2") -- catches circuits that are valid but not the one asked
  5. if anything fails, the exact errors go back to the LLM and it tries again
  6. after `max_attempts` failures with the cheap model, one last try with fallback_parser (Sonnet 5)
  7. if the only remaining problem is a consistency mismatch (the check is phrase-based, so it can
     misfire), the netlist is still returned, with the mismatch as a WARNING for the confirmation step
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from llm_client import LLMError, VelonaClient, extract_json
from parsing.prompts import NOT_JSON_FEEDBACK, RETRY_FEEDBACK, TEXT_PARSER_SYSTEM, TEXT_PARSER_USER
from schemas import Netlist
from validator import ValidationReport, validate

# LLM spelling -> (type, default model)
TYPE_ALIASES: Dict[str, tuple] = {
    "resistor": ("R", None), "res": ("R", None),
    "capacitor": ("C", None), "cap": ("C", None),
    "inductor": ("L", None),
    "voltage_source": ("V", None), "vsource": ("V", None), "battery": ("V", None),
    "dc_voltage_source": ("V", None), "ac_voltage_source": ("V", None), "vs": ("V", None),
    "current_source": ("I", None), "isource": ("I", None),
    "dc_current_source": ("I", None), "ac_current_source": ("I", None),
    "diode": ("D", None),
    "bjt": ("Q", None), "transistor": ("Q", None), "npn": ("Q", "NPN"), "pnp": ("Q", "PNP"),
    "mosfet": ("M", None), "nmos": ("M", "NMOS"), "pmos": ("M", "PMOS"),
    "opamp": ("OPAMP", "ideal"), "op_amp": ("OPAMP", "ideal"),
    "operational_amplifier": ("OPAMP", "ideal"),
    "vcvs": ("E", None), "vccs": ("G", None), "cccs": ("F", None), "ccvs": ("H", None),
}
VALID_TYPES = {"R", "C", "L", "V", "I", "D", "Q", "M", "OPAMP", "E", "G", "F", "H"}

# when the LLM uses named keys instead of a "nodes" list
NODE_KEY_ORDERS = [
    ("n+", "n-"), ("n1", "n2"), ("positive", "negative"), ("anode", "cathode"),
    ("collector", "base", "emitter"), ("drain", "gate", "source"),
    ("out", "in+", "in-"), ("n+", "n-", "nc+", "nc-"),
]

QUERY_NAMES = {"v": "V", "i": "I", "p": "P", "z": "Z", "h": "H", "rth": "Rth", "vth": "Vth",
               "zth": "Zth", "isc": "Isc", "rn": "Rn", "in": "In", "bias": "bias"}
BARE_NAMES = {"rin": "Rin", "rout": "Rout", "gain": "gain", "fc": "fc", "f0": "f0", "poles": "poles",
              "zeros": "zeros", "tau": "tau", "qfactor": "Qfactor", "bandwidth": "bandwidth"}


@dataclass
class ParseResult:
    ok: bool
    netlist: Optional[Netlist]
    report: Optional[ValidationReport]
    missing_info: List[str] = field(default_factory=list)
    attempts: int = 0
    model_used: str = ""
    errors: List[str] = field(default_factory=list)   # last errors if ok is False
    raw: Dict[str, Any] = field(default_factory=dict)  # last JSON from the model


# ----------------------------------------------------------------------
# Normalisation of common LLM output slips
# ----------------------------------------------------------------------
def _to_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return f"{v:g}"
    s = str(v).strip()
    return s or None


def _norm_query(q: Any) -> str:
    s = str(q).replace(" ", "")
    m = re.match(r"^(\w+)\((.*)\)$", s)
    if m:
        name = QUERY_NAMES.get(m.group(1).lower(), m.group(1))
        return f"{name}({m.group(2)})"
    return BARE_NAMES.get(s.lower(), s)


def normalize(data: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(data)
    out.pop("missing_info", None)

    comps = []
    for raw in data.get("components") or []:
        c = dict(raw)

        t = str(c.get("type", "")).strip()
        if t.upper() in VALID_TYPES:
            c["type"] = t.upper()
        else:
            key = t.lower().replace("-", "_").replace(" ", "_")
            if key in TYPE_ALIASES:
                c["type"], default_model = TYPE_ALIASES[key]
                if default_model and not c.get("model"):
                    c["model"] = default_model

        if "nodes" not in c:
            for keys in NODE_KEY_ORDERS:
                if all(k in c for k in keys):
                    c["nodes"] = [c.pop(k) for k in keys]
                    break
        if "nodes" in c:
            c["nodes"] = [_to_str(n) or "" for n in c["nodes"]]

        for k in ("value", "ac"):
            if k in c:
                c[k] = _to_str(c[k])
        if "ac_phase" in c:
            try:
                c["ac_phase"] = float(c["ac_phase"] or 0)
            except (TypeError, ValueError):
                c["ac_phase"] = 0.0
        if c.get("model") is not None:
            c["model"] = str(c["model"]).strip()
        if c.get("params") is None:
            c.pop("params", None)
        comps.append(c)
    out["components"] = comps

    q = data.get("query") or []
    out["query"] = [_norm_query(x) for x in (q if isinstance(q, list) else [q])]
    out["frequency"] = _to_str(data.get("frequency"))
    out["assumptions"] = [str(a) for a in (data.get("assumptions") or [])]
    return out


# ----------------------------------------------------------------------
# Question <-> netlist consistency
# ----------------------------------------------------------------------
_VALUE_WORDS = r"(?:[\d.]+\s*[a-zA-Zµμ]*\s+)?"           # optional "10 Ohm " / "1 uF "
_ELEMENT = r"(resistor|capacitor|inductor|[RCL]\d*)\b(?:\s+([RCL]\d+)\b)?"

# "output (voltage) is taken across (the) (10 Ohm) resistor (R2)", "Vout across C1"
_OUT_ACROSS = re.compile(r"\b(?:output|v_?out|v_?o)\b[^.;]*?\bacross\s+(?:the\s+)?" + _VALUE_WORDS + _ELEMENT,
                         re.IGNORECASE)
# "voltage (drop) across (the) (resistor) R2"  -- explicit ids only
_VOLT_ACROSS = re.compile(r"\bvoltage(?:\s+drop)?\s+across\s+(?:the\s+)?" + _VALUE_WORDS
                          + r"(?:resistor\s+|capacitor\s+|inductor\s+)?([RCL]\d+)\b", re.IGNORECASE)

_WORD_TYPE = {"resistor": "R", "r": "R", "capacitor": "C", "c": "C", "inductor": "L", "l": "L"}
_OUT_NODES = ("out", "vout", "Vout", "vo", "Vo", "o", "output")


def _output_node(nl: Netlist) -> Optional[str]:
    nodes = nl.nodes()
    for q in nl.query:
        m = re.match(r"^H\((\w+),", q)
        if m:
            return m.group(1)
    for n in _OUT_NODES:
        if n in nodes:
            return n
    vq = [m.group(1) for q in nl.query if (m := re.match(r"^V\((\w+)\)$", q))]
    return vq[0] if len(vq) == 1 else None


def _candidates(nl: Netlist, word: str, explicit_id: Optional[str]) -> Tuple[str, List]:
    """Components a phrase like 'resistor R2' / 'R' / 'capacitor' can refer to -> (description, list)."""
    ref = (explicit_id or word).strip()
    if re.fullmatch(r"[RCLrcl]\d+", ref):
        cid = ref[0].upper() + ref[1:]
        comp = nl.get(cid)
        if comp:
            return cid, [comp]
        typ = cid[0]                      # parser renamed it: fall back to any element of that type
    else:
        typ = _WORD_TYPE.get(ref.lower(), "")
    desc = {"R": "the resistor", "C": "the capacitor", "L": "the inductor"}.get(typ, ref)
    return desc, [c for c in nl.components if c.type.value == typ]


def consistency_errors(question: str, nl: Netlist) -> List[str]:
    """Check phrases in the question against the netlist. Returns problems as LLM feedback."""
    if nl.analysis.value == "conceptual" or not question:
        return []
    errors: List[str] = []

    # --- "output taken across X": X must connect the output node to ground
    out = _output_node(nl)
    for m in _OUT_ACROSS.finditer(question):
        desc, cands = _candidates(nl, m.group(1), m.group(2))
        if not cands:
            errors.append(f"The question says the output is across {desc}, but the netlist has no such component.")
            continue
        if out is None:
            continue
        if not any(set(c.nodes[:2]) == {out, "0"} for c in cands):
            where = ", ".join(f"{c.id} ({c.nodes[0]}–{c.nodes[1]})" for c in cands)
            errors.append(f'The question says the output is across {desc}, but {where} does not connect the '
                          f'output node "{out}" to ground "0". Re-order the series elements so {desc} '
                          f'connects "{out}" to "0".')

    # --- "voltage across R2": some V query must measure exactly R2's nodes
    measured = []
    for q in nl.query:
        if (m := re.match(r"^V\((\w+)\)$", q)):
            measured.append({m.group(1), "0"})
        elif (m := re.match(r"^V\((\w+),(\w+)\)$", q)):
            measured.append({m.group(1), m.group(2)})
    if measured:
        for m in _VOLT_ACROSS.finditer(question):
            cid = m.group(1)[0].upper() + m.group(1)[1:]
            comp = nl.get(cid)
            if comp is None:
                continue
            a, b = comp.nodes[0], comp.nodes[1]
            if set(comp.nodes[:2]) not in measured:
                fix = f"V({a})" if b == "0" else f"V({b})" if a == "0" else f"V({a},{b})"
                errors.append(f"The question asks for the voltage across {cid}, which connects {a} and {b}, "
                              f"but no query measures those two nodes. Use {fix}.")
    return errors


def _pydantic_errors(e: ValidationError, data: Dict[str, Any]) -> List[str]:
    """Turn pydantic errors into messages the LLM can act on ('component R3: ...')."""
    msgs = []
    comps = data.get("components") or []
    for err in e.errors():
        loc = list(err.get("loc", ()))
        where = ".".join(str(x) for x in loc) or "netlist"
        if len(loc) >= 2 and loc[0] == "components" and isinstance(loc[1], int) and loc[1] < len(comps):
            cid = comps[loc[1]].get("id", f"#{loc[1]}")
            field_path = ".".join(str(x) for x in loc[2:])
            where = f"component {cid}" + (f" ({field_path})" if field_path else "")
        msg = str(err.get("msg", "")).replace("Value error, ", "")
        msgs.append(f"{where}: {msg}")
    return msgs


# ----------------------------------------------------------------------
def parse_text(question: str, client: Optional[VelonaClient] = None,
               max_attempts: int = 2, use_fallback: bool = True) -> ParseResult:
    client = client or VelonaClient()
    base = [{"role": "system", "content": TEXT_PARSER_SYSTEM},
            {"role": "user", "content": TEXT_PARSER_USER.format(question=question.strip())}]

    plan = [(client.model_for("text_parser"), max_attempts)]
    if use_fallback and client.fallback_parser != plan[0][0]:
        plan.append((client.fallback_parser, 1))

    attempts = 0
    last_errors: List[str] = []
    raw: Dict[str, Any] = {}
    missing: List[str] = []
    model = plan[0][0]
    best: Optional[tuple] = None     # valid netlist that only failed the consistency check

    for model, n_tries in plan:
        messages = list(base)                 # each model starts fresh
        for _ in range(n_tries):
            attempts += 1
            try:
                text = client.chat("text_parser", messages, json_mode=True, model=model)
            except LLMError as e:             # request itself failed
                last_errors = [str(e)]
                continue
            try:
                raw = extract_json(text)
            except LLMError as e:
                last_errors = [str(e)]
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": NOT_JSON_FEEDBACK}]
                continue

            missing = [str(m) for m in (raw.get("missing_info") or [])]
            data = normalize(raw)
            data["question"] = question.strip()

            try:
                nl = Netlist.model_validate(data)
            except ValidationError as e:
                errs = _pydantic_errors(e, data)
            else:
                report = validate(nl)
                if report.ok:
                    errs = consistency_errors(question, nl)
                    if not errs:
                        return ParseResult(True, nl, report, missing, attempts, model, [], raw)
                    best = (nl, report, errs, list(missing), model, raw)
                else:
                    errs = report.errors

            last_errors = errs
            messages += [{"role": "assistant", "content": json.dumps(raw)},
                         {"role": "user",
                          "content": RETRY_FEEDBACK.format(errors="\n".join(f"- {x}" for x in errs))}]

    if best is not None:
        nl, report, errs, missing, model, raw = best
        report.warnings += [f"Possible mismatch with the question: {e}" for e in errs]
        return ParseResult(True, nl, report, missing, attempts, model, [], raw)
    return ParseResult(False, None, None, missing, attempts, model, last_errors, raw)