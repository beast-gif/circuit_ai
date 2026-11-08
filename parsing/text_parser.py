"""
Text question -> validated Netlist.

Loop:
  1. LLM (text_parser role) writes the netlist JSON
  2. normalise common LLM slips ("resistor" -> "R", "gnd" -> "0", numbers -> strings)
  3. schemas.Netlist checks each component, validator.validate checks the whole circuit
  4. consistency_errors() compares the netlist with phrases in the question
     ("output taken across R", "voltage across R2") -- catches circuits that are valid but not the one asked --
     and rejects elements the LLM invented to "measure" something (0 V shorts, a source across Z/Rth/Isc terminals)
  5. if anything fails, the exact errors go back to the LLM and it tries again
  6. after `max_attempts` failures with the cheap model, one last try with fallback_parser (Sonnet 5)
  7. if the only remaining problem is a consistency mismatch (the check is phrase-based, so it can
     misfire), the netlist is still returned, with the mismatch as a WARNING for the confirmation step
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from llm_client import LLMError, VelonaClient, extract_json
from parsing.prompts import NOT_JSON_FEEDBACK, RETRY_FEEDBACK, TEXT_PARSER_SYSTEM, TEXT_PARSER_USER
from schemas import Netlist, parse_value
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
    history: List[Dict[str, Any]] = field(default_factory=list)  # why each failed attempt was rejected


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


_COMPONENT_KEYS = ("components", "elements", "parts", "netlist_components")
_QUERY_KEYS = ("query", "queries", "find", "unknowns")


def unwrap(data: Dict[str, Any]) -> Dict[str, Any]:
    """Accept common variations of the JSON shape instead of mistaking them for 'no circuit':
    {"netlist": {...}} / {"circuit": {...}} wrappers, "elements" instead of "components",
    "queries" instead of "query"."""
    if not isinstance(data, dict):
        return {}
    if not any(isinstance(data.get(k), list) for k in _COMPONENT_KEYS):
        nested = [v for v in data.values()
                  if isinstance(v, dict) and any(isinstance(v.get(k), list) for k in _COMPONENT_KEYS)]
        if len(nested) == 1:                     # keep top-level scalars (e.g. "analysis") the wrapper lacks
            top = {k: v for k, v in data.items() if not isinstance(v, dict)}
            data = {**top, **nested[0]}
    data = dict(data)
    if not isinstance(data.get("components"), list):
        for k in _COMPONENT_KEYS[1:]:
            if isinstance(data.get(k), list):
                data["components"] = data.pop(k)
                break
    if "query" not in data:
        for k in _QUERY_KEYS[1:]:
            if k in data:
                data["query"] = data.pop(k)
                break
    return data


def normalize(data: Dict[str, Any]) -> Dict[str, Any]:
    data = unwrap(data)
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
        # an op-amp WITH data is not ideal: the default "ideal" label would be misleading in the netlist
        if c.get("type") == "OPAMP" and c.get("params") and str(c.get("model") or "").lower() == "ideal":
            c["model"] = "nonideal"
        comps.append(c)
    out["components"] = comps

    q = data.get("query") or []
    out["query"] = [_norm_query(x) for x in (q if isinstance(q, list) else [q])]
    out["frequency"] = _to_str(data.get("frequency"))
    out["assumptions"] = [str(a) for a in (data.get("assumptions") or [])]

    # "analysis": tidy the spelling, or work it out when the LLM left it out (saves a whole retry)
    a = str(out.get("analysis") or "").strip().lower().replace("-", "_").replace(" ", "_")
    a = a.removesuffix("_analysis")
    inferred = infer_analysis(out)
    if a in ANALYSES:
        out["analysis"] = a
    elif inferred:
        out["analysis"] = inferred
    else:
        out.pop("analysis", None)        # nothing to infer from -> schema error -> the LLM retries
    return out


ANALYSES = {"dc", "ac", "transient", "small_signal", "conceptual"}
_AC_KINDS = {"fc", "f0", "poles", "zeros", "Qfactor", "bandwidth", "H", "gain"}


def infer_analysis(data: Dict[str, Any]) -> Optional[str]:
    """Analysis type from the rest of the netlist, used only when the LLM omitted it.
    No components -> None (NOT "conceptual"): an empty or oddly shaped reply must be retried,
    never silently turned into a theory question."""
    comps = data.get("components") or []
    if not comps:
        return None
    kinds = {re.match(r"^(\w+)", q).group(1) for q in data.get("query") or [] if re.match(r"^(\w+)", q)}
    types = {str(c.get("type", "")) for c in comps}
    if "tau" in kinds:
        return "transient"
    if types & {"Q", "M"} and kinds & {"Rin", "Rout", "gain"}:
        return "small_signal"
    if kinds & _AC_KINDS or data.get("frequency") or any(c.get("ac") not in (None, "") for c in comps):
        return "ac"
    return "dc"


# ----------------------------------------------------------------------
# Question <-> netlist consistency
# ----------------------------------------------------------------------
_VALUE_WORDS = r"(?:[\d.]+\s*[a-zA-Zµμ]*\s+)?"           # optional "10 Ohm " / "1 uF "
_ELEMENT = r"(resistor|capacitor|inductor|[RCL]\d*)\b(?:\s+([RCL]\d+)\b)?"

# "output (voltage) is taken across (the) (10 Ohm) resistor (R2)", "Vout across C1"
_OUT_ACROSS = re.compile(r"\b(?:output|v_?out|v_?o)\b[^.;]*?\bacross\s+(?:the\s+)?" + _VALUE_WORDS + _ELEMENT,
                         re.IGNORECASE)
# "voltage (drop) across (the) (1 kOhm) resistor (R2)" / "voltage across R2"
#   group 1 = type word, group 2 = id after the word, group 3 = id alone
_VOLT_ACROSS = re.compile(r"\bvoltage(?:\s+drop)?\s+across\s+(?:the\s+)?" + _VALUE_WORDS
                          + r"(?:(resistor|capacitor|inductor)\b(?:\s+([RCL]\d+)\b)?|([RCL]\d+)\b)",
                          re.IGNORECASE)

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


_VALUE_IN_TEXT = re.compile(r"(?<![\w.])\d+(?:\.\d+)?\s*(?:meg|[kmuµnp])?\s*(?:ohms?|Ω|v|a|f|h|hz)\b",
                            re.IGNORECASE)


def consistency_errors(question: str, nl: Netlist) -> List[str]:
    """Check phrases in the question against the netlist. Returns problems as LLM feedback."""
    if not question:
        return []
    if nl.analysis.value == "conceptual":
        # "A 12 V source ... R1 = 4 kOhm ... find" is a circuit problem, not theory
        values = [m.group(0) for m in _VALUE_IN_TEXT.finditer(question)]
        if len(values) >= 2:
            return [f"The question gives circuit values ({', '.join(values[:4])}), so it is a numeric circuit "
                    "problem, not a theory question. Return the full netlist: analysis, components and query."]
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
            word, explicit = m.group(1), (m.group(2) or m.group(3))
            if explicit:
                comp = nl.get(explicit[0].upper() + explicit[1:])
                name = comp.id if comp else ""
            else:                                   # "the resistor": only when there is exactly one
                same = [c for c in nl.components if c.type.value == _WORD_TYPE[word.lower()]]
                comp = same[0] if len(same) == 1 else None
                name = f"the {word.lower()} ({comp.id})" if comp else ""
            if comp is None:
                continue
            a, b = comp.nodes[0], comp.nodes[1]
            if set(comp.nodes[:2]) not in measured:
                fix = f"V({a})" if b == "0" else f"V({b})" if a == "0" else f"V({a},{b})"
                errors.append(f"The question asks for the voltage across {name}, which connects {a} and {b}, "
                              f"but no query measures those two nodes. Use {fix}.")

    errors += _dangling_errors(question, nl)
    errors += _added_element_errors(question, nl)
    errors += _opamp_errors(question, nl)
    return errors


# ----------------------------------------------------------------------
# Op-amp checks (Phase 2)
# ----------------------------------------------------------------------
_PASSIVE = {"R", "C", "L"}
# "inverting amplifier / amp / summer ..." but NOT "non-inverting" and NOT "inverting input"
_INVERTING = re.compile(r"(?<!non-)(?<!non )\binverting\s+(?:op[- ]?amp\s+)?(?:amplifier|amp\b|configuration|summ)",
                        re.IGNORECASE)
_NONINVERTING = re.compile(r"\bnon[- ]?inverting\s+(?:op[- ]?amp\s+)?(?:amplifier|amp\b|configuration)", re.IGNORECASE)
_FOLLOWER = re.compile(r"\b(?:voltage\s+)?follower\b|\bunity[- ]gain\s+buffer\b|\bbuffer\s+amplifier\b", re.IGNORECASE)


def _has_feedback(nl: Netlist, out: str, inm: str) -> bool:
    """Is there a path of R/C/L elements from the output to the − input that does not pass through ground?
    (a single Rf, a T-network, an RC network ... all count)"""
    if out == inm:
        return True
    adj: Dict[str, set] = defaultdict(set)
    for c in nl.components:
        if c.type.value in _PASSIVE:
            a, b = c.nodes
            adj[a].add(b)
            adj[b].add(a)
    seen, stack = {out}, [out]
    while stack:
        n = stack.pop()
        for m in adj[n]:
            if m == inm:
                return True
            if m != "0" and m not in seen:
                seen.add(m)
                stack.append(m)
    return False


def _to_ground_only(nl: Netlist, node: str, skip: str) -> bool:
    """True if `node` is ground, or every element touching it (except op-amp `skip`) goes to ground."""
    if node == "0":
        return True
    touching = [c for c in nl.components if c.id != skip and node in c.nodes]
    return bool(touching) and all(c.type.value in _PASSIVE and set(c.nodes) == {node, "0"} for c in touching)


def _opamp_errors(question: str, nl: Netlist) -> List[str]:
    ops = [c for c in nl.components if c.type.value == "OPAMP"]
    if not ops:
        return []
    errors: List[str] = []

    for u in ops:
        out, inp, inm = u.nodes
        if not _has_feedback(nl, out, inm):
            errors.append(f"{u.id} has no feedback path from its output '{out}' to its − input '{inm}'. A linear "
                          f"op-amp circuit needs negative feedback (e.g. Rf between out and in−). Remember the node "
                          f"order is ['out', 'in+', 'in−'].")

    grounded = [u for u in ops if _to_ground_only(nl, u.nodes[1], u.id)]
    if _INVERTING.search(question) and not grounded:
        u = ops[0]
        errors.append(f"The question describes an inverting amplifier: the op-amp's + input goes to ground and the "
                      f"signal reaches the − input through the input resistor. Here {u.id}'s + input is "
                      f"'{u.nodes[1]}'. Use nodes ['{u.nodes[0]}', '0', '<node of the input resistor>'].")
    if _NONINVERTING.search(question) and len(grounded) == len(ops):
        u = ops[0]
        errors.append(f"The question describes a non-inverting amplifier: the signal goes to the op-amp's + input. "
                      f"Here {u.id}'s + input is grounded. Use nodes ['{u.nodes[0]}', '<input node>', "
                      f"'<node between R1 and Rf>'].")
    if _FOLLOWER.search(question) and not any(u.nodes[0] == u.nodes[2] for u in ops):
        u = ops[0]
        errors.append(f"A voltage follower ties the − input directly to the output. Use nodes "
                      f"['{u.nodes[0]}', '<input node>', '{u.nodes[0]}'] for {u.id}.")

    caps = [c for c in nl.components if c.type.value == "C"]
    if re.search(r"\bintegrator\b", question, re.IGNORECASE) and not any(
            set(c.nodes) == {u.nodes[0], u.nodes[2]} for u in ops for c in caps):
        errors.append("An op-amp integrator has the capacitor in the feedback path, between the output and the "
                      "− input (the resistor goes from the input to the − input).")
    if re.search(r"\bdifferentiator\b", question, re.IGNORECASE) and not any(
            u.nodes[2] in c.nodes and u.nodes[0] not in c.nodes for u in ops for c in caps):
        errors.append("An op-amp differentiator has the capacitor at the input (from the input to the − input) "
                      "and the resistor in the feedback path (between the output and the − input).")
    return errors


def _dangling_errors(question: str, nl: Netlist) -> List[str]:
    """A node touched by only ONE element is an open end: no current can flow through that element.
    Fine for a terminal being measured (Thevenin 'a'), almost always a parse mistake otherwise
    (e.g. 'a source and resistor form a loop' parsed as R1 n1 x with x connected to nothing)."""
    conn: Dict[str, List[str]] = defaultdict(list)
    for c in nl.components:
        # every pin counts, including op-amp inputs and E/G sensing pins: they carry no current, but a
        # source feeding only an op-amp's + input (non-inverting amp, follower) is wired, not a loose end
        for n in c.nodes:
            conn[n].append(c.id)

    queried = set()
    for q in nl.query:
        m = re.match(r"^\w+\((.*)\)$", q)
        if m:
            queried |= {x.strip() for x in m.group(1).split(",")}
    out = _output_node(nl)
    if out:
        queried.add(out)

    errors = []
    for n, ids in conn.items():
        if n == "0" or len(ids) != 1 or n in queried:
            continue
        # the question itself names it as a terminal ("terminals a and b") -> leave it
        if re.search(rf"\b(?:terminal|node|point)s?\s+(?:\w+\s+(?:and|to|-)\s+)?{re.escape(n)}\b", question, re.I):
            continue
        errors.append(f'Node {n} connects only to {ids[0]}, so no current can flow through {ids[0]} (an open end). '
                      f'If {ids[0]} is part of a loop or series path, connect its other end to the rest of the '
                      f'circuit (often ground "0").')
    return errors


_PORT_QUERY = re.compile(r"^(Z|Zth|Rth|Rn|Isc|In|Vth)\((\w+),(\w+)\)$")


def _is_zero(raw: Optional[str]) -> bool:
    if raw in (None, ""):
        return True
    try:
        return parse_value(raw) == 0
    except ValueError:
        return False


def _added_element_errors(question: str, nl: Netlist) -> List[str]:
    """Catch elements the LLM invented to 'measure' something (they change the circuit)."""
    errors: List[str] = []
    q = question.lower()

    # a 0 V voltage source is a short / ammeter the question never describes
    if not re.search(r"\b0\s*v\b|\bzero[- ]volt|\bshort", q):
        for c in nl.components:
            if c.type.value == "V" and _is_zero(c.value) and _is_zero(c.ac):
                errors.append(f"{c.id} is a 0 V source (a short) that the question does not describe. "
                              "Remove it: use Isc(a,b) for a short-circuit current and I(X) for a current; "
                              "the solver adds shorts and test sources itself.")

    # a voltage source straight across the terminals being measured forces Z = 0 / Isc undefined
    for qry in nl.query:
        m = _PORT_QUERY.match(qry)
        if not m or m.group(1) == "Vth":
            continue
        ends = {m.group(2), m.group(3)}
        for c in nl.components:
            if c.type.value == "V" and set(c.nodes[:2]) == ends:
                errors.append(f"{c.id} is connected directly across the terminals of {qry}. With sources turned "
                              f"off it becomes a short, so the answer would be 0. If the question does not "
                              f"describe this source, remove it.")
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
               max_attempts: int = 2, use_fallback: bool = True, hint: Optional[str] = None) -> ParseResult:
    """hint: a correction from the user ("R2 is 2k, not 1k") or from the solver, used on a re-parse."""
    client = client or VelonaClient()
    user = TEXT_PARSER_USER.format(question=question.strip())
    if hint:
        user += (f"\n\nA previous attempt at this circuit was wrong. Correction: {hint.strip()}\n"
                 "Apply this correction.")
    base = [{"role": "system", "content": TEXT_PARSER_SYSTEM},
            {"role": "user", "content": user}]

    plan = [(client.model_for("text_parser"), max_attempts)]
    if use_fallback and client.fallback_parser != plan[0][0]:
        plan.append((client.fallback_parser, 1))

    attempts = 0
    last_errors: List[str] = []
    raw: Dict[str, Any] = {}
    missing: List[str] = []
    model = plan[0][0]
    best: Optional[tuple] = None     # valid netlist that only failed the consistency check
    history: List[Dict[str, Any]] = []

    def failed(stage: str, errs: List[str]) -> None:
        history.append({"attempt": attempts, "model": model, "stage": stage, "errors": list(errs)})

    for model, n_tries in plan:
        messages = list(base)                 # each model starts fresh
        for _ in range(n_tries):
            attempts += 1
            try:
                text = client.chat("text_parser", messages, json_mode=True, model=model)
            except LLMError as e:             # request itself failed
                last_errors = [str(e)]
                failed("request", last_errors)
                continue
            try:
                raw = extract_json(text)
            except LLMError as e:
                last_errors = [str(e)]
                failed("json", last_errors)
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": NOT_JSON_FEEDBACK}]
                continue

            missing = [str(m) for m in (raw.get("missing_info") or [])]
            data = normalize(raw)
            data["question"] = question.strip()

            try:
                nl = Netlist.model_validate(data)
            except ValidationError as e:
                errs, stage = _pydantic_errors(e, data), "schema"
            else:
                report = validate(nl)
                if report.ok:
                    errs, stage = consistency_errors(question, nl), "consistency"
                    if not errs:
                        return ParseResult(True, nl, report, missing, attempts, model, [], raw, history)
                    if nl.analysis.value != "conceptual":    # a "theory" reply to a numeric question is never kept
                        best = (nl, report, errs, list(missing), model, raw)
                else:
                    errs, stage = report.errors, "validator"

            failed(stage, errs)
            last_errors = errs
            messages += [{"role": "assistant", "content": json.dumps(raw)},
                         {"role": "user",
                          "content": RETRY_FEEDBACK.format(errors="\n".join(f"- {x}" for x in errs))}]

    if best is not None:
        nl, report, errs, missing, model, raw = best
        report.warnings += [f"Possible mismatch with the question: {e}" for e in errs]
        return ParseResult(True, nl, report, missing, attempts, model, [], raw, history)
    return ParseResult(False, None, None, missing, attempts, model, last_errors, raw, history)