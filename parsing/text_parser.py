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
from schemas import Netlist, is_symbolic, parse_value
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
    "diode": ("D", None), "zener": ("D", "zener"), "zener_diode": ("D", "zener"), "led": ("D", "led"),
    "bjt": ("Q", None), "transistor": ("Q", None), "npn": ("Q", "NPN"), "pnp": ("Q", "PNP"),
    "mosfet": ("M", None), "nmos": ("M", "NMOS"), "pmos": ("M", "PMOS"), "fet": ("M", None),
    "nfet": ("M", "NMOS"), "pfet": ("M", "PMOS"), "mos": ("M", None),
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
               "zth": "Zth", "isc": "Isc", "rn": "Rn", "in": "In", "bias": "bias",
               "vmax": "Vmax", "vpeak": "Vmax", "vp": "Vmax", "vmin": "Vmin", "vavg": "Vavg", "vdc": "Vavg",
               "vrms": "Vrms", "vtc": "vtc", "imax": "Imax", "ipeak": "Imax", "iavg": "Iavg", "idc": "Iavg",
               "irms": "Irms", "piv": "PIV", "prv": "PIV", "conduction": "conduction",
               "conduction_angle": "conduction", "vripple": "Vripple", "vr": "Vripple", "vpp": "Vpp",
               "ripple": "ripple", "ripple_factor": "ripple", "rf": "ripple",
               "ic": "IC", "ib": "IB", "ie": "IE", "vce": "VCE", "vbe": "VBE", "vec": "VEC", "veb": "VEB",
               "region": "region", "q_point": "bias", "qpoint": "bias", "operating_point": "bias",
               "state": "state",
               # NOTE: "vth" with node arguments stays Thevenin -- Vth(a,b) -- which is what Phase 1 means by
               # it. The Schmitt-trigger thresholds VTH / VTL are BARE queries, in BARE_NAMES below.
               "gm": "gm", "rpi": "rpi", "r_pi": "rpi", "rpie": "rpi", "ro": "ro", "r_o": "ro", "re": "re",
               "r_e": "re", "id": "ID", "idss": "ID", "is": "IS", "ig": "IG", "vgs": "VGS", "vds": "VDS",
               "vsg": "VSG", "vsd": "VSD", "vov": "VOV", "vdsat": "VOV", "vgsq": "VGS"}
BARE_NAMES = {"rin": "Rin", "rout": "Rout", "gain": "gain", "av": "gain", "a_v": "gain",
              "voltage_gain": "gain", "fc": "fc", "f0": "f0", "poles": "poles",
              "zeros": "zeros", "tau": "tau", "qfactor": "Qfactor", "bandwidth": "bandwidth",
              "filter": "filter", "filter_type": "filter", "type": "filter", "response": "filter",
              "vth": "VTH", "vut": "VTH", "vh": "VTH", "upper_threshold": "VTH", "utp": "VTH",
              "vtl": "VTL", "vlt": "VTL", "vl": "VTL", "lower_threshold": "VTL", "ltp": "VTL",
              "hysteresis": "hysteresis", "hysteresis_width": "hysteresis", "vhys": "hysteresis",
              "trip": "trip", "trip_point": "trip", "threshold": "VTH",
              "fosc": "fosc", "f_osc": "fosc", "oscillation_frequency": "fosc",
              "frequency_of_oscillation": "fosc", "f_oscillation": "fosc",
              "barkhausen": "barkhausen", "barkhausen_criterion": "barkhausen",
              "gain_condition": "gain_condition", "condition": "gain_condition",
              "gain_required": "gain_condition", "rf": "gain_condition",
              "oscillates": "oscillates", "will_oscillate": "oscillates",
              "period": "period", "t": "period", "time_period": "period",
              "duty": "duty", "duty_cycle": "duty", "duty_ratio": "duty",
              "t_high": "T_high", "thigh": "T_high", "pulse_width": "T_high",
              "t_on": "T_high", "t_low": "T_low", "tlow": "T_low", "t_off": "T_low"}


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


_BARE_ONLY = {"vtl", "vlt", "vl", "lower_threshold", "ltp", "vut", "utp", "upper_threshold",
              "hysteresis", "hysteresis_width", "vhys", "trip", "trip_point", "threshold"}


def _norm_query(q: Any) -> str:
    s = str(q).replace(" ", "")
    m = re.match(r"^(\w+)\((.*)\)$", s)
    if m:
        low = m.group(1).lower()
        if low in _BARE_ONLY:                      # "VTL(U1)" / "hysteresis(U1)": these take no arguments
            return BARE_NAMES[low]
        name = QUERY_NAMES.get(low, m.group(1))
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
_AC_KINDS = {"fc", "f0", "poles", "zeros", "Qfactor", "bandwidth", "H", "gain", "filter"}


def infer_analysis(data: Dict[str, Any]) -> Optional[str]:
    """Analysis type from the rest of the netlist, used only when the LLM omitted it.
    No components -> None (NOT "conceptual"): an empty or oddly shaped reply must be retried,
    never silently turned into a theory question."""
    comps = data.get("components") or []
    if not comps:
        return None
    kinds = {re.match(r"^(\w+)", q).group(1) for q in data.get("query") or [] if re.match(r"^(\w+)", q)}
    types = {str(c.get("type", "")) for c in comps}
    if "tau" in kinds or kinds & {"Vmax", "Vmin", "Vavg", "Vrms", "vtc", "Imax", "Iavg", "Irms", "PIV", "conduction",
                                  "Vripple", "Vpp", "ripple"}:
        return "transient"
    if "D" in types and any(c.get("ac") not in (None, "") for c in comps):
        return "transient"                      # a diode with a sinusoidal input: rectifier / clipper
    if types & {"Q", "M"} and (kinds & {"Rin", "Rout", "gain", "H", "gm", "rpi", "re", "ro"}
                               or (("C" in types) and any(c.get("ac") not in (None, "") for c in comps))):
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
    errors += _comparator_errors(question, nl)
    errors += _oscillator_errors(question, nl)
    errors += _filter_errors(question, nl)
    errors += _loop_errors(question, nl)
    errors += _series_source_errors(question, nl)
    errors += _diode_errors(question, nl)
    errors += _bjt_errors(question, nl)
    errors += _mos_errors(question, nl)
    errors += _value_errors(nl)
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
    switching = _is_switching(question, nl)

    for u in ops:
        out, inp, inm = u.nodes
        if not switching and not _has_feedback(nl, out, inm):
            errors.append(f"{u.id} has no feedback path from its output '{out}' to its − input '{inm}'. A linear "
                          f"op-amp circuit needs negative feedback (e.g. Rf between out and in−). Remember the node "
                          f"order is ['out', 'in+', 'in−'].")

    grounded = [u for u in ops if _to_ground_only(nl, u.nodes[1], u.id)]
    if _INVERTING.search(question) and not switching and not grounded:
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


# ----------------------------------------------------------------------
# Comparator / Schmitt-trigger checks (Phase 3, step 6)
# ----------------------------------------------------------------------
_SCHMITT = re.compile(r"\bschmitt\b|\bhyster|\bsnap[- ]?action\b|\bregenerative\s+comparator\b", re.IGNORECASE)
_COMPARATOR = re.compile(r"\bcomparator\b|\bzero[- ]crossing\s+detector\b|\blevel\s+detector\b|"
                         r"\bwindow\s+detector\b", re.IGNORECASE)
_THRESHOLD_Q = {"VTH", "VTL", "hysteresis", "trip"}


def _is_switching(question: str, nl: Netlist) -> bool:
    """True when the question is about a comparator / Schmitt trigger, which has no negative feedback."""
    if any(q in _THRESHOLD_Q or q.startswith("state(") for q in nl.query):
        return True
    return bool(_SCHMITT.search(question) or _COMPARATOR.search(question))


def _comparator_errors(question: str, nl: Netlist) -> List[str]:
    ops = [c for c in nl.components if c.type.value == "OPAMP"]
    if not ops or not _is_switching(question, nl):
        return []
    errors: List[str] = []
    schmitt = bool(_SCHMITT.search(question)) or any(q in ("VTH", "VTL", "hysteresis", "trip") for q in nl.query)

    for u in ops:
        out, inp, inm = u.nodes
        fb_plus = _has_feedback(nl, out, inp)
        fb_minus = _has_feedback(nl, out, inm)
        if schmitt and not fb_plus:
            if fb_minus:
                errors.append(f"{u.id}'s feedback resistor goes to its − input '{inm}', which is NEGATIVE feedback "
                              f"(an ordinary amplifier). A Schmitt trigger needs POSITIVE feedback: put the "
                              f"feedback resistor between the output '{out}' and the + input, and remember the "
                              f"node order is ['out', 'in+', 'in−'].")
            else:
                errors.append(f"The question describes a Schmitt trigger (hysteresis), but {u.id} has no feedback "
                              f"path from its output '{out}' back to its + input '{inp}'. Add the positive-feedback "
                              f"resistor between the output and the + input.")
        elif schmitt and fb_minus:
            errors.append(f"{u.id} has feedback to BOTH inputs. A Schmitt trigger has positive feedback only: "
                          f"remove the path from '{out}' to the − input '{inm}'.")

    if _SCHMITT.search(question) and nl.analysis.value not in ("dc",):
        errors.append('Threshold / hysteresis questions use analysis "dc": the trip points do not depend on the '
                      'input waveform. Put the input source in with value "0".')

    if any(q in _THRESHOLD_Q for q in nl.query) and not any(
            c.type.value in ("V", "I") and c.id.upper() not in ("VREF", "VCC", "VEE", "VDD", "VSS")
            for c in nl.components):
        errors.append("A threshold question needs the input source in the netlist (the trip point is the value of "
                      'that source), e.g. {"id": "V1", "type": "V", "nodes": ["in", "0"], "value": "0"}.')
    return errors


# ----------------------------------------------------------------------
# "in series with" checks (Phase 1)
# ----------------------------------------------------------------------
_IN_SERIES = re.compile(r"\bin\s+series\s+with\b|\bthrough\s+(?:a\s+)?\w*\s*resistor\b", re.IGNORECASE)
_LOOP_WORD = re.compile(r"\bloop\b|\bmesh\b", re.IGNORECASE)


def _series_source_errors(question: str, nl: Netlist) -> List[str]:
    """A resistor wired directly ACROSS an ideal voltage source does nothing: the source fixes the voltage,
    so the resistor carries its own current and changes no other node. When the question says the source is
    "in series with" that resistor, the resistor belongs BETWEEN the source terminal and the next node, and
    putting it in parallel silently changes the answer (the rest of the circuit then sees the full source
    voltage instead of a divided one).

    The exception is a question that says the source and the resistor "form a loop" -- there the parallel
    wiring is right, and _loop_errors already checks it -- so this test stands down when "loop" is mentioned.
    """
    if not _IN_SERIES.search(question) or _LOOP_WORD.search(question):
        return []
    errors: List[str] = []
    for src in [c for c in nl.components if c.type.value == "V"]:
        pair = set(src.nodes[:2])
        for r in [c for c in nl.components if c.type.value == "R" and set(c.nodes[:2]) == pair]:
            other = [c for c in nl.components
                     if c.id not in (src.id, r.id) and set(c.nodes[:2]) & (pair - {"0"})]
            if not other:
                continue                      # nothing else is fed from that node: harmless
            errors.append(
                f"{r.id} is wired directly across {src.id} (both on nodes {sorted(pair)}), but the question "
                f"says they are in series. In parallel, {src.id} fixes that node's voltage and {r.id} has no "
                f"effect on the rest of the circuit. Put {r.id} BETWEEN the source and the node it feeds: "
                f"{src.id} on ['n1', '0'] and {r.id} on ['n1', '<the node it feeds>'].")
    return errors


# ----------------------------------------------------------------------
# Oscillator checks (Phase 4, step 1)
# ----------------------------------------------------------------------
_OSC_Q = {"fosc", "barkhausen", "gain_condition", "oscillates", "period", "duty",
          "T_high", "T_low"}
_PHASE_SHIFT = re.compile(r"\bphase[- ]shift\s+oscillator\b|\bRC\s+phase[- ]shift\b", re.IGNORECASE)


def _oscillator_errors(question: str, nl: Netlist) -> List[str]:
    ops = [c for c in nl.components if c.type.value == "OPAMP"]
    srcs = [c for c in nl.components if c.type.value in ("V", "I")]
    amps = ops + [c for c in nl.components if c.type.value in ("Q", "M", "E", "G", "F", "H")]
    if not amps or not (set(nl.query) & _OSC_Q):
        return []
    errors: List[str] = []
    if srcs:
        errors.append("An oscillator has no input: remove "
                      + ", ".join(c.id for c in srcs)
                      + ". The oscillation comes from the feedback loop itself, and the solver finds it from "
                        "det A(s) = 0, which needs the circuit to be source-free.")

    # an LC oscillator is given as the AC equivalent circuit: the transistor needs gm, not a bias
    for c in nl.components:
        if c.type.value in ("Q", "M"):
            p = {str(k).lower(): v for k, v in (c.params or {}).items()}
            if not any(p.get(k) for k in ("gm", "g_m", "ic", "i_c", "id", "i_d")):
                errors.append(f"{c.id} has no gm: an oscillator netlist is the AC equivalent circuit (no "
                              f"supply), so the transistor is used through its small-signal model. Put gm in "
                              f'its params, e.g. "params": {{"gm": 0.001}}, or I_C if the question gives the '
                              f"bias current instead.")
            if len(c.nodes) >= 3 and c.nodes[2] != "0":
                errors.append(f"{c.id}: in the AC equivalent circuit of a common-emitter LC oscillator the "
                              f"emitter is at AC ground, so its third node should be '0', not "
                              f"'{c.nodes[2]}'.")
    if not ops:
        return errors

    op = ops[0]
    out, inp, inm = op.nodes[:3]
    rs = [c for c in nl.components if c.type.value == "R"]
    cs = [c for c in nl.components if c.type.value == "C"]
    feedback = [c for c in rs if set(c.nodes[:2]) == {out, inm}]
    if not feedback:
        errors.append(f"{op.id} has no feedback resistor between its output '{out}' and its − input '{inm}'. "
                      "Every op-amp oscillator needs one: it sets the gain that makes up for the loss in the "
                      "RC network.")

    # a resistor from the − input to ground does nothing when the + input is grounded: the − input is a
    # virtual ground, so both of its ends are at 0 V
    if inp == "0":
        dead = [c for c in rs if set(c.nodes[:2]) == {inm, "0"}]
        if dead:
            errors.append(f"{dead[0].id} connects the − input '{inm}' to ground, but with the + input grounded "
                          f"the − input is a VIRTUAL ground: both of its ends sit at 0 V, so it carries no "
                          f"signal and does nothing. In an RC phase-shift oscillator the last section's "
                          f"resistor goes from the previous node TO the − input, not from the − input to "
                          f"ground.")

    # every section of a phase-shift ladder is one C and one R: they must come in pairs
    if _PHASE_SHIFT.search(question) or len(cs) >= 3:
        ladder_r = [c for c in rs if c not in feedback and not (inp == "0" and set(c.nodes[:2]) == {inm, "0"})]
        if len(cs) >= 3 and len(ladder_r) < len(cs):
            errors.append(f"The ladder has {len(cs)} capacitors but only {len(ladder_r)} resistors beside the "
                          f"feedback resistor. Each RC section is one capacitor AND one resistor, so write "
                          f"three complete sections: C1 ['out','a'], R1 ['a','0'], C2 ['a','b'], "
                          f"R2 ['b','0'], C3 ['b','c'], R3 ['c','{inm}'] -- the last resistor ends at the "
                          f"− input.")
        elif len(cs) >= 3 and not any(inm in c.nodes[:2] for c in ladder_r):
            errors.append(f"The last section of the ladder must reach the − input '{inm}' through its "
                          f"RESISTOR (R3 ['c','{inm}']), with its capacitor before it (C3 ['b','c']).")
    return errors


# ----------------------------------------------------------------------
# Filter checks (Phase 2, step 4)
# ----------------------------------------------------------------------
_SALLEN = re.compile(r"sallen[\s-]*key", re.IGNORECASE)
_MFB = re.compile(r"multiple[\s-]*feedback|\bmfb\b|deliyannis|\brauch\b", re.IGNORECASE)
_TYPE_WORDS = [("low-pass", re.compile(r"\blow[\s-]?pass\b", re.IGNORECASE)),
               ("high-pass", re.compile(r"\bhigh[\s-]?pass\b", re.IGNORECASE)),
               ("band-pass", re.compile(r"\bband[\s-]?pass\b", re.IGNORECASE)),
               ("notch", re.compile(r"\bnotch\b|\bband[\s-]?(?:stop|reject)\b", re.IGNORECASE))]
_TYPE_HINT = {
    "low-pass": "a low-pass has a CONSTANT numerator: series resistors (or inductors) carry the signal and "
                "capacitors go to ground or back to the output",
    "high-pass": "a high-pass has s (or s²) alone in the numerator: series capacitors carry the signal and "
                 "resistors go to ground or back to the output",
    "band-pass": "a band-pass has s alone in the numerator: it blocks both DC and high frequencies",
    "notch": "a notch has (s² + ω0²) in the numerator: it blocks one frequency and passes DC and high frequencies",
}


def _out_feedback(nl: Netlist, u) -> List[str]:
    """Nodes (other than ground and the op-amp's own input pins) joined to the output by one passive element."""
    out, inp, inm = u.nodes
    nodes = []
    for c in nl.components:
        if c.type.value in _PASSIVE and out in c.nodes:
            other = c.nodes[1] if c.nodes[0] == out else c.nodes[0]
            if other not in ("0", inp, inm, out):
                nodes.append(other)
    return nodes


def _parsed_response(nl: Netlist):
    """Filter type of the parsed netlist (same solver code as the answer), or None if it cannot be found."""
    try:
        from llm_client import load_conventions
        from solvers.dc_ac import _Job
        from solvers.filters import analyse

        job = _Job(nl, load_conventions())
        out, inp = job._default_transfer()
        H, _ = job._transfer(out, inp)
        return analyse(H)
    except Exception:          # any problem here is not the parser's fault: no feedback, no blocking
        return None


def _filter_errors(question: str, nl: Netlist) -> List[str]:
    errors: List[str] = []
    ops = [c for c in nl.components if c.type.value == "OPAMP"]
    if _SALLEN.search(question):
        if not ops:
            errors.append("A Sallen-Key filter is built around an op-amp (often a unity-gain follower); add it "
                          "with nodes ['out', 'p', 'out'] or ['out', 'p', 'n1'] with gain resistors.")
        elif all(_to_ground_only(nl, u.nodes[1], u.id) for u in ops):
            errors.append("In a Sallen-Key filter the signal reaches the op-amp's + input through the RC network "
                          "(the + input is NOT grounded). Use the Sallen-Key template.")
        elif not any(_out_feedback(nl, u) for u in ops):
            errors.append("A Sallen-Key filter has a positive-feedback element from the junction 'a' of the first "
                          "two input elements to the OUTPUT (C1 in the low-pass, R1 in the high-pass). Here no "
                          "element joins the output to such a node.")
    if _MFB.search(question):
        if not ops:
            errors.append("A multiple-feedback (MFB) filter is built around an op-amp; add it with nodes "
                          "['out', '0', 'n1'].")
        elif not any(_to_ground_only(nl, u.nodes[1], u.id) for u in ops):
            errors.append("In a multiple-feedback (MFB) filter the op-amp's + input is grounded and the signal "
                          "reaches the − input. Use the MFB template.")
        elif not any(_out_feedback(nl, u) and _has_feedback(nl, u.nodes[0], u.nodes[2]) for u in ops):
            errors.append("A multiple-feedback (MFB) filter has TWO feedback paths from the output: one element to "
                          "the − input and one to node 'a' (the junction after the input element).")

    # the response the question names must be the response of the parsed circuit
    named = [kind for kind, pat in _TYPE_WORDS if pat.search(question)]
    if len(named) == 1 and not errors and (ops or any(c.type.value in ("C", "L") for c in nl.components)):
        resp = _parsed_response(nl)
        if resp is not None and resp.kind not in ("other", named[0]):
            errors.append(f"The question describes a {named[0]} filter, but the circuit you built has a "
                          f"{resp.kind} response. Remember: {_TYPE_HINT[named[0]]}. Re-check which elements are "
                          f"in series with the signal and which go to ground or to the output.")
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
    errors += _dead_end_errors(question, nl, conn)
    return errors


def _dead_end_errors(question: str, nl: Netlist, conn: Dict[str, List[str]]) -> List[str]:
    """A node whose elements ALL go to one and the same other node is a dead end too: e.g. "R1 in series
    with the parallel combination of R2 and R3" parsed as R2, R3 both n1–out with "out" connected to nothing
    else. No current flows through R2 and R3, so every voltage across them is 0. Only Thevenin / impedance
    terminals are exempt (there the open end is the point of the question)."""
    by_id = {c.id: c for c in nl.components}
    terminals = set()
    for q in nl.query:
        m = _PORT_QUERY.match(q.replace(" ", ""))
        if m:
            terminals |= {m.group(2), m.group(3)}
    errors = []
    for n, ids in conn.items():
        if n == "0" or len(ids) < 2 or n in terminals:
            continue
        comps = [by_id[i] for i in ids]
        if not all(c.type.value in _PASSIVE for c in comps):
            continue
        others = {c.nodes[1] if c.nodes[0] == n else c.nodes[0] for c in comps}
        if len(others) != 1:
            continue
        if re.search(rf"\b(?:terminal|point)s?\s+(?:\w+\s+(?:and|to|-)\s+)?{re.escape(n)}\b", question, re.I):
            continue
        m = others.pop()
        names = ", ".join(ids)
        errors.append(f'Node {n} is reached only through {names}, which all connect it to node {m} and to nothing '
                      f'else, so no current flows through them (the voltage across them would always be 0). '
                      f'Elements "in parallel" share BOTH nodes with the rest of the circuit: e.g. a parallel '
                      f'combination in series after R1 goes from R1\'s far node to ground, ["{m}", "0"].')
    return errors


# "A 6 V source in series with a 2 kOhm resistor forms a loop": those two parts close a loop on their own
_LOOP = re.compile(r"\bforms?\s+(?:a\s+)?(?:closed\s+)?loop\b", re.IGNORECASE)
_V_IN_TEXT = re.compile(r"(\d+(?:\.\d+)?)\s*V\b")
_R_IN_TEXT = re.compile(r"(\d+(?:\.\d+)?)\s*(k|K|M|Meg|m)?\s*(?:Ohm|ohm|Ω)")


def _loop_errors(question: str, nl: Netlist) -> List[str]:
    errors = []
    for sentence in re.split(r"(?<=[.!?;])\s+", question):
        if not _LOOP.search(sentence):
            continue
        vs = {float(x) for x in _V_IN_TEXT.findall(sentence)}
        rs = {parse_value(a + (b or "")) for a, b in _R_IN_TEXT.findall(sentence)}
        if len(vs) != 1 or len(rs) != 1:
            continue
        v, r = vs.pop(), rs.pop()
        vc = [c for c in nl.components if c.type.value == "V" and not _is_zero(c.value) and _num_eq(c.value, v)]
        rc = [c for c in nl.components if c.type.value == "R" and _num_eq(c.value, r)]
        if len(vc) == 1 and len(rc) == 1 and set(rc[0].nodes) != set(vc[0].nodes):
            errors.append(f"The question says the {v:g} V source and the resistor {rc[0].id} form a loop on their "
                          f"own: {rc[0].id} must connect the source's two terminals, nodes {vc[0].nodes}. Here it "
                          f"is {rc[0].nodes}. A current I in that loop that controls a dependent source does NOT "
                          f"mean the loop is wired to the dependent source's node; that is a separate part.")
    return errors


# ----------------------------------------------------------------------
# Diode checks (Phase 3)
# ----------------------------------------------------------------------
_ZENER = re.compile(r"\bzener\b", re.IGNORECASE)
_IDEAL_DIODE = re.compile(r"\bideal\s+(?:silicon\s+)?diodes?\b|\bdiodes?\s+(?:is|are)\s+ideal\b", re.IGNORECASE)
_FORWARD = re.compile(r"\bforward", re.IGNORECASE)


def _is_zener(c) -> bool:
    return "zener" in (c.model or "").lower() or any(str(k).lower() in ("vz", "bv", "vbr", "v_z")
                                                       for k in (c.params or {}))


def _diode_errors(question: str, nl: Netlist) -> List[str]:
    ds = [c for c in nl.components if c.type.value == "D"]
    if not ds:
        return []
    errors: List[str] = []
    if _ZENER.search(question):
        zs = [c for c in ds if _is_zener(c)]
        if not zs:
            errors.append('The question has a Zener diode: give it "model": "zener" and its Zener voltage in '
                          'params, e.g. {"VZ": 5.1}.')
        elif not _FORWARD.search(question):
            for z in zs:
                if z.nodes[1] == "0" and z.nodes[0] != "0":
                    errors.append(f"{z.id} is a Zener working in breakdown, so it is REVERSE-connected: cathode on "
                                  f"the regulated (+) node and anode at ground -> nodes ['0', '{z.nodes[0]}']. "
                                  f"Remember the order is ['anode', 'cathode'].")
    if _IDEAL_DIODE.search(question) and not all((c.model or "").lower() == "ideal" for c in ds if not _is_zener(c)):
        errors.append('The question says the diodes are ideal: set "model": "ideal" on each of them.')
    errors += _rectifier_errors(question, nl, ds)
    return errors


_RMS_IN_TEXT = re.compile(r"(\d+(?:\.\d+)?)\s*V\s*(?:\(\s*)?(?:rms|RMS|r\.m\.s\.)", re.IGNORECASE)
_PP_IN_TEXT = re.compile(r"(\d+(?:\.\d+)?)\s*V\s*(?:peak[- ]to[- ]peak|p-p|pp)\b", re.IGNORECASE)
_BRIDGE = re.compile(r"\bbridge\b", re.IGNORECASE)
_CTAP = re.compile(r"cent(?:er|re)[- ]?tap", re.IGNORECASE)


def _rectifier_errors(question: str, nl: Netlist, ds) -> List[str]:
    if nl.analysis.value != "transient":
        return []
    errors: List[str] = []
    waves = [c for c in nl.components if c.type.value in ("V", "I") and (
        c.ac not in (None, "") or {str(k).lower() for k in (c.params or {})} & {"vrms", "rms", "vpp", "vp", "vpeak"})]
    # an RMS or peak-to-peak value put straight into "ac" (which means PEAK) gives a wrong amplitude
    for pat, key in ((_RMS_IN_TEXT, "Vrms"), (_PP_IN_TEXT, "Vpp")):
        for mtxt in pat.finditer(question):
            x = float(mtxt.group(1))
            for c in waves:
                if c.ac not in (None, "") and _num_eq(c.ac, x):
                    errors.append(f'{x:g} V in the question is a{"n RMS" if key == "Vrms" else " peak-to-peak"} '
                                  f'value, but {c.id} has "ac": "{c.ac}" (which means the PEAK). Use params '
                                  f'{{"{key}": {x:g}}} instead of "ac".')
    if _BRIDGE.search(question) and len(ds) != 4:
        errors.append(f"A bridge rectifier has FOUR diodes (D1 a→out, D2 b→out, D3 0→a, D4 0→b with the source "
                      f"between a and b); the netlist has {len(ds)}.")
    if _CTAP.search(question) and len(waves) != 2:
        errors.append("A centre-tapped secondary is TWO equal sinusoidal halves with the centre tap at ground "
                      "(V1 ['a','0'] and V2 ['0','b'], each with half the total voltage).")
    return errors


# ----------------------------------------------------------------------
# MOSFET checks (Phase 3, step 5)
# ----------------------------------------------------------------------
_PMOS = re.compile(r"\bp-?\s?mos\b|\bp-?channel\b|\bpmosfet\b", re.IGNORECASE)
_NMOS = re.compile(r"\bn-?\s?mos\b|\bn-?channel\b|\bnmosfet\b", re.IGNORECASE)
_VT_IN_TEXT = re.compile(r"(?:V_?t(?:h|n|p|0)?|threshold(?:\s+voltage)?)\s*(?:=|is|of)?\s*"
                         r"[-+]?(\d+(?:\.\d+)?)\s*V", re.IGNORECASE)
_K_PARAMS = ("k", "kn", "kp", "beta", "k2", "kprime", "k_prime", "mucox", "mu_cox", "kn_prime", "kp_prime")
_VT_PARAMS = ("vt", "vth", "vto", "vtn", "vtp", "v_t", "v_th", "threshold")


def _mos_errors(question: str, nl: Netlist) -> List[str]:
    ms = [c for c in nl.components if c.type.value == "M"]
    if not ms:
        return []
    errors: List[str] = []
    models = {(c.model or "").strip().upper() for c in ms}
    if _PMOS.search(question) and not _NMOS.search(question) and "PMOS" not in models:
        errors.append('The question has a PMOS (p-channel) transistor: set "model": "PMOS" (its source goes '
                      'toward the positive supply; nodes are still ["drain", "gate", "source"]).')
    if _NMOS.search(question) and not _PMOS.search(question) and models - {"NMOS", ""}:
        errors.append('The question has an NMOS (n-channel) transistor: set "model": "NMOS".')
    for c in ms:
        p = {str(k).strip().lower().replace(" ", "_").replace("-", "_") for k, v in (c.params or {}).items()
             if v not in (None, "")}
        if not p & set(_K_PARAMS):
            errors.append(f'{c.id} has no conduction parameter: add params {{"k": ...}} in A/V² (I_D = '
                          f'½·k·V_OV²), or "kprime" with "WL", or "K2" when the question writes '
                          f'I_D = K·(V_GS − V_t)² without the ½.')
        if not p & set(_VT_PARAMS):
            errors.append(f'{c.id} has no threshold voltage: add params {{"Vt": ...}} (a magnitude; the sign '
                          f'comes from the NMOS / PMOS model).')
    vts = {float(m.group(1)) for m in _VT_IN_TEXT.finditer(question)}
    if len(vts) == 1:
        want = vts.pop()
        for c in ms:
            p = {str(k).strip().lower().replace(" ", "_"): v for k, v in (c.params or {}).items()}
            got = next((p[k] for k in _VT_PARAMS if k in p), None)
            if got is not None and not _num_eq(str(got), want) and not _num_eq(str(got), -want):
                errors.append(f"The question gives V_t = {want:g} V: set params {{\"Vt\": {want:g}}} on {c.id}.")
    return errors


def _value_errors(nl: Netlist) -> List[str]:
    """A component whose value is a NAME in an otherwise numeric circuit: usually an invented placeholder
    ("bypass", "large", "C") where the question gave no number. A fully symbolic circuit is fine."""
    vals = [(c, c.value) for c in nl.components if c.type.value in ("R", "C", "L") and c.value not in (None, "")]
    numeric = [c for c, v in vals if not is_symbolic(v)]
    named = [(c, v) for c, v in vals if is_symbolic(v)]
    if not numeric or not named:
        return []
    # A design question leaves exactly one value as a symbol ON PURPOSE: "what must Rf be for it to
    # oscillate" is answered by solving for that symbol, so it is not an invented placeholder.
    if len(named) == 1 and any(q in ("gain_condition", "fosc", "barkhausen", "oscillates") for q in nl.query):
        return []
    out = []
    for c, v in named:
        unit = {"R": '"1k"', "C": '"10u"', "L": '"1m"'}[c.type.value]
        out.append(f'{c.id} has "{v}" as its value, which is not a number, while the rest of the circuit has '
                   f'numbers. Put a number there (e.g. {unit}); if the question gives none, choose a typical '
                   f'value and say so in "assumptions".')
    return out


# ----------------------------------------------------------------------
# BJT checks (Phase 3)
# ----------------------------------------------------------------------
_BETA_IN_TEXT = re.compile(r"(?:β|\bbeta\b|\bh_?fe\b|\bhFE\b|\bβdc\b|\bbeta_?dc\b)\s*(?:=|of|is|:)?\s*(\d+(?:\.\d+)?)",
                           re.IGNORECASE)
_EMITTER_GROUNDED = re.compile(r"emitter\s+(?:is\s+|terminal\s+is\s+)?(?:grounded|connected\s+(?:directly\s+)?to\s+ground|"
                               r"at\s+ground|tied\s+to\s+ground)", re.IGNORECASE)


def _bjt_errors(question: str, nl: Netlist) -> List[str]:
    qs = [c for c in nl.components if c.type.value == "Q"]
    if not qs:
        return []
    errors: List[str] = []
    models = {(c.model or "").strip().upper() for c in qs}
    if re.search(r"\bpnp\b", question, re.IGNORECASE) and not re.search(r"\bnpn\b", question, re.IGNORECASE) \
            and "PNP" not in models:
        errors.append('The question has a PNP transistor: set "model": "PNP" (its emitter goes toward the positive '
                      'supply; nodes are still ["collector", "base", "emitter"]).')
    if re.search(r"\bnpn\b", question, re.IGNORECASE) and not re.search(r"\bpnp\b", question, re.IGNORECASE) \
            and models - {"NPN", ""}:
        errors.append('The question has an NPN transistor: set "model": "NPN".')
    betas = {float(m.group(1)) for m in _BETA_IN_TEXT.finditer(question)}
    if len(betas) == 1 and len(qs) >= 1:
        want = betas.pop()
        for c in qs:
            p = {str(k).lower(): v for k, v in (c.params or {}).items()}
            got = next((p[k] for k in ("beta", "β", "hfe", "bf") if k in p), None)
            if got is None or not _num_eq(str(got), want):
                errors.append(f"The question gives β = {want:g}: set params {{\"beta\": {want:g}}} on {c.id}.")
    if nl.analysis.value == "small_signal":
        if not any(c.type.value == "C" for c in nl.components) and re.search(
                r"\bcoupl|\bbypass|\bcapacitor", question, re.IGNORECASE):
            errors.append("The question has coupling / bypass capacitors: keep them in the netlist (the solver "
                          "opens them for the DC bias and shorts them for the signal).")
        if not any(c.type.value in ("V", "I") and c.ac not in (None, "") for c in nl.components):
            errors.append('A small-signal amplifier needs a signal source with an AC magnitude: add '
                          '{"id": "V1", "type": "V", "nodes": ["in", "0"], "ac": "1"} where the signal enters.')
    if _EMITTER_GROUNDED.search(question) and len(qs) == 1 and qs[0].nodes[2] != "0":
        errors.append(f"The question says the emitter is grounded: {qs[0].id} nodes are [collector, base, emitter], "
                      f"so the third node must be \"0\" (it is \"{qs[0].nodes[2]}\").")
    return errors


def _num_eq(raw: Optional[str], x: float) -> bool:
    try:
        return abs(parse_value(raw) - x) <= 1e-9 * max(abs(x), 1e-30)
    except (ValueError, TypeError):
        return False


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
    # (a threshold question is the exception: its input source is a placeholder the solver replaces by a symbol)
    threshold = any(x in _THRESHOLD_Q for x in nl.query)
    if not threshold and not re.search(r"\b0\s*v\b|\bzero[- ]volt|\bshort", q):
        for c in nl.components:
            # (a sinusoid given only as params {"Vrms": ..} / {"Vpp": ..} has no value and no ac: not a short)
            wave = {str(k).strip().lower() for k in (c.params or {})} & {
                "vrms", "rms", "vpp", "vp", "vpeak", "vm", "peak"}
            if c.type.value == "V" and _is_zero(c.value) and _is_zero(c.ac) and not wave:
                errors.append(f"{c.id} is a 0 V source (a short) that the question does not describe. "
                              "Remove it: use Isc(a,b) for a short-circuit current and I(X) for a current; "
                              "the solver adds shorts and test sources itself.")

    sources = [c for c in nl.components if c.type.value in ("V", "I")]
    needs_input = [x for x in nl.query if re.match(r"^(H\(|gain$|Rin$|fc$|bandwidth$|filter$)", x)]

    # a voltage source straight across the terminals being measured forces Z = 0 / Isc undefined
    for qry in nl.query:
        m = _PORT_QUERY.match(qry)
        if not m or m.group(1) == "Vth":
            continue
        ends = {m.group(2), m.group(3)}
        for c in nl.components:
            if c.type.value == "V" and set(c.nodes[:2]) == ends:
                if needs_input and len(sources) == 1:
                    # the source IS needed (as the input of the gain); the port query is the wrong one
                    errors.append(f"Keep {c.id}: it is the input for {', '.join(needs_input)}. For the input "
                                  f"resistance use the query \"Rin\" instead of {qry} (Rin removes the input source "
                                  f"by itself; {qry} would short it and give 0).")
                else:
                    errors.append(f"{c.id} is connected directly across the terminals of {qry}. With sources "
                                  f"turned off it becomes a short, so the answer would be 0. If the question does "
                                  f"not describe this source, remove it.")

    # a gain / transfer-function / input-resistance question needs an input source to work with
    if needs_input and not sources:
        errors.append(f"{', '.join(needs_input)} needs an input source, but the netlist has none. Add one: "
                      f'{{"id": "V1", "type": "V", "nodes": ["in", "0"], "value": "1"}} (or "ac": "1" for a '
                      f"frequency response), at the node where the signal enters.")
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