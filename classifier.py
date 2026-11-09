"""
Question classifier: decides the TOPIC (used by the explainer and the benchmark)
and which SOLVER will run.

Rules first (free, instant, same answer every time), in this priority:
  1. analysis == conceptual                                          -> conceptual
  2. oscillator words (oscillat, Barkhausen, Wien, Colpitts, Hartley) -> oscillator
  3. filter words (filter, cutoff, -3 dB, Bode, low/high/band-pass)
     or filter queries (fc, poles, zeros, Qfactor, bandwidth)        -> filter
  4. a device in the netlist: OPAMP -> opamp, Q -> bjt, M -> mosfet, D -> diode
  5. otherwise the analysis type: dc / ac / transient / small_signal

Then it checks the question text against the netlist. Disagreements lower the
confidence to "low" and are listed in `notes`, e.g.
  "transistor amplifier" in the question but no Q/M in the netlist -> parse probably dropped it
  "filter" in the question but no C or L in the circuit
Only when confidence is low AND a client is passed does GPT-5 Nano
(classifier role) pick the topic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from llm_client import LLMError, VelonaClient
from schemas import AnalysisType, Netlist

TOPICS = ["dc", "ac", "transient", "small_signal", "filter", "oscillator",
          "opamp", "bjt", "mosfet", "diode", "conceptual"]
DEVICE_TOPIC = {"OPAMP": "opamp", "Q": "bjt", "M": "mosfet", "D": "diode"}
DEVICE_LABEL = {"opamp": "an op-amp", "bjt": "a BJT", "mosfet": "a MOSFET", "diode": "a diode"}
FILTER_QUERIES = {"fc", "poles", "zeros", "Qfactor", "bandwidth", "filter"}

WORDS = {
    "oscillator": ["oscillat", "barkhausen", "wien", "colpitts", "hartley", "crystal oscillator",
                   "frequency of oscillation"],
    "filter": ["filter", "cutoff", "cut-off", "corner frequency", "-3 db", "−3 db", "3 db", "3db", "bode",
               "low-pass", "low pass", "lowpass", "high-pass", "high pass", "highpass",
               "band-pass", "band pass", "bandpass", "notch", "band-stop", "bandstop",
               "sallen", "multiple feedback", "multiple-feedback", "mfb", "butterworth", "chebyshev", "bessel"],
    "opamp": ["op-amp", "op amp", "opamp", "operational amplifier", "741", "non-inverting",
              "inverting amplifier", "voltage follower", "unity-gain buffer"],
    "bjt": ["bjt", "npn", "pnp", "common emitter", "common-emitter", "common collector",
            "emitter follower", "common base", "common-base"],
    "mosfet": ["mosfet", "nmos", "pmos", "common source", "common-source", "source follower", "fet"],
    "diode": ["diode", "zener", "rectifier", "clipper", "clamper"],
}

CLASSIFIER_SYSTEM = (
    "You label analog electronics questions with ONE topic from this list: " + ", ".join(TOPICS) + ".\n"
    "Pick the topic that best describes what the question is mainly about.\n"
    'Reply with JSON only: {"topic": "<one topic>", "reason": "<one short sentence>"}'
)


@dataclass
class Classification:
    topic: str
    tags: List[str] = field(default_factory=list)
    confidence: str = "high"            # "high" | "low"
    notes: List[str] = field(default_factory=list)
    source: str = "rules"               # "rules" | "llm"
    solver: Optional[str] = None        # solver name, or None if not supported yet
    unsupported_reason: str = ""


def _mentions(text: str, phrase: str) -> bool:
    return re.search(r"\b" + re.escape(phrase), text) is not None if phrase[0].isalnum() else phrase in text


def classify(nl: Netlist, client: Optional[VelonaClient] = None) -> Classification:
    q = (nl.question or "").lower()
    mentioned = {topic for topic, words in WORDS.items() if any(_mentions(q, w) for w in words)}
    says_transistor = _mentions(q, "transistor")

    if nl.analysis == AnalysisType.CONCEPTUAL:
        return Classification(topic="conceptual", tags=sorted(mentioned),
                              unsupported_reason="Conceptual question: answered by the theory route, not a solver")

    types = {c.type.value for c in nl.components}
    kinds = {re.match(r"^(\w+)", x).group(1) for x in nl.query if re.match(r"^(\w+)", x)}
    has_lc = bool(types & {"C", "L"})
    devices = [DEVICE_TOPIC[t] for t in ("OPAMP", "Q", "M", "D") if t in types]
    is_osc = "oscillator" in mentioned
    is_filter = "filter" in mentioned or bool(kinds & FILTER_QUERIES)

    tags: List[str] = []
    if is_osc:
        tags.append("oscillator")
    if is_filter:
        tags.append("filter")
    tags += devices
    if "resonan" in q:
        tags.append("resonance")
    tags.append(nl.analysis.value)

    topic = ("oscillator" if is_osc else "filter" if is_filter
             else devices[0] if devices else nl.analysis.value)

    # --- question vs netlist disagreements ---------------------------
    notes: List[str] = []
    for dev in ("opamp", "bjt", "mosfet", "diode"):
        if dev in mentioned and dev not in devices:
            notes.append(f"The question mentions {DEVICE_LABEL[dev]} but the netlist has none; "
                         "the parse may have dropped it.")
    if says_transistor and not ({"bjt", "mosfet"} & set(devices)):
        notes.append("The question mentions a transistor but the netlist has no BJT or MOSFET.")
    if (is_osc or "filter" in mentioned) and not has_lc:
        notes.append("The question talks about a filter/oscillator but the circuit has no capacitor or inductor.")

    result = Classification(topic=topic, tags=list(dict.fromkeys(tags)),
                            confidence="low" if notes else "high", notes=notes)

    # --- LLM tie-break only when the rules disagree ------------------
    if result.confidence == "low" and client is not None:
        try:
            user = (f"Question: {nl.question}\nNetlist:\n{nl.to_spice()}\n"
                    f"Rule-based guess: {topic}\nConflicts: {notes}")
            data = client.chat_json("classifier", CLASSIFIER_SYSTEM, user)
            llm_topic = str(data.get("topic", "")).strip().lower()
            if llm_topic in TOPICS:
                result.topic, result.source = llm_topic, "llm"
                result.notes.append(f"LLM picked '{llm_topic}': {data.get('reason', '')}")
        except LLMError as e:
            result.notes.append(f"LLM tie-break failed: {e}")

    # --- which solver runs -------------------------------------------
    from solvers import NotSupportedError, get_solver      # local import keeps startup light
    try:
        result.solver = get_solver(nl).name
    except NotSupportedError as e:
        result.solver, result.unsupported_reason = None, str(e)
    return result