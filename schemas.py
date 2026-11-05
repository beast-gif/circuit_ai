"""
Core data structures shared by every module.

Netlist    -> what the parser (LLM) must produce and the solver consumes
Step       -> one line of a worked solution
StepTrace  -> ordered list of steps; both "key" and "full" views come from it
SolveResult-> final answers + trace + verification status
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ----------------------------------------------------------------------
# Value parsing:  "2.2k" -> 2200.0,  "4.7uF" -> 4.7e-6,  "1M" -> 1e6
# ----------------------------------------------------------------------
# Case-sensitive on purpose:
#   "M"  = mega  (textbook: 1 MΩ)      SPICE would read "1M" as milli -- we don't.
#   "m"  = milli (10 mA, 5 mH)
#   "meg"/"Meg"/"MEG" = mega (SPICE style, also accepted)
#   "F" alone = farad (1F = 1 farad), "f" = femto
_PREFIX = {
    "T": 1e12, "G": 1e9, "g": 1e9, "M": 1e6,
    "k": 1e3, "K": 1e3,
    "m": 1e-3,
    "u": 1e-6, "U": 1e-6, "µ": 1e-6, "μ": 1e-6,
    "n": 1e-9, "p": 1e-12, "f": 1e-15,
}
_NUM = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*?)\s*$")
_SYMBOL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_value(raw: Union[str, int, float]) -> float:
    """Convert an engineering-notation string to a float (SI units)."""
    if isinstance(raw, (int, float)):
        return float(raw)
    m = _NUM.match(str(raw))
    if not m:
        raise ValueError(f"Cannot parse value: {raw!r}")
    number = float(m.group(1))
    rest = m.group(2).replace(" ", "")
    if not rest:
        return number
    if rest.lower().startswith("meg"):
        return number * 1e6
    first = rest[0]
    if first == "F" and len(rest) == 1:          # "1F" -> one farad
        return number
    return number * _PREFIX.get(first, 1.0)      # unit letters (V, A, Ω, H, Hz) -> x1


def is_symbolic(raw: Union[str, int, float, None]) -> bool:
    """True for symbolic values like 'R', 'R1', 'C_f' (for answers in terms of symbols)."""
    return isinstance(raw, str) and bool(_SYMBOL.match(raw.strip())) and not _NUM.match(raw)


# ----------------------------------------------------------------------
# Components
# ----------------------------------------------------------------------
class ComponentType(str, Enum):
    R = "R"          # resistor                 nodes: [n+, n-]
    C = "C"          # capacitor                nodes: [n+, n-]
    L = "L"          # inductor                 nodes: [n+, n-]
    V = "V"          # independent voltage src  nodes: [n+, n-]
    I = "I"          # independent current src  nodes: [n+, n-]  (current flows n+ -> n- through the source)
    D = "D"          # diode                    nodes: [anode, cathode]
    Q = "Q"          # BJT                      nodes: [collector, base, emitter]
    M = "M"          # MOSFET                   nodes: [drain, gate, source]  (body tied to source)
    OPAMP = "OPAMP"  # op-amp                   nodes: [out, in+, in-]
    E = "E"          # VCVS                     nodes: [n+, n-, nc+, nc-], value = gain
    G = "G"          # VCCS                     nodes: [n+, n-, nc+, nc-], value = gm
    F = "F"          # CCCS                     nodes: [n+, n-], control = source id, value = gain
    H = "H"          # CCVS                     nodes: [n+, n-], control = source id, value = rm


NODE_COUNT = {
    "R": 2, "C": 2, "L": 2, "V": 2, "I": 2, "D": 2,
    "Q": 3, "M": 3, "OPAMP": 3,
    "E": 4, "G": 4, "F": 2, "H": 2,
}
NEEDS_VALUE = {"R", "C", "L", "E", "G", "F", "H"}
MUST_BE_POSITIVE = {"R", "C", "L"}


class Component(BaseModel):
    id: str = Field(..., description="Unique name, e.g. R1, C2, V1, Q1")
    type: ComponentType
    nodes: List[str] = Field(..., description="Node names in the order listed in ComponentType")
    value: Optional[str] = Field(None, description="'1k', '10', '4.7u', or a symbol like 'R'")
    ac: Optional[str] = Field(None, description="AC magnitude for sources, e.g. '1'")
    ac_phase: float = Field(0.0, description="AC phase in degrees")
    model: Optional[str] = Field(None, description="NPN/PNP, NMOS/PMOS, ideal, ...")
    params: Dict[str, Union[float, str]] = Field(default_factory=dict,
                                                 description="beta, VA, kn, Vt, A0, ...")
    control: Optional[str] = Field(None, description="Controlling source id (F/H only)")

    @field_validator("nodes")
    @classmethod
    def _clean_nodes(cls, v: List[str]) -> List[str]:
        cleaned = [str(n).strip() for n in v]
        # common LLM spellings of ground
        return ["0" if n.lower() in {"gnd", "ground", "0"} else n for n in cleaned]

    @model_validator(mode="after")
    def _check(self) -> "Component":
        t = self.type.value
        if len(self.nodes) != NODE_COUNT[t]:
            raise ValueError(f"{self.id}: type {t} needs {NODE_COUNT[t]} nodes, got {len(self.nodes)}")
        if t in NEEDS_VALUE and self.value in (None, ""):
            raise ValueError(f"{self.id}: type {t} needs a value")
        if t in {"V", "I"} and self.value in (None, "") and self.ac in (None, ""):
            raise ValueError(f"{self.id}: source needs a DC value or an AC magnitude")
        if t in {"F", "H"} and not self.control:
            raise ValueError(f"{self.id}: {t} source needs 'control' (id of controlling source)")
        if self.value not in (None, "") and not is_symbolic(self.value):
            num = parse_value(self.value)                 # raises if unparseable
            if t in MUST_BE_POSITIVE and num <= 0:
                raise ValueError(f"{self.id}: {t} value must be > 0, got {self.value}")
        if len(set(self.nodes)) < 2 and t in NODE_COUNT and NODE_COUNT[t] == 2:
            raise ValueError(f"{self.id}: both terminals on the same node ({self.nodes[0]})")
        return self

    def numeric_value(self) -> Optional[float]:
        if self.value in (None, "") or is_symbolic(self.value):
            return None
        return parse_value(self.value)


# ----------------------------------------------------------------------
# Netlist (the contract between parser and solver)
# ----------------------------------------------------------------------
class AnalysisType(str, Enum):
    DC = "dc"                    # bias point / DC node voltages
    AC = "ac"                    # phasor at one frequency, or H(s) / H(jw)
    TRANSIENT = "transient"      # time response (step, RC charging, ...)
    SMALL_SIGNAL = "small_signal"  # transistor amp gain / Rin / Rout
    CONCEPTUAL = "conceptual"    # no circuit to solve; theory question


# Accepted query forms (the validator in Step 2 enforces these):
#   V(out)  V(a,b)  I(R1)  P(R1)  Z(a,b)  Rth(a,b)  Vth(a,b)  Rin  Rout
#   H(out,in)  gain  fc  f0  poles  zeros  bias(Q1)
class Netlist(BaseModel):
    question: str = Field("", description="Original question text")
    components: List[Component] = Field(default_factory=list)
    analysis: AnalysisType
    query: List[str] = Field(default_factory=list, description="What to find, e.g. ['V(out)']")
    frequency: Optional[str] = Field(None, description="Single AC frequency in Hz, e.g. '1k'")
    assumptions: List[str] = Field(default_factory=list,
                                   description="Assumptions stated in the question, e.g. 'ideal op-amp'")

    @model_validator(mode="after")
    def _check(self) -> "Netlist":
        if self.analysis == AnalysisType.CONCEPTUAL:
            return self
        if not self.components:
            raise ValueError("Netlist has no components")
        if not self.query:
            raise ValueError("Netlist has no query (what should be solved?)")
        ids = [c.id for c in self.components]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"Duplicate component ids: {sorted(dupes)}")
        if self.frequency is not None:
            parse_value(self.frequency)
        return self

    # --- helpers -------------------------------------------------------
    def nodes(self) -> List[str]:
        seen: List[str] = []
        for c in self.components:
            for n in c.nodes:
                if n not in seen:
                    seen.append(n)
        return seen

    def get(self, comp_id: str) -> Optional[Component]:
        return next((c for c in self.components if c.id == comp_id), None)

    def to_spice(self) -> str:
        """Human-readable SPICE-style listing (also used in the confirmation step)."""
        lines = []
        for c in self.components:
            parts = [c.id, *c.nodes]
            if c.control:
                parts.append(c.control)
            if c.model:
                parts.append(c.model)
            if c.value not in (None, ""):
                parts.append(str(c.value))
            if c.ac not in (None, ""):
                parts.append(f"AC {c.ac} {c.ac_phase:g}")
            if c.params:
                parts.append(" ".join(f"{k}={v}" for k, v in c.params.items()))
            lines.append(" ".join(parts))
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Step trace (drives "key steps first, full solution on request")
# ----------------------------------------------------------------------
class Step(BaseModel):
    step: int
    level: Literal["key", "full"] = "key"
    action: str = Field(..., description="e.g. 'KCL at node out'")
    equation: Optional[str] = None       # e.g. "(Vout-10)/1000 + Vout/2000 = 0"
    result: Optional[str] = None         # e.g. "Vout = 6.667 V"


class StepTrace(BaseModel):
    steps: List[Step] = Field(default_factory=list)

    def add(self, action: str, equation: Optional[str] = None,
            result: Optional[str] = None, level: Literal["key", "full"] = "key") -> Step:
        s = Step(step=len(self.steps) + 1, level=level, action=action,
                 equation=equation, result=result)
        self.steps.append(s)
        return s

    def key_steps(self) -> List[Step]:
        return [s for s in self.steps if s.level == "key"]

    def all_steps(self) -> List[Step]:
        return list(self.steps)


# ----------------------------------------------------------------------
# Solver output
# ----------------------------------------------------------------------
class SolveResult(BaseModel):
    answers: Dict[str, str] = Field(default_factory=dict,
                                    description="Pretty answers, e.g. {'V(out)': '6.667 V'}")
    numeric: Dict[str, Optional[float]] = Field(default_factory=dict,
                                                description="Raw numbers for verification")
    symbolic: Dict[str, str] = Field(default_factory=dict,
                                     description="Closed-form expressions, e.g. H(s)")
    trace: StepTrace = Field(default_factory=StepTrace)
    method: str = ""                       # "MNA (Lcapy)", "small-signal", ...
    verified: Optional[bool] = None        # None = not yet checked
    verification_notes: List[str] = Field(default_factory=list)
