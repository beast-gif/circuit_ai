"""
Modified Nodal Analysis engine (SymPy). Shared by every linear solver.

For every non-ground node n it writes      sum of currents LEAVING n = 0      (KCL)
and for every voltage-type element (V, E, H, and L in DC) one constraint
                                           V(n+) - V(n-) = value
Unknowns = node voltages V_n  +  branch currents I_X of voltage-type elements.
Same count of equations and unknowns -> one linear system A·x = b.

Element models
    DC:  capacitor = open (no term),  inductor = short (0 V constraint)
    AC:  capacitor current = s·C·(Va - Vb),  inductor current = (Va - Vb)/(s·L)

Every current term is kept twice:
    expr : SymPy expression  -> used to solve
    text : template like "({V_in} - {V_out})/{R1}"  -> used for the step-by-step display.
           The solver fills the {tokens} with names (V_out, R1) or values (6.667, 1k).

Sign conventions (identical to SPICE, so the ngspice check in Step 4 lines up):
    branch current I_X of V/E/H/L : flows from nodes[0] THROUGH the element to nodes[1]
                                    (a source delivering power has negative I_X)
    I source ["a", "b"]           : current flows from a through the source into b
    G / F ["n+", "n-", ...]       : controlled current flows from n+ through the source to n-

Ideal op-amp ["out", "in+", "in-"]  (the "nullor" model, Phase 2):
    - no current enters in+ or in-            -> the inputs add NOTHING to KCL
    - the output supplies whatever current is needed
                                              -> one unknown I_U (flows from out into the op-amp,
                                                 i.e. the op-amp DELIVERS -I_U into node out)
    - virtual short                           -> one equation  V(in-) = V(in+)
    One unknown + one equation, so the system stays square. Example, inverting amp
    (Rin = 1k, Rf = 10k, Vin = 1): V- = V+ = 0, KCL at in-: (0 - 1)/1k + (0 - Vout)/10k = 0 -> Vout = -10.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import sympy as sp

from schemas import Component, is_symbolic, parse_value
from solvers.base import NotSupportedError, SolverError, pretty

s = sp.Symbol("s")
GROUND = "0"
LINEAR_TYPES = {"R", "C", "L", "V", "I", "E", "G", "F", "H", "OPAMP"}
INDEPENDENT = {"V", "I"}
CONTROLLED = {"E", "G", "F", "H"}
POSITIVE_TYPES = {"R", "C", "L"}
BRANCH_TYPES = {"V", "E", "H", "OPAMP"}           # elements that get their own current unknown
LATER_PHASE = {"Q": "BJTs (Phase 3)", "M": "MOSFETs (Phase 3)", "D": "diodes (Phase 3)"}


@dataclass
class Term:
    expr: sp.Expr
    text: str


# ----------------------------------------------------------------------
# Small text helpers for the display templates
# ----------------------------------------------------------------------
def _wrapped(t: str) -> bool:
    if not (t.startswith("(") and t.endswith(")")):
        return False
    depth = 0
    for i, ch in enumerate(t):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and i != len(t) - 1:
                return False
    return True


def paren(t: str) -> str:
    """Wrap in () when the text is not a single token: '1k' stays, '-5' -> '(-5)'."""
    t = str(t).strip()
    if not t or _wrapped(t):
        return t
    return f"({t})" if any(ch in t for ch in " +-/*∠") else t


def neg(t: str) -> str:
    return t[1:] if t.startswith("-") else "-" + t


def join_terms(texts: List[str]) -> str:
    return " + ".join(texts).replace("+ -", "- ") if texts else "0"


def tidy(e) -> sp.Expr:
    """Simplify a rational expression: combine fractions, cancel, factor (real cases)."""
    e = sp.cancel(sp.together(sp.sympify(e)))
    if not e.has(sp.I):
        try:
            e = sp.factor(e)
        except Exception:
            pass
    return e


# ----------------------------------------------------------------------
# Non-ideal op-amp parameters (Phase 2, step 3)
# ----------------------------------------------------------------------
_IDEAL_NAMES = {"", "ideal"}
_GENERIC_NONIDEAL = {"nonideal", "non-ideal", "non_ideal", "real", "practical", "finite", "finite_gain"}
_ALIASES = {
    "A0": ("a0", "a", "aol", "avol", "ao", "open_loop_gain", "openloopgain", "dc_gain"),
    "GBW": ("gbw", "gbp", "gb", "ft", "f_t", "unity_gain_bandwidth", "unitygainbandwidth", "ugb", "ugbw"),
    "Rin": ("rin", "ri", "r_in", "input_resistance", "rid"),
    "Rout": ("rout", "ro", "r_out", "output_resistance"),
    "SR": ("sr", "slew_rate", "slewrate", "slew"),
}


def _op_value(raw, gain: bool = False) -> float:
    """'100 dB' -> 1e5 (gains only), '1 MHz' -> 1e6, '2 MΩ' -> 2e6, 75 -> 75."""
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    if gain and "db" in s.lower():
        return 10 ** (float(re.match(r"\s*([-+]?[\d.]+)", s).group(1)) / 20)
    return parse_value(s)


def resolve_opamp(c: Component, conventions: Optional[dict] = None) -> Optional[Dict[str, object]]:
    """None for an ideal op-amp; otherwise {A0, GBW, Rin, Rout, SR, notes} with missing values filled
    from conventions.yaml (A0 = 1e5 and GBW = 1 MHz, 741-class) and a note saying which were assumed.

    Rules:
      - nothing given and model "ideal"            -> ideal (None)
      - a real part named (model "741") but no data -> A0 and GBW from conventions
      - any data given but no A0                    -> A0 from conventions (the gain is what makes it non-ideal)
      - A0 given, no GBW                            -> no pole: A is frequency-independent
    """
    p = {str(k).strip().lower().replace(" ", "_").replace("-", "_"): v for k, v in (c.params or {}).items()}
    raw = {key: next((p[a] for a in aliases if p.get(a) not in (None, "")), None) for key, aliases in _ALIASES.items()}
    model = (c.model or "").strip().lower()
    if all(v is None for v in raw.values()) and model in _IDEAL_NAMES:
        return None

    op = (conventions or {}).get("opamp") or {}
    notes: List[str] = []
    A0 = _op_value(raw["A0"], gain=True) if raw["A0"] is not None else None
    GBW = _op_value(raw["GBW"]) if raw["GBW"] is not None else None
    named_part = model not in _IDEAL_NAMES | _GENERIC_NONIDEAL
    if A0 is None:
        A0 = float(op.get("A0", 1e5))
        notes.append(f"A0 = {A0:g} assumed (typical, not given)")
    if GBW is None and named_part and raw["A0"] is None:
        GBW = float(op.get("GBW", 1e6))
        notes.append(f"GBW = {GBW:g} Hz assumed (typical {c.model}-class value, not given)")
    return {
        "A0": A0, "GBW": GBW,
        "Rin": _op_value(raw["Rin"]) if raw["Rin"] is not None else None,
        "Rout": _op_value(raw["Rout"]) if raw["Rout"] is not None else None,
        "SR": raw["SR"], "notes": notes,
    }


# ----------------------------------------------------------------------
class MNA:
    def __init__(self, components: List[Component], mode: str,
                 overrides: Optional[Dict[str, object]] = None,
                 removed: Optional[set] = None, conventions: Optional[dict] = None):
        if mode not in ("dc", "ac"):
            raise ValueError("mode must be 'dc' or 'ac'")
        removed = set(removed or ())
        self.mode = mode
        self.conventions = conventions or {}
        self.opamp_info: Dict[str, Optional[Dict[str, object]]] = {}   # id -> None (ideal) or resolved params
        self.tokens: Dict[str, str] = {}                               # extra display tokens, e.g. U1_A -> "1e5"
        self.components = [c for c in components if c.id not in removed]
        self.by_id = {c.id: c for c in self.components}
        self.overrides = {k: sp.sympify(v) for k, v in (overrides or {}).items()}

        later = sorted({c.type.value for c in self.components} - LINEAR_TYPES)
        if later:
            raise NotSupportedError("The DC/AC solver handles R, L, C, sources, controlled sources and op-amps; "
                                    + ", ".join(LATER_PHASE.get(t, t) for t in later) + " come later.")

        # AC analysis: if any source has an AC magnitude, only AC parts are used (like SPICE).
        has_ac = any(c.ac not in (None, "") for c in self.components if c.type.value in INDEPENDENT)
        self.ac_sources = has_ac and mode == "ac"
        self._params: Dict[str, sp.Symbol] = {}

        self.nodes: List[str] = []
        for c in self.components:
            for n in c.nodes:
                if n != GROUND and n not in self.nodes:
                    self.nodes.append(n)
        self.V: Dict[str, sp.Expr] = {n: sp.Symbol(f"V_{n}") for n in self.nodes}
        self.V[GROUND] = sp.Integer(0)

        self.branch: Dict[str, sp.Symbol] = {}
        for c in self.components:
            t = c.type.value
            if t in BRANCH_TYPES or (t == "L" and mode == "dc"):
                self.branch[c.id] = sp.Symbol(f"I_{c.id}")

        self.kcl: Dict[str, List[Term]] = {n: [] for n in self.nodes}
        self.constraints: List[Tuple[sp.Expr, str, str]] = []   # (expr = 0, text, component id)
        self.solution: Optional[Dict[sp.Symbol, sp.Expr]] = None
        self.A = self.b = None
        self.unknowns: List[sp.Symbol] = []
        self._build()

    # ------------------------------------------------------------------
    # Values
    # ------------------------------------------------------------------
    def _param(self, raw, positive: bool) -> sp.Expr:
        raw = str(raw).strip()
        if is_symbolic(raw):
            if raw not in self._params:
                self._params[raw] = sp.Symbol(raw, positive=True) if positive else sp.Symbol(raw, real=True)
            return self._params[raw]
        return sp.nsimplify(parse_value(raw), rational=True)    # exact: 4.7u -> 47/10000000

    def value(self, c: Component) -> sp.Expr:
        if c.value in (None, ""):
            raise SolverError(f"{c.id} has no value")
        return self._param(c.value, c.type.value in POSITIVE_TYPES)

    def source_value(self, c: Component) -> sp.Expr:
        if c.id in self.overrides:
            return self.overrides[c.id]
        if self.ac_sources:
            if c.ac in (None, ""):
                return sp.Integer(0)                            # DC-only source in AC analysis
            mag = self._param(c.ac, False)
            if c.ac_phase:
                mag = mag * sp.exp(sp.I * sp.pi * sp.nsimplify(c.ac_phase, rational=True) / 180)
            return mag
        if c.value in (None, ""):
            return sp.Integer(0)                                # AC-only source in DC analysis
        return self._param(c.value, False)

    def display_value(self, c: Component) -> str:
        if c.type.value in INDEPENDENT:
            if c.id in self.overrides:
                return pretty(self.overrides[c.id])
            if self.ac_sources:
                if c.ac in (None, ""):
                    return "0"
                return f"{c.ac}∠{c.ac_phase:g}°" if c.ac_phase else str(c.ac)
            return str(c.value) if c.value not in (None, "") else "0"
        return str(c.value)

    # ------------------------------------------------------------------
    # Currents
    # ------------------------------------------------------------------
    def _dv(self, a: str, b: str) -> Tuple[sp.Expr, str]:
        expr = self.V[a] - self.V[b]
        if a == GROUND and b == GROUND:
            return expr, "0"
        if b == GROUND:
            return expr, f"{{V_{a}}}"
        if a == GROUND:
            return expr, f"-{{V_{b}}}"
        return expr, f"({{V_{a}}} - {{V_{b}}})"

    @staticmethod
    def _lhs(a: str, b: str) -> str:
        if b == GROUND:
            return f"{{V_{a}}}"
        if a == GROUND:
            return f"-{{V_{b}}}"
        return f"{{V_{a}}} - {{V_{b}}}"

    def _passive(self, c: Component, a: str, b: str) -> Optional[Tuple[sp.Expr, str]]:
        """Current from a to b through R, C or L (None = open circuit)."""
        t = c.type.value
        dv, dt = self._dv(a, b)
        x = self.value(c)
        tok = f"{{{c.id}}}"
        if t == "R":
            return dv / x, f"{dt}/{tok}"
        if t == "C":
            if self.mode == "dc":
                return None
            return s * x * dv, f"s*{tok}*{dt}"
        if t == "L" and self.mode == "ac":
            return dv / (s * x), f"{dt}/(s*{tok})"
        raise AssertionError(f"_passive called for {c.id}")

    def _control(self, c: Component) -> Component:
        cid = (c.control or "").strip()
        if cid not in self.by_id:
            raise SolverError(f"{c.id}: controlling component '{cid}' is not in the circuit")
        ctrl = self.by_id[cid]
        if cid == c.id or ctrl.type.value in ("F", "G"):
            raise SolverError(f"{c.id}: control must be a voltage source, resistor, inductor or capacitor")
        return ctrl

    def current(self, c: Component) -> Tuple[sp.Expr, str]:
        """Current through c from nodes[0] to nodes[1] (in terms of the unknowns)."""
        t = c.type.value
        if c.id in self.branch:
            return self.branch[c.id], f"{{I_{c.id}}}"
        if t in ("R", "C", "L"):
            r = self._passive(c, c.nodes[0], c.nodes[1])
            return r if r is not None else (sp.Integer(0), "0")
        if t == "I":
            return self.source_value(c), f"{{{c.id}}}"
        if t == "G":
            dv, dt = self._dv(c.nodes[2], c.nodes[3])
            return self.value(c) * dv, f"{{{c.id}}}*{paren(dt)}"
        if t == "F":
            ie, it = self.current(self._control(c))
            return self.value(c) * ie, f"{{{c.id}}}*{paren(it)}"
        raise SolverError(f"No current model for {c.id} ({t})")

    def voltage(self, c: Component) -> sp.Expr:
        return self.V[c.nodes[0]] - self.V[c.nodes[1]]

    # ------------------------------------------------------------------
    def _build(self) -> None:
        for c in self.components:
            t = c.type.value
            a, b = c.nodes[0], c.nodes[1]

            if t == "OPAMP":                          # see module docstring
                self._stamp_opamp(c)
                continue

            # KCL contributions
            if t in ("R", "C") or (t == "L" and self.mode == "ac"):
                for p, q in ((a, b), (b, a)):
                    if p == GROUND:
                        continue
                    r = self._passive(c, p, q)
                    if r is not None:
                        self.kcl[p].append(Term(*r))
            else:
                e, txt = self.current(c)
                if a != GROUND:
                    self.kcl[a].append(Term(e, txt))
                if b != GROUND:
                    self.kcl[b].append(Term(-e, neg(txt)))

            # constraints of voltage-type elements
            lhs_e, lhs_t = self.V[a] - self.V[b], self._lhs(a, b)
            if t == "V":
                self.constraints.append((lhs_e - self.source_value(c), f"{lhs_t} = {{{c.id}}}", c.id))
            elif t == "L" and self.mode == "dc":
                self.constraints.append((lhs_e, f"{lhs_t} = 0", c.id))
            elif t == "E":
                dv, dt = self._dv(c.nodes[2], c.nodes[3])
                self.constraints.append((lhs_e - self.value(c) * dv,
                                         f"{lhs_t} = {{{c.id}}}*{paren(dt)}", c.id))
            elif t == "H":
                ie, it = self.current(self._control(c))
                self.constraints.append((lhs_e - self.value(c) * ie,
                                         f"{lhs_t} = {{{c.id}}}*{paren(it)}", c.id))

    def _stamp_opamp(self, c: Component) -> None:
        """Ideal:     V(in-) = V(in+)                                  (virtual short)
        Non-ideal:    V(out) = A(s)·(V(in+) - V(in-)) + Rout·I_U         (Thevenin output)
                      A(s) = A0 / (1 + s/ωp),  ωp = 2π·GBW/A0  (A = A0 in DC or without GBW)
                      plus Rin between in+ and in- (adds KCL terms at both inputs)
        Either way the output carries the unknown current I_U (from out into the op-amp)."""
        out, inp, inm = c.nodes
        I_U = self.branch[c.id]
        if out != GROUND:
            self.kcl[out].append(Term(I_U, f"{{I_{c.id}}}"))
        vp = "0" if inp == GROUND else f"{{V_{inp}}}"
        vm = "0" if inm == GROUND else f"{{V_{inm}}}"

        info = resolve_opamp(c, self.conventions)
        self.opamp_info[c.id] = info
        if info is None:
            self.constraints.append((self.V[inm] - self.V[inp], f"{vm} = {vp}", c.id))
            return

        A0 = sp.nsimplify(info["A0"], rational=True)
        if self.mode == "ac" and info["GBW"]:
            wp = 2 * sp.pi * sp.nsimplify(info["GBW"], rational=True) / A0
            A = A0 / (1 + s / wp)
            self.tokens[f"{c.id}_A"] = f"{info['A0']:g}/(1 + s/{float(wp):.4g})"
        else:
            A = A0
            self.tokens[f"{c.id}_A"] = f"{info['A0']:g}"

        if info["Rin"]:                                     # input resistance between the inputs
            Rin = sp.nsimplify(info["Rin"], rational=True)
            self.tokens[f"{c.id}_Rin"] = f"{info['Rin']:g}"
            for p, q in ((inp, inm), (inm, inp)):
                if p != GROUND:
                    dv, dt = self._dv(p, q)
                    self.kcl[p].append(Term(dv / Rin, f"{dt}/{{{c.id}_Rin}}"))

        vd_e, vd_t = self._dv(inp, inm)
        expr = self.V[out] - A * vd_e
        text = f"{{V_{out}}} = {{{c.id}_A}}*{paren(vd_t)}"
        if info["Rout"]:
            Rout = sp.nsimplify(info["Rout"], rational=True)
            self.tokens[f"{c.id}_Rout"] = f"{info['Rout']:g}"
            expr -= Rout * I_U
            text += f" + {{{c.id}_Rout}}*{{I_{c.id}}}"
        self.constraints.append((expr, text, c.id))

    def known_nodes(self) -> Dict[str, Tuple[sp.Expr, str, str]]:
        """Nodes fixed directly by a grounded independent voltage source: node -> (value, display, source id)."""
        known: Dict[str, Tuple[sp.Expr, str, str]] = {}
        for c in self.components:
            if c.type.value != "V":
                continue
            a, b = c.nodes
            if b == GROUND and a != GROUND:
                known[a] = (self.source_value(c), self.display_value(c), c.id)
            elif a == GROUND and b != GROUND:
                known[b] = (-self.source_value(c), "-" + paren(self.display_value(c)), c.id)
        return known

    # ------------------------------------------------------------------
    def solve(self) -> Dict[sp.Symbol, sp.Expr]:
        empty = [n for n in self.nodes if not self.kcl[n]]
        if empty:
            why = "in DC (capacitors are open circuits)" if self.mode == "dc" else "here"
            raise SolverError(f"Node(s) {empty} have nothing that carries current {why}, "
                              "so their voltage is undefined.")

        self.unknowns = [self.V[n] for n in self.nodes] + list(self.branch.values())
        eqs = [sp.Add(*[t.expr for t in self.kcl[n]]) for n in self.nodes] + [e for e, _, _ in self.constraints]
        self.A, self.b = sp.linear_eq_to_matrix(eqs, self.unknowns)

        sol = sp.linsolve((self.A, self.b), self.unknowns)
        if sol == sp.S.EmptySet or len(sol) == 0:
            raise SolverError("The circuit equations contradict each other (e.g. a voltage source shorted "
                              "by a wire or an inductor in DC).")
        vals = list(sol)[0]
        unknown_set = set(self.unknowns)
        if any(v.free_symbols & unknown_set for v in vals):
            raise SolverError("The circuit has no unique solution: some voltage or current is undefined "
                              "(e.g. a part connected only through capacitors in DC, or a loop of current sources).")
        self.solution = {u: tidy(v) for u, v in zip(self.unknowns, vals)}
        return self.solution

    def evaluate(self, expr) -> sp.Expr:
        if self.solution is None:
            raise SolverError("solve() must run before evaluate()")
        return tidy(sp.sympify(expr).subs(self.solution))