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
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import sympy as sp

from schemas import Component, is_symbolic, parse_value
from solvers.base import NotSupportedError, SolverError, pretty

s = sp.Symbol("s")
GROUND = "0"
LINEAR_TYPES = {"R", "C", "L", "V", "I", "E", "G", "F", "H"}
INDEPENDENT = {"V", "I"}
CONTROLLED = {"E", "G", "F", "H"}
POSITIVE_TYPES = {"R", "C", "L"}
LATER_PHASE = {"OPAMP": "op-amps (Phase 2)", "Q": "BJTs (Phase 3)",
               "M": "MOSFETs (Phase 3)", "D": "diodes (Phase 3)"}


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
class MNA:
    def __init__(self, components: List[Component], mode: str,
                 overrides: Optional[Dict[str, object]] = None,
                 removed: Optional[set] = None):
        if mode not in ("dc", "ac"):
            raise ValueError("mode must be 'dc' or 'ac'")
        removed = set(removed or ())
        self.mode = mode
        self.components = [c for c in components if c.id not in removed]
        self.by_id = {c.id: c for c in self.components}
        self.overrides = {k: sp.sympify(v) for k, v in (overrides or {}).items()}

        later = sorted({c.type.value for c in self.components} - LINEAR_TYPES)
        if later:
            raise NotSupportedError("The DC/AC solver handles R, L, C, independent and controlled sources; "
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
            if t in ("V", "E", "H") or (t == "L" and mode == "dc"):
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
