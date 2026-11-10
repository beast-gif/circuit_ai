"""
Comparators and Schmitt triggers (Phase 3, step 6): op-amps whose output sits at a supply rail.

An op-amp in a linear circuit obeys the virtual short, V+ = V−, because negative feedback holds it there.
A comparator has NO negative feedback (or POSITIVE feedback), so nothing holds the inputs together: the
output slams to a rail. The op-amp then has three possible states, and they are decided the same way as a
diode's or a transistor's — assume, solve, check:

    output HIGH   V(out) = +V_sat         valid while V+ > V−
    output LOW    V(out) = −V_sat         valid while V+ < V−
    linear        V+ = V− (virtual short)  valid while |V(out)| <= V_sat   (needs negative feedback)

Thresholds (Schmitt trigger). With the output stuck at one rail the circuit is linear, so write the input as
a symbol and solve once: the differential input V+ − V− comes out as a straight line in the input voltage,
and the state lasts until that line reaches zero. Solving  V+ − V− = 0  therefore gives the exact trip point
of that state, with no guessing:

    while the output is HIGH it stays high until vin reaches the boundary of the HIGH state
    while the output is LOW  it stays low  until vin reaches the boundary of the LOW state
    V_UT = the higher of the two,  V_LT = the lower,  hysteresis = V_UT − V_LT

For the classic inverting Schmitt trigger (input to in−, feedback R2 from out to in+, R1 from in+ to ground)
this reproduces V_UT, V_LT = ±V_sat·R1/(R1 + R2); the non-inverting one gives ±V_sat·R1/R2. Both come out of
the same two linear solves, so no formula is hard-coded.
"""
from __future__ import annotations

import itertools
import json
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import sympy as sp

from schemas import AnalysisType, Component, Netlist, SolveResult, StepTrace, is_symbolic, parse_value
from solvers.base import BaseSolver, NotSupportedError, SolverError, fmt_num, fmt_si, sig_figs
from solvers.mna import GROUND, MNA

X = sp.Symbol("vin_x", real=True)
STATES = ("hi", "lo", "lin")
KINDS = {"V", "I", "P", "VTH", "VTL", "hysteresis", "trip", "state", "bias"}
_Q = re.compile(r"^(\w+)\((\w+)(?:,(\w+))?\)$")
_SAT_WORD = re.compile(r"saturat\w*\s*(?:at|voltage|output)?\s*(?:of|=|is)?\s*(?:±|\+/-|\+-)?\s*"
                       r"(\d+(?:\.\d+)?)\s*v", re.IGNORECASE)
_RAIL_SYM = re.compile(r"(?:±|\+/-|\+/−|\+-)\s*(\d+(?:\.\d+)?)\s*v\b", re.IGNORECASE)
_RAIL_POS = re.compile(r"\bv(?:cc|dd|\+|pos|sat)\s*=\s*\+?\s*(\d+(?:\.\d+)?)\s*v\b", re.IGNORECASE)
_RAIL_NEG = re.compile(r"\bv(?:ee|ss|-|neg)\s*=\s*[-−]?\s*(\d+(?:\.\d+)?)\s*v\b", re.IGNORECASE)
DEFAULT_SAT = 13.0          # only used when the question gives no supply at all, and it is said in the notes


def rails(nl: Netlist, conventions: Optional[dict] = None) -> Tuple[float, float, List[str]]:
    """(+V_sat, −V_sat magnitude, notes) from the question, its assumptions or an op-amp's params."""
    notes: List[str] = []
    for c in nl.components:
        if c.type.value != "OPAMP":
            continue
        p = {str(k).strip().lower(): v for k, v in (c.params or {}).items()}
        for key in ("vsat", "v_sat", "vo_max", "vsat_pos", "vcc", "vs"):
            if p.get(key) not in (None, ""):
                v = abs(parse_value(str(p[key])))
                return v, v, notes
    text = " ".join([nl.question or ""] + list(nl.assumptions))
    m = _SAT_WORD.search(text) or _RAIL_SYM.search(text)
    if m:
        return float(m.group(1)), float(m.group(1)), notes
    pos, neg = _RAIL_POS.search(text), _RAIL_NEG.search(text)
    if pos or neg:
        p = float(pos.group(1)) if pos else float(neg.group(1))
        n = float(neg.group(1)) if neg else float(pos.group(1))
        return p, n, notes
    notes.append(f"no supply given: the output is taken to saturate at ±{fmt_num(DEFAULT_SAT)} V")
    return DEFAULT_SAT, DEFAULT_SAT, notes


def has_negative_feedback(nl: Netlist, op: Component) -> bool:
    """Is there a path from the output to the INVERTING input through passive components (not via ground)?
    That is what holds an op-amp linear. A path to the non-inverting input is positive feedback instead."""
    out, inm = op.nodes[0], op.nodes[2]
    adj: Dict[str, List[str]] = {}
    for c in nl.components:
        if c.type.value not in ("R", "C", "L"):
            continue
        a, b = c.nodes[0], c.nodes[1]
        if GROUND in (a, b):
            continue
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    seen, stack = {out}, [out]
    while stack:
        n = stack.pop()
        if n == inm:
            return True
        for m in adj.get(n, []):
            if m not in seen:
                seen.add(m)
                stack.append(m)
    return False


class ComparatorSolver(BaseSolver):
    name = "comparator"
    method = ("Assume the op-amp's output state (high rail / low rail / linear), solve the linear circuit "
              "exactly, check the assumption; trip points from the input that makes V+ − V− zero")

    def supports(self, nl: Netlist) -> Tuple[bool, str]:
        ops = [c for c in nl.components if c.type.value == "OPAMP"]
        if not ops:
            return False, ""
        if any(c.type.value in ("Q", "M", "D") for c in nl.components):
            return False, ""                       # the device solvers deal with those circuits
        kinds = {(_Q.match(q.replace(" ", "")).group(1) if _Q.match(q.replace(" ", "")) else q.replace(" ", ""))
                 for q in nl.query}
        wants_trip = bool(kinds & {"VTH", "VTL", "hysteresis", "trip", "state"})
        comparator = any(not has_negative_feedback(nl, op) for op in ops)
        if not (wants_trip or comparator):
            return False, ""                       # a normal linear op-amp circuit: the DC/AC solver has it
        if nl.analysis == AnalysisType.CONCEPTUAL:
            return False, "Conceptual question: no circuit to solve"
        if nl.analysis != AnalysisType.DC:
            return False, ("A comparator / Schmitt trigger is solved at DC (and by its trip points); a "
                           "time-varying input through one is not supported yet")
        if any(c.value not in (None, "") and is_symbolic(c.value) for c in nl.components):
            return False, "Comparator states can only be decided with numeric values"
        return True, ""

    def solve(self, nl: Netlist, conventions: Dict) -> SolveResult:
        ok, why = self.supports(nl)
        if not ok:
            raise NotSupportedError(why or "Not a comparator circuit")
        return _CompJob(nl, conventions).run()


@dataclass
class Trip:
    state: str                 # the state this is the boundary of
    vin: float                 # input voltage at the boundary
    out: float                 # output voltage while in that state


class _CompJob:
    def __init__(self, nl: Netlist, conventions: Dict):
        self.nl, self.conv = nl, conventions or {}
        self.sig = sig_figs(conventions)
        self.ops = [c for c in nl.components if c.type.value == "OPAMP"]
        if len(self.ops) > 2:
            raise NotSupportedError("At most two comparators in one circuit")
        self.pos, self.neg, self.notes = rails(nl, conventions)
        self.others = [c for c in nl.components if c.type.value != "OPAMP"]
        self.trace = StepTrace()
        self.res = SolveResult(method=ComparatorSolver.method)
        self.queries = [q.replace(" ", "") for q in nl.query]

    # ------------------------------------------------------------------
    def _parts(self, op: Component, state: str) -> List[Component]:
        """The op-amp replaced by what its state means."""
        out, inp, inm = op.nodes
        if state == "lin":
            return [op]                                     # the ideal nullor: V+ = V−
        v = self.pos if state == "hi" else -self.neg
        return [Component(id=f"{op.id}_sat", type="V", nodes=[out, GROUND], value=repr(v))]

    def _components(self, states: Dict[str, str], src: Optional[Component] = None,
                    sym: bool = False) -> List[Component]:
        comps = []
        for c in self.others:
            if sym and src is not None and c.id == src.id:
                continue                                    # replaced by the symbolic input below
            comps.append(c)
        for op in self.ops:
            comps += self._parts(op, states[op.id])
        return comps

    def _mna(self, states: Dict[str, str], overrides: Optional[Dict] = None) -> MNA:
        m = MNA(self._components(states), "dc", overrides=overrides, conventions=self.conv)
        m.solve()
        return m

    def _vd(self, m: MNA, op: Component):
        return m.evaluate(m.V[op.nodes[1]] - m.V[op.nodes[2]])

    def _f(self, m: MNA, expr) -> float:
        # always substitute the solution: m.V[n] is the raw unknown, not its value
        e = m.evaluate(expr)
        if getattr(e, "free_symbols", set()):
            raise NotSupportedError("Comparator states need numeric values")
        return float(sp.N(e))

    # ------------------------------------------------------------------
    def _checks(self, m: MNA, states: Dict[str, str]) -> List[Tuple[bool, str]]:
        out = []
        for op in self.ops:
            st = states[op.id]
            vd = self._f(m, self._vd(m, op))
            vo = self._f(m, m.V[op.nodes[0]])
            if st == "hi":
                ok = vd >= -1e-12
                out.append((ok, f"{op.id} output high needs V+ > V−: V+ − V− = {fmt_si(vd, 'V', self.sig)}"
                                + (" ✓" if ok else " ✗")))
            elif st == "lo":
                ok = vd <= 1e-12
                out.append((ok, f"{op.id} output low needs V+ < V−: V+ − V− = {fmt_si(vd, 'V', self.sig)}"
                                + (" ✓" if ok else " ✗")))
            else:
                ok = -self.neg - 1e-9 <= vo <= self.pos + 1e-9
                out.append((ok, f"{op.id} linear needs the output inside the rails: V(out) = "
                                f"{fmt_si(vo, 'V', self.sig)} vs ±{fmt_si(min(self.pos, self.neg), 'V')}"
                                + (" ✓" if ok else " ✗ (it is saturated)")))
        return out

    def _state_text(self, op_id: str, st: str) -> str:
        return {"hi": f"{op_id} output at the + rail", "lo": f"{op_id} output at the − rail",
                "lin": f"{op_id} linear (virtual short)"}[st]

    # ------------------------------------------------------------------
    def _input_source(self) -> Optional[Component]:
        """The source that carries the input: the one the question's trip points are measured against."""
        srcs = [c for c in self.others if c.type.value in ("V", "I")]
        if len(srcs) == 1:
            return srcs[0]
        named = [c for c in srcs if c.id.lower() in ("vin", "v1", "vi", "vs", "in")
                 or any(n in ("in", "vin") for n in c.nodes)]
        return named[0] if len(named) == 1 else None

    def trips(self) -> List[Trip]:
        """For each saturated state, the input voltage at which it ends (V+ − V− reaches zero)."""
        src = self._input_source()
        if src is None or len(self.ops) != 1:
            raise SolverError("Trip points need exactly one op-amp and one input source")
        op = self.ops[0]
        out: List[Trip] = []
        for st in ("hi", "lo"):
            states = {op.id: st}
            m = MNA(self._components(states), "dc", overrides={src.id: X}, conventions=self.conv)
            m.solve()
            vd = sp.expand(m.evaluate(self._vd(m, op)))
            a = sp.simplify(vd.coeff(X, 1))
            b = sp.simplify(vd.subs(X, 0))
            if a == 0:
                raise SolverError(f"the input does not reach {op.id}: V+ − V− does not depend on {src.id}, so "
                                  f"there is no trip point (check how the input is connected)")
            x = float(sp.N(-b / a))
            vo = self.pos if st == "hi" else -self.neg
            out.append(Trip(st, x, vo))
            self.trace.add(f"With the output at the {'+' if st == 'hi' else '−'} rail "
                           f"({fmt_si(vo, 'V', self.sig)}), write the input as a symbol and solve once",
                           f"V+ − V− = {fmt_num(float(sp.N(a)), self.sig)}·{src.id} + "
                           f"{fmt_num(float(sp.N(b)), self.sig)} = 0",
                           f"this state ends at {src.id} = {fmt_si(x, 'V', self.sig)}")
        return out

    # ------------------------------------------------------------------
    def run(self) -> SolveResult:
        kinds = {(_Q.match(q).group(1) if _Q.match(q) else q) for q in self.queries}
        for op in self.ops:
            fb = "positive feedback" if not has_negative_feedback(self.nl, op) else "negative feedback"
            kind = ("a comparator / Schmitt trigger: with no negative feedback the output goes to a rail"
                    if fb == "positive feedback" else "an op-amp with negative feedback")
            self.trace.add(f"{op.id} is {kind}",
                           f"output states: +{fmt_si(self.pos, 'V')} (high), −{fmt_si(self.neg, 'V')} (low), "
                           f"or linear while |V(out)| ≤ the rail", "; ".join(self.notes) or None)

        trips: List[Trip] = []
        if kinds & {"VTH", "VTL", "hysteresis", "trip"}:
            trips = self.trips()
            hi = next(t for t in trips if t.state == "hi")
            lo = next(t for t in trips if t.state == "lo")
            vut, vlt = max(hi.vin, lo.vin), min(hi.vin, lo.vin)
            self.trace.add("The two boundaries are the trip points",
                           f"while high, it switches at {fmt_si(hi.vin, 'V', self.sig)};  while low, it "
                           f"switches at {fmt_si(lo.vin, 'V', self.sig)}",
                           f"V_UT = {fmt_si(vut, 'V', self.sig)}, V_LT = {fmt_si(vlt, 'V', self.sig)}, "
                           f"hysteresis = {fmt_si(vut - vlt, 'V', self.sig)}")

        solved = None
        if kinds - {"VTH", "VTL", "hysteresis", "trip"}:
            tried, good = [], []
            for combo in itertools.product(STATES, repeat=len(self.ops)):
                states = {op.id: st for op, st in zip(self.ops, combo)}
                if any(st == "lin" and not has_negative_feedback(self.nl, op)
                       for op, st in zip(self.ops, combo)):
                    continue                        # no negative feedback: it cannot stay linear
                try:
                    m = self._mna(states)
                except SolverError as e:
                    tried.append((states, f"no unique solution ({e})"))
                    continue
                checks = self._checks(m, states)
                bad = [t for ok, t in checks if not ok]
                if bad:
                    tried.append((states, bad[0]))
                    continue
                good.append((states, m, checks))
            # Positive feedback makes the circuit BISTABLE: inside the hysteresis band both rail states pass
            # their own check, so there is no single DC answer. Only one consistent state means a real answer.
            solved = good[0] if len(good) == 1 else None
            if solved is None:
                if len(good) > 1:
                    tried += [(st_, "consistent as well, so the output is bistable here") for st_, _, _ in good]
                if not trips:
                    try:
                        trips = self.trips()                  # show WHERE the two boundaries are
                    except SolverError:
                        raise SolverError(
                            "No output state is consistent: with hysteresis the output depends on the input's "
                            "history, so ask for the trip points (VTH, VTL, hysteresis) instead of a single "
                            "DC output")
                lo_, hi_ = min(t.vin for t in trips), max(t.vin for t in trips)
                for st_, why in tried[:6]:
                    self.trace.add("Try " + ", ".join(self._state_text(i, s) for i, s in st_.items()), None,
                                   why, level="full")
                self.trace.add("The input sits between the two trip points, so the output depends on which way "
                               "the input last crossed one (that is what hysteresis means) and a single DC "
                               "answer is not defined",
                               f"{fmt_si(lo_, 'V', self.sig)} < {self._input_source().id} "
                               f"< {fmt_si(hi_, 'V', self.sig)}",
                               "both output states are stable here")
            else:
                states, m, checks = solved
                for st_, why in tried[:6]:
                    self.trace.add("Try " + ", ".join(self._state_text(i, s) for i, s in st_.items()), None,
                                   f"✗ {why}: this assumption is wrong", level="full")
                self.trace.add("Assume " + ", ".join(self._state_text(i, s) for i, s in states.items()),
                               ";  ".join(f"{i}: V(out) = "
                                          + ("+" + fmt_si(self.pos, 'V') if s == "hi" else
                                             "−" + fmt_si(self.neg, 'V') if s == "lo" else "V+ = V−")
                                          for i, s in states.items()))
                self._linear_answers(states, m)
                self.trace.add("Check the assumptions", ";  ".join(t for _, t in checks),
                               "all assumptions hold ✓")

        for q in self.queries:
            mq = _Q.match(q)
            kind, a, b = (mq.group(1), mq.group(2), mq.group(3)) if mq else (q, None, None)
            if kind in ("VTH", "VTL", "hysteresis", "trip") and trips:
                hi = next(t for t in trips if t.state == "hi")
                lo = next(t for t in trips if t.state == "lo")
                vut, vlt = max(hi.vin, lo.vin), min(hi.vin, lo.vin)
                val = {"VTH": vut, "VTL": vlt, "hysteresis": vut - vlt, "trip": vut}[kind]
                self.res.answers[q] = fmt_si(val, "V", self.sig) + (
                    f" (upper trip point: the output goes {'low' if hi.vin >= lo.vin else 'high'} here)"
                    if kind in ("VTH", "trip") else
                    " (lower trip point)" if kind == "VTL" else
                    f" wide (from {fmt_si(vlt, 'V', self.sig)} to {fmt_si(vut, 'V', self.sig)})")
                self.res.numeric[q] = val
            elif q in self.res.answers:
                continue
            elif kind in ("state", "bias"):
                if solved:
                    st = solved[0].get(a or self.ops[0].id, "?")
                    self.res.answers[q] = (f"{a or self.ops[0].id} output "
                                           + {"hi": f"saturated HIGH at +{fmt_si(self.pos, 'V', self.sig)}",
                                              "lo": f"saturated LOW at −{fmt_si(self.neg, 'V', self.sig)}",
                                              "lin": "in the linear region"}[st])
                else:
                    self.res.answers[q] = "depends on the input's history (hysteresis): ask for the trip points"
                self.res.numeric[q] = None
            elif solved is None and kind in ("V", "I", "P"):
                self.res.answers[q] = ("depends on the input's history (hysteresis): the input lies between the "
                                       "two trip points, so both output states are stable — ask for VTH, VTL "
                                       "and the hysteresis instead")
                self.res.numeric[q] = None
            elif q not in self.res.answers:
                self.res.answers[q] = (f"Could not compute: '{q}' is not available for a comparator "
                                       "(ask for V, I, VTH, VTL, hysteresis or state)")
                self.res.numeric[q] = None

        for op in self.ops:
            if solved:
                self.res.symbolic[f"state.{op.id}"] = solved[0][op.id]
        self.res.symbolic["comp.rails"] = json.dumps([self.pos, self.neg])
        if trips:
            self.res.symbolic["comp.trips"] = json.dumps([[t.state, t.vin, t.out] for t in trips])
        self.res.verification_notes += [f"info: {n}" for n in self.notes]
        if self.queries and all(self.res.answers.get(q, "").startswith("Could not compute")
                                for q in self.queries):
            raise SolverError(" | ".join(self.res.answers.values()))
        self.res.trace = self.trace
        return self.res

    # ------------------------------------------------------------------
    def _linear_answers(self, states: Dict[str, str], m: MNA) -> None:
        """V / I / P of the circuit once the op-amp states are fixed: hand it to the DC/AC solver so the
        steps and the formatting are the same as everywhere else."""
        want = [q for q in self.queries if (_Q.match(q) or [None]) and _Q.match(q)
                and _Q.match(q).group(1) in ("V", "I", "P")]
        if not want:
            return
        from solvers.dc_ac import _Job
        comps = self._components(states)
        lin = Netlist(question=self.nl.question, components=comps, analysis="dc", query=want,
                      assumptions=list(self.nl.assumptions))
        try:
            job = _Job(lin, self.conv)
            res = job.run()
        except SolverError as e:
            raise SolverError(f"the circuit with the output at the rail could not be solved: {e}")
        for st in job.trace.all_steps():
            self.trace.add(st.action, st.equation, st.result, level=st.level)
        for q in want:
            if q in res.answers:
                self.res.answers[q] = res.answers[q]
                for k, v in res.numeric.items():
                    if k == q or k.startswith(q + "."):
                        self.res.numeric[k] = v