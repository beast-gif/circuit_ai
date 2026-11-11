"""
Relaxation oscillators (Phase 4, step 4): the op-amp astable multivibrator, a.k.a. the square-wave generator.

WHY det A(s) = 0 DOES NOT WORK HERE
The RC and LC oscillators of P4-1..P4-3 are linear circuits sitting exactly on the edge of stability, so a
pole pair on the jw axis describes them completely and the answer is a sine wave. A relaxation oscillator is
not like that at all. Its op-amp is always hard against a rail, the circuit is linear only BETWEEN switching
instants, and the waveform is a square wave with an exponential ramp on the capacitor. There is no sinusoid
and no pole on the axis: the frequency comes from how long the capacitor takes to crawl from one trip point
to the other.

THE METHOD, which is the same assume-solve-check as every nonlinear device in this project
An inverting Schmitt trigger with the capacitor at its − input has exactly two states, and in each state the
circuit is a plain RC charging toward a fixed target:

    state         what the op-amp is        the capacitor charges        and switches when
    output HIGH   a +Vsat source at "out"   toward Vf_hi with tau_hi     v_C reaches V+ = V_UT
    output LOW    a -Vsat source at "out"   toward Vf_lo with tau_lo     v_C reaches V+ = V_LT

Each state's three numbers are found from the circuit, not from a formula:
    V_UT, V_LT   the + input voltage in that state  (the Schmitt trip points of P3-6)
    Vf           the open-circuit voltage at the capacitor's terminals, with the capacitor removed
    tau = R_th*C where R_th is what a test current sees across those terminals, sources off  (Thevenin)

Then the time for an RC charging from v0 toward Vf, stopping at v1, is the standard exponential solve:

    v(t) = Vf + (v0 - Vf)*exp(-t/tau)   ->   t = tau * ln((v0 - Vf)/(v1 - Vf))

so  T_high = tau_hi * ln((V_LT - Vf_hi)/(V_UT - Vf_hi))   and   T_low = tau_lo * ln((V_UT - Vf_lo)/(V_LT - Vf_lo)),
with T = T_high + T_low,  f = 1/T  and  duty = T_high/T.

If one of those logarithms has a non-positive argument, the capacitor is charging toward a target it will
never pass, so that state never ends: the output LATCHES instead of oscillating. The solver says so rather
than returning a number.

For the textbook circuit (symmetric rails, R1 from the + input to ground, R2 as the positive feedback) this
reduces to the familiar result, with beta = R1/(R1+R2):

    V_UT, V_LT = ±beta*Vsat,  Vf = ±Vsat,  tau = R*C   ->   T = 2*R*C*ln((1+beta)/(1-beta))
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import sympy as sp

from schemas import AnalysisType, Component, Netlist, SolveResult, StepTrace, is_symbolic
from solvers.base import BaseSolver, NotSupportedError, SolverError, fmt_num, fmt_si, sig_figs
from solvers.comparator import rails
from solvers.mna import GROUND, MNA

KINDS = {"fosc", "f0", "period", "duty", "T_high", "T_low", "VTH", "VTL", "hysteresis", "barkhausen"}
_Q = re.compile(r"^(\w+)\((\w+)(?:,(\w+))?\)$")
_ASTABLE = re.compile(r"\bastable\b|\bsquare[- ]wave\s+(?:generator|oscillator)\b|\brelaxation\s+oscillator\b|"
                      r"\bmultivibrator\b|\bfree[- ]running\b", re.IGNORECASE)


def _has_path(nl: Netlist, a: str, b: str, types=("R",)) -> bool:
    """Is there a path from a to b through the given element types, not via ground?"""
    seen, stack = {a}, [a]
    while stack:
        n = stack.pop()
        for c in nl.components:
            if c.type.value in types and n in c.nodes[:2]:
                other = c.nodes[1] if c.nodes[0] == n else c.nodes[0]
                if other == b:
                    return True
                if other not in seen and other != GROUND:
                    seen.add(other)
                    stack.append(other)
    return False


def astable_shape(nl: Netlist) -> Optional[Component]:
    """The op-amp of an astable multivibrator, if this circuit is one: positive feedback to the + input AND
    the timing capacitor on the − input. Used by this solver to claim the circuit, and by the oscillator
    solver to decline it with a useful reason instead of a confusing one."""
    for op in [c for c in nl.components if c.type.value == "OPAMP"]:
        out, inp, inm = op.nodes[:3]
        cap = [c for c in nl.components if c.type.value == "C" and inm in c.nodes[:2]]
        if cap and _has_path(nl, out, inp) and _has_path(nl, out, inm):
            return op
    return None


class RelaxationSolver(BaseSolver):
    name = "relaxation"
    method = ("two rail states, each a plain RC charging toward its own Thevenin target: the time in each "
              "state is tau*ln((v_start - Vf)/(v_stop - Vf)) between the Schmitt trip points")

    def supports(self, nl: Netlist) -> Tuple[bool, str]:
        ops = [c for c in nl.components if c.type.value == "OPAMP"]
        if not ops or nl.analysis == AnalysisType.CONCEPTUAL:
            return False, ""
        kinds = {(_Q.match(q.replace(" ", "")).group(1) if _Q.match(q.replace(" ", "")) else q.replace(" ", ""))
                 for q in nl.query}
        timing = [c for c in nl.components if c.type.value == "C" and ops[0].nodes[2] in c.nodes[:2]]
        if astable_shape(nl) is None:
            return False, ""
        if not (kinds & KINDS or _ASTABLE.search(nl.question or "")):
            return False, ""
        if any(c.type.value in ("D", "Q", "M") for c in nl.components):
            return False, ("Relaxation oscillators with diodes or transistors (separate charge and discharge "
                           "paths, or a 555 built from parts) are not supported yet")
        if any(c.type.value in ("V", "I") for c in nl.components):
            return False, ("An astable multivibrator has no input source: it free-runs. Remove the source "
                           "(the supply rails belong in the assumptions, not the netlist).")
        if len(timing) > 1:
            return False, (f"More than one capacitor at the − input ({', '.join(c.id for c in timing)}): "
                           "the timing capacitor cannot be identified")
        if any(is_symbolic(c.value) for c in nl.components if c.value not in (None, "")):
            return False, "Relaxation oscillator values must all be numbers"
        return True, ""

    def solve(self, nl: Netlist, conventions: Dict) -> SolveResult:
        ok, why = self.supports(nl)
        if not ok:
            raise NotSupportedError(why or "Not a relaxation oscillator")
        return _RelaxJob(nl, conventions).run()


class _RelaxJob:
    def __init__(self, nl: Netlist, conventions: Dict):
        self.nl, self.conv = nl, conventions or {}
        self.sig = sig_figs(conventions)
        self.trace = StepTrace()
        self.res = SolveResult(method=RelaxationSolver.method)
        self.queries = [q.replace(" ", "") for q in nl.query]
        self.op = next(c for c in nl.components if c.type.value == "OPAMP")
        self.cap = next(c for c in nl.components if c.type.value == "C" and self.op.nodes[2] in c.nodes[:2])
        self.pos, self.neg, notes = rails(nl, conventions)
        for n in notes:
            self.trace.notes.append(n)
            self.res.verification_notes.append(n)

    # ------------------------------------------------------------------
    def _state_circuit(self, high: bool, zero_sources: bool = False) -> List[Component]:
        """The circuit in one rail state: the op-amp becomes a voltage source at that rail (0 V when the
        sources are turned off, which is what Thevenin asks for), and the timing capacitor is removed."""
        v = 0.0 if zero_sources else (self.pos if high else -self.neg)
        out = self.op.nodes[0]
        comps = [c for c in self.nl.components
                 if c.id not in (self.op.id, self.cap.id) and c.type.value != "C"]
        comps.append(Component(id=f"{self.op.id}_sat", type="V", nodes=[out, GROUND], value=repr(v)))
        return comps

    def _dc(self, comps: List[Component]) -> MNA:
        m = MNA(comps, "dc", conventions=self.conv)
        m.solve()
        return m

    def _v(self, m: MNA, node: str) -> float:
        if node == GROUND:
            return 0.0
        e = m.evaluate(m.V[node])
        if getattr(e, "free_symbols", set()):
            raise SolverError("the circuit does not reduce to numbers")
        return float(sp.N(e))

    def _state(self, high: bool) -> Tuple[float, float, float]:
        """(trip point, Thevenin target at the capacitor, tau) for one rail state."""
        a, b = self.cap.nodes[0], self.cap.nodes[1]
        m = self._dc(self._state_circuit(high))
        trip = self._v(m, self.op.nodes[1])        # the + input: where this state ends
        vf = self._v(m, a) - self._v(m, b)                 # open-circuit voltage across the capacitor
        # R_th: 1 A test current across the capacitor's terminals, every source off
        test = self._state_circuit(high, zero_sources=True)
        test.append(Component(id="I_test", type="I", nodes=[b, a], value="1"))
        mt = self._dc(test)
        rth = abs(self._v(mt, a) - self._v(mt, b))
        C = float(sp.N(MNA(self.nl.components, "dc", conventions=self.conv).value(self.cap)))
        if rth <= 0:
            raise SolverError(f"the capacitor {self.cap.id} sees no resistance, so it cannot set a time "
                              f"constant (is the timing resistor missing?)")
        return trip, vf, rth * C

    # ------------------------------------------------------------------
    def _time(self, tau: float, v0: float, v1: float, vf: float, what: str) -> float:
        """How long an RC charging from v0 toward vf takes to reach v1."""
        num, den = v0 - vf, v1 - vf
        if den == 0 or num / den <= 0 or num / den < 1:
            raise SolverError(
                f"the output latches instead of oscillating: in the {what} state the capacitor charges "
                f"toward {fmt_si(vf, 'V', self.sig)} but has to reach {fmt_si(v1, 'V', self.sig)} starting "
                f"from {fmt_si(v0, 'V', self.sig)}, which it never does. For an astable the trip points must "
                f"both lie between the two rails.")
        return tau * math.log(num / den)

    def _textbook(self, f0: float) -> None:
        """The book's formula, for the standard symmetric circuit."""
        if abs(self.pos - self.neg) > 1e-12:
            return
        inp, inm, out = self.op.nodes[1], self.op.nodes[2], self.op.nodes[0]
        r1 = [c for c in self.nl.components if c.type.value == "R" and set(c.nodes[:2]) == {inp, GROUND}]
        r2 = [c for c in self.nl.components if c.type.value == "R" and set(c.nodes[:2]) == {inp, out}]
        rt = [c for c in self.nl.components if c.type.value == "R" and set(c.nodes[:2]) == {inm, out}]
        if not (len(r1) == len(r2) == len(rt) == 1):
            return
        m = MNA(self.nl.components, "dc", conventions=self.conv)
        try:
            R1, R2 = float(sp.N(m.value(r1[0]))), float(sp.N(m.value(r2[0])))
            R, C = float(sp.N(m.value(rt[0]))), float(sp.N(m.value(self.cap)))
        except SolverError:
            return
        beta = R1 / (R1 + R2)
        T = 2 * R * C * math.log((1 + beta) / (1 - beta))
        self.res.numeric["fosc.approx"] = 1 / T
        self.trace.add("The book's formula for this circuit, for comparison",
                       f"β = R1/(R1+R2) = {fmt_num(beta, 4)};  T = 2·R·C·ln((1+β)/(1−β)) "
                       f"= 2·{fmt_si(R, 'Ω')}·{fmt_si(C, 'F')}·ln({fmt_num((1 + beta) / (1 - beta), 4)})",
                       f"T = {fmt_si(T, 's', self.sig)},  f = {fmt_si(1 / T, 'Hz', self.sig)} "
                       f"({'agrees' if abs(1 / T - f0) < 0.01 * f0 else 'differs'} with the circuit solve)")

    # ------------------------------------------------------------------
    def run(self) -> SolveResult:
        self.trace.add(f"{self.op.id} has the timing capacitor on its − input and positive feedback to "
                           f"its + input: an astable multivibrator",
                           "the output is always at a rail, so the circuit is linear only between switches",
                           "two states, each an RC charging between the Schmitt trip points")
        v_ut, vf_hi, tau_hi = self._state(True)
        v_lt, vf_lo, tau_lo = self._state(False)
        self.trace.add("Find each state's trip point, target and time constant from the circuit",
                       f"output HIGH (+{fmt_si(self.pos, 'V')}): switches at V+ = {fmt_si(v_ut, 'V', self.sig)}"
                       f", charges toward {fmt_si(vf_hi, 'V', self.sig)}, τ = {fmt_si(tau_hi, 's', self.sig)}\n"
                       f"          output LOW  (−{fmt_si(self.neg, 'V')}): switches at V+ = "
                       f"{fmt_si(v_lt, 'V', self.sig)}, charges toward {fmt_si(vf_lo, 'V', self.sig)}, "
                       f"τ = {fmt_si(tau_lo, 's', self.sig)}",
                       f"the capacitor shuttles between {fmt_si(v_lt, 'V', self.sig)} and "
                       f"{fmt_si(v_ut, 'V', self.sig)}")
        if v_ut <= v_lt:
            raise SolverError("the two trip points are not in the right order (the upper one must be above "
                              "the lower one): check that the feedback resistor goes to the + input")

        t_hi = self._time(tau_hi, v_lt, v_ut, vf_hi, "high")
        t_lo = self._time(tau_lo, v_ut, v_lt, vf_lo, "low")
        T = t_hi + t_lo
        f0 = 1 / T
        self.trace.add("Time each state with the exponential charging law",
                       f"t = τ·ln((v_start − Vf)/(v_stop − Vf))\n"
                       f"          T_high = {fmt_si(tau_hi, 's', self.sig)}·ln(("
                       f"{fmt_num(v_lt, self.sig)} − {fmt_num(vf_hi, self.sig)})/({fmt_num(v_ut, self.sig)} − "
                       f"{fmt_num(vf_hi, self.sig)})) = {fmt_si(t_hi, 's', self.sig)}\n"
                       f"          T_low  = {fmt_si(tau_lo, 's', self.sig)}·ln(("
                       f"{fmt_num(v_ut, self.sig)} − {fmt_num(vf_lo, self.sig)})/({fmt_num(v_lt, self.sig)} − "
                       f"{fmt_num(vf_lo, self.sig)})) = {fmt_si(t_lo, 's', self.sig)}",
                       f"T = {fmt_si(T, 's', self.sig)},  f = {fmt_si(f0, 'Hz', self.sig)},  duty = "
                       f"{fmt_num(100 * t_hi / T, 4)} %")
        self._textbook(f0)

        for key, val in (("fosc", f0), ("period", T), ("T_high", t_hi), ("T_low", t_lo),
                         ("duty", t_hi / T), ("VTH", v_ut), ("VTL", v_lt), ("hysteresis", v_ut - v_lt)):
            self.res.numeric[key] = val

        for q in self.queries:
            mq = _Q.match(q)
            kind = mq.group(1) if mq else q
            if kind in ("fosc", "f0"):
                self.res.answers[q] = fmt_si(f0, "Hz", self.sig)
            elif kind == "period":
                self.res.answers[q] = fmt_si(T, "s", self.sig)
            elif kind == "T_high":
                self.res.answers[q] = fmt_si(t_hi, "s", self.sig) + " (output at the + rail)"
            elif kind == "T_low":
                self.res.answers[q] = fmt_si(t_lo, "s", self.sig) + " (output at the − rail)"
            elif kind == "duty":
                self.res.answers[q] = f"{fmt_num(100 * t_hi / T, 4)} %"
            elif kind == "VTH":
                self.res.answers[q] = fmt_si(v_ut, "V", self.sig) + " (the capacitor stops charging up here)"
            elif kind == "VTL":
                self.res.answers[q] = fmt_si(v_lt, "V", self.sig) + " (and stops discharging here)"
            elif kind == "hysteresis":
                self.res.answers[q] = (fmt_si(v_ut - v_lt, "V", self.sig) + " wide (from "
                                       + fmt_si(v_lt, "V", self.sig) + " to " + fmt_si(v_ut, "V", self.sig) + ")")
            elif kind == "barkhausen":
                self.res.answers[q] = ("not applicable: a relaxation oscillator is not a linear feedback loop. "
                                       "It switches between two saturated states, so there is no loop gain of "
                                       "1 at a single frequency -- the period comes from the RC charging time.")
                self.res.numeric[q] = None
            elif kind in ("V", "I", "P"):
                self.res.answers[q] = ("the output is a square wave between the two rails and the capacitor a "
                                       "ramp between the trip points: ask for fosc, period, duty, T_high, "
                                       "T_low, VTH or VTL")
                self.res.numeric[q] = None
            else:
                self.res.answers[q] = (f"Could not compute: '{q}' is not available for a relaxation "
                                       "oscillator (ask for fosc, period, duty, T_high, T_low, VTH, VTL)")
                self.res.numeric[q] = None
            if kind in ("fosc", "f0", "period", "T_high", "T_low", "duty", "VTH", "VTL", "hysteresis"):
                self.res.numeric[q] = self.res.numeric["fosc" if kind == "f0" else kind]

        self.res.symbolic["relax.states"] = (f"high: trip {v_ut:.9g} V, target {vf_hi:.9g} V, tau {tau_hi:.9g} s; "
                                             f"low: trip {v_lt:.9g} V, target {vf_lo:.9g} V, tau {tau_lo:.9g} s")
        self.res.trace = self.trace
        if self.queries and all(self.res.answers.get(q, "").startswith("Could not compute")
                                for q in self.queries):
            raise SolverError(" | ".join(self.res.answers.values()))
        return self.res