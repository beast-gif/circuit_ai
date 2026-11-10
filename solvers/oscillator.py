"""
Oscillators (Phase 4, step 1): Wien-bridge and RC phase-shift oscillators built around an op-amp.

THE IDEA, AND WHY IT NEEDS NO FORMULA
An oscillator has no input. Every other circuit in this project is solved as A·x = b with a source on the
right-hand side; here b = 0, so the only way to get a non-zero output is for the matrix itself to be
singular:

        A(s)·x = 0  has a non-zero solution   <=>   det A(s) = 0

det A(s) is the characteristic polynomial of the circuit, and its roots are the circuit's poles. A pole pair
at s = ±jw0 (real part exactly zero) is a sine wave that neither grows nor decays: that IS oscillation. So
the whole problem is "find where det A(jw) = 0", and it is solved by splitting that one complex equation
into its real and imaginary parts:

        Im[det A(jw)] = 0   ->   the PHASE condition   (the loop comes back in phase: angle = 0)
        Re[det A(jw)] = 0   ->   the MAGNITUDE condition (the loop gain is exactly 1)

Those two equations are the Barkhausen criterion, written out exactly. Two equations, two unknowns: the
frequency w0 and whatever is still unknown about the gain (the feedback resistor). Nothing about Wien
bridges or sqrt(6) is built in -- those come out of the determinant.

WHAT THE REAL PART OF THE POLE MEANS, when every value is numeric:
        real part  = 0   the amplitude is constant: it oscillates, right at the Barkhausen condition
        real part  > 0   the amplitude grows: it oscillates, and the rails (or an AGC lamp / diodes) limit it
        real part  < 0   the amplitude decays: it does NOT oscillate, the gain is too low
and then the solver re-solves with the feedback resistor as a symbol to say what value WOULD oscillate.

For the classic circuits this reproduces the textbook results exactly:
    Wien bridge, equal R and C      f0 = 1/(2*pi*R*C),        needs Rf/R1 = 2  (amplifier gain 3)
    RC phase shift, 3 equal CR legs f0 = 1/(2*pi*R*C*sqrt(6)), needs Rf/R  = 29
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import sympy as sp

from schemas import AnalysisType, Component, Netlist, SolveResult, StepTrace, is_symbolic
from solvers.base import BaseSolver, NotSupportedError, SolverError, fmt_num, fmt_si, sig_figs
from solvers.mna import GROUND, MNA, s as S

W = sp.Symbol("w_osc", positive=True)
KINDS = {"fosc", "f0", "barkhausen", "gain_condition", "oscillates"}
_Q = re.compile(r"^(\w+)\((\w+)(?:,(\w+))?\)$")
_OSC_WORD = re.compile(r"\boscillat|\bbarkhausen\b|\bwien\b|\bphase[- ]shift\b|\bcolpitts\b|\bhartley\b|"
                       r"\btwin[- ]?t\b", re.IGNORECASE)
SUSTAINED = 1e-6            # |real part| <= this * |imaginary part| counts as "on the axis"


def _free_values(nl: Netlist) -> List[Component]:
    """Components whose value is a symbol rather than a number ("Rf", "R"), i.e. things to solve FOR."""
    return [c for c in nl.components if c.value not in (None, "") and is_symbolic(c.value)]


class OscillatorSolver(BaseSolver):
    name = "oscillator"
    method = ("det A(s) = 0 for the source-free circuit (the characteristic equation); s = jw splits it into "
              "the Barkhausen phase and magnitude conditions, which give w0 and the gain needed")

    def supports(self, nl: Netlist) -> Tuple[bool, str]:
        ops = [c for c in nl.components if c.type.value == "OPAMP"]
        kinds = {(_Q.match(q.replace(" ", "")).group(1) if _Q.match(q.replace(" ", "")) else q.replace(" ", ""))
                 for q in nl.query}
        asked = bool(kinds & KINDS) and (kinds & {"fosc", "barkhausen", "gain_condition", "oscillates"}
                                         or _OSC_WORD.search(nl.question or ""))
        sources = [c for c in nl.components if c.type.value in ("V", "I")]
        looks_like = bool(ops) and not sources and bool(_OSC_WORD.search(nl.question or ""))
        if not (asked or looks_like):
            return False, ""
        if any(c.type.value in ("D", "Q", "M") for c in nl.components):
            return False, ("Oscillators with diodes or transistors in the loop are not supported yet: "
                           "this step covers op-amp RC oscillators")
        if not ops:
            return False, "An oscillator needs the amplifier in the netlist (an op-amp in this step)"
        if sources:
            return False, ("An oscillator has no input source. Remove the source from the netlist: the "
                           "oscillation comes from the feedback loop itself")
        if nl.analysis == AnalysisType.CONCEPTUAL:
            return False, "Conceptual question: no circuit to solve"
        return True, ""

    def solve(self, nl: Netlist, conventions: Dict) -> SolveResult:
        ok, why = self.supports(nl)
        if not ok:
            raise NotSupportedError(why or "Not an oscillator circuit")
        return _OscJob(nl, conventions).run()


class _OscJob:
    def __init__(self, nl: Netlist, conventions: Dict):
        self.nl, self.conv = nl, conventions or {}
        self.sig = sig_figs(conventions)
        self.trace = StepTrace()
        self.res = SolveResult(method=OscillatorSolver.method)
        self.queries = [q.replace(" ", "") for q in nl.query]
        self.ops = [c for c in nl.components if c.type.value == "OPAMP"]

    # ------------------------------------------------------------------
    def _char_poly(self, comps: List[Component]) -> sp.Expr:
        """The characteristic polynomial: the numerator of det A(s) for the source-free circuit."""
        m = MNA(comps, "ac", conventions=self.conv)
        A, b = m.system()
        if any(x != 0 for x in b):
            raise SolverError("the circuit still has a source in it, so it is not source-free")
        det = sp.cancel(sp.together(A.det()))
        num = sp.numer(det)
        if num == 0:
            raise SolverError("det A(s) is identically zero: the circuit is under-determined (check that "
                              "every node is connected and the op-amp has feedback)")
        return sp.expand(num)

    def _split(self, poly: sp.Expr) -> Tuple[sp.Expr, sp.Expr]:
        """Re and Im of the characteristic polynomial at s = jw (w real and positive)."""
        at_jw = sp.expand(poly.subs(S, sp.I * W))
        re_part = sp.simplify(sp.re(at_jw))
        im_part = sp.simplify(sp.im(at_jw))
        return re_part, im_part

    # ------------------------------------------------------------------
    def _feedback_resistor(self) -> Optional[Component]:
        """The resistor from the output to the − input: the one that sets the amplifier's gain."""
        for op in self.ops:
            out, inm = op.nodes[0], op.nodes[2]
            for c in self.nl.components:
                if c.type.value == "R" and set(c.nodes[:2]) == {out, inm}:
                    return c
        return None

    def _gain_partner(self, rf: Component) -> Optional[Component]:
        """The other resistor at the − input: the ratio Rf/this is the gain condition textbooks quote."""
        op = self.ops[0]
        inm = op.nodes[2]
        for c in self.nl.components:
            if c.type.value == "R" and c.id != rf.id and inm in c.nodes[:2]:
                return c
        return None

    # ------------------------------------------------------------------
    def _solve_symbolic(self, comps: List[Component], unknown: Component) -> Tuple[float, float]:
        """Both Barkhausen equations at once: returns (w0, the value the unknown must have)."""
        poly = self._char_poly(comps)
        re_part, im_part = self._split(poly)
        x = sp.Symbol(str(unknown.value), positive=True)
        self.trace.add("Set the characteristic equation det A(jw) = 0 and split it into real and imaginary parts",
                       f"Im: {sp.nsimplify(im_part)} = 0  (phase condition)\n"
                       f"          Re: {sp.nsimplify(re_part)} = 0  (magnitude condition)",
                       f"two equations for the two unknowns w and {unknown.id}")
        sols = sp.solve([re_part, im_part], [W, x], dict=True)
        good = [t for t in sols if t.get(W) and t[W].is_real and float(t[W]) > 0
                and t.get(x) is not None and sp.im(t[x]) == 0 and float(t[x]) > 0]
        if not good:
            raise SolverError(f"no positive solution of the two Barkhausen equations: this circuit cannot "
                              f"oscillate for any value of {unknown.id}")
        t = good[0]
        return float(t[W]), float(t[x])

    def _poles(self, poly: sp.Expr) -> List[complex]:
        p = sp.Poly(poly, S)
        if p.degree() < 1:
            raise SolverError("the characteristic equation has no s in it: there is no frequency to find "
                              "(the loop has no reactive elements)")
        return [complex(r) for r in sp.nroots(p, n=20)]

    # ------------------------------------------------------------------
    def _textbook(self, w0: float) -> None:
        """Name the topology and compare with the formula from the book, when the parts are all equal."""
        rs = [c for c in self.nl.components if c.type.value == "R"]
        cs = [c for c in self.nl.components if c.type.value == "C"]
        rf = self._feedback_resistor()
        loop_r = [c for c in rs if rf is None or c.id != rf.id]
        vals_r = {str(c.value) for c in loop_r}
        vals_c = {str(c.value) for c in cs}
        if len(vals_c) != 1 or len(vals_r) > 2:
            return
        try:
            R = float(sp.N(MNA(self.nl.components, "ac", conventions=self.conv).value(loop_r[0])))
            C = float(sp.N(MNA(self.nl.components, "ac", conventions=self.conv).value(cs[0])))
        except (SolverError, IndexError):
            return
        if len(cs) == 2:
            name, f_book, gain = "Wien bridge", 1 / (2 * sp.pi * R * C), "Rf/R1 = 2 (amplifier gain 3)"
        elif len(cs) >= 3:
            name, f_book = "RC phase-shift", 1 / (2 * sp.pi * R * C * sp.sqrt(6))
            gain = "Rf/R = 29"
        else:
            return
        f_book = float(f_book)
        self.res.numeric["fosc.approx"] = f_book
        self.trace.add(f"This is the classic {name} oscillator, so the book's formula applies",
                       f"f0 = 1/(2·π·R·C{'·√6' if len(cs) >= 3 else ''}) with R = {fmt_si(R, 'Ω')}, "
                       f"C = {fmt_si(C, 'F')};  condition {gain}",
                       f"textbook f0 = {fmt_si(f_book, 'Hz', self.sig)} "
                       f"({'agrees' if abs(f_book - w0 / (2 * 3.141592653589793)) < 0.01 * f_book else 'differs'}"
                       " with the determinant)")

    # ------------------------------------------------------------------
    def run(self) -> SolveResult:
        comps = list(self.nl.components)
        for op in self.ops:
            self.trace.add(f"{op.id} is the amplifier; the loop has no input source, so the circuit is "
                           f"source-free and A(s)·x = 0",
                           "a non-zero output with no input is only possible where det A(s) = 0",
                           "that determinant is the characteristic equation of the circuit")

        unknowns = _free_values(self.nl)
        if len(unknowns) > 1:
            raise SolverError("More than one component value is left as a symbol ("
                              + ", ".join(c.id for c in unknowns)
                              + "). The two Barkhausen equations can fix the frequency and ONE unknown: "
                                "give numbers for the rest.")

        w0 = x_needed = None
        unknown = unknowns[0] if unknowns else None
        grows = None
        if unknown is not None:
            w0, x_needed = self._solve_symbolic(comps, unknown)
            self.trace.add("Solve the two equations together",
                           f"w0 = {fmt_num(w0, self.sig)} rad/s,  {unknown.id} = "
                           f"{fmt_si(x_needed, 'Ω' if unknown.type.value == 'R' else '', self.sig)}",
                           f"the loop oscillates at f0 = {fmt_si(w0 / (2 * 3.141592653589793), 'Hz', self.sig)} "
                           f"when {unknown.id} has exactly that value")
        else:
            poly = self._char_poly(comps)
            self.trace.add("Write out det A(s) = 0", f"{sp.nsimplify(sp.factor(poly))} = 0",
                           "its roots are the poles of the circuit")
            poles = self._poles(poly)
            osc = [p for p in poles if abs(p.imag) > 1e-12]
            if not osc:
                raise SolverError("every pole is real, so nothing in this circuit rings: it cannot oscillate")
            p = max(osc, key=lambda z: (z.real, abs(z.imag)))
            w0 = abs(p.imag)
            grows = p.real
            kind = ("exactly on the jw axis: the amplitude stays constant, so this is the Barkhausen condition"
                    if abs(p.real) <= SUSTAINED * abs(p.imag) else
                    "in the right half plane: the amplitude GROWS, so it oscillates and the supply rails "
                    "(or a limiting network) set the final amplitude" if p.real > 0 else
                    "in the left half plane: the amplitude DECAYS, so it does not oscillate -- the gain is "
                    "too low")
            self.trace.add("Look at the pole pair",
                           f"s = {fmt_num(p.real, self.sig)} ± j{fmt_num(abs(p.imag), self.sig)} rad/s",
                           f"the real part is {kind}")
            if p.real < -SUSTAINED * abs(p.imag):
                rf = self._feedback_resistor()
                if rf is not None:
                    sym = [Component(**{**c.model_dump(), "value": "Rf_needed"}) if c.id == rf.id else c
                           for c in comps]
                    try:
                        w_need, rf_need = self._solve_symbolic(sym, next(c for c in sym if c.id == rf.id))
                        self.res.numeric["gain_condition"] = rf_need
                        self.trace.add(f"Ask what {rf.id} would have to be",
                                       f"re-solve the same two equations with {rf.id} as the unknown",
                                       f"{rf.id} = {fmt_si(rf_need, 'Ω', self.sig)} would put the poles on the "
                                       f"axis (f0 = {fmt_si(w_need / (2 * 3.141592653589793), 'Hz', self.sig)})")
                    except SolverError:
                        pass

        f0 = w0 / (2 * 3.141592653589793)
        self._textbook(w0)

        # ---------------- answers
        for q in self.queries:
            mq = _Q.match(q)
            kind = mq.group(1) if mq else q
            if kind in ("fosc", "f0"):
                self.res.answers[q] = fmt_si(f0, "Hz", self.sig) + (
                    "" if grows is None or abs(grows) <= SUSTAINED * w0 else
                    " (the amplitude grows until the rails limit it)" if grows > 0 else
                    " (the ringing frequency -- as built, the amplitude decays and it does not oscillate)")
                self.res.numeric[q] = f0
                self.res.numeric.setdefault("fosc", f0)
            elif kind == "barkhausen":
                self.res.answers[q] = (f"the loop gain must be 1 with zero phase shift; here that happens at "
                                       f"f0 = {fmt_si(f0, 'Hz', self.sig)}"
                                       + (f" with {unknown.id} = {fmt_si(x_needed, 'Ω', self.sig)}"
                                          if unknown is not None else ""))
                self.res.numeric[q] = f0
            elif kind == "gain_condition":
                if unknown is not None:
                    partner = self._gain_partner(unknown) if unknown.type.value == "R" else None
                    ratio = ""
                    if partner is not None:
                        try:
                            pv = float(sp.N(MNA(comps, "ac", conventions=self.conv).value(partner)))
                            ratio = f"  ({unknown.id}/{partner.id} = {fmt_num(x_needed / pv, self.sig)})"
                        except SolverError:
                            ratio = ""
                    self.res.answers[q] = fmt_si(x_needed, "Ω", self.sig) + ratio
                    self.res.numeric[q] = x_needed
                elif "gain_condition" in self.res.numeric:
                    self.res.answers[q] = (fmt_si(self.res.numeric["gain_condition"], "Ω", self.sig)
                                           + " (the value the feedback resistor needs)")
                else:
                    self.res.answers[q] = ("already satisfied: the poles sit on the jw axis with the values "
                                           "given" if grows is not None and abs(grows) <= SUSTAINED * w0 else
                                           "could not be determined from the values given")
                    self.res.numeric[q] = None
            elif kind == "oscillates":
                yes = unknown is not None or (grows is not None and grows > -SUSTAINED * w0)
                self.res.answers[q] = ("yes" if yes else "no, the loop gain is below 1 and the amplitude decays")
                self.res.numeric[q] = None
            else:
                self.res.answers[q] = (f"Could not compute: '{q}' is not available for an oscillator "
                                       "(ask for fosc, barkhausen, gain_condition or oscillates)")
                self.res.numeric[q] = None

        self.res.symbolic["osc.char_poly"] = str(sp.nsimplify(self._char_poly(comps))) if unknown is None else ""
        self.res.numeric["w_osc"] = w0
        if grows is not None:
            self.res.numeric["pole.re"] = grows
        self.res.trace = self.trace
        if self.queries and all(self.res.answers.get(q, "").startswith("Could not compute")
                                for q in self.queries):
            raise SolverError(" | ".join(self.res.answers.values()))
        return self.res