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

import math
import re
from typing import Dict, List, Optional, Tuple

import sympy as sp

from schemas import AnalysisType, Component, Netlist, SolveResult, StepTrace, is_symbolic
from solvers.base import BaseSolver, NotSupportedError, SolverError, fmt_num, fmt_si, sig_figs
from solvers.mna import GROUND, MNA, s as S
from solvers.relaxation import astable_shape

W = sp.Symbol("w_osc", positive=True)
AMPLIFIERS = ("OPAMP", "E", "G", "F", "H")      # an op-amp, or a transistor's small-signal model
_GM_KEYS = ("gm", "g_m", "transconductance")
_IC_KEYS = ("ic", "i_c", "id", "i_d", "iq", "i_q")
_BETA_KEYS = ("beta", "hfe", "bf", "b")
KINDS = {"fosc", "f0", "barkhausen", "gain_condition", "oscillates"}
_Q = re.compile(r"^(\w+)\((\w+)(?:,(\w+))?\)$")
_OSC_WORD = re.compile(r"\boscillat|\bbarkhausen\b|\bwien\b|\bphase[- ]shift\b|\bcolpitts\b|\bhartley\b|"
                       r"\btwin[- ]?t\b", re.IGNORECASE)
SUSTAINED = 1e-6            # |real part| <= this * |imaginary part| counts as "on the axis"


def _params(c: Component) -> Dict[str, float]:
    out = {}
    for k, v in (c.params or {}).items():
        try:
            out[str(k).strip().lower().replace(" ", "_").replace("-", "_")] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _gm_raw(c: Component) -> Optional[str]:
    """A gm given as a SYMBOL rather than a number: params {"gm": "gm"} is the natural way to write
    "what transconductance does this transistor need?", so it is accepted as the unknown."""
    for k, v in (c.params or {}).items():
        if str(k).strip().lower().replace(" ", "_").replace("-", "_") in _GM_KEYS and is_symbolic(str(v)):
            return str(v).strip()
    return None


def _gm_of(c: Component, VT: float = 0.025) -> Optional[float]:
    """gm straight from the question: given directly, or from the bias current (gm = I_C/V_T for a BJT)."""
    p = _params(c)
    for k in _GM_KEYS:
        if p.get(k):
            return p[k]
    for k in _IC_KEYS:
        if p.get(k):
            if c.type.value == "Q":
                return p[k] / VT                              # gm = I_C / V_T
            kp = next((p[x] for x in ("k", "kprime", "k_prime", "kn", "kp", "k2") if p.get(x)), None)
            if kp:
                return math.sqrt(2 * kp * p[k])                # gm = sqrt(2·k·I_D)
    return None


def _small_signal(nl: Netlist, conv: Dict, trace: StepTrace) -> List[Component]:
    """Replace each transistor by its small-signal model, which is what the AC equivalent circuit of an LC
    oscillator is drawn with anyway:

        BJT  : a transconductance  i_c = gm·v_be  from collector to emitter, plus r_π = β/gm base-emitter
        FET  : a transconductance  i_d = gm·v_gs  from drain to source (the gate draws no current)

    The DC bias is NOT solved here: an oscillator netlist is the AC equivalent and carries no supply, so the
    question has to give gm (or the bias current it comes from).
    """
    VT = float(((conv or {}).get("semiconductor") or {}).get("VT", 0.025))
    out: List[Component] = []
    for c in nl.components:
        if c.type.value not in ("Q", "M"):
            out.append(c)
            continue
        gm = _gm_of(c, VT)
        sym = _gm_raw(c)
        if gm is None and sym is None:
            raise SolverError(f"{c.id}: no gm given, so its small-signal model cannot be built")
        gm_val = sym if sym else repr(gm)
        gm_txt = sym if sym else fmt_si(gm, "S")
        p = _params(c)
        if c.type.value == "Q":
            col, base, emit = c.nodes
            out.append(Component(id=f"{c.id}_gm", type="G", nodes=[col, emit, base, emit], value=gm_val))
            beta = next((p[k] for k in _BETA_KEYS if p.get(k)), None)
            note = ""
            if beta and gm:
                out.append(Component(id=f"{c.id}_rpi", type="R", nodes=[base, emit], value=repr(beta / gm)))
                note = f", r_π = β/gm = {fmt_si(beta / gm, 'Ω')}"
            trace.add(f"Replace {c.id} by its small-signal model (the AC equivalent circuit)",
                      f"i_c = gm·v_be with gm = {gm_txt}{note}",
                      "the collector becomes a current source controlled by the base-emitter voltage"
                      + ("" if beta else "; with no β given, r_π is taken as infinite"))
        else:
            drain, gate, src = c.nodes
            out.append(Component(id=f"{c.id}_gm", type="G", nodes=[drain, src, gate, src], value=gm_val))
            trace.add(f"Replace {c.id} by its small-signal model (the AC equivalent circuit)",
                      f"i_d = gm·v_gs with gm = {gm_txt}",
                      "the gate draws no current, so there is no r_π")
    return out


def _unit(c: Optional[Component]) -> str:
    """The unit the solved-for value is printed in: a resistor in Ω, a transconductance in S."""
    if c is None:
        return ""
    return {"R": "Ω", "C": "F", "L": "H", "G": "S", "E": "", "F": "", "H": "Ω"}.get(c.type.value, "")


# public alias: the verifier needs the same substitution before it can hand the circuit to ngspice
small_signal_circuit = _small_signal


def _free_values(nl: Netlist) -> List[Component]:
    """Components whose value is a symbol rather than a number ("Rf", "R"), i.e. things to solve FOR."""
    return [c for c in nl.components if c.value not in (None, "") and is_symbolic(c.value)]


class OscillatorSolver(BaseSolver):
    name = "oscillator"
    method = ("det A(s) = 0 for the source-free circuit (the characteristic equation); s = jw splits it into "
              "the Barkhausen phase and magnitude conditions, which give w0 and the gain needed")

    def supports(self, nl: Netlist) -> Tuple[bool, str]:
        ops = [c for c in nl.components if c.type.value in AMPLIFIERS]
        kinds = {(_Q.match(q.replace(" ", "")).group(1) if _Q.match(q.replace(" ", "")) else q.replace(" ", ""))
                 for q in nl.query}
        asked = bool(kinds & KINDS) and (kinds & {"fosc", "barkhausen", "gain_condition", "oscillates"}
                                         or _OSC_WORD.search(nl.question or ""))
        sources = [c for c in nl.components if c.type.value in ("V", "I")]
        looks_like = bool(ops) and not sources and bool(_OSC_WORD.search(nl.question or ""))
        if not (asked or looks_like):
            return False, ""
        op = astable_shape(nl)
        if op is not None:
            # A square-wave generator is source-free and asked about with "fosc" too, but it is NOT a linear
            # feedback loop: its output sits at a rail and the period comes from the RC charging time. Say
            # that, rather than reporting that the poles are real.
            return False, (f"{op.id} is wired as an astable multivibrator (positive feedback to its + input "
                           f"and the timing capacitor on its − input), which is a relaxation oscillator, not "
                           f"a linear one: det A(s) = 0 does not describe it. The relaxation solver handles "
                           f"it -- if you are seeing this message, check that solvers/__init__.py lists "
                           f"RelaxationSolver in SOLVERS.")
        if any(c.type.value == "D" for c in nl.components):
            return False, ("Oscillators with diodes in the loop (amplitude-limiting networks) are not "
                           "supported yet: leave the limiting diodes out and ask for the frequency")
        if any(c.type.value in ("Q", "M") for c in nl.components) and not ops:
            # a transistor is allowed, but only as its small-signal model, which needs gm
            for c in nl.components:
                if c.type.value in ("Q", "M") and _gm_of(c) is None and _gm_raw(c) is None:
                    return False, (f"{c.id}: a transistor oscillator is solved from the small-signal model, "
                                   f"so give gm (or I_C / I_D, from which gm = I_C/V_T). A DC bias cannot be "
                                   f"found here: the netlist is the AC equivalent circuit and has no supply.")
        if not ops and not any(c.type.value in ("Q", "M") for c in nl.components):
            return False, ("An oscillator needs its amplifier in the netlist: an op-amp, a transistor, or the "
                           "transistor's small-signal model as a G (transconductance) element")
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
        self.ops = [c for c in nl.components if c.type.value in AMPLIFIERS]
        self.comps = list(nl.components)

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
        """The resistor from an op-amp's output to its − input: the one that sets the amplifier's gain."""
        for op in [c for c in self.ops if c.type.value == "OPAMP"]:
            out, inm = op.nodes[0], op.nodes[2]
            for c in self.nl.components:
                if c.type.value == "R" and set(c.nodes[:2]) == {out, inm}:
                    return c
        return None

    def _gain_element(self) -> Optional[Component]:
        """What to solve for when the gain is too low: an op-amp's feedback resistor, or -- in an LC
        oscillator built round a transistor -- the transconductance itself ("what gm is needed?")."""
        rf = self._feedback_resistor()
        if rf is not None:
            return rf
        gs = [c for c in self.comps if c.type.value in ("E", "G", "F", "H")]
        return gs[0] if len(gs) == 1 else None

    def _gain_partner(self, rf: Component) -> Optional[Component]:
        """The other resistor at the − input: the ratio Rf/this is the gain condition textbooks quote."""
        ops = [c for c in self.ops if c.type.value == "OPAMP"]
        if not ops or rf.type.value != "R":
            return None
        inm = ops[0].nodes[2]
        for c in self.nl.components:
            if c.type.value == "R" and c.id != rf.id and inm in c.nodes[:2]:
                return c
        return None

    # ------------------------------------------------------------------
    def _solve_symbolic(self, comps: List[Component], unknown: Component) -> Tuple[float, float]:
        """Both Barkhausen equations at once: returns (w0, the value the unknown must have)."""
        poly = self._char_poly(comps)
        re_part, im_part = self._split(poly)
        # Take the symbol OUT of the polynomial rather than building a new one: MNA makes a resistance
        # positive=True but a transconductance only real=True, and Symbol("gm", positive=True) is a
        # different symbol from Symbol("gm", real=True), which would leave nothing to solve for.
        name = str(unknown.value).strip()
        x = next((f for f in poly.free_symbols if f.name == name), None)
        if x is None:
            raise SolverError(f"{unknown.id}'s value '{name}' does not appear in the characteristic "
                              f"equation, so it cannot be solved for")
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

    def _val(self, c: Component) -> Optional[float]:
        try:
            return float(sp.N(MNA(self.comps, "ac", conventions=self.conv).value(c)))
        except (SolverError, TypeError, ValueError):
            return None

    def _textbook_lc(self, f0: float) -> bool:
        """LC oscillators: a tapped tank. Colpitts taps the capacitance, Hartley the inductance.

            Colpitts   two C and one L   ->   f0 = 1/(2*pi*sqrt(L*C1*C2/(C1+C2)))
            Hartley    two L and one C   ->   f0 = 1/(2*pi*sqrt((L1+L2)*C))

        The tap divides the tank, so the feedback ratio is fixed by the component ratio, and the amplifier
        only has to make up for it: gm*R = (the ratio of the element at the input to the one at the output).
        """
        ls = [c for c in self.comps if c.type.value == "L"]
        cs = [c for c in self.comps if c.type.value == "C"]
        vals_l = [self._val(c) for c in ls]
        vals_c = [self._val(c) for c in cs]
        if any(v is None for v in vals_l + vals_c):
            return False
        if len(cs) == 2 and len(ls) == 1:
            name = "Colpitts"
            eq = vals_c[0] * vals_c[1] / (vals_c[0] + vals_c[1])
            f_book = 1 / (2 * 3.141592653589793 * math.sqrt(vals_l[0] * eq))
            how = (f"C_eq = C1·C2/(C1+C2) = {fmt_si(eq, 'F')} in series with L = {fmt_si(vals_l[0], 'H')}; "
                   f"f0 = 1/(2·π·√(L·C_eq))")
        elif len(ls) == 2 and len(cs) == 1:
            name = "Hartley"
            eq = vals_l[0] + vals_l[1]
            f_book = 1 / (2 * 3.141592653589793 * math.sqrt(eq * vals_c[0]))
            how = (f"L_eq = L1 + L2 = {fmt_si(eq, 'H')} with C = {fmt_si(vals_c[0], 'F')}; "
                   f"f0 = 1/(2·π·√(L_eq·C))")
        else:
            return False
        self.res.numeric["fosc.approx"] = f_book
        self.trace.add(f"This is the classic {name} oscillator: a tapped tank, so the book's formula applies",
                       how,
                       f"textbook f0 = {fmt_si(f_book, 'Hz', self.sig)} "
                       f"({'agrees' if abs(f_book - f0) < 0.01 * f_book else 'differs'} with the determinant)")
        return True

    # ------------------------------------------------------------------
    def _textbook(self, w0: float) -> None:
        """Name the topology and compare with the formula from the book, when the parts are all equal."""
        f0 = w0 / (2 * 3.141592653589793)
        if self._textbook_lc(f0):
            return
        rs = [c for c in self.comps if c.type.value == "R"]
        cs = [c for c in self.comps if c.type.value == "C"]
        rf = self._feedback_resistor()
        loop_r = [c for c in rs if rf is None or c.id != rf.id]
        vals_r = {str(c.value) for c in loop_r}
        vals_c = {str(c.value) for c in cs}
        if len(vals_c) != 1 or len(vals_r) > 2:
            return
        try:
            R = float(sp.N(MNA(self.comps, "ac", conventions=self.conv).value(loop_r[0])))
            C = float(sp.N(MNA(self.comps, "ac", conventions=self.conv).value(cs[0])))
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
        comps = _small_signal(self.nl, self.conv, self.trace)
        self.comps = comps
        for op in self.ops:
            self.trace.add(f"{op.id} is the amplifier; the loop has no input source, so the circuit is "
                           f"source-free and A(s)·x = 0",
                           "a non-zero output with no input is only possible where det A(s) = 0",
                           "that determinant is the characteristic equation of the circuit")

        unknowns = _free_values(Netlist(analysis=self.nl.analysis, query=self.nl.query,
                                        question=self.nl.question, components=comps))
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
                           f"{fmt_si(x_needed, _unit(unknown), self.sig)}",
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
                rf = self._gain_element()
                if rf is not None:
                    unit = "Ω" if rf.type.value == "R" else ("S" if rf.type.value == "G" else "")
                    sym = [Component(**{**c.model_dump(), "value": "x_needed"}) if c.id == rf.id else c
                           for c in comps]
                    try:
                        w_need, rf_need = self._solve_symbolic(sym, next(c for c in sym if c.id == rf.id))
                        self.res.numeric["gain_condition"] = rf_need
                        self.trace.add(f"Ask what {rf.id} would have to be",
                                       f"re-solve the same two equations with {rf.id} as the unknown",
                                       f"{rf.id} = {fmt_si(rf_need, unit, self.sig)} would put the poles on the "
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
                    partner = self._gain_partner(unknown)
                    ratio = ""
                    if partner is not None:
                        try:
                            pv = float(sp.N(MNA(comps, "ac", conventions=self.conv).value(partner)))
                            ratio = f"  ({unknown.id}/{partner.id} = {fmt_num(x_needed / pv, self.sig)})"
                        except SolverError:
                            ratio = ""
                    self.res.answers[q] = fmt_si(x_needed, _unit(unknown), self.sig) + ratio
                    self.res.numeric[q] = x_needed
                elif "gain_condition" in self.res.numeric:
                    el = self._gain_element()
                    self.res.answers[q] = (fmt_si(self.res.numeric["gain_condition"], _unit(el), self.sig)
                                           + f" (the value {el.id if el else 'the gain element'} needs)")
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