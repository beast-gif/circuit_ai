"""
BJT circuits in DC: bias point / Q-point (Phase 3, step 3). Diodes may be in the same circuit.

Large-signal BJT model (the textbook one), per region, NPN shown (PNP: every voltage and current reversed):
  active      V_BE = V_BE(on) (0.7 V)   and   I_C = β·I_B   ->  a 0.7 V source B->E + a CCCS β·I_B from C to E
  saturation  V_BE = V_BE(on)           and   V_CE = V_CE(sat) (0.2 V)  ->  two voltage sources
  cutoff      I_B = 0 and I_C = 0  ->  open circuits
Checks (the same a student writes after assuming a region):
  active      I_B >= 0  and  V_CE >= V_CE(sat)
  saturation  I_B >= 0, I_C >= 0  and  I_C <= β·I_B   (the "forced β" I_C/I_B is below β)
  cutoff      V_BE < V_BE(on)  and  V_BC < V_BE(on)

Method: exactly as for diodes: assume a region for every transistor (and a state for every diode), solve the
linear circuit with MNA, check, keep the combination that passes. Active is tried first (the usual guess).

For one active transistor with a resistive base network, the textbook formula is shown as a cross-check:
  I_B = (V_BB − V_BE − V_X) / (R_BB + (β + 1)·R_E)
with V_BB, R_BB the Thevenin equivalent seen by the base and V_X the node the emitter resistor returns to.
"""
from __future__ import annotations

import itertools
import json
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import sympy as sp

from schemas import AnalysisType, Component, Netlist, SolveResult, StepTrace, is_symbolic, parse_value
from solvers.base import BaseSolver, NotSupportedError, SolverError, fmt_num, fmt_si, sig_figs
from solvers.mna import GROUND, MNA
from solvers.mosfet import MOSFET, resolve_mosfet

MAX_COMBOS = 729
_Q = re.compile(r"^(\w+)\((\w+)(?:,(\w+))?\)$")
BJT_KINDS = {"IC", "IB", "IE", "VCE", "VBE", "VEC", "VEB", "bias", "region", "I", "P"}
MOS_KINDS = {"ID", "IS", "IG", "VGS", "VDS", "VSG", "VSD", "VOV", "bias", "region", "I", "P"}

_ALIASES = {
    "beta": ("beta", "β", "hfe", "bf", "b", "beta_dc", "betadc", "h_fe"),
    "VBE": ("vbe", "vbe_on", "vbeon", "veb", "veb_on", "v_be", "vbe(on)"),
    "VCEsat": ("vcesat", "vce_sat", "vce(sat)", "vecsat", "vec_sat", "v_ce_sat"),
}


def _num(raw) -> float:
    return float(raw) if isinstance(raw, (int, float)) else parse_value(str(raw).strip())


@dataclass
class BJT:
    id: str
    c: str
    b: str
    e: str
    pnp: bool
    beta: float
    VBE: float
    VCEsat: float
    notes: List[str] = field(default_factory=list)
    states = ("active", "sat", "cutoff")

    @property
    def kind(self) -> str:
        return "PNP" if self.pnp else "NPN"

    def as_dict(self) -> Dict[str, object]:
        return {"pnp": self.pnp, "beta": self.beta, "VBE": self.VBE, "VCEsat": self.VCEsat}

    # node pairs in the transistor's "positive" direction (NPN: B->E, C->E; PNP: E->B, E->C)
    @property
    def be(self) -> Tuple[str, str]:
        return (self.e, self.b) if self.pnp else (self.b, self.e)

    @property
    def ce(self) -> Tuple[str, str]:
        return (self.e, self.c) if self.pnp else (self.c, self.e)

    @property
    def vbe_name(self) -> str:
        return "V_EB" if self.pnp else "V_BE"

    @property
    def vce_name(self) -> str:
        return "V_EC" if self.pnp else "V_CE"

    def label(self) -> Tuple[str, str]:
        return (f"{self.id}: {self.kind}, β = {fmt_num(self.beta)}, {self.vbe_name}(on) = {fmt_si(self.VBE, 'V')}, "
                f"{self.vce_name}(sat) = {fmt_si(self.VCEsat, 'V')}",
                f"active → {fmt_si(self.VBE, 'V')} source {'E→B' if self.pnp else 'B→E'} + current source β·I_B "
                f"{'E→C' if self.pnp else 'C→E'};  saturation → {fmt_si(self.VBE, 'V')} and "
                f"{fmt_si(self.VCEsat, 'V')} sources;  cutoff → open")

    def _vbe(self, rmin: bool) -> List[Component]:
        """The 0.7 V base-emitter source; with rmin, 1 mΩ in series so that junctions in parallel (a current
        mirror) share the base current equally instead of leaving the split undefined."""
        bp, bn = self.be
        if not rmin:
            return [Component(id=f"{self.id}_be", type="V", nodes=[bp, bn], value=repr(self.VBE))]
        x = f"{self.id}_bx"
        return [Component(id=f"{self.id}_be", type="V", nodes=[bp, x], value=repr(self.VBE)),
                Component(id=f"{self.id}_br", type="R", nodes=[x, bn], value="1e-3")]

    def parts(self, state: str, off_as_r: bool = False, rmin: bool = False) -> List[Component]:
        (bp, bn), (cp, cn) = self.be, self.ce
        if state == "active":
            return self._vbe(rmin) + [Component(id=f"{self.id}_c", type="F", nodes=[cp, cn], value=repr(self.beta),
                                                control=f"{self.id}_be")]
        if state == "sat":
            return self._vbe(rmin) + [Component(id=f"{self.id}_ce", type="V", nodes=[cp, cn], value=repr(self.VCEsat))]
        t, v = ("R", "1e12") if off_as_r else ("I", "0")
        return [Component(id=f"{self.id}_be", type=t, nodes=[bp, bn], value=v),
                Component(id=f"{self.id}_ce", type=t, nodes=[cp, cn], value=v)]


def resolve_bjt(c: Component, conventions: Optional[dict] = None) -> BJT:
    p = {str(k).strip().lower().replace(" ", "_").replace("-", "_"): v for k, v in (c.params or {}).items()}
    raw = {key: next((p[a] for a in aliases if p.get(a) not in (None, "")), None) for key, aliases in _ALIASES.items()}
    sc = (conventions or {}).get("semiconductor") or {}
    model = (c.model or "").strip().lower()
    q = BJT(id=c.id, c=c.nodes[0], b=c.nodes[1], e=c.nodes[2], pnp="pnp" in model,
            beta=_num(raw["beta"]) if raw["beta"] is not None else float(sc.get("beta", 100)),
            VBE=abs(_num(raw["VBE"])) if raw["VBE"] is not None else float(sc.get("VBE_on", 0.7)),
            VCEsat=abs(_num(raw["VCEsat"])) if raw["VCEsat"] is not None else float(sc.get("VCE_sat", 0.2)))
    if raw["beta"] is None:
        q.notes.append(f"β = {fmt_num(q.beta)} assumed (not given)")
    return q


# ======================================================================
class TransistorSolver(BaseSolver):
    name = "transistor"
    method = "Assume each transistor's region (and each diode's state), solve the linear circuit exactly (MNA), check"

    def supports(self, nl: Netlist) -> Tuple[bool, str]:
        types = {c.type.value for c in nl.components}
        if not types & {"Q", "M"}:
            return False, ""
        if nl.analysis == AnalysisType.CONCEPTUAL:
            return False, "Conceptual question: no circuit to solve"
        if nl.analysis in (AnalysisType.SMALL_SIGNAL, AnalysisType.AC):
            return True, ""                       # small-signal amplifier: solvers/small_signal.py
        if nl.analysis == AnalysisType.TRANSIENT:
            return False, "Transistor circuits with time-varying inputs are not supported yet"
        return True, ""

    def solve(self, nl: Netlist, conventions: Dict) -> SolveResult:
        ok, why = self.supports(nl)
        if not ok:
            raise NotSupportedError(why)
        if nl.analysis in (AnalysisType.SMALL_SIGNAL, AnalysisType.AC):
            from solvers.small_signal import SmallSignalJob
            return SmallSignalJob(nl, conventions).run()
        return DeviceJob(nl, conventions).run()


class DeviceJob:
    """Assume-solve-check for circuits with BJTs, MOSFETs and (piecewise-linear) diodes. A MOSFET region is
    nonlinear, so inside the assumed state it is solved by Newton-Raphson (its tangent model, re-solved until
    the operating point stops moving); the region checks are then exactly the same idea as for a BJT."""

    def __init__(self, nl: Netlist, conventions: Dict):
        from solvers.diode import _DiodeJob             # diodes reuse their own models and checks
        self.nl, self.conv = nl, conventions or {}
        self.sig = sig_figs(conventions)
        for c in nl.components:
            for v in (c.value, c.ac):
                if v not in (None, "") and is_symbolic(v):
                    raise NotSupportedError(f"Transistor regions can only be decided with numeric values "
                                            f"({c.id} = {v} is symbolic)")
        self.dj = _DiodeJob(nl, conventions)
        if any(d.kind == "exp" for d in self.dj.diodes):
            raise NotSupportedError("Diodes next to transistors use the piecewise-linear models, not the exponential one")
        self.bjts = [resolve_bjt(c, self.conv) for c in nl.components if c.type.value == "Q"]
        self.by_q = {q.id: q for q in self.bjts}
        self.mos = [resolve_mosfet(c, self.conv) for c in nl.components if c.type.value == "M"]
        self.by_m = {m.id: m for m in self.mos}
        self.others = [c for c in nl.components if c.type.value not in ("D", "Q", "M")]
        self._variant: Dict[tuple, Tuple[bool, bool]] = {}      # which stand-in worked for each state combination
        self._op: Dict[tuple, Dict[str, Tuple[float, float]]] = {}   # converged (V_GS, V_DS) per MOSFET
        self.trace = StepTrace()
        self.res = SolveResult(method=TransistorSolver.method)

    # ------------------------------------------------------------------
    def _components(self, states: Dict[str, str], off_as_r: bool = False, rmin: bool = False,
                    op: Optional[Dict[str, Tuple[float, float]]] = None) -> List[Component]:
        comps = list(self.others)
        for d in self.dj.diodes:
            comps += self.dj._parts(d, states[d.id], off_as_r=off_as_r)
        for q in self.bjts:
            comps += q.parts(states[q.id], off_as_r, rmin)
        for m in self.mos:
            vgs, vds = (op or {}).get(m.id, (m.Vt + 0.5, 1.0))
            comps += m.parts(states[m.id], vgs, vds, off_as_r)
        return comps

    def _mna(self, states: Dict[str, str]) -> MNA:
        last = None
        for off_as_r, rmin in ((False, False), (True, False), (False, True), (True, True)):
            try:
                m = MNA(self._components(states, off_as_r, rmin), "dc", conventions=self.conv)
                m.solve()
                self._variant[tuple(sorted(states.items()))] = (off_as_r, rmin)
                return m
            except SolverError as e:
                last = e
        raise last

    def _solve(self, states: Dict[str, str]) -> Tuple[MNA, Dict[str, Tuple[float, float]]]:
        """Solve the circuit with these states. With no MOSFET this is one linear solve; with MOSFETs it is
        Newton-Raphson on their operating points (each step is one linear solve of the tangent model)."""
        key = tuple(sorted(states.items()))
        if not self.mos:
            m = self._mna(states)
            return m, {}
        last: Optional[SolverError] = None
        n_starts = max(len(m.starts(states[m.id])) for m in self.mos)
        for off_as_r in (False, True):
            for s_i in range(n_starts):
                op = {}
                for m in self.mos:
                    st = m.starts(states[m.id])
                    op[m.id] = st[min(s_i, len(st) - 1)]
                try:
                    mm, op = self._newton(states, op, off_as_r)
                except SolverError as e:
                    last = e
                    continue
                self._variant[key] = (off_as_r, False)
                self._op[key] = op
                return mm, op
        raise last or SolverError("could not solve the circuit")

    def _newton(self, states: Dict[str, str], op: Dict[str, Tuple[float, float]], off_as_r: bool):
        """Newton-Raphson from one starting point. Returns (MNA, operating point). If the iterates stop
        moving without solving the circuit, the assumed region is the wrong guess — the clamp keeps pushing
        the point back into a region the circuit does not want — and the error says which way it was pushed."""
        for _ in range(300):
            mm = MNA(self._components(states, off_as_r, False, op), "dc", conventions=self.conv)
            mm.solve()
            raw, done, moved = {}, True, False
            for mos in self.mos:
                vgs = self._f(mm, mm.V[mos.gs[0]] - mm.V[mos.gs[1]])
                vds = self._f(mm, mm.V[mos.ds[0]] - mm.V[mos.ds[1]])
                o_gs, o_ds = op[mos.id]
                if (abs(vgs - o_gs) > 1e-10 + 1e-9 * abs(o_gs)
                        or abs(vds - o_ds) > 1e-10 + 1e-9 * abs(o_ds)):
                    done = False                     # the circuit disagrees with the point the model was built at
                raw[mos.id] = (vgs, vds)
            if done:
                return mm, op                        # a true fixed point: the nonlinear equations hold
            new = {}
            for mos in self.mos:
                vgs, vds = raw[mos.id]
                o_gs, o_ds = op[mos.id]
                vgs = mos.limit(vgs, o_gs, "vgs")
                vds = mos.limit(vds, o_ds, "vds")
                new[mos.id] = mos.clamp(states[mos.id], vgs, vds)
                if abs(new[mos.id][0] - o_gs) > 1e-13 or abs(new[mos.id][1] - o_ds) > 1e-13:
                    moved = True
            if not moved:                            # stuck on the boundary of the assumed region
                raise SolverError(self._stuck_reason(states, op, raw))
            op = new
        raise SolverError("the MOSFET square-law equations did not settle in 300 iterations")

    def _stuck_reason(self, states, op, raw) -> str:
        """Why an assumed region cannot hold: the circuit pushes the operating point out of it."""
        why = []
        for mos in self.mos:
            st = states[mos.id]
            vgs, vds = raw[mos.id]
            ov = op[mos.id][0] - mos.Vt
            if st in ("sat", "tri") and vgs < mos.Vt - 1e-9:
                why.append(f"{mos.id} cannot conduct: {mos.vgs_name} stays below V_t (it is cut off)")
            elif st == "sat" and vds < ov - 1e-9:
                why.append(f"{mos.id} cannot be saturated: the circuit pulls {mos.vds_name} below V_OV, so it "
                           f"is in the triode region")
            elif st == "tri" and vds > ov + 1e-9:
                why.append(f"{mos.id} cannot be in the triode region: the circuit holds {mos.vds_name} above "
                           f"V_OV, so it is saturated")
        return "; ".join(why) or "the MOSFET square-law equations did not settle in this region"

    def m_values(self, m: MNA, mos: MOSFET, st: str, op: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """I_D, I_S, I_G, V_GS / V_SG, V_DS / V_SD, V_OV at the converged point."""
        vgs = self._f(m, m.V[mos.gs[0]] - m.V[mos.gs[1]])
        vds = self._f(m, m.V[mos.ds[0]] - m.V[mos.ds[1]])
        idq = mos.current(st, vgs, vds)
        return {"ID": idq, "IS": idq, "IG": 0.0, "VGS": vgs, "VDS": vds, "VOV": vgs - mos.Vt}

    @staticmethod
    def _f(m: MNA, expr) -> float:
        e = m.evaluate(expr)
        if e.free_symbols:
            raise NotSupportedError("Transistor regions can only be decided with numeric values")
        return float(sp.N(e))

    def q_values(self, m: MNA, q: BJT, st: str) -> Dict[str, float]:
        """I_B, I_C, I_E (positive in normal operation), V_BE / V_EB, V_CE / V_EC."""
        vbe = self._f(m, m.V[q.be[0]] - m.V[q.be[1]])
        vce = self._f(m, m.V[q.ce[0]] - m.V[q.ce[1]])
        if st == "cutoff":
            ib = ic = 0.0
        else:
            ib = self._f(m, m.branch[f"{q.id}_be"])
            ic = q.beta * ib if st == "active" else self._f(m, m.branch[f"{q.id}_ce"])
        return {"IB": ib, "IC": ic, "IE": ib + ic, "VBE": vbe, "VCE": vce}

    def _q_checks(self, m: MNA, q: BJT, st: str) -> List[Tuple[bool, str]]:
        v = self.q_values(m, q, st)
        s = self.sig
        if st == "active":
            ok1, ok2 = v["IB"] >= -1e-12, v["VCE"] >= q.VCEsat - 1e-9
            return [(ok1, f"{q.id} active needs I_B ≥ 0: I_B = {fmt_si(v['IB'], 'A', s)}" + (" ✓" if ok1 else " ✗")),
                    (ok2, f"{q.id} active needs {q.vce_name} ≥ {fmt_si(q.VCEsat, 'V')}: {q.vce_name} = "
                          f"{fmt_si(v['VCE'], 'V', s)}" + (" ✓" if ok2 else " ✗ (it would be saturated)"))]
        if st == "sat":
            ok1 = v["IB"] >= -1e-12 and v["IC"] >= -1e-12
            ok2 = v["IC"] <= q.beta * v["IB"] + 1e-12
            bf = v["IC"] / v["IB"] if v["IB"] > 0 else math.inf
            return [(ok1, f"{q.id} saturated needs I_B, I_C ≥ 0: I_B = {fmt_si(v['IB'], 'A', s)}, I_C = "
                          f"{fmt_si(v['IC'], 'A', s)}" + (" ✓" if ok1 else " ✗")),
                    (ok2, f"{q.id} saturated needs I_C ≤ β·I_B: forced β = I_C/I_B = "
                          f"{fmt_num(bf, s) if bf != math.inf else '∞'} ≤ {fmt_num(q.beta)}" + (" ✓" if ok2 else " ✗"))]
        vbc = v["VBE"] - v["VCE"]                       # V_BC (PNP: V_CB)
        ok1, ok2 = v["VBE"] <= q.VBE + 1e-9, vbc <= q.VBE + 1e-9
        return [(ok1, f"{q.id} cutoff needs {q.vbe_name} < {fmt_si(q.VBE, 'V')}: {q.vbe_name} = "
                      f"{fmt_si(v['VBE'], 'V', s)}" + (" ✓" if ok1 else " ✗")),
                (ok2, f"{q.id} cutoff needs the base-collector junction off: {'V_CB' if q.pnp else 'V_BC'} = "
                      f"{fmt_si(vbc, 'V', s)}" + (" ✓" if ok2 else " ✗"))]

    def _checks(self, m: MNA, states: Dict[str, str], op: Optional[Dict] = None) -> List[Tuple[bool, str]]:
        out = self.dj._check(m, states, {}) if self.dj.diodes else []
        for q in self.bjts:
            out += self._q_checks(m, q, states[q.id])
        for mos in self.mos:
            v = self.m_values(m, mos, states[mos.id], op or {})
            out += mos.checks(states[mos.id], v["VGS"], v["VDS"], self.sig)
        return out

    def _region(self, st: str, i: str = "") -> str:
        if i in self.by_m:
            return self.by_m[i].region(st)
        return {"active": "active", "sat": "saturation", "cutoff": "cutoff"}[st]

    def _state_text(self, i: str, st: str) -> str:
        if i in self.by_q or i in self.by_m:
            return f"{i} {self._region(st, i)}"
        return self.dj._state_text(self.dj.by_id[i], st)

    def _replace_text(self, i: str, st: str) -> str:
        if i in self.by_m:
            mos = self.by_m[i]
            if st == "cutoff":
                return f"{i}: drain open (I_D = 0)"
            law = ("I_D = ½·k·V_OV²" if st == "sat" else f"I_D = k·(V_OV·{mos.vds_name} − {mos.vds_name}²/2)")
            return f"{i}: {law}, solved by Newton-Raphson (its tangent model, re-solved until it settles)"
        if i not in self.by_q:
            return self.dj._replace_text(self.dj.by_id[i], st)
        q = self.by_q[i]
        if st == "active":
            return (f"{i}: {q.vbe_name} = {fmt_si(q.VBE, 'V')} source, collector current = β·I_B = "
                    f"{fmt_num(q.beta)}·I_B")
        if st == "sat":
            return f"{i}: {q.vbe_name} = {fmt_si(q.VBE, 'V')} and {q.vce_name} = {fmt_si(q.VCEsat, 'V')} sources"
        return f"{i}: all terminals open (I_B = I_C = 0)"

    # ------------------------------------------------------------------
    def run(self) -> SolveResult:
        queries = [q.replace(" ", "") for q in self.nl.query]
        devs = ([(d.id, list(d.states)) for d in self.dj.diodes] + [(q.id, list(q.states)) for q in self.bjts]
                + [(m.id, list(m.states)) for m in self.mos])
        n = math.prod(len(s) for _, s in devs)
        if n > MAX_COMBOS:
            raise NotSupportedError(f"Too many devices to try every state ({n} combinations)")
        for d in self.dj.diodes:
            desc, rule = d.label(self.dj.VT)
            self.trace.add(f"Diode model — {desc}", rule, "; ".join(d.notes) or None)
        for q in self.bjts:
            desc, rule = q.label()
            self.trace.add(f"Transistor model — {desc}", rule, "; ".join(q.notes) or None)
        for mos in self.mos:
            desc, rule = mos.label()
            self.trace.add(f"MOSFET model — {desc}", rule, "; ".join(mos.notes) or None)

        tried, found = [], None
        for combo in itertools.product(*[s for _, s in devs]):
            states = {i: st for (i, _), st in zip(devs, combo)}
            try:
                m, op = self._solve(states)
            except SolverError as e:
                tried.append((states, f"no unique solution ({e})"))
                continue
            checks = self._checks(m, states, op)
            bad = [t for ok, t in checks if not ok]
            if bad:
                tried.append((states, bad[0]))
                continue
            found = (states, m, checks, op)
            break
        if found is None:
            raise SolverError("No combination of transistor regions / diode states is consistent: check the "
                              "device type (NPN/PNP, NMOS/PMOS), the pin order ([collector, base, emitter], "
                              "[drain, gate, source]) and the supplies")
        states, m, checks, op = found
        for st_, why in tried[:8]:
            self.trace.add("Try " + ", ".join(self._state_text(i, s) for i, s in st_.items()), None,
                           f"✗ {why}: this assumption is wrong", level="full")
        self.trace.add("Assume " + ", ".join(self._state_text(i, s) for i, s in states.items()),
                       ";  ".join(self._replace_text(i, s) for i, s in states.items()))

        own = BJT_KINDS | MOS_KINDS
        lin_q = []
        for q in queries:
            mq = _Q.match(q)
            kind, ref = (mq.group(1), mq.group(2)) if mq else (q, None)
            if kind in own and (ref in self.by_q or ref in self.by_m or ref in self.dj.by_id):
                continue
            if kind in ("V", "I", "P"):
                lin_q.append(q)
        from solvers.dc_ac import _Job
        key = tuple(sorted(states.items()))
        comps = self._components(states, *self._variant.get(key, (False, False)), op=self._op.get(key))
        if not lin_q:
            node = next(nd for c in comps for nd in c.nodes if nd != GROUND)
            lin_q = [f"V({node})"]
        lin_nl = Netlist(question=self.nl.question, components=comps, analysis="dc", query=lin_q,
                         assumptions=list(self.nl.assumptions))
        job = _Job(lin_nl, self.conv)
        job_res = job.run()
        if job._main is None:
            job.main()
        for st in job.trace.all_steps():
            self.trace.add(st.action, st.equation, st.result, level=st.level)
        self.trace.add("Check the assumptions", ";  ".join(t for _, t in checks), "all assumptions hold ✓")
        jm = job._main
        self._textbook_step(states, jm)
        self._mos_steps(states, jm, op)

        for q in queries:
            mq = _Q.match(q)
            kind, ref = (mq.group(1), mq.group(2)) if mq else (q, None)
            if ref in self.by_q and kind in BJT_KINDS:
                self._q_answer(q, kind, self.by_q[ref], states[ref], jm)
            elif ref in self.by_m and kind in MOS_KINDS:
                self._m_answer(q, kind, self.by_m[ref], states[ref], jm, op)
            elif ref in self.dj.by_id and kind in ("I", "P", "bias"):
                self.dj._diode_answer(q, kind, self.dj.by_id[ref], states[ref], jm, {})
                self.res.answers[q] = self.dj.res.answers[q]
                for k, v in self.dj.res.numeric.items():
                    if k == q or k.startswith(q + "."):
                        self.res.numeric[k] = v
            elif q in job_res.answers:
                self.res.answers[q] = job_res.answers[q]
                for k, v in job_res.numeric.items():
                    if k == q or k.startswith(q + "."):
                        self.res.numeric[k] = v
            else:
                self.res.answers[q] = (f"Could not compute: '{q}' is not available for transistor bias yet "
                                       "(ask for V, I, P, bias, IC, IB, IE, VCE, VBE, ID, VGS or VDS)")
                self.res.numeric[q] = None
        self.res.verification_notes += job_res.verification_notes
        for d in self.dj.diodes:
            self.res.symbolic[f"state.{d.id}"] = states[d.id]
            self.res.symbolic[f"diode.{d.id}"] = json.dumps(d.as_dict())
        for q in self.bjts:
            self.res.symbolic[f"state.{q.id}"] = states[q.id]
            self.res.symbolic[f"bjt.{q.id}"] = json.dumps(q.as_dict())
        for mos in self.mos:
            self.res.symbolic[f"state.{mos.id}"] = states[mos.id]
            self.res.symbolic[f"mos.{mos.id}"] = json.dumps(mos.as_dict())
            v = self.m_values(jm, mos, states[mos.id], op)
            self.res.symbolic[f"op.{mos.id}"] = json.dumps([v["VGS"], v["VDS"], v["ID"]])
        if queries and all(self.res.answers.get(q, "").startswith("Could not compute") for q in queries):
            raise SolverError(" | ".join(self.res.answers.values()))
        self.res.trace = self.trace
        return self.res

    def _mos_steps(self, states: Dict[str, str], m: MNA, op: Dict[str, Tuple[float, float]]) -> None:
        """Show the square law with the numbers put in, the way it is written by hand."""
        for mos in self.mos:
            st = states[mos.id]
            v = self.m_values(m, mos, st, op)
            s = self.sig
            if st == "cutoff":
                self.trace.add(f"{mos.id} is cut off", f"{mos.vgs_name} = {fmt_si(v['VGS'], 'V', s)} < V_t = "
                               f"{fmt_si(mos.Vt, 'V')}", "I_D = 0")
                continue
            ov = v["VOV"]
            if st == "sat":
                eq = (f"I_D = ½·k·V_OV² = ½ × {fmt_si(mos.k, 'A/V²')} × ({fmt_si(v['VGS'], 'V', s)} − "
                      f"{fmt_si(mos.Vt, 'V')})² = ½ × {fmt_si(mos.k, 'A/V²')} × ({fmt_si(ov, 'V', s)})²")
                if mos.lam:
                    eq += f" × (1 + {fmt_num(mos.lam)} × {fmt_si(v['VDS'], 'V', s)})"
            else:
                eq = (f"I_D = k·(V_OV·{mos.vds_name} − {mos.vds_name}²/2) = {fmt_si(mos.k, 'A/V²')} × "
                      f"({fmt_si(ov, 'V', s)} × {fmt_si(v['VDS'], 'V', s)} − {fmt_si(v['VDS'], 'V', s)}²/2)")
            self.trace.add(f"Square law for {mos.id} in {mos.region(st)}", eq,
                           f"I_D = {fmt_si(v['ID'], 'A', s)}  ({mos.vgs_name} = {fmt_si(v['VGS'], 'V', s)}, "
                           f"V_OV = {fmt_si(ov, 'V', s)}, {mos.vds_name} = {fmt_si(v['VDS'], 'V', s)})")

    def _m_answer(self, qq: str, kind: str, mos: MOSFET, st: str, m: MNA,
                  op: Dict[str, Tuple[float, float]]) -> None:
        v = self.m_values(m, mos, st, op)
        s = self.sig
        if kind in ("bias", "region"):
            self.res.answers[qq] = (f"{mos.id} in {mos.region(st)}: I_D = {fmt_si(v['ID'], 'A', s)}, "
                                    f"{mos.vgs_name} = {fmt_si(v['VGS'], 'V', s)}, {mos.vds_name} = "
                                    f"{fmt_si(v['VDS'], 'V', s)}, V_OV = {fmt_si(v['VOV'], 'V', s)}, I_G = 0"
                                    f"  (Q-point: I_D = {fmt_si(v['ID'], 'A', s)}, {mos.vds_name} = "
                                    f"{fmt_si(v['VDS'], 'V', s)})")
            self.res.numeric[qq] = None
            for k in ("ID", "VGS", "VDS", "VOV", "IS", "IG"):
                self.res.numeric[f"{qq}.{k}"] = v[k]
            self.res.symbolic[f"{qq}.region"] = mos.region(st)
            return
        if kind == "P":
            p = v["VDS"] * v["ID"]
            self.res.answers[qq] = f"{fmt_si(p, 'W', s)} dissipated"
            self.res.numeric[qq] = p
            self.trace.add(f"Power in {mos.id}", f"P = {mos.vds_name}·I_D = {fmt_si(v['VDS'], 'V', s)} × "
                           f"{fmt_si(v['ID'], 'A', s)}", f"P({mos.id}) = {self.res.answers[qq]}")
            return
        key = {"I": "ID", "ID": "ID", "IS": "IS", "IG": "IG", "VGS": "VGS", "VSG": "VGS", "VDS": "VDS",
               "VSD": "VDS", "VOV": "VOV"}[kind]
        unit = "A" if key.startswith("I") else "V"
        name = {"ID": "I_D", "IS": "I_S", "IG": "I_G", "VGS": mos.vgs_name, "VDS": mos.vds_name,
                "VOV": "V_OV"}[key]
        self.res.answers[qq] = fmt_si(v[key], unit, s) + (" (drain current)" if kind == "I" else "")
        self.res.numeric[qq] = v[key]
        self.trace.add(f"{name} of {mos.id}", None, f"{name} = {self.res.answers[qq]}")

    # ------------------------------------------------------------------
    def _q_answer(self, qq: str, kind: str, q: BJT, st: str, m: MNA) -> None:
        v = self.q_values(m, q, st)
        s = self.sig
        if kind in ("bias", "region"):
            self.res.answers[qq] = (f"{q.id} in the {self._region(st)} region: I_B = {fmt_si(v['IB'], 'A', s)}, "
                                    f"I_C = {fmt_si(v['IC'], 'A', s)}, I_E = {fmt_si(v['IE'], 'A', s)}, "
                                    f"{q.vce_name} = {fmt_si(v['VCE'], 'V', s)}, {q.vbe_name} = {fmt_si(v['VBE'], 'V', s)}"
                                    f"  (Q-point: I_C = {fmt_si(v['IC'], 'A', s)}, {q.vce_name} = {fmt_si(v['VCE'], 'V', s)})")
            self.res.numeric[qq] = None
            for k in ("IB", "IC", "IE", "VCE", "VBE"):
                self.res.numeric[f"{qq}.{k}"] = v[k]
            self.res.symbolic[f"{qq}.region"] = self._region(st)
            return
        if kind == "P":
            p = v["VCE"] * v["IC"] + v["VBE"] * v["IB"]
            self.res.answers[qq] = f"{fmt_si(p, 'W', s)} dissipated"
            self.res.numeric[qq] = p
            self.trace.add(f"Power in {q.id}", f"P = {q.vce_name}·I_C + {q.vbe_name}·I_B = {fmt_si(v['VCE'], 'V', s)}"
                           f" × {fmt_si(v['IC'], 'A', s)} + {fmt_si(v['VBE'], 'V', s)} × {fmt_si(v['IB'], 'A', s)}",
                           f"P({q.id}) = {self.res.answers[qq]}")
            return
        key = {"I": "IC", "IC": "IC", "IB": "IB", "IE": "IE", "VCE": "VCE", "VEC": "VCE", "VBE": "VBE",
               "VEB": "VBE"}[kind]
        unit = "A" if key.startswith("I") else "V"
        val = v[key]
        name = {"IC": "I_C", "IB": "I_B", "IE": "I_E", "VCE": q.vce_name, "VBE": q.vbe_name}[key]
        self.res.answers[qq] = f"{fmt_si(val, unit, s)}" + (" (collector current)" if kind == "I" else "")
        self.res.numeric[qq] = val
        eq = {"IC": "I_C = β·I_B" if st == "active" else None, "IE": "I_E = I_B + I_C",
              "VCE": f"{q.vce_name} = V({q.ce[0]}) − V({q.ce[1]})",
              "VBE": f"{q.vbe_name} = V({q.be[0]}) − V({q.be[1]})", "IB": None}[key]
        self.trace.add(f"{name} of {q.id}", eq, f"{name} = {self.res.answers[qq]}")

    # ------------------------------------------------------------------
    def _textbook_step(self, states: Dict[str, str], m: MNA) -> None:
        """One active BJT, resistive base network, emitter through one resistor (or directly) to a fixed node:
        show I_B = (V_BB − V_BE − V_X)/(R_BB + (β+1)R_E) as the textbook cross-check."""
        if len(self.bjts) != 1 or self.dj.diodes or self.mos:
            return
        q = self.bjts[0]
        if states[q.id] != "active":
            return
        if any({q.c, q.b} == set(c.nodes[:2]) for c in self.others):      # collector-feedback: other formula
            return
        at_e = [c for c in self.others if q.e in c.nodes[:2]]
        if q.e == GROUND:
            RE, vx_node = 0.0, GROUND
        elif len(at_e) == 1 and at_e[0].type.value == "R":
            RE = parse_value(at_e[0].value)
            vx_node = at_e[0].nodes[1] if at_e[0].nodes[0] == q.e else at_e[0].nodes[0]
        else:
            return
        try:
            from solvers.dc_ac import _Job
            # the base network with the transistor removed (its emitter resistor and collector parts too)
            base_side = [c for c in self.others if not (at_e and c is at_e[0])]
            th = _Job(Netlist(components=base_side, analysis="dc", query=[f"Vth({q.b},0)", f"Rth({q.b},0)"]),
                      self.conv).run()
            VBB, RBB = th.numeric[f"Vth({q.b},0)"], th.numeric[f"Rth({q.b},0)"]
            if vx_node == GROUND:
                VX = 0.0
            else:
                vx = _Job(Netlist(components=base_side, analysis="dc", query=[f"V({vx_node})"]), self.conv).run()
                VX = vx.numeric[f"V({vx_node})"]
        except Exception:                                   # not a simple base network: no textbook step
            return
        if VBB is None or RBB is None or VX is None:
            return
        sgn = -1.0 if q.pnp else 1.0
        ib = (sgn * (VBB - VX) - q.VBE) / (RBB + (q.beta + 1) * RE)
        exact = self.q_values(m, q, "active")["IB"]
        if abs(ib - exact) > 1e-6 * max(abs(exact), 1e-12):
            return
        s = self.sig
        num = (f"({fmt_si(VBB, 'V', s)} − {fmt_si(q.VBE, 'V')}" + (f" − {fmt_si(VX, 'V', s)})" if VX else ")")
               if not q.pnp else f"({fmt_si(VX, 'V', s)} − {fmt_si(q.VBE, 'V')} − {fmt_si(VBB, 'V', s)})")
        self.trace.add("Textbook check: Thevenin equivalent of the base network (V_BB, R_BB), then the base loop",
                       f"V_BB = {fmt_si(VBB, 'V', s)}, R_BB = {fmt_si(RBB, 'Ω', s)};  I_B = "
                       + ("(V_X − V_EB − V_BB)" if q.pnp else "(V_BB − V_BE − V_X)") + "/(R_BB + (β + 1)·R_E) = "
                       f"{num}/({fmt_si(RBB, 'Ω', s)} + {fmt_num(q.beta + 1)} × {fmt_si(RE, 'Ω', s)})",
                       f"I_B = {fmt_si(ib, 'A', s)}  ✓ same as the full solution")