"""
BJT small-signal amplifiers (Phase 3, step 4): voltage gain, input and output resistance.

Three stages, exactly as a textbook does it:

  1. DC bias with the capacitors OPEN (they block DC) and the signal source off -> the Q-point, from the
     P3-3 solver. Every small-signal parameter comes from that bias current, so the bias is shown first.
  2. Hybrid-pi parameters of each transistor at its Q-point:
         gm = I_C / V_T          (V_T = 25 mV -> 1 mA gives 40 mA/V)
         r_pi = β / gm = β·V_T / I_C
         r_e = V_T / I_E = 1/gm approximately  (the "re model": gm·r_e ≈ 1)
         r_o = V_A / I_C          only when an Early voltage V_A is given (otherwise r_o = ∞, ignored)
     The npn and pnp small-signal models are identical: a resistor r_pi from base to emitter, a current
     source gm·v_be into the collector, and r_o from collector to emitter.
  3. Small-signal circuit: every DC supply becomes ground (a battery holds no signal voltage, so it is a
     short to ground for the signal), and every coupling / bypass capacitor becomes a short (mid-band).
     Shorting means the two nodes become ONE node, so the circuit that is left is purely linear and the
     existing DC/AC solver finds the gain, R_in and R_out.
     If the question asks for a cutoff frequency or the poles, the capacitors are KEPT instead and the same
     solver works in the s-domain (the low-frequency response of the amplifier).

Textbook results the method reproduces (shown as a check when they apply):
  common emitter, emitter bypassed : Av = −gm·(R_C ‖ R_L),         R_in = R_B ‖ r_pi
  common emitter with R_E          : Av = −gm·R_C/(1 + gm·R_E) ≈ −R_C/R_E
  emitter follower                 : Av = gm·R'_E/(1 + gm·R'_E) ≈ 1,  R_out = R_E ‖ (1/gm)
  common base                      : Av = +gm·(R_C ‖ R_L),         R_in = r_e
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import sympy as sp

from schemas import AnalysisType, Component, Netlist, SolveResult, StepTrace, is_symbolic, parse_value
from solvers.base import NotSupportedError, SolverError, fmt_num, fmt_si, sig_figs
from solvers.bjt import DeviceJob, resolve_bjt
from solvers.mosfet import resolve_mosfet
from solvers.mna import GROUND

FREQ_KINDS = {"fc", "f0", "poles", "zeros", "bandwidth", "tau"}
SS_KINDS = {"gain", "H", "Rin", "Rout", "gm", "rpi", "ro", "re", "V", "I", "Ai"}
_Q = re.compile(r"^(\w+)\((\w+)(?:,(\w+))?\)$")


def _grounded_only(comps: List[Component]) -> List[Component]:
    """Keep only the components that have a path to ground. Removing the capacitors can leave an island
    (a coupling capacitor's own node, a source resistance behind one): nodal analysis has no solution for a
    floating island, and no DC current flows there anyway."""
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for c in comps:
        a = find(c.nodes[0])
        for n in c.nodes[1:]:
            b = find(n)
            if a != b:
                parent[b] = a
    g = find(GROUND) if GROUND in parent else None
    if g is None:
        return comps
    return [c for c in comps if any(find(n) == g for n in c.nodes)]


@dataclass
class Hybrid:
    id: str
    gm: float
    rpi: Optional[float]         # BJT only: the gate of a MOSFET draws no current, so r_π is infinite
    re: Optional[float]
    ro: Optional[float]
    IC: float                    # I_C (BJT) or I_D (MOSFET)
    IE: float
    beta: Optional[float]
    region: str
    fet: bool = False
    VOV: float = 0.0

    def text(self, sig: int) -> str:
        if self.fet:
            parts = [f"gm = √(2·k·I_D) = 2·I_D/V_OV = {fmt_si(self.gm, 'S', sig)}",
                     "r_π = ∞ (the gate draws no current)"]
            parts.append(f"r_o = 1/(λ·I_D) = {fmt_si(self.ro, 'Ω', sig)}" if self.ro
                         else "r_o = ∞ (λ = 0, no channel-length modulation)")
            return ";  ".join(parts)
        parts = [f"gm = I_C/V_T = {fmt_si(self.gm, 'S', sig)}", f"r_π = β/gm = {fmt_si(self.rpi, 'Ω', sig)}",
                 f"r_e = V_T/I_E = {fmt_si(self.re, 'Ω', sig)}"]
        parts.append(f"r_o = V_A/I_C = {fmt_si(self.ro, 'Ω', sig)}" if self.ro else "r_o = ∞ (no Early voltage given)")
        return ";  ".join(parts)


class SmallSignalJob:
    def __init__(self, nl: Netlist, conventions: Dict):
        self.nl, self.conv = nl, conventions or {}
        self.sig = sig_figs(conventions)
        sc = self.conv.get("semiconductor") or {}
        self.VT = float(sc.get("VT", 0.025))
        self.VA = sc.get("VA", None)
        self.queries = [q.replace(" ", "") for q in nl.query]
        kinds0 = {(_Q.match(q).group(1) if _Q.match(q) else q) for q in self.queries}
        midband0 = not (kinds0 & FREQ_KINDS)
        for c in nl.components:
            for v in (c.value, c.ac):
                if v in (None, "") or not is_symbolic(v):
                    continue
                if c.type.value == "C" and midband0:
                    # at mid-band every coupling / bypass capacitor is a short, so its value cannot affect
                    # the answer: a question that only says "bypassed" is still answerable
                    continue
                if c.type.value == "C":
                    raise NotSupportedError(f"{c.id} has no numeric value ({v}), and the question asks about "
                                            f"frequency, where the capacitor value matters")
                raise NotSupportedError(f"Small-signal analysis needs numeric values ({c.id} = {v})")
        self.bjts = [resolve_bjt(c, self.conv) for c in nl.components if c.type.value == "Q"]
        self.mos = [resolve_mosfet(c, self.conv) for c in nl.components if c.type.value == "M"]
        self.devs = self.bjts + self.mos
        if not self.devs:
            raise NotSupportedError("No transistor in the circuit")
        self.caps = [c for c in nl.components if c.type.value == "C"]
        self.midband = midband0
        self._out_hint: Optional[str] = None
        self.trace = StepTrace()
        self.res = SolveResult(method="DC bias -> hybrid-pi parameters at the Q-point -> small-signal circuit "
                                      "(supplies grounded, capacitors shorted) solved exactly")

    # ------------------------------------------------------------------
    # 1. DC bias
    # ------------------------------------------------------------------
    def _signal_sources(self) -> List[Component]:
        """Sources that carry the signal: an AC magnitude, or a 0 V DC source used as the input."""
        return [c for c in self.nl.components if c.type.value in ("V", "I") and c.ac not in (None, "")]

    def dc_bias(self) -> Dict[str, Hybrid]:
        sig_ids = {c.id for c in self._signal_sources()}
        dc_comps: List[Component] = []
        for c in self.nl.components:
            if c.type.value == "C":                     # capacitors are open circuits in DC
                continue
            if c.id in sig_ids:
                # the signal source is off for the bias: a voltage source becomes 0 V (a short), a current
                # source 0 A (an open). Keeping it (instead of deleting it) holds its node at a known voltage.
                if c.type.value == "I":
                    continue
                dc_comps.append(c.model_copy(update={"ac": None, "value": c.value or "0"}))
            else:
                dc_comps.append(c)
        dc_comps = _grounded_only(dc_comps)
        q_ids = [q.id for q in self.devs]
        dc_nl = Netlist(question=self.nl.question, components=dc_comps, analysis="dc",
                        query=[f"bias({i})" for i in q_ids], assumptions=list(self.nl.assumptions))
        try:
            job = DeviceJob(dc_nl, self.conv)
            dc_res = job.run()
        except SolverError as e:
            raise SolverError(f"Could not find the DC bias, which the small-signal parameters come from: {e}")
        self.trace.add("Step 1 — DC bias with the capacitors open (they block DC) and the signal source off",
                       "the small-signal parameters all come from the Q-point", None)
        for st in dc_res.trace.all_steps():
            self.trace.add(st.action, st.equation, st.result, level="full")
        out: Dict[str, Hybrid] = {}
        for q in self.devs:
            key = f"bias({q.id})"
            region = dc_res.symbolic.get(f"{key}.region", "?")
            fet = q in self.mos
            want = "saturation" if fet else "active"
            if region != want:
                raise SolverError(f"{q.id} is in {region}, not {want}: a small-signal amplifier must be biased "
                                  f"in {'saturation' if fet else 'the active region'} (check the bias resistors)")
            if fet:
                idq = abs(dc_res.numeric.get(f"{key}.ID") or 0.0)
                vds = dc_res.numeric.get(f"{key}.VDS")
                vov = dc_res.numeric.get(f"{key}.VOV") or 0.0
                if idq <= 0:
                    raise SolverError(f"{q.id} has no bias current, so it cannot amplify")
                self.trace.add(f"Q-point of {q.id} (from the DC bias)", None,
                               f"I_D = {fmt_si(idq, 'A', self.sig)}, V_OV = {fmt_si(vov, 'V', self.sig)}, "
                               f"{q.vds_name} = {fmt_si(vds, 'V', self.sig)} — saturation ✓")
                gm, ro = q.gm_ro(idq, vov + q.Vt, vds or 0.0)
                out[q.id] = Hybrid(q.id, gm, None, None, ro, idq, idq, None, region, fet=True, VOV=vov)
                self.res.numeric[f"bias({q.id}).ID"] = idq
                self.res.numeric[f"bias({q.id}).VDS"] = vds
            else:
                ic = abs(dc_res.numeric.get(f"{key}.IC") or 0.0)
                ie = abs(dc_res.numeric.get(f"{key}.IE") or 0.0)
                vce = dc_res.numeric.get(f"{key}.VCE")
                if ic <= 0:
                    raise SolverError(f"{q.id} has no bias current, so it cannot amplify")
                self.trace.add(f"Q-point of {q.id} (from the DC bias)", None,
                               f"I_C = {fmt_si(ic, 'A', self.sig)}, I_E = {fmt_si(ie, 'A', self.sig)}, "
                               f"{q.vce_name} = {fmt_si(vce, 'V', self.sig)} — active region ✓")
                gm = ic / self.VT
                out[q.id] = Hybrid(q.id, gm, q.beta / gm, self.VT / ie,
                                   (float(self.VA) / ic) if self.VA else None, ic, ie, q.beta, region)
                self.res.numeric[f"bias({q.id}).IC"] = ic
                self.res.numeric[f"bias({q.id}).VCE"] = vce
            h = out[q.id]
            self.res.numeric[f"gm({q.id})"] = h.gm
            for nm, val in (("rpi", h.rpi), ("re", h.re), ("ro", h.ro)):
                if val is not None:
                    self.res.numeric[f"{nm}({q.id})"] = val
        self.trace.add("Step 2 — hybrid-π parameters at the Q-point (V_T = " + fmt_si(self.VT, "V") + ")",
                       ";  ".join(f"{i}: " + h.text(self.sig) for i, h in out.items()),
                       None)
        return out

    # ------------------------------------------------------------------
    # 2. small-signal circuit
    # ------------------------------------------------------------------
    def ss_netlist(self, hp: Dict[str, Hybrid]) -> Tuple[Netlist, Dict[str, str]]:
        """Supplies -> ground, capacitors -> shorts (mid-band): both are node merges. Then each transistor
        becomes its hybrid-pi model. Returns (netlist, node renaming)."""
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rb == GROUND:                            # ground always wins the name
                ra, rb = rb, ra
            parent[rb] = ra

        for n in self.nl.nodes():
            find(n)
        sig_ids = {c.id for c in self._signal_sources()}
        shorted, grounded, kept = [], [], []
        for c in self.nl.components:
            if c.id in sig_ids:
                continue
            if c.type.value == "V":                     # a DC supply holds no signal: short it
                union(c.nodes[0], c.nodes[1])
                grounded.append(c.id)
            elif c.type.value == "I":                   # a DC current source is an open circuit for the signal
                grounded.append(c.id)
            elif c.type.value == "C" and self.midband:
                union(c.nodes[0], c.nodes[1])
                shorted.append(c.id)
            elif c.type.value == "C":
                kept.append(c.id)
        ren = {n: find(n) for n in self.nl.nodes()}
        drop = set(grounded) | set(shorted)
        self._name_output(ren, drop)

        comps: List[Component] = []
        for c in self.nl.components:
            if c.id in drop or c.type.value in ("Q", "M"):
                continue
            nodes = [ren.get(n, n) for n in c.nodes]
            if c.type.value in ("R", "C", "L") and nodes[0] == nodes[1]:
                continue                                # shorted out by a capacitor
            upd = {"nodes": nodes}
            if c.id in sig_ids and self.midband:
                # mid-band: the signal is a plain linear excitation, so a 1 V DC source stands for it
                upd.update({"value": "1", "ac": None, "ac_phase": 0.0})
            elif c.id in sig_ids:
                upd.update({"value": None, "ac": c.ac or "1"})
            comps.append(c.model_copy(update=upd))
        for q in self.devs:
            h = hp[q.id]
            if h.fet:
                g, s_, d = ren.get(q.g, q.g), ren.get(q.s, q.s), ren.get(q.d, q.d)
                # the gate draws no current, so there is no r_π: only gm and r_o between drain and source
                comps.append(Component(id=f"{q.id}_gm", type="G", nodes=[d, s_, g, s_], value=repr(h.gm)))
                if h.ro and h.ro > 0:
                    comps.append(Component(id=f"{q.id}_ro", type="R", nodes=[d, s_], value=repr(h.ro)))
                continue
            b, e, cc = ren.get(q.b, q.b), ren.get(q.e, q.e), ren.get(q.c, q.c)
            comps.append(Component(id=f"{q.id}_rpi", type="R", nodes=[b, e], value=repr(h.rpi)))
            comps.append(Component(id=f"{q.id}_gm", type="G", nodes=[cc, e, b, e], value=repr(h.gm)))
            if h.ro:
                comps.append(Component(id=f"{q.id}_ro", type="R", nodes=[cc, e], value=repr(h.ro)))
        parts = []
        if grounded:
            parts.append(f"DC supplies {', '.join(grounded)} → short to ground (no signal across a battery)")
        if shorted:
            parts.append(f"capacitors {', '.join(shorted)} → shorts (mid-band)")
        if kept:
            parts.append(f"capacitors {', '.join(kept)} kept (the question asks about frequency)")
        merged = sorted({f"{n} = {r}" for n, r in ren.items() if n != r and r != "out"})
        parts.append("hybrid-π model for " + ", ".join(q.id for q in self.devs))
        if self._out_hint:
            orig = sorted({n for n, r in ren.items() if r == self._out_hint and n != r})
            if orig:
                parts.append(f"output node: {' = '.join(orig)}")
        self.trace.add("Step 3 — small-signal circuit", ";  ".join(parts),
                       ("nodes joined: " + ", ".join(merged)) if merged else None)
        ss = Netlist(question=self.nl.question, components=comps,
                     analysis="ac" if not self.midband else "dc",
                     query=self._ss_queries(ren), assumptions=list(self.nl.assumptions),
                     frequency=self.nl.frequency)
        return ss, ren

    def _name_output(self, ren: Dict[str, str], drop: set) -> None:
        """Make sure the small-signal circuit has a node the solver recognises as the output, so that
        "gain", "Rout" and "fc" need no extra hint. A node the question itself asks for wins; otherwise the
        output is the transistor's collector (or its emitter when the collector sits at signal ground, i.e.
        an emitter follower), renamed to "out"."""
        live = set(ren.values())
        for q in self.queries:
            m = _Q.match(q)
            if m and m.group(1) == "V" and not m.group(3):
                node = ren.get(m.group(2), m.group(2))
                if node in live and node != GROUND:
                    self._out_hint = node
                    return
        if live & {"out", "vout", "Vout", "o", "output"}:
            self._out_hint = next(iter(live & {"out", "vout", "Vout", "o", "output"}))
            return
        q = self.devs[-1]
        out_t, src_t = (q.d, q.s) if q in self.mos else (q.c, q.e)
        cand = ren.get(out_t, out_t)
        if cand == GROUND:                              # output terminal at signal ground -> follower
            cand = ren.get(src_t, src_t)
        if cand == GROUND:
            # both output terminals sit at signal ground: there is no output to measure (e.g. a follower
            # whose source resistor is bypassed, which shorts the output to ground)
            self._out_hint = None
            name = "source" if q in self.mos else "emitter"
            raise SolverError(f"{q.id} has no signal output: its {name} and its "
                              f"{'drain' if q in self.mos else 'collector'} are both at signal ground "
                              f"(check which capacitors the question really has — a bypass capacitor across "
                              f"the output resistor would short the output)")
        for n, r in list(ren.items()):                  # rename the chosen node to "out"
            if r == cand:
                ren[n] = "out"
        self._out_hint = "out"

    def _ss_queries(self, ren: Dict[str, str]) -> List[str]:
        out = []
        for q in self.queries:
            m = _Q.match(q)
            kind, a, b = (m.group(1), m.group(2), m.group(3)) if m else (q, None, None)
            if kind in ("gm", "rpi", "ro", "re") or kind in ("bias", "region", "IC", "IB", "IE", "VCE", "VBE",
                                                             "ID", "IS", "IG", "VGS", "VDS", "VSG", "VSD", "VOV"):
                continue
            if kind in ("V", "I") and a:
                out.append(f"{kind}({ren.get(a, a)}" + (f",{ren.get(b, b)})" if b else ")"))
            elif kind == "H" and a:
                out.append(f"H({ren.get(a, a)},{b})")
            elif kind in ("gain", "Rin", "Rout", "fc", "f0", "poles", "zeros", "bandwidth", "tau"):
                out.append(q)
        return out or ["gain"]

    # ------------------------------------------------------------------
    def run(self) -> SolveResult:
        for q in self.devs:
            desc, _ = q.label()
            self.trace.add(f"Transistor — {desc}", None, "; ".join(q.notes) or None)
        hp = self.dc_bias()
        ss, ren = self.ss_netlist(hp)

        from solvers.dc_ac import _Job
        job = _Job(ss, self.conv)
        try:
            ss_res = job.run()
        except SolverError as e:
            raise SolverError(f"Small-signal circuit could not be solved: {e}")
        for st in job.trace.all_steps():
            self.trace.add(st.action, st.equation, st.result, level=st.level)
        ss_res = self._fix_rin(ss, ss_res)

        for q in self.queries:
            m = _Q.match(q)
            kind, a, b = (m.group(1), m.group(2), m.group(3)) if m else (q, None, None)
            if kind in ("gm", "rpi", "ro", "re") and a in hp:
                h = hp[a]
                val = {"gm": h.gm, "rpi": h.rpi, "ro": h.ro, "re": h.re}[kind]
                unit = "S" if kind == "gm" else "Ω"
                if val is None:
                    self.res.answers[q] = "∞ (no Early voltage V_A given, so r_o is ignored)"
                    self.res.numeric[q] = None
                else:
                    self.res.answers[q] = fmt_si(val, unit, self.sig)
                    self.res.numeric[q] = val
                continue
            if kind in ("bias", "region", "IC", "IB", "IE", "VCE", "VBE", "ID", "IS", "IG", "VGS", "VDS",
                        "VSG", "VSD", "VOV") and a in hp:
                h = hp[a]
                if kind in ("bias", "region"):
                    self.res.answers[q] = (f"{a} in {h.region}, Q-point "
                                           + (f"I_D = {fmt_si(h.IC, 'A', self.sig)}, V_OV = "
                                              f"{fmt_si(h.VOV, 'V', self.sig)}" if h.fet else
                                              f"I_C = {fmt_si(h.IC, 'A', self.sig)}, I_E = "
                                              f"{fmt_si(h.IE, 'A', self.sig)}"))
                    self.res.numeric[q] = None
                else:
                    pick = {"IC": h.IC, "ID": h.IC, "IS": h.IE, "IE": h.IE, "IG": 0.0,
                            "IB": (h.IC / h.beta) if h.beta else 0.0, "VOV": h.VOV}.get(kind)
                    if pick is None:
                        pick = self.res.numeric.get(f"bias({a}).{kind}")
                    unit = "A" if kind.startswith("I") else "V"
                    self.res.answers[q] = fmt_si(pick, unit, self.sig) if pick is not None else "-"
                    self.res.numeric[q] = pick
                continue
            key = q
            if kind in ("V", "I") and a:
                key = f"{kind}({ren.get(a, a)}" + (f",{ren.get(b, b)})" if b else ")")
            elif kind == "H" and a:
                key = f"H({ren.get(a, a)},{b})"
            if key in ss_res.answers:
                self.res.answers[q] = ss_res.answers[key]
                for k, v in ss_res.numeric.items():
                    if k == key or k.startswith(key + "."):
                        self.res.numeric[q + k[len(key):]] = v
                if key in ss_res.symbolic:
                    self.res.symbolic[q] = ss_res.symbolic[key]
            else:
                self.res.answers[q] = (f"Could not compute: '{q}' is not available for small-signal amplifiers "
                                       "(ask for gain, Rin, Rout, gm, rpi, re or V of a node)")
                self.res.numeric[q] = None
        self._textbook(hp, ss, ren)
        self.res.verification_notes += ss_res.verification_notes
        for q in self.devs:
            h = hp[q.id]
            self.res.symbolic[f"hybrid.{q.id}"] = json.dumps(
                {"gm": h.gm, "rpi": h.rpi, "ro": h.ro, "IC": h.IC, "IE": h.IE,
                 "beta": h.beta, "fet": h.fet, "VOV": h.VOV})
            self.res.symbolic[f"state.{q.id}"] = "sat" if h.fet else "active"
        self.res.symbolic["ss.nodes"] = json.dumps(ren)
        self.res.symbolic["ss.midband"] = "1" if self.midband else "0"
        if self.queries and all(self.res.answers.get(q, "").startswith("Could not compute") for q in self.queries):
            raise SolverError(" | ".join(self.res.answers.values()))
        self.res.trace = self.trace
        return self.res

    def _fix_rin(self, ss: Netlist, ss_res: SolveResult) -> SolveResult:
        """R_in means the resistance looking INTO the amplifier's input terminal. The source's own series
        resistance R_S belongs to the source, not to the amplifier, so it is not part of R_in (the gain, on
        the other hand, does include the attenuation R_S causes). Re-solve R_in with the source moved to the
        amplifier's input node, i.e. with any series resistors between them taken out."""
        if "Rin" not in [q for q in self.queries]:
            return ss_res
        src = next((c for c in ss.components if c.type.value in ("V", "I")
                    and c.id in {s.id for s in self._signal_sources()}), None)
        if src is None:
            return ss_res
        comps = list(ss.components)
        hot, ret = src.nodes[0], src.nodes[1]
        dropped: List[str] = []
        for _ in range(4):                                   # follow a chain of series resistors, if any
            at = [c for c in comps if hot in c.nodes and c.id != src.id and c.id not in dropped]
            if len(at) != 1 or at[0].type.value != "R":
                break
            r = at[0]
            dropped.append(r.id)
            hot = r.nodes[1] if r.nodes[0] == hot else r.nodes[0]
        if not dropped:
            return ss_res
        comps = [c for c in comps if c.id not in dropped]
        comps = [c.model_copy(update={"nodes": [hot, ret]}) if c.id == src.id else c for c in comps]
        try:
            from solvers.dc_ac import _Job as _J
            res2 = _J(Netlist(question=self.nl.question, components=comps, analysis=ss.analysis,
                              query=["Rin"], frequency=ss.frequency), self.conv).run()
        except SolverError:
            return ss_res
        self.trace.add(f"Input resistance of the amplifier: {', '.join(dropped)} belongs to the SOURCE, not to "
                       f"the amplifier, so R_in is measured looking into node {hot}",
                       f"(the gain does include the attenuation {', '.join(dropped)} causes)",
                       f"R_in = {res2.answers.get('Rin')}")
        ss_res.answers["Rin"] = res2.answers.get("Rin", ss_res.answers.get("Rin", "-"))
        for k, v in res2.numeric.items():
            if k == "Rin" or k.startswith("Rin."):
                ss_res.numeric[k] = v
        return ss_res

    # ------------------------------------------------------------------
    def _textbook(self, hp: Dict[str, Hybrid], ss: Netlist, ren: Dict[str, str]) -> None:
        """One transistor: name the configuration and show the textbook gain formula next to the exact answer."""
        if len(self.devs) != 1 or not self.midband:
            return
        q = self.devs[0]
        h = hp[q.id]
        if h.fet:
            b, e, c = ren.get(q.g, q.g), ren.get(q.s, q.s), ren.get(q.d, q.d)
        else:
            b, e, c = ren.get(q.b, q.b), ren.get(q.e, q.e), ren.get(q.c, q.c)
        gain = next((v for k, v in self.res.numeric.items()
                     if k.split("(")[0] in ("gain", "H") and isinstance(v, float)), None)
        if gain is None:
            return
        r_at = lambda node: [x for x in ss.components if x.type.value == "R" and node in x.nodes
                             and not x.id.startswith(q.id)]

        def par(rs: List[Component]) -> Optional[float]:
            vals = [parse_value(x.value) for x in rs if GROUND in x.nodes]
            if not vals:
                return None
            g = sum(1 / v for v in vals)
            return 1 / g if g else None

        RC = par(r_at(c)) if c != GROUND else None             # a terminal at signal ground has no R of its own
        RE = par(r_at(e)) if e != GROUND else None
        fet = h.fet
        if c != GROUND and e == GROUND and RC:                 # emitter / source at signal ground
            av = -h.gm * RC
            name = "common source (source at signal ground)" if fet else "common emitter (emitter at signal ground)"
            formula = "Av = −gm·(R_D ‖ R_L)" if fet else "Av = −gm·(R_C ‖ R_L)"
        elif RE and RC and b != GROUND:                        # un-bypassed emitter / source resistor
            av = -h.gm * RC / (1 + h.gm * RE)
            name = ("common source with an un-bypassed source resistor" if fet else
                    "common emitter with an un-bypassed emitter resistor")
            formula = "Av = −gm·R_D/(1 + gm·R_S)" if fet else "Av = −gm·R_C/(1 + gm·R_E)"
        elif RE and c == GROUND:                               # follower
            av = h.gm * RE / (1 + h.gm * RE)
            name = "source follower (common drain)" if fet else "emitter follower (common collector)"
            formula = "Av = gm·R'_S/(1 + gm·R'_S)" if fet else "Av = gm·R'_E/(1 + gm·R'_E)"
        elif b == GROUND and RC:                               # common base / common gate
            av = h.gm * RC
            name, formula = ("common gate", "Av = +gm·(R_D ‖ R_L)") if fet else ("common base", "Av = +gm·(R_C ‖ R_L)")
        else:
            return
        ok = abs(av - gain) <= 0.005 * max(abs(gain), 1e-12)
        self.trace.add(f"Textbook check: this is a {name}",
                       f"{formula} = {fmt_num(av, self.sig)}" + ("" if ok else "  (the exact solution differs: the "
                                                                 "full circuit has loading the formula ignores)"),
                       "✓ same as the exact solution" if ok else None)
        self.res.numeric["gain.approx"] = av