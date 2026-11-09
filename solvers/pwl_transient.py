"""
Diode circuits with capacitors (or inductors) and a sinusoidal input (Phase 3, step 2b):
capacitor-filtered rectifiers (ripple), clampers, peak detectors, voltage doublers.

A capacitor gives the circuit memory, so the output depends on the past. Method (exact for the
piecewise-linear diode models):

  1. For every diode-state combination the circuit is LINEAR. Replace each capacitor by a voltage source vC
     (and each inductor by a current source iL) and solve with MNA, keeping vC, iL and s = sin(ωt) as symbols:
         dvC/dt = iC / C,   diL/dt = vL / L      ->   y' = A·y + b·sin(ωt) + c
  2. Add s = sin ωt and k = cos ωt to the state (s' = ωk, k' = −ωs): z = [y, s, k, 1] obeys z' = M·z, whose
     EXACT solution over a step h is z(t+h) = e^{M·h}·z(t) (matrix exponential: no step-size error).
  3. A diode check (ON: I_D >= 0, OFF: V_AK < Von) is linear in z. When one reaches zero inside a step, the
     instant is found by bisection and the next state is chosen. So switching instants are exact too.
  4. Steady state: the cycle that repeats, y(T) = y(0). Newton's method on the cycle map y0 -> y(T).
  5. Peak, minimum, average, rms, ripple and conduction angle are measured on that cycle.

Textbook approximations are reported next to the exact values for a capacitor-input filter:
  Vr ≈ Vp/(k·f·R·C) (k = 1 half-wave, 2 full-wave),  Vdc ≈ Vp − Vr/2,  ripple factor ≈ Vr/(2√3·Vdc),
  conduction angle ≈ √(2·Vr/Vp) rad,  peak diode current ≈ I_L·(1 + (2π/k)·√(Vp/(2Vr))).
"""
from __future__ import annotations

import itertools
import json
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import sympy as sp

from schemas import Component, Netlist, SolveResult, StepTrace, parse_value
from solvers.base import NotSupportedError, SolverError, fmt_num, fmt_si, sig_figs
from solvers.mna import GROUND, MNA
from solvers.waveform import sinusoids

S = sp.Symbol("s_in", real=True)                 # s = sin(ωt)
R_MIN = 1e-3                                       # Ω in series with an ON diode, only when needed
STEPS = 4000                                       # exact steps per cycle (sampling for the measurements)
KINDS = {"V", "Vmax", "Vmin", "Vavg", "Vrms", "Vripple", "Vpp", "ripple", "I", "Imax", "Iavg", "Irms",
         "PIV", "conduction", "bias"}
_Q = re.compile(r"^(\w+)\((\w+)(?:,(\w+))?\)$")


def expm(M: np.ndarray) -> np.ndarray:
    """Matrix exponential: scaling and squaring with a 20-term Taylor series (exact to rounding for the
    small matrices here)."""
    n = M.shape[0]
    nrm = float(np.max(np.sum(np.abs(M), axis=0))) if M.size else 0.0
    sc = max(0, int(math.ceil(math.log2(nrm / 0.5)))) if nrm > 0.5 else 0
    A = M / (2.0 ** sc)
    E, term = np.eye(n), np.eye(n)
    for k in range(1, 21):
        term = term @ A / k
        E = E + term
    for _ in range(sc):
        E = E @ E
    return E


@dataclass
class Combo:
    states: Dict[str, str]
    M: np.ndarray                              # z' = M z
    margins: List[np.ndarray]                  # each must stay >= 0
    q: Dict[str, np.ndarray] = field(default_factory=dict)    # quantity -> row vector (value = row · z)
    E: Optional[np.ndarray] = None             # e^{M h}
    Ed: Optional[np.ndarray] = None            # e^{M δ}, a tiny look-ahead
    rmin: bool = False

    def margin(self, z: np.ndarray) -> float:
        return min((float(g @ z) for g in self.margins), default=math.inf)


class PWLTransient:
    def __init__(self, nl: Netlist, conventions: Dict, djob):
        self.nl, self.conv, self.dj = nl, conventions or {}, djob
        self.sig = sig_figs(conventions)
        self.trace = StepTrace()
        self.res = SolveResult(method="Piecewise-exact simulation (matrix exponential in each diode state, exact "
                                      "switching instants), steady state by Newton's method on one cycle")
        if any(d.kind == "exp" for d in djob.diodes):
            raise NotSupportedError("Waveforms use the piecewise-linear diode models, not the exponential model")
        self.src = sinusoids(nl)
        if not self.src:
            raise NotSupportedError("A filtered rectifier / clamper needs a sinusoidal input (\"ac\" peak, Vrms or Vpp)")
        self.notes: List[str] = []
        if nl.frequency not in (None, ""):
            self.f = parse_value(nl.frequency)
        else:
            self.f = 1000.0
            self.notes.append("input frequency not given: 1 kHz assumed")
        self.w, self.T = 2 * math.pi * self.f, 1 / self.f
        self.caps = [c for c in nl.components if c.type.value == "C"]
        self.inds = [c for c in nl.components if c.type.value == "L"]
        self.ysyms = [sp.Symbol(f"y_{c.id}", real=True) for c in self.caps + self.inds]
        self.n = len(self.ysyms)
        self.main = self.src[0]
        self.Vm = self.main.peak

    # ------------------------------------------------------------------
    # one linear circuit per diode-state combination
    # ------------------------------------------------------------------
    def _components(self, states: Dict[str, str], off_as_r: bool, rmin: bool) -> List[Component]:
        comps = []
        for c in self.nl.components:
            if c.type.value == "D":
                continue
            if c.type.value == "C":                 # capacitor -> voltage source vC (nodes[0] +), value via overrides
                comps.append(Component(id=c.id, type="V", nodes=list(c.nodes), value="0"))
            elif c.type.value == "L":               # inductor -> current source iL (nodes[0] -> nodes[1])
                comps.append(Component(id=c.id, type="I", nodes=list(c.nodes), value="0"))
            else:
                comps.append(c)
        for d in self.dj.diodes:
            parts = self.dj._parts(d, states[d.id], off_as_r=off_as_r)
            if rmin and states[d.id] in ("on", "bz") and not (d.rd if states[d.id] == "on" else d.rz):
                p = parts[0]                         # V source p -> n: put R_MIN after it
                x = f"{d.id}_m"
                parts = [Component(id=p.id, type="V", nodes=[p.nodes[0], x], value=p.value),
                         Component(id=f"{d.id}_rm", type="R", nodes=[x, p.nodes[1]], value=repr(R_MIN))]
            comps += parts
        return comps

    def _row(self, expr) -> np.ndarray:
        """expr linear in y, s -> row over z = [y..., s, k, 1]."""
        e = sp.expand(sp.sympify(expr))
        extra = e.free_symbols - set(self.ysyms) - {S}
        if extra:
            raise NotSupportedError(f"Waveforms need numeric component values ({sorted(map(str, extra))})")
        row = [float(sp.N(e.coeff(y, 1))) for y in self.ysyms] + [float(sp.N(e.coeff(S, 1))), 0.0]
        const = e.subs({**{y: 0 for y in self.ysyms}, S: 0})
        row.append(float(sp.N(const)))
        r = np.array(row)
        r[np.abs(r) < 1e-15] = 0.0
        return r

    def _overrides(self) -> Dict[str, sp.Expr]:
        ov = {s.id: sp.nsimplify(s.offset, rational=True) + s.sign * sp.nsimplify(s.peak, rational=True) * S
              for s in self.src}
        ov.update({c.id: y for c, y in zip(self.caps + self.inds, self.ysyms)})     # the state variables
        return ov

    def _combo(self, states: Dict[str, str], wanted: List[Tuple[str, str, str, Optional[str]]]) -> Optional[Combo]:
        m, rmin = None, False
        for off_as_r, rm in ((False, False), (True, False), (False, True), (True, True)):
            try:
                m = MNA(self._components(states, off_as_r, rm), "dc", overrides=self._overrides(),
                        conventions=self.conv)
                m.solve()
                rmin = rm
                break
            except (SolverError, NotSupportedError):
                m = None
        if m is None:
            return None
        nz = self.n + 3
        M = np.zeros((nz, nz))
        for i, c in enumerate(self.caps):           # dvC/dt = iC / C (iC = branch current, nodes[0] -> nodes[1])
            M[i] = self._row(m.evaluate(m.branch[c.id])) / parse_value(c.value)
        for j, c in enumerate(self.inds):           # diL/dt = vL / L
            k = len(self.caps) + j
            M[k] = self._row(m.evaluate(m.V[c.nodes[0]] - m.V[c.nodes[1]])) / parse_value(c.value)
        M[self.n, self.n + 1] = self.w              # s' = ω k
        M[self.n + 1, self.n] = -self.w             # k' = −ω s
        margins = []
        for d in self.dj.diodes:
            st = states[d.id]
            if st == "off":
                vak = m.evaluate(m.V[d.a] - m.V[d.k])
                margins.append(self._row(d.Von - vak))
                if d.kind == "zener":
                    margins.append(self._row(d.VZ + vak))
            else:
                margins.append(self._row(m.evaluate(m.branch[d.id])))
        cb = Combo(states, M, margins, rmin=rmin)
        for name, kind, a, b in wanted:
            cb.q[name] = self._row(self._quantity(m, states, kind, a, b))
        return cb

    def _quantity(self, m: MNA, states, kind: str, a: str, b: Optional[str]):
        if kind == "node":
            return m.evaluate(m.V[a] - m.V[b or GROUND])
        if kind == "piv":
            d = self.dj.by_id[a]
            return -m.evaluate(m.V[d.a] - m.V[d.k])
        if kind == "cur":
            if a in self.dj.by_id:
                return sp.Integer(0) if states[a] == "off" else m.evaluate(m.branch[a])
            c = next((x for x in self.nl.components if x.id == a), None)
            if c is None:
                raise SolverError(f"No component {a}")
            if c.type.value == "C":
                return m.evaluate(m.branch[c.id])
            if c.type.value == "L":
                return self.ysyms[len(self.caps) + self.inds.index(c)]
            return m.evaluate(m.current(m.by_id[a])[0])
        raise SolverError(kind)

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------
    def _z(self, y: np.ndarray, t: float) -> np.ndarray:
        return np.concatenate([y, [math.sin(self.w * t), math.cos(self.w * t), 1.0]])

    def _pick(self, z: np.ndarray, tol: float) -> Combo:
        """The diode states valid at this instant (look a tiny bit ahead to break ties at a switch)."""
        best, best_m = None, -math.inf
        for cb in self.combos:
            if cb.margin(z) < -tol:
                continue
            mg = cb.margin(cb.Ed @ z)
            if mg > best_m:
                best, best_m = cb, mg
        if best is None:
            best = max(self.combos, key=lambda c: c.margin(z))
        return best

    def _cycle(self, y0: np.ndarray, record: bool = False):
        h = self.T / STEPS
        z, t = self._z(y0, 0.0), 0.0
        tol = 1e-9 * max(1.0, abs(self.Vm))
        cb = self._pick(z, tol)
        samples = [(0.0, z.copy(), cb)] if record else None
        switches: List[Tuple[float, Dict[str, str]]] = [(0.0, cb.states)]
        fine_until = -1.0                                  # record: dense samples for 2 steps after a switch
        for _ in range(STEPS):
            remaining = h
            for _guard in range(20):                       # a few switches inside one step at most
                zn = cb.E @ z if remaining == h else expm(cb.M * remaining) @ z
                if cb.margin(zn) >= -tol:
                    if record and t < fine_until:          # the turn-on current spike lasts ~R·C: sample it
                        for k in range(1, 40):
                            zk = expm(cb.M * (remaining * k / 40)) @ z
                            samples.append((t + remaining * k / 40, zk, cb))
                    z, t = zn, t + remaining
                    break
                lo, hi = 0.0, remaining                    # bisection for the switching instant
                for _b in range(60):
                    mid = 0.5 * (lo + hi)
                    if cb.margin(expm(cb.M * mid) @ z) >= -tol:
                        lo = mid
                    else:
                        hi = mid
                z = expm(cb.M * lo) @ z
                t += lo
                remaining -= lo
                new = self._pick(z, tol)
                if record:
                    samples.append((t, z.copy(), cb))       # end of the old state
                    samples.append((t, z.copy(), new))      # start of the new one
                if new is cb:                               # numerically stuck at the boundary: finish the step
                    z = expm(cb.M * remaining) @ z
                    t += remaining
                    break
                cb = new
                switches.append((t, cb.states))
                fine_until = t + 2 * h
                if remaining <= 1e-15 * self.T:
                    break
            if record:
                samples.append((t, z.copy(), cb))
        return z[: self.n], samples, switches

    def _steady(self) -> np.ndarray:
        """y0 with y(T) = y0: Newton's method on the cycle map (piecewise affine: a few iterations)."""
        y0 = np.zeros(self.n)
        scale = max(1.0, abs(self.Vm))
        for _ in range(3):                                   # a few plain cycles first: a good starting point
            y0, _, _ = self._cycle(y0)
        for _ in range(40):
            y1, _, _ = self._cycle(y0)
            F = y1 - y0
            if np.max(np.abs(F)) <= 1e-10 * scale:
                return y0
            J = np.zeros((self.n, self.n))
            eps = 1e-6 * scale
            for i in range(self.n):
                e = np.zeros(self.n)
                e[i] = eps
                J[:, i] = (self._cycle(y0 + e)[0] - y1) / eps
            try:
                step = np.linalg.solve(J - np.eye(self.n), -F)
            except np.linalg.LinAlgError:
                step = F                                     # fall back to one plain cycle
            y0 = y0 + step
        y1, _, _ = self._cycle(y0)
        if np.max(np.abs(y1 - y0)) > 1e-6 * scale:
            raise SolverError("The circuit did not settle to a periodic steady state")
        return y0

    # ------------------------------------------------------------------
    def run(self, queries: List[str]) -> SolveResult:
        wanted, plan = [], []
        for q in queries:
            mq = _Q.match(q)
            kind, a, b = (mq.group(1), mq.group(2), mq.group(3)) if mq else (q, None, None)
            if kind not in KINDS or a is None:
                self.res.answers[q] = (f"Could not compute: '{q}' is not available for filtered rectifiers / "
                                       f"clampers (ask for Vmax, Vmin, Vavg, Vripple, ripple, Imax, conduction, PIV)")
                self.res.numeric[q] = None
                continue
            if kind == "bias" or kind == "conduction":
                if a not in self.dj.by_id:
                    self.res.answers[q] = f"Could not compute: {kind}() needs a diode"
                    self.res.numeric[q] = None
                    continue
                plan.append((q, kind, a, b, None))
                continue
            if kind == "PIV":
                if a not in self.dj.by_id:
                    self.res.answers[q] = "Could not compute: PIV() needs a diode"
                    self.res.numeric[q] = None
                    continue
                spec = ("piv", a, None)
            elif kind.startswith("V") or kind == "ripple":
                spec = ("node", a, b)
            else:
                spec = ("cur", a, None)
            name = f"{spec[0]}:{spec[1]}:{spec[2]}"
            if name not in [w[0] for w in wanted]:
                wanted.append((name, *spec))
            plan.append((q, kind, a, b, name))

        self.combos: List[Combo] = []
        for combo in itertools.product(*[d.states for d in self.dj.diodes]):
            states = {d.id: st for d, st in zip(self.dj.diodes, combo)}
            cb = self._combo(states, wanted)
            if cb is not None:
                cb.E = expm(cb.M * (self.T / STEPS))
                cb.Ed = expm(cb.M * (self.T / STEPS * 1e-3))
                self.combos.append(cb)
        if not self.combos:
            raise SolverError("No diode state gives a solvable circuit")

        y0 = self._steady()
        _, samples, switches = self._cycle(y0, record=True)
        self._steps(y0, switches)

        for q, kind, a, b, name in plan:
            try:
                self._answer(q, kind, a, b, name, samples, switches)
            except (SolverError, NotSupportedError) as e:
                self.res.answers[q] = f"Could not compute: {e}"
                self.res.numeric[q] = None
        # for the verifier: the steady-state capacitor voltages / inductor currents at t = 0
        self.res.symbolic["tran.ic"] = json.dumps({c.id: float(v) for c, v in zip(self.caps + self.inds, y0)})
        self.res.symbolic["tran.f"] = repr(self.f)
        self.res.symbolic["wave.Vm"] = repr(self.main.peak * self.main.sign)
        self.res.symbolic["wave.sources"] = json.dumps({s.id: [s.peak * s.sign, s.offset] for s in self.src})
        for d in self.dj.diodes:
            self.res.symbolic[f"diode.{d.id}"] = json.dumps(d.as_dict())
        self.res.verification_notes += [f"info: {n}" for n in self.notes]
        if queries and all(self.res.answers.get(q, "").startswith("Could not compute") for q in queries):
            raise SolverError(" | ".join(self.res.answers.values()))
        self.res.trace = self.trace
        return self.res

    # ------------------------------------------------------------------
    def _deg(self, t: float) -> float:
        return 360.0 * t / self.T

    def _intervals(self, switches, did: str) -> List[Tuple[float, float]]:
        """(start, end) times in the cycle where diode `did` conducts."""
        out, start = [], None
        seq = list(switches) + [(self.T, None)]
        for (t, st), (t2, _) in zip(seq, seq[1:]):
            on = st[did] != "off"
            if on and start is None:
                start = t
            if not on and start is not None:
                out.append((start, t))
                start = None
        if start is not None:
            out.append((start, self.T))
        if len(out) > 1 and out[0][0] == 0.0 and out[-1][1] == self.T:     # wraps around t = 0
            out = [(out[-1][0], out[0][1] + self.T)] + out[1:-1]
        return out

    def _steps(self, y0: np.ndarray, switches) -> None:
        for d in self.dj.diodes:
            desc, rule = d.label(self.dj.VT)
            self.trace.add(f"Diode model — {desc}", rule, "; ".join(d.notes) or None)
        for s in self.src:
            self.trace.add(f"Input {s.id}: v(t) = {'−' if s.sign < 0 else ''}Vm·sin(ωt)", s.how
                           + f";  f = {fmt_si(self.f, 'Hz')}" + (" (assumed)" if self.notes else ""),
                           f"Vm = {fmt_si(s.peak, 'V', self.sig)}")
        self.trace.add("The capacitor remembers, so solve in time: in each diode state the circuit is linear "
                       "(exact exponential + sinusoid); a diode switches when its check reaches zero; the steady "
                       "state is the cycle that repeats exactly",
                       "dvC/dt = iC/C in each state;  z(t + h) = e^{M·h}·z(t);  find vC(0) with vC(T) = vC(0) "
                       "(Newton's method)")
        ic = ";  ".join(f"v_{c.id} = {fmt_si(v, 'V', self.sig)}" for c, v in zip(self.caps, y0))
        if ic:
            self.trace.add("Steady state: capacitor voltage at the start of each cycle (t = 0, vin = 0)", None, ic)
        for d in self.dj.diodes:
            iv = self._intervals(switches, d.id)
            if iv:
                txt = ";  ".join(f"from {fmt_num(self._deg(a), 4)}° to {fmt_num(self._deg(b), 4)}°" for a, b in iv)
            else:
                txt = "never"
            self.trace.add(f"{d.id} conducts (angle in the input cycle, 0° = vin rising through 0)", None, txt)
        tau = self._discharge_tau()
        if tau:
            self.trace.add("With every diode OFF the capacitor discharges through the load",
                           f"τ = R·C = {fmt_si(tau, 's', self.sig)}  (period T = {fmt_si(self.T, 's', self.sig)})",
                           f"τ/T = {fmt_num(tau / self.T, self.sig)}")

    def _discharge_tau(self) -> Optional[float]:
        if len(self.caps) != 1 or self.inds:
            return None
        off = next((c for c in self.combos if all(s == "off" for s in c.states.values())), None)
        if off is None:
            return None
        a = off.M[0, 0]
        return -1.0 / a if a < 0 else None

    def _pulses(self, switches) -> int:
        """Charging pulses per cycle: how many times 'some diode conducts' starts."""
        on = [any(s != "off" for s in st.values()) for _, st in switches]
        starts = sum(1 for i in range(len(on)) if on[i] and not on[i - 1]) if len(on) > 1 else int(on[0])
        return max(starts, 1 if any(on) else 0)

    def _approx(self, samples, switches, vmax: float, node: Optional[str] = None) -> Optional[Dict[str, Tuple[float, str]]]:
        """Textbook capacitor-input filter approximations: ONE capacitor from the output to ground, discharging
        through the load R (not a clamper, where the capacitor is in series)."""
        if len(self.caps) != 1 or GROUND not in self.caps[0].nodes or (node and node not in self.caps[0].nodes):
            return None
        tau = self._discharge_tau()
        k = self._pulses(switches)
        if not tau or k not in (1, 2) or vmax <= 0:
            return None
        c = self.caps[0]
        C = parse_value(c.value)
        R = tau / C
        vp = vmax
        vr = vp / (k * self.f * R * C)
        vdc = vp - vr / 2
        il = vp / R
        return {"Vripple": (vr, f"Vp/({'2' if k == 2 else ''}f·R·C)"), "Vavg": (vdc, "Vp − Vr/2"),
                "ripple": (vr / (2 * math.sqrt(3) * vdc), "Vr/(2√3·Vdc)"),
                "conduction": (math.degrees(math.sqrt(2 * vr / vp)), "√(2Vr/Vp)"),
                "Imax": (il * (1 + (2 * math.pi / k) * math.sqrt(vp / (2 * vr))), "I_L·(1 + (2π/k)·√(Vp/2Vr))"),
                "Vmin": (vp - vr, "Vp − Vr")}

    def _series(self, samples, name: str):
        t = np.array([s[0] for s in samples])
        v = np.array([float(s[2].q[name] @ s[1]) for s in samples])
        return t, v

    def _answer(self, q, kind, a, b, name, samples, switches) -> None:
        if kind == "conduction":
            iv = self._intervals(switches, a)
            deg = sum(self._deg(e - s) for s, e in iv)
            ap = None
            vq = self._out_name(samples)
            if vq:
                tt, vv = self._series(samples, vq)
                ap = (self._approx(samples, switches, float(np.max(vv)), vq.split(":")[1]) or {}).get("conduction")
            self.res.answers[q] = f"{fmt_num(deg, self.sig)}° per cycle" + (
                f" (exact; textbook approximation {ap[1]} = {fmt_num(ap[0], self.sig)}°)" if ap else "")
            self.res.numeric[q] = deg
            if ap:
                self.res.numeric[q + ".approx"] = ap[0]
            return
        if kind == "bias":
            iv = self._intervals(switches, a)
            self.res.answers[q] = (f"{a} conducts " + ";  ".join(f"from {fmt_num(self._deg(s), 4)}° to "
                                                                   f"{fmt_num(self._deg(e), 4)}°" for s, e in iv)
                                   if iv else f"{a} never conducts in the steady state")
            self.res.numeric[q] = None
            return
        t, v = self._series(samples, name)
        mx, mn = float(np.max(v)), float(np.min(v))
        avg = float(np.trapezoid(v, t) / self.T) if hasattr(np, "trapezoid") else float(np.trapz(v, t) / self.T)
        sq = float(np.trapezoid(v * v, t) / self.T) if hasattr(np, "trapezoid") else float(np.trapz(v * v, t) / self.T)
        rms = math.sqrt(max(sq, 0.0))
        unit = "A" if kind in ("I", "Imax", "Iavg", "Irms") else "V"
        scale = max(abs(mx), abs(mn), 1e-30)
        avg = 0.0 if abs(avg) < 1e-12 * scale else avg
        if unit == "V" and kind != "PIV":
            ap = self._approx(samples, switches, mx, a if b in (None, GROUND) else "?")
        elif kind == "Imax":
            vq = self._out_name(samples)
            ap = self._approx(samples, switches, float(np.max(self._series(samples, vq)[1])), vq.split(":")[1]) \
                if vq else None
        else:
            ap = None
        vals = {"max": mx, "min": mn, "avg": avg, "rms": rms, "pp": mx - mn,
                "ripple": math.sqrt(max(sq - avg * avg, 0.0)) / abs(avg) if abs(avg) > 1e-30 else math.inf}

        def txt(val: float, akey: Optional[str], u: str = unit) -> str:
            s = fmt_si(val, u, self.sig) if u else fmt_num(val, self.sig)
            if ap and akey in ap:
                av, f = ap[akey]
                s += f" (exact steady state; textbook approximation {f} = " + (
                    fmt_si(av, u, self.sig) if u else fmt_num(av, self.sig)) + ")"
                self.res.numeric[q + ".approx"] = av
            return s

        if kind in ("V", "I"):
            self.res.answers[q] = (f"peak {fmt_si(mx, unit, self.sig)}, minimum {fmt_si(mn, unit, self.sig)}, "
                                   f"average {fmt_si(avg, unit, self.sig)}, peak-to-peak "
                                   f"{fmt_si(mx - mn, unit, self.sig)}")
            self.res.numeric[q] = mx
            for k_, v_ in (("max", mx), ("min", mn), ("avg", avg), ("rms", rms), ("pp", mx - mn)):
                self.res.numeric[f"{q}.{k_}"] = v_
            return
        sel = {"Vmax": ("max", None), "Imax": ("max", "Imax"), "PIV": ("max", None), "Vmin": ("min", "Vmin"),
               "Vavg": ("avg", "Vavg"), "Iavg": ("avg", None), "Vrms": ("rms", None), "Irms": ("rms", None),
               "Vripple": ("pp", "Vripple"), "Vpp": ("pp", "Vripple"), "ripple": ("ripple", "ripple")}[kind]
        val = vals[sel[0]]
        if kind == "PIV":
            val = max(val, 0.0)
        self.res.answers[q] = txt(val, sel[1], "" if kind == "ripple" else unit)
        self.res.numeric[q] = val
        label = {"max": "Peak", "min": "Minimum", "avg": "Average (DC) value", "rms": "RMS value",
                 "pp": "Peak-to-peak ripple", "ripple": "Ripple factor (rms of the ripple / DC value)"}[sel[0]]
        self.trace.add(f"{label}: {q}", "measured on the steady-state cycle", f"{q} = {self.res.answers[q]}")

    def _out_name(self, samples) -> Optional[str]:
        names = [n for n in (samples[0][2].q if samples else {}) if n.startswith("node:")]
        return names[0] if names else None