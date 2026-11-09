"""
Diode circuits with a sinusoidal input and no capacitors (Phase 3, step 2a):
half-wave / full-wave / bridge rectifiers, clippers (limiters), diode logic with AC inputs.

Why this is exact and simple: with no capacitor or inductor the circuit has no memory, so the output at
any instant depends only on the input at that instant. Write the input as vin = Vm·x with x = sin(ωt):

  1. For every combination of diode states, solve the linear circuit with x kept as a SYMBOL (MNA).
     Every voltage and current is then a straight line in x:  vo = a·x + b.
  2. Each diode's check (ON: I_D >= 0, OFF: V_AK < Von, breakdown: I_Z >= 0) is a linear inequality in x,
     so each state combination is valid on an exact interval of x. The intervals tile [-1, 1]: that is the
     transfer characteristic vo(vin), piecewise linear with exact breakpoints.
  3. Over one cycle x = sin θ, so averages and RMS values are exact integrals of a·sin θ + b:
        ∫(a sin θ + b) dθ   = −a cos θ + b θ
        ∫(a sin θ + b)² dθ  = a²(θ/2 − sin 2θ/4) − 2ab cos θ + b² θ
     Peaks are at the interval ends (straight lines), PIV = the largest reverse voltage −V_AK.

Sinusoidal sources: "ac" = peak amplitude, or params {"Vrms": ..} / {"Vpp": ..}, optional {"ratio": N1/N2}
for a transformer (the value given is then the primary). All sinusoids share one sin(ωt); "ac_phase" 180
inverts one (the second half of a centre-tapped secondary).
"""
from __future__ import annotations

import itertools
import json
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import sympy as sp

from schemas import Component, Netlist, SolveResult, StepTrace, parse_value
from solvers.base import NotSupportedError, SolverError, fmt_num, fmt_si, sig_figs
from solvers.mna import GROUND, MNA

X = sp.Symbol("x", real=True)                   # x = sin(ωt) = vin / Vm
WAVE_KINDS = {"V", "Vmax", "Vmin", "Vavg", "Vrms", "I", "Imax", "Iavg", "Irms", "PIV", "conduction", "vtc", "bias"}
_Q = re.compile(r"^(\w+)\((\w+)(?:,(\w+))?\)$")
EPS = 1e-12


def _param(c: Component, *names):
    p = {str(k).strip().lower().replace(" ", "_"): v for k, v in (c.params or {}).items()}
    for n in names:
        if p.get(n) not in (None, ""):
            v = p[n]
            return float(v) if isinstance(v, (int, float)) else parse_value(str(v))
    return None


@dataclass
class Sinusoid:
    id: str
    peak: float            # secondary peak if a transformer ratio is given
    sign: int              # +1, or -1 for ac_phase = 180
    offset: float
    how: str               # text for the steps


def sinusoids(nl: Netlist) -> List[Sinusoid]:
    out = []
    for c in nl.components:
        if c.type.value not in ("V", "I"):
            continue
        rms, vpp = _param(c, "vrms", "rms", "irms"), _param(c, "vpp", "ipp")
        vp = _param(c, "vp", "vpeak", "vm", "peak")
        if c.ac in (None, "") and rms is None and vpp is None and vp is None:
            continue
        unit = "V" if c.type.value == "V" else "A"
        if c.ac not in (None, "") or vp is not None:
            peak = parse_value(c.ac) if c.ac not in (None, "") else vp
            how = f"peak {fmt_si(peak, unit)}"
        elif rms is not None:
            peak, how = rms * math.sqrt(2), f"{fmt_si(rms, unit)} rms × √2 = {fmt_si(rms * math.sqrt(2), unit)} peak"
        else:
            peak, how = vpp / 2, f"{fmt_si(vpp, unit)} peak-to-peak ÷ 2 = {fmt_si(vpp / 2, unit)} peak"
        ratio = _param(c, "ratio", "turns_ratio", "n1_n2", "n")
        if ratio:
            how += f"; transformer {fmt_num(ratio)}:1 → secondary peak = {fmt_si(peak, unit)} / {fmt_num(ratio)}"
            peak /= ratio
            how += f" = {fmt_si(peak, unit)}"
        ph = float(c.ac_phase or 0.0) % 360
        if abs(ph) < 1e-9:
            sign = 1
        elif abs(ph - 180) < 1e-9:
            sign = -1
        else:
            raise NotSupportedError("Sinusoidal sources must be in phase or 180° apart (a common sin(ωt))")
        offset = parse_value(c.value) if c.value not in (None, "") else 0.0
        out.append(Sinusoid(c.id, peak, sign, offset, how))
    return out


@dataclass
class Region:
    lo: float               # x range (x = vin/Vm)
    hi: float
    states: Dict[str, str]
    m: MNA
    conds: List[str]        # the checks, as text in vin


def lin(expr) -> Tuple[float, float]:
    """expr (linear in x) -> (slope, constant)."""
    e = sp.expand(sp.sympify(expr))
    if e.free_symbols - {X}:
        raise NotSupportedError("Waveforms need numeric component values")
    a = float(sp.N(e.coeff(X, 1)))
    b = float(sp.N(e.subs(X, 0)))
    # leakage of OFF diodes modelled as 1 TΩ leaves ~1e-10-size terms: they are zero
    return (0.0 if abs(a) < 1e-9 else a), (0.0 if abs(b) < 1e-9 else b)


def theta_pieces(lo: float, hi: float) -> List[Tuple[float, float]]:
    """θ intervals (one period, θ in [-π/2, 3π/2)) where lo <= sin θ <= hi."""
    a, b = math.asin(max(-1.0, min(1.0, lo))), math.asin(max(-1.0, min(1.0, hi)))
    return [(a, b), (math.pi - b, math.pi - a)]


def integrals(a: float, b: float, t1: float, t2: float) -> Tuple[float, float]:
    """(∫ f dθ, ∫ f² dθ) for f = a sin θ + b over [t1, t2]."""
    def F1(t):
        return -a * math.cos(t) + b * t

    def F2(t):
        return a * a * (t / 2 - math.sin(2 * t) / 4) - 2 * a * b * math.cos(t) + b * b * t
    return F1(t2) - F1(t1), F2(t2) - F2(t1)


class WaveJob:
    def __init__(self, nl: Netlist, conventions: Dict, djob):
        self.nl, self.conv, self.dj = nl, conventions or {}, djob      # djob: the _DiodeJob (models, parts)
        self.sig = sig_figs(conventions)
        self.trace = StepTrace()
        self.res = SolveResult(method="Diode states solved with the input as a variable (exact piecewise-linear "
                                      "transfer characteristic), then exact integrals over one cycle")
        self.src = sinusoids(nl)
        if not self.src:
            raise NotSupportedError("A rectifier / clipper needs a sinusoidal input: give the source an \"ac\" "
                                    "peak amplitude (or Vrms / Vpp)")
        if any(d.kind == "exp" for d in djob.diodes):
            raise NotSupportedError("Waveforms use the piecewise-linear diode models (ideal / 0.7 V / Zener), "
                                    "not the exponential model")
        self.main = self.src[0]
        self.Vm = self.main.peak * self.main.sign      # vin = Vm·x for the main source

    # ------------------------------------------------------------------
    def _overrides(self) -> Dict[str, sp.Expr]:
        return {s.id: sp.nsimplify(s.offset, rational=True) + s.sign * sp.nsimplify(s.peak, rational=True) * X
                for s in self.src}

    def _interval(self, f, lo: float, hi: float) -> Tuple[float, float]:
        """Narrow [lo, hi] to where f(x) = a·x + b >= 0."""
        a, b = lin(f)
        if abs(a) < 1e-15:
            return (lo, hi) if b >= -1e-12 else (1.0, -1.0)
        root = -b / a
        return (max(lo, root), hi) if a > 0 else (lo, min(hi, root))

    def _vin(self, x: float) -> str:
        return fmt_si(self.Vm * x, "V", self.sig)

    def _line(self, f, unit: str = "V", var: str = "vin") -> str:
        """a·x + b as text in vin (vin = Vm·x)."""
        a, b = lin(f)
        k = a / self.Vm if self.Vm else 0.0                 # slope per volt of vin
        if abs(k) < 1e-12:
            return fmt_si(b, unit, self.sig)
        term = var if abs(k - 1) < 1e-12 else (f"−{var}" if abs(k + 1) < 1e-12 else f"{fmt_num(k, self.sig)}·{var}")
        if abs(b) < 1e-12 * max(1.0, abs(self.Vm)):
            return term
        sign = "+" if b > 0 else "−"
        return f"{term} {sign} {fmt_si(abs(b), unit, self.sig)}"

    # ------------------------------------------------------------------
    def regions(self) -> List[Region]:
        pwl = self.dj.diodes
        combos = list(itertools.product(*[d.states for d in pwl]))
        found: List[Region] = []
        for combo in combos:
            states = {d.id: st for d, st in zip(pwl, combo)}
            m = None
            for off_as_r in (False, True):       # True: OFF diodes as 1 TΩ, when they leave nodes floating
                try:
                    m = MNA(self.dj._components(states, off_as_r=off_as_r), "dc", overrides=self._overrides(),
                            conventions=self.conv)
                    m.solve()
                    break
                except SolverError:
                    m = None
            if m is None:
                continue
            lo, hi, conds = -1.0, 1.0, []
            for d in pwl:
                st = states[d.id]
                if st == "off":
                    vak = m.evaluate(m.V[d.a] - m.V[d.k])
                    lo, hi = self._interval(d.Von - vak, lo, hi)
                    conds.append(f"{d.id} OFF: V_AK = {self._line(vak)} < {fmt_si(d.Von, 'V')}")
                    if d.kind == "zener":
                        lo, hi = self._interval(d.VZ + vak, lo, hi)
                        conds[-1] += f" and V_KA < {fmt_si(d.VZ, 'V')}"
                else:
                    i = m.evaluate(m.branch[d.id])
                    lo, hi = self._interval(i, lo, hi)
                    what = "I_D" if st == "on" else "I_Z"
                    conds.append(f"{d.id} {'ON' if st == 'on' else 'in breakdown'}: {what} = {self._line(i, 'A')} ≥ 0")
                if hi - lo <= 1e-12:
                    break
            if hi - lo > 1e-12:
                found.append(Region(lo, hi, states, m, conds))
        found.sort(key=lambda r: r.lo)
        # keep a tiling of [-1, 1]: drop regions that overlap one already kept (equal answers there anyway)
        tiles: List[Region] = []
        for r in found:
            if tiles and r.lo < tiles[-1].hi - 1e-9:
                if r.hi <= tiles[-1].hi + 1e-9:
                    continue
                r = Region(tiles[-1].hi, r.hi, r.states, r.m, r.conds)
            tiles.append(r)
        if not tiles or tiles[0].lo > -1 + 1e-9 or tiles[-1].hi < 1 - 1e-9 or any(
                b.lo - a.hi > 1e-9 for a, b in zip(tiles, tiles[1:])):
            raise SolverError("Could not find consistent diode states over the whole input cycle: check the diode "
                              "directions and the source polarities")
        return tiles

    # ------------------------------------------------------------------
    def _expr(self, r: Region, kind: str, a: str, b: Optional[str]):
        """The asked quantity in region r, as an expression in x."""
        m = r.m
        if kind.startswith("V") or kind == "vtc":
            return m.evaluate(m.V[a] - m.V[b or GROUND])
        if kind == "PIV":
            d = self.dj.by_id[a]
            return -m.evaluate(m.V[d.a] - m.V[d.k])         # reverse voltage across the diode
        if a in self.dj.by_id:
            d = self.dj.by_id[a]
            st = r.states[a]
            return sp.Integer(0) if st == "off" else m.evaluate(m.branch[d.id])
        c = m.by_id.get(a)
        if c is None:
            raise SolverError(f"No component {a}")
        return m.evaluate(m.current(c)[0])

    def _stats(self, tiles: List[Region], kind: str, a: str, b: Optional[str]) -> Dict[str, float]:
        mx, mn, s1, s2 = -math.inf, math.inf, 0.0, 0.0
        for r in tiles:
            fa, fb = lin(self._expr(r, kind, a, b))
            for x in (r.lo, r.hi):
                mx, mn = max(mx, fa * x + fb), min(mn, fa * x + fb)
            for t1, t2 in theta_pieces(r.lo, r.hi):
                i1, i2 = integrals(fa, fb, t1, t2)
                s1, s2 = s1 + i1, s2 + i2
        out = {"max": mx, "min": mn, "avg": s1 / (2 * math.pi), "rms": math.sqrt(max(s2, 0.0) / (2 * math.pi))}
        scale = max(abs(mx), abs(mn), 1e-30)
        for k in out:                                   # rounding noise of exact zeros (7e-16 V) -> 0
            if abs(out[k]) < 1e-12 * scale:
                out[k] = 0.0
        return out

    def _shape(self, tiles: List[Region], a: str, b: Optional[str]) -> Optional[str]:
        """'half' / 'full' when the output looks like an (ideal-shaped) rectifier output, for the textbook
        approximations Vavg = Vp/π (half-wave) or 2Vp/π (full-wave)."""
        def f(x):
            r = next(t for t in tiles if t.lo - 1e-12 <= x <= t.hi + 1e-12)
            fa, fb = lin(self._expr(r, "V", a, b))
            return fa * x + fb
        pts = [0.13, 0.37, 0.61, 0.89, 1.0]
        pos, neg = [f(p) for p in pts], [f(-p) for p in pts]
        scale = max(abs(v) for v in pos + neg) or 1.0
        if all(abs(p - n) <= 1e-9 * scale for p, n in zip(pos, neg)) and max(pos) > 1e-9 * scale:
            return "full"
        if all(abs(n) <= 1e-9 * scale for n in neg) and all(p >= -1e-9 * scale for p in pos) and max(pos) > 0:
            return "half"
        if all(abs(p) <= 1e-9 * scale for p in pos) and all(n <= 1e-9 * scale for n in neg) and min(neg) < 0:
            return "half"
        return None

    # ------------------------------------------------------------------
    def run(self, queries: List[str]) -> SolveResult:
        for d in self.dj.diodes:
            desc, rule = d.label(self.dj.VT)
            self.trace.add(f"Diode model — {desc}", rule, "; ".join(d.notes) or None)
        for s in self.src:
            self.trace.add(f"Input {s.id}: v(t) = {'−' if s.sign < 0 else ''}Vm·sin(ωt)"
                           + (f" + {fmt_si(s.offset, 'V')}" if s.offset else ""),
                           s.how + (f";  f = {fmt_si(parse_value(self.nl.frequency), 'Hz')}" if self.nl.frequency else ""),
                           f"Vm = {fmt_si(s.peak, 'V', self.sig)}")
        self.trace.add("No capacitors: at every instant the circuit is resistive, so the output depends only on "
                       "the present input. Solve it with vin as a variable for each diode state; each state holds "
                       "over a range of vin",
                       f"vin = Vm·sin(ωt) runs from {self._vin(-1)} to {self._vin(1)}")
        tiles = self.regions()
        out_node = self._out_node(queries)
        for r in tiles:
            lo, hi = sorted((self.Vm * r.lo, self.Vm * r.hi))
            st = ", ".join(self.dj._state_text(self.dj.by_id[i], s) for i, s in r.states.items())
            eq = f"vo = {self._line(r.m.evaluate(r.m.V[out_node]))}" if out_node else None
            self.trace.add(f"vin from {fmt_si(lo, 'V', self.sig)} to {fmt_si(hi, 'V', self.sig)}: {st}", eq,
                           ";  ".join(r.conds) or None)

        for q in queries:
            mq = _Q.match(q)
            kind, a, b = (mq.group(1), mq.group(2), mq.group(3)) if mq else (q, None, None)
            try:
                self._answer(q, kind, a, b, tiles)
            except (SolverError, NotSupportedError) as e:
                self.res.answers[q] = f"Could not compute: {e}"
                self.res.numeric[q] = None
        # for the verifier: the transfer characteristic of every node, the input scale, the diode models
        self.res.symbolic["wave.Vm"] = repr(self.Vm)
        self.res.symbolic["wave.sources"] = json.dumps({s.id: [s.peak * s.sign, s.offset] for s in self.src})
        for d in self.dj.diodes:
            self.res.symbolic[f"diode.{d.id}"] = json.dumps(d.as_dict())
        if queries and all(self.res.answers.get(q, "").startswith("Could not compute") for q in queries):
            raise SolverError(" | ".join(self.res.answers.values()))
        self.res.trace = self.trace
        return self.res

    def _out_node(self, queries: List[str]) -> Optional[str]:
        for q in queries:
            mq = _Q.match(q)
            if mq and (mq.group(1).startswith("V") or mq.group(1) == "vtc") and mq.group(3) in (None, GROUND):
                return mq.group(2)
        return "out" if any("out" in c.nodes for c in self.nl.components) else None

    def _answer(self, q: str, kind: str, a: Optional[str], b: Optional[str], tiles: List[Region]) -> None:
        if kind not in WAVE_KINDS or a is None:
            raise SolverError(f"'{q}' is not available for waveforms (ask for Vmax, Vmin, Vavg, Vrms, PIV, "
                              f"Imax, Iavg, conduction or vtc)")
        diodes = self.dj.by_id
        if kind in ("PIV", "conduction") and a not in diodes:
            raise SolverError(f"{kind}() needs a diode, {a} is not one")
        unit = "V" if kind.startswith("V") or kind in ("PIV", "vtc") else "A"
        name = f"V_{a}{b}" if b else (f"v_{a}" if unit == "V" else f"i_{a}")

        if kind == "vtc":
            parts = []
            for r in tiles:
                lo, hi = sorted((self.Vm * r.lo, self.Vm * r.hi))
                parts.append(f"{fmt_si(lo, 'V', self.sig)} ≤ vin ≤ {fmt_si(hi, 'V', self.sig)}: "
                             f"vo = {self._line(self._expr(r, 'V', a, b))}")
            self.res.answers[q] = ";  ".join(parts)
            self.res.numeric[q] = None
            self.res.symbolic[q] = json.dumps([[r.lo, r.hi, *lin(self._expr(r, "V", a, b))] for r in tiles])
            self.trace.add(f"Transfer characteristic of {a}", None, self.res.answers[q])
            return

        if kind == "conduction":
            on = sum(t2 - t1 for r in tiles if r.states[a] != "off" for t1, t2 in theta_pieces(r.lo, r.hi))
            deg = math.degrees(on)
            self.res.answers[q] = f"{fmt_num(deg, self.sig)}° of each 360° cycle"
            self.res.numeric[q] = deg
            self.trace.add(f"Conduction angle of {a}: the part of the cycle where it conducts",
                           "θ where sin θ is inside the ON ranges of vin", f"{fmt_num(deg, self.sig)}°")
            return

        if kind == "bias":
            d = diodes.get(a)
            if d is None:
                raise SolverError(f"bias() needs a diode, {a} is not one")
            on = [r for r in tiles if r.states[a] != "off"]
            if not on:
                txt = f"{a} never conducts (OFF over the whole cycle)"
            else:
                rng = "; ".join(f"{fmt_si(min(self.Vm * r.lo, self.Vm * r.hi), 'V', self.sig)} ≤ vin ≤ "
                                f"{fmt_si(max(self.Vm * r.lo, self.Vm * r.hi), 'V', self.sig)}" for r in on)
                txt = f"{a} conducts for {rng}"
            self.res.answers[q] = txt
            self.res.numeric[q] = None
            return

        st = self._stats(tiles, "PIV" if kind == "PIV" else kind, a, b)
        if kind == "PIV":
            piv = max(st["max"], 0.0)
            self.res.answers[q] = fmt_si(piv, "V", self.sig)
            self.res.numeric[q] = piv
            self.trace.add(f"Peak inverse voltage of {a}: the largest reverse voltage −V_AK over the cycle", None,
                           f"PIV = {fmt_si(piv, 'V', self.sig)}")
            return

        approx = None
        if unit == "V" and kind in ("Vavg", "Vrms", "V"):
            shape = self._shape(tiles, a, b)
            vp = max(abs(st["max"]), abs(st["min"]))
            if shape == "half":
                approx = {"avg": (vp / math.pi, "Vp/π"), "rms": (vp / 2, "Vp/2")}
            elif shape == "full":
                approx = {"avg": (2 * vp / math.pi, "2Vp/π"), "rms": (vp / math.sqrt(2), "Vp/√2")}

        def with_approx(which: str, val: float) -> str:
            txt = fmt_si(val, unit, self.sig)
            if approx:
                av, f = approx[which]
                txt += f" (exact for the diode model; textbook approximation {f} = {fmt_si(av, unit, self.sig)})"
                self.res.numeric[f"{q}.approx"] = av
            return txt

        prefix = {"max": "max", "min": "min", "avg": "avg", "rms": "rms"}
        if kind in ("V", "I"):
            self.res.answers[q] = (f"peak {fmt_si(st['max'], unit, self.sig)}, minimum {fmt_si(st['min'], unit, self.sig)}, "
                                   f"average {fmt_si(st['avg'], unit, self.sig)}, rms {fmt_si(st['rms'], unit, self.sig)}")
            self.res.numeric[q] = st["max"]
            for k in prefix:
                self.res.numeric[f"{q}.{k}"] = st[k]
            self.trace.add(f"Waveform of {name}", "peaks at the ends of the vin ranges; average and rms by exact "
                           "integration over one cycle", self.res.answers[q])
            return
        which = {"Vmax": "max", "Imax": "max", "Vmin": "min", "Vavg": "avg", "Iavg": "avg",
                 "Vrms": "rms", "Irms": "rms"}[kind]
        val = st[which]
        eq = {"max": f"largest value of {name} (at an end of a vin range)",
              "min": f"smallest value of {name}",
              "avg": f"{name},avg = (1/2π)·∫ {name} dθ over one cycle (exact)",
              "rms": f"{name},rms = √[(1/2π)·∫ {name}² dθ] over one cycle (exact)"}[which]
        self.res.answers[q] = with_approx(which, val) if which in ("avg", "rms") else fmt_si(val, unit, self.sig)
        self.res.numeric[q] = val
        self.trace.add({"max": "Peak", "min": "Minimum", "avg": "Average (DC) value", "rms": "RMS value"}[which]
                       + f" of {name}", eq, f"{kind}({a}{',' + b if b else ''}) = {self.res.answers[q]}")