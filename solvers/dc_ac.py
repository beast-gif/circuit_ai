"""
DC / AC solver for linear circuits (R, L, C, independent and controlled sources).

Runs the MNA engine and answers each query, writing a step trace as it goes:

  V(n), V(a,b)          node / differential voltage
  I(X), P(X)            current through X (nodes[0] -> nodes[1]), power absorbed by X
                        For a voltage source, I(X) is reported as the current DELIVERED out of its
                        + terminal (= -SPICE branch current); numeric value uses the same sign.
  Vth, Rth, Zth, Z      Thevenin voltage = open-circuit V(a)-V(b)
                        Rth = turn off independent sources, inject 1 A test current, Rth = V(a)-V(b)
  Isc, In, Rn           Norton: Isc = Vth / Rth
  H(out,in), gain       transfer function with the input source set to 1 and others to 0
  poles, zeros          roots of the denominator / numerator of H(s)
  fc                    -3 dB frequency: |H(j2πf)| = |H|max / √2   (closed form for first order)
  f0, Qfactor,          second-order denominator a2·s² + a1·s + a0:
  bandwidth                 ω0 = √(a0/a2),  Q = √(a0·a2)/a1,  BW = a1/(2π·a2)  (= ω0/Q in Hz)
  tau                   first-order: τ = Rth·C  or  L/Rth  (Rth seen by the C or L)
  Rin, Rout             resistance seen by the input source / looking into the output node
"""
from __future__ import annotations

import cmath
import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import sympy as sp

from schemas import AnalysisType, Component, Netlist, SolveResult, StepTrace, parse_value
from solvers.base import (BaseSolver, NotSupportedError, SolverError, fmt_complex, fmt_num, fmt_si,
                          pretty, sig_figs)
from solvers.mna import (CONTROLLED, GROUND, INDEPENDENT, LATER_PHASE, LINEAR_TYPES, MNA, join_terms,
                         paren, resolve_opamp, s, tidy)

_Q = re.compile(r"^(\w+)\((\w+)(?:,(\w+))?\)$")

# Supply rails in the question or assumptions: "+/-12 V", "±15 V", "Vcc = 12 V", "VEE = -12 V"
_RAIL_SYM = re.compile(r"(?:±|\+/-|\+/−|\+-)\s*(\d+(?:\.\d+)?)\s*v\b", re.IGNORECASE)
_RAIL_POS = re.compile(r"\bv(?:cc|dd|\+|pos)\s*=\s*\+?\s*(\d+(?:\.\d+)?)\s*v\b", re.IGNORECASE)
_RAIL_NEG = re.compile(r"\bv(?:ee|ss|-|neg)\s*=\s*[-−]?\s*(\d+(?:\.\d+)?)\s*v\b", re.IGNORECASE)
FREQ_KINDS = {"fc", "f0", "poles", "zeros", "Qfactor", "bandwidth"}
MAIN_KINDS = {"V", "I", "P", "Vth", "Isc", "In"}
OUT_NAMES = ("out", "vout", "Vout", "o", "output")


class _Tokens(dict):
    def __missing__(self, key):          # unknown token -> keep its name
        return key


def fill(template: str, mapping: Optional[Dict[str, str]] = None) -> str:
    return template.format_map(_Tokens(mapping or {}))


def node_name(n: str) -> str:
    return "ground" if n == GROUND else n


class DCACSolver(BaseSolver):
    name = "dc_ac"
    method = "Nodal analysis (MNA) solved symbolically with SymPy"

    def supports(self, nl: Netlist) -> Tuple[bool, str]:
        types = {c.type.value for c in nl.components}
        later = sorted(types - LINEAR_TYPES)
        if later:
            return False, "Needs " + ", ".join(LATER_PHASE.get(t, t) for t in later)
        if nl.analysis == AnalysisType.CONCEPTUAL:
            return False, "Conceptual question: no circuit to solve"
        if nl.analysis == AnalysisType.SMALL_SIGNAL and types & {"Q", "M"}:
            return False, "Small-signal transistor analysis comes in Phase 3"
        # (an op-amp "amplifier" labelled small_signal is solved like DC: the op-amp models are linear)
        if nl.analysis == AnalysisType.TRANSIENT:
            other = [q for q in nl.query if q.replace(" ", "") != "tau"]
            if other:
                return False, (f"Time-domain responses ({', '.join(other)}) come in a later phase; "
                               "only tau is supported now")
        return True, ""

    def solve(self, nl: Netlist, conventions: Dict) -> SolveResult:
        ok, why = self.supports(nl)
        if not ok:
            raise NotSupportedError(why)
        return _Job(nl, conventions).run()


# ======================================================================
class _Job:
    """State for one solve() call."""

    def __init__(self, nl: Netlist, conventions: Dict):
        self.nl = nl
        self.conv = conventions or {}
        self.sig = sig_figs(conventions)
        self.mode = "ac" if nl.analysis == AnalysisType.AC else "dc"
        self.freq = parse_value(nl.frequency) if nl.frequency else None
        self.omega = 2 * math.pi * self.freq if self.freq else None
        self.trace = StepTrace()
        self.res = SolveResult(method=DCACSolver.method)
        self.types = {c.type.value for c in nl.components}
        self.opamps = {c.id: resolve_opamp(c, self.conv) for c in nl.components if c.type.value == "OPAMP"}
        # an op-amp with a gain-bandwidth limit makes the circuit frequency-dependent, like a C or L
        self.has_reactive = bool(self.types & {"C", "L"})
        self.has_LC = self.has_reactive or any(i and i["GBW"] for i in self.opamps.values())
        # an op-amp's GBW pole only matters when the question is about frequency (or the analysis is AC);
        # a DC question about a 741 amplifier just uses A0
        self.freq_asked = bool({self._kind(q.replace(" ", ""))[0] for q in nl.query} & (FREQ_KINDS | {"tau"}))
        self.indep = [c for c in nl.components if c.type.value in INDEPENDENT]
        self.node_set = set(nl.nodes())
        self.queries = [q.replace(" ", "") for q in nl.query]
        self._main: Optional[MNA] = None
        self._main_err: Optional[str] = None
        self._transfers: Dict[Tuple[str, str], Tuple[sp.Expr, Component]] = {}

    # ------------------------------------------------------------------
    def run(self) -> SolveResult:
        self._intro_steps()
        for q in self.queries:
            try:
                self._answer(q)
            except NotSupportedError:
                raise
            except SolverError as e:
                self.res.answers[q] = f"Could not compute: {e}"
                self.res.numeric[q] = None
                self.res.verification_notes.append(f"{q}: {e}")
        if self.queries and all(self.res.answers.get(q, "").startswith("Could not compute") for q in self.queries):
            raise SolverError(" | ".join(self.res.verification_notes))
        self._saturation_check()
        self.res.trace = self.trace
        return self.res

    # ------------------------------------------------------------------
    # Op-amp saturation (the linear models cannot clip, so warn instead)
    # ------------------------------------------------------------------
    def _rails(self) -> Optional[Tuple[float, float]]:
        """(+rail, -rail magnitude) from the question / assumptions, or None if no supply is given."""
        text = " ".join([self.nl.question or ""] + list(self.nl.assumptions))
        m = _RAIL_SYM.search(text)
        if m:
            return float(m.group(1)), float(m.group(1))
        pos, neg = _RAIL_POS.search(text), _RAIL_NEG.search(text)
        if pos or neg:
            return (float(pos.group(1)) if pos else math.inf), (float(neg.group(1)) if neg else math.inf)
        return None

    def _saturation_check(self) -> None:
        rails = self._rails()
        if not rails or not self.opamps or self._main is None:
            return
        pos, neg = rails
        m = self._main
        for cid in self.opamps:
            out = m.by_id[cid].nodes[0]
            v = self._at(m.solution[m.V[out]])
            if v.free_symbols:
                continue
            z = complex(sp.N(v))
            if abs(z.imag) > 1e-9 * max(1.0, abs(z)):     # AC phasor: its peak swings both ways
                value, limit, rail = abs(z), min(pos, neg), "±"
            else:                                          # DC (or a real phasor): check the matching rail
                value = z.real
                limit, rail = (pos, "+") if value >= 0 else (neg, "−")
            if abs(value) > limit:
                msg = (f"{cid} would saturate: the linear answer needs V_{out} = {fmt_si(value, 'V', self.sig)}, "
                       f"beyond the {rail}{limit:g} V supply. A real op-amp clips about 1-2 V inside the rail, "
                       f"so this linear result is not actually reached.")
                self.res.verification_notes.append(f"saturation: {msg}")
                self.trace.add(f"⚠️ Saturation check for {cid}",
                               f"|V_{out}| = {fmt_si(abs(value), 'V', self.sig)} > {limit:g} V supply", msg)

    @staticmethod
    def _kind(q: str) -> Tuple[str, Optional[str], Optional[str]]:
        m = _Q.match(q)
        return (m.group(1), m.group(2), m.group(3)) if m else (q, None, None)

    def _answer(self, q: str) -> None:
        kind, a, b = self._kind(q)
        handlers = {
            "V": self._q_V, "I": self._q_I, "P": self._q_P,
            "Vth": self._q_Vth, "Rth": self._q_Z, "Zth": self._q_Z, "Z": self._q_Z, "Rn": self._q_Z,
            "Isc": self._q_Isc, "In": self._q_Isc,
            "H": self._q_H, "gain": self._q_gain,
            "poles": self._q_roots, "zeros": self._q_roots,
            "fc": self._q_fc, "f0": self._q_f0, "Qfactor": self._q_f0, "bandwidth": self._q_bw,
            "tau": self._q_tau, "Rin": self._q_Rin, "Rout": self._q_Rout,
        }
        if kind == "bias":
            raise SolverError("bias() needs the diode/transistor solver (Phase 3)")
        if kind not in handlers:
            raise SolverError(f"Query '{q}' is not handled by the DC/AC solver")
        handlers[kind](q, kind, a, b)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    def _at(self, expr) -> sp.Expr:
        """Substitute s = jω when a frequency is given."""
        e = sp.sympify(expr)
        if self.omega is not None and e.has(s):
            e = e.subs(s, sp.I * sp.Float(self.omega, 15))
        return e

    def _fmt(self, expr, unit: str, rect: bool = False):
        """-> (text, numeric value or |value|, phase in degrees or None, symbolic string or None)."""
        e = self._at(expr)
        if e.has(sp.oo, sp.zoo, sp.nan):
            raise SolverError("The result is infinite or undefined")
        if not e.free_symbols:
            z = complex(sp.N(e, 15))
            if abs(z.imag) <= 1e-9 * max(1.0, abs(z.real)):
                return fmt_si(z.real, unit, self.sig), z.real, None, None
            return fmt_complex(z, unit, self.sig, rect), abs(z), math.degrees(cmath.phase(z)), None
        txt = pretty(e, self.sig)
        return (f"{txt} {unit}".strip() if unit else txt), None, None, str(e)

    def _short(self, expr) -> str:
        """Compact value for substituting into an equation line."""
        e = self._at(expr)
        if e.has(sp.oo, sp.zoo, sp.nan):
            return "∞"
        if not e.free_symbols:
            z = complex(sp.N(e, 15))
            if abs(z.imag) <= 1e-9 * max(1.0, abs(z.real)):
                return paren(fmt_num(z.real, self.sig))
            return paren(fmt_complex(z, "", self.sig))
        return paren(pretty(e, self.sig))

    def _record(self, q: str, expr, unit: str, rect: bool = False) -> Tuple[str, Optional[float]]:
        text, num, ph, sym = self._fmt(expr, unit, rect)
        self.res.answers[q] = text
        self.res.numeric[q] = num
        if ph is not None:
            self.res.numeric[q + ".phase_deg"] = ph
        if sym is not None:
            self.res.symbolic[q] = sym
        return text, num

    def _solved_vals(self, m: MNA) -> Dict[str, str]:
        vals = {c.id: paren(m.display_value(c)) for c in m.components}
        vals.update({k: paren(v) for k, v in m.tokens.items()})       # op-amp A, Rin, Rout
        for n in m.nodes:
            vals[f"V_{n}"] = self._short(m.solution[m.V[n]])
        for cid, sym in m.branch.items():
            vals[f"I_{cid}"] = self._short(m.solution[sym])
        return vals

    def _zero_text(self, removed=()) -> str:
        parts = [f"{c.id} → short" if c.type.value == "V" else f"{c.id} → open"
                 for c in self.indep if c.id not in removed]
        txt = ", ".join(parts) or "no independent sources to turn off"
        if self.types & CONTROLLED:
            txt += "; dependent sources stay active"
        if "OPAMP" in self.types:
            txt += "; op-amps stay active (they are not independent sources)"
        return txt

    # ------------------------------------------------------------------
    # Step builders
    # ------------------------------------------------------------------
    def _intro_steps(self) -> None:
        nodes = [n for n in self.nl.nodes() if n != GROUND]
        self.trace.add("Label the nodes",
                       "Ground (reference) = node 0; unknown node voltages: " + ", ".join(f"V_{n}" for n in nodes),
                       level="full")
        kinds = {self._kind(q)[0] for q in self.queries}
        if self.nl.analysis == AnalysisType.DC and self.has_reactive and kinds & (MAIN_KINDS | {"Rth", "Zth", "Z", "Rn"}):
            self.trace.add("DC steady state: capacitors act as open circuits, inductors as short circuits")
        s_domain = self.has_reactive or (self.has_LC and self.freq_asked)
        if self.mode == "ac" or (s_domain and kinds & (FREQ_KINDS | {"H", "gain"})):
            eq = "Z_R = R,  Z_C = 1/(sC),  Z_L = sL"
            if self.omega:
                eq += f";  s = jω with ω = 2π × {fmt_si(self.freq, 'Hz', self.sig)} = {fmt_num(self.omega, self.sig)} rad/s"
            self.trace.add("Use the s-domain (impedance) model", eq)
        if self.mode == "ac" and any(c.ac not in (None, "") for c in self.indep) \
                and any(c.ac in (None, "") for c in self.indep):
            self.trace.add("Only the AC parts of the sources are used; DC-only sources are set to 0",
                           level="full")

        for cid, info in self.opamps.items():                 # non-ideal op-amp models, stated up front
            if info is None:
                continue
            parts = [f"A0 = {fmt_num(info['A0'], self.sig)}"]
            eq = "Vout = A·(V+ − V−)"
            if info["GBW"]:
                fp = info["GBW"] / info["A0"]
                parts.append(f"GBW = {fmt_si(info['GBW'], 'Hz', self.sig)} → open-loop pole "
                             f"fp = GBW/A0 = {fmt_si(fp, 'Hz', self.sig)} (ωp = {fmt_num(2 * math.pi * fp, self.sig)} rad/s)")
                if self.mode == "ac" or self.freq_asked:
                    eq = "Vout = A(s)·(V+ − V−),  A(s) = A0 / (1 + s/ωp)"
                else:
                    eq = "Vout = A0·(V+ − V−)  (DC: the open-loop pole does not affect the answer)"
            if info["Rin"]:
                parts.append(f"Rin = {fmt_si(info['Rin'], 'Ω', self.sig)} between the inputs")
            if info["Rout"]:
                parts.append(f"Rout = {fmt_si(info['Rout'], 'Ω', self.sig)} in series with the output")
                eq += " + Rout·I"
            self.trace.add(f"Non-ideal op-amp {cid} model: {'; '.join(parts)}", eq,
                           "; ".join(info["notes"]) or None)
            if info["SR"] is not None:
                self.trace.add(f"Slew rate of {cid} ({info['SR']}) limits large, fast signals only; "
                               "it does not enter this linear analysis", level="full")

    def _system_steps(self, m: MNA, level: str = "key") -> None:
        known = m.known_nodes()
        vals = {c.id: paren(m.display_value(c)) for c in m.components}
        vals.update({k: paren(v) for k, v in m.tokens.items()})        # op-amp A, Rin, Rout
        vals.update({f"V_{n}": paren(disp) for n, (_, disp, _) in known.items()})

        for n, (_, disp, cid) in known.items():
            self.trace.add(f"Source {cid} fixes node {n}", None, f"V_{n} = {disp}", level=level)

        # Ideal op-amps first, the way a student writes it: virtual short, then KCL at the inputs
        opamp_out = {}
        for _, tmpl, cid in m.constraints:
            c = m.by_id[cid]
            if c.type.value != "OPAMP":
                continue
            opamp_out[c.nodes[0]] = cid
            names, nums = fill(tmpl), fill(tmpl, vals)
            if m.opamp_info.get(cid) is None:
                action = f"Ideal op-amp {cid}: virtual short V− = V+, and no current flows into its inputs"
            else:
                action = (f"Non-ideal op-amp {cid}: output = gain × (V+ − V−)"
                          + (" + Rout × output current" if m.opamp_info[cid]["Rout"] else "")
                          + ("; Rin carries current between the inputs" if m.opamp_info[cid]["Rin"] else ""))
            self.trace.add(action, names + (f"   →   {nums}" if nums != names else ""), level=level)

        for n in m.nodes:
            tmpl = join_terms([t.text for t in m.kcl[n]])
            names, nums = fill(tmpl), fill(tmpl, vals)
            eq = f"{names} = 0" + (f"   →   {nums} = 0" if nums != names else "")
            if n in known:
                self.trace.add(f"KCL at node {n} (gives the current of {known[n][2]})", eq, level="full")
            elif n in opamp_out:                               # only gives the op-amp's output current
                self.trace.add(f"KCL at node {n} (gives the output current of {opamp_out[n]})", eq, level="full")
            else:
                self.trace.add(f"KCL at node {n}: sum of currents leaving = 0", eq, level=level)

        labels = {"V": "Voltage source {cid} between {a} and {b}", "L": "{cid} is a short circuit in DC",
                  "E": "VCVS {cid}", "H": "CCVS {cid}"}
        for _, tmpl, cid in m.constraints:
            c = m.by_id[cid]
            if c.type.value == "OPAMP":
                continue                                       # already shown above
            if c.type.value == "V" and GROUND in c.nodes[:2]:
                continue                                       # already shown as "fixes node"
            names, nums = fill(tmpl), fill(tmpl, vals)
            action = labels[c.type.value].format(cid=cid, a=node_name(c.nodes[0]), b=node_name(c.nodes[1]))
            self.trace.add(action, names + (f"   →   {nums}" if nums != names else ""), level=level)

    def _solution_step(self, m: MNA, level: str = "key") -> None:
        x = ", ".join(str(u) for u in m.unknowns)
        rows = []
        for i in range(m.A.rows):
            row = "  ".join(pretty(a, self.sig) for a in m.A.row(i))
            rows.append(f"[ {row} | {pretty(m.b[i], self.sig)} ]")
        self.trace.add(f"Matrix form A·x = b with x = [{x}]", "  ".join(rows), level="full")

        node_txt = ", ".join(f"V_{n} = {self._fmt(m.solution[m.V[n]], 'V')[0]}" for n in m.nodes)
        self.trace.add("Solve the equations", None, node_txt, level=level)
        if m.branch:
            cur = ", ".join(f"I_{cid} = {self._fmt(m.solution[sym], 'A')[0]}" for cid, sym in m.branch.items())
            self.trace.add("Branch currents (SPICE direction: into the first node of the element)",
                           None, cur, level="full")

    # ------------------------------------------------------------------
    # Circuit runs
    # ------------------------------------------------------------------
    def main(self) -> MNA:
        if self._main_err:
            raise SolverError(self._main_err)
        if self._main is None:
            try:
                m = MNA(self.nl.components, self.mode, conventions=self.conv)
                self._system_steps(m, level="key")
                m.solve()
            except NotSupportedError:
                raise
            except SolverError as e:
                self._main_err = str(e)
                raise
            self._solution_step(m, level="key")
            self._main = m
        return self._main

    def _zth(self, a: str, b: str, removed=(), mode: Optional[str] = None,
             show: bool = True) -> Tuple[sp.Expr, MNA]:
        """Impedance between a and b: independent sources off, 1 A test current into a (out of b)."""
        if a == b:
            raise SolverError("The two terminals must be different nodes")
        mode = mode or self.mode
        zero = {c.id: 0 for c in self.indep if c.id not in removed}
        test = Component(id="I_test", type="I", nodes=[b, a], value="1")
        m = MNA(list(self.nl.components) + [test], mode, overrides=zero, removed=set(removed),
                conventions=self.conv)
        if show:
            self._system_steps(m, level="full")
        m.solve()
        z = m.evaluate(m.V[a] - m.V[b])
        if show:
            self.trace.add(f"Apply a 1 A test current into {node_name(a)} (out of {node_name(b)}) and solve",
                           None, f"V_{a} - V_{b} = {self._short(z)}" if b != GROUND else f"V_{a} = {self._short(z)}",
                           level="full")
        return z, m

    def _driver(self, inp: str) -> Component:
        comp = next((c for c in self.indep if c.id == inp), None)
        if comp:
            return comp
        if any(c.id == inp for c in self.nl.components):
            raise SolverError(f"{inp} is not an independent source, so it cannot be the input")
        if not self.indep:
            raise SolverError("There is no independent source to act as the input")
        touching = [c for c in self.indep if inp in c.nodes]
        if len(touching) == 1:
            return touching[0]
        if len(self.indep) == 1:
            return self.indep[0]
        raise SolverError(f"Several sources could be the input; ask for H(out, <source>) with one of "
                          f"{[c.id for c in self.indep]}")

    def _default_transfer(self) -> Tuple[str, str]:
        for q in self.queries:
            kind, a, b = self._kind(q)
            if kind == "H" and b:
                return a, b
        out = next((n for n in OUT_NAMES if n in self.node_set), None)
        if out is None:
            vnodes = [a for k, a, b in map(self._kind, self.queries) if k == "V" and a and not b]
            out = vnodes[0] if vnodes else None
        if out is None:
            raise SolverError("Cannot tell which node is the output; ask for H(out_node, input)")
        if len(self.indep) != 1:
            raise SolverError("Cannot tell which source is the input; ask for H(out, <source>)")
        return out, self.indep[0].id

    def _transfer(self, out: str, inp: str) -> Tuple[sp.Expr, Component]:
        key = (out, inp)
        if key in self._transfers:
            return self._transfers[key]
        drv = self._driver(inp)
        mode = "ac" if (self.mode == "ac" or self.has_reactive or (self.has_LC and self.freq_asked)) else "dc"
        ov = {c.id: (1 if c.id == drv.id else 0) for c in self.indep}
        m = MNA(self.nl.components, mode, overrides=ov, conventions=self.conv)

        others = [c.id for c in self.indep if c.id != drv.id]
        self.trace.add("Transfer function: apply a unit input"
                       + (" and solve in the s-domain" if mode == "ac" else ""),
                       f"{drv.id} = 1" + (f",  {', '.join(others)} = 0 (turned off)" if others else ""))
        self._system_steps(m, level="key" if self._main is None else "full")
        m.solve()
        self._solution_step(m, level="full")

        vout = m.evaluate(m.V[out])
        if inp in self.node_set:
            vin = m.evaluate(m.V[inp])
            if vin == 0:
                raise SolverError(f"V_{inp} is 0 with {drv.id} = 1, so V_{out}/V_{inp} is undefined")
            H = tidy(vout / vin)
            ratio = f"V_{out} / V_{inp}"
        else:
            H = vout
            ratio = f"V_{out} / {drv.id}"
        label = "H(s)" if H.has(s) else "H"
        self.trace.add("Transfer function", f"{label} = {ratio}", f"{label} = {pretty(H, self.sig)}")
        # recorded even when H itself was not asked, so the verifier can check f0/Q/BW/poles/fc against it
        self.res.symbolic.setdefault(f"H({out},{inp})", str(H))
        self._transfers[key] = (H, drv)
        return H, drv

    # ------------------------------------------------------------------
    # Query handlers
    # ------------------------------------------------------------------
    def _q_V(self, q, kind, a, b):
        m = self.main()
        if b is None:
            text, _ = self._record(q, m.evaluate(m.V[a]), "V")
            # already shown by "Solve the equations", so only in the complete view
            self.trace.add(f"Voltage at node {node_name(a)}", None, f"V_{a} = {text}", level="full")
            return
        expr = m.evaluate(m.V[a] - m.V[b])
        text, _ = self._record(q, expr, "V")
        vals = self._solved_vals(m)
        va = "0" if a == GROUND else vals[f"V_{a}"]
        vb = "0" if b == GROUND else vals[f"V_{b}"]
        self.trace.add(f"Voltage between {node_name(a)} and {node_name(b)}",
                       f"V_{a}{b} = V_{a} - V_{b} = {va} - {vb}", f"V_{a}{b} = {text}")

    def _q_I(self, q, kind, cid, _):
        m = self.main()
        c = m.by_id[cid]
        expr, tmpl = m.current(c)
        val = m.evaluate(expr)
        n0, n1 = c.nodes[0], c.nodes[1]
        if c.type.value == "OPAMP":
            text, _ = self._record(q, tidy(-val), "A")
            self.res.answers[q] = f"{text} (delivered by the op-amp output into node {node_name(n0)})"
            self.trace.add(f"Output current of op-amp {cid}",
                           f"from KCL at node {node_name(n0)}: I_{cid} = {self._short(val)};  delivered = -I_{cid}",
                           f"I({cid}) = {text} out of the op-amp output")
            return
        if c.type.value in ("V", "E", "H"):
            text, _ = self._record(q, tidy(-val), "A")
            self.res.answers[q] = f"{text} (out of the + terminal, node {node_name(n0)})"
            self.trace.add(f"Current supplied by {cid}",
                           f"I_{cid} (SPICE direction, + → − through the source) = {self._short(val)};  "
                           f"delivered = -I_{cid}",
                           f"I({cid}) = {text} out of the + terminal")
            return
        text, _ = self._record(q, val, "A")
        self.res.answers[q] = f"{text} (from {node_name(n0)} to {node_name(n1)})"
        names, nums = fill(tmpl), fill(tmpl, self._solved_vals(m))
        eq = f"I({cid}) = {names}" + (f" = {nums}" if nums != names else "")
        self.trace.add(f"Current through {cid} (from {node_name(n0)} to {node_name(n1)})", eq, f"I({cid}) = {text}")

    def _q_P(self, q, kind, cid, _):
        m = self.main()
        c = m.by_id[cid]
        if c.type.value == "OPAMP":
            raise SolverError("An ideal op-amp's power is not defined (its supply pins are not modelled)")
        v = m.evaluate(m.voltage(c))
        i = m.evaluate(m.current(c)[0])
        if self.mode == "dc":
            p = tidy(v * i)
            eq = f"P = V·I = {self._short(v)} × {self._short(i)}"
        else:
            if self.omega is None and (v.has(s) or i.has(s)):
                raise SolverError("Average AC power needs a frequency")
            ve, ie = self._at(v), self._at(i)
            if ve.free_symbols or ie.free_symbols:
                raise SolverError("Average AC power needs numeric values")
            V, I = complex(sp.N(ve)), complex(sp.N(ie))
            p = sp.Float(0.5 * (V * I.conjugate()).real)
            eq = f"P = ½·Re(V·I*) with peak phasors V = {self._short(v)}, I = {self._short(i)}"
        text, num = self._record(q, p, "W")
        if num is not None:
            word = "delivered" if num < 0 else "absorbed"
            text = f"{fmt_si(abs(num), 'W', self.sig)} {word}"
        else:
            text = f"{text} absorbed (negative = delivered)"
        self.res.answers[q] = text
        self.trace.add(f"Power in {cid} (passive sign convention: positive = absorbed)", eq, f"P({cid}) = {text}")

    def _q_Vth(self, q, kind, a, b):
        m = self.main()
        expr = m.evaluate(m.V[a] - m.V[b])
        text, _ = self._record(q, expr, "V")
        vals = self._solved_vals(m)
        va = "0" if a == GROUND else vals[f"V_{a}"]
        vb = "0" if b == GROUND else vals[f"V_{b}"]
        self.trace.add(f"Thevenin voltage = open-circuit voltage between {node_name(a)} and {node_name(b)}",
                       f"V_th = V_{a} - V_{b} = {va} - {vb}", f"V_th = {text}")

    def _q_Z(self, q, kind, a, b):
        label = {"Rth": "R_th", "Zth": "Z_th", "Z": "Z", "Rn": "R_N"}[kind]
        self.trace.add(f"Find {label} between {node_name(a)} and {node_name(b)}: turn off independent sources",
                       self._zero_text())
        z, _ = self._zth(a, b)
        text, _ = self._record(q, z, "Ω", rect=True)
        self.trace.add(f"{label} = (V_{a} - V_{b}) / 1 A", None, f"{label} = {text}")

    def _q_Isc(self, q, kind, a, b):
        m = self.main()
        vth = m.evaluate(m.V[a] - m.V[b])
        self.trace.add(f"Norton: find R_th between {node_name(a)} and {node_name(b)}", self._zero_text(),
                       level="full")
        z, _ = self._zth(a, b)
        if z == 0:
            raise SolverError("R_th = 0, so the short-circuit current is unbounded")
        isc = tidy(vth / z)
        text, _ = self._record(q, isc, "A")
        self.trace.add("Norton (short-circuit) current", f"I_N = V_th / R_th = {self._short(vth)} / {self._short(z)}",
                       f"I_N = {text}")

    def _report_transfer(self, q: str, H: sp.Expr, out: str, inp: str, drv: Component) -> None:
        # H itself is already shown by the "Transfer function" step in _transfer();
        # here we only add what is new (DC gain, value at the given frequency).
        if not H.has(s):
            self._record(q, H, "")
            return
        self.res.symbolic[q] = str(H)
        text = f"H(s) = {pretty(H, self.sig)}"
        extras: List[Tuple[str, str, str]] = []          # (action, equation, result)
        num: Optional[float] = None
        _, den = sp.fraction(sp.cancel(sp.together(H)))
        if den.subs(s, 0) != 0:
            h0 = tidy(H.subs(s, 0))
            if not h0.free_symbols:
                dc = float(sp.re(sp.N(h0)))
                self.res.numeric[q + ".dc_gain"] = dc
                text += f";  DC gain H(0) = {fmt_num(dc, self.sig)}"
                extras.append(("DC gain: put s = 0", "H(0)", f"H(0) = {fmt_num(dc, self.sig)}"))
            else:
                text += f";  H(0) = {pretty(h0, self.sig)}"
                extras.append(("DC gain: put s = 0", "H(0)", f"H(0) = {pretty(h0, self.sig)}"))
        hj = self._at(H)
        if self.omega is not None and not hj.free_symbols:
            z = complex(sp.N(hj))
            mag, ph = abs(z), math.degrees(cmath.phase(z))
            db = f" ({fmt_num(20 * math.log10(mag), self.sig)} dB)" if mag > 0 else ""
            at = (f"|H| = {fmt_num(mag, self.sig)}{db}, phase = {fmt_num(ph, self.sig)}°")
            text += f";  at f = {fmt_si(self.freq, 'Hz', self.sig)}: {at}"
            extras.append((f"Evaluate at f = {fmt_si(self.freq, 'Hz', self.sig)}",
                           f"s = jω = j{fmt_num(self.omega, self.sig)}", at))
            num = mag
            self.res.numeric[q + ".phase_deg"] = ph
        self.res.answers[q] = text
        self.res.numeric[q] = num
        for action, eq, result in extras:
            self.trace.add(action, eq, result)

    def _q_H(self, q, kind, out, inp):
        H, drv = self._transfer(out, inp)
        self._report_transfer(q, H, out, inp, drv)

    def _q_gain(self, q, kind, a, b):
        out, inp = self._default_transfer()
        H, drv = self._transfer(out, inp)
        self._report_transfer(q, H, out, inp, drv)

    # --- poles / zeros / cutoff / resonance ---------------------------
    def _polys(self, H: sp.Expr) -> Tuple[sp.Poly, sp.Poly]:
        if not H.has(s):
            raise SolverError("The circuit has no capacitors or inductors, so H has no poles, zeros or cutoff")
        num, den = sp.fraction(sp.cancel(sp.together(H)))
        return sp.Poly(num, s), sp.Poly(den, s)

    @staticmethod
    def _coeff_symbols(*polys: sp.Poly) -> set:
        out = set()
        for p in polys:
            for cf in p.all_coeffs():
                out |= sp.sympify(cf).free_symbols
        return out

    def _roots(self, poly: sp.Poly) -> List:
        if poly.degree() <= 0:
            return []
        if self._coeff_symbols(poly):
            r = sp.roots(poly)
            if sum(r.values()) != poly.degree():
                raise SolverError("The roots have no closed form with symbolic values")
            out = []
            for root, k in r.items():
                out += [root] * k
            return out
        return [complex(z) for z in poly.nroots(n=15)]

    def _root_text(self, r) -> str:
        if isinstance(r, complex):
            if abs(r.imag) <= 1e-9 * max(1.0, abs(r.real)):
                return fmt_num(r.real, self.sig)
            sign = "+" if r.imag >= 0 else "-"
            return f"{fmt_num(r.real, self.sig)} {sign} j{fmt_num(abs(r.imag), self.sig)}"
        return pretty(r, self.sig)

    def _q_roots(self, q, kind, a, b):
        out, inp = self._default_transfer()
        H, _ = self._transfer(out, inp)
        num, den = self._polys(H)
        poly = den if kind == "poles" else num
        roots = self._roots(poly)
        word = "denominator" if kind == "poles" else "numerator"
        if not roots:
            self.res.answers[q] = f"no finite {kind}"
            self.res.numeric[q] = None
        else:
            self.res.answers[q] = "s = " + ", ".join(self._root_text(r) for r in roots) + " rad/s"
            self.res.numeric[q] = float(len(roots))
            for i, r in enumerate(roots, 1):
                if isinstance(r, complex):
                    self.res.numeric[f"{q}[{i}].re"] = r.real
                    self.res.numeric[f"{q}[{i}].im"] = r.imag
            self.res.symbolic[q] = str(roots)
        self.trace.add(f"{kind.capitalize()}: roots of the {word} of H(s)",
                       f"{pretty(poly.as_expr(), self.sig)} = 0", self.res.answers[q])

    def _cutoffs(self, H: sp.Expr) -> Tuple[List[float], float]:
        fn = sp.lambdify(s, H, modules="numpy")
        f = np.logspace(-3, 12, 4001)
        with np.errstate(all="ignore"):
            mag = np.abs(np.broadcast_to(fn(1j * 2 * np.pi * f), f.shape)).astype(float)
        finite = np.isfinite(mag)
        if not finite.any():
            raise SolverError("|H(jω)| could not be evaluated")
        hmax = float(np.max(mag[finite]))
        if hmax == 0:
            raise SolverError("|H| is zero at every frequency")
        target = hmax / math.sqrt(2)
        d = mag - target

        def g(lf: float) -> float:
            return abs(complex(fn(1j * 2 * math.pi * 10 ** lf))) - target

        cuts: List[float] = []
        for i in range(len(f) - 1):
            if not (finite[i] and finite[i + 1]) or d[i] == 0 or np.sign(d[i]) == np.sign(d[i + 1]):
                continue
            lo, hi = math.log10(f[i]), math.log10(f[i + 1])
            glo = g(lo)
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                gm = g(mid)
                if (gm > 0) == (glo > 0):
                    lo, glo = mid, gm
                else:
                    hi = mid
            cuts.append(10 ** (0.5 * (lo + hi)))
        return cuts, hmax

    def _q_fc(self, q, kind, a, b):
        out, inp = self._default_transfer()
        H, _ = self._transfer(out, inp)
        num, den = self._polys(H)

        if self._coeff_symbols(num, den):
            if den.degree() != 1:
                raise SolverError("fc of a higher-order circuit needs numeric component values")
            a1, a0 = den.all_coeffs()
            fc = tidy(a0 / (2 * sp.pi * a1))
            self.trace.add("First-order circuit: one pole at s = -a0/a1", f"fc = |p|/(2π) = a0/(2π·a1)",
                           f"fc = {pretty(fc, self.sig)}")
            self._record(q, fc, "Hz")
            return

        if den.degree() == 1:
            a1, a0 = [float(sp.N(x)) for x in den.all_coeffs()]
            p = -a0 / a1
            self.trace.add("First-order circuit: single pole",
                           f"p = {fmt_num(p, self.sig)} rad/s  →  fc = |p|/(2π) = {fmt_num(abs(p), self.sig)}/(2π)",
                           f"fc = {fmt_si(abs(p) / (2 * math.pi), 'Hz', self.sig)}")
        cuts, hmax = self._cutoffs(H)
        if not cuts:
            raise SolverError("|H| never falls 3 dB below its peak between 1 mHz and 1 THz")
        self.trace.add("−3 dB condition: |H(j2πf)| = |H|max / √2",
                       f"|H|max = {fmt_num(hmax, self.sig)}  →  solve |H(j2πf)| = "
                       f"{fmt_num(hmax, self.sig)}/√2 = {fmt_num(hmax / math.sqrt(2), self.sig)}"
                       + ("" if den.degree() == 1 else "  (solved numerically)"),
                       level="key" if den.degree() != 1 else "full")
        self.res.numeric[q] = cuts[0]
        if len(cuts) == 1:
            self.res.answers[q] = fmt_si(cuts[0], "Hz", self.sig)
        else:
            self.res.answers[q] = ", ".join(fmt_si(x, "Hz", self.sig) for x in cuts)
            for i, x in enumerate(cuts[1:], 2):
                self.res.numeric[f"{q}_{i}"] = x
        # a first-order circuit already showed fc in the "single pole" step
        self.trace.add("Cutoff frequency", None, f"fc = {self.res.answers[q]}",
                       level="full" if den.degree() == 1 else "key")

    def _second_order(self) -> Optional[Dict[str, sp.Expr]]:
        """ω0, Q, BW from a second-order H(s); None if no usable transfer function. Cached."""
        if hasattr(self, "_so"):
            if isinstance(self._so, SolverError):
                raise self._so
            return self._so
        try:
            out, inp = self._default_transfer()
        except SolverError:
            self._so = None
            return None
        H, _ = self._transfer(out, inp)
        _, den = self._polys(H)
        if den.degree() != 2:
            self._so = SolverError(f"Needs a second-order circuit (the denominator of H(s) has degree "
                                   f"{den.degree()})")
            raise self._so
        a2, a1, a0 = den.all_coeffs()
        w0 = tidy(sp.sqrt(a0 / a2))
        Q = sp.oo if a1 == 0 else tidy(sp.sqrt(a0 * a2) / a1)
        BW = tidy(a1 / (2 * sp.pi * a2))

        # numbers (or symbols) to substitute into each formula
        c2, c1, c0 = (self._short(x) for x in (a2, a1, a0))
        w0_txt = self._short(w0)
        self.trace.add("Read the coefficients of the second-order denominator a2·s² + a1·s + a0",
                       f"{pretty(den.as_expr(), self.sig)} = 0",
                       f"a2 = {c2},  a1 = {c1},  a0 = {c0}")
        self._so = {
            "f0": tidy(w0 / (2 * sp.pi)), "Q": Q, "BW": BW,
            "f0_eq": f"ω0 = √(a0/a2) = √({c0}/{c2}) = {w0_txt} rad/s  →  f0 = ω0/(2π) = {w0_txt}/(2π)",
            "Q_eq": (f"Q = √(a0·a2)/a1 = √({c0}·{c2})/{c1}" if Q != sp.oo else "a1 = 0, so Q = √(a0·a2)/a1 → ∞"),
            "BW_eq": f"BW = a1/(2π·a2) = {c1}/(2π·{c2})   (same as f0/Q)",
        }
        return self._so

    def _lc_resonance(self) -> sp.Expr:
        Ls = [c for c in self.nl.components if c.type.value == "L"]
        Cs = [c for c in self.nl.components if c.type.value == "C"]
        if len(Ls) != 1 or len(Cs) != 1:
            raise SolverError("Cannot find f0: no clear output node, and the circuit is not a single L with a single C")
        m = MNA(self.nl.components, "ac", conventions=self.conv)
        Lv, Cv = m.value(Ls[0]), m.value(Cs[0])
        f0 = tidy(1 / (2 * sp.pi * sp.sqrt(Lv * Cv)))
        self.trace.add("Resonance: X_L = X_C  →  ω0 = 1/√(LC)",
                       f"f0 = 1/(2π·√({Ls[0].id}·{Cs[0].id})) = 1/(2π·√({Ls[0].value}·{Cs[0].value}))")
        return f0

    def _q_f0(self, q, kind, a, b):
        so = self._second_order()
        if kind == "f0":
            if so:
                text, _ = self._record(q, so["f0"], "Hz")
                self.trace.add("Resonant frequency", so["f0_eq"], f"f0 = {text}")
            else:
                text, _ = self._record(q, self._lc_resonance(), "Hz")
                self.trace.add("Resonant frequency", None, f"f0 = {text}")
            return
        if so is None:
            raise SolverError("Q needs a transfer function; ask for H(out, in)")
        if so["Q"] == sp.oo:
            self.res.answers[q] = "∞ (lossless LC: a1 = 0)"
            self.res.numeric[q] = None
        else:
            self._record(q, so["Q"], "")
        self.trace.add("Quality factor", so["Q_eq"], f"Q = {self.res.answers[q]}")

    @staticmethod
    def _is_lowpass(H: sp.Expr, hmax: float) -> bool:
        """|H| at DC is (within 0.1 %) the maximum -> the response starts in its pass band."""
        h0 = H.subs(s, 0)
        if h0.has(sp.oo, sp.zoo, sp.nan) or h0.free_symbols:
            return False
        return abs(complex(sp.N(h0))) >= hmax * (1 - 1e-3)

    def _q_bw(self, q, kind, a, b):
        try:
            so = self._second_order()
        except SolverError:
            so = None
        if so is not None:
            text, _ = self._record(q, so["BW"], "Hz")
            self.trace.add("Bandwidth", so["BW_eq"], f"BW = {text}")
            return
        out, inp = self._default_transfer()
        H, _ = self._transfer(out, inp)
        cuts, hmax = self._cutoffs(H)
        if len(cuts) == 1 and self._is_lowpass(H, hmax):
            # low-pass (e.g. an op-amp amplifier limited by GBW): the band is 0 → fc, so BW = fc
            fc = cuts[0]
            self.trace.add("Low-pass response: the pass band runs from DC to the −3 dB point",
                           f"|H|max = |H(0)| = {fmt_num(hmax, self.sig)}  →  solve |H(j2πf)| = "
                           f"{fmt_num(hmax, self.sig)}/√2 = {fmt_num(hmax / math.sqrt(2), self.sig)}")
            text, _ = self._record(q, sp.Float(fc), "Hz")
            self.trace.add("Bandwidth = fc − 0", None, f"BW = {text}")
            return
        if len(cuts) != 2:
            raise SolverError("Bandwidth needs two −3 dB points (a band-pass response), or one for a low-pass")
        text, _ = self._record(q, sp.Float(cuts[1] - cuts[0]), "Hz")
        self.trace.add("Bandwidth = f_H − f_L",
                       f"{fmt_si(cuts[1], 'Hz', self.sig)} − {fmt_si(cuts[0], 'Hz', self.sig)}", f"BW = {text}")

    # --- time constant ------------------------------------------------
    def _q_tau(self, q, kind, a, b):
        storage = [c for c in self.nl.components if c.type.value in ("C", "L")]
        if len(storage) != 1:
            raise SolverError(f"τ is defined here for first-order circuits (one C or L); this one has {len(storage)}")
        e = storage[0]
        n0, n1 = e.nodes
        self.trace.add(f"First-order circuit: find the resistance seen by {e.id}",
                       f"turn off sources ({self._zero_text()}), remove {e.id}, look into nodes "
                       f"{node_name(n0)}–{node_name(n1)}")
        z, m = self._zth(n0, n1, removed={e.id}, mode="dc")
        if z == 0:
            raise SolverError(f"{e.id} sees zero resistance, so τ is undefined")
        self.trace.add(f"Resistance seen by {e.id}", None, f"R_th = {self._fmt(z, 'Ω')[0]}")
        val = m.value(e)
        if e.type.value == "C":
            tau, eq = tidy(z * val), f"τ = R_th·C = {self._short(z)} × {e.value}"
        else:
            tau, eq = tidy(val / z), f"τ = L/R_th = {e.value} / {self._short(z)}"
        text, _ = self._record(q, tau, "s")
        self.trace.add("Time constant", eq, f"τ = {text}")

    # --- input / output resistance ------------------------------------
    def _input_source(self) -> Component:
        for q in self.queries:
            kind, a, b = self._kind(q)
            if kind == "H" and b:
                return self._driver(b)
        if len(self.indep) == 1:
            return self.indep[0]
        raise SolverError("Cannot tell which source is the input")

    def _q_Rin(self, q, kind, a, b):
        drv = self._input_source()
        n0, n1 = drv.nodes
        self.trace.add(f"Input resistance: resistance seen by the source {drv.id}",
                       f"remove {drv.id}, turn off other sources ({self._zero_text(removed={drv.id})}), "
                       f"look into nodes {node_name(n0)}–{node_name(n1)}")
        z, _ = self._zth(n0, n1, removed={drv.id})
        text, _ = self._record(q, z, "Ω", rect=True)
        self.trace.add("Input resistance", None, f"R_in = {text}")

    def _q_Rout(self, q, kind, a, b):
        out, _ = self._default_transfer()
        if out == GROUND:
            raise SolverError("The output node cannot be ground")
        self.trace.add(f"Output resistance: look into node {out} with sources off", self._zero_text())
        z, _ = self._zth(out, GROUND)
        text, _ = self._record(q, z, "Ω", rect=True)
        self.trace.add("Output resistance", None, f"R_out = {text}")