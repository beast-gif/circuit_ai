"""
Diode circuits in DC (Phase 3, step 1).

Method (the way a student does it, but exhaustively):
  1. Every diode gets a model (below). Piecewise-linear models make each diode a simple linear element once
     its STATE is known: ON -> a voltage source, OFF -> an open circuit.
  2. Assume a state for every diode, replace the diodes, and solve the now-linear circuit EXACTLY with the
     MNA engine (the same one that solves every DC circuit).
  3. Check the assumption:  ON needs I_D >= 0 (forward current),  OFF needs V_AK < Von,
     Zener breakdown needs I_Z >= 0 (reverse current).
  4. The combination that passes every check is the answer (for these monotone models there is exactly one).

Models (picked from the netlist's "model" / "params"; defaults from conventions.yaml):
  ideal        ON: V_AK = 0 (short)                OFF: open
  cvd          ON: V_AK = Von (0.7 V Si, 0.3 V Ge)  OFF: open             <- default ("silicon diode")
  pwl          ON: V_AK = Von + rd·I_D               OFF: open
  zener        forward ON: V_AK = Von;  breakdown: V_KA = VZ + rz·I_Z;  otherwise OFF
  exp          I_D = IS·(e^(V_AK/(n·VT)) − 1)  (Shockley), solved by Newton-Raphson iteration
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

MAX_COMBOS = 729                        # 3^6: six Zeners, or nine ordinary diodes
ALLOWED_KINDS = {"V", "I", "P", "bias"}
_Q = re.compile(r"^(\w+)\((\w+)(?:,(\w+))?\)$")

_ALIASES = {
    "Von": ("von", "vd", "vf", "vd_on", "v_on", "vgamma", "vγ", "cut_in", "cutin", "v_cut_in", "vfwd",
            "forward_voltage", "vdon", "vd0", "v_d"),
    "rd": ("rd", "rf", "r_on", "ron", "rs", "r_d", "forward_resistance"),
    "IS": ("is", "i_s", "isat", "i_sat", "saturation_current", "i0"),
    "n": ("n", "eta", "ideality", "ideality_factor"),
    "VZ": ("vz", "bv", "vbr", "v_z", "zener_voltage", "breakdown_voltage"),
    "rz": ("rz", "zz", "r_z", "zener_resistance"),
}
_IDEAL = {"ideal"}
_EXP = {"exp", "exponential", "shockley", "diode_equation", "real", "spice"}
_PWL = {"pwl", "piecewise", "piecewise_linear", "piecewise-linear"}
_GE = {"germanium", "ge"}
_LED = {"led"}


def _num(raw) -> float:
    return float(raw) if isinstance(raw, (int, float)) else parse_value(str(raw).strip())


@dataclass
class Diode:
    id: str
    a: str                      # anode
    k: str                      # cathode
    kind: str                   # ideal | cvd | pwl | zener | exp
    Von: float = 0.7
    rd: float = 0.0
    IS: float = 1e-14
    n: float = 1.0
    VZ: Optional[float] = None
    rz: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def states(self) -> List[str]:
        if self.kind == "zener":
            return ["bz", "off", "on"]
        if self.kind == "exp":
            return ["exp"]
        return ["on", "off"]

    def as_dict(self) -> Dict[str, object]:
        return {"kind": self.kind, "Von": self.Von, "rd": self.rd, "IS": self.IS, "n": self.n,
                "VZ": self.VZ, "rz": self.rz}

    def label(self, VT: float) -> Tuple[str, str]:
        """(model description, replacement rule) for the steps."""
        v = fmt_si(self.Von, "V")
        if self.kind == "ideal":
            return f"{self.id}: ideal diode", "ON → short circuit (V_AK = 0);  OFF → open circuit"
        if self.kind == "cvd":
            return (f"{self.id}: constant-voltage-drop model, Von = {v}",
                    f"ON → {v} source (anode +);  OFF → open circuit")
        if self.kind == "pwl":
            return (f"{self.id}: piecewise-linear model, Von = {v}, rd = {fmt_si(self.rd, 'Ω')}",
                    f"ON → {v} source in series with {fmt_si(self.rd, 'Ω')};  OFF → open circuit")
        if self.kind == "zener":
            vz = fmt_si(self.VZ, "V")
            rz = f" in series with rz = {fmt_si(self.rz, 'Ω')}" if self.rz else ""
            return (f"{self.id}: Zener diode, VZ = {vz}",
                    f"breakdown → {vz} source{rz} (cathode +);  forward ON → {v};  otherwise OFF (open)")
        return (f"{self.id}: exponential (Shockley) model, IS = {fmt_num(self.IS)} A, n = {fmt_num(self.n)}, "
                f"VT = {fmt_si(VT, 'V')}",
                "I_D = IS·(e^(V_AK/(n·VT)) − 1)  (nonlinear: solved by Newton-Raphson iteration)")


def resolve_diode(c: Component, conventions: Optional[dict] = None) -> Diode:
    p = {str(k).strip().lower().replace(" ", "_").replace("-", "_"): v for k, v in (c.params or {}).items()}
    raw = {key: next((p[a] for a in aliases if p.get(a) not in (None, "")), None) for key, aliases in _ALIASES.items()}
    model = (c.model or "").strip().lower().replace(" ", "_")
    sc = (conventions or {}).get("semiconductor") or {}
    von_si = float(sc.get("VD_on", 0.7))
    d = Diode(id=c.id, a=c.nodes[0], k=c.nodes[1], kind="cvd", Von=von_si)

    if raw["Von"] is not None:
        d.Von = _num(raw["Von"])
    elif model in _GE:
        d.Von = 0.3
    elif model in _LED:
        d.Von = 2.0
        d.notes.append("LED forward voltage 2 V assumed (typical red LED, not given)")
    d.rd = _num(raw["rd"]) if raw["rd"] is not None else 0.0

    if "zener" in model or raw["VZ"] is not None:
        if raw["VZ"] is None:
            raise SolverError(f"{c.id} is a Zener diode but no Zener voltage VZ is given")
        d.kind, d.VZ = "zener", abs(_num(raw["VZ"]))
        d.rz = _num(raw["rz"]) if raw["rz"] is not None else 0.0
    elif model in _EXP or raw["IS"] is not None:
        d.kind = "exp"
        d.IS = _num(raw["IS"]) if raw["IS"] is not None else 1e-14
        d.n = _num(raw["n"]) if raw["n"] is not None else 1.0
        if raw["IS"] is None:
            d.notes.append("IS = 1e-14 A assumed (typical small-signal silicon diode)")
    elif model in _IDEAL:
        d.kind, d.Von, d.rd = "ideal", 0.0, 0.0
    elif model in _PWL or d.rd:
        d.kind = "pwl"
    return d


# ======================================================================
class DiodeSolver(BaseSolver):
    name = "diode"
    method = "Assume each diode's state, solve the linear circuit exactly (MNA), check the assumption"

    def supports(self, nl: Netlist) -> Tuple[bool, str]:
        types = {c.type.value for c in nl.components}
        if "D" not in types:
            return False, ""
        if types & {"Q", "M"}:
            return False, ""                      # the transistor solver handles (or explains) these
        if nl.analysis == AnalysisType.CONCEPTUAL:
            return False, "Conceptual question: no circuit to solve"
        if nl.analysis == AnalysisType.TRANSIENT:
            return True, ""      # no C/L: solvers/waveform.py;  with C/L (ripple, clampers): solvers/pwl_transient.py
        if nl.analysis != AnalysisType.DC:
            return False, ("Small-signal (AC) analysis of diode circuits comes later; DC circuits and "
                           "rectifiers / clippers (analysis 'transient') are supported")
        return True, ""

    def solve(self, nl: Netlist, conventions: Dict) -> SolveResult:
        ok, why = self.supports(nl)
        if not ok:
            raise NotSupportedError(why)
        if nl.analysis == AnalysisType.TRANSIENT:
            queries = [q.replace(" ", "") for q in nl.query]
            if any(c.type.value in ("C", "L") for c in nl.components):
                from solvers.pwl_transient import PWLTransient
                return PWLTransient(nl, conventions, _DiodeJob(nl, conventions)).run(queries)
            from solvers.waveform import WaveJob
            return WaveJob(nl, conventions, _DiodeJob(nl, conventions)).run(queries)
        return _DiodeJob(nl, conventions).run()


class _DiodeJob:
    def __init__(self, nl: Netlist, conventions: Dict):
        self.nl = nl
        self.conv = conventions or {}
        self.sig = sig_figs(conventions)
        self.VT = float(((self.conv.get("semiconductor") or {}).get("VT", 0.025)))
        self.diodes = [resolve_diode(c, self.conv) for c in nl.components if c.type.value == "D"]
        self.by_id = {d.id: d for d in self.diodes}
        self.others = [c for c in nl.components if c.type.value != "D"]
        for c in nl.components:
            for v in (c.value, c.ac):
                if v not in (None, "") and is_symbolic(v):
                    raise NotSupportedError("Diode states can only be decided with numeric values "
                                            f"({c.id} = {v} is symbolic)")
        self.trace = StepTrace()
        self.res = SolveResult(method=DiodeSolver.method)

    # ------------------------------------------------------------------
    # Linear stand-ins for each diode state
    # ------------------------------------------------------------------
    @staticmethod
    def _v(x: float) -> str:
        return repr(float(x))

    def _parts(self, d: Diode, state: str, lin: Optional[Tuple[float, float]] = None,
               off_as_r: bool = False) -> List[Component]:
        if state == "off":            # 0 A source keeps both nodes in the circuit (an open circuit)
            if off_as_r:              # ... or 1 TΩ (its leakage) when an all-OFF bridge leaves nodes floating
                return [Component(id=d.id, type="R", nodes=[d.a, d.k], value="1e12")]
            return [Component(id=d.id, type="I", nodes=[d.a, d.k], value="0")]
        if state in ("on", "bz"):
            p, n_ = (d.a, d.k) if state == "on" else (d.k, d.a)
            v, r = (d.Von, d.rd) if state == "on" else (d.VZ, d.rz)
            if r:
                x = f"{d.id}_x"
                return [Component(id=d.id, type="V", nodes=[p, x], value=self._v(v)),
                        Component(id=f"{d.id}_r", type="R", nodes=[x, n_], value=self._v(r))]
            return [Component(id=d.id, type="V", nodes=[p, n_], value=self._v(v))]
        G, Ieq = lin                  # "exp": tangent line at the present V_D (companion model)
        return [Component(id=f"{d.id}_g", type="R", nodes=[d.a, d.k], value=self._v(1 / G)),
                Component(id=d.id, type="I", nodes=[d.a, d.k], value=self._v(Ieq))]

    def _components(self, states: Dict[str, str], lin: Optional[Dict[str, Tuple[float, float]]] = None,
                    off_as_r: bool = False):
        comps = list(self.others)
        for d in self.diodes:
            comps += self._parts(d, states[d.id], (lin or {}).get(d.id), off_as_r)
        return comps

    def _mna(self, states, lin=None) -> MNA:
        m = MNA(self._components(states, lin), "dc", conventions=self.conv)
        m.solve()
        return m

    @staticmethod
    def _f(m: MNA, expr) -> float:
        e = m.evaluate(expr)
        if e.free_symbols:
            raise NotSupportedError("Diode states can only be decided with numeric values")
        return float(sp.N(e))

    def _vak(self, m: MNA, d: Diode) -> float:
        return self._f(m, m.V[d.a] - m.V[d.k])

    def _exp_current(self, d: Diode, v: float) -> float:
        return d.IS * math.expm1(v / (d.n * self.VT))

    def _current(self, m: MNA, d: Diode, state: str, vexp: Optional[float] = None) -> float:
        """Forward current (anode -> cathode); for breakdown, the reverse current I_Z (cathode -> anode)."""
        if state == "off":
            return 0.0
        if state == "exp":
            return self._exp_current(d, vexp)
        return self._f(m, m.branch[d.id])

    # ------------------------------------------------------------------
    def _newton(self, states: Dict[str, str]) -> Tuple[MNA, Dict[str, float], List[Dict[str, float]]]:
        """Solve the exponential diodes (given the states of the piecewise-linear ones)."""
        exps = [d for d in self.diodes if d.kind == "exp"]
        V = {d.id: 0.6 for d in exps}
        hist: List[Dict[str, float]] = []
        for _ in range(150):
            lin = {}
            for d in exps:
                nvt = d.n * self.VT
                e = math.exp(min(V[d.id] / nvt, 700))
                G = d.IS * e / nvt + 1e-12                      # + gmin keeps reverse-biased diodes solvable
                lin[d.id] = (G, d.IS * (e - 1) - G * V[d.id])
            m = self._mna(states, lin)
            new = {}
            for d in exps:                                       # SPICE-style junction voltage limiting
                nvt, vold, vnew = d.n * self.VT, V[d.id], self._vak(m, d)
                vcrit = nvt * math.log(nvt / (math.sqrt(2) * d.IS))
                if vnew > vcrit and abs(vnew - vold) > 2 * nvt:
                    if vold > 0:
                        arg = 1 + (vnew - vold) / nvt
                        vnew = vold + nvt * math.log(arg) if arg > 0 else vcrit
                    else:
                        vnew = nvt * math.log(vnew / nvt)
                new[d.id] = vnew
            hist.append(dict(new))
            done = all(abs(new[i] - V[i]) <= 1e-12 + 1e-10 * abs(V[i]) for i in V)
            V = new
            if done:
                return m, V, hist
        raise SolverError("The diode equation did not converge in 150 Newton iterations")

    # ------------------------------------------------------------------
    def _check(self, m: MNA, states: Dict[str, str], vexp: Dict[str, float]) -> List[Tuple[bool, str]]:
        out = []
        for d in self.diodes:
            st = states[d.id]
            if st == "exp":
                v = vexp[d.id]
                out.append((True, f"{d.id}: V_AK = {fmt_si(v, 'V', self.sig)}, "
                                  f"I_D = {fmt_si(self._exp_current(d, v), 'A', self.sig)} (diode equation)"))
                continue
            if st == "off":
                v = self._vak(m, d)
                ok = v <= d.Von + 1e-9
                txt = f"{d.id} OFF needs V_AK < {fmt_si(d.Von, 'V')}: V_AK = {fmt_si(v, 'V', self.sig)}"
                if d.kind == "zener":
                    ok = ok and -v <= d.VZ + 1e-9
                    txt += f" and V_KA < VZ = {fmt_si(d.VZ, 'V')}"
                out.append((ok, txt + (" ✓" if ok else " ✗")))
                continue
            i = self._current(m, d, st)
            ok = i >= -1e-12
            what = "I_D" if st == "on" else "I_Z (cathode → anode)"
            need = "ON needs I_D ≥ 0" if st == "on" else "in breakdown needs I_Z ≥ 0"
            out.append((ok, f"{d.id} {need}: {what} = {fmt_si(i, 'A', self.sig)}" + (" ✓" if ok else " ✗")))
        return out

    @staticmethod
    def _state_text(d: Diode, st: str) -> str:
        return {"on": f"{d.id} ON", "off": f"{d.id} OFF", "bz": f"{d.id} in breakdown",
                "exp": f"{d.id} (diode equation)"}[st]

    def _replace_text(self, d: Diode, st: str) -> str:
        if st == "off":
            return f"{d.id} → open circuit"
        if st == "on":
            if d.kind == "ideal":
                return f"{d.id} → short circuit ({node_name(d.a)} = {node_name(d.k)})"
            return (f"{d.id} → {fmt_si(d.Von, 'V')} source, + at anode {node_name(d.a)}"
                    + (f", in series with {fmt_si(d.rd, 'Ω')}" if d.rd else ""))
        if st == "bz":
            return (f"{d.id} → {fmt_si(d.VZ, 'V')} source, + at cathode {node_name(d.k)}"
                    + (f", in series with {fmt_si(d.rz, 'Ω')}" if d.rz else ""))
        return f"{d.id} → tangent-line model at each iteration"

    # ------------------------------------------------------------------
    def run(self) -> SolveResult:
        queries = [q.replace(" ", "") for q in self.nl.query]
        pwl = [d for d in self.diodes if d.kind != "exp"]
        n_combos = math.prod(len(d.states) for d in pwl) if pwl else 1
        if n_combos > MAX_COMBOS:
            raise NotSupportedError(f"Too many diodes to try every state ({n_combos} combinations)")

        for d in self.diodes:
            desc, rule = d.label(self.VT)
            self.trace.add(f"Diode model — {desc}", rule, "; ".join(d.notes) or None)

        tried: List[Tuple[Dict[str, str], str]] = []
        found = []
        for combo in itertools.product(*[d.states for d in pwl]) if pwl else [()]:
            states = {d.id: st for d, st in zip(pwl, combo)}
            states.update({d.id: "exp" for d in self.diodes if d.kind == "exp"})
            try:
                if any(d.kind == "exp" for d in self.diodes):
                    m, vexp, hist = self._newton(states)
                else:
                    m, vexp, hist = self._mna(states), {}, []
            except SolverError as e:
                tried.append((states, f"no unique solution ({e})"))
                continue
            checks = self._check(m, states, vexp)
            bad = [t for ok, t in checks if not ok]
            if bad:
                tried.append((states, bad[0]))
                continue
            found.append((states, m, vexp, hist, checks, len(tried)))
        if not found and not pwl and tried:
            raise SolverError(tried[0][1])
        if not found:
            raise SolverError("No combination of diode states is consistent: check the diode directions and "
                              "the source polarities")
        states, m, vexp, hist, checks, n_before = found[0]
        if len(found) > 1:
            self.res.verification_notes.append(
                "more than one set of diode states is consistent; the first one tried is reported")

        # --- steps ----------------------------------------------------
        for st_, why in tried[:min(n_before, 8)]:       # the guesses rejected before the answer was found
            self.trace.add("Try " + ", ".join(self._state_text(self.by_id[i], s) for i, s in st_.items()),
                           None, f"✗ {why}: this assumption is wrong", level="full")
        if pwl:
            self.trace.add("Assume " + ", ".join(self._state_text(d, states[d.id]) for d in pwl),
                           ";  ".join(self._replace_text(d, states[d.id]) for d in pwl))

        lin_q = []
        for q in queries:
            mq = _Q.match(q)
            kind, ref = (mq.group(1), mq.group(2)) if mq else (q, None)
            if kind in ("I", "P", "bias") and ref in self.by_id:
                continue
            if kind not in ALLOWED_KINDS:
                continue
            lin_q.append(q)
        from solvers.dc_ac import _Job                  # local import: dc_ac imports nothing from here
        if not lin_q:
            # every query is about the diodes themselves (I, P, bias): the linear solve still needs a query,
            # so ask for one node voltage; its answer is not reported (only the asked queries are copied)
            node = next(n for c in self._components(states, self._lin_at(vexp)) for n in c.nodes if n != GROUND)
            lin_q = [f"V({node})"]
        lin_nl = Netlist(question=self.nl.question, components=self._components(states, self._lin_at(vexp)),
                         analysis="dc", query=lin_q, assumptions=list(self.nl.assumptions))
        job = _Job(lin_nl, self.conv)
        job_res = job.run()
        if job._main is None:
            job.main()
        exp_mode = any(d.kind == "exp" for d in self.diodes)
        if exp_mode:
            self._newton_steps(hist, vexp)
        for st in job.trace.all_steps():                # the linear solve (companion model: detail only)
            self.trace.add(st.action, st.equation, st.result, level="full" if exp_mode else st.level)
        self.trace.add("Check the assumptions" if pwl else "Diode operating point",
                       ";  ".join(t for _, t in checks),
                       "all assumptions hold ✓" if pwl else None)

        # --- answers in the order asked ----------------------------------
        jm = job._main
        for q in queries:
            mq = _Q.match(q)
            kind, ref = (mq.group(1), mq.group(2)) if mq else (q, None)
            if kind in ("I", "P", "bias") and ref in self.by_id:
                self._diode_answer(q, kind, self.by_id[ref], states[ref], jm, vexp)
            elif q in job_res.answers:
                self.res.answers[q] = job_res.answers[q]
                for k, v in job_res.numeric.items():
                    if k == q or k.startswith(q + "."):
                        self.res.numeric[k] = v
                if q in job_res.symbolic:
                    self.res.symbolic[q] = job_res.symbolic[q]
            else:
                self.res.answers[q] = (f"Could not compute: '{q}' is not available for diode circuits yet "
                                       "(ask for V, I, P or bias)")
                self.res.numeric[q] = None
        self.res.verification_notes += job_res.verification_notes
        for d in self.diodes:                           # for the verifier: the state and the model used
            self.res.symbolic[f"state.{d.id}"] = states[d.id]
            self.res.symbolic[f"diode.{d.id}"] = json.dumps(d.as_dict())
        if queries and all(self.res.answers.get(q, "").startswith("Could not compute") for q in queries):
            raise SolverError(" | ".join(self.res.answers.values()))
        self.res.trace = self.trace
        return self.res

    def _lin_at(self, vexp: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
        lin = {}
        for i, v in vexp.items():
            d = self.by_id[i]
            nvt = d.n * self.VT
            e = math.exp(min(v / nvt, 700))
            G = d.IS * e / nvt + 1e-12
            lin[i] = (G, d.IS * (e - 1) - G * v)
        return lin

    def _newton_steps(self, hist: List[Dict[str, float]], vexp: Dict[str, float]) -> None:
        ids = list(vexp)
        rows = [f"it {k}: " + ", ".join(f"V_{i} = {fmt_num(h[i], 6)} V" for i in ids)
                for k, h in enumerate(hist, 1)]
        if len(rows) > 9:
            rows = rows[:6] + ["…"] + rows[-2:]
        self.trace.add("Newton-Raphson: replace each diode by its tangent line at the present V_D, solve, repeat",
                       "start V_D = 0.6 V;  " + ";  ".join(rows),
                       f"converged after {len(hist)} iterations")
        for i in ids:
            d = self.by_id[i]
            v = vexp[i]
            I = self._exp_current(d, v)
            self.trace.add(f"Diode current of {i} from the diode equation",
                           f"I_D = IS·(e^(V_D/(n·VT)) − 1) = {fmt_num(d.IS)}·(e^({fmt_num(v, 6)}/{fmt_num(d.n * self.VT)})"
                           f" − 1)", f"I_D = {fmt_si(I, 'A', self.sig)}")

    def _diode_answer(self, q: str, kind: str, d: Diode, st: str, m: MNA, vexp: Dict[str, float]) -> None:
        v = vexp[d.id] if st == "exp" else self._vak(m, d)
        i = self._current(m, d, st, vexp.get(d.id))
        if kind == "I":
            if st == "bz":
                self.res.answers[q] = f"{fmt_si(i, 'A', self.sig)} (Zener current, cathode → anode, in breakdown)"
            else:
                self.res.answers[q] = f"{fmt_si(i, 'A', self.sig)} (anode {node_name(d.a)} → cathode {node_name(d.k)})"
            self.res.numeric[q] = i
            return
        if kind == "P":
            p = (-v) * i if st == "bz" else v * i            # both are V·I in the conducting direction
            self.res.answers[q] = f"{fmt_si(p, 'W', self.sig)} absorbed"
            self.res.numeric[q] = p
            self.trace.add(f"Power in {d.id}", f"P = V·I = {fmt_si(abs(v), 'V', self.sig)} × {fmt_si(i, 'A', self.sig)}",
                           f"P({d.id}) = {self.res.answers[q]}")
            return
        word = {"on": "ON (forward-biased, conducting)", "off": "OFF (not conducting)",
                "bz": "in reverse breakdown (regulating)", "exp": "conducting" if i > 1e-9 else "reverse-biased"}[st]
        cur = f"I_Z = {fmt_si(i, 'A', self.sig)}" if st == "bz" else f"I_D = {fmt_si(i, 'A', self.sig)}"
        self.res.answers[q] = f"{d.id} {word}: V_AK = {fmt_si(v, 'V', self.sig)}, {cur}"
        self.res.numeric[q] = None
        self.res.numeric[q + ".V"] = v
        self.res.numeric[q + ".I"] = i


def node_name(n: str) -> str:
    return "ground" if n == GROUND else n