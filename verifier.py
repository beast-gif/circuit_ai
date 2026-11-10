"""
Independent cross-check of the solver's answers with ngspice.

Idea: our solver (SymPy MNA) and ngspice are completely different programs. If both
give the same number, the answer is very likely right; if they disagree, the answer
is flagged instead of being shown with confidence.

For each query the circuit is rebuilt in SPICE syntax and measured directly:

  V(n), V(a,b), Vth   .op (DC) or .ac at the given frequency  -> v(a) - v(b)
  I(X), P(X)          a 0 V "sense" source in series with X measures its current;
                      P = (v(n0) - v(n1)) · i  (AC: ½·Re(V·I*))
  Rth / Z / Rin / Rout  sources off + 1 A test source -> v(a) - v(b)
  Isc / In            a 0 V source shorts a-b and measures the current
  tau                 Rth seen by the C (or L) measured as above, then Rth·C (or L/Rth)
  H(s), gain          driver = 1, others = 0; H(jω) compared at 10 frequencies (0.1 Hz .. 100 MHz)
  poles, zeros, f0,   these are exact algebra on H(s), so they are verified when H(s) matches
  Qfactor, bandwidth  ngspice at every test frequency
  fc                  H(s) check + ngspice |H(j2πfc)| / |H|max must equal 1/√2

Comparison:  relative error = |solver − ngspice| / |ngspice|  <=  tolerance (config.yaml).

Result: SolveResult.verified = True (all checks passed), False (any mismatch),
        None (nothing could be checked, e.g. symbolic values or ngspice missing).
"""
from __future__ import annotations

import cmath
import json
import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp

from llm_client import load_config, load_conventions
from schemas import AnalysisType, Component, Netlist, SolveResult, is_symbolic, parse_value
from solvers.bjt import resolve_bjt
from solvers.diode import resolve_diode
from solvers.mna import resolve_opamp
from solvers.mosfet import resolve_mosfet
from solvers.base import fmt_complex, fmt_num, fmt_si

GROUND = "0"
INDEP = {"V", "I"}
LINEAR = {"R", "C", "L", "V", "I", "E", "G", "F", "H", "OPAMP", "D"}   # D: exponential model, or linearised
MOS_LEVEL1 = "LEVEL=1"               # ngspice MOSFET level 1 is exactly the square law (W = L: I_D = ½·KP·V_OV²)
K_OVER_Q = 8.617333262e-5                 # V/K: VT = (k/q)·T
# An ideal op-amp is simulated as a VCVS  out-0 = A·(V+ − V−)  with a huge A. The closed-loop error is
# about |closed-loop gain| / A: an integrator with |H| = 15 900 at 0.1 Hz is off by 1.6e-5 with A = 1e9
# (it would be 1.6 % with A = 1e6, which is why 1e9 is used).
OPAMP_GAIN = 1e9
OUT_NAMES = ("out", "vout", "Vout", "o", "output")
SAMPLE_FREQS = [0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
UNITS = {"V": "V", "Vth": "V", "I": "A", "Isc": "A", "In": "A", "P": "W",
         "Rth": "Ω", "Zth": "Ω", "Z": "Ω", "Rn": "Ω", "Rin": "Ω", "Rout": "Ω",
         "tau": "s", "fc": "Hz", "f0": "Hz", "bandwidth": "Hz",
         "IC": "A", "IB": "A", "IE": "A", "VCE": "V", "VEC": "V", "VBE": "V", "VEB": "V",
         "ID": "A", "IS": "A", "IG": "A", "VGS": "V", "VSG": "V", "VDS": "V", "VSD": "V", "VOV": "V"}
DERIVED = {"poles", "zeros", "f0", "Qfactor", "bandwidth", "filter"}
# differences below these are treated as zero (textbook values are mV / µA and up)
ABS_FLOOR = {"V": 1e-6, "A": 1e-9, "W": 1e-9, "Ω": 1e-3, "s": 1e-12, "Hz": 1e-6, "": 1e-9}
_Q = re.compile(r"^(\w+)\((\w+)(?:,(\w+))?\)$")
_PRINT = re.compile(r"^\s*(q\d+)\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)", re.MULTILINE)
_s = sp.Symbol("s")

Src = Tuple[float, float, float]          # (DC value, AC magnitude, AC phase in degrees)


class SpiceError(RuntimeError):
    """ngspice could not run the circuit."""


class _Skip(Exception):
    """This query cannot be checked numerically (not a failure)."""


@dataclass
class Check:
    query: str
    solver: str
    spice: str
    rel_err: Optional[float]
    status: str                    # "ok" | "MISMATCH" | "skipped"
    note: str = ""


@dataclass
class VerifyReport:
    checks: List[Check] = field(default_factory=list)
    verified: Optional[bool] = None
    notes: List[str] = field(default_factory=list)
    ngspice: Optional[str] = None

    def table(self) -> str:
        if not self.checks:
            return "  (no checks)" + ("".join(f"\n  note: {n}" for n in self.notes))
        rows = [("query", "solver", "ngspice", "error", "status")]
        for c in self.checks:
            err = "-" if c.rel_err is None else f"{100 * c.rel_err:.4f}%"
            rows.append((c.query, c.solver, c.spice, err, c.status))
        w = [min(34, max(len(r[i]) for r in rows)) for i in range(5)]
        out = ["  " + "  ".join(r[i][:w[i]].ljust(w[i]) for i in range(5)) for r in rows]
        out.insert(1, "  " + "  ".join("-" * x for x in w))
        for c in self.checks:
            if c.note:
                out.append(f"  {c.query}: {c.note}")
        out += [f"  note: {n}" for n in self.notes]
        return "\n".join(out)


# ----------------------------------------------------------------------
# ngspice
# ----------------------------------------------------------------------
def find_ngspice(configured: Optional[str] = None) -> Optional[str]:
    candidates = [configured, r"C:\Spice64\bin\ngspice_con.exe",
                  shutil.which("ngspice_con"), shutil.which("ngspice")]
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    return None


def _num(x: float) -> str:
    return f"{float(x):.12g}"


def vexpr(a: str, b: str = GROUND) -> str:
    """ngspice expression for V(a) - V(b)."""
    va = "0" if a == GROUND else f"v({a})"
    if b == GROUND:
        return va
    vb = f"v({b})"
    return f"({va} - {vb})" if a != GROUND else f"(-{vb})"


def build_deck(nl: Netlist, sources: Dict[str, Src], removed: Sequence[str] = (),
               sense: Sequence[str] = (), extra: Sequence[str] = (), conventions: Optional[dict] = None,
               diode_mode: str = "model") -> str:
    """Netlist -> SPICE deck. Element X becomes <type>_<id> (e.g. R_R1) so names always start
    with the right letter. Components in `sense` get a 0 V source V_sense_<id> in series at nodes[0]."""
    comps = [c for c in nl.components if c.id not in set(removed)]
    by_id = {c.id: c for c in comps}
    sense = set(sense)
    for c in comps:                                   # F/H in SPICE must be controlled by a V source
        if c.type.value in ("F", "H"):
            ctrl = by_id.get((c.control or "").strip())
            if ctrl is None:
                raise _Skip(f"{c.id}: controlling component {c.control} is not in this circuit")
            if ctrl.type.value != "V":
                sense.add(ctrl.id)

    def ctrl_name(cid: str) -> str:
        return f"V_{cid}" if by_id[cid].type.value == "V" else f"V_sense_{cid}"

    lines = ["* circuit check"]
    models: List[str] = []
    for c in comps:
        t = c.type.value
        nodes = list(c.nodes)
        if c.id in sense:
            xs = f"xs_{c.id}"
            lines.append(f"V_sense_{c.id} {nodes[0]} {xs} DC 0 AC 0")
            nodes[0] = xs
        name = f"{t}_{c.id}"
        if t in ("R", "C", "L"):
            lines.append(f"{name} {nodes[0]} {nodes[1]} {_num(parse_value(c.value))}")
        elif t in INDEP:
            dc, ac, ph = sources.get(c.id, (0.0, 0.0, 0.0))
            lines.append(f"{name} {nodes[0]} {nodes[1]} DC {_num(dc)} AC {_num(ac)} {_num(ph)}")
        elif t in ("E", "G"):
            lines.append(f"{name} {' '.join(nodes)} {_num(parse_value(c.value))}")
        elif t == "OPAMP":                                  # nodes = [out, in+, in-]
            info = resolve_opamp(c, conventions)
            out, inp, inm = nodes
            if info is None:
                lines.append(f"E_{c.id} {out} 0 {inp} {inm} {_num(OPAMP_GAIN)}")
            else:
                # Independent macro-model (not the solver's equation):
                #   G (1 S) pushes Vd into node x, R = A0 Ω to ground  -> V(x) = A0·Vd
                #   C = 1/(2π·GBW) across R                             -> pole at GBW/A0
                #   unity buffer E from x, then Rout to the output; Rin between the inputs
                x, o = f"{c.id}_x", (f"{c.id}_o" if info["Rout"] else out)
                lines.append(f"G_{c.id} 0 {x} {inp} {inm} 1")
                lines.append(f"R_{c.id}_a {x} 0 {_num(info['A0'])}")
                if info["GBW"]:
                    lines.append(f"C_{c.id}_p {x} 0 {_num(1 / (2 * math.pi * info['GBW']))}")
                lines.append(f"E_{c.id} {o} 0 {x} 0 1")
                if info["Rout"]:
                    lines.append(f"R_{c.id}_o {o} {out} {_num(info['Rout'])}")
                if info["Rin"]:
                    lines.append(f"R_{c.id}_i {inp} {inm} {_num(info['Rin'])}")
        elif t in ("F", "H"):
            lines.append(f"{name} {nodes[0]} {nodes[1]} {ctrl_name(c.control.strip())} {_num(parse_value(c.value))}")
        elif t == "D":
            # diode_mode "model": only the exponential model reaches here (piecewise-linear diodes are
            # replaced by their state first); "real": a typical silicon diode, for the information note
            d = resolve_diode(c, conventions)
            if diode_mode == "real":
                params = f"IS={_num(d.IS if d.kind == 'exp' else 1e-14)} N={_num(d.n if d.kind == 'exp' else 1)}"
                if d.kind == "pwl" and d.rd:
                    params += f" RS={_num(d.rd)}"
                if d.kind == "zener":
                    params += f" BV={_num(d.VZ)} IBV=1e-3" + (f" RS={_num(d.rz)}" if d.rz else "")
            elif d.kind == "exp":
                params = f"IS={_num(d.IS)} N={_num(d.n)}"
            else:
                raise _Skip(f"{c.id}: a piecewise-linear diode must be replaced by its state first")
            lines.append(f"D_{c.id} {nodes[0]} {nodes[1]} DM_{c.id}")
            models.append(f".model DM_{c.id} D({params})")
        elif t == "M" and diode_mode == "real":
            # ngspice level-1 MOSFET with W = L (the default): I_D = ½·KP·(V_GS − VTO)²·(1 + LAMBDA·V_DS),
            # the same square law the solver uses. The bulk is tied to the source.
            mo = resolve_mosfet(c, conventions)
            lines.append(f"M_{c.id} {nodes[0]} {nodes[1]} {nodes[2]} {nodes[2]} MO_{c.id}")
            models.append(f".model MO_{c.id} {'PMOS' if mo.pmos else 'NMOS'}({MOS_LEVEL1} "
                          f"VTO={_num(-mo.Vt if mo.pmos else mo.Vt)} KP={_num(mo.k)} LAMBDA={_num(mo.lam)})")
        elif t == "Q" and diode_mode == "real":
            # information only: a real BJT (Gummel-Poon with the question's β, IS = 1e-14 A, no Early effect)
            q = resolve_bjt(c, conventions)
            lines.append(f"Q_{c.id} {nodes[0]} {nodes[1]} {nodes[2]} QM_{c.id}")
            models.append(f".model QM_{c.id} {'PNP' if q.pnp else 'NPN'}(BF={_num(q.beta)} IS=1e-14)")
        else:
            raise _Skip(f"{c.id}: type {t} is not checked yet")
    if models:
        # ngspice's VT is (k/q)·T: pick the temperature that gives the VT of conventions.yaml (25 mV -> 17 °C),
        # with TNOM = TEMP so IS is used exactly as given
        vt = float(((conventions or {}).get("semiconductor") or {}).get("VT", 0.025))
        tc = vt / K_OVER_Q - 273.15
        lines += models + [f".options temp={tc:.6f} tnom={tc:.6f}"]
    lines += list(extra)
    # 1 TΩ from every node to ground: keeps ngspice's DC bias point solvable when a node is reached
    # only through capacitors (needed before an AC run). Effect on kΩ-MΩ circuits: < 1e-6 relative.
    # Only added when the circuit has a C or L: without them it is not needed, and it would spoil
    # very large answers (a non-inverting amp's input resistance is ~1e10 Ω, 1 % of 1 TΩ).
    if any(c.type.value in ("C", "L") for c in comps):
        lines.append(".options rshunt=1e12")
    return "\n".join(lines) + "\n"


class _Ctl:
    """Builds the .control block: analyses + 'let qN = expr' / 'print qN' probes."""

    def __init__(self) -> None:
        self.lines: List[str] = []
        self.n = 0

    def cmd(self, line: str) -> None:
        self.lines.append(line)

    def probe(self, expr: str) -> str:
        q = f"q{self.n}"
        self.n += 1
        self.lines += [f"let {q} = {expr}", f"print {q}"]
        return q


# ----------------------------------------------------------------------
class Verifier:
    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or load_config()
        sv = cfg.get("solver", {}) or {}
        self.tol = float(sv.get("spice_tolerance_linear", 1e-3))
        self.tol_sweep = float(sv.get("spice_tolerance_sweep", 2e-3))
        self.timeout = float(sv.get("ngspice_timeout_s", 30))
        self.exe = find_ngspice(sv.get("ngspice_path"))
        self.conv = load_conventions(cfg)

    # ------------------------------------------------------------------
    def run(self, deck: str, control: List[str], keys: Sequence[str] = (),
            files: Sequence[str] = ()) -> Tuple[Dict[str, float], Dict[str, str]]:
        """Run ngspice in batch mode. Returns printed probe values and the text of written files."""
        if self.exe is None:
            raise SpiceError("ngspice not found")
        text = (deck + ".control\nset noaskquit\nset wr_singlescale\nset numdgt=12\n"
                + "\n".join(control) + "\n.endc\n.end\n")
        with tempfile.TemporaryDirectory() as td:
            Path(td, "check.cir").write_text(text, encoding="ascii", errors="replace")
            try:
                proc = subprocess.run([self.exe, "-b", "check.cir"], cwd=td, capture_output=True,
                                      text=True, errors="replace", timeout=self.timeout)
            except subprocess.TimeoutExpired as e:
                raise SpiceError(f"ngspice timed out after {self.timeout:g} s") from e
            out = (proc.stdout or "") + "\n" + (proc.stderr or "")
            vals = {m.group(1).lower(): float(m.group(2)) for m in _PRINT.finditer(out)}
            data = {f: Path(td, f).read_text(errors="replace") for f in files if Path(td, f).exists()}
        missing = [k for k in keys if k not in vals] + [f for f in files if f not in data]
        if missing:
            tail = "\n".join(l for l in out.strip().splitlines()[-12:])
            raise SpiceError(f"no result for {missing}. ngspice said:\n{tail}")
        return vals, data

    # ------------------------------------------------------------------
    def verify(self, nl: Netlist, res: SolveResult) -> VerifyReport:
        rep = VerifyReport(ngspice=self.exe)
        self.nl, self.res = nl, res
        self.mode = "ac" if nl.analysis == AnalysisType.AC else "dc"
        self.freq = parse_value(nl.frequency) if nl.frequency else None
        self.indep = [c for c in nl.components if c.type.value in INDEP]
        self.nodes = set(nl.nodes())
        self._hchecks: Dict[str, Check] = {}
        self.forward_ids: set = set()            # diode ids: their current is reported anode -> cathode
        self.diodes: Dict[str, Tuple[object, str]] = {}
        self.bjts: Dict[str, Tuple[object, str]] = {}
        self.fets: Dict[str, Tuple[object, str, tuple]] = {}
        orig = self.orig = nl
        if nl.analysis in (AnalysisType.SMALL_SIGNAL, AnalysisType.AC) and any(
                c.type.value in ("Q", "M") for c in nl.components):
            return self._verify_amp(nl, res, rep)               # small-signal BJT amplifier
        if nl.analysis == AnalysisType.TRANSIENT and any(c.type.value == "D" for c in nl.components):
            if any(c.type.value in ("C", "L") for c in nl.components):
                return self._verify_tran(nl, res, rep)          # filtered rectifiers / clampers
            return self._verify_waveform(nl, res, rep)          # rectifiers / clippers
        if any(c.type.value in ("D", "Q", "M") for c in nl.components):
            lin, why = self._linearize(nl, res)
            if lin is None:
                rep.notes.append(why)
                res.verified = None
                res.verification_notes.append(f"not verified: {why}")
                return rep
            self.nl = nl = lin
            # the diodes' stand-in sources are part of the circuit now: give them their values in the deck
            self.indep = [c for c in nl.components if c.type.value in INDEP]
            self.nodes = set(nl.nodes())

        reason = self._uncheckable()
        if reason:
            rep.notes.append(reason)
            res.verified = None
            res.verification_notes.append(f"not verified: {reason}")
            return rep

        for q in [x.replace(" ", "") for x in nl.query]:
            if res.answers.get(q, "").startswith("Could not compute") or q not in res.answers:
                rep.checks.append(Check(q, "-", "-", None, "skipped", "the solver gave no answer"))
                continue
            try:
                rep.checks.append(self._check_query(q))
            except _Skip as e:
                rep.checks.append(Check(q, res.answers.get(q, "-"), "-", None, "skipped", str(e)))
            except SpiceError as e:
                rep.checks.append(Check(q, res.answers.get(q, "-"), "-", None, "skipped", f"ngspice failed: {e}"))

        if self.diodes or self.bjts or self.fets:
            rep.checks += self._state_checks()
            self._real_diode_note(orig)

        statuses = {c.status for c in rep.checks}
        rep.verified = False if "MISMATCH" in statuses else (True if "ok" in statuses else None)
        res.verified = rep.verified
        for c in rep.checks:
            if c.status == "MISMATCH":
                res.verification_notes.append(f"{c.query}: solver {c.solver} vs ngspice {c.spice}")
            elif c.status == "skipped":
                res.verification_notes.append(f"{c.query}: not checked ({c.note})")
        return rep

    # ------------------------------------------------------------------
    # Diodes
    # ------------------------------------------------------------------
    def _linearize(self, nl: Netlist, res: SolveResult) -> Tuple[Optional[Netlist], str]:
        """Replace every piecewise-linear diode by the state the solver reported (ON -> its voltage source,
        OFF -> 1 TΩ). ngspice then checks the circuit equations, and _state_checks() checks that each
        state is right using ngspice's own currents and voltages. Exponential diodes stay real diodes."""
        comps = []
        for c in nl.components:
            if c.type.value == "M":                  # MOSFET: its tangent model at the reported Q-point
                st = res.symbolic.get(f"state.{c.id}")
                opj = res.symbolic.get(f"op.{c.id}")
                if st is None:
                    return None, "the solver did not report the MOSFET regions"
                mo = resolve_mosfet(c, self.conv)
                vgs, vds = (json.loads(opj)[:2] if opj else (mo.Vt + 0.5, 1.0))
                self.fets[c.id] = (mo, st, (vgs, vds))
                comps += mo.parts(st, vgs, vds, off_as_r=True)
                continue
            if c.type.value == "Q":                  # BJT: its region's linear model (cutoff: 1 TΩ)
                st = res.symbolic.get(f"state.{c.id}")
                if st is None:
                    return None, "the solver did not report the transistor regions"
                q = resolve_bjt(c, self.conv)
                self.bjts[c.id] = (q, st)
                comps += q.parts(st, off_as_r=True, rmin=True)     # 1 mΩ: parallel junctions (mirrors)
                continue
            if c.type.value != "D":
                comps.append(c)
                continue
            st = res.symbolic.get(f"state.{c.id}")
            if st is None:
                return None, "the solver did not report the diode states"
            d = resolve_diode(c, self.conv)
            self.diodes[c.id] = (d, st)
            self.forward_ids.add(c.id)
            if st == "exp":
                comps.append(c)
            elif st == "off":
                comps.append(Component(id=c.id, type="R", nodes=[d.a, d.k], value="1e12"))
            else:
                p, n_ = (d.a, d.k) if st == "on" else (d.k, d.a)
                v, r = (d.Von, d.rd) if st == "on" else (d.VZ, d.rz)
                if r:
                    comps += [Component(id=c.id, type="V", nodes=[p, f"{c.id}_x"], value=_num(v)),
                              Component(id=f"{c.id}_r", type="R", nodes=[f"{c.id}_x", n_], value=_num(r))]
                else:
                    comps.append(Component(id=c.id, type="V", nodes=[p, n_], value=_num(v)))
        return nl.model_copy(update={"components": comps}), ""

    def _state_checks(self) -> List[Check]:
        """ON needs I_D >= 0, breakdown needs I_Z >= 0, OFF needs V_AK below the turn-on voltage:
        all read from ngspice's solution of the circuit."""
        out_q = self._bjt_state_checks() + self._fet_state_checks()
        exprs, sense = {}, []
        for cid, (d, st) in self.diodes.items():
            if st in ("on", "bz"):
                exprs[cid] = f"i(V_sense_{cid})"
                sense.append(cid)
            elif st == "off":
                exprs[cid] = vexpr(d.a, d.k)
        if not exprs:
            return out_q
        try:
            vals = self._dc(exprs, sources=self._natural("dc"), sense=sense)
        except (SpiceError, _Skip) as e:
            return out_q + [Check("diode states", "-", "-", None, "skipped", str(e))]
        out = out_q
        for cid, (d, st) in self.diodes.items():
            if cid not in vals:
                continue
            x = vals[cid]
            if st == "off":
                ok = x <= d.Von + 1e-6 and (d.kind != "zener" or -x <= d.VZ + 1e-6)
                out.append(Check(f"state({cid})", "OFF", f"V_AK = {fmt_si(x, 'V')}", None,
                                 "ok" if ok else "MISMATCH", f"OFF needs V_AK < {fmt_si(d.Von, 'V')}"
                                 + (f" and V_KA < {fmt_si(d.VZ, 'V')}" if d.kind == "zener" else "")))
            else:
                ok = x >= -1e-9
                name = "ON" if st == "on" else "breakdown"
                what = "I_D" if st == "on" else "I_Z"
                out.append(Check(f"state({cid})", name, f"{what} = {fmt_si(x, 'A')}", None,
                                 "ok" if ok else "MISMATCH", f"{name} needs {what} ≥ 0"))
        return out

    def _fet_state_checks(self) -> List[Check]:
        """ngspice solves the circuit with real level-1 MOSFETs; its own V_GS and V_DS must land in the
        region the solver assumed."""
        out = []
        for cid, (mo, st, _op) in self.fets.items():
            try:
                v = self._real_dc({"VGS": vexpr(*mo.gs), "VDS": vexpr(*mo.ds)})
            except (SpiceError, _Skip) as e:
                out.append(Check(f"region({cid})", st, "-", None, "skipped", str(e)))
                continue
            vgs, vds, ov = v["VGS"], v["VDS"], v["VGS"] - mo.Vt
            if st == "sat":
                ok = ov >= -1e-6 and vds >= ov - 1e-6
                need = f"saturation needs V_OV ≥ 0 and {mo.vds_name} ≥ V_OV"
            elif st == "tri":
                ok = ov >= -1e-6 and -1e-6 <= vds <= ov + 1e-6
                need = f"the triode region needs 0 ≤ {mo.vds_name} < V_OV"
            else:
                ok = ov <= 1e-6
                need = f"cutoff needs {mo.vgs_name} < V_t"
            out.append(Check(f"region({cid})", mo.region(st),
                             f"V_OV = {fmt_si(ov, 'V')}, {mo.vds_name} = {fmt_si(vds, 'V')}", None,
                             "ok" if ok else "MISMATCH", need))
        return out

    def _real_dc(self, exprs: Dict[str, str]) -> Dict[str, float]:
        """Solve the ORIGINAL circuit in ngspice with real level-1 MOSFETs (and real diodes / BJTs if there
        are any): ngspice finds the nonlinear operating point itself, so nothing is taken from the solver."""
        lin, lin_indep = self.nl, self.indep
        try:
            self.nl = self.orig
            self.indep = [c for c in self.orig.components if c.type.value in INDEP]
            return self._dc(exprs, sources=self._natural("dc"), diode_mode="real")
        finally:
            self.nl, self.indep = lin, lin_indep

    def _fet_query(self, q: str, kind: str, a: str, unit: str, nat) -> Check:
        mo, st, (vgs0, vds0) = self.fets[a]
        try:
            got = self._real_dc({"VGS": vexpr(*mo.gs), "VDS": vexpr(*mo.ds)})
        except (SpiceError, _Skip) as e:
            raise _Skip(str(e))
        vgs, vds = got["VGS"], got["VDS"]
        idq = mo.current(st, vgs, vds)                   # the square law at ngspice's own node voltages
        if kind in ("bias", "region"):
            c1 = self._compare(q, self.res.numeric.get(f"{q}.ID"), complex(idq), "A")
            c2 = self._compare(q, self.res.numeric.get(f"{q}.VDS"), complex(vds), "V")
            worst = c1 if (c1.rel_err or 0) >= (c2.rel_err or 0) else c2
            status = "MISMATCH" if "MISMATCH" in (c1.status, c2.status) else worst.status
            return Check(q, f"I_D = {c1.solver}, {mo.vds_name} = {c2.solver}",
                         f"I_D = {c1.spice}, {mo.vds_name} = {c2.spice}", worst.rel_err, status,
                         "Q-point: ngspice solved the square law itself (level-1 MOSFET)")
        if kind == "P":
            return self._compare(q, self._solver_value(q), complex(vds * idq), "W")
        val = {"I": idq, "ID": idq, "IS": idq, "IG": 0.0, "VGS": vgs, "VSG": vgs, "VDS": vds, "VSD": vds,
               "VOV": vgs - mo.Vt}.get(kind)
        if val is None:
            raise _Skip(f"no ngspice check for '{kind}' yet")
        return self._compare(q, self._solver_value(q), complex(val), unit)

    def _bjt_exprs(self, cid: str) -> Tuple[Dict[str, str], List[str]]:
        """ngspice expressions for I_B, I_C, V_BE, V_CE of a linearised BJT (+ the sense sources needed)."""
        q, st = self.bjts[cid]
        ex = {"VBE": vexpr(*q.be), "VCE": vexpr(*q.ce)}
        sense = []
        if st == "cutoff":
            ex["IB"], ex["IC"] = "0", "0"
        else:
            ex["IB"] = f"i(V_{cid}_be)"
            if st == "active":
                ex["IC"] = f"i(V_sense_{cid}_c)"          # through the β·I_B source, measured by a 0 V sense
                sense.append(f"{cid}_c")
            else:
                ex["IC"] = f"i(V_{cid}_ce)"
        return ex, sense

    def _bjt_state_checks(self) -> List[Check]:
        out = []
        for cid, (q, st) in self.bjts.items():
            ex, sense = self._bjt_exprs(cid)
            try:
                v = self._dc({k: e for k, e in ex.items() if e != "0"}, sources=self._natural("dc"), sense=sense)
            except (SpiceError, _Skip) as e:
                out.append(Check(f"region({cid})", st, "-", None, "skipped", str(e)))
                continue
            ib, ic = v.get("IB", 0.0), v.get("IC", 0.0)
            vbe, vce = v["VBE"], v["VCE"]
            if st == "active":
                ok = ib >= -1e-9 and vce >= q.VCEsat - 1e-6
                txt, need = f"I_B = {fmt_si(ib, 'A')}, {q.vce_name} = {fmt_si(vce, 'V')}", \
                    f"active needs I_B ≥ 0 and {q.vce_name} ≥ {fmt_si(q.VCEsat, 'V')}"
            elif st == "sat":
                ok = ib >= -1e-9 and ic >= -1e-9 and ic <= q.beta * ib + 1e-9
                txt, need = f"I_B = {fmt_si(ib, 'A')}, I_C = {fmt_si(ic, 'A')}", "saturation needs I_C ≤ β·I_B"
            else:
                ok = vbe <= q.VBE + 1e-6 and (vbe - vce) <= q.VBE + 1e-6
                txt, need = f"{q.vbe_name} = {fmt_si(vbe, 'V')}", f"cutoff needs {q.vbe_name} < {fmt_si(q.VBE, 'V')}"
            region = {"active": "active", "sat": "saturation", "cutoff": "cutoff"}[st]
            out.append(Check(f"region({cid})", region, txt, None, "ok" if ok else "MISMATCH", need))
        return out

    def _real_diode_note(self, orig: Netlist) -> None:
        """Information only: the same circuit with typical real silicon diodes (IS = 1e-14 A, n = 1)."""
        # only a fair comparison for ordinary silicon diodes (0.7 V) and Zeners: not ideal, exponential,
        # germanium or LED models
        if any(d.kind in ("ideal", "exp") or (d.kind != "zener" and abs(d.Von - 0.7) > 0.05)
               for d, _ in self.diodes.values()):
            return
        if any(abs(q.VBE - 0.7) > 0.05 for q, _ in self.bjts.values()):
            return
        if self.fets:
            return                                      # the square law IS the model ngspice would use
        vq = [q for q in (x.replace(" ", "") for x in orig.query)
              if (m := _Q.match(q)) and m.group(1) == "V" and self.res.numeric.get(q) is not None]
        if not vq:
            return
        lin, lin_indep = self.nl, self.indep
        try:
            self.nl = orig
            self.indep = [c for c in orig.components if c.type.value in INDEP]
            exprs = {q: vexpr(_Q.match(q).group(2), _Q.match(q).group(3) or GROUND) for q in vq}
            vals = self._dc(exprs, sources=self._natural("dc"), diode_mode="real")
        except (SpiceError, _Skip):
            return
        finally:
            self.nl, self.indep = lin, lin_indep
        parts, worst = [], 0.0
        for q in vq:
            mine = self.res.numeric[q]
            parts.append(f"{q} = {fmt_si(vals[q], 'V')}")
            worst = max(worst, abs(vals[q] - mine) / max(abs(mine), 1e-3))
        what = ("real silicon devices (IS = 1e-14 A, the question's β)" if self.bjts
                else "a real silicon diode (IS = 1e-14 A, n = 1)")
        self.res.verification_notes.append(
            f"info: with {what} ngspice gives " + ", ".join(parts)
            + f", within {100 * worst:.1f} % of the piecewise-linear model")

    # ------------------------------------------------------------------
    # Rectifiers / clippers (diodes + sinusoidal input, no capacitors)
    # ------------------------------------------------------------------
    def _verify_waveform(self, nl: Netlist, res: SolveResult, rep: VerifyReport) -> VerifyReport:
        """Independent check: ngspice sweeps x = sin(ωt) from -1 to 1 (every sinusoidal source becomes
        amplitude·x) with near-ideal switching diodes (IS = 1e-9 A, N = 0.01: about 4 mV at 1 mA) in series
        with each diode's turn-on voltage, then peak / average / rms / PIV / conduction are measured
        numerically over 20 000 points of one cycle. Nothing is taken from the solver except the queries."""
        def done(why: str) -> VerifyReport:
            rep.notes.append(why)
            res.verified = None
            res.verification_notes.append(f"not verified: {why}")
            return rep

        if self.exe is None:
            return done("ngspice not found (set solver.ngspice_path in config.yaml)")
        if "wave.sources" not in res.symbolic:
            return done("the solver did not report the input sources")
        srcs = json.loads(res.symbolic["wave.sources"])
        Vm = abs(float(res.symbolic.get("wave.Vm", "1")))
        diodes = {c.id: resolve_diode(c, self.conv) for c in nl.components if c.type.value == "D"}
        extra = ["V_xs xs 0 DC 0", ".model DIDEAL D(IS=1e-9 N=0.01)"]
        for c in nl.components:
            n0, n1 = c.nodes[0], c.nodes[1]
            if c.id in srcs:
                amp, off = srcs[c.id]
                el = "E" if c.type.value == "V" else "G"
                if off and c.type.value == "V":
                    extra += [f"E_{c.id} {n0} {c.id}_o xs 0 {_num(amp)}", f"V_{c.id}_off {c.id}_o {n1} DC {_num(off)}"]
                else:
                    extra.append(f"{el}_{c.id} {n0} {n1} xs 0 {_num(amp)}")
                    if off:
                        extra.append(f"I_{c.id}_off {n0} {n1} DC {_num(off)}")
            elif c.id in diodes:
                d = diodes[c.id]
                f_end = f"{c.id}_fr" if d.rd else d.k
                extra += [f"D_{c.id}_f {d.a} {c.id}_f DIDEAL", f"V_{c.id}_on {c.id}_f {f_end} DC {_num(d.Von)}"]
                if d.rd:
                    extra.append(f"R_{c.id}_rd {c.id}_fr {d.k} {_num(d.rd)}")
                if d.kind == "zener":
                    z_end = f"{c.id}_zr" if d.rz else d.a
                    extra += [f"D_{c.id}_z {d.k} {c.id}_z DIDEAL", f"V_{c.id}_vz {c.id}_z {z_end} DC {_num(d.VZ)}"]
                    if d.rz:
                        extra.append(f"R_{c.id}_rz {c.id}_zr {d.a} {_num(d.rz)}")
        rest = nl.model_copy(update={"components": [c for c in nl.components if c.id not in srcs and c.id not in diodes]})
        sources = {c.id: (parse_value(c.value) if c.value not in (None, "") else 0.0, 0.0, 0.0)
                   for c in rest.components if c.type.value in INDEP}

        # one vector per quantity the queries need
        vec: Dict[str, str] = {}
        plan = []
        for q in [x.replace(" ", "") for x in nl.query]:
            m = _Q.match(q)
            if not m or res.answers.get(q, "").startswith("Could not compute") or m.group(1) == "bias":
                rep.checks.append(Check(q, res.answers.get(q, "-")[:40], "-", None, "skipped", "nothing to measure"))
                continue
            kind, a, b = m.group(1), m.group(2), m.group(3)
            if kind.startswith("V") or kind == "vtc":
                expr = vexpr(a, b or GROUND)
            elif kind in ("PIV",):
                expr = f"-({vexpr(diodes[a].a, diodes[a].k)})"
            elif a in diodes:
                expr = f"i(V_{a}_on)" + (f" + i(V_{a}_vz)" if diodes[a].kind == "zener" else "")
            else:
                c = next((x for x in nl.components if x.id == a), None)
                if c is None or c.type.value != "R":
                    rep.checks.append(Check(q, res.answers.get(q, "-")[:40], "-", None, "skipped",
                                            "only diode and resistor currents are measured"))
                    continue
                expr = f"({vexpr(c.nodes[0], c.nodes[1])})/{_num(parse_value(c.value))}"
            key = f"w{len(vec)}"
            vec[key] = expr
            plan.append((q, kind, a, key))
        if not plan:
            return done("no measurable query")

        ctl = _Ctl()
        ctl.cmd("dc V_xs -1 1 0.0005")
        for k, e in vec.items():
            ctl.cmd(f"let {k} = {e}")
        ctl.cmd("wrdata sweep.txt " + " ".join(vec))
        try:
            deck = build_deck(rest, sources, extra=extra, conventions=self.conv)
            _, data = self.run(deck, ctl.lines, files=["sweep.txt"])
        except (SpiceError, _Skip) as e:
            return done(f"ngspice failed: {e}")
        rows = []
        for line in data["sweep.txt"].splitlines():
            try:
                rows.append([float(p) for p in line.split()])
            except ValueError:
                continue
        rows = [r for r in rows if len(r) >= 1 + len(vec)]
        if not rows:
            return done("empty ngspice sweep")
        arr = np.array(rows)
        xs = arr[:, 0]
        if xs[0] > -0.999 or xs[-1] < 0.999:
            return done("ngspice sweep stopped early (convergence)")
        cols = {k: arr[:, 1 + i] for i, k in enumerate(vec)}
        theta = np.linspace(0, 2 * math.pi, 20000, endpoint=False)
        xt = np.sin(theta)

        def cmp(q: str, label: str, mine: Optional[float], theirs: float, unit: str, tol_abs: float) -> Check:
            if mine is None:
                return Check(q, "-", fmt_si(theirs, unit), None, "skipped", "no numeric answer")
            err = abs(mine - theirs)
            ok = err <= max(0.01 * abs(theirs), tol_abs)
            rel = err / max(abs(theirs), 1e-30)
            return Check(q if not label else f"{q}.{label}", fmt_si(mine, unit) if unit != "°" else f"{mine:.4g}°",
                         fmt_si(theirs, unit) if unit != "°" else f"{theirs:.4g}°", rel, "ok" if ok else "MISMATCH",
                         "ngspice sweep with switching diodes over one cycle")

        for q, kind, a, key in plan:
            v = np.interp(xt, xs, cols[key])
            unit = "A" if kind in ("I", "Imax", "Iavg", "Irms") else "V"
            tol = 0.005 * Vm if unit == "V" else 1e-6
            stats = {"max": float(np.max(v)), "min": float(np.min(v)), "avg": float(np.mean(v)),
                     "rms": float(np.sqrt(np.mean(v * v)))}
            if kind in ("V", "I"):
                for lab in ("max", "min", "avg", "rms"):
                    rep.checks.append(cmp(q, lab, res.numeric.get(f"{q}.{lab}"), stats[lab], unit, tol))
            elif kind in ("Vmax", "Imax", "PIV"):
                theirs = max(stats["max"], 0.0) if kind == "PIV" else stats["max"]
                rep.checks.append(cmp(q, "", res.numeric.get(q), theirs, unit, tol))
            elif kind == "Vmin":
                rep.checks.append(cmp(q, "", res.numeric.get(q), stats["min"], unit, tol))
            elif kind in ("Vavg", "Iavg"):
                rep.checks.append(cmp(q, "", res.numeric.get(q), stats["avg"], unit, tol))
            elif kind in ("Vrms", "Irms"):
                rep.checks.append(cmp(q, "", res.numeric.get(q), stats["rms"], unit, tol))
            elif kind == "conduction":
                peak = max(float(np.max(v)), 1e-12)
                deg = 360.0 * float(np.mean(v > max(1e-6, 1e-3 * peak)))
                mine = res.numeric.get(q)
                ok = mine is not None and abs(mine - deg) <= 2.0
                rep.checks.append(Check(q, f"{mine:.4g}°" if mine is not None else "-", f"{deg:.4g}°", None,
                                        "ok" if ok else "MISMATCH", "within 2° (switching-diode sweep)"))
            elif kind == "vtc":
                pieces = json.loads(res.symbolic.get(q, "[]"))
                worst = 0.0
                for x, y in zip(xs, cols[key]):
                    p = next((p for p in pieces if p[0] - 1e-9 <= x <= p[1] + 1e-9), None)
                    if p is not None:
                        worst = max(worst, abs(p[2] * x + p[3] - y))
                ok = bool(pieces) and worst <= 0.005 * Vm + 0.01
                rep.checks.append(Check(q, "piecewise-linear vo(vin)", f"{len(xs)}-point sweep", worst / max(Vm, 1e-30),
                                        "ok" if ok else "MISMATCH", f"largest difference {fmt_si(worst, 'V')}"))

        statuses = {c.status for c in rep.checks}
        rep.verified = False if "MISMATCH" in statuses else (True if "ok" in statuses else None)
        res.verified = rep.verified
        for c in rep.checks:
            if c.status == "MISMATCH":
                res.verification_notes.append(f"{c.query}: solver {c.solver} vs ngspice {c.spice}")
        return rep

    # ------------------------------------------------------------------
    # Filtered rectifiers / clampers (diodes + C/L + sinusoidal input): ngspice transient
    # ------------------------------------------------------------------
    def _verify_tran(self, nl: Netlist, res: SolveResult, rep: VerifyReport) -> VerifyReport:
        """ngspice .tran with switching diodes (IS = 1e-9 A, N = 0.01, + 1 mΩ like the solver's ideal
        diodes) in series with each diode's turn-on voltage. It starts from the solver's steady-state capacitor
        voltages (uic) and runs 10 cycles: if that state were not the true steady state, the 10th cycle would
        have drifted away from it. The 10th cycle is measured numerically."""
        def done(why: str) -> VerifyReport:
            rep.notes.append(why)
            res.verified = None
            res.verification_notes.append(f"not verified: {why}")
            return rep

        if self.exe is None:
            return done("ngspice not found (set solver.ngspice_path in config.yaml)")
        if "tran.ic" not in res.symbolic or "wave.sources" not in res.symbolic:
            return done("the solver did not report its steady state")
        ic = json.loads(res.symbolic["tran.ic"])
        srcs = json.loads(res.symbolic["wave.sources"])
        f = float(res.symbolic["tran.f"])
        T = 1 / f
        Vm = abs(float(res.symbolic.get("wave.Vm", "1")))
        diodes = {c.id: resolve_diode(c, self.conv) for c in nl.components if c.type.value == "D"}
        # gear integration: no trapezoidal ringing on the diode turn-on spikes; rshunt: no floating bridge nodes
        extra = [".options method=gear rshunt=1e12 itl4=200 gmin=1e-10"]          # the .model line is added per attempt
        for c in nl.components:
            n0, n1 = c.nodes[0], c.nodes[1]
            if c.id in srcs:
                amp, off = srcs[c.id]
                extra.append(f"{c.type.value}_{c.id} {n0} {n1} SIN({_num(off)} {_num(amp)} {_num(f)})")
            elif c.type.value == "C":
                extra.append(f"C_{c.id} {n0} {n1} {_num(parse_value(c.value))} IC={_num(ic.get(c.id, 0.0))}")
            elif c.type.value == "L":
                extra.append(f"L_{c.id} {n0} {n1} {_num(parse_value(c.value))} IC={_num(ic.get(c.id, 0.0))}")
            elif c.id in diodes:
                d = diodes[c.id]
                r_f = d.rd if d.rd else 1e-3
                extra += [f"D_{c.id}_f {d.a} {c.id}_f DIDEAL", f"V_{c.id}_on {c.id}_f {c.id}_fr DC {_num(d.Von)}",
                          f"R_{c.id}_rd {c.id}_fr {d.k} {_num(r_f)}",
                          # 100 MΩ across each diode: defines floating bridge nodes when every diode is OFF
                          # (leaks < 1 µA, against mA loads)
                          f"R_{c.id}_leak {d.a} {d.k} 1e8"]
                if d.kind == "zener":
                    r_z = d.rz if d.rz else 1e-3
                    extra += [f"D_{c.id}_z {d.k} {c.id}_z DIDEAL", f"V_{c.id}_vz {c.id}_z {c.id}_zr DC {_num(d.VZ)}",
                              f"R_{c.id}_rz {c.id}_zr {d.a} {_num(r_z)}"]
        skip = set(srcs) | set(diodes) | {c.id for c in nl.components if c.type.value in ("C", "L")}
        rest = nl.model_copy(update={"components": [c for c in nl.components if c.id not in skip]})
        sources = {c.id: (parse_value(c.value) if c.value not in (None, "") else 0.0, 0.0, 0.0)
                   for c in rest.components if c.type.value in INDEP}

        vec: Dict[str, str] = {}
        plan = []
        for q in [x.replace(" ", "") for x in nl.query]:
            m = _Q.match(q)
            if not m or res.answers.get(q, "").startswith("Could not compute") or m.group(1) == "bias":
                rep.checks.append(Check(q, res.answers.get(q, "-")[:40], "-", None, "skipped", "nothing to measure"))
                continue
            kind, a, b = m.group(1), m.group(2), m.group(3)
            if kind.startswith("V") or kind == "ripple":
                expr = vexpr(a, b or GROUND)
            elif kind == "PIV":
                expr = f"-({vexpr(diodes[a].a, diodes[a].k)})"
            elif a in diodes:
                expr = f"i(V_{a}_on)" + (f" + i(V_{a}_vz)" if diodes[a].kind == "zener" else "")
            else:
                c = next((x for x in nl.components if x.id == a), None)
                if c is None or c.type.value != "R":
                    rep.checks.append(Check(q, res.answers.get(q, "-")[:40], "-", None, "skipped",
                                            "only diode and resistor currents are measured"))
                    continue
                expr = f"({vexpr(c.nodes[0], c.nodes[1])})/{_num(parse_value(c.value))}"
            key = f"w{len(vec)}"
            vec[key] = expr
            plan.append((q, kind, a, key))
        if not plan:
            return done("no measurable query")

        # 10 cycles, data saved for the last one only; max step 0.5 µs-scale (T/40000) so the diode turn-on
        # current spike is resolved (it lasts ~ R·C with the 1 mΩ series resistance)
        cycles, step = 10, T / 40000
        ctl = _Ctl()
        ctl.cmd(f"tran {_num(step)} {_num(cycles * T)} {_num((cycles - 1) * T)} {_num(step)} uic")
        for k, e in vec.items():
            ctl.cmd(f"let {k} = {e}")
        ctl.cmd("wrdata tran.txt " + " ".join(vec))
        # Very sharp switching diodes (N = 0.01, ~4 mV off the ideal drop) first. A bridge whose nodes float
        # before it conducts can defeat ngspice's Newton iteration with them; then once more with N = 0.05
        # (~25 mV more drop at 0.5 A, inside the 0.5 %·Vm tolerance).
        # CJO: a small junction capacitance (as every real diode has) keeps nodes that touch only diodes
        # (a bridge's a and b) from jumping in zero time, which otherwise stops ngspice's time stepping.
        # 10 pF at 50 Hz is ~300 MΩ: no effect on the answers.
        arr, why, n_used = None, "", 0.01
        for n_sw, cjo in ((0.01, "10p"), (0.05, "100p"), (0.1, "1n")):
            try:
                deck = build_deck(rest, sources, extra=[f".model DIDEAL D(IS=1e-9 N={n_sw} CJO={cjo})"] + extra,
                                  conventions=self.conv)
                _, data = self.run(deck, ctl.lines, files=["tran.txt"])
            except (SpiceError, _Skip) as e:
                why = f"ngspice failed: {str(e)[:300]}"
                continue
            rows = []
            for line in data["tran.txt"].splitlines():
                try:
                    rows.append([float(p) for p in line.split()])
                except ValueError:
                    continue
            rows = [r for r in rows if len(r) >= 1 + len(vec) and r[0] >= (cycles - 1) * T * (1 - 1e-9)]
            if len(rows) < 100 or rows[-1][0] < 0.999 * cycles * T:   # never interpolate past ngspice's data
                why = (f"ngspice stopped at t = {fmt_si(rows[-1][0], 's')} of {fmt_si(cycles * T, 's')} "
                       "(convergence)" if rows else "ngspice transient gave too few points")
                continue
            arr = np.array(rows)
            n_used = n_sw
            if n_sw != 0.01:
                rep.notes.append(f"switching diodes softened to N = {n_sw} for ngspice to converge")
            break
        if arr is None:
            return done(why)
        tt = arr[:, 0]
        t0 = (cycles - 1) * T
        grid = np.linspace(t0, cycles * T, 8001)

        def cmp(q: str, mine: Optional[float], theirs: float, unit: str, rel: float, floor: float) -> Check:
            if mine is None:
                return Check(q, "-", fmt_si(theirs, unit), None, "skipped", "no numeric answer")
            err = abs(mine - theirs)
            ok = err <= max(rel * abs(theirs), floor)
            show = (lambda x: fmt_si(x, unit)) if unit not in ("°", "") else (lambda x: f"{x:.4g}{unit}")
            return Check(q, show(mine), show(theirs), err / max(abs(theirs), 1e-30), "ok" if ok else "MISMATCH",
                         "ngspice transient, 10th cycle from the solver's steady state")

        for q, kind, a, key in plan:
            v = np.interp(grid, tt, arr[:, 1 + list(vec).index(key)])
            mx, mn = float(np.max(v)), float(np.min(v))
            avg = float(np.mean(v[:-1]))
            rms = float(np.sqrt(np.mean(v[:-1] ** 2)))
            # the switching diodes' extra forward drop, n·VT·ln(I/IS) ≈ n·0.536 V at ~0.5 A, for up to 2 in series
            vfl = 0.005 * Vm + 2 * n_used * 0.536
            if kind in ("V", "I"):
                unit = "V" if kind == "V" else "A"
                fl = vfl if unit == "V" else 1e-5
                for lab, val in (("max", mx), ("min", mn), ("avg", avg)):
                    rep.checks.append(cmp(f"{q}.{lab}", res.numeric.get(f"{q}.{lab}"), val, unit, 0.01, fl))
            elif kind in ("Vmax", "Vmin", "Vavg", "Vrms"):
                val = {"Vmax": mx, "Vmin": mn, "Vavg": avg, "Vrms": rms}[kind]
                rep.checks.append(cmp(q, res.numeric.get(q), val, "V", 0.01, vfl))
            elif kind == "PIV":
                rep.checks.append(cmp(q, res.numeric.get(q), max(mx, 0.0), "V", 0.01, vfl))
            elif kind in ("Vripple", "Vpp"):
                rep.checks.append(cmp(q, res.numeric.get(q), mx - mn, "V", 0.03, 0.002 * Vm))
            elif kind == "ripple":
                rf = float(np.sqrt(max(np.mean(v[:-1] ** 2) - avg * avg, 0.0)) / abs(avg)) if abs(avg) > 1e-30 else math.inf
                rep.checks.append(cmp(q, res.numeric.get(q), rf, "", 0.05, 1e-4))
            elif kind in ("Imax",):
                rep.checks.append(cmp(q, res.numeric.get(q), mx, "A", 0.05, 1e-4))
            elif kind in ("Iavg", "Irms"):
                rep.checks.append(cmp(q, res.numeric.get(q), avg if kind == "Iavg" else rms, "A", 0.02, 1e-5))
            elif kind == "conduction":
                peak = max(mx, 1e-12)
                deg = 360.0 * float(np.mean(v[:-1] > max(1e-6, 1e-3 * peak)))
                mine = res.numeric.get(q)
                ok = mine is not None and abs(mine - deg) <= 3.0
                rep.checks.append(Check(q, f"{mine:.4g}°" if mine is not None else "-", f"{deg:.4g}°", None,
                                        "ok" if ok else "MISMATCH", "within 3° (ngspice transient)"))

        statuses = {c.status for c in rep.checks}
        rep.verified = False if "MISMATCH" in statuses else (True if "ok" in statuses else None)
        res.verified = rep.verified
        for c in rep.checks:
            if c.status == "MISMATCH":
                res.verification_notes.append(f"{c.query}: solver {c.solver} vs ngspice {c.spice}")
        return rep

    # ------------------------------------------------------------------
    # Small-signal BJT amplifier: ngspice solves the real nonlinear circuit and linearises it itself
    # ------------------------------------------------------------------
    def _verify_amp(self, nl: Netlist, res: SolveResult, rep: VerifyReport) -> VerifyReport:
        """Independent check: ngspice gets the ORIGINAL circuit with real BJTs (Gummel-Poon, the question's β,
        IS chosen so that V_BE comes out at the model's 0.7 V at the solver's bias current) and runs its own
        .op + .ac. So ngspice finds the bias, its own gm and the gain by itself; nothing is taken from the
        solver except the queries. Mid-band is checked at a frequency where the coupling capacitors are
        shorts (or at the question's frequency)."""
        def done(why: str) -> VerifyReport:
            rep.notes.append(why)
            res.verified = None
            res.verification_notes.append(f"not verified: {why}")
            return rep

        if self.exe is None:
            return done("ngspice not found (set solver.ngspice_path in config.yaml)")
        hyb = {c.id: json.loads(res.symbolic[f"hybrid.{c.id}"]) for c in nl.components
               if c.type.value in ("Q", "M") and f"hybrid.{c.id}" in res.symbolic}
        if not hyb:
            return done("the solver did not report the hybrid-π parameters")
        vt = float((self.conv.get("semiconductor") or {}).get("VT", 0.025))
        models, extra = [], []
        for c in nl.components:
            if c.type.value != "Q":
                continue
            q = resolve_bjt(c, self.conv)
            ic = hyb[c.id]["IC"]
            is_ = ic / math.exp(q.VBE / vt)              # so that V_BE = 0.7 V at this bias current
            va = (self.conv.get("semiconductor") or {}).get("VA")
            params = f"BF={_num(q.beta)} IS={_num(is_)}" + (f" VAF={_num(float(va))}" if va else "")
            extra.append(f"Q_{c.id} {c.nodes[0]} {c.nodes[1]} {c.nodes[2]} QM_{c.id}")
            models.append(f".model QM_{c.id} {'PNP' if q.pnp else 'NPN'}({params})")
        tc = vt / K_OVER_Q - 273.15
        extra += models + [f".options temp={tc:.6f} tnom={tc:.6f}"]
        # BJTs are written out by hand above (their IS is chosen to land on the model's V_BE); MOSFETs stay in
        # the netlist and build_deck writes them as real level-1 devices, which IS the square law
        rest = nl.model_copy(update={"components": [c for c in nl.components if c.type.value != "Q"]})

        sig = [c for c in nl.components if c.type.value in INDEP and c.ac not in (None, "")]
        if len(sig) != 1:
            return done("the amplifier needs exactly one signal source for an independent check")
        src = sig[0]
        caps = [c for c in nl.components if c.type.value == "C"]
        midband = res.symbolic.get("ss.midband") == "1"
        if self.freq is not None:
            f_test = self.freq
        elif midband and caps:
            # A frequency where every coupling / bypass capacitor really is a short. The smallest resistance
            # in the circuit may be the transistor's own 1/gm (about 19 Ω in a common-base stage), not a
            # resistor, so that counts too; 2000 / (2π·R_min·C_min) leaves a wide margin. The BJT model has
            # no junction capacitances, so a high test frequency costs nothing.
            rmin = min([parse_value(c.value) for c in nl.components if c.type.value == "R"]
                       + [1.0 / h["gm"] for h in hyb.values() if h.get("gm")]
                       + [h["rpi"] for h in hyb.values() if h.get("rpi")])
            cmin = min(parse_value(c.value) for c in caps)
            f_test = 2000.0 / (2 * math.pi * max(rmin, 1e-6) * cmin)
        else:
            f_test = 1e3
        sources = {c.id: (parse_value(c.value) if c.value not in (None, "") else 0.0,
                          parse_value(c.ac) if c.ac not in (None, "") else 0.0, float(c.ac_phase or 0.0))
                   for c in rest.components if c.type.value in INDEP}

        wants = []
        for q in [x.replace(" ", "") for x in nl.query]:
            m = _Q.match(q)
            kind, a, b = (m.group(1), m.group(2), m.group(3)) if m else (q, None, None)
            if res.numeric.get(q) is None and kind not in ("gain", "H"):
                rep.checks.append(Check(q, res.answers.get(q, "-")[:40], "-", None, "skipped", "no numeric answer"))
                continue
            wants.append((q, kind, a, b))
        if not wants:
            return done("no measurable query")

        out_node = None
        for q, kind, a, b in wants:
            if kind == "H" and a:
                out_node = a
            elif kind == "V" and a and not b:
                out_node = out_node or a
        if out_node is None:
            # the solver renamed its output node to "out": map that back to a node of the ORIGINAL circuit
            # (several original nodes can map to it once the coupling capacitors are shorted; at the test
            # frequency they carry the same AC voltage, so any of them measures the output)
            ren = json.loads(res.symbolic.get("ss.nodes", "{}"))
            same = sorted(n for n, r in ren.items() if r == "out" and n in self.nodes)
            term = {n for c in nl.components if c.type.value in ("Q", "M") for n in c.nodes}
            out_node = next((n for n in same if n not in term), None) or (same[0] if same else None)
        if out_node is None:
            out_node = next((n for n in ("out", "vo", "vout", "o", "output", "c", "e", "d", "s")
                             if n in self.nodes), None)

        def ac_run(exprs: Dict[str, str], src_ac: float, test: Optional[str] = None) -> Dict[str, complex]:
            s2 = dict(sources)
            s2[src.id] = (s2.get(src.id, (0.0, 0.0, 0.0))[0], src_ac, 0.0)
            ctl = _Ctl()
            ctl.cmd(f"ac lin 1 {_num(f_test)} {_num(f_test)}")
            pairs = {k: (ctl.probe(f"real({e})"), ctl.probe(f"imag({e})")) for k, e in exprs.items()}
            deck = build_deck(rest, s2, extra=extra + ([test] if test else []), conventions=self.conv,
                              diode_mode="real")
            vals, _ = self.run(deck, ctl.lines, [k for pr in pairs.values() for k in pr])
            return {k: complex(vals[r], vals[i]) for k, (r, i) in pairs.items()}

        gains: Dict[str, complex] = {}
        try:
            if out_node is None:
                return done("could not tell which node is the amplifier output")
            need = {"vo": vexpr(out_node, GROUND)}
            need["iin"] = f"i(V_{src.id})" if src.type.value == "V" else "0"
            need["vin"] = vexpr(src.nodes[0], src.nodes[1])
            amp_in = self._amp_input(nl, src)
            if amp_in is not None:
                need["vamp"] = vexpr(amp_in, GROUND)
            gains = ac_run(need, 1.0)
        except (SpiceError, _Skip) as e:
            return done(f"ngspice failed: {str(e)[:200]}")

        note_out = f"output measured at node {out_node}"
        rep.notes.append(note_out)
        rout = None
        if any(k == "Rout" for _, k, _, _ in wants) and out_node:
            try:
                rout = ac_run({"vo": vexpr(out_node, GROUND)}, 0.0, test=f"I_test_rout 0 {out_node} AC 1")["vo"]
            except (SpiceError, _Skip) as e:
                rep.notes.append(f"R_out not checked: {str(e)[:120]}")

        what = "real MOSFETs" if any(c.type.value == "M" for c in nl.components) else "real BJTs"
        note = f"ngspice .op + .ac at {fmt_si(f_test, 'Hz')} with {what} (its own bias and gm)"
        for q, kind, a, b in wants:
            try:
                if kind in ("gain", "H"):
                    val = gains["vo"] / gains["vin"] if abs(gains["vin"]) > 1e-12 else None
                    if val is None:
                        raise _Skip("no input voltage")
                    rep.checks.append(self._compare(q, self._solver_value(q), val, "", tol=0.05, note=note))
                elif kind == "Rin":
                    iin = -gains["iin"]                       # current delivered by the source
                    if abs(iin) < 1e-15:
                        raise _Skip("no input current")
                    # R_in is the resistance of the AMPLIFIER input: v at the amplifier's input node divided
                    # by the same current, so a source resistance in between is not counted
                    vamp = gains.get("vamp", gains["vin"])
                    rep.checks.append(self._compare(q, self._solver_value(q), vamp / iin, "Ω",
                                                    tol=0.05, note=note))
                elif kind == "Rout":
                    if rout is None:
                        raise _Skip("output resistance not measured")
                    rep.checks.append(self._compare(q, self._solver_value(q), rout, "Ω", tol=0.05, note=note))
                elif kind in ("gm", "rpi", "re", "ro", "IC", "IE", "IB", "VCE", "bias", "region",
                              "ID", "IS", "IG", "VGS", "VDS", "VSG", "VSD", "VOV"):
                    rep.checks.append(Check(q, res.answers.get(q, "-")[:40], "-", None, "skipped",
                                            "a model parameter, checked through the gain it produces"))
                elif kind == "V" and a:
                    v = gains["vo"] if a == out_node else None
                    if v is None:
                        raise _Skip("only the output node is measured")
                    rep.checks.append(self._compare(q, self._solver_value(q), v, "V", tol=0.05, note=note))
                else:
                    rep.checks.append(Check(q, res.answers.get(q, "-")[:40], "-", None, "skipped",
                                            f"no ngspice check for '{kind}' in a small-signal amplifier"))
            except _Skip as e:
                rep.checks.append(Check(q, res.answers.get(q, "-")[:40], "-", None, "skipped", str(e)))

        statuses = {c.status for c in rep.checks}
        rep.verified = False if "MISMATCH" in statuses else (True if "ok" in statuses else None)
        res.verified = rep.verified
        for c in rep.checks:
            if c.status == "MISMATCH":
                res.verification_notes.append(f"{c.query}: solver {c.solver} vs ngspice {c.spice}")
        return rep

    def _uncheckable(self) -> str:
        if self.exe is None:
            return "ngspice not found (set solver.ngspice_path in config.yaml)"
        if self.nl.analysis == AnalysisType.CONCEPTUAL:
            return "conceptual question"
        later = {c.type.value for c in self.nl.components} - LINEAR
        if later:
            return f"components {sorted(later)} are not checked yet"
        for c in self.nl.components:
            for v in (c.value, c.ac):
                if v not in (None, "") and is_symbolic(v):
                    return "the answer uses symbolic values, which ngspice cannot simulate"
        return ""

    # ------------------------------------------------------------------
    # Source settings
    # ------------------------------------------------------------------
    def _natural(self, mode: str) -> Dict[str, Src]:
        """Sources as written in the question (same rule as the solver for AC)."""
        has_ac = any(c.ac not in (None, "") for c in self.indep)
        vals: Dict[str, Src] = {}
        for c in self.indep:
            dc = parse_value(c.value) if c.value not in (None, "") else 0.0
            if mode == "ac":
                if has_ac:
                    ac = parse_value(c.ac) if c.ac not in (None, "") else 0.0
                    vals[c.id] = (dc, ac, float(c.ac_phase or 0.0))
                else:
                    vals[c.id] = (dc, dc, 0.0)
            else:
                vals[c.id] = (dc, 0.0, 0.0)
        return vals

    def _zeroed(self) -> Dict[str, Src]:
        return {c.id: (0.0, 0.0, 0.0) for c in self.indep}

    def _driven(self, drv_id: str) -> Dict[str, Src]:
        return {c.id: ((1.0, 1.0, 0.0) if c.id == drv_id else (0.0, 0.0, 0.0)) for c in self.indep}

    # ------------------------------------------------------------------
    # Measurements
    # ------------------------------------------------------------------
    def _dc(self, exprs: Dict[str, str], **deck) -> Dict[str, float]:
        ctl = _Ctl()
        ctl.cmd("op")
        keys = {label: ctl.probe(e) for label, e in exprs.items()}
        vals, _ = self.run(build_deck(self.nl, conventions=self.conv, **deck), ctl.lines, list(keys.values()))
        return {label: vals[k] for label, k in keys.items()}

    def _ac(self, exprs: Dict[str, str], freqs: Sequence[float], **deck) -> Dict[str, np.ndarray]:
        ctl = _Ctl()
        keys: Dict[str, List[Tuple[str, str]]] = {label: [] for label in exprs}
        for f in freqs:
            ctl.cmd(f"ac lin 1 {_num(f)} {_num(f)}")
            for label, e in exprs.items():
                keys[label].append((ctl.probe(f"real({e})"), ctl.probe(f"imag({e})")))
        flat = [k for pairs in keys.values() for pair in pairs for k in pair]
        vals, _ = self.run(build_deck(self.nl, conventions=self.conv, **deck), ctl.lines, flat)
        return {label: np.array([complex(vals[r], vals[i]) for r, i in pairs]) for label, pairs in keys.items()}

    def _sweep(self, ratio: str, start: float, stop: float, points: int, lin: bool, **deck):
        ctl = _Ctl()
        ctl.cmd(f"ac {'lin' if lin else 'dec'} {points} {_num(start)} {_num(stop)}")
        ctl.cmd(f"let hr = real({ratio})")
        ctl.cmd(f"let hi = imag({ratio})")
        ctl.cmd("wrdata sweep.txt hr hi")
        _, data = self.run(build_deck(self.nl, conventions=self.conv, **deck), ctl.lines, files=["sweep.txt"])
        f, h = [], []
        for line in data["sweep.txt"].splitlines():
            parts = line.split()
            try:
                nums = [float(p) for p in parts]
            except ValueError:
                continue
            if len(nums) >= 4 and len(nums) % 2 == 0 and nums[0] == nums[2]:
                nums = [nums[0], nums[1], nums[3]]         # older format repeats the scale
            if len(nums) >= 3:
                f.append(nums[0])
                h.append(complex(nums[1], nums[2]))
        if not f:
            raise SpiceError("empty sweep output")
        return np.array(f), np.array(h)

    # ------------------------------------------------------------------
    # Comparison helpers
    # ------------------------------------------------------------------
    def _solver_value(self, q: str) -> Optional[complex]:
        num = self.res.numeric.get(q)
        if num is None:
            return None
        ph = self.res.numeric.get(q + ".phase_deg")
        return num * cmath.exp(1j * math.radians(ph)) if ph is not None else complex(num)

    def _compare(self, q: str, got: Optional[complex], want: complex, unit: str,
                 tol: Optional[float] = None, note: str = "") -> Check:
        tol = self.tol if tol is None else tol
        if got is None:
            raise _Skip("no numeric value to compare")
        diff = abs(got - want)
        # Absolute floor: values that are really 0 come back from ngspice as tiny leftovers
        # (e.g. 2.5e-8 V through the 1 TΩ rshunt on an open node). Below the floor they count as equal;
        # dividing by a near-zero value would otherwise report a 100 % "error".
        err = 0.0 if diff <= ABS_FLOOR.get(unit, 1e-12) else diff / max(abs(want), 1e-9)
        is_complex = abs(want.imag) > 1e-9 * max(1.0, abs(want.real)) or abs(got.imag) > 1e-9 * max(1.0, abs(got.real))
        show = (lambda z: fmt_complex(z, unit)) if is_complex else (lambda z: fmt_si(z.real, unit))
        return Check(q, show(got), show(want), err, "ok" if err <= tol else "MISMATCH", note)

    def _value(self, q: str, expr: str, unit: str, sources: Dict[str, Src], sign: float = 1.0,
               mode: Optional[str] = None, **deck) -> Check:
        """Check one measured quantity in DC, at the given AC frequency, or as a function of s."""
        mode = mode or self.mode
        if mode == "dc":
            x = self._dc({"x": expr}, sources=sources, **deck)["x"] * sign
            return self._compare(q, self._solver_value(q), complex(x), unit)
        if self.freq is not None:
            z = self._ac({"x": expr}, [self.freq], sources=sources, **deck)["x"][0] * sign
            return self._compare(q, self._solver_value(q), z, unit, note=f"at {fmt_si(self.freq, 'Hz')}")
        sym = self.res.symbolic.get(q)
        if sym is None:                               # constant answer in an AC question
            z = self._ac({"x": expr}, [1e3], sources=sources, **deck)["x"][0] * sign
            return self._compare(q, self._solver_value(q), z, unit, note="frequency-independent; checked at 1 kHz")
        return self._sdomain(q, sym, expr, None, sources, sign=sign, **deck)

    def _sdomain(self, q: str, sym: str, num: str, den: Optional[str], sources: Dict[str, Src],
                 sign: float = 1.0, **deck) -> Check:
        """Compare a solver expression in s with ngspice at SAMPLE_FREQS (plus the given frequency)."""
        expr = sp.sympify(sym)
        if expr.free_symbols - {_s}:
            raise _Skip("the expression has symbols other than s")
        fn = sp.lambdify(_s, expr, modules="numpy")
        freqs = list(SAMPLE_FREQS) + ([self.freq] if self.freq else [])
        mine = np.array([complex(fn(1j * 2 * math.pi * f)) for f in freqs])
        exprs = {"n": num} if den is None else {"n": num, "d": den}
        m = self._ac(exprs, freqs, sources=sources, **deck)
        theirs = m["n"] * sign if den is None else m["n"] / m["d"]
        scale = float(np.max(np.abs(theirs))) or 1.0
        errs = np.abs(mine - theirs) / np.maximum(np.abs(theirs), 1e-6 * scale)
        worst = float(np.max(errs))
        at = freqs[int(np.argmax(errs))]
        return Check(q, "expression in s", f"{len(freqs)} frequencies", worst,
                     "ok" if worst <= self.tol else "MISMATCH",
                     f"compared H(j2πf) from 0.1 Hz to 100 MHz; worst at {fmt_si(at, 'Hz')}")

    # ------------------------------------------------------------------
    # Transfer-function helpers (same rules as the solver)
    # ------------------------------------------------------------------
    def _driver(self, inp: str):
        comp = next((c for c in self.indep if c.id == inp), None)
        if comp:
            return comp
        touching = [c for c in self.indep if inp in c.nodes]
        if len(touching) == 1:
            return touching[0]
        if len(self.indep) == 1:
            return self.indep[0]
        raise _Skip("cannot tell which source is the input")

    def _default_io(self) -> Tuple[str, str]:
        for q in self.nl.query:
            m = _Q.match(q.replace(" ", ""))
            if m and m.group(1) == "H" and m.group(3):
                return m.group(2), m.group(3)
        out = next((n for n in OUT_NAMES if n in self.nodes), None)
        if out is None:
            vq = [m.group(2) for q in self.nl.query if (m := _Q.match(q)) and m.group(1) == "V" and not m.group(3)]
            out = vq[0] if vq else None
        if out is None or len(self.indep) != 1:
            raise _Skip("cannot tell which transfer function the solver used")
        return out, self.indep[0].id

    def _transfer_check(self, out: str, inp: str, label: Optional[str] = None) -> Check:
        key = f"H({out},{inp})"
        if key in self._hchecks:
            return self._hchecks[key]
        sym = self.res.symbolic.get(key)
        if sym is None:
            raise _Skip(f"the solver did not record {key}")
        drv = self._driver(inp)
        den = vexpr(inp) if inp in self.nodes else None
        chk = self._sdomain(label or key, sym, vexpr(out), den, self._driven(drv.id))
        self._hchecks[key] = chk
        return chk

    # ------------------------------------------------------------------
    # Per-query checks
    # ------------------------------------------------------------------
    def _check_query(self, q: str) -> Check:
        m = _Q.match(q)
        kind, a, b = (m.group(1), m.group(2), m.group(3)) if m else (q, None, None)
        unit = UNITS.get(kind, "")
        nat = self._natural(self.mode)

        if a in getattr(self, "bjts", {}):                # IC(Q1), VCE(Q1), I(Q1), P(Q1) ...
            if kind == "I":
                unit = "A"
            return self._bjt_query(q, kind, a, unit, nat)

        if a in getattr(self, "fets", {}):                # ID(M1), VDS(M1), I(M1), P(M1) ...
            if kind == "I":
                unit = "A"
            return self._fet_query(q, kind, a, unit, nat)

        if kind in ("V", "Vth"):
            return self._value(q, vexpr(a, b or GROUND), unit, nat)

        if kind == "I":
            c = next(x for x in self.nl.components if x.id == a)
            sign = -1.0 if c.type.value in ("V", "E", "H", "OPAMP") else 1.0   # solver reports current DELIVERED
            if a in self.forward_ids:            # a diode: forward current (or I_Z in breakdown), like a resistor
                sign = 1.0
            return self._value(q, f"i(V_sense_{a})", unit, nat, sign=sign, sense=[a])

        if kind == "P":
            c = next(x for x in self.nl.components if x.id == a)
            if a in self.forward_ids and any(x.id == f"{a}_r" for x in self.nl.components):
                raise _Skip("power of a diode with series resistance is covered by its I and V checks")
            v = vexpr(c.nodes[0], c.nodes[1])
            if self.mode == "dc":
                return self._value(q, f"{v}*i(V_sense_{a})", unit, nat, sense=[a])
            if self.freq is None:
                raise _Skip("average AC power needs a frequency")
            mm = self._ac({"v": v, "i": f"i(V_sense_{a})"}, [self.freq], sources=nat, sense=[a])
            p = 0.5 * (mm["v"][0] * np.conj(mm["i"][0])).real
            return self._compare(q, self._solver_value(q), complex(p), unit)

        if kind in ("Rth", "Zth", "Z", "Rn"):
            return self._value(q, vexpr(a, b), unit, self._zeroed(), extra=[f"I_test {b} {a} DC 1 AC 1"])

        if kind in ("Isc", "In"):
            return self._value(q, "i(V_short)", unit, nat, extra=[f"V_short {a} {b} DC 0 AC 0"])

        if kind in ("H", "gain"):
            out, inp = (a, b) if kind == "H" else self._default_io()
            if self.res.symbolic.get(f"H({out},{inp})") is not None and sp.sympify(
                    self.res.symbolic[f"H({out},{inp})"]).has(_s):
                chk = self._transfer_check(out, inp, label=q)
                return Check(q, chk.solver, chk.spice, chk.rel_err, chk.status, chk.note)
            drv = self._driver(inp)
            exprs = {"n": vexpr(out)}
            if inp in self.nodes:
                exprs["d"] = vexpr(inp)
            mm = self._dc(exprs, sources=self._driven(drv.id))
            h = mm["n"] / mm["d"] if "d" in mm else mm["n"]
            return self._compare(q, self._solver_value(q), complex(h), unit)

        if kind == "bandwidth":
            # a low-pass bandwidth IS a -3 dB point, so ngspice can check it directly like fc
            # (a band-pass BW = f_H - f_L is not, and falls through to the algebra check on H)
            try:
                chk = self._check_fc(q)
                if chk.status == "ok":
                    return Check(q, chk.solver, chk.spice, chk.rel_err, chk.status, "low-pass: BW = fc, " + chk.note)
            except _Skip:
                pass

        if kind in DERIVED:
            out, inp = self._default_io()
            chk = self._transfer_check(out, inp)
            note = (f"exact algebra on H({out},{inp}), which matches ngspice at {chk.spice}"
                    if chk.status == "ok" else f"H({out},{inp}) does NOT match ngspice")
            return Check(q, self.res.answers.get(q, "-"), f"via H({out},{inp})", chk.rel_err, chk.status, note)

        if kind == "fc":
            return self._check_fc(q)

        if kind == "tau":
            storage = [c for c in self.nl.components if c.type.value in ("C", "L")]
            if len(storage) != 1:
                raise _Skip("not a first-order circuit")
            e = storage[0]
            n0, n1 = e.nodes
            rth = self._dc({"r": vexpr(n0, n1)}, sources=self._zeroed(), removed=[e.id],
                           extra=[f"I_test {n1} {n0} DC 1 AC 1"])["r"]
            val = parse_value(e.value)
            tau = rth * val if e.type.value == "C" else val / rth
            return self._compare(q, self._solver_value(q), complex(tau), unit,
                                 note=f"R seen by {e.id} from ngspice = {fmt_si(rth, 'Ω')}")

        if kind == "Rin":
            inp = None
            for x in self.nl.query:
                mx = _Q.match(x.replace(" ", ""))
                if mx and mx.group(1) == "H" and mx.group(3):
                    inp = mx.group(3)
            drv = self._driver(inp) if inp else (self.indep[0] if len(self.indep) == 1 else None)
            if drv is None:
                raise _Skip("cannot tell which source is the input")
            n0, n1 = drv.nodes
            return self._value(q, vexpr(n0, n1), unit, self._zeroed(), removed=[drv.id],
                               extra=[f"I_test {n1} {n0} DC 1 AC 1"])

        if kind == "Rout":
            out, _ = self._default_io()
            return self._value(q, vexpr(out), unit, self._zeroed(), extra=[f"I_test 0 {out} DC 1 AC 1"])

        raise _Skip(f"no ngspice check for '{kind}' yet")

    @staticmethod
    def _amp_input(nl: Netlist, src: Component) -> Optional[str]:
        """The node where the signal actually enters the amplifier: follow any chain of series resistors
        (the source's own resistance) away from the source."""
        hot, node, dropped = src.nodes[0], None, set()
        for _ in range(4):
            at = [c for c in nl.components if hot in c.nodes and c.id != src.id and c.id not in dropped]
            if len(at) != 1 or at[0].type.value != "R":
                break
            r = at[0]
            dropped.add(r.id)
            hot = r.nodes[1] if r.nodes[0] == hot else r.nodes[0]
            node = hot
        return node

    def _bjt_query(self, q: str, kind: str, a: str, unit: str, nat) -> Check:
        if True:
            ex, sense = self._bjt_exprs(a)
            if kind in ("bias", "region"):            # the Q-point: I_C and V_CE together
                need = {k: e for k, e in ex.items() if k in ("IC", "VCE") and e != "0"}
                got = self._dc(need, sources=nat, sense=sense) if need else {}
                ic, vce = got.get("IC", 0.0), got["VCE"] if "VCE" in got else 0.0
                c1 = self._compare(q, self.res.numeric.get(f"{q}.IC"), complex(ic), "A")
                c2 = self._compare(q, self.res.numeric.get(f"{q}.VCE"), complex(vce), "V")
                worst = c1 if (c1.rel_err or 0) >= (c2.rel_err or 0) else c2
                status = "MISMATCH" if "MISMATCH" in (c1.status, c2.status) else worst.status
                return Check(q, f"I_C = {c1.solver}, V_CE = {c2.solver}", f"I_C = {c1.spice}, V_CE = {c2.spice}",
                             worst.rel_err, status, "Q-point")
            if kind == "P":
                expr = f"({ex['VCE']})*({ex['IC']}) + ({ex['VBE']})*({ex['IB']})"
            else:
                key = {"I": "IC", "IC": "IC", "IB": "IB", "VCE": "VCE", "VEC": "VCE", "VBE": "VBE",
                       "VEB": "VBE"}.get(kind)
                if kind == "IE":
                    expr = f"({ex['IB']}) + ({ex['IC']})"
                elif key:
                    expr = ex[key]
                else:
                    raise _Skip(f"no ngspice check for '{kind}' yet")
            if expr.strip() in ("0", "(0) + (0)"):
                return self._compare(q, self._solver_value(q), complex(0.0), unit)
            return self._value(q, expr, unit, nat, sense=sense)

    def _check_fc(self, q: str) -> Check:
        out, inp = self._default_io()
        hchk = self._transfer_check(out, inp)
        if hchk.status != "ok":
            return Check(q, self.res.answers.get(q, "-"), "-", hchk.rel_err, "MISMATCH",
                         f"H({out},{inp}) does not match ngspice, so fc is not trusted")
        cuts = [self.res.numeric.get(q)] + [self.res.numeric[k] for k in sorted(self.res.numeric)
                                            if k.startswith(f"{q}_")]
        cuts = [x for x in cuts if x is not None]
        if not cuts:
            raise _Skip("no numeric fc to check (symbolic answer)")
        drv = self._driver(inp)
        src = self._driven(drv.id)
        ratio = vexpr(out) if inp not in self.nodes else f"{vexpr(out)}/{vexpr(inp)}"

        # |H|max: coarse sweep, then a fine linear sweep around the peak (sharp resonances)
        f, h = self._sweep(ratio, 1e-3, 1e12, 100, lin=False, sources=src)
        mag = np.abs(h)
        i = int(np.argmax(mag))
        lo, hi = f[max(i - 1, 0)], f[min(i + 1, len(f) - 1)]
        hmax = float(mag[i])
        if hi > lo:
            _, h2 = self._sweep(ratio, lo, hi, 2001, lin=True, sources=src)
            hmax = max(hmax, float(np.max(np.abs(h2))))

        # Reference gain, decided from ngspice's own sweep (same definition as the solver, found independently):
        # starts in its pass band and ends far below -> low-pass, measured from |H(0)| even if it peaks;
        # the mirror case -> high-pass, from |H(∞)|; anything else (band-pass, notch) -> from the peak.
        h_lo, h_hi = float(mag[0]), float(mag[-1])
        if h_lo >= hmax / math.sqrt(2) and h_hi < h_lo / math.sqrt(2):
            ref, name = h_lo, "|H(0)|"
        elif h_hi >= hmax / math.sqrt(2) and h_lo < h_hi / math.sqrt(2):
            ref, name = h_hi, "|H(∞)|"
        else:
            ref, name = hmax, "|H|max"

        mm = self._ac({"n": vexpr(out)} | ({"d": vexpr(inp)} if inp in self.nodes else {}), cuts, sources=src)
        hc = mm["n"] / mm["d"] if "d" in mm else mm["n"]
        ratios = np.abs(hc) / ref
        worst = float(np.max(np.abs(ratios * math.sqrt(2) - 1)))
        return Check(q, self.res.answers.get(q, "-"),
                     f"|H(fc)|/{name} = " + ", ".join(fmt_num(r, 5) for r in ratios),
                     worst, "ok" if worst <= self.tol_sweep else "MISMATCH",
                     "must equal 1/√2 = 0.70711 at each cutoff")


def verify(nl: Netlist, res: SolveResult, verifier: Optional[Verifier] = None) -> VerifyReport:
    return (verifier or Verifier()).verify(nl, res)