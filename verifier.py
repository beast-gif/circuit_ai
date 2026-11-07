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

from llm_client import load_config
from schemas import AnalysisType, Netlist, SolveResult, is_symbolic, parse_value
from solvers.base import fmt_complex, fmt_num, fmt_si

GROUND = "0"
INDEP = {"V", "I"}
LINEAR = {"R", "C", "L", "V", "I", "E", "G", "F", "H"}
OUT_NAMES = ("out", "vout", "Vout", "o", "output")
SAMPLE_FREQS = [0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
UNITS = {"V": "V", "Vth": "V", "I": "A", "Isc": "A", "In": "A", "P": "W",
         "Rth": "Ω", "Zth": "Ω", "Z": "Ω", "Rn": "Ω", "Rin": "Ω", "Rout": "Ω",
         "tau": "s", "fc": "Hz", "f0": "Hz", "bandwidth": "Hz"}
DERIVED = {"poles", "zeros", "f0", "Qfactor", "bandwidth"}
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
               sense: Sequence[str] = (), extra: Sequence[str] = ()) -> str:
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
        elif t in ("F", "H"):
            lines.append(f"{name} {nodes[0]} {nodes[1]} {ctrl_name(c.control.strip())} {_num(parse_value(c.value))}")
        else:
            raise _Skip(f"{c.id}: type {t} is not checked yet")
    lines += list(extra)
    # 1 TΩ from every node to ground: keeps ngspice's DC bias point solvable when a node is reached
    # only through capacitors (needed before an AC run). Effect on kΩ-MΩ circuits: < 1e-6 relative.
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

        statuses = {c.status for c in rep.checks}
        rep.verified = False if "MISMATCH" in statuses else (True if "ok" in statuses else None)
        res.verified = rep.verified
        for c in rep.checks:
            if c.status == "MISMATCH":
                res.verification_notes.append(f"{c.query}: solver {c.solver} vs ngspice {c.spice}")
            elif c.status == "skipped":
                res.verification_notes.append(f"{c.query}: not checked ({c.note})")
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
        vals, _ = self.run(build_deck(self.nl, **deck), ctl.lines, list(keys.values()))
        return {label: vals[k] for label, k in keys.items()}

    def _ac(self, exprs: Dict[str, str], freqs: Sequence[float], **deck) -> Dict[str, np.ndarray]:
        ctl = _Ctl()
        keys: Dict[str, List[Tuple[str, str]]] = {label: [] for label in exprs}
        for f in freqs:
            ctl.cmd(f"ac lin 1 {_num(f)} {_num(f)}")
            for label, e in exprs.items():
                keys[label].append((ctl.probe(f"real({e})"), ctl.probe(f"imag({e})")))
        flat = [k for pairs in keys.values() for pair in pairs for k in pair]
        vals, _ = self.run(build_deck(self.nl, **deck), ctl.lines, flat)
        return {label: np.array([complex(vals[r], vals[i]) for r, i in pairs]) for label, pairs in keys.items()}

    def _sweep(self, ratio: str, start: float, stop: float, points: int, lin: bool, **deck):
        ctl = _Ctl()
        ctl.cmd(f"ac {'lin' if lin else 'dec'} {points} {_num(start)} {_num(stop)}")
        ctl.cmd(f"let hr = real({ratio})")
        ctl.cmd(f"let hi = imag({ratio})")
        ctl.cmd("wrdata sweep.txt hr hi")
        _, data = self.run(build_deck(self.nl, **deck), ctl.lines, files=["sweep.txt"])
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

        if kind in ("V", "Vth"):
            return self._value(q, vexpr(a, b or GROUND), unit, nat)

        if kind == "I":
            c = next(x for x in self.nl.components if x.id == a)
            sign = -1.0 if c.type.value in ("V", "E", "H") else 1.0   # solver reports current DELIVERED
            return self._value(q, f"i(V_sense_{a})", unit, nat, sign=sign, sense=[a])

        if kind == "P":
            c = next(x for x in self.nl.components if x.id == a)
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

        mm = self._ac({"n": vexpr(out)} | ({"d": vexpr(inp)} if inp in self.nodes else {}), cuts, sources=src)
        hc = mm["n"] / mm["d"] if "d" in mm else mm["n"]
        ratios = np.abs(hc) / hmax
        worst = float(np.max(np.abs(ratios * math.sqrt(2) - 1)))
        return Check(q, self.res.answers.get(q, "-"),
                     "|H(fc)|/|H|max = " + ", ".join(fmt_num(r, 5) for r in ratios),
                     worst, "ok" if worst <= self.tol_sweep else "MISMATCH",
                     "must equal 1/√2 = 0.70711 at each cutoff")


def verify(nl: Netlist, res: SolveResult, verifier: Optional[Verifier] = None) -> VerifyReport:
    return (verifier or Verifier()).verify(nl, res)