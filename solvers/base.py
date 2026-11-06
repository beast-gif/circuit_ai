"""
Common solver interface + number formatting used by every solver.

Every solver (dc_ac now; opamp, filters, transistor, oscillator later) implements:
    supports(netlist) -> (True, "")  or  (False, "reason")
    solve(netlist, conventions) -> SolveResult
"""
from __future__ import annotations

import cmath
import math
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import sympy as sp

from schemas import Netlist, SolveResult


class SolverError(Exception):
    """The circuit or a query could not be solved (singular system, undefined node, bad query)."""


class NotSupportedError(SolverError):
    """No solver handles this circuit yet (e.g. op-amps before Phase 2)."""


class BaseSolver(ABC):
    name: str = "base"
    method: str = ""

    @abstractmethod
    def supports(self, nl: Netlist) -> Tuple[bool, str]:
        ...

    @abstractmethod
    def solve(self, nl: Netlist, conventions: Dict[str, Any]) -> SolveResult:
        ...


# ----------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------
def sig_figs(conventions: Optional[Dict[str, Any]]) -> int:
    return int(((conventions or {}).get("output") or {}).get("significant_figures", 4))


_PREFIXES = [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""),
             (1e-3, "m"), (1e-6, "µ"), (1e-9, "n"), (1e-12, "p"), (1e-15, "f")]
PLAIN_UNITS = {"", "rad/s", "°", "dB"}      # no SI prefix for these


def fmt_num(x: float, sig: int = 4) -> str:
    """6.66667 -> '6.667',  1.5e-05 -> '1.5e-5'."""
    x = float(x)
    if x == 0:
        return "0"
    txt = f"{x:.{sig}g}"
    if "e" in txt:
        mant, exp = txt.split("e")
        txt = f"{mant}e{int(exp)}"
    return txt


def fmt_si(x: float, unit: str = "", sig: int = 4) -> str:
    """0.003333, 'A' -> '3.333 mA';  4000, 'Ω' -> '4 kΩ';  159.15, 'Hz' -> '159.2 Hz'."""
    x = float(x)
    if unit in PLAIN_UNITS or not math.isfinite(x):
        return f"{fmt_num(x, sig)} {unit}".strip()
    if x == 0:
        return f"0 {unit}".strip()
    xr = float(f"{x:.{sig}g}")              # round first so 999.96 becomes 1 k, not 1000
    a = abs(xr)
    for factor, prefix in _PREFIXES:
        if a >= factor:
            return f"{fmt_num(xr / factor, sig)} {prefix}{unit}"
    return f"{fmt_num(xr, sig)} {unit}"


def fmt_complex(z: complex, unit: str = "", sig: int = 4, rect: bool = False) -> str:
    """Polar '157.2 mV ∠ -80.96°'; rect=True (impedances) gives 'a + jb Ω (= polar)'."""
    mag, ph = abs(z), math.degrees(cmath.phase(z))
    polar = f"{fmt_si(mag, unit, sig)} ∠ {fmt_num(ph, sig)}°"
    if not rect:
        return polar
    sign = "+" if z.imag >= 0 else "-"
    return f"{fmt_si(z.real, unit, sig)} {sign} j{fmt_si(abs(z.imag), unit, sig)}  (= {polar})"


def pretty(expr: Any, sig: int = 4) -> str:
    """Readable SymPy expression: decimals to `sig` figures, ** -> ^, I -> j."""
    e = sp.sympify(expr)
    if e.atoms(sp.Float) or any(not n.is_Integer for n in e.atoms(sp.Rational)):
        e = sp.N(e, sig)
    txt = str(e)
    txt = re.sub(r"(\.\d*?[1-9])0+(?=\D|$)", r"\1", txt)   # 0.004700 -> 0.0047
    txt = re.sub(r"\.0+(?=\D|$)", "", txt)                  # 1000.00  -> 1000
    txt = txt.replace("**", "^")
    txt = re.sub(r"\bI\b", "j", txt)
    return txt
