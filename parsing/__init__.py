"""
Solver registry. The first solver whose supports() accepts the netlist is used.
Later phases add more solvers to SOLVERS.
"""
from typing import Dict, List

from schemas import Netlist, SolveResult
from solvers.base import BaseSolver, NotSupportedError, SolverError
from solvers.bjt import TransistorSolver
from solvers.comparator import ComparatorSolver
from solvers.dc_ac import DCACSolver
from solvers.diode import DiodeSolver
from solvers.oscillator import OscillatorSolver
from solvers.relaxation import RelaxationSolver

# The oscillator solver goes FIRST. An LC oscillator's netlist contains a transistor, which the transistor
# solver would otherwise grab and try to bias -- but an oscillator netlist is the AC equivalent circuit and
# has no supply, so that always fails. Its supports() is narrow (oscillator queries, or oscillator wording
# with no source), so amplifier questions still fall through to the transistor solver. It also comes before
# the comparator solver: both see an op-amp without negative feedback, and only the source-free test tells
# them apart.
# The relaxation solver comes before the oscillator one: an astable multivibrator is also a
# source-free op-amp circuit asked about with "fosc", but it is not a linear feedback loop and
# det A(s) = 0 says nothing useful about it. Its supports() needs the astable's shape: positive
# feedback to the + input AND the timing capacitor on the − input.
SOLVERS: List[BaseSolver] = [RelaxationSolver(), OscillatorSolver(), TransistorSolver(), DiodeSolver(),
                            ComparatorSolver(), DCACSolver()]


def get_solver(nl: Netlist) -> BaseSolver:
    reasons = []
    for sv in SOLVERS:
        ok, why = sv.supports(nl)
        if ok:
            return sv
        if why:
            reasons.append(why)
    raise NotSupportedError("; ".join(reasons) or "No solver for this circuit yet")


def solve(nl: Netlist, conventions: Dict) -> SolveResult:
    return get_solver(nl).solve(nl, conventions)


__all__ = ["BaseSolver", "SolverError", "NotSupportedError", "SOLVERS", "get_solver", "solve"]