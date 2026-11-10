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

# the oscillator solver comes before the comparator one: both see an op-amp without negative
# feedback, and only the oscillator's source-free test tells them apart
SOLVERS: List[BaseSolver] = [TransistorSolver(), DiodeSolver(), OscillatorSolver(),
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