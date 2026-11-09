"""
Solver registry. The first solver whose supports() accepts the netlist is used.
Later phases add OpAmpSolver, TransistorSolver, OscillatorSolver to SOLVERS.
"""
from typing import Dict, List

from schemas import Netlist, SolveResult
from solvers.base import BaseSolver, NotSupportedError, SolverError
from solvers.dc_ac import DCACSolver
from solvers.diode import DiodeSolver

SOLVERS: List[BaseSolver] = [DiodeSolver(), DCACSolver()]


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