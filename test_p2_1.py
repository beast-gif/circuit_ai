"""
Phase 2, Step 1 checks: the ideal op-amp in the MNA engine (plus one non-ideal check since P2-3).

    python test_p2_1.py           # hand-written op-amp circuits (free, no LLM)
    python test_p2_1.py --steps   # also print the key steps of each circuit

Every expected value is worked by hand in the comment above it.
"""
import cmath
import math
import sys

import sympy as sp

from classifier import classify
from llm_client import load_conventions
from schemas import Component, Netlist
from solvers import NotSupportedError, SolverError, get_solver

CONV = load_conventions()
SHOW = "--steps" in sys.argv
s = sp.Symbol("s")


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"\n       {detail}" if detail and not cond else ""))
    return cond


def C(id, type, nodes, value=None, **kw):
    return Component(id=id, type=type, nodes=nodes, value=value, **kw)


def U(id, out, plus, minus, **kw):
    """Ideal op-amp: nodes = [out, in+, in-]."""
    return Component(id=id, type="OPAMP", nodes=[out, plus, minus], model="ideal", **kw)


def net(components, query, analysis="dc", question=""):
    return Netlist(analysis=analysis, query=query, components=components, question=question)


def close(got, want, rel=1e-3):
    return got is not None and abs(got - want) <= rel * max(1e-12, abs(want))


def show(res):
    for st in res.trace.key_steps():
        print(f"      {st.step:>2}. {st.action}" + (f"\n          {st.equation}" if st.equation else "")
              + (f"\n          => {st.result}" if st.result else ""))


CASES = [
    # Inverting: V- = V+ = 0 (virtual ground). KCL at n1: (0 - 1)/1k + (0 - Vout)/10k = 0 -> Vout = -10.
    # Rin seen by the source = R1 = 1k (n1 is at 0 V). Rout of an ideal op-amp amplifier = 0.
    ("Inverting amplifier (gain -Rf/R1 = -10)",
     net([C("V1", "V", ["in", "0"], "1"), C("R1", "R", ["in", "n1"], "1k"), C("R2", "R", ["n1", "out"], "10k"),
          U("U1", "out", "0", "n1")], ["V(out)", "H(out,in)", "Rin", "Rout"]),
     {"V(out)": -10.0, "H(out,in)": -10.0, "Rin": 1000.0, "Rout": 0.0}),

    # Non-inverting: V- = V+ = 1. KCL at n1: 1/1k + (1 - Vout)/9k = 0 -> Vout = 10 (gain 1 + 9k/1k).
    ("Non-inverting amplifier (gain 1 + Rf/R1 = 10)",
     net([C("V1", "V", ["in", "0"], "1"), U("U1", "out", "in", "n1"), C("R1", "R", ["n1", "0"], "1k"),
          C("R2", "R", ["n1", "out"], "9k")], ["V(out)"]),
     {"V(out)": 10.0}),

    # Voltage follower: in- tied to out, so Vout = V+ = 2.5 regardless of the 1k load.
    ("Voltage follower",
     net([C("V1", "V", ["in", "0"], "2.5"), U("U1", "out", "in", "out"), C("RL", "R", ["out", "0"], "1k")],
         ["V(out)"]),
     {"V(out)": 2.5}),

    # Summing: Vout = -Rf (V1/R1 + V2/R2) = -10k (1/10k + 2/10k) = -3.
    ("Inverting summer",
     net([C("V1", "V", ["a", "0"], "1"), C("V2", "V", ["b", "0"], "2"), C("R1", "R", ["a", "n"], "10k"),
          C("R2", "R", ["b", "n"], "10k"), C("Rf", "R", ["n", "out"], "10k"), U("U1", "out", "0", "n")],
         ["V(out)"]),
     {"V(out)": -3.0}),

    # Difference amp, all 10k: V+ = 3 * 10k/20k = 1.5 = V-.  KCL at n: (1.5 - 1)/10k + (1.5 - Vout)/10k = 0
    # -> Vout = 2 = Va - Vb.
    ("Difference amplifier (Vout = Va - Vb)",
     net([C("V1", "V", ["a", "0"], "3"), C("V2", "V", ["b", "0"], "1"), C("R1", "R", ["b", "n"], "10k"),
          C("R2", "R", ["n", "out"], "10k"), C("R3", "R", ["a", "p"], "10k"), C("R4", "R", ["p", "0"], "10k"),
          U("U1", "out", "p", "n")], ["V(out)"]),
     {"V(out)": 2.0}),

    # Two inverting stages: -2 then -5 -> overall +10, so Vout = 10 for 1 V in.
    ("Two cascaded inverting stages (+10)",
     net([C("V1", "V", ["in", "0"], "1"), C("R1", "R", ["in", "a"], "1k"), C("R2", "R", ["a", "o1"], "2k"),
          U("U1", "o1", "0", "a"), C("R3", "R", ["o1", "b"], "1k"), C("R4", "R", ["b", "out"], "5k"),
          U("U2", "out", "0", "b")], ["V(out)", "V(o1)"]),
     {"V(out)": 10.0, "V(o1)": -2.0}),

    # Inverting low-pass (C across Rf): H = -(Rf/R1) / (1 + s Rf C) = -10 / (1 + 1e-4 s)
    # -> DC gain -10, fc = 1/(2 pi Rf C) = 1/(2 pi 1e-4) = 1591.5 Hz.
    ("Active first-order low-pass",
     net([C("V1", "V", ["in", "0"], ac="1"), C("R1", "R", ["in", "n"], "1k"), C("R2", "R", ["n", "out"], "10k"),
          C("C1", "C", ["n", "out"], "10n"), U("U1", "out", "0", "n")], ["H(out,in)", "fc"], analysis="ac"),
     {"H(out,in).dc_gain": -10.0, "fc": 1591.55}),

    # Op-amp amplifier the parser labelled "small_signal": solved like DC (gain -4.7).
    ("Inverting amp labelled small_signal (gain -Rf/R1 = -4.7)",
     net([C("V1", "V", ["in", "0"], "1"), C("R1", "R", ["in", "n1"], "10k"), C("R2", "R", ["n1", "out"], "47k"),
          U("U1", "out", "0", "n1")], ["gain"], analysis="small_signal"),
     {"gain": -4.7}),
]


def solver_checks() -> bool:
    ok = True
    for name, nl, expected in CASES:
        print(f"\n--- {name}")
        try:
            res = get_solver(nl).solve(nl, CONV)
        except SolverError as e:
            ok &= check(name, False, f"solver error: {e}")
            continue
        for q, a in res.answers.items():
            print(f"     {q:<10} = {a}")
        if SHOW:
            show(res)
        for key, want in expected.items():
            ok &= check(f"{name}: {key} = {want:g}", close(res.numeric.get(key), want), f"got {res.numeric.get(key)}")
    return ok


def special_checks() -> bool:
    ok = True
    print()

    # Integrator: H = -1/(s R C) = -1/(1e-4 s) = -10000/s.  |H| = 1 at f = 1/(2 pi R C) = 1591.5 Hz.
    integ = net([C("V1", "V", ["in", "0"], ac="1"), C("R1", "R", ["in", "n"], "10k"), C("C1", "C", ["n", "out"], "10n"),
                 U("U1", "out", "0", "n")], ["H(out,in)", "poles"], analysis="ac")
    res = get_solver(integ).solve(integ, CONV)
    print(f"     integrator: {res.answers}")
    H = sp.sympify(res.symbolic.get("H(out,in)", "0"))
    ok &= check("integrator H(s)·s = -10000", sp.simplify(H * s + 10000) == 0, str(H))
    mag = abs(complex(H.subs(s, 2j * math.pi * 1591.549)))
    ok &= check("integrator |H| = 1 at 1591.5 Hz", abs(mag - 1) < 1e-3, f"{mag}")
    ok &= check("integrator pole at s = 0", close(res.numeric.get("poles[1].re"), 0.0) or
                abs(res.numeric.get("poles[1].re") or 1) < 1e-9, str(res.numeric))

    # Symbolic inverting amp: H = -Rf/R1
    sym = net([C("V1", "V", ["in", "0"], "Vs"), C("R1", "R", ["in", "n1"], "R1"), C("R2", "R", ["n1", "out"], "Rf"),
               U("U1", "out", "0", "n1")], ["H(out,in)"])
    res = get_solver(sym).solve(sym, CONV)
    print(f"     symbolic: {res.answers}")
    Hs = res.symbolic.get("H(out,in)")
    good = False
    if Hs:
        val = sp.sympify(Hs).subs({sp.Symbol("R1"): 2, sp.Symbol("Rf"): 6})
        good = abs(float(val) + 3) < 1e-12
    ok &= check("symbolic inverting gain = -Rf/R1", good, str(Hs) + " " + str(res.answers))

    # The key steps show the virtual short, the way a student writes it
    nl = CASES[0][1]
    res = get_solver(nl).solve(nl, CONV)
    vs = [st for st in res.trace.key_steps() if st.action.startswith("Ideal op-amp U1: virtual short")]
    ok &= check("key steps include 'virtual short V− = V+' with V_n1 = 0",
                bool(vs) and "V_n1 = 0" in (vs[0].equation or ""), str([st.action for st in res.trace.key_steps()]))
    kcl_out = [st for st in res.trace.key_steps() if st.action.startswith("KCL at node out")]
    ok &= check("KCL at the op-amp output is hidden in key steps (only gives the output current)", not kcl_out)
    if SHOW:
        show(res)

    # Non-ideal op-amp (solved since P2-3): A0 = 1e5 -> Vout = -10 / (1 + 11/1e5) = -9.99890012
    nonideal = net([C("V1", "V", ["in", "0"], "1"), C("R1", "R", ["in", "n1"], "1k"),
                    C("R2", "R", ["n1", "out"], "10k"),
                    Component(id="U1", type="OPAMP", nodes=["out", "0", "n1"], params={"A0": 100000})], ["V(out)"])
    try:
        res = get_solver(nonideal).solve(nonideal, CONV)
        ok &= check("op-amp with A0 = 1e5 -> solved, Vout = -9.99890", close(res.numeric.get("V(out)"),
                    -10 / (1 + 11 / 1e5), 1e-6), str(res.answers))
    except (NotSupportedError, SolverError) as e:
        ok &= check("op-amp with A0 = 1e5 -> solved, Vout = -9.99890", False, str(e))

    # Floating op-amp input (in+ connected to nothing) -> clear error, not a wrong number
    floating = net([C("V1", "V", ["in", "0"], "1"), C("R1", "R", ["in", "n1"], "1k"),
                    C("R2", "R", ["n1", "out"], "10k"), U("U1", "out", "p", "n1")], ["V(out)"])
    try:
        get_solver(floating).solve(floating, CONV)
        ok &= check("floating in+ -> SolverError", False)
    except SolverError as e:
        ok &= check("floating in+ -> SolverError", "undefined" in str(e), str(e))

    # Classifier now routes op-amps to the solver
    c = classify(net(CASES[0][1].components, ["V(out)"], question="Find Vout of the inverting op-amp amplifier."))
    ok &= check("classifier: op-amp -> topic opamp, solver dc_ac", c.topic == "opamp" and c.solver == "dc_ac", str(c))
    return ok


if __name__ == "__main__":
    all_ok = solver_checks()
    all_ok &= special_checks()
    print("\nPHASE 2 STEP 1:", "ALL PASSED" if all_ok else "SOME CHECKS FAILED")