"""
Step 3 checks.

    python test_step3.py           # solver + classifier on hand-written circuits (free, no LLM)
    python test_step3.py --steps   # also prints the key steps of every case
    python test_step3.py --full    # prints ALL steps (what "show complete" will use)
    python test_step3.py --live    # also: real questions -> parse -> classify -> solve (Velona)
"""
import sys

import sympy as sp

from classifier import classify
from llm_client import load_conventions
from schemas import Component, Netlist
from solvers import NotSupportedError, SolverError, get_solver

CONV = load_conventions()
SHOW_STEPS = "--steps" in sys.argv or "--full" in sys.argv
FULL = "--full" in sys.argv


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"\n       {detail}" if detail and not cond else ""))
    return cond


def C(id, type, nodes, value=None, **kw):
    return Component(id=id, type=type, nodes=nodes, value=value, **kw)


def net(components, query, analysis="dc", question="", frequency=None):
    return Netlist(analysis=analysis, query=query, components=components, question=question, frequency=frequency)


def show(res):
    for st in (res.trace.all_steps() if FULL else res.trace.key_steps()):
        line = f"     {st.step:>2}. [{st.level}] {st.action}"
        if st.equation:
            line += f"\n           {st.equation}"
        if st.result:
            line += f"\n           => {st.result}"
        print(line)


def close(got, want, rel=1e-3):
    return got is not None and abs(got - want) <= rel * max(1e-12, abs(want))


# ----------------------------------------------------------------------
# Hand-checked circuits: (name, netlist, {numeric key: expected})
# ----------------------------------------------------------------------
CASES = [
    ("Voltage divider",
     net([C("V1", "V", ["in", "0"], "10"), C("R1", "R", ["in", "out"], "1k"), C("R2", "R", ["out", "0"], "2k")],
         ["V(out)", "I(R1)", "P(R2)", "I(V1)", "P(V1)"]),
     # Vout = 10·2k/3k ; I = 10/3k ; P(R2) = I²·2k ; source delivers 10 V × 3.333 mA
     {"V(out)": 6.6667, "I(R1)": 3.3333e-3, "P(R2)": 0.022222, "I(V1)": 3.3333e-3, "P(V1)": -0.033333}),

    ("Series-parallel",
     net([C("V1", "V", ["n1", "0"], "12"), C("R1", "R", ["n1", "n2"], "4k"),
          C("R2", "R", ["n2", "0"], "6k"), C("R3", "R", ["n2", "0"], "12k")],
         ["I(R1)", "V(n2)", "I(R3)"]),
     # 6k ∥ 12k = 4k ; total 8k ; I = 1.5 mA ; V(n2) = 6 V ; I(R3) = 0.5 mA
     {"I(R1)": 1.5e-3, "V(n2)": 6.0, "I(R3)": 0.5e-3}),

    ("Thevenin / Norton",
     net([C("V1", "V", ["n1", "0"], "20"), C("R1", "R", ["n1", "a"], "5k"), C("R2", "R", ["a", "0"], "20k")],
         ["Vth(a,0)", "Rth(a,0)", "Isc(a,0)"]),
     # Vth = 20·20/25 = 16 V ; Rth = 5k ∥ 20k = 4k ; Isc = 4 mA
     {"Vth(a,0)": 16.0, "Rth(a,0)": 4000.0, "Isc(a,0)": 4e-3}),

    ("Floating voltage source",
     net([C("V1", "V", ["a", "0"], "10"), C("R1", "R", ["a", "b"], "1k"),
          C("V2", "V", ["b", "c"], "5"), C("R2", "R", ["c", "0"], "1k")],
         ["V(c)", "V(b)", "I(R1)"]),
     # loop: I = (10 - 5)/2k = 2.5 mA ; V(c) = 2.5 V ; V(b) = 7.5 V
     {"V(c)": 2.5, "V(b)": 7.5, "I(R1)": 2.5e-3}),

    ("VCCS (G)",
     net([C("I1", "I", ["0", "a"], "1m"), C("R1", "R", ["a", "0"], "1k"),
          C("G1", "G", ["b", "0", "a", "0"], "2m"), C("R2", "R", ["b", "0"], "1k")],
         ["V(a)", "V(b)"]),
     # V(a) = 1 mA·1k = 1 V ; G1 pulls 2 mA out of b -> V(b) = -2 V
     {"V(a)": 1.0, "V(b)": -2.0}),

    ("VCVS (E)",
     net([C("V1", "V", ["in", "0"], "1"), C("R1", "R", ["in", "x"], "1k"), C("R2", "R", ["x", "0"], "1k"),
          C("E1", "E", ["out", "0", "x", "0"], "10"), C("RL", "R", ["out", "0"], "1k")],
         ["V(out)"]),
     # V(x) = 0.5 V ; V(out) = 10 × 0.5 = 5 V
     {"V(out)": 5.0}),

    ("CCVS (H), SPICE sign",
     net([C("V1", "V", ["a", "0"], "10"), C("R1", "R", ["a", "0"], "2k"),
          C("H1", "H", ["b", "0"], "1k", control="V1"), C("R2", "R", ["b", "0"], "1k")],
         ["V(b)"]),
     # SPICE current of V1 (+ to - inside) = -5 mA ; V(b) = 1k × (-5 mA) = -5 V
     {"V(b)": -5.0}),

    ("DC: inductor short, capacitor open",
     net([C("V1", "V", ["a", "0"], "10"), C("L1", "L", ["a", "b"], "1m"),
          C("R1", "R", ["b", "0"], "1k"), C("C1", "C", ["b", "0"], "1u")],
         ["V(b)", "I(L1)"]),
     {"V(b)": 10.0, "I(L1)": 10e-3}),

    ("RC low-pass: H(s), pole, fc",
     net([C("V1", "V", ["in", "0"], ac="1"), C("R1", "R", ["in", "out"], "1k"), C("C1", "C", ["out", "0"], "1u")],
         ["H(out,in)", "poles", "fc"], analysis="ac"),
     # H = 1/(1 + sRC), RC = 1 ms -> pole -1000 rad/s -> fc = 1000/(2π) = 159.15 Hz
     {"H(out,in).dc_gain": 1.0, "poles[1].re": -1000.0, "fc": 159.155}),

    ("RC low-pass at 1 kHz (phasor)",
     net([C("V1", "V", ["in", "0"], ac="1"), C("R1", "R", ["in", "out"], "1k"), C("C1", "C", ["out", "0"], "1u")],
         ["V(out)"], analysis="ac", frequency="1k"),
     # ωRC = 6.2832 -> |H| = 1/√(1 + 39.48) = 0.15718 ; phase = -atan(6.2832) = -80.957°
     {"V(out)": 0.15718, "V(out).phase_deg": -80.957}),

    ("Series RLC band-pass",
     net([C("V1", "V", ["in", "0"], ac="1"), C("L1", "L", ["in", "n1"], "10m"),
          C("C1", "C", ["n1", "out"], "1u"), C("R1", "R", ["out", "0"], "10")],
         ["f0", "Qfactor", "bandwidth"], analysis="ac"),
     # ω0 = 1/√(LC) = 10⁴ rad/s -> 1591.5 Hz ; Q = (1/R)√(L/C) = 10 ; BW = f0/Q = 159.15 Hz
     {"f0": 1591.55, "Qfactor": 10.0, "bandwidth": 159.155}),

    ("RC time constant",
     net([C("V1", "V", ["in", "0"], "10"), C("R1", "R", ["in", "out"], "1k"),
          C("R2", "R", ["out", "0"], "1k"), C("C1", "C", ["out", "0"], "1u")],
         ["tau"], analysis="transient"),
     # C sees 1k ∥ 1k = 500 Ω -> τ = 500 × 1 µF = 0.5 ms
     {"tau": 5e-4}),
]


def solver_checks() -> bool:
    ok = True
    for name, nl, expected in CASES:
        try:
            res = get_solver(nl).solve(nl, CONV)
        except SolverError as e:
            ok &= check(name, False, f"solver error: {e}")
            continue
        print(f"\n--- {name}")
        for q, a in res.answers.items():
            print(f"     {q:<12} = {a}")
        if SHOW_STEPS:
            show(res)
        for key, want in expected.items():
            got = res.numeric.get(key)
            ok &= check(f"{name}: {key} ≈ {want:g}", close(got, want), f"got {got}")
    return ok


def symbolic_and_error_checks() -> bool:
    ok = True
    print("\n--- Symbolic divider (R1, R2, Vs)")
    nl = net([C("V1", "V", ["in", "0"], "Vs"), C("R1", "R", ["in", "out"], "R1"), C("R2", "R", ["out", "0"], "R2")],
             ["V(out)"])
    res = get_solver(nl).solve(nl, CONV)
    print(f"     V(out) = {res.answers.get('V(out)')}")
    sym = res.symbolic.get("V(out)")
    good = False
    if sym:
        val = sp.sympify(sym).subs({sp.Symbol("R1"): 1000, sp.Symbol("R2"): 2000, sp.Symbol("Vs"): 10})
        good = abs(float(val) - 20 / 3) < 1e-9
    ok &= check("symbolic V(out) = R2·Vs/(R1+R2)", good, f"got {sym}")

    # op-amp -> not supported until Phase 2
    opamp = net([C("V1", "V", ["in", "0"], "1"), C("R1", "R", ["in", "n1"], "1k"),
                 C("R2", "R", ["n1", "out"], "10k"), Component(id="U1", type="OPAMP", nodes=["out", "0", "n1"])],
                ["V(out)"], question="Find Vout of the inverting op-amp amplifier.")
    try:
        get_solver(opamp)
        ok &= check("op-amp -> NotSupportedError", False)
    except NotSupportedError as e:
        ok &= check("op-amp -> NotSupportedError", "Phase 2" in str(e), str(e))

    # node joined only through capacitors in DC -> undefined
    caps = net([C("V1", "V", ["a", "0"], "10"), C("R1", "R", ["a", "0"], "1k"),
                C("C1", "C", ["a", "x"], "1u"), C("C2", "C", ["x", "0"], "1u")], ["V(x)"])
    try:
        get_solver(caps).solve(caps, CONV)
        ok &= check("capacitor-only node in DC -> SolverError", False)
    except SolverError as e:
        ok &= check("capacitor-only node in DC -> SolverError", "undefined" in str(e), str(e))

    # one failing query must not hide the others
    rc = net([C("V1", "V", ["in", "0"], ac="1"), C("R1", "R", ["in", "out"], "1k"), C("C1", "C", ["out", "0"], "1u")],
             ["fc", "f0"], analysis="ac")
    res = get_solver(rc).solve(rc, CONV)
    ok &= check("f0 on a first-order circuit fails alone, fc still answered",
                close(res.numeric.get("fc"), 159.155) and res.answers.get("f0", "").startswith("Could not compute"),
                str(res.answers))
    return ok


def classifier_checks() -> bool:
    ok = True
    divider = net([C("V1", "V", ["in", "0"], "10"), C("R1", "R", ["in", "out"], "1k"),
                   C("R2", "R", ["out", "0"], "2k")], ["V(out)"], question="Find the voltage across R2.")
    c = classify(divider)
    ok &= check("divider -> topic dc, solver dc_ac", c.topic == "dc" and c.solver == "dc_ac", str(c))

    rc = net([C("V1", "V", ["in", "0"], ac="1"), C("R1", "R", ["in", "out"], "1k"), C("C1", "C", ["out", "0"], "1u")],
             ["H(out,in)", "fc"], analysis="ac", question="An RC low-pass filter has R = 1k and C = 1uF. Find fc.")
    c = classify(rc)
    ok &= check("RC low-pass -> topic filter, solver dc_ac", c.topic == "filter" and c.solver == "dc_ac", str(c))

    opamp = net([C("V1", "V", ["in", "0"], "1"), C("R1", "R", ["in", "n1"], "1k"),
                 C("R2", "R", ["n1", "out"], "10k"), Component(id="U1", type="OPAMP", nodes=["out", "0", "n1"])],
                ["V(out)"], question="Find Vout of the inverting op-amp amplifier.")
    c = classify(opamp)
    ok &= check("op-amp -> topic opamp, no solver yet", c.topic == "opamp" and c.solver is None, str(c))

    wrong = net([C("V1", "V", ["in", "0"], "10"), C("R1", "R", ["in", "out"], "5k"),
                 C("R2", "R", ["out", "0"], "5k")], ["V(out)"],
                question="A common-emitter transistor amplifier has RC = 5k. Find the gain.")
    c = classify(wrong)
    ok &= check("transistor question, resistor netlist -> low confidence", c.confidence == "low", str(c))

    concept = Netlist(analysis="conceptual", question="Why does negative feedback stabilize gain?")
    c = classify(concept)
    ok &= check("conceptual -> topic conceptual, no solver", c.topic == "conceptual" and c.solver is None, str(c))
    return ok


LIVE = [
    "A 10 V DC source is connected in series with R1 = 1 kOhm and R2 = 2 kOhm. Find the voltage across R2 "
    "and the power delivered by the source.",
    "A 12 V battery supplies R1 = 4 kOhm in series with the parallel combination of R2 = 6 kOhm and "
    "R3 = 12 kOhm. Find the current through R1 and through R3.",
    "Find the Thevenin equivalent at terminals a-b: a 20 V source in series with 5 kOhm connects to terminal a, "
    "and a 20 kOhm resistor connects a to b, where b is the negative terminal of the source.",
    "An RC low-pass filter has R = 1 kOhm and C = 1 uF. Find the transfer function Vout/Vin and the cutoff "
    "frequency.",
    "A series RLC circuit driven by a 1 V AC source has R = 10 Ohm, L = 10 mH and C = 1 uF, with the output "
    "taken across R. Find the resonant frequency, quality factor and bandwidth.",
]


def live_checks() -> bool:
    from llm_client import VelonaClient
    from parsing import parse_text

    client = VelonaClient()
    ok = True
    for i, question in enumerate(LIVE, 1):
        print(f"\n=== L{i}: {question}")
        pr = parse_text(question, client=client)
        if not pr.ok:
            ok &= check(f"L{i} parsed", False, str(pr.errors))
            continue
        print(pr.netlist.to_spice())
        print(f"query={pr.netlist.query}")
        cl = classify(pr.netlist, client=client)
        print(f"topic={cl.topic}  tags={cl.tags}  confidence={cl.confidence}  solver={cl.solver}")
        if cl.notes:
            print("notes:", cl.notes)
        if cl.solver is None:
            ok &= check(f"L{i} has a solver", False, cl.unsupported_reason)
            continue
        try:
            res = get_solver(pr.netlist).solve(pr.netlist, CONV)
        except SolverError as e:
            ok &= check(f"L{i} solved", False, str(e))
            continue
        for q, a in res.answers.items():
            print(f"     {q:<12} = {a}")
        show(res)
        ok &= check(f"L{i} solved", not res.verification_notes, str(res.verification_notes))
    print("\n" + client.usage_report())
    return ok


if __name__ == "__main__":
    all_ok = solver_checks()
    all_ok &= symbolic_and_error_checks()
    print()
    all_ok &= classifier_checks()
    if "--live" in sys.argv:
        all_ok &= live_checks()
    print("\nSTEP 3:", "ALL PASSED" if all_ok else "SOME CHECKS FAILED")
