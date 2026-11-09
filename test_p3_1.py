"""
Phase 3, Step 1 checks: diode circuits in DC.

    python test_p3_1.py           # hand-worked diode circuits + ngspice verification + parser checks (free)
    python test_p3_1.py --steps   # also print the key steps of each circuit
    python test_p3_1.py --live    # 5 real diode questions: parse -> solve -> ngspice verify (Velona)

Method under test: assume each diode's state, solve the linear circuit (MNA), check
  ON -> I_D >= 0,   OFF -> V_AK < Von,   Zener breakdown -> I_Z >= 0.
Every expected value is worked by hand in the comment above it.
"""
import sys

from benchmark.run_benchmark import check_expectation
from classifier import classify
from llm_client import load_conventions
from parsing.text_parser import consistency_errors
from schemas import Component, Netlist
from solvers import NotSupportedError, SolverError, get_solver
from verifier import Verifier

CONV = load_conventions()
SHOW = "--steps" in sys.argv


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"\n       {detail}" if detail and not cond else ""))
    return cond


def C(id, type, nodes, value=None, **kw):
    return Component(id=id, type=type, nodes=nodes, value=value, **kw)


def D(id, anode, cathode, model=None, **params):
    return Component(id=id, type="D", nodes=[anode, cathode], model=model, params=params)


def net(components, query, analysis="dc", question=""):
    return Netlist(analysis=analysis, query=query, components=components, question=question)


def close(got, want, rel=1e-6, floor=1e-12):
    return got is not None and abs(got - want) <= max(rel * abs(want), floor)


def show(res, full=False):
    for st in (res.trace.all_steps() if full else res.trace.key_steps()):
        print(f"      {st.step:>2}. {st.action}" + (f"\n          {st.equation}" if st.equation else "")
              + (f"\n          => {st.result}" if st.result else ""))


SRC5 = C("V1", "V", ["in", "0"], "5")
R1K = C("R1", "R", ["in", "a"], "1k")

CASES = [
    # Silicon diode, constant 0.7 V drop: I = (5 - 0.7)/1k = 4.3 mA, V(a) = 0.7 V, P = 0.7 × 4.3m = 3.01 mW
    ("Series silicon diode (0.7 V model)",
     net([SRC5, R1K, D("D1", "a", "0")], ["V(a)", "I(D1)", "P(D1)", "bias(D1)"]),
     {"V(a)": 0.7, "I(D1)": 4.3e-3, "P(D1)": 3.01e-3}, {"bias(D1)": "ON"}),

    # Reverse-biased (cathode at a): ON would need I_D = (0 - 0.7 - 5)/1k = -5.7 mA < 0 -> OFF, V(a) = 5 V, I = 0
    ("Reverse-biased diode -> OFF",
     net([SRC5, R1K, D("D1", "0", "a")], ["V(a)", "I(D1)", "bias(D1)"]),
     {"V(a)": 5.0, "I(D1)": 0.0}, {"bias(D1)": "OFF"}),

    # Ideal diode: ON = short -> V(a) = 0, I = 5 mA
    ("Ideal diode",
     net([SRC5, R1K, D("D1", "a", "0", "ideal")], ["V(a)", "I(D1)"]),
     {"V(a)": 0.0, "I(D1)": 5e-3}, {}),

    # Two diodes in series: I = (5 - 1.4)/1k = 3.6 mA, V(a) = 1.4 V, V(b) = 0.7 V
    ("Two silicon diodes in series",
     net([SRC5, R1K, D("D1", "a", "b"), D("D2", "b", "0")], ["V(a)", "V(b)", "I(D1)"]),
     {"V(a)": 1.4, "V(b)": 0.7, "I(D1)": 3.6e-3}, {}),

    # Si (0.7 V) and Ge (0.3 V) in parallel: Ge turns on first and clamps a at 0.3 V < 0.7 V -> Si stays OFF
    #   I(Ge) = (5 - 0.3)/1k = 4.7 mA, I(Si) = 0
    ("Silicon and germanium diodes in parallel",
     net([SRC5, R1K, D("D1", "a", "0"), D("D2", "a", "0", "germanium")], ["V(a)", "I(D1)", "I(D2)", "bias(D1)"]),
     {"V(a)": 0.3, "I(D1)": 0.0, "I(D2)": 4.7e-3}, {"bias(D1)": "OFF"}),

    # Piecewise-linear: Von = 0.7, rd = 20 Ω: I = 4.3/1020 = 4.21569 mA, V(a) = 0.7 + 20 I = 0.784314 V
    ("Piecewise-linear diode (rd = 20 Ω)",
     net([SRC5, R1K, D("D1", "a", "0", rd=20)], ["V(a)", "I(D1)"]),
     {"V(a)": 0.7 + 20 * 4.3 / 1020, "I(D1)": 4.3 / 1020}, {}),

    # Zener regulator: 12 V, R = 470, VZ = 5.1, RL = 1k
    #   V(out) = 5.1;  I_R = 6.9/470 = 14.6809 mA;  I_L = 5.1 mA;  I_Z = 9.58085 mA
    ("Zener regulator (in breakdown)",
     net([C("V1", "V", ["in", "0"], "12"), C("R1", "R", ["in", "out"], "470"), D("D1", "0", "out", "zener", VZ=5.1),
          C("RL", "R", ["out", "0"], "1k")], ["V(out)", "I(D1)", "I(R1)", "bias(D1)"]),
     {"V(out)": 5.1, "I(D1)": 6.9 / 470 - 5.1e-3, "I(R1)": 6.9 / 470}, {"bias(D1)": "breakdown"}),

    # Same regulator with RL = 200: without the Zener V = 12 × 200/670 = 3.582 V < 5.1 V -> Zener OFF
    ("Zener drops out of regulation (heavy load)",
     net([C("V1", "V", ["in", "0"], "12"), C("R1", "R", ["in", "out"], "470"), D("D1", "0", "out", "zener", VZ=5.1),
          C("RL", "R", ["out", "0"], "200")], ["V(out)", "I(D1)", "bias(D1)"]),
     {"V(out)": 12 * 200 / 670, "I(D1)": 0.0}, {"bias(D1)": "OFF"}),

    # Zener with rz = 10 Ω: (12 - Vo)/470 = Vo/1000 + (Vo - 5.1)/10 -> Vo = 5.192903 V, I_Z = 9.29028 mA
    ("Zener with series resistance rz = 10 Ω",
     net([C("V1", "V", ["in", "0"], "12"), C("R1", "R", ["in", "out"], "470"),
          D("D1", "0", "out", "zener", VZ=5.1, rz=10), C("RL", "R", ["out", "0"], "1k")], ["V(out)", "I(D1)"]),
     {"V(out)": 5.192902826490613, "I(D1)": 0.009290282649061333}, {}),

    # Exponential model, IS = 1e-14 A, n = 1, VT = 25 mV: solve (5 - V)/1k = IS(e^(V/0.025) - 1)
    #   -> V = 0.669851 V, I = 4.33015 mA
    ("Exponential diode (IS = 1e-14 A)",
     net([SRC5, R1K, D("D1", "a", "0", IS=1e-14, n=1)], ["V(a)", "I(D1)"]),
     {"V(a)": 0.6698509496766559, "I(D1)": 0.004330149050323345}, {}),

    # Sedra-Smith style: a diode that conducts 1 mA at 0.7 V (IS = 1e-3/(e^28 - 1) = 6.9144e-16 A),
    #   5 V and 1 kΩ -> V = 0.736254 V, I = 4.26375 mA
    ("Exponential diode, 1 mA at 0.7 V",
     net([SRC5, R1K, D("D1", "a", "0", "exponential", IS=6.914400106945009e-16)], ["V(a)", "I(D1)"]),
     {"V(a)": 0.7362537046469805, "I(D1)": 0.0042637462953530195}, {}),

    # LED with Vf = 2 V from 9 V through 330 Ω: I = 7/330 = 21.2121 mA
    ("LED current",
     net([C("V1", "V", ["in", "0"], "9"), C("R1", "R", ["in", "a"], "330"), D("D1", "a", "0", "led", Von=2)],
         ["I(D1)"]),
     {"I(D1)": 7 / 330}, {}),

    # Precision rectifier (super-diode), +2 V in: the op-amp drives D1 ON, V(out) = 2 V, op-amp output 2.7 V
    ("Op-amp super-diode, positive input",
     net([C("V1", "V", ["in", "0"], "2"), Component(id="U1", type="OPAMP", nodes=["x", "in", "out"], model="ideal"),
          D("D1", "x", "out"), C("RL", "R", ["out", "0"], "1k")], ["V(out)", "V(x)", "I(D1)"]),
     {"V(out)": 2.0, "V(x)": 2.7, "I(D1)": 2e-3}, {}),
]


def solver_checks(v: Verifier) -> bool:
    ok = True
    if v.exe is None:
        print("[SKIP] ngspice not found: verification checks skipped")
    for name, nl, expected, words in CASES:
        print(f"\n--- {name}")
        try:
            res = get_solver(nl).solve(nl, CONV)
        except (SolverError, NotSupportedError) as e:
            ok &= check(name, False, f"solver error: {e}")
            continue
        for q, a in res.answers.items():
            print(f"     {q:<10} = {a}")
        if SHOW:
            show(res)
        for key, want in expected.items():
            ok &= check(f"{name}: {key} = {want:.7g}", close(res.numeric.get(key), want), f"got {res.numeric.get(key)}")
        for key, word in words.items():
            ok &= check(f"{name}: {key} says '{word}'", word in res.answers.get(key, ""), res.answers.get(key, ""))
        if v.exe:
            rep = v.verify(nl, res)
            print(rep.table())
            ok &= check(f"{name}: verified by ngspice", rep.verified is True, "; ".join(res.verification_notes))
            info = [n for n in res.verification_notes if n.startswith("info:")]
            if info:
                print("     " + info[0])
    return ok


def step_checks() -> bool:
    ok = True
    print()
    nl = CASES[1][1]                                           # the reverse-biased diode
    res = get_solver(nl).solve(nl, CONV)
    full = [st for st in res.trace.all_steps()]
    tried = [st for st in full if st.action.startswith("Try D1 ON")]
    ok &= check("reverse diode: full steps show the rejected guess 'Try D1 ON ... ✗'",
                bool(tried) and "✗" in (tried[0].result or ""), str([st.action for st in full]))
    key = [st.action for st in res.trace.key_steps()]
    ok &= check("key steps: model, 'Assume D1 OFF', check of the assumption",
                any(a.startswith("Diode model") for a in key) and "Assume D1 OFF" in key
                and "Check the assumptions" in key, str(key))
    chk = next(st for st in res.trace.key_steps() if st.action == "Check the assumptions")
    ok &= check("check step shows V_AK = -5 V < 0.7 V ✓", "V_AK = -5 V" in (chk.equation or "") and "✓" in chk.equation,
                chk.equation)
    if SHOW:
        show(res, full=True)

    nl = CASES[9][1]                                           # exponential
    res = get_solver(nl).solve(nl, CONV)
    ok &= check("exponential: key step shows the Newton-Raphson iteration",
                any(st.action.startswith("Newton-Raphson") for st in res.trace.key_steps()))
    ok &= check("exponential: no 'Assume ...' step (nothing to assume)",
                not any(st.action.startswith("Assume") for st in res.trace.all_steps()))
    if SHOW:
        show(res)

    nl = CASES[6][1]                                           # zener: the model step names VZ
    res = get_solver(nl).solve(nl, CONV)
    ok &= check("zener: model step names VZ = 5.1 V",
                any("Zener diode, VZ = 5.1 V" in st.action for st in res.trace.key_steps()))
    return ok


def misc_checks() -> bool:
    ok = True
    print()
    # (rectifiers, clippers, filtered rectifiers and clampers are solved since P3-2; small-signal AC is not yet)
    acd = net([C("V1", "V", ["in", "0"], ac="10"), C("R1", "R", ["in", "a"], "1k"), D("D1", "a", "0")],
              ["V(a)"], analysis="ac")
    try:
        get_solver(acd)
        ok &= check("small-signal AC analysis of a diode circuit -> NotSupported (later step)", False)
    except NotSupportedError as e:
        ok &= check("small-signal AC analysis of a diode circuit -> NotSupported (later step)",
                    "Small-signal" in str(e), str(e))
    sym = net([C("V1", "V", ["in", "0"], "Vs"), C("R1", "R", ["in", "a"], "R"), D("D1", "a", "0")], ["V(a)"])
    try:
        get_solver(sym).solve(sym, CONV)
        ok &= check("symbolic diode circuit -> NotSupported (states need numbers)", False)
    except NotSupportedError as e:
        ok &= check("symbolic diode circuit -> NotSupported (states need numbers)", "numeric" in str(e), str(e))
    c = classify(net(CASES[0][1].components, ["I(D1)"], question="Find the diode current."))
    ok &= check("classifier: diode -> topic diode, solver diode", c.topic == "diode" and c.solver == "diode", str(c))

    ok &= check("Rth with a diode -> clear 'not available' message (other answers still given)",
                "not available for diode circuits" in get_solver(CASES[0][1]).solve(
                    net(CASES[0][1].components, ["V(a)", "Rth(a,0)"]), CONV).answers.get("Rth(a,0)", ""))
    return ok


def parser_checks() -> bool:
    ok = True
    print()
    q = ("A 12 V supply feeds a 5.1 V Zener diode through R = 470 Ohm, and a 1 kOhm load is connected across the "
         "Zener. Find the load voltage.")
    parts = [C("V1", "V", ["in", "0"], "12"), C("R1", "R", ["in", "out"], "470"), C("RL", "R", ["out", "0"], "1k")]
    e = consistency_errors(q, net(parts + [D("D1", "0", "out", "zener", VZ=5.1)], ["V(out)"], question=q))
    ok &= check("Zener reverse-connected (cathode on out) -> consistent", not e, str(e))
    e = consistency_errors(q, net(parts + [D("D1", "out", "0", "zener", VZ=5.1)], ["V(out)"], question=q))
    ok &= check("Zener with its cathode at ground -> rejected (must be reverse-connected)",
                any("REVERSE-connected" in x for x in e), str(e))
    e = consistency_errors(q, net(parts + [D("D1", "0", "out")], ["V(out)"], question=q))
    ok &= check("Zener question but plain diode -> rejected (model zener + VZ)", any('"zener"' in x for x in e), str(e))
    qi = "An ideal diode is in series with a 1 kOhm resistor across a 5 V source. Find the current."
    e = consistency_errors(qi, net([SRC5, R1K, D("D1", "a", "0")], ["I(D1)"], question=qi))
    ok &= check("'ideal diode' but no ideal model -> rejected", any('"ideal"' in x for x in e), str(e))
    e = consistency_errors(qi, net([SRC5, R1K, D("D1", "a", "0", "ideal")], ["I(D1)"], question=qi))
    ok &= check("'ideal diode' with model ideal -> consistent", not e, str(e))
    return ok


LIVE = [
    ("A 5 V DC source is connected to a 1 kOhm resistor in series with a silicon diode, anode toward the resistor "
     "and cathode to ground. Find the diode current.",
     [{"kind": "I", "value": 4.3e-3, "abs": True}]),
    ("A 15 V supply feeds a 470 Ohm load through a series resistor R = 220 Ohm. A 6.2 V Zener diode is connected "
     "across the load to regulate its voltage. Find the load voltage and the Zener current.",
     [{"kind": "V", "value": 6.2}, {"kind": "I", "value": (15 - 6.2) / 220 - 6.2 / 470, "abs": True}]),
    ("Two ideal diodes form an OR gate: D1 from node a to the output and D2 from node b to the output, and a "
     "1 kOhm resistor connects the output to ground. Node a is driven by a 5 V source and node b by a 3 V source, "
     "both referenced to ground. Find the output voltage.",
     [{"kind": "V", "value": 5.0}]),
    ("A diode with IS = 1e-14 A and n = 1 is in series with a 1 kOhm resistor across a 5 V supply. Using the diode "
     "equation with VT = 25 mV, find the diode current.",
     [{"kind": "I", "value": 0.004330149050323345, "abs": True, "tol": 0.001}]),
    ("An LED with a forward voltage of 2 V is driven from a 9 V battery through a 330 Ohm resistor. "
     "Find the LED current.",
     [{"kind": "I", "value": 7 / 330, "abs": True}]),
]


def live_checks() -> bool:
    from graph import CircuitAI

    ai = CircuitAI(auto_confirm=True, do_explain=False)
    ok = True
    for i, (question, expect) in enumerate(LIVE, 1):
        print(f"\n=== L{i}: {question}")
        r = ai.ask(question)
        st = r.state
        if st.get("netlist"):
            print(Netlist.model_validate(st["netlist"]).to_spice())
            print("query:", st["netlist"].get("query"), "| assumptions:", st["netlist"].get("assumptions"))
        for h in st.get("parse_history") or []:
            print(f"   rejected attempt {h['attempt']} at {h['stage']}: {h['errors'][0][:160]}")
        res = st.get("result") or {}
        print("answers:", res.get("answers"), "| verified:", res.get("verified"))
        passed = r.status == "answered"
        for e in expect:
            good, detail = check_expectation(e, res.get("numeric") or {}, res.get("symbolic") or {}, 0.01,
                                             res.get("verification_notes"))
            passed &= good
            print("  ", detail)
        ok &= check(f"L{i} correct and verified", passed and res.get("verified") is True,
                    f"status={r.status}, verified={res.get('verified')}, error={st.get('error')}")
    print("\n" + ai.client.usage_report())
    return ok


if __name__ == "__main__":
    ver = Verifier()
    all_ok = solver_checks(ver)
    all_ok &= step_checks()
    all_ok &= misc_checks()
    all_ok &= parser_checks()
    if "--live" in sys.argv:
        all_ok &= live_checks()
    print("\nPHASE 3 STEP 1:", "ALL PASSED" if all_ok else "SOME CHECKS FAILED")