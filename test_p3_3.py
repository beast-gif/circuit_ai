"""
Phase 3, Step 3 checks: BJT DC bias (Q-point), with diodes allowed in the same circuit.

    python test_p3_3.py           # hand-worked bias circuits + ngspice verification + parser checks (free)
    python test_p3_3.py --steps   # also print the key steps of each circuit
    python test_p3_3.py --live    # 5 real bias questions: parse -> solve -> ngspice verify (Velona)

Model: active  V_BE = 0.7 V, I_C = β·I_B;  saturation  V_BE = 0.7 V, V_CE = 0.2 V;  cutoff  I_B = I_C = 0.
Hand formula for one transistor (base network as a Thevenin source V_BB, R_BB, emitter resistor R_E to V_X):
    I_B = (V_BB − V_BE − V_X) / (R_BB + (β + 1)·R_E)
Every expected value is worked by hand in the comment above it (independently checked).
"""
import sys

from benchmark.run_benchmark import check_expectation
from classifier import classify
from llm_client import load_conventions
from parsing.text_parser import consistency_errors
from schemas import Component, Netlist
from solvers import NotSupportedError, SolverError, get_solver
from validator import validate
from verifier import Verifier

CONV = load_conventions()
SHOW = "--steps" in sys.argv


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"\n       {detail}" if detail and not cond else ""))
    return cond


def C(id, type, nodes, value=None, **kw):
    return Component(id=id, type=type, nodes=nodes, value=value, **kw)


def Q(id, c, b, e, model="NPN", beta=100):
    return Component(id=id, type="Q", nodes=[c, b, e], model=model, params={"beta": beta})


def net(components, query, question=""):
    return Netlist(analysis="dc", query=query, components=components, question=question)


def close(got, want, rel=1e-6, floor=1e-12):
    return got is not None and abs(got - want) <= max(rel * abs(want), floor)


def show(res, full=False):
    for st in (res.trace.all_steps() if full else res.trace.key_steps()):
        print(f"      {st.step:>2}. {st.action}" + (f"\n          {st.equation}" if st.equation else "")
              + (f"\n          => {st.result}" if st.result else ""))


VCC12 = C("VCC", "V", ["vcc", "0"], "12")
BQ = ["bias(Q1)", "IC(Q1)", "IB(Q1)", "VCE(Q1)"]

CASES = [
    # Fixed bias: VCC = 12, RB = 470k, RC = 2.2k, β = 100:
    #   I_B = (12 − 0.7)/470k = 24.0426 µA, I_C = 2.40426 mA, V_CE = 12 − 2.2k·I_C = 6.71064 V  -> active
    ("Fixed bias (active)",
     net([VCC12, C("RB", "R", ["vcc", "b"], "470k"), C("RC", "R", ["vcc", "c"], "2.2k"), Q("Q1", "c", "b", "0")], BQ),
     {"IB(Q1)": 2.4042553191489362e-05, "IC(Q1)": 0.002404255319148936, "VCE(Q1)": 6.7106382978723405},
     {"bias(Q1)": "active"}),

    # Fixed bias with RB = 47k: β·I_B = 24.04 mA but RC allows only (12 − 0.2)/2.2k = 5.36364 mA -> saturated,
    #   V_CE = 0.2 V, forced β = 22.31 < 100
    ("Fixed bias (saturated)",
     net([VCC12, C("RB", "R", ["vcc", "b"], "47k"), C("RC", "R", ["vcc", "c"], "2.2k"), Q("Q1", "c", "b", "0")], BQ),
     {"IC(Q1)": 0.005363636363636364, "IB(Q1)": 0.00024042553191489364, "VCE(Q1)": 0.2},
     {"bias(Q1)": "saturation"}),

    # Voltage divider: V_BB = 12·10/57 = 2.10526 V, R_BB = 47k‖10k = 8.24561k,
    #   I_B = (2.10526 − 0.7)/(8.24561k + 101·1k) = 12.8633 µA, I_C = 1.28633 mA, V_CE = 7.87087 V
    ("Voltage-divider bias",
     net([VCC12, C("R1", "R", ["vcc", "b"], "47k"), C("R2", "R", ["b", "0"], "10k"), C("RC", "R", ["vcc", "c"], "2.2k"),
          C("RE", "R", ["e", "0"], "1k"), Q("Q1", "c", "b", "e")], BQ + ["IE(Q1)", "V(e)", "V(c)"]),
     {"IB(Q1)": 1.2863337080456077e-05, "IC(Q1)": 0.0012863337080456078, "IE(Q1)": 0.001299197045126064,
      "VCE(Q1)": 7.8708687971736, "V(e)": 1.299197045126064, "V(c)": 9.170065842299664}, {"bias(Q1)": "active"}),

    # Dual supply: +10 / −10 V, RB = 10k to ground, RE = 10k to −10 V, RC = 4.7k:
    #   I_B = (0 − 0.7 + 10)/(10k + 101·10k) = 9.11765 µA, I_C = 0.911765 mA, V_E = −0.791176, V_CE = 6.50588 V
    ("Dual-supply emitter bias",
     net([C("VCC", "V", ["vcc", "0"], "10"), C("VEE", "V", ["vee", "0"], "-10"), C("RB", "R", ["b", "0"], "10k"),
          C("RC", "R", ["vcc", "c"], "4.7k"), C("RE", "R", ["e", "vee"], "10k"), Q("Q1", "c", "b", "e")],
         ["IC(Q1)", "VCE(Q1)", "V(e)"]),
     {"IC(Q1)": 0.000911764705882353, "VCE(Q1)": 6.505882352941176, "V(e)": -0.7911764705882353}, {}),

    # Collector feedback: RB = 220k from c to b, RC = 2.2k, emitter grounded:
    #   I_B = (12 − 0.7)/(RB + (β+1)·RC) = 11.3/442.2k = 25.5540 µA, I_C = 2.55540 mA, V_CE = 6.32189 V
    ("Collector-feedback bias",
     net([VCC12, C("RC", "R", ["vcc", "c"], "2.2k"), C("RB", "R", ["c", "b"], "220k"), Q("Q1", "c", "b", "0")],
         ["IC(Q1)", "VCE(Q1)"]),
     {"IC(Q1)": 0.0025554047942107643, "VCE(Q1)": 6.321890547263682}, {}),

    # PNP: emitter at +12 V, RB = 470k base to ground, RC = 2.2k collector to ground:
    #   I_B = (12 − 0.7)/470k = 24.0426 µA (out of the base), I_C = 2.40426 mA, V_C = 5.28936 V, V_EC = 6.71064 V
    ("PNP fixed bias",
     net([VCC12, C("RB", "R", ["b", "0"], "470k"), C("RC", "R", ["c", "0"], "2.2k"), Q("Q1", "c", "b", "vcc", "PNP")],
         ["IC(Q1)", "VEC(Q1)", "V(c)", "bias(Q1)"]),
     {"IC(Q1)": 0.002404255319148936, "VEC(Q1)": 6.7106382978723405, "V(c)": 5.2893617021276595},
     {"bias(Q1)": "active"}),

    # Cutoff: base driven by 0.5 V through 10k (< 0.7 V): I_C = 0, V_C = VCC = 12 V
    ("Cutoff (0.5 V at the base)",
     net([VCC12, C("VIN", "V", ["in", "0"], "0.5"), C("RB", "R", ["in", "b"], "10k"), C("RC", "R", ["vcc", "c"], "1k"),
          Q("Q1", "c", "b", "0")], ["IC(Q1)", "V(c)", "bias(Q1)"]),
     {"IC(Q1)": 0.0, "V(c)": 12.0}, {"bias(Q1)": "cutoff"}),

    # Transistor switch driving an LED (2 V) from 5 V: I_B = 4.3/10k = 430 µA, β·I_B = 43 mA >
    #   (5 − 2 − 0.2)/220 = 12.7273 mA -> saturated, LED current 12.7273 mA
    ("BJT switch with an LED (diode + transistor)",
     net([C("VCC", "V", ["vcc", "0"], "5"), C("RC", "R", ["vcc", "a"], "220"),
          Component(id="D1", type="D", nodes=["a", "c"], model="led", params={"Von": 2}),
          C("RB", "R", ["vcc", "b"], "10k"), Q("Q1", "c", "b", "0")], ["IC(Q1)", "I(D1)", "bias(Q1)"]),
     {"IC(Q1)": 0.012727272727272726, "I(D1)": 0.012727272727272726}, {"bias(Q1)": "saturation"}),

    # Darlington, RB = 10 MΩ: I_B1 = (12 − 1.4)/10M = 1.06 µA, I_B2 = I_E1 = 101·I_B1, I_C2 = 100·I_B2 = 10.706 mA,
    #   V_C2 = 12 − 1k·I_C2 = 1.294 V
    ("Darlington pair",
     net([VCC12, C("RB", "R", ["vcc", "b1"], "10Meg"), Q("Q1", "vcc", "b1", "e1"), Q("Q2", "c2", "e1", "0"),
          C("RC", "R", ["vcc", "c2"], "1k")], ["IC(Q2)", "V(c2)"]),
     {"IC(Q2)": 0.010706, "V(c2)": 1.2940000000000005}, {}),

    # Current mirror: I_REF = (12 − 0.7)/11.3k = 1 mA = I_C + 2·I_B -> I_C2 = I_REF·β/(β + 2) = 0.980392 mA,
    #   V_C2 = 12 − 2k·I_C2 = 10.0392 V
    ("Current mirror",
     net([VCC12, C("RREF", "R", ["vcc", "x"], "11.3k"), Q("Q1", "x", "x", "0"), Q("Q2", "out", "x", "0"),
          C("RL", "R", ["vcc", "out"], "2k")], ["IC(Q2)", "V(out)"]),
     {"IC(Q2)": 0.000980392156862745, "V(out)": 10.03921568627451}, {}),
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
    nl = CASES[2][1]                                    # voltage divider: the textbook cross-check step
    res = get_solver(nl).solve(nl, CONV)
    tb = [st for st in res.trace.key_steps() if st.action.startswith("Textbook check")]
    ok &= check("voltage divider: textbook step with V_BB = 2.105 V and R_BB = 8.246 kΩ",
                bool(tb) and "V_BB = 2.105 V" in (tb[0].equation or "") and "8.246 kΩ" in (tb[0].equation or ""),
                str([(st.action, st.equation) for st in tb]))
    acts = [st.action for st in res.trace.key_steps()]
    ok &= check("key steps: model, 'Assume Q1 active', 'Check the assumptions'",
                any(a.startswith("Transistor model") for a in acts) and "Assume Q1 active" in acts
                and "Check the assumptions" in acts, str(acts))
    if SHOW:
        show(res)

    nl = CASES[1][1]                                    # saturated: the rejected 'active' guess in the full view
    res = get_solver(nl).solve(nl, CONV)
    tried = [st for st in res.trace.all_steps() if st.action.startswith("Try Q1 active")]
    ok &= check("saturated: full view shows 'Try Q1 active ... ✗ (it would be saturated)'",
                bool(tried) and "saturated" in (tried[0].result or ""), str([st.result for st in tried]))
    chk = next(st for st in res.trace.key_steps() if st.action == "Check the assumptions")
    ok &= check("saturated: check shows forced β = 22.31 ≤ 100", "forced β = I_C/I_B = 22.31" in (chk.equation or ""),
                chk.equation)

    nl = CASES[4][1]                                    # collector feedback: no (wrong) textbook formula step
    res = get_solver(nl).solve(nl, CONV)
    ok &= check("collector feedback: no textbook voltage-divider step",
                not any(st.action.startswith("Textbook check") for st in res.trace.all_steps()))
    return ok


def misc_checks() -> bool:
    ok = True
    print()
    r = validate(CASES[2][1])
    ok &= check("validator accepts bias / IC / IB / VCE / IE", not r.errors, str(r.errors))
    bad = net(CASES[2][1].components, ["IC(RC)"])
    ok &= check("validator: IC() of a resistor -> error", any("needs a BJT" in x for x in validate(bad).errors))
    c = classify(net(CASES[0][1].components, ["IC(Q1)"], question="Find the collector current of the BJT."))
    ok &= check("classifier: BJT -> topic bjt, solver transistor", c.topic == "bjt" and c.solver == "transistor", str(c))
    ss = Netlist(analysis="small_signal", query=["gain"], components=CASES[2][1].components, question="CE amplifier")
    try:
        get_solver(ss)
        ok &= check("small-signal BJT amplifier -> NotSupported until P3-4", False)
    except NotSupportedError as e:
        ok &= check("small-signal BJT amplifier -> NotSupported until P3-4", "P3-4" in str(e), str(e))
    return ok


def parser_checks() -> bool:
    ok = True
    print()
    q = "A fixed-bias NPN circuit has VCC = 12 V, RB = 470 kOhm, RC = 2.2 kOhm and beta = 150. The emitter is grounded."
    good = net([VCC12, C("RB", "R", ["vcc", "b"], "470k"), C("RC", "R", ["vcc", "c"], "2.2k"),
                Q("Q1", "c", "b", "0", beta=150)], ["IC(Q1)"], q)
    e = consistency_errors(q, good)
    ok &= check("fixed bias, β = 150 given and set, emitter grounded -> consistent", not e, str(e))
    e = consistency_errors(q, net(good.components[:3] + [Q("Q1", "c", "b", "0", beta=100)], ["IC(Q1)"], q))
    ok &= check("β = 150 in the question but 100 in the netlist -> rejected", any('"beta": 150' in x for x in e), str(e))
    e = consistency_errors(q, net(good.components[:3] + [Q("Q1", "c", "b", "e", beta=150),
                                                       C("RE", "R", ["e", "0"], "1k")], ["IC(Q1)"], q))
    ok &= check("'emitter is grounded' but emitter on node e -> rejected", any("emitter is grounded" in x for x in e),
                str(e))
    qp = "A PNP transistor has its emitter at +12 V, RB = 470 kOhm from base to ground and RC = 2.2 kOhm. beta = 100."
    e = consistency_errors(qp, net(CASES[5][1].components[:3] + [Q("Q1", "c", "b", "vcc", "NPN")], ["IC(Q1)"], qp))
    ok &= check("'PNP' in the question but model NPN -> rejected", any('"PNP"' in x for x in e), str(e))
    return ok


LIVE = [
    ("A voltage-divider biased NPN transistor has VCC = 12 V, R1 = 47 kOhm from VCC to the base, R2 = 10 kOhm from "
     "the base to ground, RC = 2.2 kOhm and RE = 1 kOhm, with beta = 100. Find IC and VCE.",
     [{"kind": "IC", "value": 0.0012863337080456078}, {"kind": "VCE", "value": 7.8708687971736}]),
    ("A fixed-bias NPN transistor circuit has VCC = 12 V, RB = 47 kOhm from VCC to the base, RC = 2.2 kOhm and the "
     "emitter grounded. With beta = 100, find the collector current and the region of operation.",
     [{"kind": "IC", "value": 0.005363636363636364}]),
    ("In a collector-feedback bias circuit, RC = 2.2 kOhm connects VCC = 12 V to the collector, RB = 220 kOhm "
     "connects the collector to the base, and the emitter is grounded. With beta = 100, find IC and VCE.",
     [{"kind": "IC", "value": 0.0025554047942107643}, {"kind": "VCE", "value": 6.321890547263682}]),
    ("A PNP transistor has its emitter connected to +12 V, a 470 kOhm resistor from the base to ground and a "
     "2.2 kOhm resistor from the collector to ground. With beta = 100, find the collector current and VEC.",
     [{"kind": "IC", "value": 0.002404255319148936}, {"kind": "VEC", "value": 6.7106382978723405}]),
    ("An NPN transistor is biased from +10 V and -10 V supplies: RC = 4.7 kOhm from +10 V to the collector, "
     "RE = 10 kOhm from the emitter to -10 V, and RB = 10 kOhm from the base to ground. With beta = 100 find IC "
     "and VCE.",
     [{"kind": "IC", "value": 0.000911764705882353}, {"kind": "VCE", "value": 6.505882352941176}]),
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
            print("query:", st["netlist"].get("query"))
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
    print("\nPHASE 3 STEP 3:", "ALL PASSED" if all_ok else "SOME CHECKS FAILED")