"""
Phase 2, Step 2 checks: op-amps from a real question.

    python test_p2_2.py           # parser checks, validator, drawings, ngspice verification (free, no LLM)
    python test_p2_2.py --live    # 6 real op-amp questions: parse -> solve -> ngspice verify (Velona)
"""
import sys
from pathlib import Path

import sympy as sp

from benchmark.run_benchmark import check_expectation
from llm_client import load_conventions
from parsing.text_parser import consistency_errors
from renderer import describe, render_svg
from schemas import Component, Netlist
from solvers import get_solver
from test_p2_1 import CASES as P21_CASES
from validator import validate
from verifier import Verifier

CONV = load_conventions()
RENDERS = Path(__file__).with_name("renders")


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"\n       {detail}" if detail and not cond else ""))
    return cond


def C(id, type, nodes, value=None, **kw):
    return Component(id=id, type=type, nodes=nodes, value=value, **kw)


def U(id, out, plus, minus, **kw):
    return Component(id=id, type="OPAMP", nodes=[out, plus, minus], model="ideal", **kw)


def nl(components, query, question, analysis="dc"):
    return Netlist(analysis=analysis, query=query, components=components, question=question)


def parser_checks() -> bool:
    ok = True
    inv_parts = [C("V1", "V", ["in", "0"], "1"), C("R1", "R", ["in", "n1"], "1k"), C("R2", "R", ["n1", "out"], "10k")]

    q = "An inverting amplifier has Rin = 1 kOhm and Rf = 10 kOhm. Find Vout for a 1 V input."
    e = consistency_errors(q, nl(inv_parts + [U("U1", "out", "0", "n1")], ["V(out)"], q))
    ok &= check("inverting amp wired correctly -> consistent", not e, str(e))
    e = consistency_errors(q, nl(inv_parts + [U("U1", "out", "in", "n1")], ["V(out)"], q))
    ok &= check("inverting amp with + input on the signal -> rejected", any("inverting amplifier" in x for x in e), str(e))
    e = consistency_errors(q, nl(inv_parts + [U("U1", "out", "n1", "0")], ["V(out)"], q))
    ok &= check("inputs swapped (no feedback to − input) -> rejected", any("no feedback" in x for x in e), str(e))

    qn = "A non-inverting amplifier has R1 = 1 kOhm to ground and Rf = 9 kOhm. Find Vout for 1 V in."
    good = [C("V1", "V", ["in", "0"], "1"), U("U1", "out", "in", "n1"), C("R1", "R", ["n1", "0"], "1k"),
            C("R2", "R", ["n1", "out"], "9k")]
    e = consistency_errors(qn, nl(good, ["V(out)"], qn))
    ok &= check("non-inverting amp wired correctly -> consistent", not e, str(e))
    e = consistency_errors(qn, nl(inv_parts + [U("U1", "out", "0", "n1")], ["V(out)"], qn))
    ok &= check("non-inverting question but + input grounded -> rejected",
                any("non-inverting" in x for x in e), str(e))

    qf = "A voltage follower is driven by a 3.3 V source and drives a 100 Ohm load. Find Vout."
    e = consistency_errors(qf, nl([C("V1", "V", ["in", "0"], "3.3"), U("U1", "out", "in", "out"),
                                   C("RL", "R", ["out", "0"], "100")], ["V(out)"], qf))
    ok &= check("follower with − input tied to out -> consistent", not e, str(e))

    qd = ("A difference amplifier: R1 = 10 kOhm from V2 to the inverting input, R2 = 10 kOhm from the inverting "
          "input to the output, R3 = 10 kOhm from V1 to the non-inverting input, R4 = 10 kOhm to ground.")
    diff = [C("V1", "V", ["a", "0"], "5"), C("V2", "V", ["b", "0"], "2"), C("R1", "R", ["b", "n"], "10k"),
            C("R2", "R", ["n", "out"], "10k"), C("R3", "R", ["a", "p"], "10k"), C("R4", "R", ["p", "0"], "10k"),
            U("U1", "out", "p", "n")]
    e = consistency_errors(qd, nl(diff, ["V(out)"], qd))
    ok &= check("'inverting input' wording does not trigger the inverting-amplifier rule", not e, str(e))

    qi = "An op-amp integrator has R = 10 kOhm and C = 100 nF. Find Vout/Vin."
    wrong_int = [C("V1", "V", ["in", "0"], ac="1"), C("C1", "C", ["in", "n1"], "100n"),
                 C("R1", "R", ["n1", "out"], "10k"), U("U1", "out", "0", "n1")]
    e = consistency_errors(qi, nl(wrong_int, ["H(out,in)"], qi, "ac"))
    ok &= check("integrator with C at the input (that's a differentiator) -> rejected",
                any("integrator" in x for x in e), str(e))

    # T-network feedback (no single element from out to in−) still counts as feedback
    t_net = inv_parts[:2] + [C("Ra", "R", ["n1", "x"], "10k"), C("Rb", "R", ["x", "out"], "10k"),
                             C("Rc", "R", ["x", "0"], "1k"), U("U1", "out", "0", "n1")]
    e = consistency_errors(q, nl(t_net, ["V(out)"], q))
    ok &= check("T-network feedback path is recognised", not any("no feedback" in x for x in e), str(e))

    r = validate(nl([C("V1", "V", ["in", "0"], "1"), C("R1", "R", ["in", "n1"], "1k"),
                     U("U1", "0", "in", "n1")], ["V(n1)"], "x"))
    ok &= check("validator: op-amp output on ground -> error", any("output is connected to ground" in x
                                                                    for x in r.errors), str(r))
    return ok


def verifier_and_drawing_checks() -> bool:
    ok = True
    v = Verifier()
    RENDERS.mkdir(exist_ok=True)
    if v.exe is None:
        print("[SKIP] ngspice not found: verification checks skipped")
    cases = list(P21_CASES) + [(
        "Follower delivering 33 mA",
        nl([C("V1", "V", ["in", "0"], "3.3"), U("U1", "out", "in", "out"), C("RL", "R", ["out", "0"], "100")],
           ["V(out)", "I(U1)"], "follower"),
        {"V(out)": 3.3, "I(U1)": 0.033})]
    for name, net, expected in cases:
        res = get_solver(net).solve(net, CONV)
        for key, want in expected.items():
            got = res.numeric.get(key)
            ok &= check(f"{name}: {key} = {want:g}", got is not None and abs(got - want) <= 1e-3 * max(1e-12, abs(want)),
                        f"got {got}")
        if v.exe:
            rep = v.verify(net, res)
            print(rep.table())
            ok &= check(f"{name}: verified by ngspice", rep.verified is True, "; ".join(res.verification_notes))
        svg = render_svg(net)
        if svg:
            (RENDERS / ("opamp_" + name.split("(")[0].strip().lower().replace(" ", "_") + ".svg")).write_text(
                svg, encoding="utf-8")
        ok &= check(f"{name}: drawing made", bool(svg))
    print(f"\n     op-amp drawings saved to {RENDERS} (opamp_*.svg)")
    print("\n" + describe(P21_CASES[0][1]))
    return ok


LIVE = [
    ("An inverting amplifier uses Rin = 4.7 kOhm and Rf = 47 kOhm with an ideal op-amp. The input is 0.2 V DC. "
     "Find the output voltage.", [{"kind": "V", "value": -2.0}]),
    ("A non-inverting amplifier has R1 = 1 kOhm from the inverting input to ground and Rf = 4 kOhm from the output "
     "to the inverting input. The input is 0.5 V. Find the output voltage.", [{"kind": "V", "value": 2.5}]),
    ("An inverting summing amplifier has Rf = 20 kOhm and two inputs: V1 = 1 V through 10 kOhm and V2 = 0.5 V "
     "through 5 kOhm. Find the output voltage.", [{"kind": "V", "value": -4.0}]),
    ("A voltage follower is driven by a 3.3 V source and drives a 100 Ohm load. Find the output voltage and the "
     "current supplied by the op-amp.", [{"kind": "V", "value": 3.3}, {"kind": "I", "value": 0.033, "abs": True}]),
    ("A difference amplifier uses four equal 10 kOhm resistors: R1 from V2 to the inverting input, R2 from the "
     "inverting input to the output, R3 from V1 to the non-inverting input and R4 from the non-inverting input to "
     "ground. V1 = 5 V and V2 = 2 V. Find the output voltage.", [{"kind": "V", "value": 3.0}]),
    ("An op-amp integrator has R = 10 kOhm from the input to the inverting input and C = 100 nF from the output to "
     "the inverting input. Find the transfer function Vout/Vin.", [{"kind": "H_times_s", "value": -1000.0}]),
]


def live_checks() -> bool:
    from graph import CircuitAI

    ai = CircuitAI(auto_confirm=True, do_explain=False)
    ok = True
    s = sp.Symbol("s")
    for i, (question, expect) in enumerate(LIVE, 1):
        print(f"\n=== L{i}: {question}")
        r = ai.ask(question)
        st = r.state
        if st.get("netlist"):
            print(Netlist.model_validate(st["netlist"]).to_spice())
            print("query:", st["netlist"].get("query"))
        for h in st.get("parse_history") or []:
            print(f"   rejected attempt {h['attempt']} at {h['stage']}: {h['errors'][0][:140]}")
        res = st.get("result") or {}
        print("answers:", res.get("answers"), "| verified:", res.get("verified"))
        passed = r.status == "answered"
        for e in expect:
            if e["kind"] == "H_times_s":                     # integrator: H(s)·s must equal -1/(RC)
                hs = [v for k, v in (res.get("symbolic") or {}).items() if k.startswith("H(")]
                good = any(sp.simplify(sp.sympify(h) * s - e["value"]) == 0 for h in hs)
                passed &= good
                print(f"   H(s)·s = {e['value']}: {'ok' if good else 'NO'}  (H = {hs})")
            else:
                good, detail = check_expectation(e, res.get("numeric") or {}, res.get("symbolic") or {}, 0.01)
                passed &= good
                print("  ", detail)
        ok &= check(f"L{i} correct and verified", passed and res.get("verified") is True,
                    f"status={r.status}, verified={res.get('verified')}")
    print("\n" + ai.client.usage_report())
    return ok


if __name__ == "__main__":
    all_ok = parser_checks()
    print()
    all_ok &= verifier_and_drawing_checks()
    if "--live" in sys.argv:
        all_ok &= live_checks()
    print("\nPHASE 2 STEP 2:", "ALL PASSED" if all_ok else "SOME CHECKS FAILED")