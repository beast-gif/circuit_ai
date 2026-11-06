"""
Step 2 checks.

    python test_step2.py          # validator + question/netlist consistency checks (free, no LLM)
    python test_step2.py --live   # also parses 6 real questions with Velona
"""
import sys

from schemas import Component, Netlist
from validator import validate


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"\n       {detail}" if detail and not cond else ""))
    return cond


def C(id, type, nodes, value=None, **kw):
    return Component(id=id, type=type, nodes=nodes, value=value, **kw)


def nl(components, query, analysis="dc"):
    return Netlist(analysis=analysis, query=query, components=components)


def validator_checks() -> bool:
    ok = True
    divider = [C("V1", "V", ["in", "0"], "10"), C("R1", "R", ["in", "out"], "1k"),
               C("R2", "R", ["out", "0"], "2k")]

    r = validate(nl(divider, ["V(out)"]))
    ok &= check("divider: no errors, no warnings", r.ok and not r.warnings, str(r))

    r = validate(nl([C("V1", "V", ["a", "b"], "10"), C("R1", "R", ["a", "b"], "1k")], ["V(a)"]))
    ok &= check("no ground -> error", any("ground" in e.lower() for e in r.errors), str(r))

    r = validate(nl(divider + [C("R3", "R", ["x", "y"], "1k")], ["V(out)"]))
    ok &= check("floating section -> error", any("not connected" in e for e in r.errors), str(r))

    r = validate(nl([C("V1", "V", ["a", "0"], "10"), C("V2", "V", ["a", "0"], "5"),
                     C("R1", "R", ["a", "0"], "1k")], ["V(a)"]))
    ok &= check("parallel voltage sources -> error", any("loop" in e for e in r.errors), str(r))

    r = validate(nl([C("I1", "I", ["0", "x"], "1m"), C("I2", "I", ["x", "0"], "2m"),
                     C("V1", "V", ["a", "0"], "5"), C("R1", "R", ["a", "0"], "1k")], ["V(a)"]))
    ok &= check("node fed only by current sources -> error",
                any("only to current sources" in e for e in r.errors), str(r))

    r = validate(nl([C("V1", "V", ["a", "0"], "5"), C("L1", "L", ["a", "0"], "1m"),
                     C("R1", "R", ["a", "0"], "1k")], ["I(R1)"]))
    ok &= check("inductor across source in DC -> error", any("short" in e for e in r.errors), str(r))

    r = validate(nl([C("V1", "V", ["in", "0"], "10"), C("R1", "R", ["in", "Out"], "1k"),
                     C("R2", "R", ["out", "0"], "2k")], ["V(out)"]))
    ok &= check("'Out' vs 'out' -> error", any("case" in e for e in r.errors), str(r))

    r = validate(nl(divider, ["V(nope)"]))
    ok &= check("query on missing node -> error", any("nope" in e for e in r.errors), str(r))

    r = validate(nl(divider, ["I(R9)"]))
    ok &= check("query on missing component -> error", any("R9" in e for e in r.errors), str(r))

    r = validate(nl(divider, ["bias(R1)"]))
    ok &= check("bias() on a resistor -> error", any("bias()" in e for e in r.errors), str(r))

    r = validate(nl(divider, ["voltage of out"]))
    ok &= check("unknown query form -> error", any("Unknown query" in e for e in r.errors), str(r))

    thev = [C("V1", "V", ["n1", "0"], "20"), C("R1", "R", ["n1", "a"], "5k"),
            C("R2", "R", ["a", "0"], "20k"), C("R3", "R", ["a", "b"], "1k")]
    r = validate(nl(thev, ["Rth(b,0)"]))
    ok &= check("open Thevenin terminal -> warning only", r.ok and any("open terminal" in w for w in r.warnings),
                str(r))

    r = validate(nl([C("V1", "V", ["in", "0"], "10"), C("R1", "R", ["in", "out"], "1k"),
                     C("C1", "C", ["out", "0"], "1u")], ["H(out,in)", "fc"], analysis="ac"))
    ok &= check("AC with no AC magnitude -> warning only", r.ok and any("AC magnitude" in w for w in r.warnings),
                str(r))

    r = validate(Netlist(analysis="conceptual", question="Why does feedback stabilize gain?"))
    ok &= check("conceptual question -> no checks", r.ok)
    return ok


def consistency_checks() -> bool:
    from parsing.text_parser import consistency_errors

    ok = True
    q_rlc = ("A series RLC circuit driven by a 1 V AC source has R = 10 Ohm, L = 10 mH and C = 1 uF, "
             "with the output taken across R. Find the resonant frequency.")
    right = Netlist(analysis="ac", query=["f0"], question=q_rlc, components=[
        C("V1", "V", ["in", "0"], ac="1"), C("L1", "L", ["in", "n1"], "10m"),
        C("C1", "C", ["n1", "out"], "1u"), C("R1", "R", ["out", "0"], "10")])
    wrong = Netlist(analysis="ac", query=["f0"], question=q_rlc, components=[
        C("V1", "V", ["in", "0"], ac="1"), C("R1", "R", ["in", "out"], "10"),
        C("L1", "L", ["out", "n1"], "10m"), C("C1", "C", ["n1", "0"], "1u")])
    e = consistency_errors(q_rlc, right)
    ok &= check("RLC, R between out and 0 -> consistent", not e, str(e))
    e = consistency_errors(q_rlc, wrong)
    ok &= check("RLC, R between in and out -> mismatch", bool(e) and "R1" in e[0], str(e))

    q_rc = "An RC circuit has R = 1 kOhm and C = 1 uF; the output voltage is taken across the 1 uF capacitor."
    rc = Netlist(analysis="ac", query=["H(out,in)"], question=q_rc, components=[
        C("V1", "V", ["in", "0"], ac="1"), C("R1", "R", ["in", "out"], "1k"), C("C1", "C", ["out", "0"], "1u")])
    e = consistency_errors(q_rc, rc)
    ok &= check("output across the 1 uF capacitor, C at out-0 -> consistent", not e, str(e))

    q_div = "A 10 V source feeds R1 = 1 kOhm in series with R2 = 2 kOhm. Find the voltage across R2."
    comps = [C("V1", "V", ["in", "0"], "10"), C("R1", "R", ["in", "out"], "1k"), C("R2", "R", ["out", "0"], "2k")]
    e = consistency_errors(q_div, Netlist(analysis="dc", query=["V(out)"], question=q_div, components=comps))
    ok &= check("voltage across R2 with V(out) -> consistent", not e, str(e))
    e = consistency_errors(q_div, Netlist(analysis="dc", query=["V(in)"], question=q_div, components=comps))
    ok &= check("voltage across R2 with V(in) -> mismatch", bool(e) and "V(out)" in e[0], str(e))

    q_missing = "The output is taken across the inductor. Find Vout."
    e = consistency_errors(q_missing, Netlist(analysis="dc", query=["V(out)"], question=q_missing,
                                              components=comps))
    ok &= check("output across an inductor that is not in the netlist -> mismatch", bool(e), str(e))
    return ok


LIVE_CASES = [
    ("A 10 V DC source is connected in series with R1 = 1 kOhm and R2 = 2 kOhm. "
     "Find the voltage across R2.",
     dict(analysis="dc", min_comps=3, query_has="V(")),
    ("A 12 V battery supplies R1 = 4 kOhm in series with the parallel combination of "
     "R2 = 6 kOhm and R3 = 12 kOhm. Find the current through R1.",
     dict(analysis="dc", min_comps=4, query_has="I(R1)")),
    ("An RC low-pass filter has R = 1 kOhm and C = 1 uF. Find the transfer function "
     "Vout/Vin and the cutoff frequency.",
     dict(analysis="ac", min_comps=3, query_has="fc")),
    ("Find the Thevenin equivalent at terminals a-b: a 20 V source in series with 5 kOhm "
     "connects to terminal a, and a 20 kOhm resistor connects a to b, where b is the "
     "negative terminal of the source.",
     dict(analysis="dc", min_comps=3, query_has="th(")),
    ("Find Vout of a voltage divider made of R1 and R2 fed by a source Vs, with Vout across R2.",
     dict(analysis="dc", min_comps=3, query_has="V(")),
    ("Why does negative feedback stabilize the gain of an amplifier?",
     dict(analysis="conceptual", min_comps=0, query_has="")),
]


def live_checks() -> bool:
    from llm_client import VelonaClient
    from parsing import parse_text

    client = VelonaClient()
    ok = True
    for i, (question, exp) in enumerate(LIVE_CASES, 1):
        print(f"\n--- Q{i}: {question[:80]}{'...' if len(question) > 80 else ''}")
        res = parse_text(question, client=client)
        if not res.ok:
            ok &= check(f"Q{i} parsed", False, f"errors after {res.attempts} attempts: {res.errors}")
            continue
        n = res.netlist
        print(n.to_spice() or "(no components)")
        print(f"analysis={n.analysis.value}  query={n.query}  attempts={res.attempts}  model={res.model_used}")
        if res.report.warnings:
            print("warnings:", res.report.warnings)
        if res.missing_info:
            print("missing_info:", res.missing_info)
        if n.assumptions:
            print("assumptions:", n.assumptions)
        good = (n.analysis.value == exp["analysis"]
                and len(n.components) >= exp["min_comps"]
                and (not exp["query_has"] or any(exp["query_has"].lower() in q.lower() for q in n.query)))
        ok &= check(f"Q{i} matches expected", good, f"expected {exp}")

    print("\n" + client.usage_report())
    return ok


if __name__ == "__main__":
    all_ok = validator_checks()
    print()
    all_ok &= consistency_checks()
    if "--live" in sys.argv:
        all_ok &= live_checks()
    print("\nSTEP 2:", "ALL PASSED" if all_ok else "SOME CHECKS FAILED")