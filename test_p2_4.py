"""
Phase 2, Step 4 checks: active filters (Sallen-Key, MFB) and filter identification from H(s).

    python test_p2_4.py           # hand-worked filters + ngspice verification + parser checks (free, no LLM)
    python test_p2_4.py --steps   # also print the key steps of each circuit
    python test_p2_4.py --live    # 4 real filter questions: parse -> solve -> ngspice verify (Velona)

Standard second-order form:  D(s) = s² + (ω0/Q)·s + ω0²   ->  ω0 = √(a0/a2),  Q = √(a0·a2)/a1
Every expected value is worked by hand in the comment above it.
"""
import math
import sys

import sympy as sp

from benchmark.run_benchmark import check_expectation
from classifier import classify
from llm_client import load_conventions
from parsing.text_parser import _norm_query, consistency_errors
from schemas import Component, Netlist
from solvers import SolverError, get_solver
from validator import validate
from verifier import Verifier

CONV = load_conventions()
SHOW = "--steps" in sys.argv
F0_1K = 1 / (2 * math.pi * 1e4 * math.sqrt(2e-16))          # 1125.395 Hz


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"\n       {detail}" if detail and not cond else ""))
    return cond


def C(id, type, nodes, value=None, **kw):
    return Component(id=id, type=type, nodes=nodes, value=value, **kw)


def U(id, out, plus, minus):
    return Component(id=id, type="OPAMP", nodes=[out, plus, minus], model="ideal")


def net(components, query, question=""):
    return Netlist(analysis="ac", query=query, components=[C("V1", "V", ["in", "0"], ac="1")] + components,
                   question=question)


def close(got, want, rel=1e-4):
    return got is not None and abs(got - want) <= rel * max(1e-12, abs(want))


def show(res):
    for st in res.trace.key_steps():
        print(f"      {st.step:>2}. {st.action}" + (f"\n          {st.equation}" if st.equation else "")
              + (f"\n          => {st.result}" if st.result else ""))


# --- the circuits (templates from the parser prompt) ---------------------------------------------
def sk_lp(R1="10k", R2="10k", C1="20n", C2="10n", gain=None, query=("filter",)):
    parts = [C("R1", "R", ["in", "a"], R1), C("R2", "R", ["a", "p"], R2), C("C1", "C", ["a", "out"], C1),
             C("C2", "C", ["p", "0"], C2)]
    if gain:                                       # K = 1 + RB/RA
        parts += [C("RA", "R", ["n1", "0"], gain[0]), C("RB", "R", ["n1", "out"], gain[1]), U("U1", "out", "p", "n1")]
    else:
        parts += [U("U1", "out", "p", "out")]
    return net(parts, list(query))


def sk_hp(query=("filter",)):
    return net([C("C1", "C", ["in", "a"], "10n"), C("C2", "C", ["a", "p"], "10n"), C("R1", "R", ["a", "out"], "10k"),
                C("R2", "R", ["p", "0"], "20k"), U("U1", "out", "p", "out")], list(query))


def mfb_lp(query=("filter",)):
    return net([C("R1", "R", ["in", "a"], "10k"), C("C1", "C", ["a", "0"], "45n"), C("R2", "R", ["a", "out"], "10k"),
                C("R3", "R", ["a", "n1"], "10k"), C("C2", "C", ["n1", "out"], "10n"), U("U1", "out", "0", "n1")],
               list(query))


def mfb_bp(query=("filter",)):
    return net([C("R1", "R", ["in", "a"], "10k"), C("C1", "C", ["a", "n1"], "10n"), C("C2", "C", ["a", "out"], "10n"),
                C("R3", "R", ["n1", "out"], "160k"), U("U1", "out", "0", "n1")], list(query))


CASES = [
    # Unity-gain Sallen-Key low-pass: H = 1 / (R1R2C1C2·s² + C2(R1+R2)·s + 1)
    #   ω0 = 1/√(R1R2C1C2) = 1/√(1e8 × 2e-16) = 7071.07 rad/s -> f0 = 1125.40 Hz
    #   Q = √(R1R2C1C2) / (C2(R1+R2)) = 1.41421e-4 / (10n × 20k) = 0.70711 (Butterworth) -> fc = f0, K = 1
    ("Sallen-Key low-pass, unity gain (Butterworth)",
     sk_lp(query=("filter", "f0", "Qfactor", "fc")),
     {"filter.K": 1.0, "filter.f0": F0_1K, "filter.Q": 1 / math.sqrt(2), "f0": F0_1K,
      "Qfactor": 1 / math.sqrt(2), "fc": F0_1K}, ("low-pass", "Butterworth")),

    # Equal-component Sallen-Key with gain K = 1 + RB/RA = 2:  H = K / (s²R²C² + sRC(3 − K) + 1)
    #   f0 = 1/(2πRC) = 1591.55 Hz,  Q = 1/(3 − K) = 1  (peaks above K)
    #   fc (−3 dB from the DC gain K): (1 − x²)² + x²/Q² = 2 -> x⁴ − x² − 1 = 0 -> x² = (1+√5)/2
    #   fc = 1591.55 × √1.618034 = 2024.48 Hz;  low-pass bandwidth = fc
    ("Sallen-Key low-pass, equal components, K = 2 (Q = 1, peaking)",
     sk_lp(C1="10n", gain=("10k", "10k"), query=("filter", "fc", "bandwidth")),
     {"filter.K": 2.0, "filter.f0": 1591.5494309, "filter.Q": 1.0,
      "fc": 1591.5494309 * math.sqrt((1 + math.sqrt(5)) / 2), "bandwidth": 1591.5494309 * math.sqrt((1 + math.sqrt(5)) / 2)},
     ("low-pass",)),

    # Unity-gain Sallen-Key high-pass: H = s² / (s² + s(C1+C2)/(R2C1C2) + 1/(R1R2C1C2))
    #   C1 = C2 = C: ω0 = 1/(C√(R1R2)) = 1/(1e-8 × √2e8) = 7071.07 rad/s -> f0 = 1125.40 Hz
    #   Q = ½√(R2/R1) = ½√2 = 0.70711 -> fc = f0; K = H(∞) = 1. Bandwidth of a high-pass -> not finite.
    ("Sallen-Key high-pass, unity gain (Butterworth)",
     sk_hp(query=("filter", "fc", "bandwidth")),
     {"filter.K": 1.0, "filter.f0": F0_1K, "filter.Q": 1 / math.sqrt(2), "fc": F0_1K}, ("high-pass",)),

    # MFB low-pass: H = −(1/(R1R3C1C2)) / (s² + s(1/C1)(1/R1 + 1/R2 + 1/R3) + 1/(R2R3C1C2))
    #   K = −R2/R1 = −1;  all R = 10k: ω0 = 1/(R√(C1C2)) = 1/(1e4 × 2.1213e-8) = 4714.05 -> f0 = 750.26 Hz
    #   Q = √(C1/C2)/3 = √4.5/3 = 0.70711
    ("MFB low-pass (inverting, Butterworth)",
     mfb_lp(query=("filter", "f0", "Qfactor")),
     {"filter.K": -1.0, "filter.f0": 750.2635968, "filter.Q": 1 / math.sqrt(2), "f0": 750.2635968,
      "Qfactor": 1 / math.sqrt(2)}, ("low-pass",)),

    # MFB band-pass: H = −(s/(R1C)) / (s² + s·2/(R3C) + 1/(R1R3C²))  with C1 = C2 = C = 10n
    #   ω0 = 1/(C√(R1R3)) = 1/(1e-8 × 4e4) = 2500 rad/s -> f0 = 397.887 Hz
    #   Q = ½√(R3/R1) = ½ × 4 = 2;  BW = f0/Q = 198.944 Hz;  K = H(jω0) = −R3/(2R1) = −8
    ("MFB band-pass (Q = 2, K = −8)",
     mfb_bp(query=("filter", "f0", "Qfactor", "bandwidth")),
     {"filter.K": -8.0, "filter.f0": 397.8873577, "filter.Q": 2.0, "f0": 397.8873577, "Qfactor": 2.0,
      "bandwidth": 198.9436789}, ("band-pass",)),

    # Passive notch: R from in to out, then L and C in series from out to ground.
    #   H = (s²LC + 1) / (s²LC + sRC + 1) -> notch, K = 1, ω0 = 1/√(LC) = 1e4 -> f0 = 1591.55 Hz
    #   Q = ω0L/R = 1e4 × 0.01/100 = 1;  BW = f0/Q = 1591.55 Hz
    ("Passive RLC notch",
     net([C("R1", "R", ["in", "out"], "100"), C("L1", "L", ["out", "x"], "10m"), C("C1", "C", ["x", "0"], "1u")],
         ["filter", "f0", "Qfactor", "bandwidth"]),
     {"filter.K": 1.0, "filter.f0": 1591.5494309, "filter.Q": 1.0, "bandwidth": 1591.5494309}, ("notch",)),

    # First-order RC low-pass: K = 1, fc = 1/(2πRC) = 1/(2π × 1e-3) = 159.155 Hz
    ("First-order RC low-pass identified",
     net([C("R1", "R", ["in", "out"], "1k"), C("C1", "C", ["out", "0"], "1u")], ["filter", "fc"]),
     {"filter.K": 1.0, "filter.fc": 159.1549431, "fc": 159.1549431}, ("first-order low-pass",)),
]


def solver_checks(v: Verifier) -> bool:
    ok = True
    if v.exe is None:
        print("[SKIP] ngspice not found: verification checks skipped")
    for name, nl, expected, words in CASES:
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
            ok &= check(f"{name}: {key} = {want:.7g}", close(res.numeric.get(key), want), f"got {res.numeric.get(key)}")
        ans = res.answers.get("filter", "")
        ok &= check(f"{name}: 'filter' answer names {words}", all(w in ans for w in words), ans)
        if v.exe:
            rep = v.verify(nl, res)
            print(rep.table())
            ok &= check(f"{name}: verified by ngspice", rep.verified is True, "; ".join(res.verification_notes))
    return ok


def formula_checks() -> bool:
    """The design formulas come from the circuit's own symbolic H(s): compare them with the textbook ones."""
    ok = True
    print()
    R1, R2, R3, C1, C2, RA, RB = sp.symbols("R1 R2 R3 C1 C2 RA RB", positive=True)
    loc = {str(x): x for x in (R1, R2, R3, C1, C2, RA, RB)}

    def formulas(nl):
        res = get_solver(nl).solve(nl, CONV)
        f = {k: sp.sympify(res.symbolic[k], locals=loc) for k in ("formula.w0", "formula.Q", "formula.K")
             if k in res.symbolic}
        return res, f

    res, f = formulas(sk_lp())
    ok &= check("Sallen-Key LP: ω0 = 1/√(R1·R2·C1·C2)",
                "formula.w0" in f and sp.simplify(f["formula.w0"] - 1 / sp.sqrt(R1 * R2 * C1 * C2)) == 0, str(f))
    ok &= check("Sallen-Key LP: Q = √(R1·R2·C1·C2)/(C2·(R1 + R2))",
                "formula.Q" in f and sp.simplify(f["formula.Q"] - sp.sqrt(R1 * R2 * C1 * C2) / (C2 * (R1 + R2))) == 0,
                str(f))
    step = [st for st in res.trace.key_steps() if st.action.startswith("Design formulas")]
    ok &= check("design-formula step is a key step", bool(step), str([st.action for st in res.trace.key_steps()]))
    if step:
        print("     " + step[0].equation)

    # with gain resistors: textbook Q = √(R1R2C1C2) / (C2(R1+R2) + R1C1(1 − K)),  K = 1 + RB/RA
    res, f = formulas(sk_lp(C1="10n", gain=("10k", "10k")))
    K = 1 + RB / RA
    ok &= check("Sallen-Key LP with gain: Q = √(R1R2C1C2)/(C2(R1+R2) + R1C1(1 − K)), K = 1 + RB/RA",
                "formula.Q" in f and sp.simplify(f["formula.Q"] - sp.sqrt(R1 * R2 * C1 * C2) /
                                                 (C2 * (R1 + R2) + R1 * C1 * (1 - K))) == 0
                and sp.simplify(f["formula.K"] - K) == 0, str(f))

    res, f = formulas(mfb_bp())
    ok &= check("MFB BP: ω0 = 1/√(R1·R3·C1·C2), K = −R3·C1/(R1·(C1 + C2))",
                "formula.w0" in f and sp.simplify(f["formula.w0"] - 1 / sp.sqrt(R1 * R3 * C1 * C2)) == 0
                and sp.simplify(f["formula.K"] + R3 * C1 / (R1 * (C1 + C2))) == 0, str(f))

    # non-ideal op-amp -> no formula step (it would be unreadable), numbers still exact
    nl = sk_lp()
    nl.components[-1] = Component(id="U1", type="OPAMP", nodes=["out", "p", "out"], params={"A0": 1e5})
    res = get_solver(nl).solve(nl, CONV)
    ok &= check("non-ideal op-amp: no design-formula step", "formula.w0" not in res.symbolic)
    return ok


def misc_checks() -> bool:
    ok = True
    print()
    # integrator: pole at s = 0 -> not a standard filter, but no crash
    integ = net([C("R1", "R", ["in", "n1"], "10k"), C("C1", "C", ["n1", "out"], "10n"), U("U1", "out", "0", "n1")],
                ["filter"])
    res = get_solver(integ).solve(integ, CONV)
    ok &= check("integrator -> 'not a standard type'", "not a standard type" in res.answers.get("filter", ""),
                str(res.answers))
    ans = get_solver(CASES[2][1]).solve(CASES[2][1], CONV).answers.get("bandwidth", "")
    ok &= check("high-pass bandwidth -> clear 'not finite' message", "not finite" in ans, ans)

    # pass-band gain is reported even when the parser asks "gain" / "H(out,in)" instead of "filter"
    # (L2 and L3 of the first --live run): K = 2 for the equal-component SK, K = −8 for the MFB band-pass
    nl = sk_lp(C1="10n", gain=("10k", "10k"), query=("gain",))
    res = get_solver(nl).solve(nl, CONV)
    ok &= check("SK low-pass 'gain' query -> gain.K = 2", close(res.numeric.get("gain.K"), 2.0), str(res.numeric))
    nl = mfb_bp(query=("H(out,in)",))
    res = get_solver(nl).solve(nl, CONV)
    ans = res.answers.get("H(out,in)", "")
    ok &= check("MFB band-pass 'H(out,in)' -> H(out,in).K = −8 and 'pass-band gain' in the answer",
                close(res.numeric.get("H(out,in).K"), -8.0) and "pass-band gain K = -8" in ans, ans)
    good, detail = check_expectation({"kind": "K", "value": -8.0}, res.numeric, res.symbolic, 0.01)
    ok &= check("benchmark kind 'K' matches H(out,in).K", good, detail)

    ok &= check("'filter_type' / 'type' normalise to the 'filter' query",
                _norm_query("filter_type") == "filter" and _norm_query("type") == "filter")
    r = validate(sk_lp())
    ok &= check("validator accepts the 'filter' query", not r.errors, str(r.errors))
    c = classify(Netlist(analysis="ac", query=["f0"], components=sk_lp().components,
                         question="Find f0 of the Sallen-Key circuit."))
    ok &= check("classifier: Sallen-Key -> topic filter", c.topic == "filter", str(c))
    return ok


def parser_checks() -> bool:
    ok = True
    print()
    q = ("A unity-gain Sallen-Key low-pass filter has R1 = R2 = 10 kOhm, C1 = 20 nF from the junction of R1 and R2 "
         "to the output and C2 = 10 nF from the non-inverting input to ground. Find f0 and Q.")
    good = sk_lp(query=("f0", "Qfactor"))
    e = consistency_errors(q, good.model_copy(update={"question": q}))
    ok &= check("Sallen-Key LP wired by the template -> consistent", not e, str(e))

    # R and C swapped -> it is a high-pass now, so the "low-pass" in the question must reject it
    swapped = net([C("C1", "C", ["in", "a"], "20n"), C("C2", "C", ["a", "p"], "10n"), C("R1", "R", ["a", "out"], "10k"),
                   C("R2", "R", ["p", "0"], "10k"), U("U1", "out", "p", "out")], ["f0"], q)
    e = consistency_errors(q, swapped)
    ok &= check("'low-pass' question but a high-pass circuit -> rejected",
                any("low-pass filter" in x and "high-pass response" in x for x in e), str(e))

    # + input grounded (that is an MFB shape) for a Sallen-Key question
    e = consistency_errors(q, mfb_lp(query=("f0",)).model_copy(update={"question": q}))
    ok &= check("Sallen-Key question with the + input grounded -> rejected", any("Sallen-Key" in x for x in e), str(e))

    qm = ("A multiple-feedback band-pass filter has R1 = 10 kOhm from the input to node A, C1 = 10 nF from A to the "
          "inverting input, C2 = 10 nF from A to the output and R3 = 160 kOhm from the inverting input to the output. "
          "Find the centre frequency.")
    e = consistency_errors(qm, mfb_bp(query=("f0",)).model_copy(update={"question": qm}))
    ok &= check("MFB band-pass wired by the template -> consistent", not e, str(e))
    e = consistency_errors(qm, sk_lp(query=("f0",)).model_copy(update={"question": qm}))
    ok &= check("MFB question with a Sallen-Key circuit -> rejected", any("multiple-feedback" in x for x in e), str(e))

    # Phase 1 benefit: an RC "high-pass" parsed with C to ground (a low-pass) is now caught
    qh = "An RC high-pass filter has C = 0.47 uF and R = 1 kOhm. Find the cutoff frequency."
    wrong = net([C("R1", "R", ["in", "out"], "1k"), C("C1", "C", ["out", "0"], "0.47u")], ["fc"], qh)
    e = consistency_errors(qh, wrong)
    ok &= check("RC 'high-pass' parsed as a low-pass -> rejected", any("high-pass filter" in x for x in e), str(e))
    right = net([C("C1", "C", ["in", "out"], "0.47u"), C("R1", "R", ["out", "0"], "1k")], ["fc"], qh)
    ok &= check("RC high-pass wired right -> consistent", not consistency_errors(qh, right))
    return ok


LIVE = [
    ("A unity-gain Sallen-Key low-pass filter has R1 = R2 = 10 kOhm, C1 = 20 nF from the junction of R1 and R2 to "
     "the output, and C2 = 10 nF from the non-inverting input to ground. Find the natural frequency f0 and the "
     "Q factor.",
     [{"kind": "f0", "value": F0_1K}, {"kind": "Qfactor", "value": 1 / math.sqrt(2)}]),
    ("An equal-component Sallen-Key low-pass filter uses R = 10 kOhm and C = 10 nF. Its non-inverting amplifier "
     "has RA = 10 kOhm from the inverting input to ground and RB = 10 kOhm from the output to the inverting input. "
     "Find f0, Q and the pass-band gain.",
     [{"kind": "f0", "value": 1591.5494309}, {"kind": "Qfactor", "value": 1.0}, {"kind": "K", "value": 2.0}]),
    ("A multiple-feedback band-pass filter has R1 = 10 kOhm from the input to node A, C1 = 10 nF from A to the "
     "inverting input, C2 = 10 nF from A to the output and R3 = 160 kOhm from the inverting input to the output; "
     "the non-inverting input is grounded. Find the centre frequency, Q, the bandwidth and the gain at the centre "
     "frequency.",
     [{"kind": "f0", "value": 397.8873577}, {"kind": "Qfactor", "value": 2.0},
      {"kind": "bandwidth", "value": 198.9436789}, {"kind": "K", "value": -8.0}]),
    ("A unity-gain Sallen-Key high-pass filter has C1 = C2 = 10 nF, R1 = 10 kOhm from the junction of the two "
     "capacitors to the output and R2 = 20 kOhm from the non-inverting input to ground. Find the cutoff frequency.",
     [{"kind": "fc", "value": F0_1K}]),
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
            good, detail = check_expectation(e, res.get("numeric") or {}, res.get("symbolic") or {}, 0.01)
            passed &= good
            print("  ", detail)
        ok &= check(f"L{i} correct and verified", passed and res.get("verified") is True,
                    f"status={r.status}, verified={res.get('verified')}")
    print("\n" + ai.client.usage_report())
    return ok


if __name__ == "__main__":
    ver = Verifier()
    all_ok = solver_checks(ver)
    all_ok &= formula_checks()
    all_ok &= misc_checks()
    all_ok &= parser_checks()
    if "--live" in sys.argv:
        all_ok &= live_checks()
    print("\nPHASE 2 STEP 4:", "ALL PASSED" if all_ok else "SOME CHECKS FAILED")