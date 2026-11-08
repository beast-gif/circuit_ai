"""
Phase 2, Step 3 checks: non-ideal op-amps (finite gain A0, GBW pole, Rin, Rout) and the saturation warning.

    python test_p2_3.py           # hand-worked circuits + ngspice verification (free, no LLM)
    python test_p2_3.py --steps   # also print the key steps of each circuit
    python test_p2_3.py --live    # 3 real questions: parse -> solve -> ngspice verify (Velona)

Every expected value is worked by hand in the comment above it.
Closed-loop formula used throughout:  gain = G_ideal × A0β / (1 + A0β),  β = feedback fraction.
"""
import sys

from benchmark.run_benchmark import check_expectation
from explainer import render_verification
from llm_client import load_conventions
from parsing.prompts import TEXT_PARSER_SYSTEM
from schemas import Component, Netlist
from solvers import SolverError, get_solver
from solvers.mna import resolve_opamp
from verifier import Verifier

CONV = load_conventions()
SHOW = "--steps" in sys.argv


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"\n       {detail}" if detail and not cond else ""))
    return cond


def C(id, type, nodes, value=None, **kw):
    return Component(id=id, type=type, nodes=nodes, value=value, **kw)


def U(id, out, plus, minus, model=None, **params):
    """Op-amp, nodes = [out, in+, in-]. With params (A0, GBW, Rin, Rout) it is non-ideal."""
    return Component(id=id, type="OPAMP", nodes=[out, plus, minus], model=model or ("" if params else "ideal"),
                     params=params)


def net(components, query, analysis="dc", question="", assumptions=()):
    return Netlist(analysis=analysis, query=query, components=components, question=question,
                   assumptions=list(assumptions))


def close(got, want, rel=1e-6):
    return got is not None and abs(got - want) <= rel * max(1e-12, abs(want))


def show(res):
    for st in res.trace.key_steps():
        print(f"      {st.step:>2}. {st.action}" + (f"\n          {st.equation}" if st.equation else "")
              + (f"\n          => {st.result}" if st.result else ""))


def noninv(op, query=("V(out)",), analysis="dc", src=None):
    """Non-inverting amp, R1 = 1k to ground, Rf = 9k  ->  β = 1k/(1k+9k) = 0.1, ideal gain 10."""
    src = src or C("V1", "V", ["in", "0"], "1")
    return net([src, op, C("R1", "R", ["n1", "0"], "1k"), C("R2", "R", ["n1", "out"], "9k")],
               list(query), analysis)


def inv(op, query=("V(out)",), analysis="dc", src=None, **kw):
    """Inverting amp, R1 = 1k, Rf = 10k  ->  β = R1/(R1+Rf) = 1/11, ideal gain -10."""
    src = src or C("V1", "V", ["in", "0"], "1")
    return net([src, C("R1", "R", ["in", "n1"], "1k"), C("R2", "R", ["n1", "out"], "10k"), op],
               list(query), analysis, **kw)


AC_IN = C("V1", "V", ["in", "0"], ac="1")

CASES = [
    # Non-inverting, A0 = 1e5, β = 0.1:  Vout = A0/(1 + A0β) = 1e5/10001 = 9.99900010  (ideal: 10)
    ("Non-inverting, finite A0 = 1e5",
     noninv(U("U1", "out", "in", "n1", A0=100000), ("V(out)", "gain")),
     {"V(out)": 1e5 / 10001, "gain": 1e5 / 10001}),

    # Inverting, A0 = 1e4:  Vout = -(Rf/R1) / (1 + (1 + Rf/R1)/A0) = -10 / (1 + 11/1e4) = -9.98901209
    # The "virtual ground" is no longer exactly 0:  V(n1) = -Vout/A0 = +9.98901e-4 V
    ("Inverting, finite A0 = 1e4",
     inv(U("U1", "out", "0", "n1", A0=10000), ("V(out)", "V(n1)")),
     {"V(out)": -10 / (1 + 11 / 1e4), "V(n1)": (10 / (1 + 11 / 1e4)) / 1e4}),

    # A0 given in dB: 100 dB = 10^(100/20) = 1e5 -> same answer as the first case
    ("Non-inverting, A0 = '100 dB'",
     noninv(U("U1", "out", "in", "n1", A0="100 dB")),
     {"V(out)": 1e5 / 10001}),

    # A named part with no data: model "741" -> A0 = 1e5 and GBW = 1 MHz from conventions.yaml.
    # A DC question ignores the pole, so Vout = 9.99900010 again.
    ("741 with no data (typical values assumed), DC",
     noninv(U("U1", "out", "in", "n1", model="741")),
     {"V(out)": 1e5 / 10001}),

    # GBW: A(s) = A0/(1 + s/ωp), fp = GBW/A0 = 1e6/1e5 = 10 Hz.
    # H(s) = A0 / (1 + A0β + s/ωp)  -> DC gain 9.9990001, closed-loop pole fc = fp(1 + A0β) = 10 × 10001 = 100010 Hz
    # (rule of thumb: GBW / closed-loop gain = 1 MHz / 10 = 100 kHz)
    ("Non-inverting, A0 = 1e5, GBW = 1 MHz: bandwidth",
     noninv(U("U1", "out", "in", "n1", A0=100000, GBW="1 MHz"), ("H(out,in)", "fc"), "ac", AC_IN),
     {"H(out,in).dc_gain": 1e5 / 10001, "fc": 100010.0}),

    # Inverting with GBW: β = 1/11, fc = fp(1 + A0β) = 10 × (1 + 1e5/11) = 90919.09 Hz
    # (smaller than the non-inverting gain-10 case: the noise gain is 11, not 10)
    # DC gain = -10/(1 + 11/1e5) = -9.99890012
    ("Inverting, A0 = 1e5, GBW = 1 MHz: bandwidth",
     inv(U("U1", "out", "0", "n1", A0=100000, GBW=1e6), ("H(out,in)", "fc"), "ac", AC_IN),
     {"H(out,in).dc_gain": -10 / (1 + 11 / 1e5), "fc": 10 * (1 + 1e5 / 11)}),

    # Same circuit asked for "bandwidth" (how L2 of --live was parsed): it is a low-pass, so the band is
    # 0 → fc and BW = fc = 90919.09 Hz. ngspice checks |H(BW)|/|H|max = 1/√2.
    ("Inverting, A0 = 1e5, GBW = 1 MHz: 'bandwidth' query",
     inv(U("U1", "out", "0", "n1", A0=100000, GBW=1e6), ("bandwidth",), "ac", AC_IN),
     {"bandwidth": 10 * (1 + 1e5 / 11)}),

    # Follower with Rout = 75 Ω into RL = 100 Ω, A0 = 1e5, Vin = 1:
    #   Vout = A0(1 - Vout) + Rout·I_U, I_U = -Vout/RL  ->  Vout = A0 / (1 + A0 + Rout/RL) = 1e5/100001.75
    #   I(U1) delivered = Vout/RL = 9.999825e-3 A
    ("Follower, A0 = 1e5, Rout = 75 Ω, RL = 100 Ω",
     net([C("V1", "V", ["in", "0"], "1"), U("U1", "out", "in", "out", A0=100000, Rout=75),
          C("RL", "R", ["out", "0"], "100")], ["V(out)", "I(U1)"]),
     {"V(out)": 1e5 / 100001.75, "I(U1)": 1e5 / 100001.75 / 100}),

    # Non-inverting with Rin = 1 MΩ between the inputs, A0 = 1e5 (β = 0.1):
    #   Input resistance (1 A test current into "in"): vd = I·Rin, V- = I(900 + 0.1·A0·Rin) ... (see P2-3 notes)
    #   Rin_closed = 900 + Rin(1 + A0β) = 900 + 1e6 × 10001 = 1.00010009e10 Ω   (≈ Rin(1 + A0β))
    #   Vout (KCL at n1 with the Rin current, ×9e6): x = V- = (9 + 1e8)/(10009 + 1e8),
    #   Vout = A0(1 - x) = 1e5 × 1e4 / 100010009 = 9.99899920
    ("Non-inverting, A0 = 1e5, Rin = 1 MΩ: input resistance",
     noninv(U("U1", "out", "in", "n1", A0=100000, Rin="1Meg"), ("V(out)", "Rin")),
     {"V(out)": 1e9 / 100010009, "Rin": 900 + 1e6 * 10001}),
]


def solver_checks(v: Verifier) -> bool:
    ok = True
    if v.exe is None:
        print("[SKIP] ngspice not found: verification checks skipped")
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
            ok &= check(f"{name}: {key} = {want:.9g}", close(res.numeric.get(key), want),
                        f"got {res.numeric.get(key)}")
        if v.exe:
            rep = v.verify(nl, res)
            print(rep.table())
            ok &= check(f"{name}: verified by ngspice", rep.verified is True, "; ".join(res.verification_notes))
    return ok


def model_checks() -> bool:
    ok = True
    print()
    r = resolve_opamp(U("U1", "out", "in", "n1"), CONV)
    ok &= check("model 'ideal', no params -> ideal (None)", r is None, str(r))
    r = resolve_opamp(U("U1", "out", "in", "n1", A0="100 dB"), CONV)
    ok &= check("A0 = '100 dB' -> 1e5", r is not None and close(r["A0"], 1e5), str(r))
    r = resolve_opamp(U("U1", "out", "in", "n1", Aol=2e5, ft="1 MHz", ro="75", ri="2Meg"), CONV)
    ok &= check("aliases Aol / ft / ro / ri -> A0 / GBW / Rout / Rin",
                r is not None and r["A0"] == 2e5 and r["GBW"] == 1e6 and r["Rout"] == 75 and r["Rin"] == 2e6, str(r))
    r = resolve_opamp(U("U1", "out", "in", "n1", model="741"), CONV)
    ok &= check("model '741', no params -> A0 1e5 and GBW 1 MHz assumed, with notes",
                r is not None and r["A0"] == 1e5 and r["GBW"] == 1e6 and len(r["notes"]) == 2, str(r))
    r = resolve_opamp(U("U1", "out", "in", "n1", model="nonideal"), CONV)
    ok &= check("model 'nonideal', no params -> A0 assumed, no GBW pole",
                r is not None and r["A0"] == 1e5 and r["GBW"] is None, str(r))
    r = resolve_opamp(U("U1", "out", "in", "n1", Rout=50), CONV)
    ok &= check("only Rout given -> A0 assumed from conventions", r is not None and r["A0"] == 1e5 and
                r["Rout"] == 50 and any("assumed" in n for n in r["notes"]), str(r))

    # The key steps state the model and the assumption
    nl = noninv(U("U1", "out", "in", "n1", model="741"))
    res = get_solver(nl).solve(nl, CONV)
    model_step = [st for st in res.trace.key_steps() if st.action.startswith("Non-ideal op-amp U1 model")]
    ok &= check("key steps state the 741 model (A0, GBW, fp = 10 Hz) and that values were assumed",
                bool(model_step) and "10 Hz" in model_step[0].action and "assumed" in (model_step[0].result or ""),
                str([(st.action, st.result) for st in res.trace.key_steps()]))
    ok &= check("DC question with a 741: no 's-domain' step (the pole does not matter in DC)",
                not any("s-domain" in st.action for st in res.trace.all_steps()))
    gain_q = noninv(U("U1", "out", "in", "n1", model="741"), ("gain",))
    res = get_solver(gain_q).solve(gain_q, CONV)
    ok &= check("DC 'gain' with a 741 is a number (9.999), not H(s)", close(res.numeric.get("gain"), 1e5 / 10001),
                str(res.answers))
    eq = [st for st in res.trace.key_steps() if st.action.startswith("Non-ideal op-amp U1: output")]
    ok &= check("key steps show 'Non-ideal op-amp U1: output = gain × (V+ − V−)' (no virtual short)",
                bool(eq) and not any("virtual short" in st.action for st in res.trace.all_steps()),
                str([st.action for st in res.trace.key_steps()]))
    if SHOW:
        show(res)

    ok &= check("parser prompt lists the op-amp params keys and the named-part rule",
                all(k in TEXT_PARSER_SYSTEM for k in ('"A0"', '"GBW"', '"Rin"', '"Rout"', "741")))
    return ok


def saturation_checks(v: Verifier) -> bool:
    ok = True
    print()
    ideal = U("U1", "out", "0", "n1")

    # Inverting gain -10 with 2 V in -> linear answer -20 V, beyond a ±12 V supply -> warning
    nl = inv(ideal, src=C("V1", "V", ["in", "0"], "2"), assumptions=["supply +/-12 V"])
    res = get_solver(nl).solve(nl, CONV)
    sat = [n for n in res.verification_notes if n.startswith("saturation:")]
    ok &= check("-20 V needed with ±12 V rails -> saturation note", close(res.numeric.get("V(out)"), -20.0) and
                bool(sat) and "−12 V" in sat[0], str(res.verification_notes))
    ok &= check("saturation also shown as a key step",
                any(st.action.startswith("⚠️ Saturation check") for st in res.trace.key_steps()))
    if v.exe:
        v.verify(nl, res)
    text = render_verification(res)
    ok &= check("explainer shows the saturation warning as a ⚠️ line", "⚠️ U1 would saturate" in text, text)
    if v.exe:
        ok &= check("saturation note does not make verification fail (the linear answer still matches)",
                    res.verified is True, str(res.verification_notes))

    # Same circuit, 1 V in -> -10 V, inside ±15 V -> no warning
    nl = inv(ideal, src=C("V1", "V", ["in", "0"], "1"), question="Supply is ±15 V.")
    res = get_solver(nl).solve(nl, CONV)
    ok &= check("-10 V with ±15 V rails -> no saturation note",
                not any(n.startswith("saturation:") for n in res.verification_notes), str(res.verification_notes))

    # Rails written as Vcc / Vee in the question; only the + rail is exceeded here (+20 V > 15 V)
    nl = inv(ideal, src=C("V1", "V", ["in", "0"], "-2"), question="Vcc = 15 V and Vee = -15 V.")
    res = get_solver(nl).solve(nl, CONV)
    sat = [n for n in res.verification_notes if n.startswith("saturation:")]
    ok &= check("+20 V with Vcc = 15 V -> saturation note on the + rail", bool(sat) and "+15 V" in sat[0],
                str(res.verification_notes))

    # No supply given -> no check at all (we do not invent rails)
    nl = inv(ideal, src=C("V1", "V", ["in", "0"], "5"))
    res = get_solver(nl).solve(nl, CONV)
    ok &= check("no supply stated -> no saturation note even at -50 V",
                not any(n.startswith("saturation:") for n in res.verification_notes))
    return ok


LIVE = [
    ("A non-inverting amplifier has R1 = 1 kOhm from the inverting input to ground and Rf = 9 kOhm from the output "
     "to the inverting input. The op-amp has a finite open-loop gain of 10^4. The input is 1 V DC. "
     "Find the output voltage.",
     [{"kind": "V", "value": 1e4 / 1001}], False),                     # 9.99001 (ideal would be 10)
    ("An inverting amplifier uses R1 = 1 kOhm and Rf = 10 kOhm. The op-amp has an open-loop gain of 100 dB and a "
     "gain-bandwidth product of 1 MHz. Find the closed-loop -3 dB bandwidth.",
     [{"kind": "bandwidth", "value": 10 * (1 + 1e5 / 11)}], False),    # 90.919 kHz (answered as bandwidth or fc)
    ("An inverting amplifier with R1 = 1 kOhm and Rf = 10 kOhm is powered from +/-12 V supplies. "
     "The input is 2 V DC. Find the output voltage.",
     [{"kind": "V", "value": -20.0}], True),                            # linear -20 V + saturation warning
]


def live_checks() -> bool:
    from graph import CircuitAI

    ai = CircuitAI(auto_confirm=True, do_explain=False)
    ok = True
    for i, (question, expect, want_sat) in enumerate(LIVE, 1):
        print(f"\n=== L{i}: {question}")
        r = ai.ask(question)
        st = r.state
        if st.get("netlist"):
            print(Netlist.model_validate(st["netlist"]).to_spice())
            print("query:", st["netlist"].get("query"), "| assumptions:", st["netlist"].get("assumptions"))
            print("op-amp params:", [(c["id"], c.get("model"), c.get("params")) for c in st["netlist"]["components"]
                                     if c["type"] == "OPAMP"])
        for h in st.get("parse_history") or []:
            print(f"   rejected attempt {h['attempt']} at {h['stage']}: {h['errors'][0][:140]}")
        res = st.get("result") or {}
        print("answers:", res.get("answers"), "| verified:", res.get("verified"))
        passed = r.status == "answered"
        for e in expect:                                   # tight tolerance: must differ from the ideal answer
            good, detail = check_expectation(e, res.get("numeric") or {}, res.get("symbolic") or {}, 1e-4)
            passed &= good
            print("  ", detail)
        sat = any(str(n).startswith("saturation:") for n in res.get("verification_notes") or [])
        if want_sat:
            print("   saturation note:", "yes" if sat else "NO")
            passed &= sat
        ok &= check(f"L{i} correct and verified", passed and res.get("verified") is True,
                    f"status={r.status}, verified={res.get('verified')}")
    print("\n" + ai.client.usage_report())
    return ok


if __name__ == "__main__":
    ver = Verifier()
    all_ok = solver_checks(ver)
    all_ok &= model_checks()
    all_ok &= saturation_checks(ver)
    if "--live" in sys.argv:
        all_ok &= live_checks()
    print("\nPHASE 2 STEP 3:", "ALL PASSED" if all_ok else "SOME CHECKS FAILED")