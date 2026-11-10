"""
Phase 4, Step 1 checks: op-amp RC oscillators (Wien bridge, RC phase shift) via det A(s) = 0.

    python test_p4_1.py           # 8 hand-worked oscillators + step / validator / parser checks (free)
    python test_p4_1.py --steps   # also print the key steps
    python test_p4_1.py --live    # 5 real questions: parse -> solve (Velona)

Hand-worked with R = 10 kOhm and C = 10 nF, so 1/RC = 10^4 rad/s:
    Wien bridge        char. poly s^2 + 10^8  ->  poles ±j10^4,  f0 = 1/(2 pi R C)   = 1591.5494 Hz
                       with R2 = 15 k the poles move to -2500 ± j9682.46 (decays, no oscillation)
                       with R2 = 30 k they move to +5000 ± j8660.25 (grows, rails limit it)
                       with R2 symbolic the two Barkhausen equations give R2 = 20 k exactly
    RC phase shift     3 CR sections, last R into the virtual ground:
                       f0 = 1/(2 pi R C sqrt 6) = 649.7473 Hz  and  Rf/R = 29  ->  Rf = 290 k
"""
import math
import sys

from benchmark.run_benchmark import check_expectation
from llm_client import load_conventions
from parsing.text_parser import consistency_errors
from schemas import Component, Netlist
from solvers import NotSupportedError, SolverError, get_solver
from validator import validate

CONV = load_conventions()
SHOW = "--steps" in sys.argv
F_WIEN = 1 / (2 * math.pi * 10e3 * 10e-9)                   # 1591.5494 Hz
F_PS = F_WIEN / math.sqrt(6)                                # 649.7473 Hz


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"\n       {detail}" if detail and not cond else ""))
    return cond


def C(id, type, nodes, value=None, **kw):
    return Component(id=id, type=type, nodes=nodes, value=value, **kw)


def U(id, out, plus, minus):
    return Component(id=id, type="OPAMP", nodes=[out, plus, minus], model="ideal")


def close(got, want, rel=1e-6):
    return got is not None and abs(got - want) <= rel * max(abs(want), 1e-12)


def show(res, full=False):
    for st in (res.trace.all_steps() if full else res.trace.key_steps()):
        print(f"      {st.step:>2}. {st.action}" + (f"\n          {st.equation}" if st.equation else "")
              + (f"\n          => {st.result}" if st.result else ""))


# Wien bridge: RC network to the + input, R1/R2 divider at the − input
def wien(r2="20k", query=None):
    return Netlist(analysis="ac", query=query or ["fosc"], question="A Wien-bridge oscillator.",
                   components=[C("R1", "R", ["n", "0"], "10k"), C("R2", "R", ["n", "out"], r2),
                               C("Rs", "R", ["out", "x"], "10k"), C("Cs", "C", ["x", "p"], "10n"),
                               C("Rp", "R", ["p", "0"], "10k"), C("Cp", "C", ["p", "0"], "10n"),
                               U("U1", "out", "p", "n")])


# RC phase-shift: three CR sections from the output back to the − input (a virtual ground)
def phase_shift(rf="290k", query=None):
    return Netlist(analysis="ac", query=query or ["fosc"], question="An RC phase-shift oscillator.",
                   components=[C("Rf", "R", ["n", "out"], rf),
                               C("C1", "C", ["out", "a"], "10n"), C("Ra", "R", ["a", "0"], "10k"),
                               C("C2", "C", ["a", "b"], "10n"), C("Rb", "R", ["b", "0"], "10k"),
                               C("C3", "C", ["b", "c"], "10n"), C("Rc", "R", ["c", "n"], "10k"),
                               U("U1", "out", "0", "n")])


CASES = [
    # 1. Wien bridge at exactly the Barkhausen condition (R2/R1 = 2): poles on the jw axis
    ("Wien bridge, R2/R1 = 2", wien(), {"fosc": F_WIEN, "pole.re": 0.0}),

    # 2. The same bridge with the gain too LOW: the poles move left, so it does not oscillate
    ("Wien bridge with R2 = 15 k (gain too low)", wien("15k", ["fosc", "oscillates", "gain_condition"]),
     {"pole.re": -2500.0, "gain_condition": 20e3}),

    # 3. Gain too HIGH: the poles move right, it oscillates and the rails limit the amplitude
    ("Wien bridge with R2 = 30 k (gain too high)", wien("30k", ["fosc", "oscillates"]),
     {"pole.re": 5000.0}),

    # 4. R2 left as a symbol: the two Barkhausen equations give the frequency AND the resistor
    ("Wien bridge with R2 unknown", wien("Rf", ["fosc", "gain_condition"]),
     {"fosc": F_WIEN, "gain_condition": 20e3}),

    # 5. RC phase-shift oscillator, Rf = 290 k = 29 R: on the condition
    ("RC phase-shift, Rf = 29 R", phase_shift(), {"fosc": F_PS, "pole.re": 0.0}),

    # 6. RC phase-shift with Rf unknown -> Rf = 29 R comes out of the determinant
    ("RC phase-shift with Rf unknown", phase_shift("Rf", ["fosc", "gain_condition"]),
     {"fosc": F_PS, "gain_condition": 290e3}),

    # 7. A different RC: R = 4.7 k, C = 22 nF -> f0 = 1/(2 pi R C) = 1539.3 Hz
    ("Wien bridge with R = 4.7 k, C = 22 nF",
     Netlist(analysis="ac", query=["fosc"], question="A Wien-bridge oscillator.",
             components=[C("R1", "R", ["n", "0"], "10k"), C("R2", "R", ["n", "out"], "20k"),
                         C("Rs", "R", ["out", "x"], "4.7k"), C("Cs", "C", ["x", "p"], "22n"),
                         C("Rp", "R", ["p", "0"], "4.7k"), C("Cp", "C", ["p", "0"], "22n"),
                         U("U1", "out", "p", "n")]),
     {"fosc": 1 / (2 * math.pi * 4.7e3 * 22e-9)}),

    # 8. Barkhausen asked for by name
    ("Barkhausen criterion of the Wien bridge", wien("20k", ["barkhausen"]), {"barkhausen": F_WIEN}),
]


def solver_checks() -> bool:
    ok = True
    for name, nl, expected in CASES:
        print(f"\n--- {name}")
        try:
            res = get_solver(nl).solve(nl, CONV)
        except (SolverError, NotSupportedError) as e:
            ok &= check(name, False, f"solver error: {type(e).__name__}: {e}")
            continue
        for q, a in res.answers.items():
            print(f"     {q:<16} = {a}")
        if SHOW:
            show(res)
        for key, want in expected.items():
            got = res.numeric.get(key)
            good = close(got, want, 1e-6) if want else (got is not None and abs(got) <= 1e-3)
            ok &= check(f"{name}: {key} = {want:.7g}", good, f"got {got}")
    return ok


def step_checks() -> bool:
    ok = True
    print()
    nl = wien("Rf", ["fosc", "gain_condition"])
    res = get_solver(nl).solve(nl, CONV)
    acts = [st.action for st in res.trace.key_steps()]
    ok &= check("a step says the circuit is source-free, so det A(s) = 0",
                any("source-free" in a for a in acts), str(acts))
    sp_ = next((st for st in res.trace.key_steps() if "real and imaginary parts" in st.action), None)
    ok &= check("a step splits det A(jw) into the phase and magnitude conditions",
                sp_ is not None and "phase condition" in (sp_.equation or "")
                and "magnitude condition" in (sp_.equation or ""), str(sp_.equation if sp_ else None))
    ok &= check("a step names the Wien bridge and compares the textbook formula",
                any("Wien bridge" in a for a in acts), str(acts))
    ok &= check("the textbook frequency is reported separately as fosc.approx",
                close(res.numeric.get("fosc.approx"), F_WIEN, 1e-9), str(res.numeric.get("fosc.approx")))
    if SHOW:
        show(res)

    nl = wien("15k", ["fosc", "oscillates"])
    res = get_solver(nl).solve(nl, CONV)
    acts = " | ".join(st.action + " " + (st.result or "") for st in res.trace.key_steps())
    ok &= check("gain too low: a step says the amplitude decays", "DECAYS" in acts, acts)
    ok &= check("gain too low: it says what the feedback resistor would have to be",
                "would put the poles on the axis" in acts, acts)
    ok &= check("gain too low: the answer to 'oscillates' is no",
                res.answers.get("oscillates", "").startswith("no"), str(res.answers))

    nl = wien("30k", ["fosc", "oscillates"])
    res = get_solver(nl).solve(nl, CONV)
    acts = " | ".join(st.action + " " + (st.result or "") for st in res.trace.key_steps())
    ok &= check("gain too high: a step says the amplitude grows and the rails limit it",
                "GROWS" in acts, acts)
    ok &= check("gain too high: the answer to 'oscillates' is yes",
                res.answers.get("oscillates", "").startswith("yes"), str(res.answers))
    return ok


def misc_checks() -> bool:
    ok = True
    print()
    ok &= check("validator accepts fosc / gain_condition / barkhausen / oscillates",
                not validate(wien("20k", ["fosc", "gain_condition", "barkhausen", "oscillates"])).errors,
                str(validate(wien("20k", ["fosc", "gain_condition", "barkhausen", "oscillates"])).errors))
    ok &= check("an oscillator goes to the oscillator solver", get_solver(wien()).name == "oscillator",
                get_solver(wien()).name)

    # a Sallen-Key filter has an op-amp and an RC network but it HAS an input source: still a filter
    flt = Netlist(analysis="ac", query=["fc"], question="A Sallen-Key low-pass filter.",
                  components=[C("V1", "V", ["in", "0"], None, ac="1"), C("R1", "R", ["in", "a"], "10k"),
                              C("R2", "R", ["a", "p"], "10k"), C("C1", "C", ["a", "out"], "10n"),
                              C("C2", "C", ["p", "0"], "10n"), U("U1", "out", "p", "out")])
    ok &= check("a filter with an input source still goes to the DC/AC solver",
                get_solver(flt).name == "dc_ac", get_solver(flt).name)

    # two unknowns cannot be solved from two equations plus the frequency
    two = wien("Rf", ["fosc", "gain_condition"])
    two.components[0].value = "R1x"
    try:
        get_solver(two).solve(two, CONV)
        ok &= check("two symbolic values -> clear error", False)
    except SolverError as e:
        ok &= check("two symbolic values -> clear error", "more than one component value" in str(e).lower(),
                    str(e))

    # an op-amp with plain negative feedback and no reactance cannot oscillate
    flat = Netlist(analysis="ac", query=["fosc"], question="Does this oscillate?",
                   components=[C("R1", "R", ["n", "0"], "10k"), C("R2", "R", ["n", "out"], "20k"),
                               C("R3", "R", ["out", "p"], "10k"), C("R4", "R", ["p", "0"], "10k"),
                               U("U1", "out", "p", "n")])
    try:
        get_solver(flat).solve(flat, CONV)
        ok &= check("no reactive element -> says it cannot oscillate", False)
    except SolverError as e:
        ok &= check("no reactive element -> says it cannot oscillate",
                    "no s in it" in str(e) or "cannot oscillate" in str(e) or "no frequency" in str(e), str(e))

    # an oscillator question that still carries a source must be rejected with a clear reason
    from solvers.oscillator import OscillatorSolver
    withsrc = wien()
    withsrc.components.append(C("V1", "V", ["in", "0"], None, ac="1"))
    good, why = OscillatorSolver().supports(withsrc)
    ok &= check("a source in an oscillator netlist is rejected with a reason",
                not good and "no input source" in why, f"{good} {why}")
    return ok


def parser_checks() -> bool:
    ok = True
    print()
    q = ("A Wien-bridge oscillator uses R = 10 kOhm and C = 10 nF in both arms, with R1 = 10 kOhm from the "
         "inverting input to ground and R2 = 20 kOhm of feedback. Find the frequency of oscillation.")
    good = wien()
    good.question = q
    ok &= check("Wien-bridge netlist -> consistent", not consistency_errors(q, good),
                str(consistency_errors(q, good)))
    q2 = ("An RC phase-shift oscillator uses three identical sections of R = 10 kOhm and C = 10 nF around an "
          "inverting op-amp. Find the frequency of oscillation and the feedback resistor needed.")
    ps = phase_shift("Rf", ["fosc", "gain_condition"])
    ps.question = q2
    ok &= check("phase-shift netlist -> consistent", not consistency_errors(q2, ps),
                str(consistency_errors(q2, ps)))

    # the live L3 failure: the ladder stopped at C3 -> n, with a useless R from the − input to ground.
    # That is a different circuit (f0 = 1/(2 pi R C sqrt 3), Rf/R = 12), so it must be rejected.
    short = Netlist(analysis="ac", query=["fosc", "gain_condition"], question=q2,
                    components=[C("R1", "R", ["n", "0"], "10k"), C("Rf", "R", ["n", "out"], "Rf"),
                                C("C1", "C", ["out", "a"], "10n"), C("R2", "R", ["a", "0"], "10k"),
                                C("C2", "C", ["a", "b"], "10n"), C("R3", "R", ["b", "0"], "10k"),
                                C("C3", "C", ["b", "n"], "10n"), U("U1", "out", "0", "n")])
    e = consistency_errors(q2, short)
    ok &= check("a ladder with a missing section resistor -> rejected",
                any("virtual ground" in x or "capacitors but only" in x for x in e), str(e))

    # an oscillator netlist that still carries a source -> rejected at the consistency stage too
    src = wien()
    src.question = q
    src.components.append(C("V1", "V", ["in", "0"], None, ac="1"))
    ok &= check("a source in an oscillator netlist -> rejected by the parser checks",
                any("no input" in x for x in consistency_errors(q, src)),
                str(consistency_errors(q, src)))
    return ok


LIVE = [
    ("A Wien-bridge oscillator uses R = 10 kOhm and C = 10 nF in both arms of the bridge, with R1 = 10 kOhm "
     "from the inverting input to ground and R2 = 20 kOhm as the feedback resistor. Find the frequency of "
     "oscillation.", [{"kind": "fosc", "value": F_WIEN}]),
    ("A Wien-bridge oscillator has R = 4.7 kOhm and C = 22 nF in both arms. What is its oscillation "
     "frequency?", [{"kind": "fosc", "value": 1 / (2 * math.pi * 4.7e3 * 22e-9), "tol": 0.001}]),
    ("An RC phase-shift oscillator uses three identical RC sections with R = 10 kOhm and C = 10 nF around an "
     "inverting op-amp. Find the frequency of oscillation and the value of the feedback resistor needed to "
     "sustain it.", [{"kind": "fosc", "value": F_PS, "tol": 0.001},
                     {"kind": "gain_condition", "value": 290e3, "tol": 0.001}]),
    ("In a Wien-bridge oscillator with R = 10 kOhm and C = 10 nF, R1 = 10 kOhm. What must the feedback "
     "resistor R2 be for the circuit to oscillate, and at what frequency?",
     [{"kind": "fosc", "value": F_WIEN, "tol": 0.001}, {"kind": "gain_condition", "value": 20e3, "tol": 0.001}]),
    ("A Wien-bridge oscillator has R = 10 kOhm and C = 10 nF in both arms, R1 = 10 kOhm and R2 = 15 kOhm. "
     "Will it oscillate?", [{"kind": "pole.re", "value": -2500.0, "tol": 0.01}]),
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
        print("answers:", res.get("answers"))
        passed = r.status == "answered"
        for e in expect:
            good, detail = check_expectation(e, res.get("numeric") or {}, res.get("symbolic") or {}, 0.01,
                                             res.get("verification_notes"))
            passed &= good
            print("  ", detail)
        ok &= check(f"L{i} correct", passed, f"status={r.status}, error={st.get('error')}")
    print("\n" + ai.client.usage_report())
    return ok


if __name__ == "__main__":
    all_ok = solver_checks()
    all_ok &= step_checks()
    all_ok &= misc_checks()
    all_ok &= parser_checks()
    if "--live" in sys.argv:
        all_ok &= live_checks()
    print("\nPHASE 4 STEP 1:", "ALL PASSED" if all_ok else "SOME CHECKS FAILED")