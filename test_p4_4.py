"""
Phase 4, Step 4 checks: the op-amp astable multivibrator (square-wave / relaxation oscillator).

    python test_p4_4.py           # 8 hand-worked astables + step / routing / parser checks (free)
    python test_p4_4.py --steps   # also print the key steps
    python test_p4_4.py --live    # 4 real questions: parse -> solve (Velona)

Every expected number below was produced twice, independently of the project: once from the closed form
    t = tau * ln((v0 - Vf)/(v1 - Vf))
and once by brute-force time stepping of dv/dt = (Vout - v)/(R*C) at 1 ns steps, flipping the output
whenever v crosses the + input. The two agreed to 5-6 digits, which is what fixes the values here.

    R = 10 k, C = 100 n, R1 = R2 = 10 k, rails ±12 V     (beta = 0.5, tau = 1 ms)
        trip points ±6 V, T_high = T_low = tau*ln 3 = 1.0986123 ms
        T = 2.1972246 ms,  f = 455.1196 Hz,  duty 50 %
    R1 = 10 k, R2 = 100 k (beta = 0.0909)                 T_high = tau*ln(1.2) = 0.1823216 ms
        T = 0.3646431 ms,  f = 2742.4075 Hz
    asymmetric rails +12 / -6 V, beta = 0.5               trips +6 and -3 V
        T_high = tau*ln(15/6) = 0.9162907 ms, T_low = tau*ln(12/3) = 1.3862944 ms
        T = 2.3025851 ms,  f = 434.2945 Hz,  duty 39.794 %
    R = 4.7 k, C = 22 n, R1 = 22 k, R2 = 47 k             f = 7318.7571 Hz
"""
import math
import sys

from llm_client import load_conventions
from parsing.text_parser import consistency_errors
from schemas import Component, Netlist
from solvers import NotSupportedError, SolverError, get_solver
from validator import validate

CONV = load_conventions()
SHOW = "--steps" in sys.argv
TAU = 10e3 * 100e-9                                   # 1 ms
T_SYM = 2 * TAU * math.log(3)                         # 2.1972246 ms
F_SYM = 1 / T_SYM                                     # 455.1196 Hz


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"\n       {detail}" if detail and not cond else ""))
    return cond


def C(id, type, nodes, value=None, **kw):
    return Component(id=id, type=type, nodes=nodes, value=value, **kw)


def close(got, want, rel=1e-6):
    return got is not None and abs(got - want) <= rel * max(abs(want), 1e-12)


def show(res):
    for st in res.trace.key_steps():
        print(f"      {st.step:>2}. {st.action}" + (f"\n          {st.equation}" if st.equation else "")
              + (f"\n          => {st.result}" if st.result else ""))


def astable(r="10k", c="100n", r1="10k", r2="10k", query=None, question="An op-amp astable multivibrator "
                                                                        "with ±12 V supplies."):
    return Netlist(analysis="ac", query=query or ["fosc"], question=question,
                   components=[C("R1", "R", ["p", "0"], r1), C("R2", "R", ["out", "p"], r2),
                               C("R", "R", ["out", "n"], r), C("C1", "C", ["n", "0"], c),
                               Component(id="U1", type="OPAMP", nodes=["out", "p", "n"], model="ideal")])


CASES = [
    # 1. The textbook symmetric astable
    ("Astable, beta = 0.5, tau = 1 ms", astable(query=["fosc", "period", "duty", "VTH", "VTL"]),
     {"fosc": F_SYM, "period": T_SYM, "duty": 0.5, "VTH": 6.0, "VTL": -6.0}),

    # 2. Weaker positive feedback -> narrower hysteresis -> a much higher frequency
    ("Astable with R2 = 100 k (beta = 0.0909)", astable(r2="100k", query=["fosc", "VTH"]),
     {"fosc": 1 / (2 * TAU * math.log(1.2)), "VTH": 12 * 10 / 110}),

    # 3. Asymmetric rails: the duty cycle stops being 50 %
    ("Astable with +12 / -6 V rails",
     astable(query=["fosc", "duty", "T_high", "T_low"],
             question="An op-amp astable multivibrator. The output saturates at Vcc = 12 V and Vee = -6 V."),
     {"fosc": 1 / (TAU * (math.log(15 / 6) + math.log(12 / 3))), "T_high": TAU * math.log(15 / 6),
      "T_low": TAU * math.log(12 / 3), "duty": math.log(15 / 6) / (math.log(15 / 6) + math.log(12 / 3))}),

    # 4. A different RC pair and divider
    ("Astable with R = 4.7 k, C = 22 n, R1 = 22 k, R2 = 47 k",
     astable("4.7k", "22n", "22k", "47k", query=["fosc", "period"]),
     {"fosc": 1 / (2 * 4.7e3 * 22e-9 * math.log((1 + 22 / 69) / (1 - 22 / 69)))}),

    # 5. Ten times the capacitor -> ten times the period, exactly
    ("Astable with C = 1 uF (ten times the period)", astable(c="1u", query=["period"]),
     {"period": 10 * T_SYM}),

    # 6. The timing resistor scales the period the same way
    ("Astable with R = 100 k", astable(r="100k", query=["fosc"]), {"fosc": F_SYM / 10}),

    # 7. The hysteresis asked for by name: the same ±6 V as the Schmitt trigger of P3-6
    ("Astable: the trip points are the Schmitt trip points", astable(query=["VTH", "VTL", "hysteresis"]),
     {"VTH": 6.0, "VTL": -6.0, "hysteresis": 12.0}),

    # 8. Barkhausen does not apply here, and the solver should say so rather than inventing a loop gain
    ("Astable asked about Barkhausen", astable(query=["fosc", "barkhausen"]), {"fosc": F_SYM}),
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
            print(f"     {q:<12} = {a}")
        if SHOW:
            show(res)
        for key, want in expected.items():
            got = res.numeric.get(key)
            ok &= check(f"{name}: {key} = {want:.7g}", close(got, want, 1e-6), f"got {got}")
    return ok


def step_checks() -> bool:
    ok = True
    print()
    nl = astable(query=["fosc", "duty"])
    res = get_solver(nl).solve(nl, CONV)
    acts = " | ".join(st.action + " " + (st.equation or "") + " " + (st.result or "")
                      for st in res.trace.key_steps())
    ok &= check("a step says the output is always at a rail, so the circuit is linear only between switches",
                "linear only between switches" in acts, acts[:200])
    ok &= check("a step gives each state's trip point, target and time constant",
                "charges toward" in acts and "τ =" in acts, acts[:300])
    ok &= check("a step shows the exponential charging law being used",
                "τ·ln((v_start − Vf)/(v_stop − Vf))" in acts, acts[:300])
    ok &= check("the book's formula 2RC·ln((1+β)/(1−β)) is shown for comparison",
                "2·R·C·ln((1+β)/(1−β))" in acts, acts[:300])
    ok &= check("and it agrees with the circuit solve", "agrees with the circuit solve" in acts, acts[:300])
    ok &= check("the textbook frequency is recorded separately as fosc.approx",
                close(res.numeric.get("fosc.approx"), F_SYM, 1e-9), str(res.numeric.get("fosc.approx")))
    ok &= check("Barkhausen is refused with a reason, not answered with a number",
                "not applicable" in get_solver(CASES[7][1]).solve(CASES[7][1], CONV).answers["barkhausen"],
                str(get_solver(CASES[7][1]).solve(CASES[7][1], CONV).answers.get("barkhausen")))
    if SHOW:
        show(res)

    # a circuit that cannot oscillate: the trip points lie outside the rails the capacitor charges toward,
    # so the output latches. beta = 1 (no R1 to ground -> V+ = Vout) is the clearest case.
    latch = astable(query=["fosc"])
    latch.components = [c for c in latch.components if c.id != "R1"]
    try:
        get_solver(latch).solve(latch, CONV)
        ok &= check("a latching circuit -> says so instead of giving a frequency", False)
    except (SolverError, NotSupportedError) as e:
        ok &= check("a latching circuit -> says so instead of giving a frequency",
                    "latches" in str(e) or "never" in str(e), str(e))
    return ok


def misc_checks() -> bool:
    ok = True
    print()
    nl = astable(query=["fosc", "period", "duty", "T_high", "T_low"])
    ok &= check("validator accepts period / duty / T_high / T_low", not validate(nl).errors,
                str(validate(nl).errors))
    ok &= check("an astable goes to the relaxation solver", get_solver(nl).name == "relaxation",
                get_solver(nl).name)

    # the Wien bridge must NOT be stolen by the relaxation solver: no timing capacitor on the − input
    wien = Netlist(analysis="ac", query=["fosc"], question="A Wien-bridge oscillator.",
                   components=[C("R1", "R", ["n", "0"], "10k"), C("R2", "R", ["n", "out"], "20k"),
                               C("Rs", "R", ["out", "x"], "10k"), C("Cs", "C", ["x", "p"], "10n"),
                               C("Rp", "R", ["p", "0"], "10k"), C("Cp", "C", ["p", "0"], "10n"),
                               Component(id="U1", type="OPAMP", nodes=["out", "p", "n"], model="ideal")])
    ok &= check("a Wien bridge still goes to the oscillator solver", get_solver(wien).name == "oscillator",
                get_solver(wien).name)

    # a plain Schmitt trigger (no timing capacitor) still goes to the comparator solver
    schmitt = Netlist(analysis="dc", query=["VTH", "VTL"], question="An inverting Schmitt trigger, ±12 V.",
                      components=[C("V1", "V", ["in", "0"], "0"), C("R1", "R", ["p", "0"], "10k"),
                                  C("R2", "R", ["out", "p"], "100k"),
                                  Component(id="U1", type="OPAMP", nodes=["out", "p", "in"], model="ideal")])
    ok &= check("a Schmitt trigger still goes to the comparator solver",
                get_solver(schmitt).name == "comparator", get_solver(schmitt).name)

    # an integrator (capacitor at the − input but NEGATIVE feedback only) is not an astable
    integ = Netlist(analysis="ac", query=["gain"], question="An op-amp integrator.",
                    components=[C("V1", "V", ["in", "0"], None, ac="1"), C("R1", "R", ["in", "n"], "10k"),
                                C("C1", "C", ["n", "out"], "100n"),
                                Component(id="U1", type="OPAMP", nodes=["out", "0", "n"], model="ideal")])
    ok &= check("an integrator is not taken for an astable", get_solver(integ).name == "dc_ac",
                get_solver(integ).name)

    q = ("An op-amp astable multivibrator with ±12 V supplies has R1 = 10 kOhm from the non-inverting input "
         "to ground, R2 = 10 kOhm of feedback, a timing resistor of 10 kOhm and a 100 nF capacitor. Find the "
         "frequency of the output square wave.")
    good = astable(query=["fosc"], question=q)
    ok &= check("astable netlist -> consistent", not consistency_errors(q, good),
                str(consistency_errors(q, good)))
    return ok


LIVE = [
    ("An op-amp astable multivibrator runs from +/-12 V supplies with R1 = 10 kOhm from the non-inverting "
     "input to ground, R2 = 10 kOhm of positive feedback, a timing resistor R = 10 kOhm and a timing "
     "capacitor C = 100 nF. Find the frequency of the square wave.",
     [{"kind": "fosc", "value": F_SYM, "tol": 0.001}]),
    ("A square-wave generator uses an op-amp with +/-12 V saturation, a 100 nF timing capacitor charged "
     "through 10 kOhm from the output to the inverting input, and a positive-feedback divider of "
     "R1 = 10 kOhm to ground and R2 = 100 kOhm. Find the period of the output.",
     [{"kind": "period", "value": 2 * TAU * math.log(1.2), "tol": 0.001}]),
    ("An op-amp relaxation oscillator with +/-12 V rails has R1 = 22 kOhm, R2 = 47 kOhm, a timing resistor "
     "of 4.7 kOhm and a 22 nF capacitor. What is its output frequency and the duty cycle?",
     [{"kind": "fosc", "value": 1 / (2 * 4.7e3 * 22e-9 * math.log((1 + 22 / 69) / (1 - 22 / 69))),
       "tol": 0.001}, {"kind": "duty", "value": 0.5, "tol": 0.01}]),
    ("In an op-amp astable multivibrator with +/-12 V supplies, R1 = 10 kOhm, R2 = 10 kOhm, R = 10 kOhm and "
     "C = 100 nF, find the upper and lower threshold voltages of the capacitor voltage.",
     [{"kind": "VTH", "value": 6.0, "tol": 0.001}, {"kind": "VTL", "value": -6.0, "tol": 0.001}]),
]


def live_checks() -> bool:
    from benchmark.run_benchmark import check_expectation
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
    if "--live" in sys.argv:
        all_ok &= live_checks()
    print("\nPHASE 4 STEP 4:", "ALL PASSED" if all_ok else "SOME CHECKS FAILED")