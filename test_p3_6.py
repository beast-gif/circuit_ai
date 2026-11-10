"""
Phase 3, Step 6 checks: comparators and Schmitt triggers (op-amps whose output sits at a rail).

    python test_p3_6.py           # hand-worked comparators + trip points + parser checks (free)
    python test_p3_6.py --steps   # also print the key steps
    python test_p3_6.py --live    # 5 real questions: parse -> solve (Velona)

The trip point of a state is the input voltage that makes V+ − V− zero while the output is held at that
rail. Worked by hand for the circuits below (±12 V rails, R1 = 10 k, R2 = 100 k):
    inverting Schmitt     V_UT, V_LT = ±V_sat·R1/(R1 + R2) = ±1.090909 V,  hysteresis 2.181818 V
    non-inverting Schmitt V_UT, V_LT = ±V_sat·R1/R2 = ±1.2 V,              hysteresis 2.4 V
    with a 5 V reference  the pair shifts to 5.636364 V and 3.454545 V
"""
import sys

from benchmark.run_benchmark import check_expectation
from llm_client import load_conventions
from parsing.text_parser import consistency_errors
from schemas import Component, Netlist
from solvers import NotSupportedError, SolverError, get_solver
from validator import validate

CONV = load_conventions()
SHOW = "--steps" in sys.argv
VSAT = 12.0
BETA = 10e3 / 110e3                     # R1/(R1+R2)


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"\n       {detail}" if detail and not cond else ""))
    return cond


def C(id, type, nodes, value=None, **kw):
    return Component(id=id, type=type, nodes=nodes, value=value, **kw)


def U(id, out, plus, minus):
    return Component(id=id, type="OPAMP", nodes=[out, plus, minus], model="ideal")


def net(components, query, question="±12 V supplies."):
    return Netlist(analysis="dc", query=query, components=components, question=question)


def close(got, want, rel=1e-9, floor=1e-12):
    return got is not None and abs(got - want) <= max(rel * abs(want), floor)


def show(res, full=False):
    for st in (res.trace.all_steps() if full else res.trace.key_steps()):
        print(f"      {st.step:>2}. {st.action}" + (f"\n          {st.equation}" if st.equation else "")
              + (f"\n          => {st.result}" if st.result else ""))


# inverting Schmitt trigger: input to in−, R1 from in+ to ground, R2 (positive feedback) from out to in+
def inv_schmitt(vin="0", vref=None):
    parts = [C("V1", "V", ["in", "0"], vin), C("R2", "R", ["out", "p"], "100k"), U("U1", "out", "p", "in")]
    parts.append(C("R1", "R", ["p", "ref"], "10k") if vref else C("R1", "R", ["p", "0"], "10k"))
    if vref:
        parts.append(C("VREF", "V", ["ref", "0"], vref))
    return parts


# non-inverting Schmitt trigger: input through R1 to in+, R2 from out to in+, in− grounded
def noninv_schmitt(vin="0"):
    return [C("V1", "V", ["in", "0"], vin), C("R1", "R", ["in", "p"], "10k"),
            C("R2", "R", ["out", "p"], "100k"), U("U1", "out", "p", "0")]


CASES = [
    # 1. Inverting Schmitt trigger: the two boundaries are ±V_sat·β = ±1.090909 V
    ("Inverting Schmitt trigger: trip points",
     net(inv_schmitt(), ["VTH", "VTL", "hysteresis"]),
     {"VTH": VSAT * BETA, "VTL": -VSAT * BETA, "hysteresis": 2 * VSAT * BETA}),

    # 2. Non-inverting Schmitt trigger: V+ = (vin·R2 + Vo·R1)/(R1 + R2) = 0 -> vin = −Vo·R1/R2 = ∓1.2 V
    ("Non-inverting Schmitt trigger: trip points",
     net(noninv_schmitt(), ["VTH", "VTL", "hysteresis"]),
     {"VTH": VSAT * 10e3 / 100e3, "VTL": -VSAT * 10e3 / 100e3, "hysteresis": 2.4}),

    # 3. Inverting Schmitt trigger with the divider returned to a 5 V reference instead of ground:
    #    the pair shifts to (5·R2 + Vo·R1)/(R1 + R2) = 5.636364 V and 3.454545 V (same 2.181818 V width)
    ("Inverting Schmitt trigger with a 5 V reference",
     net(inv_schmitt(vref="5"), ["VTH", "VTL", "hysteresis"]),
     {"VTH": (5 * 100e3 + VSAT * 10e3) / 110e3, "VTL": (5 * 100e3 - VSAT * 10e3) / 110e3,
      "hysteresis": 2 * VSAT * BETA}),

    # 4. A plain comparator (no feedback at all): 3 V on the + input against a 2 V reference on the − input,
    #    so the output is at the + rail
    ("Plain comparator, input above the reference",
     net([C("V1", "V", ["in", "0"], "3"), C("VREF", "V", ["ref", "0"], "2"), U("U1", "out", "in", "ref")],
         ["V(out)", "state(U1)"]),
     {"V(out)": 12.0}),

    # 5. The same comparator with the input below the reference -> the − rail
    ("Plain comparator, input below the reference",
     net([C("V1", "V", ["in", "0"], "1"), C("VREF", "V", ["ref", "0"], "2"), U("U1", "out", "in", "ref")],
         ["V(out)", "state(U1)"]),
     {"V(out)": -12.0}),

    # 6. An inverting Schmitt trigger with the input well above the upper trip point: the output must be low,
    #    whatever the history, and the + input then sits at −V_sat·β = −1.090909 V
    ("Inverting Schmitt trigger driven past its trip point",
     net(inv_schmitt(vin="5"), ["V(out)", "V(p)", "state(U1)"]),
     {"V(out)": -12.0, "V(p)": -VSAT * BETA}),

    # 7. Rails given as Vcc / Vee rather than ±: +15 / −15, so the trip points scale with 15 V
    ("Rails written as Vcc = 15 V and Vee = -15 V",
     Netlist(analysis="dc", query=["VTH", "hysteresis"], components=inv_schmitt(),
             question="The op-amp runs from Vcc = 15 V and Vee = -15 V."),
     {"VTH": 15 * BETA, "hysteresis": 2 * 15 * BETA}),

    # 8. Saturation voltage given directly ("the output saturates at 13 V")
    ("Saturation voltage stated as 13 V",
     Netlist(analysis="dc", query=["VTH"], components=inv_schmitt(),
             question="The output saturates at 13 V."),
     {"VTH": 13 * BETA}),
]


def solver_checks() -> bool:
    ok = True
    for name, nl, expected in CASES:
        print(f"\n--- {name}")
        try:
            res = get_solver(nl).solve(nl, CONV)
        except (SolverError, NotSupportedError) as e:
            ok &= check(name, False, f"solver error: {e}")
            continue
        for q, a in res.answers.items():
            print(f"     {q:<12} = {a}")
        if SHOW:
            show(res)
        for key, want in expected.items():
            ok &= check(f"{name}: {key} = {want:.7g}", close(res.numeric.get(key), want, 1e-9),
                        f"got {res.numeric.get(key)}")
    return ok


def step_checks() -> bool:
    ok = True
    print()
    res = get_solver(CASES[0][1]).solve(CASES[0][1], CONV)
    acts = [st.action for st in res.trace.key_steps()]
    ok &= check("steps name it a comparator / Schmitt trigger with positive feedback",
                any("comparator / Schmitt trigger" in a for a in acts), str(acts))
    sym = [st for st in res.trace.key_steps() if "write the input as a symbol" in st.action]
    ok &= check("steps show both rails solved with the input as a symbol", len(sym) == 2,
                str([st.action for st in sym]))
    ok &= check("the symbolic step shows V+ − V− = a·V1 + b = 0",
                bool(sym) and "V+ − V− =" in (sym[0].equation or "") and "= 0" in (sym[0].equation or ""),
                str(sym[0].equation if sym else None))
    trip = next((st for st in res.trace.key_steps() if st.action.startswith("The two boundaries")), None)
    ok &= check("a step states V_UT, V_LT and the hysteresis together",
                trip is not None and "hysteresis =" in (trip.result or ""), str(trip.result if trip else None))
    if SHOW:
        show(res)

    res = get_solver(CASES[5][1]).solve(CASES[5][1], CONV)
    tried = [st for st in res.trace.all_steps() if st.action.startswith("Try U1 output at the + rail")]
    ok &= check("driven past the trip point: the full view shows the rejected '+ rail' guess",
                bool(tried) and "✗" in (tried[0].result or ""), str([st.result for st in tried]))
    chk = next(st for st in res.trace.key_steps() if st.action == "Check the assumptions")
    ok &= check("the check shows V+ − V− with its sign", "V+ − V−" in (chk.equation or ""), chk.equation)
    return ok


def misc_checks() -> bool:
    ok = True
    print()
    r = validate(CASES[0][1])
    ok &= check("validator accepts VTH / VTL / hysteresis", not r.errors, str(r.errors))
    ok &= check("validator accepts state(U1)", not validate(CASES[3][1]).errors, str(validate(CASES[3][1]).errors))

    # an ordinary inverting amplifier (negative feedback, no threshold asked) must NOT go to this solver
    amp = Netlist(analysis="dc", query=["V(out)"], question="Find the output.",
                  components=[C("V1", "V", ["in", "0"], "1"), C("R1", "R", ["in", "n1"], "1k"),
                              C("R2", "R", ["n1", "out"], "10k"), U("U1", "out", "0", "n1")])
    ok &= check("a normal op-amp amplifier still goes to the DC/AC solver", get_solver(amp).name == "dc_ac",
                get_solver(amp).name)
    ok &= check("a Schmitt trigger goes to the comparator solver", get_solver(CASES[0][1]).name == "comparator",
                get_solver(CASES[0][1]).name)

    # Asking for a single DC output of a Schmitt trigger INSIDE the hysteresis band has no one answer: both
    # rail states pass their own check there, which is exactly what "bistable" means.
    mid = net(inv_schmitt(vin="0"), ["V(out)", "state(U1)"])
    res = get_solver(mid).solve(mid, CONV)
    print("     inside the band:", res.answers)
    ok &= check("input inside the hysteresis band -> says it depends on the history",
                "history" in res.answers.get("V(out)", ""), str(res.answers))
    ok &= check("inside the band: a step shows both states are stable",
                any("bistable" in (st.result or "") or "both output states are stable" in (st.result or "")
                    for st in res.trace.all_steps()),
                str([st.result for st in res.trace.all_steps()]))
    # just outside the band there IS one answer again: 1.2 V > V_UT = 1.0909 V -> the output must be low
    out = net(inv_schmitt(vin="1.2"), ["V(out)"])
    ok &= check("just outside the band -> a single answer again",
                close(get_solver(out).solve(out, CONV).numeric.get("V(out)"), -VSAT),
                str(get_solver(out).solve(out, CONV).answers))

    # no supply stated anywhere -> a stated assumption, not a silent guess
    bare = Netlist(analysis="dc", query=["hysteresis"], components=inv_schmitt(), question="Find the hysteresis.")
    res = get_solver(bare).solve(bare, CONV)
    ok &= check("no supply given -> ±13 V assumed and said in the notes",
                any("13" in n for n in res.verification_notes) and close(res.numeric["hysteresis"], 2 * 13 * BETA),
                f"{res.numeric.get('hysteresis')} {res.verification_notes}")

    # an input that never reaches the op-amp cannot have a trip point
    stuck = net([C("V1", "V", ["in", "0"], "1"), C("R1", "R", ["p", "0"], "10k"),
                 C("R2", "R", ["out", "p"], "100k"), U("U1", "out", "p", "0")], ["VTH"])
    try:
        get_solver(stuck).solve(stuck, CONV)
        ok &= check("input not connected to the comparator -> clear error", False)
    except SolverError as e:
        ok &= check("input not connected to the comparator -> clear error", "does not depend on" in str(e), str(e))
    return ok


def parser_checks() -> bool:
    ok = True
    print()
    q = ("An inverting Schmitt trigger uses an op-amp with ±12 V supplies, R1 = 10 kOhm from the non-inverting "
         "input to ground and R2 = 100 kOhm from the output to the non-inverting input. The input is applied to "
         "the inverting input. Find the upper and lower threshold voltages and the hysteresis.")
    good = net(inv_schmitt(), ["VTH", "VTL", "hysteresis"], q)
    ok &= check("Schmitt trigger netlist -> consistent", not consistency_errors(q, good),
                str(consistency_errors(q, good)))
    # the feedback must go to the NON-inverting input: to in− it would be an ordinary amplifier
    wrong = net([C("V1", "V", ["in", "0"], "0"), C("R1", "R", ["p", "0"], "10k"),
                 C("R2", "R", ["out", "in"], "100k"), U("U1", "out", "p", "in")], ["VTH"], q)
    e = [x.lower() for x in consistency_errors(q, wrong)]
    ok &= check("feedback taken to the inverting input -> rejected",
                any("positive feedback" in x or "+ input" in x for x in e), str(e))
    return ok


LIVE = [
    ("An inverting Schmitt trigger uses an op-amp with +/-12 V supplies, R1 = 10 kOhm from the non-inverting "
     "input to ground and R2 = 100 kOhm from the output back to the non-inverting input. The input signal goes "
     "to the inverting input. Find the upper and lower trip points and the hysteresis.",
     [{"kind": "VTH", "value": VSAT * BETA}, {"kind": "VTL", "value": -VSAT * BETA},
      {"kind": "hysteresis", "value": 2 * VSAT * BETA}]),
    ("A non-inverting Schmitt trigger has the input applied through R1 = 10 kOhm to the non-inverting input, "
     "R2 = 100 kOhm from the output to the non-inverting input, and the inverting input grounded. The op-amp "
     "saturates at +/-12 V. Find the two threshold voltages.",
     [{"kind": "VTH", "value": 1.2}, {"kind": "VTL", "value": -1.2}]),
    ("An op-amp comparator with +/-12 V supplies has a 2 V reference on its inverting input and 3 V applied to "
     "its non-inverting input. Find the output voltage.",
     [{"kind": "V", "value": 12.0}]),
    ("An inverting Schmitt trigger with +/-15 V supplies has R1 = 10 kOhm from the non-inverting input to "
     "ground and R2 = 100 kOhm of feedback. Find the hysteresis width.",
     [{"kind": "hysteresis", "value": 2 * 15 * BETA}]),
    ("An op-amp comparator with +/-12 V supplies compares a 1 V input on its non-inverting input against a 2 V "
     "reference on its inverting input. What is the output voltage?",
     [{"kind": "V", "value": -12.0}]),
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
    print("\nPHASE 3 STEP 6:", "ALL PASSED" if all_ok else "SOME CHECKS FAILED")