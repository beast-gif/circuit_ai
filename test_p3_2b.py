"""
Phase 3, Step 2b checks: capacitor-filtered rectifiers (ripple), clampers and a voltage doubler.

    python test_p3_2b.py           # hand-worked steady states + ngspice transient verification (free)
    python test_p3_2b.py --steps   # also print the key steps
    python test_p3_2b.py --live    # 4 real questions: parse -> solve -> ngspice verify (Velona)

Expected values were computed independently from the ideal-switching equations of the peak rectifier:
  conduction: vC = vin − Von;  turn-off when iD = C·dvin/dt + vC/R = 0;  discharge vC = Voff·e^(−t/RC)
  until it meets vin − Von again (turn-on).  Textbook approximations: Vr ≈ Vp/(k·f·R·C), Vdc ≈ Vp − Vr/2.
"""
import math
import sys

from benchmark.run_benchmark import check_expectation
from llm_client import load_conventions
from schemas import Component, Netlist
from solvers import NotSupportedError, SolverError, get_solver
from validator import validate
from verifier import Verifier

CONV = load_conventions()
SHOW = "--steps" in sys.argv
VB = 12 * math.sqrt(2)


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"\n       {detail}" if detail and not cond else ""))
    return cond


def C(id, type, nodes, value=None, **kw):
    return Component(id=id, type=type, nodes=nodes, value=value, **kw)


def D(id, anode, cathode, model=None, **params):
    return Component(id=id, type="D", nodes=[anode, cathode], model=model, params=params)


def net(components, query, frequency="50", question=""):
    return Netlist(analysis="transient", query=query, components=components, frequency=frequency, question=question)


def close(got, want, rel, floor=1e-6):
    return got is not None and abs(got - want) <= max(rel * abs(want), floor)


def show(res):
    for st in res.trace.key_steps():
        print(f"      {st.step:>2}. {st.action}" + (f"\n          {st.equation}" if st.equation else "")
              + (f"\n          => {st.result}" if st.result else ""))


RL = C("RL", "R", ["out", "0"], "1k")
FILT_Q = ["Vmax(out)", "Vmin(out)", "Vripple(out)", "Vavg(out)", "ripple(out)", "conduction(D1)", "Imax(D1)"]

CASES = [
    # Half-wave, Si, Vm = 10 V, 50 Hz, R = 1k, C = 100 µF (RC = 0.1 s = 5 periods):
    #   turn-off at 91.70° (iD = 0), discharge to 7.75558 V, turn-on at 57.73° of the next cycle
    #   Vr = 9.3 − 7.75558 = 1.54442 V (textbook Vp/(fRC) = 1.86 V);  Vavg = 8.53097 V (textbook 8.37 V)
    #   ripple factor 0.053346 (textbook 0.06415);  conduction 33.964° (textbook √(2Vr/Vp) = 36.24°)
    #   peak diode current at turn-on: C·ω·Vm·cos θon + vC/R = 175.48 mA for an ideal switch (the solver's 1 mΩ
#   in series makes the spike rise in ~0.1 µs and peak slightly lower, hence the 1 % tolerance)
    ("Half-wave rectifier with a 100 µF filter",
     net([C("V1", "V", ["a", "0"], ac="10"), D("D1", "a", "out"), RL, C("C1", "C", ["out", "0"], "100u")], FILT_Q),
     {"Vmax(out)": (9.3, 1e-5), "Vmin(out)": (7.755577606261043, 2e-4), "Vripple(out)": (1.5444223937389578, 1e-3),
      "Vripple(out).approx": (1.86, 1e-5), "Vavg(out)": (8.530973731822087, 2e-4), "Vavg(out).approx": (8.37, 1e-5),
      "ripple(out)": (0.05334609758273376, 3e-3), "conduction(D1)": (33.96382090590148, 2e-3),
      "conduction(D1).approx": (36.23703271523067, 1e-5), "Imax(D1)": (0.1754801675304583, 1e-2)}),

    # Bridge, 12 V rms, 50 Hz, R = 1k, C = 470 µF: Vp = Vm − 1.4 = 15.5706, Vmin = 15.26280,
    #   Vr = 0.307765 V (textbook Vp/(2fRC) = 0.331289), Vavg = 15.41946, each diode conducts 11.2844°,
    #   peak diode current 490.32 mA
    ("Bridge rectifier with a 470 µF filter",
     net([C("V1", "V", ["a", "b"], params={"Vrms": 12}), D("D1", "a", "out"), D("D2", "b", "out"),
          D("D3", "0", "a"), D("D4", "0", "b"), RL, C("C1", "C", ["out", "0"], "470u")], FILT_Q),
     {"Vmax(out)": (VB - 1.4, 1e-5), "Vmin(out)": (15.262797788261453, 2e-4), "Vripple(out)": (0.3077649602156889, 3e-3),
      "Vripple(out).approx": (0.3312885691165349, 1e-5), "Vavg(out)": (15.419458664350236, 2e-4),
      "conduction(D1)": (22.568896037761128 / 2, 3e-3), "Imax(D1)": (0.4903162171661329, 1e-2)}),

    # Positive clamper (D anode at ground), Si, no load: C charges to Vm − 0.7 on the negative peak and holds:
    #   vo = vin + 9.3 V -> Vmax = 19.3 V, Vmin = −0.7 V, Vavg = 9.3 V
    ("Positive clamper (silicon, no load)",
     net([C("V1", "V", ["in", "0"], ac="10"), C("C1", "C", ["in", "out"], "1u"), D("D1", "0", "out")],
         ["Vmax(out)", "Vmin(out)", "Vavg(out)"], frequency="1k"),
     {"Vmax(out)": (19.3, 1e-5), "Vmin(out)": (-0.7, 1e-5), "Vavg(out)": (9.3, 1e-5)}),

    # Negative clamper with an ideal diode (D anode at the output): vo = vin − Vm -> 0 to −20 V, average −10 V
    ("Negative clamper (ideal diode)",
     net([C("V1", "V", ["in", "0"], ac="10"), C("C1", "C", ["in", "out"], "1u"), D("D1", "out", "0", "ideal")],
         ["Vmax(out)", "Vmin(out)", "Vavg(out)"], frequency="1k"),
     {"Vmax(out)": (0.0, 1e-9, 1e-4), "Vmin(out)": (-20.0, 1e-5), "Vavg(out)": (-10.0, 1e-5)}),

    # Half-wave voltage doubler, Si, no load: C1 charges to Vm − 0.7, C2 to 2·Vm − 1.4 = 18.6 V, no ripple
    ("Half-wave voltage doubler (no load)",
     net([C("V1", "V", ["a", "0"], ac="10"), C("C1", "C", ["a", "x"], "100u"), D("D1", "0", "x"), D("D2", "x", "out"),
          C("C2", "C", ["out", "0"], "100u")], ["Vmax(out)", "Vmin(out)", "Vripple(out)"]),
     {"Vmax(out)": (18.6, 1e-5), "Vmin(out)": (18.6, 1e-5), "Vripple(out)": (0.0, 1e-9, 1e-5)}),
]


def solver_checks(v: Verifier) -> bool:
    ok = True
    if v.exe is None:
        print("[SKIP] ngspice not found: verification checks skipped")
    for name, nl, expected in CASES:
        print(f"\n--- {name}")
        try:
            res = get_solver(nl).solve(nl, CONV)
        except (SolverError, NotSupportedError) as e:
            ok &= check(name, False, f"solver error: {e}")
            continue
        for q, a in res.answers.items():
            print(f"     {q:<15} = {a}")
        if SHOW:
            show(res)
        for key, spec in expected.items():
            want, rel = spec[0], spec[1]
            floor = spec[2] if len(spec) > 2 else 1e-6
            ok &= check(f"{name}: {key} = {want:.7g}", close(res.numeric.get(key), want, rel, floor),
                        f"got {res.numeric.get(key)}")
        if v.exe:
            rep = v.verify(nl, res)
            print(rep.table())
            ok &= check(f"{name}: verified by ngspice (transient from the steady state)", rep.verified is True,
                        "; ".join(res.verification_notes))
    return ok


def step_checks() -> bool:
    ok = True
    print()
    nl = CASES[0][1]
    res = get_solver(nl).solve(nl, CONV)
    acts = [st.action for st in res.trace.key_steps()]
    ok &= check("steps: the piecewise-exact method, the steady-state capacitor voltage, when D1 conducts, τ = RC",
                any(a.startswith("The capacitor remembers") for a in acts)
                and any(a.startswith("Steady state: capacitor voltage") for a in acts)
                and any(a.startswith("D1 conducts") for a in acts)
                and any(a.startswith("With every diode OFF") for a in acts), str(acts))
    tau = next(st for st in res.trace.key_steps() if st.action.startswith("With every diode OFF"))
    ok &= check("τ = R·C = 100 ms", "100 ms" in (tau.equation or ""), tau.equation)
    vr = res.answers.get("Vripple(out)", "")
    ok &= check("ripple answer gives the exact value and the textbook Vp/(f·R·C)", "Vp/(f·R·C)" in vr, vr)
    if SHOW:
        show(res)
    clamp = CASES[2][1]
    res = get_solver(clamp).solve(clamp, CONV)
    ok &= check("clamper: no capacitor-filter approximation (the capacitor is in series)",
                "textbook" not in res.answers.get("Vavg(out)", ""), res.answers.get("Vavg(out)"))
    nof = net(CASES[2][1].components, ["Vmax(out)"], frequency=None)
    res = get_solver(nof).solve(nof, CONV)
    ok &= check("no frequency given -> 1 kHz assumed, said in the notes",
                any("1 kHz assumed" in n for n in res.verification_notes), str(res.verification_notes))
    return ok


def misc_checks() -> bool:
    ok = True
    print()
    r = validate(CASES[0][1])
    ok &= check("validator accepts Vripple / ripple", not r.errors, str(r.errors))
    ex = net([C("V1", "V", ["a", "0"], ac="10"), D("D1", "a", "out", IS=1e-14), RL, C("C1", "C", ["out", "0"], "100u")],
             ["Vmax(out)"])
    try:
        get_solver(ex).solve(ex, CONV)
        ok &= check("exponential diode with a capacitor -> NotSupported (piecewise-linear only)", False)
    except NotSupportedError as e:
        ok &= check("exponential diode with a capacitor -> NotSupported (piecewise-linear only)",
                    "piecewise-linear" in str(e), str(e))
    return ok


LIVE = [
    ("A half-wave rectifier with a silicon diode is fed from a 10 V peak, 50 Hz sine wave and drives a 1 kOhm "
     "load with a 100 uF filter capacitor across it. Find the peak-to-peak ripple voltage and the DC output "
     "voltage.",
     [{"kind": "Vripple", "value": 1.5444223937389578, "tol": 0.02}, {"kind": "Vavg", "value": 8.530973731822087}]),
    ("A bridge rectifier with silicon diodes is fed from a 12 V rms, 50 Hz secondary. A 470 uF capacitor is "
     "connected across the 1 kOhm load. Find the peak-to-peak ripple and the average output voltage.",
     [{"kind": "Vripple", "value": 0.3077649602156889, "tol": 0.02}, {"kind": "Vavg", "value": 15.419458664350236}]),
    ("A clamper has a 1 uF capacitor in series between a 10 V peak, 1 kHz sine source and the output, and a silicon "
     "diode with its anode at ground and its cathode at the output. There is no load. Find the maximum and minimum "
     "output voltages.",
     [{"kind": "Vmax", "value": 19.3}, {"kind": "Vmin", "value": -0.7}]),
    ("A half-wave voltage doubler: C1 = 100 uF from a 10 V peak, 50 Hz source to node x, a silicon diode D1 with "
     "its anode at ground and cathode at x, a silicon diode D2 from x (anode) to the output, and C2 = 100 uF from "
     "the output to ground. With no load, find the output voltage.",
     [{"kind": "Vout", "value": 18.6}]),         # no ripple: V, Vmax or Vavg are all 18.6 V
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
            print("query:", st["netlist"].get("query"), "| analysis:", st["netlist"].get("analysis"),
                  "| f:", st["netlist"].get("frequency"))
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
    if "--live" in sys.argv:
        all_ok &= live_checks()
    print("\nPHASE 3 STEP 2b:", "ALL PASSED" if all_ok else "SOME CHECKS FAILED")