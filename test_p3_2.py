"""
Phase 3, Step 2a checks: rectifiers and clippers (diodes + sinusoidal input, no capacitors).

    python test_p3_2.py           # hand-worked waveforms + ngspice sweep verification + parser checks (free)
    python test_p3_2.py --steps   # also print the key steps of each circuit
    python test_p3_2.py --live    # 5 real questions: parse -> solve -> ngspice verify (Velona)

vin = Vm·sin θ. Exact results for the 0.7 V model, e.g. the half-wave output vo = vin − 0.7 while vin > 0.7:
    Vavg = (1/2π)·[2·Vm·cos α − 0.7·(π − 2α)],  α = asin(0.7/Vm)
The textbook approximation Vp/π ignores that the diode conducts for slightly less than half a cycle; both
are reported. Expected values were computed independently (4-million-point numerical integration).
"""
import math
import sys

from benchmark.run_benchmark import check_expectation
from llm_client import load_conventions
from parsing.text_parser import consistency_errors
from schemas import Component, Netlist
from solvers import NotSupportedError, SolverError, get_solver
from validator import validate
from verifier import Verifier

CONV = load_conventions()
SHOW = "--steps" in sys.argv
VB = 12 * math.sqrt(2)                        # 12 V rms -> 16.9706 V peak


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"\n       {detail}" if detail and not cond else ""))
    return cond


def C(id, type, nodes, value=None, **kw):
    return Component(id=id, type=type, nodes=nodes, value=value, **kw)


def D(id, anode, cathode, model=None, **params):
    return Component(id=id, type="D", nodes=[anode, cathode], model=model, params=params)


def net(components, query, question="", frequency=None):
    return Netlist(analysis="transient", query=query, components=components, question=question,
                   frequency=frequency)


def close(got, want, rel=1e-6, floor=1e-9):
    return got is not None and abs(got - want) <= max(rel * abs(want), floor)


def show(res):
    for st in res.trace.key_steps():
        print(f"      {st.step:>2}. {st.action}" + (f"\n          {st.equation}" if st.equation else "")
              + (f"\n          => {st.result}" if st.result else ""))


SIN10 = C("V1", "V", ["a", "0"], ac="10")
RL = C("RL", "R", ["out", "0"], "1k")
ALL = ["Vmax(out)", "Vmin(out)", "Vavg(out)", "Vrms(out)"]

CASES = [
    # Half-wave, silicon (0.7 V), Vm = 10 V, RL = 1k:
    #   Vp = 9.3 V, Vavg = 2.840901 V exact (textbook Vp/π = 2.960282), Vrms = 4.559059 V (textbook Vp/2 = 4.65)
    #   PIV = Vm = 10 V (negative peak, diode OFF), Imax = 9.3 mA, conduction = 180 − 2·asin(0.07) = 171.972°
    ("Half-wave rectifier, silicon diode",
     net([SIN10, D("D1", "a", "out"), RL], ALL + ["PIV(D1)", "Imax(D1)", "Iavg(D1)", "conduction(D1)"]),
     {"Vmax(out)": 9.3, "Vmin(out)": 0.0, "Vavg(out)": 2.8409006431657593, "Vavg(out).approx": 9.3 / math.pi,
      "Vrms(out)": 4.559059262371793, "Vrms(out).approx": 4.65, "PIV(D1)": 10.0, "Imax(D1)": 9.3e-3,
      "Iavg(D1)": 2.8409006431657593e-3, "conduction(D1)": 171.97209}),

    # Ideal diode: Vavg = Vm/π = 3.183099 V, Vrms = Vm/2 = 5 V, conduction 180°
    ("Half-wave rectifier, ideal diode",
     net([SIN10, D("D1", "a", "out", "ideal"), RL], ALL + ["conduction(D1)"]),
     {"Vmax(out)": 10.0, "Vavg(out)": 10 / math.pi, "Vrms(out)": 5.0, "conduction(D1)": 180.0}),

    # Bridge, 12 V rms (Vm = 16.97056 V): two diodes conduct -> Vp = Vm − 1.4 = 15.57056 V
    #   Vavg = 9.440580 exact (textbook 2Vp/π = 9.912528), Vrms = 10.755234; PIV of each diode = Vm − 0.7 = 16.27056
    ("Bridge rectifier, 12 V rms",
     net([C("V1", "V", ["a", "b"], params={"Vrms": 12}), D("D1", "a", "out"), D("D2", "b", "out"),
          D("D3", "0", "a"), D("D4", "0", "b"), RL], ALL + ["PIV(D1)", "PIV(D3)"], frequency="50"),
     {"Vmax(out)": VB - 1.4, "Vmin(out)": 0.0, "Vavg(out)": 9.440579602164584, "Vavg(out).approx": 2 * (VB - 1.4) / math.pi,
      "Vrms(out)": 10.755233518118695, "PIV(D1)": VB - 0.7, "PIV(D3)": VB - 0.7}),

    # Centre-tapped, each half 12 V rms: Vp = Vm − 0.7 = 16.27056, Vavg = 10.112988, PIV = 2Vm − 0.7 = 33.24113
    ("Centre-tapped full-wave rectifier",
     net([C("V1", "V", ["a", "0"], params={"Vrms": 12}), C("V2", "V", ["0", "b"], params={"Vrms": 12}),
          D("D1", "a", "out"), D("D2", "b", "out"), RL], ALL + ["PIV(D1)"]),
     {"Vmax(out)": VB - 0.7, "Vavg(out)": 10.112987826696045, "Vrms(out)": 11.373671197844128,
      "PIV(D1)": 2 * VB - 0.7}),

    # Transformer 120 V rms, 10:1 -> 12 V rms secondary, ideal half-wave: Vmax = 16.97056, Vavg = Vm/π = 5.401898
    ("Transformer 10:1 into an ideal half-wave rectifier",
     net([C("V1", "V", ["a", "0"], params={"Vrms": 120, "ratio": 10}), D("D1", "a", "out", "ideal"), RL],
         ["Vmax(out)", "Vavg(out)"]),
     {"Vmax(out)": VB, "Vavg(out)": VB / math.pi}),

    # Series-R clipper with a 2 V battery: clips at 2.7 V, passes the negative half:
    #   Vmax = 2.7, Vmin = −10, Vavg = −1.949844, Vrms = 5.312645
    ("Positive clipper at 2.7 V (diode + 2 V battery)",
     net([SIN10, C("R1", "R", ["a", "out"], "1k"), D("D1", "out", "x"), C("V2", "V", ["x", "0"], "2")],
         ALL + ["vtc(out)"]),
     {"Vmax(out)": 2.7, "Vmin(out)": -10.0, "Vavg(out)": -1.9498435983636946, "Vrms(out)": 5.312644819198832}),

    # Double limiter at ±3.7 V (two diodes with 3 V batteries): Vavg = 0, Vrms = 3.392551, D1 conducts 136.569°
    ("Double limiter ±3.7 V",
     net([SIN10, C("R1", "R", ["a", "out"], "1k"), D("D1", "out", "x"), C("V2", "V", ["x", "0"], "3"),
          D("D2", "y", "out"), C("V3", "V", ["0", "y"], "3")], ALL + ["conduction(D1)"]),
     {"Vmax(out)": 3.7, "Vmin(out)": -3.7, "Vavg(out)": 0.0, "Vrms(out)": 3.392551466196041,
      "conduction(D1)": 136.56879}),

    # Zener clipper (VZ = 5.1 V): clips at +5.1 V (breakdown) and −0.7 V (forward):
    #   Vavg = 1.784072, Vrms = 3.216764
    ("Zener clipper +5.1 V / −0.7 V",
     net([SIN10, C("R1", "R", ["a", "out"], "1k"), D("D1", "0", "out", "zener", VZ=5.1)], ALL),
     {"Vmax(out)": 5.1, "Vmin(out)": -0.7, "Vavg(out)": 1.7840719772440354, "Vrms(out)": 3.2167644347633075}),
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
        for key, want in expected.items():
            rel = 2e-5 if key.startswith("conduction") else 1e-6     # numeric reference for the angle
            ok &= check(f"{name}: {key} = {want:.7g}", close(res.numeric.get(key), want, rel), f"got {res.numeric.get(key)}")
        if v.exe:
            rep = v.verify(nl, res)
            print(rep.table())
            ok &= check(f"{name}: verified by ngspice (sweep over one cycle)", rep.verified is True,
                        "; ".join(res.verification_notes))
    return ok


def step_checks() -> bool:
    ok = True
    print()
    nl = CASES[0][1]
    res = get_solver(nl).solve(nl, CONV)
    key = [st for st in res.trace.key_steps()]
    ranges = [st for st in key if st.action.startswith("vin from")]
    ok &= check("half-wave: two vin ranges (D1 OFF below 0.7 V, ON above)", len(ranges) == 2,
                str([st.action for st in ranges]))
    on = next((st for st in ranges if "D1 ON" in st.action), None)
    ok &= check("ON range shows vo = vin − 700 mV", on is not None and "vo = vin − 700 mV" in (on.equation or ""),
                str(on.equation if on else None))
    avg = res.answers.get("Vavg(out)", "")
    ok &= check("Vavg answer gives the exact value AND the textbook Vp/π", "textbook approximation Vp/π" in avg, avg)
    if SHOW:
        show(res)
    clip = CASES[5][1]
    res = get_solver(clip).solve(clip, CONV)
    vtc = res.answers.get("vtc(out)", "")
    ok &= check("clipper vtc: 'vo = vin' below 2.7 V and 'vo = 2.7 V' above",
                "vo = vin" in vtc and "vo = 2.7 V" in vtc, vtc)
    return ok


def misc_checks() -> bool:
    ok = True
    print()
    r = validate(CASES[0][1])
    ok &= check("validator accepts Vmax / Vavg / PIV / Imax / conduction", not r.errors, str(r.errors))
    bad = net([SIN10, D("D1", "a", "out"), RL], ["PIV(RL)"])
    ok &= check("validator: PIV() of a resistor -> error", any("needs a diode" in x for x in validate(bad).errors))
    ok &= check("source given only as Vrms is a valid component",
                Component(id="V1", type="V", nodes=["a", "0"], params={"Vrms": 12}).params["Vrms"] == 12)
    th = net([SIN10, C("R1", "R", ["a", "out"], "1k"), D("D1", "out", "0")], ["Rth(out,0)"])
    try:
        res = get_solver(th).solve(th, CONV)
        ok &= check("Rth on a waveform circuit -> clear 'not available'", False, str(res.answers))
    except SolverError as e:
        ok &= check("Rth on a waveform circuit -> clear 'not available'", "not available for waveforms" in str(e), str(e))
    return ok


def parser_checks() -> bool:
    ok = True
    print()
    q = "A half-wave rectifier is fed from a 12 V rms source and drives a 1 kOhm load. Find the peak output."
    wrong = net([C("V1", "V", ["a", "0"], ac="12"), D("D1", "a", "out"), RL], ["Vmax(out)"], q)
    e = consistency_errors(q, wrong)
    ok &= check("'12 V rms' put into ac (= peak) -> rejected, use Vrms", any('"Vrms": 12' in x for x in e), str(e))
    right = net([C("V1", "V", ["a", "0"], params={"Vrms": 12}), D("D1", "a", "out"), RL], ["Vmax(out)"], q)
    e = consistency_errors(q, right)
    ok &= check("'12 V rms' as params Vrms -> consistent", not e, str(e))
    qb = "A bridge rectifier with silicon diodes is fed from a 10 V peak source and drives 1 kOhm. Find Vavg."
    e = consistency_errors(qb, net([C("V1", "V", ["a", "0"], ac="10"), D("D1", "a", "out"), RL], ["Vavg(out)"], qb))
    ok &= check("'bridge' with one diode -> rejected (needs four)", any("FOUR diodes" in x for x in e), str(e))
    qc = "A centre-tapped full-wave rectifier has 12 V rms on each half of the secondary. Find the PIV."
    e = consistency_errors(qc, net([C("V1", "V", ["a", "0"], params={"Vrms": 12}), D("D1", "a", "out"),
                                    D("D2", "0", "out"), RL], ["PIV(D1)"], qc))
    ok &= check("'centre-tapped' with one source -> rejected (two halves)", any("TWO equal" in x for x in e), str(e))
    return ok


LIVE = [
    ("A half-wave rectifier uses a silicon diode and a 1 kOhm load. The input is a 10 V peak, 50 Hz sine wave. "
     "Find the peak output voltage and the peak inverse voltage of the diode.",
     [{"kind": "Vmax", "value": 9.3}, {"kind": "PIV", "value": 10.0}]),
    ("A bridge rectifier with four silicon diodes is fed from a 12 V rms, 50 Hz transformer secondary and drives "
     "a 1 kOhm load. Find the peak output voltage and the PIV of each diode.",
     [{"kind": "Vmax", "value": VB - 1.4}, {"kind": "PIV", "value": VB - 0.7}]),
    ("A centre-tapped full-wave rectifier has 12 V rms on each half of the secondary and drives a 1 kOhm load "
     "through two silicon diodes. Find the peak output voltage and the PIV of each diode.",
     [{"kind": "Vmax", "value": VB - 0.7}, {"kind": "PIV", "value": 2 * VB - 0.7}]),
    ("A series clipper has a 1 kOhm resistor from a 10 V peak sine source to the output, and from the output a "
     "silicon diode (anode at the output) in series with a 2 V battery (positive terminal toward the diode) to "
     "ground. Find the maximum and minimum output voltages.",
     [{"kind": "Vmax", "value": 2.7}, {"kind": "Vmin", "value": -10.0}]),
    ("An ideal-diode half-wave rectifier is fed from 120 V rms through a 10:1 step-down transformer and drives "
     "a 1 kOhm load. Find the average (DC) output voltage.",
     [{"kind": "Vavg", "value": VB / math.pi}]),
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
            print("query:", st["netlist"].get("query"), "| analysis:", st["netlist"].get("analysis"))
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
    print("\nPHASE 3 STEP 2a:", "ALL PASSED" if all_ok else "SOME CHECKS FAILED")