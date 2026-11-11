"""
Phase 4, Step 3 checks: LC oscillators (Colpitts, Hartley) through the transistor's small-signal model.

    python test_p4_3.py           # 8 hand-worked LC oscillators + step / parser checks (free)
    python test_p4_3.py --steps   # also print the key steps
    python test_p4_3.py --spice   # also verify each one against ngspice
    python test_p4_3.py --live    # 4 real questions: parse -> solve (Velona)

An LC oscillator question is always given as the AC EQUIVALENT circuit: no supply, the emitter at ground,
and the transistor replaced by i_c = gm·v_be (plus r_π = β/gm if β is given). The determinant then does the
rest, exactly as for the RC oscillators.

HAND-WORKED, twice over (determinant, then the loop-gain picture), for the common-emitter Colpitts with
L = 1 mH, C1 = 10 nF at the base, C2 = 100 nF at the collector, RL = 10 kOhm and r_π ignored:

    Im[det] = 0  ->  w^2 = 1/(L·C_eq),  C_eq = C1·C2/(C1+C2) = 9.0909 nF  ->  f0 = 52785.723 Hz
    Re[det] = 0  ->  gm·RL = C1/C2 = 0.1                                  ->  gm = 10 uS

The second one is worth checking by hand, because the feedback here is a step-UP. Driving the tank at the
collector, the base sits at V_b/V_c = 1/(1 - w^2·L·C1), and at w^2 = (C1+C2)/(L·C1·C2) that is exactly
-C2/C1 = -10: the divider multiplies by 10 and inverts. So the amplifier only needs |A| = gm·RL = 1/10 to
close the loop -- not 10. (Books that write "gm·R = C2/C1" label C1 as the collector-side capacitor.)

Hartley, same idea with the inductance tapped: L1 = 1 mH at the base, L2 = 100 uH at the collector,
C = 10 nF, RL = 10 kOhm:
    f0 = 1/(2·pi·sqrt((L1+L2)·C)) = 47987.021 Hz      and      gm·RL = L2/L1 = 0.1  ->  gm = 10 uS
"""
import math
import sys

from llm_client import load_config, load_conventions
from schemas import Component, Netlist, is_symbolic
from solvers import NotSupportedError, SolverError, get_solver

CONV = load_conventions()
SHOW = "--steps" in sys.argv
SPICE = "--spice" in sys.argv

L1, C1_, C2_, RL_ = 1e-3, 10e-9, 100e-9, 10e3
CEQ = C1_ * C2_ / (C1_ + C2_)
F_COLP = 1 / (2 * math.pi * math.sqrt(L1 * CEQ))          # 52785.723 Hz
GM_COLP = (C1_ / C2_) / RL_                               # 10 µS
LH1, LH2, CH = 1e-3, 100e-6, 10e-9
F_HART = 1 / (2 * math.pi * math.sqrt((LH1 + LH2) * CH))  # 47987.021 Hz
GM_HART = (LH2 / LH1) / RL_                               # 10 µS


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


def colpitts(gm=None, params=None, query=None, c1="10n", c2="100n", l="1m", rl="10k"):
    """Common-emitter Colpitts as the AC equivalent circuit. gm given -> a G element; params -> a BJT."""
    # params={} is "a transistor with nothing given", which is a real case to check -- so test for None,
    # not for truthiness
    amp = (C("Q1", "Q", ["c", "b", "0"], model="NPN", params=params) if params is not None else
           C("G1", "G", ["c", "0", "b", "0"], gm))
    return Netlist(analysis="ac", query=query or ["fosc"], question="A Colpitts oscillator.",
                   components=[C("C1", "C", ["b", "0"], c1), C("C2", "C", ["c", "0"], c2),
                               C("L1", "L", ["c", "b"], l), C("RL", "R", ["c", "0"], rl), amp])


def hartley(gm=None, params=None, query=None):
    amp = (C("Q1", "Q", ["c", "b", "0"], model="NPN", params=params) if params is not None else
           C("G1", "G", ["c", "0", "b", "0"], gm))
    return Netlist(analysis="ac", query=query or ["fosc"], question="A Hartley oscillator.",
                   components=[C("La", "L", ["b", "0"], "1m"), C("Lb", "L", ["c", "0"], "100u"),
                               C("C1", "C", ["c", "b"], "10n"), C("RL", "R", ["c", "0"], "10k"), amp])


CASES = [
    # 1. Colpitts exactly on the condition: gm = 10 µS -> poles on the jw axis at 52785.723 Hz
    ("Colpitts on the condition (gm = 10 µS)", colpitts(repr(GM_COLP)), {"fosc": F_COLP, "pole.re": 0.0}),

    # 2. gm too small by half: the poles move into the left half plane, so it dies away
    ("Colpitts with gm = 5 µS (too low)", colpitts("5e-6", query=["fosc", "oscillates", "gain_condition"]),
     {"gain_condition": GM_COLP}),

    # 3. gm ten times too big: it grows
    ("Colpitts with gm = 100 µS (too high)", colpitts("100e-6", query=["fosc", "oscillates"]), {}),

    # 4. gm left as a symbol: both Barkhausen equations give f0 AND the gm needed
    ("Colpitts with gm unknown", colpitts("gm", query=["fosc", "gain_condition"]),
     {"fosc": F_COLP, "gain_condition": GM_COLP}),

    # 5. The same, but written as a real BJT: the solver builds the small-signal model itself
    ("Colpitts with a BJT, gm = 10 µS in params", colpitts(params={"gm": GM_COLP}),
     {"fosc": F_COLP, "pole.re": 0.0}),

    # 6. A BJT given by its bias current instead: gm = I_C/V_T = 0.25 mA / 25 mV = 10 mA/V
    #    (that is far above the 10 µS needed, so it grows -- but f0 is unchanged by gm)
    ("Colpitts with a BJT given I_C = 0.25 mA", colpitts(params={"IC": 0.25e-3}, query=["fosc"]),
     {"fosc.approx": F_COLP}),

    # 7. Hartley on its condition
    ("Hartley on the condition (gm = 10 µS)", hartley(repr(GM_HART)), {"fosc": F_HART, "pole.re": 0.0}),

    # 8. Hartley with gm unknown
    ("Hartley with gm unknown", hartley("gm", query=["fosc", "gain_condition"]),
     {"fosc": F_HART, "gain_condition": GM_HART}),
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
        if SPICE:
            from verifier import Verifier
            rep = Verifier(load_config()).verify(nl, res)
            print(rep.table())
            ok &= check(f"{name}: ngspice agrees", rep.verified is not False,
                        "; ".join(f"{c.query}: {c.solver} vs {c.spice}" for c in rep.checks))
    return ok


def step_checks() -> bool:
    ok = True
    print()
    nl = colpitts(params={"gm": GM_COLP, "beta": 100})
    res = get_solver(nl).solve(nl, CONV)
    acts = " | ".join(st.action + " " + (st.equation or "") + " " + (st.result or "")
                      for st in res.trace.key_steps())
    ok &= check("a step replaces the BJT by its small-signal model", "small-signal model" in acts, acts[:300])
    ok &= check("that step shows gm and r_pi = beta/gm", "r_π = β/gm" in acts, acts[:300])
    ok &= check("a step names the Colpitts tapped tank", "Colpitts" in acts, acts[:300])
    ok &= check("the textbook C_eq formula is shown", "C1·C2/(C1+C2)" in acts, acts[:300])
    if SHOW:
        show(res)

    # with r_pi included the required gm is slightly higher: the base resistance loads the tank
    nl = colpitts("gm", query=["gain_condition"])
    nl.components.append(C("rpi", "R", ["b", "0"], "10k"))
    res = get_solver(nl).solve(nl, CONV)
    gm_loaded = res.numeric.get("gain_condition")
    ok &= check("adding r_pi = 10 k raises the gm needed (the base loads the tank)",
                gm_loaded is not None and gm_loaded > GM_COLP,
                f"{gm_loaded} vs {GM_COLP} with no r_pi")
    print(f"       (gm needed: {GM_COLP * 1e6:.4f} µS with no r_π, {gm_loaded * 1e6:.4f} µS with r_π = 10 k)")

    nl = hartley(repr(GM_HART))
    res = get_solver(nl).solve(nl, CONV)
    acts = " | ".join(st.action + " " + (st.equation or "") for st in res.trace.key_steps())
    ok &= check("a step names the Hartley tank and shows L1 + L2", "Hartley" in acts and "L1 + L2" in acts,
                acts[:300])
    return ok


def misc_checks() -> bool:
    ok = True
    print()
    from parsing.text_parser import consistency_errors
    from solvers.oscillator import OscillatorSolver

    ok &= check("an LC oscillator goes to the oscillator solver",
                get_solver(colpitts(repr(GM_COLP))).name == "oscillator",
                get_solver(colpitts(repr(GM_COLP))).name)

    # a transistor with no gm cannot be used: there is no supply to bias it from
    bare = colpitts(params={})
    good, why = OscillatorSolver().supports(bare)
    ok &= check("a transistor without gm -> clear reason", not good and "gm" in why, f"{good} {why}")
    q = "A Colpitts oscillator uses a BJT. Find the frequency of oscillation."
    bare.question = q
    ok &= check("the parser checks also ask for gm",
                any("no gm" in x for x in consistency_errors(q, bare)), str(consistency_errors(q, bare)))

    # the emitter must be at AC ground in the equivalent circuit
    moved = colpitts(params={"gm": GM_COLP})
    moved.components[-1] = C("Q1", "Q", ["c", "b", "e"], model="NPN", params={"gm": GM_COLP})
    moved.components.append(C("RE", "R", ["e", "0"], "1k"))
    moved.question = q
    ok &= check("an emitter that is not at AC ground -> flagged",
                any("AC ground" in x for x in consistency_errors(q, moved)),
                str(consistency_errors(q, moved)))

    # an amplitude-limiting diode is out of scope, with a reason that says what to do
    with_d = colpitts(repr(GM_COLP))
    with_d.components.append(C("D1", "D", ["c", "0"]))
    good, why = OscillatorSolver().supports(with_d)
    ok &= check("a limiting diode -> out of scope, with a reason", not good and "diodes" in why, f"{good} {why}")
    return ok


LIVE = [
    ("A Colpitts oscillator uses L = 1 mH with C1 = 10 nF at the base and C2 = 100 nF at the collector, a "
     "collector load of 10 kOhm, and a transistor with gm = 1 mS. Find the frequency of oscillation.",
     [{"kind": "fosc", "value": F_COLP, "tol": 0.001}]),
    ("A Hartley oscillator has L1 = 1 mH at the base, L2 = 100 uH at the collector and C = 10 nF across the "
     "tank, with a 10 kOhm collector load and a transistor with gm = 1 mS. What is its oscillation "
     "frequency?", [{"kind": "fosc", "value": F_HART, "tol": 0.001}]),
    ("In a Colpitts oscillator with L = 1 mH, C1 = 10 nF at the base, C2 = 100 nF at the collector and a "
     "10 kOhm collector load, what transconductance gm does the transistor need to sustain oscillation, and "
     "at what frequency does it oscillate?",
     [{"kind": "fosc", "value": F_COLP, "tol": 0.001}, {"kind": "gain_condition", "value": GM_COLP,
                                                        "tol": 0.001}]),
    ("A Colpitts oscillator has L = 100 uH, C1 = 1 nF and C2 = 1 nF, a 5 kOhm load and a transistor with "
     "gm = 2 mS. Find the oscillation frequency.",
     [{"kind": "fosc", "value": 1 / (2 * math.pi * math.sqrt(100e-6 * 0.5e-9)), "tol": 0.001}]),
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
    print("\nPHASE 4 STEP 3:", "ALL PASSED" if all_ok else "SOME CHECKS FAILED")