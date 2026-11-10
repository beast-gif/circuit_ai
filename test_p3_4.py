"""
Phase 3, Step 4 checks: BJT small-signal amplifiers (voltage gain, R_in, R_out).

    python test_p3_4.py           # hand-worked amplifiers + ngspice verification + parser checks (free)
    python test_p3_4.py --steps   # also print the key steps of each circuit
    python test_p3_4.py --live    # 5 real amplifier questions: parse -> solve -> ngspice verify (Velona)

All five amplifiers share the same bias (VCC = 12 V, R1 = 47 k, R2 = 10 k, RC = 2.2 k, RE = 1 k, β = 100):
    I_C = 1.286334 mA  ->  gm = I_C/V_T = 51.45335 mA/V,  r_π = β/gm = 1943.508 Ω,  r_e = V_T/I_E = 19.2427 Ω
Expected gains and resistances were solved independently (my own nodal equations on the hybrid-π model),
and ngspice checks the model itself by biasing real BJTs and running its own .ac analysis.
"""
import sys

from benchmark.run_benchmark import check_expectation
from llm_client import load_conventions
from schemas import Component, Netlist
from solvers import NotSupportedError, SolverError, get_solver
from validator import validate
from verifier import Verifier

CONV = load_conventions()
CONV_EARLY = {**CONV, "semiconductor": {**(CONV.get("semiconductor") or {}), "VA": 100.0}}
SHOW = "--steps" in sys.argv

GM = 0.051453348321824306
RPI = 1943.5081148564298
RE_SMALL = 19.242654602538906


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"\n       {detail}" if detail and not cond else ""))
    return cond


def C(id, type, nodes, value=None, **kw):
    return Component(id=id, type=type, nodes=nodes, value=value, **kw)


def Q(id, c, b, e, model="NPN", beta=100):
    return Component(id=id, type="Q", nodes=[c, b, e], model=model, params={"beta": beta})


def net(components, query, question="", analysis="small_signal"):
    return Netlist(analysis=analysis, query=query, components=components, question=question)


def close(got, want, rel=1e-6, floor=1e-12):
    return got is not None and abs(got - want) <= max(rel * abs(want), floor)


def show(res, full=False):
    for st in (res.trace.all_steps() if full else res.trace.key_steps()):
        print(f"      {st.step:>2}. {st.action}" + (f"\n          {st.equation}" if st.equation else "")
              + (f"\n          => {st.result}" if st.result else ""))


# the common bias network: VCC, divider, RC, RE  (+ the signal source through C1)
def bias(rc="2.2k", re="1k"):
    out = [C("VCC", "V", ["vcc", "0"], "12"), C("R1", "R", ["vcc", "b"], "47k"), C("R2", "R", ["b", "0"], "10k")]
    if rc:
        out.append(C("RC", "R", ["vcc", "c"], rc))
    if re:
        out.append(C("RE", "R", ["e", "0"], re))
    return out


def ce(bypass=True, rl=None, rs=None, cap_out=True):
    """Common-emitter amplifier: signal in through C1 to the base, out at the collector."""
    comps = bias() + [Q("Q1", "c", "b", "e")]
    src_node = "s" if rs else "in"
    comps.append(C("V1", "V", [src_node, "0"], ac="1"))
    if rs:
        comps.append(C("RS", "R", ["s", "in"], rs))
    comps.append(C("C1", "C", ["in", "b"], "10u"))
    if bypass:
        comps.append(C("CE", "C", ["e", "0"], "100u"))
    if rl:
        comps += [C("C2", "C", ["c", "out"], "10u"), C("RL", "R", ["out", "0"], rl)]
    return comps


CASES = [
    # 1. CE, emitter fully bypassed, no load: Av = −gm·RC = −113.1974, Rin = R1‖R2‖r_π = 1572.797 Ω,
    #    Rout = RC = 2.2 kΩ
    ("Common emitter, bypassed emitter",
     net(ce(), ["gain", "Rin", "Rout", "gm(Q1)", "rpi(Q1)", "re(Q1)"]),
     {"gain": -113.19736630801347, "Rin": 1572.7967094059125, "Rout": 2200.0, "gm(Q1)": GM, "rpi(Q1)": RPI,
      "re(Q1)": RE_SMALL}, CONV),

    # 2. Same with a 4.7 kΩ load coupled at the collector: Av = −gm·(RC‖RL) = −77.10545, Rout = RC‖RL = 1498.551 Ω
    ("Common emitter with a 4.7 kΩ load",
     net(ce(rl="4.7k"), ["gain", "Rout"]),
     {"gain": -77.10545241270484, "Rout": 1498.5507246376812}, CONV),

    # 3. CE with the emitter resistor NOT bypassed: Av = −2.137094 exactly
    #    (the textbook −gm·RC/(1 + gm·RE) = −2.158058 is 1 % off: it drops the base current in R_E)
    #    Rin = R1‖R2‖(r_π + (β+1)·R_E) = 7634.132 Ω
    ("Common emitter, un-bypassed emitter resistor",
     net(ce(bypass=False), ["gain", "Rin"]),
     {"gain": -2.13709445140087, "Rin": 7634.131998886845}, CONV),

    # 4. Emitter follower (output at the emitter, collector straight to VCC):
    #    Av = +0.9811206, Rout = R_E ‖ (r_π/(β+1)) = 18.87937 Ω, Rin = 7634.132 Ω
    ("Emitter follower",
     net(bias(rc=None) + [Q("Q1", "vcc", "b", "e"), C("V1", "V", ["in", "0"], ac="1"),
                          C("C1", "C", ["in", "b"], "10u")],
         ["gain", "Rin", "Rout"]),
     {"gain": 0.981120634506763, "Rin": 7634.131998886845, "Rout": 18.879365493237447}, CONV),

    # 5. Common base (base bypassed to ground, signal into the emitter):
    #    Av = +gm·RC = +113.1974 (non-inverting), Rin = R_E ‖ r_e-ish = 18.87937 Ω
    ("Common base",
     net(bias() + [Q("Q1", "c", "b", "e"), C("CB", "C", ["b", "0"], "100u"),
                   C("V1", "V", ["in", "0"], ac="1"), C("C1", "C", ["in", "e"], "10u")],
         ["gain", "Rin"]),
     {"gain": 113.197366308013, "Rin": 18.879365493237447}, CONV),

    # 6. CE with a 600 Ω source resistance: the divider and r_π load the source, so the overall gain drops
    #    Av = (Rin/(Rin + Rs))·(−gm·RC) = −81.93884.  R_in is still the AMPLIFIER's input resistance,
    #    1572.797 Ω: the 600 Ω belongs to the source, not to the amplifier.
    ("Common emitter driven through 600 Ω",
     net(ce(rs="600"), ["gain", "Rin"]),
     {"gain": -81.9388415271202, "Rin": 1572.7967094059125}, CONV),

    # 7. Same as 1 but with an Early voltage V_A = 100 V: r_o = V_A/I_C = 77.740 kΩ and it appears in
    #    parallel with RC: Av = −gm·(RC ‖ r_o) = −110.0821
    ("Common emitter with the Early effect (V_A = 100 V)",
     net(ce(), ["gain", "ro(Q1)"]),
     {"gain": -110.08211493592286, "ro(Q1)": 77740.32459425718}, CONV_EARLY),
]


def solver_checks(v: Verifier) -> bool:
    ok = True
    if v.exe is None:
        print("[SKIP] ngspice not found: verification checks skipped")
    for name, nl, expected, conv in CASES:
        print(f"\n--- {name}")
        try:
            res = get_solver(nl).solve(nl, conv)
        except (SolverError, NotSupportedError) as e:
            ok &= check(name, False, f"solver error: {e}")
            continue
        for q, a in res.answers.items():
            print(f"     {q:<10} = {a}")
        if SHOW:
            show(res)
        for key, want in expected.items():
            ok &= check(f"{name}: {key} = {want:.7g}", close(res.numeric.get(key), want), f"got {res.numeric.get(key)}")
        if v.exe:
            rep = v.verify(nl, res)
            print(rep.table())
            ok &= check(f"{name}: verified by ngspice (its own bias and gm)", rep.verified is True,
                        "; ".join(res.verification_notes))
    return ok


def step_checks() -> bool:
    ok = True
    print()
    nl = CASES[0][1]
    res = get_solver(nl).solve(nl, CONV)
    acts = [st.action for st in res.trace.key_steps()]
    ok &= check("key steps: the three stages are named (DC bias, hybrid-π, small-signal circuit)",
                any(a.startswith("Step 1") for a in acts) and any(a.startswith("Step 2") for a in acts)
                and any(a.startswith("Step 3") for a in acts), str(acts))
    qp = next((st for st in res.trace.key_steps() if st.action.startswith("Q-point of Q1")), None)
    ok &= check("Q-point step shows I_C = 1.286 mA and 'active region ✓'",
                qp is not None and "1.286 mA" in (qp.result or "") and "active" in (qp.result or ""),
                str(qp.result if qp else None))
    hyb = next(st for st in res.trace.key_steps() if st.action.startswith("Step 2"))
    ok &= check("hybrid-π step shows gm, r_π, r_e and r_o = ∞",
                all(x in (hyb.equation or "") for x in ("gm = I_C/V_T", "r_π = β/gm", "r_e = V_T/I_E", "r_o = ∞")),
                hyb.equation)
    ss = next(st for st in res.trace.key_steps() if st.action.startswith("Step 3"))
    ok &= check("small-signal step says the supplies are grounded and the capacitors shorted",
                "short to ground" in (ss.equation or "") and "shorts (mid-band)" in (ss.equation or ""), ss.equation)
    tb = next((st for st in res.trace.key_steps() if st.action.startswith("Textbook check")), None)
    ok &= check("textbook step names the common emitter and matches Av = −gm·(R_C ‖ R_L)",
                tb is not None and "common emitter" in tb.action and "same as the exact solution" in (tb.result or ""),
                str((tb.action, tb.equation, tb.result) if tb else None))
    if SHOW:
        show(res)

    # the un-bypassed case: the textbook formula is 1 % off, and the step must say so instead of claiming equality
    res = get_solver(CASES[2][1]).solve(CASES[2][1], CONV)
    tb = next((st for st in res.trace.key_steps() if st.action.startswith("Textbook check")), None)
    ok &= check("un-bypassed R_E: the step shows the approximate formula and flags the difference",
                tb is not None and "un-bypassed" in tb.action and "the exact solution differs" in (tb.equation or ""),
                str((tb.action, tb.equation) if tb else None))
    ok &= check("un-bypassed R_E: gain.approx = −2.158 is recorded next to the exact −2.137",
                close(res.numeric.get("gain.approx"), -2.1580579682634933, 1e-6), str(res.numeric.get("gain.approx")))
    tb = next((st for st in get_solver(CASES[3][1]).solve(CASES[3][1], CONV).trace.key_steps()
               if st.action.startswith("Textbook check")), None)
    ok &= check("emitter follower is named as such", tb is not None and "emitter follower" in tb.action,
                str(tb.action if tb else None))
    tb = next((st for st in get_solver(CASES[4][1]).solve(CASES[4][1], CONV).trace.key_steps()
               if st.action.startswith("Textbook check")), None)
    ok &= check("common base is named as such", tb is not None and "common base" in tb.action,
                str(tb.action if tb else None))
    res = get_solver(CASES[5][1]).solve(CASES[5][1], CONV)
    st = next((s for s in res.trace.key_steps() if s.action.startswith("Input resistance of the amplifier")), None)
    ok &= check("source resistance: a step says R_S belongs to the source, not the amplifier",
                st is not None and "RS" in st.action, str(st.action if st else None))
    return ok


def parser_checks() -> bool:
    from parsing.text_parser import consistency_errors
    ok = True
    print()
    q = ("A common-emitter amplifier has VCC = 12 V, R1 = 47 kOhm, R2 = 10 kOhm, RC = 2.2 kOhm and RE = 1 kOhm "
         "bypassed by a 100 uF capacitor, with the signal coupled to the base through a 10 uF capacitor. "
         "beta = 100. Find the voltage gain.")
    good = net(ce(), ["gain"], q)
    ok &= check("amplifier with its capacitors and AC source -> consistent", not consistency_errors(q, good),
                str(consistency_errors(q, good)))
    no_caps = net([c for c in ce() if c.type.value != "C"], ["gain"], q)
    e = consistency_errors(q, no_caps)
    ok &= check("capacitors removed by the parser -> rejected", any("keep them in the netlist" in x for x in e), str(e))
    no_src = net([c for c in ce() if c.id != "V1"], ["gain"], q)
    e = consistency_errors(q, no_src)
    ok &= check("no AC signal source -> rejected", any("needs a signal source" in x for x in e), str(e))
    # L5 of the first live run: the parser wrote CE's value as the word "bypass"
    worded = net([c.model_copy(update={"value": "bypass"}) if c.id == "CE" else c for c in ce()], ["gain"], q)
    e = consistency_errors(q, worded)
    ok &= check('capacitor value written as the word "bypass" -> rejected, asked for a number',
                any('not a number' in x and "CE" in x for x in e), str(e))
    # ... and the solver still answers it at mid-band, where the value cannot matter
    res = get_solver(worded).solve(worded, CONV)
    ok &= check('mid-band gain still exact with a word-valued bypass capacitor (it is a short anyway)',
                close(res.numeric.get("gain"), -113.19736630801347), str(res.answers))
    freq = Netlist(analysis="small_signal", query=["fc"], components=worded.components, question=q)
    try:
        get_solver(freq).solve(freq, CONV)
        ok &= check("word-valued capacitor + a frequency question -> clear error", False)
    except NotSupportedError as e2:
        ok &= check("word-valued capacitor + a frequency question -> clear error",
                    "value matters" in str(e2), str(e2))
    return ok


def misc_checks() -> bool:
    ok = True
    print()
    r = validate(CASES[0][1])
    ok &= check("validator accepts gain / Rin / Rout / gm / rpi / re", not r.errors, str(r.errors))
    bad = net(CASES[0][1].components, ["gm(RC)"])
    ok &= check("validator: gm() of a resistor -> error", any("needs a BJT" in x for x in validate(bad).errors))

    # a transistor biased in saturation cannot be a linear amplifier: say so instead of giving a number
    sat = net([C("VCC", "V", ["vcc", "0"], "12"), C("RB", "R", ["vcc", "b"], "47k"), C("RC", "R", ["vcc", "c"], "2.2k"),
               Q("Q1", "c", "b", "0"), C("V1", "V", ["in", "0"], ac="1"), C("C1", "C", ["in", "b"], "10u")],
              ["gain"])
    try:
        get_solver(sat).solve(sat, CONV)
        ok &= check("saturated bias -> clear error, not a gain", False)
    except SolverError as e:
        ok &= check("saturated bias -> clear error, not a gain", "saturation" in str(e) and "active" in str(e), str(e))

    # the DC bias ignores the capacitors: shorting the emitter capacitor must not change the Q-point
    a = get_solver(CASES[0][1]).solve(CASES[0][1], CONV).numeric["bias(Q1).IC"]
    b = get_solver(CASES[2][1]).solve(CASES[2][1], CONV).numeric["bias(Q1).IC"]
    ok &= check("bypassed and un-bypassed amplifiers share the same Q-point (capacitors are open in DC)",
                close(a, b, 1e-12), f"{a} vs {b}")
    return ok


LIVE = [
    ("A common-emitter amplifier has VCC = 12 V, R1 = 47 kOhm from VCC to the base, R2 = 10 kOhm from the base to "
     "ground, RC = 2.2 kOhm, RE = 1 kOhm bypassed by a 100 uF capacitor, and the signal is coupled to the base "
     "through a 10 uF capacitor. With beta = 100 find the mid-band voltage gain and the input resistance.",
     [{"kind": "gain", "value": -113.19736630801347, "tol": 0.02},
      {"kind": "Rin", "value": 1572.7967094059125, "tol": 0.02}]),
    ("In the same common-emitter amplifier (VCC = 12 V, R1 = 47 kOhm, R2 = 10 kOhm, RC = 2.2 kOhm, RE = 1 kOhm "
     "bypassed), a 4.7 kOhm load is coupled to the collector through a 10 uF capacitor. With beta = 100 find the "
     "voltage gain and the output resistance.",
     [{"kind": "gain", "value": -77.10545241270484, "tol": 0.02},
      {"kind": "Rout", "value": 1498.5507246376812, "tol": 0.02}]),
    ("A common-emitter amplifier with VCC = 12 V, R1 = 47 kOhm, R2 = 10 kOhm, RC = 2.2 kOhm and RE = 1 kOhm has NO "
     "bypass capacitor across RE. The input is coupled to the base through a 10 uF capacitor. With beta = 100 find "
     "the voltage gain.",
     [{"kind": "gain", "value": -2.13709445140087, "tol": 0.03}]),
    ("An emitter follower has VCC = 12 V, R1 = 47 kOhm from VCC to the base, R2 = 10 kOhm from the base to ground, "
     "the collector connected directly to VCC and RE = 1 kOhm from the emitter to ground. The signal is coupled to "
     "the base through a 10 uF capacitor. With beta = 100 find the voltage gain and the output resistance.",
     [{"kind": "gain", "value": 0.981120634506763, "tol": 0.02},
      {"kind": "Rout", "value": 18.879365493237447, "tol": 0.05}]),
    ("A common-emitter amplifier (VCC = 12 V, R1 = 47 kOhm, R2 = 10 kOhm, RC = 2.2 kOhm, RE = 1 kOhm bypassed, "
     "beta = 100) is driven from a source with a 600 Ohm internal resistance through a 10 uF coupling capacitor. "
     "Find the overall voltage gain from the source to the collector.",
     [{"kind": "gain", "value": -81.9388415271202, "tol": 0.03}]),
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
            good, detail = check_expectation(e, res.get("numeric") or {}, res.get("symbolic") or {}, 0.02,
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
    print("\nPHASE 3 STEP 4:", "ALL PASSED" if all_ok else "SOME CHECKS FAILED")