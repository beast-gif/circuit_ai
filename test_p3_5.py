"""
Phase 3, Step 5 checks: MOSFET DC bias (square law) and MOSFET small-signal amplifiers.

    python test_p3_5.py           # hand-worked circuits + ngspice verification + parser checks (free)
    python test_p3_5.py --steps   # also print the key steps of each circuit
    python test_p3_5.py --live    # 5 real MOSFET questions: parse -> solve -> ngspice verify (Velona)

Square law (NMOS; PMOS the same with V_SG and V_SD):
    saturation  I_D = ½·k·V_OV²·(1 + λ·V_DS)     triode  I_D = k·(V_OV·V_DS − V_DS²/2)     cutoff  I_D = 0
Every expected value was found independently by solving the quadratic by hand (sympy, outside the project),
and ngspice checks the model itself with its own level-1 MOSFETs, which ARE this square law.
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

# bias A, shared by the amplifiers: VDD 10 V, divider 1M/1M (V_G = 5 V), RD 2k, RS 1k, k 0.5 mA/V², Vt 1 V
ID_A = 0.0015278640450004206
VOV_A = 2.4721359549995796
VDS_A = 5.416407864998738
GM_A = 0.0012360679774997896          # √(2·k·I_D) = 2·I_D/V_OV


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"\n       {detail}" if detail and not cond else ""))
    return cond


def C(id, type, nodes, value=None, **kw):
    return Component(id=id, type=type, nodes=nodes, value=value, **kw)


def M(id, d, g, s, model="NMOS", k=0.5e-3, Vt=1, **extra):
    return Component(id=id, type="M", nodes=[d, g, s], model=model, params={"k": k, "Vt": Vt, **extra})


def net(components, query, question="", analysis="dc"):
    return Netlist(analysis=analysis, query=query, components=components, question=question)


def close(got, want, rel=1e-6, floor=1e-12):
    return got is not None and abs(got - want) <= max(rel * abs(want), floor)


def show(res, full=False):
    for st in (res.trace.all_steps() if full else res.trace.key_steps()):
        print(f"      {st.step:>2}. {st.action}" + (f"\n          {st.equation}" if st.equation else "")
              + (f"\n          => {st.result}" if st.result else ""))


VDD = C("VDD", "V", ["vdd", "0"], "10")
DIV = [C("R1", "R", ["vdd", "g"], "1Meg"), C("R2", "R", ["g", "0"], "1Meg")]
MQ = ["bias(M1)", "ID(M1)", "VGS(M1)", "VDS(M1)", "VOV(M1)"]

CASES = [
    # A. Divider gate bias, source resistor: V_G = 5 V, and I_D = ½k(V_G − I_D·R_S − V_t)² gives
    #    I_D = 1.527864 mA, V_OV = 2.472136 V, V_DS = 10 − I_D(2k + 1k) = 5.416408 V -> saturation
    ("Divider gate bias with a source resistor (saturation)",
     net([VDD] + DIV + [C("RD", "R", ["vdd", "d"], "2k"), C("RS", "R", ["s", "0"], "1k"), M("M1", "d", "g", "s")],
         MQ + ["V(d)", "V(s)"]),
     {"ID(M1)": ID_A, "VGS(M1)": 5.0 - ID_A * 1e3, "VDS(M1)": VDS_A, "VOV(M1)": VOV_A,
      "V(s)": ID_A * 1e3, "V(d)": 10 - ID_A * 2e3}, {"bias(M1)": "saturation"}),

    # B. Diode-connected (gate tied to the drain, so V_DS = V_GS and it is always saturated):
    #    I_D = ½k(10 − I_D·10k − 1)²  ->  I_D = 0.7292122 mA, V_GS = V_DS = 2.707878 V
    #    (the other root of the quadratic has V_OV < 0, so it is rejected)
    ("Diode-connected NMOS",
     net([VDD, C("RD", "R", ["vdd", "d"], "10k"), M("M1", "d", "d", "0")], ["ID(M1)", "VDS(M1)", "bias(M1)"]),
     {"ID(M1)": 0.0007292121597166108, "VDS(M1)": 2.7078784028338916}, {"bias(M1)": "saturation"}),

    # C. Gate at 5 V with a 10 kΩ drain resistor: saturation would need I_D = 4 mA and V_D = −30 V, so it is
    #    in the TRIODE region: (10 − V_DS)/10k = k(4·V_DS − V_DS²/2) -> V_DS = 0.5067629 V, I_D = 0.9493237 mA
    ("Triode (linear) region",
     net([VDD, C("VG", "V", ["g", "0"], "5"), C("RD", "R", ["vdd", "d"], "10k"), M("M1", "d", "g", "0")],
         ["ID(M1)", "VDS(M1)", "bias(M1)"]),
     {"ID(M1)": 0.0009493237062523878, "VDS(M1)": 0.5067629374761224}, {"bias(M1)": "triode"}),

    # D. PMOS, source at +10 V through R_S = 1 kΩ, gate divider to 5 V, R_D = 2 kΩ to ground:
    #    V_SG = 10 − I_D·1k − 5, I_D = ½k(V_SG − V_t)² -> I_D = 1.527864 mA, V_SG = 3.472136 V,
    #    V_SD = 10 − I_D·3k = 5.416408 V (the mirror image of case A)
    ("PMOS, divider gate bias",
     net([VDD] + DIV + [C("RS", "R", ["vdd", "s"], "1k"), C("RD", "R", ["d", "0"], "2k"),
                        M("M1", "d", "g", "s", "PMOS")],
         ["ID(M1)", "VSG(M1)", "VSD(M1)", "bias(M1)"]),
     {"ID(M1)": ID_A, "VSG(M1)": 5.0 - ID_A * 1e3, "VSD(M1)": VDS_A}, {"bias(M1)": "saturation"}),

    # E. Gate at 0.5 V < V_t = 1 V: cut off, so I_D = 0 and the drain sits at V_DD
    ("Cutoff (gate below the threshold)",
     net([VDD, C("VG", "V", ["g", "0"], "0.5"), C("RD", "R", ["vdd", "d"], "10k"), M("M1", "d", "g", "0")],
         ["ID(M1)", "V(d)", "bias(M1)"]),
     {"ID(M1)": 0.0, "V(d)": 10.0}, {"bias(M1)": "cutoff"}),

    # F. Case A with channel-length modulation λ = 0.02 /V: the (1 + λ·V_DS) factor raises the current to
    #    I_D = 1.595756 mA (V_OV = 2.404244 V, V_DS = 5.212733 V), and r_o = 1/(λ·I_D) = 31.333 kΩ
    ("With channel-length modulation (λ = 0.02 /V)",
     net([VDD] + DIV + [C("RD", "R", ["vdd", "d"], "2k"), C("RS", "R", ["s", "0"], "1k"),
                        M("M1", "d", "g", "s", **{"lambda": 0.02})],
         ["ID(M1)", "VDS(M1)", "VOV(M1)"]),
     {"ID(M1)": 0.0015957557522781566, "VDS(M1)": 5.21273274316553, "VOV(M1)": 2.4042442477218433}, {}),

    # G. k given as k' and W/L: k = k'·(W/L) = 0.1 mA/V² × 5 = 0.5 mA/V², so case A again
    ("k given as k' and W/L",
     net([VDD] + DIV + [C("RD", "R", ["vdd", "d"], "2k"), C("RS", "R", ["s", "0"], "1k"),
                        Component(id="M1", type="M", nodes=["d", "g", "s"], model="NMOS",
                                  params={"kprime": 0.1e-3, "WL": 5, "Vt": 1})],
         ["ID(M1)"]),
     {"ID(M1)": ID_A}, {}),

    # H. A question that writes I_D = K(V_GS − V_t)² with no ½: K = 0.25 mA/V² means k = 0.5 mA/V²
    ("K without the ½ factor (K2 = k/2)",
     net([VDD] + DIV + [C("RD", "R", ["vdd", "d"], "2k"), C("RS", "R", ["s", "0"], "1k"),
                        Component(id="M1", type="M", nodes=["d", "g", "s"], model="NMOS",
                                  params={"K2": 0.25e-3, "Vt": 1})],
         ["ID(M1)"]),
     {"ID(M1)": ID_A}, {}),
]

# --- small-signal amplifiers (same bias as case A) -----------------------------------------------
def cs(bypass=True, rd="2k", drain_to_vdd=False):
    comps = [VDD] + DIV + [C("RS", "R", ["s", "0"], "1k"), C("V1", "V", ["in", "0"], ac="1"),
                           C("C1", "C", ["in", "g"], "10u")]
    comps.append(M("M1", "vdd" if drain_to_vdd else "d", "g", "s"))
    if rd and not drain_to_vdd:
        comps.append(C("RD", "R", ["vdd", "d"], rd))
    if bypass:
        comps.append(C("CS", "C", ["s", "0"], "100u"))
    return comps


SS_CASES = [
    # Common source, source bypassed: Av = −gm·R_D = −2.472136, R_in = R1‖R2 = 500 kΩ, R_out = R_D = 2 kΩ
    ("Common source, bypassed source resistor",
     Netlist(analysis="small_signal", query=["gain", "Rin", "Rout", "gm(M1)", "ID(M1)"], components=cs()),
     {"gain": -GM_A * 2e3, "Rin": 500e3, "Rout": 2000.0, "gm(M1)": GM_A, "ID(M1)": ID_A}),

    # Un-bypassed source resistor: Av = −gm·R_D/(1 + gm·R_S) = −1.105573 (exact: the gate draws no current,
    # so unlike the BJT there is no (β+1) correction and the textbook formula is EXACT here)
    ("Common source, un-bypassed source resistor",
     Netlist(analysis="small_signal", query=["gain", "Rin"], components=cs(bypass=False)),
     {"gain": -GM_A * 2e3 / (1 + GM_A * 1e3), "Rin": 500e3}),

    # Source follower: Av = gm·R_S/(1 + gm·R_S) = 0.5527864, R_out = R_S ‖ (1/gm) = 447.2136 Ω
    ("Source follower",
     Netlist(analysis="small_signal", query=["gain", "Rout"],
             components=cs(bypass=False, drain_to_vdd=True)),
     {"gain": GM_A * 1e3 / (1 + GM_A * 1e3), "Rout": 1 / (1 / 1e3 + GM_A)}),
]


def solver_checks(v: Verifier) -> bool:
    ok = True
    if v.exe is None:
        print("[SKIP] ngspice not found: verification checks skipped")
    for name, nl, expected, words in CASES:
        print(f"\n--- {name}")
        try:
            res = get_solver(nl).solve(nl, CONV)
        except (SolverError, NotSupportedError) as e:
            ok &= check(name, False, f"solver error: {e}")
            continue
        for q, a in res.answers.items():
            print(f"     {q:<10} = {a}")
        if SHOW:
            show(res)
        for key, want in expected.items():
            ok &= check(f"{name}: {key} = {want:.7g}", close(res.numeric.get(key), want, 1e-5),
                        f"got {res.numeric.get(key)}")
        for key, word in words.items():
            ok &= check(f"{name}: {key} says '{word}'", word in res.answers.get(key, ""), res.answers.get(key, ""))
        if v.exe:
            rep = v.verify(nl, res)
            print(rep.table())
            ok &= check(f"{name}: verified by ngspice (its own square-law solve)", rep.verified is True,
                        "; ".join(res.verification_notes))
    return ok


def ss_checks(v: Verifier) -> bool:
    ok = True
    for name, nl, expected in SS_CASES:
        print(f"\n--- {name}")
        try:
            res = get_solver(nl).solve(nl, CONV)
        except (SolverError, NotSupportedError) as e:
            ok &= check(name, False, f"solver error: {e}")
            continue
        for q, a in res.answers.items():
            print(f"     {q:<10} = {a}")
        if SHOW:
            show(res)
        for key, want in expected.items():
            ok &= check(f"{name}: {key} = {want:.7g}", close(res.numeric.get(key), want, 1e-5),
                        f"got {res.numeric.get(key)}")
        if v.exe:
            rep = v.verify(nl, res)
            print(rep.table())
            ok &= check(f"{name}: verified by ngspice", rep.verified is True, "; ".join(res.verification_notes))
    return ok


def step_checks() -> bool:
    ok = True
    print()
    res = get_solver(CASES[0][1]).solve(CASES[0][1], CONV)
    acts = [st.action for st in res.trace.key_steps()]
    ok &= check("key steps: the MOSFET model, 'Assume M1 saturation', the square law, the checks",
                any(a.startswith("MOSFET model") for a in acts) and "Assume M1 saturation" in acts
                and any(a.startswith("Square law for M1") for a in acts)
                and "Check the assumptions" in acts, str(acts))
    law = next(st for st in res.trace.key_steps() if st.action.startswith("Square law for M1"))
    ok &= check("the square-law step shows ½·k·V_OV² with the numbers in it",
                "½·k·V_OV²" in (law.equation or "") and "2.472 V" in (law.equation or ""), law.equation)
    chk = next(st for st in res.trace.key_steps() if st.action == "Check the assumptions")
    ok &= check("the check shows V_DS ≥ V_OV with both numbers", "V_DS ≥ V_OV" in (chk.equation or "")
                and "5.416 V" in (chk.equation or ""), chk.equation)

    res = get_solver(CASES[2][1]).solve(CASES[2][1], CONV)      # the triode case
    tried = [st for st in res.trace.all_steps() if st.action.startswith("Try M1 saturation")]
    ok &= check("triode: the full view shows the rejected saturation guess",
                bool(tried) and "triode" in (tried[0].result or ""), str([st.result for st in tried]))
    if SHOW:
        show(res, full=True)

    res = get_solver(SS_CASES[0][1]).solve(SS_CASES[0][1], CONV)
    hyb = next(st for st in res.trace.key_steps() if st.action.startswith("Step 2"))
    ok &= check("the amplifier's hybrid-π step uses √(2·k·I_D) and says r_π is infinite",
                "√(2·k·I_D)" in (hyb.equation or "") and "r_π = ∞" in (hyb.equation or ""), hyb.equation)
    tb = next((st for st in res.trace.key_steps() if st.action.startswith("Textbook check")), None)
    ok &= check("the amplifier is named a common source", tb is not None and "common source" in tb.action,
                str(tb.action if tb else None))
    tb = next((st for st in get_solver(SS_CASES[2][1]).solve(SS_CASES[2][1], CONV).trace.key_steps()
               if st.action.startswith("Textbook check")), None)
    ok &= check("the follower is named a source follower", tb is not None and "source follower" in tb.action,
                str(tb.action if tb else None))
    # a follower with its source resistor bypassed has no output: say so instead of a confusing error
    bad = Netlist(analysis="small_signal", query=["gain"], components=cs(bypass=True, drain_to_vdd=True))
    try:
        get_solver(bad).solve(bad, CONV)
        ok &= check("bypassed follower -> clear 'no signal output' error", False)
    except SolverError as e:
        ok &= check("bypassed follower -> clear 'no signal output' error", "no signal output" in str(e), str(e))
    return ok


def misc_checks() -> bool:
    ok = True
    print()
    r = validate(CASES[0][1])
    ok &= check("validator accepts ID / VGS / VDS / VOV / bias", not r.errors, str(r.errors))
    bad = net(CASES[0][1].components, ["ID(RD)"])
    ok &= check("validator: ID() of a resistor -> error", any("needs a MOSFET" in x for x in validate(bad).errors))
    bad2 = net(CASES[0][1].components, ["IC(M1)"])
    ok &= check("validator: IC() of a MOSFET -> error naming the right query",
                any("ID(M1)" in x for x in validate(bad2).errors), str(validate(bad2).errors))

    no_k = net([VDD] + DIV + [C("RD", "R", ["vdd", "d"], "2k"),
                              Component(id="M1", type="M", nodes=["d", "g", "0"], model="NMOS",
                                        params={"Vt": 1})], ["ID(M1)"])
    try:
        get_solver(no_k).solve(no_k, CONV)
        ok &= check("MOSFET with no k -> clear error", False)
    except SolverError as e:
        ok &= check("MOSFET with no k -> clear error", "conduction parameter k" in str(e), str(e))
    no_vt = net([VDD] + DIV + [C("RD", "R", ["vdd", "d"], "2k"),
                               Component(id="M1", type="M", nodes=["d", "g", "0"], model="NMOS",
                                         params={"k": 0.5e-3})], ["ID(M1)"])
    try:
        get_solver(no_vt).solve(no_vt, CONV)
        ok &= check("MOSFET with no Vt -> clear error", False)
    except SolverError as e:
        ok &= check("MOSFET with no Vt -> clear error", "threshold voltage" in str(e), str(e))

    # a MOSFET biased in the triode region cannot be a linear amplifier
    tri = Netlist(analysis="small_signal", query=["gain"],
                  components=[VDD, C("VG", "V", ["g", "0"], "5"), C("RD", "R", ["vdd", "d"], "10k"),
                              M("M1", "d", "g", "0"), C("V1", "V", ["in", "0"], ac="1"),
                              C("C1", "C", ["in", "g"], "10u")])
    try:
        get_solver(tri).solve(tri, CONV)
        ok &= check("a MOSFET in triode -> clear error, not a gain", False)
    except SolverError as e:
        ok &= check("a MOSFET in triode -> clear error, not a gain",
                    "triode" in str(e) and "saturation" in str(e), str(e))
    return ok


def parser_checks() -> bool:
    ok = True
    print()
    q = ("An NMOS transistor with k = 0.5 mA/V^2 and Vt = 1 V is biased from VDD = 10 V with RD = 2 kOhm, "
         "RS = 1 kOhm and a gate divider of two 1 MOhm resistors. Find the drain current.")
    good = net(CASES[0][1].components, ["ID(M1)"], q)
    ok &= check("NMOS bias netlist -> consistent", not consistency_errors(q, good), str(consistency_errors(q, good)))
    qp = ("A PMOS transistor with k = 0.5 mA/V^2 and Vt = 1 V has its source at +10 V through RS = 1 kOhm, "
          "a gate divider of two 1 MOhm resistors and RD = 2 kOhm from the drain to ground. Find ID.")
    wrong = net([c.model_copy(update={"model": "NMOS"}) if c.id == "M1" else c for c in CASES[3][1].components],
                ["ID(M1)"], qp)
    e = consistency_errors(qp, wrong)
    ok &= check("'PMOS' in the question but model NMOS -> rejected", any("PMOS" in x for x in e), str(e))
    ok &= check("the right PMOS netlist -> consistent", not consistency_errors(qp, CASES[3][1]),
                str(consistency_errors(qp, CASES[3][1])))
    # a MOSFET with no k / no Vt would be unsolvable: catch it at parse time
    bare = net([VDD] + DIV + [C("RD", "R", ["vdd", "d"], "2k"),
                              Component(id="M1", type="M", nodes=["d", "g", "0"], model="NMOS")],
               ["ID(M1)"], q)
    e = consistency_errors(q, bare)
    ok &= check("MOSFET with no k and no Vt -> both reported",
                any("conduction parameter" in x for x in e) and any("threshold voltage" in x for x in e), str(e))
    qv = ("An NMOS with k = 0.5 mA/V^2 and Vt = 2 V is biased from VDD = 10 V with RD = 2 kOhm, RS = 1 kOhm "
          "and a gate divider of two 1 MOhm resistors. Find ID.")
    e = consistency_errors(qv, net(CASES[0][1].components, ["ID(M1)"], qv))
    ok &= check("Vt = 2 V in the question but 1 V in the netlist -> rejected",
                any('"Vt": 2' in x for x in e), str(e))
    return ok


LIVE = [
    ("An NMOS transistor with k = 0.5 mA/V^2 and Vt = 1 V is biased from VDD = 10 V. A gate divider of two "
     "1 MOhm resistors sets the gate voltage, RD = 2 kOhm is in the drain and RS = 1 kOhm in the source. "
     "Find the drain current and VDS.",
     [{"kind": "ID", "value": ID_A}, {"kind": "VDS", "value": VDS_A}]),
    ("A diode-connected NMOS (gate tied to the drain) with k = 0.5 mA/V^2 and Vt = 1 V is fed from VDD = 10 V "
     "through a 10 kOhm resistor. Find the drain current and VGS.",
     [{"kind": "ID", "value": 0.0007292121597166108}, {"kind": "VGS", "value": 2.7078784028338916}]),
    ("An NMOS transistor with k = 0.5 mA/V^2 and Vt = 1 V has its gate held at 5 V, its source grounded and a "
     "10 kOhm resistor from VDD = 10 V to the drain. Find the drain current and the region of operation.",
     [{"kind": "ID", "value": 0.0009493237062523878}]),
    ("A PMOS transistor with k = 0.5 mA/V^2 and Vt = 1 V has its source connected to +10 V through "
     "RS = 1 kOhm, a gate divider of two 1 MOhm resistors, and RD = 2 kOhm from the drain to ground. "
     "Find the drain current.",
     [{"kind": "ID", "value": ID_A}]),
    ("A common-source amplifier uses an NMOS with k = 0.5 mA/V^2 and Vt = 1 V, VDD = 10 V, a gate divider of "
     "two 1 MOhm resistors, RD = 2 kOhm, and RS = 1 kOhm bypassed by a 100 uF capacitor. The signal is coupled "
     "to the gate through a 10 uF capacitor. Find the mid-band voltage gain and the input resistance.",
     [{"kind": "gain", "value": -GM_A * 2e3, "tol": 0.02}, {"kind": "Rin", "value": 500e3, "tol": 0.02}]),
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
            print("params:", [(c["id"], c.get("params")) for c in st["netlist"]["components"]
                              if c["type"] == "M"])
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
    all_ok &= ss_checks(ver)
    all_ok &= step_checks()
    all_ok &= misc_checks()
    all_ok &= parser_checks()
    if "--live" in sys.argv:
        all_ok &= live_checks()
    print("\nPHASE 3 STEP 5:", "ALL PASSED" if all_ok else "SOME CHECKS FAILED")