"""
Phase 2, Step 5 checks: fixes from the 120-run Phase 1 rerun (free, no LLM).

    python test_p2_5.py

  dc03#2  R2 || R3 parsed as a dead-end pair (both n1–out, "out" goes nowhere) -> V across R3 = 0
  dep04#1 "6 V source in series with 2 kOhm forms a loop" parsed with R1 going to node z -> 1.2 V, not 1.5 V
  flt04#2 "DC gain" asked as the "filter" query -> K was there but not recorded as a DC gain
  opa06#1 gain + input resistance with no source: the parser was told to remove V1, then H had no input
"""
from benchmark.run_benchmark import check_expectation
from llm_client import load_conventions
from parsing.text_parser import consistency_errors
from schemas import Component, Netlist
from solvers import get_solver

CONV = load_conventions()


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"\n       {detail}" if detail and not cond else ""))
    return cond


def C(id, type, nodes, value=None, **kw):
    return Component(id=id, type=type, nodes=nodes, value=value, **kw)


def nl(components, query, question, analysis="dc"):
    return Netlist(analysis=analysis, query=query, components=components, question=question)


ok = True

# --- dc03: dead-end parallel pair ------------------------------------------------------------------
q = ("A 24 V supply drives R1 = 4 kOhm in series with the parallel combination of R2 = 12 kOhm and R3 = 6 kOhm. "
     "Find the voltage across R3.")
bad = nl([C("V1", "V", ["in", "0"], "24"), C("R1", "R", ["in", "n1"], "4k"), C("R2", "R", ["n1", "out"], "12k"),
          C("R3", "R", ["n1", "out"], "6k")], ["V(n1,out)"], q)
e = consistency_errors(q, bad)
ok &= check("dc03: R2, R3 both n1–out with 'out' going nowhere -> rejected (dead end)",
            any("Node out is reached only through R2, R3" in x for x in e), str(e))
good = nl([C("V1", "V", ["in", "0"], "24"), C("R1", "R", ["in", "out"], "4k"), C("R2", "R", ["out", "0"], "12k"),
           C("R3", "R", ["out", "0"], "6k")], ["V(out)"], q)
ok &= check("dc03: R2 || R3 from out to ground -> consistent", not consistency_errors(q, good),
            str(consistency_errors(q, good)))
qz = "Find the equivalent resistance between terminals a and b of R1 = 10 Ohm in parallel with R2 = 20 Ohm."
port = nl([C("R1", "R", ["a", "b"], "10"), C("R2", "R", ["a", "b"], "20")], ["Z(a,b)"], qz)
ok &= check("Z(a,b) of two parallel resistors (terminals are open on purpose) -> not flagged",
            not any("reached only through" in x for x in consistency_errors(qz, port)),
            str(consistency_errors(qz, port)))

# --- dep04: the source and resistor close their own loop ------------------------------------------
q = ("A 6 V source in series with a 2 kOhm resistor forms a loop with current I. A current-controlled voltage "
     "source with transresistance 500 Ohm produces a voltage of 500 times I at node z (negative terminal at "
     "ground), and a 1 kOhm resistor connects z to ground. Find the voltage at node z.")
wrong = nl([C("V1", "V", ["n1", "0"], "6"), C("R1", "R", ["n1", "z"], "2k"),
            C("H1", "H", ["z", "0"], "500", control="R1"), C("R2", "R", ["z", "0"], "1k")], ["V(z)"], q)
e = consistency_errors(q, wrong)
ok &= check("dep04: R1 wired to z instead of closing the 6 V loop -> rejected",
            any("form a loop on their own" in x for x in e), str(e))
right = nl([C("V1", "V", ["n1", "0"], "6"), C("R1", "R", ["n1", "0"], "2k"),
            C("H1", "H", ["z", "0"], "500", control="R1"), C("R2", "R", ["z", "0"], "1k")], ["V(z)"], q)
e = consistency_errors(q, right)
ok &= check("dep04: R1 across the source (its own loop) -> consistent", not e, str(e))
res = get_solver(right).solve(right, CONV)
ok &= check("dep04: V(z) = 500 × 3 mA = 1.5 V (magnitude)", abs(abs(res.numeric["V(z)"]) - 1.5) < 1e-9,
            str(res.answers))

# --- flt04: "DC gain" answered through the filter query --------------------------------------------
q = ("An RC low-pass filter has R1 = 1 kOhm in series and C = 1 uF from the output to ground, and a load RL = 1 kOhm "
     "is connected across C. Find the DC gain and the cutoff frequency.")
f = nl([C("V1", "V", ["in", "0"], ac="1"), C("R1", "R", ["in", "out"], "1k"), C("C1", "C", ["out", "0"], "1u"),
        C("RL", "R", ["out", "0"], "1k")], ["filter", "fc"], q, "ac")
res = get_solver(f).solve(f, CONV)
good, detail = check_expectation({"kind": "dc_gain", "value": 0.5}, res.numeric, res.symbolic, 0.01)
ok &= check("flt04: 'filter' query records filter.dc_gain = 0.5 (benchmark kind dc_gain matches)", good, detail)
ok &= check("flt04: no consistency complaint for the correct netlist", not consistency_errors(q, f),
            str(consistency_errors(q, f)))

# --- opa06 (op-amp benchmark): "voltage gain and input resistance" with no source in the question -------
q = "An inverting amplifier has Rin = 2.2 kOhm and Rf = 22 kOhm. Find its voltage gain and its input resistance."
amp = [C("Rin", "R", ["in", "n1"], "2.2k"), C("Rf", "R", ["n1", "out"], "22k"),
       Component(id="U1", type="OPAMP", nodes=["out", "0", "n1"], model="ideal")]
v1 = C("V1", "V", ["in", "0"], "1")
e = consistency_errors(q, nl([v1] + amp, ["H(out,in)", "Z(in,0)"], q))
ok &= check("opa06 attempt 1: V1 + Z(in,0) -> 'keep V1, use Rin' (not 'remove it')",
            any("Keep V1" in x and '"Rin"' in x for x in e) and not any("remove it" in x for x in e), str(e))
e = consistency_errors(q, nl(amp, ["H(out,in)", "Z(in,0)"], q))
ok &= check("opa06 attempt 2: no source but a gain query -> 'needs an input source'",
            any("needs an input source" in x for x in e), str(e))
good = nl([v1] + amp, ["H(out,in)", "Rin"], q)
ok &= check("opa06: V1 + H(out,in) + Rin -> consistent", not consistency_errors(q, good),
            str(consistency_errors(q, good)))
res = get_solver(good).solve(good, CONV)
ok &= check("opa06: gain = -10, Rin = 2.2 kΩ", abs(res.numeric["H(out,in)"] + 10) < 1e-9
            and abs(res.numeric["Rin"] - 2200) < 1e-6, str(res.answers))
qz = "Find the equivalent resistance between terminals a and b of R1 = 10 Ohm in parallel with R2 = 20 Ohm."
ok &= check("impedance-only question with no source -> still no 'needs an input source'",
            not any("needs an input source" in x for x in consistency_errors(qz, port)))

print("\nPHASE 2 STEP 5 FIXES:", "ALL PASSED" if ok else "SOME CHECKS FAILED")