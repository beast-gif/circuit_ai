"""
Phase 4, Step 2 checks: ngspice verification of oscillators (free -- no LLM calls).

    python test_p4_2.py            # verify 6 oscillators against ngspice
    python test_p4_2.py --decks    # also print the deck ngspice was given
    python test_p4_2.py --steps    # also print the solver's key steps

The solver gets its answer from det A(s) = 0. ngspice is given the same circuit with the op-amp as a plain
VCVS, a 1 mV charge on one capacitor and nothing else, and integrates it in time. Two independent numbers
come out of V(out):

    the zero crossings  -> the frequency          compared with w0/2pi
    ln(peak) vs t       -> the real part of the pole, i.e. whether the amplitude holds, grows or dies

Hand-worked for R = 10 kOhm, C = 10 nF (1/RC = 10^4 rad/s):
    Wien, R2 = 20 k     poles  0     ± j10000   -> 1591.5494 Hz, amplitude constant
    Wien, R2 = 15 k     poles -2500  ± j9682.5  -> 1541.0 Hz,    amplitude decays (6 cycles: x 6e-5)
    Wien, R2 = 30 k     poles +5000  ± j8660.3  -> 1378.8 Hz,    amplitude grows  (6 cycles: x 3e9)
    phase shift, 29R    poles  0     ± j4082.5  -> 649.7473 Hz,  amplitude constant
"""
import math
import sys

from llm_client import load_config, load_conventions
from schemas import Component, Netlist, is_symbolic
from solvers import get_solver
from verifier import Verifier, build_deck

CONV = load_conventions()
SHOW = "--steps" in sys.argv
DECKS = "--decks" in sys.argv
F_WIEN = 1 / (2 * math.pi * 10e3 * 10e-9)
F_PS = F_WIEN / math.sqrt(6)


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"\n       {detail}" if detail and not cond else ""))
    return cond


def C(id, type, nodes, value=None, **kw):
    return Component(id=id, type=type, nodes=nodes, value=value, **kw)


def U(id, out, plus, minus):
    return Component(id=id, type="OPAMP", nodes=[out, plus, minus], model="ideal")


def wien(r2="20k", r="10k", c="10n", query=None):
    return Netlist(analysis="ac", query=query or ["fosc"], question="A Wien-bridge oscillator.",
                   components=[C("R1", "R", ["n", "0"], "10k"), C("R2", "R", ["n", "out"], r2),
                               C("Rs", "R", ["out", "x"], r), C("Cs", "C", ["x", "p"], c),
                               C("Rp", "R", ["p", "0"], r), C("Cp", "C", ["p", "0"], c),
                               U("U1", "out", "p", "n")])


def phase_shift(rf="290k", query=None):
    return Netlist(analysis="ac", query=query or ["fosc"], question="An RC phase-shift oscillator.",
                   components=[C("Rf", "R", ["n", "out"], rf),
                               C("C1", "C", ["out", "a"], "10n"), C("Ra", "R", ["a", "0"], "10k"),
                               C("C2", "C", ["a", "b"], "10n"), C("Rb", "R", ["b", "0"], "10k"),
                               C("C3", "C", ["b", "c"], "10n"), C("Rc", "R", ["c", "n"], "10k"),
                               U("U1", "out", "0", "n")])


# (name, netlist, what ngspice must agree about)
CASES = [
    ("Wien bridge on the condition (R2 = 2 R1)", wien(), "verified"),
    ("Wien bridge, gain too low (R2 = 15 k)", wien("15k", query=["fosc", "oscillates"]), "verified"),
    ("Wien bridge, gain too high (R2 = 30 k)", wien("30k", query=["fosc", "oscillates"]), "verified"),
    ("Wien bridge designed by the solver (R2 unknown)", wien("Rf", query=["fosc", "gain_condition"]),
     "verified"),
    ("RC phase-shift on the condition (Rf = 29 R)", phase_shift(), "verified"),
    ("RC phase-shift designed by the solver (Rf unknown)", phase_shift("Rf", ["fosc", "gain_condition"]),
     "verified"),
]


def main() -> bool:
    cfg = load_config()
    v = Verifier(cfg)
    if v.exe is None:
        print("ngspice not found: set solver.ngspice_path in config.yaml")
        return False
    print(f"ngspice: {v.exe}\n")

    ok = True
    for name, nl, want in CASES:
        print(f"--- {name}")
        res = get_solver(nl).solve(nl, CONV)
        for q, a in res.answers.items():
            print(f"     {q:<16} = {a}")
        if SHOW:
            for st in res.trace.key_steps():
                print(f"      {st.step:>2}. {st.action}"
                      + (f"\n          {st.equation}" if st.equation else "")
                      + (f"\n          => {st.result}" if st.result else ""))
        if DECKS:
            # same substitution the verifier makes: a design question's symbolic value is simulated at the
            # value the solver computed for it, and the capacitors are added separately (with their IC=)
            shown = []
            for c in nl.components:
                if c.type.value == "C":
                    continue
                if c.value not in (None, "") and is_symbolic(c.value):
                    c = c.model_copy(update={"value": repr(res.numeric.get("gain_condition"))})
                shown.append(c)
            print(build_deck(nl.model_copy(update={"components": shown}), {}, conventions=CONV)
                  + "* (the capacitors are added by the verifier with their IC= kick)\n")
        rep = v.verify(nl, res)
        print(rep.table())                      # the table already prints the notes
        ok &= check(f"{name}: ngspice agrees", rep.verified is True,
                    f"verified={rep.verified}; " + "; ".join(f"{c.query}: {c.solver} vs {c.spice} ({c.note})"
                                                             for c in rep.checks))
        # the frequency ngspice measured must also match the hand value, not just the solver
        fq = next((c for c in rep.checks if c.query.startswith("fosc")), None)
        if fq is not None and fq.rel_err is not None:
            ok &= check(f"{name}: ngspice frequency within 2 % of the solver's", fq.rel_err <= 0.02,
                        f"rel = {fq.rel_err}")
        print()

    # the amplitude verdict must be the discriminating one: a wrong-gain circuit cannot pass as "constant"
    nl = wien("15k", query=["fosc"])
    res = get_solver(nl).solve(nl, CONV)
    rep = v.verify(nl, res)
    line = next((c for c in rep.checks if "pole.re" in c.query), None)
    ok &= check("gain too low: ngspice measures a NEGATIVE envelope slope",
                line is not None and line.spice.lstrip().startswith("-"),
                f"{line.query if line else None}: {line.spice if line else None}")

    nl = wien("30k", query=["fosc"])
    res = get_solver(nl).solve(nl, CONV)
    rep = v.verify(nl, res)
    line = next((c for c in rep.checks if "pole.re" in c.query), None)
    ok &= check("gain too high: ngspice measures a POSITIVE envelope slope",
                line is not None and not line.spice.lstrip().startswith("-"),
                f"{line.query if line else None}: {line.spice if line else None}")

    # a different RC pair, to be sure the frequency is measured and not assumed
    nl = wien("20k", "4.7k", "22n")
    res = get_solver(nl).solve(nl, CONV)
    rep = v.verify(nl, res)
    fq = next((c for c in rep.checks if c.query.startswith("fosc")), None)
    ok &= check("R = 4.7 k, C = 22 nF: ngspice measures 1539.2 Hz",
                fq is not None and fq.rel_err is not None and fq.rel_err <= 0.02,
                f"{fq.solver if fq else None} vs {fq.spice if fq else None}")
    return ok


if __name__ == "__main__":
    print("\nPHASE 4 STEP 2:", "ALL PASSED" if main() else "SOME CHECKS FAILED")