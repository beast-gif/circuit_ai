"""
Step 4 checks: every Step 3 circuit is solved, then cross-checked with ngspice.

    python test_step4.py           # all hand-checked circuits + a planted wrong answer (free, no LLM)
    python test_step4.py --live    # also: real questions -> parse -> solve -> verify (Velona)
    python test_step4.py --deck    # also prints the SPICE deck generated for the divider
"""
import sys

from schemas import Component, Netlist
from solvers import SolverError, get_solver
from test_step3 import CASES, CONV, LIVE
from verifier import Verifier, build_deck


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"\n       {detail}" if detail and not cond else ""))
    return cond


def C(id, type, nodes, value=None, **kw):
    return Component(id=id, type=type, nodes=nodes, value=value, **kw)


def offline_checks(v: Verifier) -> bool:
    ok = True
    for name, nl, _ in CASES:
        res = get_solver(nl).solve(nl, CONV)
        rep = v.verify(nl, res)
        print(f"\n--- {name}")
        print(rep.table())
        ok &= check(f"{name}: verified by ngspice", rep.verified is True,
                    "; ".join(res.verification_notes) or "no checks ran")

    # A planted wrong answer MUST be caught
    print("\n--- Planted mistake: V(out) of the divider changed from 6.667 V to 7 V")
    nl = CASES[0][1]
    res = get_solver(nl).solve(nl, CONV)
    res.numeric["V(out)"] = 7.0
    res.answers["V(out)"] = "7 V"
    rep = v.verify(nl, res)
    print(rep.table())
    bad = next((c for c in rep.checks if c.query == "V(out)"), None)
    ok &= check("wrong V(out) is flagged MISMATCH and the result is not verified",
                bad is not None and bad.status == "MISMATCH" and rep.verified is False)

    # Values that are really 0: ngspice leaves ~2.5e-8 V through the 1 TΩ rshunt on an open node
    near0 = v._compare("V(y)", complex(0.0), complex(2.5e-8), "V")
    ok &= check("0 V vs 2.5e-8 V counts as equal (absolute floor 1 µV)", near0.status == "ok", str(near0))
    real = v._compare("V(y)", complex(0.0), complex(1e-3), "V")
    ok &= check("0 V vs 1 mV is still a MISMATCH", real.status == "MISMATCH", str(real))

    # Symbolic values cannot be simulated -> verified stays None (not False)
    print("\n--- Symbolic divider")
    nl = Netlist(analysis="dc", query=["V(out)"], components=[
        C("V1", "V", ["in", "0"], "Vs"), C("R1", "R", ["in", "out"], "R1"), C("R2", "R", ["out", "0"], "R2")])
    res = get_solver(nl).solve(nl, CONV)
    rep = v.verify(nl, res)
    print(rep.table())
    ok &= check("symbolic circuit -> verified = None (skipped, not failed)", rep.verified is None)
    return ok


def live_checks(v: Verifier) -> bool:
    from llm_client import VelonaClient
    from parsing import parse_text

    client = VelonaClient()
    ok = True
    for i, question in enumerate(LIVE, 1):
        print(f"\n=== L{i}: {question}")
        pr = parse_text(question, client=client)
        if not pr.ok:
            ok &= check(f"L{i} parsed", False, str(pr.errors))
            continue
        print(pr.netlist.to_spice())
        try:
            res = get_solver(pr.netlist).solve(pr.netlist, CONV)
        except SolverError as e:
            ok &= check(f"L{i} solved", False, str(e))
            continue
        rep = v.verify(pr.netlist, res)
        print(rep.table())
        ok &= check(f"L{i} verified by ngspice", rep.verified is True, "; ".join(res.verification_notes))
    print("\n" + client.usage_report())
    return ok


if __name__ == "__main__":
    v = Verifier()
    if v.exe is None:
        print("[FAIL] ngspice not found.\n"
              "  1. Download ngspice-XX_64.zip from https://ngspice.sourceforge.io/download.html\n"
              "  2. Extract it to C:\\Spice64  (so C:\\Spice64\\bin\\ngspice_con.exe exists)\n"
              "  3. Or set solver.ngspice_path in config.yaml to where ngspice_con.exe is")
        sys.exit(1)
    print(f"ngspice: {v.exe}")

    if "--deck" in sys.argv:
        nl = CASES[0][1]
        print("\nSPICE deck for the divider (with a current sense source on R1):")
        print(build_deck(nl, sources={"V1": (10.0, 0.0, 0.0)}, sense=["R1"]))

    all_ok = offline_checks(v)
    if "--live" in sys.argv:
        all_ok &= live_checks(v)
    print("\nSTEP 4:", "ALL PASSED" if all_ok else "SOME CHECKS FAILED")