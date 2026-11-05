"""
Step 1 checks.

    python test_step1.py          # offline checks (schemas, value parsing)
    python test_step1.py --ping   # also calls every Velona model once (costs a few paise)
"""
import sys

from pydantic import ValidationError

from schemas import Component, Netlist, StepTrace, parse_value


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    return cond


def offline_checks() -> bool:
    ok = True

    # --- value parsing ---
    cases = {"1k": 1e3, "2.2k": 2.2e3, "4.7uF": 4.7e-6, "10mA": 1e-2, "1M": 1e6,
             "1meg": 1e6, "100n": 1e-7, "15pF": 1.5e-11, "10V": 10.0, "1F": 1.0,
             "5Ω": 5.0, "1e3": 1e3, 47: 47.0}
    for raw, want in cases.items():
        got = parse_value(raw)
        ok &= check(f"parse_value({raw!r}) = {want:g}", abs(got - want) <= 1e-12 * max(1, abs(want)))

    # --- valid voltage-divider netlist ---
    divider = Netlist(
        question="Find Vout for a 10 V source with R1 = 1k and R2 = 2k in series.",
        analysis="dc",
        query=["V(out)"],
        components=[
            Component(id="V1", type="V", nodes=["in", "gnd"], value="10"),
            Component(id="R1", type="R", nodes=["in", "out"], value="1k"),
            Component(id="R2", type="R", nodes=["out", "0"], value="2k"),
        ],
    )
    ok &= check("divider netlist builds", True)
    ok &= check("'gnd' normalised to '0'", divider.get("V1").nodes == ["in", "0"])
    ok &= check("nodes() = [in, 0, out]", divider.nodes() == ["in", "0", "out"])
    print("\n" + divider.to_spice() + "\n")

    # --- symbolic values allowed ---
    sym = Component(id="R1", type="R", nodes=["a", "b"], value="R")
    ok &= check("symbolic value 'R' accepted", sym.numeric_value() is None)

    # --- invalid netlists must be rejected ---
    bad = [
        ("BJT with 2 nodes", dict(id="Q1", type="Q", nodes=["c", "b"], model="NPN")),
        ("negative resistor", dict(id="R1", type="R", nodes=["a", "b"], value="-1k")),
        ("resistor shorted to itself", dict(id="R1", type="R", nodes=["a", "a"], value="1k")),
        ("resistor without value", dict(id="R1", type="R", nodes=["a", "b"])),
        ("garbage value", dict(id="R1", type="R", nodes=["a", "b"], value="abc$")),
    ]
    for name, kw in bad:
        try:
            Component(**kw)
            ok &= check(f"rejects {name}", False)
        except (ValidationError, ValueError):
            ok &= check(f"rejects {name}", True)

    try:
        Netlist(analysis="dc", query=["V(out)"], components=[
            Component(id="R1", type="R", nodes=["a", "0"], value="1k"),
            Component(id="R1", type="R", nodes=["a", "0"], value="2k")])
        ok &= check("rejects duplicate ids", False)
    except ValidationError:
        ok &= check("rejects duplicate ids", True)

    # --- step trace levels ---
    t = StepTrace()
    t.add("Label nodes: in, out, ground", level="full")
    t.add("KCL at node out", "(Vout-10)/1000 + Vout/2000 = 0")
    t.add("Solve", result="Vout = 6.667 V")
    ok &= check("trace: 2 key steps, 3 total", len(t.key_steps()) == 2 and len(t.all_steps()) == 3)

    return ok


def ping_models() -> bool:
    from llm_client import VelonaClient
    client = VelonaClient()
    ok = True
    for role, model in client.models.items():
        try:
            data = client.chat_json(role, "Reply only with JSON.", 'Return {"ok": true}')
            ok &= check(f"{role:<13} -> {model}", data.get("ok") is True)
        except Exception as e:
            print(f"[FAIL] {role:<13} -> {model}: {e}")
            ok = False
    print("\n" + client.usage_report())
    return ok


if __name__ == "__main__":
    all_ok = offline_checks()
    if "--ping" in sys.argv:
        all_ok &= ping_models()
    print("\nSTEP 1:", "ALL PASSED" if all_ok else "SOME CHECKS FAILED")
