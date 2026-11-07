"""
Step 5 checks.

    python test_step5.py           # template, number guard, fallback, LaTeX cleanup, arithmetic checker (free)
    python test_step5.py --live    # real explanations (key + full) and a theory question via Velona
"""
import sys

from conceptual import answer_conceptual
from explainer import allowed_numbers, explain, render_steps, select_steps, unsupported_numbers
from solvers import get_solver
from test_step3 import CASES, CONV


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"\n       {detail}" if detail and not cond else ""))
    return cond


class FakeClient:
    """Stands in for VelonaClient: returns canned replies, counts calls."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def chat(self, role, messages, **kw):
        self.calls += 1
        return self.replies.pop(0) if self.replies else ""


def case(name):
    nl = next(n for t, n, _ in CASES if t == name)
    nl = nl.model_copy(update={"question": f"({name})"})
    return nl, get_solver(nl).solve(nl, CONV)


def offline_checks() -> bool:
    ok = True
    nl, res = case("Voltage divider")

    # --- template rendering --------------------------------------------
    key, full = select_steps(res, "key"), select_steps(res, "full")
    ok &= check(f"divider: key steps ({len(key)}) fewer than full steps ({len(full)})", len(key) < len(full))
    ex = explain(nl, res, "key", use_llm=False)
    print("\n--- Template, key level (divider) ---\n" + ex.text + "\n")
    ok &= check("template has final answers + verification line + 'complete solution' hint",
                "**Final answers**" in ex.text and "ngspice" in ex.text and ex.has_more)
    ok &= check("template keeps the exact answer 6.667 V", "6.667 V" in ex.text)

    # --- number guard ---------------------------------------------------
    allowed = allowed_numbers([nl.question, nl.to_spice(), *res.answers.values()]
                              + [f"{s.action} {s.equation or ''} {s.result or ''}" for s in res.trace.all_steps()])
    good = "Approach: nodal analysis. `(V_out - 10)/1k + V_out/2k = 0` gives `V_out = 6.667 V`, I = 3.333 mA = 0.003333 A."
    ok &= check("guard accepts numbers from the solution (incl. 3.333 mA = 0.003333 A)",
                unsupported_numbers(good, allowed) == [], str(unsupported_numbers(good, allowed)))
    ok &= check("guard accepts rounding within 0.5 % (6.67 for 6.667)",
                unsupported_numbers("V_out = 6.67 V", allowed) == [])
    bad = unsupported_numbers("so V_out = 7.5 V and I = 2.5 mA", allowed)
    ok &= check("guard flags invented numbers 7.5 and 2.5", bad == ["2.5", "7.5"], str(bad))
    ok &= check("guard ignores digits inside names (R1, V_n1, C2) and small integers",
                unsupported_numbers("KCL at n1 using R1, R2 and step 3; 2π", allowed) == [])

    # --- LLM path with a fake client -------------------------------------
    good_reply = ("Approach: nodal analysis.\n\n**1. KCL at node out**\nCurrents leaving the node sum to zero.\n"
                  "`(V_out - 10)/1k + V_out/2k = 0`\n→ `V_out = 6.667 V`")
    fake = FakeClient([good_reply])
    ex = explain(nl, res, "key", client=fake)
    ok &= check("clean LLM reply is used (source = llm, 1 call)", ex.source == "llm" and fake.calls == 1,
                str(ex.warnings))
    ok &= check("final answers are appended by code, not the LLM", ex.text.count("**Final answers**") == 1)

    fake = FakeClient(["V_out = 7.5 V", "V_out = 7.5 V again"])
    ex = explain(nl, res, "key", client=fake)
    ok &= check("LLM inventing a number twice -> falls back to the template", ex.source == "template"
                and fake.calls == 2 and "6.667 V" in ex.text, str(ex.warnings))
    print("     warnings:", ex.warnings)

    fake = FakeClient(["V_out = 7.5 V", good_reply])
    ex = explain(nl, res, "key", client=fake)
    ok &= check("LLM fixes the number on retry -> LLM text used", ex.source == "llm" and fake.calls == 2)

    # --- full level shows every step -------------------------------------
    nl2, res2 = case("Series RLC band-pass")
    ex = explain(nl2, res2, "full", use_llm=False)
    ok &= check("full level: no 'ask for complete' hint", not ex.has_more)
    return ok


def fixes_checks() -> bool:
    from conceptual import arithmetic_errors
    from explainer import clean_latex

    ok = True
    print()
    # --- 1. LaTeX cleanup -------------------------------------------------
    raw = r"The source \(V_1\) fixes V_{\text{out}}; \frac{1}{2} \cdot 10 \approx 5 and \omega = 2\pi f"
    got = clean_latex(raw)
    ok &= check("LaTeX -> plain text", "\\" not in got and "V_1" in got and "V_out" in got
                and "(1)/(2) · 10 ≈ 5" in got and "ω = 2π f" in got, got)

    # --- 2. trace: no duplicate / odd key steps -----------------------------
    nl, res = case("Voltage divider")
    results = [s.result for s in res.trace.key_steps() if s.result]
    ok &= check("divider: no repeated result in key steps", len(results) == len(set(results)), str(results))
    nl, res = case("RC low-pass: H(s), pole, fc")
    key_actions = [s.action for s in res.trace.key_steps()]
    ok &= check("RC: fc shown once in key steps", key_actions.count("Cutoff frequency") == 0, str(key_actions))
    tf = next(s for s in res.trace.key_steps() if s.action.startswith("Transfer function: apply"))
    ok &= check("transfer step equation is 'V1 = 1', not prose", tf.equation == "V1 = 1", str(tf.equation))

    # --- 3. arithmetic checker -------------------------------------------
    correct = ["A_f = 1000 / (1 + 1000 × 0.1) = 1000 / 101 ≈ 9.90",
               "gm = IC/VT = 0.002 / 0.025 = 0.08 A/V",
               "fc = 1/(2π × 10⁻³) = 159.2 Hz",
               "ΔAf/Af = 0.5/51 ≈ 1%",
               "gm = 1 mA / 25 mV = 40 mA/V (units: not checked, must not be flagged)",
               "1. the 5 + 3 = 8 rule"]
    for line in correct:
        e = arithmetic_errors(line)
        ok &= check(f"arithmetic OK: {line[:45]}", not e, str(e))
    for line in ["A_f = 500 / 51 ≈ 9.70", "change = 0.5/51 ≈ 0.5%", "rπ = 100 / 0.08 = 125"]:
        e = arithmetic_errors(line)
        ok &= check(f"arithmetic slip caught: {line}", len(e) == 1, str(e))
        print("      ", e)

    from conceptual import answer_conceptual as ac
    bad = "**Worked example**\nA_f = 500 / 51 ≈ 9.70"
    good = "**Worked example**\nA_f = 500 / 51 ≈ 9.80"
    fake = FakeClient([bad, good])
    ex = ac("q", "key", client=fake, conventions=CONV)
    ok &= check("theory: slip fixed on retry -> no warning", fake.calls == 2 and "⚠️" not in ex.text, ex.text)
    fake = FakeClient([bad, bad])
    ex = ac("q", "key", client=fake, conventions=CONV)
    ok &= check("theory: slip twice -> answer shown WITH a warning naming the line",
                "⚠️ Possible arithmetic slip" in ex.text and "9.80" in ex.text, ex.text)
    return ok


def live_checks() -> bool:
    from llm_client import VelonaClient

    client = VelonaClient()
    ok = True
    for name, level in [("Voltage divider", "key"), ("Series RLC band-pass", "key"),
                        ("Series RLC band-pass", "full"), ("RC low-pass: H(s), pole, fc", "key")]:
        nl, res = case(name)
        ex = explain(nl, res, level, topic="ac" if "RC" in name else "dc", client=client)
        print(f"\n===== {name}  [{level}]  source={ex.source} =====\n{ex.text}")
        if ex.warnings:
            print("warnings:", ex.warnings)
        ok &= check(f"{name} [{level}] explained by the LLM", ex.source == "llm", str(ex.warnings))

    q = "Why does negative feedback stabilize the gain of an amplifier?"
    for level in ("key", "full"):
        ex = answer_conceptual(q, level, client=client)
        print(f"\n===== Theory [{level}] =====\n{ex.text}")
        ok &= check(f"theory answer [{level}]", ex.source == "llm" and "Worked example" in ex.text)

    print("\n" + client.usage_report())
    return ok


if __name__ == "__main__":
    all_ok = offline_checks()
    all_ok &= fixes_checks()
    if "--live" in sys.argv:
        all_ok &= live_checks()
    print("\nSTEP 5:", "ALL PASSED" if all_ok else "SOME CHECKS FAILED")