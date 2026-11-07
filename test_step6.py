"""
Step 6 checks.

    python test_step6.py                # renderer, benchmark file, graph flow with a fake parser (free)
    python test_step6.py --live         # one real question through the whole graph, you confirm in the terminal
Then the real benchmark:
    python benchmark/run_benchmark.py --limit 5
    python benchmark/run_benchmark.py
"""
import sys
from pathlib import Path

from benchmark.run_benchmark import candidates, check_expectation, load
from parsing.text_parser import ParseResult
from renderer import describe, render_svg
from schemas import Component, Netlist
from test_step3 import CASES
from validator import ValidationReport

RENDERS = Path(__file__).with_name("renders")


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"\n       {detail}" if detail and not cond else ""))
    return cond


def C(id, type, nodes, value=None, **kw):
    return Component(id=id, type=type, nodes=nodes, value=value, **kw)


DIVIDER = Netlist(question="A 10 V source feeds R1 = 1k and R2 = 2k in series. Find the voltage across R2.",
                  analysis="dc", query=["V(out)"],
                  components=[C("V1", "V", ["in", "0"], "10"), C("R1", "R", ["in", "out"], "1k"),
                              C("R2", "R", ["out", "0"], "2k")])
WRONG = DIVIDER.model_copy(update={"components": [C("V1", "V", ["in", "0"], "10"),
                                                  C("R1", "R", ["in", "out"], "2k"),
                                                  C("R2", "R", ["out", "0"], "1k")]})
CAPS_ONLY = Netlist(analysis="dc", query=["V(x)"], question="(unsolvable on purpose)",
                    components=[C("V1", "V", ["a", "0"], "10"), C("R1", "R", ["a", "0"], "1k"),
                                C("C1", "C", ["a", "x"], "1u"), C("C2", "C", ["x", "0"], "1u")])


class FakeParser:
    """Returns the given netlists in order and records the hints it was called with."""

    def __init__(self, *netlists):
        self.netlists = list(netlists)
        self.hints = []

    def __call__(self, question, hint):
        self.hints.append(hint)
        nl = self.netlists.pop(0) if len(self.netlists) > 1 else self.netlists[0]
        return ParseResult(True, nl, ValidationReport(), [], 1, "fake")


def renderer_checks() -> bool:
    ok = True
    RENDERS.mkdir(exist_ok=True)
    text = describe(DIVIDER)
    print(text)
    ok &= check("describe(): one line per component + the 'Find' line", len(text.splitlines()) == 4
                and "R1: 1 kΩ resistor, between in and out" in text, text)
    drawn = 0
    for name, nl, _ in CASES:
        svg = render_svg(nl)
        if svg:
            drawn += 1
            fname = RENDERS / (name.split(":")[0].replace(" ", "_").replace("/", "_").lower() + ".svg")
            fname.write_text(svg, encoding="utf-8")
    ok &= check(f"render_svg(): drew {drawn}/{len(CASES)} Step 3 circuits", drawn == len(CASES),
                "is schemdraw installed? pip install schemdraw")
    print(f"     open the files in {RENDERS} in a browser to look at them")
    return ok


def benchmark_file_checks() -> bool:
    ok = True
    items = load(Path("benchmark/dc_ac.jsonl"))
    ids = [it["id"] for it in items]
    ok &= check(f"benchmark has {len(items)} problems (>= 40)", len(items) >= 40)
    ok &= check("benchmark ids are unique", len(ids) == len(set(ids)))
    ok &= check("every problem has a question and at least one expectation",
                all(it.get("question") and it.get("expect") and all("kind" in e for e in it["expect"])
                    for it in items))

    numeric = {"V(out)": 8.0001, "V(out).phase_deg": 3.0, "I(V1)": -0.001, "poles[1].re": -100.0,
               "H(out,in).dc_gain": 0.5, "fc": 159.2, "fc_2": 1000.0}
    ok &= check("matcher: V answer within 1 %", check_expectation({"kind": "V", "value": 8}, numeric, {}, 0.01)[0])
    ok &= check("matcher: ignores phase keys", [k for k, _ in candidates(numeric, "V")] == ["V(out)"])
    ok &= check("matcher: abs for current direction",
                check_expectation({"kind": "I", "value": 0.001, "abs": True}, numeric, {}, 0.01)[0])
    ok &= check("matcher: wrong value fails", not check_expectation({"kind": "V", "value": 9}, numeric, {}, 0.01)[0])
    ok &= check("matcher: pole / dc_gain / second cutoff",
                check_expectation({"kind": "pole", "value": -100}, numeric, {}, 0.01)[0]
                and check_expectation({"kind": "dc_gain", "value": 0.5}, numeric, {}, 0.01)[0]
                and check_expectation({"kind": "fc", "value": 1000}, numeric, {}, 0.01)[0])
    ok &= check("matcher: symbolic answers compared by value",
                check_expectation({"kind": "V", "symbolic": "R2*Vs/(R1 + R2)"}, {},
                                  {"V(out)": "Vs*R2/(R2 + R1)"}, 0.01)[0])
    return ok


def graph_checks() -> bool:
    from graph import CircuitAI

    ok = True
    # 1. wrong circuit -> user corrects -> right circuit -> answer
    fake = FakeParser(WRONG, DIVIDER)
    ai = CircuitAI(use_llm=False, parse_fn=fake)
    r = ai.ask(DIVIDER.question)
    ok &= check("graph pauses for confirmation", r.status == "needs_confirmation", r.status)
    if r.confirmation:
        print("     confirmation text:\n" + "\n".join("       " + l for l in r.confirmation["netlist_text"].splitlines()))
        ok &= check("confirmation includes the drawing", bool(r.confirmation.get("svg")))
    r = ai.confirm(r.thread_id, ok=False, correction="R1 is 1k and R2 is 2k")
    ok &= check("correction goes back to the parser as a hint",
                r.status == "needs_confirmation" and fake.hints[-1] == "R1 is 1k and R2 is 2k", str(fake.hints))
    r = ai.confirm(r.thread_id, ok=True)
    ok &= check("after confirming: answered with V(out) = 6.667 V",
                r.status == "answered" and "6.667 V" in r.explanation, r.status + " " + r.explanation[:200])
    ver = (r.state.get("result") or {}).get("verified")
    print(f"     ngspice verified: {ver}")
    full = ai.show_complete(r.thread_id)
    ok &= check("show_complete gives the full trace (matrix step) without re-solving",
                "Matrix form" in full and len(full) > len(r.explanation), full[:200])

    # 2. solver fails -> automatic re-parse with the solver's error as the hint
    fake = FakeParser(CAPS_ONLY, DIVIDER)
    ai = CircuitAI(use_llm=False, parse_fn=fake, auto_confirm=True)
    r = ai.ask("(test)")
    ok &= check("unsolvable circuit -> re-parsed with the solver's error -> answered",
                r.status == "answered" and len(fake.hints) == 2 and "solver could not solve" in (fake.hints[1] or ""),
                f"{r.status} hints={fake.hints}")

    # 3. op-amp -> unsupported (no crash)
    opamp = Netlist(analysis="dc", query=["V(out)"], question="inverting op-amp amplifier",
                    components=[C("V1", "V", ["in", "0"], "1"), C("R1", "R", ["in", "n1"], "1k"),
                                C("R2", "R", ["n1", "out"], "10k"),
                                Component(id="U1", type="OPAMP", nodes=["out", "0", "n1"])])
    ai = CircuitAI(use_llm=False, parse_fn=FakeParser(opamp), auto_confirm=True)
    r = ai.ask("op-amp")
    ok &= check("op-amp -> status unsupported with a reason", r.status == "unsupported" and "Phase 2" in r.explanation,
                r.explanation)

    # 4. conceptual -> theory route
    concept = Netlist(analysis="conceptual", question="Why does feedback stabilize gain?")
    ai = CircuitAI(use_llm=False, parse_fn=FakeParser(concept))
    r = ai.ask(concept.question)
    ok &= check("conceptual -> theory route, no confirmation step", r.status == "answered"
                and (r.state.get("classification") or {}).get("topic") == "conceptual", r.status)
    return ok


def live_check() -> bool:
    from graph import CircuitAI

    ai = CircuitAI()
    q = ("A series RLC circuit driven by a 1 V AC source has R = 10 Ohm, L = 10 mH and C = 1 uF, "
         "with the output taken across R. Find the resonant frequency, quality factor and bandwidth.")
    r = ai.ask(q)
    rounds = 0
    while r.status == "needs_confirmation" and rounds < 3:
        rounds += 1
        RENDERS.mkdir(exist_ok=True)
        if r.confirmation.get("svg"):
            (RENDERS / "live.svg").write_text(r.confirmation["svg"], encoding="utf-8")
            print(f"\n(drawing saved to {RENDERS / 'live.svg'})")
        print("\nIs this your circuit?\n" + r.confirmation["netlist_text"])
        for w in r.confirmation.get("warnings", []):
            print("  warning:", w)
        ans = input("\n[y] yes / or type a correction: ").strip()
        r = ai.confirm(r.thread_id, ok=ans.lower() in ("y", "yes", ""), correction=ans)
    print(f"\nstatus: {r.status}\n\n{r.explanation}")
    if r.status == "answered" and input("\nShow the complete solution? [y/n] ").strip().lower().startswith("y"):
        print("\n" + ai.show_complete(r.thread_id))
    print("\n" + ai.client.usage_report())
    return check("live question answered", r.status == "answered")


if __name__ == "__main__":
    all_ok = renderer_checks()
    print()
    all_ok &= benchmark_file_checks()
    print()
    all_ok &= graph_checks()
    if "--live" in sys.argv:
        all_ok &= live_check()
    print("\nSTEP 6:", "ALL PASSED" if all_ok else "SOME CHECKS FAILED")
