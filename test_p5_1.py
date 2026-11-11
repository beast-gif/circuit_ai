"""
Phase 5, Step 1 checks: reading a circuit out of an IMAGE.

    python test_p5_1.py            # file checks, prompt checks, and the whole retry loop against a
                                   # STUB vision model -- free, no API calls, no images needed
    python test_p5_1.py --live     # the real Sonnet 5 on every image in benchmark/images/

HOW THE OFFLINE CHECKS WORK, and why they are worth more than they look
A vision model cannot be unit-tested for eyesight. What CAN be tested is everything around it: that a
mis-traced netlist is rejected and sent back with the reason, that a value it could not read becomes a
symbol and a warning instead of an invented number, that the question printed on the image is what the
consistency checks run against, that a typed question overrides the printed one, and that a bad file is
refused before any money is spent. All of that is exercised here with a stub client that returns canned
JSON -- including a stub that gets the circuit WRONG on its first reply and right on its second, which is
the behaviour the retry loop exists for.

HOW TO ADD A REAL IMAGE CASE (for --live)
Put the picture in benchmark/images/ and a file of the same name with .json added, holding the question and
what the answer must be:

    benchmark/images/divider.png
    benchmark/images/divider.png.json   {"question": "Find the voltage across R2.",
                                         "expect": [{"kind": "V", "value": 8.0}]}

"question" may be "" when the question is printed in the picture. Each expectation is the same {kind, value,
tol} the other benchmarks use. A photo of a textbook page, a phone picture of your own drawing and a
screenshot all work -- those three are the cases that matter, and they fail in different ways.
"""
import json
import sys
from pathlib import Path

from llm_client import LLMError, load_conventions
from parsing.image_parser import MAX_IMAGE_BYTES, check_image, parse_image
from parsing.prompts import IMAGE_PARSER_SYSTEM
from schemas import Netlist

CONV = load_conventions()
IMAGES = Path("benchmark/images")
TMP = Path("_p5_tmp")
# the smallest valid PNG (1x1, transparent): enough for every check that does not involve the model
PNG_1PX = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000a"
                        "49444154789c6300010000050001"
                        "0d0a2db40000000049454e44ae426082")


def check(name, cond, detail=""):
    cond = bool(cond)                      # a condition may be a list/dict: only its truth matters here
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"\n       {detail}" if detail and not cond else ""))
    return cond


class StubClient:
    """A vision model that returns whatever JSON the test tells it to, one reply per call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.fallback_parser = "stub"

    def model_for(self, role):
        return f"stub/{role}"

    def chat(self, role, messages, json_mode=False, model=None, max_tokens=None):
        self.calls.append(messages)
        if not self.replies:
            raise LLMError("stub ran out of replies")
        return json.dumps(self.replies.pop(0))

    def usage_report(self):
        return "stub: no tokens used"


DIVIDER_OK = {
    "analysis": "dc",
    "components": [{"id": "V1", "type": "V", "nodes": ["in", "0"], "value": "12"},
                   {"id": "R1", "type": "R", "nodes": ["in", "out"], "value": "2k"},
                   {"id": "R2", "type": "R", "nodes": ["out", "0"], "value": "4k"}],
    "query": ["V(out)"], "frequency": None, "assumptions": [], "missing_info": [],
    "question_text": "A 12 V source feeds R1 = 2 kOhm in series with R2 = 4 kOhm. Find the voltage "
                     "across R2.",
    "reading_notes": [],
}

# the same picture read WRONGLY: R1 and R2 both put across the source (a wire traced through a crossing),
# which leaves node "out" as a dead end -- exactly what the consistency checks are for
DIVIDER_BAD = {**DIVIDER_OK,
               "components": [{"id": "V1", "type": "V", "nodes": ["in", "0"], "value": "12"},
                              {"id": "R1", "type": "R", "nodes": ["in", "0"], "value": "2k"},
                              {"id": "R2", "type": "R", "nodes": ["in", "0"], "value": "4k"}],
               "query": ["V(out)"]}


def tmp_image(name="c.png", data=PNG_1PX) -> str:
    TMP.mkdir(exist_ok=True)
    p = TMP / name
    p.write_bytes(data)
    return str(p)


def file_checks() -> bool:
    ok = True
    print()
    for name, path, word in [
        ("a missing file is refused by name", "benchmark/images/does_not_exist.png", "not found"),
        ("a PDF is refused with what to do instead", tmp_image("page.pdf", b"%PDF-1.4"), "PNG"),
        ("an empty file is refused", tmp_image("empty.png", b""), "empty"),
    ]:
        try:
            check_image(path)
            ok &= check(name, False, "no error raised")
        except LLMError as e:
            ok &= check(name, word.lower() in str(e).lower(), str(e))
    big = tmp_image("big.png", PNG_1PX + b"\0" * (MAX_IMAGE_BYTES + 1))
    try:
        check_image(big)
        ok &= check("an oversized image is refused with the limit", False)
    except LLMError as e:
        ok &= check("an oversized image is refused with the limit", "MB" in str(e), str(e))
    ok &= check("a valid PNG passes the file check", Path(check_image(tmp_image())).exists())
    return ok


def prompt_checks() -> bool:
    ok = True
    print()
    for word, why in [
        ("A node is a whole piece of wire", "node tracing is spelled out"),
        ("CROSS with no dot", "crossing wires are not a connection"),
        ("the bar is the CATHODE", "diode orientation"),
        ("pointing OUT of the base = NPN", "BJT type from the arrow"),
        # phrases must not straddle a line wrap in the prompt: keep them short
        ("NOT assume the + is on top", "op-amp inputs are read, not assumed"),
        ("2k2", "value shorthand"),
        ("Do not invent a number", "an unreadable value is never guessed"),
        ("reading_notes", "the doubt channel exists"),
        ("question_text", "the printed question is captured"),
    ]:
        ok &= check(f"prompt covers: {why}", word in IMAGE_PARSER_SYSTEM, f"missing {word!r}")
    ok &= check("the image prompt keeps the whole text-parser contract",
                "ONLY WHAT THE QUESTION DESCRIBES" in IMAGE_PARSER_SYSTEM
                and "Wien bridge" in IMAGE_PARSER_SYSTEM)
    return ok


def stub_checks() -> bool:
    ok = True
    print()
    img = tmp_image()

    # 1. a clean read goes straight through, and the printed question becomes the netlist's question
    cl = StubClient([DIVIDER_OK])
    pr = parse_image(img, client=cl)
    ok &= check("a clean read parses on the first attempt", pr.ok and pr.attempts == 1,
                f"{pr.ok} {pr.attempts} {pr.errors}")
    ok &= check("the question printed in the image is kept",
                pr.ok and "12 V source" in (pr.netlist.question or ""), pr.netlist.question if pr.ok else "")
    ok &= check("the image is attached to the first message only",
                any(isinstance(m.get("content"), list) for m in cl.calls[0]))

    # 2. a mis-traced circuit is rejected, the reason goes back, and the second reply is accepted
    cl = StubClient([DIVIDER_BAD, DIVIDER_OK])
    pr = parse_image(img, client=cl)
    ok &= check("a mis-traced netlist is rejected and the retry is accepted", pr.ok and pr.attempts == 2,
                f"{pr.ok} attempts={pr.attempts} {pr.errors}")
    ok &= check("the rejection reason was sent back to the model",
                len(cl.calls) == 2 and any("look at the image again" in str(m.get("content", "")).lower()
                                           for m in cl.calls[1]),
                str(cl.calls[1][-1])[:200] if len(cl.calls) == 2 else "only one call")
    ok &= check("the failed attempt is recorded with its stage",
                bool(pr.history) and pr.history[0]["stage"] in ("validator", "consistency"),
                str(pr.history))

    # 3. a value it could not read: a symbol plus a note, never an invented number
    unsure = {**DIVIDER_OK,
              "components": [{"id": "V1", "type": "V", "nodes": ["in", "0"], "value": "12"},
                             {"id": "R1", "type": "R", "nodes": ["in", "out"], "value": "2k"},
                             {"id": "R2", "type": "R", "nodes": ["out", "0"], "value": "R2"}],
              "query": ["Rth(out,0)"],
              "reading_notes": ["R2's value is blurred; it could be 4k7 or 47k"]}
    pr = parse_image(img, client=StubClient([unsure]))
    ok &= check("an unreadable value becomes a warning the person will see",
                pr.ok and any("blurred" in w for w in pr.report.warnings),
                str(pr.report.warnings if pr.ok else pr.errors))
    ok &= check("and the component is still there, as a symbol",
                pr.ok and any(c.id == "R2" and c.value == "R2" for c in pr.netlist.components))

    # 4. a typed question overrides the one printed in the image, and both are kept
    pr = parse_image(img, question="Find the power in R2.", client=StubClient([DIVIDER_OK]))
    ok &= check("a typed question is kept as the question",
                pr.ok and pr.netlist.question.startswith("Find the power in R2."), pr.netlist.question)
    ok &= check("and what was printed in the image is kept alongside it",
                pr.ok and "question printed in the image" in pr.netlist.question)

    # 5. a picture with no question at all: say so rather than guessing silently
    bare = {**DIVIDER_OK, "question_text": "", "query": ["V(out)"]}
    pr = parse_image(img, client=StubClient([bare]))
    ok &= check("no question in the image -> recorded in missing_info",
                pr.ok and any("no question" in m for m in pr.missing_info), str(pr.missing_info))

    # 6. the model never returning usable JSON must fail with the reason, not crash
    pr = parse_image(img, client=StubClient([{"nonsense": 1}, {"nonsense": 2}, {"nonsense": 3}]))
    ok &= check("a model that never produces a netlist -> clean failure",
                not pr.ok and pr.attempts == 3 and bool(pr.errors), f"{pr.ok} {pr.attempts} {pr.errors}")

    # 7. reading_notes are carried on the raw result too, for the frontend to show
    pr = parse_image(img, client=StubClient([unsure]))
    ok &= check("reading notes are available on the result for the UI",
                pr.ok and pr.raw.get("_reading_notes"), str(pr.raw.get("_reading_notes")))
    return ok


def graph_checks() -> bool:
    """The image goes through the same graph as text: one path in, same solver, same verification."""
    ok = True
    print()
    from graph import CircuitAI

    img = tmp_image()
    ai = CircuitAI(use_llm=False, auto_confirm=True, do_explain=False, do_verify=False,
                   image_parse_fn=lambda i, q, h: parse_image(i, question=q, client=StubClient([DIVIDER_OK])))
    r = ai.ask(image=img)
    ok &= check("an image question is solved through the normal graph", r.status == "answered",
                f"{r.status}: {r.state.get('error')}")
    ok &= check("and gives the right answer (12 V across 2k + 4k -> 8 V across R2)",
                abs((r.state.get("result") or {}).get("numeric", {}).get("V(out)", 0) - 8.0) < 1e-9,
                str((r.state.get("result") or {}).get("numeric")))
    ok &= check("the question read off the image reaches the state",
                "12 V source" in (r.state.get("question") or ""), r.state.get("question", "")[:80])

    missing = CircuitAI(use_llm=False, auto_confirm=True, do_explain=False, do_verify=False)
    r = missing.ask(image="benchmark/images/nope.png")
    ok &= check("a missing image fails with the file's name, not a stack trace",
                r.status == "parse_failed" and "not found" in (r.state.get("error") or ""),
                f"{r.status}: {r.state.get('error')}")
    try:
        missing.ask()
        ok &= check("ask() with neither question nor image -> clear error", False)
    except ValueError as e:
        ok &= check("ask() with neither question nor image -> clear error", "needs a question" in str(e))
    return ok


def live_checks() -> bool:
    from benchmark.run_benchmark import check_expectation
    from graph import CircuitAI
    from llm_client import VelonaClient

    if not IMAGES.exists():
        print(f"\n{IMAGES} does not exist: create it and drop in images (see the top of this file)")
        return False
    cases = sorted(p for p in IMAGES.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if not cases:
        print(f"\nno images in {IMAGES}: see the top of this file for how to add one")
        return False

    ai = CircuitAI(auto_confirm=True, do_explain=False)
    ok = True
    for img in cases:
        meta_path = Path(str(img) + ".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        print(f"\n=== {img.name}  {meta.get('question', '(question is in the image)')}")
        r = ai.ask(question=meta.get("question", ""), image=str(img))
        st = r.state
        if st.get("netlist"):
            print(Netlist.model_validate(st["netlist"]).to_spice())
            print("query:", st["netlist"].get("query"))
            print("read as:", (st.get("question") or "")[:200])
        for w in st.get("parse_warnings") or []:
            print("   warning:", w)
        for h in st.get("parse_history") or []:
            print(f"   rejected attempt {h['attempt']} at {h['stage']}: {h['errors'][0][:160]}")
        res = st.get("result") or {}
        print("answers:", res.get("answers"))
        passed = r.status == "answered"
        for e in meta.get("expect", []):
            good, detail = check_expectation(e, res.get("numeric") or {}, res.get("symbolic") or {}, 0.01,
                                             res.get("verification_notes"))
            passed &= good
            print("  ", detail)
        if not meta.get("expect"):
            print("   (no expectations in the .json: read the netlist above and check it yourself)")
        ok &= check(f"{img.name}", passed, f"status={r.status}, error={st.get('error')}")
    print("\n" + ai.client.usage_report())
    return ok


if __name__ == "__main__":
    all_ok = file_checks()
    all_ok &= prompt_checks()
    all_ok &= stub_checks()
    all_ok &= graph_checks()
    if "--live" in sys.argv:
        all_ok &= live_checks()
    print("\nPHASE 5 STEP 1:", "ALL PASSED" if all_ok else "SOME CHECKS FAILED")