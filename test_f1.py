"""
Step 7, part 1 checks: the HTTP API.

    pip install fastapi uvicorn python-multipart httpx
    python test_f1.py            # free: a stubbed pipeline, no API calls, no images needed
    python test_f1.py --live     # one real question and one real image through the running pipeline

The stub replaces CircuitAI with a fake that returns whatever Reply the test wants, so the API's own
contract can be checked exactly: the three-call session, the shape of each reply, the uploaded file
reaching the pipeline, and -- the part that matters most -- the error contract.

THE ERROR CONTRACT IS THE POINT
A provider failure must never reach the user as text. The 402 that stopped the benchmark earlier said
"Balance Rs 2.98. Top up from Rs 10 at https://velona.in/wallet" -- a message that tells whoever is using
the app what the wallet holds and where to pay. The checks below assert that none of that survives: the
response carries a short code and one neutral sentence, and the real reason goes to the server log.
"""
import io
import sys
from pathlib import Path

from fastapi.testclient import TestClient

import server
from graph import Reply

PNG_1PX = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000a"
                        "49444154789c630001000005000100"
                        "0d0a2db40000000049454e44ae426082")


def check(name, cond, detail=""):
    cond = bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"\n       {detail}" if detail and not cond else ""))
    return cond


class StubAI:
    """Stands in for CircuitAI: records what it was called with, returns what the test set up."""

    def __init__(self):
        self.calls = []
        self.next = None            # a Reply, or an Exception to raise
        self.full = "the complete solution"

    def _go(self, kind, **kw):
        self.calls.append((kind, kw))
        if isinstance(self.next, Exception):
            raise self.next
        return self.next

    def ask(self, question="", level="key", thread_id=None, image=None):
        return self._go("ask", question=question, level=level, image=image)

    def confirm(self, thread_id, ok=True, correction=""):
        return self._go("confirm", thread_id=thread_id, ok=ok, correction=correction)

    def show_complete(self, thread_id):
        self.calls.append(("full", {"thread_id": thread_id}))
        if isinstance(self.next, Exception):
            raise self.next
        return self.full


NEEDS = Reply("t1", "needs_confirmation",
              confirmation={"question": "Find the voltage across R2.",
                            "netlist_text": "a 12 V source, R1 = 2 kOhm, R2 = 4 kOhm",
                            "spice": "V1 in 0 12\nR1 in out 2k\nR2 out 0 4k",
                            "svg": "<svg/>", "warnings": ["R2's value is blurred"], "topic": "dc"},
              state={"question": "Find the voltage across R2."})

ANSWERED = Reply("t1", "answered", explanation="V(out) = 8 V because ...",
                 state={"question": "Find the voltage across R2.",
                        "netlist": {"analysis": "dc", "query": ["V(out)"], "question": "q",
                                    "components": [{"id": "V1", "type": "V", "nodes": ["in", "0"],
                                                    "value": "12"},
                                                   {"id": "R1", "type": "R", "nodes": ["in", "out"],
                                                    "value": "2k"},
                                                   {"id": "R2", "type": "R", "nodes": ["out", "0"],
                                                    "value": "4k"}]},
                        "result": {"answers": {"V(out)": "8 V"}, "numeric": {"V(out)": 8.0},
                                   "verified": True, "verification_notes": [], "method": "MNA",
                                   "trace": {"steps": []}},
                        "verify_table": "  query   solver  ngspice  ok",
                        "parse_warnings": []})

FAILED = Reply("t1", "parse_failed", state={"error": "Could not turn the question into a circuit: ..."})


def api_checks() -> bool:
    ok = True
    stub = StubAI()
    server.AI = stub
    c = TestClient(server.app)

    r = c.get("/health")
    ok &= check("GET /health", r.status_code == 200 and r.json()["ok"])

    # --- the three-call session
    stub.next = NEEDS
    r = c.post("/ask", data={"question": "Find the voltage across R2."})
    j = r.json()
    ok &= check("POST /ask returns needs_confirmation, not an answer",
                r.status_code == 200 and j["status"] == "needs_confirmation", str(j)[:200])
    ok &= check("the confirmation carries the circuit three ways (words, netlist, drawing)",
                all(j["confirmation"].get(k) for k in ("netlist_text", "spice", "svg")), str(j)[:200])
    ok &= check("and the doubts about what was read",
                j["confirmation"]["warnings"] == ["R2's value is blurred"])
    ok &= check("and a thread_id to continue with", j["thread_id"] == "t1")

    stub.next = ANSWERED
    r = c.post("/confirm", json={"thread_id": "t1", "ok": True})
    j = r.json()
    ok &= check("POST /confirm returns the answer", j["status"] == "answered"
                and j["answer"]["answers"] == {"V(out)": "8 V"}, str(j)[:200])
    ok &= check("with the key steps, the numbers and the verification",
                j["answer"]["explanation"] and j["answer"]["numeric"]["V(out)"] == 8.0
                and j["answer"]["verified"] is True and j["answer"]["verification"])
    ok &= check("and the circuit it solved", j["circuit"]["spice"].startswith("V1"))
    ok &= check("confirm passed the thread through",
                stub.calls[-1][1]["thread_id"] == "t1" and stub.calls[-1][1]["ok"] is True)

    r = c.get("/full/t1")
    ok &= check("GET /full returns the complete solution", r.json()["explanation"] == stub.full)

    # --- a correction re-parses instead of answering
    stub.next = NEEDS
    r = c.post("/confirm", json={"thread_id": "t1", "ok": False, "correction": "R2 is 4.7k"})
    ok &= check("a correction is passed to the pipeline",
                stub.calls[-1][1]["correction"] == "R2 is 4.7k" and stub.calls[-1][1]["ok"] is False)

    # --- an image reaches the pipeline as a real file on disk
    stub.next = NEEDS
    seen = {}
    r = c.post("/ask", data={"question": ""},
               files={"image": ("circuit.png", io.BytesIO(PNG_1PX), "image/png")})
    path = stub.calls[-1][1]["image"]
    seen["path"] = path
    ok &= check("an uploaded image is saved and its path given to the pipeline",
                path and Path(path).exists() and Path(path).read_bytes() == PNG_1PX, str(path))
    ok &= check("an image with no typed question is allowed", r.json()["status"] == "needs_confirmation")

    # --- nothing at all
    r = c.post("/ask", data={"question": ""})
    ok &= check("neither question nor image -> 400", r.status_code == 400, str(r.json())[:160])

    # --- a file that is not an image: the user can fix this, so they see the real reason
    r = c.post("/ask", data={"question": ""},
               files={"image": ("page.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")})
    j = r.json()
    ok &= check("a PDF upload -> 400 with a reason the user can act on",
                r.status_code == 400 and j["error"]["code"] == "IMAGE-BAD"
                and j["error"]["ours"] is False and "PNG" in j["error"]["message"], str(j)[:200])
    return ok


def error_contract_checks() -> bool:
    """The provider's own words must never reach the browser."""
    ok = True
    print()
    from llm_client import LLMError
    from solvers import NotSupportedError, SolverError

    stub = StubAI()
    server.AI = stub
    c = TestClient(server.app)

    leaky = "Error code: 402 - Balance ₹2.98. Top up from ₹10 at https://velona.in/wallet"
    stub.next = LLMError(leaky)
    r = c.post("/ask", data={"question": "anything"})
    j = r.json()
    body = r.text
    ok &= check("a 402 -> 503 with code LLM-402", r.status_code == 503 and j["error"]["code"] == "LLM-402",
                str(j)[:200])
    ok &= check("marked as ours, so the UI shows the contact popup", j["error"]["ours"] is True)
    ok &= check("the balance is NOT in the response", "2.98" not in body and "Balance" not in body, body[:200])
    ok &= check("the top-up URL is NOT in the response", "velona.in/wallet" not in body, body[:200])
    ok &= check("the user is told who to contact", "Amrith" in j["error"]["message"], j["error"]["message"])

    for text, code in [("Error code: 401 - Unauthorized", "LLM-401"),
                       ("Error code: 429 - rate limit exceeded", "LLM-429"),
                       ("Connection timed out after 60s", "LLM-503"),
                       ("something strange happened", "LLM-ERR")]:
        stub.next = LLMError(text)
        j = c.post("/ask", data={"question": "x"}).json()
        ok &= check(f"{text[:28]!r} -> {code}", j["error"]["code"] == code and j["error"]["ours"] is True,
                    str(j)[:160])
        ok &= check(f"   and {code} says nothing about the cause",
                    j["error"]["message"] == server.CONTACT, j["error"]["message"])

    stub.next = NotSupportedError("Oscillators with diodes in the loop are not supported yet")
    j = c.post("/ask", data={"question": "x"}).json()
    ok &= check("an unsupported circuit -> the real reason, not the popup",
                j["error"]["code"] == "UNSUPPORTED" and j["error"]["ours"] is False
                and "not supported yet" in j["error"]["message"], str(j)[:200])

    stub.next = SolverError("the output latches instead of oscillating")
    j = c.post("/ask", data={"question": "x"}).json()
    ok &= check("a solver refusal -> the real reason", j["error"]["code"] == "SOLVE-FAILED"
                and j["error"]["ours"] is False and "latches" in j["error"]["message"], str(j)[:200])

    stub.next = RuntimeError("a bug in our code with a stack trace")
    r = c.post("/ask", data={"question": "x"})
    j = r.json()
    ok &= check("an unexpected crash -> ours, with nothing leaked",
                r.status_code == 503 and j["error"]["ours"] is True
                and "stack trace" not in r.text, str(j)[:200])

    stub.next = FAILED
    j = c.post("/ask", data={"question": "what is the meaning of life"}).json()
    ok &= check("a parse failure -> the reason, marked as not ours",
                j["status"] == "parse_failed" and j["error"]["code"] == "PARSE-FAILED"
                and j["error"]["ours"] is False, str(j)[:200])
    return ok


def live_checks() -> bool:
    """One real question and one real image, through the actual pipeline."""
    print()
    import importlib
    importlib.reload(server)                      # a fresh CircuitAI, not the stub
    c = TestClient(server.app)
    ok = True

    r = c.post("/ask", data={"question": "A 12 V source feeds R1 = 2 kOhm in series with R2 = 4 kOhm. "
                                         "Find the voltage across R2."})
    j = r.json()
    print("ask ->", j.get("status"), str(j.get("confirmation", j.get("error")))[:300])
    ok &= check("a real question reaches the confirmation step", j.get("status") == "needs_confirmation")
    if j.get("status") == "needs_confirmation":
        j2 = c.post("/confirm", json={"thread_id": j["thread_id"], "ok": True}).json()
        print("confirm ->", j2.get("status"), str(j2.get("answer", {}).get("answers"))[:200])
        ok &= check("and is answered correctly",
                    abs((j2.get("answer", {}).get("numeric") or {}).get("V(out)", 0) - 8.0) < 1e-6
                    or abs((j2.get("answer", {}).get("numeric") or {}).get("V(a)", 0) - 8.0) < 1e-6,
                    str(j2.get("answer", {}).get("numeric"))[:200])
        full = c.get(f"/full/{j['thread_id']}").json()
        ok &= check("and the full solution is available", bool(full.get("explanation")))

    img = Path("benchmark/images/divider.png")
    if img.exists():
        with img.open("rb") as f:
            j = c.post("/ask", data={"question": ""},
                       files={"image": (img.name, f, "image/png")}).json()
        print("image ask ->", j.get("status"), str(j.get("confirmation", j.get("error")))[:300])
        ok &= check("an uploaded image reaches the confirmation step",
                    j.get("status") == "needs_confirmation")
        if j.get("status") == "needs_confirmation":
            j2 = c.post("/confirm", json={"thread_id": j["thread_id"], "ok": True}).json()
            print("image confirm ->", str(j2.get("answer", {}).get("answers"))[:200])
            ok &= check("and is solved from the picture alone",
                        any(abs(v - 8.0) < 1e-6 for v in (j2.get("answer", {}).get("numeric") or {}).values()
                            if isinstance(v, (int, float))),
                        str(j2.get("answer", {}).get("numeric"))[:200])
    else:
        print(f"({img} not there: skipping the image half -- run benchmark/make_image_cases.py)")
    return ok


if __name__ == "__main__":
    all_ok = api_checks()
    all_ok &= error_contract_checks()
    if "--live" in sys.argv:
        all_ok &= live_checks()
    print("\nSTEP 7 PART 1:", "ALL PASSED" if all_ok else "SOME CHECKS FAILED")