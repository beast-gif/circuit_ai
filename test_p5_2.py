"""
Phase 5, Step 2: the image benchmark -- how well the circuit is read off a picture.

    python benchmark/make_image_cases.py --photos     # draw the cases first (once)
    python test_p5_2.py                               # run every image, score by category
    python test_p5_2.py --only clean                  # just the crisp drawings
    python test_p5_2.py --only photo                  # just the degraded "phone photo" versions
    python test_p5_2.py --ids inv_amp,pnp              # a few by name
    python test_p5_2.py --json out.json                # write the full record for later comparison

WHY THE SCORE IS SPLIT BY CATEGORY
Reading a crisp screenshot and reading a photograph of a page are different problems, and averaging them
gives a number that describes neither. A drop from "clean" to "photo" is the cost of the camera, and it is
the number worth watching as the prompt changes.

WHAT COUNTS AS A PASS
The same thing as every other benchmark: the circuit is solved and the numbers come out right. That is
deliberately strict -- the netlist is not compared component by component, because there are many correct
netlists for one drawing (node names differ, a source may be split, parts may be ordered differently) and
only one correct set of answers. A wrong answer from a plausible-looking netlist is exactly the failure
this is meant to catch.

Each run also reports WHY failures happened, because with images the reason matters more than the count:
    parse_failed   the model never produced a netlist the checks would accept
    wrong answer   it produced a circuit confidently, and that circuit was not the one in the picture
    crashed        a bug in the pipeline, not in the reading
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from benchmark.run_benchmark import check_expectation
from schemas import Netlist

IMAGES = Path("benchmark/images")
SUFFIX = {".png", ".jpg", ".jpeg", ".webp"}


def arg(name: str, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def load_cases():
    only = arg("--only")
    ids = [x.strip() for x in (arg("--ids") or "").split(",") if x.strip()]
    out = []
    for img in sorted(IMAGES.iterdir() if IMAGES.exists() else []):
        if img.suffix.lower() not in SUFFIX:
            continue
        meta_path = Path(str(img) + ".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta.setdefault("category", "clean")
        meta.setdefault("question", "")
        meta.setdefault("expect", [])
        if only and meta["category"] != only:
            continue
        if ids and not any(i in img.name for i in ids):
            continue
        out.append((img, meta))
    return out


def main() -> bool:
    from graph import CircuitAI

    cases = load_cases()
    if not cases:
        print(f"No image cases in {IMAGES}.\nDraw them first:  python benchmark/make_image_cases.py --photos")
        return False

    ai = CircuitAI(auto_confirm=True, do_explain=False)
    rows, score = [], defaultdict(lambda: [0, 0])
    for img, meta in cases:
        t0 = time.time()
        print(f"\n=== {img.name}   [{meta['category']}]")
        try:
            r = ai.ask(question=meta["question"], image=str(img))
            st = r.state
        except Exception as e:                                  # noqa: BLE001 - a crash is a result too
            msg = str(e)
            if any(w in msg.lower() for w in ("402", "401", "balance", "insufficient", "quota",
                                              "top up", "unauthorized")):
                # an account problem is not a result: stop now instead of marking every case "crashed"
                print(f"   STOPPING: the API refused the request -- {msg[:200]}")
                print("   Nothing is wrong with the images or the code; top up and run it again.")
                break
            print(f"   CRASHED: {type(e).__name__}: {e}")
            rows.append({"id": img.name, "category": meta["category"], "passed": False,
                         "status": "crashed", "detail": f"{type(e).__name__}: {e}"})
            score[meta["category"]][1] += 1
            continue

        res = st.get("result") or {}
        if st.get("netlist"):
            print(Netlist.model_validate(st["netlist"]).to_spice())
            print("   query:", st["netlist"].get("query"))
            if not meta["question"]:
                print("   read the question as:", (st.get("question") or "")[:160])
        for w in st.get("parse_warnings") or []:
            print("   warning:", w)
        for h in st.get("parse_history") or []:
            print(f"   rejected attempt {h['attempt']} at {h['stage']}: {h['errors'][0][:140]}")
        print("   answers:", res.get("answers"))

        passed = r.status == "answered"
        details = []
        for e in meta["expect"]:
            good, detail = check_expectation(e, res.get("numeric") or {}, res.get("symbolic") or {}, 0.01,
                                             res.get("verification_notes"))
            passed &= good
            details.append(detail)
            print("  ", detail)
        if not meta["expect"]:
            print("   (no expectations: not scored)")
        why = ("ok" if passed else
               "parse_failed" if r.status != "answered" else "wrong answer")
        print(f"   -> {'PASS' if passed else 'FAIL'} ({why}, {time.time() - t0:.1f}s)")

        rows.append({"id": img.name, "category": meta["category"], "passed": bool(passed),
                     "status": r.status, "why": why, "seconds": round(time.time() - t0, 1),
                     "netlist": st.get("netlist"), "answers": res.get("answers"),
                     "question_read": st.get("question"), "warnings": st.get("parse_warnings"),
                     "parse_history": st.get("parse_history"), "details": details,
                     "error": st.get("error")})
        score[meta["category"]][0] += int(bool(passed))
        score[meta["category"]][1] += 1

    print("\n" + "-" * 46)
    print(f"  {'category':<12} {'score':>9}   {'rate':>6}")
    print("-" * 46)
    for cat, (good, total) in sorted(score.items()):
        print(f"  {cat:<12} {good:>4}/{total:<4} {100 * good / max(total, 1):>7.1f} %")
    allg = sum(g for g, _ in score.values())
    allt = sum(t for _, t in score.values())
    print("-" * 46)
    print(f"  {'all':<12} {allg:>4}/{allt:<4} {100 * allg / max(allt, 1):>7.1f} %")

    bad = [r for r in rows if not r["passed"]]
    if bad:
        print("\nfailures:")
        for r in bad:
            print(f"  {r['id']:<26} {r.get('why', r['status'])}"
                  + (f": {r['error'][:90]}" if r.get("error") else ""))
    out = arg("--json")
    if out:
        Path(out).write_text(json.dumps({"rows": rows, "score": dict(score)}, indent=1, default=str))
        print(f"\nwrote {out}")
    print("\n" + ai.client.usage_report())
    return allg == allt


if __name__ == "__main__":
    print("\nPHASE 5 STEP 2:", "ALL PASSED" if main() else "SOME CASES FAILED")