"""
Runs the benchmark: every question goes through the full pipeline (parse -> classify -> solve ->
ngspice verify), with the confirmation step auto-accepted, and the numbers are compared with
hand-worked answers.

    python benchmark/run_benchmark.py                 # all 40 (about ₹0.3 with V4 Flash)
    python benchmark/run_benchmark.py --limit 5       # first 5
    python benchmark/run_benchmark.py --ids dc,flt    # only ids starting with dc or flt
    python benchmark/run_benchmark.py --explain       # also generate explanations (costs more)

Matching is by KIND, not by exact query name, because the parser may call a node "out" or "n2":
an expectation {"kind": "V", "value": 8} passes if ANY V(...) answer is within --tol (default 1 %)
of 8. "abs": true compares magnitudes (for currents/powers whose sign depends on direction).
Results are saved to benchmark/results/run_<time>.json.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sympy as sp  # noqa: E402

PREFIXES = {
    "V": ("V(",), "Vth": ("Vth(", "V("), "I": ("I(",), "Isc": ("Isc(", "In("), "P": ("P(",),
    "Rth": ("Rth(", "Zth(", "Z(", "Rn(", "Rin", "Rout"), "Z": ("Z(", "Zth(", "Rth(", "Rin"),
    "fc": ("fc",), "f0": ("f0",), "Qfactor": ("Qfactor",), "bandwidth": ("bandwidth", "fc"), "tau": ("tau",),
    "gain": ("gain", "H("),
}


def load(path: Path) -> List[Dict[str, Any]]:
    items = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path.name} line {i}: {e}")
    return items


def candidates(numeric: Dict[str, Optional[float]], kind: str) -> List[Tuple[str, float]]:
    out = []
    for k, v in numeric.items():
        if v is None:
            continue
        if kind == "dc_gain":
            ok = k.endswith(".dc_gain")
        elif kind == "pole":
            ok = re.fullmatch(r"poles\[\d+\]\.re", k) is not None
        else:
            ok = (k.startswith(PREFIXES.get(kind, (kind,))) and ".phase_deg" not in k
                  and ".dc_gain" not in k and "[" not in k)
        if ok:
            out.append((k, float(v)))
    return out


def same_expression(a: str, b: str) -> bool:
    """True if two symbolic expressions agree at 3 random points (symbols matched by name)."""
    try:
        ea, eb = sp.sympify(a), sp.sympify(b)
    except (sp.SympifyError, TypeError):
        return False
    names = sorted({s.name for s in ea.free_symbols | eb.free_symbols})
    for _ in range(3):
        vals = {n: random.uniform(1, 10) for n in names}
        va = complex(ea.subs({s: vals[s.name] for s in ea.free_symbols}).evalf())
        vb = complex(eb.subs({s: vals[s.name] for s in eb.free_symbols}).evalf())
        if abs(va - vb) > 1e-9 * max(1.0, abs(vb)):
            return False
    return True


def check_expectation(exp: Dict[str, Any], numeric: Dict, symbolic: Dict, tol: float) -> Tuple[bool, str]:
    kind = exp["kind"]
    if "symbolic" in exp:
        keys = [k for k in symbolic if k.startswith(PREFIXES.get(kind, (kind,)))]
        for k in keys:
            if same_expression(symbolic[k], exp["symbolic"]):
                return True, f"{k} = {symbolic[k]}"
        return False, f"no {kind} answer equal to {exp['symbolic']} (got {[symbolic[k] for k in keys]})"
    want = float(exp["value"])
    use_abs = bool(exp.get("abs"))
    cands = candidates(numeric, kind)
    for k, v in cands:
        got = abs(v) if use_abs else v
        target = abs(want) if use_abs else want
        if abs(got - target) <= tol * max(abs(target), 1e-12):
            return True, f"{k} = {v:.6g}"
    shown = ", ".join(f"{k}={v:.4g}" for k, v in cands) or "no answer of this kind"
    return False, f"want {kind} = {want:g}; got {shown}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(Path(__file__).with_name("dc_ac.jsonl")))
    ap.add_argument("--ids", default="", help="comma-separated id prefixes, e.g. dc,flt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run every problem N times; the parser varies between runs, so N=3 gives a steadier score")
    args = ap.parse_args()

    from graph import CircuitAI

    items = load(Path(args.file))
    if args.ids:
        prefixes = tuple(p.strip() for p in args.ids.split(",") if p.strip())
        items = [it for it in items if it["id"].startswith(prefixes)]
    if args.limit:
        items = items[: args.limit]
    if args.repeat > 1:
        items = [dict(it, run=r + 1) for it in items for r in range(args.repeat)]

    ai = CircuitAI(auto_confirm=True, do_explain=args.explain)
    rows, by_topic = [], defaultdict(lambda: [0, 0, 0])            # topic -> [passed, total, verified]
    print(f"{'id':<8} {'topic':<10} {'result':<6} {'verified':<9} {'time':>6}  {'parse':<18} detail")
    for it in items:
        t0 = time.time()
        try:
            reply = ai.ask(it["question"])
            st = reply.state
        except Exception as e:                                       # keep going on unexpected errors
            st = {"status": "crashed", "error": f"{type(e).__name__}: {e}"}
        dt = time.time() - t0
        res = st.get("result") or {}
        numeric, symbolic = res.get("numeric") or {}, res.get("symbolic") or {}
        verified = res.get("verified")

        if st.get("status") != "answered":
            passed, details = False, [f"status={st.get('status')}: {st.get('error', '')[:120]}"]
        else:
            checks = [check_expectation(e, numeric, symbolic, args.tol) for e in it["expect"]]
            passed, details = all(ok for ok, _ in checks), [d for _, d in checks]

        t = by_topic[it["topic"]]
        t[0] += passed
        t[1] += 1
        t[2] += verified is True
        mark = "PASS" if passed else "FAIL"
        attempts, model = st.get("parse_attempts"), (st.get("parse_model") or "").split("/")[-1]
        parse = f"{attempts}x {model}" if attempts else "-"          # e.g. "1x deepseek-v4-flash"
        label = it["id"] + (f"#{it['run']}" if it.get("run") else "")       # dc01#2 = second run of dc01
        print(f"{label:<8} {it['topic']:<10} {mark:<6} {str(verified):<9} {dt:>5.1f}s  {parse:<18} "
              f"{' | '.join(details)}")
        for h in st.get("parse_history") or []:                     # why earlier attempts were rejected
            first = (h.get("errors") or ["?"])[0]
            print(f"{'':>9}attempt {h.get('attempt')} ({(h.get('model') or '').split('/')[-1]}) rejected at "
                  f"{h.get('stage')}: {first[:150]}" + (f"  (+{len(h['errors']) - 1} more)" if len(h.get('errors') or []) > 1 else ""))
        if not passed and st.get("netlist"):
            for c in st["netlist"].get("components", []):
                print(f"{'':>9}{c['id']} {c['type']} {c['nodes']} {c.get('value') or ''} "
                      f"{('AC ' + c['ac']) if c.get('ac') else ''}")
            print(f"{'':>9}query={st['netlist'].get('query')}  hand answer: {it.get('work', '')}")
        rows.append({"id": it["id"], "run": it.get("run", 1), "topic": it["topic"], "passed": passed,
                     "verified": verified,
                     "status": st.get("status"), "details": details, "seconds": round(dt, 2),
                     "parse_attempts": attempts, "parse_model": st.get("parse_model"),
                     "parse_history": st.get("parse_history"),
                     "parse_raw": None if passed else st.get("parse_raw"),   # what the model sent, failures only
                     "netlist": st.get("netlist"), "answers": res.get("answers")})

    total_pass = sum(v[0] for v in by_topic.values())
    total = sum(v[1] for v in by_topic.values())
    print("\nBy topic:")
    for topic, (p, n, v) in sorted(by_topic.items()):
        print(f"  {topic:<11} {p}/{n} correct   {v}/{n} verified by ngspice")
    print(f"\nOVERALL: {total_pass}/{total} correct ({100 * total_pass / max(total, 1):.0f} %)")
    if ai.client:
        print("\n" + ai.client.usage_report())

    out_dir = Path(__file__).with_name("results")
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"tol": args.tol, "passed": total_pass, "total": total, "rows": rows},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()