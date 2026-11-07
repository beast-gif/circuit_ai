"""
The whole pipeline as a LangGraph state graph.

    parse ─► classify ─┬─► theory ───────────────────────────────► END   (conceptual question)
       ▲               ├─► unsupported ──────────────────────────► END   (op-amp/BJT/... not yet)
       │               └─► confirm ──(ok)──► solve ─► verify ─► explain ─► END
       │                      │                │
       └──(user correction)───┘                │
       └──(solver could not solve it)──────────┘   (one automatic re-parse with the solver's error)

confirm uses LangGraph's interrupt(): the graph pauses and returns the drawing + description.
The app calls confirm(thread_id, ok=True) to continue, or ok=False with a correction
("R2 is 2k, not 1k"), which goes back to the parser as a hint. Up to 3 correction rounds.

    ai = CircuitAI()
    r = ai.ask("A 10 V source ... find the voltage across R2.")
    r.status == "needs_confirmation"  -> show r.confirmation["svg"] / ["netlist_text"]
    r = ai.confirm(r.thread_id, ok=True)
    print(r.explanation)                       # key steps
    print(ai.show_complete(r.thread_id))       # every step, same trace, no re-solving

State holds plain dicts (netlist / result as JSON) so the checkpointer can store it.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from classifier import classify
from conceptual import answer_conceptual
from explainer import explain
from llm_client import VelonaClient, load_conventions
from parsing.text_parser import ParseResult, parse_text
from renderer import describe, render_svg
from schemas import Netlist, SolveResult
from solvers import NotSupportedError, SolverError, get_solver
from verifier import Verifier

MAX_CONFIRM_ROUNDS = 3
MAX_SOLVE_RETRIES = 1


class State(TypedDict, total=False):
    question: str
    level: str
    hint: str
    netlist: Dict[str, Any]
    parse_warnings: List[str]
    parse_attempts: int
    parse_model: str
    parse_history: List[Dict[str, Any]]      # why each rejected attempt failed (stage + errors)
    parse_raw: Dict[str, Any]                # the model's last JSON reply
    classification: Dict[str, Any]
    confirmed: bool
    confirm_rounds: int
    solve_retries: int
    result: Dict[str, Any]
    verify_table: str
    explanation: str
    explain_warnings: List[str]
    status: str
    error: str


@dataclass
class Reply:
    thread_id: str
    status: str                     # needs_confirmation | answered | parse_failed | unsupported | ...
    explanation: str = ""
    confirmation: Optional[Dict[str, Any]] = None
    state: Dict[str, Any] = field(default_factory=dict)


class CircuitAI:
    def __init__(self, client: Optional[VelonaClient] = None, use_llm: bool = True,
                 auto_confirm: bool = False, do_explain: bool = True, do_verify: bool = True,
                 parse_fn: Optional[Callable[[str, Optional[str]], ParseResult]] = None):
        self.client = client if client is not None else (VelonaClient() if use_llm else None)
        self.use_llm = use_llm
        self.auto_confirm = auto_confirm
        self.do_explain = do_explain
        self.do_verify = do_verify
        self.conv = load_conventions()
        self.verifier = Verifier() if do_verify else None
        self.parse_fn = parse_fn or (lambda q, hint: parse_text(q, client=self.client, hint=hint))
        self.graph = self._build()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ask(self, question: str, level: str = "key", thread_id: Optional[str] = None) -> Reply:
        tid = thread_id or str(uuid.uuid4())
        self.graph.invoke({"question": question, "level": level}, self._cfg(tid))
        return self._reply(tid)

    def confirm(self, thread_id: str, ok: bool = True, correction: str = "") -> Reply:
        self.graph.invoke(Command(resume={"ok": ok, "correction": correction}), self._cfg(thread_id))
        return self._reply(thread_id)

    def show_complete(self, thread_id: str) -> str:
        """Full solution from the stored trace (or the full theory answer). Nothing is re-solved."""
        s = self.graph.get_state(self._cfg(thread_id)).values
        topic = (s.get("classification") or {}).get("topic", "")
        if topic == "conceptual":
            return answer_conceptual(s["question"], "full", client=self.client).text
        if not s.get("result"):
            return s.get("explanation", "") or "Nothing has been solved in this conversation yet."
        nl, res = Netlist.model_validate(s["netlist"]), SolveResult.model_validate(s["result"])
        return explain(nl, res, "full", topic=topic, client=self.client,
                       use_llm=self.use_llm and self.client is not None).text

    # ------------------------------------------------------------------
    @staticmethod
    def _cfg(tid: str) -> Dict[str, Any]:
        return {"configurable": {"thread_id": tid}}

    def _reply(self, tid: str) -> Reply:
        snap = self.graph.get_state(self._cfg(tid))
        pending = [i for t in snap.tasks for i in (getattr(t, "interrupts", None) or ())]
        if pending:
            return Reply(tid, "needs_confirmation", confirmation=pending[0].value, state=dict(snap.values))
        v = dict(snap.values)
        return Reply(tid, v.get("status", "unknown"), explanation=v.get("explanation", "") or v.get("error", ""),
                     state=v)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    def _parse(self, s: State) -> Dict[str, Any]:
        pr = self.parse_fn(s["question"], s.get("hint"))
        stats = {"parse_attempts": s.get("parse_attempts", 0) + (pr.attempts or 0),   # total over re-parses
                 "parse_model": pr.model_used,
                 "parse_history": list(s.get("parse_history", [])) + list(pr.history or []),
                 "parse_raw": pr.raw}                          # the model's last JSON, for debugging
        if not pr.ok:
            return {**stats, "status": "parse_failed",
                    "error": "Could not turn the question into a circuit: " + "; ".join(pr.errors)}
        warnings = list(pr.report.warnings) if pr.report else []
        warnings += [f"Missing information: {m}" for m in pr.missing_info]
        return {**stats, "netlist": pr.netlist.model_dump(mode="json"), "parse_warnings": warnings,
                "status": "parsed"}

    def _classify(self, s: State) -> Dict[str, Any]:
        nl = Netlist.model_validate(s["netlist"])
        return {"classification": asdict(classify(nl, client=self.client))}

    def _theory(self, s: State) -> Dict[str, Any]:
        if not (self.use_llm and self.client):
            return {"status": "answered", "explanation": "(theory answers need the LLM)"}
        ex = answer_conceptual(s["question"], s.get("level", "key"), client=self.client, conventions=self.conv)
        return {"status": "answered", "explanation": ex.text}

    def _unsupported(self, s: State) -> Dict[str, Any]:
        why = (s.get("classification") or {}).get("unsupported_reason") or s.get("error", "")
        return {"status": "unsupported", "explanation": f"This question is not supported yet: {why}"}

    def _confirm(self, s: State) -> Dict[str, Any]:
        if self.auto_confirm:
            return {"confirmed": True}
        nl = Netlist.model_validate(s["netlist"])
        cl = s.get("classification") or {}
        reply = interrupt({
            "question": s["question"],
            "netlist_text": describe(nl),
            "spice": nl.to_spice(),
            "svg": render_svg(nl),
            "warnings": s.get("parse_warnings", []) + cl.get("notes", []),
            "topic": cl.get("topic", ""),
        })
        if isinstance(reply, dict) and reply.get("ok"):
            return {"confirmed": True}
        correction = reply.get("correction", "") if isinstance(reply, dict) else str(reply or "")
        return {"confirmed": False, "hint": correction or "The circuit was not drawn correctly; re-read the question.",
                "confirm_rounds": s.get("confirm_rounds", 0) + 1}

    def _not_confirmed(self, s: State) -> Dict[str, Any]:
        return {"status": "not_confirmed",
                "explanation": f"The circuit was still wrong after {MAX_CONFIRM_ROUNDS} corrections. "
                               "Try rewording the question with every connection spelled out."}

    def _solve(self, s: State) -> Dict[str, Any]:
        nl = Netlist.model_validate(s["netlist"])
        try:
            res = get_solver(nl).solve(nl, self.conv)
        except NotSupportedError as e:
            return {"status": "unsupported", "error": str(e)}
        except SolverError as e:
            if s.get("solve_retries", 0) < MAX_SOLVE_RETRIES:
                return {"status": "retry_parse", "solve_retries": s.get("solve_retries", 0) + 1,
                        "hint": f"The solver could not solve the circuit you produced ({e}). "
                                "Re-read the question and fix the connections."}
            return {"status": "solve_failed", "error": f"Could not solve the circuit: {e}",
                    "explanation": f"Could not solve the circuit: {e}"}
        return {"result": res.model_dump(mode="json"), "status": "solved"}

    def _verify(self, s: State) -> Dict[str, Any]:
        if not self.verifier:
            return {}
        nl, res = Netlist.model_validate(s["netlist"]), SolveResult.model_validate(s["result"])
        rep = self.verifier.verify(nl, res)
        return {"result": res.model_dump(mode="json"), "verify_table": rep.table()}

    def _explain(self, s: State) -> Dict[str, Any]:
        if not self.do_explain:
            return {"status": "answered"}
        nl, res = Netlist.model_validate(s["netlist"]), SolveResult.model_validate(s["result"])
        topic = (s.get("classification") or {}).get("topic", "")
        ex = explain(nl, res, s.get("level", "key"), topic=topic, client=self.client,
                     use_llm=self.use_llm and self.client is not None)
        return {"status": "answered", "explanation": ex.text, "explain_warnings": ex.warnings}

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    @staticmethod
    def _after_parse(s: State) -> str:
        return "classify" if s.get("status") == "parsed" else END

    @staticmethod
    def _after_classify(s: State) -> str:
        cl = s.get("classification") or {}
        if cl.get("topic") == "conceptual":
            return "theory"
        if not cl.get("solver"):
            return "unsupported"
        return "confirm"

    @staticmethod
    def _after_confirm(s: State) -> str:
        if s.get("confirmed"):
            return "solve"
        return "parse" if s.get("confirm_rounds", 0) < MAX_CONFIRM_ROUNDS else "not_confirmed"

    def _after_solve(self, s: State) -> str:
        st = s.get("status")
        if st == "solved":
            return "verify" if self.do_verify else "explain"
        if st == "retry_parse":
            return "parse"
        if st == "unsupported":
            return "unsupported"
        return END

    def _build(self):
        g = StateGraph(State)
        for name, fn in [("parse", self._parse), ("classify", self._classify), ("theory", self._theory),
                         ("unsupported", self._unsupported), ("confirm", self._confirm),
                         ("not_confirmed", self._not_confirmed), ("solve", self._solve),
                         ("verify", self._verify), ("explain", self._explain)]:
            g.add_node(name, fn)
        g.add_edge(START, "parse")
        g.add_conditional_edges("parse", self._after_parse)
        g.add_conditional_edges("classify", self._after_classify)
        g.add_conditional_edges("confirm", self._after_confirm)
        g.add_conditional_edges("solve", self._after_solve)
        g.add_edge("verify", "explain")
        for end in ("theory", "unsupported", "not_confirmed", "explain"):
            g.add_edge(end, END)
        return g.compile(checkpointer=MemorySaver())