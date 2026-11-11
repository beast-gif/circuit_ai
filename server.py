"""
The HTTP API (Step 7, part 1): the pipeline, wrapped so a browser can talk to it.

    pip install fastapi uvicorn python-multipart
    uvicorn server:app --reload            # http://127.0.0.1:8000/docs for the interactive docs

THE SHAPE OF A SESSION, and why it is three calls and not one
Solving is not a single request/response, because the circuit has to be confirmed before it is trusted:

    POST /ask        question and/or image  ->  what it READ: netlist, drawing, warnings   + a thread_id
    POST /confirm    thread_id, ok|correction ->  the answer, the key steps, the verification
    GET  /full/{id}  the complete solution, from the trace that is already stored (nothing is re-solved)

The thread_id is the whole session. It is LangGraph's checkpoint key, so the server keeps no state of its
own beyond the uploaded file -- a browser refresh loses nothing, and a correction re-parses the SAME image
because the image path lives in the checkpoint too.

The confirm step matters most for images: with a typed question the person can see what they typed, but a
single wire traced wrongly in a photograph is invisible to them. They would just get a confident wrong
answer. So /ask answers with what it understood, not with a number.

ERRORS COME IN TWO KINDS, and the frontend must treat them differently
  OURS   -- no balance, bad key, rate limit, provider down, timeout. The user can do nothing about these
            and must not see the raw text (the provider's message leaks the wallet balance and a top-up
            URL). They get one line plus a short code, and the code identifies which one it was.
  THEIRS -- not an image, a circuit that is not supported yet, a value that could not be read, a question
            that is not about a circuit. These are actionable, so the real reason is shown.
Every response carries {"error": {"code", "message", "ours": bool}} or nothing at all, so the UI has one
rule: `ours` -> the contact popup, otherwise -> show the message.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from graph import CircuitAI
from llm_client import LLMError
from solvers import NotSupportedError, SolverError

log = logging.getLogger("circuit_ai.api")

UPLOADS = Path(tempfile.gettempdir()) / "circuit_ai_uploads"
UPLOAD_MAX_AGE = 6 * 3600          # an abandoned session's image is deleted after this
OK_SUFFIX = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

app = FastAPI(title="Analog circuit solver", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000"
                                        ).split(",") if o],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# One instance for the process: it holds the LangGraph checkpointer, which is what makes a thread_id mean
# anything. Not auto-confirmed -- the confirmation step is the point of this API.
AI = CircuitAI(auto_confirm=False, do_explain=True, do_verify=True)


# ----------------------------------------------------------------------
# errors
# ----------------------------------------------------------------------
_OURS = [
    (re.compile(r"\b402\b|balance|insufficient|top up|credit", re.I), "LLM-402",
     "The service is out of credit."),
    (re.compile(r"\b401\b|\b403\b|unauthor|api[_ -]?key|forbidden", re.I), "LLM-401",
     "The service is not configured correctly."),
    (re.compile(r"\b429\b|rate limit|too many requests|quota", re.I), "LLM-429",
     "The service is busy."),
    (re.compile(r"timeout|timed out|connection|unreachable|\b5\d\d\b", re.I), "LLM-503",
     "The service is unreachable."),
]
CONTACT = "Something went wrong on our side. Please contact Amrith (amrithdev.vercel.app) and try again."


def error_body(exc: Exception) -> Dict[str, Any]:
    """Decide whether a failure is ours or the user's, and what they get told."""
    text = str(exc)
    if isinstance(exc, LLMError):
        for pattern, code, _ in _OURS:
            if pattern.search(text):
                log.error("%s: %s", code, text)             # the real reason goes to the log, not the user
                return {"code": code, "message": CONTACT, "ours": True}
        # an image that is missing, empty, too big or not an image: the user CAN fix that
        if any(w in text.lower() for w in ("image", "png", "jpg", "mb", "empty", "not found")):
            return {"code": "IMAGE-BAD", "message": text, "ours": False}
        log.error("LLM-ERR: %s", text)
        return {"code": "LLM-ERR", "message": CONTACT, "ours": True}
    if isinstance(exc, NotSupportedError):
        return {"code": "UNSUPPORTED", "message": text, "ours": False}
    if isinstance(exc, SolverError):
        return {"code": "SOLVE-FAILED", "message": text, "ours": False}
    log.exception("unhandled error")
    return {"code": "SERVER", "message": CONTACT, "ours": True}


def fail(exc: Exception) -> JSONResponse:
    body = error_body(exc)
    status = 503 if body["ours"] else 400
    return JSONResponse({"error": body}, status_code=status)


# ----------------------------------------------------------------------
# uploads
# ----------------------------------------------------------------------
def sweep_uploads() -> None:
    """Delete images from sessions nobody came back to."""
    now = time.time()
    for p in UPLOADS.glob("*"):
        try:
            if now - p.stat().st_mtime > UPLOAD_MAX_AGE:
                p.unlink()
        except OSError:
            pass


def save_upload(f: UploadFile) -> str:
    suffix = Path(f.filename or "upload.png").suffix.lower()
    if suffix not in OK_SUFFIX:
        raise LLMError(f"{f.filename}: not an image this can read (use PNG, JPG, WEBP, GIF or BMP). "
                       f"A PDF page must be exported as an image first.")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    sweep_uploads()
    path = UPLOADS / f"{uuid.uuid4().hex}{suffix}"
    with path.open("wb") as out:
        shutil.copyfileobj(f.file, out)
    return str(path)


# ----------------------------------------------------------------------
# responses
# ----------------------------------------------------------------------
class ConfirmIn(BaseModel):
    thread_id: str
    ok: bool = True
    correction: str = ""


def reply_body(r) -> Dict[str, Any]:
    """One shape for every reply, so the frontend has one branch: look at "status"."""
    s = r.state or {}
    out: Dict[str, Any] = {"thread_id": r.thread_id, "status": r.status}

    if r.status == "needs_confirmation":
        c = r.confirmation or {}
        out["confirmation"] = {
            "question": c.get("question", ""),           # what it will answer (read off the image, maybe)
            "netlist_text": c.get("netlist_text", ""),   # the circuit in words
            "spice": c.get("spice", ""),                 # and as a netlist
            "svg": c.get("svg", ""),                     # and as a drawing, to compare with the upload
            "warnings": c.get("warnings", []),           # anything read with doubt
            "topic": c.get("topic", ""),
        }
        return out

    if r.status == "answered":
        res = s.get("result") or {}
        out["answer"] = {
            "explanation": r.explanation,                        # the key steps, already written out
            "answers": res.get("answers") or {},
            "numeric": res.get("numeric") or {},
            "verified": res.get("verified"),                     # True / False / None (= not checked)
            "verification": s.get("verify_table", ""),
            "verification_notes": res.get("verification_notes") or [],
            "warnings": s.get("parse_warnings", []) + (s.get("explain_warnings") or []),
            "has_full": bool(res) or (s.get("classification") or {}).get("topic") == "conceptual",
        }
        if s.get("netlist"):
            out["circuit"] = {"spice": _spice(s), "question": s.get("question", "")}
        return out

    # parse_failed / unsupported / anything else that ended without an answer
    reason = s.get("error") or r.explanation or "The question could not be answered."
    code = {"parse_failed": "PARSE-FAILED", "unsupported": "UNSUPPORTED"}.get(r.status, "FAILED")
    out["error"] = {"code": code, "message": reason, "ours": False}
    return out


def _spice(s: Dict[str, Any]) -> str:
    from schemas import Netlist
    try:
        return Netlist.model_validate(s["netlist"]).to_spice()
    except Exception:                                           # noqa: BLE001 - cosmetic only
        return ""


# ----------------------------------------------------------------------
# endpoints
# ----------------------------------------------------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "version": app.version}


@app.post("/ask")
async def ask(question: str = Form(""), level: str = Form("key"),
              image: Optional[UploadFile] = File(None)) -> Any:
    """Read the question and/or the picture, and reply with what was understood -- not with the answer."""
    if not question.strip() and image is None:
        raise HTTPException(status_code=400, detail="Send a question, an image, or both.")
    path = None
    try:
        if image is not None:
            path = save_upload(image)
        r = AI.ask(question=question, level=level, image=path)
        return reply_body(r)
    except Exception as e:                                      # noqa: BLE001 - every failure is a reply
        if path:
            Path(path).unlink(missing_ok=True)
        return fail(e)


@app.post("/confirm")
def confirm(body: ConfirmIn) -> Any:
    """Accept the circuit as read, or correct it ("R2 is 4.7k") and have it read again."""
    try:
        r = AI.confirm(body.thread_id, ok=body.ok, correction=body.correction)
        return reply_body(r)
    except Exception as e:                                      # noqa: BLE001
        return fail(e)


@app.get("/full/{thread_id}")
def full(thread_id: str) -> Any:
    """The complete solution. Nothing is re-solved: it is written from the trace that is already stored."""
    try:
        return {"thread_id": thread_id, "explanation": AI.show_complete(thread_id)}
    except Exception as e:                                      # noqa: BLE001
        return fail(e)


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="127.0.0.1", port=8000)