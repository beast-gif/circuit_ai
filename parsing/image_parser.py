"""
Image -> netlist (Phase 5, step 1).

The pipeline does not change: a vision model produces the SAME netlist JSON the text parser produces, and
it then goes through the SAME schema, validator, consistency checks and retry loop. Nothing downstream knows
or cares that the circuit came from a picture -- the solver still does all the mathematics, and ngspice
still verifies it.

WHAT IS DIFFERENT ABOUT AN IMAGE, and how each difference is handled:

1. The question may be in the picture. The model copies it into "question_text", and that text is what the
   consistency checks are run against. If the person also typed a question, the typed one wins (they can
   see the image and may be asking about one part of it), and both are kept in the netlist's question.

2. Reading can fail in ways text cannot. A value can be blurred, a wire can be ambiguous, a symbol can be
   drawn unusually. The model returns "reading_notes" for anything it was not sure of; those become
   warnings on the parse and are shown with the netlist, because a wrong value read off a photo is the one
   error no amount of careful solving can recover from. A value it cannot read at all becomes a symbol with
   a note, never an invented number.

3. The image is sent once. On a retry the correction goes back as text in the same conversation, so the
   model still has the picture in context but it is not re-uploaded (an image costs far more tokens than
   the error message).

4. Node tracing is the dominant failure mode, so the prompt makes the model do it first, and the existing
   consistency checks (dead ends, loops, op-amp feedback, ladders...) catch most of what it gets wrong --
   the same checks that catch the text parser's mistakes.
"""
from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from llm_client import LLMError, VelonaClient, extract_json
from parsing.prompts import (IMAGE_PARSER_SYSTEM, IMAGE_PARSER_USER, IMAGE_PARSER_USER_WITH_TEXT,
                             NOT_JSON_FEEDBACK, RETRY_FEEDBACK)
from parsing.text_parser import ParseResult, _pydantic_errors, consistency_errors, normalize
from schemas import Netlist
from validator import validate

MAX_IMAGE_BYTES = 8 * 1024 * 1024
OK_SUFFIX = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def check_image(image_path: str) -> Path:
    """Fail early and clearly: a missing or unreadable file is the user's problem, not the model's."""
    path = Path(image_path)
    if not path.exists():
        raise LLMError(f"Image not found: {image_path}")
    if path.suffix.lower() not in OK_SUFFIX:
        raise LLMError(f"{path.name}: not an image this can read (use PNG, JPG, WEBP, GIF or BMP). "
                       f"A PDF page must be exported as an image first.")
    size = path.stat().st_size
    if size == 0:
        raise LLMError(f"{path.name} is empty")
    if size > MAX_IMAGE_BYTES:
        raise LLMError(f"{path.name} is {size / 1e6:.1f} MB, over the {MAX_IMAGE_BYTES / 1e6:.0f} MB limit. "
                       f"Save it smaller (a schematic needs no more than ~1500 px across).")
    return path


def parse_image(image_path: str, question: str = "", client: Optional[VelonaClient] = None,
                max_attempts: int = 3, hint: Optional[str] = None) -> ParseResult:
    """Read a schematic image into a validated Netlist.

    question: what the person typed, if anything. It overrides any question printed in the image.
    hint:     a correction ("R2 is 2.2k, the label is hard to read") used on a re-parse.
    """
    client = client or VelonaClient()
    path = check_image(image_path)

    prompt = (IMAGE_PARSER_USER_WITH_TEXT.format(question=question.strip()) if question.strip()
              else IMAGE_PARSER_USER)
    if hint:
        prompt += (f"\n\nA previous reading of this image was wrong. Correction: {hint.strip()}\n"
                   "Apply this correction.")
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": IMAGE_PARSER_SYSTEM},
        {"role": "user", "content": [{"type": "text", "text": prompt},
                                     {"type": "image_url", "image_url": {"url": _data_url(path)}}]},
    ]

    model = client.model_for("image_parser")
    attempts, last_errors, raw, missing = 0, [], {}, []
    notes: List[str] = []
    best: Optional[tuple] = None
    history: List[Dict[str, Any]] = []

    def failed(stage: str, errs: List[str]) -> None:
        history.append({"attempt": attempts, "model": model, "stage": stage, "errors": list(errs)})

    for _ in range(max_attempts):
        attempts += 1
        try:
            text = client.chat("image_parser", messages, json_mode=True, model=model)
        except LLMError as e:
            last_errors = [str(e)]
            failed("request", last_errors)
            continue
        try:
            raw = extract_json(text)
        except LLMError as e:
            last_errors = [str(e)]
            failed("json", last_errors)
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": NOT_JSON_FEEDBACK}]
            continue

        missing = [str(m) for m in (raw.get("missing_info") or [])]
        notes = [str(n) for n in (raw.get("reading_notes") or [])]
        seen = str(raw.get("question_text") or "").strip()

        data = normalize(raw)
        # What the checks are run against: the typed question if there is one, else what was read off the
        # image. The netlist keeps both, so the explainer and the confirm step can show the person what the
        # model thought it was reading.
        asked = question.strip() or seen
        data["question"] = (f"{question.strip()}\n(question printed in the image: {seen})"
                            if question.strip() and seen else asked)

        try:
            nl = Netlist.model_validate(data)
        except ValidationError as e:
            errs, stage = _pydantic_errors(e, data), "schema"
        else:
            report = validate(nl)
            if report.ok:
                errs, stage = consistency_errors(asked, nl), "consistency"
                if not errs:
                    return _done(nl, report, missing, notes, attempts, model, raw, history, seen, question)
                if nl.analysis.value != "conceptual":
                    best = (nl, report, errs, list(missing), list(notes), raw, seen)
            else:
                errs, stage = report.errors, "validator"

        failed(stage, errs)
        last_errors = errs
        messages += [{"role": "assistant", "content": json.dumps(raw)},
                     {"role": "user", "content": RETRY_FEEDBACK.format(
                         errors="\n".join(f"- {x}" for x in errs))
                      + "\n\nLook at the image again before you answer: most of these come from a wire "
                        "traced wrongly or a symbol read wrongly, not from the JSON."}]

    if best is not None:
        nl, report, errs, missing, notes, raw, seen = best
        report.warnings += [f"Possible mismatch with the question: {e}" for e in errs]
        return _done(nl, report, missing, notes, attempts, model, raw, history, seen, question)
    return ParseResult(False, None, None, missing, attempts, model, last_errors, raw, history)


def _done(nl: Netlist, report, missing: List[str], notes: List[str], attempts: int, model: str,
          raw: Dict[str, Any], history: List[Dict[str, Any]], seen: str, question: str) -> ParseResult:
    """Attach what was read off the picture to the result, so it reaches the person."""
    for n in notes:
        report.warnings.append(f"Read from the image with some doubt: {n}")
    if not question.strip() and not seen:
        missing = list(missing) + ["the image showed no question, so the output node's voltage was assumed"]
    res = ParseResult(True, nl, report, missing, attempts, model, [], raw, history)
    res.raw = dict(raw)
    res.raw["_reading_notes"] = notes
    res.raw["_question_text"] = seen
    return res