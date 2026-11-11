import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { fullSolution } from "../api.js";

function Verdict({ verified }) {
  if (verified === true) return <p className="verdict verdict-ok">Checked against an ngspice simulation — it agrees.</p>;
  if (verified === false) return <p className="verdict verdict-bad">The ngspice simulation disagrees. Treat this answer with care.</p>;
  return <p className="verdict verdict-none">Not checked by simulation.</p>;
}

export default function AnswerView({ threadId, sent, confirmation, answer, onRestart, onError }) {
  const [full, setFull] = useState("");
  const [loadingFull, setLoadingFull] = useState(false);
  const entries = Object.entries(answer.answers || {});

  async function showFull() {
    setLoadingFull(true);
    const r = await fullSolution(threadId);
    setLoadingFull(false);
    if (r.error) return onError(r.error);
    setFull(r.explanation || "");
  }

  return (
    <section className="stage" aria-labelledby="answer-title">
      <div className="panes panes-answer">
        <aside className="pane pane-pinned">
          <h3>The circuit</h3>
          {confirmation?.svg ? (
            <div className="paper" dangerouslySetInnerHTML={{ __html: confirmation.svg }} />
          ) : sent.preview ? (
            <img className="sent-image" src={sent.preview} alt="The picture you sent" />
          ) : null}
          {(confirmation?.question || sent.question) && (
            <p className="sent-question">{confirmation?.question || sent.question}</p>
          )}
          <button className="btn btn-quiet" onClick={onRestart}>Solve another circuit</button>
        </aside>

        <div className="pane">
          <h2 id="answer-title">Answer</h2>
          {entries.length > 0 && (
            <dl className="answers">
              {entries.map(([q, a]) => (
                <div key={q}>
                  <dt>{q}</dt>
                  <dd>{a}</dd>
                </div>
              ))}
            </dl>
          )}
          <Verdict verified={answer.verified} />
          {answer.verification && (
            <details className="compare">
              <summary>See the comparison with ngspice</summary>
              <pre>{answer.verification}</pre>
            </details>
          )}
          {answer.warnings?.length > 0 && (
            <div className="doubts" role="note">
              <strong>Worth knowing</strong>
              <ul>{answer.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
            </div>
          )}

          <h3 className="steps-title">{full ? "Full solution" : "Key steps"}</h3>
          <div className="prose">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{full || answer.explanation || ""}</ReactMarkdown>
          </div>
          {!full && answer.has_full && (
            <button className="btn btn-main" onClick={showFull} disabled={loadingFull}>
              {loadingFull ? "Writing the full solution…" : "Show the full solution"}
            </button>
          )}
        </div>
      </div>
    </section>
  );
}
