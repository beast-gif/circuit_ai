import { useState } from "react";

// Before anything is solved, show what was understood. A wire traced wrongly in a photo is
// invisible to the person unless the circuit is drawn back to them.
export default function ConfirmView({ sent, confirmation, busy, onAccept, onCorrect, onRestart }) {
  const [fixing, setFixing] = useState(false);
  const [correction, setCorrection] = useState("");
  const lines = (confirmation.netlist_text || "").split("\n").filter(Boolean);

  return (
    <section className="stage" aria-labelledby="confirm-title">
      <h2 id="confirm-title">Is this your circuit?</h2>
      <p className="lede">Check the drawing against yours before it is solved. One wrong wire changes every answer.</p>

      <div className="panes">
        <div className="pane">
          <h3>You sent</h3>
          {sent.preview && <img className="sent-image" src={sent.preview} alt="The picture you sent" />}
          {sent.question && <p className="sent-question">{sent.question}</p>}
        </div>

        <div className="pane">
          <h3>Circuit AI read</h3>
          {confirmation.svg && (
            <div className="paper" aria-label="The circuit as it was read"
                 dangerouslySetInnerHTML={{ __html: confirmation.svg }} />
          )}
          <ul className="parts">
            {lines.map((l, i) => <li key={i}>{l}</li>)}
          </ul>
          {confirmation.question && sent.preview && !sent.question && (
            <p className="read-question"><strong>Question it read:</strong> {confirmation.question}</p>
          )}
          {confirmation.warnings?.length > 0 && (
            <div className="doubts" role="note">
              <strong>Read with some doubt</strong>
              <ul>{confirmation.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
            </div>
          )}
        </div>
      </div>

      {fixing ? (
        <form className="fix" onSubmit={(e) => { e.preventDefault(); if (correction.trim()) onCorrect(correction.trim()); }}>
          <label className="field">
            <span>What is wrong?</span>
            <textarea rows={2} autoFocus value={correction} onChange={(e) => setCorrection(e.target.value)}
                      placeholder="R2 is 4.7 kΩ, not 47 kΩ" />
          </label>
          <div className="row">
            <button className="btn btn-main" disabled={busy || !correction.trim()}>
              {busy ? "Reading the circuit…" : "Read the circuit again"}
            </button>
            <button type="button" className="btn btn-quiet" onClick={() => setFixing(false)}>Cancel</button>
          </div>
        </form>
      ) : (
        <div className="row">
          <button className="btn btn-main" disabled={busy} onClick={onAccept}>
            {busy ? "Solving…" : "Yes, solve it"}
          </button>
          <button className="btn btn-quiet" disabled={busy} onClick={() => setFixing(true)}>No, fix something</button>
          <button className="link" disabled={busy} onClick={onRestart}>Start over</button>
        </div>
      )}
    </section>
  );
}
