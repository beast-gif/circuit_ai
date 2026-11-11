import { useCallback, useState } from "react";
import { askCircuit, confirmCircuit } from "./api.js";
import AskPanel from "./components/AskPanel.jsx";
import ConfirmView from "./components/ConfirmView.jsx";
import AnswerView from "./components/AnswerView.jsx";
import ErrorDialog from "./components/ErrorDialog.jsx";
import { BrandMark, HeroSchematic, Symbol, Sweep } from "./components/Graphics.jsx";

const SOLVES = [
  ["network", "DC and AC networks", "Dividers, bridges, Thevenin and Norton, dependent sources, phasors."],
  ["opamp", "Op-amps and filters", "Amplifiers, integrators, Sallen-Key and multiple-feedback filters, Schmitt triggers."],
  ["diode", "Diodes and rectifiers", "Clippers, clampers, Zener regulators, half-wave and bridge rectifiers with ripple."],
  ["transistor", "BJT and MOSFET stages", "Bias points in every region, then gain, input and output resistance."],
  ["oscillator", "Oscillators", "Wien bridge, phase shift, Colpitts, Hartley and the astable multivibrator."],
];

export default function App() {
  // ask -> confirm -> answer, with the thread id carrying the session on the server
  const [stage, setStage] = useState("ask");
  const [busy, setBusy] = useState(false);
  const [sent, setSent] = useState({ question: "", preview: "" });
  const [threadId, setThreadId] = useState("");
  const [confirmation, setConfirmation] = useState(null);
  const [answer, setAnswer] = useState(null);
  const [problem, setProblem] = useState("");      // something the person can fix: shown inline
  const [ourError, setOurError] = useState(null);  // something they cannot: the contact dialog

  const handleError = useCallback((error) => {
    if (error.ours) setOurError(error);
    else setProblem(error.message);
  }, []);

  function take(reply) {
    if (reply.error) {
      handleError(reply.error);
      return;
    }
    setProblem("");
    setThreadId(reply.thread_id);
    if (reply.status === "needs_confirmation") {
      setConfirmation(reply.confirmation);
      setStage("confirm");
    } else if (reply.status === "answered") {
      setAnswer(reply.answer);
      setStage("answer");
    }
  }

  async function ask({ question, image, preview }) {
    setSent({ question, preview });
    setBusy(true);
    take(await askCircuit({ question, image }));
    setBusy(false);
  }

  async function accept() {
    setBusy(true);
    take(await confirmCircuit({ threadId, ok: true }));
    setBusy(false);
  }

  async function correct(correction) {
    setBusy(true);
    take(await confirmCircuit({ threadId, ok: false, correction }));
    setBusy(false);
  }

  function restart() {
    setStage("ask");
    setConfirmation(null);
    setAnswer(null);
    setProblem("");
    setThreadId("");
    setSent({ question: "", preview: "" });
  }

  return (
    <div className="page">
      <header className="top">
        <a className="brand" href="/" onClick={(e) => { e.preventDefault(); restart(); }}>
          <BrandMark />
          <span>Circuit AI</span>
        </a>
        <p className="top-note">Analog circuits, solved and checked</p>
      </header>

      <main>
        {stage === "ask" && (
          <>
            <section className="hero">
              <div className="hero-copy">
                <h1>Photograph a circuit. Get the worked solution.</h1>
                <p className="lede">
                  Circuit AI reads the schematic, solves it exactly with nodal analysis, and checks every
                  number against a SPICE simulation before showing it to you.
                </p>
                <AskPanel onSubmit={ask} busy={busy} problem={problem} />
              </div>
              <div className="hero-graphic">
                <HeroSchematic />
              </div>
            </section>

            <section className="solves" aria-labelledby="solves-title">
              <h2 id="solves-title">What it solves</h2>
              <ul>
                {SOLVES.map(([sym, title, text]) => (
                  <li key={sym}>
                    <Symbol name={sym} />
                    <h3>{title}</h3>
                    <p>{text}</p>
                  </li>
                ))}
              </ul>
            </section>

            <section className="how" aria-labelledby="how-title">
              <h2 id="how-title">How an answer is made</h2>
              <ol>
                <li>
                  <h3>It reads the circuit and shows it back</h3>
                  <p>The picture or question becomes a netlist, drawn out for you to confirm. Nothing is solved until you agree it is your circuit.</p>
                </li>
                <li>
                  <h3>It solves with algebra, not guesswork</h3>
                  <p>The equations are built and solved symbolically, so the answer is exact. The language model never does the arithmetic.</p>
                </li>
                <li>
                  <h3>A simulator checks the result</h3>
                  <p>The same circuit runs in ngspice, and the two answers are compared. You see whether they agree.</p>
                </li>
              </ol>
            </section>
          </>
        )}

        {stage === "confirm" && confirmation && (
          busy ? <Sweep label="Solving the circuit and checking it with ngspice…" /> : null
        )}
        {stage === "confirm" && confirmation && (
          <div className={busy ? "dimmed" : ""}>
            <ConfirmView sent={sent} confirmation={confirmation} busy={busy}
                         onAccept={accept} onCorrect={correct} onRestart={restart} />
            {problem && <p className="problem" role="alert">{problem}</p>}
          </div>
        )}

        {stage === "answer" && answer && (
          <AnswerView threadId={threadId} sent={sent} confirmation={confirmation} answer={answer}
                      onRestart={restart} onError={handleError} />
        )}
      </main>

      <footer className="foot">
        <p>Built by Amrith. Answers are checked by simulation, but check them against your own working too.</p>
      </footer>

      {ourError && <ErrorDialog error={ourError} onClose={() => setOurError(null)} />}
    </div>
  );
}
