import { useRef, useState } from "react";

const OK_TYPES = ["image/png", "image/jpeg", "image/webp", "image/gif", "image/bmp"];

// The one form on the site: a picture, a question, or both.
export default function AskPanel({ onSubmit, busy, problem }) {
  const [question, setQuestion] = useState("");
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState("");
  const [over, setOver] = useState(false);
  const [localProblem, setLocalProblem] = useState("");
  const picker = useRef(null);

  function take(file) {
    if (!file) return;
    if (!OK_TYPES.includes(file.type)) {
      setLocalProblem("That file is not an image. Use a PNG or JPG — export a PDF page as an image first.");
      return;
    }
    setLocalProblem("");
    setImage(file);
    setPreview(URL.createObjectURL(file));
  }

  function clearImage() {
    setImage(null);
    setPreview("");
    if (picker.current) picker.current.value = "";
  }

  function submit(e) {
    e.preventDefault();
    if (!image && !question.trim()) {
      setLocalProblem("Add a picture of the circuit, type the question, or do both.");
      return;
    }
    setLocalProblem("");
    onSubmit({ question: question.trim(), image, preview });
  }

  const shown = localProblem || problem;

  return (
    <form className="ask" onSubmit={submit} aria-label="Ask about a circuit">
      <div
        className={"drop" + (over ? " drop-over" : "") + (preview ? " drop-filled" : "")}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); take(e.dataTransfer.files[0]); }}
      >
        {preview ? (
          <>
            <img src={preview} alt="The circuit picture you chose" />
            <button type="button" className="link" onClick={clearImage}>Remove picture</button>
          </>
        ) : (
          <>
            <p className="drop-title">Drop a picture of the circuit here</p>
            <p className="drop-note">A textbook figure, a screenshot or a photo of your notebook. If the question is printed beside the circuit, that is all you need.</p>
            <button type="button" className="btn btn-quiet" onClick={() => picker.current.click()}>
              Choose a picture
            </button>
          </>
        )}
        <input ref={picker} type="file" accept="image/*" hidden
               onChange={(e) => take(e.target.files[0])} />
      </div>

      <label className="field">
        <span>Question {image ? "(optional with a picture)" : ""}</span>
        <textarea
          rows={3}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="A 12 V source feeds R1 = 2 kΩ in series with R2 = 4 kΩ. Find the voltage across R2."
        />
      </label>

      {shown && <p className="problem" role="alert">{shown}</p>}

      <button className="btn btn-main" disabled={busy}>
        {busy ? "Reading the circuit…" : "Read the circuit"}
      </button>
    </form>
  );
}
