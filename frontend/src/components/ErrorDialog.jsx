import { useEffect, useRef } from "react";

// Shown only for failures on our side (error.ours === true): no credit, a bad key, the
// provider down, a crash. The person can do nothing about those, so they get one plain
// sentence, who to contact, and a code Amrith can identify the cause from.
export default function ErrorDialog({ error, onClose }) {
  const button = useRef(null);

  useEffect(() => {
    button.current?.focus();
    const esc = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [onClose]);

  return (
    <div className="veil" onClick={onClose}>
      <div className="dialog" role="alertdialog" aria-modal="true" aria-labelledby="err-title"
           aria-describedby="err-body" onClick={(e) => e.stopPropagation()}>
        <svg viewBox="0 0 120 48" width="120" height="48" aria-hidden="true" className="fuse">
          <path d="M0 24 H30 M90 24 H120" />
          <rect x="30" y="12" width="60" height="24" rx="6" />
          <path d="M38 24 H52 L56 17 M64 31 L68 24 H82" />
        </svg>
        <h2 id="err-title">Circuit AI cannot answer right now</h2>
        <p id="err-body">
          The problem is on our side, not with your circuit. Please contact Amrith at{" "}
          <a href="https://amrithdev.vercel.app" target="_blank" rel="noreferrer">amrithdev.vercel.app</a>{" "}
          and mention this code.
        </p>
        <p className="code">{error.code}</p>
        <button ref={button} className="btn btn-main" onClick={onClose}>Close</button>
      </div>
    </div>
  );
}
