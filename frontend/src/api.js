// The three-call session with the FastAPI server (see server.py).
//
// Every call resolves to the server's JSON. A failure is never thrown as raw text:
// it comes back as { error: { code, message, ours } }, and `ours` decides what the
// person sees -- true means the "contact Amrith" dialog, false means the message
// itself, because it is something they can act on.

const BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const UNREACHABLE = {
  error: {
    code: "NET-DOWN",
    message: "Something went wrong on our side. Please contact Amrith (amrithdev.vercel.app) and try again.",
    ours: true,
  },
};

async function send(path, options) {
  let res;
  try {
    res = await fetch(BASE + path, options);
  } catch {
    return UNREACHABLE; // the server is not running, or the network dropped
  }
  let body;
  try {
    body = await res.json();
  } catch {
    return UNREACHABLE;
  }
  if (body && body.error) return body;
  if (!res.ok) {
    // FastAPI's own 400 for "neither question nor image"
    const detail = typeof body?.detail === "string" ? body.detail : "That request could not be read.";
    return { error: { code: "BAD-REQUEST", message: detail, ours: false } };
  }
  return body;
}

export function askCircuit({ question, image }) {
  const form = new FormData();
  form.append("question", question || "");
  form.append("level", "key");
  if (image) form.append("image", image);
  return send("/ask", { method: "POST", body: form });
}

export function confirmCircuit({ threadId, ok, correction }) {
  return send("/confirm", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ thread_id: threadId, ok, correction: correction || "" }),
  });
}

export function fullSolution(threadId) {
  return send(`/full/${encodeURIComponent(threadId)}`, { method: "GET" });
}
