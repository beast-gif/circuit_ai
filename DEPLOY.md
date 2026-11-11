# Deploying Circuit AI

Two services, because they are two different kinds of thing:

| part | runs on | why |
|---|---|---|
| `frontend/` (React) | Vercel | static files; nothing to install |
| everything else (FastAPI) | Render, as **Docker** | needs Python *and* the ngspice program |

The Docker part is not optional. Render's Python runtime cannot `apt-get install ngspice`, and
without ngspice every answer comes back "not checked by simulation" — the verification is the
part of this project worth showing.

---

## 0. Before the first push

```bat
git status --porcelain | findstr /i ".env"
```

Nothing but `.env.example` may appear. The key lives in `.env` (ignored); `config.yaml` only
names the variable it is read from.

```bat
cd C:\Users\amrit\Desktop\ece_question_solver
git init
git add .
git commit -m "Circuit AI: solver, API and frontend"
git branch -M main
git remote add origin https://github.com/beast-gif/circuit-ai.git
git push -u origin main
```

---

## 1. Backend on Render

1. **New + → Blueprint**, pick the repo. Render reads `render.yaml` and proposes
   **circuit-ai-api** (Docker, free, Singapore).
2. It asks for the two values marked `sync: false`:
   - `VELONA_API_KEY` — your key.
   - `ALLOWED_ORIGINS` — leave it as `http://localhost:5173` for now; the real URL comes in step 3.
3. Apply. The first build takes a few minutes (it installs ngspice and the Python packages).
4. Check it: open `https://circuit-ai-api.onrender.com/health` → `{"ok": true, ...}`, then
   `/docs` and run the three-call flow by hand as you did locally.

**Confirm ngspice really made it in.** Ask a question through `/docs` and look at the answer's
`verified` field. `true` means the simulator ran. `null` with a note about ngspice means the
image is wrong — check the build log for the apt step.

---

## 2. Frontend on Vercel

1. **Add New → Project**, pick the same repo.
2. **Root Directory: `frontend`** — this is the setting people miss; without it Vercel tries to
   build the Python project. Framework preset: Vite. Build `npm run build`, output `dist`.
3. Environment variable: `VITE_API_URL` = `https://circuit-ai-api.onrender.com` (no trailing slash).
4. Deploy. Note the URL it gives you, e.g. `https://circuit-ai.vercel.app`.

`VITE_` variables are baked in **at build time**, not read at runtime — changing `VITE_API_URL`
later means redeploying the frontend, not just restarting it.

---

## 3. Let them talk to each other

Back on Render → **Environment** → set

```
ALLOWED_ORIGINS = https://circuit-ai.vercel.app
```

exactly as Vercel shows it: `https://`, no trailing slash, no path. Save, which restarts the
service. If you later add your own domain, this becomes a comma-separated list:

```
ALLOWED_ORIGINS = https://circuit-ai.vercel.app,https://circuitai.amrithdev.in
```

A mismatch here shows up as the **contact-Amrith popup** with code `NET-DOWN`: the browser
blocked the response before the app could read it. The real reason is in the browser console as
a CORS error.

---

## 4. What to expect from the free tier

**The first request after 15 idle minutes takes about 50 seconds.** Render stops free instances,
and the next request wakes one. The app shows its "Reading the circuit…" state the whole time, so
it looks slow rather than broken. Two honest ways to deal with it:

- leave it, and mention it when you demo;
- or hit `/health` from a cron service (e.g. cron-job.org) every 10 minutes, which keeps it awake
  within the free 750 hours/month if it is your only service.

**Keep it to one instance.** A `thread_id` lives in this process's memory, so a second instance
would not recognise a session the first one started: the user would get an error between "read
the circuit" and "solve it". Do not enable autoscaling.

**Uploaded pictures are temporary.** They go to the container's temp directory and are deleted
after six hours, or whenever Render restarts the instance. That is fine for a session, but if an
instance restarts between the read and a correction, the re-read has no image to work from.

---

## 5. After every deploy, check these four

1. `GET /health` on Render → `{"ok": true}`
2. The site loads on Vercel and the hero circuit draws itself.
3. A typed question: 12 V, 2 kΩ, 4 kΩ → confirm → **8 V**, with the ngspice line saying it agrees.
4. An uploaded picture (`benchmark/images/divider.png`) → confirm → **8 V**.

If 3 works and 4 does not, it is the image request specifically — check Render's log for the
file size or the parse error.

---

## Cost

Everything above is free: Render's free web service, Vercel's hobby plan, and GitHub. The only
thing you pay for is Velona usage, which is per question — around 20k prompt tokens for an image
question, far less for a typed one.
