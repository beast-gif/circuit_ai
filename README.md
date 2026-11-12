# Circuit AI

**Photograph an analog circuit. Get the worked solution — and a simulator's word that it is right.**

Circuit AI reads a question or a schematic, turns it into a netlist, solves it symbolically with nodal
analysis, and runs the same circuit through ngspice before showing you anything. The language model
reads and explains. It never does the arithmetic.

| | |
|---|---|
| Live site | https://circuit-ai.vercel.app |
| API | https://circuit-ai-api.onrender.com/docs |
| Stack | Python · SymPy · LangGraph · FastAPI · ngspice · React (Vite) |

---

## Why not just ask a language model?

Ask any LLM for the voltage across a resistor and it will give you a confident number, arrived at by
predicting plausible text. It is right most of the time, which is worse than being wrong all of the
time — you cannot tell the difference without doing the problem yourself.

Circuit AI splits the job along the line where each tool is actually reliable:

| Step | Done by | Because |
|---|---|---|
| Read the question or the picture | LLM | Language and vision are what it is good at |
| Build the equations | SymPy (`solvers/mna.py`) | Modified nodal analysis is mechanical, not a judgement call |
| Solve them | SymPy | Exact algebra — no floating-point guesswork, no hallucinated step |
| Check the result | ngspice | An independent program that has never seen the symbolic working |
| Write up the steps | LLM | It is handed the solved trace and asked to narrate it |

Every number you see came out of a solver and survived a simulation. The model's job was to read and
to explain.

---

## How an answer is made

```
question / image
      │
   ┌──▼──┐   LLM → JSON netlist, then a schema check, a validator
   │parse│   and consistency rules; two retries with the errors fed back
   └──┬──┘
      │
  ┌───▼────┐
  │classify│  dc · ac · transient · oscillator · comparator · diode · transistor
  └───┬────┘
      │
  ┌───▼───┐  the netlist is drawn back as a schematic and shown to you.
  │confirm│  LangGraph interrupt() — nothing is solved until you agree it
  └───┬───┘  is your circuit
      │
   ┌──▼──┐   MNA → A·x = b, solved symbolically.
   │solve│   The right solver is chosen by topology, not by the model
   └──┬──┘
      │
  ┌───▼──┐   the same circuit as a SPICE deck. Answers compared within
  │verify│   0.1 % (linear) or 5 % (BJT/MOSFET textbook approximations)
  └──┬───┘
      │
  ┌───▼───┐  key steps first; the full solution on request
  │explain│
  └───────┘
```

---

## The method, with numbers

### Linear networks — modified nodal analysis

Every linear circuit becomes one matrix equation **A·x = b**, where `x` holds the node voltages and
the currents through voltage sources. For a 12 V source feeding R1 = 2 kΩ in series with R2 = 4 kΩ,
KCL at the output node is

```
(V_out − V_in)/R1 + V_out/R2 = 0
```

With V_in fixed at 12 V by the source:

```
V_out · (1/2000 + 1/4000) = 12/2000
V_out · 7.5e-4            = 6.0e-3
V_out                     = 8 V exactly
```

SymPy solves it as algebra, so the answer is `8`, not `7.999999999999998`. ngspice is then asked the
same question and must agree to within 0.1 %.

### Oscillators — one condition for all of them

A linear oscillator has no input. Set the source vector to zero and the MNA system becomes

```
A(s) · x = 0
```

which has a non-trivial solution only where the matrix is singular:

```
det A(s) = 0          ← the characteristic equation
```

Substituting **s = jω** splits it into two real equations, and those two *are* the Barkhausen
criterion — not asserted, derived from the topology:

| | |
|---|---|
| `Im{det A(jω)} = 0` | the phase condition → the frequency of oscillation |
| `Re{det A(jω)} = 0` | the magnitude condition → the gain needed to sustain it |

The real part of the pole says what the circuit actually does: `σ = 0` sustained, `σ > 0` growing
until the rails clip it, `σ < 0` dying away.

The textbook formulas fall out of this on their own. A Wien bridge with R = 10 kΩ and C = 10 nF:

```
f0 = 1/(2πRC) = 1/(2π · 1e4 · 1e-8) = 1591.5494 Hz,   R2/R1 = 2
```

A three-section phase-shift oscillator with the same parts:

```
f0 = 1/(2πRC√6) = 649.7473 Hz,   Rf = 29R
```

Colpitts and Hartley go through the transistor's small-signal equivalent (gm, rπ = β/gm) and produce
the usual `gm·R_L = C_base/C_coll` and `gm·R_L = L_coll/L_base` conditions the same way.

### Relaxation oscillators — where det A(s) = 0 does not apply

An astable multivibrator is not a linear system; it is a comparator that keeps changing state. There
is no characteristic equation to take the determinant of. `solvers/relaxation.py` handles it per
state: find the trip point, find the Thevenin target the capacitor is charging towards, and time the
exponential:

```
t = τ · ln((v0 − Vf)/(v1 − Vf)),   τ = R_th · C
```

For the symmetric case with β = R2/(R1+R2), the two halves collapse to

```
T = 2RC · ln((1+β)/(1−β))
```

With R = 10 kΩ, C = 10 nF, β = 0.5: `T = 2e-4 · ln 3 = 219.72 µs`, so **f = 4551.2 Hz**. Asymmetric
supplies (say +12 V and −6 V) give unequal halves and the duty cycle is reported as well.

### Comparators — when there is no single answer

Inside a Schmitt trigger's hysteresis band, *both* output rails are self-consistent. A solver that
returns the first state it finds would claim +12 V for an input that might equally be at −12 V.
`solvers/comparator.py` collects **every** consistent state, and when more than one survives it says
so: the output depends on the input's history. That is the correct answer, and it is the one a
careless solver gets wrong.

---

## What it solves

| Area | Covered |
|---|---|
| DC and AC networks | Dividers, bridges, Thevenin and Norton, dependent sources, phasors, mesh and nodal |
| Op-amps | Inverting, non-inverting, summing, difference, integrators, differentiators |
| Filters | Sallen-Key, multiple-feedback, first-order RC/RL, cutoff and roll-off |
| Comparators | Schmitt triggers, trip points, hysteresis width, bistability |
| Diodes | Clippers, clampers, Zener regulators, half-wave and bridge rectifiers, ripple, PIV |
| Transistors | BJT and MOSFET bias in every region, then gain, Rin, Rout |
| Oscillators | Wien bridge, phase shift, Colpitts, Hartley, crystal, astable multivibrator |

Values may be written any way a textbook writes them, including RKM / BS 1852 shorthand where the
prefix replaces the decimal point: `2k2` = 2.2 kΩ, `4R7` = 4.7 Ω, `R47` = 0.47 Ω, `1M5` = 1.5 MΩ,
`2u2` = 2.2 µF.

---

## Reading pictures

A schematic goes to a vision model that returns **the same netlist JSON** as the text parser, plus
the question if it is printed beside the circuit. Everything downstream is unchanged, so every
solver works on a photograph the day it works on typed text.

It copes with hand-drawn sketches, textbook scans and phone photos; with IEC rectangles as well as
zig-zag resistor symbols; with wires that cross without a junction dot; and with several ground
symbols that are the same node. What it reads is always drawn back and shown to you before anything
is solved — that confirmation step exists because a misread schematic is the one failure a simulator
cannot catch.

---

## Layout

```
├── server.py            FastAPI: /ask → /confirm → /full
├── graph.py             LangGraph pipeline and the interrupt() confirmation
├── schemas.py           netlist schema, value parsing (SI + RKM shorthand)
├── validator.py         electrical sanity: floating nodes, shorted sources, missing ground
├── classifier.py        picks the analysis type
├── renderer.py          netlist → schematic (schemdraw) for the confirmation step
├── verifier.py          builds the SPICE deck, runs ngspice, compares
├── explainer.py         solved trace → key steps / full solution
├── parsing/
│   ├── prompts.py       the parser prompts and their worked examples
│   ├── text_parser.py   question → netlist, with retries on validation errors
│   └── image_parser.py  schematic → netlist (vision)
├── solvers/
│   ├── mna.py           modified nodal analysis: system() and solve()
│   ├── dc_ac.py         linear DC, AC and phasor circuits
│   ├── diode.py         piecewise-linear diodes, Zeners, rectifiers
│   ├── bjt.py           mosfet.py   small_signal.py   filters.py
│   ├── comparator.py    three op-amp states, trip points, bistability
│   ├── oscillator.py    det A(s) = 0
│   └── relaxation.py    astable timing
├── benchmark/           105 problems with hand-checked expected values
├── frontend/            Vite + React 18 single-page app
├── Dockerfile           Python + ngspice, for Render
└── DEPLOY.md            Render + Vercel, step by step
```

---

## Running it locally

**Prerequisites:** Python 3.11+, Node 18+, and the ngspice **console** program.

```bat
:: Windows: unzip ngspice-XX_64.zip to C:\Spice64 (config.yaml points at ngspice_con.exe)
:: Linux:   sudo apt-get install -y ngspice
```

```bat
git clone https://github.com/beast-gif/circuit-ai.git
cd circuit-ai

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

copy .env.example .env
:: put your Velona key in .env as VELONA_API_KEY=...
```

The key is read from the environment and only *named* in `config.yaml` (`api_key_env:
VELONA_API_KEY`). It is never written into a file in the repo, and never into a `VITE_` variable —
those are compiled into the public browser bundle.

Backend:

```bat
uvicorn server:app --reload --port 8000
```

Frontend, in a second terminal:

```bat
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The dev server talks to `http://127.0.0.1:8000` unless `VITE_API_URL`
says otherwise.

---

## The API

Three calls, tied together by a `thread_id` that holds the LangGraph checkpoint. The confirmation in
the middle is the point: the circuit is never solved until it has been read back to you.

| | |
|---|---|
| `POST /ask` | `question` and/or `image` (multipart) → what was understood: netlist, schematic SVG, warnings |
| `POST /confirm` | `{thread_id, ok, correction}` → the answer, or a re-read if you corrected it |
| `GET /full/{thread_id}` | the complete solution for a question already answered |
| `GET /health` | `{ok, version, origins, ngspice}` — `ngspice: false` means answers cannot be verified |

Errors come back as `{code, message, ours}`. `ours: true` means the fault is this service's, and the
app shows a contact dialog with the code rather than the raw message — a provider's billing or rate
limit reply is never put in front of a user.

---

## Benchmarks

```bat
python benchmark/run_benchmark.py benchmark/phase3.jsonl
python benchmark/make_image_cases.py --photos    :: generates the image set
```

105 problems across four sets — DC/AC networks, op-amps and filters, diodes and transistors,
oscillators — each with an expected value worked out by hand, independently of the solver. Grading
is strict: a textbook approximation does not count as a match for an exact expectation, and device
names must match by kind, not by string similarity. Results land in `benchmark/results/` and are not
committed; re-run before quoting a figure.

The image set is generated rather than stored: ten circuits rendered cleanly, then re-rendered as
simulated photographs (skewed, shadowed, slightly blurred) to test the vision parser against the
clean version of the same circuit.

---

## What it does not do

- **Large-signal nonlinear transients.** Diodes and transistors are solved piecewise-linearly, the
  way a textbook does. Anything needing a full nonlinear time-step solution is ngspice's job, not
  this solver's.
- **Distributed and RF effects.** Lumped elements only — no transmission lines, no S-parameters.
- **Digital logic and microcontrollers.** This is analog.
- **Sessions across restarts.** A `thread_id` lives in one process's memory, so the service runs as a
  single instance. Confirm a circuit and solve it in the same sitting.

---

Built by **Amrith** · [amrithdev.vercel.app](https://amrithdev.vercel.app) ·
[github.com/beast-gif](https://github.com/beast-gif)

Answers are checked by simulation, but check them against your own working too.
