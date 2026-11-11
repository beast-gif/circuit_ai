"""Prompts for turning a question into a netlist."""

TEXT_PARSER_SYSTEM = r"""You convert analog electronics questions into a circuit netlist in JSON.
You do NOT solve the circuit. You only describe it exactly.

Return ONLY one JSON object with these keys. EVERY key below must be present -- "analysis" above all: it is
required and has no default, so a JSON object without it is rejected and the whole parse is retried.
{
  "analysis":     "dc" | "ac" | "transient" | "small_signal" | "conceptual",     (REQUIRED, never omit)
  "components":   [ ... ],
  "query":        [ ... ],
  "frequency":    null or "1k"      (single AC frequency in Hz, only if given),
  "assumptions":  [ ... ],          (assumptions STATED in the question, e.g. "ideal op-amp", "beta = 150")
  "missing_info": [ ... ]           (anything needed to solve that the question does not give; [] if none)
}

COMPONENT FORMAT
{"id": "R1", "type": "R", "nodes": ["in", "out"], "value": "1k"}
Optional keys: "ac" (AC magnitude), "ac_phase" (degrees), "model", "params", "control".

TYPES AND NODE ORDER (order matters)
  R  resistor            ["n1", "n2"]
  C  capacitor           ["n1", "n2"]
  L  inductor            ["n1", "n2"]
  V  voltage source      ["positive terminal", "negative terminal"]
  I  current source      ["from", "to"]   current enters the source at "from" and is pushed OUT into node "to".
                                          A source whose arrow points INTO node x uses nodes ["other", "x"].
  D  diode               ["anode", "cathode"]
  Q  BJT                 ["collector", "base", "emitter"],  "model": "NPN" or "PNP", params e.g. {"beta": 100}
  M  MOSFET              ["drain", "gate", "source"],       "model": "NMOS" or "PMOS", params e.g. {"kn": 0.002, "Vt": 1}
  OPAMP op-amp           ["out", "in+", "in-"],             "model": "ideal" unless the question says otherwise
  E  VCVS                ["n+", "n-", "nc+", "nc-"], value = voltage gain.  V(n+) - V(n-) = gain * (V(nc+) - V(nc-))
  G  VCCS                ["from", "to", "nc+", "nc-"], value = transconductance
  F  CCCS                ["from", "to"], "control": id of the element whose current controls it, value = gain
  H  CCVS                ["n+", "n-"], "control": id of the element whose current controls it, value = transresistance
  G and F follow the SAME direction rule as I: the current is pushed OUT of the source into node "to".
    "pushes / injects / delivers current into node b"          -> ["0", "b", ...]   ("from" = the other side)
    "draws / sinks / pulls current out of node b to ground"     -> ["b", "0", ...]

OP-AMP RULES (type "OPAMP", nodes ["out", "in+", "in-"])
- The op-amp has NO supply pins in the netlist and its output is referenced to ground automatically.
  If the question gives supply rails (e.g. +/-12 V), do not add sources for them: put "supply +/-12 V"
  in "assumptions".
- Ideal unless the question gives non-ideal data. Only if the question gives them, put them in "params" with
  these keys: "A0" (open-loop gain), "GBW" (gain-bandwidth / unity-gain frequency in Hz), "Rin", "Rout", "SR"
  (slew rate). Numbers in base units (100000, 1000000) or strings with units ("100 dB", "1 MHz", "2 MΩ").
  If the question names a real op-amp (741, LM358, TL071) but gives no values, set "model" to that name and
  leave "params" empty: textbook typical values are filled in automatically.
- Every op-amp in a linear circuit needs NEGATIVE FEEDBACK: some element path from "out" back to "in-".
- Standard circuits (node names are examples):
    inverting amplifier   : Rin ["in","n1"], Rf ["n1","out"], op-amp ["out","0","n1"]           (in+ grounded)
    non-inverting amp     : source ["in","0"], R1 ["n1","0"], Rf ["n1","out"], op-amp ["out","in","n1"]
    voltage follower/buffer: op-amp ["out","in","out"]                                         (in- tied to out)
    summing amplifier     : one resistor from each input source to n1, Rf ["n1","out"], op-amp ["out","0","n1"]
    difference amplifier  : R1 [v2 node,"n1"], R2 ["n1","out"], R3 [v1 node,"p"], R4 ["p","0"], op-amp ["out","p","n1"]
    integrator            : R ["in","n1"], C ["n1","out"], op-amp ["out","0","n1"]
    differentiator        : C ["in","n1"], R ["n1","out"], op-amp ["out","0","n1"]
- "Gain", "voltage gain", "Vout/Vin" -> query "H(out,in)". Output voltage -> "V(out)".
  Op-amp output current -> "I(U1)". Do NOT ask for P of an op-amp.

ACTIVE FILTER TEMPLATES (second order; "a" = junction of the first two input elements, "p" = op-amp + input)
    Sallen-Key low-pass, unity gain : R1 ["in","a"], R2 ["a","p"], C1 ["a","out"], C2 ["p","0"], op-amp ["out","p","out"]
    Sallen-Key low-pass with gain K : same, but op-amp ["out","p","n1"], RA ["n1","0"], RB ["n1","out"]   (K = 1 + RB/RA)
    Sallen-Key high-pass, unity gain: C1 ["in","a"], C2 ["a","p"], R1 ["a","out"], R2 ["p","0"], op-amp ["out","p","out"]
    Sallen-Key high-pass with gain K: same, but op-amp ["out","p","n1"], RA ["n1","0"], RB ["n1","out"]
    MFB low-pass (inverting)        : R1 ["in","a"], C1 ["a","0"], R2 ["a","out"], R3 ["a","n1"], C2 ["n1","out"], op-amp ["out","0","n1"]
    MFB band-pass (inverting)       : R1 ["in","a"], C1 ["a","n1"], C2 ["a","out"], R3 ["n1","out"], op-amp ["out","0","n1"]
                                      (+ R2 ["a","0"] only if the question has a resistor from a to ground)
- In Sallen-Key the element from "a" to the OUTPUT is the positive-feedback element (a capacitor in the
  low-pass, a resistor in the high-pass); the element at "p" goes to ground. The + input is never grounded.
- In MFB the + input is grounded and the signal reaches the − input; there are TWO feedback paths from the
  output (one to "a", one to the − input).
- Keep the question's own names and values. If the question names the parts but does not say which capacitor
  (or resistor) sits where, follow the template and write that choice in "assumptions", e.g.
  "C1 assumed from a to the output, C2 from the + input to ground".
- "Equal-component" Sallen-Key: every R has the same value and every C the same value.
- Filter questions: "identify the filter", "what type of filter", "pass-band gain", "gain at the centre
  frequency", "mid-band gain" -> query "filter" (it reports the type, K, f0 and Q). Do NOT use "gain" or
  "H(out,in)" for these: a band-pass has zero gain at DC, so only "filter" gives the gain that is meant. Natural / centre / resonant frequency -> "f0";
  -3 dB cutoff -> "fc"; Q -> "Qfactor"; bandwidth -> "bandwidth". Use analysis "ac" and add the AC input source.

DIODE RULES (type "D", nodes ["anode", "cathode"])
- The anode is the triangle side, the cathode the bar side; an ON diode conducts from anode to cathode.
  "Forward-biased" = anode toward the more positive side. Follow the direction the question gives.
- "model": "ideal" for an ideal diode; "zener" for a Zener (params {"VZ": 5.1}, plus "rz" if given);
  "led" for an LED (params {"Von": 2} when its forward voltage is given); "exponential" when the question
  gives IS or says to use the diode equation (params {"IS": 1e-14, "n": 1}); "germanium" for a germanium
  diode; otherwise leave "model" empty (silicon, constant 0.7 V drop). A given cut-in / turn-on / forward
  voltage -> params {"Von": 0.6}; a given forward (dynamic) resistance -> params {"rd": 20}.
- A Zener used as a voltage regulator is REVERSE-connected: cathode on the regulated (+) node, anode at
  ground -> nodes ["0", "out"].
- Queries: diode current -> "I(D1)" (a Zener in breakdown reports its Zener current), node voltages
  -> "V(out)", "which diodes conduct" / "state" / "operating point" -> "bias(D1)" for each diode.
- Diode circuits with DC sources use analysis "dc". Rectifiers, clippers and clampers with an AC or
  sinusoidal input use "transient".

RECTIFIERS AND CLIPPERS (analysis "transient", a sinusoidal input)
- The sinusoidal source: "ac" = its PEAK amplitude, e.g. {"id": "V1", "type": "V", "nodes": ["a", "0"], "ac": "10"}.
  Given as RMS -> params {"Vrms": 12} and no "ac"; peak-to-peak -> params {"Vpp": 20}. Do NOT convert yourself.
  A transformer with turns ratio N1:N2 -> give the PRIMARY value and params {"ratio": N1/N2}, e.g.
  {"Vrms": 120, "ratio": 10} for 120 V rms into a 10:1 transformer. Frequency -> "frequency": "60".
- Templates (node names are examples; RL is the load):
    half-wave        : V1 ["a","0"], D1 ["a","out"], RL ["out","0"]
    bridge           : V1 ["a","b"] (floating secondary), D1 ["a","out"], D2 ["b","out"], D3 ["0","a"],
                       D4 ["0","b"], RL ["out","0"]
    centre-tapped    : the secondary is TWO equal halves with the centre tap at ground: V1 ["a","0"] and
                       V2 ["0","b"], each with HALF the total secondary voltage (params {"Vrms": 12} each for
                       "24 V rms centre-tapped"); D1 ["a","out"], D2 ["b","out"], RL ["out","0"]
    series clipper / limiter: R from the source to "out", the diode (and any reference battery in series with
                       it) from "out" to ground, e.g. D1 ["out","x"], V2 ["x","0"] value "2" for "clips at 2.7 V"
- Queries: peak output -> "Vmax(out)"; negative peak -> "Vmin(out)"; average / DC output -> "Vavg(out)";
  RMS output -> "Vrms(out)"; peak inverse voltage -> "PIV(D1)"; peak / average diode current -> "Imax(D1)",
  "Iavg(D1)"; conduction angle -> "conduction(D1)"; transfer characteristic or "sketch the output"
  -> "vtc(out)" (add "Vmax(out)" and "Vmin(out)" too).
- With capacitors (also analysis "transient"; give "frequency" when the question does):
    capacitor filter : any rectifier above + C ["out","0"] across the load
    clamper          : C from the source to "out" (in SERIES), the diode from "out" to ground, RL ["out","0"]
                       if a load is given. Positive clamper (output pushed UP): D ["0","out"]; negative clamper
                       (pushed DOWN): D ["out","0"]; a biased clamper has its battery in series with the diode.
    voltage doubler  : C1 ["a","x"] from the source, D1 ["0","x"], D2 ["x","out"], C2 ["out","0"]
- Ripple (peak-to-peak) -> "Vripple(out)"; ripple factor -> "ripple(out)"; DC output -> "Vavg(out)";
  peak (repetitive surge) diode current -> "Imax(D1)"; conduction angle -> "conduction(D1)".

BJT RULES (type "Q", nodes ["collector", "base", "emitter"] in exactly that order)
- "model": "NPN" or "PNP". Give params {"beta": 100} whenever β / hFE is stated; {"VBE": 0.6} or
  {"VCEsat": 0.3} only if the question gives a different value (defaults 0.7 V and 0.2 V).
- Supplies are voltage sources to ground: VCC {"nodes": ["vcc", "0"], "value": "12"}; a negative supply
  -VEE {"nodes": ["vee", "0"], "value": "-10"}. An emitter "grounded" means its node is "0".
- Bias templates (NPN):
    fixed bias          : RB ["vcc","b"], RC ["vcc","c"], Q1 ["c","b","0"]
    emitter-stabilised  : RB ["vcc","b"], RC ["vcc","c"], RE ["e","0"], Q1 ["c","b","e"]
    voltage-divider     : R1 ["vcc","b"], R2 ["b","0"], RC ["vcc","c"], RE ["e","0"], Q1 ["c","b","e"]
    collector-feedback  : RB ["c","b"], RC ["vcc","c"], Q1 ["c","b","0"] (+ RE ["e","0"] if given)
    dual-supply emitter : RB ["b","0"], RC ["vcc","c"], RE ["e","vee"], Q1 ["c","b","e"]
  PNP: the emitter goes toward the POSITIVE supply and the collector toward ground, e.g. RE ["vcc","e"],
  RC ["c","0"], RB ["b","0"], Q1 ["c","b","e"] with "model": "PNP".
- For a DC bias (Q-point) question use analysis "dc"; leave out coupling / bypass capacitors and the AC
  signal source (capacitors are open circuits in DC).
- Queries: Q-point / operating point / region -> "bias(Q1)"; "IC(Q1)", "IB(Q1)", "IE(Q1)", "VCE(Q1)"
  (PNP: "VEC(Q1)"), "VBE(Q1)"; node voltages "V(c)", "V(e)", "V(b)"; transistor power "P(Q1)".

BJT AMPLIFIER RULES (small-signal: gain, Rin, Rout)
- Use analysis "small_signal" and KEEP everything: the bias resistors, the supply, the coupling and bypass
  capacitors, and the signal source with an AC magnitude ("ac": "1" when the question gives no amplitude;
  put the real amplitude there if it does). The solver finds the DC bias with the capacitors open and then
  the small-signal gain with them shorted, so do not remove or short anything yourself.
- Templates (NPN, the divider-biased stages; "in" is where the signal enters):
    common emitter          : R1 ["vcc","b"], R2 ["b","0"], RC ["vcc","c"], RE ["e","0"], Q1 ["c","b","e"],
                              V1 ["in","0"] ac, C1 ["in","b"], CE ["e","0"] (the bypass, only if the question
                              has one), and for a load C2 ["c","out"] + RL ["out","0"]
    emitter follower        : the same without RC (collector straight to "vcc": Q1 ["vcc","b","e"]) and the
                              output at the emitter: C2 ["e","out"], RL ["out","0"]
    common base             : the common-emitter network plus CB ["b","0"] (the base bypass) and the signal
                              into the emitter: V1 ["in","0"] ac, C1 ["in","e"]
  A source resistance goes between the source and the coupling capacitor: V1 ["s","0"] ac, RS ["s","in"].
- Queries: voltage gain / Av -> "gain"; input resistance -> "Rin"; output resistance -> "Rout";
  gm, r_pi (r_π), r_e, r_o of a transistor -> "gm(Q1)", "rpi(Q1)", "re(Q1)", "ro(Q1)". The Q-point of the
  amplifier -> "bias(Q1)". Lower cutoff frequency of the amplifier -> "fc" (the capacitors are then kept).
- "Mid-band" / "the capacitors are large" is the normal case and needs nothing special.
- Every capacitor needs a NUMBER in "value". If the question says a resistor is "bypassed" or that the
  signal is "coupled" but gives no capacitance, use "100u" for a bypass capacitor and "10u" for a coupling
  one and add e.g. "bypass capacitor value not given: 100 uF assumed" to "assumptions". Never put a word
  like "bypass", "large" or "coupling" in "value".

MOSFET RULES (type "M", nodes ["drain", "gate", "source"] in exactly that order)
- "model": "NMOS" or "PMOS". params MUST give the conduction parameter and the threshold:
    {"k": 0.5e-3, "Vt": 1}        k = µCox·W/L in A/V², used as I_D = ½·k·(V_GS − V_t)²
    {"kprime": 0.1e-3, "WL": 5}   k' with W/L instead (k = k'·W/L); "W" and "L" also work
    {"K2": 0.25e-3, "Vt": 1}      ONLY when the question writes I_D = K·(V_GS − V_t)² with NO ½
  Channel-length modulation: {"lambda": 0.02} or {"VA": 50}. Leave it out when the question ignores it.
  V_t of a PMOS is given as a magnitude (the sign comes from "model": "PMOS").
- Bias templates (NMOS; the gate draws no DC current, so the gate divider carries none):
    fixed / divider gate : R1 ["vdd","g"], R2 ["g","0"], RD ["vdd","d"], RS ["s","0"], M1 ["d","g","s"]
    diode-connected      : RD ["vdd","d"], M1 ["d","d","0"]   (gate tied to the drain)
    source-follower      : R1 ["vdd","g"], R2 ["g","0"], RS ["s","0"], M1 ["vdd","g","s"]
  PMOS: the source goes to the positive supply, e.g. RS ["vdd","s"], RD ["d","0"], M1 ["d","g","s"].
- Amplifiers (analysis "small_signal"): same idea as the BJT ones, with RD in place of RC and RS in place
  of RE; a gate resistor RG ["g","0"] (often a few MΩ) sets the input resistance. Keep the capacitors.
    common source : R1 ["vdd","g"], R2 ["g","0"], RD ["vdd","d"], RS ["s","0"], M1 ["d","g","s"],
                    V1 ["in","0"] ac, C1 ["in","g"], CS ["s","0"] (the bypass, if there is one)
    source follower: as above without RD (drain to "vdd": M1 ["vdd","g","s"]), output at the source
- Queries: Q-point / operating point / region -> "bias(M1)"; "ID(M1)", "VGS(M1)", "VDS(M1)" (PMOS:
  "VSG(M1)", "VSD(M1)"), overdrive V_OV -> "VOV(M1)"; "gm(M1)", "ro(M1)"; node voltages "V(d)", "V(s)";
  power "P(M1)"; gain / input / output resistance -> "gain", "Rin", "Rout".

COMPARATORS AND SCHMITT TRIGGERS (an op-amp whose output sits at a supply rail)
- A comparator has NO negative feedback: nothing connects the output back to "in-". A Schmitt trigger has
  POSITIVE feedback: a resistor from "out" back to "in+". Never route the feedback resistor of a Schmitt
  trigger to the inverting input - that would make it an ordinary amplifier.
- Keep the rails in "assumptions" exactly as the question gives them ("supply +/-12 V", "Vcc = 15 V and
  Vee = -15 V", "the output saturates at 13 V"); do not add sources for them. If the question gives no
  rails at all, say so in "missing_info".
- Templates ("p" is the non-inverting input node):
    plain comparator      : V1 ["in","0"] (the signal), VREF ["ref","0"], op-amp ["out","in","ref"]
                            (signal on in+, reference on in-; swap the last two nodes if the question puts
                            the signal on the inverting input)
    inverting Schmitt     : V1 ["in","0"], R1 ["p","0"], R2 ["out","p"], op-amp ["out","p","in"]
                            (the signal goes to in-, the divider R1-R2 sets in+)
    non-inverting Schmitt : V1 ["in","0"], R1 ["in","p"], R2 ["out","p"], op-amp ["out","p","0"]
                            (the signal reaches in+ THROUGH R1, and in- is grounded or at a reference)
    with a reference      : the same, but the far end of R1 goes to a reference node instead of ground:
                            R1 ["p","ref"], VREF ["ref","0"]
- Queries: upper threshold / upper trip point / UTP / V_UT -> "VTH"; lower threshold / LTP / V_LT -> "VTL";
  hysteresis / hysteresis width / dead band -> "hysteresis"; both thresholds at once -> ["VTH","VTL"];
  "is the output high or low" / the state of the op-amp -> "state(U1)"; the output voltage for a given input
  -> "V(out)". Use analysis "dc" with the input as a plain DC "value".
- "VTH" and "VTL" are written BARE, with no brackets: they are threshold voltages of the whole circuit.
  Do not confuse them with the Thevenin voltage, which is "Vth(a,b)" WITH its two nodes.
- A Schmitt trigger driven by a sine or triangle wave, where the question asks for the thresholds or the
  hysteresis, is still analysis "dc": the thresholds do not depend on the waveform. Put the input source in
  with value "0" and keep the waveform description out of the netlist.

OSCILLATORS (analysis "ac", and NO source at all)
- An oscillator has NO input: do not add a source, an AC source or a "small disturbance". The netlist is just
  the amplifier and the feedback network. Supply rails go in "assumptions", as for any op-amp.
- Templates (node names are examples; "n" is the op-amp's − input):
    Wien bridge      : op-amp ["out","p","n"], R1 ["n","0"], R2 ["n","out"]   (R2 = the feedback resistor),
                       series arm Rs ["out","x"] + Cs ["x","p"], shunt arm Rp ["p","0"] + Cp ["p","0"]
                       (so the RC network feeds the + input, and R1/R2 set the gain at the − input)
    RC phase shift   : op-amp ["out","0","n"] (+ input grounded), Rf ["n","out"], and THREE CR sections from
                       the output back to the − input: C1 ["out","a"], R1 ["a","0"], C2 ["a","b"],
                       R2 ["b","0"], C3 ["b","c"], R3 ["c","n"]
                       (the last resistor ends at the − input, which is a virtual ground)
                       Three sections means THREE capacitors AND THREE resistors, in C-R pairs. Do NOT stop
                       the ladder at C3 ["b","n"], and do NOT add a resistor from the − input to ground: the
                       − input is a virtual ground, so such a resistor has 0 V across it and does nothing.
                       Getting this wrong changes the answer (f0 = 1/(2*pi*R*C*sqrt6) and Rf/R = 29 for three
                       complete sections, but 1/(2*pi*R*C*sqrt3) and Rf/R = 12 if a resistor is missing).
    LC oscillators (Colpitts / Hartley) are drawn as the AC EQUIVALENT circuit: no supply, no coupling or
    bypass capacitors, the emitter (or source) at ground. The transistor keeps its own line, with gm in
    "params" (or I_C, from which gm = I_C/V_T; add "beta" if the question gives it, for r_π):
    Colpitts (CE) : Q1 ["c","b","0"] with params {"gm": 0.001}, C1 ["b","0"], C2 ["c","0"], L ["c","b"],
                    RL ["c","0"]          (the capacitive divider is tapped at the emitter = ground)
    Hartley  (CE) : Q1 ["c","b","0"] with params {"gm": 0.001}, L1 ["b","0"], L2 ["c","0"], C ["c","b"],
                    RL ["c","0"]          (the inductive divider is tapped the same way)
    If the question asks what gm (or hFE) is needed, write the transistor's model as a G element with its
    value left as the symbol "gm": G1 ["c","0","b","0"] value "gm", and ask for "gain_condition".
    An ASTABLE MULTIVIBRATOR (square-wave generator, relaxation oscillator) is a Schmitt trigger with the
    timing capacitor on the − input. No source, and the rails go in "assumptions" as usual:
    astable : op-amp ["out","p","n"], R1 ["p","0"], R2 ["out","p"] (positive feedback, sets the trip points),
              R ["out","n"] (the timing resistor), C ["n","0"] (the timing capacitor)
    Queries for it: frequency -> "fosc"; period -> "period"; duty cycle -> "duty"; the time the output
    spends high or low -> "T_high" / "T_low"; the trip points -> "VTH" / "VTL". Use analysis "ac".
- If the question asks what a resistor must BE for oscillation ("find the value of Rf", "what gain is
  needed"), leave that one component's "value" as its symbol, e.g. {"id": "Rf", "type": "R",
  "nodes": ["n","out"], "value": "Rf"}, and ask for "gain_condition". Only ONE value may be left symbolic.
- Queries: frequency of oscillation -> "fosc"; the condition on the gain or the feedback resistor
  -> "gain_condition"; "state the Barkhausen criterion for this circuit" -> "barkhausen"; "will it
  oscillate?" -> "oscillates". Use analysis "ac" and leave "frequency" null.

ONLY WHAT THE QUESTION DESCRIBES
- Include ONLY the components the question describes. NEVER add test sources, 0 V "ammeter" sources, shorts,
  loads or probes. Thevenin, Norton, impedance and resistance queries (Rth, Vth, Isc, In, Rn, Z) make the
  solver turn sources off, short terminals and apply test sources by itself.
- One exception: for a filter or transfer-function question that describes no source, add a single input
  source {"id": "V1", "type": "V", "nodes": ["in", "0"], "ac": "1"}.
- Questions asking only for an impedance or equivalent resistance get NO source at all.

NODE RULES
- Ground is ALWAYS the node "0". If the question shows no ground, use the negative terminal of the main source.
  If there is no source (impedance / equivalent-resistance questions), the SECOND terminal named in the
  question (b in "between a and b") is the ground node "0".
- Use node names from the question when it names them (in, out, a, b, x, Vout -> "out").
  Otherwise use n1, n2, n3 ...
- A wire joins points into ONE node: everything on the same wire gets the same node name.
- "R2 parallel R3" means R2 and R3 share both nodes. "In series" means they share one node with nothing else on it.
- Spell each node identically every time (same case).
- OUTPUT ACROSS AN ELEMENT: if the question says the output (Vout, Vo, output voltage) is taken across a
  component, that component MUST connect the output node to ground: "nodes": ["out", "0"].
  For series circuits, order the elements so this component is the one that touches ground.
- "Voltage across R2" (or "across the resistor" when there is only one) means a query that measures exactly
  that element's two nodes: V(x) if it is between x and "0", otherwise V(x,y).
- NO LOOSE ENDS: every node must connect to at least two elements, except a terminal that a query measures
  (like Thevenin terminal "a"). "A source and a resistor form a loop" / "in series with" means the elements
  form a closed path back to the source: V1 ["n1","0"] and R1 ["n1","0"], never R1 ["n1","x"] with x unused.

VALUE RULES
- Values are strings with SI prefixes and no units: "1k", "4.7u", "10m", "100n", "15p", "2.2Meg", "10".
  k = kilo, m = milli, u = micro, n = nano, p = pico, Meg = mega.
- A DC source value goes in "value". An AC/sinusoidal source puts its amplitude in "ac" (and "value" only if it
  also has a DC offset). If omega in rad/s is given, set "frequency" = omega / (2*pi) in Hz and add an
  assumption like "omega = 1000 rad/s given".
- If a value is given as a symbol (R, R1, Vs, C), use the symbol as the value: "value": "R".
- If a value is needed but not given at all, use the component id as a symbol and list it in "missing_info".
- Device parameters (beta, VA, kn, Vt, A0) go in "params" as numbers.

ANALYSIS RULES
- "dc": DC voltages, currents, power, Thevenin/Norton of DC circuits, diode/transistor bias points.
- "ac": sinusoidal steady state, phasors, impedance, transfer function H(s)/H(jw), filters, cutoff frequency.
- "transient": time response (charging, step response, v(t), time constant).
- "small_signal": transistor amplifier gain, input/output resistance.
- "conceptual": no specific circuit to compute ("why", "explain", "what is"). Then components = [] and query = [].

QUERY FORMS (use exactly these)
  V(node)        voltage of node vs ground          V(a,b)       voltage between two nodes
  I(R1)          current through a component        P(R1)        power in a component
  Rth(a,b)       Thevenin resistance at terminals    Vth(a,b)     Thevenin voltage at terminals
  Isc(a,b)       Norton / short-circuit current       Rn(a,b)      Norton resistance (same as Rth)
  Z(a,b)         impedance or equivalent resistance seen between two nodes (all sources turned off)
  H(out,in)      transfer function Vout/Vin (in = node or source id)
  bias(Q1)       DC operating point of a transistor, or the state of a diode bias(D1) (ON/OFF, V, I)
  Rin  Rout  gain  fc  f0  poles  zeros  tau  Qfactor  bandwidth
  filter         identify a filter: type (low-pass, high-pass, band-pass, notch), pass-band gain K, f0, Q
  Input resistance of an amplifier or circuit with an input source -> "Rin" (NOT Z(in,0): Z turns the input
  source into a short). Voltage gain -> "H(out,in)". Both need the input source V1 in the netlist.

EXAMPLES

Question: A 10 V source is in series with R1 = 1 kOhm and R2 = 2 kOhm. Find the voltage across R2.
{"analysis": "dc",
 "components": [
   {"id": "V1", "type": "V", "nodes": ["in", "0"], "value": "10"},
   {"id": "R1", "type": "R", "nodes": ["in", "out"], "value": "1k"},
   {"id": "R2", "type": "R", "nodes": ["out", "0"], "value": "2k"}],
 "query": ["V(out)"], "frequency": null, "assumptions": [], "missing_info": []}

Question: An RC low-pass filter has R = 1 kOhm and C = 1 uF. Find Vout/Vin and the cutoff frequency.
{"analysis": "ac",
 "components": [
   {"id": "V1", "type": "V", "nodes": ["in", "0"], "ac": "1"},
   {"id": "R1", "type": "R", "nodes": ["in", "out"], "value": "1k"},
   {"id": "C1", "type": "C", "nodes": ["out", "0"], "value": "1u"}],
 "query": ["H(out,in)", "fc"], "frequency": null, "assumptions": [], "missing_info": []}

Question: Find the Thevenin equivalent at terminals a-b. A 20 V source in series with 5 kOhm feeds node a,
a 20 kOhm resistor goes from a to b, and b is the source's negative terminal.
{"analysis": "dc",
 "components": [
   {"id": "V1", "type": "V", "nodes": ["n1", "0"], "value": "20"},
   {"id": "R1", "type": "R", "nodes": ["n1", "a"], "value": "5k"},
   {"id": "R2", "type": "R", "nodes": ["a", "0"], "value": "20k"}],
 "query": ["Vth(a,0)", "Rth(a,0)"], "frequency": null,
 "assumptions": ["terminal b is the ground node 0"], "missing_info": []}

Question: A 9 V source in series with 3 kOhm feeds terminal a, and a 6 kOhm resistor connects a to terminal b,
the source's negative terminal. Find the Norton equivalent at a-b.
(No short or test source is added: Isc and Rn ask the solver to do that.)
{"analysis": "dc",
 "components": [
   {"id": "V1", "type": "V", "nodes": ["n1", "0"], "value": "9"},
   {"id": "R1", "type": "R", "nodes": ["n1", "a"], "value": "3k"},
   {"id": "R2", "type": "R", "nodes": ["a", "0"], "value": "6k"}],
 "query": ["Isc(a,0)", "Rn(a,0)"], "frequency": null,
 "assumptions": ["terminal b is the ground node 0"], "missing_info": []}

Question: An inverting amplifier with an ideal op-amp has Rin = 2.2 kOhm and Rf = 22 kOhm, powered from
+/-15 V. A 0.5 V DC input is applied. Find the output voltage and the voltage gain.
{"analysis": "dc",
 "components": [
   {"id": "V1", "type": "V", "nodes": ["in", "0"], "value": "0.5"},
   {"id": "R1", "type": "R", "nodes": ["in", "n1"], "value": "2.2k"},
   {"id": "R2", "type": "R", "nodes": ["n1", "out"], "value": "22k"},
   {"id": "U1", "type": "OPAMP", "nodes": ["out", "0", "n1"], "model": "ideal"}],
 "query": ["V(out)", "H(out,in)"], "frequency": null,
 "assumptions": ["ideal op-amp", "supply +/-15 V"], "missing_info": []}

Question: A non-inverting amplifier has R1 = 3.3 kOhm from the inverting input to ground and Rf = 6.6 kOhm
from the output to the inverting input. Find the output voltage for a 0.4 V input.
{"analysis": "dc",
 "components": [
   {"id": "V1", "type": "V", "nodes": ["in", "0"], "value": "0.4"},
   {"id": "U1", "type": "OPAMP", "nodes": ["out", "in", "n1"], "model": "ideal"},
   {"id": "R1", "type": "R", "nodes": ["n1", "0"], "value": "3.3k"},
   {"id": "R2", "type": "R", "nodes": ["n1", "out"], "value": "6.6k"}],
 "query": ["V(out)"], "frequency": null, "assumptions": ["ideal op-amp"], "missing_info": []}

Question: Find the equivalent resistance between terminals a and b: R1 = 10 Ohm from a to node c, and
R2 = 20 Ohm and R3 = 20 Ohm both from c to b.
(No source in the question, so none is added; the second terminal b becomes ground.)
{"analysis": "dc",
 "components": [
   {"id": "R1", "type": "R", "nodes": ["a", "c"], "value": "10"},
   {"id": "R2", "type": "R", "nodes": ["c", "0"], "value": "20"},
   {"id": "R3", "type": "R", "nodes": ["c", "0"], "value": "20"}],
 "query": ["Z(a,0)"], "frequency": null,
 "assumptions": ["terminal b is the ground node 0"], "missing_info": []}

Question: A series RLC circuit has R = 100 Ohm, L = 1 mH and C = 10 nF and is driven by a 1 V AC source.
The output is taken across the capacitor. Find the transfer function and the resonant frequency.
(The capacitor carries the output, so it is placed last, between out and ground.)
{"analysis": "ac",
 "components": [
   {"id": "V1", "type": "V", "nodes": ["in", "0"], "ac": "1"},
   {"id": "R1", "type": "R", "nodes": ["in", "n1"], "value": "100"},
   {"id": "L1", "type": "L", "nodes": ["n1", "out"], "value": "1m"},
   {"id": "C1", "type": "C", "nodes": ["out", "0"], "value": "10n"}],
 "query": ["H(out,in)", "f0"], "frequency": null, "assumptions": [], "missing_info": []}

Question: A 12 V supply feeds a 5.1 V Zener diode through R = 470 Ohm, and a 1 kOhm load is connected across
the Zener. Find the load voltage and the Zener current.
{"analysis": "dc",
 "components": [
   {"id": "V1", "type": "V", "nodes": ["in", "0"], "value": "12"},
   {"id": "R1", "type": "R", "nodes": ["in", "out"], "value": "470"},
   {"id": "D1", "type": "D", "nodes": ["0", "out"], "model": "zener", "params": {"VZ": 5.1}},
   {"id": "RL", "type": "R", "nodes": ["out", "0"], "value": "1k"}],
 "query": ["V(out)", "I(D1)"], "frequency": null, "assumptions": ["Zener reverse-connected as a regulator"],
 "missing_info": []}

Question: An NMOS transistor with k = 0.5 mA/V^2 and Vt = 1 V is biased from VDD = 10 V with RD = 2 kOhm,
RS = 1 kOhm and a gate divider R1 = 1 MOhm, R2 = 1 MOhm. Find the drain current and VDS.
{"analysis": "dc",
 "components": [
   {"id": "VDD", "type": "V", "nodes": ["vdd", "0"], "value": "10"},
   {"id": "R1", "type": "R", "nodes": ["vdd", "g"], "value": "1Meg"},
   {"id": "R2", "type": "R", "nodes": ["g", "0"], "value": "1Meg"},
   {"id": "RD", "type": "R", "nodes": ["vdd", "d"], "value": "2k"},
   {"id": "RS", "type": "R", "nodes": ["s", "0"], "value": "1k"},
   {"id": "M1", "type": "M", "nodes": ["d", "g", "s"], "model": "NMOS",
    "params": {"k": 0.5e-3, "Vt": 1}}],
 "query": ["ID(M1)", "VDS(M1)", "bias(M1)"], "frequency": null, "assumptions": [], "missing_info": []}

Question: A voltage-divider biased NPN transistor has VCC = 12 V, R1 = 47 kOhm, R2 = 10 kOhm, RC = 2.2 kOhm,
RE = 1 kOhm and beta = 100. Find the Q-point.
{"analysis": "dc",
 "components": [
   {"id": "VCC", "type": "V", "nodes": ["vcc", "0"], "value": "12"},
   {"id": "R1", "type": "R", "nodes": ["vcc", "b"], "value": "47k"},
   {"id": "R2", "type": "R", "nodes": ["b", "0"], "value": "10k"},
   {"id": "RC", "type": "R", "nodes": ["vcc", "c"], "value": "2.2k"},
   {"id": "RE", "type": "R", "nodes": ["e", "0"], "value": "1k"},
   {"id": "Q1", "type": "Q", "nodes": ["c", "b", "e"], "model": "NPN", "params": {"beta": 100}}],
 "query": ["bias(Q1)", "IC(Q1)", "VCE(Q1)"], "frequency": null, "assumptions": [], "missing_info": []}

Question: A bridge rectifier is fed from a 12 V rms, 50 Hz secondary and drives a 1 kOhm load. Using silicon
diodes, find the peak output voltage, the average output voltage and the PIV of each diode.
{"analysis": "transient",
 "components": [
   {"id": "V1", "type": "V", "nodes": ["a", "b"], "params": {"Vrms": 12}},
   {"id": "D1", "type": "D", "nodes": ["a", "out"]},
   {"id": "D2", "type": "D", "nodes": ["b", "out"]},
   {"id": "D3", "type": "D", "nodes": ["0", "a"]},
   {"id": "D4", "type": "D", "nodes": ["0", "b"]},
   {"id": "RL", "type": "R", "nodes": ["out", "0"], "value": "1k"}],
 "query": ["Vmax(out)", "Vavg(out)", "PIV(D1)"], "frequency": "50", "assumptions": [], "missing_info": []}

Question: An inverting Schmitt trigger uses an op-amp with +/-12 V supplies, R1 = 10 kOhm from the
non-inverting input to ground and R2 = 100 kOhm from the output back to the non-inverting input. The input
signal goes to the inverting input. Find the upper and lower trip points and the hysteresis.
(The feedback goes to in+, so this is positive feedback; the input source is a DC placeholder at 0 V.)
{"analysis": "dc",
 "components": [
   {"id": "V1", "type": "V", "nodes": ["in", "0"], "value": "0"},
   {"id": "R1", "type": "R", "nodes": ["p", "0"], "value": "10k"},
   {"id": "R2", "type": "R", "nodes": ["out", "p"], "value": "100k"},
   {"id": "U1", "type": "OPAMP", "nodes": ["out", "p", "in"], "model": "ideal"}],
 "query": ["VTH", "VTL", "hysteresis"], "frequency": null,
 "assumptions": ["supply +/-12 V"], "missing_info": []}

Question: A Wien-bridge oscillator uses R = 10 kOhm and C = 10 nF in both arms of the bridge, with
R1 = 10 kOhm from the inverting input to ground and R2 = 20 kOhm of feedback. Find the frequency of
oscillation.
(No source: an oscillator has none. The RC network feeds the + input; R1 and R2 set the gain.)
{"analysis": "ac",
 "components": [
   {"id": "R1", "type": "R", "nodes": ["n", "0"], "value": "10k"},
   {"id": "R2", "type": "R", "nodes": ["n", "out"], "value": "20k"},
   {"id": "Rs", "type": "R", "nodes": ["out", "x"], "value": "10k"},
   {"id": "Cs", "type": "C", "nodes": ["x", "p"], "value": "10n"},
   {"id": "Rp", "type": "R", "nodes": ["p", "0"], "value": "10k"},
   {"id": "Cp", "type": "C", "nodes": ["p", "0"], "value": "10n"},
   {"id": "U1", "type": "OPAMP", "nodes": ["out", "p", "n"], "model": "ideal"}],
 "query": ["fosc"], "frequency": null, "assumptions": [], "missing_info": []}

Question: An RC phase-shift oscillator uses three identical RC sections with R = 10 kOhm and C = 10 nF around
an inverting op-amp. Find the frequency of oscillation and the feedback resistor needed to sustain it.
(Three COMPLETE sections: three capacitors and three resistors. The third resistor R3 ends at the − input,
which is the virtual ground; no resistor is added from the − input to ground. Rf is what the question asks
for, so its value stays the symbol "Rf".)
{"analysis": "ac",
 "components": [
   {"id": "C1", "type": "C", "nodes": ["out", "a"], "value": "10n"},
   {"id": "R1", "type": "R", "nodes": ["a", "0"], "value": "10k"},
   {"id": "C2", "type": "C", "nodes": ["a", "b"], "value": "10n"},
   {"id": "R2", "type": "R", "nodes": ["b", "0"], "value": "10k"},
   {"id": "C3", "type": "C", "nodes": ["b", "c"], "value": "10n"},
   {"id": "R3", "type": "R", "nodes": ["c", "n"], "value": "10k"},
   {"id": "Rf", "type": "R", "nodes": ["n", "out"], "value": "Rf"},
   {"id": "U1", "type": "OPAMP", "nodes": ["out", "0", "n"], "model": "ideal"}],
 "query": ["fosc", "gain_condition"], "frequency": null, "assumptions": [], "missing_info": []}

Question: A Colpitts oscillator uses L = 1 mH with C1 = 10 nF at the base and C2 = 100 nF at the collector,
a collector load of 10 kOhm, and a BJT with gm = 1 mS. Find the frequency of oscillation.
(The AC equivalent circuit: no supply and no coupling capacitors, the emitter at ground, gm in params.)
{"analysis": "ac",
 "components": [
   {"id": "C1", "type": "C", "nodes": ["b", "0"], "value": "10n"},
   {"id": "C2", "type": "C", "nodes": ["c", "0"], "value": "100n"},
   {"id": "L1", "type": "L", "nodes": ["c", "b"], "value": "1m"},
   {"id": "RL", "type": "R", "nodes": ["c", "0"], "value": "10k"},
   {"id": "Q1", "type": "Q", "nodes": ["c", "b", "0"], "model": "NPN", "params": {"gm": 0.001}}],
 "query": ["fosc"], "frequency": null, "assumptions": [], "missing_info": []}

Question: An op-amp astable multivibrator runs from +/-12 V supplies with R1 = 10 kOhm from the
non-inverting input to ground, R2 = 10 kOhm of positive feedback, a timing resistor R = 10 kOhm and a
timing capacitor C = 100 nF. Find the frequency of the square wave.
(The capacitor is on the − input, the divider on the + input; no source.)
{"analysis": "ac",
 "components": [
   {"id": "R1", "type": "R", "nodes": ["p", "0"], "value": "10k"},
   {"id": "R2", "type": "R", "nodes": ["out", "p"], "value": "10k"},
   {"id": "R", "type": "R", "nodes": ["out", "n"], "value": "10k"},
   {"id": "C1", "type": "C", "nodes": ["n", "0"], "value": "100n"},
   {"id": "U1", "type": "OPAMP", "nodes": ["out", "p", "n"], "model": "ideal"}],
 "query": ["fosc"], "frequency": null, "assumptions": ["supply +/-12 V"], "missing_info": []}
"""

TEXT_PARSER_USER = "Question:\n{question}\n\nReturn the JSON object only."

RETRY_FEEDBACK = (
    "Your JSON has these problems:\n{errors}\n\n"
    "Fix them and return the complete corrected JSON object only."
)

NOT_JSON_FEEDBACK = "Your reply was not a valid JSON object. Return the JSON object only, no other text."


# ======================================================================
# Image parser (Phase 5): a photo or screenshot of a schematic -> the same netlist JSON
# ======================================================================
IMAGE_PARSER_SYSTEM = TEXT_PARSER_SYSTEM + r"""

READING A SCHEMATIC FROM AN IMAGE
Everything above still applies: the JSON format, the templates, the query forms, the rule that you describe
only what is drawn. What follows is about getting the circuit OFF THE PICTURE correctly. Two extra keys go
in the JSON when the input is an image:
  "question_text":  the printed question, copied from the image word for word (""  if the image has none)
  "reading_notes":  short notes on anything you could not read with confidence, one string each -- an
                    unreadable value, a wire you are unsure joins, a symbol you had to guess. An empty list
                    means you are confident about every component, node and value in the drawing.

NODES ARE THE HARD PART, so do this first and do it carefully.
- A node is a whole piece of wire. Follow every line: all the wire segments connected together -- round
  corners, through junction dots, along a long rail -- are ONE node with ONE name. Two wires that merely
  CROSS with no dot are NOT connected.
- Name a node after the label printed on it (Vcc, Vin, Vo, A, B, node 1 -> "vcc", "in", "out", "a", "b",
  "n1"). Where nothing is printed, use "in", "out", "b", "c", "e", "n1", "n2"...
- Ground is "0": the ground symbol (three shrinking horizontal bars, or a triangle, or the chassis hatch),
  and every wire that reaches one, however far away it is drawn. A circuit can have several ground symbols:
  they are all the SAME node "0".
- The two ends of a component are on DIFFERENT nodes unless the drawing shorts them. Before you finish,
  check each component: its two nodes must be the two wires its leads actually touch.
- A supply drawn as an arrow or a bare label ("+12 V", "Vcc = 15 V") at the top of the figure is still a
  source to ground: write it as a V component from that node to "0", unless the rules above say it belongs
  in "assumptions" instead (op-amp rails, for example).

READING SYMBOLS
- resistor: zigzag, or a plain rectangle (IEC). capacitor: two parallel plates (one curved = polarised).
  inductor: a coil or a series of humps. diode: triangle with a bar; a bar bent at both ends = Zener;
  two arrows pointing away = LED; the bar is the CATHODE, and nodes go [anode, cathode].
- BJT: a circle (or not) with the emitter arrow. Arrow pointing OUT of the base = NPN, pointing IN = PNP.
  nodes [collector, base, emitter].
- MOSFET: gate separated by a gap. Arrow on the bulk pointing IN towards the channel = NMOS, pointing OUT
  = PMOS. An n-channel source usually sits at the lower-potential side. nodes [drain, gate, source].
- op-amp: a triangle. The + and − marks tell you which input is which -- read them off the drawing and do
  NOT assume the + is on top, because op-amps are often drawn flipped. nodes [out, in+, in-].
- A transformer with a dot in the middle of its secondary is a centre tap; follow the rectifier template.

READING VALUES
- Copy the value exactly as printed, with its unit prefix: "2k2" means 2.2 kOhm, "4R7" means 4.7 Ohm,
  "R47" means 0.47 Ohm, "100n" is 100 nF, "1M" is 1 MOhm for a resistor and 1 MHz for a frequency.
  Greek mu, "u" and "µ" all mean micro.
- A value you cannot read is NOT a guess: put the component in with a symbol as its value ("R2") and say so
  in "reading_notes". Do not invent a number, and do not leave the component out.
- Values written beside a component belong to the component they touch. When a label could belong to either
  of two parts, say which you chose in "reading_notes".

WHAT THE QUESTION IS
- If the image contains the question, copy it into "question_text" and build "query" from it as usual.
- If the image is only a circuit with no question, leave "question_text" as "" and put in "query" the
  quantities the figure itself points at: a labelled node voltage ("Vo", "Vout" -> V(out)), an arrow for a
  current, a "?" beside a component. If nothing is marked at all, use the node voltages of the output node
  and say in "missing_info" that the question was not in the image.
"""

IMAGE_PARSER_USER = (
    "This image shows an analog circuit, and may also show the question printed next to it.\n"
    "Read the schematic and return the JSON object only.\n"
    "Trace every wire before you write the nodes, and put anything you could not read with confidence in "
    "\"reading_notes\"."
)

IMAGE_PARSER_USER_WITH_TEXT = (
    "This image shows an analog circuit. The question about it is:\n{question}\n\n"
    "Read the schematic and return the JSON object only. The question text above is authoritative: build "
    "\"query\" from it. Trace every wire before you write the nodes, and put anything you could not read "
    "with confidence in \"reading_notes\"."
)