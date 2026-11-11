// Every drawing on the site lives here, as inline SVG so it stays sharp and themeable.
// The hero is the one animated moment: the schematic draws itself wire by wire, and the
// output lands on an oscilloscope whose trace keeps sweeping.

const RES_IN = "M150 200 L160 200 L167 186 L181 214 L195 186 L209 214 L223 186 L237 214 L244 200 L262 200";
const RES_FB = "M330 110 L340 110 L347 96 L361 124 L375 96 L389 124 L403 96 L417 124 L424 110 L442 110";

export function BrandMark({ size = 34 }) {
  // a resistor body with its colour bands: brown, black, orange = 10 k
  return (
    <svg width={size * 1.9} height={size} viewBox="0 0 76 40" aria-hidden="true" className="brandmark">
      <path d="M0 20 H14 M62 20 H76" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
      <rect x="14" y="8" width="48" height="24" rx="9" fill="#E8D5A8" />
      <rect x="22" y="8" width="5" height="24" fill="#7A4A21" />
      <rect x="32" y="8" width="5" height="24" fill="#16242A" />
      <rect x="42" y="8" width="5" height="24" fill="#FF8A2A" />
      <rect x="53" y="8" width="4" height="24" fill="#C9A227" />
    </svg>
  );
}

export function HeroSchematic() {
  return (
    <svg className="hero-art" viewBox="40 60 680 320" role="img"
         aria-label="An inverting amplifier circuit whose output is shown as a sine wave on an oscilloscope">
      {/* --- the circuit: each path draws in, staggered by its --d delay */}
      <g className="draw" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="90" cy="290" r="27" pathLength="1" style={{ "--d": "0s" }} />
        <path d="M90 263 V200 H150" pathLength="1" style={{ "--d": ".25s" }} />
        <path d={RES_IN} pathLength="1" style={{ "--d": ".5s" }} />
        <path d="M262 200 H340" pathLength="1" style={{ "--d": ".8s" }} />
        <path d="M340 150 V290 L462 220 Z" pathLength="1" style={{ "--d": "1s" }} />
        <path d="M340 240 H310 V318" pathLength="1" style={{ "--d": "1.3s" }} />
        <path d="M300 200 V110 H330" pathLength="1" style={{ "--d": "1.3s" }} />
        <path d={RES_FB} pathLength="1" style={{ "--d": "1.55s" }} />
        <path d="M442 110 H496 V220" pathLength="1" style={{ "--d": "1.85s" }} />
        <path d="M462 220 H548" pathLength="1" style={{ "--d": "2.05s" }} />
        <path d="M90 317 V344" pathLength="1" style={{ "--d": "1.3s" }} />
        {/* grounds */}
        <path d="M72 344 H108 M79 352 H101 M86 360 H94" pathLength="1" style={{ "--d": "1.5s" }} />
        <path d="M292 318 H328 M299 326 H321 M306 334 H314" pathLength="1" style={{ "--d": "1.5s" }} />
      </g>
      <g className="fade" style={{ "--d": "2.1s" }} fill="currentColor">
        <circle cx="300" cy="200" r="5" />
        <circle cx="496" cy="220" r="5" />
        <text x="82" y="284" className="art-sign">+</text>
        <text x="82" y="306" className="art-sign">−</text>
        <text x="350" y="207" className="art-sign">−</text>
        <text x="350" y="247" className="art-sign">+</text>
        <text x="182" y="172" className="art-label">1 kΩ</text>
        <text x="362" y="82" className="art-label">10 kΩ</text>
        <text x="46" y="246" className="art-label">1 V</text>
      </g>

      {/* --- the oscilloscope the output lands on */}
      <g className="fade" style={{ "--d": "2.2s" }}>
        <rect x="548" y="128" width="160" height="184" rx="14" className="scope-body" />
        <rect x="560" y="140" width="136" height="136" rx="6" className="scope-glass" />
        <g className="scope-grid">
          <path d="M594 140 V276 M628 140 V276 M662 140 V276 M560 174 H696 M560 208 H696 M560 242 H696" />
        </g>
        <clipPath id="glass"><rect x="560" y="140" width="136" height="136" rx="6" /></clipPath>
        <g clipPath="url(#glass)">
          {/* two periods wider than the glass, slid left one period per loop */}
          <path className="trace"
                d="M560 208 q17 -86 34 0 t34 0 t34 0 t34 0 t34 0 t34 0 t34 0 t34 0" />
        </g>
        <circle cx="574" cy="294" r="6" className="scope-knob" />
        <circle cx="596" cy="294" r="6" className="scope-knob" />
        <text x="618" y="299" className="scope-read">−10 V</text>
      </g>
    </svg>
  );
}

// Small symbols for the "what it solves" strip. Each one is the real schematic symbol.
const SYMBOLS = {
  network: (
    <path d="M4 24 H16 L20 14 L28 34 L36 14 L44 34 L52 14 L56 24 H68" />
  ),
  opamp: (
    <>
      <path d="M20 6 V42 L54 24 Z" />
      <path d="M6 15 H20 M6 33 H20 M54 24 H68" />
    </>
  ),
  diode: (
    <>
      <path d="M4 24 H26 M46 24 H68" />
      <path d="M26 10 V38 L46 24 Z" />
      <path d="M46 10 V38" />
    </>
  ),
  transistor: (
    <>
      <path d="M6 24 H28 M28 10 V38" />
      <path d="M28 18 L50 8 V2 M28 30 L50 40 V46" />
      <path d="M50 40 L40 39.5 L44.5 31.5 Z" fill="currentColor" />
    </>
  ),
  oscillator: (
    <>
      <circle cx="36" cy="24" r="20" />
      <path d="M22 24 q7 -18 14 0 t14 0" />
    </>
  ),
};

export function Symbol({ name }) {
  return (
    <svg viewBox="0 0 72 48" width="72" height="48" aria-hidden="true" className="symbol"
         fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      {SYMBOLS[name]}
    </svg>
  );
}

export function Sweep({ label }) {
  // the busy state: a scope trace sweeping while the server reads or solves
  return (
    <div className="sweep" role="status" aria-live="polite">
      <svg viewBox="0 0 320 60" aria-hidden="true">
        <path className="sweep-grid" d="M0 30 H320 M80 0 V60 M160 0 V60 M240 0 V60" />
        <path className="sweep-trace" pathLength="1"
              d="M0 30 q20 -44 40 0 t40 0 t40 0 t40 0 t40 0 t40 0 t40 0 t40 0" />
      </svg>
      <p>{label}</p>
    </div>
  );
}
