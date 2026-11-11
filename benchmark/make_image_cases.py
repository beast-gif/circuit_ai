"""
Draw the Phase 5 image benchmark (Phase 5, step 2).

    python benchmark/make_image_cases.py            # write benchmark/images/*.png and their .json
    python benchmark/make_image_cases.py --photos   # also write "phone photo" versions of each

Nothing here imports the project: these are just pictures, drawn with matplotlib, plus a .json beside each
saying what the answer must be. That matters -- a benchmark drawn by the thing being tested would prove
nothing. Every expected value was worked by hand (see "work" in each case).

The clean drawings are the easy half. --photos then degrades each one the way a phone camera does -- a few
degrees of rotation, perspective, uneven lighting, blur and JPEG noise -- which is what a student actually
uploads. Reading those is a different problem from reading a crisp screenshot, and the score is reported
per category so the two never get averaged into one meaningless number.

WHAT EACH CASE IS FOR
    divider      the baseline: a junction dot, a ground symbol, two resistors
    diode        symbol orientation: the cathode bar decides whether current flows at all
    inv_amp      the op-amp is drawn FLIPPED (+ on the bottom): reading it wrongly gives +11 V, not -10 V
    bjt_bias     the emitter arrow decides NPN vs PNP; the supply is a bare "+12 V" label, not a symbol
    pnp_bias     the same circuit mirrored, with the arrow pointing IN: it must come out as a PNP
    crossing     two wires that CROSS with no dot are not connected -- the classic tracing trap
    shorthand    values written as 2k2 and 100n rather than 2.2 kOhm and 0.1 uF
    iec_boxes    resistors drawn as plain rectangles (IEC style) instead of zigzags
    many_grounds three separate ground symbols that are all the same node
    zener        a Zener's bent bar, and a reverse-connected diode that is meant to be in breakdown
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:                                     # pragma: no cover
    sys.exit("matplotlib is needed to draw the cases:  pip install matplotlib")

OUT = Path(__file__).resolve().parent / "images"
LW = 2.0
OHM, MU, MINUS, BETA = "Ω", "µ", "−", "β"


# ----------------------------------------------------------------------
# drawing primitives
# ----------------------------------------------------------------------
def fig(w=6.6, h=4.4):
    f, ax = plt.subplots(figsize=(w, h), dpi=170)
    ax.set_axis_off()
    ax.set_aspect("equal")
    return f, ax


def wire(ax, *pts):
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color="k", lw=LW,
            solid_capstyle="round", zorder=1)


def dot(ax, p):
    ax.plot([p[0]], [p[1]], "o", color="k", ms=5, zorder=3)


def _label(ax, mid, perp, side, label, gap=0.34, fs=12.5):
    s = 1 if side == "right" else -1
    pos = mid + perp * gap * s
    vertical = abs(perp[0]) > abs(perp[1])
    ha = ("left" if (perp[0] * s) > 0 else "right") if vertical else "center"
    va = "center" if vertical else ("bottom" if (perp[1] * s) > 0 else "top")
    ax.text(pos[0], pos[1], label, ha=ha, va=va, fontsize=fs, zorder=4)


def _axis(p, q):
    p, q = np.array(p, float), np.array(q, float)
    d = q - p
    L = np.linalg.norm(d)
    u = d / L
    return p, q, L, u, np.array([-u[1], u[0]])


def res(ax, p, q, label, side="right", n=6, amp=0.16):
    p, q, L, u, perp = _axis(p, q)
    a, b = p + u * (L / 2 - 0.45), p + u * (L / 2 + 0.45)
    wire(ax, p, a)
    pts = [a] + [a + u * (np.linalg.norm(b - a) / n) * (i + 0.5)
                 + perp * (amp if i % 2 == 0 else -amp) for i in range(n)] + [b]
    wire(ax, *pts)
    wire(ax, b, q)
    _label(ax, (a + b) / 2, perp, side, label)


def res_box(ax, p, q, label, side="right"):
    """IEC style: a plain rectangle instead of a zigzag."""
    p, q, L, u, perp = _axis(p, q)
    a, b = p + u * (L / 2 - 0.45), p + u * (L / 2 + 0.45)
    wire(ax, p, a)
    wire(ax, b, q)
    corners = [a + perp * 0.22, b + perp * 0.22, b - perp * 0.22, a - perp * 0.22]
    ax.add_patch(plt.Polygon(corners, closed=True, facecolor="white", edgecolor="k", lw=LW, zorder=2))
    _label(ax, (a + b) / 2, perp, side, label, gap=0.4)


def cap(ax, p, q, label, side="right", gap=0.14):
    p, q, L, u, perp = _axis(p, q)
    a, b = p + u * (L / 2 - gap), p + u * (L / 2 + gap)
    wire(ax, p, a)
    wire(ax, b, q)
    for c in (a, b):
        wire(ax, c - perp * 0.33, c + perp * 0.33)
    _label(ax, (a + b) / 2, perp, side, label, gap=0.55)


def diode(ax, p, q, label, side="right", zener=False):
    p, q, L, u, perp = _axis(p, q)
    a, b = p + u * (L / 2 - 0.22), p + u * (L / 2 + 0.22)
    wire(ax, p, a)
    wire(ax, b, q)
    ax.add_patch(plt.Polygon([a + perp * 0.28, a - perp * 0.28, b], closed=True,
                             facecolor="none", edgecolor="k", lw=LW))
    if zener:                                   # the cathode bar is bent at both ends
        wire(ax, b + perp * 0.28 + u * 0.16, b + perp * 0.28, b - perp * 0.28, b - perp * 0.28 - u * 0.16)
    else:
        wire(ax, b - perp * 0.28, b + perp * 0.28)
    _label(ax, (a + b) / 2, perp, side, label, gap=0.5)


def source(ax, p, q, label, plus_at="end", side="left"):
    p, q, L, u, perp = _axis(p, q)
    c = (p + q) / 2
    r = 0.42
    wire(ax, p, c - u * r)
    wire(ax, c + u * r, q)
    ax.add_patch(plt.Circle(c, r, facecolor="white", edgecolor="k", lw=LW, zorder=2))
    hi = c - u * r * 0.5 if plus_at == "start" else c + u * r * 0.5
    lo = c + u * r * 0.5 if plus_at == "start" else c - u * r * 0.5
    ax.text(hi[0], hi[1], "+", ha="center", va="center", fontsize=15, zorder=3)
    ax.text(lo[0], lo[1], MINUS, ha="center", va="center", fontsize=17, zorder=3)
    _label(ax, c, perp, side, label, gap=0.75)


def ground(ax, p):
    x, y = p
    wire(ax, (x, y), (x, y - 0.3))
    for i, w in enumerate((0.34, 0.22, 0.1)):
        wire(ax, (x - w, y - 0.3 - i * 0.13), (x + w, y - 0.3 - i * 0.13))


def hop(ax, p, size=0.22):
    """A wire crossing another one WITHOUT connecting: drawn as a little hop."""
    x, y = p
    t = np.linspace(0, np.pi, 24)
    ax.plot(x + size * np.cos(t), y + size * np.sin(t), color="k", lw=LW, zorder=2)


def opamp(ax, c, flip=False, size=0.95):
    cx, cy = c
    ax.add_patch(plt.Polygon([[cx - size, cy + size], [cx - size, cy - size], [cx + size * 1.15, cy]],
                             closed=True, facecolor="white", edgecolor="k", lw=LW, zorder=2))
    top, bot = (cx - size, cy + size * 0.5), (cx - size, cy - size * 0.5)
    inp, inm = (top, bot) if not flip else (bot, top)
    ax.text(inp[0] + 0.2, inp[1], "+", ha="center", va="center", fontsize=15, zorder=3)
    ax.text(inm[0] + 0.2, inm[1], MINUS, ha="center", va="center", fontsize=17, zorder=3)
    return inp, inm, (cx + size * 1.15, cy)


def bjt(ax, c, label="Q1", pnp=False, emitter_up=False):
    """Returns (collector, base, emitter).

    The arrow always sits on the EMITTER lead, and its direction is the whole point: pointing AWAY from the
    base is an NPN, pointing INTO the base is a PNP. A PNP is normally drawn with its emitter at the top
    (emitter_up), so the arrow has to move to the upper lead with it -- drawing a PNP with the arrow still
    on the lower lead would say the emitter is at the bottom, which contradicts the rest of the figure.
    """
    cx, cy = c
    bx = cx - 0.55
    wire(ax, (bx - 0.55, cy), (bx, cy))
    wire(ax, (bx, cy - 0.55), (bx, cy + 0.55))
    wire(ax, (bx, cy + 0.28), (cx + 0.35, cy + 0.8), (cx + 0.35, cy + 1.25))
    wire(ax, (bx, cy - 0.28), (cx + 0.35, cy - 0.8), (cx + 0.35, cy - 1.25))

    up = 1 if emitter_up else -1                      # which diagonal carries the emitter arrow
    start = np.array([bx, cy + 0.28 * up])
    along = np.array([cx + 0.35 - bx, 0.52 * up])
    along = along / np.linalg.norm(along)             # base -> emitter
    head = along if not pnp else -along               # NPN points away from the base, PNP into it
    tip = start + along * (0.52 if not pnp else 0.20)
    perp = np.array([-head[1], head[0]])
    ax.add_patch(plt.Polygon([tip, tip - head * 0.26 + perp * 0.13, tip - head * 0.26 - perp * 0.13],
                             closed=True, facecolor="k", edgecolor="k", zorder=3))
    ax.text(cx + 0.8, cy + 0.15, label, fontsize=12.5)
    top, bot = (cx + 0.35, cy + 1.25), (cx + 0.35, cy - 1.25)
    return (bot, (bx - 0.55, cy), top) if emitter_up else (top, (bx - 0.55, cy), bot)


def title(ax, text):
    ax.text(0.5, -0.04, text, transform=ax.transAxes, ha="center", va="top", fontsize=13)


def save(f, ax, name, meta, pad=0.4):
    OUT.mkdir(parents=True, exist_ok=True)
    ax.relim()
    ax.autoscale_view()
    f.savefig(OUT / name, bbox_inches="tight", pad_inches=pad, facecolor="white")
    plt.close(f)
    (OUT / (name + ".json")).write_text(json.dumps(meta, indent=1))
    print("wrote", name)


# ----------------------------------------------------------------------
# the cases
# ----------------------------------------------------------------------
def case_divider():
    f, ax = fig(6.4, 4.0)
    source(ax, (0, 0), (0, 3.2), "12 V")
    wire(ax, (0, 3.2), (3.4, 3.2))
    res(ax, (3.4, 3.2), (3.4, 1.8), f"R1 = 2 k{OHM}")
    res(ax, (3.4, 1.8), (3.4, 0.4), f"R2 = 4 k{OHM}")
    wire(ax, (3.4, 0.4), (3.4, 0), (0, 0))
    ground(ax, (1.7, 0))
    dot(ax, (3.4, 1.8))
    ax.text(3.75, 1.8, "A", fontsize=13, va="center")
    title(ax, "Find the voltage across R2.")
    save(f, ax, "divider.png", {"category": "clean", "question": "",
                                "expect": [{"kind": "V", "value": 8.0}],
                                "work": "12 V across 2k + 4k: V(A) = 12*4/6 = 8 V"})


def case_diode():
    f, ax = fig(6.8, 3.6)
    source(ax, (0, 0), (0, 2.8), "10 V")
    wire(ax, (0, 2.8), (2.2, 2.8))
    res(ax, (2.2, 2.8), (4.4, 2.8), f"1 k{OHM}")
    wire(ax, (4.4, 2.8), (5.2, 2.8))
    diode(ax, (5.2, 2.8), (5.2, 0.6), "D1 (Si)")
    wire(ax, (5.2, 0.6), (5.2, 0), (0, 0))
    ground(ax, (2.6, 0))
    title(ax, "The diode is silicon. Find the diode current.")
    save(f, ax, "diode.png", {"category": "clean", "question": "",
                              "expect": [{"kind": "I", "value": 0.0093, "abs": True}],
                              "work": "I = (10 - 0.7)/1k = 9.3 mA"})


def case_inv_amp():
    f, ax = fig(7.6, 4.6)
    ip, im, out = opamp(ax, (4.2, 1.9), flip=True)
    source(ax, (0.4, 0.3), (0.4, im[1]), "v_in = 1 V")
    ground(ax, (0.4, 0.3))
    wire(ax, (0.4, im[1]), (1.3, im[1]))
    res(ax, (1.3, im[1]), (2.7, im[1]), f"R_in = 1 k{OHM}")
    wire(ax, (2.7, im[1]), im)
    dot(ax, (3.0, im[1]))
    wire(ax, (3.0, im[1]), (3.0, im[1] + 1.6))
    res(ax, (3.0, im[1] + 1.6), (6.1, im[1] + 1.6), f"R_f = 10 k{OHM}")
    wire(ax, (6.1, im[1] + 1.6), (6.1, out[1]))
    wire(ax, out, (6.9, out[1]))
    dot(ax, (6.1, out[1]))
    ax.text(7.05, out[1], "v_o", fontsize=13, va="center")
    wire(ax, ip, (ip[0] - 0.8, ip[1]))
    ground(ax, (ip[0] - 0.8, ip[1]))
    title(ax, "The op-amp is ideal. Find the output voltage v_o.")
    save(f, ax, "inv_amp.png",
         {"category": "clean", "question": "", "expect": [{"kind": "V", "value": -10.0}],
          "work": "inverting amp, Av = -Rf/Rin = -10 so vo = -10 V. The op-amp is drawn FLIPPED, with + on "
                  "the bottom: reading the inputs the other way gives +11 V"})


def case_bjt_bias():
    f, ax = fig(6.2, 5.6)
    col, base, emit = bjt(ax, (3.2, 2.9))
    wire(ax, (1.0, 5.6), (col[0], 5.6))
    ax.text(0.85, 5.6, "+12 V", fontsize=13, ha="right", va="center")
    res(ax, (col[0], 5.6), col, f"R_C = 2.2 k{OHM}")
    res(ax, (1.0, 5.6), (1.0, 3.6), f"R1 = 47 k{OHM}", side="left")
    wire(ax, (1.0, 3.6), (1.0, base[1]))
    dot(ax, (1.0, base[1]))
    wire(ax, (1.0, base[1]), base)
    res(ax, (1.0, base[1]), (1.0, 0.8), f"R2 = 10 k{OHM}", side="left")
    wire(ax, (1.0, 0.8), (1.0, 0.3))
    res(ax, emit, (emit[0], 0.8), f"R_E = 1 k{OHM}")
    wire(ax, (emit[0], 0.8), (emit[0], 0.3), (1.0, 0.3))
    ground(ax, (2.1, 0.3))
    title(ax, f"The transistor is silicon with {BETA} = 100. Find IC and VCE.")
    save(f, ax, "bjt_bias.png",
         {"category": "clean", "question": "",
          "expect": [{"kind": "IC", "value": 0.00128633, "tol": 0.02},
                     {"kind": "VCE", "value": 7.87087, "tol": 0.02}],
          "work": "VBB = 2.105 V, RBB = 8.246k, IB = 12.86 uA, IC = 1.286 mA, VCE = 7.871 V"})


def case_pnp_bias():
    """The mirror of bjt_bias: a PNP with its emitter at the top, so the arrow points INTO the base."""
    f, ax = fig(6.2, 5.6)
    col, base, emit = bjt(ax, (3.2, 2.9), pnp=True, emitter_up=True)
    wire(ax, (1.0, 5.6), (emit[0], 5.6))
    ax.text(0.85, 5.6, "+12 V", fontsize=13, ha="right", va="center")
    res(ax, (emit[0], 5.6), emit, f"R_E = 1 k{OHM}")
    res(ax, (1.0, 5.6), (1.0, 3.6), f"R1 = 10 k{OHM}", side="left")
    wire(ax, (1.0, 3.6), (1.0, base[1]))
    dot(ax, (1.0, base[1]))
    wire(ax, (1.0, base[1]), base)
    res(ax, (1.0, base[1]), (1.0, 0.8), f"R2 = 47 k{OHM}", side="left")
    wire(ax, (1.0, 0.8), (1.0, 0.3))
    res(ax, col, (col[0], 0.8), f"R_C = 2.2 k{OHM}")
    wire(ax, (col[0], 0.8), (col[0], 0.3), (1.0, 0.3))
    ground(ax, (2.1, 0.3))
    title(ax, f"The transistor is a silicon PNP with {BETA} = 100. Find IC and VEC.")
    save(f, ax, "pnp_bias.png",
         {"category": "clean", "question": "",
          "expect": [{"kind": "IC", "value": 0.00128633, "tol": 0.03},
                     {"kind": "VEC", "value": 7.87087, "tol": 0.03}],
          "work": "the mirror of bjt_bias: VBB = 9.895 V, RBB = 8.246k, IB = 12.86 uA, IC = 1.286 mA, "
                  "VEC = 7.871 V. The emitter is at the TOP and its arrow points IN: a PNP"})


def case_crossing():
    """Two independent dividers from one supply. The feed to the second one CROSSES the first branch's
    wire with a hop, so the two are not joined. Read the hop as a junction and node x jumps to 10 V."""
    f, ax = fig(7.0, 4.6)
    source(ax, (0, 0.2), (0, 4.5), "10 V", side="right")
    wire(ax, (0, 4.5), (2.5, 4.5))
    dot(ax, (1.2, 4.5))
    res(ax, (2.5, 4.5), (2.5, 3.0), f"R1 = 1 k{OHM}")
    dot(ax, (2.5, 3.0))
    ax.text(2.72, 3.0, "x", fontsize=13, va="center")
    wire(ax, (2.5, 3.0), (2.5, 2.6))
    res(ax, (2.5, 2.6), (2.5, 0.6), f"R2 = 3 k{OHM}", side="left")
    wire(ax, (2.5, 0.6), (2.5, 0.2))
    # the feed to the second divider, crossing branch 1 without touching it
    wire(ax, (1.2, 4.5), (1.2, 2.4), (2.32, 2.4))
    hop(ax, (2.5, 2.4))
    wire(ax, (2.68, 2.4), (5.4, 2.4))
    res(ax, (5.4, 2.4), (5.4, 1.4), f"R3 = 2 k{OHM}")
    dot(ax, (5.4, 1.4))
    ax.text(5.62, 1.4, "y", fontsize=13, va="center")
    res(ax, (5.4, 1.4), (5.4, 0.6), f"R4 = 2 k{OHM}", side="left")
    wire(ax, (5.4, 0.6), (5.4, 0.2), (0, 0.2))
    ground(ax, (3.9, 0.2))
    title(ax, "The hop means the wires are not joined. Find the voltages at x and y.")
    save(f, ax, "crossing.png",
         {"category": "clean", "question": "",
          "expect": [{"kind": "V", "value": 7.5, "tol": 0.01}, {"kind": "V", "value": 5.0, "tol": 0.01}],
          "work": "two separate dividers from the same 10 V: x = 10*3/4 = 7.5 V, y = 10*2/4 = 5 V. If the "
                  "hop is read as a junction, node x is tied to the 10 V feed and becomes 10 V"})


def case_shorthand():
    f, ax = fig(6.8, 3.6)
    source(ax, (0, 0), (0, 2.8), "1 V ac")
    wire(ax, (0, 2.8), (1.6, 2.8))
    res(ax, (1.6, 2.8), (4.0, 2.8), "2k2")
    wire(ax, (4.0, 2.8), (4.8, 2.8))
    cap(ax, (4.8, 2.8), (4.8, 0.6), "100n")
    wire(ax, (4.8, 0.6), (4.8, 0), (0, 0))
    ground(ax, (2.4, 0))
    dot(ax, (4.8, 2.8))
    ax.text(5.15, 2.8, "out", fontsize=13, va="center")
    title(ax, "Find the cut-off frequency of this filter.")
    save(f, ax, "shorthand.png",
         {"category": "clean", "question": "",
          "expect": [{"kind": "fc", "value": 723.4315595086151, "tol": 0.01}],
          "work": "2k2 = 2.2 kOhm and 100n = 100 nF: fc = 1/(2*pi*R*C) = 723.43 Hz"})


def case_iec_boxes():
    f, ax = fig(6.4, 4.0)
    source(ax, (0, 0), (0, 3.2), "12 V")
    wire(ax, (0, 3.2), (3.4, 3.2))
    res_box(ax, (3.4, 3.2), (3.4, 1.8), f"R1 = 2 k{OHM}")
    res_box(ax, (3.4, 1.8), (3.4, 0.4), f"R2 = 4 k{OHM}")
    wire(ax, (3.4, 0.4), (3.4, 0), (0, 0))
    ground(ax, (1.7, 0))
    dot(ax, (3.4, 1.8))
    title(ax, "The resistors are drawn as boxes. Find the voltage across R2.")
    save(f, ax, "iec_boxes.png",
         {"category": "clean", "question": "", "expect": [{"kind": "V", "value": 8.0}],
          "work": "the same divider as divider.png, with IEC rectangle resistors: 8 V"})


def case_many_grounds():
    f, ax = fig(7.4, 4.6)
    ip, im, out = opamp(ax, (4.0, 2.2))
    source(ax, (0.4, 0.6), (0.4, ip[1]), "v_in = 1 V")
    ground(ax, (0.4, 0.6))                       # ground 1
    wire(ax, (0.4, ip[1]), ip)
    res(ax, (2.2, im[1]), im, f"R1 = 10 k{OHM}", side="right")
    wire(ax, (2.2, im[1]), (1.6, im[1]))
    ground(ax, (1.6, im[1]))                     # ground 2: the same node as ground 1
    dot(ax, (2.9, im[1]))
    wire(ax, (2.9, im[1]), (2.9, im[1] - 1.5))
    res(ax, (2.9, im[1] - 1.5), (5.8, im[1] - 1.5), f"R_f = 47 k{OHM}", side="left")
    wire(ax, (5.8, im[1] - 1.5), (5.8, out[1]))
    wire(ax, out, (6.6, out[1]))
    dot(ax, (5.8, out[1]))
    ax.text(6.75, out[1], "v_o", fontsize=13, va="center")
    ground(ax, (6.9, 0.6))                       # ground 3: drawn but unused, still the same node
    wire(ax, (6.9, 0.6), (6.9, 1.0))
    ax.text(6.9, 1.15, "0 V", fontsize=11, ha="center")
    title(ax, "Three ground symbols, one node. Find the voltage gain v_o/v_in.")
    save(f, ax, "many_grounds.png",
         {"category": "clean", "question": "",
          "expect": [{"kind": "gain", "value": 5.7, "tol": 0.01}],
          "work": "non-inverting amp: Av = 1 + Rf/R1 = 1 + 47/10 = 5.7"})


def case_zener():
    f, ax = fig(6.6, 4.0)
    source(ax, (0, 0), (0, 3.0), "15 V")
    wire(ax, (0, 3.0), (1.4, 3.0))
    res(ax, (1.4, 3.0), (3.6, 3.0), f"Rs = 1 k{OHM}")
    wire(ax, (3.6, 3.0), (4.2, 3.0))
    dot(ax, (4.2, 3.0))
    diode(ax, (4.2, 0.6), (4.2, 3.0), "6.2 V", zener=True)   # cathode UP: reverse biased, in breakdown
    wire(ax, (4.2, 0.6), (4.2, 0))
    res(ax, (5.8, 3.0), (5.8, 0.6), f"RL = 2.2 k{OHM}")
    wire(ax, (4.2, 3.0), (5.8, 3.0))
    wire(ax, (5.8, 0.6), (5.8, 0), (0, 0))
    ground(ax, (2.9, 0))
    ax.text(4.55, 3.3, "out", fontsize=12.5)
    title(ax, "Find the load voltage and the Zener current.")
    save(f, ax, "zener.png",
         {"category": "clean", "question": "",
          "expect": [{"kind": "V", "value": 6.2, "tol": 0.01},
                     {"kind": "I", "value": 0.00598182, "abs": True, "tol": 0.02}],
          "work": "V = 6.2 V, I(Rs) = 8.8 mA, I(RL) = 2.818 mA, IZ = 5.982 mA"})


CASES = [case_divider, case_diode, case_inv_amp, case_bjt_bias, case_pnp_bias,
         case_crossing, case_shorthand, case_iec_boxes, case_many_grounds, case_zener]


# ----------------------------------------------------------------------
# "phone photo" versions
# ----------------------------------------------------------------------
def photos() -> None:
    """Degrade each clean drawing the way a phone camera does. This is the half that actually matters:
    a student photographs a page at an angle, under a desk lamp, slightly out of focus."""
    try:
        from PIL import Image, ImageEnhance, ImageFilter
    except ImportError:
        print("Pillow is needed for the photo versions:  pip install pillow")
        return
    rng = np.random.default_rng(7)
    for p in sorted(OUT.glob("*.png")):
        if p.name.startswith("photo_"):
            continue
        meta_path = Path(str(p) + ".json")
        im = Image.open(p).convert("RGB")
        im = im.rotate(float(rng.uniform(-3.5, 3.5)), resample=Image.BICUBIC,
                       expand=True, fillcolor=(255, 255, 255))
        w, h = im.size
        # uneven lighting: a soft diagonal gradient, like a lamp on one side
        grad = np.linspace(0.78, 1.0, w)[None, :, None] * np.linspace(1.0, 0.86, h)[:, None, None]
        arr = np.asarray(im, dtype=float) * grad
        arr += rng.normal(0, 4.0, arr.shape)                       # sensor noise
        im = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))
        im = im.filter(ImageFilter.GaussianBlur(0.7))              # not quite in focus
        im = ImageEnhance.Contrast(im).enhance(0.92)
        im = im.resize((int(w * 0.72), int(h * 0.72)), Image.LANCZOS)
        out = OUT / ("photo_" + p.name.replace(".png", ".jpg"))
        im.save(out, quality=72)                                   # phone JPEG artefacts
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta["category"] = "photo"
            Path(str(out) + ".json").write_text(json.dumps(meta, indent=1))
        print("wrote", out.name)


if __name__ == "__main__":
    for c in CASES:
        c()
    if "--photos" in sys.argv:
        photos()
    print(f"\n{len(CASES)} cases in {OUT}")