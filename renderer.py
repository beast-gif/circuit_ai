"""
Netlist -> picture + plain-text description, for the confirmation step ("Is this your circuit?").

Most wrong answers come from a wrong PARSE, not wrong maths (Step 4 cannot see those), so the
student checks the circuit before it is solved.

Layout ("ladder", fits most textbook circuits):
  - non-ground nodes sit on a top rail (y = 0), left to right, in the order you reach them
    starting from the main source
  - elements from a node to ground hang down to a ground rail (y = GROUND_Y)
  - elements between two neighbouring nodes sit on the top rail
  - parallel elements and elements that skip over nodes go on lanes above the rail
Every element is labelled with its id and value, and sources say where their + terminal is,
so polarity never depends on the symbol alone.

Op-amps (Phase 2) are drawn as what an ideal op-amp is electrically: a controlled voltage source
from the output node to ground, labelled with its + and − input nodes. Its inputs carry no
current, so no wires are drawn to them (describe() lists every pin as well).

describe() always works (text only). render_svg() needs schemdraw and returns None if
drawing fails -- the pipeline never stops because of a picture.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from schemas import Component, Netlist, is_symbolic, parse_value
from solvers.base import fmt_si

GROUND = "0"
GROUND_Y = -3.5
NODE_GAP = 3.5          # horizontal space for one series element
SHUNT_GAP = 2.5         # horizontal space between parallel shunt elements at one node
LANE_STEP = 2.2         # vertical space between lanes above the rail

UNIT = {"R": "Ω", "C": "F", "L": "H", "V": "V", "I": "A"}
NAMES = {"R": "resistor", "C": "capacitor", "L": "inductor", "V": "voltage source", "I": "current source",
         "E": "VCVS (voltage-controlled voltage source)", "G": "VCCS (voltage-controlled current source)",
         "F": "CCCS (current-controlled current source)", "H": "CCVS (current-controlled voltage source)",
         "D": "diode", "Q": "BJT", "M": "MOSFET", "OPAMP": "op-amp"}


def _node(n: str) -> str:
    return "ground" if n == GROUND else n


def _val(raw: Optional[str], unit: str) -> str:
    if raw in (None, ""):
        return ""
    if is_symbolic(raw):
        return str(raw)
    try:
        return fmt_si(parse_value(raw), unit)
    except ValueError:
        return str(raw)


def _ctrl_v(c: Component) -> str:
    a, b = c.nodes[2], c.nodes[3]
    return f"V({a})" if b == GROUND else f"V({a},{b})"


def value_text(c: Component) -> str:
    t = c.type.value
    if t in ("R", "C", "L"):
        return _val(c.value, UNIT[t])
    if t in ("V", "I"):
        parts = []
        if c.value not in (None, ""):
            parts.append(_val(c.value, UNIT[t]) + (" DC" if c.ac not in (None, "") else ""))
        if c.ac not in (None, ""):
            ph = f" ∠{c.ac_phase:g}°" if c.ac_phase else ""
            parts.append(f"{_val(c.ac, UNIT[t])} AC{ph}")
        return " + ".join(parts) or "0"
    if t == "E":
        return f"{c.value}·{_ctrl_v(c)}"
    if t == "G":
        return f"{c.value}·{_ctrl_v(c)} (A)"
    if t == "F":
        return f"{c.value}·I({c.control})"
    if t == "H":
        return f"{c.value}·I({c.control}) (V)"
    return c.model or ""


def describe(nl: Netlist) -> str:
    """One readable line per element, e.g. 'R1: 1 kΩ resistor between in and out'."""
    if not nl.components:
        return "(no circuit: theory question)"
    lines = []
    for c in nl.components:
        t = c.type.value
        n = c.nodes
        if t == "V" or t == "E" or t == "H":
            where = f"+ at {_node(n[0])}, − at {_node(n[1])}"
        elif t in ("I", "G", "F"):
            where = f"current flows from {_node(n[0])} through the source into {_node(n[1])}"
        elif t == "D":
            where = f"anode {_node(n[0])}, cathode {_node(n[1])}"
        elif t == "Q":
            where = f"C={_node(n[0])}, B={_node(n[1])}, E={_node(n[2])}"
        elif t == "M":
            where = f"D={_node(n[0])}, G={_node(n[1])}, S={_node(n[2])}"
        elif t == "OPAMP":
            where = f"out={_node(n[0])}, +in={_node(n[1])}, −in={_node(n[2])}"
        else:
            where = f"between {_node(n[0])} and {_node(n[1])}"
        lines.append(f"{c.id}: {value_text(c)} {NAMES.get(t, t)}, {where}".replace("  ", " "))
    lines.append(f"Find: {', '.join(nl.query)}   ({nl.analysis.value} analysis"
                 + (f", f = {_val(nl.frequency, 'Hz')}" if nl.frequency else "") + ")")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
def _order_nodes(nl: Netlist) -> List[str]:
    nodes = [n for n in nl.nodes() if n != GROUND]
    adj: Dict[str, List[str]] = defaultdict(list)
    for c in nl.components:
        if c.type.value == "OPAMP":               # its inputs carry no current: not a wire in the layout
            continue
        a, b = c.nodes[0], c.nodes[1]
        if GROUND not in (a, b):
            adj[a].append(b)
            adj[b].append(a)
    src = next((c for c in nl.components if c.type.value in ("V", "I")), None)
    start = next((n for n in (src.nodes[:2] if src else []) if n != GROUND), nodes[0] if nodes else None)
    order: List[str] = []
    for root in ([start] if start else []) + nodes:
        if root in order:
            continue
        q = deque([root])
        while q:
            n = q.popleft()
            if n in order:
                continue
            order.append(n)
            q.extend(m for m in adj[n] if m not in order)
    return order


def layout(nl: Netlist):
    """-> (node x positions, bus end x, placements[(component, start_xy, end_xy)], lane wires, ground xs)."""
    order = _order_nodes(nl)
    shunts: Dict[str, List[Component]] = defaultdict(list)
    series: List[Component] = []
    for c in nl.components:
        # an op-amp is drawn as its equivalent dependent source from its output to ground
        a, b = (c.nodes[0], GROUND) if c.type.value == "OPAMP" else (c.nodes[0], c.nodes[1])
        if a == GROUND or b == GROUND:
            shunts[b if a == GROUND else a].append(c)
        else:
            series.append(c)

    xpos, bus_end, x = {}, {}, 0.0
    for n in order:
        xpos[n] = x
        bus_end[n] = x + SHUNT_GAP * (max(1, len(shunts[n])) - 1)
        x = bus_end[n] + NODE_GAP

    placements: List[Tuple[Component, Tuple[float, float], Tuple[float, float]]] = []
    wires: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    ground_xs: List[float] = []

    for n, comps in shunts.items():
        if n not in xpos:
            continue
        if len(comps) > 1:
            wires.append(((xpos[n], 0.0), (bus_end[n], 0.0)))
        for k, c in enumerate(comps):
            gx = xpos[n] + SHUNT_GAP * k
            top, bottom = (gx, 0.0), (gx, GROUND_Y)
            start, end = (top, bottom) if c.nodes[0] == n else (bottom, top)
            placements.append((c, start, end))
            ground_xs.append(gx)

    rank = {n: i for i, n in enumerate(order)}
    used_direct = set()
    lanes: List[List[Tuple[float, float]]] = []          # occupied x-intervals per lane
    for c in series:
        a, b = c.nodes[0], c.nodes[1]
        left, right = (a, b) if rank[a] < rank[b] else (b, a)
        if rank[right] - rank[left] == 1 and (left, right) not in used_direct:
            used_direct.add((left, right))
            p_left, p_right = (bus_end[left], 0.0), (xpos[right], 0.0)
        else:
            x0, x1 = xpos[left], xpos[right]
            lane = 0
            while lane < len(lanes) and any(not (x1 < s or x0 > e) for s, e in lanes[lane]):
                lane += 1
            if lane == len(lanes):
                lanes.append([])
            lanes[lane].append((x0 - 0.3, x1 + 0.3))
            y = LANE_STEP * (lane + 1)
            p_left, p_right = (x0, y), (x1, y)
            wires += [((x0, 0.0), (x0, y)), ((x1, 0.0), (x1, y))]
        start, end = (p_left, p_right) if a == left else (p_right, p_left)
        placements.append((c, start, end))
    return xpos, bus_end, placements, wires, ground_xs


def _add_background(svg: str, color: str) -> str:
    """Insert a filled rectangle behind everything, covering the whole viewBox.
    (schemdraw's SVG is transparent: black lines vanish in dark viewers.)"""
    import re
    m = re.search(r"<svg\b[^>]*>", svg)
    if not m:
        return svg
    vb = re.search(r'viewBox="\s*([-\d.eE]+)[\s,]+([-\d.eE]+)[\s,]+([-\d.eE]+)[\s,]+([-\d.eE]+)\s*"', m.group(0))
    if vb:
        x, y, w, h = vb.groups()
        rect = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}"/>'
    else:
        rect = f'<rect x="0" y="0" width="100%" height="100%" fill="{color}"/>'
    return svg[:m.end()] + rect + svg[m.end():]


def render_svg(nl: Netlist, background: Optional[str] = "white") -> Optional[str]:
    """SVG text of the schematic, or None if it cannot be drawn.
    background: fill colour behind the drawing ("white" by default, None = transparent)."""
    if not nl.components:
        return None
    try:
        import schemdraw
        import schemdraw.elements as elm
    except ImportError:
        return None
    if {c.type.value for c in nl.components} & {"D", "Q", "M"}:
        return None                                   # diode / transistor symbols come with Phase 3
    try:
        schemdraw.use("svg")
        xpos, _, placements, wires, ground_xs = layout(nl)
        d = schemdraw.Drawing(show=False)
        d.config(fontsize=11)
        symbol = {"R": elm.Resistor, "C": elm.Capacitor, "L": elm.Inductor2,
                  "I": elm.SourceI, "E": elm.SourceControlledV, "H": elm.SourceControlledV,
                  "G": elm.SourceControlledI, "F": elm.SourceControlledI}
        for c, start, end in placements:
            t = c.type.value
            if t == "OPAMP":
                # ideal op-amp = output voltage source to ground set by its inputs; the label names the pins
                _, inp, inm = c.nodes
                label = f"{c.id} op-amp  (+ in: {_node(inp)}, − in: {_node(inm)})"
                d.add(elm.SourceControlledV().endpoints(end, start).label(label))
                continue
            if t == "V":
                cls = elm.SourceSin if c.ac not in (None, "") and c.value in (None, "") else elm.SourceV
                # draw from - to + so the symbol's + is at the + node; the label states it anyway
                start, end = end, start
                label = f"{c.id} {value_text(c)} (+ at {_node(c.nodes[0])})"
            else:
                cls = symbol[t]
                label = f"{c.id} {value_text(c)}"
            d.add(cls().endpoints(start, end).label(label))
        for p, q in wires:
            d.add(elm.Line().endpoints(p, q))
        if ground_xs:
            gx0, gx1 = min(ground_xs), max(ground_xs)
            if gx1 > gx0:
                d.add(elm.Line().endpoints((gx0, GROUND_Y), (gx1, GROUND_Y)))
            d.add(elm.Ground().at((gx0, GROUND_Y)))
        for n, x in xpos.items():
            d.add(elm.Dot().at((x, 0.0)).label(n, loc="top"))
        data = d.get_imagedata("svg")
        svg = data.decode("utf-8") if isinstance(data, bytes) else str(data)
        return _add_background(svg, background) if background else svg
    except Exception:                                  # never let a drawing problem stop the pipeline
        return None