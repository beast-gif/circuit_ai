"""
Circuit-level checks on a Netlist. Pure Python, no LLM.

schemas.py checks each component on its own (node count, value format).
This file checks the circuit as a whole -- things that would make MNA
fail or give an answer for the wrong circuit:

  ERRORS (parser must fix, triggers a retry)
    - no ground node "0"
    - part of the circuit not connected to ground
    - loop made only of voltage sources          -> MNA matrix singular
    - node fed only by current sources           -> node voltage undefined
    - inductor directly across a voltage source in DC (short circuit)
    - node names that differ only by case ("out" vs "Out")
    - query refers to a node/component that does not exist, or has unknown form

  WARNINGS (shown to the user at the confirmation step)
    - node with a single connection (open terminal)
    - AC analysis with no AC magnitude on any source
    - small-signal analysis without a transistor, transient without C/L
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from schemas import AnalysisType, Netlist

# ----------------------------------------------------------------------
# Accepted query forms
# ----------------------------------------------------------------------
_Q_NODE_V = re.compile(r"^V\((\w+)\)$")                       # V(out)
_Q_DIFF_V = re.compile(r"^V\((\w+),(\w+)\)$")                 # V(a,b)
_Q_COMP = re.compile(r"^(I|P)\((\w+)\)$")                     # I(R1), P(R1)
_Q_PORT = re.compile(r"^(Z|Rth|Vth|Zth|Isc|Rn|In)\((\w+),(\w+)\)$")  # Rth(a,b)
_Q_TRANSFER = re.compile(r"^H\((\w+),(\w+)\)$")               # H(out,in) or H(out,V1)
_Q_BIAS = re.compile(r"^bias\((\w+)\)$")                      # bias(Q1)
_Q_DEV = re.compile(r"^(IC|IB|IE|VCE|VBE|VEC|VEB|region|gm|rpi|ro|re)\((\w+)\)$")   # IC(Q1), gm(Q1): BJT
_Q_FET = re.compile(r"^(ID|IS|IG|VGS|VDS|VSG|VSD|VOV)\((\w+)\)$")                  # ID(M1), VDS(M1): MOSFET
_Q_STATE = re.compile(r"^state\((\w+)\)$")                                        # state(U1): comparator
_Q_BARE = re.compile(r"^(Rin|Rout|gain|fc|f0|poles|zeros|tau|Qfactor|bandwidth|filter|VTH|VTL|hysteresis|"
                     r"trip|fosc|barkhausen|gain_condition|oscillates)$")
_Q_WAVE_V = re.compile(r"^(Vmax|Vmin|Vavg|Vrms|vtc|Vripple|Vpp|ripple)\((\w+)(?:,(\w+))?\)$")   # waveform of a node: Vavg(out)
_Q_WAVE_C = re.compile(r"^(Imax|Iavg|Irms|PIV|conduction)\((\w+)\)$")       # of a component: PIV(D1)

QUERY_HELP = ("V(node), V(a,b), I(comp), P(comp), Rth(a,b), Vth(a,b), Z(a,b), "
              "H(out,in), bias(Q1), Rin, Rout, gain, fc, f0, poles, zeros, tau, Qfactor, bandwidth, filter, "
              "IC(Q1), IB(Q1), IE(Q1), VCE(Q1), VBE(Q1), gm(Q1), rpi(Q1), re(Q1), ID(M1), VGS(M1), VDS(M1), VOV(M1), "
              "VTH, VTL, hysteresis, state(U1), fosc, barkhausen, gain_condition, oscillates, Vmax(n), Vmin(n), Vavg(n), Vrms(n), vtc(n), Vripple(n), ripple(n), Imax(X), Iavg(X), Irms(X), PIV(D1), conduction(D1)")

VOLTAGE_TYPES = {"V", "E", "H"}            # elements that force a voltage
CURRENT_TYPES = {"I", "G", "F"}            # elements that force a current
CONTROL_ONLY = {"E": (2, 3), "G": (2, 3)}  # sensing terminals: carry no current


@dataclass
class ValidationReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def feedback(self) -> str:
        """Text sent back to the parser LLM on a retry."""
        return "\n".join(f"- {e}" for e in self.errors)

    def __str__(self) -> str:
        lines = [f"ERROR: {e}" for e in self.errors] + [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines) if lines else "OK"


class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> bool:
        """Join a and b. Returns False if they were already joined (= a loop)."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[ra] = rb
        return True


# ----------------------------------------------------------------------
def validate(nl: Netlist) -> ValidationReport:
    rep = ValidationReport()
    if nl.analysis == AnalysisType.CONCEPTUAL:
        return rep

    nodes = nl.nodes()
    types = {c.id: c.type.value for c in nl.components}
    is_dc = nl.analysis == AnalysisType.DC

    # --- node names differing only by case -----------------------------
    by_lower: Dict[str, Set[str]] = defaultdict(set)
    for n in nodes:
        by_lower[n.lower()].add(n)
    for variants in by_lower.values():
        if len(variants) > 1:
            rep.errors.append(f"Node names differ only by case: {sorted(variants)}. Use one spelling.")

    # --- op-amp pins ----------------------------------------------------
    for c in nl.components:
        if c.type.value == "OPAMP":
            out, inp, inm = c.nodes
            if out == "0":
                rep.errors.append(f"{c.id}: the op-amp output is connected to ground. Its nodes are "
                                  "['out', 'in+', 'in-']; the output must be its own node.")
            if inp == inm:
                rep.errors.append(f"{c.id}: both op-amp inputs are on node {inp}; in+ and in- must differ.")

    # --- ground ---------------------------------------------------------
    if "0" not in nodes:
        rep.errors.append('No ground node. One node must be named "0" (usually the source\'s negative terminal).')

    # --- current-carrying connections per node --------------------------
    # (component id, terminal index); E/G sensing terminals are excluded
    conn: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for c in nl.components:
        skip = CONTROL_ONLY.get(c.type.value, ())
        for i, n in enumerate(c.nodes):
            if i not in skip:
                conn[n].append((c.id, i))

    # --- connectivity: every node must reach ground ---------------------
    if "0" in nodes:
        uf = _UnionFind()
        for n in nodes:
            uf.find(n)
        for c in nl.components:
            t = c.type.value
            if t in CONTROL_ONLY:
                uf.union(c.nodes[0], c.nodes[1])
                uf.union(c.nodes[2], c.nodes[3])
            else:
                for n in c.nodes[1:]:
                    uf.union(c.nodes[0], n)
        ground_root = uf.find("0")
        floating = sorted(n for n in nodes if uf.find(n) != ground_root)
        if floating:
            rep.errors.append(f"Nodes {floating} are not connected to ground (node \"0\"). "
                              "Check that shared nodes use the same name.")

    # --- loops of voltage sources ---------------------------------------
    uf_v = _UnionFind()
    for c in nl.components:
        if c.type.value in VOLTAGE_TYPES:
            a, b = c.nodes[0], c.nodes[1]
            if not uf_v.union(a, b):
                rep.errors.append(f"{c.id} closes a loop made only of voltage sources "
                                  f"(across nodes {a}, {b}); the circuit has no unique solution.")

    # --- inductor across a voltage source in DC -------------------------
    if is_dc:
        v_pairs = {frozenset(c.nodes[:2]) for c in nl.components if c.type.value in VOLTAGE_TYPES}
        for c in nl.components:
            if c.type.value == "L" and frozenset(c.nodes) in v_pairs:
                rep.errors.append(f"{c.id} is directly across a voltage source; in DC an inductor is a "
                                  "short, so the source would be shorted.")

    # --- nodes fed only by current sources (and capacitors in DC) -------
    for n, items in conn.items():
        if n == "0":
            continue
        kinds = {types[cid] for cid, _ in items}
        if "OPAMP" in kinds:
            continue                      # ideal op-amp virtual short defines it
        current_like = CURRENT_TYPES | ({"C"} if is_dc else set())
        if kinds and kinds <= current_like and (kinds & CURRENT_TYPES):
            rep.errors.append(f"Node {n} connects only to current sources"
                              f"{' / capacitors (open in DC)' if is_dc else ''}; its voltage is undefined.")

    # --- queries ---------------------------------------------------------
    queried_nodes = _check_queries(nl, nodes, types, rep)

    # --- single-connection nodes (warning) ------------------------------
    for n, items in conn.items():
        if n != "0" and len(items) == 1:
            cid = items[0][0]
            if n in queried_nodes:
                rep.warnings.append(f"Node {n} is an open terminal (only {cid} connects to it). "
                                    "Fine for open-circuit / Thevenin questions.")
            else:
                rep.warnings.append(f"Node {n} has only one connection ({cid}). Is a wire missing?")

    # --- analysis-specific warnings -------------------------------------
    sources = [c for c in nl.components if c.type.value in {"V", "I"}]
    if nl.analysis == AnalysisType.AC and sources and not any(c.ac not in (None, "") for c in sources):
        rep.warnings.append("AC analysis but no source has an AC magnitude; its DC value will be "
                            "used as the AC amplitude.")
    if nl.analysis == AnalysisType.SMALL_SIGNAL and not any(t in {"Q", "M"} for t in types.values()):
        rep.warnings.append("Small-signal analysis requested but the circuit has no BJT or MOSFET.")
    if nl.analysis == AnalysisType.TRANSIENT and not any(t in {"C", "L", "D"} for t in types.values()):
        rep.warnings.append("Transient analysis requested but the circuit has no capacitor or inductor.")

    return rep


def _check_queries(nl: Netlist, nodes: List[str], types: Dict[str, str],
                   rep: ValidationReport) -> Set[str]:
    """Validate each query string. Returns the set of nodes the queries mention."""
    node_set = set(nodes)
    mentioned: Set[str] = set()

    def need_node(n: str, q: str) -> None:
        mentioned.add(n)
        if n not in node_set:
            rep.errors.append(f"Query {q} refers to node '{n}', which is not in the circuit. "
                              f"Nodes are: {nodes}")

    def need_comp(cid: str, q: str) -> None:
        if cid not in types:
            rep.errors.append(f"Query {q} refers to component '{cid}', which is not in the circuit.")

    for q in nl.query:
        q = q.replace(" ", "")
        if m := _Q_NODE_V.match(q):
            need_node(m.group(1), q)
        elif m := _Q_DIFF_V.match(q):
            need_node(m.group(1), q); need_node(m.group(2), q)
        elif m := _Q_COMP.match(q):
            need_comp(m.group(2), q)
        elif m := _Q_PORT.match(q):
            need_node(m.group(2), q); need_node(m.group(3), q)
        elif m := _Q_WAVE_V.match(q):
            need_node(m.group(2), q)
            if m.group(3):
                need_node(m.group(3), q)
        elif m := _Q_WAVE_C.match(q):
            need_comp(m.group(2), q)
            if m.group(1) in ("PIV", "conduction") and m.group(2) in types and types[m.group(2)] != "D":
                rep.errors.append(f"Query {q}: {m.group(1)}() needs a diode, {m.group(2)} is {types[m.group(2)]}.")
        elif m := _Q_TRANSFER.match(q):
            need_node(m.group(1), q)
            inp = m.group(2)
            if inp not in node_set and inp not in types:
                rep.errors.append(f"Query {q}: input '{inp}' is neither a node nor a source id.")
            if inp in node_set:
                mentioned.add(inp)
        elif m := _Q_STATE.match(q):
            need_comp(m.group(1), q)
        elif m := _Q_FET.match(q):
            cid = m.group(2)
            need_comp(cid, q)
            if cid in types and types[cid] != "M":
                rep.errors.append(f"Query {q}: {m.group(1)}() needs a MOSFET, {cid} is {types[cid]}.")
        elif m := _Q_DEV.match(q):
            cid = m.group(2)
            need_comp(cid, q)
            if cid in types and types[cid] not in ("Q", "M"):
                rep.errors.append(f"Query {q}: {m.group(1)}() needs a transistor, {cid} is {types[cid]}.")
            elif cid in types and types[cid] == "M" and m.group(1) not in ("gm", "ro", "region"):
                rep.errors.append(f"Query {q}: {m.group(1)}() is a BJT quantity; for a MOSFET ask for "
                                  f"ID({cid}), VGS({cid}), VDS({cid}) or gm({cid}).")
        elif m := _Q_BIAS.match(q):
            cid = m.group(1)
            need_comp(cid, q)
            if cid in types and types[cid] not in {"Q", "M", "D"}:
                rep.errors.append(f"Query {q}: bias() needs a transistor or diode, {cid} is {types[cid]}.")
        elif _Q_BARE.match(q):
            pass
        else:
            rep.errors.append(f"Unknown query form '{q}'. Allowed: {QUERY_HELP}")
    return mentioned