"""
MOSFET device model (Phase 3, step 5): square-law DC regions and the small-signal parameters.

Square law, NMOS (PMOS: every voltage reversed, so the same formulas with V_SG and V_SD):
    cutoff      V_GS < V_t                       I_D = 0
    saturation  V_GS >= V_t and V_DS >= V_OV     I_D = ½·k·V_OV²·(1 + λ·V_DS)        V_OV = V_GS − V_t
    triode      V_GS >= V_t and V_DS <  V_OV     I_D = k·(V_OV·V_DS − V_DS²/2)
with k = µ·Cox·W/L in A/V² (so I_D = ½·k·V_OV² in saturation). A question that writes the law WITHOUT the
½, I_D = K·(V_GS − V_t)², gives K in params as "K2": then k = 2·K.
The gate carries no DC current, so I_G = 0 always and I_S = I_D.

Unlike the BJT's active region (I_C = β·I_B is linear in the unknowns), these are NONLINEAR in the node
voltages, so each region is solved by Newton-Raphson: the device is replaced by its tangent at the present
operating point (a conductance, a transconductance and a current source), the linear circuit is solved, and
the point is updated until it stops moving. The region is then checked exactly as for a BJT.

Small signal in saturation:
    gm = k·V_OV = √(2·k·I_D) = 2·I_D/V_OV        r_o = 1/(λ·I_D)   (infinite when λ = 0)
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from schemas import Component, parse_value
from solvers.base import SolverError, fmt_num, fmt_si

_ALIASES = {
    "k": ("k", "kn", "kp", "k_n", "k_p", "beta", "kprime_wl", "k_prime_w_l", "mucox_wl", "conduction_parameter"),
    "K2": ("k2", "kcap", "k_no_half", "k_half"),
    "kprime": ("kprime", "k_prime", "mucox", "mu_cox", "mun_cox", "mup_cox", "kn_prime", "kp_prime", "k_n_prime"),
    "WL": ("wl", "w_l", "w_over_l", "w_by_l", "aspect_ratio"),
    "W": ("w", "width"),
    "L": ("l", "length"),
    "Vt": ("vt", "vth", "vto", "vtn", "vtp", "v_t", "v_th", "vgsth", "vgs_th", "threshold", "vt0"),
    "lam": ("lambda", "lam", "lambda_", "channel_length_modulation"),
    "VA": ("va", "early", "v_a"),
}


def _num(raw) -> float:
    return float(raw) if isinstance(raw, (int, float)) else parse_value(str(raw).strip())


@dataclass
class MOSFET:
    id: str
    d: str
    g: str
    s: str
    pmos: bool
    k: float                     # A/V², with I_D = ½·k·V_OV²
    Vt: float                    # magnitude
    lam: float = 0.0             # 1/V
    notes: List[str] = field(default_factory=list)
    states = ("sat", "tri", "cutoff")

    @property
    def kind(self) -> str:
        return "PMOS" if self.pmos else "NMOS"

    def as_dict(self) -> Dict[str, object]:
        return {"pmos": self.pmos, "k": self.k, "Vt": self.Vt, "lam": self.lam}

    # terminal pairs in the conducting direction (NMOS: G–S, D–S;  PMOS: S–G, S–D)
    @property
    def gs(self) -> Tuple[str, str]:
        return (self.s, self.g) if self.pmos else (self.g, self.s)

    @property
    def ds(self) -> Tuple[str, str]:
        return (self.s, self.d) if self.pmos else (self.d, self.s)

    @property
    def vgs_name(self) -> str:
        return "V_SG" if self.pmos else "V_GS"

    @property
    def vds_name(self) -> str:
        return "V_SD" if self.pmos else "V_DS"

    def label(self) -> Tuple[str, str]:
        lam = f", λ = {fmt_num(self.lam)} /V" if self.lam else ", λ = 0 (r_o ignored)"
        return (f"{self.id}: {self.kind}, k = {fmt_si(self.k, 'A/V²')}, V_t = {fmt_si(self.Vt, 'V')}{lam}",
                f"saturation → I_D = ½·k·({self.vgs_name} − V_t)²"
                + ("·(1 + λ·" + self.vds_name + ")" if self.lam else "")
                + f";  triode → I_D = k·(V_OV·{self.vds_name} − {self.vds_name}²/2);  cutoff → I_D = 0")

    # ------------------------------------------------------------------
    def current(self, state: str, vgs: float, vds: float) -> float:
        """I_D in the conducting direction for the given operating point."""
        ov = vgs - self.Vt
        if state == "cutoff" or ov <= 0:
            return 0.0
        if state == "sat":
            return 0.5 * self.k * ov * ov * (1 + self.lam * max(vds, 0.0))
        return self.k * (ov * vds - 0.5 * vds * vds)

    def slopes(self, state: str, vgs: float, vds: float) -> Tuple[float, float]:
        """(gm, gds) = the partial derivatives of I_D at this operating point."""
        ov = vgs - self.Vt
        if state == "cutoff" or ov <= 0:
            return 0.0, 0.0
        if state == "sat":
            return self.k * ov * (1 + self.lam * max(vds, 0.0)), 0.5 * self.k * ov * ov * self.lam
        return self.k * vds, self.k * (ov - vds)

    def parts(self, state: str, vgs: float = 0.0, vds: float = 0.0, off_as_r: bool = False) -> List[Component]:
        """The device replaced by its tangent at (vgs, vds): a transconductance, a conductance and a current
        source, all between the drain and source (the gate draws no current)."""
        gp, gn = self.gs
        dp, dn = self.ds
        if state == "cutoff":
            t, v = ("R", "1e12") if off_as_r else ("I", "0")
            return [Component(id=f"{self.id}_ds", type=t, nodes=[dp, dn], value=v)]
        gm, gds = self.slopes(state, vgs, vds)
        i0 = self.current(state, vgs, vds)
        ieq = i0 - gm * vgs - gds * vds
        out = [Component(id=f"{self.id}_gm", type="G", nodes=[dp, dn, gp, gn], value=repr(gm)),
               Component(id=f"{self.id}_ds", type="I", nodes=[dp, dn], value=repr(ieq))]
        if abs(gds) > 1e-18:
            # The output conductance of the tangent model, as a current source controlled by its OWN two
            # nodes (that is exactly a conductance). It cannot be written as a resistor: in the triode
            # region ∂I_D/∂V_DS = k·(V_OV − V_DS) is NEGATIVE whenever an iterate has V_DS above V_OV, and
            # a negative resistance is not a valid component — the conductance form takes any sign.
            out.append(Component(id=f"{self.id}_go", type="G", nodes=[dp, dn, dp, dn], value=repr(gds)))
        elif off_as_r:
            out.append(Component(id=f"{self.id}_go", type="R", nodes=[dp, dn], value="1e12"))
        return out

    # ------------------------------------------------------------------
    # Newton helpers: where to start, and keeping an iterate inside the assumed region
    # ------------------------------------------------------------------
    def starts(self, state: str) -> List[Tuple[float, float]]:
        """Starting points for the iteration. The square law is a parabola in V_GS, so a start on the wrong
        side of it (V_DS above V_OV while solving the triode region, say) can send Newton the wrong way.
        A couple of starts per region covers that."""
        if state == "sat":
            return [(self.Vt + 1.0, 3.0), (self.Vt + 3.0, 6.0), (self.Vt + 0.2, 1.0)]
        if state == "tri":
            return [(self.Vt + 4.0, 0.2), (self.Vt + 1.0, 0.1), (self.Vt + 0.5, 0.05)]
        return [(0.0, 1.0)]

    def clamp(self, state: str, vgs: float, vds: float) -> Tuple[float, float]:
        """Hold an iterate inside the region being assumed. If the real solution is elsewhere the iterate
        sticks on the boundary and stops matching the circuit's own solution, which is how the caller
        notices that this region is the wrong guess."""
        if state == "sat":
            vgs = max(vgs, self.Vt)
            return vgs, max(vds, vgs - self.Vt)
        if state == "tri":
            vgs = max(vgs, self.Vt)
            return vgs, min(max(vds, 0.0), max(vgs - self.Vt, 0.0))
        return vgs, vds

    @staticmethod
    def limit(new: float, old: float, kind: str) -> float:
        """Limit one Newton step, as SPICE does, so a far-off iterate cannot overshoot."""
        cap = max(0.5, 0.25 * abs(old)) if kind == "vgs" else max(2.0, 0.5 * abs(old))
        return old + max(-cap, min(cap, new - old))

    # ------------------------------------------------------------------
    def checks(self, state: str, vgs: float, vds: float, sig: int = 4) -> List[Tuple[bool, str]]:
        ov = vgs - self.Vt
        vg, vd = self.vgs_name, self.vds_name
        if state == "sat":
            ok1, ok2 = ov >= -1e-9, vds >= ov - 1e-9
            return [(ok1, f"{self.id} saturated needs {vg} ≥ V_t: V_OV = {fmt_si(ov, 'V', sig)}"
                          + (" ✓" if ok1 else " ✗ (it is cut off)")),
                    (ok2, f"{self.id} saturated needs {vd} ≥ V_OV: {vd} = {fmt_si(vds, 'V', sig)} vs V_OV = "
                          f"{fmt_si(ov, 'V', sig)}" + (" ✓" if ok2 else " ✗ (it is in the triode region)"))]
        if state == "tri":
            ok1, ok2 = ov >= -1e-9, -1e-9 <= vds <= ov + 1e-9
            return [(ok1, f"{self.id} in triode needs {vg} ≥ V_t: V_OV = {fmt_si(ov, 'V', sig)}"
                          + (" ✓" if ok1 else " ✗ (it is cut off)")),
                    (ok2, f"{self.id} in triode needs 0 ≤ {vd} < V_OV: {vd} = {fmt_si(vds, 'V', sig)}"
                          + (" ✓" if ok2 else " ✗ (it is saturated)"))]
        ok = ov <= 1e-9
        return [(ok, f"{self.id} cutoff needs {vg} < V_t: V_OV = {fmt_si(ov, 'V', sig)}" + (" ✓" if ok else " ✗"))]

    def region(self, state: str) -> str:
        return {"sat": "saturation", "tri": "triode (linear / ohmic)", "cutoff": "cutoff"}[state]

    def gm_ro(self, idq: float, vgs: float, vds: float) -> Tuple[float, Optional[float]]:
        """Small-signal gm and r_o in saturation at this operating point."""
        gm = math.sqrt(2 * self.k * idq) if idq > 0 else 0.0
        ro = (1.0 / (self.lam * idq)) if (self.lam and idq > 0) else None
        return gm, ro


def resolve_mosfet(c: Component, conventions: Optional[dict] = None) -> MOSFET:
    p = {str(k).strip().lower().replace(" ", "_").replace("-", "_"): v for k, v in (c.params or {}).items()}
    raw = {key: next((p[a] for a in aliases if p.get(a) not in (None, "")), None) for key, aliases in _ALIASES.items()}
    mc = (conventions or {}).get("mosfet") or {}
    model = (c.model or "").strip().lower()
    pmos = "pmos" in model or "p-channel" in model or "pchannel" in model or "p_channel" in model
    notes: List[str] = []

    if raw["k"] is not None:
        k = _num(raw["k"])
    elif raw["K2"] is not None:
        k = 2 * _num(raw["K2"])
        notes.append(f"the question's K = {fmt_si(_num(raw['K2']), 'A/V²')} is used as I_D = K·V_OV², i.e. k = 2K")
    elif raw["kprime"] is not None:
        wl = _num(raw["WL"]) if raw["WL"] is not None else (
            _num(raw["W"]) / _num(raw["L"]) if raw["W"] is not None and raw["L"] is not None else None)
        if wl is None:
            raise SolverError(f"{c.id}: k' is given but not W/L, so k = k'·(W/L) cannot be worked out")
        k = _num(raw["kprime"]) * wl
        notes.append(f"k = k'·(W/L) = {fmt_si(_num(raw['kprime']), 'A/V²')} × {fmt_num(wl)}")
    else:
        raise SolverError(f"{c.id}: the MOSFET needs its conduction parameter k (µCox·W/L, A/V²) — give it as "
                          f'params {{"k": 0.5e-3}}, or k\' with W/L')
    if raw["Vt"] is None:
        raise SolverError(f"{c.id}: the MOSFET needs its threshold voltage — give it as params {{\"Vt\": 1}}")
    vt = abs(_num(raw["Vt"]))
    lam = _num(raw["lam"]) if raw["lam"] is not None else float(mc.get("lambda", 0.0) or 0.0)
    if raw["lam"] is None and raw["VA"] is not None:
        lam = 1.0 / abs(_num(raw["VA"]))
        notes.append(f"λ = 1/V_A = {fmt_num(lam)} /V")
    return MOSFET(id=c.id, d=c.nodes[0], g=c.nodes[1], s=c.nodes[2], pmos=pmos, k=k, Vt=vt, lam=lam, notes=notes)