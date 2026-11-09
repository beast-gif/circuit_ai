"""
Filter identification from a transfer function H(s) (Phase 2, step 4).

Works on ANY H(s) the MNA engine produces (passive RC/RLC, Sallen-Key, MFB, ...): it never looks at the
topology, only at the polynomials. So a new filter circuit needs no new code.

First order,  D(s) = a1·s + a0                (ωc = a0/a1)
    N = n0                      -> low-pass     K = n0/a0          (gain at DC)
    N = n1·s                    -> high-pass    K = n1/a1          (gain at high f)
    N = n1·s + n0, n0/n1 = -a0/a1 -> all-pass   K = n1/a1

Second order, D(s) = a2·s² + a1·s + a0        (ω0 = √(a0/a2),  Q = √(a0·a2)/a1)
    N = n0                      -> low-pass     K = n0/a0          H = K·ω0² / (s² + (ω0/Q)s + ω0²)
    N = n2·s²                   -> high-pass    K = n2/a2          H = K·s²  / (...)
    N = n1·s                    -> band-pass    K = n1/a1          H = K·(ω0/Q)·s / (...)   (K = gain at f0)
    N = n2·(s² + ω0²)           -> notch        K = n2/a2          H = K·(s² + ω0²) / (...)
    N = n2·(s² - (ω0/Q)s + ω0²) -> all-pass     K = n2/a2

Higher order: low-pass / high-pass / band-pass from the behaviour at s -> 0 and s -> ∞.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import sympy as sp

from solvers.mna import s

# well-known second-order Q values (alignments)
ALIGNMENTS = [(1 / sp.sqrt(2), "Butterworth: maximally flat"), (1 / sp.sqrt(3), "Bessel: maximally flat delay"),
              (sp.Rational(1, 2), "critically damped: two equal real poles")]

STANDARD_FORM = {
    ("low-pass", 1): "H(s) = K·ωc / (s + ωc)",
    ("high-pass", 1): "H(s) = K·s / (s + ωc)",
    ("all-pass", 1): "H(s) = K·(s − ωc) / (s + ωc)",
    ("low-pass", 2): "H(s) = K·ω0² / (s² + (ω0/Q)·s + ω0²)",
    ("high-pass", 2): "H(s) = K·s² / (s² + (ω0/Q)·s + ω0²)",
    ("band-pass", 2): "H(s) = K·(ω0/Q)·s / (s² + (ω0/Q)·s + ω0²)",
    ("notch", 2): "H(s) = K·(s² + ω0²) / (s² + (ω0/Q)·s + ω0²)",
    ("all-pass", 2): "H(s) = K·(s² − (ω0/Q)·s + ω0²) / (s² + (ω0/Q)·s + ω0²)",
}

# what K means for each type, for the explanation
K_MEANING = {"low-pass": "gain at DC, H(0)", "high-pass": "gain at high frequency, H(∞)",
             "band-pass": "gain at the centre frequency, H(jω0)", "notch": "gain far from the notch",
             "all-pass": "gain magnitude at every frequency"}


@dataclass
class Response:
    kind: str                       # low-pass | high-pass | band-pass | notch | all-pass | other
    order: int
    K: Optional[sp.Expr] = None     # pass-band gain, with its sign (an inverting filter has K < 0)
    w0: Optional[sp.Expr] = None    # ω0 (2nd order) or ωc (1st order), rad/s
    Q: Optional[sp.Expr] = None     # 2nd order only
    a: tuple = ()                   # denominator coefficients, highest power first
    n: tuple = ()                   # numerator coefficients, same length as `a`

    @property
    def standard(self) -> str:
        return STANDARD_FORM.get((self.kind, self.order), "")

    @property
    def label(self) -> str:
        words = {1: "first-order", 2: "second-order"}.get(self.order, f"order-{self.order}")
        return f"{words} {self.kind}" if self.kind != "other" else f"{words} response (not a standard type)"


def _zero(x) -> bool:
    x = sp.simplify(x)
    if x == 0:
        return True
    if x.free_symbols:
        return False
    return abs(complex(sp.N(x))) < 1e-12


def _same(x, y) -> bool:
    """x == y for exact symbols, or within 1e-9 relative for numbers."""
    d = sp.simplify(x - y)
    if d == 0:
        return True
    if d.free_symbols:
        return False
    return abs(complex(sp.N(d))) <= 1e-9 * max(abs(complex(sp.N(x))), abs(complex(sp.N(y))), 1e-30)


def analyse(H: sp.Expr) -> Optional[Response]:
    """Classify H(s). None when H has no s (a pure gain) or cannot be split into polynomials."""
    H = sp.sympify(H)
    if not H.has(s):
        return None
    try:
        num, den = sp.fraction(sp.cancel(sp.together(H)))
        N, D = sp.Poly(num, s), sp.Poly(den, s)
    except (sp.PolynomialError, sp.GeneratorsNeeded):
        return None
    order = D.degree()
    if order < 1 or N.degree() > order:
        return Response("other", max(order, 0))
    a = [sp.simplify(x) for x in D.all_coeffs()]                     # a[0] = highest power
    nn = [sp.Integer(0)] * (order - N.degree()) + [sp.simplify(x) for x in N.all_coeffs()]
    lead = a[0]
    a = [sp.simplify(x / lead) for x in a]                           # monic denominator
    nn = [sp.simplify(x / lead) for x in nn]
    z = [_zero(x) for x in nn]
    if order <= 2 and _zero(a[-1]):
        # a pole at s = 0 (ideal integrator, -1/(sRC)): no finite pass-band gain, not a standard filter
        return Response("other", order, None, None, None, tuple(a), tuple(nn))

    if order == 1:
        a1, a0 = a
        n1, n0 = nn
        wc = sp.simplify(a0 / a1)
        if z[0] and not z[1]:
            return Response("low-pass", 1, sp.simplify(n0 / a0), wc, None, tuple(a), tuple(nn))
        if z[1] and not z[0]:
            return Response("high-pass", 1, sp.simplify(n1 / a1), wc, None, tuple(a), tuple(nn))
        if not z[0] and not z[1] and _same(n0 / n1, -a0 / a1):
            return Response("all-pass", 1, sp.simplify(n1 / a1), wc, None, tuple(a), tuple(nn))
        return Response("other", 1, None, wc, None, tuple(a), tuple(nn))

    if order == 2:
        a2, a1, a0 = a
        n2, n1, n0 = nn
        w0 = sp.simplify(sp.sqrt(a0 / a2))
        Q = sp.oo if _zero(a1) else sp.simplify(sp.sqrt(a0 * a2) / a1)
        kind, K = "other", None
        if z[0] and z[1] and not z[2]:
            kind, K = "low-pass", n0 / a0
        elif z[1] and z[2] and not z[0]:
            kind, K = "high-pass", n2 / a2
        elif z[0] and z[2] and not z[1]:
            kind, K = "band-pass", n1 / a1
        elif not z[0] and z[1] and not z[2] and _same(n2 / n0, a2 / a0):
            kind, K = "notch", n2 / a2
        elif not any(z) and _same(n1 / n2, -a1 / a2) and _same(n0 / n2, a0 / a2):
            kind, K = "all-pass", n2 / a2
        return Response(kind, 2, sp.simplify(K) if K is not None else None, w0, Q, tuple(a), tuple(nn))

    # higher order: from the asymptotes
    h0_zero, hinf_zero = z[-1], z[0]
    kind = ("low-pass" if not h0_zero and hinf_zero else
            "high-pass" if h0_zero and not hinf_zero else
            "band-pass" if h0_zero and hinf_zero else "other")
    K = {"low-pass": nn[-1] / a[-1], "high-pass": nn[0]}.get(kind)
    return Response(kind, order, sp.simplify(K) if K is not None else None, None, None, tuple(a), tuple(nn))


def alignment(Q) -> Optional[str]:
    """'Butterworth: maximally flat' when Q is (within 0.5 %) a well-known value."""
    if Q is None or Q == sp.oo or sp.sympify(Q).free_symbols:
        return None
    q = float(sp.N(Q))
    for val, name in ALIGNMENTS:
        if abs(q - float(val)) <= 5e-3 * float(val):
            return name
    return None


def passband_reference(resp: Optional[Response], hmax: float) -> tuple:
    """Gain the −3 dB points are measured from: the pass-band gain |K| for a low-pass or high-pass
    (the textbook definition, also when the response peaks), the peak |H|max for everything else.
    -> (reference value, description)"""
    if resp is not None and resp.kind in ("low-pass", "high-pass") and resp.K is not None \
            and not sp.sympify(resp.K).free_symbols:
        k = abs(complex(sp.N(resp.K)))
        where = "|H(0)|" if resp.kind == "low-pass" else "|H(∞)|"
        if k > 0:
            return k, f"pass-band gain {where}"
    return hmax, "|H|max"