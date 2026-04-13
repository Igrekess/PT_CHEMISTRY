"""
dblock.py — Shared d-block primitives for PTC molecular engines.

Single source of truth for d-block classification and PT-derived
correction functions. All three code paths (diatomic, triatomic,
polyatomic) import from here.

April 2026 — Theorie de la Persistance
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from ptc.constants import (
    P1, P2, P3, S_HALF,
    S3, S5, S7, C3, C5, C7,
    D3, D5, D7,
    GAMMA_3, GAMMA_5, GAMMA_7,
    AEM, RY, COULOMB_EV_A, D_FULL,
    NS1, IE2_DBLOCK,
    R35_DARK, R57_DARK, R37_DARK,
)
from ptc.periodic import period, l_of, n_fill, ns_config


@dataclass(frozen=True)
class DBlockState:
    """Immutable d-block state for one atom.

    Created once per atom via from_Z(). All d-block classification
    flags are pre-computed and cached.
    """
    Z: int
    nd: int
    ns: int
    l: int
    per: int
    is_d: bool
    is_ns1: bool
    is_d10: bool
    is_d10s2: bool
    is_d5: bool
    is_d5s1: bool
    is_d5s2: bool
    is_early: bool
    is_late: bool
    ie2: float

    @staticmethod
    @lru_cache(maxsize=128)
    def from_Z(Z: int) -> DBlockState:
        """Construct DBlockState from atomic number Z."""
        l = l_of(Z)
        per = period(Z)
        is_d = (l == 2)
        nd = n_fill(Z) if is_d else 0
        ns = ns_config(Z) if is_d else (min(Z, 2) if l == 0 else 2)
        ie2 = IE2_DBLOCK.get(Z, 0.0)

        return DBlockState(
            Z=Z, nd=nd, ns=ns, l=l, per=per,
            is_d=is_d,
            is_ns1=(Z in NS1),
            is_d10=(is_d and nd >= 2 * P2),
            is_d10s2=(is_d and nd >= 2 * P2 and ns >= 2),
            is_d5=(is_d and nd == P2),
            is_d5s1=(is_d and nd == P2 and ns == 1),
            is_d5s2=(is_d and nd == P2 and ns >= 2),
            is_early=(is_d and nd < P2),
            is_late=(is_d and P2 < nd < 2 * P2),
            ie2=ie2,
        )


# ---------------------------------------------------------------------------
# Shared d-block primitives
# ---------------------------------------------------------------------------

def pi_gate(bo: float) -> float:
    """LP→d π gate: max(S5, min(bo-1, 1))."""
    return max(S5, min(bo - 1.0, 1.0))


def vacancy_fraction(dbs: DBlockState) -> float:
    """d-shell vacancy fraction: (2P₂ - nd) / (2P₂), clamped [0,1]."""
    if not dbs.is_d:
        return 0.0
    return max(0.0, float(2 * P2 - dbs.nd)) / (2.0 * P2)


def mertens_factor(n_unpaired: int) -> float:
    """Chebyshev: n/2^(n-1). Returns 0 for n<=0, linear for n=1."""
    if n_unpaired <= 0:
        return 0.0
    if n_unpaired <= 1:
        return float(n_unpaired)
    return float(n_unpaired) / (2.0 ** (n_unpaired - 1))


def d_crt_screening(dbs: DBlockState) -> float:
    """CRT screening: C5^(nd/P2). Returns 1.0 for non-d-block."""
    if not dbs.is_d or dbs.nd <= 0:
        return 1.0
    return C5 ** (float(dbs.nd) / P2)


def sd_hybrid_eps(dbs: DBlockState, ie1: float) -> float:
    """sd-hybrid rotation for late-d. Returns ie1 unchanged otherwise."""
    if not dbs.is_late or dbs.ie2 <= 0:
        return ie1
    frac = float(dbs.nd - P2) / P2
    return ie1 * (1.0 - frac) + dbs.ie2 * frac


def d10_ionic_contraction(dbs: DBlockState) -> float:
    """d10 radius contraction: C5 for d10 or NS1 with nd>=2P2-1."""
    if not dbs.is_d:
        return 1.0
    if dbs.nd >= 2 * P2:
        return C5
    if dbs.is_ns1 and dbs.nd >= 2 * P2 - 1:
        return C5
    return 1.0


def cfse_energy(dbs: DBlockState, f_lp: float = 1.0) -> float:
    """CFSE: sin²(2π·r7/P3) × D7/P2 × f_lp. Zero for d10/non-d."""
    if not dbs.is_d or dbs.nd <= 0 or dbs.nd >= 2 * P2:
        return 0.0
    r7 = dbs.nd % P3
    sin2_r7 = math.sin(2.0 * math.pi * r7 / P3) ** 2
    return sin2_r7 * D7 / P2 * f_lp


_OMEGA_10 = 2.0 * math.pi / (2 * P2)


def dark_modes(dbs_A: DBlockState, dbs_B: DBlockState) -> float:
    """Dark k=2,4 Fourier modes on Z/10Z for each d-block atom."""
    corr = 0.0
    for dbs in (dbs_A, dbs_B):
        if not dbs.is_d or dbs.nd <= 0:
            continue
        nd = dbs.nd
        corr += (R35_DARK + R57_DARK) * math.cos(2.0 * _OMEGA_10 * nd)
        corr += R37_DARK * math.cos(4.0 * _OMEGA_10 * nd)
    return corr


def half_fill_exchange(dbs_d: DBlockState, dbs_partner: DBlockState, bo: float) -> float:
    """d5 half-fill exchange: C5 * S_HALF if either atom is d5, else 0."""
    if dbs_d.is_d5 or dbs_partner.is_d5:
        return C5 * S_HALF
    return 0.0


def d10s2_ie2_correction(dbs: DBlockState, q_rel: float, r_ion: float) -> float:
    """IE2 polarization for d10s2 (Zn/Cd/Hg) in ionic regime."""
    if not dbs.is_d10s2 or dbs.ie2 <= 0 or q_rel >= S_HALF:
        return 0.0
    return COULOMB_EV_A ** 2 * S3 / (r_ion ** 2 * dbs.ie2)
