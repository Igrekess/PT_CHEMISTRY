"""
Periodic table structure functions.

Derived from PT first principles: period boundaries follow 2k² shells
where k = per // 2 + 1 (prime-indexed shells P1=3, P2=5, P3=7).
Zero adjustable parameters.
"""
from ptc.constants import P1, P2, P3

_BLOCK_MAP = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
_CAP_MAP   = {0: 2, 1: 2 * P1, 2: 2 * P2, 3: 2 * P3}  # 2, 6, 10, 14


def period(Z: int) -> int:
    """Return the period number for atomic number Z."""
    cumul, per = 0, 1
    while True:
        k = per // 2 + 1
        cap = 2 * k * k
        if Z <= cumul + cap:
            return per
        cumul += cap
        per += 1


def period_start(per: int) -> int:
    """Return the first Z of period per."""
    z0 = 1
    for p in range(1, per):
        k = p // 2 + 1
        z0 += 2 * k * k
    return z0


def l_of(Z: int) -> int:
    """Angular momentum of the valence sub-shell: s=0, p=1, d=2, f=3."""
    per = period(Z)
    z0 = period_start(per)
    pos = Z - z0
    if pos < 2:
        return 0
    if per <= 3:
        return 1
    if per <= 5:
        return 2 if pos < 12 else 1
    if pos == 2:
        return 2
    if pos < 16:
        return 3
    if pos < 26:
        return 2
    return 1


def _n_fill_aufbau(Z: int) -> int:
    """Number of electrons in the valence sub-shell (Aufbau, no promotions).

    This is the RADIAL filling — used for screening (geometric γ₅ decay).
    The screening is a spatial phenomenon and follows the standard Aufbau
    ordering regardless of Madelung promotions.
    """
    per = period(Z)
    z0 = period_start(per)
    pos = Z - z0
    l = l_of(Z)
    if l == 0:
        return min(pos + 1, 2)
    cap = 2 * (per // 2 + 1) ** 2
    if l == 1:
        return Z - (z0 + cap - 6) + 1
    if l == 2:
        nd = max(0, pos + 1 - 2)
        if per >= 6:
            nd = max(0, nd - 14)
        return min(nd, 2 * P2)
    if l == 3:
        return min(max(0, pos + 1 - 2), 2 * P3)
    return 1


# ── Madelung anomalies: PT predictions from the pentagon ──────────────
# d5 = half-fill = D_KL maximum on Z/10Z (persistence peak)
# d10 = closure = H minimum on Z/10Z (entropy floor)
# These are INFORMATIONAL, not radial — the polygon structure demands them.

# Z → (nd_madelung, ns_madelung) for elements with s→d promotions
_MADELUNG_PROMOTIONS: dict[int, tuple[int, int]] = {
    24: (5, 1),   # Cr: d4s2 → d5s1  (half-fill)
    29: (10, 1),  # Cu: d9s2 → d10s1 (closure)
    41: (4, 1),   # Nb: d3s2 → d4s1  (quasi-half)
    42: (5, 1),   # Mo: d4s2 → d5s1  (half-fill)
    44: (7, 1),   # Ru: d6s2 → d7s1  (post-half)
    45: (8, 1),   # Rh: d7s2 → d8s1  (post-half)
    46: (10, 0),  # Pd: d8s2 → d10s0 (double closure)
    47: (10, 1),  # Ag: d9s2 → d10s1 (closure)
    78: (9, 1),   # Pt: d8s2 → d9s1  (quasi-closure)
    79: (10, 1),  # Au: d9s2 → d10s1 (closure)
    111: (10, 1), # Rg: d9s2 → d10s1 (closure, relativistic)
}


def n_fill(Z: int) -> int:
    """Number of electrons in the valence sub-shell (Madelung).

    Returns the INFORMATIONAL filling — includes d5/d10 promotions
    predicted by the pentagon Z/(2×5)Z structure.  Used for polygon
    construction (structure fine = harmonique).

    For screening (radial, geometric), use _n_fill_aufbau().
    """
    if Z in _MADELUNG_PROMOTIONS and l_of(Z) == 2:
        return _MADELUNG_PROMOTIONS[Z][0]
    return _n_fill_aufbau(Z)


def ns_config(Z: int) -> int:
    """Number of s-electrons, detecting s→d promotions.

    In PT: promotion occurs when Hund half-filling stability or d-shell
    closure exceeds the s→d promotion cost (gap decreases with period).

    Uses _n_fill_aufbau for the detection logic (avoids recursion with
    the Madelung n_fill).
    """
    if Z in _MADELUNG_PROMOTIONS and l_of(Z) == 2:
        return _MADELUNG_PROMOTIONS[Z][1]
    l = l_of(Z)
    if l == 0:
        return min(_n_fill_aufbau(Z), 2)
    if l != 2:
        return 2
    return 2


def block_of(Z: int) -> str:
    """Return the block letter ('s', 'p', 'd', or 'f') for element Z."""
    return _BLOCK_MAP[l_of(Z)]


def capacity(Z: int) -> int:
    """Return the capacity (2(2l+1)) of the valence sub-shell for element Z."""
    return _CAP_MAP[l_of(Z)]


def _np_of(Z: int) -> int:
    """Number of p-electrons for element Z.

    Returns the valence p-electron count:
      s-block (l=0): 0 — no p-electrons
      p-block (l=1): n_fill(Z) — the p sub-shell filling
      d-block (l=2): 0 — d-block bonds through s+d, not p
      f-block (l=3): 0 — same logic
    """
    l = l_of(Z)
    if l == 1:
        return min(n_fill(Z), _CAP_MAP[1])
    return 0


def _nd_of(Z: int) -> int:
    """Number of d-electrons for element Z.

    Returns the d sub-shell filling only for d-block (l=2).
    s-block, p-block, and f-block return 0.
    """
    if l_of(Z) != 2:
        return 0
    return min(n_fill(Z), 2 * P2)


def _valence_electrons(Z: int) -> int:
    """Total valence electrons."""
    return n_fill(Z) + ns_config(Z)


def _lp_pairs(Z: int, bo: float) -> int:
    """Lone pairs available for bonding."""
    l = l_of(Z)
    if l == 0:
        return 0
    np_val = _np_of(Z)
    P1 = 3
    if np_val <= P1:
        return 0
    return max(0, np_val - P1 - int(bo - 1))
