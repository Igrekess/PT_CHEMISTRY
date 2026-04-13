"""ea_geo.py — Electron Affinity from polygon geometry.

Derives EA from the geometric shell operator (ShellPolygon / AtomicShell).
The EA is computed as a fraction of the IE, weighted by the ratio of the
capture amplitude to the ejection amplitude of the active polygon.

When the shell is full (no vacancies) or empty, capture amplitude is zero
and EA returns 0.  This naturally gives zero EA for noble gases.

Zero adjustable parameters.

March 2026 — Persistence Theory
"""
from __future__ import annotations

import math

from ptc.shell_polygon import build_atomic_shell, ShellPolygon
from ptc.ie_geo import IE_geo_eV


def _closed_polygon_tunnel(Z: int, shell) -> float:
    """Multi-channel tunneling capture through a closed active polygon.

    PT insights #16-#18: when the active polygon is CLOSED, the captured
    electron tunnels to the next available shells via up to 2 channels:

    Tree-level (p-channel):
        EA_p = IE × cap_p(n=1) × s × δ_blocking
        Tunneling through the blocking polygon's gap.

    1-loop NLO (d-channel, insight #18):
        When cap_d/cap_p > s  (d quasi-accessible):
            EA_d = IE × cap_d(n=1) × s × sin²₅
            The d-channel is an open propagation channel (sin²₅),
            not a barrier (δ₅).

    2-loop resonant (pd-degenerate):
        When cap_d/cap_p > cos²₃  (pd quasi-degenerate):
            EA_d uses (1/s) × sin²₅ instead (spin inversion via
            virtual d-loop, factor 1/s²).

    Noble gases (closed p-hexagon) excluded.
    """
    from ptc.constants import S_HALF, P1, P2, S3, S5, MU_STAR, C3
    from ptc.periodic import period as per_fn

    active = shell.active_polygon
    if active.vacancies > 0:
        return 0.0

    per = per_fn(Z)

    # No p-orbital at per=1 (He): no tunneling target.
    if per < 2:
        return 0.0

    # Noble gas exclusion: closed hexagon (p6) has full-octet closure.
    if active.prime == P1 and active.n_occupied == active.capacity:
        return 0.0

    # Alkaline earth exclusion (Be, Mg): closed s-pair at per < 4.
    # No d-pentagon available for tunneling → only p-hexagon target,
    # but the s²→p promotion gap exceeds the capture energy.
    # At per ≥ 4 (Ca, Sr, Ba), the d-pentagon IS accessible → EA > 0.
    if active.prime <= 1 and per < 4:
        return 0.0

    _C3 = C3  # cos²₃ from constants (Fraction-derived)

    # Blocking gap: s-pair (prime=1) couples through P₁=3 channel.
    q = 1.0 - 2.0 / MU_STAR
    p_block = active.prime if active.prime > 1 else P1
    delta_block = (1.0 - q ** p_block) / p_block
    tunnel_p = S_HALF * delta_block

    # p-channel target: fresh hexagon at n=1
    target_p = ShellPolygon(P1, 1, per=per)
    cap_p = target_p.capture_amplitude()
    # The j-j unpaired deficit in _p_heavy_capture applies to RESIDENT
    # electrons (Tl), not VIRTUAL tunneling targets.  Undo it here.
    if per >= 6:
        cap_p /= (1.0 - S3 / P1)
    if cap_p <= 0.0:
        return 0.0

    ie = IE_geo_eV(Z)
    ea_p = ie * cap_p * tunnel_p

    # d-channel: pentagon at n=1 (1-loop correction).
    # Only for s-pair blocking (prime ≤ 1): the s-pair sits OUTSIDE both
    # p and d targets.  For f-block blocking (prime=7), the d-shell is
    # between f and p — not an independent capture target.
    if active.prime <= 1:
        # Virtual d-target: the tunneling electron traverses the core,
        # which includes filled inner shells → f_buried=True (core-screened).
        target_d = ShellPolygon(P2, 1, per=per, f_buried=(per >= 6))
        cap_d = target_d.capture_amplitude()
    else:
        cap_d = 0.0
    ratio = cap_d / cap_p if cap_p > 0 else 0.0

    if ratio > S_HALF:
        # d-channel open: propagation coupling sin²₅
        if ratio > _C3:
            # pd quasi-degenerate: spin inversion s → 1/s
            d_coupling = (1.0 / S_HALF) * S5
        else:
            # d accessible but non-resonant
            d_coupling = S_HALF * S5
        ea_d = ie * cap_d * d_coupling
        return max(0.0, ea_p + ea_d)

    return max(0.0, ea_p)


def _madelung_d10s1_ea(Z: int, shell, ie: float) -> float:
    """EA for d10s1 Madelung elements (Cu, Ag, Au).

    PT mechanism: the captured electron completes the s-pair (s1→s2).
    The d10 closure behind provides a resonance — the closed pentagon
    Z/(2×5)Z at H_min amplifies the s-pair capture.

    Three PT contributions (0 adjustable parameters):
      1. d10 closure coupling = sin²₅/s (pentagon mixing angle / spin)
      2. Radial support = √(per_eff/2) where per_eff = per - 4 + 2
         (first d-shell appears at period 4)
      3. f14 buried resonance (per ≥ 6 only) = 1 + sin²₇/s
         (heptagon Z/14Z feedback through the d10 closure)
    """
    from ptc.constants import S_HALF, S5, S7

    s_poly = shell.polygons[0]  # s-shell
    cap_s = s_poly.capture_amplitude()
    ej_s = s_poly.ejection_amplitude()
    if cap_s <= 0 or ej_s <= 0:
        return 0.0

    per = s_poly.per
    # d10 closure resonance: sin²₅/s = pentagon coupling / spin
    c_d10 = S5 / S_HALF
    # Radial support: first d-shell at per=4, so per_eff = per - 2
    per_eff = per - 4 + 2
    f_per = math.sqrt(per_eff / 2.0)
    d10_boost = 1.0 + c_d10 * f_per
    # f14 buried resonance (per ≥ 6): the heptagon Z/14Z couples
    # through the d10 closure via sin²₇/s
    if per >= 6:
        c_f14 = S7 / S_HALF
        d10_boost *= (1.0 + c_f14)

    xgap = shell.cross_gap_correction()
    return max(0.0, ie * cap_s / ej_s * d10_boost * xgap)


def _madelung_d10s0_ea(Z: int, shell, ie: float) -> float:
    """EA for d10s0 Madelung element (Pd).

    PT mechanism: both d10 and s0 are closed/empty. The captured electron
    enters the empty s-orbital, creating s1.  This is pure tunneling into
    a vacant s-state through the d10 closure barrier.

    The amplitude is s-block alkali-like (s0→s1 creation) but attenuated
    by the d10 barrier: tunnel = sin²₅ × δ₅ (pentagon gap).
    """
    from ptc.constants import S_HALF, S3, S5, D5

    # Alkali-like capture into empty s-orbital
    # Factor: sin²₅ (d-shell coupling) × sin²₃ (p-channel propagation)
    # with spin doubling (1/s) since both spin channels are open for s0→s1
    # d10 self-screening: the tunnel traverses the closed pentagon
    # bidirectionally (in and out), losing δ₅ per passage → (1 - 2δ₅).
    tunnel_amp = S5 * S3 / S_HALF * (1.0 - 2.0 * D5)
    return max(0.0, ie * tunnel_amp)


def _madelung_d_halffill_ea(Z: int, shell, ie: float) -> float:
    """EA for d-half-fill Madelung elements (Cr d5s1, Mo d5s1).

    PT mechanism: the d-shell at half-fill has D_KL maximum on Z/10Z.
    Capture into d6 BREAKS the half-fill (costs exchange stability).
    Capture into s2 is the preferred channel — completes the s-pair
    without disturbing the d5 persistence peak.

    Attenuation: the d5 persistence peak ΔD_KL = log₂(1+1/5) RESISTS
    the s-pair capture.  Factor = 1 - ΔD_KL × s × (4/per)², where
    (4/per)² is the radial dilution (first d-shell at per=4).
    """
    from ptc.constants import S_HALF

    s_poly = shell.polygons[0]
    cap_s = s_poly.capture_amplitude()
    ej_s = s_poly.ejection_amplitude()
    if cap_s <= 0 or ej_s <= 0:
        return 0.0

    per = s_poly.per
    # d5 persistence barrier: ΔD_KL at half-fill attenuates capture
    d_poly = shell.polygons[2]  # d-shell polygon
    hund_barrier = d_poly.polygon_symmetry() * S_HALF * (4.0 / per) ** 2
    attenuation = 1.0 - hund_barrier

    xgap = shell.cross_gap_correction()
    return max(0.0, ie * cap_s / ej_s * attenuation * xgap)


def _madelung_partial_d_ea(Z: int, shell, ie: float, nd: int) -> float:
    """EA for Madelung d-promoted elements with partial d-shell (Nb, Ru, Rh, Pt).

    PT mechanism: the s-pair capture is boosted by the d-shell's feedback.
    Two phases on the pentagon Z/(2×5)Z:

    Hund phase (nd ≤ P₂ = 5): persistence growing → weak coupling
        coupling = sin²₅ × s × (nd/P₂)
        (Exchange pairs build up, coupling grows linearly with filling)

    Pairing phase (nd > P₂): persistence decreasing → stronger coupling
        coupling = sin²₅/s × (nd - P₂)/P₂
        (Pairing instability assists capture, factor 1/s vs s)

    Both scaled by radial support f_per = √(per_eff/2).
    Per ≥ 6: additional f14 buried resonance (sin²₇/s).
    """
    from ptc.constants import S_HALF, P2, S5, S7

    s_poly = shell.polygons[0]
    cap_s = s_poly.capture_amplitude()
    ej_s = s_poly.ejection_amplitude()
    if cap_s <= 0 or ej_s <= 0:
        return _closed_polygon_tunnel(Z, shell)

    per = s_poly.per
    # Phase-dependent coupling on the pentagon
    if nd <= P2:
        # Hund phase: weak coupling (exchange stability, factor s)
        c_partial = S5 * S_HALF * (nd / float(P2))
    else:
        # Pairing phase: stronger coupling (instability assists, factor 1/s)
        c_partial = S5 / S_HALF * ((nd - P2) / float(P2))

    # Radial support: first d-shell at per=4
    per_eff = per - 4 + 2
    f_per = math.sqrt(per_eff / 2.0)
    boost = 1.0 + c_partial * f_per
    # f14 buried resonance (per ≥ 6)
    if per >= 6:
        c_f14 = S7 / S_HALF
        boost *= (1.0 + c_f14)

    xgap = shell.cross_gap_correction()
    return max(0.0, ie * cap_s / ej_s * boost * xgap)


def EA_geo_eV(Z: int) -> float:
    """Electron affinity (eV) from polygon geometry.

    Parameters
    ----------
    Z:
        Atomic number (1–118).

    Returns
    -------
    float
        Estimated electron affinity in eV (≥ 0).
    """
    from ptc.periodic import _MADELUNG_PROMOTIONS

    shell = build_atomic_shell(Z)
    ie = IE_geo_eV(Z)

    # ── Madelung-promoted elements: dedicated PT channels ──
    if Z in _MADELUNG_PROMOTIONS:
        nd_mad, ns_mad = _MADELUNG_PROMOTIONS[Z]
        if nd_mad == 10 and ns_mad == 1:
            return _madelung_d10s1_ea(Z, shell, ie)
        if nd_mad == 10 and ns_mad == 0:
            return _madelung_d10s0_ea(Z, shell, ie)
        if nd_mad == 5 and ns_mad == 1:
            return _madelung_d_halffill_ea(Z, shell, ie)
        # Other promotions (Nb d4s1, Ru d7s1, Rh d8s1, Pt d9s1):
        # s-pair capture with partial d-closure boost.
        # The d-shell has nd_mad/10 of the closure coupling.
        return _madelung_partial_d_ea(Z, shell, ie, nd_mad)

    # ── Standard (non-promoted) elements ──
    from ptc.periodic import period as _per_fn
    per = _per_fn(Z)
    active = shell.active_polygon
    cap = active.capture_amplitude()
    ej = active.ejection_amplitude()

    if cap <= 0.0 or ej <= 0.0:
        # Closed polygon: try tunneling to next available shell.
        return _closed_polygon_tunnel(Z, shell)

    # ── Half-fill exchange lock (Hund's rule) ──
    # At exact half-fill (n_occ = capacity/2), ALL spins are aligned.
    # Adding an electron requires anti-parallel pairing against maximum
    # exchange stabilization.
    #
    # The lock applies ONLY at the smallest period for each block:
    #   p-block per=2 (N): polygon compact, exchange > capture → EA = 0
    #   p-block per≥3 (P, As): polygon diffuse, capture > exchange → EA > 0
    #   d-block per=4 (Mn): compact pentagon, exchange locked → EA = 0
    #   d-block per≥5 (Tc, Re): diffuse, capture wins → EA > 0
    from ptc.constants import P1 as _P1, P2 as _P2, P3 as _P3
    # Half-fill lock applies to MULTI-ELECTRON polygons only (prime >= P1).
    # The s-pair (prime=1, capacity=2) with 1 electron is NOT half-fill-locked:
    # a single electron has no exchange partner, so no Hund exchange cost.
    # N (p3), Mn (d5), Eu (f7): these have multiple electrons with parallel
    # spins that resist antiparallel pairing → genuine half-fill lock.
    _HALFFILL_MAX_PER = {_P1: 2, _P2: 4, _P3: 6}  # prime → max period for lock
    if active.prime >= _P1 and active.n_occupied == active.capacity // 2:
        max_per = _HALFFILL_MAX_PER.get(active.prime, 1)
        if per <= max_per:
            # ── f-block half-fill tunnel (Eu: 4f⁷ 6s²) ──
            # The f-shell is BURIED inside the s² pair (unlike valence p³/d⁵).
            # The captured electron tunnels through s² (transmission cos²₃)
            # and couples to the empty p-shell via P₂ channel (sin²₅).
            # EA = IE × sin²₅ × cos²₃ (P₂ × complement P₁)
            if active.prime == _P3 and per >= 6:
                from ptc.constants import S5 as _S5, C3 as _C3
                return max(0.0, ie * _S5 * _C3)
            return 0.0

    # EA = IE × (capture / ejection) on the active polygon,
    # boosted by cross-gap intercouche (NOT in screening for capture).
    xgap = shell.cross_gap_correction()
    return max(0.0, ie * cap / ej * xgap)


def benchmark_ea_geo() -> dict:
    """Benchmark EA_geo_eV against NIST electron affinity data.

    Only benchmarks elements where EA_NIST > 0 (excludes noble gases and
    elements with negative or negligible EA).

    Returns
    -------
    dict with keys:
        count       : int — number of elements benchmarked
        mae_percent : float — mean absolute error in percent
        by_block    : dict[str, float] — MAE per block (s, p, d, f)
        rows        : list[dict] — per-element details
    """
    from ptc.data.experimental import EA_NIST
    from ptc.periodic import block_of

    rows = []
    block_errors: dict[str, list[float]] = {}

    for Z, ea_ref in sorted(EA_NIST.items()):
        if ea_ref <= 0:
            continue
        ea_calc = EA_geo_eV(Z)
        err_pct = abs(ea_calc - ea_ref) / ea_ref * 100.0
        blk = block_of(Z)
        block_errors.setdefault(blk, []).append(err_pct)
        rows.append({
            "Z": Z,
            "block": blk,
            "ea_ref": ea_ref,
            "ea_calc": ea_calc,
            "err_pct": err_pct,
        })

    mae = sum(r["err_pct"] for r in rows) / len(rows) if rows else 0.0
    by_block = {blk: sum(errs) / len(errs) for blk, errs in block_errors.items()}

    return {
        "count": len(rows),
        "mae_percent": mae,
        "by_block": by_block,
        "rows": rows,
    }
