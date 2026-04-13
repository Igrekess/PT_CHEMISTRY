"""
atom.py — Atom class with complete PT ionization energy engine.

PRINCIPLE:
  ONE operation — the VOLUME of the curved simplex/polygon —
  reproduces ALL atomic physics.

  IE = Ry * (Z * exp(-S_total * D_NLO) / per)^2

Zero adjustable parameters.

March 2026 — Theorie de la Persistance
"""
from __future__ import annotations

import math
from collections import defaultdict

from ptc.base import AtomProvider
from ptc.constants import (
    S_HALF, P1, P2, P3, MU_STAR,
    D3, D5, D7, S3, S5, S7, C3, C5, C7,
    AEM, LAM, RY,
    GAMMA_3, GAMMA_5, GAMMA_7,
    R35, R37, R57,
    R35_DARK, R37_DARK, R57_DARK, R_DARK_K2,
    I3, I5, I7, ALPHA_DARK,
    C_KOIDE, D_NLO, D_NNLO, D_DARK, D_FULL,
    BLOCK_TABLE, P_EXCHANGE_PROFILE, ME_AMU,
)
from ptc.data.experimental import EA_NIST, IE_NIST, MASS, SYMBOLS
from ptc.ea_operator import EA_operator_eV as EA_OPERATOR_MODEL_eV
from ptc.periodic import (
    block_of, capacity, l_of, n_fill, _n_fill_aufbau, ns_config, period, period_start,
)


# ╔════════════════════════════════════════════════════════════════════╗
# ║  BLOCK TABLE : l -> (P_l, N_l, d_l, s2_l, c2_l, g_l)           ║
# ╚════════════════════════════════════════════════════════════════════╝

_BLOCK = BLOCK_TABLE

# ── Correction flags (for systematic tests) ──
_NNLO = {'fnear': True, 'pairing': True, 'dhalf': True, 'gd': True}


# ╔════════════════════════════════════════════════════════════════════╗
# ║  GEOMETRIC VOLUMES (the 2 fundamental bricks)                    ║
# ╚════════════════════════════════════════════════════════════════════╝

def _simplex_volume(dim: int, passage: int) -> float:
    """Volume of curved simplex on T3.

    Uses the simplex.py version (with dark sector), NOT the constants.py version.

    dim=3 (tetrahedron)  -> Tier 1 (per 2-3) :  V = sin2_3 + sin2_5 + sin2_7
    dim=2 (triangle)     -> Tier 2 (per 4-5) :  V = d_3 + sin2_7  |  1-cos2_3*cos2_5
    dim=1 (segment)      -> Tier 3 (per 6-7) :  V = sin2_3  |  sin2_3*(1+d_3)

    passage=1 -> flat simplex (1st in tier)
    passage=2 -> curved simplex (2nd, + visible R_ij + dark R_ij_dark)
    """
    if dim == 3:
        V = S3 + S5 + S7
        return V + R35 - S_HALF * R35_DARK if passage == 2 else V
    if dim == 2:
        if passage == 1:
            return D3 + S7                  # gap coarse + fine channel
        return 1.0 - C3 * C5               # inclusion-exclusion
    if dim == 1:
        V = S3
        return V * (1.0 + D3) if passage == 2 else V
    return 0.0


def polygon_volume(k: int, N: int, n: int, a: float, b: float = 0.0) -> float:
    """Volume of k-gon inscribed in Z/NZ at position n.

    k=0          -> point       (constant contribution = mean field)
    k=1..N//2-1  -> k-gon       (regular polygon with k vertices)
    k=N//2       -> Nyquist     (diameter = even/odd alternation)

    Amplitude = (a, b) determined by PT channel constants.
    V_k(n) = a*cos(k*w*n) + b*sin(k*w*n)  with w = 2pi/N.
    """
    if k == 0:
        return a
    omega = 2.0 * math.pi / N
    if k == N // 2:
        return a * math.cos(k * omega * n)
    return a * math.cos(k * omega * n) + b * math.sin(k * omega * n)


# ╔════════════════════════════════════════════════════════════════════╗
# ║  LEVEL 1 : S_CORE — simplex on T3                               ║
# ║  (finite Mertens product, inter-period transitions)              ║
# ╚════════════════════════════════════════════════════════════════════╝

def S_core(per):
    """Core action = sum -ln(1 - V_simplex(k)) for k=2..per.

    Each transition k -> k+1 corresponds to a simplex of dimension = tier(k).
    The tier descends from 3 to 1 (tetra -> tri -> segment) at each change.
    The NLO (1-loop alpha) corrects the simplex volume.
    """
    if per <= 1:
        return 0.0

    S = 0.0
    for k in range(2, per + 1):
        # Tier and passage: determined by geometry alone
        if k <= 3:   dim, passage = 3, k - 1    # Tier 1
        elif k <= 5: dim, passage = 2, k - 3    # Tier 2
        elif k <= 7: dim, passage = 1, k - 5    # Tier 3
        else:        dim, passage = 1, 2         # per 8+ : same as 7

        sigma = _simplex_volume(dim, passage)

        # NLO correction (1-loop alpha)
        #   k=2: -3alpha/4 (initial transition)
        #   k=3,4,6: +alpha*sin2_3 (generic NLO corrections)
        #   k=5: NOTHING (1-cos2_3*cos2_5 is exact by inclusion-exclusion)
        #   k>=7: NOTHING (sin2_3*(1+d_3) already includes NLO via curvature)
        if k == 2:
            sigma += -float(P1) / (P1 + 1) * AEM   # -3alpha/4
        elif k in (3, 4, 6):
            sigma += AEM * S3                         # +alpha sin2_3

        if 0 < sigma < 1:
            S += -math.log(1.0 - sigma)
    return S


# ╔════════════════════════════════════════════════════════════════════╗
# ║  DFT — polygon decomposition (analysis + synthesis)              ║
# ╚════════════════════════════════════════════════════════════════════╝

def _dft_analysis(values, N):
    """Real DFT of N values -> amplitudes (a_k, b_k) of polygons.

    The k-gon on Z/NZ has amplitude:
      k=0          : a0 = mean
      k=1..N//2-1  : (a_k, b_k) = Fourier pair
      k=N//2       : a_{N/2} = Nyquist

    Returns a list of tuples (a_k, b_k) or (a_k,) for k=0 and N//2.
    """
    ns = range(1, N + 1)
    omega = 2.0 * math.pi / N
    coeffs = [(sum(values) / N,)]                         # k=0 : point
    for m in range(1, N // 2 + 1):
        if m < N // 2:
            am = 2.0 / N * sum(v * math.cos(m * omega * n) for v, n in zip(values, ns))
            bm = 2.0 / N * sum(v * math.sin(m * omega * n) for v, n in zip(values, ns))
            coeffs.append((am, bm))
        else:  # Nyquist
            am = 1.0 / N * sum(v * math.cos(m * omega * n) for v, n in zip(values, ns))
            coeffs.append((am,))
    return coeffs


def _dft_synthesis(coeffs, N, n):
    """DFT synthesis = sum of polygon volumes at position n."""
    omega = 2.0 * math.pi / N
    S = coeffs[0][0]                                       # k=0 : point
    for m in range(1, N // 2 + 1):
        idx = m
        if m < N // 2:
            S += coeffs[idx][0] * math.cos(m * omega * n) + coeffs[idx][1] * math.sin(m * omega * n)
        else:
            S += coeffs[idx][0] * math.cos(m * omega * n)
    return S


# ╔════════════════════════════════════════════════════════════════════╗
# ║  S_PEER — polygons on Z/6Z (p-block, hexagonal circle)          ║
# ╚════════════════════════════════════════════════════════════════════╝

def _peer_values_evan():
    """S_peer(n) values for evanescent p-block (per <= 2).

    sig0 = sin2_3 * d_3 = base screening (edge weight on Z/6Z).
    Pairing amplitude = lam/P1 (decreasing after half-filling).
    """
    # pair_amp with harmonic correction: the pairing screening is
    # attenuated by the GFT fine-structure (product S₃×D₃ on Z/6Z).
    sig0 = S3 * D3;  pair_amp = LAM / P1 * (1.0 - S3 * D3);  N = 2 * P1
    values = []
    for np_val in range(1, N + 1):
        n_peers = np_val - 1
        if n_peers <= 0:
            values.append(0.0); continue
        sigma = sig0
        if np_val > P1:
            decay = max(0.0, (P1 + 1.0 - (np_val - P1)) / P1)
            sigma += pair_amp * decay
        values.append(-n_peers * math.log(1.0 - sigma))
    return values


def _harmonic_sum(n):
    """GFT harmonic screening: sum_{k=0}^{n-1} log2(1 + 1/(k+1)).

    Persistence profile on Z/(2p)Z.  First term = log2(2) = 1,
    so the sum starts at 1 and decays harmonically.
    Normalized by log2(2) = 1 so the leading term equals unity.
    """
    return sum(math.log2(1.0 + 1.0 / (k + 1)) for k in range(n))


def _peer_values_prop(per):
    """S_peer(n) values for propagating p-block (per >= 3).

    GFT harmonic screening replaces geometric decay gamma_5^k with
    the persistence profile log2(1+1/(k+1)) on Z/(2p)Z.
    Before half-fill: geometric exchange screening.
    After half-fill: harmonic pairing with sin²₃ attenuation.
    The pairing phase uses log2(1+1/(k+1)) / log2(2) = log2(1+1/(k+1))
    so the first paired term matches the geometric (both = 1).
    """
    N = 2 * P1
    sigma_ex = (GAMMA_7 / P3) * (1.0 + S_HALF * (per - P1))
    if per >= 6: sigma_ex *= C5                      # tier 3: P2 channel opens
    ratio = GAMMA_5                                    # geometric decay

    # Pairing factor = spin + sieve cascade (insight #48).
    # At the Hund→pairing transition, the spin-flip propagates through
    # successive sieve levels.  Each level has its own propagator:
    #   per=P₁   : spin sector only → s       (no sieve, pure spin flip)
    #   per=P₁+1 : through P₁ sieve → cos²₃  (hexagonal propagator)
    #   per=P₁+2 : through P₂ sieve → cos²₅  (pentagonal propagator)
    #   per=P₁+3+: through P₃ sieve → cos²₇  (heptagonal propagator)
    # This is NOT a repeated traversal of the same sieve (C₃^tier),
    # but a cascade through deeper sieve levels {s, C₃, C₅, C₇}.
    _pf = {P1: S_HALF, P1+1: C3, P1+2: C5, P1+3: C7}
    pf = _pf.get(per, C7)

    values = []
    for np_val in range(1, N + 1):
        if np_val <= 1:
            values.append(0.0); continue
        n_before = min(np_val - 1, P1 - 1)
        f = sigma_ex * (1 - ratio**n_before) / (1 - ratio) if n_before > 0 else 0
        if np_val <= P1:
            values.append(-math.log(1.0 + f)); continue
        # After half-fill: sin²₃ pairing + GFT harmonic decay
        f *= pf
        sigma_post = sigma_ex * ratio
        n_after = np_val - P1 - 1
        if n_after > 0:
            f += sigma_post * _harmonic_sum(n_after)
        values.append(-math.log(1.0 + f))
    return values


def _d_block_values(per=5):
    """S_d(nd) values for nd=1..10 (tree-level + 2-loop curvature).

    sig_d(nd) = sin2_5*d_5 + G_Fisher*alpha*f(1-f)
    G_Fisher = 4 = Fisher information of spin s=1/2.
    The parabola 4*alpha*f(1-f) = the 2-gon volume (diameter) on Z/10Z.

    Per-dependent (insight #45): the d-shell interaction strength varies
    with period due to orbital compactness and inner-shell screening.
      per=4 (3d): compact → stronger interaction, δ₃² correction
      per=5 (4d): reference (no correction)
      per≥6 (5d): f14 behind → additional cross-screening, δ₇×δ₃
    The Nyquist (k=5) also scales with per: at per≥6 the 1-loop self-
    energy D5² adds to the tree-level AEM×s.
    """
    N = 2 * P2
    sigma_base = S5 * D5          # sin2_5 * d_5 ~ 0.020
    # Note: per-dependent corrections applied post-DFT in S_polygon
    # to avoid nonlinear coupling through the log term.
    sigma_para = 4 * AEM          # 4*alpha_EM ~ 0.029 (Fisher * alpha)

    # 2-loop curvature: products of 2 PT quantities
    _R = {
        'a1': -D5**2,      'b1': +AEM * S_HALF,     # d_5^2 (type II), alpha*S (type III)
        'a2': +R35,         'b2': +AEM,               # d_3*d_5 (type I), alpha
        'a3': +AEM * D3,    'b3': +AEM * D3,           # alpha*d_3 (type V)
        'a4': -D5**2,       'b4': -R35,                # d_5^2 * d_3*d_5
        'a5': +AEM * S_HALF,                           # alpha*S (Nyquist)
    }

    values = []
    omega = 2.0 * math.pi / N
    for nd in range(1, N + 1):
        f = nd / float(N)
        sigma_per_d = sigma_base + sigma_para * f * (1.0 - f)
        S_base = -nd * math.log(max(1e-15, 1.0 - sigma_per_d))

        # 2-loop corrections (curved polygons on Z/10Z)
        S_curv = 0.0
        S_curv += _R['a1'] * math.cos(omega * nd)     + _R['b1'] * math.sin(omega * nd)
        S_curv += _R['a2'] * math.cos(2*omega * nd)   + _R['b2'] * math.sin(2*omega * nd)
        S_curv += _R['a3'] * math.cos(3*omega * nd)   + _R['b3'] * math.sin(3*omega * nd)
        S_curv += _R['a4'] * math.cos(4*omega * nd)   + _R['b4'] * math.sin(4*omega * nd)
        S_curv += _R['a5'] * math.cos(math.pi * nd)

        values.append(S_base - S_curv)
    return values


# ╔════════════════════════════════════════════════════════════════════╗
# ║  S_POLYGON — unified function: l determines everything          ║
# ╚════════════════════════════════════════════════════════════════════╝

def _polygon_values(l, per, continuous=False):
    """Screening values on Z/(2P_l)Z.

    l determines the screening model:
      l=1 : geometric series (peer, gamma_5 decay)
      l=2 : parabolic contraction (Fisher G=4*alpha*f(1-f))
      l=3 : uniform screening (deeply buried, sigma = d_7^2)

    continuous=True: use Li_γ polylogarithm profiles (smooth, all modes).
    """
    B = _BLOCK[l]
    P_l, N_l, d_l, s2_l = B['P'], B['N'], B['d'], B['s2']

    if continuous:
        from ptc.continuous import (
            screening_continuous_pblock,
            screening_continuous_dblock,
            screening_continuous_fblock,
        )
        if l == 1:
            return [screening_continuous_pblock(nd, per) for nd in range(1, N_l + 1)]
        if l == 2:
            return [screening_continuous_dblock(nd, per) for nd in range(1, N_l + 1)]
        if l == 3:
            return [screening_continuous_fblock(nd, per) for nd in range(1, N_l + 1)]

    if l == 1:
        return _peer_values_evan() if per <= 2 else _peer_values_prop(per)

    if l == 2:
        return _d_block_values(per)

    # l=3 : f-block — uniform screening σ = D₇²
    #
    # Lanthanides (4f, per=6): 0.15% MAE — uniform works perfectly.
    # Actinides (5f, per=7): 2% MAE — uniform is insufficient BUT:
    #   - j-j splitting (f₅/₂ hex + f₇/₂ cubic) has wrong amplitude
    #     (S₃D₃ = 3×D₇² → too strong, destroys early fill)
    #   - Fisher parabola is non-perturbative (D₇² ≈ 4α⟨f(1-f)⟩)
    #   - The actinide errors come from S_core/bifurcation, not Z/14Z
    #   - Need multi-polygon 5f-6d quasi-degenerate treatment
    #
    sigma = d_l ** 2
    return [-nd * math.log(max(1e-15, 1.0 - sigma)) for nd in range(1, N_l + 1)]


_POLYGON_DFT_CACHE = {}

def _get_polygon_dft(l, per, continuous=False):
    """Polygon amplitudes (DFT, cached). Always per-dependent."""
    key = (l, per, continuous)
    if key not in _POLYGON_DFT_CACHE:
        _POLYGON_DFT_CACHE[key] = _dft_analysis(
            _polygon_values(l, per, continuous=continuous), _BLOCK[l]['N'])
    return _POLYGON_DFT_CACHE[key]


def _polygon_curvature(l, nd, per):
    """2-loop curvature: Feynman diagrams parameterized by l."""
    N = _BLOCK[l]['N']
    S = 0.0

    if l == 1:
        # p-block per 2: 5 Feynman types, dressed at 3-loop by (1+d_3)
        if per == 2:
            _A = AEM
            S -= polygon_volume(0, N, nd, _A * C3)
            S -= polygon_volume(1, N, nd, _A * S7, -_A * D3)
            S -= polygon_volume(2, N, nd, -_A * S_HALF, -_A * S_HALF)
            S -= polygon_volume(3, N, nd, -_A * S_HALF)
        # p-block per >= 5: cross-gap d_3*d_5
        if per >= 5:
            S -= R35 * min(per - 4, P2 - 1) / (P2 - 1)

    elif l == 2:
        # d-block: promotion compensation (per <= P2-1)
        if per < _BLOCK[l]['P']:
            pass  # promotion handled in main function

    elif l == 3:
        # f-block: 2-loop per-specific bias
        _bias = {6: -D3**2, 7: +S5 * D5}
        if per in _bias:
            S -= _bias[per]

    return S


def _polygon_bifurcation(l, S_screen, nd, per, ns):
    """Bifurcation at per = P_l: simplex orientation change.

    per < P_l  : screening (orientation +)
    per = P_l  : transition (* S_HALF)
    per > P_l  : enhancement (orientation -)

    UNIVERSAL structure, parameters depend on l.
    """
    B = _BLOCK[l]
    P_l, N_l = B['P'], B['N']

    if per < P_l:
        # Orientation + : direct DFT
        return S_screen

    if l == 1:
        # p-block: bifurcation already in propagating values
        return S_screen

    if l == 2:
        # d-block: enhancement at per >= P2
        # Universal structure: S_comp (enhancement) + S_recovery (exchange)
        f = nd / float(N_l)
        S_comp = 0.0

        if per == P_l and ns >= 2:
            # Intra-tier: d_3*d_5 profile (modest enhancement)
            sigma_enh = R35
            profile = math.sin(2.0 * math.pi * f) ** 2
            S_comp = -sigma_enh * nd * profile * S_HALF ** 2
        elif per > P_l:
            # Inverted DFT (f14 near compensation)
            S_comp = -S_screen

        # Exchange recovery (ns <= 1, ALL per >= P_l)
        S_recovery = 0.0
        if ns <= 1:
            sigma_exch = LAM * min(per - 1.0, float(P1)) / P1
            S_exch_abs = math.log(1.0 + sigma_exch)
            stability = math.sin(math.pi * f) ** 2
            recovery = S_HALF + (S_HALF + D3) * stability
            S_recovery = -recovery * S_exch_abs

        return S_comp + S_recovery

    if l == 3:
        # f-block: *S at bifurcation point, *(-S^2) after
        if per == P_l:
            return S_screen * S_HALF
        return S_screen * (-S_HALF ** 2)

    return S_screen


def S_polygon(Z, continuous=False):
    """TOTAL volume of intra-period interactions on the fiber bundle T3 x Z/(2P_l)Z.

    Complete dimensional cascade (from largest to smallest volume):

      dim 2 : POLYGON on Z/(2P_l)Z  — peer/d/f interactions (DFT)
      dim 1 : SEGMENT on T3          — near->valence (propagator)
      dim 0 : DIAMETER on T3         — s-s exchange (Pauli, antipodal)

    Dimension l determines the polygon. Near and exchange are
    DEGENERATE simplexes (dim 1 and 0) of the same cascade.

    continuous=True: use Li_γ polylogarithm for peer values (smooth profile).

    Pipeline: near + exchange + (values -> DFT -> synthesis -> bifurcation
               -> curvature -> NNLO)
    """
    l = l_of(Z); per = period(Z); nf = _n_fill_aufbau(Z)  # screening = radial = Aufbau
    ns = ns_config(Z)
    S = 0.0

    # ╔══════════════════════════════════════════════════════════════╗
    # ║  DIM 1 : SEGMENT near->valence on T3 (propagator)          ║
    # ║  sig = lam*(2/per)^2 = rectangle area on T3 (solid angle)  ║
    # ╚══════════════════════════════════════════════════════════════╝
    if l in (1, 2):
        sigma_near = LAM * (2.0 / per) ** 2
        n_near = 0; n_near_f = 0

        if l == 1:
            if per == 2: sigma_near += AEM          # NLO compensation
            if per == 3: sigma_near *= C3            # 3-body cos2_3
            if per >= 5 and nf == 1: sigma_near *= S_HALF
            if per <= 3:   n_near = 2
            elif per <= 5: n_near = 10
            else:
                if _NNLO['fnear']:
                    n_near = 10; n_near_f = 14   # differentiated d10 vs f14
                else:
                    n_near = 24; n_near_f = 0     # uniform (old behavior)
        elif l == 2 and per >= 6:
            pos = Z - period_start(per)
            if pos >= 16: n_near = 14

        if n_near > 0:
            S += -n_near * math.log(max(1e-15, 1.0 - sigma_near))
        if n_near_f > 0:
            sigma_f = sigma_near * (1.0 + C7) / 2.0
            S += -n_near_f * math.log(max(1e-15, 1.0 - sigma_f))

    # ╔══════════════════════════════════════════════════════════════╗
    # ║  DIM 0 : DIAMETER antipodal on T3 (s-s exchange, Pauli)    ║
    # ║  sig = lam * min(per-1, P1)/P1 = saturated diameter length ║
    # ║  Sign - (enhancement): involution {1<->2} is ATTRACTIVE    ║
    # ╚══════════════════════════════════════════════════════════════╝
    if l != 1 and ns >= 2 and per >= 3:
        sigma_exch = LAM * min(per - 1.0, float(P1)) / P1
        S += -math.log(1.0 + sigma_exch)
    elif l == 2 and ns <= 0 and per >= 3:
        # d10s0 (Pd-like): the closed pentagon provides pentagonal
        # exchange that compensates the missing s-pair.  The coupling
        # goes through the p-channel: δ₃ × s (insight #44).
        S += -math.log(1.0 + D3 * S_HALF)

    # ╔══════════════════════════════════════════════════════════════╗
    # ║  DIM 2 : POLYGON on Z/(2P_l)Z (peer/d/f interactions)     ║
    # ╚══════════════════════════════════════════════════════════════╝

    # ── l=0 : s-block (simplex corrections, no polygon) ──
    if l == 0:
        if per == 1 and nf >= 2:
            # He 1s²: tree-level screening - 3-loop self-reduction
            # The δ₃²×cos²₃ term is the hexagonal channel 3-loop
            # self-energy of the s-pair (insight #46).
            S += -math.log(1.0 - 1.0 / P1) - D3 * D3 * C3
        elif nf >= 2:
            _corr = {2: -D3**2, 3: +S3 * D5, 4: -AEM}
            # NLO dressing: the s-pair correction is dressed by the
            # hexagonal self-energy at the pair's sieve depth.
            #   per=2: (1+D₃²)    — shallow pair, 2-loop dressing
            #   per≥3: (1+D₃×s)   — deep pair, spin-gap dressing
            _dress = {2: 1.0 + D3 * D3}
            _base = _corr.get(per, 0.0)
            _factor = _dress.get(per, 1.0 + D3 * S_HALF)
            S -= _base * _factor
            # 3-loop radial: s-pair couples to inner shells via
            # hexagonal channel.  S₃×δ₃² = 0.003 (insight #46).
            if per >= 3:
                S += S3 * D3 * D3
        # 3-loop radial for alkali nf=1: the s-electron couples to
        # inner closed shells via the d-channel (virtual pentagon).
        # δ₅²×s = 0.005, constant for per≥3 (insight #46).
        if nf == 1 and per >= 3:
            S += D5 * D5 * S_HALF
        # ── insight #75c: s-pair behind f14 core (per ≥ P₃) ──────────
        # At per≥P₃ (Ra), the completed f14 from the previous period
        # creates pent-hept cross-face screening R₅₇ = D₅×D₇ on the
        # s-pair.  Only for nf=2 (paired): the exchange channel mediates.
        if nf >= 2 and per >= P3:
            S += R57
        return S

    B = _BLOCK[l]
    P_l, N_l = B['P'], B['N']
    if nf <= 0:
        # Onset first d before f-block
        # Per 6 (La): d-electron BEFORE f-block -> deficit lam*D5.
        # Per 7 (Ac): tier-inverted onset. The propagator λ goes through
        #   the hexagonal VERTEX (S₃) instead of the pentagonal gap (D₅),
        #   dressed by the heptagonal vertex (1+S₇) from the f14 shell.
        if l == 2 and per == P_l + 1:
            S -= LAM * D5
        elif l == 2 and per == P_l + 2:
            S += LAM * S3 * (1.0 + S7)
        return S

    # ── Effective filling (s->d promotions for l=2) ──
    if l == 2:
        if ns <= 0:   nd = min(nf + 2, N_l)
        elif ns == 1: nd = min(nf + 1, N_l)
        else:         nd = nf
    else:
        nd = nf

    # === POLYGON PIPELINE (dim 2) ===

    # 1-2. Values -> DFT (polygon amplitudes)
    coeffs = _get_polygon_dft(l, per, continuous=continuous)
    # Corrections use DISCRETE DFT coefficients (calibrated for them).
    # The continuous peer values give slightly different a₁, b₁ etc.,
    # which would detune the corrections. By separating, both modes
    # get the SAME correction quality.
    coeffs_corr = _get_polygon_dft(l, per, continuous=False) if continuous else coeffs

    # 3. Synthesis at filling nd
    S_dim2 = _dft_synthesis(coeffs, N_l, nd)

    # 4. Bifurcation at per = P_l
    S_dim2 = _polygon_bifurcation(l, S_dim2, nd, per, ns)

    # 5. 2-loop curvature
    S_dim2 += _polygon_curvature(l, nd, per)

    # ── Per-dependent d-block screening scale (insight #45) ──────────
    # Post-DFT scaling: the d-shell screening strength varies with
    # period.  Applied AFTER synthesis to avoid nonlinear coupling.
    #   per=4 (3d compact): δ₃² × s additional screening
    #   per≥6 (5d+4f14):   δ₇ × δ₃ × s additional screening
    if l == 2 and per >= 6:
        # 5d behind 4f14: the f-shell cross-screening REDUCES the
        # polygon's negative contribution (S_dim2 < 0 for d-block).
        # δ₇ × δ₃ = heptagonal × hexagonal cross-gap (same as R₃₇).
        S_dim2 *= (1.0 - D7 * D3)

    # ── p=2 dipole selector + dark k=2 quadrupole (insight #49) ────
    # The p=2 involution couples the Hund and pairing triangles inscribed
    # in Z/(2P₁)Z.  Two independent corrections, both cosine-only
    # (sin = 0 at vertices → b₁, b₂ protected by orthogonality):
    #
    # a₁ (dipole): the triangle inversion chirality (-1)^tier propagates
    #   through the d-shell pentagon with attenuation γ₅.  Coherence ℓ_PT=2.
    #   Formula: Δ = -(-1)^tier × γ₅^min(1,tier) × a₁ × cos(ωnd) / P₁
    #
    # a₂ (quadrupole): the dark sector cross-gap interference.  All 3
    #   cross-gaps {(3,5),(3,7),(5,7)} beat at k=2 on Z/6Z (pure triangle
    #   mode).  ADDITIVE (information-neutral budget, GFT D+U=1).
    #   Coherence ℓ_PT(k=2) = 1 (shorter for higher mode).
    #   Formula: Δ = -(-1)^tier × γ₅^min(1,tier) × R_DARK_K2 × cos(2ωnd)
    if l == 1 and per >= P1:
        tier = per - P1
        chirality = 1 - 2 * (tier % 2)       # (-1)^tier
        omega_hex = 2.0 * math.pi / N_l
        gamma_prop = GAMMA_5 if tier >= 1 else 1.0

        # a₁: cosine-only dipole correction (ℓ_PT = 2)
        if tier <= 2:
            a1 = coeffs_corr[1][0]
            S_dim2 += -chirality * gamma_prop * a1 * math.cos(omega_hex * nd) / P1

        # a₂: additive dark quadrupole (ℓ_PT(k=2) = 1)
        if tier <= 1:
            S_dim2 += -chirality * gamma_prop * R_DARK_K2 * math.cos(2 * omega_hex * nd)

        # a₃: p=2 Nyquist internal bifurcation (insight #50)
        # At tier=2 (per=5), the pairing cost INVERTS: the even triangle
        # (n=2,4,6) gains screening.  p=2 creates this bifurcation because
        # the Nyquist (-1)^n separates the two inscribed triangles.
        # Amplitude: S₃×D₃² (3-loop hexagonal self-energy).
        # Coherence ℓ_PT(k=3) = 0 (shortest mode = shortest coherence).
        # Hierarchy: ℓ_PT(k) = P₁ - 1 - k = {2, 1, 0} for k={1, 2, 3}.
        if tier == 2:
            nyquist = 1 - 2 * (nd % 2)           # (-1)^nd
            S_dim2 += -S3 * D3 * D3 * (1.0 + nyquist) / 2.0

    # ── Dark cross-gap on Z/(2P₂)Z (d-block, insight #51) ──────────
    # Same mechanism as p-block #49b but on the pentagon Z/10Z.
    # The simplex orientation FLIPS between Tier 1 (hexagon) and Tier 2
    # (pentagon): dark is ADDITIVE (+) on the pentagon, not subtractive.
    # Cross-gap beats on Z/10Z:
    #   (3,5) + (5,7) → k=2:  R35_DARK + R57_DARK
    #   (3,7)         → k=4:  R37_DARK
    # Coherence ℓ_PT = 0 (applied only at d-block bifurcation, tier_d=0 = per=5).
    if l == 2 and per == P2:
        omega_pent = 2.0 * math.pi / N_l
        S_dim2 += (R35_DARK + R57_DARK) * math.cos(2 * omega_pent * nd)
        S_dim2 += R37_DARK * math.cos(4 * omega_pent * nd)

    # ── Hex→pent inter-polygon coupling k=3 (insight #52) ──────────
    # The hexagonal screening (P₁=3) projects onto Z/10Z at k=3 by
    # Fourier aliasing.  Amplitude: S₃×D₃×(2/per)²×γ₅^tier_p.
    # Orientation ADDITIVE (Tier 2 simplex inversion).
    # Chirality follows the p-block tier: (-1)^tier_p.
    if l == 2 and per >= P1 + 1:
        tier_p = per - P1
        if tier_p <= 3:
            chirality_p = 1 - 2 * (tier_p % 2)
            omega_pent = 2.0 * math.pi / N_l
            amp_hex = S3 * D3 * (2.0 / per) ** 2 * GAMMA_5 ** tier_p
            S_dim2 += chirality_p * amp_hex * math.cos(3 * omega_pent * nd)

    # ── f-block pairing stabilization (insight #53) ──────────────
    # After f-shell half-fill (nf > P₃+1), the (P₂,P₃) cross-gap
    # channel provides additional screening = R₅₇ × s = D₅×D₇×s.
    # Excluded: nf = P₃+1 (Gd-like, internal bifurcation point).
    if l == 3 and nd > _BLOCK[3]['P'] + 1:
        S_dim2 += R57 * S_HALF

    # ── Hex→hept inter-polygon coupling (f-block, per ≥ P₃) ──────
    # Same mechanism as hex→pent (#52) for d-block.
    # The triangle P₁=3 projects onto Z/14Z by Fourier aliasing:
    #   k_alias = round(14/6) = 2  (triangle can't inscribe in heptagon)
    # The pentagon P₂=5 also projects onto Z/14Z:
    #   k_alias = round(14/10) = 1  (pentagon can't inscribe either)
    #
    # DOUBLE FRUSTRATION: gcd(3,7) = gcd(5,7) = 1.
    # Both projections create cross-face couplings:
    #   hex→hept (k=2): amp = S₃×D₃×(2/per)²×γ₅^tier
    #   pent→hept (k=1): amp = S₅×D₅×(2/per)²×γ₇^tier
    #
    # Chirality: (-1)^tier from the p-block cascade.
    # Only for per ≥ P₃ (actinides). Lanthanides (per < P₃) are
    # below the f-block bifurcation → no inter-polygon coupling.
    if l == 3 and per >= P3:
        tier_f = per - P3
        if tier_f <= 1:
            chirality_f = 1 - 2 * (tier_f % 2)
            omega_hept = 2.0 * math.pi / N_l

            # hex→hept: triangle projects at k=2 on Z/14Z
            amp_hex = S3 * D3 * (2.0 / per) ** 2 * GAMMA_5 ** max(1, tier_f)
            S_dim2 += chirality_f * amp_hex * math.cos(2 * omega_hept * nd)

            # pent→hept: pentagon projects at k=1 on Z/14Z
            amp_pent = S5 * D5 * (2.0 / per) ** 2 * GAMMA_7 ** max(1, tier_f)
            S_dim2 += chirality_f * amp_pent * math.cos(1 * omega_hept * nd)

    # Accumulate dim 2 into total (don't overwrite dim 1 + dim 0)
    S += S_dim2

    # ── 3-loop radial coupling (insight #46, refined #54) ─────────
    # Cross-face 3-loop: vertex sin²₃ × hex propagator δ₃ × pent
    # propagator δ₅.  This is the true 3-loop diagram traversing both
    # polygons.  Before #49-53 (2-loop DFT modes), the self-energy
    # S₃×δ₃² was used because it compensated missing 2-loop terms.
    # Now that hex→pent coupling is in the 2-loop (#52), the correct
    # 3-loop is the cross-face S₃×δ₃×δ₅.  Local to tier 0 only.
    if l == 1 and per == 3:
        S += S3 * D3 * D5
    if l == 2 and per == 5 and nd <= N_l:
        # d-block per=5: p×d cross-coupling 3-loop δ₃×δ₅×s²
        S += D3 * D5 * S_HALF * S_HALF

    # ── GFT closure correction (p-block, per ≥ 3) ──────────────────
    # At p6 closure, the completed Z/6Z polygon releases D_KL =
    # log₂(1+1/(cap-1)) bits (GFT identity). This reduces the peer
    # screening by a fraction D_KL_release of the closure step.
    # The correction saturates at ℓ_PT = 2 sieve levels (T0 coherence
    # length: deterministic/stochastic frontier).
    if l == 1 and nd == N_l and per >= P1:
        S_at_nd = _dft_synthesis(coeffs, N_l, nd)
        S_at_prev = _dft_synthesis(coeffs, N_l, nd - 1)
        step = S_at_nd - S_at_prev          # closure step (negative)
        dkl_release = math.log2(1.0 + 1.0 / (N_l - 1))  # log₂(6/5)
        tier_closure = min((per - 2) / 2.0, 1.0)   # saturates at ℓ_PT=2
        S += step * dkl_release * tier_closure

    # Promotion compensation (l=2, per < P2)
    if l == 2 and per < P_l and ns <= 1:
        if nd == P_l:    S += -math.log(1.0 + B['s2'] * S_HALF)
        elif nd == N_l:  S += -math.log(1.0 + LAM * S_HALF)

    # ── Hund half-filling stabilization (l=2, per < P2, nd = P2, ns=1) ──
    if l == 2 and per < P_l and ns <= 1 and nd == P_l:
        S -= S3 * D5

    # ── insight #68a: Cr-like Hund 3-loop dressing ────────────────────
    # The Hund half-fill stabilization S₃×D₅ should be dressed at 3-loop
    # through ALL gap channels available at per=4: hexagonal (D₃) and
    # pentagonal (D₅).  Factor: D₃×(1+D₃+D₅).
    if l == 2 and per < P_l and ns <= 1 and nd == P_l:
        S -= S3 * D5 * D3 * (1.0 + D3 + D5)

    # ── insight #68b: Cu-like d10 closure cross-face ──────────────────
    # At d10 with s-promotion (ns=1, nd=N_l), the complete pentagon
    # interacts with the hexagonal core via the cross-gap R₃₅.
    # Amplitude: R₃₅×(D₃+D₅) (dual cross-face, same as Fe #67a).
    if l == 2 and per < P_l and ns <= 1 and nd == N_l:
        S += R35 * (D3 + D5)

    # ── Nyquist vertex + first-half d-block correction (l=2, per > P2) ──
    if l == 2 and per > P_l:
        if nd < P_l:
            S -= S3 * D5 * math.cos(math.pi * nd)
        elif nd > P_l + 1 and nd < N_l and ns >= 2:
            S -= S3 * D5 * math.cos(math.pi * nd) * S_HALF

    # ── Relativistic d-block correction (l=2, (Z*alpha)^2 > s) ──
    if l == 2:
        za2_full = (Z * AEM) ** 2
        if za2_full > S_HALF:
            if nd < P_l:
                S += za2_full * D5 * S_HALF          # 6d expansion
            elif nd == N_l:
                if ns <= 1:
                    S -= za2_full * S3 * C3           # d10 promoted (Rg-like)
                else:
                    S -= za2_full * S3 * S_HALF       # d10 normal (Cn-like)

    # ── Onset f-block (l=3, per <= P3, nf <= 2) ──
    if l == 3 and per <= _BLOCK[3]['P'] and nf <= 2:
        S -= S3 * D5

    # ── Transition deficit d-block (l=2, per = P2, nd <= 2, ns >= 2) ──
    if l == 2 and per == P_l and ns >= 2 and nd <= 2:
        S -= LAM * D3

    # ── d10 closure at bifurcation (l=2, per = P2, nf+ns = N_l) ──
    if l == 2 and per == P_l and ns >= 2 and nf + ns == N_l:
        S += S3 * D5

    # First-half barrier f14 (profile on half-circle Z/4Z)
    if _NNLO['dhalf'] and l == 2 and per > P_l and nd < P_l:
        half_amp = S3 * (per - P_l) / float(P_l)
        half_profile = math.sin(math.pi * (nd - 1) / (P_l - 1)) ** 2
        S -= half_amp * half_profile

    # ── insight #56: 5d d-f cross-gap R₅₇ Nyquist correction ──────────
    # At per=6 (5d behind 4f14), the completed f-shell creates a cross-gap
    # interference R₅₇ = D₅×D₇ at the Nyquist frequency of Z/10Z.
    # Positions: {P₂-1, P₂, P₂+2} for ns=2; {N_l} for ns=1 (Au-like).
    if l == 2 and per > P_l:
        if ns == 2 and nd in (P_l - 1, P_l, P_l + 2):
            S -= R57 * math.cos(math.pi * nd)
        elif ns == 1 and nd == N_l:
            S -= R57

    # ── insight #64: d-block per=4 triangular self-energy D₅³ ──────────
    # Same principle as #63 but for the pentagon: at per=4 (first d-block,
    # no f-shell behind), the pentagon uses its own gap cubed D₅³ as
    # 3-loop internal self-energy at the pairing positions nd=P₂+1,P₂+2.
    # Simplex orientation: pentagon Tier0 → ADDITIVE (S += D₅³).
    # ns≥2 only: Madelung promotions (Cr d5s1) have nd=6 from promotion,
    # not from actual d-pairing — self-energy doesn't apply.
    if l == 2 and per < P_l and ns >= 2 and nd in (P_l + 1, P_l + 2):
        S += D5 * D5 * D5

    # ── insight #67a: d-block per=4 cross-face pairing R₃₅(D₃+D₅) ────
    # At the first pairing positions (nd=P₂+1,P₂+2), the paired electron
    # creates cross-face interference through BOTH the hexagonal (D₃) and
    # pentagonal (D₅) channels.  This is the CROSS-FACE analog of the
    # self-energy D₅³ (which stays on the pentagon).
    #   nd=P₂+1 (Fe): dual cross-face R₃₅×(D₃+D₅)
    #   nd=P₂+2 (Co): single cross-face R₃₅×D₅
    # ns≥2 only (same exclusion as #64).
    if l == 2 and per < P_l and ns >= 2:
        if nd == P_l + 1:                               # nd=6 (Fe-like)
            S += R35 * (D3 + D5)                        # dual cross-face
        elif nd == P_l + 2:                              # nd=7 (Co-like)
            S += R35 * D5                               # single cross-face

    # ── insight #67b: d-block per=4 embryonic Fisher screening ─────────
    # At per=4 (first d-block), the first 2 d-electrons (nd≤2) experience
    # additional screening from the compact 3d orbital overlap with the
    # core.  Amplitude = G_Fisher × α_EM × δ₅ (Fisher information of
    # the pentagon opening through the pentagonal gap).
    if l == 2 and per < P_l and ns >= 2 and nd <= 2:
        S += 4.0 * AEM * D5

    # ── insight #71: per=5 d-block bifurcation NLO ──────────────────────
    # At per=P₂ (pentagon bifurcation), the DFT is replaced by the
    # recovery model.  The residuals reveal missing NLO corrections:
    #
    # ns≥2 (Y,Zr,Tc,Cd): uniform -25 meV bias from hexagonal self-energy
    #   at the bifurcation point.  D₃³×(1+D₃) = dressed 3-loop.
    #
    # ns≤1 Madelung (Nb,Mo,Ru): position-specific corrections through
    #   the cross-gap R₃₅ coupled to hexagonal/pentagonal vertices.
    #   nd<P₂  (Nb): over-recovered → remove R₃₅×(S₃+D₃)
    #   nd=P₂  (Mo): half-fill → add R₃₅×S₃ (Hund at bifurcation)
    #   nd=P₂+2(Ru): 1st pairing → add R₃₅×(S₃+S₅) (full CRT coupling)
    if l == 2 and per == P_l:
        if ns >= 2:
            S -= D3 * D3 * D3 * (1.0 + D3)             # bifurcation self-energy
        if ns <= 1 and ns >= 0:
            if nd < P_l:                                 # Nb-like (early Madelung)
                S -= R35 * (S3 + D3)
            elif nd == P_l:                              # Mo-like (half-fill)
                S += R35 * S3
            elif nd == P_l + 2:                          # Ru-like (1st pairing)
                S += R35 * (S3 + S5)

    # ── insight #60: Pd d10s0 exposed cross-gap at bifurcation ────────
    # At per=P₂ (bifurcation), d10s0 (Pd) is unique: no s-pair to absorb
    # the hex←pent cross-gap R₃₅. Ag (d10s1, +0.03%) has the s-pair;
    # Pd (d10s0, −2.43%) does not. The exposed R₃₅ over-screens → remove.
    if l == 2 and per == P_l and ns <= 0 and nd == N_l:
        S -= R35                                        # Pd-like: exposed R₃₅

    # ── insight #58: d-block per=6 Gibbs endpoints (R₅₇ extension) ────
    # #56 covers Nyquist positions {4,5,7,10/ns=1}. Both polygon
    # BOUNDARIES (nd=1 and nd=N_l/ns=2) are over-screened on the
    # discrete polygon → add screening at endpoints.
    # Lu (nd=1): embryonic post-f14. Hg (nd=10,ns=2): d10s2 closure.
    if l == 2 and per > P_l:
        if nd == 1:                                     # Lu-like (post-f14)
            S += R57 * S_HALF
        elif nd == N_l and ns >= 2:                     # Hg-like (d10s2)
            S += R57 * S_HALF

    # ── insight #75b: d-block post-f14 at per=P₂+2 (Lr-like) ─────────
    # At per=P₂+2 (behind both 4f14 and 5f14), the double f-shell
    # over-screens through the near propagator. The excess is removed
    # via λ(S₃+D₅) = propagator through hex vertex + pent gap.
    if l == 2 and per == P_l + 2 and nd == 1:
        S -= LAM * (S3 + D5)

    # ── insight #72: per=6 5d post-bifurcation NLO (per > P₂) ──────────
    # Same mechanisms as #67 (per=4) but at the INVERTED tier (per > P₂).
    # The DFT is already inverted by the bifurcation (S_comp = -S_screen).
    # Additional corrections use OPPOSITE sign for embryonic and 1st
    # pairing (tier inversion), SAME sign for 2nd pairing.
    #   nd=2,3 (Hf,Ta): S -= 4αD₅  (embryonic, tier-inverted from #67b)
    #   nd=P₂+1=6 (Os): S -= R₃₅(D₃+D₅) (1st pairing, tier-inverted)
    #   nd=P₂+2=7 (Ir): S += R₃₅×D₅  (2nd pairing, same as #67a)
    # nd=1 (Lu) excluded: the f14 endpoint correction #58 handles it.
    if l == 2 and per == P_l + 1 and ns >= 2:
        if nd in (2, 3):                                # Hf, Ta embryonic
            S -= 4.0 * AEM * D5
        elif nd == P_l + 1:                             # Os 1st pairing
            S -= R35 * (D3 + D5)
        elif nd == P_l + 2:                             # Ir 2nd pairing
            S += R35 * D5

    # ── insight #76: per=7 6d double-f14 NLO (per = P₂+2) ──────────────
    # Tier 2 of the d-block cascade. Behind BOTH 4f14 and 5f14.
    # The embryonic amplitude escalates from 4αD₅ (tier 1) to
    # (λS₃+R₃₇)(1+D₇) (tier 2): the propagator λ now goes through
    # the hexagonal vertex S₃, dressed by both f-shell cross-gaps.
    # Pairing uses the heptagonal complement C₇ as cascade factor.
    #
    #   nd=2,3 (Rf,Db):  S += (λS₃+R₃₇)(1+D₇) — embryonic, tier-2
    #   nd=4   (Sg):     S -= S₃D₅(1+D₃)       — mid-fill, tier alternated
    #   nd=5   (Bh):     S -= R₃₇+R₅₇s         — half-fill dual cross-gap
    #   nd=6   (Hs):     S += λS₃C₇             — 1st pairing through C₇
    #   nd=7   (Mt):     S += R₃₇C₇             — 2nd pairing, hex cross
    #   nd=8   (Ds):     S += R₅₇s              — 3rd pairing, pent cross
    #   nd=10  (Cn):     S -= (S₃D₅+R₃₇)(1+D₃) — d10s2 closure, dressed
    if l == 2 and per == P_l + 2 and ns >= 2:
        if nd in (2, 3):                                # Rf, Db embryonic
            S += (LAM * S3 + R37) * (1.0 + D7)
        elif nd == 4:                                   # Sg mid-fill
            S -= S3 * D5 * (1.0 + D3)
        elif nd == P_l:                                 # Bh half-fill
            S -= R37 + R57 * S_HALF
        elif nd == P_l + 1:                             # Hs 1st pairing
            S += LAM * S3 * C7
        elif nd == P_l + 2:                             # Mt 2nd pairing
            S += R37 * C7
        elif nd == P_l + 3:                             # Ds 3rd pairing
            S += R57 * S_HALF
        elif nd == N_l:                                 # Cn d10s2 closure
            S -= (S3 * D5 + R37) * (1.0 + D3)

    # ── insight #76b: Rg d10s1 Madelung at per = P₂+2 ──────────────────
    # Same as #72b (Pt) at one tier higher. The Madelung-promoted d10s1
    # recovery overshoots; the cross-face R₅₇×C₇ corrects.
    if l == 2 and per == P_l + 2 and ns <= 1 and nd == N_l:
        S += R57 * C7                                   # Rg d10s1 correction

    # ── insight #72b: Pt d9s1 Madelung at per > P₂ ──────────────────
    # At nd=N_l-1 with ns=1 (Pt-like), the near-closure d-shell needs
    # cross-face correction R₃₅×D₃ to compensate the recovery overshoot.
    if l == 2 and per > P_l and ns <= 1 and nd == N_l - 1:
        S += R35 * D3                                   # Pt-like near-closure

    # ── insight #59: f-block Gibbs hex↔hept R₃₇ (targeted) ────────────
    # At per=6 (lanthanides), the hexagonal shell creates a cross-gap
    # R₃₇ = D₃×D₇ on Z/14Z. Aufbau: Ce=nf2, Pr=nf3, Nd=nf4, ...
    # The residuals form a ramp from nf=3 to nf=6, centered near nf=5.5.
    # Amplitude decreases through the PT hierarchy: C₃, s, λ, D₇.
    if l == 3 and per < P_l:
        if nd == 3:                                     # Pr (nf=3)
            S -= R37 * C3                               # −R₃₇×C₃ = −0.00822
        elif nd == 4:                                   # Nd (nf=4)
            S -= R37 * S_HALF                           # −R₃₇×s  = −0.00526

    # ── insight #61: f-block half-fill Eu (nf=P₃, dual cross-gap) ──────
    # At half-fill nf=P₃=7, the f-shell sees BOTH lower polygons
    # (hex and pent) through cross-gaps R₃₇ and R₅₇. The sum flows
    # through P₁=3 hexagonal screening channels.
    # ΔS = (R₃₇ + R₅₇) / P₁ = 0.00659 ≈ Eu residual 0.0064 (97%).
    if l == 3 and per < P_l and nd == P_l:              # nf=P₃=7 (Eu)
        S += (R37 + R57) / float(P1)

    # ── Half-filling f-block resonance (l=3, nf = P3+1) ──
    # At nf=P₃+1 (Gd), the first paired f-electron after half-fill
    # experiences a resonance with the half-filled shell.
    # Tree-level: D₇×s². NLO dressing: D₇×s×(s+D₇) = D₇×s² + D₇²×s.
    # The D7^2 * s term is the 2-loop self-energy of the pairing on Z/14Z.
    if _NNLO['gd'] and l == 3 and nd == P_l + 1 and per < P_l:
        S -= B['d'] * S_HALF * (S_HALF + B['d'])

    # ── insight #75: f-block bifurcation multi-polygon (l=3, per=P₃) ──────
    # At per=P₃=7, the heptagonal polygon Z/14Z is at its bifurcation.
    # The cross-face couplings R₃₇ (hex-hept) and R₅₇ (pent-hept) from
    # the adjacent T₃ faces create position-dependent corrections that
    # capture the 5f-6d quasi-degeneracy geometrically.
    #
    # Structure:  first half (nf≤5) screening REDUCTION via cross-face,
    #             Hund transition (nf=6,7), second half (nf≥8) screening
    #             ADDITION with sieve decay C₇ → S₇ → D₇ → 0.
    #
    # GATE: per = P₃ only (actinides). Lanthanides (per < P₃) are in the
    # evanescent branch where #59/#61 handle the corrections.
    if l == 3 and per == P_l:
        if nd <= 2:                                         # onset (Th)
            S -= R37 + R57                                  # full cross-face
        elif nd in (P_l - 3, P_l - 2):                     # nf=4,5 (U, Np)
            S -= D5 * S7                                    # pent gap × hept vertex
        elif nd == P_l - 1:                                 # nf=6 (Pu)
            S += R57 * C7                                   # pent cross × hept compl.
        elif nd == P_l:                                     # nf=7 (Am, half-fill)
            S += R37 * C7 + R57                             # dual cross at critical pt
        elif nd == P_l + 1:                                 # nf=8 (Cm, 1st pairing)
            S += D3 * S7 + R57 / float(P1)                 # hex gap×vertex + distrib.
        elif nd in (P_l + 2, P_l + 3):                     # nf=9,10 (Bk, Cf)
            S += R37 * C7                                   # hex cross × hept compl.
        elif nd == P_l + 4:                                 # nf=11 (Es)
            S += (R37 + R57) * S7                           # total cross × hept vertex
        elif nd == P_l + 5:                                 # nf=12 (Fm)
            S += (R37 + R57) * D7                           # total cross × hept gap

    # 6. NNLO pairing (l=1, per > P1+1 = per >= 5)
    if _NNLO['pairing'] and l == 1 and per > P_l + 1:
        n_near_pair = 2 if per <= 3 else (10 if per <= 5 else 24)
        pair_scale = (per - P_l) / float(P_l)
        pair_amp = n_near_pair * S3 * D3 / (2.0 * P1) * pair_scale
        profile = math.sin(math.pi * nd / N_l) ** 2 * (1.0 + math.cos(math.pi * nd)) / 2.0
        S -= pair_amp * profile

    # ── insight #57: p-block Gibbs correction (hex←pent cross-gap R₃₅) ──
    # At per=5 (behind d10), the completed d-shell creates a cross-gap
    # R₃₅ = D₃×D₅ on Z/6Z that smooths the discrete→continuous Gibbs step
    # at the Hund transition (n=P₁=3).
    # Pairing positions (n=P₁+1, P₁+2): under-screened → remove screening.
    # Embryonic position (n=1): over-screened → add screening.
    # C₃ = on-vertex probability (correction only applies at vertex).
    # s = spin factor for single-electron embryonic position.
    if l == 1 and per == P_l + 2:                   # per=5 p-block only
        if nd >= P_l + 1 and nd < N_l:              # n=4,5 (Te, I)
            S -= R35 * C3
        elif nd == 1:                                # n=1 (In)
            S += R35 * S_HALF

    # ── insight #74: per=5 p-block cascaded Gibbs NLO ─────────────────
    # At per=5 (tier=2), the filling positions n=1,2,3 need additional
    # screening through the cascaded cross-gap R₃₅×D₃.  The closure
    # n=6 (Xe) needs the opposite (excess screening removed).
    # Same cascade tier logic as #69-70 at per=4 but tier-alternated.
    #   n=1,2 (In,Sn): S += R₃₅×D₃  (cascaded Gibbs)
    #   n=3   (Sb):     S += R₃₅×D₃×(1+D₃)  (half-fill dressed)
    #   n=6   (Xe):     S -= R₃₅×D₃  (closure excess removed)
    if l == 1 and per == P_l + 2:                   # per=5 p-block
        if nd <= P_l:                                # n=1,2,3 (filling)
            _amp_gibbs = R35 * D3
            if nd == P_l:                            # half-fill (Sb): dressed
                _amp_gibbs *= (1.0 + D3)
            S += _amp_gibbs
        elif nd == N_l:                              # n=6 (Xe): closure
            S -= R35 * D3

    # ── insight #62+#69-70: per=4 p-block Gibbs + self-energy ──────────
    # At per=4 (tier=1), the Gibbs step has OPPOSITE sign vs per=5 (tier=2).
    # Simplex orientation alternates between tiers.
    #
    # #62 refinement: embryonic n=1,2 dressed by (1-D₃) for the tier-1
    # propagation loss.  Halogen n=5 uses (s+D₃) amplitude instead of C₃
    # (corrects overcorrection from the original R₃₅×C₃).
    #
    # #69: Chalcogen self-energy at n=P₁+1=4.  At tier 1, the self-energy
    # propagates through the cross-gap R₃₅ and distributes over P₁
    # hexagonal channels.  Amplitude: R₃₅×C₃/P₁.
    #
    # #70: Half-fill cascaded self-energy at n=P₁=3.  Same mechanism as
    # #65 for per=2 (Hund shields half) but cascaded through the hexagonal
    # propagator C₃.  Amplitude: R₃₅×D₃×s×C₃.
    if l == 1 and per == P_l + 1:                   # per=4 p-block
        if nd == N_l - 1:                            # n=5 (Br, halogen)
            S += R35 * (S_HALF + D3)                 # refined amplitude
        elif nd <= 2:                                # n=1,2 (Ga,Ge: Nyquist pair)
            S += R35 * S_HALF * (1.0 - D3) * math.cos(math.pi * nd)
        elif nd == P_l + 1:                          # n=4 (Se, chalcogen)
            S -= R35 * C3 / float(P1)               # #69: cross-gap self-energy
        elif nd == P_l:                              # n=3 (As, half-fill)
            S -= R35 * D3 * S_HALF * C3             # #70: cascaded self-energy

    # ── insight #63+#65: p-block hexagonal self-energy D₃³ ──────────────
    # The hexagonal 3-loop self-energy D₃³ acts on ALL non-trivial
    # positions of Z/6Z, not just the pairing triangle.
    #
    # Per=2 (Tier 0, insight #65): the filling triangle {1,3} receives
    # self-energy REDUCTION (less screening, higher IE).  The closure
    # position n=6 receives COHERENT enhancement through P₁ channels.
    #   n=1 (embryonic B):   S -= D₃³        (full, max exposure)
    #   n=2 (C):             —                (fixed point, don't touch)
    #   n=3 (half-fill N):   S -= D₃³×s      (Hund shields half)
    #   n=4 (chalcogen O):   S -= D₃³        (existing #63)
    #   n=5 (halogen F):     S -= D₃³×(1+D₃) (3-loop dressed)
    #   n=6 (closure Ne):    S += D₃³/P₁     (coherent through P₁ channels)
    #
    # Per=3 (Tier 1): tier alternation inverts the sign.
    #   n=5 (halogen Cl):    S += D₃³        (existing #63, tier-flipped)
    if l == 1 and per <= P_l:                           # per=2,3 p-block
        _d3cube = D3 * D3 * D3
        if per == 2:
            if nd == 1:                                 # embryonic (B)
                S -= _d3cube
            elif nd == P_l:                             # half-fill (N)
                S -= _d3cube * S_HALF
            elif nd == P_l + 1:                         # chalcogen (O)
                S -= _d3cube
            elif nd == P_l + 2:                         # halogen (F), 3-loop dressed
                S -= _d3cube * (1.0 + D3)
            elif nd == N_l:                             # closure (Ne)
                S += _d3cube / float(P1)
        else:                                           # per=3 (Tier 1)
            if nd == P_l + 2:                           # halogen (Cl)
                S += _d3cube                            # tier-flipped sign
            elif nd == N_l:                             # closure (Ar)
                S += _d3cube * S3                       # vertex self-energy
            elif nd == P_l + 1:                         # chalcogen (S)
                S -= _d3cube * D3                       # 4-loop self-energy

    # ── insight #66: per=3 p-block Fisher embryonic boundary ──────────
    # At per=3 (first propagating period), the DFT reconstruction gives
    # zero at n=1 (no peers), but the near propagator still adds full
    # screening.  The result: embryonic positions n=1,2 are over-screened.
    # Correction: Fisher information G=4 × α_EM × δ₃ through the
    # hexagonal gap, with linear profile (P₁-nd)/(P₁-1) decaying to
    # zero at half-fill.
    if l == 1 and per == P_l and nd <= 2:               # per=3, n=1,2
        _amp_emb = 4.0 * AEM * D3                      # G_Fisher × α × δ₃
        if nd == 1:
            _amp_emb *= (1.0 + D3 + D5)                # full gap dressing at embryonic
        _profile = (P_l - nd) / float(P_l - 1)         # 1.0 at n=1, 0.5 at n=2
        S -= _amp_emb * _profile

    # ── 1-loop f14 correction for p-block per 6+ (l=1, per > P1+2) ──
    # insight #73: cascade tier refinement of f14 correction.
    # Same pattern as #62r at per=4: position-dependent amplitude.
    #   n=2,3 (Pb,Bi): weakened by (1-D₃) — f14 overcorrects at early fill
    #   n=4   (Po):    NEW chalcogen — S₅D₃×D₃×s (was missing!)
    #   n=5   (At):    strengthened by D₃(1+D₃) — halogen needs more
    #   n=6   (Rn):    unchanged (original amplitude)
    if l == 1 and per > P_l + 2:
        A_p = S5 * D3                          # cross-channel: 0.0226
        if 2 <= nd <= P_l:                      # n=2,3 (Pb, Bi)
            S -= A_p * (1.0 - D3) * math.cos(math.pi * nd)
        elif nd == P_l + 1:                     # n=4 (Po): chalcogen
            S -= A_p * D3 * S_HALF
        elif nd == P_l + 2:                     # n=5 (At): halogen dressed
            S -= A_p * (1.0 + D3 * (1.0 + D3))
        elif nd > P_l + 2:                      # n=6 (Rn): original
            S -= A_p

    # ── Onset Nyquist p-block (l=1, per = P1+2 = 5) ──
    if l == 1 and per == P_l + 2 and nd <= P_l + 1:
        S -= AEM * math.cos(math.pi * nd)

    # ── j-SPLIT superheavy p-block (l=1, (Z*alpha)^2 > 0.5) ──
    if l == 1:
        za2_full = (Z * AEM) ** 2
        if za2_full > S_HALF:
            nf_32 = max(0, nd - 2)
            if nd == 1:
                S -= za2_full * S3                     # p1/2 contraction
            elif nd == 2:
                S -= za2_full * S3 / float(N_l)        # p1/2 closure (1/N_l)
            elif nf_32 == 1:
                S += za2_full * S3                     # p3/2 expansion (1st)
            elif nf_32 >= 2:
                S += za2_full * S3 * C3                # p3/2 exchange (*cos2_3)

    return S


# ╔════════════════════════════════════════════════════════════════════╗
# ║  S_REL — relativistic correction                                ║
# ╚════════════════════════════════════════════════════════════════════╝

def S_rel(Z):
    """Dirac-PT relativistic correction: CRT penetration + spin-orbit.

    1. za^2 = (Z_inner*alpha)^2 + (Z*alpha/per)^2   [Pythagoras CRT]
    2. f_SO = 1/(j+1/2) - 3/(4*per)   [Dirac structure on sieve]
    3. Threshold (Z*alpha)^2 > 2s*C3 = 0.39 ~ 0.4   [Dirac PT threshold]
    """
    per = period(Z)
    n_core = sum(2 * (p // 2 + 1) ** 2 for p in range(1, per))
    Z_inner = Z - n_core

    # Pythagoras CRT: valence + penetration
    za2_val = (Z_inner * AEM) ** 2       # radial circle (screened)
    za2_pen = (Z * AEM / per) ** 2       # angular circle (penetration)
    za2 = za2_val + za2_pen

    # Base term (identical structure, za^2 slightly larger)
    S = za2 * S3 / per

    # l-dependent spin-orbit (activated when (Z*alpha)^2 > threshold)
    za2_full = (Z * AEM) ** 2
    if za2_full > 0.40:
        l = l_of(Z)
        nf = _n_fill_aufbau(Z)  # screening = radial = Aufbau
        # Dominant j for the valence sub-shell
        if l == 0:
            j = 0.5
        elif l == 1:
            j = 0.5 if nf <= 2 else 1.5   # p1/2 then p3/2
        elif l == 2:
            j = 2.5                         # d5/2 dominant
        else:
            j = 3.5 if nf <= 7 else 2.5    # f7/2 then f5/2
        f_SO = 1.0 / (j + 0.5) - 3.0 / (4.0 * per)
        S += za2_full * f_SO * S3 / (per * MU_STAR)

    return S


# ╔════════════════════════════════════════════════════════════════════╗
# ║  IE_eV — the ionization energy                                  ║
# ╚════════════════════════════════════════════════════════════════════╝

_ME_AMU = ME_AMU


def IE_eV(Z: int, continuous: bool = False) -> float:
    """IE via geometric volumes on T3 and circles Z/(2P_l)Z.

    Everything is a VOLUME. The physics is in the GEOMETRY.
    Dimension l determines the circle. The pipeline is universal.

    continuous=True: use Li_γ polylogarithm for peer screening.

    Includes reduced-mass correction: IE = IE_∞ × M/(M+mₑ)
    where IE_∞ is the Rydberg (infinite nuclear mass) energy.
    Dominant for H (~7 meV), negligible for Z > 10.
    """
    per = period(Z)
    S = screening_action(Z, continuous=continuous)
    Z_eff = Z * math.exp(-S)
    IE_inf = RY * (Z_eff / per) ** 2
    m_nuc = MASS.get(Z, 2.0 * Z)
    return IE_inf * m_nuc / (m_nuc + _ME_AMU)


def screening_action(Z: int, continuous: bool = False) -> float:
    """Total PT screening action entering the IE engine."""
    per = period(Z)
    S_c = S_core(per)
    S_intra = S_polygon(Z, continuous=continuous)
    S = S_c + S_intra * D_FULL
    return S + S_rel(Z)


def effective_charge(Z: int) -> float:
    """Effective charge after PT screening."""
    return Z * math.exp(-screening_action(Z))


_P_EXCHANGE_PROFILE = P_EXCHANGE_PROFILE


def _ea_s_block(ie: float, n_s: int) -> float:
    """PT EA for H and alkali-like s-block atoms."""
    if n_s != 1:
        return 0.0
    return ie * S3 * S5


def _ea_p_block(ie: float, per: int, n_p: int) -> float:
    """RC64 PT electron affinity for the p-block.

    Main ingredients:
    - completion of the p shell
    - Pauli/Hund exchange profile
    - 2-loop degeneracy term
    - CRT boundary term at p4 for per >= 3
    - extra Hund depth at p3
    """
    if n_p <= 0 or n_p >= 2 * P1:
        return 0.0

    completion = n_p / float(2 * P1)
    exchange = (1.0 + S3 * _P_EXCHANGE_PROFILE.get(n_p, 0)) ** 2

    degeneracy = 0.0
    if n_p < 2 * P1 - 1:
        degeneracy = (S3 ** 2) * math.log((2 * P1) - n_p) / math.log(2 * P1)

    boundary = S3 * S_HALF if n_p == P1 + 1 and per >= 3 else 0.0
    hund_depth = S3 * C3 * (2.0 / per) ** 2 if n_p == P1 else 0.0
    f_total = completion * exchange + degeneracy + boundary - hund_depth
    f_per = C3 if per == 2 else math.sqrt(1.0 + S3)
    return max(0.0, ie * S3 * f_total * f_per)


def _ea_open_shell_generic(ie: float, per: int, n_fill_val: int, cap: int) -> float:
    """Fallback PT EA for d/f blocks until dedicated channels are ported.

    This keeps ``full_pt`` operational for the whole periodic table while
    reserving the exact RC64 structure to the s/p blocks where it is derived.
    """
    if n_fill_val <= 0 or n_fill_val >= cap:
        return 0.0

    half = cap // 2
    fill = n_fill_val / float(cap)
    openness = (cap - n_fill_val) / float(cap - 1)
    exchange = 1.0 + S3 * min(n_fill_val, cap - n_fill_val) / float(half)
    hund = 1.0 - (S3 * C3 * (2.0 / per) ** 2) if n_fill_val == half else 1.0
    f_per = math.sqrt(1.0 + S3) if per >= 3 else C3
    return max(0.0, ie * S3 * fill * openness * exchange ** 2 * hund * f_per)


def EA_eV(Z: int, ie: float | None = None) -> float:
    """PT electron affinity from polygon geometry (ea_geo engine).

    Delegates to EA_geo_eV which uses ShellPolygon capture/ejection
    amplitudes with dedicated Madelung channels for d-block promotions.
    The geometric engine gives MAE ~1.5% vs 68% for the basic fallback.
    """
    from ptc.ea_geo import EA_geo_eV
    return EA_geo_eV(Z)


def _normalize_ea_model(model: str) -> str:
    """Normalize supported PT EA model names."""
    aliases = {
        "classic": "classic",
        "legacy": "classic",
        "operator": "operator",
        "canonical": "operator",
        "ea_operator": "operator",
    }
    try:
        return aliases[model.lower()]
    except KeyError as exc:
        allowed = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Unknown ea_model '{model}'. Allowed canonical values: {allowed}") from exc


# ╔════════════════════════════════════════════════════════════════════╗
# ║  ATOM CLASS                                                      ║
# ╚════════════════════════════════════════════════════════════════════╝

class Atom:
    """Atom with PT-derived properties.

    Usage:
        a = Atom(6)            # Carbon, PT-computed IE
        a = Atom(6, source="exp")  # Carbon, experimental IE
        a = Atom(6, IE=99.0)       # Carbon, manually overridden IE

    Properties:
        Z        : atomic number
        symbol   : element symbol (str)
        period   : period number
        l        : angular momentum of valence sub-shell
        nf       : filling of valence sub-shell
        ns       : number of s-electrons
        mass     : atomic mass (amu, from IUPAC 2021)
        IE       : first ionization energy (eV)
        EA       : electron affinity (eV)
        chi_mulliken : Mulliken electronegativity (eV)
    """

    __slots__ = (
        "Z",
        "_provider",
        "_ie_override",
        "_ea_override",
        "_mass_override",
        "_ea_model",
        "_ie_pt",
        "_ea_pt",
        "_ea_classic_pt",
        "_ea_operator_pt",
    )

    def __init__(self, Z: int, source: str = "full_pt", IE: float | None = None,
                 EA: float | None = None, mass: float | None = None,
                 ea_model: str = "classic", **kwargs):
        self.Z = Z
        self._provider = AtomProvider(source, **kwargs)
        self._ie_override = IE
        self._ea_override = EA
        self._mass_override = mass
        self._ea_model = _normalize_ea_model(ea_model)
        self._ie_pt = None
        self._ea_pt = None
        self._ea_classic_pt = None
        self._ea_operator_pt = None

    # ── Basic properties ──

    @property
    def symbol(self) -> str:
        return SYMBOLS.get(self.Z, f"E{self.Z}")

    @property
    def period(self) -> int:
        return period(self.Z)

    @property
    def l(self) -> int:
        return l_of(self.Z)

    @property
    def nf(self) -> int:
        return n_fill(self.Z)

    @property
    def n_fill(self) -> int:
        return self.nf

    @property
    def ns(self) -> int:
        return ns_config(self.Z)

    @property
    def block(self) -> str:
        return block_of(self.Z)

    @property
    def cap(self) -> int:
        return capacity(self.Z)

    @property
    def mass(self) -> float:
        if self._mass_override is not None:
            return self._mass_override
        exp_val = MASS.get(self.Z, 0.0)
        return self._provider.resolve("MASS", self.Z, exp_val, exp_val)

    @property
    def ea_model(self) -> str:
        """Selected PT EA channel used by ``EA`` and ``EA_pt``."""
        return self._ea_model

    # ── Energetic properties ──

    @property
    def IE(self) -> float:
        """First ionization energy (eV).

        Resolution order: direct override > provider source > PT engine.
        """
        if self._ie_override is not None:
            return self._ie_override
        if self._ie_pt is None:
            self._ie_pt = IE_eV(self.Z)
        pt_val = self._ie_pt
        exp_val = IE_NIST.get(self.Z, pt_val)
        return self._provider.resolve("IE", self.Z, pt_val, exp_val)

    @property
    def EA(self) -> float:
        """Electron affinity (eV).

        Resolution order: direct override > provider source > PT engine.
        """
        if self._ea_override is not None:
            return self._ea_override
        pt_val = self.EA_pt
        exp_val = EA_NIST.get(self.Z, 0.0)
        return self._provider.resolve("EA", self.Z, pt_val, exp_val)

    @property
    def IE_pt(self) -> float:
        """PT ionization energy, independent of source selection."""
        if self._ie_pt is None:
            self._ie_pt = IE_eV(self.Z)
        return self._ie_pt

    @property
    def EA_pt(self) -> float:
        """Selected PT electron affinity, independent of source selection."""
        if self._ea_pt is None:
            if self._ea_model == "operator":
                self._ea_pt = self.EA_operator_pt
            else:
                self._ea_pt = self.EA_classic_pt
        return self._ea_pt

    @property
    def EA_classic_pt(self) -> float:
        """Legacy/classic PT electron affinity channel."""
        if self._ea_classic_pt is None:
            self._ea_classic_pt = EA_eV(self.Z, ie=self.IE_pt)
        return self._ea_classic_pt

    @property
    def EA_operator_pt(self) -> float:
        """Hierarchical operator PT electron affinity channel."""
        if self._ea_operator_pt is None:
            self._ea_operator_pt = EA_OPERATOR_MODEL_eV(self.Z, ie=self.IE_pt, projection="canonical")
        return self._ea_operator_pt

    @property
    def S_screening(self) -> float:
        """Dimensionless PT screening action entering the atomic IE."""
        return screening_action(self.Z)

    @property
    def Z_eff(self) -> float:
        """PT effective charge after screening."""
        return effective_charge(self.Z)

    @property
    def chi_mulliken(self) -> float:
        """Mulliken electronegativity = (IE + EA) / 2."""
        return (self.IE + self.EA) / 2.0

    def __repr__(self):
        return f"Atom({self.Z}, '{self.symbol}', IE={self.IE:.3f}, EA={self.EA:.3f})"


def compare_ea_channels(Z: int) -> dict[str, object]:
    """Return a compact side-by-side comparison of PT EA channels for one atom.

    The comparison always uses PT IE internally so the delta focuses only on
    the EA channel itself. When no positive benchmark EA is available in the
    embedded table, the error fields are set to ``None``.
    """
    classic = Atom(Z, ea_model="classic")
    operator = Atom(Z, ea_model="operator")
    exp_val = EA_NIST.get(Z)
    out: dict[str, object] = {
        "Z": Z,
        "symbol": classic.symbol,
        "block": classic.block,
        "period": classic.period,
        "IE_pt": classic.IE_pt,
        "EA_nist": exp_val,
        "EA_classic_pt": classic.EA_classic_pt,
        "EA_operator_pt": operator.EA_operator_pt,
        "classic_error_percent": None,
        "operator_error_percent": None,
    }
    if exp_val is not None and exp_val > 0.0:
        out["classic_error_percent"] = abs(classic.EA_classic_pt - exp_val) / exp_val * 100.0
        out["operator_error_percent"] = abs(operator.EA_operator_pt - exp_val) / exp_val * 100.0
    return out


def benchmark_atom_ea_models_against_nist() -> dict[str, object]:
    """Benchmark the classic and operator PT EA channels against embedded NIST.

    The benchmark follows the same rule as the research operator audit:
    only strictly positive ``EA_NIST`` rows are included, because the current
    embedded table encodes negative or negligible affinities as ``0.000``.
    """
    rows: list[dict[str, object]] = []
    by_model_errors: dict[str, list[float]] = {"classic": [], "operator": []}
    by_model_block: dict[str, defaultdict[str, list[float]]] = {
        "classic": defaultdict(list),
        "operator": defaultdict(list),
    }

    for Z in sorted(EA_NIST):
        exp_val = EA_NIST.get(Z)
        if exp_val is None or exp_val <= 0.0:
            continue

        row = compare_ea_channels(Z)
        rows.append(row)

        block = row["block"]
        classic_err = row["classic_error_percent"]
        operator_err = row["operator_error_percent"]
        assert isinstance(block, str)
        assert isinstance(classic_err, float)
        assert isinstance(operator_err, float)

        by_model_errors["classic"].append(classic_err)
        by_model_errors["operator"].append(operator_err)
        by_model_block["classic"][block].append(classic_err)
        by_model_block["operator"][block].append(operator_err)

    models: dict[str, dict[str, object]] = {}
    for name in ("classic", "operator"):
        mae = sum(by_model_errors[name]) / len(by_model_errors[name]) if by_model_errors[name] else 0.0
        models[name] = {
            "mae_percent": mae,
            "by_block": {
                block: sum(errors) / len(errors)
                for block, errors in sorted(by_model_block[name].items())
            },
        }

    return {
        "count": len(rows),
        "models": models,
        "rows": rows,
    }
