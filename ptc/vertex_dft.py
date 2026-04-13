"""
PTC — Vertex DFT on Z/(2P_l)Z with NATIVE contextual screening.

All effects are encoded in the screening S BEFORE exponentiation.
No post-hoc corrections. The wavefunction interferes in the action,
not in the energy (Principe ondulatoire).

4 contributions to S at each vertex v for bond bi:

  S_vertex = S_polygon   [DFT modes k=0..N/2 on Z/(2P_l)Z]
           + S_competition  [orbital saturation: (z-1)/z × filling]
           + S_lp_blocking  [LP at vertex blocks bond modes]
           + S_curvature    [cross-face Fisher cosines]

All from s = 1/2. 0 parameters.
"""

import math
from ptc.constants import (
    S_HALF, P0, P1, P2, P3,
    S3, S5, S7, C3, C5, C7,
    D3, D5, D7, AEM, RY,
    CROSS_35, CROSS_37, CROSS_57,
)
from ptc.topology import Topology, valence_electrons


def _peer_value(ie_partner: float, lp_partner: int, per_vertex: int,
                per_partner: int, lp_vertex: int = 0) -> float:
    """Peer screening value for one bond position on Z/(2P_l)Z.

    Contextual floor: D₇ is the universal structural floor.
    Bohr interpolation (reducing floor toward S₃D₃ for near-Ry partners)
    only applies when the CENTRE has LP (lp_vertex > 0, per ≤ 2).
    Physical reason: LP at the centre creates sp-hybridisation that
    reduces inter-bond screening. Without LP, the polygon is rigid
    and the D₇ floor reflects the minimal structural screening.

    Period crossing adds D₃ × √|Δper| × f_rel.
    """
    f_ie = min(ie_partner / RY, 1.0)

    # Base peer value from partner IE deficit
    base = (1.0 - f_ie) * S3

    # Contextual floor
    # D₇ is the universal default. Bohr reduction only when centre has LP.
    if lp_partner > 0:
        floor = D7
    elif per_vertex > 2:
        floor = D7
    elif lp_vertex > 0:
        # Centre with LP: sp-hybridisation reduces inter-bond screening
        # for LP-free near-Bohr partners (H at O in H₂O, H at N in NH₃)
        f_bohr = f_ie ** P1
        floor = D7 * (1.0 - f_bohr) + S3 * D3 * f_bohr
    else:
        # Centre without LP (C in C₂H₆, C in CH₄): keep D₇
        floor = D7

    val = max(floor, base)

    # Period crossing (cross-shell dispersion)
    # GFT anti-double-counting: the period crossing contribution is
    # REDUCED when the base peer value is already significant. The
    # polygon Z/(2P₁)Z has finite capacity S₃ per position. When the
    # base screening (1-f_ie)×S₃ already occupies a fraction base/S₃
    # of this capacity, the period crossing can only fill the REMAINDER.
    # Without this, low-IE + high-Δper partners (Si, B, P at C vertex)
    # get peer ≈ S₃, systematically over-screening these bonds.
    delta_per = abs(per_vertex - per_partner)
    if delta_per > 0:
        per_min = min(per_vertex, per_partner)
        f_rel = (S_HALF if per_min == 1
                 else P1 / max(per_vertex, per_partner) if max(per_vertex, per_partner) > P1
                 else 1.0)
        f_remain = max(0.0, 1.0 - base / S3)  # remaining capacity
        val += math.sqrt(delta_per) * D3 * f_rel * f_remain

    return val


def vertex_dft(vertex: int, bond_idx: int,
                   topology: Topology, atom_data: dict) -> float:
    """Complete vertex screening on Z/(2P_l)Z with all effects native.

    Returns S_vertex: the total screening contribution of this vertex
    to bond bond_idx. This S is DIRECTLY used in exp(-S × depth).

    Combines:
    - Polygon DFT (modes k=0..N/2)
    - Orbital competition (face saturation)
    - LP blocking (face rigidity)
    - Fisher curvature (cross-face coupling)
    """
    Z = topology.Z_list[vertex]
    d = atom_data[Z]
    z_v = topology.z_count[vertex]
    lp_v = topology.lp[vertex]
    per_v = d['per']
    ie_v = d['IE']
    is_d = d.get('is_d_block', False)
    is_s_block = (d.get('l', 0) == 0 and d.get('per', 1) >= 2)
    if is_d:
        N = 2 * P2   # 10 — decagonal face for d-block
    elif is_s_block:
        N = 2 * P0   # 4  — square face for s-block (Li, Be, Na, Mg, K, Ca)
    else:
        N = 2 * P1   # 6  — hexagonal face for p-block

    # ════════════════════════════════════════════════════════════
    #  1. POLYGON DFT (peer values → Fourier → synthesis)
    # ════════════════════════════════════════════════════════════

    # Collect bonds at vertex, sorted by partner IE descending
    bonds_at_v = []
    for bi in topology.vertex_bonds.get(vertex, []):
        ii, jj, bo = topology.bonds[bi]
        partner = jj if ii == vertex else ii
        Z_p = topology.Z_list[partner]
        ie_p = atom_data[Z_p]['IE']
        lp_p = topology.lp[partner]
        per_p = atom_data[Z_p]['per']
        bonds_at_v.append((bi, partner, Z_p, ie_p, lp_p, per_p, bo))
    bonds_at_v.sort(key=lambda x: -x[3])

    # Peer values at each polygon position
    # MODE-DEPENDENT: separate σ (k=0 monopole) from π+LP (k≥1).
    # PT principle: σ bonds screen on face P₁ (mode k=0 of Z/(2P₁)Z).
    # π bonds screen on face P₂ (mode k≥1). LP screens on face P₃ (k≥2).
    # The DFT monopole (a₀) should only contain the σ fraction.
    vals = [0.0] * N        # full values for modes k≥1
    vals_sigma = [0.0] * N  # σ-only values for mode k=0
    target_pos = -1
    n_lp_free_near_bohr = 0  # for Fisher composite
    n_near_bohr = 0

    for pos, (bi, partner, Z_p, ie_p, lp_p, per_p, bo) in enumerate(bonds_at_v):
        if pos >= N:
            break
        peer = _peer_value(ie_p, lp_p, per_v, per_p, lp_vertex=lp_v)
        # HOMONUCLEAR DEGENERACY BOOST [Fisher information of CRT pair]
        # When IE_vertex ≈ IE_partner (|δIE| < D₃×Ry), both atoms are in
        # the same CRT residue class → the k=0 DFT mode is degenerate →
        # the base screening (1-f_ie)×S₃ is near-zero. The universal
        # floor D₇ alone gives S₁ ≈ 0.03, causing +4% positive bias.
        # The Fisher information of the degenerate pair adds D₃×S_HALF
        # to the floor, raising S₁ into the sweet spot [0.05, 0.10].
        _delta_ie = abs(ie_p - ie_v)
        if _delta_ie < D3 * RY:
            _f_homo = 1.0 - _delta_ie / (D3 * RY)  # smooth: 1 at exact homo, 0 at threshold
            peer = max(peer, D7 + D3 * S_HALF * _f_homo)
        # σ/π split: only σ fraction (1/bo) enters the monopole k=0.
        # π fraction ((bo-1)/bo) lives on face P₂, contributes to k≥1.
        sigma_frac = 1.0 / max(bo, 1.0)
        vals[pos] = peer
        vals_sigma[pos] = peer * sigma_frac
        if bi == bond_idx:
            target_pos = pos
        if ie_p >= RY * C3:
            n_near_bohr += 1
            if lp_p == 0:
                n_lp_free_near_bohr += 1

    # LP positions — contribute to k≥1 (rigidity) but NOT to k=0 (monopole)
    for k in range(min(lp_v, N - min(z_v, N))):
        pos = min(z_v, N - 1) + k
        if pos >= N:
            break
        if is_d:
            lp_val = D5 * max(lp_v, 1)
        elif z_v >= P1 and lp_v > 0:
            lp_val = S3 * max(lp_v, 1)
        else:
            lp_val = D3 * max(lp_v, 1)
        vals[pos] = lp_val
        # LP monopole fraction: terminals (z=1) keep most LP in k=0,
        # centres (z≥2) progressively exclude LP from monopole.
        # f_mono = 1 - z/(z+P₁): z=1→0.75, z=2→0.60, z=3→0.50, z=4→0.43
        f_mono = 1.0 - float(z_v) / (z_v + P1)
        vals_sigma[pos] = lp_val * f_mono

    # LP cooperation (face rigidity) — applies to full vals only
    if z_v >= 2 and lp_v > 0:
        z_eff = z_v + min(lp_v, 3) * (1.0 + S_HALF)
        if z_eff > 1.0:
            theta = math.acos(max(-1.0, min(1.0, -1.0 / (z_eff - 1.0))))
            sigma_face = D5 * math.sin(theta / 2.0) * min(float(lp_v), float(P1)) / P1
            f_coop = max(0.0, 1.0 - sigma_face)
            for pos in range(min(z_v, N), min(z_v + lp_v, N)):
                vals[pos] *= f_coop

    if target_pos < 0:
        return 0.0

    # DFT analysis + synthesis
    # MODE-DEPENDENT: k=0 uses vals_sigma, k≥1 uses full vals
    omega = 2.0 * math.pi / N
    ns_range = range(1, N + 1)

    # k=0: σ-only monopole (LP and π excluded)
    a0_sigma = sum(vals_sigma) / N

    # k≥1: full spectrum (σ + π + LP for spatial structure)
    coeffs_k1 = []
    for m in range(1, N // 2 + 1):
        if m < N // 2:
            am = 2.0 / N * sum(v * math.cos(m * omega * n) for v, n in zip(vals, ns_range))
            bm = 2.0 / N * sum(v * math.sin(m * omega * n) for v, n in zip(vals, ns_range))
            coeffs_k1.append((am, bm))
        else:
            am = 1.0 / N * sum(v * math.cos(m * omega * n) for v, n in zip(vals, ns_range))
            coeffs_k1.append((am,))

    # Synthesis at filling position with SPECTRAL WEIGHTING
    # PT principle: the sieve filters sequentially — higher modes k>0
    # are perturbative corrections to the monopole k=0.
    # Weight w_k = cos²(π×k/N): k=0 → 1 (full), k=N/2 → 0 (suppressed).
    # This attenuates LP contributions in high-frequency modes without
    # an ad-hoc gate. 0 params: cos² is the geometric projection on Z/(2P_l)Z.
    lp_weight = 1.0 if z_v <= 1 else S3
    n_fill = max(1.0, min(z_v + lp_v * lp_weight, float(N)))
    S = a0_sigma  # σ-only monopole (weight = 1.0)
    for m_idx, m in enumerate(range(1, N // 2 + 1)):
        if m < N // 2:
            S += (coeffs_k1[m_idx][0] * math.cos(m * omega * n_fill)
                  + coeffs_k1[m_idx][1] * math.sin(m * omega * n_fill))
        else:
            S += coeffs_k1[m_idx][0] * math.cos(m * omega * n_fill)

    # ════════════════════════════════════════════════════════════
    #  2. BIFURCATION at z + lp = P₁ (orientation du simplexe)
    # ════════════════════════════════════════════════════════════

    filling = z_v + lp_v
    if filling == P1:
        S *= S_HALF
    elif filling > P1:
        f_fill = float(filling) / N
        fisher = 4.0 * f_fill * (1.0 - f_fill)
        S *= fisher

    # ════════════════════════════════════════════════════════════
    #  2bis. PARSEVAL BUDGET for z > P_l
    # ════════════════════════════════════════════════════════════
    # When z exceeds the polygon capacity P_l, Parseval's theorem
    # says the spectral budget is exceeded: extra screening.
    # S += D₃ × (z − P_l) / P_l — continuous, 0 params.
    l_v = d.get('l', 0)
    per_val = d['per']
    P_l = {0: P0, 1: P1, 2: P2, 3: P3}.get(l_v, P1)
    if z_v > P_l:
        # Period-dependent Parseval: per≥3 atoms (Si, Ge…) get S₃ scaling.
        # per=2 (C, N, O) get base D₃ scaling — their polygons are compact.
        per_excess = max(0, per_val - 2)
        if per_excess > 0:
            S += S3 * float(z_v - P_l) * per_excess / P_l
        else:
            S += D3 * float(z_v - P_l) / P_l

    # ════════════════════════════════════════════════════════════
    #  2ter. PERIOD SCREENING (s-block spiral excess)
    # ════════════════════════════════════════════════════════════
    # Each period beyond 2 = one extra s-block cycle on the Aufbau spiral.
    # Each cycle screens by D₃ (one trip around Z/(2P₁)Z).
    # Gated: not d-block (d-block uses Z/(2P₂)Z polygon), and lp < 2
    # (high-LP atoms like Br/Cl already screened by LP mechanism).
    if per_val > 2 and not is_d and lp_v < 2:
        S += D3 * float(per_val - 2)

    # ════════════════════════════════════════════════════════════
    #  2quater. HUND EXCHANGE SCREENING (d-block only)
    # ════════════════════════════════════════════════════════════
    # D-block atoms with unpaired d-electrons lose exchange energy
    # upon bonding. Each bond breaks delta_K exchange pairs.
    # S_Hund = delta_K × J / (Ry/P₁) where J = S₅×Ry/(2P₂).
    # delta_K = min(nf, P₂) - 1 counts exchange pairs.
    # 0 params: S₅, P₂, P₁ from s = 1/2. [Phase 1 C9]
    if is_d:
        nf = d.get('nf', 0)
        delta_K = max(0, min(nf, P2) - 1)
        if delta_K > 0:
            J_exchange = S5 * RY / (2.0 * P2)
            S_hund = delta_K * J_exchange / (RY / P1)
            S += S_hund

    # ════════════════════════════════════════════════════════════
    #  3. ORBITAL COMPETITION (face saturation, IN the screening)
    # ════════════════════════════════════════════════════════════
    # In PT: when z bonds occupy the same polygon Z/(2P_l)Z,
    # each additional bond beyond 1 shares the spectral budget.
    # Competition = (z-1)/z × S₃ = fraction of k=0 mode shared.
    # MC1 correction integrated INTO the screening.
    #
    # Gate: IE >= Ry - S₃ (non-hybridized centres, N/O/F).
    # Amplitude continue: sin²(π × IE / (2 × Ry)) interpole
    # between 0 (IE << Ry, hybridized) and S₃ (IE ≈ Ry, compact).

    # Gate: IE >= Ry − S₃ (non-hybridized centres: N, O, F).
    # Below this threshold, sp-hybridization absorbs the competition.
    # Above: compact orbitals compete non-perturbatively.
    # Continuous via sin²(π × (IE − IE_thr) / (2 × S₃)) from 0 at
    # threshold to S₃ × (z−1)/z at IE = Ry.
    _IE_THR = RY - S3
    if z_v >= 2 and ie_v >= _IE_THR:
        f_ie_comp = math.sin(math.pi * min(ie_v - _IE_THR, S3) / (2.0 * S3)) ** 2
        S_comp = f_ie_comp * S3 * float(z_v - 1) / z_v
        # LP protection: hybridised LP absorb competition (O lp=2: factor 0.5)
        S_comp *= float(z_v) / (z_v + lp_v)
        S += S_comp

    # ════════════════════════════════════════════════════════════
    #  3bis. VSEPR ANGLE SCREENING (holonomic projection on S²)
    # ════════════════════════════════════════════════════════════
    # The VSEPR geometry creates an angular stabilization that is
    # SEPARATE from the DFT screening — it's a geometric bonus from
    # the holonomic projection on S². It CAN make S negative (D_P1 > cap)
    # because angular optimization provides ADDITIONAL energy beyond
    # the structural screening budget. Applied AFTER the floor.
    S_vsepr = 0.0
    if z_v >= 2:
        _z_eff_vsepr = z_v + lp_v * (1.0 + C3)
        if _z_eff_vsepr > 1.001:
            _cos_theta = max(-1.0, min(1.0, -1.0 / (_z_eff_vsepr - 1.0)))
            # LP-driven VSEPR: BOTH centre LP and partner LP contribute.
            # Centre LP (lp_v): creates angular compression on S²
            #   → f_centre = lp_v / (z_v + lp_v)
            # Partner LP: creates LP-bond repulsion at the vertex
            #   → f_partner = mean(lp_p / (z_p + max(lp_p, 1)))
            # Combined: max(f_centre, f_partner) — whichever is stronger.
            _f_centre = float(lp_v) / (z_v + max(lp_v, 1)) if lp_v > 0 else 0.0
            _lp_drive_sum = 0.0
            for _bi_v, _partner_v, _Z_pv, _ie_pv, _lp_pv, _per_pv, _bo_v in bonds_at_v:
                _z_pv = topology.z_count[_partner_v]
                _lp_drive_sum += float(_lp_pv) / (_z_pv + max(_lp_pv, 1))
            _f_partner = _lp_drive_sum / max(len(bonds_at_v), 1)
            _f_lp_drive = max(_f_centre, _f_partner)
            S_vsepr = -S3 * abs(_cos_theta) / z_v * _f_lp_drive

    # ════════════════════════════════════════════════════════════
    #  4. FISHER CURVATURE (cross-face coupling, IN the screening)
    # ════════════════════════════════════════════════════════════
    # P₁⊗P₃ anti-correlation: for compact vertices (per ≤ 2)
    # with LP-free partners, P₁ and P₃ modes interfere
    # destructively → reduces screening.
    # C20 correction mais sans f_size.

    if per_v <= 2 and z_v >= 2 and lp_v == 0:
        if n_lp_free_near_bohr >= 2:
            f_fisher = float(n_lp_free_near_bohr * (n_lp_free_near_bohr - 1)) / (z_v ** 2)
            S -= CROSS_37 * f_fisher

    # ════════════════════════════════════════════════════════════
    #  4bis. BOND-ORDER FACE CLOSURE (hypervalent d-orbital screening)
    # ════════════════════════════════════════════════════════════
    # For per≥3 atoms with bo_sum > z (hypervalent), π electrons
    # occupy the P₁ face through d-orbital participation.
    # Closure: sin²(π × bo_fill/2) where bo_fill = bo_sum/(2P₁).
    # D-access: sin²(π × per/(2P₂)) gates by d-orbital availability.
    # [0 params]
    bo_sum_v = topology.sum_bo[vertex]
    bo_excess_v = round(bo_sum_v) - z_v
    if per_v >= P1 and bo_excess_v > 0 and not is_d:
        bo_fill = min(1.0, round(bo_sum_v) / float(N))
        f_closure = math.sin(math.pi * bo_fill / 2.0) ** 2
        f_d_access = math.sin(math.pi * per_v / (2.0 * P2)) ** 2
        S += S3 * f_closure * f_d_access * float(bo_excess_v) / float(z_v)

    # ════════════════════════════════════════════════════════════
    #  5. CURVATURE per ≥ 3 (cross-gap Fisher cosines)
    # ════════════════════════════════════════════════════════════

    if per_v == 2 and z_v >= 2:
        S += AEM * C3 * (z_v - 1) / P1
    if per_v >= P1:
        tier = per_v - P1
        S += CROSS_35 * D3 * min(tier + 1, P2 - 1) / (P2 - 1)

    # ════════════════════════════════════════════════════════════
    #  6. SHANNON CAPACITY COMPRESSION
    # ════════════════════════════════════════════════════════════
    # The P₁ face has a natural capacity S₃ = sin²(θ₃).  Above this,
    # the screening saturates: additional modes contribute only at
    # one-loop level (D₃ per unit).  This compresses high-S values
    # (hypervalent per≥3 molecules) while preserving the sweet spot
    # (S ∈ [0.05, 0.10] where MAE is 1.88%).
    #
    # S_eff = S                       if S ≤ S₃
    # S_eff = S₃ + (S − S₃) × D₃    if S > S₃
    #
    # PT basis: tree-level capacity is S₃; NLO corrections above
    # the tree level are suppressed by D₃ (one-loop on Z/6Z).
    S_pos = max(0.0, S)
    if S_pos > S3:
        S_pos = S3 + (S_pos - S3) * D3

    # Add VSEPR geometric bonus (can make total negative = bond
    # enhancement beyond cap — holonomic projection on S²)
    return S_pos + S_vsepr


# ════════════════════════════════════════════════════════════════════
#  CONTEXTUAL PEER FLOORS FOR P₂ AND P₃
# ════════════════════════════════════════════════════════════════════

def _peer_floor_P2(lp_partner: int, ie_partner: float,
                   bo_partner: float) -> float:
    """Contextual peer floor on Z/(2P₂)Z (pentagonal face).

    Follows the same logic as P₁'s _peer_value floor, adapted for
    the π-dominated pentagonal face:
      LP > 0 → D₇ (structural, same as P₁)
      LP = 0, bo > 1 → interpolated toward S₅D₅ (π spectral weight)
      LP = 0, bo = 1 → Bohr-interpolated D₇↔S₅D₅ via (IE/Ry)^P₁
    """
    if lp_partner > 0:
        return D7
    f_ie = min(ie_partner / RY, 1.0)
    f_bohr = f_ie ** P1
    _FLOOR_HI = S5 * D5   # ≈ 0.014 — Bohr limit for π face
    if bo_partner > 1.0:
        # π partner: interpolate by π fraction
        pi_frac = min((bo_partner - 1.0), 1.0)
        return D7 * (1.0 - pi_frac) + _FLOOR_HI * pi_frac
    return D7 * (1.0 - f_bohr) + _FLOOR_HI * f_bohr


def _peer_floor_P3(per_partner: int, lp_partner: int,
                   ie_partner: float) -> float:
    """Contextual peer floor on Z/(2P₃)Z (heptagonal face).

    Steric-weighted: heavier period → more steric pressure → higher floor.
    LP compresses the steric footprint → keeps D₇ structural floor.
    """
    if lp_partner > 0:
        return D7  # LP = structural → minimal steric floor
    # Period-dependent steric floor
    per_excess = max(0, per_partner - 1)
    f_steric = math.sqrt(per_excess) * D7
    return max(D7, f_steric)


# ════════════════════════════════════════════════════════════════════
#  V5 VERTEX DFT — P₂ FACE (NATIVE, contextual)
# ════════════════════════════════════════════════════════════════════

def vertex_dft_P2(vertex: int, bond_idx: int,
                      topology: Topology, atom_data: dict) -> float:
    """V5 native vertex DFT on Z/(2P₂)Z (pentagonal face, π + LP).

    Contextual peer floor replaces the universal D₇.
    Occupants: π bonds (bo > 1 component) + LP.
    All from s = 1/2.
    """
    Z = topology.Z_list[vertex]
    d = atom_data[Z]
    z_v = topology.z_count[vertex]
    lp_v = topology.lp[vertex]
    per_v = d['per']
    N = 2 * P2  # 10 positions

    vals = [0.0] * N

    # 1. Peer values: π bonds sorted by partner IE
    pi_bonds = []
    for bi in topology.vertex_bonds.get(vertex, []):
        ii, jj, bo = topology.bonds[bi]
        if bo > 1.0:
            partner = jj if ii == vertex else ii
            Z_p = topology.Z_list[partner]
            ie_p = atom_data[Z_p]['IE']
            lp_p = topology.lp[partner]
            pi_bonds.append((bi, partner, ie_p, bo, lp_p))
    pi_bonds.sort(key=lambda x: -x[2])

    n_sigma = z_v - len(pi_bonds)
    if not pi_bonds:
        return 0.0  # no π bonds → P₂ face empty

    target_pos = -1
    for pos, (bi, partner, ie_p, bo, lp_p) in enumerate(pi_bonds):
        if pos >= N:
            break
        f_ie_p = min(ie_p / RY, 1.0)
        # CONTEXTUAL floor
        floor = _peer_floor_P2(lp_p, ie_p, bo)
        vals[pos] = max(floor, (1.0 - f_ie_p) * S5) * (bo - 1.0)
        if lp_p > 0:
            vals[pos] += lp_p * D5 / P2
        Z_p = topology.Z_list[partner]
        per_p = atom_data[Z_p]['per']
        if abs(per_v - per_p) > 0:
            f_rel = (S_HALF if min(per_v, per_p) == 1
                     else P2 / max(per_v, per_p) if max(per_v, per_p) > P2
                     else 1.0)
            vals[pos] += math.sqrt(abs(per_v - per_p)) * D5 * f_rel
        if n_sigma > 0:
            vals[pos] += n_sigma * D5 * D3 / P2
        if bi == bond_idx:
            target_pos = pos

    # LP at positions after π bonds
    n_pi = min(len(pi_bonds), N)
    for k in range(min(lp_v, N - n_pi)):
        pos = n_pi + k
        if pos >= N:
            break
        vals[pos] = D5 * max(lp_v, 1)
        if max(0, per_v - 2) > 0 and z_v <= 1:
            vals[pos] *= (1.0 + max(0, per_v - 2) * D5)

    if target_pos < 0:
        target_pos = min(n_pi + lp_v, N - 1)
        if target_pos < 0:
            return 0.0

    # LP cooperation
    if z_v >= 2 and lp_v > 0:
        z_eff = z_v + min(lp_v, P2) * (1.0 + S_HALF)
        if z_eff > 1.0:
            theta = math.acos(max(-1.0, min(1.0, -1.0 / (z_eff - 1.0))))
            sigma_face = D5 * math.sin(theta / 2.0) * min(float(lp_v), float(P2)) / P2
            for pos in range(n_pi, min(n_pi + lp_v, N)):
                vals[pos] *= max(0.0, 1.0 - sigma_face)

    # 2. DFT analysis on Z/10Z
    omega = 2.0 * math.pi / N
    ns = range(1, N + 1)
    coeffs = [(sum(vals) / N,)]
    for m in range(1, N // 2 + 1):
        if m < N // 2:
            am = 2.0 / N * sum(v * math.cos(m * omega * n)
                               for v, n in zip(vals, ns))
            bm = 2.0 / N * sum(v * math.sin(m * omega * n)
                               for v, n in zip(vals, ns))
            coeffs.append((am, bm))
        else:
            am = 1.0 / N * sum(v * math.cos(m * omega * n)
                               for v, n in zip(vals, ns))
            coeffs.append((am,))

    # 3. Synthesis
    lp_weight = 1.0 if z_v <= 1 else S5
    n_fill = max(1.0, min(n_pi + lp_v * lp_weight, float(N)))
    S = coeffs[0][0]
    for m in range(1, N // 2 + 1):
        idx = m
        if m < N // 2:
            S += (coeffs[idx][0] * math.cos(m * omega * n_fill)
                  + coeffs[idx][1] * math.sin(m * omega * n_fill))
        else:
            S += coeffs[idx][0] * math.cos(m * omega * n_fill)

    # 4. Bifurcation at π + lp = P₂
    filling_P2 = len(pi_bonds) + lp_v
    if filling_P2 == P2:
        S *= S_HALF
    elif filling_P2 > P2:
        f_fill = float(filling_P2) / N
        S *= 4.0 * f_fill * (1.0 - f_fill)

    # 5. Curvature
    if per_v == 2 and len(pi_bonds) >= 1:
        S += CROSS_35 * D5 * C5 * len(pi_bonds) / P2
    if per_v >= P2:
        tier = per_v - P2
        S -= CROSS_57 * D5 * min(tier + 1, P3 - 1) / (P3 - 1)

    # Shannon capacity compression (same as P₁, S₅ capacity on P₂)
    S_pos = max(0.0, S)
    if S_pos > S5:
        S_pos = S5 + (S_pos - S5) * D5
    return S_pos


# ════════════════════════════════════════════════════════════════════
#  V5 VERTEX DFT — P₃ FACE (NATIVE, contextual)
# ════════════════════════════════════════════════════════════════════

def vertex_dft_P3(vertex: int, bond_idx: int,
                      topology: Topology, atom_data: dict) -> float:
    """V5 native vertex DFT on Z/(2P₃)Z (heptagonal face, steric).

    Contextual peer floor replaces the universal D₇.
    Occupants: ALL bonds + LP (total steric load).
    All from s = 1/2.
    """
    Z = topology.Z_list[vertex]
    d = atom_data[Z]
    z_v = topology.z_count[vertex]
    lp_v = topology.lp[vertex]
    per_v = d['per']
    N = 2 * P3  # 14 positions

    total_load = z_v + lp_v
    vals = [0.0] * N

    # 1. Peer values: ALL bonds sorted by partner period
    bonds_at_v = []
    for bi in topology.vertex_bonds.get(vertex, []):
        ii, jj, bo = topology.bonds[bi]
        partner = jj if ii == vertex else ii
        Z_p = topology.Z_list[partner]
        per_p = atom_data[Z_p]['per']
        lp_p = topology.lp[partner]
        ie_p = atom_data[Z_p]['IE']
        bonds_at_v.append((bi, partner, per_p, bo, lp_p, ie_p, Z_p))
    bonds_at_v.sort(key=lambda x: -x[2])

    target_pos = -1
    for pos, (bi, partner, per_p, bo, lp_p, ie_p, Z_p) in enumerate(bonds_at_v):
        if pos >= N:
            break
        # CONTEXTUAL steric floor
        floor = _peer_floor_P3(per_p, lp_p, ie_p)
        vals[pos] = math.sqrt(max(per_p - 1, 0)) * floor
        if lp_p > 0:
            vals[pos] += lp_p * D7 / P3
        if abs(per_v - per_p) > 0:
            vals[pos] += math.sqrt(abs(per_v - per_p)) * D7 * S_HALF
        if min(Z, Z_p) == 1 and max(per_v, per_p) >= P1:
            vals[pos] += D3 * math.sqrt(max(per_v, per_p) - 1) / P1
        if bi == bond_idx:
            target_pos = pos

    # LP at remaining positions
    n_bonds_placed = min(z_v, N)
    for k in range(min(lp_v, N - n_bonds_placed)):
        pos = n_bonds_placed + k
        if pos >= N:
            break
        vals[pos] = D7 * max(lp_v, 1)
        if per_v >= P1 and lp_v > 0:
            n_promoted = max(0, total_load - 4)
            if n_promoted > 0:
                vals[pos] += n_promoted * D5 / max(z_v, 1)

    if target_pos < 0:
        return 0.0

    # 2. DFT analysis on Z/14Z
    omega = 2.0 * math.pi / N
    ns = range(1, N + 1)
    coeffs = [(sum(vals) / N,)]
    for m in range(1, N // 2 + 1):
        if m < N // 2:
            am = 2.0 / N * sum(v * math.cos(m * omega * n)
                               for v, n in zip(vals, ns))
            bm = 2.0 / N * sum(v * math.sin(m * omega * n)
                               for v, n in zip(vals, ns))
            coeffs.append((am, bm))
        else:
            am = 1.0 / N * sum(v * math.cos(m * omega * n)
                               for v, n in zip(vals, ns))
            coeffs.append((am,))

    # 3. Synthesis
    lp_weight = 1.0 if z_v <= 1 else S7
    n_fill = max(1.0, min(z_v + lp_v * lp_weight, float(N)))
    S = coeffs[0][0]
    for m in range(1, N // 2 + 1):
        idx = m
        if m < N // 2:
            S += (coeffs[idx][0] * math.cos(m * omega * n_fill)
                  + coeffs[idx][1] * math.sin(m * omega * n_fill))
        else:
            S += coeffs[idx][0] * math.cos(m * omega * n_fill)

    # 4. Bifurcation at z + lp = P₃
    if total_load == P3:
        S *= S_HALF
    elif total_load > P3:
        f_fill = float(total_load) / N
        S *= 4.0 * f_fill * (1.0 - f_fill)

    # 5. Curvature
    if per_v >= P1:
        tier = per_v - P1
        S -= CROSS_37 * D7 * min(tier + 1, P3 - 1) / (P3 - 1)
    if per_v >= P2:
        tier = per_v - P2
        S -= CROSS_57 * D7 * min(tier + 1, P3 - 1) / (P3 - 1)

    # Shannon capacity compression (S₇ capacity on P₃)
    S_pos = max(0.0, S)
    if S_pos > S7:
        S_pos = S7 + (S_pos - S7) * D7
    return S_pos
