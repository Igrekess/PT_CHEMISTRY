"""
PTC CASCADE — Moteur D_at unifié sur T³.

Contient les DEUX phases de la convergence spectrale moléculaire :

  PHASE 1 (n ≤ P₃ = 7) : délègue à transfer_matrix.py (Perron λ₀)
    + post-corrections LP→σ, charges formelles, VSEPR.

  PHASE 2 (n > P₃) : cascade per-bond P₀→P₁→P₂→P₃
    + 17 corrections NLO (coopérative, Dicke, T³ perturbatif,
      LP→π, ring strain, halogen π-drain, etc.)

La bifurcation à n = P₃ = 7 est la transition Phase 1 / Phase 2
de la convergence de Mertens moléculaire [T4, D08] :
  Phase 1 : le graphe moléculaire est dense → T⁴ Perron exact.
  Phase 2 : le graphe est sparse → corrections NLO convergent.

Architecture per-bond (Phase 2) :
   Face P₀ (Z/2Z) — parity/LP gate
   Face P₁ (Z/6Z) — σ+π bonding  ← DOMINANT
   Face P₂ (Z/10Z) — d/π² conditionné par P₁
   Face P₃ (Z/14Z) — ionique conditionné par P₁+P₂

0 paramètre ajusté. Tout depuis s = 1/2.
"""

import math
import numpy as np
from typing import Dict, NamedTuple, Optional, List, Set

from ptc.constants import (
    S_HALF, P0, P1, P2, P3,
    S3, S5, S7, C3, C5, C7,
    D3, D5, D7, AEM, RY,
    D_FULL, D_FULL_P1, D_FULL_P2, DEPTH_P3,
    BETA_GHOST, CROSS_35, CROSS_37, CROSS_57,
    F_BIFURC, COULOMB_EV_A,
    GAMMA_7, R35_DARK, R57_DARK, R37_DARK,
    IE2_DBLOCK,
)
from ptc.topology import Topology, valence_electrons
from ptc.atom import IE_eV, EA_eV
from ptc.periodic import period, l_of, ns_config
from ptc.data.experimental import IE_NIST, EA_NIST, MASS
from ptc.bond import r_equilibrium

# ── Import v4 screening modules (pure PT, no calibration) ──
# v4 engine (delegation for n ≤ 7)
from ptc.transfer_matrix import compute_D_at_transfer
# v4 per-bond seed (diatomic quality, for triatomic v5 solver)
from ptc.screening_bond_v4 import D0_screening as _D0_screening_v4

# Shared screening functions (used by both v4 and v5)
from ptc.screening import (
    _resolve_atom_data,
    BondSeed,
    _cap_P0, _cap_P2, _cap_P3,
    _dim3_overcrowding,
    _dim2_lp_mutual,
    _dim1_exchange,
    _screening_P0,
    _dim2_vacancy_boost,
    _huckel_aromatic,
    _apply_shell_attenuation,
    _build_T4, _scf_iterate,
)
from ptc.vertex_dft_v5 import vertex_dft_v5, vertex_dft_v5_P2, vertex_dft_v5_P3


# ════════════════════════════════════════════════════════════════════
#  VERTEX CONTEXT
# ════════════════════════════════════════════════════════════════════

class VertexCtx:
    __slots__ = ('Z', 'ie', 'ea', 'per', 'l', 'z', 'lp', 'nf',
                 'is_d', 'n_bohr', 'lp_adj')

    def __init__(self, v: int, topology: Topology, atom_data: dict):
        self.Z = topology.Z_list[v]
        d = atom_data[self.Z]
        self.ie = d['IE']
        self.ea = d['EA']
        self.per = d['per']
        self.l = d.get('l', 0)
        self.z = topology.z_count[v]
        self.lp = topology.lp[v]
        self.nf = d.get('nf', 0)
        self.is_d = d.get('is_d_block', False)
        self.n_bohr = 0
        self.lp_adj = 0
        for bi in topology.vertex_bonds.get(v, []):
            ii, jj, bo = topology.bonds[bi]
            partner = jj if ii == v else ii
            Z_p = topology.Z_list[partner]
            d_p = atom_data[Z_p]
            if d_p['IE'] >= RY * C3:
                self.n_bohr += 1
            self.lp_adj += topology.lp[partner]


# ════════════════════════════════════════════════════════════════════
#  CONTEXTUAL PEER FLOOR (replaces D7 universally)
# ════════════════════════════════════════════════════════════════════

def _peer_floor(lp_partner: int, ie_partner: float) -> float:
    """Contextual peer floor on Z/(2P₁)Z.

    LP > 0: structural screening → D₇
    LP = 0: Bohr-interpolated → D₇·(1-f) + S₃D₃·f
    f_Bohr = (IE/Ry)^P₁ : P₁ modes all Bohr-localized.
    """
    if lp_partner > 0:
        return D7
    f_ie = min(ie_partner / RY, 1.0)
    f_bohr = f_ie ** P1
    return D7 * (1.0 - f_bohr) + S3 * D3 * f_bohr


# ════════════════════════════════════════════════════════════════════
#  FACE P₁ SCREENING — CONTEXTUAL (replaces _screening_P1)
# ════════════════════════════════════════════════════════════════════

def _screening_P1_cascade(i: int, j: int, bo: float, bi: int,
                           topology: Topology, atom_data: dict,
                           ctx_i: VertexCtx, ctx_j: VertexCtx) -> float:
    """P₁ screening with contextual peer floor.

    Uses the SAME v4 pipeline (dim3 + vertex_DFT + dim1 + bifurcation)
    but with a contextual peer floor instead of universal D₇.
    The v4 _vertex_polygon_dft is called directly — it already
    implements the full DFT on Z/(2P_l)Z with all the modes.

    The ONLY change vs v4: the peer floor is per-bond contextual
    via _peer_floor(), which replaces f_size-gated C17/C18.
    """
    Zi, Zj = topology.Z_list[i], topology.Z_list[j]
    di, dj = atom_data[Zi], atom_data[Zj]
    z_A, z_B = topology.z_count[i], topology.z_count[j]

    S = 0.0

    # ── dim 3: overcrowding ──
    S_dim3, sigma_3 = _dim3_overcrowding(i, j, topology, atom_data)
    S += S_dim3

    # ── dim 2: vertex DFT v5 (NATIVE contextual + competition + Fisher)
    # All effects integrated in S: no post-hoc corrections needed.
    S_dft_A = vertex_dft_v5(i, bi, topology, atom_data)
    S_dft_B = vertex_dft_v5(j, bi, topology, atom_data)

    # ── dim 2: LP mutual (cross-polygon LP coupling, v4 function) ──
    S_mutual = _dim2_lp_mutual(i, j, topology, atom_data)

    # Degree-aware amplification for z=2 (triatomic regime)
    # Modulated by: (a) z/(z+lp) — LP already provides screening
    #               (b) 1/bo_max — π bonds on different face, less monopole deficit
    z_max_bond = max(z_A, z_B)
    if z_max_bond == 2:
        for v_sh, z_v_sh, label in ((i, z_A, 'A'), (j, z_B, 'B')):
            if z_v_sh >= 2 and z_v_sh < P1:
                lp_sh = topology.lp[v_sh]
                f_lp = float(z_v_sh) / (z_v_sh + lp_sh) if lp_sh > 0 else 1.0
                # Bond-order modulation: double/triple bonds have π on P₂/P₃
                # face → less monopole deficit on P₁ → less amplification needed
                bo_max_v = max(topology.bonds[bk][2]
                               for bk in topology.vertex_bonds.get(v_sh, [bi]))
                f_bo = 1.0 / max(bo_max_v, 1.0)
                f_amp = 1.0 + float(P1 - z_v_sh) / z_v_sh * f_lp * f_bo
                if label == 'A':
                    S_dft_A *= f_amp
                else:
                    S_dft_B *= f_amp

    S += (S_dft_A + S_dft_B) / 2.0 + S_mutual

    # ── dim 1: exchange/Bohr/polarity (v4 function) ──
    S_dim1, Q_eff = _dim1_exchange(i, j, topology, atom_data)
    S += S_dim1

    # ── Bifurcation (v4 logic) ──
    per_i, per_j = di['per'], dj['per']
    delta_per = abs(per_i - per_j)
    f_bif = 1.0 - (1.0 - F_BIFURC) * min(delta_per, P1) / P1 * Q_eff
    if f_bif < 1.0:
        S_dim3_val = -math.log(1.0 - sigma_3) if 0 < sigma_3 < 1 else max(sigma_3, 0.0)
        S_dft_total = S_dft_A + S_dft_B + S_mutual
        S_geom = max(0.0, S_dft_total + S_dim3_val)
        S_edge_vert = S - S_geom
        S = S_edge_vert + S_geom * f_bif

    S = max(0.0, S)

    # ── Ionic same-period floor ──
    # Use charge-modulated IE for ionic floor
    _charges = getattr(topology, 'charges', None)
    _qi = _charges[i] if _charges is not None and i < len(_charges) else 0
    _qj = _charges[j] if _charges is not None and j < len(_charges) else 0
    _ie_i_eff = di['IE'] + _qi * RY / P1
    _ie_j_eff = dj['IE'] + _qj * RY / P1
    f_cov = min(1.0, min(_ie_i_eff, _ie_j_eff) / RY)
    IE_acc = max(_ie_i_eff, _ie_j_eff)
    if f_cov < C3 and delta_per == 0 and IE_acc < RY:
        S = max(S, (1.0 - f_cov) * S3 * P1)

    return S, Q_eff


# ════════════════════════════════════════════════════════════════════
#  LP TERMINAL CROWDING (size-independent, replaces C2/C2b)
# ════════════════════════════════════════════════════════════════════

def _lp_crowding(v: int, topology: Topology, atom_data: dict,
                 bond_energies: Dict[int, float]) -> None:
    """Apply LP crowding and halogen pair enhancement at vertex v.

    Modifies bond_energies IN PLACE. Size-independent.
    """
    z_v = topology.z_count[v]
    if z_v < 2:
        return

    d_v = atom_data[topology.Z_list[v]]
    per_v = d_v['per']
    lp_v = topology.lp[v]

    adj_bonds = [bi for bi in topology.vertex_bonds.get(v, [])
                 if bi in bond_energies]
    if not adj_bonds:
        return

    # Sum LP on adjacent terminals, weighted by compactness.
    # Super-Bohr gate: F/O LP (IE > Ry×(1+S₃)) is structural
    # screening that is EXCLUDED from crowding UNLESS the centre
    # has no d-vacancy, non-deficient, and no LP (same as v4 C2).
    _center_has_d_vacancy = per_v >= P1 and d_v.get('l', 0) >= 1
    # Super-Bohr gate: include high-IE terminal LP when the centre
    # itself has LP (e.g. N in NF3: lp=1, F terminals lp=3).
    # Centres without LP (C in CF4: lp=0) → F LP stays excluded.
    # This differentiates NF3 (overbinding) from CF4 (ok).
    _include_super_bohr = (not _center_has_d_vacancy
                           and (lp_v > 0 or d_v.get('nf', 0) > 1))
    lp_w = 0.0
    lp_raw = 0
    n_hal = 0
    n_above_bohr = 0
    for bi in adj_bonds:
        ii, jj, _ = topology.bonds[bi]
        t = jj if ii == v else ii
        lp_t = topology.lp[t]
        lp_raw += lp_t
        ie_t = atom_data[topology.Z_list[t]]['IE']
        per_t = atom_data[topology.Z_list[t]]['per']
        Z_t = topology.Z_list[t]
        if ie_t > RY * (1 + S3):
            n_above_bohr += 1
            if _include_super_bohr:
                lp_w += lp_t * (2.0 / max(per_t, 2)) ** (1.0 / P1)
        else:
            lp_w += lp_t * (2.0 / max(per_t, 2)) ** (1.0 / P1)
        if Z_t in (9, 17, 35, 53):
            n_hal += 1

    # LP crowding
    S_crowd = 0.0
    if lp_w > z_v:
        f_crowd = (lp_w - z_v) / max(lp_w, 1)
        lp_per_term = lp_raw / max(z_v, 1.0)
        S_crowd = D3 * S_HALF * f_crowd * (1.0 + lp_per_term / P1)
        for bi in adj_bonds:
            bond_energies[bi] *= math.exp(-S_crowd)

    # Super-Bohr LP pair Fisher correlation at LP centres.
    # When n ≥ 2 high-IE terminals surround an LP centre (e.g. N in NF3),
    # their LP modes constructively interfere on the centre's polygon,
    # creating cross-vertex screening: D₃ × n(n−1)/(z×P₁).
    # This is the converse of vertex_dft_v5 §4 (Fisher anti-correlation
    # for LP-free partners), applied to super-Bohr LP partners.
    if _include_super_bohr and n_above_bohr >= 2 and lp_v > 0:
        S_pair_super = D3 * float(n_above_bohr * (n_above_bohr - 1)) / (z_v * P1)
        for bi in adj_bonds:
            bond_energies[bi] *= math.exp(-S_pair_super)

    # Halogen LP pair enhancement (C2b with per-dependent lp_eff)
    _center_has_d_vacancy = per_v >= P1 and d_v.get('l', 0) >= 1
    if (not _center_has_d_vacancy and d_v.get('nf', 0) > 1
            and lp_v == 0 and n_hal >= 2):
        n_pairs = n_hal * (n_hal - 1) / 2.0
        lp_hal_sum = 0.0
        for bi in adj_bonds:
            ii, jj, _ = topology.bonds[bi]
            t = jj if ii == v else ii
            Z_t = topology.Z_list[t]
            if Z_t in (9, 17, 35, 53):
                per_t = atom_data[Z_t]['per']
                lp_t = topology.lp[t]
                lp_hal_sum += float(lp_t) * (float(per_t) / P0) ** (1.0 / P1)
        lp_hal = lp_hal_sum / n_hal
        S_pair = n_pairs / z_v * (lp_hal / (2.0 * P1)) ** 2 / P1
        S_extra = max(0.0, S_pair - S_crowd)
        if S_extra > 0:
            for bi in adj_bonds:
                bond_energies[bi] *= math.exp(-S_extra)


# ════════════════════════════════════════════════════════════════════
#  HÜCKEL AROMATIC (imported from v4)
# ════════════════════════════════════════════════════════════════════
# Uses _huckel_aromatic from transfer_matrix.py directly.


# ════════════════════════════════════════════════════════════════════
#  NLO VERTEX COUPLING (with C16 degeneracy built in)
# ════════════════════════════════════════════════════════════════════

def _nlo_correction(topology: Topology, atom_data: dict,
                    bond_energies: Dict[int, float]) -> None:
    """NLO pairwise vertex coupling. Modifies bond_energies IN PLACE."""
    ring_set = topology.ring_bonds or set()
    for v in range(topology.n_atoms):
        z_v = topology.z_count[v]
        if z_v < 2:
            continue
        v_bonds = topology.vertex_bonds.get(v, [])
        if len(v_bonds) < 2:
            continue

        lp_v = topology.lp[v]
        per_v = atom_data[topology.Z_list[v]]['per']

        Ev = [(bi, bond_energies.get(bi, 0.0)) for bi in v_bonds]
        E_total = sum(e for _, e in Ev)
        if E_total <= 0:
            continue

        E_nlo = 0.0
        for a in range(len(Ev)):
            bi_a, D_a = Ev[a]
            if D_a <= 0:
                continue
            for b in range(a + 1, len(Ev)):
                bi_b, D_b = Ev[b]
                if D_b <= 0:
                    continue
                V = D3 * math.sqrt(D_a * D_b) / (z_v * P1)
                if bi_a in ring_set and bi_b in ring_set:
                    V *= math.sqrt(P1)
                delta_E = abs(D_a - D_b)
                f_degen = 1.0
                if (z_v >= P1 and lp_v == 0 and per_v <= 2
                        and bi_a not in ring_set and bi_b not in ring_set):
                    f_degen = delta_E / (delta_E + V) if V > 1e-12 else 1.0
                E_pair = -(V * V) / (delta_E + V) * f_degen
                E_nlo += E_pair

        if E_nlo < 0:
            for bi, D_bi in Ev:
                frac = D_bi / E_total
                bond_energies[bi] = max(bond_energies[bi] + E_nlo * frac, 0.01)


# ════════════════════════════════════════════════════════════════════
#  GHOST ATTENUATION
# ════════════════════════════════════════════════════════════════════

def _ghost(ctx_i: VertexCtx, ctx_j: VertexCtx) -> float:
    n_shells = ((ctx_i.per - 1) + (ctx_j.per - 1)) / 2.0
    z_min = min(ctx_i.z, ctx_j.z)
    f_z = 1.0 + max(0, z_min - 1) * D3
    S_ghost = BETA_GHOST * D3 * S_HALF * max(n_shells, 1.0) * f_z
    return math.exp(-S_ghost * D_FULL)


# ════════════════════════════════════════════════════════════════════
#  MAIN: compute_D_at_cascade
# ════════════════════════════════════════════════════════════════════

class CascadeResult(NamedTuple):
    D_at: float
    D_at_P1: float
    D_at_P2: float
    D_at_P3: float
    iterations: int


# ════════════════════════════════════════════════════════════════════
#  TRIATOMIC V5 NATIVE SOLVER
#  v4 diatomic-quality seeds + 3×3 cooperation + v5-improved face
#  corrections (GFT peer saturation, σ-π D₃ gate, d-block routing).
#  0 adjustable parameters.  All from s = 1/2.
# ════════════════════════════════════════════════════════════════════

def _triatomic_v5(topology: Topology, atom_data: dict) -> CascadeResult:
    """V5 native triatomic solver: v4 seeds + 3×3 cooperation + v5 corrections.

    Architecture:
      1. v4 diatomic-quality per-bond seeds (_D0_screening_v4)
      2. 3×3 Hamiltonian → f_renorm (bilateral cooperation)
      3. Per-type routing (radical/linear/bent/sblock/carbene)
      4. Face corrections F1-F7 with v5 improvements
      5. Ghost VP attenuation

    0 adjustable parameters.  All from s = 1/2.
    """
    # ── 1. Identify roles: central B (z=2), terminals A and C ──
    zc = topology.z_count
    ctr = max(range(3), key=lambda k: zc[k])
    terms = [k for k in range(3) if k != ctr]
    A, B, C = terms[0], ctr, terms[1]

    Z_B = topology.Z_list[B]
    d_B = atom_data[Z_B]
    d_A = atom_data[topology.Z_list[A]]
    d_C = atom_data[topology.Z_list[C]]
    lp_B = topology.lp[B]
    per_B = d_B['per']
    l_B = d_B.get('l', 0)
    ie_B = d_B['IE']
    nf_B = d_B.get('nf', 0)

    # ── 2. Match bonds to A-B and B-C ──
    bond_AB = bond_BC = None
    for bi, (i, j, bo) in enumerate(topology.bonds):
        s = frozenset((i, j))
        if A in s and B in s:
            bond_AB = (i, j, bo, bi)
        elif B in s and C in s:
            bond_BC = (i, j, bo, bi)
    if bond_AB is None or bond_BC is None:
        return CascadeResult(0.0, 0.0, 0.0, 0.0, 0)

    bo_AB, bo_BC = bond_AB[2], bond_BC[2]
    bi_AB, bi_BC = bond_AB[3], bond_BC[3]

    # ── 3. Detect molecule type ──
    total_ve = sum(valence_electrons(Z) for Z in topology.Z_list)
    is_radical = (total_ve % 2 == 1)
    n_pi_total = max(0, bo_AB - 1) + max(0, bo_BC - 1)
    bo_span = abs(bo_AB - bo_BC)
    is_linear = (lp_B == 0 and n_pi_total >= 2 and zc[B] == 2
                 and not is_radical)
    is_sblock_linear = (lp_B == 0 and l_B == 0 and zc[B] == 2
                        and per_B >= 2)

    # ── 4. v4 per-bond seeds (diatomic quality) ──
    def _v4_seed(ia, ib, bo):
        Zi, Zj = topology.Z_list[ia], topology.Z_list[ib]
        lp_i, lp_j = topology.lp[ia], topology.lp[ib]
        return _D0_screening_v4(Zi, Zj, bo, lp_A=lp_i, lp_B=lp_j)

    # ── 4a. d-block terminal effective bond order ──────────────
    # When a d-block terminal (z=1) bonds to a partner with LP,
    # the d-vacancy enables LP→d back-donation, increasing the
    # effective bond order. The SMILES bond order (1 for single)
    # misses this channel.
    #
    # PT basis: on Z/(2P₂)Z, the d-vacancy creates P₂ − nf/2
    # empty positions. Terminal LP can donate into these positions,
    # effectively creating a fractional π bond:
    #   bo_eff = bo + d_vac × min(lp_partner, P₁−1) / P₂
    #
    # Gate: d-block (21≤Z≤30 or 39≤Z≤48), z=1 (terminal),
    #        partner has LP > 0.
    # 0 adjustable parameters. All from s = 1/2.
    _bo_eff_AB = bo_AB
    _bo_eff_BC = bo_BC
    for _bond_data, _lbl_bo in [(bond_AB, 'AB'), (bond_BC, 'BC')]:
        _ia_bo, _ib_bo, _bo_bo, _ = _bond_data
        for _idx_d_bo, _idx_p_bo in [(_ia_bo, _ib_bo), (_ib_bo, _ia_bo)]:
            _Zd_bo = topology.Z_list[_idx_d_bo]
            if not (21 <= _Zd_bo <= 30 or 39 <= _Zd_bo <= 48):
                continue
            if topology.z_count[_idx_d_bo] != 1:
                continue  # d-block must be terminal
            _lp_p_bo = topology.lp[_idx_p_bo]
            if _lp_p_bo == 0:
                continue
            _dd_bo = atom_data[_Zd_bo]
            _nf_bo = _dd_bo.get('nf', 0)
            _dv_bo = max(0.0, (2.0 * P2 - _nf_bo)) / (2.0 * P2)
            if _dv_bo <= 0:
                continue
            _lp_eff_bo = min(_lp_p_bo, P1 - 1)
            _delta_bo = _dv_bo * _lp_eff_bo / P2
            if _lbl_bo == 'AB':
                _bo_eff_AB = _bo_bo + _delta_bo
            else:
                _bo_eff_BC = _bo_bo + _delta_bo
            break  # one d-block per bond

    sr_AB = _v4_seed(bond_AB[0], bond_AB[1], _bo_eff_AB)
    sr_BC = _v4_seed(bond_BC[0], bond_BC[1], _bo_eff_BC)
    D0_AB = sr_AB.D0
    D0_BC = sr_BC.D0

    # ── 4b. Deep ionic S_ion restoration ──
    # v4 seeds attenuate S_ion (Pythagorean anti-screening) when
    # EA_acc < S₃×Ry (Born-Haber gate inactive). For deep ionic bonds
    # (ie_ratio < s), restore the attenuated fraction so the ionic P₃
    # contribution enters D₀ at full strength.
    # 0 adjustable parameters — uses the v4 seed's stored face components.
    for sr, lbl in [(sr_AB, 'AB'), (sr_BC, 'BC')]:
        if sr.D_P3 > 0 and sr.D_P1 > 0.01:
            _ia_sr = bond_AB[0] if lbl == 'AB' else bond_BC[0]
            _ib_sr = bond_AB[1] if lbl == 'AB' else bond_BC[1]
            _di_sr = atom_data[topology.Z_list[_ia_sr]]
            _dj_sr = atom_data[topology.Z_list[_ib_sr]]
            _ie_min_sr = min(_di_sr['IE'], _dj_sr['IE'])
            _ie_max_sr = max(_di_sr['IE'], _dj_sr['IE'])
            _ie_ratio_sr = _ie_min_sr / _ie_max_sr if _ie_max_sr > 0 else 1.0
            _ea_max_sr = max(_di_sr.get('EA', 0.0), _dj_sr.get('EA', 0.0))
            if _ie_ratio_sr < S_HALF and _ea_max_sr < S3 * RY:
                # Reconstruct full vs attenuated Pythagorean anti-screening.
                _r_ion_sr = sr.D_P3 / sr.D_P1
                _S_pyth_full = -0.5 * math.log(1.0 + _r_ion_sr ** 2)
                _S_ion_actual = sr.terms.get('S_ion', 0.0)
                if _S_pyth_full < _S_ion_actual < 0:
                    # Partial restoration, ramped by depth into ionic regime.
                    # sqrt ramp: the bifurcation at s activates quickly once
                    # ie_ratio crosses below s — the bond is genuinely ionic.
                    _f_deep_sr = math.sqrt(
                        (S_HALF - _ie_ratio_sr) / S_HALF)
                    _delta_S_sr = _S_pyth_full - _S_ion_actual  # negative
                    _boost_sr = math.exp(-_delta_S_sr * _f_deep_sr)
                    if lbl == 'AB':
                        D0_AB *= _boost_sr
                    else:
                        D0_BC *= _boost_sr

    # ── 4c. Coulomb vertex boost (dim 0) ──────────────────────────
    # When ie_ratio < s AND EA_acc < S₃×Ry (Born-Haber gate inactive),
    # the v4 P₃ face under-contributes because cap_P3 depends on EA.
    # The dim 0 vertex Coulomb energy supplements the deficit:
    #   D_coulomb = q_ct² × COULOMB_EV_A / r_eq × f_Born
    #   boost = max(0, D_coulomb - D_P3_current) × f_depth
    #
    # PT basis (Principe 4): the bifurcation at s = 1/2 switches the
    # bond from dim 1 (edge, covalent) to dim 0 (vertex, Coulomb).
    # The vertex energy scale is COULOMB_EV_A/r, not Ry/P₃.
    # When the Born-Haber gate blocks P₃, the Coulomb vertex energy
    # is the correct dim 0 contribution.
    #
    # Gate: only fire when ONE bond is ionic (MOH type).
    # When BOTH are ionic (M₂O type), F8 bilateral handles it.
    # This avoids double-counting.
    #
    # 0 adjustable parameters. All from s = 1/2.
    _n_deep_ionic = 0
    for (ia_ic, ib_ic, bo_ic, _), sr_ic in [
            (bond_AB, sr_AB), (bond_BC, sr_BC)]:
        _di_ic = atom_data[topology.Z_list[ia_ic]]
        _dj_ic = atom_data[topology.Z_list[ib_ic]]
        _ratio_ic = (min(_di_ic['IE'], _dj_ic['IE'])
                     / max(_di_ic['IE'], _dj_ic['IE'], 1e-10))
        if _ratio_ic < S_HALF:
            _n_deep_ionic += 1

    if _n_deep_ionic == 1:  # exactly one ionic bond (MOH type)
        for (ia_ic, ib_ic, bo_ic, _), sr_ic, lbl_ic in [
                (bond_AB, sr_AB, 'AB'), (bond_BC, sr_BC, 'BC')]:
            _di_ic = atom_data[topology.Z_list[ia_ic]]
            _dj_ic = atom_data[topology.Z_list[ib_ic]]
            _ie_min_ic = min(_di_ic['IE'], _dj_ic['IE'])
            _ie_max_ic = max(_di_ic['IE'], _dj_ic['IE'])
            _ratio_ic = _ie_min_ic / _ie_max_ic if _ie_max_ic > 0 else 1.0
            if _ratio_ic >= S_HALF:
                continue
            _ea_acc_ic = max(_di_ic.get('EA', 0.0), _dj_ic.get('EA', 0.0))
            if _ea_acc_ic >= S3 * RY:
                continue  # Born-Haber gate active → P₃ ok
            # Coulomb vertex energy (dim 0)
            _r_ic = r_equilibrium(
                _di_ic['per'], _dj_ic['per'], bo_ic, 0.0,
                topology.z_count[ia_ic], topology.z_count[ib_ic],
                topology.lp[ia_ic], topology.lp[ib_ic])
            _q_ct = 1.0 - _ratio_ic
            _f_Born = 1.0 - 1.0 / (P1 + 1)  # = 3/4
            _D_coul = _q_ct ** 2 * COULOMB_EV_A / max(_r_ic, 0.5) * _f_Born
            # Boost = Coulomb - existing P3
            _D_P3_cur = sr_ic.D_P3
            _deficit = max(0.0, _D_coul - _D_P3_cur)
            if _deficit > 0:
                # Ramp by depth into ionic regime (sqrt activation)
                _f_dep = math.sqrt((S_HALF - _ratio_ic) / S_HALF)
                if lbl_ic == 'AB':
                    D0_AB += _deficit * _f_dep
                else:
                    D0_BC += _deficit * _f_dep

    # ── 4d. d-block vertex reorganisation boost (dim 0) ──────────
    # Computed here, applied later (after D_face initialised in §7).
    _D_dblock_face = 0.0
    # Cross-face coupling: P₁/P₂ = σ-d overlap on Z/(2P₁)Z × Z/(2P₂)Z.
    # Gates:
    # (1) Partner must have LP > 0 (LP donation into d-vacancy).
    # (2) d-block atom must be a TERMINAL (z=1), not the centre.
    #     When the d-block atom is the centre, the 3×3 Hamiltonian
    #     and face corrections (F4/F6/F7) already capture d-orbital
    #     bonding. The boost is only needed when the d-block vertex
    #     is treated as a simple endpoint by the v4 bilateral seed.
    _cross_P1P2 = float(P1) / P2  # = 3/5 = 0.60
    for (ia_d, ib_d, bo_d, _), lbl_d in [
            (bond_AB, 'AB'), (bond_BC, 'BC')]:
        for _Zd_v, _Zp_v, _id_d, _id_p in [
                (topology.Z_list[ia_d], topology.Z_list[ib_d], ia_d, ib_d),
                (topology.Z_list[ib_d], topology.Z_list[ia_d], ib_d, ia_d)]:
            if not (21 <= _Zd_v <= 30 or 39 <= _Zd_v <= 48):
                continue
            # Gate: d-block must be terminal (z=1)
            if topology.z_count[_id_d] != 1:
                continue
            # Gate: partner must have LP (donation source)
            if topology.lp[_id_p] == 0:
                continue
            _dd_v = atom_data[_Zd_v]
            _dp_v = atom_data[_Zp_v]
            _ea_pv = _dp_v.get('EA', 0.0)
            if _ea_pv >= S3 * RY:
                continue
            _nf_dv = _dd_v.get('nf', 0)
            _d_vac_v = max(0.0, (2.0 * P2 - _nf_dv)) / (2.0 * P2)
            if _d_vac_v <= 0:
                continue
            _f_ea_v = (S3 * RY - _ea_pv) / (S3 * RY)
            _D_dblock_face += _d_vac_v * _f_ea_v * RY * _cross_P1P2
            break  # one d-block vertex per bond

    # ── 5. 3×3 Hamiltonian → f_renorm ──
    eps_A = d_A['IE'] / RY
    eps_B_h = ie_B / RY
    eps_C = d_C['IE'] / RY

    def _hopping(ia, ib):
        di_h = atom_data[topology.Z_list[ia]]
        dj_h = atom_data[topology.Z_list[ib]]
        ea_h, eb_h = di_h['IE'] / RY, dj_h['IE'] / RY
        eg_h = math.sqrt(ea_h * eb_h)
        pm = max(di_h['per'], dj_h['per'])
        lm = min(topology.lp[ia], topology.lp[ib])
        fp = ((2.0 / pm) ** (1.0 / (P1 * P1) if pm == P1
              else 1.0 / P1) if pm > 2 else 1.0)
        fl = max(0.01, 1.0 - lm / (2.0 * max(pm, 1)))
        return S_HALF * eg_h * fl * fp

    t_AB = _hopping(A, B)
    t_BC = _hopping(B, C)
    t_sum = t_AB + t_BC

    eff_lp = lp_B + (S_HALF if is_radical else 0.0)
    if eff_lp > 0 and not is_linear and not is_sblock_linear:
        t_AC = (S_HALF * math.sqrt(eps_A * eps_C)
                * eff_lp / (P2 * max(per_B, 2)))
    else:
        t_AC = 0.0

    _per_max_tri = max(d_A['per'], per_B, d_C['per'])
    if _per_max_tri >= P1:
        _per3_wt = float(_per_max_tri - 2) / P1
        t_AC += D5 * math.sqrt(eps_A * eps_C) * _per3_wt

    H = np.array([
        [eps_A,   -t_AB,  -t_AC],
        [-t_AB,  eps_B_h, -t_BC],
        [-t_AC,  -t_BC,   eps_C]
    ])
    eigvals = np.sort(np.linalg.eigvalsh(H))
    gap_bonding = eigvals[1] - eigvals[0]
    f_renorm = gap_bonding / t_sum if t_sum > 0 else 1.0
    f_renorm = max(0.5, min(1.2, f_renorm))

    # ── 6. Bilateral cooperation on SCF energies ──
    # f_renorm captures the 3-body cooperation that SCF misses.
    # Apply perturbatively: delta = (f_renorm - 1) modulates
    # the sigma channel (P₁ face, dominant contribution).
    if per_B >= P1 and l_B <= 1 and not is_linear and not is_sblock_linear:
        _lp_term_avg = (sum(float(topology.lp[t]) for t in terms)
                        / max(zc[B], 1))
        _f_bilat = (S3 * float(per_B - 2) / P1
                    * (S_HALF + _lp_term_avg * D3))
        _Z_A_b, _Z_C_b = topology.Z_list[A], topology.Z_list[C]
        if _Z_A_b != _Z_C_b:
            _q_asym = (min(d_A['IE'], d_C['IE'])
                       / max(d_A['IE'], d_C['IE']))
            _f_bilat *= _q_asym
        D0_AB *= (1.0 + _f_bilat)
        D0_BC *= (1.0 + _f_bilat)

    # ── 7. Face corrections ──
    cap_face = RY / P2
    D_face = _D_dblock_face  # dim 0 d-block vertex boost (§4d)

    # F1: Vacancy cooperation
    if l_B == 2 and not is_linear:
        nd_B = nf_B
        p2_vac_B = max(0, 2 * P2 - nd_B)
        cap_f1 = RY / P2
        if p2_vac_B > 0 and nd_B < 2 * P2:
            f_vac_frac = p2_vac_B / (2.0 * P2)
            for t in terms:
                lp_t = topology.lp[t]
                if lp_t > 0:
                    d_t = atom_data[topology.Z_list[t]]
                    f_donate = min(lp_t, p2_vac_B) * S_HALF / (2.0 * P2)
                    f_bohr = math.sin(math.pi * min(d_t['IE'], RY) / (2 * RY)) ** 2
                    f_per = 2.0 / max(d_t['per'], 2)
                    D_face += f_donate * f_bohr * f_per * f_vac_frac * cap_f1 * D_FULL
    else:
        p1_vac_B = 0
        if l_B >= 1:
            p1_vac_B = max(0, 2 * P1 - nf_B - round(topology.sum_bo[B]))
        elif l_B == 0:
            p1_vac_B = max(0, 2 - round(topology.sum_bo[B]))
        if p1_vac_B > 0:
            for t in terms:
                lp_t = topology.lp[t]
                if lp_t > 0:
                    d_t = atom_data[topology.Z_list[t]]
                    f_donate = min(lp_t, p1_vac_B) * S_HALF / P2
                    f_bohr = math.sin(math.pi * min(d_t['IE'], RY) / (2 * RY)) ** 2
                    f_per = 2.0 / max(d_t['per'], 2)
                    D_face += f_donate * f_bohr * f_per * cap_face * D_FULL

    # F2: LP cooperation at center (BENT)
    _cos_theta_lp = None
    if lp_B > 0 and not is_linear and not is_sblock_linear:
        z_eff_angle = zc[B] + lp_B * (1.0 + C3)
        if is_radical:
            odd_B = (valence_electrons(Z_B) - round(topology.sum_bo[B])) % 2
            z_eff_angle += odd_B * S_HALF
        if z_eff_angle > 1.001:
            cos_theta = max(-1.0, min(1.0, -1.0 / (z_eff_angle - 1.0)))
            _cos_theta_lp = cos_theta
            sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)
            coop_amp = sin_theta * S_HALF
            f_per_lp = 2.0 / max(per_B, 2)
            D_face += coop_amp * D5 * lp_B / P1 * f_per_lp * cap_face * D_FULL

    # F3: LP directional Coulomb
    if (lp_B >= zc[B] and per_B <= 2
            and not is_linear and not is_sblock_linear):
        if all(topology.lp[t] == 0 for t in terms):
            r_sum, r_count = 0.0, 0
            for (ia, ib, bo_f, _) in [bond_AB, bond_BC]:
                di_f = atom_data[topology.Z_list[ia]]
                dj_f = atom_data[topology.Z_list[ib]]
                r_b = r_equilibrium(di_f['per'], dj_f['per'], bo_f, 0.0,
                                    zc[ia], zc[ib], topology.lp[ia], topology.lp[ib])
                if r_b > 0:
                    r_sum += r_b; r_count += 1
            if r_count > 0:
                r_avg = r_sum / r_count
                f_comp_lp = 1.0 if ie_B >= RY - S3 else 2.0 / 3.0
                D_F3_raw = lp_B * S3 * S_HALF * COULOMB_EV_A / (zc[B] * r_avg) * f_comp_lp
                f_coop = (1.0 - _cos_theta_lp) / 2.0 if _cos_theta_lp is not None else 1.0
                D_face += D_F3_raw * f_coop

    # F4: d-block Madelung bilateral cooperation
    if l_B == 2 and not is_linear:
        ea_terms = [atom_data[topology.Z_list[t]]['EA'] for t in terms]
        if all(ea > S3 * RY for ea in ea_terms):
            ie2_d = IE2_DBLOCK.get(Z_B, 0.0)
            if ie2_d > 0:
                nd = nf_B
                if nd <= P2: f_vac_ionic = 1.0
                elif nd < 2 * P2: f_vac_ionic = float(2 * P2 - nd) / P2
                else:
                    _ns = ns_config(Z_B)
                    f_vac_ionic = float(_ns) / P1 if _ns >= 2 else S_HALF
                for t in terms:
                    lp_t = topology.lp[t]
                    if lp_t > 0:
                        d_t = atom_data[topology.Z_list[t]]
                        f_ie = min(d_t['IE'], RY) / RY
                        f_per = 1.0 + S_HALF * d_t['per'] / P1
                        D_face += (lp_t * S3 / P1 * cap_face * f_ie * f_per
                                   * RY / ie2_d * f_vac_ionic * D_FULL)

    # F4b: d-block bilateral Madelung
    if l_B == 2 and not is_linear:
        nd = nf_B; _ns = ns_config(Z_B)
        _is_d10s2 = (nd >= 2 * P2 and _ns >= 2)
        _is_late_d = (nd > P2 and nd < 2 * P2)
        if _is_d10s2 or _is_late_d:
            ea_terms = [atom_data[topology.Z_list[t]]['EA'] for t in terms]
            if all(ea > S3 * RY for ea in ea_terms):
                ie2_d = IE2_DBLOCK.get(Z_B, 0.0)
                if ie2_d > 0:
                    _f_d = float(_ns) / P1 if _is_d10s2 else float(2*P2 - nd) / P2 * float(_ns) / P1
                    for t in terms:
                        lp_t = topology.lp[t]
                        if lp_t > 0:
                            d_t = atom_data[topology.Z_list[t]]
                            if _is_d10s2 and d_t['per'] < P1: continue
                            if _is_late_d and d_t['per'] >= P1: continue
                            f_per = (float(max(d_t['per'], 2)) / 2.0) ** (1.0 / P1)
                            D_face += float(lp_t) * _f_d * S3 / P1 * cap_face * RY / ie2_d * f_per * D_FULL

    # F5: Late-d coordination screening
    if l_B == 2 and not is_linear and P2 < nf_B < 2 * P2:
        _per_avg = sum(atom_data[topology.Z_list[t]]['per'] for t in terms) / max(zc[B], 1)
        S_late = S3 * float(nf_B - P2) / (2.0 * P2) * min(1.0, _per_avg / P1)
        D0_AB *= math.exp(-S_late)
        D0_BC *= math.exp(-S_late)

    # F5b: d10s1 competition
    _NS1_D10S1 = frozenset({29, 47, 79})
    if l_B == 2 and not is_linear and Z_B in _NS1_D10S1:
        _per_avg = sum(atom_data[topology.Z_list[t]]['per'] for t in terms) / max(zc[B], 1)
        _dp = max(_per_avg - 2.0, 0.0)
        if _dp > 0:
            D0_AB *= math.exp(-S3 * S_HALF * _dp)
            D0_BC *= math.exp(-S3 * S_HALF * _dp)

    # F6: Hund exchange + orbital polarization + CFSE
    _NS1_METALS = frozenset({24, 29, 41, 42, 44, 45, 46, 47, 78, 79})
    if l_B == 2:
        _nf = min(nf_B + (1 if Z_B in _NS1_METALS else 0), 2 * P2)
        if _nf > 0:
            if _nf <= P2:
                _n_unp = min(_nf, 2 * P2 - _nf)
                if _n_unp > 0:
                    _fill = (float(_n_unp) / (2.0 ** (_n_unp - 1)) if _nf >= P2 - 1 and _n_unp > 1
                             else float(_n_unp) / (2.0 * P2))
                    for t in terms:
                        if topology.lp[t] == 0:
                            _ie_t = atom_data[topology.Z_list[t]]['IE']
                            D_face += math.sqrt(ie_B * _ie_t) / RY * _fill * (RY / P2) * S_HALF * D_FULL
            if _nf > P2:
                _n_dn = _nf - P2
                if 0 < _n_dn < P2:
                    _k1 = 2.0 * math.sin(math.pi * _n_dn / P2) ** 2 / (P2 * P2 * math.sin(math.pi / P2) ** 2)
                    for t in terms:
                        if topology.lp[t] == 0:
                            _ie_t = atom_data[topology.Z_list[t]]['IE']
                            D_face += math.sqrt(ie_B * _ie_t) / RY * _k1 * (RY / P2) * S_HALF * D_FULL
            _r7 = _nf % P3
            _sin2 = math.sin(2.0 * math.pi * _r7 / P3) ** 2
            if _sin2 > 1e-10:
                for t in terms:
                    _lp_t = topology.lp[t]
                    if _lp_t > 0:
                        D_face += _sin2 * D7 / P2 * min(float(_lp_t) / P1, 1.0) * (RY / P2) * S_HALF * D_FULL

    # F7: Dark modes
    if l_B == 2:
        _nf = min(nf_B + (1 if Z_B in _NS1_METALS else 0), 2 * P2)
        if _nf > 0:
            _tier = per_B - P2
            if _tier >= 0:
                _w10 = 2.0 * math.pi / (2 * P2)
                _ch = 1 - 2 * (_tier % 2)
                _gd = GAMMA_7 if _tier >= 1 else 1.0
                _Dd = 0.0
                if _tier <= 1:
                    _Dd += -_ch * _gd * (R35_DARK + R57_DARK) * math.cos(2 * _w10 * _nf) * (RY / P2)
                if _tier == 0:
                    _Dd += -_ch * R37_DARK * math.cos(4 * _w10 * _nf) * (RY / P2)
                D_face += _Dd * D_FULL

    # F8: Deep ionic bilateral Madelung cooperation (s/p-block)
    # PT derivation: when BOTH bonds are deep ionic (ie_ratio < s),
    # the bifurcation at s = 1/2 places them on the P₃ face.
    # The bilateral Madelung energy arises from the cooperative
    # charge transfer: both terminals donate to the center, creating
    # a stronger Coulomb field than two independent diatomic bonds.
    #
    # Physics: in diatomic Li-O, the bond is weakly ionic (EA(O) low).
    # In Li₂O, BOTH Li atoms donate → O effectively captures 2 electrons,
    # forming a 3-ion cluster Li⁺-O²⁻-Li⁺ with Madelung stabilisation.
    # This cooperative energy is ABSENT from 2×diatomic seeds.
    #
    # Gate: ie_ratio(A-B) < s AND ie_ratio(B-C) < s AND lp_B > 0.
    # 0 adjustable parameters — all from s = 1/2.
    if not is_radical and lp_B > 0:
        _ie_ratios_f8 = []
        _r_eqs_f8 = []
        for (ia, ib, bo_b, _) in [bond_AB, bond_BC]:
            di_b = atom_data[topology.Z_list[ia]]
            dj_b = atom_data[topology.Z_list[ib]]
            ie_min_b = min(di_b['IE'], dj_b['IE'])
            ie_max_b = max(di_b['IE'], dj_b['IE'])
            _ie_ratios_f8.append(ie_min_b / ie_max_b
                                 if ie_max_b > 0 else 1.0)
            _r_eqs_f8.append(r_equilibrium(
                di_b['per'], dj_b['per'], bo_b, 0.0,
                topology.z_count[ia], topology.z_count[ib],
                topology.lp[ia], topology.lp[ib]))
        if all(r < S_HALF for r in _ie_ratios_f8):
            # Both bonds are deep ionic — bilateral Madelung activates.
            # Charge transfer fraction per bond: f_ct = 1 - ie_ratio.
            # f_ct → 1 for fully ionic, → 0 at ie_ratio = 1.
            _f_ct_avg = sum(1.0 - r for r in _ie_ratios_f8) / 2.0
            _r_avg_f8 = sum(_r_eqs_f8) / 2.0
            if _r_avg_f8 > 0.1:
                # Madelung cooperation:
                #   E = f_ct² × (COULOMB/r) × (n_ionic-1)/n_ionic × C₃
                # where (n_ionic-1)/n_ionic = S_HALF for 2 ionic bonds,
                # C₃ = cos²(π/3) screens higher-face contributions.
                # LP gating: center acceptor capacity.
                _f_lp_f8 = min(float(lp_B) / P1, 1.0)
                E_bilat = (_f_ct_avg ** 2 * COULOMB_EV_A / _r_avg_f8
                           * S_HALF * C3 * _f_lp_f8)
                # ── P₃ deficit gate ──
                # When the v4 seeds already capture significant
                # ionic character (high D_P3), F8 bilateral is
                # double-counting. Gate by the P₃ deficit ratio:
                # if sum(D_P3) ≥ q_ct × COULOMB/r → F8 not needed.
                _p3_sum_f8 = max(sr_AB.D_P3, 0.0) + max(sr_BC.D_P3, 0.0)
                _coul_ref_f8 = (_f_ct_avg * COULOMB_EV_A
                                / max(_r_avg_f8, 0.5))
                _f_p3_deficit = max(0.0, 1.0 - _p3_sum_f8
                                    / max(_coul_ref_f8, 0.01))
                # ── Madelung vertex-vertex repulsion ──
                # The bilateral cluster A⁺-B²⁻-C⁺ has cation-cation
                # repulsion at distance 2r. The Madelung factor for a
                # linear 3-ion chain is (1 - 1/(2n)) where n=2 bonds.
                # f_Mad = 1 - 1/(2×2) = 3/4.
                # PT basis: on Z/(2P₁)Z, the two cations at positions
                # k₀ and k₀+2 repel with coupling 1/(2×n_ionic).
                _n_ionic_f8 = len(_ie_ratios_f8)
                _f_madelung = 1.0 - 1.0 / (2.0 * _n_ionic_f8)
                D_face += E_bilat * _f_p3_deficit * _f_madelung

    # ── 8. Per-type routing (cooperation on SCF energies) ──

    if is_radical:
        # Sigma/pi split with f_renorm (using v4 seed ratios)
        for lbl, sr, bo_r in [('AB', sr_AB, bo_AB), ('BC', sr_BC, bo_BC)]:
            sig_frac = 1.0 / max(bo_r, 1)
            pi_frac = max(0.0, bo_r - 1) / max(bo_r, 1)
            D_sig = sr.D0 * sig_frac * f_renorm
            D_pi = sr.D0 * pi_frac * (1.0 - S_HALF)
            D_rest = (sr.D_P2 + sr.D_P3) * (1.0 - S_HALF)
            if lbl == 'AB':
                D0_AB = D_sig + D_pi + D_rest
            else:
                D0_BC = D_sig + D_pi + D_rest
        # ── Radical budget = IE_centre + d/p-vacancy LP donation ──
        # PT basis (dim 0 vertex capacity) : the centre's IE₁ bounds
        # the σ-channel capacity. But vacant d-orbitals (l=2) or
        # p-orbitals (l=1, per≥3) accept LP donation from terminals,
        # opening additional bonding channels on Z/(2P₂)Z.
        # The additional capacity = d_vac × (Ry/P₂) × LP_total.
        # Gate: terminal LP only (structural donation, not bonding).
        # 0 adjustable parameters. All from s = 1/2.
        _lp_term_rad = sum(topology.lp[t] for t in terms)
        if l_B == 2:
            # d-block: d-vacancy accepts LP on Z/(2P₂)Z
            _d_vac_rad = max(0.0, (2.0 * P2 - nf_B)) / (2.0 * P2)
            _rad_cap = _d_vac_rad * (RY / P2)
        elif l_B >= 1 and per_B >= P1:
            # p-block per≥3: d-orbitals accessible but weaker coupling
            # Coupling via cross-face S₃ (P₁→P₂ off-diagonal)
            _p_vac_rad = max(0.0, (2.0 * P1 - nf_B)) / (2.0 * P1)
            _rad_cap = _p_vac_rad * (RY / P2) * S3
        else:
            # s-block or per<3: no d-orbital bridge
            _rad_cap = 0.0
        radical_budget = (ie_B * D_FULL
                          + _rad_cap * _lp_term_rad * D_FULL)
        D_total_pre = D0_AB + D0_BC + D_face
        if D_total_pre > radical_budget:
            scale_rad = radical_budget / max(D_total_pre, 1e-12)
            D0_AB *= scale_rad; D0_BC *= scale_rad; D_face *= scale_rad

    elif is_linear:
        _dblock_linear = (l_B == 2)
        if _dblock_linear:
            f_renorm = 1.0

        # L1: σ renormalization via 3×3 eigensystem
        # Use ABSOLUTE CRT components (D_P1, D_P2+D_P3), not fractions.
        # The v4 reconstructed components capture more than D0 because
        # D_P1 + D_P2 + D_P3 may exceed D0 (CRT non-orthogonality).
        D_sig_AB, D_nonsig_AB = sr_AB.D_P1, sr_AB.D_P2 + sr_AB.D_P3
        D_sig_BC, D_nonsig_BC = sr_BC.D_P1, sr_BC.D_P2 + sr_BC.D_P3

        if bo_span >= 1.5 and max(bo_AB, bo_BC) >= 3:
            f_triple = 1.0 - (1.0 - f_renorm) / max(bo_AB, bo_BC)
            if bo_AB >= bo_BC:
                D0_AB = D_sig_AB * f_triple + D_nonsig_AB
                D0_BC = D_sig_BC * f_renorm + D_nonsig_BC
            else:
                D0_AB = D_sig_AB * f_renorm + D_nonsig_AB
                D0_BC = D_sig_BC * f_triple + D_nonsig_BC
        else:
            D0_AB = D_sig_AB * f_renorm + D_nonsig_AB
            D0_BC = D_sig_BC * f_renorm + D_nonsig_BC

        # L1b: Symmetric linear bilateral cooperation (CO₂, CS₂, BeH₂)
        # For A=B=A symmetric linears, bilateral stabilization from
        # identical terminals is under-estimated by the tree-level 3×3
        # Hamiltonian. The NLO correction = δ₃² (second-order holonomic
        # on Z/(2P₁)Z): the square of the gap parameter captures the
        # cooperative spectral weight that the 3-body eigensystem misses.
        # 0 parameters. Gate: same terminal element.
        if topology.Z_list[A] == topology.Z_list[C]:
            D0_AB *= (1.0 + D3 * D3)
            D0_BC *= (1.0 + D3 * D3)

        if _dblock_linear:
            _S_comp_d = (1.0 - float(nf_B) / (2.0 * P2)) * float(zc[B]) * S_HALF / P2
            D0_AB *= math.exp(-_S_comp_d)
            D0_BC *= math.exp(-_S_comp_d)

        _HUCKEL_3C = 4.0 - 2.0 * math.sqrt(2.0)
        lp_terms = [topology.lp[t] for t in terms]
        lp_asym = (min(lp_terms) > 0 and min(lp_terms) != max(lp_terms))
        f_huckel = C3 if lp_asym else 1.0
        if not _dblock_linear:
            D_face += _HUCKEL_3C * D5 * S_HALF * n_pi_total * cap_face * D_FULL * f_huckel

        Z_A_lin, Z_C_lin = topology.Z_list[A], topology.Z_list[C]
        if Z_A_lin != Z_C_lin:
            q_rel = min(d_A['IE'], d_C['IE']) / max(d_A['IE'], d_C['IE'])
            asym_boost = (1.0 - q_rel) * S3
            if D0_AB < D0_BC: D0_AB *= (1.0 + asym_boost)
            else: D0_BC *= (1.0 + asym_boost)

        if bo_span >= 1.5 and max(bo_AB, bo_BC) >= 3:
            if bo_AB > bo_BC:
                D0_BC += sr_AB.D_P1 * (bo_AB - 1) / bo_AB * S3 * S_HALF
            else:
                D0_AB += sr_BC.D_P1 * (bo_BC - 1) / bo_BC * S3 * S_HALF

        if lp_asym:
            for t_idx, t in enumerate(terms):
                lp_t = topology.lp[t]
                if lp_t > 0:
                    S_lp = lp_t * S3 / P1
                    if t == A or t_idx == 0: D0_AB *= math.exp(-S_lp)
                    else: D0_BC *= math.exp(-S_lp)

        if bo_span >= 1.5 and max(bo_AB, bo_BC) >= 3:
            for t in terms:
                ve_t = valence_electrons(topology.Z_list[t])
                unused = ve_t - round(topology.sum_bo[t]) - 2 * topology.lp[t]
                if unused > 0:
                    D_face += unused * RY / P1 * S_HALF * D_FULL

    elif is_sblock_linear:
        for t in terms:
            lp_t = topology.lp[t]
            if lp_t > 0:
                d_t = atom_data[topology.Z_list[t]]
                f_ie = min(d_t['IE'], RY) / RY
                f_per = 1.0 + S_HALF * d_t['per'] / P1
                D_face += lp_t * S3 / P1 * cap_face * f_ie * f_per * D_FULL

    else:
        # BENT: LP crowding
        if lp_B >= zc[B]:
            lp_adj_w = sum(topology.lp[t] * (2.0 / max(atom_data[topology.Z_list[t]]['per'], 2)) ** (1.0 / P1)
                           for t in terms)
            if lp_adj_w > zc[B]:
                min_term_lp = min(topology.lp[t] for t in terms)
                f_crowd = (lp_adj_w - zc[B]) / max(lp_adj_w, 1)
                lp_per_term = sum(topology.lp[t] for t in terms) / max(zc[B], 1.0)
                S_crowd = D3 * S_HALF * f_crowd * (1.0 + lp_per_term / P1)
                if min_term_lp >= P1:
                    ie_terms = [atom_data[topology.Z_list[t]]['IE'] for t in terms]
                    S_crowd *= (1.0 + (max(ie_terms) / max(min(ie_terms), 0.1) - 1.0) * S3)
                lp_total = lp_B + sum(topology.lp[t] for t in terms)
                if per_B <= 2 and lp_total > P2:
                    S_crowd += (lp_total - P2) * S3 * S_HALF / zc[B]
                D0_AB *= math.exp(-S_crowd)
                D0_BC *= math.exp(-S_crowd)

        # B2: Coordination from 3×3
        if abs(f_renorm - 1.0) > 0.01:
            D0_AB *= (1.0 + (f_renorm - 1.0) * D3)
            D0_BC *= (1.0 + (f_renorm - 1.0) * D3)

        # B3: Singlet carbene
        if Z_B == 6 and lp_B == 1 and n_pi_total == 0:
            if max(topology.lp[t] for t in terms) > 0:
                polarity = max(d_A['IE'], d_C['IE']) / max(min(d_A['IE'], d_C['IE']), 0.1)
                avg_lp = sum(topology.lp[t] for t in terms) / 2.0
                atten = D5 + D3 * S_HALF * polarity / P1
                atten *= (1.0 + (avg_lp - 1) * S3 / P1)
                atten = max(0.0, min(atten, S3 + D5))
                D0_AB *= (1.0 - atten); D0_BC *= (1.0 - atten)

        # B4: Mixed bond-order bent
        elif bo_span > 0 and n_pi_total > 0 and 0 < lp_B < zc[B]:
            S_mixed = lp_B * S3 / max(zc[B], 1)
            D0_AB *= (1.0 - S_mixed); D0_BC *= (1.0 - S_mixed)
            if D0_AB < D0_BC: D0_AB *= (1.0 - D5)
            else: D0_BC *= (1.0 - D5)

        # B5: Bent π
        elif n_pi_total >= 2 and per_B >= 3:
            if topology.Z_list[A] != topology.Z_list[C]:
                q_rel = min(d_A['IE'], d_C['IE']) / max(d_A['IE'], d_C['IE'])
                D0_AB *= (1.0 - D5 * S_HALF - (1.0 - q_rel) * S3)
                D0_BC *= (1.0 - D5 * S_HALF - (1.0 - q_rel) * S3)
            else:
                D0_AB *= (1.0 - D5 * S_HALF); D0_BC *= (1.0 - D5 * S_HALF)

    # ── 9. Ghost + assembly ──
    D_sum = 0.0
    D_sum_P1, D_sum_P2, D_sum_P3 = 0.0, 0.0, 0.0

    for D0_bond, bi_bond, (ia, ib, bo_b, _) in [
        (D0_AB, bi_AB, bond_AB), (D0_BC, bi_BC, bond_BC)
    ]:
        di_b = atom_data[topology.Z_list[ia]]
        dj_b = atom_data[topology.Z_list[ib]]
        n_shells = ((di_b['per'] - 1) + (dj_b['per'] - 1)) / 2.0
        z_min_b = min(zc[ia], zc[ib])
        f_z_ghost = 1.0 + max(0, z_min_b - 1) * D3
        S_ghost = BETA_GHOST * D3 * S_HALF * max(n_shells, 1.0) * f_z_ghost
        ghost_atten = math.exp(-S_ghost * D_FULL)

        D0_final = D0_bond * ghost_atten
        # Per-face decomposition from v4 seed ratios
        sr = sr_AB if bi_bond == bi_AB else sr_BC
        D0_seed = sr.D0
        scale_b = D0_bond / D0_seed if D0_seed > 1e-10 else 1.0
        D_sum_P1 += sr.D_P1 * scale_b * ghost_atten
        D_sum_P2 += sr.D_P2 * scale_b * ghost_atten
        D_sum_P3 += sr.D_P3 * scale_b * ghost_atten
        D_sum += D0_final

    return CascadeResult(
        D_at=D_sum + D_face,
        D_at_P1=D_sum_P1,
        D_at_P2=D_sum_P2,
        D_at_P3=D_sum_P3,
        iterations=1,
    )


def compute_D_at_cascade(topology: Topology) -> CascadeResult:
    """Compute D_at via cascade on T³.

    Uses v4's proven screening functions for each face,
    with contextual peer floor (no f_size) and sequential
    cascade conditioning P₁ → P₂ → P₃.

    0 adjustable parameters. All from s = 1/2.
    """
    n_atoms = topology.n_atoms
    n_bonds = len(topology.bonds)

    if n_bonds == 0:
        return CascadeResult(0.0, 0.0, 0.0, 0.0, 0)

    # ── ROUTING: Phase 1 / Phase 2 bifurcation ──
    # n=2:   bilateral exact (diatomiques)
    # n=3:   triatomic solver (Phase 1 seeds + 3×3 cooperation)
    # n=4-7: PHASE 1 — Perron eigenvalue λ₀(T⁴) + post-corrections
    # n≥8:   PHASE 2 — per-bond cascade (17 corrections NLO)
    if n_atoms == P1 and n_bonds == 2:
        atom_data = _resolve_atom_data(topology, "v4")
        return _triatomic_v5(topology, atom_data)
    if n_atoms <= P3:
        r4 = compute_D_at_transfer(topology)
        D_at_base = r4.D_at
        # ── Formal charge correction for v4-delegated molecules ──
        # v4 atom_data is keyed by Z (not per-vertex), so it cannot
        # represent charge-shifted IEs natively.  Apply a post-cascade
        # correction: each charge-bearing bond is weakened because the
        # charge separation widens the effective IE gap on Z/(2P₁)Z,
        # increasing the screening by D₃ × |q| per CRT channel.
        #
        # PT basis: formal charge ±1 shifts IE_eff by ±Ry/P₁.  This
        # changes the Bohr factor f_bohr and the polarity Q_eff.
        # Leading-order correction per bond:
        #   δD/D ≈ -D₃ × (|q_i| + |q_j|) / P₁
        # The factor 1/P₁ comes from the P₁ modes sharing the polygon.
        _charges_v4 = getattr(topology, 'charges', None)
        if _charges_v4 is not None and any(q != 0 for q in _charges_v4):
            _delta = 0.0
            D_bond_avg = D_at_base / max(n_bonds, 1)
            for bi_v4, (i_v4, j_v4, bo_v4) in enumerate(topology.bonds):
                qi_v4 = _charges_v4[i_v4] if i_v4 < len(_charges_v4) else 0
                qj_v4 = _charges_v4[j_v4] if j_v4 < len(_charges_v4) else 0
                if qi_v4 == 0 and qj_v4 == 0:
                    continue
                q_abs = abs(qi_v4) + abs(qj_v4)
                # Charge separation increases screening → bond weakening
                _delta -= D3 * q_abs / P1 * D_bond_avg
            D_at_base += _delta
            D_at_base = max(D_at_base, 0.0)
        # ── LP→σ BACK-DONATION (per≤2, isolated LP reference mode) ──
        # v4 captures LP screening but underestimates LP stabilization.
        # PT mechanism: on Z/(2P₁)Z, a SINGLE LP at k=0 creates a
        # reference mode that lifts the Fourier floor for bonding modes
        # (k≥1).  This constructive DFT shift STRENGTHENS each bond.
        #
        # The shift is sin²(θ₃) per steric position (holonomic coupling
        # on the hexagonal face), applied to the LP-vertex's share of
        # the total bond energy.
        #
        # Gate: per ≤ 2 (compact LP, Bohr-localized).
        #       lp = 1 exactly (O with lp=2 has mutual LP stabilization
        #       already captured by v4's bilateral LP balance).
        #       ALL bonding partners have lp = 0 (LP asymmetry is
        #       UNCOMPENSATED — the missing mechanism).
        #       Not for d-block (l ≥ 2).
        #
        # PT basis: sin²(θ₃) = holonomic weight on Z/3Z.  Division by
        # steric = z + lp: the LP occupies 1/(z+lp) of the vertex
        # polygon, so its back-donation is diluted by the steric count.
        _lp_back_boost = 0.0
        for _v_lb in range(n_atoms):
            _Z_lb = topology.Z_list[_v_lb]
            _lp_lb = topology.lp[_v_lb]
            if _lp_lb != 1:
                continue
            _per_lb = period(_Z_lb)
            _l_lb = l_of(_Z_lb)
            if _per_lb > 2 or _l_lb >= 2:
                continue
            _z_lb = topology.z_count[_v_lb]
            if _z_lb == 0:
                continue
            # Check: ALL bonding partners have lp = 0 (uncompensated LP)
            _all_partners_no_lp = True
            for _bk_lb in topology.vertex_bonds.get(_v_lb, []):
                _a_lb, _b_lb, _ = topology.bonds[_bk_lb]
                _p_lb = _b_lb if _a_lb == _v_lb else _a_lb
                if topology.lp[_p_lb] > 0:
                    _all_partners_no_lp = False
                    break
            if not _all_partners_no_lp:
                continue
            # Coupling: S₃ × n_LP / z_v.  The LP creates a k=0 reference
            # on Z/(2P₁)Z with holonomic weight sin²(θ₃).  Each bond
            # receives 1/z_v of the back-donation.  The /z_v (not /steric)
            # is correct because the LP mode couples to BONDING modes only,
            # and there are z_v of them on the vertex polygon.
            _f_lb = S3 * float(_lp_lb) / _z_lb
            _lp_back_boost += _f_lb * _z_lb / max(n_bonds, 1)
        if _lp_back_boost > 0:
            D_at_base *= (1.0 + _lp_back_boost)
        # ── VSEPR hypervalent axial attenuation ──────────────────────
        # [dim 0+2: simplicial complex, Phase 2]
        #
        # PT basis (Principe 4, bifurcation géométrique) :
        # For p-block vertices with steric number (z+lp) > P₁,
        # the VSEPR geometry places bonds in AXIAL positions that
        # live on the pentagonal face Z/(2P₂)Z instead of the
        # hexagonal face Z/(2P₁)Z.
        #
        # The bifurcation is DISCRETE (transition de phase) :
        #   steric ≤ P₁ : all bonds on Z/(2P₁)Z (no attenuation)
        #   steric > P₁ : n_ax bonds on Z/(2P₂)Z (attenuated)
        #
        # Steric = 5 (TBP → T-shape / seesaw) :
        #   LP preferentially occupy equatorial positions.
        #   n_eq_bonds = P₁ - n_eq_lp, n_ax_bonds = z - n_eq_bonds.
        #   Axial attenuation: f_ax = (P₁/P₂) × exp(-lp × S₃).
        #   The exp(-lp × S₃) factor: equatorial LP modes on Z/(2P₁)Z
        #   screen the axial face Z/(2P₂)Z via the cross-face coupling
        #   sin²(π/P₁) (triangle factor on the simplicial boundary).
        #
        # Steric = 6 (octahedral → square pyramidal) :
        #   v4 shell_attenuation already handles z > P₁ excess bonds.
        #   Additional LP trans screening: exp(-lp × S₃).
        #   This is the converse of the TBP case: LP doesn't create
        #   axial/equatorial splitting but screens ALL bonds uniformly
        #   via the P₂ cross-face.
        #
        # Gate: l=1 (p-block), per ≥ P₁ (d-orbitals available for the
        # axial face Z/(2P₂)Z), z+lp > P₁.
        #
        # 0 adjustable parameters. All from s = 1/2.
        for _v_vs in range(n_atoms):
            _Zv_vs = topology.Z_list[_v_vs]
            _lv_vs = l_of(_Zv_vs)
            _perv_vs = period(_Zv_vs)
            if _lv_vs != 1 or _perv_vs < P1:
                continue
            _zv_vs = topology.z_count[_v_vs]
            _lpv_vs = topology.lp[_v_vs]
            _steric_vs = _zv_vs + _lpv_vs
            if _steric_vs <= P1:
                continue

            # ── Steric = 5, z ≤ P₁: TBP-derived T-shape ──
            # Gate: z ≤ P₁ ensures shell_attenuation did NOT fire.
            # These are the molecules where VSEPR splitting is MISSING:
            # ClF₃ (z=3,lp=2), BrF₃, IF₃ — T-shape with 2 LP in
            # equatorial plane. The 2 axial bonds are 3c-4e character
            # and live on Z/(2P₂)Z. Without this correction, v4
            # treats all 3 bonds equally → 50-70% over-estimation.
            #
            # For z > P₁ (seesaw: SF₄, z=4, lp=1), shell_attenuation
            # already reduces the excess bond. No additional correction.
            if _steric_vs == P2 and _zv_vs <= P1:
                _n_eq_lp = min(_lpv_vs, P1 - 1)
                _n_eq_bonds = P1 - _n_eq_lp
                _n_ax_bonds = max(0, _zv_vs - _n_eq_bonds)
                if _n_ax_bonds > 0:
                    _f_ax_vs = (float(P1) / P2) * math.exp(
                        -float(_lpv_vs) * S3)
                    # Fraction of total D_at from bonds at this vertex
                    _n_v_bonds = len(topology.vertex_bonds.get(_v_vs, []))
                    _f_vert = float(_n_v_bonds) / max(n_bonds, 1)
                    # Reduction: axial bonds attenuated from 1.0 to f_ax
                    _f_ax_frac = float(_n_ax_bonds) / max(_zv_vs, 1)
                    _red_vs = (1.0 - _f_ax_vs) * _f_ax_frac * _f_vert
                    D_at_base *= (1.0 - _red_vs)

            # ── Steric ≥ 2P₁, z > P₁, lp > 0: sq pyr phase transition ──
            # shell_attenuation already reduces n_excess = z-P₁ bonds
            # by ×P₁/P₂.  But the remaining P₁ bonds also live on the
            # SATURATED hexagonal face — with lp filling position(s),
            # their effective cap shifts from Ry/P₁ towards Ry/P₂.
            #
            # PT basis: when z+lp = 2P₁, the face Z/(2P₁)Z is fully
            # occupied. The LP at position k₀ screens the bond at k₀+P₁
            # (trans) with full coupling S₃ (antipodal interference).
            # Non-trans bonds see partial screening: S₃ × sin²(angle).
            # Average over the P₁ non-excess bonds: LP screening per
            # non-excess bond = S₃ × lp / z_v (mean-field on polygon).
            #
            # 0 adjustable parameters. All from s = 1/2.
            elif _steric_vs >= 2 * P1 and _zv_vs > P1 and _lpv_vs > 0:
                _n_non_excess = P1  # bonds NOT handled by shell_atten
                _S_lp_sqpyr = S3 * float(_lpv_vs) / _zv_vs
                _n_v_bonds_sq = len(topology.vertex_bonds.get(_v_vs, []))
                _f_vert_sq = float(_n_v_bonds_sq) / max(n_bonds, 1)
                _f_non_excess = float(_n_non_excess) / max(_zv_vs, 1)
                _red_sq = (1.0 - math.exp(-_S_lp_sqpyr)) * _f_non_excess * _f_vert_sq
                D_at_base *= (1.0 - _red_sq)

        D_at_base = max(D_at_base, 0.0)

        D_at_base = max(D_at_base, 0.0)

        # ════════════════════════════════════════════════════════════
        #  v5 CORRECTIONS PORTED TO v4 PATH — STATUS
        #
        #  ✅ LP→σ back-donation (above): VALIDATED, NH₃ -7.6→-0.9%
        #
        #  ❌ LP→π cross-channel: v4 already handles LP+π correctly
        #     via bilateral screening balance. Adding boost = double-
        #     counting → cyanamide +3→+14%, dicyanog +0.25→+14%.
        #
        #  ❌ π-drain, cooperative, ring strain: these are SCREENING
        #     corrections. v4's Perron eigenvalue implicitly captures
        #     the same screening via the spectral gap. Adding them
        #     double-counts → COF₂ -6→-19%, CH₃NH₂ -12→-11%.
        #
        #  Conclusion: v4's transfer matrix is a COMPLETE screening
        #  engine for n≤7. Only the LP→σ reference mode mechanism is
        #  genuinely missing (uncompensated LP creates a k=0 boost
        #  that T⁴ cannot represent).
        # ════════════════════════════════════════════════════════════

        return CascadeResult(
            D_at=D_at_base,
            D_at_P1=r4.D_at_P1,
            D_at_P2=r4.D_at_P2,
            D_at_P3=r4.D_at_P3,
            iterations=0,
        )

    atom_data = _resolve_atom_data(topology, "v4")

    # ── Vertex contexts ──
    ctxs = [VertexCtx(v, topology, atom_data) for v in range(n_atoms)]

    # ── Hückel aromatic caps (v4 function) ──
    huckel_caps = _huckel_aromatic(topology, atom_data)

    # ══════════════════════════════════════════════════════════════
    #  CASCADE: P₀ → P₁ → P₂ → P₃
    # ══════════════════════════════════════════════════════════════
    #
    # ARCHITECTURE — Bifurcation Phase 1 / Phase 2 (avril 2026)
    #
    # Le code ci-dessus (n ≤ P₃) est PHASE 1 : Perron eigenvalue.
    # Le code ci-dessous (n > P₃) est PHASE 2 : cascade per-bond.
    #
    # Le gap Perron entre Phase 1 (λ₀ exact) et Phase 2 (per-bond)
    # est structurellement analogue à T4 Phase 1 (k ≤ 6) : les
    # corrections NLO n'ont pas encore convergé.  Le vecteur propre
    # de Perron EST la limite convergente pour les graphes denses.
    # Aucune correction locale ne peut remplacer l'eigenvalue globale.
    # n = P₃ = 7 est le seuil structurel [T4, Mertens moléculaire].
    # ══════════════════════════════════════════════════════════════

    D_P0 = {}
    D_P1 = {}
    _D_P1_sigma = {}  # sigma-only P₁ for cascade conditioning (CRT k=0)
    D_P2 = {}
    D_P3 = {}
    Q_effs = {}

    # Pre-fetch formal charges (avoid per-bond getattr)
    _charges = getattr(topology, 'charges', None)

    for bi, (i, j, bo) in enumerate(topology.bonds):
        ctx_i, ctx_j = ctxs[i], ctxs[j]
        Zi, Zj = topology.Z_list[i], topology.Z_list[j]
        di, dj = atom_data[Zi], atom_data[Zj]

        # ── Formal charge modulation ──────────────────────────────────
        # A formal charge ±1 represents a DISPLACEMENT of D_KL between
        # CRT channels: the vertex has donated (+1) or accepted (-1) an
        # electron, shifting its effective IE reference by ±Ry/P₁.
        # +1: donated electron → harder to ionize → IE_eff increases
        # -1: accepted electron → easier to ionize → IE_eff decreases
        # Only active for atoms WITH charges; q=0 gives ie_eff = IE.
        qi = _charges[i] if _charges is not None and i < len(_charges) else 0
        qj = _charges[j] if _charges is not None and j < len(_charges) else 0
        ie_i_eff = di['IE'] + qi * RY / P1
        ie_j_eff = dj['IE'] + qj * RY / P1

        # ── FACE P₀ (Z/2Z) ──
        S0 = _screening_P0(i, j, topology, atom_data)
        cap_p0 = _cap_P0(i, j, topology, atom_data)
        D_P0[bi] = cap_p0 * math.exp(-S0) if cap_p0 > 0 else 0.0

        # ── FACE P₁ (Z/6Z) — with contextual floor ──
        S1, Q_eff = _screening_P1_cascade(
            i, j, bo, bi, topology, atom_data, ctx_i, ctx_j)
        Q_effs[bi] = Q_eff

        ie_min = min(ie_i_eff, ie_j_eff)
        f_bohr_P1 = 1.0 if ie_min >= RY * S_HALF else math.sin(math.pi * ie_min / RY) ** 2
        # Parseval vertex filling: z bonds share the polygon Z/(2P₁)Z.
        # Each bond's cap is reduced by the filling fraction.
        # f_fill = (z_i + z_j) / (4P₁) : average filling of both vertices.
        # Parseval filling: only for quasi-homonuclear bonds (|δIE| small)
        # where the DFT peer floor is the ONLY screening source.
        _dIE_cap = abs(ie_i_eff - ie_j_eff)
        _f_fill = (topology.z_count[i] + topology.z_count[j]) / (4.0 * P1)
        if _dIE_cap < D3 * RY:
            _f_homo_cap = 1.0 - _dIE_cap / (D3 * RY)
            cap_P1 = (RY / P1) * f_bohr_P1 * (1.0 - D3 * D3 * _f_fill * _f_homo_cap)
        else:
            cap_P1 = (RY / P1) * f_bohr_P1


        if bi in huckel_caps:
            _D_P1_sigma[bi] = cap_P1 * math.exp(-S1 * D_FULL_P1)
            D_P1[bi] = (cap_P1 + huckel_caps[bi]) * math.exp(-S1 * D_FULL_P1)
        else:
            D_P1_sig = cap_P1 * math.exp(-S1 * D_FULL_P1)
            _D_P1_sigma[bi] = D_P1_sig
            n_pi = max(0.0, bo - 1.0)
            D_pi = 0.0
            if n_pi > 0:
                cap_pi = min(n_pi, 1.0) * (RY / P2)
                if n_pi > 1.0:
                    cap_pi += min(n_pi - 1.0, 1.0) * (RY / P3)
                D_pi = cap_pi * math.exp(-S1 * math.sqrt(C3) * D_FULL_P1)
                # σ-π spectral overlap (GFT anti-double-counting)
                # σ monopole (k=0) doesn't interfere with π (k≥1), but
                # σ dipole (k=1, amplitude D₃) does on the P₁ face.
                # Gate: NOT for d-block bonds. At d-block vertices, the
                # π channel lives on the pentagonal face P₂ (not P₁),
                # so there is no σ-π overlap on P₁. Additionally,
                # d→π* back-donation STRENGTHENS the d-block π channel.
                _is_d_bond = (di.get('is_d_block', False)
                              or dj.get('is_d_block', False))
                if not _is_d_bond:
                    _z_avg = (topology.z_count[i] + topology.z_count[j]) / 2.0
                    _f_sp = D3 * min(_z_avg / P1, 1.0)
                    # Ring enhancement: constrained ring dihedral forces
                    # σ-π orbital alignment, increasing spectral overlap.
                    # Factor (1 + P₁×D₃): total hexagonal face holonomic
                    # correction (P₁ modes × D₃ per mode) activated by
                    # the ring curvature constraint on Z/(2P₁)Z.
                    _ring_bonds = topology.ring_bonds or set()
                    if bi in _ring_bonds:
                        _f_sp *= (1.0 + P1 * D3 * (1.0 + S_HALF))
                    D_pi *= (1.0 - _f_sp)
            D_P1[bi] = D_P1_sig + D_pi

        # ── FACE P₂ (Z/10Z) — conditioned by P₁ ──
        S_P2_dft = (vertex_dft_v5_P2(i, bi, topology, atom_data)
                     + vertex_dft_v5_P2(j, bi, topology, atom_data)) / 2.0
        cap_p2, _ = _cap_P2(i, j, bo, topology, atom_data)
        if cap_p2 > 0:
            ie_min_p2 = min(ie_i_eff, ie_j_eff)
            f_bohr_p2 = math.sin(math.pi * min(ie_min_p2, RY) / (2.0 * RY)) ** 2
            cap_p2 *= f_bohr_p2
        S_P2_eff = S_P2_dft * S_HALF
        D_P2[bi] = cap_p2 * math.exp(-S_P2_eff * D_FULL_P2) if cap_p2 > 0 else 0.0

        # CASCADE: P₂ attenuated by P₁ saturation
        cap_P1_full = (RY / P1) * D_FULL
        # Use sigma-only P₁ for cascade (CRT k=0 orthogonal to pi k=1)
        f_P1_used = min(_D_P1_sigma[bi] / cap_P1_full, 1.0) if cap_P1_full > 0 else 0
        f_P2_cascade = C3 + (1.0 - C3) * (1.0 - f_P1_used)
        D_P2[bi] *= f_P2_cascade

        # Vacancy boost (gated for triple bonds — C22)
        if bo < P1 - S_HALF:
            D_P2[bi] += _dim2_vacancy_boost(i, j, bo, topology, atom_data)

        # ── FACE P₃ (Z/14Z) — conditioned by P₁+P₂ ──
        S_P3_dft = (vertex_dft_v5_P3(i, bi, topology, atom_data)
                     + vertex_dft_v5_P3(j, bi, topology, atom_data)) / 2.0
        cap_p3, _, Q_p3 = _cap_P3(i, j, bo, topology, atom_data)
        S_P3_eff = S_P3_dft * S5 * (1.0 - Q_eff)
        D_P3_raw = cap_p3 * math.exp(-S_P3_eff * DEPTH_P3) if cap_p3 > 0 else 0.0

        # ── cos²(θ_σ) — σ-channel Rabi transmission ──
        # Fraction of the bond NOT resolved by the σ channel.
        # Modulates the ionic mixing Q: polar bonds get more ionic
        # character in the CRT Pythagore combination.
        # cos²(θ) = δIE² / (δIE² + 4 × S₃² × C₃² × IE_A × IE_B)
        _dIE = abs(ie_i_eff - ie_j_eff)
        _denom = _dIE ** 2 + 4.0 * S3 * S3 * C3 * C3 * ie_i_eff * ie_j_eff
        _cos2_sigma = _dIE ** 2 / _denom if _denom > 1e-20 else 0.0

        # Pythagore CRT (Principe 3) with polarity-enhanced Q
        # Q_ionic = Q_eff + cos²(θ_σ) × D₃ × (1 - Q_eff)
        # This shifts polar bonds toward more ionic mixing.
        Q_ionic = Q_eff + _cos2_sigma * D3 * (1.0 - Q_eff)

        # Deep ionic bifurcation: when ie_ratio < s = 1/2, the bond
        # crosses the bifurcation into the P₃-dominant regime.
        # The covalent and ionic channels become ADDITIVE (not orthogonal):
        # the electron has transferred, creating Coulomb attraction PLUS
        # a covalent residual.  Push Q_ionic → 0 (additive limit).
        _ie_ratio_b = (min(ie_i_eff, ie_j_eff)
                       / max(ie_i_eff, ie_j_eff, 1e-10))
        if _ie_ratio_b < S_HALF:
            _f_deep_b = (S_HALF - _ie_ratio_b) / S_HALF
            Q_ionic *= (1.0 - _f_deep_b)

        D_cov = D_P0[bi] + D_P1[bi] + D_P2[bi]
        D_ion = D_P3_raw
        D_add = D_cov + D_ion
        D_pyth = math.sqrt(D_cov ** 2 + D_ion ** 2) if D_ion > 0 else D_cov
        D_P3[bi] = D_add * (1.0 - Q_ionic ** 2) + D_pyth * Q_ionic ** 2 - D_cov

    # ── Build BondSeeds for SCF ──
    bond_seeds: Dict[int, BondSeed] = {}
    _S1_cache = {}  # screening values for cross-face
    for bi, (i, j, bo) in enumerate(topology.bonds):
        D0_total = (D_P0[bi] + D_P1[bi] + D_P2[bi] + D_P3[bi]) * D_FULL
        # Approximate per-face screenings for cross-face coupling
        S1_approx = max(0.01, -math.log(max(D_P1[bi] / (RY / P1), 0.01)) / D_FULL_P1) if D_P1[bi] > 0.01 else 0.5
        S2_approx = 0.0 if D_P2[bi] < 0.01 else 0.1
        S3_approx = 0.0 if D_P3[bi] < 0.01 else 0.1
        bond_seeds[bi] = BondSeed(
            D0=D0_total,
            D0_P0=D_P0[bi], D0_P1=D_P1[bi],
            D0_P2=D_P2[bi], D0_P3=D_P3[bi],
            S0=0.0, S1=S1_approx, S2=S2_approx, S3=S3_approx,
            Q_eff=Q_effs.get(bi, 0.0),
        )

    # ── SCF T⁴ (Perron eigenvector from v4) ──
    if n_atoms >= 3:
        bond_energies_scf, _ = _scf_iterate(topology, atom_data, bond_seeds)
        bond_energies: Dict[int, float] = dict(bond_energies_scf)
    else:
        bond_energies: Dict[int, float] = {bi: s.D0 for bi, s in bond_seeds.items()}

    # ── Shell attenuation (hypervalent vertices z > P_l) ──
    # Excess bonds at per≥3 vertices use sparser circle: ×P_l/P_{l+1}.
    # Imported from v4. Critical for SF₆, PCl₅, SO₃, SCl₂.
    _apply_shell_attenuation(topology, atom_data, bond_energies)

    # ── CROSS-FACE COUNTER-ATTENUATION (excess bond LP restoration) ──
    # Shell attenuation reduces excess bonds by P_l/P_{l+1} (0.6 for l=1).
    # The v4 d-vacancy bridge restores partially (ratio_eff ≈ 0.76-0.80).
    # But a SECOND restoration channel exists: terminal LP couples back
    # into the sparse circle via cross-face σ→P₂ donation (CROSS₃₅).
    # This channel is NOT in v4's shell_attenuation.
    #
    # PT basis: the excess bond lives on Z/(2P_{l+1})Z (sparse circle).
    # Terminal LP (on Z/(2P_l)Z, dense circle) can couple INTO the
    # sparse circle via the off-diagonal block of T³. The coupling
    # strength is d_vacancy × S₃ × lp_terminal / z_v.
    #
    # Only fires at vertices where shell_attenuation already ran
    # (l≥1, per≥3, z > P_l). Targets the n_excess weakest bonds.
    _BLOCK_P_ca = {0: P0, 1: P1, 2: P2, 3: P3}
    for v in range(n_atoms):
        Z_v = topology.Z_list[v]
        d_v = atom_data[Z_v]
        z_v = topology.z_count[v]
        l_v = d_v.get('l', 0)
        per_v = d_v.get('per', 1)
        P_l = _BLOCK_P_ca.get(l_v, P1)
        if l_v < 1 or per_v < P1 or z_v <= P_l:
            continue  # shell_attenuation didn't fire here
        n_excess = z_v - P_l
        nf_v = d_v.get('nf', 0)
        d_vac = max(0.0, (2.0 * P2 - nf_v)) / (2.0 * P2)
        if d_vac <= 0:
            continue
        # Sort bonds by energy (weakest first = attenuated bonds)
        v_bonds = topology.vertex_bonds.get(v, [])
        v_bonds_sorted = sorted(v_bonds,
                                key=lambda bi: bond_energies.get(bi, 0.0))
        for k in range(min(n_excess, len(v_bonds_sorted))):
            bi = v_bonds_sorted[k]
            ii, jj, _ = topology.bonds[bi]
            partner = jj if ii == v else ii
            lp_p = topology.lp[partner]
            if lp_p == 0:
                continue  # no LP to back-donate
            # Cross-face restore: d_vac × S₃ × lp / z_v
            _restore = d_vac * S3 * float(lp_p) / z_v
            bond_energies[bi] *= (1.0 + _restore)

    # ── High-IE back-donation boost (hypervalent + per=2 terminals) ──
    # Shell attenuation removes energy at hypervalent vertices.  But
    # per=2 high-IE terminals (F, O) couple efficiently into the
    # d-vacancy bridge: their compact LP enables strong σ→d
    # back-donation that partially restores the attenuated capacity.
    # v4 captures this through post-hoc corrections; v5 needs an
    # explicit back-donation boost.  Coupling: D₃ × f_IE where
    # f_IE = (IE_terminal / Ry)^{P₁} measures the Bohr-localization
    # fraction.  Only fires at hypervalent vertices (z > P_l) with
    # per=2 partners (the d-bridge targets).
    _BLOCK_P_local = {0: P0, 1: P1, 2: P2, 3: P3}
    for v in range(n_atoms):
        Z_v = topology.Z_list[v]
        d_v = atom_data[Z_v]
        z_v = topology.z_count[v]
        l_v = d_v.get('l', 0)
        per_v = d_v.get('per', 1)
        P_l = _BLOCK_P_local.get(l_v, P1)
        if l_v < 1 or per_v < P1 or z_v <= P_l:
            continue  # not hypervalent or no d-bridge available
        n_excess = z_v - P_l
        nf_v = d_v.get('nf', 0)
        d_vac = max(0.0, (2.0 * P2 - nf_v)) / (2.0 * P2)
        if d_vac <= 0:
            continue
        v_bonds = topology.vertex_bonds.get(v, [])
        for bi in v_bonds:
            ii, jj, _ = topology.bonds[bi]
            partner = jj if ii == v else ii
            Z_p = topology.Z_list[partner]
            d_p = atom_data[Z_p]
            per_p = d_p.get('per', 1)
            if per_p > 2:
                continue  # only per=2 terminals (F, O)
            ie_p = d_p['IE']
            f_ie = min(ie_p / RY, 1.0) ** P1
            # Boost: restore D₃ × d_vacancy × f_IE per excess bond
            boost = D3 * d_vac * f_ie * n_excess / z_v
            D_old = bond_energies.get(bi, 0.0)
            bond_energies[bi] = D_old * (1.0 + boost)

    # ── LP crowding (size-independent, super-Bohr gated) ──
    for v in range(n_atoms):
        _lp_crowding(v, topology, atom_data, bond_energies)

    # ── NLO vertex coupling ──
    _nlo_correction(topology, atom_data, bond_energies)

    # ── π-π VERTEX COMPETITION (cumulene/nitro cross-face) ────────
    # At vertices with 2+ π bonds (bo > 1), the π modes compete on
    # Z/(2P₂)Z.  This spectral competition creates cross-face screening
    # on P₁ via CROSS₃₅ (P₁⊗P₂ off-diagonal coupling).
    # Fires for: cumulenes (C=C=C), nitro groups (N with 2×N=O),
    #            conjugated ketones, etc.
    # Does NOT fire for vertices with only 1 π bond (vinyl, carbonyl).
    # Coupling: CROSS₃₅ × Π(bo_i − 1) / z_v — product of π fractions.
    for v in range(n_atoms):
        z_v = topology.z_count[v]
        if z_v < 2:
            continue
        v_bonds = topology.vertex_bonds.get(v, [])
        _pi_product = 1.0
        _n_pi_bonds = 0
        for _bk in v_bonds:
            _, _, _bo_k = topology.bonds[_bk]
            if _bo_k > 1.0:
                _pi_product *= (_bo_k - 1.0)
                _n_pi_bonds += 1
        if _n_pi_bonds < 2 or _pi_product < S_HALF:
            continue  # need 2+ substantial π bonds (not aromatic 1.5)
        _S_pi_comp = CROSS_35 * _pi_product / z_v
        _f_pi = math.exp(-_S_pi_comp)
        for _bk in v_bonds:
            if _bk in bond_energies:
                bond_energies[_bk] *= _f_pi

    # ── STAR VERTEX CROSS-TERM (2×2 = 2+2 degeneracy) ─────────────
    # At star vertices (all neighbours z=1), the NLO pairwise coupling
    # is killed by f_degen when bonds are degenerate (delta_E ≈ 0).
    # This is correct for the ADDITIVE channel (2+2: sum unchanged by
    # symmetric splitting).  But the MULTIPLICATIVE channel (2×2:
    # correlated pairs) contributes an additional screening that the
    # NLO formula misses at exact degeneracy.
    #
    # 4 = 2+2 = 2×2 is the ONLY integer where addition and multiplication
    # coincide.  This arithmetic identity creates a degeneracy on the
    # P₁ face: 4 mod 3 = 1 (unit element), so the P₁ DFT collapses
    # to a single mode.  The cross-term restores the missing screening.
    #
    # Coupling: D₃ × C(z,2) / (z × P₁) per bond at the star vertex.
    # Fires only at star vertices (all terminals z=1) with z ≥ P₁.
    for v in range(n_atoms):
        z_v = topology.z_count[v]
        if z_v != P1:
            continue  # exactly z=P₁=3 (the 2×2=2+2 point)
        # Check: ALL neighbours are terminals (z=1) → star topology
        v_bonds = topology.vertex_bonds.get(v, [])
        _is_star = True
        for _bk in v_bonds:
            _ii, _jj, _ = topology.bonds[_bk]
            _p = _jj if _ii == v else _ii
            if topology.z_count[_p] != 1:
                _is_star = False
                break
        if not _is_star:
            continue
        # Gate: skip if ANY terminal is super-Bohr (IE > Ry·(1+S₃)).
        # Super-Bohr LP (F) is excluded from LP crowding, so these
        # molecules are ALREADY under-screened. Adding cross-term
        # would over-correct. The cross-term only fires when LP
        # crowding is active (non-super-Bohr terminals: Cl, Br, O, S).
        _has_super_bohr = False
        for _bk in v_bonds:
            _ii, _jj, _ = topology.bonds[_bk]
            _p = _jj if _ii == v else _ii
            if atom_data[topology.Z_list[_p]]['IE'] > RY * (1.0 + S3):
                _has_super_bohr = True
                break
        if _has_super_bohr:
            continue
        # Cross-term: pairwise correlation D₃ × C(z,2) / (z × P₁)
        _n_pairs = z_v * (z_v - 1) // 2
        _S_star = D3 * float(_n_pairs) / (z_v * P1)
        for _bk in v_bonds:
            if _bk in bond_energies:
                bond_energies[_bk] *= math.exp(-_S_star)

    # ── HALOGEN π-DRAIN (inductive withdrawal at conjugated vertices) ─
    # Halogen terminals at conjugated vertices drain π spectral weight.
    # Their LP anti-correlates with the vertex's π modes on Z/(2P₂)Z,
    # transferring weight from P₂ (π) to P₁ (σ).
    #
    # Two tiers (avoids double-counting with LP crowding):
    #  (a) Super-Bohr (F: IE > Ry·(1+S₃)): coupling S₃/z_v (full).
    #      LP crowding excludes these, so π-drain is their ONLY correction.
    #  (b) Non-super-Bohr (Cl, Br, I): coupling D₃·S_HALF/z_v (residual).
    #      LP crowding already provides partial correction; this handles
    #      the π-specific residual that LP crowding misses.
    _RY_SUPER = RY * (1.0 + S3)
    _HALOGENS = frozenset((9, 17, 35, 53))
    for v in range(n_atoms):
        z_v = topology.z_count[v]
        if z_v < 2:
            continue
        v_bonds = topology.vertex_bonds.get(v, [])
        # Check conjugation (any bond at vertex with bo > 1)
        _has_pi = False
        for _bk in v_bonds:
            if topology.bonds[_bk][2] > 1.0:
                _has_pi = True
                break
        if not _has_pi:
            continue
        # Gate super-Bohr tier: only fires for DELOCALIZED π (homonuclear
        # or aromatic ring bonds).  Heteronuclear π (C=O, C=N, S=O) is
        # localized — halogen F doesn't drain a localized π bond.
        _ring_bonds = topology.ring_bonds or set()
        _has_deloc_pi = False
        for _bk in v_bonds:
            _ii, _jj, _bo_k = topology.bonds[_bk]
            if _bo_k <= 1.0:
                continue
            if _bk in _ring_bonds:
                _has_deloc_pi = True   # aromatic
                break
            if topology.Z_list[_ii] == topology.Z_list[_jj]:
                _has_deloc_pi = True   # homonuclear (C=C, N=N)
                break
        # Count halogen terminals by three tiers:
        #   (a) super-Bohr + delocalized π → S₃ coupling (full)
        #   (a') super-Bohr + heteronuclear π → D₃ coupling (mild)
        #   (b) non-super-Bohr → D₃·S_HALF coupling (residual)
        _n_bohr_deloc = 0     # tier (a)
        _n_bohr_hetero = 0    # tier (a')
        _n_nonbohr = 0        # tier (b)
        for _bk in v_bonds:
            _ii, _jj, _ = topology.bonds[_bk]
            _p = _jj if _ii == v else _ii
            if topology.z_count[_p] != 1:
                continue
            _Zp = topology.Z_list[_p]
            if _Zp not in _HALOGENS:
                continue
            _ie_p = atom_data[_Zp]['IE']
            if _ie_p > _RY_SUPER:
                if _has_deloc_pi:
                    _n_bohr_deloc += 1
                else:
                    _n_bohr_hetero += 1
            else:
                _n_nonbohr += 1
        if _n_bohr_deloc == 0 and _n_bohr_hetero == 0 and _n_nonbohr == 0:
            continue
        # Combined drain (three tiers additive, separately capped)
        _S_drain = 0.0
        if _n_bohr_deloc > 0:
            _S_drain += min(S3, S3 * float(_n_bohr_deloc) / z_v)
        if _n_bohr_hetero > 0:
            _S_drain += min(D3, D3 * float(_n_bohr_hetero) / z_v)
        if _n_nonbohr > 0:
            _S_drain += min(D3 * S_HALF,
                            D3 * S_HALF * float(_n_nonbohr) / z_v)
        _f_drain = math.exp(-_S_drain)
        for _bk in v_bonds:
            if _bk in bond_energies:
                bond_energies[_bk] *= _f_drain

    # ── COOPERATIVE SCREENING (graph topology, 0 params) ──────────
    # Two contributions, both topological (connectivity, not size):
    #
    # (a) Second-shell: bonds at neighbor vertices propagate screening
    #     (T₄ iterative convergence). Coupling √(S₃×D₃) attenuated by hop.
    #
    # (b) First-shell LP: lone pairs on sibling terminals block positions
    #     on the vertex's Z/(2P₁)Z bonding circle. LP = structural shield.
    #     Coupling D₃ (delta on Z/(2P₁)Z, direct — no hop attenuation).
    #     Only fires when sibling partners carry LP (H has lp=0 → no screen).
    #     PT basis: LP fills non-bonding slots, reducing the vertex capacity
    #     for each bond. This is the mechanism that v4 captures via C2/C7/C17.
    if n_atoms >= 3:
        for bi, (i, j, bo) in enumerate(topology.bonds):
            D_bi = bond_energies.get(bi, 0.0)
            if D_bi <= 0:
                continue
            n_2nd = 0.0
            lp_sibling_i = 0.0
            lp_sibling_j = 0.0
            for v, lp_acc in ((i, 'i'), (j, 'j')):
                _lp_sib = 0.0
                for bk in topology.vertex_bonds.get(v, []):
                    if bk == bi:
                        continue
                    a, b, _ = topology.bonds[bk]
                    nb = b if a == v else a
                    n_2nd += max(0, topology.z_count[nb] - 1)
                    # LP screening gated by period AND ionization:
                    # Per≥3 (Cl,Br,S): diffuse LP → full blocking.
                    # Per=2, IE ≤ Ry (O,N): moderate LP → S_HALF weight.
                    # Per=2, IE > Ry (F): Bohr-localized → LP stabilizes
                    #   the bond rather than blocking. No vertex screening.
                    Z_nb = topology.Z_list[nb]
                    per_nb = atom_data[Z_nb].get('per', 1)
                    ie_nb = atom_data[Z_nb]['IE']
                    z_nb = topology.z_count[nb]
                    if per_nb >= P1:           # per ≥ 3
                        _lp_sib += topology.lp[nb]
                    elif per_nb == 2 and (ie_nb <= RY or z_nb >= 2):
                        # Per=2 LP screening:
                        # - IE ≤ Ry (O-like): diffuse LP → S_HALF weight
                        # - z ≥ 2 (ring/bridge, e.g. N in pyridine): LP
                        #   occupies polygon positions structurally →
                        #   S_HALF weight regardless of IE
                        # - z=1 terminal, IE > Ry (F): Bohr-localized LP
                        #   stabilizes bond → excluded
                        _lp_sib += topology.lp[nb] * S_HALF
                if lp_acc == 'i':
                    lp_sibling_i = _lp_sib
                else:
                    lp_sibling_j = _lp_sib

            f_total = 1.0

            # (a) Second-shell cooperative screening
            # Coupling depends on cross-channel activity at bond vertices.
            # Full coupling √(S₃×D₃): when LP or π present → T₄ cross-channel
            # (P₁⊗P₂ or P₁⊗P₃) propagates screening through both faces.
            # Reduced coupling D₃: pure σ bonds at LP-free vertices →
            # only single-channel P₁ screening (no cross-face amplification).
            # PT basis: √(S₃×D₃) = geometric mean of two face couplings.
            # D₃ = single-face holonomic correction on Z/(2P₁)Z.
            if n_2nd > 0:
                z_pair = topology.z_count[i] + topology.z_count[j]
                _has_cross = (
                    topology.lp[i] > 0 or topology.lp[j] > 0
                    or bo > 1.0
                    or lp_sibling_i > 0 or lp_sibling_j > 0
                )
                _COOP = math.sqrt(S3 * D3) if _has_cross else D3
                f_coop = math.exp(-_COOP * n_2nd / (max(z_pair, 2) * P1))
                f_total *= f_coop

            # (b) First-shell LP vertex screening
            z_i, z_j = topology.z_count[i], topology.z_count[j]
            if z_i >= 2 and lp_sibling_i > 0:
                f_lp_i = 1.0 - D3 * lp_sibling_i / (z_i * P1)
                f_total *= max(f_lp_i, 1.0 - D3)
            if z_j >= 2 and lp_sibling_j > 0:
                f_lp_j = 1.0 - D3 * lp_sibling_j / (z_j * P1)
                f_total *= max(f_lp_j, 1.0 - D3)

            # (c) Cross-face P₁⊗P₂ (σ-π correlation)
            # When a vertex sees BOTH LP (σ on P₁) and π bonds (on P₂)
            # from its neighbours, the two CRT faces interfere.  This is
            # the CROSS₃₅ channel: the off-diagonal coupling of the
            # Z/6Z × Z/10Z product.
            # Selective: only fires at vertices with LP AND π neighbors.
            # Does NOT fire for saturated fluorides (PF₅ has π=0).
            for v, z_v in ((i, z_i), (j, z_j)):
                if z_v < 2:
                    continue
                _lp_cross = 0.0
                _pi_cross = max(0.0, bo - 1.0)  # π of current bond
                for bk in topology.vertex_bonds.get(v, []):
                    if bk == bi:
                        continue
                    a_k, b_k, bo_k = topology.bonds[bk]
                    nb_k = b_k if a_k == v else a_k
                    Z_nb_k = topology.Z_list[nb_k]
                    per_nb_k = atom_data[Z_nb_k].get('per', 1)
                    lp_nb_k = topology.lp[nb_k]
                    if per_nb_k >= P1 and lp_nb_k > 0:
                        _lp_cross += lp_nb_k
                    elif per_nb_k == 2 and lp_nb_k > 0:
                        _lp_cross += lp_nb_k * S_HALF
                    _pi_cross += max(0.0, bo_k - 1.0)
                if _lp_cross > 0 and _pi_cross > 0:
                    f_cross = 1.0 - CROSS_35 * _lp_cross * _pi_cross / (z_v * P1)
                    f_total *= max(f_cross, 1.0 - CROSS_35)

            if f_total < 1.0:
                bond_energies[bi] = D_bi * f_total

    # ── LP→π CROSS-CHANNEL RESONANCE (inclusion-exclusion P₁∪P₂) ──
    # When an atom has LP (channel P₁) adjacent to LOCALIZED π bonds
    # (channel P₂), LP→π resonance BOOSTS bond energy through the
    # union of orthogonal CRT channels.
    #
    # PT basis (Principe 8, inclusion-exclusion) :
    #   P(P₁ ∪ P₂) = S₃ + S₅ − S₃×S₅ = 1 − C₃×C₅ = 0.371
    # This is the probability of coupling through EITHER channel.
    # The LP on P₁ donates density into π* on P₂, creating additional
    # bonding capacity not captured by either channel alone.
    #
    # Classic example: amide resonance N−C(=O) ↔ N⁺=C(−O⁻).
    # The N lone pair delocalizes into the C=O π*, strengthening
    # the C−N bond and weakening C=O.
    #
    # Gate: LOCALIZED π only (bo > 1 and NOT aromatic ring bond).
    #       Aromatic rings already capture LP→π via Hückel eigenvalues.
    #       LP from ring atoms excluded (σ-type LP, like pyridine N,
    #       does NOT donate into ring π).
    #
    # Coefficient: (S₃+S₅−S₃S₅) × n_LP × n_π_adj / P₁ per bond.
    # The /P₁ comes from sharing the hexagonal face Z/(2P₁)Z with
    # P₁ bonding modes.
    _UNION_P1P2 = S3 + S5 - S3 * S5
    _ring_bonds_lpr = topology.ring_bonds or set()
    _ring_atoms_lpr = topology.ring_atoms if hasattr(topology, 'ring_atoms') else set()
    for v in range(n_atoms):
        lp_v = topology.lp[v]
        if lp_v == 0:
            continue
        # Gate: exclude ring atoms (σ-LP like pyridine N)
        if v in _ring_atoms_lpr:
            continue
        z_v = topology.z_count[v]
        if z_v < 2:
            continue
        # Count LOCALIZED π bonds at or adjacent to this vertex
        v_bonds_lpr = topology.vertex_bonds.get(v, [])
        n_pi_loc = 0.0
        # (a) Direct: π bonds at this vertex (non-ring)
        for bk in v_bonds_lpr:
            _, _, bo_k = topology.bonds[bk]
            if bo_k > 1.0 and bk not in _ring_bonds_lpr:
                n_pi_loc += min(bo_k - 1.0, 1.0)
        # (b) Neighbor: π bonds at adjacent vertices (cross-bond LP→π)
        for bk in v_bonds_lpr:
            a_k, b_k, _ = topology.bonds[bk]
            nb_k = b_k if a_k == v else a_k
            for bm in topology.vertex_bonds.get(nb_k, []):
                if bm == bk:
                    continue
                _, _, bo_m = topology.bonds[bm]
                if bo_m > 1.0 and bm not in _ring_bonds_lpr:
                    n_pi_loc += min(bo_m - 1.0, 1.0) * S_HALF
        if n_pi_loc <= 0:
            continue
        # LP→π boost: union coupling × LP × π_adj / P₁, per bond
        f_lp_pi = _UNION_P1P2 * float(lp_v) * min(n_pi_loc, 2.0) / P1
        for bk in v_bonds_lpr:
            if bk in bond_energies:
                bond_energies[bk] *= (1.0 + f_lp_pi / z_v)

    # ── RING STRAIN (small rings, angular deficit) ──────────────
    # In small rings (3-4 membered), bond angles deviate from the
    # tetrahedral reference (109.47°). The DFT on Z/(2P₁)Z loses
    # angular coverage. Coverage deficit: D₃ × 2/ring_size.
    # PT basis: 3-ring has 60° angles (deficit from 109.47°),
    # 4-ring has 90° angles (still strained). 5-rings (108°) and
    # 6-rings (120°) are unstrained on Z/(2P₁)Z.
    # Gate: ring_size ≤ P₁+1 = 4 (structural, not post-hoc).
    if topology.rings:
        _edge_to_bi = {}
        for _bi_r, (_ii_r, _jj_r, _) in enumerate(topology.bonds):
            _edge_to_bi[(min(_ii_r, _jj_r), max(_ii_r, _jj_r))] = _bi_r
        _bond_min_ring: Dict[int, int] = {}
        for _ring in topology.rings:
            _rsize = len(_ring)
            if _rsize > P1 + 1:     # 3 and 4-membered rings
                continue
            for _ridx in range(_rsize):
                _a_r = _ring[_ridx]
                _b_r = _ring[(_ridx + 1) % _rsize]
                _bi_r = _edge_to_bi.get((min(_a_r, _b_r), max(_a_r, _b_r)))
                if _bi_r is not None:
                    if _bi_r not in _bond_min_ring or _rsize < _bond_min_ring[_bi_r]:
                        _bond_min_ring[_bi_r] = _rsize
        for _bi_r, _rsize in _bond_min_ring.items():
            _f_strain = 1.0 - D3 * 2.0 / _rsize
            if _bi_r in bond_energies:
                bond_energies[_bi_r] *= _f_strain

        # Exocyclic strain: bonds FROM strained-ring vertices to non-ring
        # atoms feel angular distortion (Walsh orbital rehybridization).
        # Weaker than endocyclic: D₃/ring_size (half the ring factor).
        _strained_verts: Dict[int, int] = {}  # vertex → min ring size
        for _ring in topology.rings:
            _rsize = len(_ring)
            if _rsize > P1 + 1:
                continue
            for _v in _ring:
                if _v not in _strained_verts or _rsize < _strained_verts[_v]:
                    _strained_verts[_v] = _rsize
        for _v, _rsize in _strained_verts.items():
            _f_exo = 1.0 - D3 / _rsize
            for _bk in topology.vertex_bonds.get(_v, []):
                if _bk in _bond_min_ring:
                    continue  # already handled as endocyclic
                if _bk in bond_energies:
                    bond_energies[_bk] *= _f_exo

    # ── RING LP DICKE SCREENING [cooperative LP coherence in rings] ──
    # LP modes at ring vertices interfere on the ring torus Z/n_ring Z.
    # Cooperative (Dicke) scaling: n_LP² amplitudes, like vertex coherence
    # (C6) but for non-bonding LP modes. PT basis:
    #   - 6-ring LP is σ-type (in-plane, ⊥ to π) → couples on P₁ face
    #     with strength S₃ (full sin²(θ₃) occupation on Z/6Z).
    #   - 5-ring LP is partly π-donated (Hückel 4n+2 needs heteroatom LP)
    #     → weaker coupling D₃ (one-loop correction on P₁).
    #   - Screening per LP bond: coupling × n_LP² / n_ring, capped at coupling.
    if topology.rings:
        # Reuse edge lookup if already built
        if '_edge_to_bi' not in dir():
            _edge_to_bi = {}
            for _bi_r, (_ii_r, _jj_r, _) in enumerate(topology.bonds):
                _edge_to_bi[(min(_ii_r, _jj_r), max(_ii_r, _jj_r))] = _bi_r
        for _ring in topology.rings:
            _rsize = len(_ring)
            if _rsize < 4 or _rsize > P3:
                continue  # 3-rings: strain; very large: too diffuse
            # Only CONJUGATED non-aromatic rings (avg bo > 1.0, not 1.5).
            # Saturated rings: σ LP → lone-pair crowding (separate).
            # AROMATIC rings (bo=1.5): LP already in Hückel eigenvalues
            # → LP Dicke on top is GFT double-counting.
            _bo_sum = 0.0
            _n_rb = 0
            _n_aro = 0
            for _ridx in range(_rsize):
                _a_r = _ring[_ridx]
                _b_r = _ring[(_ridx + 1) % _rsize]
                _bi_r = _edge_to_bi.get(
                    (min(_a_r, _b_r), max(_a_r, _b_r)))
                if _bi_r is not None:
                    _bo_r = topology.bonds[_bi_r][2]
                    _bo_sum += _bo_r
                    _n_rb += 1
                    if _bo_r == 1.5:
                        _n_aro += 1
            if _n_rb == 0 or _bo_sum / _n_rb <= 1.0:
                continue  # saturated ring — skip
            if _n_aro == _n_rb:
                continue  # fully aromatic — Hückel handles LP
            # Ring-size dependent coupling: 6+ ring σ-LP, 5-ring π-LP
            _coupling = S3 if _rsize >= 2 * P1 else D3
            # Accumulate LP in ring (weighted by connectivity)
            _n_lp_ring = 0.0
            _lp_verts = []
            for _v in _ring:
                _lp_v = topology.lp[_v]
                if _lp_v > 0:
                    _n_lp_ring += _lp_v
                    _lp_verts.append(_v)
            if _n_lp_ring == 0:
                continue
            # Dicke cooperative factor: n_LP² / n_ring, capped
            _f_dicke = min(_coupling,
                          _coupling * _n_lp_ring ** 2 / _rsize)
            # Apply to ring bonds adjacent to LP vertices
            for _v in _lp_verts:
                _v_bonds = topology.vertex_bonds.get(_v, [])
                for _bk in _v_bonds:
                    if _bk not in topology.ring_bonds:
                        continue
                    # Check this bond is in THIS ring
                    _a, _b, _ = topology.bonds[_bk]
                    _other = _b if _a == _v else _a
                    if _other not in _ring:
                        continue
                    if _bk in bond_energies:
                        bond_energies[_bk] *= (1.0 - _f_dicke)

    # ── C5: T³ PERTURBATIVE SCREENING [Z mod {3,5,7}, 0 params] ──
    # Cross-vertex NLO on T³ = Z/3Z × Z/5Z × Z/7Z.
    # Ported from v4 (transfer_matrix.py L6154-6203): pure PT, no calibration.
    _TWO_PI = 2.0 * math.pi
    if n_atoms >= 3:
        _crt = [(Z % P1, Z % P2, Z % P3) for Z in topology.Z_list]
        _aro_bonds = set(bi for bi, (_, _, bo) in enumerate(topology.bonds) if bo == 1.5)
        for bi, (i, j, bo) in enumerate(topology.bonds):
            D_bi = bond_energies.get(bi, 0.0)
            if D_bi <= 0.0:
                continue
            # Gate: skip aromatic bonds. Hückel eigenvalues already
            # capture inter-bond cooperation in aromatic rings. Adding
            # C5 T³ screening on top is GFT double-counting.
            if bi in _aro_bonds:
                continue
            surv_total = 1.0
            for v_center, v_partner in ((i, j), (j, i)):
                r_partner = _crt[v_partner]
                f_screen = [0.0, 0.0, 0.0]
                for bk in topology.vertex_bonds.get(v_center, []):
                    if bk == bi:
                        continue
                    a, b, bo_k = topology.bonds[bk]
                    v_nb = b if a == v_center else a
                    Z_nb = topology.Z_list[v_nb]
                    lp_nb = topology.lp[v_nb]
                    r_nb = _crt[v_nb]
                    for k, (Pk, r_nb_k, r_p_k) in enumerate(
                        zip((P1, P2, P3), r_nb, r_partner)
                    ):
                        if r_nb_k:
                            occ_k = math.sin(_TWO_PI * r_nb_k / Pk) ** 2
                        else:
                            # Identity element on Z/P_kZ: atom is P_k-smooth.
                            # sin²(0) = 0 but LP still screens isotropically.
                            # Direct LP channel: D₃ occupation (one-loop on P₁).
                            if lp_nb > 0:
                                f_screen[k] += lp_nb * D3 / Pk
                            continue
                        dr = abs(r_nb_k - r_p_k)
                        interf = math.cos(_TWO_PI * dr / Pk) ** 2
                        if lp_nb > 0:
                            f_screen[k] += lp_nb * occ_k * interf / Pk
                        if k == 1:
                            for bm in topology.vertex_bonds.get(v_nb, []):
                                if bm == bk:
                                    continue
                                _, _, bo_m = topology.bonds[bm]
                                if bo_m > 1.0:
                                    f_screen[1] += (bo_m - 1.0) * occ_k * interf / P2
                        if k == 2:
                            per_nb = atom_data[Z_nb].get('per', 1)
                            if per_nb > 2:
                                f_screen[2] += (per_nb - 2) * occ_k * interf / P3
                surv_v = (
                    (1.0 - S3 * S3 * min(f_screen[0], 1.0))
                    * (1.0 - S3 * S5 * min(f_screen[1], 1.0))
                    * (1.0 - S3 * S7 * min(f_screen[2], 1.0))
                )
                surv_total *= surv_v
            # Perturbative: D₃ = one-loop attenuation on T³ hexagonal face.
            delta = 1.0 - surv_total
            bond_energies[bi] = D_bi * (1.0 - delta * D3)

    # ── C6: DICKE VERTEX COHERENCE [T1 on molecular graph] ──
    # Ported from v4 (transfer_matrix.py L6205-6235): pure PT.
    if not topology.is_diatomic:
        for v in range(n_atoms):
            z_v = topology.z_count[v]
            if z_v < 2:
                continue
            v_bonds_d = topology.vertex_bonds.get(v, [])
            if len(v_bonds_d) < 2:
                continue
            partner_Z_count: Dict[int, int] = {}
            for bk in v_bonds_d:
                ii_d, jj_d, _ = topology.bonds[bk]
                p_d = jj_d if ii_d == v else ii_d
                Zp = topology.Z_list[p_d]
                partner_Z_count[Zp] = partner_Z_count.get(Zp, 0) + 1
            for Zp_cl, n_cl in partner_Z_count.items():
                if n_cl < 2:
                    continue
                f_dicke = D3 * D3 * (n_cl - 1.0) / (z_v * P1)
                for bk in v_bonds_d:
                    ii_d, jj_d, _ = topology.bonds[bk]
                    p_d = jj_d if ii_d == v else ii_d
                    if topology.Z_list[p_d] == Zp_cl:
                        D_old = bond_energies.get(bk, 0.0)
                        if D_old > 0:
                            bond_energies[bk] = D_old * (1.0 + f_dicke)

    # ── RADICAL BUDGET CAP (triatomics only) ──
    # For small odd-electron molecules (n≤3), the unpaired electron
    # blocks half the pi manifold. Total D_at capped by IE_centre × D_FULL.
    # Only for triatomics — larger radicals use different mechanisms.
    if n_atoms <= P1:
        _total_ve = sum(valence_electrons(Z) for Z in topology.Z_list)
        if _total_ve % 2 == 1:
            D_total_pre = sum(bond_energies.get(bi, 0.0) for bi in range(n_bonds))
            # Find centre with highest IE
            _ctr_rad = -1
            _ie_max_rad = 0.0
            for v in range(n_atoms):
                if topology.z_count[v] >= 2:
                    _ie_v = atom_data[topology.Z_list[v]]['IE']
                    if _ie_v > _ie_max_rad:
                        _ie_max_rad = _ie_v
                        _ctr_rad = v
            if _ctr_rad >= 0:
                _Z_ctr_rad = topology.Z_list[_ctr_rad]
                _d_ctr_rad = atom_data[_Z_ctr_rad]
                _l_ctr_rad = _d_ctr_rad.get('l', 0)
                _per_ctr_rad = _d_ctr_rad.get('per', 1)
                _nf_ctr_rad = _d_ctr_rad.get('nf', 0)
                if _l_ctr_rad == 2:
                    _dv_rad = max(0.0, (2.0 * P2 - _nf_ctr_rad)) / (2.0 * P2)
                    _rad_cap_c = _dv_rad * (RY / P2)
                elif _l_ctr_rad >= 1 and _per_ctr_rad >= P1:
                    _pv_rad = max(0.0, (2.0 * P1 - _nf_ctr_rad)) / (2.0 * P1)
                    _rad_cap_c = _pv_rad * (RY / P2) * S3
                else:
                    _rad_cap_c = 0.0
                _lp_term_rad_c = sum(topology.lp[v] for v in range(n_atoms)
                                     if topology.z_count[v] == 1)
                budget = (_ie_max_rad * D_FULL
                          + _rad_cap_c * _lp_term_rad_c * D_FULL)
                if D_total_pre > budget:
                    scale = budget / D_total_pre
                    for bi in range(n_bonds):
                        bond_energies[bi] = bond_energies.get(bi, 0.0) * scale

    # ── Ghost + assembly ──
    D_sum = 0.0
    D_sum_P1 = 0.0
    D_sum_P2 = 0.0
    D_sum_P3 = 0.0

    for bi, (i, j, bo) in enumerate(topology.bonds):
        g = _ghost(ctxs[i], ctxs[j])
        D_final = bond_energies[bi] * g
        ratio = bond_energies[bi] / max((D_P0[bi] + D_P1[bi] + D_P2[bi] + D_P3[bi]) * D_FULL, 1e-10)
        D_sum += D_final
        D_sum_P1 += D_P1[bi] * D_FULL * g * ratio
        D_sum_P2 += D_P2[bi] * D_FULL * g * ratio
        D_sum_P3 += D_P3[bi] * D_FULL * g * ratio

    return CascadeResult(
        D_at=D_sum,
        D_at_P1=D_sum_P1,
        D_at_P2=D_sum_P2,
        D_at_P3=D_sum_P3,
        iterations=1,
    )
