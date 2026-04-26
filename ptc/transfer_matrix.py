"""PTC Transfer Matrix Engine — SCF on T⁴.

Phase 0: Foundation.

The molecule IS a polygon on T³ = Z/P₁Z × Z/P₂Z × Z/P₃Z.
Each ATOM occupies a vertex. Each BOND is an edge.
The energy of atomization decomposes by CRT face:

    D_at = Σ_bonds [D₀(P₁) + D₀(P₂) + D₀(P₃)] × D_FULL × ghost
         + E_spectral

where D₀(P_l) = T_l[i,j] (off-diagonal of per-face transfer matrix).

The T_mol spectrum encodes multi-center effects:
  λ₀ (Perron)   → total molecular screening
  λ₁ gap        → bifurcation σ/π, screening uniformity
  Degeneracies  → resonance, conjugation (Hückel emerges)
  Eigenvectors  → charge distribution, VSEPR angles

CONSTRAINTS:
  - 0 adjustable parameters (PT constants only: sin²θ_p, γ_p, s, P_l, Ry)
  - 0 external data (no Pauling EN, no VSEPR tables)
  - All screening from: IE/Ry, LP/(2P_l), per/P_l

March 2026 — Théorie de la Persistance
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque

from ptc.constants import (
    S_HALF, P0, P1, P2, P3,
    S3, S5, S7, C3, C5, C7,
    D3, D5, D7, AEM, RY,
    D_FULL, BETA_GHOST,
    D3_TH, COULOMB_EV_A,
    CROSS_35, CROSS_37, CROSS_57, COS_37,
    GAMMA_3, GAMMA_7, R35_DARK, R57_DARK, R37_DARK,
    IE2_DBLOCK,
    F_BIFURC, EPS_3, EPS_5, EPS_7,    # ε-field [A8]
    F_EPS, D_FULL_P1, D_FULL_P2, D_FULL_P3, DEPTH_P3,  # depth corrections
)
from ptc.atom import IE_eV, EA_eV
from ptc.topology import Topology, valence_electrons
from ptc.data.experimental import IE_NIST, EA_NIST, MASS
from ptc.periodic import period, l_of, _n_fill_aufbau, ns_config, _np_of
from ptc.bond import BondResult, r_equilibrium, omega_e
from ptc.dft_polygon import (electron_density, dft_spectrum,
                              electron_density_Z30, dft_spectrum_Z30,
                              vertex_cross_spectrum_Z30)
from ptc.screening_bond_v4 import _np_of as _np_of_v4, D0_screening as _D0_screening_v4

# ── T³ screening weight [Z mod {3,5,7}] ────────────────────────────
_TWO_PI = 2.0 * math.pi

# ── Element group sets ──────────────────────────────────────────────
_HALOGENS = frozenset({9, 17, 35, 53})
_CHALCOGENS = frozenset({16, 34, 52})
_BLOCK_P = {1: P1, 2: P2, 3: P3}  # angular momentum l → prime P_l

# ── Dimer D₀ cache for C9c sub-saturated polygon override ──────────
# Memoized homo/heteronuclear D₀ from a direct engine call on the
# 2-atom topology. Re-entrancy is safe: dimers have no rings, so
# C9c does not trigger inside the recursive call.
_DIMER_D_CACHE: Dict[Tuple[int, int], float] = {}


def _dimer_D_cached(Za: int, Zb: int) -> float:
    """Memoized dimer D₀ (eV) via the engine itself."""
    key = (min(Za, Zb), max(Za, Zb))
    if key in _DIMER_D_CACHE:
        return _DIMER_D_CACHE[key]
    from ptc.smiles_parser import _SYM
    from ptc.topology import build_topology
    if Za >= len(_SYM) or Zb >= len(_SYM):
        _DIMER_D_CACHE[key] = 0.0
        return 0.0
    sa, sb = _SYM[Za], _SYM[Zb]
    if not sa or not sb:
        _DIMER_D_CACHE[key] = 0.0
        return 0.0
    smiles = f"[{sa}][{sb}]"
    try:
        topo = build_topology(smiles)
        res = compute_D_at_transfer(topo)
        D = float(res.D_at)
    except Exception:
        D = 0.0
    _DIMER_D_CACHE[key] = D
    return D


def _huckel_polygon_multiplier(N: int, n_e: int) -> float:
    """Sum_k n_k cos(2π k / N) over Aufbau-filled cycle modes.

    For a homonuclear ring of N atoms with n_e σ-delocalized electrons,
    cycle-Hückel modes ε_k = α - 2t cos(2πk/N) (β = -t < 0). Filling in
    pairs from largest cosine (most bonding) gives the per-D_dimer
    multiplier of the polygon stabilization, calibrated by D_dimer = 2t.
    """
    if N < 2 or n_e <= 0:
        return 0.0
    cosines = sorted([math.cos(_TWO_PI * k / N) for k in range(N)], reverse=True)
    rem = n_e
    mult = 0.0
    for c in cosines:
        if rem >= 2:
            mult += 2.0 * c
            rem -= 2
        elif rem == 1:
            mult += 1.0 * c
            rem -= 1
        else:
            break
    return mult


def _is_s1_outer(Z: int) -> bool:
    """True if outer shell is ns¹ with no p valence (Group 1 or Group 11).

    These are the σ-aromatic sub-saturated atoms: alkali (Li, Na, K, …)
    and coinage (Cu, Ag, Au). The np filter excludes excited-state s¹
    p-block species (would never occur from neutral SMILES).
    """
    return ns_config(Z) == 1 and _np_of(Z) == 0


def _chi(d: dict) -> float:
    """Mulliken electronegativity: (IE + EA) / 2."""
    return (d['IE'] + d['EA']) / 2.0


# ╔════════════════════════════════════════════════════════════════════╗
# ║  DATA TYPES                                                        ║
# ╚════════════════════════════════════════════════════════════════════╝

@dataclass
class TransferResult:
    """Result of T_mol computation."""
    D_at: float
    D_at_P1: float = 0.0
    D_at_P2: float = 0.0
    D_at_P3: float = 0.0
    E_spectral: float = 0.0
    bonds: List[BondResult] = field(default_factory=list)
    spectrum_P1: Optional[np.ndarray] = None
    spectrum_P2: Optional[np.ndarray] = None
    spectrum_P3: Optional[np.ndarray] = None
    topology: Optional[Topology] = None
    mechanism: Optional[str] = None
    mechanism_score: Optional[float] = None
    functional_state: Optional["PolyFunctionalState"] = None
    functional_weights: Optional["PolyFunctionalWeights"] = None


@dataclass(frozen=True)
class AtomState:
    """Local PT descriptors for one atom inside one molecular topology."""
    idx: int
    Z: int
    IE: float
    EA: float
    mass: float
    nf: int
    ns: int
    per: int
    l: int
    chi: float
    formal_charge: int
    lp: int
    z_count: int
    sum_bo: float
    vacancy: int
    p1_vacancy: int
    remaining_valence: int
    odd_signature: bool
    radical: bool
    is_terminal: bool
    is_d_block: bool


@dataclass(frozen=True)
class BondState:
    """Local PT descriptors for one bond before collective couplings."""
    bond_idx: int
    i: int
    j: int
    bo: float
    atom_i: AtomState
    atom_j: AtomState
    chi_i: float
    chi_j: float
    q_eff: float
    q_rel: float
    ie_geom: float
    per_min: int
    per_max: int
    l_min: int
    l_max: int
    lp_min: int
    lp_max: int
    z_min: int
    z_max: int
    is_homonuclear: bool
    is_cross_period: bool
    r_eq: float


# ── NEW: 4-face DFT seed and T⁴ coordinates ────────────────────────

@dataclass(frozen=True)
class BondSeed:
    """Per-bond energy seed from 4-face DFT before SCF."""
    D0: float
    D0_P0: float
    D0_P1: float
    D0_P2: float
    D0_P3: float
    S0: float
    S1: float
    S2: float
    S3: float
    Q_eff: float


# ╔════════════════════════════════════════════════════════════════════╗
# ║  ATOM DATA                                                        ║
# ╚════════════════════════════════════════════════════════════════════╝

def _resolve_atom_data(topology: Topology, source: str = "full_pt") -> dict:
    """Build atom_data dict for all unique Z in topology."""
    unique_Z = set(topology.Z_list)
    atom_data = {}
    for Z in unique_Z:
        if source == "full_nist":
            ie = IE_NIST.get(Z, IE_eV(Z, continuous=True))
            ea = EA_NIST.get(Z, 0.0)
        else:
            ie = IE_eV(Z, continuous=True)
            ea = EA_eV(Z, ie)
        atom_data[Z] = {
            'IE': ie, 'EA': ea,
            'mass': MASS.get(Z, 2.0 * Z),
            'nf': _n_fill_aufbau(Z),
            'ns': ns_config(Z),
            'per': period(Z),
            'l': l_of(Z),
            'is_d_block': period(Z) >= 4 and l_of(Z) == 2,
        }
    return atom_data


def _resolve_atom_states(topology: Topology, atom_data: dict) -> List[AtomState]:
    """Build per-atom PT descriptors anchored in the current topology."""
    atom_states: List[AtomState] = []
    for idx, Z in enumerate(topology.Z_list):
        d = atom_data[Z]
        formal_charge = _formal_charge_at(topology, idx)
        lp = topology.lp[idx]
        z_count = topology.z_count[idx]
        sum_bo = topology.sum_bo[idx] if topology.sum_bo else 0.0
        vacancy = topology.vacancy[idx] if topology.vacancy else 0
        remaining_valence = valence_electrons(Z) - formal_charge - round(sum_bo)
        atom_states.append(AtomState(
            idx=idx,
            Z=Z,
            IE=d['IE'],
            EA=d['EA'],
            mass=d['mass'],
            nf=d['nf'],
            ns=d['ns'],
            per=d['per'],
            l=d['l'],
            chi=_chi(d),
            formal_charge=formal_charge,
            lp=lp,
            z_count=z_count,
            sum_bo=sum_bo,
            vacancy=vacancy,
            p1_vacancy=max(0, 2 * P1 - round(sum_bo) - 2 * lp),
            remaining_valence=remaining_valence,
            odd_signature=(valence_electrons(Z) - round(sum_bo)) % 2 == 1,
            radical=remaining_valence % 2 == 1,
            is_terminal=z_count <= 1,
            is_d_block=d.get('is_d_block', False),
        ))
    return atom_states


def _build_bond_state(bond_idx: int,
                      topology: Topology,
                      atom_states: List[AtomState]) -> BondState:
    """Build local descriptors for one bond from atom states."""
    i, j, bo = topology.bonds[bond_idx]
    atom_i = atom_states[i]
    atom_j = atom_states[j]
    ie_i = atom_i.IE / RY
    ie_j = atom_j.IE / RY
    return BondState(
        bond_idx=bond_idx,
        i=i,
        j=j,
        bo=bo,
        atom_i=atom_i,
        atom_j=atom_j,
        chi_i=atom_i.chi,
        chi_j=atom_j.chi,
        q_eff=abs(atom_i.chi - atom_j.chi) / max(0.1, atom_i.chi + atom_j.chi),
        q_rel=min(ie_i, ie_j) / max(ie_i, ie_j),
        ie_geom=math.sqrt(ie_i * ie_j),
        per_min=min(atom_i.per, atom_j.per),
        per_max=max(atom_i.per, atom_j.per),
        l_min=min(atom_i.l, atom_j.l),
        l_max=max(atom_i.l, atom_j.l),
        lp_min=min(atom_i.lp, atom_j.lp),
        lp_max=max(atom_i.lp, atom_j.lp),
        z_min=min(atom_i.z_count, atom_j.z_count),
        z_max=max(atom_i.z_count, atom_j.z_count),
        is_homonuclear=atom_i.Z == atom_j.Z and atom_i.formal_charge == atom_j.formal_charge,
        is_cross_period=atom_i.per != atom_j.per,
        r_eq=r_equilibrium(atom_i.per, atom_j.per, bo),
    )


def _formal_charge_at(topology: Topology, idx: int) -> int:
    """Formal charge carried by one vertex, defaulting to neutral."""
    charges = topology.charges if topology.charges else []
    return int(charges[idx]) if idx < len(charges) else 0


# ╔════════════════════════════════════════════════════════════════════╗
# ║  PER-FACE DFT SCREENING                                           ║
# ║                                                                    ║
# ║  6 screening components, ALL from PT quantities:                   ║
# ║    IE/Ry, LP/(2P₁), per/P₁                                       ║
# ║  No lookup tables. No Pauling EN. No VSEPR tiers.                 ║
# ╚════════════════════════════════════════════════════════════════════╝

def _vertex_polygon_dft(vertex: int, bond_idx: int,
                         topology: Topology, atom_data: dict,
                         r_bonds: Dict[int, float] = None) -> float:
    """Screening from one vertex via polygon DFT on Z/(2P₁)Z.

    Direct molecular analogue of atom.py's S_polygon → DFT pipeline.
    Each vertex has a polygon Z/(2P_l)Z where bonds and LP occupy positions.
    The screening at the bond's position = Fourier synthesis of peer values.

    Pipeline (same as atom.py):
      1. Peer values: screening strength at each polygon position
      2. DFT analysis: decompose into (a_k, b_k) polygon amplitudes
      3. DFT synthesis: reconstruct at the bond's position
    """
    Z = topology.Z_list[vertex]
    d = atom_data[Z]
    z_v = topology.z_count[vertex]
    lp_v = topology.lp[vertex]
    per_v = d['per']
    is_d_block = d.get('is_d_block', False)
    N = 2 * (P2 if is_d_block else P1)

    # ── 1. Peer values at each polygon position ──
    # Positions: bonds (sorted by partner IE) then LP then empty.
    # Bond position: σ = partner's screening deficit (1 - IE_p/Ry) × S₃
    # LP position: σ = S₃ × D₃ (face blocking, like atom.py sig0)
    # Empty position: σ = 0

    bonds_at_v = []
    for bi in topology.vertex_bonds.get(vertex, []):
        ii, jj, bo = topology.bonds[bi]
        partner = jj if ii == vertex else ii
        Z_p = topology.Z_list[partner]
        ie_p = atom_data[Z_p]['IE']
        bonds_at_v.append((bi, partner, ie_p, bo))
    # Sort by partner IE descending (strongest bonds first, like Aufbau)
    bonds_at_v.sort(key=lambda x: -x[2])

    vals = [0.0] * N

    # Place bonds at positions 0..z-1
    target_pos = -1  # which position has our bond
    for pos, (bi, partner, ie_p, bo) in enumerate(bonds_at_v):
        if pos >= N:
            break
        f_ie_p = min(ie_p / RY, 1.0)
        # Peer value = partner's screening deficit
        # High IE partner (F, O) → low σ → good bond
        # Low IE partner (Na, Al) → high σ → weak bond
        #
        # Floor [C18]: minimum per-bond screening on Z/(2P_l)Z.
        # Default D₇ (heptagonal self-screening ≈ 0.090).
        # For compact sp³ centres (per ≤ 2, z > P₁, lp = 0) bonding
        # with LP-free partners (H), sp-hybridisation reduces the
        # residual screening. The floor interpolates between D₇
        # (large molecules where SCF handles cooperation) and S₃×D₃
        # (small molecules where cooperation is under-counted).
        # f_size = max(0, 1 − (n − P₁)/(P₃ − P₁)) fades for n ≥ P₃.
        # 0 adjustable parameters: D₇, S₃, D₃ from s = 1/2.
        _peer_floor = D7
        if per_v <= 2 and z_v > P1 and lp_v == 0:
            _lp_partner = topology.lp[partner]
            if _lp_partner == 0:
                _f_size_c18 = max(0.0, 1.0 - float(
                    topology.n_atoms - P1) / (P3 - P1))
                _peer_floor = D7 * (1.0 - _f_size_c18) + S3 * D3 * _f_size_c18
        Z_p = topology.Z_list[partner]
        vals[pos] = max(_peer_floor, (1.0 - f_ie_p) * S3)
        # Period crossing adds to peer value
        per_p = atom_data[topology.Z_list[partner]]['per']
        if abs(per_v - per_p) > 0:
            per_min = min(per_v, per_p)
            f_rel = (S_HALF if per_min == 1
                     else P1 / max(per_v, per_p) if max(per_v, per_p) > P1
                     else 1.0)
            vals[pos] += math.sqrt(abs(per_v - per_p)) * D3 * f_rel
        # Geometric overlap feedback [SPN]
        if r_bonds is not None and bi in r_bonds:
            r_ref = r_equilibrium(per_v, per_p, bo)
            r_actual = r_bonds[bi]
            if r_ref > 0 and r_actual > r_ref:
                # Stretched bond → less overlap → peer value decreases
                vals[pos] *= math.sqrt(r_ref / r_actual)
        if bi == bond_idx:
            target_pos = pos

    # Place LP at positions z..z+lp-1
    for k in range(min(lp_v, N - min(z_v, N))):
        pos = min(z_v, N - 1) + k
        if pos >= N:
            break
        # LP screening per LP on the local polygon.
        # d-block vertices live on Z/10Z: the LP occupies the pentagonal face.
        if is_d_block:
            vals[pos] = D5 * max(lp_v, 1)
        elif z_v >= P1 and lp_v > 0:
            vals[pos] = S3 * max(lp_v, 1)
        else:
            vals[pos] = D3 * max(lp_v, 1)
        # Radial amplification for per ≥ 3
        n_rad = max(0, d['per'] - 2)
        if n_rad > 0 and z_v <= 1:
            vals[pos] *= (1.0 + n_rad * D3)

    if target_pos < 0:
        return 0.0  # bond not found at this vertex (shouldn't happen)

    # LP cooperation: z ≥ 2 with LP → face rigidity → reduces LP peer values
    if z_v >= 2 and lp_v > 0:
        z_eff = z_v + min(lp_v, 3) * (1.0 + S_HALF)
        if z_eff > 1.0:
            theta = math.acos(max(-1.0, min(1.0, -1.0 / (z_eff - 1.0))))
            # Cooperation FRACTION: what fraction of LP screening is canceled
            # by geometric cooperation. σ_face = D₅ sin(θ/2) LP/P₁.
            # Full cooperation → LP peer value × (1 - σ_face).
            sigma_face = D5 * math.sin(theta / 2.0) * min(float(lp_v), float(P1)) / P1
            f_coop = max(0.0, 1.0 - sigma_face)
            for pos in range(min(z_v, N), min(z_v + lp_v, N)):
                vals[pos] *= f_coop

    # ── 2. DFT analysis (same as atom.py _dft_analysis) ──
    omega = 2.0 * math.pi / N
    ns = range(1, N + 1)
    coeffs = [(sum(vals) / N,)]  # k=0: mean field
    for m in range(1, N // 2 + 1):
        if m < N // 2:
            am = 2.0 / N * sum(v * math.cos(m * omega * n)
                               for v, n in zip(vals, ns))
            bm = 2.0 / N * sum(v * math.sin(m * omega * n)
                               for v, n in zip(vals, ns))
            coeffs.append((am, bm))
        else:  # Nyquist
            am = 1.0 / N * sum(v * math.cos(m * omega * n)
                               for v, n in zip(vals, ns))
            coeffs.append((am,))

    # ── 3. DFT synthesis at FILLING position ──
    # LP weight on filling: terminal (z≤1) → full weight, polyatomic (z≥2) → S₃.
    # Terminal LP is structural (N₂, F₂: lp IS the polygon).
    # Polyatomic LP cooperates → reduced weight on filling position.
    lp_weight = 1.0 if z_v <= 1 else S3
    n_fill = max(1.0, min(z_v + lp_v * lp_weight, float(N)))
    S = coeffs[0][0]  # k=0 mean field
    for m in range(1, N // 2 + 1):
        idx = m
        if m < N // 2:
            S += (coeffs[idx][0] * math.cos(m * omega * n_fill)
                  + coeffs[idx][1] * math.sin(m * omega * n_fill))
        else:
            S += coeffs[idx][0] * math.cos(m * omega * n_fill)

    # ── 4. BIFURCATION at z + lp = P₁ [atom.py _polygon_bifurcation] ──
    # Before half-fill (z+lp < P₁): screening (orientation +)
    # At half-fill (z+lp = P₁): transition (× S_HALF)
    # After half-fill (z+lp > P₁): enhancement (orientation -)
    filling = z_v + lp_v
    if filling == P1:
        S *= S_HALF  # transition: half strength
    elif filling > P1:
        # Enhancement: LP-rich atoms (F, Cl, O with lp≥2) REDUCE screening
        # because they're past half-fill on the hexagonal polygon.
        # The Fisher parabola 4f(1-f) peaks at f=0.5 and drops to 0 at f=0,1.
        f_fill = float(filling) / N
        fisher = 4.0 * f_fill * (1.0 - f_fill)  # peaks at 0.5, zero at 0 and 1
        S *= fisher

    # ── 4b. PER-3 BOND-ORDER FACE CLOSURE [hypervalent d-orbital screening] ──
    # For per≥3 atoms, π electrons from multiple bonds also occupy the P₁
    # face through d-orbital participation. When bo_sum > z (hypervalent),
    # the face is more occupied than z+lp alone indicates.
    #
    # The effective face occupation fraction is bo_sum/(2P₁). When this
    # exceeds P₁/N = 0.5 (half-fill), the face approaches closure:
    # each additional π electron adds screening ∝ sin²(π·bo_fill/2).
    #
    # PT derivation: closed polygon Z/(2P₁)Z → Parseval <sin²> = s = 1/2.
    # The transition to closure goes as sin²(π·f/2) where f = bo_fill.
    # Gated by sin²(π·per/(2P₂)): d-orbital access strength.
    #
    # For SO₃ (z=3, bo=6): bo_fill = 1.0 → sin²(π/2) = 1 → full closure
    # For DMSO (z=3, bo=4): bo_fill = 0.67 → sin²(π/3) = 0.75
    # For DMS (z=2, bo=2): bo_fill = 0.33 → no excess → 0
    bo_sum_v = topology.sum_bo[vertex]
    bo_excess = round(bo_sum_v) - z_v
    if per_v >= P1 and bo_excess > 0 and not is_d_block:
        bo_fill = min(1.0, round(bo_sum_v) / float(N))
        f_closure = math.sin(math.pi * bo_fill / 2.0) ** 2
        f_d_access = math.sin(math.pi * per_v / (2.0 * P2)) ** 2
        # The closure screening adds on top of the DFT screening.
        # Scale: S₃ × closure × d-access × bo_excess_frac
        # S₃ (not D₃): full sin² coupling because the closure is a FACE
        # saturation (not just a dispersion); the closed polygon has
        # Parseval average = s = S_HALF.
        S += S3 * f_closure * f_d_access * float(bo_excess) / float(z_v)

    # ── 5. CURVATURE: 2-loop corrections [atom.py _polygon_curvature] ──
    # Cross-gap corrections: modes from other faces couple into this polygon.
    # k=0 (mean field): AEM correction for period-2
    if per_v == 2 and z_v >= 2:
        S += AEM * C3 * (z_v - 1) / P1  # 2-loop hexagonal dressing
    # k=1 (dipole): Fisher cosine CROSS₃₅ projected through D₃ [P3a]
    # COS_35 > 0 → P₁⊗P₂ aligned → adds screening (same sign as tree-level)
    if per_v >= P1:
        tier = per_v - P1
        S += CROSS_35 * D3 * min(tier + 1, P2 - 1) / (P2 - 1)

    return max(0.0, S)


def _vertex_polygon_dft_P2(vertex: int, bond_idx: int,
                            topology: Topology, atom_data: dict) -> float:
    """Screening from one vertex via polygon DFT on Z/(2P₂)Z (pentagonal face).

    Same pipeline as P₁ DFT but on the pentagonal polygon (10 positions).
    Occupants: π bonds (bo > 1 component) + LP.
    σ-only bonds do NOT occupy P₂ positions — they live on P₁.

    Pipeline:
      1. Peer values: π screening strength at each pentagonal position
      2. DFT analysis: decompose into Fourier amplitudes on Z/10Z
      3. DFT synthesis: reconstruct at the bond's position
      4. Bifurcation at n_π + lp = P₂ (half-fill)
      5. Curvature: cross-gap corrections R₃₅, R₅₇

    All from s = 1/2.
    """
    Z = topology.Z_list[vertex]
    d = atom_data[Z]
    z_v = topology.z_count[vertex]
    lp_v = topology.lp[vertex]
    per_v = d['per']
    N = 2 * P2  # 10 positions

    vals = [0.0] * N

    # ── 1. Peer values: π bonds sorted by partner IE ──
    pi_bonds = []
    for bi in topology.vertex_bonds.get(vertex, []):
        ii, jj, bo = topology.bonds[bi]
        if bo > 1.0:
            partner = jj if ii == vertex else ii
            Z_p = topology.Z_list[partner]
            ie_p = atom_data[Z_p]['IE']
            pi_bonds.append((bi, partner, ie_p, bo))
    pi_bonds.sort(key=lambda x: -x[2])

    # Also count σ bonds for LP-σ cross-coupling awareness
    n_sigma = z_v - len(pi_bonds)

    if not pi_bonds:
        return 0.0  # no π bonds → P₂ face empty (LP stays on P₁)

    # Place π bonds at positions 0..n_pi-1
    target_pos = -1
    for pos, (bi, partner, ie_p, bo) in enumerate(pi_bonds):
        if pos >= N:
            break
        f_ie_p = min(ie_p / RY, 1.0)
        Z_p = topology.Z_list[partner]
        # Peer value: partner's screening deficit × pentagonal coupling × π weight
        vals[pos] = max(D7, (1.0 - f_ie_p) * S5) * (bo - 1.0)
        # LP of partner: cross-face LP→π coupling
        lp_p = topology.lp[partner]
        if lp_p > 0:
            vals[pos] += lp_p * D5 / P2
        # Period crossing on P₂ face
        per_p = atom_data[Z_p]['per']
        if abs(per_v - per_p) > 0:
            f_rel = (S_HALF if min(per_v, per_p) == 1
                     else P2 / max(per_v, per_p) if max(per_v, per_p) > P2
                     else 1.0)
            vals[pos] += math.sqrt(abs(per_v - per_p)) * D5 * f_rel
        # σ neighbors amplify π screening (σ-π cross-coupling on polygon)
        if n_sigma > 0:
            vals[pos] += n_sigma * D5 * D3 / P2
        if bi == bond_idx:
            target_pos = pos

    # Place LP at positions after π bonds
    n_pi = min(len(pi_bonds), N)
    for k in range(min(lp_v, N - n_pi)):
        pos = n_pi + k
        if pos >= N:
            break
        # LP peer on pentagonal face: D₅ × lp_count
        vals[pos] = D5 * max(lp_v, 1)
        # Radial amplification for per ≥ 3
        n_rad = max(0, per_v - 2)
        if n_rad > 0 and z_v <= 1:
            vals[pos] *= (1.0 + n_rad * D5)

    # σ-only bond: probe at position after all π + LP
    if target_pos < 0:
        target_pos = min(n_pi + lp_v, N - 1)
        if target_pos < 0:
            return 0.0

    # LP cooperation on P₂ (same structure as P₁)
    if z_v >= 2 and lp_v > 0:
        z_eff = z_v + min(lp_v, P2) * (1.0 + S_HALF)
        if z_eff > 1.0:
            theta = math.acos(max(-1.0, min(1.0, -1.0 / (z_eff - 1.0))))
            sigma_face = D5 * math.sin(theta / 2.0) * min(float(lp_v), float(P2)) / P2
            f_coop = max(0.0, 1.0 - sigma_face)
            for pos in range(n_pi, min(n_pi + lp_v, N)):
                vals[pos] *= f_coop

    # ── 2. DFT analysis on Z/10Z ──
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

    # ── 3. DFT synthesis at filling position ──
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

    # ── 4. BIFURCATION at n_π + lp = P₂ ──
    filling_P2 = len(pi_bonds) + lp_v
    if filling_P2 == P2:
        S *= S_HALF
    elif filling_P2 > P2:
        f_fill = float(filling_P2) / N
        fisher = 4.0 * f_fill * (1.0 - f_fill)
        S *= fisher

    # ── 5. CURVATURE: Fisher cosine cross-face corrections [P3a] ──
    # CROSS₃₅ from P₁ face into P₂ (COS_35 > 0 → adds screening)
    if per_v == 2 and len(pi_bonds) >= 1:
        S += CROSS_35 * D5 * C5 * len(pi_bonds) / P2
    # CROSS₅₇ from P₃ face into P₂ (COS_57 < 0 → subtracts screening)
    if per_v >= P2:
        tier = per_v - P2
        S -= CROSS_57 * D5 * min(tier + 1, P3 - 1) / (P3 - 1)

    return max(0.0, S)


def _vertex_polygon_dft_P3(vertex: int, bond_idx: int,
                            topology: Topology, atom_data: dict) -> float:
    """Screening from one vertex via polygon DFT on Z/(2P₃)Z (heptagonal face).

    The P₃ face captures steric overcrowding — when z + LP exceeds the
    hexagonal capacity (2P₁), the excess spills onto the heptagonal polygon.

    Occupants: ALL bonds + LP (total steric load).
    The heptagonal polygon has 14 positions — room for large coordination.

    Pipeline:
      1. Peer values: steric weight at each heptagonal position
      2. DFT analysis: Fourier on Z/14Z
      3. DFT synthesis: at filling position
      4. Bifurcation at z + lp = P₃ (half-fill)
      5. Curvature: R₃₇, R₅₇

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

    # ── 1. Peer values: ALL bonds sorted by partner period (heavier = more steric) ──
    bonds_at_v = []
    for bi in topology.vertex_bonds.get(vertex, []):
        ii, jj, bo = topology.bonds[bi]
        partner = jj if ii == vertex else ii
        Z_p = topology.Z_list[partner]
        per_p = atom_data[Z_p]['per']
        bonds_at_v.append((bi, partner, per_p, bo))
    bonds_at_v.sort(key=lambda x: -x[2])

    target_pos = -1
    for pos, (bi, partner, per_p, bo) in enumerate(bonds_at_v):
        if pos >= N:
            break
        Z_p = topology.Z_list[partner]
        # Steric peer value: heavier period = more steric pressure
        vals[pos] = math.sqrt(max(per_p - 1, 0)) * D7
        # LP of partner adds steric bulk
        lp_p = topology.lp[partner]
        if lp_p > 0:
            vals[pos] += lp_p * D7 / P3
        # Period crossing: cross-period bonds create steric mismatch
        if abs(per_v - per_p) > 0:
            vals[pos] += math.sqrt(abs(per_v - per_p)) * D7 * S_HALF
        # H→heavy mismatch: 1s orbital much smaller than 3sp
        if min(Z, Z_p) == 1 and max(per_v, per_p) >= P1:
            vals[pos] += D3 * math.sqrt(max(per_v, per_p) - 1) / P1
        if bi == bond_idx:
            target_pos = pos

    # Place LP at remaining positions
    n_bonds_placed = min(z_v, N)
    for k in range(min(lp_v, N - n_bonds_placed)):
        pos = n_bonds_placed + k
        if pos >= N:
            break
        vals[pos] = D7 * max(lp_v, 1)
        # d-orbital promotion for per ≥ 3 with LP
        if per_v >= P1 and lp_v > 0:
            n_promoted = max(0, total_load - 4)
            if n_promoted > 0:
                vals[pos] += n_promoted * D5 / max(z_v, 1)

    if target_pos < 0:
        return 0.0

    # ── 2. DFT analysis on Z/14Z ──
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

    # ── 3. DFT synthesis ──
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

    # ── 4. BIFURCATION at z + lp = P₃ ──
    if total_load == P3:
        S *= S_HALF
    elif total_load > P3:
        f_fill = float(total_load) / N
        fisher = 4.0 * f_fill * (1.0 - f_fill)
        S *= fisher

    # ── 5. CURVATURE: Fisher cosine cross-face corrections [P3a] ──
    # CROSS₃₇ from P₁ face (COS_37 < 0 → subtracts screening)
    if per_v >= P1:
        tier = per_v - P1
        S -= CROSS_37 * D7 * min(tier + 1, P3 - 1) / (P3 - 1)
    # CROSS₅₇ from P₂ face (COS_57 < 0 → subtracts screening)
    if per_v >= P2:
        tier = per_v - P2
        S -= CROSS_57 * D7 * min(tier + 1, P3 - 1) / (P3 - 1)

    return max(0.0, S)


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Z/30Z VERTEX 3-BODY SCREENING  [CRT product P₁⊗P₂]            ║
# ║                                                                    ║
# ║  Physics: The hexagonal face Z/(2P₁)Z has only P₁=3 modes,       ║
# ║  which cannot distinguish molecules with same bond orders but     ║
# ║  different partner chemistry (CO₂ vs CS₂ both give V=0.111).     ║
# ║  Z/30Z = Z/(2P₁·P₂)Z has 15 modes, encoding both hex and        ║
# ║  pent faces via CRT: Z/30Z ≅ Z/2Z × Z/3Z × Z/5Z.               ║
# ║                                                                    ║
# ║  The cross-modes (k%3≠0, k%5≠0) carry inter-face coupling that   ║
# ║  resolves the hex degeneracy. Partner IE, LP, and period enter    ║
# ║  through the pentagonal coordinate on Z/30Z.                      ║
# ║                                                                    ║
# ║  0 adjustable parameters. All from s = 1/2.                       ║
# ╚════════════════════════════════════════════════════════════════════╝

def _vertex_3body_Z30(vertex: int, topology: Topology,
                      atom_data: dict,
                      bond_energies: Dict[int, float]) -> Dict[int, float]:
    """Compute per-bond Z/30Z vertex 3-body correction.

    For a vertex with z >= 2 bonds, the Z/30Z DFT encodes each bond's
    contribution using both hexagonal (p-shell) and pentagonal (screening
    environment) coordinates via CRT.

    The correction is an ADDITIONAL screening or enhancement term:
      ΔD(bond_i) = -D₃/(z×P₁) × Σ_{j≠i} V_Z30(i,j) × F_Bohr

    where V_Z30(i,j) is the cross-spectral power between bonds i and j
    on Z/30Z, weighted by CRT mode type.

    Returns dict {bond_idx: correction_eV}.
    """
    z_v = topology.z_count[vertex]
    if z_v < 2:
        return {}

    v_bonds = topology.vertex_bonds.get(vertex, [])
    if len(v_bonds) < 2:
        return {}

    Z_center = topology.Z_list[vertex]
    d_center = atom_data[Z_center]
    ie_c = d_center['IE']
    per_c = d_center['per']
    l_c = d_center.get('l', 0)
    lp_v = topology.lp[vertex]
    np_c = d_center.get('nf', 0) if l_c == 1 else 0

    f_bohr = min(ie_c / RY, 1.0)

    # Build Z/30Z density for each bond, encoding partner properties
    # in the pentagonal coordinate.
    bond_rhos = {}
    for bi in v_bonds:
        ii, jj, bo = topology.bonds[bi]
        partner = jj if ii == vertex else ii
        Z_p = topology.Z_list[partner]
        d_p = atom_data[Z_p]
        ie_p = d_p['IE']
        per_p = d_p['per']
        l_p = d_p.get('l', 0)
        lp_p = topology.lp[partner]

        # Hex electrons: from center's contribution to this bond
        if l_c == 1:
            n_hex = min(int(bo) + 2 * lp_v, 2 * P1)
        elif l_c == 0:
            n_hex = min(2, 2 * P1)
        else:
            n_hex = min(int(bo), 2 * P1)
        lp_hex = min(lp_v, P1)

        # Pent electrons: encoding partner's screening environment.
        # Partner IE maps to pent filling: high IE → more filled
        # Partner LP maps to pent LP: LP block pent positions
        ie_fill = min(ie_p / RY, 1.0) * P2  # 0 to 5 based on IE
        n_pent = max(1, min(round(ie_fill), 2 * P2))

        # Partner LP as pentagonal LP (blocking)
        lp_pent = min(lp_p, P2)

        # Period factor: cross-period partners shift pent filling
        if abs(per_c - per_p) > 0:
            n_pent = max(1, min(n_pent + abs(per_c - per_p), 2 * P2))

        rho_30 = electron_density_Z30(n_hex, n_pent, lp_hex, lp_pent)
        bond_rhos[bi] = rho_30

    # Compute pairwise cross-spectra on Z/30Z
    corrections = {}
    for bi in v_bonds:
        corrections[bi] = 0.0

    bi_list = list(v_bonds)
    for a_idx in range(len(bi_list)):
        bi_a = bi_list[a_idx]
        D_a = bond_energies.get(bi_a, 0.0)
        if D_a <= 0:
            continue

        for b_idx in range(a_idx + 1, len(bi_list)):
            bi_b = bi_list[b_idx]
            D_b = bond_energies.get(bi_b, 0.0)
            if D_b <= 0:
                continue

            # Z/30Z cross-spectrum
            V_cross = vertex_cross_spectrum_Z30(
                bond_rhos[bi_a], bond_rhos[bi_b],
                S3_weight=S3, S5_weight=S5
            )

            # NLO coupling strength modulated by cross-spectrum
            V_base = D3 * math.sqrt(D_a * D_b) / (z_v * P1)

            # The cross-spectrum modulates the coupling:
            # High V_cross → bonds share spectral modes → stronger coupling
            # Low V_cross → bonds spectrally orthogonal → weaker coupling
            # Normalization: V_cross / (S3 * S5 * 0.5) maps to [0, ~1]
            V_norm = abs(V_cross) / max(S3 * S5 * S_HALF, 1e-10)
            f_z30 = min(V_norm, 2.0)

            # 2nd order perturbation with Z/30Z modulation
            V_eff = V_base * (1.0 + f_z30 * S3)
            delta_E = abs(D_a - D_b)
            E_pair = -(V_eff * V_eff) / (delta_E + V_eff) if (delta_E + V_eff) > 0 else 0

            # The DIFFERENCE between Z/30Z-modulated and standard coupling
            V_std = V_base
            E_pair_std = -(V_std * V_std) / (delta_E + V_std) if (delta_E + V_std) > 0 else 0

            delta_corr = (E_pair - E_pair_std) * f_bohr

            # Distribute proportionally
            E_total = D_a + D_b
            if E_total > 0 and delta_corr != 0:
                corrections[bi_a] += delta_corr * D_a / E_total
                corrections[bi_b] += delta_corr * D_b / E_total

    return corrections


def _dim3_overcrowding(i: int, j: int, topology: Topology,
                       atom_data: dict) -> Tuple[float, float]:
    """Dim 3 screening: polyhedron overcrowding at vertices.

    [P₃=7 simplex level]
    Returns (S_dim3, sigma_3):
      S_dim3  — contribution to screening S (-ln(1-sigma_3) or linear fallback)
      sigma_3 — raw overcrowding fraction (needed by bifurcation block)

    Active when z+LP exceeds hexagonal capacity 2P₁.
    """
    Zi, Zj = topology.Z_list[i], topology.Z_list[j]
    di, dj = atom_data[Zi], atom_data[Zj]
    per_i, per_j = di['per'], dj['per']
    lp_i = topology.lp[i]
    lp_j = topology.lp[j]
    z_i = topology.z_count[i]
    z_j = topology.z_count[j]

    z_max = max(z_i, z_j)
    lp_max = max(lp_i, lp_j)

    # Overcrowding: z+LP exceeds hexagonal capacity 2P₁
    z_lp = z_max + lp_max
    sigma_3 = 0.0
    if z_max > P1 + 1:
        sigma_3 += max(0, z_max - P1 - 1) * D3
    if z_lp > 2 * P1:
        sigma_3 += (z_lp - 2.0 * P1) / (2.0 * P1) * D3

    # d-orbital promotion: steric > 4 with LP at per ≥ P₁
    for z, lp, per in [(z_i, lp_i, per_i), (z_j, lp_j, per_j)]:
        if per >= P1 and lp > 0:
            n_promoted = max(0, z + lp - 4)
            if n_promoted > 0:
                sigma_3 += n_promoted * D5 / max(z, 1)

    # H→heavy shell mismatch: 1s covers only fraction of 3sp
    if min(Zi, Zj) == 1 and max(per_i, per_j) >= P1:
        sigma_3 += D3 * math.sqrt(max(per_i, per_j) - 1) / P1

    S_dim3 = 0.0
    if 0 < sigma_3 < 1:
        S_dim3 = -math.log(1.0 - sigma_3)
    elif sigma_3 >= 1:
        S_dim3 = sigma_3  # linear fallback for very overcrowded

    return S_dim3, sigma_3


def _dim2_lp_mutual(i: int, j: int, topology: Topology,
                    atom_data: dict) -> float:
    """Dim 2 screening: LP cross-polygon coupling.

    [P₂=5 simplex level — 3-body: LP_i interacts with LP_j through bond]
    Returns S_mutual contribution.
    """
    Zi, Zj = topology.Z_list[i], topology.Z_list[j]
    di, dj = atom_data[Zi], atom_data[Zj]
    per_i, per_j = di['per'], dj['per']
    lp_i = topology.lp[i]
    lp_j = topology.lp[j]
    z_i = topology.z_count[i]
    z_j = topology.z_count[j]

    lp_mutual = min(lp_i, lp_j)
    S_mutual = 0.0
    if lp_mutual >= 1:
        nf_a, nf_b = atom_data[Zi]['nf'], atom_data[Zj]['nf']
        is_structural = (lp_mutual == 1
                         and nf_a in (P1, 2 * P1) and nf_b in (P1, 2 * P1))
        if is_structural:
            S_mutual = S3 * D3
        else:
            per_e = math.sqrt(per_i * per_j)
            density = (2.0 / max(per_e, 2)) ** 2
            z_max_bond = max(z_i, z_j)
            if abs(per_i - per_j) == 1 and z_max_bond <= 1:
                density *= C3
            S_mutual = S3 * lp_mutual * density
            n_radial = max(0, max(per_i, per_j) - 2)
            if n_radial > 0:
                S_mutual *= (1.0 + n_radial * D3)
            if z_max_bond <= 1:
                chi_a = _chi(di)
                chi_b = _chi(dj)
                Q_lp = abs(chi_a - chi_b) / max(0.1, chi_a + chi_b)
                S_mutual *= (1.0 + Q_lp)
            if lp_mutual >= P1:
                S_mutual *= (1.0 + S3)

    return S_mutual


def _dim1_exchange(i: int, j: int, topology: Topology,
                   atom_data: dict) -> Tuple[float, float]:
    """Dim 1 screening: exchange, Bohr resonance, polarity enhancement.

    [P₁=3 simplex level — pairwise vertex effects]
    Returns (S_dim1, Q_eff):
      S_dim1 — NEGATIVE contribution to S (enhances bond)
      Q_eff  — polarity asymmetry (needed by bifurcation block)
    """
    Zi, Zj = topology.Z_list[i], topology.Z_list[j]
    di, dj = atom_data[Zi], atom_data[Zj]
    per_i, per_j = di['per'], dj['per']
    z_i = topology.z_count[i]
    z_j = topology.z_count[j]

    S_dim1 = 0.0

    # Parity (homonuclear = true exchange, maximal)
    sigma_0 = 0.0
    if Zi == Zj:
        sigma_0 = S3 * S_HALF  # homonuclear exchange
    elif per_i == per_j:
        sigma_0 = D3 * S_HALF  # same-period partial exchange

    # Bohr resonance: IE ≈ Ry → enhanced exchange
    sigma_0_bohr = 0.0
    dev_i = abs(di['IE'] - RY) / RY
    dev_j = abs(dj['IE'] - RY) / RY
    dev_max = max(dev_i, dev_j)
    if dev_max < D3:
        sigma_0_bohr = D3 * (1.0 - dev_i / D3) * (1.0 - dev_j / D3)
    elif dev_max < S3:
        sigma_0_bohr = D3 * (1.0 - dev_i / S3) * (1.0 - dev_j / S3)
    elif (dev_max < C3 and max(z_i, z_j) <= 1
          and max(di['IE'], dj['IE']) > RY):
        sigma_0_bohr = D7 * (1.0 - dev_i / C3) * (1.0 - dev_j / C3)
    if sigma_0_bohr > 0:
        ea_prod = max(di['EA'], 0.01) * max(dj['EA'], 0.01)
        f_EA = min(1.0, math.sqrt(ea_prod) / (RY * D3 * S_HALF))
        sigma_0_bohr *= f_EA

    sigma_0_total = sigma_0 + sigma_0_bohr
    # Exchange is ATTRACTIVE → -ln(1 + σ₀) [NEGATIVE contribution]
    if sigma_0_total > 0:
        S_dim1 += -math.log(1.0 + sigma_0_total)

    # Polarity enhancement (vertex asymmetry reduces screening)
    chi_i = _chi(di)
    chi_j = _chi(dj)
    Q_eff = abs(chi_i - chi_j) / max(0.1, chi_i + chi_j)
    si_raw = max(D7, (1.0 - min(di['IE'] / RY, 1.0)) * S3)
    sj_raw = max(D7, (1.0 - min(dj['IE'] / RY, 1.0)) * S3)
    sigma_asym = S_HALF * abs(si_raw - sj_raw) * (1.0 + Q_eff)
    if sigma_asym > 0:
        S_dim1 += -math.log(1.0 + sigma_asym)  # NEGATIVE (enhancement)

    return S_dim1, Q_eff


def _screening_P1(i: int, j: int, bo: float,
                   topology: Topology, atom_data: dict,
                   bond_idx: int = -1,
                   r_bonds: Dict[int, float] = None) -> float:
    """Screening S for bond (i,j) on P₁ face — simplicial cascade.

    dim 3 (P₃) : overcrowding     → _dim3_overcrowding
    dim 2+1    : vertex DFT        → _vertex_polygon_dft (unchanged)
    dim 2 (P₂) : LP mutual        → _dim2_lp_mutual
    dim 1 (P₁) : exchange/Bohr    → _dim1_exchange
    + bifurcation + ionic floor
    """
    Zi, Zj = topology.Z_list[i], topology.Z_list[j]
    di, dj = atom_data[Zi], atom_data[Zj]
    per_i, per_j = di['per'], dj['per']
    delta_per = abs(per_i - per_j)

    S = 0.0

    # ── dim 3: overcrowding ──
    S_dim3, sigma_3 = _dim3_overcrowding(i, j, topology, atom_data)
    S += S_dim3

    # ── dim 2+1: vertex polygon DFT (both vertices) ──
    bi_found = bond_idx
    if bi_found < 0:
        for bi_s, (ii_s, jj_s, _) in enumerate(topology.bonds):
            if (ii_s == i and jj_s == j) or (ii_s == j and jj_s == i):
                bi_found = bi_s
                break

    S_dft_A = _vertex_polygon_dft(i, bi_found, topology, atom_data, r_bonds)
    S_dft_B = _vertex_polygon_dft(j, bi_found, topology, atom_data, r_bonds)

    # ── dim 2: LP mutual ──
    S_mutual = _dim2_lp_mutual(i, j, topology, atom_data)

    # ── Degree-aware vertex DFT combination ──────────────────────────
    # At a vertex of degree z, z bonds share the polygon Z/(2P₁)Z.
    # The DFT mean-field (k=0 mode) scales as z/N with z occupants
    # on N = 2P₁ positions. For sparse polygons (z < P₁), this
    # under-estimates the screening compared to the reference z ≈ P₁.
    #
    # Correction: amplify the vertex DFT from low-degree vertices
    # by the face-fill deficiency factor:
    #   f_degree(z) = 1 + D₃ × max(0, P₁ - z) / P₁   for z ≥ 2
    #
    # At z = 2:  f = 1 + D₃ × 1/3 = 1.039  (3.9% more screening)
    # At z = P₁: f = 1               (no correction at half-fill)
    # At z > P₁: f = 1               (no correction)
    # At z = 1:  f = 1               (terminal, not shared)
    #
    # Isolation gate: the correction is strongest when the BOND
    # connects a low-z vertex to a terminal vertex (isolated pair,
    # like in triatomics). When both vertices have z ≥ 2, the SCF
    # naturally handles inter-vertex sharing.
    #
    # The gate: f_gate = max(1/z_A, 1/z_B). High for terminal bonds,
    # low for embedded bonds. Multiplied by P₁ to scale correctly.
    #
    # 0 adjustable parameters. All from s = 1/2.
    z_A = topology.z_count[i]
    z_B = topology.z_count[j]

    # Face-sharing screening: when a vertex has z ≥ 2 bonds sharing
    # the face Z/(2P₁)Z, the DFT mean field (k=0 mode = sum/N) is
    # diluted by the empty positions. This is the STRUCTURAL screening
    # from face sharing that the DFT under-captures.
    #
    # At a vertex with z bonds sharing N = 2P₁ positions:
    # - The occupied fraction is z/N (bonds) + lp/N (lone pairs)
    # - The DFT places peers at z+lp positions out of N
    # - The unoccupied fraction (N - z - lp)/N represents face
    #   "headroom" that dilutes the mean field
    #
    # The face-sharing screening for THIS bond at vertex v:
    #   S_share(v) = S₃ × sin²(π·z_v/(2P₁)) × (z_v - 1) / z_v
    #
    # sin²(π·z/(2P₁)): face holonomy at the fill boundary — the
    #   DFT spectral weight of the occupancy level on Z/(2P₁)Z.
    #   This is LARGE when z fills a resonant fraction of the face
    #   (z=P₁: sin²(π/2)=1) and small at extreme fill levels.
    #
    # (z-1)/z: the fraction of the face occupied by OTHER bonds
    #   (face-sharing fraction). Zero for z=1 (terminal), 1/2 for
    #   z=2, 2/3 for z=3, etc.
    #
    # The total ADDITIVE correction to S₁:
    #   S_degree = (S_share(A) + S_share(B)) / 2
    #
    # Isolation gate: fades for embedded bonds (both z ≥ P₁)
    # because SCF inter-vertex transport handles face sharing.
    #   f_gate = P₁ / (P₁ + z_min - 1)
    #
    # 0 adjustable parameters. All from s = 1/2.
    z_A = topology.z_count[i]
    z_B = topology.z_count[j]

    z_max_bond = max(z_A, z_B)

    # Degree-aware vertex DFT combination:
    # At a z=2 vertex sharing the face Z/(2P₁)Z between 2 bonds,
    # the DFT mean-field is diluted by empty positions (only 2/6
    # positions occupied). This dilution scales as z/(2P₁).
    #
    # The correction amplifies the vertex DFT from z=2 centers by
    # a MULTIPLICATIVE factor that accounts for the diluted DFT.
    # Multiplicative (not additive) ensures the correction inherits
    # the bond-type specificity of the DFT: strong-bond vertices
    # (low screening) get a small absolute correction, while
    # weak-bond vertices (high screening) get a large correction.
    # This naturally discriminates between overbinding and
    # underbinding cases.
    #
    # Amplification factor at vertex v:
    #   f_amp(z_v) = 1 + C₃ × (P₁ - z_v) / z_v   for 2 ≤ z_v < P₁
    # C₃ = cos²(θ₃) = 0.781 is the face complement — the probability
    # that a polygon mode passes through without scattering.
    # (P₁ - z)/z: the ratio of unfilled to filled bond slots.
    #
    # At z=2: f = 1 + 0.781 × 1/2 = 1.390  (39% amplification)
    # At z=3: f = 1.0  (no correction at P₁ half-fill)
    # Terminal (z=1): not amplified (face not shared)
    #
    # Gate: only apply for bonds where z_max ≤ 2 (both vertices
    # in the low-z regime). For z_max > 2, the SCF handles sharing.
    #
    # 0 adjustable parameters. All from s = 1/2.
    if z_max_bond == 2:
        for z_v_sh, S_dft_sh in ((z_A, 'A'), (z_B, 'B')):
            if z_v_sh >= 2 and z_v_sh < P1:
                # f_amp: 1 + (2P₁/N) × (P₁-z)/z where 2P₁/N = 1 for hex face
                # At z=2: f = 1 + 1 × 1/2 = 1.5
                # This is the DFT dilution factor: the ratio of mean
                # density on the occupied sector vs the full polygon.
                f_amp = 1.0 + float(P1 - z_v_sh) / z_v_sh
                if S_dft_sh == 'A':
                    S_dft_A *= f_amp
                else:
                    S_dft_B *= f_amp

    # ── C17: sp-HYBRIDIZATION SCREENING REDUCTION [per ≤ 2, z ≥ P₁] ──
    # For compact centres (per ≤ 2) with z ≥ P₁ bonds and no LP, the
    # D₇ floor in peer values overestimates inter-bond screening.
    # Physical reason: sp-hybridisation creates constructive orbital
    # overlap that reduces inter-bond screening below the heptagonal
    # floor.
    #
    # The excess per vertex = (D₇ − S₃D₃) × n_bohr/z × f_size
    # where n_bohr = number of near-Bohr partners (IE ≥ Ry × C₃).
    # Partners with IE ≈ Ry have near-zero actual peer value, making
    # the D₇ floor excess ≈ D₇ for each such partner.
    # n_bohr/z: fraction of affected bonds at this vertex.
    # f_size fades for n ≥ P₃ (SCF handles cooperation for large mol).
    #
    # 0 adjustable parameters: D₇, S₃, D₃, C₃, P₁, P₃ from s = 1/2.
    _n_at = topology.n_atoms
    _f_size_sp = max(0.0, 1.0 - float(_n_at - P1) / (P3 - P1))
    if _f_size_sp > 0:
        for _v_sp, _z_sp, _lp_sp, _per_sp, _label_sp in [
            (i, z_A, topology.lp[i], di['per'], 'A'),
            (j, z_B, topology.lp[j], dj['per'], 'B'),
        ]:
            if _per_sp <= 2 and _z_sp >= P1:
                # LP gate: LP partially blocks sp-hybridization.
                # lp=0: full correction; lp=P₁: no correction.
                _f_lp_sp = max(0.0, 1.0 - float(_lp_sp) / P1)
                if _f_lp_sp > 0:
                    # Count near-Bohr partners (IE ≥ Ry × C₃)
                    _n_bohr = 0
                    for _bk_sp in topology.vertex_bonds.get(_v_sp, []):
                        _ik, _jk, _ = topology.bonds[_bk_sp]
                        _pk = _jk if _ik == _v_sp else _ik
                        _ie_pk = atom_data[topology.Z_list[_pk]]['IE']
                        if _ie_pk >= RY * C3:
                            _n_bohr += 1
                    if _n_bohr > 0:
                        _S_excess = ((D7 - S3 * D3)
                                     * float(_n_bohr) / _z_sp
                                     * _f_lp_sp * _f_size_sp)
                        if _label_sp == 'A':
                            S_dft_A = max(0.0, S_dft_A - _S_excess)
                        else:
                            S_dft_B = max(0.0, S_dft_B - _S_excess)

    S += (S_dft_A + S_dft_B) / 2.0 + S_mutual

    # ── dim 1: exchange/Bohr/polarity ──
    S_dim1, Q_eff = _dim1_exchange(i, j, topology, atom_data)
    S += S_dim1

    # ╔══════════════════════════════════════════════════════════════╗
    # ║  BIFURCATION [A8]: face part attenuated by q_therm/q_stat   ║
    # ╚══════════════════════════════════════════════════════════════╝
    # The face (dim 2) and polyhedron (dim 3) parts are geometric →
    # attenuated by the bifurcation factor for cross-period polar bonds.
    f_bif = 1.0 - (1.0 - F_BIFURC) * min(delta_per, P1) / P1 * Q_eff
    # Separate the geometric (face) contributions for bifurcation
    # S is already the sum of all dims. Apply bifurcation to face part.
    # Approximation: the face contribution ≈ dim2 + dim3 part of S.
    # For simplicity, apply f_bif as a global modulation when cross-period + polar.
    if f_bif < 1.0:
        # Only the face-geometric part (dim 2+3) is attenuated.
        # dim 1+0 (edge + vertex) stays at full strength.
        S_dim3 = -math.log(1.0 - sigma_3) if 0 < sigma_3 < 1 else max(sigma_3, 0.0)
        S_dft_total = S_dft_A + S_dft_B + S_mutual
        S_dim3 = -math.log(1.0 - sigma_3) if 0 < sigma_3 < 1 else max(sigma_3, 0.0)
        S_geom = max(0.0, S_dft_total + S_dim3)
        S_edge_vert = S - S_geom
        S = S_edge_vert + S_geom * f_bif

    # ── C20: FISHER COMPOSITE SCREENING REDUCTION [P₁⊗P₃] ──────────
    # From PT_PHYSICS ch.15: the Fisher metric on T³ couples the P₁
    # and P₃ faces via cos₃₇ < 0 (anti-correlated). In quark masses,
    # S_M = S₃ + S₇ + 2√(S₃S₇)|cos₃₇| (constructive interference).
    #
    # Molecular analogue: at compact sp-hybridised vertices (per ≤ 2)
    # with multiple LP-free bonds (n_coh ≥ 2), the P₁-P₃ anti-
    # correlation REDUCES the effective P₁ screening by CROSS₃₇.
    # The Fisher coherence factor n_coh(n_coh−1)/z² counts LP-free
    # bond PAIRS that interfere constructively (variance on Z/(2P₁)Z).
    #
    # Gates:
    #   1. per_v ≤ 2 (compact orbitals)
    #   2. lp_v == 0 (P₃ face unblocked)
    #   3. n_coh ≥ 2 (at least 2 coherent LP-free partners)
    #   4. f_size fades for n > 2P₁, zero at n ≥ 2P₁ + (P₃−P₁)
    #
    # 0 adjustable parameters: CROSS₃₇, P₁, P₃ from s = 1/2.
    _n_at_c20 = topology.n_atoms
    _f_size_c20 = max(0.0, 1.0 - float(_n_at_c20 - P1) / (P3 - P1))
    if _f_size_c20 > 0:
        _dS_c20 = 0.0
        for _v_c20, _per_c20, _z_c20, _lp_c20 in [
            (i, di['per'], z_A, topology.lp[i]),
            (j, dj['per'], z_B, topology.lp[j]),
        ]:
            if _per_c20 <= 2 and _z_c20 >= 2 and _lp_c20 == 0:
                # Count LP-free partners at this vertex
                _n_coh = 0
                for _bk_c20 in topology.vertex_bonds.get(_v_c20, []):
                    _ik, _jk, _ = topology.bonds[_bk_c20]
                    _pk = _jk if _ik == _v_c20 else _ik
                    if topology.lp[_pk] == 0:
                        _n_coh += 1
                if _n_coh >= 2:
                    _f_fisher = float(_n_coh * (_n_coh - 1)) / (_z_c20 * _z_c20)
                    _dS_c20 += CROSS_37 * _f_fisher * _f_size_c20
        if _dS_c20 > 0:
            S = max(0.0, S - _dS_c20 / 2.0)

    S = max(0.0, S)

    # ── Ionic same-period floor [D29] ──
    f_cov = min(1.0, min(di['IE'], dj['IE']) / RY)
    IE_acc = max(di['IE'], dj['IE'])
    if f_cov < C3 and delta_per == 0 and IE_acc < RY:
        S = max(S, (1.0 - f_cov) * S3 * P1)

    return S


# ── NEW: P₀ screening (parity gate + LP mutual on Z/2Z) ───────────

def _screening_P0(i: int, j: int, topology: Topology, atom_data: dict) -> float:
    """Screening from the binary face Z/2Z (parity, LP, Pauli).

    P₀ is the simplest face: only 2 positions (even/odd).
    Encodes LP mutual blocking and parity gate.
    """
    Zi, Zj = topology.Z_list[i], topology.Z_list[j]
    di, dj = atom_data[Zi], atom_data[Zj]
    lp_i, lp_j = topology.lp[i], topology.lp[j]
    per_i, per_j = di['per'], dj['per']

    S0 = 0.0

    # LP mutual blocking on Z/2Z
    lp_mutual = min(lp_i, lp_j)
    if lp_mutual > 0:
        per_max = max(per_i, per_j)
        S0 += S_HALF * float(lp_mutual) / max(2.0 * per_max, 1.0)

    # Parity gate: mixed parity → more screening
    same_parity = (Zi % 2) == (Zj % 2)
    if not same_parity:
        S0 += D3 * S_HALF

    return S0


# ╔════════════════════════════════════════════════════════════════════╗
# ║  CAPS                                                              ║
# ╚════════════════════════════════════════════════════════════════════╝

def _rabi_cos2(IE_A: float, IE_B: float, lp_donor: int) -> float:
    """Rabi cos²: fraction of σ-coupling transmitted to d-channel.

    cos² = Δε² / (Δε² + 4t²)  [2-level Rabi on edge of T³]
    Modulated by LP donor availability: cos² × min(lp/P₁, 1).

    High cos² (asymmetric, e.g. Si-F): d-channel opens.
    Low cos² (symmetric, e.g. C-C): σ-channel resolves everything.
    """
    delta = abs(IE_A - IE_B)
    t = S3 * C3 * math.sqrt(max(IE_A * IE_B, 0.01))
    cos2 = delta ** 2 / (delta ** 2 + 4.0 * t ** 2 + 1e-20)
    return cos2 * min(float(lp_donor) / P1, 1.0)


def _cap_P2(i: int, j: int, bo: float,
            topology: Topology, atom_data: dict) -> Tuple[float, float]:
    """Cap and screening for P₂ face.

    Returns (cap_eV, S_P2).
    Three sources:
      1. LP→vacancy (inter-atomic, Rabi-gated for heteronuclear)
      2. Self LP→d (intra-atomic, z≥2 or bo≥2)
      3. π bonds (bo > 1) — NOT Rabi-gated
    Homonuclear: inter-atomic LP→vacancy = 0 (no net transfer).
    """
    Zi, Zj = topology.Z_list[i], topology.Z_list[j]
    di, dj = atom_data[Zi], atom_data[Zj]

    def _d_vac(d):
        if d['per'] < P1 or d['l'] < 1:
            return 0.0
        nf = d['nf']
        return max(0.0, (2.0 * P2 - nf)) / (2.0 * P2)

    vac_i, vac_j = _d_vac(di), _d_vac(dj)
    lp_i = topology.lp[i]
    lp_j = topology.lp[j]

    cap = 0.0

    # ── 1. Inter-atomic LP→vacancy (Rabi-gated, z-shared) ──
    if Zi != Zj:
        # d-host = atom with vacancy. LP donor = partner.
        # Hypervalent d-host (z > P₁): cap shared among excess bonds.
        z_i_v = topology.z_count[i]
        z_j_v = topology.z_count[j]
        z_share_j = max(1, z_j_v - P1 + 1) if z_j_v > P1 else 1
        z_share_i = max(1, z_i_v - P1 + 1) if z_i_v > P1 else 1
        cap_ij = (RY / P2 * vac_j * min(float(lp_i) / P1, 1.0)
                  / z_share_j) if (vac_j > 0 and lp_i > 0) else 0.0
        cap_ji = (RY / P2 * vac_i * min(float(lp_j) / P1, 1.0)
                  / z_share_i) if (vac_i > 0 and lp_j > 0) else 0.0
        # Rabi gate: only the cos² fraction flows to P₂
        cos2_ij = _rabi_cos2(di['IE'], dj['IE'], lp_i) if cap_ij > 0 else 0.0
        cos2_ji = _rabi_cos2(di['IE'], dj['IE'], lp_j) if cap_ji > 0 else 0.0
        cap += cap_ij * cos2_ij + cap_ji * cos2_ji
    # Homonuclear: LP→vacancy cancels (no net transfer) [T1]

    # ── 2. Self LP→d (intra-atomic) ──
    for idx, lp, Z, d in [(i, lp_i, Zi, di), (j, lp_j, Zj, dj)]:
        z_sd = topology.z_count[idx]
        if d['per'] >= P1 and lp >= 2 and d['l'] >= 1 and (z_sd >= 2 or bo >= 2):
            nf_d = d['nf'] if d['l'] == 2 else 0
            vac_d = max(0.0, (2.0 * P2 - nf_d)) / (2.0 * P2)
            if vac_d > 0:
                f_self = float(lp) / (P1 + float(lp))
                per_d = max(1, d['per'] - 2)
                f_orb = math.sin(math.pi * per_d / P2) ** 2
                cap += RY / P2 * vac_d * f_self * f_orb / max(z_sd, 1)

    # ── 3. Through-bond LP→vacancy (Dewar-Chatt-Duncanson) ──────────
    # PT: when a d-block metal M (with d-vacancy) bonds to a bridge
    # atom B (lp=0, z=2) which is bonded to a terminal T with LP > 0,
    # the terminal LP couples to M's d-vacancy through B on Z/(2P₂)Z.
    # This captures the synergistic σ-donation + π-back-donation in
    # metal carbonyls M(CO)n (Dewar-Chatt-Duncanson model).
    #
    # The bridge passage attenuates the coupling by the Bohr vertex
    # amplitude sin²(π·IE_B/(2Ry)) of the bridge atom — the spectral
    # weight of B on Z/(2P₁)Z that mediates the through-bond transfer.
    # No z-sharing on M: each M-B-T chain activates its own d-orbital
    # independently (orthogonal d-channels for different ligands).
    #
    # Gate: M is d-block with d-vacancy, B has lp=0 and z=2 (bridge),
    # T is terminal (z=1) with LP > 0.
    # 0 adjustable parameters: Ry/P₂, S_HALF, sin² all from s = 1/2.
    if Zi != Zj:
        for idx_m, idx_b, d_m, d_b, vac_m in [
            (i, j, di, dj, vac_i), (j, i, dj, di, vac_j)
        ]:
            if vac_m <= 0 or not d_m.get('is_d_block', False):
                continue
            # Bridge atom: lp=0, z=2, not d-block
            lp_b = topology.lp[idx_b]
            z_b = topology.z_count[idx_b]
            if lp_b > 0 or z_b != 2 or d_b.get('is_d_block', False):
                continue
            # Find terminal neighbor(s) of the bridge atom with LP
            for bk in topology.vertex_bonds.get(idx_b, []):
                ik, jk, bo_k = topology.bonds[bk]
                t_idx = jk if ik == idx_b else ik
                if t_idx == idx_m:
                    continue  # skip the M-B bond itself
                z_t = topology.z_count[t_idx]
                if z_t > 1:
                    continue  # terminal only
                lp_t = topology.lp[t_idx]
                if lp_t <= 0:
                    continue
                # Bridge Bohr amplitude: sin²(π·IE_B/(2Ry))
                ie_b = d_b['IE']
                f_bridge = math.sin(math.pi * min(ie_b, RY) / (2.0 * RY)) ** 2
                # Through-bond cap: (Ry/P₂) × vac × (lp_T/P₁) × f_bridge
                f_lp_t = min(float(lp_t) / P1, 1.0)
                cap += (RY / P2) * vac_m * f_lp_t * f_bridge

    # Screening: steric on pentagonal face
    S_P2 = 0.0
    if cap > 0:
        z_i = topology.z_count[i]
        z_j = topology.z_count[j]
        S_P2 = D5 * (float(min(z_i, z_j)) / P2) ** 2

    return cap, S_P2


def _cap_P3(i: int, j: int, bo: float,
            topology: Topology, atom_data: dict) -> Tuple[float, float, float]:
    """Cap, screening, and Q_eff for P₃ face.

    Returns (cap_eV, S_P3, Q_eff).
    Active for heteronuclear bonds.
    """
    Zi, Zj = topology.Z_list[i], topology.Z_list[j]
    if Zi == Zj:
        return 0.0, 0.0, 0.0

    di, dj = atom_data[Zi], atom_data[Zj]

    chi_i = _chi(di)
    chi_j = _chi(dj)
    Q_eff = abs(chi_i - chi_j) / max(0.1, chi_i + chi_j)

    IE_don = min(di['IE'], dj['IE'])
    IE_acc = max(di['IE'], dj['IE'])
    EA_acc = max(di['EA'], dj['EA'])
    f_cov = min(1.0, IE_don / RY)

    # Bohr vertex amplitude [sin(π·IE/(2Ry)) product]
    f_bohr = (math.sin(math.pi * min(IE_don, RY) / (2.0 * RY))
              * math.sin(math.pi * min(IE_acc, RY) / (2.0 * RY)))
    f_bohr += (1.0 - f_bohr) * Q_eff * D3

    # Two mechanisms: polarity cap vs Born-Haber cap
    per_sum = di['per'] + dj['per']
    cap_pol = Q_eff ** 2 * (4.0 * RY / per_sum) * S3 * C3

    cap_BH = 0.0
    if EA_acc > RY * S3:
        per_avg = per_sum / 2.0
        E_mad = 2.0 * RY * C3 / per_avg
        E_BH = E_mad - IE_don + EA_acc
        if E_BH > 0:
            f_BH = min(1.0, E_BH / (RY * S3))
            # CRT face expansion for cross-period ionic bonds [D29]
            delta_per = abs(di['per'] - dj['per'])
            f_ionic = (1.0 - f_cov) * Q_eff ** 2
            f_exp = f_ionic * delta_per / P1 if (delta_per > 0 and f_ionic > D7) else 0.0
            Ry_P3 = RY / P3 + f_exp * (RY / P1 - RY / P3)
            cap_BH = f_BH * Ry_P3 * f_bohr

    cap = min(max(cap_pol, cap_BH), RY / P3 * f_bohr)

    # Screening: full transparency for Q=1
    S_P3 = (1.0 - Q_eff) / P3

    return cap, S_P3, Q_eff


def _dim2_vacancy_boost(i: int, j: int, bo: float,
                        topology: Topology, atom_data: dict) -> float:
    """Dim 2 energy boost: LP donation into partner's p-vacancy.

    [P₂=5 simplex level — 3-body: LP(A) → bond(A,B) → vacancy(B)]

    Active when:
      - One atom has LP > 0 (donor)
      - Partner has LP == 0 AND l >= 1 AND p_vac > 0 (acceptor with empty p-orbital)

    Returns additional energy (eV) to add to the bond.
    """
    boost = 0.0

    for donor_idx, acc_idx in [(i, j), (j, i)]:
        donor_lp = topology.lp[donor_idx]
        acc_lp = topology.lp[acc_idx]
        donor_d = atom_data[topology.Z_list[donor_idx]]
        acc_d = atom_data[topology.Z_list[acc_idx]]

        if donor_lp > 0 and donor_lp > acc_lp and acc_d['l'] >= 1:
            # p-orbital vacancy on acceptor (bond-corrected)
            nf_acc = acc_d['nf']
            z_acc = topology.z_count[acc_idx]
            # True vacancy: subtract both fill electrons and existing bonds
            p_vac = max(0.0, (2.0 * P1 - nf_acc - z_acc)) / (2.0 * P1)
            if p_vac > 0:
                # LP fraction available for donation
                lp_frac = min(float(donor_lp) / P1, 1.0)
                # Bohr envelope: coupling through Ry reference
                ie_don = donor_d['IE']
                ie_acc = acc_d['IE']
                f_bohr = (math.sin(math.pi * min(ie_don, RY) / (2.0 * RY))
                          * math.sin(math.pi * min(ie_acc, RY) / (2.0 * RY)))
                # Dative bond cap: IE_donor/P₁ × p_vac × s
                # The donor's IE determines the energy (it's the donor's LP).
                # s = 1/2: dative is a half-bond (unidirectional transfer).
                # f_bohr gates by Ry proximity (optimal coupling).
                cap = (ie_don / P1) * p_vac * S_HALF * f_bohr
                boost += cap

    return boost


# ── NEW: P₀ and P₁ caps ─────────────────────────────────────────────

def _cap_P0(i: int, j: int, topology: Topology, atom_data: dict) -> float:
    """Cap for binary face Z/2Z. Gates LP channel."""
    lp_i, lp_j = topology.lp[i], topology.lp[j]
    if lp_i == 0 and lp_j == 0:
        return 0.0
    Zi, Zj = topology.Z_list[i], topology.Z_list[j]
    di, dj = atom_data[Zi], atom_data[Zj]
    per_max = max(di['per'], dj['per'])
    lp_total = float(lp_i + lp_j)
    f_lp = min(1.0, lp_total / (2.0 * P1))
    ie_min = min(di['IE'], dj['IE'])
    f_bohr = math.sin(math.pi * min(ie_min, RY) / (2.0 * RY)) ** 2
    return (RY / (2.0 * per_max)) * f_lp * f_bohr * D3



# ╔════════════════════════════════════════════════════════════════════╗
# ║  REFACTORED DIATOMIC SOLVER                                        ║
# ║  Bit-exact decomposition of _diatomic_bond_Fp + mechanisms         ║
# ╚════════════════════════════════════════════════════════════════════╝

@dataclass
class _DiatCommon:
    """All shared quantities for the diatomic bond calculation."""
    Zi: int
    Zj: int
    di: dict
    dj: dict
    ie_i: float
    ie_j: float
    ea_i: float
    ea_j: float
    per_i: int
    per_j: int
    lp_i: int
    lp_j: int
    lp_mutual: int
    per_max: int
    per_min: int
    l_min: int
    l_max: int
    nf_i: int
    nf_j: int
    eps_A: float
    eps_B: float
    eps_geom: float
    f_per: float
    f_lp: float
    t_sigma: float
    t_pi: float
    q_rel: float
    bo: float
    i: int
    j: int
    topology: object  # Topology
    atom_data: dict
    _sd_hybrid_i: bool
    _sd_hybrid_j: bool
    _NS1_METALS: frozenset
    _IE2_NIST: dict


def _diat_common(i: int, j: int, bo: float,
                 topology: Topology, atom_data: dict) -> _DiatCommon:
    """Build shared state for diatomic bond calculation.

    Extracts lines 1519-1741 of _diatomic_bond_Fp exactly.
    """
    Zi, Zj = topology.Z_list[i], topology.Z_list[j]
    di, dj = atom_data[Zi], atom_data[Zj]
    ie_i, ie_j = di['IE'], dj['IE']
    ea_i, ea_j = di['EA'], dj['EA']
    per_i, per_j = di['per'], dj['per']
    lp_i = topology.lp[i]
    lp_j = topology.lp[j]
    lp_mutual = min(lp_i, lp_j)
    per_max = max(per_i, per_j)
    l_min = min(di.get('l', 0), dj.get('l', 0))
    l_max = max(di.get('l', 0), dj.get('l', 0))

    _IE2_NIST = {
        21:12.80, 22:13.58, 23:14.65, 24:16.49, 25:15.64,
        26:16.19, 27:17.08, 28:18.17, 29:20.29, 30:17.96,
        39:12.24, 40:13.13, 41:14.32, 42:16.16, 43:15.26,
        44:16.76, 45:18.08, 46:19.43, 47:21.49, 48:16.91,
        72:14.92, 73:16.2, 74:17.7, 75:16.6,
        76:17.0, 77:17.0, 78:18.56, 79:20.5, 80:18.76,
    }

    eps_A = ie_i / RY
    eps_B = ie_j / RY

    nf_i = di.get('nf', 0) if di.get('is_d_block', False) else 0
    nf_j = dj.get('nf', 0) if dj.get('is_d_block', False) else 0

    _NS1_METALS = frozenset({24, 29, 41, 42, 44, 45, 46, 47, 78, 79})

    # sd hybrid rotation
    # v65: LP blocking is PERIOD-DEPENDENT.  Compact LP orbitals (per ≤ 2:
    # O, N, F) fully block sd-hybridization.  Diffuse LP orbitals (per ≥ 3:
    # Cl, Br, S) do NOT block — their larger radius reduces spectral overlap
    # with the metal's sd-mixing zone on Z/(2P₁)Z.
    # PT: blocking probability ∝ overlap on hexagonal face; decays with
    # orbital size.  Threshold at per = P₁ − 1 = 2 (boundary s/p→d).
    # 0 adjustable parameters.
    _sd_hybrid_i = False
    _sd_hybrid_j = False
    _lp_blocks_hybrid_j = lp_j > 0 and not dj.get('is_d_block', False) and per_j <= 2
    _lp_blocks_hybrid_i = lp_i > 0 and not di.get('is_d_block', False) and per_i <= 2
    if di.get('is_d_block', False) and not _lp_blocks_hybrid_j and P2 < nf_i < 2 * P2:
        ie2_i = _IE2_NIST.get(Zi, ie_i * 2)
        if ie2_i > 0:
            _eps_hybrid_i = math.sqrt(ie_i * ie2_i) / RY
            _f_sd_i = float(nf_i - P2) / float(P2)
            eps_A = eps_A + _f_sd_i * (_eps_hybrid_i - eps_A)
            _sd_hybrid_i = True
    if dj.get('is_d_block', False) and not _lp_blocks_hybrid_i and P2 < nf_j < 2 * P2:
        ie2_j = _IE2_NIST.get(Zj, ie_j * 2)
        if ie2_j > 0:
            _eps_hybrid_j = math.sqrt(ie_j * ie2_j) / RY
            _f_sd_j = float(nf_j - P2) / float(P2)
            eps_B = eps_B + _f_sd_j * (_eps_hybrid_j - eps_B)
            _sd_hybrid_j = True

    # d-block CRT screening
    if nf_i > 0 and di.get('is_d_block', False) and not _sd_hybrid_i:
        _lp_partner_i = lp_j
        if _lp_partner_i > 0:
            eps_A *= C5 ** (float(nf_i) / P2)
        else:
            _ea_partner_i = dj['EA']
            _f_comp_i = max(_ea_partner_i / RY, 0.01)
            eps_A *= C5 ** (float(nf_i) / P2 * _f_comp_i)
    if nf_j > 0 and dj.get('is_d_block', False) and not _sd_hybrid_j:
        _lp_partner_j = lp_i
        if _lp_partner_j > 0:
            eps_B *= C5 ** (float(nf_j) / P2)
        else:
            _ea_partner_j = di['EA']
            _f_comp_j = max(_ea_partner_j / RY, 0.01)
            eps_B *= C5 ** (float(nf_j) / P2 * _f_comp_j)

    # d-block sigma stiffness [v66: exchange gain modulation]
    # PT: σ-stiffness (IE₁/IE₂)^s captures the d→s promotion cost.
    # But promotion d^n s² → d^(n+1) s¹ GAINS ΔK = n exchange pairs
    # (the new d-electron aligns with existing n-1 unpaired spins).
    # This gain partially offsets the promotion cost:
    #   f_stiff_corr = f_stiff × (1 + ΔK × J / IE₁ × f_lp)
    # where J = S₅ × Ry/(2P₂) is the exchange integral on face P₂,
    # and f_lp = 1 if partner LP=0, C₃ if partner has LP (LP partially
    # blocks the exchange gain through face P₁ screening).
    # At half-fill (nf=P₂): ΔK = 0 (promotion d5→d6 gains NO exchange
    # because the 6th electron goes spin-down). So MnH is UNCHANGED.
    # 0 adjustable parameters.
    _J_exch_stiff = S5 * RY / (2.0 * P2)
    for Z_stiff, d_stiff, is_A in [(Zi, di, True), (Zj, dj, False)]:
        if not d_stiff.get('is_d_block', False):
            continue
        if (is_A and _sd_hybrid_i) or (not is_A and _sd_hybrid_j):
            continue
        ie2 = _IE2_NIST.get(Z_stiff, 0.0)
        ie1 = d_stiff['IE']
        if ie2 <= 0 or ie1 <= 0:
            continue
        is_ns1 = Z_stiff in _NS1_METALS
        nf_stiff = d_stiff.get('nf', 0)
        if not is_ns1 and nf_stiff < 2 * P2:
            f_stiff = (ie1 / ie2) ** S_HALF
            # Exchange gain: ΔK pairs gained by promotion d^nf → d^(nf+1)
            if nf_stiff < P2:
                _delta_K = nf_stiff  # nf new parallel pairs
            else:
                _delta_K = 0  # spin-down: no new parallel pairs
            if _delta_K > 0:
                # LP-dependent gate: LP partner reduces exchange gain
                _pidx_stiff = (1 - int(is_A))  # partner index
                # Determine partner LP contribution
                _lp_part_stiff = lp_j if is_A else lp_i
                _f_lp_exch = C3 if _lp_part_stiff > 0 else 1.0
                f_stiff *= (1.0 + float(_delta_K) * _J_exch_stiff
                            / ie1 * _f_lp_exch)
            if is_A:
                eps_A *= f_stiff
            else:
                eps_B *= f_stiff

    eps_geom = math.sqrt(eps_A * eps_B)

    # period attenuation
    if Zi == Zj and l_max >= 2:
        f_per = 1.0
    elif per_max > 2:
        expo = 1.0 / (P1 * P1) if per_max == P1 else 1.0 / P1
        f_per = (2.0 / per_max) ** expo
    else:
        f_per = 1.0

    # v65: 5d Dirac contraction [relativistic orbital shrinkage]
    # PT: for period ≥ 6 d-metals, the Dirac equation contracts the
    # ns-orbital by factor γ = √(1 − (Zα)²). This REDUCES the effective
    # orbital radius, increasing overlap and partially compensating the
    # period attenuation.  Applies to heteronuclear only (homonuclear
    # already has f_per = 1).  Gate: per ≥ 6 AND d-block.
    # 0 adjustable parameters.
    if f_per < 1.0 and l_max >= 2 and per_max >= 6:
        _Z_heavy = Zi if per_i >= 6 and l_of(Zi) >= 2 else (
            Zj if per_j >= 6 and l_of(Zj) >= 2 else 0)
        if _Z_heavy > 0:
            _za2 = (_Z_heavy * AEM) ** 2
            if _za2 < 1.0:
                _gamma_rel = math.sqrt(1.0 - _za2)
                f_per /= _gamma_rel

    # LP blocking
    _per_min_lp = min(per_i, per_j)
    _orbital_space = 2.0 * max(per_max, 1)
    if _per_min_lp > P1:
        _orbital_space = min(_orbital_space, 2.0 * P1)
    f_lp = max(0.01, 1.0 - lp_mutual / _orbital_space)

    # transfer integral
    t_sigma = S_HALF * eps_geom * f_lp * f_per

    # d-block radial mismatch [v65: sd-hybrid relief]
    # PT: mismatch between d-metal s-orbital (Z/2Z) and H 1s-orbital.
    # When sd-hybrid is active, the bonding orbital has d-character that
    # contracts the orbital (face P₂ → P₁ coupling), reducing the mismatch.
    # Relief: f_sd² weights the d-component (0 for no hybrid, 0.64 for d9).
    # 0 adjustable parameters.
    if l_max >= 2:
        per_min_rm = min(per_i, per_j)
        lp_total = lp_i + lp_j
        if per_min_rm <= 1 and lp_total == 0:
            if _sd_hybrid_i or _sd_hybrid_j:
                _f_sd_rm = max(
                    (float(nf_i - P2) / float(P2)) if _sd_hybrid_i else 0.0,
                    (float(nf_j - P2) / float(P2)) if _sd_hybrid_j else 0.0,
                )
                t_sigma *= (1.0 - S5 * (1.0 - _f_sd_rm * _f_sd_rm))
            else:
                t_sigma *= (1.0 - S5)

    t_pi = S_HALF * eps_geom * f_per * S_HALF

    per_min = min(per_i, per_j)
    q_rel = min(eps_A, eps_B) / max(eps_A, eps_B)

    return _DiatCommon(
        Zi=Zi, Zj=Zj, di=di, dj=dj,
        ie_i=ie_i, ie_j=ie_j, ea_i=ea_i, ea_j=ea_j,
        per_i=per_i, per_j=per_j,
        lp_i=lp_i, lp_j=lp_j, lp_mutual=lp_mutual,
        per_max=per_max, per_min=per_min,
        l_min=l_min, l_max=l_max,
        nf_i=nf_i, nf_j=nf_j,
        eps_A=eps_A, eps_B=eps_B, eps_geom=eps_geom,
        f_per=f_per, f_lp=f_lp,
        t_sigma=t_sigma, t_pi=t_pi,
        q_rel=q_rel,
        bo=bo, i=i, j=j,
        topology=topology, atom_data=atom_data,
        _sd_hybrid_i=_sd_hybrid_i, _sd_hybrid_j=_sd_hybrid_j,
        _NS1_METALS=_NS1_METALS, _IE2_NIST=_IE2_NIST,
    )


def _diat_face_P1(c: _DiatCommon) -> float:
    """Compute initial D_cov (sigma + pi on hexagonal face P1).

    Returns D_cov = (gap_sigma + gap_pi) * cap_sigma * f_exch.
    This is the INITIAL D_cov before d-block additions and modifiers.
    """
    gap_sigma = 2.0 * c.t_sigma

    # pi spectral decomposition
    gap_pi = 0.0
    if c.l_min >= 1:
        _bo_pi_smi = min(c.bo - 1.0, 2.0)
        if _bo_pi_smi > 0:
            _val_pi_A = valence_electrons(c.Zi)
            _val_pi_B = valence_electrons(c.Zj)
            _ns_pi_A = min(2, _val_pi_A)
            _ns_pi_B = min(2, _val_pi_B)
            _np_pi_A = max(0, _val_pi_A - _ns_pi_A) if l_of(c.Zi) >= 1 else 0
            _np_pi_B = max(0, _val_pi_B - _ns_pi_B) if l_of(c.Zj) >= 1 else 0
            _unp_A = min(_np_pi_A, 2 * P1 - _np_pi_A)
            _unp_B = min(_np_pi_B, 2 * P1 - _np_pi_B)
            _lp_A = max(0, _np_pi_A - P1)
            _lp_B = max(0, _np_pi_B - P1)
            _n_shared = max(0, min(_unp_A - 1, _unp_B - 1))
            _n_full_dat = 0
            if _np_pi_A < P1:
                _n_full_dat += min(_lp_B, P1 - _np_pi_A)
            if _np_pi_B < P1:
                _n_full_dat += min(_lp_A, P1 - _np_pi_B)
            _n_full_dat = min(_n_full_dat,
                              max(0, int(_bo_pi_smi) - _n_shared))
            _n_half_dat = max(0, int(_bo_pi_smi) - _n_shared - _n_full_dat)
            if _n_half_dat > 0:
                if _np_pi_A % P1 == 0 or _np_pi_B % P1 == 0:
                    _n_half_dat = 0
            if c.Zi == c.Zj:
                _n_pi_eff = float(_n_shared + _n_full_dat + _n_half_dat)
            else:
                _n_pi_eff = float(_n_shared + _n_full_dat) + float(_n_half_dat) * S_HALF
            gap_pi = _n_pi_eff * 2.0 * c.t_pi
            # Heavy-period π attenuation [diffuse lateral overlap]
            # π modes (k≥1 on Z/(2P₁)Z) require lateral coupling that
            # degrades for diffuse 3p/4p/5p orbitals.  Factor (2/per)^s.
            # Gate: bo ≥ 3, per_max ≥ P₁, neither atom at half-fill
            # (Hund already reduces π at half-fill), both np ≥ 2
            # (np=1 atoms use dative π which has different overlap).
            # 0 adjustable parameters.
            if c.bo >= 3.0 and max(c.per_i, c.per_j) >= P1:
                if (_np_pi_A != P1 and _np_pi_B != P1
                        and _np_pi_A >= 2 and _np_pi_B >= 2):
                    gap_pi *= (2.0 / float(max(c.per_i, c.per_j))) ** S_HALF

    # exchange boost
    f_exch = 1.0
    if c.l_max >= 1:
        f_exch += S3 * S_HALF * c.q_rel
        if c.Zi == c.Zj and c.lp_mutual > 0:
            if c.lp_mutual < P1:
                f_exch += S3 * S_HALF * c.lp_mutual / P1
            else:
                f_exch -= S3 * S_HALF * (c.lp_mutual - P1 + 1) / P1

    # cap
    cap_sigma = RY / P1
    if c.l_max >= 2 and min(c.per_i, c.per_j) <= 1:
        cap_sigma = RY / math.sqrt(P1 * P2)

    D_cov = (gap_sigma + gap_pi) * cap_sigma * f_exch
    return D_cov


@dataclass
class _DiatP2Result:
    """Results from the pentagonal face (d-block channels)."""
    D_cross: float
    D_vac: float
    D_sd: float
    D_sd_promo: float
    D_P2: float
    D_d_pi: float
    D_exch_Hund: float
    D_repulsion: float
    D_hund_loss: float


def _diat_face_P2(c: _DiatCommon) -> _DiatP2Result:
    """Compute all d-block + dative quantities on the pentagonal face.

    Returns D_cross, D_vac, D_sd, D_sd_promo, D_P2, D_d_pi,
    D_repulsion, D_hund_loss.
    """
    # ── p-VACANCY BACK-DONATION ──
    D_cross = 0.0
    if c.Zi != c.Zj and c.l_min >= 1:
        _val_A = valence_electrons(c.Zi)
        _val_B = valence_electrons(c.Zj)
        _ns_A = min(2, _val_A)
        _ns_B = min(2, _val_B)
        _np_A = max(0, _val_A - _ns_A) if l_of(c.Zi) >= 1 else 0
        _np_B = max(0, _val_B - _ns_B) if l_of(c.Zj) >= 1 else 0
        for _np_acc, _lp_don_idx, _np_don, _nf_acc, _Z_acc in [
            (_np_A, c.j, _np_B, c.nf_i, c.Zi),
            (_np_B, c.i, _np_A, c.nf_j, c.Zj),
        ]:
            _raw_vac = max(0.0, float(2 * P1 - _np_acc)) / (2.0 * P1)
            _shell_dev = max(0.0, float(P1 - _np_acc) / P1)
            _eff_vac = _raw_vac * _shell_dev
            if _eff_vac < 0.01:
                continue
            _lp_don = c.topology.lp[_lp_don_idx]
            if _lp_don <= 0:
                continue
            _f_don = min(float(_lp_don) / P1, 1.0)
            _f_pol = 1.0 - c.q_rel * c.q_rel
            _f_d_pass = 1.0
            if _nf_acc > 0:
                _nf_screen = _nf_acc + (1 if _Z_acc in c._NS1_METALS else 0)
                _nf_screen = min(_nf_screen, 2 * P2)
                _excess = max(0.0, float(_nf_screen - P2)) / float(P2)
                _f_d_pass = math.cos(math.pi * _excess / 2.0) ** 2
            D_cross += (RY / P1) * _eff_vac * _f_don * c.eps_geom * c.f_per * _f_pol * _f_d_pass

        # ── v68: EXCESS DATIVE π VACANCY (p-block np < P₁ deficient) ──
        # When a p-block atom has np < P₁ and bo > np + 1, it has more
        # bonds than p-electrons.  The first dative π is in D_cross;
        # each excess dative π gets a vacancy channel on Z/(2P₁)Z.
        # Energy per vacancy: (Ry/P₁) × s × eps_geom (Bohr cap).
        # Gate: l=1 (p-block), per_max ≤ 2 (same-shell 2p overlap),
        # both atoms on the same hexagonal face for direct dative π.
        # 0 adjustable parameters.
        if c.per_max <= 2:
            for _np_ev, _lp_ev_idx, _Z_ev in [
                (_np_A, c.j, c.Zi), (_np_B, c.i, c.Zj),
            ]:
                if l_of(_Z_ev) != 1:
                    continue
                if _np_ev >= P1:
                    continue
                _lp_ev = c.topology.lp[_lp_ev_idx]
                if _lp_ev <= 0:
                    continue
                _n_excess_dat = max(0, int(round(c.bo)) - _np_ev - 1)
                if _n_excess_dat <= 0:
                    continue
                _n_vac_pi = min(_lp_ev, _n_excess_dat)
                D_cross += float(_n_vac_pi) * (RY / P1) * S_HALF * c.eps_geom

    # ── VACANCY (dative) ──
    D_vac = _dim2_vacancy_boost(c.i, c.j, c.bo, c.topology, c.atom_data)

    # ── ANTI-DOUBLE-COUNTING ──
    if D_vac > 0 and c.l_min >= 1:
        _d_don = c.dj if c.topology.lp[c.j] > c.topology.lp[c.i] else c.di
        _d_acc = c.di if c.topology.lp[c.j] > c.topology.lp[c.i] else c.dj
        _per_don = _d_don['per']
        _per_acc = _d_acc['per']
        _nf_acc_v = _d_acc['nf']
        if _per_don > _per_acc:
            D_vac = 0.0
        elif _per_don == _per_acc and _per_acc <= 2 and _nf_acc_v > P1:
            D_vac = 0.0
        elif c.l_min >= 1 and c.bo >= 2:
            _Z_acc_v = _d_acc.get('l', 0)
            if _Z_acc_v <= 1:
                _val_acc = valence_electrons(
                    c.Zi if (_d_acc is c.di) else c.Zj)
                _ns_acc = min(2, _val_acc)
                _np_acc_v2 = max(0, _val_acc - _ns_acc)
                if _np_acc_v2 <= 1:
                    D_vac = 0.0

    # ── ALKALINE EARTH sd ──
    _ALKALINE_EARTH_SD = frozenset({20, 38, 56, 88})
    D_sd = 0.0
    if c.l_max == 0 and (c.Zi in _ALKALINE_EARTH_SD or c.Zj in _ALKALINE_EARTH_SD):
        _Z_ae = c.Zi if c.Zi in _ALKALINE_EARTH_SD else c.Zj
        _d_ae = c.atom_data[_Z_ae]
        _Z_part_ae = c.Zj if _Z_ae == c.Zi else c.Zi
        _d_part_ae = c.atom_data[_Z_part_ae]
        _ie_ae = _d_ae['IE']
        _ie_part_ae = _d_part_ae['IE']
        _f_bohr_ae = (math.sin(math.pi * min(_ie_ae, RY) / (2.0 * RY))
                      * math.sin(math.pi * min(_ie_part_ae, RY) / (2.0 * RY)))
        D_sd = (_ie_ae / P2) * S_HALF * _f_bohr_ae

    # ── EARLY sd PROMOTION ──
    D_sd_promo = 0.0
    for _Z_pro, _d_pro, _nf_pro in [(c.Zi, c.di, c.nf_i), (c.Zj, c.dj, c.nf_j)]:
        if not _d_pro.get('is_d_block', False):
            continue
        if _nf_pro == 0 or _nf_pro > P2:
            continue
        _pidx_pro = c.j if _Z_pro == c.Zi else c.i
        if c.topology.lp[_pidx_pro] > 0:
            continue
        _ie2_pro = c._IE2_NIST.get(_Z_pro, 0.0)
        if _ie2_pro <= 0:
            continue
        _ie1_pro = _d_pro['IE']
        _f_fill_pro = float(_nf_pro) / float(P2)
        _D_raw_pro = (_ie2_pro - _ie1_pro) / float(P2) * _f_fill_pro * S_HALF
        _nf_hund_pro = _nf_pro + (1 if _Z_pro in c._NS1_METALS else 0)
        _nf_hund_pro = min(_nf_hund_pro, 2 * P2)
        _n_unp_pro = min(_nf_hund_pro, 2 * P2 - _nf_hund_pro)
        _n_pairs_pro = _n_unp_pro * (_n_unp_pro - 1) // 2
        _f_hund_pro = 1.0 - float(_n_pairs_pro) / float(P2 * (P2 - 1) // 2)
        _D_raw_pro *= _f_hund_pro
        D_sd_promo += min(max(0.0, _D_raw_pro), RY / P2)

    # ── D_P2 BACK-DONATION ──
    D_P2 = 0.0
    for d_m, d_lig, nf_m, Z_m in [(c.di, c.dj, c.nf_i, c.Zi), (c.dj, c.di, c.nf_j, c.Zj)]:
        if not d_m.get('is_d_block', False) or nf_m == 0:
            continue
        nf_eff = nf_m
        if Z_m in c._NS1_METALS and nf_eff >= 2 * P2 - 1:
            continue
        if nf_eff >= 2 * P2:
            continue
        f_occ = float(nf_eff) * (2.0 * P2 - nf_eff) / (P2 * P2)
        _lig_idx_hund = c.j if (d_m is c.di) else c.i
        _lp_partner_real = c.topology.lp[_lig_idx_hund]
        _lig_is_p_block = d_lig.get('l', 0) == 1
        if _lp_partner_real > 0 and _lig_is_p_block:
            _nf_hund = nf_eff + (1 if Z_m in c._NS1_METALS else 0)
            _nf_hund = min(_nf_hund, 2 * P2)
            _n_unp = min(_nf_hund, 2 * P2 - _nf_hund)
            _n_pairs = _n_unp * (_n_unp - 1) // 2
            _max_pairs = P2 * (P2 - 1) // 2
            _f_hund_block = 1.0 - float(_n_pairs) / float(_max_pairs)
            f_occ *= _f_hund_block
        ea_lig = d_lig['EA']
        _lig_idx_lp = c.j if (d_m is c.di) else c.i
        lp_lig_check = c.topology.lp[_lig_idx_lp]
        ie_m = d_m['IE']
        J0_scale = AEM * ie_m / RY
        f_accept = min(1.0, max(float(lp_lig_check) / P1,
                                ea_lig / RY,
                                J0_scale * float(nf_eff)))
        _D_P2_bond = (RY / P2) * f_occ * f_accept * c.f_per
        if nf_m > P2:
            _f_pauli_p2 = C5 ** (float(nf_m - P2) / float(P2))
            _D_P2_bond *= _f_pauli_p2
        D_P2 += _D_P2_bond

    # ── D_d_pi [v66: half-fill suppression] ──
    # PT (DFT): π back-donation requires orbital anisotropy (k=1 mode on
    # Z/P₂Z) to direct LP electron density into a specific d-vacancy.
    # At exact half-fill (nf_eff = P₂), the Dirichlet kernel k=1 mode is
    # EXACTLY zero — the d-shell is a uniform sphere with no directional
    # preference.  The π channel is therefore suppressed.
    # Gate: nf_eff = P₂ (exact half-fill, including NS1 adjustment).
    # 0 adjustable parameters.
    D_d_pi = 0.0
    for nf_dpi, d_dpi, Z_dpi in [(c.nf_i, c.di, c.Zi), (c.nf_j, c.dj, c.Zj)]:
        if not d_dpi.get('is_d_block', False) or nf_dpi >= 2 * P2:
            continue
        if Z_dpi in c._NS1_METALS and nf_dpi >= 2 * P2 - 1:
            continue
        # v66: half-fill π suppression (DFT k=1 = 0 at nf_eff = P₂)
        # Only for non-d-block partners (d→LP back-donation).
        # For d+d bonds the π channel is intra-d, not directional → keep.
        _nf_eff_dpi = nf_dpi + (1 if Z_dpi in c._NS1_METALS else 0)
        _nf_eff_dpi = min(_nf_eff_dpi, 2 * P2)
        _partner_dpi_idx = c.j if Z_dpi == c.Zi else c.i
        _partner_dpi_d = c.atom_data[c.topology.Z_list[_partner_dpi_idx]]
        if _nf_eff_dpi == P2 and not _partner_dpi_d.get('is_d_block', False):
            continue  # all k≥1 modes vanish → no directional π to non-d partner
        n_vacancy = (2.0 * P2 - nf_dpi) / P2
        partner_idx = c.j if Z_dpi == c.Zi else c.i
        lp_partner = c.topology.lp[partner_idx]
        if lp_partner == 0:
            continue
        n_d_pi = min(float(lp_partner), n_vacancy)
        f_fill_atten = (1.0 - float(nf_dpi) / (2.0 * P2)) ** 2
        if c.bo >= 2:
            D_d_pi += n_d_pi * (RY / P2) * S_HALF * c.f_per * f_fill_atten
        else:
            D_d_pi += n_d_pi * (RY / P2) * S_HALF * c.f_per * D3 * f_fill_atten

    # ── CASCADE SCREENING / D_repulsion ──
    # NOTE: this also computes D_cov modifiers (_f_gate) that are returned
    # as a multiplier on D_cov. We store them in D_repulsion and a separate
    # "D_cov_d10_factor" but to keep bit-exact, we return the factor
    # separately. Actually, to keep things simple and bit-exact, we compute
    # D_repulsion here and return the d10 gate factor.
    _NS1_METALS_rep = frozenset({24, 29, 41, 42, 44, 45, 46, 47, 78, 79})
    D_repulsion = 0.0
    # We need to track D_cov multiplier from d10 gate. Return as list of (factor,).
    _d10_factors = []
    for Z_d, nf_d, d_d in [(c.Zi, c.nf_i, c.di), (c.Zj, c.nf_j, c.dj)]:
        if not d_d.get('is_d_block', False):
            continue
        is_ns1 = Z_d in _NS1_METALS_rep
        _d10_pidx_c = c.j if Z_d == c.Zi else c.i
        _d10_partner_c = c.dj if Z_d == c.Zi else c.di
        _partner_is_p_lp = (_d10_partner_c.get('l', 0) == 1
                            and c.topology.lp[_d10_pidx_c] > 0)
        _is_d10_effective = (is_ns1 and nf_d >= 2 * P2 - 1
                             and _partner_is_p_lp)
        if (nf_d >= 2 * P2 and not is_ns1) or _is_d10_effective:
            _d10_partner = c.dj if Z_d == c.Zi else c.di
            _ea_p = _d10_partner['EA']
            _f_field = min(1.0, _ea_p / RY)
            if _is_d10_effective:
                # v65: NS1 d-hole reduces Pauli repulsion
                # PT: nf = 2P₂−1 leaves one vacancy on Z/(2P₂)Z that
                # can accommodate partner LP density.  The Pauli base
                # factor drops from cos²θ₅ to cos^(2s)θ₅ = √(cos²θ₅).
                # True d10 (nf = 2P₂, NS1 check passed above) keeps C5.
                # 0 adjustable parameters.
                if nf_d == 2 * P2 - 1:
                    _f_base = C5 ** S_HALF
                else:
                    _f_base = C5
            else:
                _f_base = C5 * S_HALF
                # v67: LP cross-face relief for true d10 non-NS1.
                # PT: partner LP on Z/(2P₁)Z couples to d10 face Z/(2P₂)Z
                # through cross-face coupling sin²θ₅.  The LP occupies
                # lp/P₁ of the hexagonal face; this fraction × sin²θ₅
                # of the blocked energy is recovered (CRT Principle 3).
                # Gate: lp_partner > 0.  0 adjustable parameters.
                _lp_partner_d10 = c.topology.lp[_d10_pidx_c]
                if _lp_partner_d10 > 0:
                    _f_LP_relief = ((1.0 - _f_base) * S5
                                    * float(_lp_partner_d10) / P1)
                    _f_base += _f_LP_relief
            _f_gate = _f_base + (1.0 - _f_base) * _f_field
            _d10_factors.append(_f_gate)
        elif nf_d > P2:
            n_excess_d = nf_d - P2
            f_ns = 1.0 + S_HALF if not is_ns1 else S_HALF
            D_repulsion += S5 * (float(n_excess_d) / P2) ** 2 * (RY / P2) * f_ns

    # ── HUND EXCHANGE LOSS ──
    D_hund_loss = 0.0
    J_exchange = S5 * RY / (2.0 * P2)
    for Z_h, nf_h, d_h in [(c.Zi, c.nf_i, c.di), (c.Zj, c.nf_j, c.dj)]:
        if not d_h.get('is_d_block', False) or nf_h == 0:
            continue
        if Z_h in c._NS1_METALS and nf_h >= 2 * P2 - 1:
            continue
        if nf_h >= 2 * P2:
            continue
        n_up = min(nf_h, P2)
        delta_K = max(0, n_up - 1)
        D_hund_loss += delta_K * J_exchange

    if D_hund_loss > 0:
        f_field = 0.0
        _partner_has_lp = False
        for idx_f in (c.i, c.j):
            d_f = c.atom_data[c.topology.Z_list[idx_f]]
            if not d_f.get('is_d_block', False):
                ea_f = d_f['EA']
                lp_f = c.topology.lp[idx_f]
                f_field = max(f_field, min(1.0, max(ea_f / RY, lp_f / P1)))
                if lp_f > 0:
                    _partner_has_lp = True
        D_hund_loss *= (1.0 - f_field)

    if D_hund_loss > 0 and (c._sd_hybrid_i or c._sd_hybrid_j):
        D_hund_loss *= S_HALF

    # ── HUND EXCHANGE COUPLING [early d: spin magnetization, nf ≤ P₂] ──
    # Early d-block metals (Sc-Mn) have n_unp aligned spins on Z/(2P₂)Z.
    # Magnetization M = n_unp/(2P₂) couples to the σ bonding electron.
    # At half-fill (nf_eff ≈ P₂), the linear factor n_unp/(2P₂) is replaced
    # by the Mertens product on Z/(2P₂)Z: ∏_{k=1}^{n-1} sin(kπ/n) = n/2^{n-1}
    # (Chebyshev identity — spectral convergence T4).
    # Gate: nf_eff ≤ P₂ only. Partner LP = 0.
    # 0 adjustable parameters.
    def _mertens_factor(n_unp, N=2 * P2):
        """Mertens product on Z/(2P₂)Z: spectral convergence (T4).

        Chebyshev identity: ∏_{k=1}^{n-1} sin(kπ/n) = n/2^{n-1}.
        At n=P₂=5: 5/16 = 0.3125 (vs linear 5/10 = 0.5).
        """
        if n_unp <= 1:
            return float(n_unp) / N  # linear for 0,1
        return float(n_unp) / (2.0 ** (n_unp - 1))

    D_exch_Hund = 0.0
    for _Z_ex, _d_ex, _nf_ex in [(c.Zi, c.di, c.nf_i), (c.Zj, c.dj, c.nf_j)]:
        if not _d_ex.get('is_d_block', False):
            continue
        _nf_eff_ex = _nf_ex + (1 if _Z_ex in c._NS1_METALS else 0)
        _nf_eff_ex = min(_nf_eff_ex, 2 * P2)
        if _nf_eff_ex == 0 or _nf_eff_ex > P2:
            continue
        _pidx_ex = c.j if _Z_ex == c.Zi else c.i
        if c.topology.lp[_pidx_ex] > 0:
            continue
        _n_unp_ex = min(_nf_eff_ex, 2 * P2 - _nf_eff_ex)
        _ie_partner = c.atom_data[c.topology.Z_list[_pidx_ex]]['IE']
        _eps_geom_ex = math.sqrt(_d_ex['IE'] * _ie_partner) / RY
        # Mertens resummation at half-fill (nf_eff >= P₂-1), linear otherwise
        if _nf_eff_ex >= P2 - 1:
            _fill_factor = _mertens_factor(_n_unp_ex)
        else:
            _fill_factor = float(_n_unp_ex) / (2.0 * P2)
        D_exch_Hund += _eps_geom_ex * _fill_factor * (RY / P2) * S_HALF

    # ── v65-DFT: ORBITAL POLARIZATION [late d: spin-down k=1 dipole] ──
    # PT (wave function): for nf > P₂, the spin-down sublattice on Z/P₂Z
    # has a non-zero k=1 Dirichlet mode |ψ̂_dn(1)|² that creates orbital
    # polarization along the bond axis, reinforcing the σ overlap.
    # This is a DIFFERENT mechanism from early-d spin magnetization:
    # early d = SPIN coupling (n_unp, peaks at d5)
    # late d = ORBITAL coupling (sin², peaks at d7-d8, zero at d5/d10)
    # Gate: nf > P₂ (late d), LP_partner = 0, AND D_repulsion exceeds
    # the natural face scale D₅×Ry/P₂ (Pauli threshold — the orbital
    # polarization only manifests when repulsion creates a driving force).
    # 0 adjustable parameters. sin² profile = exact DFT on Z/P₂Z.
    _SIN2_PI_OVER_P2 = math.sin(math.pi / P2) ** 2
    _D_PAULI_THRESH = D5 * RY / P2  # ≈ 0.278 eV
    for _Z_op, _d_op, _nf_op in [(c.Zi, c.di, c.nf_i), (c.Zj, c.dj, c.nf_j)]:
        if not _d_op.get('is_d_block', False):
            continue
        _nf_eff_op = _nf_op + (1 if _Z_op in c._NS1_METALS else 0)
        _nf_eff_op = min(_nf_eff_op, 2 * P2)
        if _nf_eff_op <= P2:
            continue  # early d handled above
        _pidx_op = c.j if _Z_op == c.Zi else c.i
        if c.topology.lp[_pidx_op] > 0:
            continue
        # Pauli threshold gate: orbital polarization only when D_rep is significant
        if D_repulsion < _D_PAULI_THRESH:
            continue
        # Spin-down k=1 mode
        _n_dn_op = _nf_eff_op - P2
        if _n_dn_op <= 0 or _n_dn_op >= P2:
            continue
        _k1_dn = (2.0 * math.sin(math.pi * _n_dn_op / P2) ** 2
                  / (P2 * P2 * _SIN2_PI_OVER_P2))
        _ie_p_op = c.atom_data[c.topology.Z_list[_pidx_op]]['IE']
        _eps_g_op = math.sqrt(_d_op['IE'] * _ie_p_op) / RY
        D_exch_Hund += _eps_g_op * _k1_dn * (RY / P2) * S_HALF

    # ── CFSE: P₃→P₂ crystal field coupling [d-block, ligand avec LP] ──
    # PT: le champ cristallin est la projection de la face heptagonale
    # (Z/14Z) sur le pentagone (Z/10Z) via le CRT.
    # r₇ = nf_eff mod P₃ = position sur le cercle heptagonal.
    # Amplitude = sin²(2π·r₇/P₃) × D₇/P₂ (Principe 3 : Pythagore CRT).
    # Gate: is_d_block AND lp_partner > 0 (champ cristallin = ligand fort).
    # 0 paramètre ajusté.
    D_CFSE = 0.0
    for _Z_cf, _d_cf, _nf_cf in [(c.Zi, c.di, c.nf_i), (c.Zj, c.dj, c.nf_j)]:
        if not _d_cf.get('is_d_block', False):
            continue
        _nf_eff_cf = _nf_cf + (1 if _Z_cf in c._NS1_METALS else 0)
        _nf_eff_cf = min(_nf_eff_cf, 2 * P2)
        if _nf_eff_cf == 0:
            continue
        _pidx_cf = c.j if _Z_cf == c.Zi else c.i
        _lp_partner = c.topology.lp[_pidx_cf]
        if _lp_partner == 0:
            continue  # no crystal field without strong-field ligand
        # Position on heptagonal circle Z/(2P₃)Z
        _r7 = _nf_eff_cf % P3
        _sin2_r7 = math.sin(2.0 * math.pi * _r7 / P3) ** 2
        # CFSE amplitude: D₇/P₂ × f_lp × (Ry/P₂) × S_HALF
        _f_lp = min(float(_lp_partner) / P1, 1.0)
        D_CFSE += _sin2_r7 * D7 / P2 * _f_lp * (RY / P2) * S_HALF
    D_exch_Hund += D_CFSE

    # ── DARK MODES k=2,4 on Z/10Z [d-block spectral structure] ──
    # PT: les modes de Fourier k=2 et k=4 sur Z/(2P₂)Z transportent
    # l'information de cross-gap entre les 3 faces CRT.
    # k=2 = beat P₁⊗P₂ (R35_DARK + R57_DARK)
    # k=4 = mode propre pentagone (R37_DARK)
    # Prototype validé sur IE atomiques (test_dblock_dark.py).
    # 0 paramètre ajusté.
    D_dark = 0.0
    _omega_10 = 2.0 * math.pi / (2 * P2)  # 2π/10
    for _Z_dk, _d_dk, _nf_dk in [(c.Zi, c.di, c.nf_i), (c.Zj, c.dj, c.nf_j)]:
        if not _d_dk.get('is_d_block', False):
            continue
        _nf_eff_dk = _nf_dk + (1 if _Z_dk in c._NS1_METALS else 0)
        _nf_eff_dk = min(_nf_eff_dk, 2 * P2)
        if _nf_eff_dk == 0:
            continue
        _per_dk = _d_dk['per']
        _tier_d = _per_dk - P2
        if _tier_d < 0:
            continue  # before bifurcation (per < P₂)
        _chir_d = 1 - 2 * (_tier_d % 2)  # (-1)^tier_d
        _gamma_d = GAMMA_7 if _tier_d >= 1 else 1.0

        # k=2 dark mode: active for tier_d = 0,1
        if _tier_d <= 1:
            D_dark += (-_chir_d * _gamma_d
                       * (R35_DARK + R57_DARK)
                       * math.cos(2 * _omega_10 * _nf_eff_dk)
                       * (RY / P2))

        # k=4 dark mode: active only tier_d = 0 (per=P₂, 3d metals)
        if _tier_d == 0:
            D_dark += (-_chir_d * R37_DARK
                       * math.cos(4 * _omega_10 * _nf_eff_dk)
                       * (RY / P2))
    D_exch_Hund += D_dark

    result = _DiatP2Result(
        D_cross=D_cross, D_vac=D_vac, D_sd=D_sd, D_sd_promo=D_sd_promo,
        D_P2=D_P2, D_d_pi=D_d_pi, D_exch_Hund=D_exch_Hund,
        D_repulsion=D_repulsion, D_hund_loss=D_hund_loss,
    )
    # Attach d10 factors as extra attribute
    result._d10_factors = _d10_factors  # type: ignore[attr-defined]
    return result


def _diat_face_P3(c: _DiatCommon) -> float:
    """Compute D_ion (ionic channel on heptagonal face P3).

    Returns D_ion.
    """
    r_cov = r_equilibrium(c.per_i, c.per_j, c.bo)
    ie_dom = max(c.ie_i, c.ie_j)
    ie_sub = min(c.ie_i, c.ie_j)
    ionic_stretch = (ie_dom / max(ie_sub, 0.1)) ** (1.0 / P1)
    r_ion = r_cov * ionic_stretch

    # d10s2 ionic radius contraction
    for nf_ion, d_ion_check, Z_ion in [(c.nf_i, c.di, c.Zi), (c.nf_j, c.dj, c.Zj)]:
        if not d_ion_check.get('is_d_block', False):
            continue
        _is_d10_eff = (nf_ion >= 2 * P2) or (
            Z_ion in c._NS1_METALS and nf_ion >= 2 * P2 - 1)
        if _is_d10_eff:
            ea_partner = max(c.ea_i, c.ea_j)
            if ea_partner > S3 * RY:
                r_ion *= C5

    D_ion_raw = max(0.0, COULOMB_EV_A / r_ion - ie_sub + max(c.ea_i, c.ea_j))
    D_ion = D_ion_raw * (1.0 - c.q_rel ** 2)

    # v59B: H⁻ ionic radius expansion [anion on Z/2Z]
    # When H accepts an electron (forming H⁻), it fills Z/2Z completely.
    # The anion radius is much larger than neutral H (152 vs 53 pm).
    # PT: treat H⁻ as period 2 (same shell size as He/Li).
    # Minimum ionic stretch for s-block: 1 + S₃·s = 1 + sin²₃/2.
    # This is the minimum vertex coupling on the triangular face.
    # PT: the extra electron on Z/2Z creates an expansion of at least
    # sin²₃ × s (transfer probability × coupling constant).
    # 0 adjustable parameters.
    per_min = c.per_min
    per_max = c.per_max
    q_rel = c.q_rel
    if per_min == 1 and per_max >= 2 and q_rel < 0.7:
        _per_ion_i = max(c.per_i, 2)
        _per_ion_j = max(c.per_j, 2)
        _r_cov_ion = r_equilibrium(_per_ion_i, _per_ion_j, c.bo)
        _r_ion_new = _r_cov_ion * ionic_stretch
        if c.l_max == 0:
            _f_H_anion = 1.0 + S3 * S_HALF  # = 1 + sin²₃/2 ≈ 1.110
            _r_ion_new = max(_r_ion_new, _r_cov_ion * _f_H_anion)
        D_ion_raw_new = max(0.0, COULOMB_EV_A / _r_ion_new - ie_sub + max(c.ea_i, c.ea_j))
        D_ion = D_ion_raw_new * (1.0 - q_rel ** 2)

    # EA-dependent ionic attenuation
    _ea_max = max(c.ea_i, c.ea_j)
    _ea_threshold = S3 * RY
    if _ea_max < _ea_threshold and D_ion > 0 and c.l_max == 0 and per_max <= 2:
        _f_ea = (_ea_max / RY) ** (S_HALF / P1)
        D_ion *= _f_ea

    # ── v68: Anion polarisation (P₁⊗P₃ Fisher coupling) ──────────────
    # When per_anion > P₁, the anion's LP overflows the hexagonal face
    # Z/(2P₁)Z.  The cation field (from Born-Haber channel) polarises
    # this extended LP cloud, creating additional binding energy.
    # PT energy: COULOMB² × sin²₃ / r_ion⁴ — dipole polarisation with
    # coupling sin²₃ on face P₁.  Radial falloff (P₁/per_anion)^(1/P₁)
    # for LP overflow extent.
    # Gate: per_anion > P₁ (LP overflow), q_rel < s (ionic regime),
    #       lp_anion > 0, D_ion > 0.
    # 0 adjustable parameters.
    if D_ion > 0 and c.q_rel < S_HALF:
        if c.ea_i >= c.ea_j:
            _per_anion = c.per_i
            _lp_anion = c.topology.lp[c.i]
            _per_cation = c.per_j
        else:
            _per_anion = c.per_j
            _lp_anion = c.topology.lp[c.j]
            _per_cation = c.per_i
        # Gate: per_anion > P₁ (LP overflow on hexagonal face),
        #        per_cation ≥ P₁ (cation on P₁ face for cross-coupling),
        #        lp_anion > 0.
        if _per_anion > P1 and _per_cation >= P1 and _lp_anion > 0:
            _f_atten_pol = (float(P1) / float(_per_anion)) ** (1.0 / P1)
            _D_pol = (COULOMB_EV_A * COULOMB_EV_A * S3
                      * float(_lp_anion) / float(P1)
                      * _f_atten_pol / (r_ion ** 4))
            D_ion += _D_pol

    return D_ion


def _diat_apply_modifiers(c: _DiatCommon, D_cov: float, p2: _DiatP2Result,
                          D_ion: float) -> float:
    """Apply all D_cov modifiers and assemble into D0_total.

    This function applies (in exact order):
    - D_cov += D_d_pi
    - d10 cascade factors on D_cov
    - T3 LP cross-face screening on D_cov
    - s-block fill gate on D_cov
    - cross-face P₀⊗P₁ σ correlation on D_cov
    - hexagonal fill (sigma_fill) on D_cov
    - half-fill exchange penalty on D_cov
    - Pauli partiel on D_cov
    - s-block P1 screening on D_cov
    - D_cov_net assembly
    - Pythagore / dative assembly
    - cross-period pi enhancement
    - D_FULL scaling
    - 6 diatomic mechanisms

    Returns D0_total (final value).
    """
    # Add D_d_pi to D_cov (π-symmetry bond on same axis = P1 face)
    D_cov += p2.D_d_pi

    # d10 cascade factors (modifies D_cov)
    for _f_gate in p2._d10_factors:  # type: ignore[attr-defined]
        D_cov *= _f_gate

    # T3 LP cross-face screening
    per_min = c.per_min
    per_max = c.per_max
    if per_min == 1 and per_max >= P1:
        lp_heavy = max(c.lp_i, c.lp_j)
        if lp_heavy > 0:
            r1_A, r1_B = c.Zi % P1, c.Zj % P1
            if r1_A != 0 and r1_B != 0:
                r2_A, r2_B = c.Zi % P2, c.Zj % P2
                r3_A, r3_B = c.Zi % P3, c.Zj % P3
                n_vis = (int(r2_A != 0 and r2_B != 0)
                         + int(r3_A != 0 and r3_B != 0))
                if n_vis > 0:
                    f_lp_occ = float(lp_heavy) / (2.0 * P1)
                    f_T3_LP = 1.0 - S3 * S_HALF * f_lp_occ * float(n_vis) / 2.0
                    D_cov *= f_T3_LP

    # s-block fill gate [DFT fill on Z/(2P₁)Z + √s on Z/2Z]
    # Two s-electrons fill 2/(2P₁) = 1/P₁ of the hexagonal face.
    # √(ns_sum/(2P₁)) = DFT spectral weight (DC mode density on P₁).
    # For ns_max=2 (closed s²): coupling reduced by √s = √(1/2).
    # PT: (1/ns_max)^s = (1/2)^(1/2) = √s. The s-shell (Z/2Z) couples
    # to the σ channel via the fundamental coupling constant s=1/2.
    # The square root is the projection amplitude (not the probability).
    # 0 adjustable parameters.
    if c.l_max == 0 and max(c.per_i, c.per_j) > 1:
        _val_s_A = min(2, valence_electrons(c.Zi))
        _val_s_B = min(2, valence_electrons(c.Zj))
        _n_s_val = float(_val_s_A + _val_s_B)
        _f_fill_s = math.sqrt(min(1.0, _n_s_val / (2.0 * P1)))
        D_cov *= _f_fill_s
        _ns_max = max(_val_s_A, _val_s_B)
        if _ns_max >= 2:
            D_cov *= (1.0 / _ns_max) ** S_HALF  # = √s for ns=2

    # v67: Unified holonomic screening on Z/(2P₁)Z [s + p-block].
    # PT: every s-block atom (Z/2Z) coupling to a p-block partner
    # undergoes holonomic rotation on the hexagonal face Z/(2P₁)Z.
    # sin²(π·np/(2P₁)) is the holonomy at fill position np.
    #
    # Previously restricted to H+p bonds (per_min ≤ 1).
    # Now extended: all s+p bonds, with continuous per-coupling:
    #   H (per=1): weight = 1/P₁          (direct face crossing)
    #   s-block (per>1): weight = 1/P₁ × (1/per_s)^s  (attenuated)
    # Gate: l_min=0 AND l_max=1.  0 adjustable parameters.
    if c.l_min == 0 and c.l_max == 1:
        # Identify s-block and p-block atoms
        if l_of(c.Zi) == 0:
            _Z_p = c.Zj
            _per_s_holo = c.per_i
        else:
            _Z_p = c.Zi
            _per_s_holo = c.per_j
        _val_p = valence_electrons(_Z_p)
        _ns_p = min(2, _val_p)
        _np_p = max(0, _val_p - _ns_p) if l_of(_Z_p) >= 1 else 0
        # Period-dependent coupling weight
        if _per_s_holo <= 1:
            _w_per_holo = 1.0 / P1
        else:
            _w_per_holo = (1.0 / P1) * (1.0 / float(_per_s_holo)) ** S_HALF
        # Holonomic screening on Z/(2P₁)Z
        _holo_screen = math.sin(math.pi * float(_np_p) / (2.0 * P1)) ** 2
        # Anti-double-counting: reduce if T3 LP also screens this atom
        _per_p = period(_Z_p)
        _w_hund = C3 if (int(_Z_p) % P1 != 0 and _per_p >= P1) else 1.0
        _f_hund = 1.0 - _w_per_holo * _w_hund * _holo_screen
        # LP restoration (multiplicative, asymmetric, np > P₁ only)
        _lp_p = max(0, _np_p - P1)
        _lp_coeff = D3 if _w_hund < 1.0 else S3
        _f_lp = 1.0 + float(_lp_p) * _lp_coeff / (P1 * S_HALF) * (_w_per_holo * P1)
        D_cov *= min(_f_hund * _f_lp, 1.0)

    # hexagonal fill (sigma_fill)
    if c.l_min >= 1:
        _val_A_f = valence_electrons(c.Zi)
        _val_B_f = valence_electrons(c.Zj)
        _ns_A_f = min(2, _val_A_f)
        _ns_B_f = min(2, _val_B_f)
        _np_A_f = max(0, _val_A_f - _ns_A_f) if l_of(c.Zi) >= 1 else 0
        _np_B_f = max(0, _val_B_f - _ns_B_f) if l_of(c.Zj) >= 1 else 0
        if c.l_min >= 2:
            if l_of(c.Zi) >= 2:
                _np_A_f = c.di.get('ns', 1)
            if l_of(c.Zj) >= 2:
                _np_B_f = c.dj.get('ns', 1)
        _np_sum_f = _np_A_f + _np_B_f
        _np_capped = min(float(_np_sum_f), 2.0 * P1)
        _sigma_fill = math.sqrt(_np_capped / (2.0 * P1))
        D_cov *= _sigma_fill

    # half-fill exchange penalty
    if c.l_min >= 1 and per_max >= P1 and c.bo >= 2:
        _val_hf_A = valence_electrons(c.Zi)
        _val_hf_B = valence_electrons(c.Zj)
        _ns_hf_A = min(2, _val_hf_A)
        _ns_hf_B = min(2, _val_hf_B)
        _np_hf_A = max(0, _val_hf_A - _ns_hf_A) if l_of(c.Zi) >= 1 else 0
        _np_hf_B = max(0, _val_hf_B - _ns_hf_B) if l_of(c.Zj) >= 1 else 0
        if _np_hf_A == P1 and _np_hf_B == P1:
            _n_per3 = int(c.per_i >= P1) + int(c.per_j >= P1)
            if _n_per3 >= 2:
                D_cov *= C3 ** (1.0 + S_HALF)
            elif _n_per3 == 1:
                D_cov *= C3
        # One half-fill + one LP-bearing (np > P₁): Hund exchange on the
        # half-fill atom weakens σ coupling.  For per ≥ P₁ the 3p/4p
        # orbitals amplify the effect.  Factor C₃^s.
        # Gate: exactly one at half-fill, the other has np > P₁.
        # Excludes: NO (per_max=2<P₁), CP (np_C=2<P₁), BN (np_B=1<P₁).
        # 0 adjustable parameters.
        elif ((_np_hf_A == P1 and _np_hf_B > P1)
              or (_np_hf_B == P1 and _np_hf_A > P1)):
            D_cov *= C3 ** S_HALF

    # Pauli partiel
    if c.l_min >= 1 and c.lp_mutual < P1 and c.bo <= 1.0:
        _val_A_m1 = valence_electrons(c.Zi)
        _val_B_m1 = valence_electrons(c.Zj)
        _ns_A_m1 = min(2, _val_A_m1)
        _ns_B_m1 = min(2, _val_B_m1)
        _np_A_m1 = max(0, _val_A_m1 - _ns_A_m1) if l_of(c.Zi) >= 1 else 0
        _np_B_m1 = max(0, _val_B_m1 - _ns_B_m1) if l_of(c.Zj) >= 1 else 0
        _np_sum_m1 = _np_A_m1 + _np_B_m1
        if _np_sum_m1 > 2 * P1:
            _tent_desc = max(0.0, float(4 * P1 - _np_sum_m1) / (2.0 * P1))
            _pauli_gap = float(P1 - c.lp_mutual) / float(P1)
            _f_pauli_partial = 1.0 - _pauli_gap * (1.0 - _tent_desc)
            D_cov *= max(0.1, _f_pauli_partial)

    # s-block P1 screening [C₃ flat for any per ≥ P₁ atom]
    # The s-shell (Z/2Z) couples to heavier atoms via the P₁ face.
    # Any inner p-shell (per ≥ P₁) screens by C₃ — ONE screening
    # event regardless of how many atoms are heavy (anti-double-counting).
    if c.l_max == 0 and per_max >= P1:
        _alkaline_earth_4p = {20, 38, 56, 88}
        _has_d_hybrid = (c.Zi in _alkaline_earth_4p or c.Zj in _alkaline_earth_4p)
        if not _has_d_hybrid:
            D_cov *= C3

    # v65: D_repulsion cross-face attenuation for d+H bonds (LP_total=0)
    # PT: D_repulsion is intra-face Pauli on Z/(2P₂)Z.  When the partner
    # has no LP and is not d-block (H-like) AND the total LP is 0 (pure
    # covalent d+H), the repulsion can only leak to the σ channel
    # (Z/(2P₁)Z) through the cross-face coupling sin²θ₅.
    # Gate: LP_total = 0 ensures this only applies to the most covalent
    # d-metal hydrides (Fe, Cu, Zn configurations) where the radial
    # mismatch also applies.  D-metals with LP > 0 (Ni, Co, Ag, Au)
    # have screening channels that justify the full D_repulsion.
    # 0 adjustable parameters.
    _D_rep_eff = p2.D_repulsion
    if c.l_max >= 2 and p2.D_repulsion > 0:
        _lp_tot_rep = c.lp_i + c.lp_j
        if _lp_tot_rep == 0 and c.per_min <= 1:
            _D_rep_eff = p2.D_repulsion * S5

    # v67: δ-bond (k=2 Dirichlet mode on Z/P₂Z) for homonuclear d+d.
    # PT: early d-block homonuclear dimers (nf < P₂) have a partially
    # filled spin-up sublattice with non-zero k=2 Fourier mode (δ
    # symmetry).  This mode creates bonding with delta angular nodes
    # (4-fold) that the σ (k=0) and π (k=1) channels do not capture.
    # Gate: homonuclear, both d-block, spin-up sublattice 0 < n_up < P₂.
    # Late d (n_up = P₂, half-fill): |ψ̂(2)|² = 0 automatically.
    # 0 adjustable parameters.  DFT on Z/P₂Z exact.
    _D_delta = 0.0
    if c.Zi == c.Zj and c.l_min >= 2:
        _nf_delta = c.nf_i
        _n_up_delta = min(_nf_delta, P2)
        if 0 < _n_up_delta < P2:
            _SIN2_2PI_P2 = math.sin(2.0 * math.pi / P2) ** 2
            _psi2_up = (math.sin(2.0 * _n_up_delta * math.pi / P2) ** 2
                        / (P2 ** 2 * _SIN2_2PI_P2))
            _D_delta = 2.0 * _psi2_up * (RY / P2)

    # D_cov_net
    D_cov_net = max(0.0, D_cov + p2.D_vac + p2.D_sd + p2.D_sd_promo
                    + p2.D_P2 + p2.D_cross + p2.D_exch_Hund + _D_delta
                    - _D_rep_eff - p2.D_hund_loss)

    # v67: sd-sd cross-term for homonuclear d+d with sd-hybrid.
    # PT: |ψ_bond|² = |cos(α)ψ_s + sin(α)ψ_d|² on BOTH atoms.
    # The engine treats s and d contributions independently (no
    # interference).  For homonuclear d+d, the cross-term
    # 2cos(α)sin(α)⟨s_A|d_B⟩ is INTRA-MOLECULAR and must be
    # subtracted to correct the implicit overcount.
    # Sign: ⟨s|d⟩ > 0 for 3d (per ≤ 4, same-phase orbitals),
    #        ⟨s|d⟩ < 0 for 5d (per ≥ 6, Dirac phase inversion).
    # Overlap: S₃/γ² (sin²θ₃ enhanced by relativistic contraction).
    # Gate: homonuclear, sd-hybrid on both.  0 adjustable parameters.
    if c.Zi == c.Zj and c._sd_hybrid_i and c._sd_hybrid_j:
        _f_sd_cr = float(c.nf_i - P2) / float(P2)
        _sin2a_cr = 2.0 * _f_sd_cr / (1.0 + _f_sd_cr ** 2)
        _za2_cr = (c.Zi * AEM) ** 2
        _gamma2_cr = max(0.01, 1.0 - _za2_cr)
        if c.per_i <= 4:
            _per_sign_cr = 1.0
        elif c.per_i >= 6:
            _per_sign_cr = -1.0
        else:
            _per_sign_cr = 0.0
        _D_cross_sd = (_sin2a_cr * S5 * (RY / P1)
                       * (S3 / _gamma2_cr) * _per_sign_cr)
        D_cov_net = max(0.0, D_cov_net - _D_cross_sd)

    # Dative-polar assemblage
    _is_strongly_dative = False
    if c.l_min >= 1:
        _val_dcheck_A = valence_electrons(c.Zi)
        _val_dcheck_B = valence_electrons(c.Zj)
        _ns_dcheck_A = min(2, _val_dcheck_A)
        _ns_dcheck_B = min(2, _val_dcheck_B)
        _np_dcheck_A = max(0, _val_dcheck_A - _ns_dcheck_A) if l_of(c.Zi) >= 1 else 0
        _np_dcheck_B = max(0, _val_dcheck_B - _ns_dcheck_B) if l_of(c.Zj) >= 1 else 0
        _np_min_dc = min(_np_dcheck_A, _np_dcheck_B)
        if _np_min_dc <= 1 and c.bo > 1.0 and c.Zi != c.Zj:
            _is_strongly_dative = True

    if _is_strongly_dative:
        D0_total = max(D_cov_net, D_ion)
    else:
        D0_total = math.sqrt(D_cov_net ** 2 + D_ion ** 2)

    # Cross-period pi enhancement
    if c.l_min >= 1 and per_max > per_min:
        _val_A_m3 = valence_electrons(c.Zi)
        _val_B_m3 = valence_electrons(c.Zj)
        _ns_A_m3 = min(2, _val_A_m3)
        _ns_B_m3 = min(2, _val_B_m3)
        _np_A_m3 = max(0, _val_A_m3 - _ns_A_m3) if l_of(c.Zi) >= 1 else 0
        _np_B_m3 = max(0, _val_B_m3 - _ns_B_m3) if l_of(c.Zj) >= 1 else 0
        _fill_m3 = max(0, min(_np_A_m3, 2 * P1 - _np_B_m3)
                        + min(_np_B_m3, 2 * P1 - _np_A_m3))
        _np_min_m3 = min(_np_A_m3, _np_B_m3)
        if _fill_m3 >= P2 and _np_min_m3 < P1:
            _fill_frac_m3 = float(_fill_m3) / (2.0 * P1)
            _f_period_gap = 1.0 - float(per_min) / float(per_max)
            _D_sat_pi = (RY / P2) * _fill_frac_m3 * c.eps_geom * _f_period_gap
            D0_total += _D_sat_pi

    D0_total *= D_FULL

    # ── 6 DIATOMIC MECHANISMS (absorbed from _apply_diatomic_mechanisms) ──
    # These replicate the exact logic using _DiatCommon fields instead of
    # BondState fields. Field mappings:
    #   bond_state.z_max → max(topology.z_count[i], topology.z_count[j])
    #   bond_state.q_rel → c.q_rel (NOTE: BondState q_rel uses raw IE/Ry,
    #                       while _DiatCommon q_rel uses modified eps_A/eps_B.
    #                       For diatomics z_count=1 always, so z_max=1.)
    #   bond_state.q_eff → |chi_i - chi_j| / max(0.1, chi_i + chi_j)
    #   bond_state.per_max → c.per_max
    #   bond_state.lp_min → min(lp_i, lp_j)
    #   formal_charge qi, qj → from topology.charges
    #
    # IMPORTANT: BondState.q_rel is computed from RAW ie_i/Ry, ie_j/Ry
    # (before sd hybrid, CRT screening, stiffness modifications).
    # We need to recompute it from the raw IEs.
    _ie_i_raw = c.ie_i / RY
    _ie_j_raw = c.ie_j / RY
    _bs_q_rel = min(_ie_i_raw, _ie_j_raw) / max(_ie_i_raw, _ie_j_raw)

    # chi for q_eff: chi = (IE + EA) / 2 (Mulliken electronegativity)
    _chi_i = (c.ie_i + c.ea_i) / 2.0
    _chi_j = (c.ie_j + c.ea_j) / 2.0
    _bs_q_eff = abs(_chi_i - _chi_j) / max(0.1, _chi_i + _chi_j)

    # z_max for diatomics is always 1 (each atom has exactly 1 neighbor)
    _bs_z_max = max(c.topology.z_count[c.i], c.topology.z_count[c.j])
    _bs_lp_min = min(c.lp_i, c.lp_j)

    # formal charges
    _charges = c.topology.charges if c.topology.charges else []
    _qi = int(_charges[c.i]) if c.i < len(_charges) else 0
    _qj = int(_charges[c.j]) if c.j < len(_charges) else 0

    alkali = {3, 11, 19, 37, 55}
    halogens = _HALOGENS
    chalcogens = _CHALCOGENS

    # Mechanism 1a: diffuse alkali (original, per_a ≥ 4)
    if (
        abs(c.bo - 1.0) < 1e-12
        and _bs_z_max == 1
        and {c.Zi, c.Zj} & alkali
        and {c.Zi, c.Zj} & halogens
    ):
        alkali_idx = c.i if c.Zi in alkali else c.j
        alkali_per = c.per_i if alkali_idx == c.i else c.per_j
        if alkali_per >= 4 and _bs_q_rel < 0.45:
            boost = 1.0 + D5 + D3 * _bs_q_eff
            D0_total *= boost

    # v59: Mechanism 1b: diffuse anion polarization [per_h > P₁]
    # When the halogen has per > P₁, the anion charge cloud is DIFFUSE.
    # Small cations (Li⁺, Na⁺) polarize this cloud more strongly.
    # Gate: alkali+halide, halogen per > P₁, compact cation (per_a < 4).
    # PT: polarization on excess period beyond P₁, modulated by P₁/per_a.
    # 0 adjustable parameters.
    if (
        abs(c.bo - 1.0) < 1e-12
        and _bs_z_max == 1
        and {c.Zi, c.Zj} & alkali
        and {c.Zi, c.Zj} & halogens
    ):
        halogen_idx = c.i if c.Zi in halogens else c.j
        halogen_per = c.per_i if halogen_idx == c.i else c.per_j
        alkali_idx = c.j if halogen_idx == c.i else c.i
        alkali_per = c.per_i if alkali_idx == c.i else c.per_j
        if halogen_per > P1:
            _f_excess = float(halogen_per - P1) / float(halogen_per)
            _f_compact = (float(P1) / float(max(alkali_per, 1))) ** 2
            boost = 1.0 + (D5 + D3 * _bs_q_eff) * _f_excess * _f_compact
            D0_total *= boost

    # Mechanism 2: heavy_homonuclear_halogen
    if (
        abs(c.bo - 1.0) < 1e-12
        and c.Zi == c.Zj
        and c.Zi in halogens
        and c.per_max >= 3
        and _bs_lp_min >= 3
    ):
        # LP polarizability saturates beyond per=4: cap at P₁-1=2
        boost = 1.0 + D5 + D3 * S_HALF * min(max(c.per_max - 2, 1), P1 - 1)
        D0_total *= boost

    # Mechanism 2b: heavy heteronuclear halide LP coupling
    # Two heavy halogens (both per > P₁) have diffuse LP clouds that
    # cross-polarize.  Boost = S₃ × q_rel (face coupling × IE overlap).
    # Gate: bo=1, both halogens, heteronuclear, both per > P₁, lp ≥ P₁.
    # Excludes ClF/BrCl/ICl (per_min ≤ P₁) and F-containing (per=2).
    # 0 adjustable parameters.
    if (
        abs(c.bo - 1.0) < 1e-12
        and c.Zi != c.Zj
        and c.Zi in halogens
        and c.Zj in halogens
        and per_min > P1
        and _bs_lp_min >= P1
    ):
        boost = 1.0 + S3 * _bs_q_rel
        D0_total *= boost

    # Mechanism 3: d_block_delta_bond (label only, no energy change)
    # (no modification to D0_total)

    # Mechanism 4: hetero_radical_pi
    total_odd = (
        (valence_electrons(c.Zi) - _qi) + (valence_electrons(c.Zj) - _qj)
    ) % 2 == 1
    if (
        abs(c.bo - 2.0) < 1e-12
        and c.Zi != c.Zj
        and c.per_max == 2
        and _qi == 0
        and _qj == 0
        and total_odd
    ):
        D0_total *= math.exp(-D5)

    # Mechanism 5: hetero_chalcogen_oxide
    if (
        abs(c.bo - 2.0) < 1e-12
        and _qi == 0
        and _qj == 0
        and _bs_z_max == 1
        and {c.Zi, c.Zj} & chalcogens
        and 8 in {c.Zi, c.Zj}
        and _bs_lp_min >= 2
        and c.per_max >= P1
    ):
        boost = math.exp(D5 * (1.1 + 0.2 * max(c.per_max - P1, 0)))
        D0_total *= boost

    # Mechanism 6: charge_separated_triple
    if (
        abs(c.bo - 3.0) < 1e-12
        and c.Zi != c.Zj
        and _qi * _qj < 0
    ):
        D0_total *= 1.0 + D5

    # Mechanism 7: dative LP→vacancy boost (bo_eff)
    # BF-like molecules: acceptor has np ≤ 1 (large vacancy), same period,
    # donor has LP. The existing dative mechanisms miss ~0.4 eV per pair
    # because the strongly_dative gate (bo > 1) excludes bo=1 molecules.
    # PT: vacancy filling on Z/(2P₁)Z by q_therm branch [Principe 4].
    # Scale: S₃ × S_HALF per pair (not S_HALF alone — avoids double-count
    # with holonomic screening already captured in D_cov).
    # Gate: bo=1, hetero, both p-block, np_acc ≤ 1, same per or per_acc ≤ 2.
    # 0 adjustable parameters.
    if (
        abs(c.bo - 1.0) < 1e-12
        and c.Zi != c.Zj
        and c.l_min >= 1
    ):
        _val_m7_A = valence_electrons(c.Zi)
        _val_m7_B = valence_electrons(c.Zj)
        _ns_m7_A = min(2, _val_m7_A)
        _ns_m7_B = min(2, _val_m7_B)
        _np_m7_A = max(0, _val_m7_A - _ns_m7_A) if l_of(c.Zi) >= 1 else 0
        _np_m7_B = max(0, _val_m7_B - _ns_m7_B) if l_of(c.Zj) >= 1 else 0
        if _np_m7_A <= _np_m7_B:
            _np_acc_m7, _lp_don_m7 = _np_m7_A, c.lp_j
            _per_acc_m7 = c.per_i if _np_m7_A == _np_m7_A else c.per_j
        else:
            _np_acc_m7, _lp_don_m7 = _np_m7_B, c.lp_i
            _per_acc_m7 = c.per_j if _np_m7_B == _np_m7_B else c.per_i
        if _np_acc_m7 <= 1 and _lp_don_m7 > 0 and max(c.per_i, c.per_j) <= 2:
            _vacancy_m7 = P1 - _np_acc_m7
            _n_dat_m7 = min(_lp_don_m7, _vacancy_m7)
            _D_dat_m7 = float(_n_dat_m7) * (RY / P1) * c.eps_geom * S3 * S_HALF
            D0_total += _D_dat_m7

    return D0_total


def _diatomic_bond(i: int, j: int, bo: float,
                   topology: Topology, atom_data: dict) -> float:
    """Refactored diatomic bond energy — calls decomposed functions.

    Bit-exact replacement for _diatomic_bond_Fp + _apply_diatomic_mechanisms.
    Returns the FINAL D0 (already includes D_FULL and all 6 mechanisms).
    """
    c = _diat_common(i, j, bo, topology, atom_data)
    D_cov = _diat_face_P1(c)
    p2 = _diat_face_P2(c)
    D_ion = _diat_face_P3(c)
    D0_total = _diat_apply_modifiers(c, D_cov, p2, D_ion)
    return D0_total


# ╔════════════════════════════════════════════════════════════════════╗
# ║  SECTION 7: BOND SEED — 4-face energy before SCF                   ║
# ╚════════════════════════════════════════════════════════════════════╝

def _compute_bond_seed(i: int, j: int, bo: float, bi: int,
                       topology: Topology, atom_data: dict,
                       huckel_caps: Dict[int, float] = None) -> BondSeed:
    """Compute 4-face seed energy for one polyatomic bond.

    Follows the same path as _compute_bond_Fp in the old engine
    (polyatomic branch, lines 2598-2860), but returns a BondSeed
    with per-face decomposition instead of a scalar D₀.

    Diatomics should use _diatomic_bond_Fp directly.
    """
    atom_states = _resolve_atom_states(topology, atom_data)
    bond_state = _build_bond_state(bi, topology, atom_states)

    Zi, Zj = bond_state.atom_i.Z, bond_state.atom_j.Z
    di, dj = atom_data[Zi], atom_data[Zj]
    per_i, per_j = bond_state.atom_i.per, bond_state.atom_j.per
    lp_i, lp_j = bond_state.atom_i.lp, bond_state.atom_j.lp
    z_i, z_j = bond_state.atom_i.z_count, bond_state.atom_j.z_count
    Q_eff = bond_state.q_eff

    # ══════════════════════════════════════════════════════════════════
    # FACE P₁ (σ + π) — vertex polygon DFT
    # ══════════════════════════════════════════════════════════════════
    S1 = _screening_P1(i, j, bo, topology, atom_data, bi)

    ie_min_bond = min(di['IE'], dj['IE'])
    if ie_min_bond < RY * S_HALF:
        f_bohr_P1 = math.sin(math.pi * ie_min_bond / RY) ** 2
    else:
        f_bohr_P1 = 1.0
    cap_P1 = (RY / P1) * f_bohr_P1

    # Hückel aromatic: replace pairwise σ+π with Hückel cap [dim 2]
    if huckel_caps and bi in huckel_caps:
        D0_P1 = (cap_P1 + huckel_caps[bi]) * math.exp(-S1 * D_FULL_P1)
    else:
        D0_P1 = cap_P1 * math.exp(-S1 * D_FULL_P1)
        # π component
        n_pi = max(0.0, bo - 1.0)
        if n_pi > 0:
            z_i_v = topology.z_count[i]
            z_j_v = topology.z_count[j]
            lp_i_v = topology.lp[i]
            lp_j_v = topology.lp[j]
            rad_i = (valence_electrons(Zi) - round(topology.sum_bo[i])) % 2 == 1
            rad_j = (valence_electrons(Zj) - round(topology.sum_bo[j])) % 2 == 1
            bent_i = (lp_i_v > 0 or rad_i) and z_i_v >= 2
            bent_j = (lp_j_v > 0 or rad_j) and z_j_v >= 2

            f_pi_share = 1.0
            if bent_i or bent_j:
                n_multi_i = sum(1 for bi2, (i2, j2, bo2) in enumerate(topology.bonds)
                               if (i2 == i or j2 == i) and bo2 > 1) if bent_i else 1
                n_multi_j = sum(1 for bi2, (i2, j2, bo2) in enumerate(topology.bonds)
                               if (i2 == j or j2 == j) and bo2 > 1) if bent_j else 1
                f_pi_share = 1.0 / max(n_multi_i, n_multi_j, 1)

            if f_pi_share < 1.0:
                cap_pi = min(n_pi, 2.0) * cap_P1 * S_HALF * f_pi_share
            else:
                cap_pi = min(n_pi, 1.0) * (RY / P2)
                if n_pi > 1.0:
                    cap_pi += min(n_pi - 1.0, 1.0) * (RY / P3)
            # π-LP vertex quenching
            f_pi_quench = 1.0
            for v_pi in (i, j):
                if topology.z_count[v_pi] < 2:
                    continue
                lp_diffuse = 0.0
                for bi_q in topology.vertex_bonds.get(v_pi, []):
                    if bi_q == bi:
                        continue
                    ii_q, jj_q, _ = topology.bonds[bi_q]
                    t_q = jj_q if ii_q == v_pi else ii_q
                    lp_q = topology.lp[t_q]
                    if lp_q < P1:
                        continue
                    per_q = atom_data[topology.Z_list[t_q]]['per']
                    if per_q <= 2:
                        continue
                    lp_diffuse += lp_q * (per_q - 2.0) / P1
                if lp_diffuse > 0:
                    q_frac = min(lp_diffuse / (2.0 * P1), 1.0)
                    f_pi_quench = min(f_pi_quench, 1.0 - S3 * q_frac)
            D0_P1 += cap_pi * f_pi_quench * math.exp(-S1 * math.sqrt(C3) * D_FULL_P1)

    if di.get("is_d_block", False) and dj.get("is_d_block", False) and bo >= 3.0:
        D0_P1 += (RY / P3) * math.exp(-S1 * D_FULL_P1)

    # ══════════════════════════════════════════════════════════════════
    # d-BLOCK POLYATOMIC CORRECTIONS (applied after D0_P1, before assembly)
    # ══════════════════════════════════════════════════════════════════
    _has_d_block = di.get('is_d_block', False) or dj.get('is_d_block', False)
    if _has_d_block:
        _NS1_METALS_POLY = frozenset({24, 29, 41, 42, 44, 45, 46, 47, 78, 79})

        nf_i_d = di.get('nf', 0) if di.get('is_d_block', False) else 0
        nf_j_d = dj.get('nf', 0) if dj.get('is_d_block', False) else 0

        # C1: ε-screening C₅^(nf/2P₂) for nf > P₂
        for nf_c1, d_c1, Z_c1 in [(nf_i_d, di, Zi), (nf_j_d, dj, Zj)]:
            if not d_c1.get('is_d_block', False) or nf_c1 <= P2:
                continue
            partner_idx_c1 = j if Z_c1 == Zi else i
            lp_partner_c1 = topology.lp[partner_idx_c1]
            if lp_partner_c1 < P1:
                continue
            D0_P1 *= C5 ** (float(nf_c1) / (2.0 * P2))

        # C6: d¹⁰s² closed-shell gate modulated by ligand field
        for Z_g, nf_g, d_g in [(Zi, nf_i_d, di), (Zj, nf_j_d, dj)]:
            if not d_g.get('is_d_block', False):
                continue
            is_ns1_g = Z_g in _NS1_METALS_POLY
            if nf_g >= 2 * P2 and not is_ns1_g:
                _partner_g = dj if Z_g == Zi else di
                _pidx_g = j if Z_g == Zi else i
                _lp_g = topology.lp[_pidx_g]
                _ea_g = _partner_g['EA']
                _f_field_g = min(1.0, max(float(_lp_g) / P1, _ea_g / RY))
                _f_gate_g = C5 * S_HALF + (1.0 - C5 * S_HALF) * _f_field_g
                D0_P1 *= _f_gate_g

        # C8: Pauli repulsion for nf > P₂
        D_repulsion_poly = 0.0
        for Z_rp, nf_rp, d_rp in [(Zi, nf_i_d, di), (Zj, nf_j_d, dj)]:
            if not d_rp.get('is_d_block', False):
                continue
            is_ns1_rp = Z_rp in _NS1_METALS_POLY
            if nf_rp > P2 and nf_rp < 2 * P2:
                n_excess_rp = nf_rp - P2
                f_ns_rp = 1.0 + S_HALF if not is_ns1_rp else S_HALF
                D_repulsion_poly += S5 * (float(n_excess_rp) / P2) ** 2 * (RY / P2) * f_ns_rp

        # C9: Hund exchange loss
        D_hund_poly = 0.0
        J_exchange_poly = S5 * RY / (2.0 * P2)
        for Z_hp, nf_hp, d_hp in [(Zi, nf_i_d, di), (Zj, nf_j_d, dj)]:
            if not d_hp.get('is_d_block', False) or nf_hp == 0:
                continue
            if Z_hp in _NS1_METALS_POLY and nf_hp >= 2 * P2 - 1:
                continue
            if nf_hp >= 2 * P2:
                continue
            n_up_hp = min(nf_hp, P2)
            delta_K_hp = max(0, n_up_hp - 1)
            D_hund_poly += delta_K_hp * J_exchange_poly

        if D_hund_poly > 0:
            f_field_poly = 0.0
            for idx_fp in (i, j):
                d_fp = atom_data[topology.Z_list[idx_fp]]
                if not d_fp.get('is_d_block', False):
                    ea_fp = d_fp['EA']
                    lp_fp = topology.lp[idx_fp]
                    # ── Through-bond LP field [DCD synergistic, 0 params] ──
                    # When the direct partner B has lp=0 and z=2 (bridge),
                    # terminal LP couples through B to create an effective
                    # ligand field on the d-metal.  The field strength is
                    # attenuated by the bridge Bohr amplitude sin²(πIE/(2Ry))
                    # — the spectral weight of B on Z/(2P₁)Z.
                    # Gate: B not d-block, B has z=2, terminal has LP > 0.
                    lp_eff = float(lp_fp)
                    if lp_fp == 0 and topology.z_count[idx_fp] == 2:
                        _ie_br = d_fp['IE']
                        _f_br = math.sin(
                            math.pi * min(_ie_br, RY) / (2.0 * RY)) ** 2
                        for _bk_fp in topology.vertex_bonds.get(idx_fp, []):
                            _ik, _jk, _ = topology.bonds[_bk_fp]
                            _tk = _jk if _ik == idx_fp else _ik
                            if _tk in (i, j):
                                continue
                            _lp_tk = topology.lp[_tk]
                            if _lp_tk > 0 and topology.z_count[_tk] <= 1:
                                lp_eff += float(_lp_tk) * _f_br
                    f_field_poly = max(f_field_poly,
                                       min(1.0, max(ea_fp / RY, lp_eff / P1)))
            D_hund_poly *= (1.0 - f_field_poly)

        D0_P1 = max(0.0, D0_P1 - D_repulsion_poly - D_hund_poly)

    # ══════════════════════════════════════════════════════════════════
    # FACE P₂ (pentagonal DFT screening)
    # ══════════════════════════════════════════════════════════════════
    S_P2_dft = (_vertex_polygon_dft_P2(i, bi, topology, atom_data)
                + _vertex_polygon_dft_P2(j, bi, topology, atom_data)) / 2.0

    cap_p2, S_p2_legacy = _cap_P2(i, j, bo, topology, atom_data)
    if cap_p2 > 0:
        ie_min_p2 = min(di['IE'], dj['IE'])
        f_bohr_p2 = math.sin(math.pi * min(ie_min_p2, RY) / (2.0 * RY)) ** 2
        cap_p2 *= f_bohr_p2
    S_P2_eff = S_P2_dft * S_HALF
    D0_P2 = cap_p2 * math.exp(-S_P2_eff * D_FULL_P2) if cap_p2 > 0 else 0.0

    # ══════════════════════════════════════════════════════════════════
    # FACE P₃ (heptagonal DFT screening — ionic)
    # ══════════════════════════════════════════════════════════════════
    S_P3_dft = (_vertex_polygon_dft_P3(i, bi, topology, atom_data)
                + _vertex_polygon_dft_P3(j, bi, topology, atom_data)) / 2.0

    cap_p3, S_p3_legacy, Q_p3 = _cap_P3(i, j, bo, topology, atom_data)
    S_P3_eff = S_P3_dft * S5 * (1.0 - Q_eff)
    D0_P3 = cap_p3 * math.exp(-S_P3_eff * DEPTH_P3) if cap_p3 > 0 else 0.0

    # ══════════════════════════════════════════════════════════════════
    # FACE P₀ (binary face Z/2Z — LP channel, NEW)
    # ══════════════════════════════════════════════════════════════════
    S0 = _screening_P0(i, j, topology, atom_data)
    cap_p0 = _cap_P0(i, j, topology, atom_data)
    D0_P0 = cap_p0 * math.exp(-S0) if cap_p0 > 0 else 0.0

    # ══════════════════════════════════════════════════════════════════
    # VACANCY BOOST (dim 2)
    # ══════════════════════════════════════════════════════════════════
    # C22: Gate out vacancy boost for triple bonds (bo >= 2.5).
    # On Z/(2P₁)Z, bo=3 occupies σ + 2π modes. The 2nd π mode
    # (cap Ry/P₃) resonates with the heteroatom LP. The vacancy
    # boost would double-count this LP as dative donation.
    # 0 adjustable parameters: P₁, S_HALF from s = 1/2.
    if bo < P1 - S_HALF:  # < 2.5: single and double bonds only
        D0_vac = _dim2_vacancy_boost(i, j, bo, topology, atom_data)
    else:
        D0_vac = 0.0

    # ══════════════════════════════════════════════════════════════════
    # TOTAL: interpolation covalent + ionic [Pythagore]
    # ══════════════════════════════════════════════════════════════════
    D_cov = D0_P0 + D0_P1 + D0_P2 + D0_vac
    D_ion = D0_P3
    D_add = D_cov + D_ion
    D_pyth = math.sqrt(D_cov ** 2 + D_ion ** 2)
    D0_total = D_add * (1.0 - Q_eff ** 2) + D_pyth * Q_eff ** 2

    return BondSeed(
        D0=D0_total * D_FULL,
        D0_P0=D0_P0,
        D0_P1=D0_P1,
        D0_P2=D0_P2,
        D0_P3=D0_P3,
        S0=S0,
        S1=S1,
        S2=S_P2_dft,
        S3=S_P3_dft,
        Q_eff=Q_eff,
    )


# ╔════════════════════════════════════════════════════════════════════╗
# ║  SECTION 7b: V4 SPECTRAL BOND SEED + PARSEVAL BUDGET              ║
# ╚════════════════════════════════════════════════════════════════════╝

def _compute_bond_seed_v4(i: int, j: int, bo: float, bi: int,
                          topology: Topology, atom_data: dict,
                          huckel_caps: Dict[int, float] = None) -> BondSeed:
    """Compute bond seed with v4's DFT-GFT spectral cross-spectrum correction.

    This function:
    1. Calls the proven _compute_bond_seed for the base computation
    2. Computes v4's DFT cross-spectrum on Z/(2P₁)Z for both atoms
    3. Applies a spectral coherence correction to the P₁ face energy

    The cross-spectrum captures inter-atomic spectral coherence:
    when both atoms have aligned modes on Z/(2P₁)Z, the bond coupling
    is enhanced (constructive interference). When modes are misaligned,
    coupling is reduced (destructive interference).

    The Parseval budget is applied AFTER all seeds are computed.

    0 adjustable parameters. All from s = 1/2.
    """
    # Step 1: Get the proven base seed
    base_seed = _compute_bond_seed(i, j, bo, bi, topology, atom_data, huckel_caps)

    # Step 2: Compute DFT cross-spectrum on Z/(2P₁)Z
    Zi, Zj = topology.Z_list[i], topology.Z_list[j]
    di, dj = atom_data[Zi], atom_data[Zj]
    l_i, l_j = di['l'], dj['l']

    np_i = _np_of_v4(Zi)
    np_j = _np_of_v4(Zj)

    # Electron counts on hex face
    if l_i == 0:
        n_hex_i = 0
    elif l_i == 2:
        n_hex_i = di.get('ns', 1)
    else:
        n_hex_i = np_i
    if l_j == 0:
        n_hex_j = 0
    elif l_j == 2:
        n_hex_j = dj.get('ns', 1)
    else:
        n_hex_j = np_j

    lp_hex_i = max(0, n_hex_i - P1) if l_i == 1 else 0
    lp_hex_j = max(0, n_hex_j - P1) if l_j == 1 else 0

    # Build electron densities and DFT spectra
    rho_i = electron_density(n_hex_i, P1, lp=lp_hex_i)
    rho_j = electron_density(n_hex_j, P1, lp=lp_hex_j)
    spec_i = dft_spectrum(rho_i, P1)
    spec_j = dft_spectrum(rho_j, P1)

    # Cross-spectrum power at k > 0 (exclude DC component k=0)
    cross_power = 0.0
    for k in range(1, P1):
        cross_power += (spec_i[k] * spec_j[k].conjugate()).real

    # Step 3: Apply spectral correction to D0_P1
    #
    # In a POLYATOMIC context, high cross-spectrum power means the
    # bond partners' spectral modes are coherent, USING UP spectral
    # budget that could be shared with other bonds at this vertex.
    # This gives ADDITIONAL screening (reduction) for bonds between
    # atoms with highly coherent spectra.
    #
    # The correction factor: exp(-D₃ × cross_power)
    # D₃ = S₃(1-S₃) ≈ 0.172 keeps it perturbative.
    # Negative sign: high coherence = used-up budget = more screening.
    f_spectral = math.exp(-D3 * cross_power)

    # ── k=3 Nyquist parity screening (C12) ──
    # Mode k=3 on Z/(2P₁)Z is the Nyquist mode (alternating ±1).
    # Non-zero only for ODD hex-face electron counts.
    # Captures parity frustration not resolved by k=0,1,2.
    # Gate: both p-block, bo ≤ 1, asymmetric parity (exactly one
    # atom has n_hex > P₁ AND n_hex odd).
    # Screening: exp(-S₃·D₃·overfill·f_per)
    # 0 parameters. S₃, D₃ from s = 1/2.
    l_min_k3 = min(l_i, l_j)
    _f_k3 = 1.0
    if l_min_k3 >= 1 and bo <= 1:
        _k3_i = (n_hex_i > P1 and n_hex_i % 2 == 1)
        _k3_j = (n_hex_j > P1 and n_hex_j % 2 == 1)
        if _k3_i != _k3_j:
            if _k3_i:
                _n_k3 = n_hex_i
                _per_k3 = di['per']
            else:
                _n_k3 = n_hex_j
                _per_k3 = dj['per']
            _overfill_k3 = float(_n_k3 - P1) / P1
            _f_per_k3 = (2.0 / float(max(_per_k3, 2))) ** (1.0 / P1) if _per_k3 > 2 else 1.0
            _f_k3 = math.exp(-S3 * D3 * _overfill_k3 * _f_per_k3)

    D0_P1_corrected = base_seed.D0_P1 * f_spectral * _f_k3

    # Scale the total D0 by the same ratio that P1 changed.
    # This preserves the vacancy boost and other contributions
    # that are baked into base_seed.D0 but not in the face decomposition.
    if base_seed.D0_P1 > 1e-10:
        scale = D0_P1_corrected / base_seed.D0_P1
        # The P1 face dominates, so scale total proportionally.
        # But only the P1 fraction of the total is affected.
        p1_fraction = base_seed.D0_P1 / max(
            base_seed.D0_P0 + base_seed.D0_P1 + base_seed.D0_P2 + base_seed.D0_P3, 1e-10)
        # Total adjustment: 1 + p1_fraction * (scale - 1)
        f_total = 1.0 + p1_fraction * (scale - 1.0)
        D0_corrected = base_seed.D0 * f_total
    else:
        D0_corrected = base_seed.D0

    return BondSeed(
        D0=D0_corrected,
        D0_P0=base_seed.D0_P0,
        D0_P1=D0_P1_corrected,
        D0_P2=base_seed.D0_P2,
        D0_P3=base_seed.D0_P3,
        S0=base_seed.S0,
        S1=base_seed.S1,
        S2=base_seed.S2,
        S3=base_seed.S3,
        Q_eff=base_seed.Q_eff,
    )


def _compute_bond_seed_spectral(topology: Topology, atom_data: dict,
                                huckel_caps: Dict[int, float] = None
                                ) -> Dict[int, BondSeed]:
    """Per-bond seeds from v4 spectral blending (v4 ⊗ 4-face DFT).

    For each bond, computes BOTH the v4 diatomic seed and the 4-face
    DFT seed (with cross-spectrum correction), then caps overbinding:

    When DFT/v4 > (1 + S₃) ≈ 1.219, the DFT seed overbinds relative
    to the v4 reference. The cap blends toward v4 × (1 + S₃), modulated
    by molecule size: full for n=3, fading to zero for n >= P₃ = 7.

    For n >= 7, the DFT + SCF pipeline is well-calibrated and no
    cap is needed. The v4 seed serves as a SPECTRAL ANCHOR: the truth
    about the per-bond energy in the limit of no multi-center effects.

    0 adjustable parameters (cap ratio = 1 + S₃, size fade = (n-P₁)/(P₃-P₁)).
    """
    n_atoms = topology.n_atoms
    bond_seeds: Dict[int, BondSeed] = {}

    for bi, (i, j, bo) in enumerate(topology.bonds):
        Zi, Zj = topology.Z_list[i], topology.Z_list[j]
        lp_i = topology.lp[i]
        lp_j = topology.lp[j]

        # ── v4 diatomic reference ──
        sr = _D0_screening_v4(Zi, Zj, bo, lp_A=lp_i, lp_B=lp_j)
        D0_v4 = sr.D0

        # ── 4-face DFT seed (with v4 cross-spectrum correction) ──
        dft_seed = _compute_bond_seed_v4(i, j, bo, bi, topology, atom_data, huckel_caps)
        D0_dft = dft_seed.D0

        # ── Spectral blending ──
        # The v4 seed is the truth for an isolated bond.
        # The DFT seed captures multi-center context (promotion sharing,
        # vacancy, LP cooperation) but systematically overbinds for
        # small molecules (n=3-6) and is well-calibrated for large ones.
        #
        # Blending strategy:
        # 1. Cap DFT at v4 × (1 + S₃) when overbinding detected
        # 2. The cap is MODULATED by molecule size: full for n=3,
        #    fading to zero for n>6 (where DFT + SCF work well).
        #
        # Size modulation: f_size = max(0, 1 - (n-3)/(P₃-3))
        # This gives f=1.0 for n=3, f=0.5 for n=5, f=0.0 for n≥7.
        # P₃=7 is the PT scale where SCF becomes reliable (T³ resolved).
        ratio = D0_dft / max(D0_v4, 1e-10)

        _CAP_RATIO = 1.0 + S3      # ≈ 1.219
        f_size = max(0.0, 1.0 - float(n_atoms - P1) / (P3 - P1))  # 1@n=3, 0@n≥7

        if ratio > _CAP_RATIO and f_size > 0:
            # DFT overbinds: blend toward v4 × cap_ratio
            D0_capped = D0_v4 * _CAP_RATIO
            # Interpolate: D0 = f_size × D0_capped + (1 - f_size) × D0_dft
            D0_blend = f_size * D0_capped + (1.0 - f_size) * D0_dft
            scale = D0_blend / max(D0_dft, 1e-10)
            bond_seeds[bi] = BondSeed(
                D0=D0_blend,
                D0_P0=dft_seed.D0_P0 * scale,
                D0_P1=dft_seed.D0_P1 * scale,
                D0_P2=dft_seed.D0_P2 * scale,
                D0_P3=dft_seed.D0_P3 * scale,
                S0=dft_seed.S0,
                S1=dft_seed.S1,
                S2=dft_seed.S2,
                S3=dft_seed.S3,
                Q_eff=dft_seed.Q_eff,
            )
        else:
            # DFT seed is within tolerance, molecule too large,
            # or linear multi-bond: keep DFT seed
            bond_seeds[bi] = dft_seed

    return bond_seeds


def _parseval_budget(topology: Topology, atom_data: dict,
                     bond_seeds: Dict[int, BondSeed]) -> Dict[int, BondSeed]:
    """Apply Parseval spectral budget constraint to bond seeds.

    Physics: on Z/(2P₁)Z, atom B's spectral density ρ̂_B(k) satisfies

        Σ_k |ρ̂_B(k)|² = n_p / (2P₁)     [Parseval on finite group]

    When B participates in z bonds, each bond gets fraction 1/z of B's
    spectral power. The per-bond attenuation at a shared vertex is:

        f_Parseval(z) = 1 / z^(S_HALF/P₁)  = 1 / z^(1/6)

    The exponent S_HALF/P₁ = 1/6 comes from the hexagonal face geometry:
    on Z/6Z with 2P₁ = 6 positions, sharing among z bonds reduces each
    bond's spectral power by z^(-1/6).

    The budget is applied only when z > P_l (hypercoordinated), because
    the NLO vertex coupling (C3), Dicke coherence (C6), face-fraction
    overcrowding (C1) and LP crowding (C2) already handle the
    multi-center budget distribution at lower coordinations.

    0 adjustable parameters: exponent = s/P₁ = (1/2)/3 = 1/6.
    """
    # ── Pre-compute per-vertex Parseval factors ──
    _expo_P1 = S_HALF / P1   # 1/6 ≈ 0.16667

    n_atoms = len(topology.Z_list)
    f_vertex = [1.0] * n_atoms

    for k in range(n_atoms):
        z_k = topology.z_count[k]
        if z_k <= 1:
            continue  # terminal atoms: full spectral budget

        # Only applies when z > P_l (coordination exceeds the face capacity)
        Zk = topology.Z_list[k]
        l_k = atom_data[Zk]['l']
        per_k = atom_data[Zk]['per']
        P_l_k = {0: 2, 1: P1, 2: P2, 3: P3}.get(l_k, P1)
        if per_k >= P1 and z_k > P_l_k:
            n_excess = z_k - P_l_k
            # Size-dependent exponent: strengthen for small molecules
            # f_size_k ∈ [0,1]: 1 at n=3, 0 at n≥7 (same PT fade)
            _f_size_k = max(0.0, 1.0 - float(n_atoms - P1) / (P3 - P1))
            _expo_eff = (D3 / P1) * (1.0 + _f_size_k * float(z_k - 1) / P1)
            f_vertex[k] = (1.0 + float(n_excess) / P_l_k) ** (-_expo_eff)

        elif (z_k == P_l_k and z_k >= P1
              and topology.lp[k] == 0):
            # ── Parseval sharing at face saturation (z = P_l, lp = 0) ──
            #
            # When z = P_l bonds fill the face Z/(2P_l)Z completely and
            # the centre carries no lone pair, the Parseval budget
            #   Σ_k |ψ̂(k)|² = 1/(2P_l)
            # is consumed entirely by bond spectral densities.
            #
            # Each bond seed was computed pairwise — as if the bond had
            # the full face.  The sharing tax per vertex:
            #
            #   f = 1 − D₃ × (z−1)/P_l × f_size
            #
            # Gates (all required):
            #   (a) z = P_l, lp = 0
            #   (b) all bonds at the vertex share the same bond order
            #       (symmetric face partitioning; heterogeneous vertices
            #       have pairwise overcounting that partially cancels)
            #   (c) all adjacent terminals are sub-Bohr (IE ≤ Ry(1+S₃))
            #       When terminals are super-Bohr (e.g. F in BF₃), LP
            #       donation through the pentagonal face Z/(2P₂)Z
            #       compensates the hexagonal sharing deficit — the
            #       sharing tax is already offset by that channel.
            #   (d) centre NOT electron-deficient with d-vacancy
            #       (nf ≤ 1 with per ≥ P₁): for d-block accessible atoms
            #       with nf ≤ 1 (Al, etc.), the d-vacancy back-donation
            #       from terminal LP already compensates the face budget
            #       deficit.  The sharing tax would double-correct.
            #
            # 0 adjustable parameters: D₃, P_l, f_size all from s = 1/2.
            _v_bonds_k = topology.vertex_bonds.get(k, [])
            _sym_bo = False
            _all_sub_bohr = True
            _deficient_d = (atom_data[Zk].get('nf', 0) <= 1
                            and per_k >= P1 and l_k >= 1)
            if _v_bonds_k and not _deficient_d:
                _bos_k = [topology.bonds[_bk][2] for _bk in _v_bonds_k]
                _sym_bo = (max(_bos_k) - min(_bos_k) < 0.01)
                # Check terminal IEs
                _IE_bohr_pars = RY * (1.0 + S3)
                for _bk in _v_bonds_k:
                    _ik, _jk, _ = topology.bonds[_bk]
                    _tk = _jk if _ik == k else _ik
                    if atom_data[topology.Z_list[_tk]]['IE'] > _IE_bohr_pars:
                        _all_sub_bohr = False
                        break
            if _sym_bo and _all_sub_bohr:
                _f_size_k = max(0.0, 1.0 - float(n_atoms - P1) / (P3 - P1))
                if _f_size_k > 0:
                    _expo_sat = D3 * float(z_k - 1) / P_l_k
                    f_vertex[k] = 1.0 - _expo_sat * _f_size_k
        # else: f_vertex[k] = 1.0 (already set)

    # ── Apply per-bond ──
    updated_seeds: Dict[int, BondSeed] = {}

    for bi, seed in bond_seeds.items():
        ii, jj, bo = topology.bonds[bi]

        # Geometric mean of the two vertex Parseval factors
        f_parseval_norm = math.sqrt(f_vertex[ii] * f_vertex[jj])

        # Scale P₁ face contribution
        D0_P1_scaled = seed.D0_P1 * f_parseval_norm

        # Scale total D0 proportionally (preserving P₀/P₂/P₃ balance)
        if seed.D0_P1 > 1e-10:
            p1_frac = seed.D0_P1 / max(
                seed.D0_P0 + seed.D0_P1 + seed.D0_P2 + seed.D0_P3, 1e-10)
            f_total = 1.0 + p1_frac * (f_parseval_norm - 1.0)
            D0_scaled = seed.D0 * f_total
        else:
            D0_scaled = seed.D0

        updated_seeds[bi] = BondSeed(
            D0=D0_scaled,
            D0_P0=seed.D0_P0,
            D0_P1=D0_P1_scaled,
            D0_P2=seed.D0_P2,
            D0_P3=seed.D0_P3,
            S0=seed.S0,
            S1=seed.S1,
            S2=seed.S2,
            S3=seed.S3,
            Q_eff=seed.Q_eff,
        )

    return updated_seeds


# ╔════════════════════════════════════════════════════════════════════╗
# ║  SPECTRAL SCF — MODE-RESOLVED PARSEVAL ITERATION                   ║
# ║                                                                    ║
# ║  Physics: each atom i has spectral density ρ̂_i(k) on Z/(2P₁)Z.  ║
# ║  Each bond draws spectral power from BOTH vertices.                ║
# ║  Self-consistent condition: total demand at each vertex ≤ budget.  ║
# ║  0 adjustable parameters — budget = Parseval sum |ρ̂(k)|².        ║
# ╚════════════════════════════════════════════════════════════════════╝

def _scf_spectral(topology: Topology, atom_data: dict,
                  bond_seeds: Dict[int, BondSeed],
                  max_iter: int = 20,
                  tol: float = None) -> Dict[int, BondSeed]:
    """Spectral SCF: cross-spectrum competition on Z/(2P₁)Z.

    For each vertex with z ≥ 2 bonds, the DFT cross-spectrum between
    bond partners measures how much the bonds COMPETE for the same
    exchange modes at the central atom. This determines the per-bond
    screening correction:

    High competition (partners spectrally similar, e.g. CO₂ with
    two O partners): bonds share the same exchange channel → large
    screening correction → D0 reduced.

    Low competition (partners spectrally orthogonal, e.g. H₂O with
    two H partners): bonds use different modes → small correction.

    The SCF iterates the vertex screening:

    For each vertex v with bonds b₁, b₂, ...:
        For each pair (bₐ, bᵦ):
            competition(a,b) = Σ_{k>0} Re[ρ̂_pₐ(k) × conj(ρ̂_pᵦ(k))]
                             / Σ_{k>0} |ρ̂_v(k)|²

        S_comp(v) = D₃ × mean_competition / P₁

    The competition metric is EXACTLY the cross-spectrum of the two
    bond partners projected through the vertex DFT, normalized by the
    vertex's total exchange power. This is a DFT inner product.

    Convergence: monotonic, bounded by Parseval.
    0 adjustable parameters.
    """
    n_atoms = topology.n_atoms
    n_bonds = len(topology.bonds)

    if n_atoms < 3 or n_bonds < 2:
        return bond_seeds

    if tol is None:
        tol = AEM  # ≈ 0.00734 eV

    # ── Step 1: compute DFT spectra for all atoms ──────────────────────
    atom_specs = {}   # Z → (fill, exchange, spectrum)
    for k in range(n_atoms):
        Z_k = topology.Z_list[k]
        if Z_k in atom_specs:
            continue
        d_k = atom_data[Z_k]
        l_k = d_k['l']
        np_k = _np_of_v4(Z_k) if l_k >= 1 else 0
        lp_k = topology.lp[k]
        P_face = P2 if d_k.get('is_d_block', False) else P1

        rho_k = electron_density(np_k, P_face, lp=lp_k)
        spec_k = dft_spectrum(rho_k, P_face)

        fill_k = abs(spec_k[0]) ** 2
        exchange_k = sum(abs(spec_k[m]) ** 2 for m in range(1, P_face))

        atom_specs[Z_k] = (fill_k, exchange_k, spec_k, P_face)

    # ── Step 2: compute per-vertex spectral factors ─────────────────────
    vertex_competition = [0.0] * n_atoms
    f_vertex = [1.0] * n_atoms

    for v in range(n_atoms):
        z_v = topology.z_count[v]
        if z_v <= 1:
            continue

        Z_v = topology.Z_list[v]
        d_v = atom_data[Z_v]
        per_v = d_v['per']
        l_v = d_v['l']
        P_l_v = {0: 2, 1: P1, 2: P2, 3: P3}.get(l_v, P1)

        fill_v, exchange_v, spec_v, P_face = atom_specs[Z_v]

        bonds_at_v = topology.vertex_bonds.get(v, [])
        n_pairs = 0
        total_competition = 0.0

        for a in range(len(bonds_at_v)):
            bi_a = bonds_at_v[a]
            ia, ja, boa = topology.bonds[bi_a]
            pa = ja if ia == v else ia
            Z_pa = topology.Z_list[pa]
            _, _, spec_pa, P_pa = atom_specs.get(Z_pa, (0, 0, None, P1))

            for b in range(a + 1, len(bonds_at_v)):
                bi_b = bonds_at_v[b]
                ib, jb, bob = topology.bonds[bi_b]
                pb = jb if ib == v else ib
                Z_pb = topology.Z_list[pb]
                _, _, spec_pb, P_pb = atom_specs.get(Z_pb, (0, 0, None, P1))

                if spec_pa is None or spec_pb is None:
                    n_pairs += 1
                    continue

                P_cross = min(P_pa, P_pb, P_face)

                overlap = 0.0
                for k in range(1, P_cross):
                    overlap += (spec_pa[k] * spec_pb[k].conjugate()).real

                if exchange_v > 1e-12:
                    competition = max(0.0, overlap / exchange_v)
                else:
                    competition = 0.0

                total_competition += min(competition, 1.0)
                n_pairs += 1

        if n_pairs > 0:
            mean_comp = total_competition / n_pairs
        else:
            mean_comp = 0.0

        vertex_competition[v] = mean_comp

        # ── Hypercoordinated: polygon promotion (same as old budget) ──
        if z_v > P_l_v and per_v >= P1:
            n_excess = z_v - P_l_v
            f_hyper = (1.0 + float(n_excess) / P_l_v) ** (-(D3 / P1))
            f_vertex[v] = f_hyper

    # ── Step 3: apply competition + hypercoordinated factors ─────────
    D0 = {bi: seed.D0 for bi, seed in bond_seeds.items()}

    # Competition screening at the seed level:
    # S_comp(v) = D₃ × competition × (z-1)/z × f_size
    # D₃ is the Fisher dispersion = one-loop coupling on Z/3Z.
    # (z-1)/z is the fraction of bonds that compete.
    # f_size fades the correction for large molecules where the
    # existing C1-C8 corrections already handle multi-center effects.
    #
    # f_size = max(0, 1 - (n - P₁) / (P₃ - P₁))
    # Gives f=1.0 for n≤3, f=0.5 for n=5, f=0 for n≥7.
    f_size = max(0.0, 1.0 - float(n_atoms - P1) / (P3 - P1))
    for v in range(n_atoms):
        z_v = topology.z_count[v]
        if z_v <= 1 or vertex_competition[v] < 1e-6:
            continue
        S_comp = D3 * vertex_competition[v] * float(z_v - 1) / float(z_v) * f_size
        f_vertex[v] *= math.exp(-S_comp)

    # Apply factors to bonds (geometric mean of vertices)
    for bi in range(n_bonds):
        ii, jj, bo = topology.bonds[bi]
        f_bond = math.sqrt(f_vertex[ii] * f_vertex[jj])
        if abs(f_bond - 1.0) > 1e-10:
            D0[bi] = bond_seeds[bi].D0 * f_bond

    # Store competition metrics for downstream C3 modulation
    topology._spectral_competition = vertex_competition

    # ── Step 4: build updated seeds ───────────────────────────────────
    updated_seeds: Dict[int, BondSeed] = {}
    for bi, seed in bond_seeds.items():
        if abs(D0[bi] - seed.D0) < 1e-12:
            updated_seeds[bi] = seed
            continue
        scale = D0[bi] / max(seed.D0, 1e-10)
        updated_seeds[bi] = BondSeed(
            D0=D0[bi],
            D0_P0=seed.D0_P0 * scale,
            D0_P1=seed.D0_P1 * scale,
            D0_P2=seed.D0_P2 * scale,
            D0_P3=seed.D0_P3 * scale,
            S0=seed.S0,
            S1=seed.S1,
            S2=seed.S2,
            S3=seed.S3,
            Q_eff=seed.Q_eff,
        )

    return updated_seeds


# ╔════════════════════════════════════════════════════════════════════╗
# ║  HÜCKEL π FOR AROMATIC RINGS                                      ║
# ╚════════════════════════════════════════════════════════════════════╝

def _huckel_aromatic(topology: Topology,
                      atom_data: dict) -> Dict[int, float]:
    """Hückel π cap per aromatic bond.

    For aromatic bonds (bo=1.5), the π energy comes from the
    Hückel eigenvalues of the ring Hamiltonian on Z/NZ.
    This REPLACES the pairwise π formula (GFT anti-double-counting).
    """
    huckel_caps: Dict[int, float] = {}
    aro_bonds = [(bi, i, j) for bi, (i, j, bo)
                 in enumerate(topology.bonds) if bo == 1.5]
    if not aro_bonds:
        return huckel_caps

    aro_adj: Dict[int, set] = defaultdict(set)
    for bi, i, j in aro_bonds:
        aro_adj[i].add((j, bi))
        aro_adj[j].add((i, bi))

    visited: set = set()
    for start in aro_adj:
        if start in visited:
            continue
        component: set = set()
        bond_indices: set = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in component:
                continue
            component.add(node)
            for nb, bi in aro_adj[node]:
                bond_indices.add(bi)
                if nb not in component:
                    queue.append(nb)
        visited.update(component)

        # Order ring atoms
        ordered = [start]
        current, prev = start, -1
        for _ in range(len(component) - 1):
            nbs = [(n, bi) for n, bi in aro_adj[current]
                   if n != prev and n in component]
            if not nbs:
                break
            nxt, _ = nbs[0]
            ordered.append(nxt)
            prev, current = current, nxt

        N_ring = len(ordered)
        if N_ring < 5:
            continue

        # Onsite ε and transfer β [T6 on Z/NZ]
        eps_ring = []
        for a in ordered:
            Z = topology.Z_list[a]
            ie, ea = atom_data[Z]['IE'], atom_data[Z]['EA']
            eps_ring.append(math.sqrt(max(ie * ea, 0.01)))

        H = np.zeros((N_ring, N_ring))
        for k in range(N_ring):
            H[k, k] = eps_ring[k]
            kn = (k + 1) % N_ring
            # β = S₃×(1+D₃): ring hops on molecular graph (P₁ face),
            # NLO 1-loop vertex correction D₃ on the hexagonal polygon.
            beta = S3 * (1.0 + D3) * math.sqrt(
                max(eps_ring[k] * eps_ring[kn], 0.0)) * D_FULL
            H[k, kn] = -beta
            H[kn, k] = -beta

        eigenvalues = sorted(np.linalg.eigvalsh(H))

        # Fill N_ring π-electrons
        E_deloc = 0.0
        placed = 0
        for ev in eigenvalues:
            if placed >= N_ring:
                break
            nf = min(2, N_ring - placed)
            E_deloc += nf * ev
            placed += nf

        alpha_avg = sum(eps_ring) / N_ring
        E_pi = abs(E_deloc - N_ring * alpha_avg)

        # Heteroatom defect attenuation
        eps_C = [e for e, a in zip(eps_ring, ordered)
                 if topology.Z_list[a] == 6]
        n_C = len(eps_C)
        if 0 < n_C < N_ring:
            eps_C_avg = sum(eps_C) / n_C
            betas = []
            for k in range(N_ring):
                kn = (k + 1) % N_ring
                betas.append(S3 * (1.0 + D3) * math.sqrt(
                    max(eps_ring[k] * eps_ring[kn], 0.0)) * D_FULL)
            f_defect = 1.0
            for k in range(N_ring):
                Z_k = topology.Z_list[ordered[k]]
                if Z_k != 6:
                    d_eps = abs(eps_ring[k] - eps_C_avg)
                    beta_k = (betas[(k - 1) % N_ring] + betas[k]) / 2
                    if beta_k > 0.01:
                        lp_k = topology.lp[ordered[k]]
                        if lp_k == 1 and Z_k == 7:
                            f_defect *= 1.0 - S3 * d_eps / (d_eps + 2.0 * beta_k)
                        else:
                            f_severe = max(0.0, min(1.0,
                                                    d_eps / (4.0 * beta_k) - 0.5))
                            f_lp_sp2 = 1.0 if lp_k == 1 else 0.0
                            power = 2.0 - f_severe * f_lp_sp2
                            f_defect *= (2 * beta_k / (d_eps + 2 * beta_k)) ** power
            E_pi *= f_defect

        # ── σ-π spectral overlap on Z/(2P₁)Z [GFT anti-double-counting] ──
        # The Hückel eigenvalues assume full hexagonal face bandwidth for
        # the π channel.  σ electrons fill k=0 (monopole) on Z/(2P₁)Z,
        # but this does NOT interfere with π (k≥1 nodal modes).  The
        # only σ-π overlap is at k=1 (dipole), with amplitude D₃ (the
        # holonomic correction on the first non-trivial mode of Z/(2P₁)Z).
        # Correction: reduce E_pi by (n_C_hex / N)² × D₃.
        # PT basis: CRT orthogonality — k=0 σ ⊥ k≥1 π.  Only the k=1
        # AC component of the σ spectrum (amplitude D₃) overlaps with π.
        if topology.rings:
            n_C_hex = 0
            for ring in topology.rings:
                ring_set = set(ring)
                if len(ring) == 6 and ring_set.issubset(component):
                    for a in ring:
                        if topology.Z_list[a] == 6:
                            n_C_hex += 1
            if n_C_hex > 0:
                f_sigma_pi = (float(n_C_hex) / N_ring) ** 2 * D3
                E_pi *= (1.0 - f_sigma_pi)

        cap_per_bond = E_pi / N_ring
        for bi in bond_indices:
            huckel_caps[bi] = cap_per_bond

    return huckel_caps


# ╔════════════════════════════════════════════════════════════════════╗
# ║  GLOBAL CAGE SPECTRUM                                             ║
# ╚════════════════════════════════════════════════════════════════════╝

def _is_full_spectrum_cage(topology: Topology) -> bool:
    """Detect closed sp2 carbon cages that require a global spectral law."""
    if topology.n_atoms < 20 or len(topology.bonds) < 30:
        return False
    if any(Z != 6 for Z in topology.Z_list):
        return False
    if any(z != 3 for z in (topology.z_count or [])):
        return False
    if any(abs(sum_bo - 4.0) > 1e-12 for sum_bo in (topology.sum_bo or [])):
        return False
    if any(lp != 0 for lp in (topology.lp or [])):
        return False
    if any(ch != 0 for ch in (topology.charges or [])):
        return False
    cyclomatic = len(topology.bonds) - topology.n_atoms + 1
    return cyclomatic >= 8


def _full_spectrum_cage_spectral_correction(
    topology: Topology,
    atom_data: dict,
) -> float:
    """Collective spectral correction for fullerene-like cages.

    These cages are not well described by summing local ring holonomies. Their
    pi network is a single closed spectral object, so we use a whole-graph
    Hückel-like correction instead of the local ring-by-ring subtraction.
    """
    heavy = [i for i, Z in enumerate(topology.Z_list) if Z > 1]
    if not heavy:
        return 0.0

    idx = {v: k for k, v in enumerate(heavy)}
    N = len(heavy)
    H = np.zeros((N, N))
    eps_ring: List[float] = []
    for v in heavy:
        Z = topology.Z_list[v]
        ie = atom_data[Z]["IE"]
        ea = atom_data[Z]["EA"]
        eps = math.sqrt(max(ie * ea, 0.01))
        eps_ring.append(eps)
        H[idx[v], idx[v]] = eps

    for i, j, _bo in topology.bonds:
        if i not in idx or j not in idx:
            continue
        beta = S5 * C5 * math.sqrt(max(H[idx[i], idx[i]] * H[idx[j], idx[j]], 0.0)) * D_FULL
        H[idx[i], idx[j]] = -beta
        H[idx[j], idx[i]] = -beta

    eigenvalues = np.sort(np.linalg.eigvalsh(H))
    n_pi = N
    placed = 0
    E_occ = 0.0
    for ev in eigenvalues:
        if placed >= n_pi:
            break
        nf = min(2, n_pi - placed)
        E_occ += nf * ev
        placed += nf

    alpha_avg = sum(eps_ring) / N
    E_pi = abs(E_occ - n_pi * alpha_avg)
    z_avg = sum(topology.z_count[v] for v in heavy) / N
    return -E_pi * (1.0 + z_avg * D3)


# ╔════════════════════════════════════════════════════════════════════╗
# ║  SPECTRAL VERTEX CORRECTIONS                                      ║
# ║                                                                    ║
# ║  Replaces v2's Star Hamiltonian + 172 if/elif gates.              ║
# ║  Uses eigenvalue analysis of the global transfer matrix.          ║
# ╚════════════════════════════════════════════════════════════════════╝

def _nnlo_spectral_curvature(
    T_P1: np.ndarray,
    topology: Topology,
    atom_data: dict,
    T_P1_nnlo: Optional[np.ndarray],
) -> float:
    """Second-order spectral curvature from one NLO-reweighted re-diagonalization."""
    if T_P1_nnlo is None or T_P1_nnlo.shape != T_P1.shape:
        return 0.0
    if not any(d.get("is_d_block", False) for d in atom_data.values()):
        return 0.0
    spec_lo = np.linalg.eigvalsh(T_P1)
    spec_nnlo = np.linalg.eigvalsh(T_P1_nnlo)
    return S3 * (
        float(np.sum(np.abs(spec_nnlo))) - float(np.sum(np.abs(spec_lo)))
    )


def _exact_ring_holonomy(
    T_P1: np.ndarray,
    ring: List[int],
    topology: Topology,
    atom_data: dict,
    bond_energies: Dict[int, float],
) -> float:
    """Exact cycle holonomy from the ordered ring transfer product."""
    N_ring = len(ring)
    if N_ring < 3:
        return 0.0

    ring_bi_list: List[int] = []
    is_aro = False
    for idx_r, a in enumerate(ring):
        b = ring[(idx_r + 1) % N_ring]
        bi_ring = None
        for bi, (i, j, bo) in enumerate(topology.bonds):
            if (i == a and j == b) or (i == b and j == a):
                bi_ring = bi
                if abs(bo - 1.5) < 0.01:
                    is_aro = True
                break
        if bi_ring is None:
            return 0.0
        ring_bi_list.append(bi_ring)
    if is_aro:
        return 0.0

    D_ring_sum = sum(bond_energies.get(bi, 0.0) for bi in ring_bi_list)
    if D_ring_sum <= 0.0:
        return 0.0

    T_cycle = np.eye(2)
    for idx_r, a in enumerate(ring):
        b = ring[(idx_r + 1) % N_ring]
        t_ab = abs(T_P1[a, b]) / RY
        eps_a = atom_data[topology.Z_list[a]]["IE"] / RY
        eps_b = atom_data[topology.Z_list[b]]["IE"] / RY
        T_local = np.array([
            [eps_a, -t_ab],
            [-t_ab, eps_b],
        ], dtype=float)
        T_cycle = T_cycle @ T_local

    T_ring = np.array(T_P1[np.ix_(ring, ring)], dtype=float) / RY
    eig_sum = float(np.sum(np.abs(np.linalg.eigvalsh(T_ring))))
    if eig_sum <= 1e-12:
        return 0.0

    cycle_trace = abs(float(np.trace(T_cycle)))
    holo_frac = min(1.0, abs(cycle_trace - eig_sum) / eig_sum)
    n_C_ring = sum(1 for k in ring if topology.Z_list[k] == 6)
    f_homo = (n_C_ring / N_ring) ** 2
    n_cyclomatic = max(len(topology.bonds) - topology.n_atoms + 1, 1)
    f_residual = (N_ring - 2.0) / N_ring
    return (
        -S3
        * math.sin(math.pi / N_ring) ** 2
        * D_ring_sum
        * f_homo
        * f_residual
        * holo_frac
        / n_cyclomatic
    )


def _spectral_correction(T_P1: np.ndarray, topology: Topology,
                          atom_data: dict,
                          bond_energies: Dict[int, float],
                          T_P1_nnlo: Optional[np.ndarray] = None) -> float:
    """Vertex corrections from the P₁ spectrum.

    Three sources:
      1. Ring strain (angular holonomy on Z/NZ)
      2. LP cooperation (structural rigidity from LP)
      3. Sub-Bohr vertex penalty
    """
    if _is_full_spectrum_cage(topology):
        return _full_spectrum_cage_spectral_correction(topology, atom_data)

    E_corr = _nnlo_spectral_curvature(T_P1, topology, atom_data, T_P1_nnlo)
    N = len(topology.Z_list)
    n_bonds = len(topology.bonds)

    # ── 1. Ring strain [T6 holonomy] ──
    if topology.rings:
        for ring in topology.rings:
            N_ring = len(ring)
            if N_ring < 3:
                continue
            theta_ring = (N_ring - 2) * 180.0 / N_ring

            # Aromatic flag for this ring
            is_aro = any(
                topology.bonds[bi][2] == 1.5
                for bi in range(n_bonds)
                if (topology.bonds[bi][0] in ring
                    and topology.bonds[bi][1] in ring))

            # σ-aromatic eligibility [mirrors C9b gate]:
            # rings of size 3-6 with only non-C non-H atoms at period ≥ 3
            # are σ-aromatic (delocalised on Z/(2P₁)Z). The angular
            # strain assumes sp-linear ideal (θ_pt = 180°) which is
            # physically wrong for these systems — their natural
            # geometry IS the ring angle, not a distortion away from it.
            ring_Zs = [topology.Z_list[k] for k in ring]
            is_sigma_aro_eligible = (
                3 <= N_ring <= 6
                and all(z not in (1, 6) for z in ring_Zs)
                and all(period(z) >= 3 for z in ring_Zs)
                and not is_aro
            )

            for k in ring:
                z_k = topology.z_count[k]
                if z_k < 2:
                    continue
                lp_k = topology.lp[k]
                z_eff = z_k + min(lp_k, 3) * (1.0 + S_HALF)
                if z_eff <= 1.0:
                    theta_pt = 180.0
                else:
                    theta_pt = math.degrees(math.acos(
                        max(-1.0, min(1.0, -1.0 / (z_eff - 1.0)))))

                bonds_at_k = topology.vertex_bonds.get(k, [])
                D_avg = (sum(bond_energies.get(bi, 0.0) for bi in bonds_at_k)
                         / max(len(bonds_at_k), 1))

                # Angular strain (skip for σ-aromatic eligible rings —
                # their ring geometry is natural, not strained)
                delta_theta = abs(theta_pt - theta_ring)
                if delta_theta > 1.0 and not is_sigma_aro_eligible:
                    delta_rad = math.radians(delta_theta)
                    E_corr -= -math.log(C3) * delta_rad ** 2 * D_avg

                # Vibrational holonomy for saturated rings ≥ 4
                if N_ring >= 4 and not is_aro and theta_pt < 179.0:
                    sin2_half = math.sin(math.radians(theta_pt) / 2.0) ** 2
                    E_corr -= S3 * sin2_half * D_avg * S_HALF / z_k

                # Ring torsion [T6, GFT: N angles sum to 0 mod 2π]
                if N_ring >= 4 and not is_aro:
                    ring_bond_set = topology.ring_bonds or set()
                    ring_bonds_k = [bi for bi in bonds_at_k
                                    if bi in ring_bond_set]
                    if ring_bonds_k:
                        D_ring = (sum(bond_energies.get(bi, 0.0)
                                      for bi in ring_bonds_k)
                                  / len(ring_bonds_k))
                        sin2_pi_N = math.sin(math.pi / N_ring) ** 2
                        E_torsion = S3 * sin2_pi_N * D_ring * S_HALF
                        if lp_k > 0:
                            E_torsion *= S3
                        E_corr -= E_torsion

            if not is_aro and any(
                atom_data[topology.Z_list[k]].get("is_d_block", False) for k in ring
            ):
                E_corr += _exact_ring_holonomy(
                    T_P1,
                    ring,
                    topology,
                    atom_data,
                    bond_energies,
                )

    # ── 2. Non-ring vertex corrections ──
    ring_atoms = topology.ring_atoms or set()
    for k in range(N):
        z_k = topology.z_count[k]
        if z_k < 2 or k in ring_atoms:
            continue

        Z_k = topology.Z_list[k]
        d_k = atom_data[Z_k]
        ie_k = d_k['IE']
        lp_k = topology.lp[k]
        is_bohr = ie_k >= RY * (1.0 - S3)

        bonds_at_k = topology.vertex_bonds.get(k, [])
        D_pair_k = sum(bond_energies.get(bi, 0.0) for bi in bonds_at_k)

        # LP cooperation now in _screening_P1 (not here)

        # Sub-Bohr penalty: vertex with IE < Ry and no LP
        if lp_k == 0 and z_k > 2 and not is_bohr:
            f_sub = max(0.0, (C3 - ie_k / RY) / S3)
            if f_sub > 0:
                E_corr -= D3 * (z_k / P1) * f_sub * D_pair_k / z_k

        # ── Star Hamiltonian vertex correction [D13 on K_{1,z}] ──
        l_k = d_k['l']
        ns_k = d_k.get('ns', 2)
        # Hybridized onsite for center
        eps_c = ie_k
        n_p = d_k['nf'] if l_k == 1 else 0
        if n_p > 0 and ie_k < RY:
            valence = d_k['nf'] + ns_k
            af = min(valence, P1 + 1) / (P1 + 1)
            eps_c = ie_k + af * C3 * (RY - ie_k)
        n_sh = max(0, d_k['per'] - P1 + 1)
        if n_sh > 0 and eps_c > ie_k + 0.01:
            eps_c *= C3 ** (n_sh * S_HALF)
        else:
            eps_c *= C3 ** n_sh

        n_h = len(bonds_at_k) + 1
        H_star = np.zeros((n_h, n_h))
        H_star[0, 0] = eps_c
        D_on_v = 0.0
        has_lp_lig = False
        for idx_h, bi_h in enumerate(bonds_at_k):
            ii_h, jj_h, bo_h = topology.bonds[bi_h]
            partner_h = jj_h if ii_h == k else ii_h
            Z_p = topology.Z_list[partner_h]
            d_p = atom_data[Z_p]
            # Partner onsite (hybridized)
            eps_p = d_p['IE']
            n_p_p = d_p['nf'] if d_p['l'] == 1 else 0
            if n_p_p > 0 and eps_p < RY:
                val_p = d_p['nf'] + d_p.get('ns', 2)
                af_p = min(val_p, P1 + 1) / (P1 + 1)
                eps_p = eps_p + af_p * C3 * (RY - eps_p)
            z_p = topology.z_count[partner_h]
            bo_w = S3 * max(bo_h - 1, 0) / P1 + S3 * S3 / P1
            if z_p > 1 and eps_p >= (RY - S3 * RY):
                eps_p /= (1.0 + (z_p - 1) * bo_w)
            H_star[idx_h + 1, idx_h + 1] = eps_p
            t_val = S3 * C3 * math.sqrt(max(eps_c * eps_p, 0.01))
            H_star[0, idx_h + 1] = t_val
            H_star[idx_h + 1, 0] = t_val
            D_on_v += bond_energies.get(bi_h, 0.0)
            lp_lig = topology.lp[partner_h]
            if lp_lig > 0:
                has_lp_lig = True

        if D_on_v > 0 and n_h >= 3:
            eigs = np.linalg.eigvalsh(H_star)
            eig_min, eig_max = eigs[0], eigs[-1]
            D_star = abs(eig_max - eig_min)
            D_pair_sum = 2.0 * sum(
                H_star[0, idx_h + 1] for idx_h in range(n_h - 1))
            R_k = min(1.0, D_star / max(D_pair_sum, 0.01))
            residual = max(0.0, 1.0 - R_k)
            # Excess = pairwise sum beyond collective spectrum
            excess = max(0.0, D_pair_sum - D_star)
            if residual > 0.001 and excess > 0.01:
                # ── Vertex diversity weighting [T1 Dicke on molecular graph] ──
                partner_Z_set = set()
                for bi_h in bonds_at_k:
                    ii_h2, jj_h2, _ = topology.bonds[bi_h]
                    p_h2 = jj_h2 if ii_h2 == k else ii_h2
                    partner_Z_set.add(topology.Z_list[p_h2])
                n_distinct = len(partner_Z_set)
                f_diversity = (n_distinct - 1.0) / max(z_k - 1.0, 1.0)

                if is_bohr or lp_k > 0:
                    # Donor: competition → reduce by S₃ × excess / z
                    E_corr -= S3 * excess / z_k
                    # ── H-partner vertex relief ──────────────────────
                    if d_k['per'] < P1 and z_k >= P1:
                        n_H_p = sum(1 for bi_h3 in bonds_at_k
                                    for ii3, jj3, _ in [topology.bonds[bi_h3]]
                                    if topology.Z_list[
                                        jj3 if ii3 == k else ii3] == 1)
                        f_H = float(n_H_p) / max(z_k, 1)
                        if f_H > 0:
                            E_corr += D3 * excess / z_k * f_H * (1.0 - f_diversity)
                elif has_lp_lig and not is_bohr:
                    # Acceptor with LP-ligands: cooperation bonus (full, no diversity gate)
                    E_corr += D3 * S_HALF * excess / z_k

    return E_corr


# ╔════════════════════════════════════════════════════════════════════╗
# ║  DIM 3 : SHELL DENSITY ATTENUATION                                ║
# ║                                                                    ║
# ║  Bonds beyond P_l at a vertex go through a SPARSER circle:        ║
# ║    first P_l bonds → Z/(2P_l)Z (dense, full energy)               ║
# ║    excess bonds    → Z/(2P_{l+1})Z (sparser, energy × P_l/P_{l+1})║
# ╚════════════════════════════════════════════════════════════════════╝

def _apply_shell_attenuation(topology: Topology, atom_data: dict,
                              bond_energies: Dict[int, float]) -> None:
    """Shell density attenuation for hypervalent vertices (dim 3, in-place).

    When z > P_l at a vertex, the excess bonds use a sparser circle:
      - p-block (l=1): first P₁=3 bonds on Z/6Z, excess on Z/10Z → ×P₁/P₂ = 3/5
      - d-block (l=2): first P₂=5 bonds on Z/10Z, excess on Z/14Z → ×P₂/P₃ = 5/7

    The attenuation factor per excess bond = P_l / P_{l+1}.
    """
    shell_cap = _BLOCK_P  # capacity per block
    shell_ratio = {1: P1 / P2, 2: P2 / P3, 3: 1.0}  # attenuation for excess

    for v in range(len(topology.Z_list)):
        z_v = topology.z_count[v]
        Z_v = topology.Z_list[v]
        d_v = atom_data[Z_v]
        l_v = d_v['l']
        per_v = d_v['per']
        # Only attenuate atoms that HAVE d-orbitals (per ≥ P₁ = 3).
        # Period-2 atoms (C, N, O) use sp³ hybridization, not d-promotion.
        if l_v < 1 or per_v < P1 or z_v <= shell_cap.get(l_v, P1):
            continue  # no excess bonds or no d-orbitals available

        P_l = shell_cap.get(l_v, P1)
        ratio = shell_ratio.get(l_v, P1 / P2)
        n_excess = z_v - P_l

        # ── d-vacancy bridge for per-3 p-block [PT: pentagonal face capacity] ──
        # For per≥3 p-block atoms (l=1, not true d-block), the d-face
        # Z/(2P₂)Z has (2P₂ - nf)/(2P₂) vacancy. This vacancy provides
        # partial capacity for the excess bond through the d-orbital bridge.
        # The bridge couples at half-strength (S_HALF) because it's an
        # indirect pathway (σ → d back-donation → σ), not direct occupancy.
        #
        # Gate: the bridge requires the bond's partner to have l ≥ 1
        # (p-orbital capacity for back-donation). H (l=0) cannot couple
        # into the d-face. This is the PT selection rule: the d-orbital
        # bridge requires angular momentum exchange l → l+1.
        d_vacancy_bridge = 0.0
        if l_v == 1 and per_v >= P1 and not d_v.get('is_d_block', False):
            nf_v = d_v.get('nf', 0)
            d_vacancy_bridge = max(0.0, (2.0 * P2 - nf_v)) / (2.0 * P2)

        # Get bonds at this vertex, sorted by energy (weakest attenuated first)
        v_bonds = topology.vertex_bonds.get(v, [])
        if not v_bonds:
            continue
        v_bonds_sorted = sorted(v_bonds, key=lambda bi: bond_energies.get(bi, 0.0))

        # Attenuate the n_excess weakest bonds, with per-bond d-vacancy modulation
        for k in range(min(n_excess, len(v_bonds_sorted))):
            bi = v_bonds_sorted[k]
            ratio_eff = ratio
            if d_vacancy_bridge > 0:
                ii, jj, _ = topology.bonds[bi]
                partner = jj if ii == v else ii
                Z_partner = topology.Z_list[partner]
                l_partner = atom_data[Z_partner].get('l', 0)
                per_partner = atom_data[Z_partner].get('per', 1)
                # Gate 1: partner has p-orbitals (l≥1) for back-donation
                # Gate 2: not homonuclear (T1: no net transfer by symmetry)
                # Gate 3: partner in different period (cross-period donation)
                if (l_partner >= 1
                        and Z_partner != Z_v
                        and per_partner != per_v):
                    ratio_eff = ratio + (1.0 - ratio) * d_vacancy_bridge * S_HALF
            reduction = bond_energies[bi] * (1.0 - ratio_eff)
            bond_energies[bi] -= reduction


# ╔════════════════════════════════════════════════════════════════════╗
# ║  SECTION 8: T⁴ MATRIX BUILDER + SCF LOOP                          ║
# ╚════════════════════════════════════════════════════════════════════╝

def _build_T4(topology: Topology, atom_data: dict,
              bond_seeds: Dict[int, 'BondSeed']) -> np.ndarray:
    """Build n×n transfer matrix on T⁴.

    Off-diagonal: bond seed energy × inter-circle repulsion factor.
    Diagonal: IE/Ry (onsite energy).

    The inter-circle repulsion exp(-S_cross) captures the REPULSION
    between electrons on different circles S¹_P. When multiple circles
    have high screening simultaneously (overcrowded vertex), the
    cross-face products S_P×S_Q are large → exp(-S_cross) << 1 →
    bond energy reduced. This is the signed kinetic coupling between
    circles that the per-face additive model misses.
    """
    N = len(topology.Z_list)
    T = np.zeros((N, N))

    for k in range(N):
        Zk = topology.Z_list[k]
        T[k, k] = atom_data[Zk]['IE'] / RY

    for bi, (ii, jj, bo) in enumerate(topology.bonds):
        seed = bond_seeds[bi]
        # Inter-circle repulsion: cross-products of screenings from
        # different faces. High S on multiple faces = crowded = repulsion.
        # Cross-face repulsion: NLO+NNLO Fisher cosines [P3a].
        # COS_35 > 0: P₁⊗P₂ aligned → repulsion.
        # COS_37 < 0, COS_57 < 0: anti-correlated → stabilisation.
        # Net S_cross ≥ 0 (floored: screening is non-negative).
        S_cross = max(0.0,
                      CROSS_35 * seed.S1 * seed.S2
                      - CROSS_37 * seed.S1 * seed.S3
                      - CROSS_57 * seed.S2 * seed.S3
                      + AEM * seed.S1 * seed.S2 * seed.S3)
        t_ij = seed.D0 * math.exp(-S_cross)
        T[ii, jj] = t_ij
        T[jj, ii] = t_ij

    return T


_SCF_MAX_ITER = 5
_SCF_ATOL = 1e-6


def _scf_iterate(topology: Topology, atom_data: dict,
                 bond_seeds: Dict[int, BondSeed]) -> Tuple[Dict[int, float], np.ndarray]:
    """SCF on T⁴: iterate Perron eigenvector until convergence.

    The T⁴ matrix includes inter-circle repulsion via exp(-S_cross).
    The SCF redistributes bond energies via the Perron eigenvector:
    vertices with high Perron weight attract energy, low weight repel.

    Returns (bond_energies, eigenvalues) after convergence.
    """
    N = len(topology.Z_list)
    n_bonds = len(topology.bonds)

    # Initial bond energies include cross-face repulsion from T⁴
    bond_energies = {}
    for bi in range(n_bonds):
        seed = bond_seeds[bi]
        # Cross-face repulsion: NLO+NNLO Fisher cosines [P3a].
        # COS_35 > 0: P₁⊗P₂ aligned → repulsion.
        # COS_37 < 0, COS_57 < 0: anti-correlated → stabilisation.
        # Net S_cross ≥ 0 (floored: screening is non-negative).
        S_cross = max(0.0,
                      CROSS_35 * seed.S1 * seed.S2
                      - CROSS_37 * seed.S1 * seed.S3
                      - CROSS_57 * seed.S2 * seed.S3
                      + AEM * seed.S1 * seed.S2 * seed.S3)
        bond_energies[bi] = seed.D0 * math.exp(-S_cross)

    if N < 3:
        return bond_energies, np.array([])

    T = _build_T4(topology, atom_data, bond_seeds)

    for iteration in range(_SCF_MAX_ITER):
        eigenvalues, eigenvectors = np.linalg.eigh(T)

        v_perron = np.abs(eigenvectors[:, -1])
        v_sum = v_perron.sum()
        if v_sum > 0:
            v_perron = v_perron / v_sum
        else:
            v_perron = np.ones(N) / N

        # Perron redistribution: δw > 0 for heavy vertices → less screening
        new_bond_energies = {}
        for bi, (ii, jj, bo) in enumerate(topology.bonds):
            seed = bond_seeds[bi]
            # Cross-face repulsion: NLO+NNLO Fisher cosines [P3a].
            S_cross = max(0.0,
                          CROSS_35 * seed.S1 * seed.S2
                          - CROSS_37 * seed.S1 * seed.S3
                          - CROSS_57 * seed.S2 * seed.S3
                          + AEM * seed.S1 * seed.S2 * seed.S3)
            delta_w = (v_perron[ii] + v_perron[jj]) - 2.0 / N
            S_perron = D3 * delta_w
            new_bond_energies[bi] = seed.D0 * math.exp(-S_cross - S_perron)

        max_delta = max(abs(new_bond_energies[bi] - bond_energies[bi])
                        for bi in range(n_bonds))
        bond_energies = new_bond_energies

        if max_delta < _SCF_ATOL:
            break

        # Rebuild T⁴ with updated seeds (D0 from Perron-adjusted energies)
        updated_seeds = {}
        for bi in range(n_bonds):
            old = bond_seeds[bi]
            updated_seeds[bi] = BondSeed(
                D0=bond_energies[bi], D0_P0=old.D0_P0, D0_P1=old.D0_P1,
                D0_P2=old.D0_P2, D0_P3=old.D0_P3,
                S0=old.S0, S1=old.S1, S2=old.S2, S3=old.S3, Q_eff=old.Q_eff,
            )
        T = _build_T4(topology, atom_data, updated_seeds)

    eigenvalues = np.linalg.eigvalsh(T)
    return bond_energies, eigenvalues


# ╔════════════════════════════════════════════════════════════════════╗
# ║  SECTION 9: GHOST ATTENUATION (Mertens tail p≥11)                  ║
# ╚════════════════════════════════════════════════════════════════════╝

def _ghost_attenuation(topology: Topology, atom_data: dict) -> float:
    """Mertens tail from inactive primes p>=11."""
    return (1.0 - BETA_GHOST / 11.0) * (1.0 - BETA_GHOST / 13.0)


# ╔════════════════════════════════════════════════════════════════════╗
# ║  TRIATOMIC FAST-PATH: 3×3 + v4 seeds                              ║
# ║                                                                    ║
# ║  Triatomic A-B-C = 2 bonds sharing a central vertex B.            ║
# ║  Architecture:                                                     ║
# ║    1. v4 per-bond seeds (diatomic-quality D₀ for each bond)       ║
# ║    2. 3×3 eigenvalue renormalization (σ sharing at center)         ║
# ║    3. Face corrections (vacancy, Hückel 3c, LP cooperation)       ║
# ║    4. Per-type routing (linear/bent/radical)                       ║
# ║    5. Ghost attenuation + assembly → TransferResult               ║
# ║                                                                    ║
# ║  0 adjustable parameters. All from s = 1/2.                       ║
# ╚════════════════════════════════════════════════════════════════════╝

def _triatomic_fast_path(topology: Topology,
                         atom_data: dict) -> Optional[TransferResult]:
    """Dedicated triatomic engine: v4 seeds + 3×3 coordination correction.

    Returns TransferResult or None if not a valid triatomic (A-B-C).

    The triatomic regime sits between the exact diatomic 2×2 limit and
    the polyatomic vertex-polygon SCF. The poly engine's vertex DFT
    shares a single polygon between 2 bonds at the central vertex,
    creating ±15-25% seed bias. This engine bypasses the poly pipeline.

    Strategy:
      - v4 seeds give diatomic-quality per-bond D₀ (baseline)
      - 3×3 Hamiltonian provides the coordination correction delta
      - Face corrections add vacancy, Hückel 3c, LP cooperation
      - Per-type routing handles bent/linear/radical/ionic specifics

    PT architecture:
      dim 1 (P₁=3): v4 per-bond D₀ (σ+π+ionic, 10 corrections)
      dim 2 (P₂=5): coordination + face corrections on Z/(2P₂)Z
      dim 0 (P₀=2): ghost VP attenuation (Mertens p≥11)

    0 adjustable parameters.
    """
    if topology.n_atoms != 3 or len(topology.bonds) != 2:
        return None

    # ── Identify roles: central B (z=2), terminals A and C ──
    zc = topology.z_count
    ctr = max(range(3), key=lambda k: zc[k])
    terms = [k for k in range(3) if k != ctr]
    A, B, C = terms[0], ctr, terms[1]

    Z_B = topology.Z_list[B]
    d_B = atom_data[Z_B]
    lp_B = topology.lp[B]
    per_B = d_B['per']
    l_B = d_B.get('l', 0)
    ie_B = d_B['IE']

    # ── Match bonds to A-B and B-C ──
    bond_AB = bond_BC = None
    for bi, (i, j, bo) in enumerate(topology.bonds):
        s = frozenset((i, j))
        if A in s and B in s:
            bond_AB = (i, j, bo, bi)
        elif B in s and C in s:
            bond_BC = (i, j, bo, bi)
    if bond_AB is None or bond_BC is None:
        return None

    # ── Detect molecule type ──
    total_ve = sum(valence_electrons(Z) for Z in topology.Z_list)
    is_radical = (total_ve % 2 == 1)
    bo_AB, bo_BC = bond_AB[2], bond_BC[2]
    n_pi_total = max(0, bo_AB - 1) + max(0, bo_BC - 1)
    bo_span = abs(bo_AB - bo_BC)

    d_A = atom_data[topology.Z_list[A]]
    d_C = atom_data[topology.Z_list[C]]

    # Linearity: center has no LP, both bonds are multi-bond (π available),
    # and molecule is NOT a radical (radicals like NO2 are bent).
    is_linear = (lp_B == 0 and n_pi_total >= 2 and zc[B] == 2 and not is_radical)
    # s-block acceptor linear: Be, Mg centers (no LP, no π, but linear)
    is_sblock_linear = (lp_B == 0 and l_B == 0 and zc[B] == 2 and per_B >= 2)

    # ── v4 per-bond seeds (diatomic quality) ──
    def _v4_seed(ia, ib, bo):
        Zi, Zj = topology.Z_list[ia], topology.Z_list[ib]
        lp_i, lp_j = topology.lp[ia], topology.lp[ib]
        return _D0_screening_v4(Zi, Zj, bo, lp_A=lp_i, lp_B=lp_j)

    sr_AB = _v4_seed(bond_AB[0], bond_AB[1], bo_AB)
    sr_BC = _v4_seed(bond_BC[0], bond_BC[1], bo_BC)

    # Start from v4 seeds as baseline
    D0_AB = sr_AB.D0
    D0_BC = sr_BC.D0

    # ── 3×3 Hamiltonian for coordination diagnostic ──
    # Used to compute the eigenvalue gap ratio f_renorm, which tells us
    # how much coordination sharing occurs at the central vertex.
    eps_A = d_A['IE'] / RY
    eps_B_h = ie_B / RY
    eps_C = d_C['IE'] / RY

    def _hopping(ia, ib):
        di_h, dj_h = atom_data[topology.Z_list[ia]], atom_data[topology.Z_list[ib]]
        ea_h, eb_h = di_h['IE'] / RY, dj_h['IE'] / RY
        eg_h = math.sqrt(ea_h * eb_h)
        pm = max(di_h['per'], dj_h['per'])
        lm = min(topology.lp[ia], topology.lp[ib])
        fp = (2.0 / pm) ** (1.0 / (P1 * P1) if pm == P1 else 1.0 / P1) if pm > 2 else 1.0
        fl = max(0.01, 1.0 - lm / (2.0 * max(pm, 1)))
        return S_HALF * eg_h * fl * fp

    t_AB = _hopping(A, B)
    t_BC = _hopping(B, C)
    t_sum = t_AB + t_BC

    # Cross coupling (bent only, LP-mediated)
    eff_lp = lp_B + (S_HALF if is_radical else 0.0)
    if eff_lp > 0 and not is_linear and not is_sblock_linear:
        t_AC = S_HALF * math.sqrt(eps_A * eps_C) * eff_lp / (P2 * max(per_B, 2))
    else:
        t_AC = 0.0

    # Per-3 bilateral through-center coupling.
    # When atoms have per >= P₁, their diffuse p-orbitals create bilateral
    # overlap through the center that the independent-hop Hamiltonian misses.
    # The coupling amplitude is D₅ × √(ε_A ε_C) × (per_max−2)/P₁,
    # using the maximum period in the triatomic.
    # This raises f_renorm for per-3 triatomics, correcting the systematic
    # underestimation of bilateral cooperation for diffuse orbitals.
    # 0 adjustable parameters.
    _per_max_tri = max(d_A['per'], per_B, d_C['per'])
    if _per_max_tri >= P1:
        _per3_wt = float(_per_max_tri - 2) / P1
        t_AC += D5 * math.sqrt(eps_A * eps_C) * _per3_wt

    H = np.array([
        [eps_A, -t_AB, -t_AC],
        [-t_AB, eps_B_h, -t_BC],
        [-t_AC, -t_BC, eps_C]
    ])
    eigvals = np.sort(np.linalg.eigvalsh(H))
    gap_bonding = eigvals[1] - eigvals[0]

    # f_renorm: how much the two bonds compete for spectral budget at center.
    # f_renorm < 1: competition reduces each bond. f_renorm > 1: cooperation.
    f_renorm = gap_bonding / t_sum if t_sum > 0 else 1.0
    f_renorm = max(0.5, min(1.2, f_renorm))

    # ── Per-3 bilateral overlap boost (BENT only) ──
    # For per_B ≥ P₁ bent centers, the diffuse valence shell creates
    # bilateral overlap that simultaneously stabilises both bonds.
    # The boost is proportional to:
    #   (per_B−2)/P₁ :  diffusivity excess beyond per=2
    #   S₃           :  hexagonal face coupling
    #   (½ + ⟨lp_t⟩ × D₃) : base ½ (orbital overlap) + terminal LP donation
    # Applied multiplicatively to v4 seeds before per-type corrections.
    # 0 adjustable parameters.
    if per_B >= P1 and l_B <= 1 and not is_linear and not is_sblock_linear:
        _lp_term_avg = sum(float(topology.lp[t]) for t in terms) / max(zc[B], 1)
        _f_bilat = S3 * float(per_B - 2) / P1 * (S_HALF + _lp_term_avg * D3)
        # Asymmetric terminals: bilateral cooperation is reduced when
        # terminal species differ (different IE → broken bilateral symmetry).
        _Z_A_b, _Z_C_b = topology.Z_list[A], topology.Z_list[C]
        if _Z_A_b != _Z_C_b:
            _ie_A_b, _ie_C_b = d_A['IE'], d_C['IE']
            _q_asym = min(_ie_A_b, _ie_C_b) / max(_ie_A_b, _ie_C_b)
            _f_bilat *= _q_asym  # attenuate by IE symmetry ratio
        D0_AB *= (1.0 + _f_bilat)
        D0_BC *= (1.0 + _f_bilat)

    # ── Face corrections (dim 2, cap = Ry/P₂) ──
    cap_face = RY / P2
    D_face = 0.0

    # F1: Vacancy cooperation (electron-deficient center + LP terminals)
    # For d-block (l_B=2), use the pentagonal face (P₂) instead of hexagonal (P₁):
    # the d-shell has 2P₂=10 slots, so vacancy = max(0, 2*P₂ - nd).
    # The vacancy fraction nd_vac/(2P₂) modulates the LP donation amplitude.
    nf_B = d_B.get('nf', 0)
    if l_B == 2 and not is_linear:
        # Fix A: d-block BENT uses pentagonal face P₂
        # Linear d-block (TiO2) uses pi-sharing in the LINEAR branch instead.
        nd_B = nf_B
        p2_vac_B = max(0, 2 * P2 - nd_B)
        cap_f1 = RY / P2
        if p2_vac_B > 0 and nd_B < 2 * P2:
            f_vac_frac = p2_vac_B / (2.0 * P2)   # vacancy fraction of d-shell
            for t in terms:
                lp_t = topology.lp[t]
                if lp_t > 0:
                    d_t = atom_data[topology.Z_list[t]]
                    per_t = d_t['per']
                    ie_t = d_t['IE']
                    f_donate = min(lp_t, p2_vac_B) * S_HALF / (2.0 * P2)
                    f_bohr = math.sin(math.pi * min(ie_t, RY) / (2 * RY)) ** 2
                    f_per = 2.0 / max(per_t, 2)
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
                    per_t = d_t['per']
                    ie_t = d_t['IE']
                    f_donate = min(lp_t, p1_vac_B) * S_HALF / P2
                    f_bohr = math.sin(math.pi * min(ie_t, RY) / (2 * RY)) ** 2
                    f_per = 2.0 / max(per_t, 2)
                    D_face += f_donate * f_bohr * f_per * cap_face * D_FULL

    # F2: LP cooperation at center (BENT, holonomic amplitude)
    # Precompute VSEPR angle for LP cooperation (used by F2 and F3).
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

    # F3: LP directional Coulomb stabilization (MC4 port)
    # When center has LP ≥ z and ALL partners have no LP (hydrides),
    # and center is compact (per=2), the LP creates directional 1/r
    # Coulomb stabilization. For per≥3, LP is too diffuse.
    # Gate: all terminals lp=0 + compact center (per ≤ 2).
    #
    # Anti-double-counting (v4.1): the diatomic seed S_holo already
    # captures intra-bond LP holonomy. F3 should only add the
    # inter-bond cooperative Coulomb stabilization — the angular
    # projection sin²(θ/2) = (1 − cos θ)/2 of the LP field onto
    # the complementary bond axis. This is exact on the VSEPR
    # polygon: the LP Coulomb kernel decomposes into an intra-bond
    # radial term (captured by S_holo) and a cooperative angular
    # term proportional to sin²(θ/2). 0 new parameters.
    if lp_B >= zc[B] and per_B <= 2 and not is_linear and not is_sblock_linear:
        all_term_no_lp = all(topology.lp[t] == 0 for t in terms)
        if all_term_no_lp:
            # Average bond distance
            r_sum, r_count = 0.0, 0
            for (ia, ib, bo_f, _) in [bond_AB, bond_BC]:
                di_f = atom_data[topology.Z_list[ia]]
                dj_f = atom_data[topology.Z_list[ib]]
                r_b = r_equilibrium(di_f['per'], dj_f['per'], bo_f,
                                    0.0, zc[ia], zc[ib],
                                    topology.lp[ia], topology.lp[ib])
                if r_b > 0:
                    r_sum += r_b
                    r_count += 1
            if r_count > 0:
                r_avg = r_sum / r_count
                f_comp_lp = 1.0 if ie_B >= RY - S3 else 2.0 / 3.0
                D_F3_raw = lp_B * S3 * S_HALF * COULOMB_EV_A / (zc[B] * r_avg) * f_comp_lp
                # Anti-double-counting: project onto cooperative channel
                if _cos_theta_lp is not None:
                    f_coop = (1.0 - _cos_theta_lp) / 2.0  # sin²(θ/2)
                else:
                    f_coop = 1.0  # fallback: no angle available
                D_face += D_F3_raw * f_coop

    # ── F4: d-block Madelung bilateral cooperation (Fix B) ──
    # M²⁺ flanked by 2 X⁻ (halides/chalcogenides with EA > S₃×Ry) creates
    # bilateral Coulomb stabilization. Same physics as the s-block ionic
    # bilateral boost (S2): LP donation from two anions stabilizes the
    # central cation through the hexagonal face (σ channel).
    # Structure: lp_t × S₃/P₁ × cap_P2 × f_ie × f_per × f_ie2 × f_vac × D_FULL
    # f_ie2 = Ry/IE₂ screens by ionization difficulty.
    # f_vac = vacancy fraction: more vacancies → more ionic character.
    #   For nd ≤ P₂ (early-d): f_vac = 1 (maximum ionic)
    #   For P₂ < nd < 2P₂ (late-d): f_vac = (2P₂-nd)/P₂ (reduced ionic)
    #   For nd = 2P₂ (full d, e.g. Zn): f_vac = S_HALF (residual ionic via s-orbital)
    # Gate: d-block center (bent) + both terminals have EA > S₃×Ry.
    if l_B == 2 and not is_linear:
        ea_terms_f4 = [atom_data[topology.Z_list[t]]['EA'] for t in terms]
        ea_gate = S3 * RY
        if all(ea > ea_gate for ea in ea_terms_f4):
            ie2_d = IE2_DBLOCK.get(Z_B, 0.0)
            nd_B_f4 = d_B.get('nf', 0)
            if ie2_d > 0:
                f_ie2 = RY / ie2_d
                # Vacancy-dependent ionic character
                # For nd ≤ P₂ (early-d): full Madelung (f_vac = 1)
                # For P₂ < nd < 2P₂ (late-d): linear decay (2P₂-nd)/P₂
                # For nd = 2P₂ (full d): d10s2 (ns ≥ 2) uses ns/P₁
                #   (s-electrons mediate ionic bonding directly);
                #   d10s1 uses S_HALF (weaker s-mediation).
                if nd_B_f4 <= P2:
                    f_vac_ionic = 1.0
                elif nd_B_f4 < 2 * P2:
                    f_vac_ionic = float(2 * P2 - nd_B_f4) / P2
                else:
                    _ns_B_f4 = ns_config(Z_B)
                    if _ns_B_f4 >= 2:
                        f_vac_ionic = float(_ns_B_f4) / P1
                    else:
                        f_vac_ionic = S_HALF
                for t in terms:
                    lp_t = topology.lp[t]
                    if lp_t > 0:
                        d_t = atom_data[topology.Z_list[t]]
                        ie_t = d_t['IE']
                        per_t = d_t['per']
                        f_ie = min(ie_t, RY) / RY
                        f_per = 1.0 + S_HALF * per_t / P1
                        D_face += lp_t * S3 / P1 * cap_face * f_ie * f_per * f_ie2 * f_vac_ionic * D_FULL

    # ── F4b: d-block bilateral Madelung cooperation ──
    # For late-d and d10s2 centres flanked by two anions, bilateral
    # Coulomb stabilisation exceeds 2× the monolateral F4 value.
    # The cooperative Madelung constant for XMX > 2× the diatomic MX.
    #
    # Two regimes:
    # (a) d10s2 (Zn, Cd, Hg): d-shell inert, ns=2 mediates.
    #     All LP terminals contribute (diffuse per ≥ P₁ only, as the
    #     compact terminals are already well-handled by F4).
    # (b) late-d (P₂ < nd < 2P₂, e.g. Ni d⁸): d-vacancies participate.
    #     Compact terminals ONLY (per < P₁, e.g. F). Diffuse terminals
    #     (Cl, Br) are already boosted by F4's bilateral term — adding
    #     F4b would double-count.  The compact terminals (F) are UNDER-
    #     served by F4 (f_vac < 1 for late-d + compact LP donation
    #     already captured at tree-level).
    #     f_d = f_vac × ns/P₁.
    #
    # Gate: d-block bent centre + both terminals EA > S₃×Ry.
    # 0 adjustable parameters.
    if l_B == 2 and not is_linear:
        nd_B_f4b = d_B.get('nf', 0)
        _ns_B_f4b = ns_config(Z_B)
        _is_d10s2 = (nd_B_f4b >= 2 * P2 and _ns_B_f4b >= 2)
        _is_late_d = (nd_B_f4b > P2 and nd_B_f4b < 2 * P2)
        if _is_d10s2 or _is_late_d:
            ea_terms_f4b = [atom_data[topology.Z_list[t]]['EA'] for t in terms]
            ea_gate_f4b = S3 * RY
            if all(ea > ea_gate_f4b for ea in ea_terms_f4b):
                ie2_d_f4b = IE2_DBLOCK.get(Z_B, 0.0)
                if ie2_d_f4b > 0:
                    if _is_d10s2:
                        _f_d_f4b = float(_ns_B_f4b) / P1
                    else:
                        _f_vac_f4b = float(2 * P2 - nd_B_f4b) / P2
                        _f_d_f4b = _f_vac_f4b * float(_ns_B_f4b) / P1
                    for t in terms:
                        lp_t = topology.lp[t]
                        if lp_t > 0:
                            d_t = atom_data[topology.Z_list[t]]
                            per_t = d_t['per']
                            # Regime (a) d10s2: diffuse terminals only
                            # Regime (b) late-d: compact terminals only
                            if _is_d10s2 and per_t < P1:
                                continue
                            if _is_late_d and per_t >= P1:
                                continue
                            f_per_f4b = (float(max(per_t, 2)) / 2.0) ** (1.0 / P1)
                            D_face += (float(lp_t) * _f_d_f4b * S3 / P1
                                       * cap_face * RY / ie2_d_f4b
                                       * f_per_f4b * D_FULL)

    # ── F5: Late-d coordination screening (Fix C) ──
    # For nd > P₂ (late-d) in bent triatomics, 2 ligands compete for the
    # same partially-filled d-orbitals. The screening scales with the
    # excess filling beyond half-fill:
    #   S_late = S₃ × (nd - P₂) / (2P₂) × f_per_avg
    #
    # Terminal modulation: compact terminals (per < P₁, e.g. F) have
    # localised orbitals that don't overlap on the pentagonal face →
    # d-orbital competition is reduced by (per/P₁).
    # Diffuse terminals (per ≥ P₁, e.g. Cl) have full competition.
    #
    # f_per_avg = min(1, per_avg / P₁)
    #   per_avg = 2 (F): f = 2/3 (reduced competition)
    #   per_avg = 3 (Cl): f = 1   (full competition)
    #
    # Gate: P₂ < nd < 2P₂ (late-d, partially filled).
    # 0 adjustable parameters.
    if l_B == 2 and not is_linear:
        nd_B_c = d_B.get('nf', 0)
        if nd_B_c > P2 and nd_B_c < 2 * P2:
            _per_avg_f5 = sum(atom_data[topology.Z_list[t]]['per']
                              for t in terms) / max(zc[B], 1)
            _f_per_f5 = min(1.0, _per_avg_f5 / P1)
            S_late = S3 * float(nd_B_c - P2) / (2.0 * P2) * _f_per_f5
            atten_blend = math.exp(-S_late)
            D0_AB *= atten_blend
            D0_BC *= atten_blend

    # ── F5b: d10s1 s-orbital competition screening ──
    # For NS1 d-block centers (Cu, Ag, Au: d10s1 promoted), the d-shell
    # is fully paired and inert. Covalent bonding uses the single ns
    # electron on Z/(2P₀)Z. With z=2 bonds sharing 1 s-electron,
    # each bond is attenuated by inter-bond competition. The competition
    # is STRONGER for diffuse terminals (per > 2) because larger
    # p-orbitals create more overlap at the center.
    # Amplitude: S₃ × s × Δper, where Δper = max(per_avg − 2, 0).
    #   - S₃: coupling through the hexagonal face Z/(2P₁)Z
    #   - s = S_HALF: s-orbital factor (only 1 electron)
    #   - Δper: terminal diffusivity excess beyond compact per=2 core
    # Gate: d-block center with Z in NS1 (d10s1 promoted), bent triatomic.
    # 0 adjustable parameters.
    _NS1_D10S1 = frozenset({29, 47, 79})  # Cu, Ag, Au (d10s1)
    if l_B == 2 and not is_linear and Z_B in _NS1_D10S1:
        _per_avg_t_d10 = sum(atom_data[topology.Z_list[t]]['per']
                             for t in terms) / max(zc[B], 1)
        _delta_per = max(_per_avg_t_d10 - 2.0, 0.0)
        if _delta_per > 0:
            _S_d10s1 = S3 * S_HALF * _delta_per
            D0_AB *= math.exp(-_S_d10s1)
            D0_BC *= math.exp(-_S_d10s1)

    # ── F6: Hund exchange + Mertens spectral convergence ──
    # Port of diatomic D_exch_Hund (early-d spin magnetization) and
    # orbital polarization (late-d spin-down k=1 dipole) to triatomic.
    # In triatomic, center B is d-block; each terminal is a partner.
    # The d-electrons are shared between BOTH bonds, so we apply ONCE
    # (not per-bond) and scale by D_FULL (matching diatomic convention).
    # Gate: d-block center. Per-terminal: LP_partner = 0 for Hund/OrbPol,
    #        LP_partner > 0 for CFSE.
    # 0 adjustable parameters.
    _NS1_METALS_TRI = frozenset({24, 29, 41, 42, 44, 45, 46, 47, 78, 79})
    if l_B == 2:
        _nf_eff_tri = nf_B + (1 if Z_B in _NS1_METALS_TRI else 0)
        _nf_eff_tri = min(_nf_eff_tri, 2 * P2)

        if _nf_eff_tri > 0:
            # ── F6a: Hund exchange (early d, nf_eff <= P2) ──
            # Spin magnetization M = n_unp/(2P2) or Mertens(n_unp) at half-fill.
            # Per terminal with LP=0 (spin-active partner).
            if _nf_eff_tri <= P2:
                _n_unp_tri = min(_nf_eff_tri, 2 * P2 - _nf_eff_tri)
                if _n_unp_tri > 0:
                    if _nf_eff_tri >= P2 - 1:
                        # Mertens resummation at half-fill
                        _fill_f = (float(_n_unp_tri) / (2.0 ** (_n_unp_tri - 1))
                                   if _n_unp_tri > 1
                                   else float(_n_unp_tri) / (2.0 * P2))
                    else:
                        _fill_f = float(_n_unp_tri) / (2.0 * P2)
                    for t in terms:
                        if topology.lp[t] == 0:
                            _ie_t_hund = atom_data[topology.Z_list[t]]['IE']
                            _eps_g_hund = math.sqrt(ie_B * _ie_t_hund) / RY
                            D_face += _eps_g_hund * _fill_f * (RY / P2) * S_HALF * D_FULL

            # ── F6b: Orbital polarization (late d, nf_eff > P2) ──
            # Spin-down k=1 Dirichlet mode on Z/P2Z.
            # Gate: nf_eff > P2, terminal LP = 0.
            if _nf_eff_tri > P2:
                _n_dn_tri = _nf_eff_tri - P2
                if 0 < _n_dn_tri < P2:
                    _SIN2_PI_P2 = math.sin(math.pi / P2) ** 2
                    _k1_dn_tri = (2.0 * math.sin(math.pi * _n_dn_tri / P2) ** 2
                                  / (P2 * P2 * _SIN2_PI_P2))
                    for t in terms:
                        if topology.lp[t] == 0:
                            _ie_t_op = atom_data[topology.Z_list[t]]['IE']
                            _eps_g_op = math.sqrt(ie_B * _ie_t_op) / RY
                            D_face += _eps_g_op * _k1_dn_tri * (RY / P2) * S_HALF * D_FULL

            # ── F6c: CFSE crystal field (P3->P2 coupling) ──
            # Gate: at least one terminal has LP > 0 (strong-field ligand).
            _r7_cf = _nf_eff_tri % P3
            _sin2_r7_cf = math.sin(2.0 * math.pi * _r7_cf / P3) ** 2
            if _sin2_r7_cf > 1e-10:
                for t in terms:
                    _lp_t_cf = topology.lp[t]
                    if _lp_t_cf > 0:
                        _f_lp_cf = min(float(_lp_t_cf) / P1, 1.0)
                        D_face += (_sin2_r7_cf * D7 / P2 * _f_lp_cf
                                   * (RY / P2) * S_HALF * D_FULL)

    # ── F7: Dark modes k=2,4 on Z/10Z ──
    # Fourier modes k=2,4 on the pentagonal face Z/(2P2)Z.
    # Gate: d-block center, per >= P2 (tier_d >= 0).
    # 0 adjustable parameters.
    if l_B == 2:
        _nf_eff_dk = nf_B + (1 if Z_B in _NS1_METALS_TRI else 0)
        _nf_eff_dk = min(_nf_eff_dk, 2 * P2)
        if _nf_eff_dk > 0:
            _tier_d = per_B - P2
            if _tier_d >= 0:
                _omega_10 = 2.0 * math.pi / (2 * P2)
                _chir_d = 1 - 2 * (_tier_d % 2)
                _gamma_d = GAMMA_7 if _tier_d >= 1 else 1.0
                _D_dark_tri = 0.0
                # k=2 dark mode: active for tier_d = 0,1
                if _tier_d <= 1:
                    _D_dark_tri += (-_chir_d * _gamma_d
                                    * (R35_DARK + R57_DARK)
                                    * math.cos(2 * _omega_10 * _nf_eff_dk)
                                    * (RY / P2))
                # k=4 dark mode: active only tier_d = 0 (per=P2, 3d metals)
                if _tier_d == 0:
                    _D_dark_tri += (-_chir_d * R37_DARK
                                    * math.cos(4 * _omega_10 * _nf_eff_dk)
                                    * (RY / P2))
                D_face += _D_dark_tri * D_FULL

    # ── Triatomic engine scope check ──
    # All 11 formerly-excluded patterns now have dedicated routing below.
    # No early returns — every triatomic is handled by the dedicated engine.

    # ── Per-type coordination corrections (on v4 baseline) ──
    mechanism = "triatomic_base"

    # ── RADICAL branch (NO2-like) ──
    # Radical = odd total electrons. The unpaired electron blocks half the
    # π channel and the 3×3 σ renormalization is strong.
    if is_radical:
        # R1: Split D_P1 into σ and π components per bond order.
        # The v4 seed lumps σ+π into D_P1. For radicals, σ gets f_renorm
        # (coordination sharing) while π gets (1-s) blocking (radical
        # occupies half the π manifold).
        for sr, bo_r, label in [(sr_AB, bo_AB, 'AB'), (sr_BC, bo_BC, 'BC')]:
            sig_frac = 1.0 / max(bo_r, 1)
            pi_frac = max(0.0, bo_r - 1) / max(bo_r, 1)
            D_sig_r = sr.D_P1 * sig_frac * f_renorm
            D_pi_r = sr.D_P1 * pi_frac * (1.0 - S_HALF)
            D_other_r = (sr.D_P2 + sr.D_P3) * (1.0 - S_HALF)
            if label == 'AB':
                D0_AB = D_sig_r + D_pi_r + D_other_r
            else:
                D0_BC = D_sig_r + D_pi_r + D_other_r
        # GFT radical budget: center IE × D_FULL (Shannon cap per vertex)
        radical_budget = ie_B * D_FULL
        D_total_pre = D0_AB + D0_BC + D_face
        if D_total_pre > radical_budget:
            scale_rad = radical_budget / max(D_total_pre, 1e-12)
            D0_AB *= scale_rad
            D0_BC *= scale_rad
            D_face *= scale_rad
        mechanism = "triatomic_radical"

    # ── LINEAR branch (CO2, CS2, N2O, OCS, HCN, HNC) ──
    elif is_linear:
        # L0: d-block linear flags and f_renorm override.
        # For d-block centers (l_B=2) in linear triatomics, the 3x3
        # eigenvalue gap reflects the large IE asymmetry (O vs Ti),
        # not bond cooperation. Reset f_renorm to 1.0 (no cooperation).
        # Gate: d-block center (l_B = 2).
        _dblock_linear = (l_B == 2)
        if _dblock_linear:
            f_renorm = 1.0

        # L1: σ renormalization via 3×3 eigensystem
        # Decompose v4 into σ (D_P1) and non-σ (D_P2 + D_P3).
        # For bo_span >= 2 (triple+single), the triple bond's σ channel
        # is partially shielded from coordination competition by its
        # large bo. Apply reduced f_renorm: only 1/bo of the compression.
        D_sig_AB = sr_AB.D_P1
        D_nonsig_AB = sr_AB.D_P2 + sr_AB.D_P3
        D_sig_BC = sr_BC.D_P1
        D_nonsig_BC = sr_BC.D_P2 + sr_BC.D_P3
        if bo_span >= 1.5 and max(bo_AB, bo_BC) >= 3:
            # Triple bond: partial shielding from coordination
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

        # L1b: d-block linear competition screening.
        # For d-block centers, the d-electrons are split across z_B bonds.
        # When the d-shell is early-filled (nf <= P2), the vacancy fraction
        # (1 - nf/2P2) is large: few d-electrons available per bond.
        # Competition screening: S_comp = (1-nf/2P2) * z_B * S_HALF / P2
        # attenuates both bonds via exp(-S_comp) on Z/(2P2)Z.
        # Gate: d-block center (l_B = 2).
        # 0 adjustable parameters.
        if _dblock_linear:
            _nf_lin = nf_B
            _S_comp_d = (1.0 - float(_nf_lin) / (2.0 * P2)) * float(zc[B]) * S_HALF / P2
            _f_comp_d = math.exp(-_S_comp_d)
            D0_AB *= _f_comp_d
            D0_BC *= _f_comp_d

        # L2: Hückel 3-center π stabilization
        # For d-block linear, d-orbital pi bonds use orthogonal d_xz/d_yz
        # channels, not a delocalised p-pi manifold — Huckel 3c does not
        # apply. Gate: not d-block center.
        _HUCKEL_3C = 4.0 - 2.0 * math.sqrt(2.0)
        # When BOTH terminals have LP but in different amounts (N2O-like),
        # the π conjugation is disrupted by the LP gradient.
        # Scale Hückel by C₃ (cos²₃) to account for broken symmetry.
        # Gate: both terminals must have LP > 0 (not HCN where H has lp=0).
        lp_terms = [topology.lp[t] for t in terms]
        lp_asym = (min(lp_terms) > 0 and min(lp_terms) != max(lp_terms))
        f_huckel = C3 if lp_asym else 1.0
        if not _dblock_linear:
            D_face += _HUCKEL_3C * D5 * S_HALF * n_pi_total * cap_face * D_FULL * f_huckel

        # L3: Asymmetric linear (OCS, HCN, HNC, N2O)
        # When terminals differ, polarity lifts the weaker bond.
        # The asymmetry boost uses the 3×3 gap asymmetry.
        Z_A_lin, Z_C_lin = topology.Z_list[A], topology.Z_list[C]
        if Z_A_lin != Z_C_lin:
            ie_A_l, ie_C_l = d_A['IE'], d_C['IE']
            q_rel_lin = min(ie_A_l, ie_C_l) / max(ie_A_l, ie_C_l)
            asym_boost = (1.0 - q_rel_lin) * S3
            if D0_AB < D0_BC:
                D0_AB *= (1.0 + asym_boost)
            else:
                D0_BC *= (1.0 + asym_boost)

        # L4: Triple+single bond cross-coupling (HCN, HNC)
        # The triple bond creates through-bond π conjugation that
        # stabilizes the single bond partner. The gain is proportional
        # to the triple bond's π energy (D_P1 × (bo-1)/bo).
        if bo_span >= 1.5 and max(bo_AB, bo_BC) >= 3:
            if bo_AB > bo_BC:
                pi_energy = sr_AB.D_P1 * (bo_AB - 1) / bo_AB
                D0_BC += pi_energy * S3 * S_HALF
            else:
                pi_energy = sr_BC.D_P1 * (bo_BC - 1) / bo_BC
                D0_AB += pi_energy * S3 * S_HALF

        # L5: Terminal LP screening for asymmetric LP linears (N2O).
        # Each terminal's LP screens the adjacent bond proportionally.
        # The LP points away from the molecule, reducing back-donation
        # to the adjacent bond. S_lp = lp × S₃ / P₁ per bond.
        if lp_asym:
            for t_idx, t in enumerate(terms):
                lp_t = topology.lp[t]
                if lp_t > 0:
                    S_lp = lp_t * S3 / P1
                    # Which bond is adjacent to this terminal?
                    if t == A or (t_idx == 0):
                        D0_AB *= math.exp(-S_lp)
                    else:
                        D0_BC *= math.exp(-S_lp)

        # L6: Formal charge ionic correction (HNC-like).
        # When a terminal of a triple bond has unused valence (all bonds
        # accounted for but fewer than normal valence), the "missing"
        # capacity represents formal charge transfer. This creates ionic
        # stabilization: D_fc = unused_ve × Ry / P₁ × s × D_FULL.
        if bo_span >= 1.5 and max(bo_AB, bo_BC) >= 3:
            for t in terms:
                Z_t = topology.Z_list[t]
                ve_t = valence_electrons(Z_t)
                sbo_t = round(topology.sum_bo[t])
                lp_t = topology.lp[t]
                unused_ve = ve_t - sbo_t - 2 * lp_t
                if unused_ve > 0:
                    D_fc = unused_ve * RY / P1 * S_HALF * D_FULL
                    D_face += D_fc

        mechanism = "triatomic_linear"

    # ── S-BLOCK LINEAR branch (BeF2, BeCl2) ──
    elif is_sblock_linear:
        # s-block centers (Be, Mg) form ionic-dominated bonds.
        # The v4 diatomic seeds undercount because the strong ionic
        # character (Pythagore) is enhanced by bilateral coordination.
        # S1: No σ renormalization for s-block — the 3×3 eigensystem
        # is designed for covalent coordination and gives misleading
        # f_renorm for strongly ionic Be/Mg centers.
        # Keep v4 seeds as-is (f_renorm = 1.0 effectively).
        # S2: Ionic bilateral boost from Pythagore.
        # Each terminal donates LP → center vacancy. The bilateral
        # donation creates an ionic field that stabilizes both bonds.
        # Heavier terminals (higher period) provide more diffuse LP
        # donation → stronger coordination ionic boost.
        for t in terms:
            lp_t = topology.lp[t]
            if lp_t > 0:
                d_t = atom_data[topology.Z_list[t]]
                ie_t = d_t['IE']
                per_t = d_t['per']
                f_ie = min(ie_t, RY) / RY
                f_per = 1.0 + S_HALF * per_t / P1
                D_face += lp_t * S3 / P1 * cap_face * f_ie * f_per * D_FULL
        mechanism = "triatomic_sblock"

    # ── BENT branch ──
    else:
        # B1: LP-LP terminal crowding
        # When center has lp ≥ z and terminals also have LP,
        # LP-LP repulsion creates screening.
        if lp_B >= zc[B]:
            lp_adj_weighted = 0.0
            min_term_lp = min(topology.lp[t] for t in terms)
            for t in terms:
                lp_t = topology.lp[t]
                d_t = atom_data[topology.Z_list[t]]
                per_t = d_t['per']
                lp_adj_weighted += lp_t * (2.0 / max(per_t, 2)) ** (1.0 / P1)
            if lp_adj_weighted > zc[B]:
                f_crowd = (lp_adj_weighted - zc[B]) / max(lp_adj_weighted, 1)
                lp_per_term = sum(topology.lp[t] for t in terms) / max(zc[B], 1.0)
                S_crowd = D3 * S_HALF * f_crowd * (1.0 + lp_per_term / P1)
                # When ALL terminals have LP ≥ P₁ (Cl2O, SCl2), bilateral
                # LP-LP repulsion is stronger: amplify by polarity factor.
                # Polarity > 1 = different IE → stronger crowding (Cl2O).
                # Polarity ≈ 1 = same IE → mild crowding (SCl2).
                if min_term_lp >= P1:
                    ie_terms = [atom_data[topology.Z_list[t]]['IE'] for t in terms]
                    term_polarity = max(ie_terms) / max(min(ie_terms), 0.1)
                    S_crowd *= (1.0 + (term_polarity - 1.0) * S3)
                # B1b: LP crowding triangle amplification.
                # When center has compact LP (per ≤ 2) and all atoms have LP,
                # the LP triangle (center + 2 terminals) creates strong
                # Coulomb repulsion. Excess LP beyond P₂ drives screening.
                lp_total = lp_B + sum(topology.lp[t] for t in terms)
                if per_B <= 2 and lp_total > P2:
                    S_crowd += (lp_total - P2) * S3 * S_HALF / zc[B]
                D0_AB *= math.exp(-S_crowd)
                D0_BC *= math.exp(-S_crowd)

        # B2: Coordination correction from 3×3 eigensystem (perturbative)
        # LP gate has already pushed f_renorm toward 1.0 for lp≥z centers.
        # The residual captures genuine coordination sharing.
        if abs(f_renorm - 1.0) > 0.01:
            coord_delta = (f_renorm - 1.0) * D3
            D0_AB *= (1.0 + coord_delta)
            D0_BC *= (1.0 + coord_delta)

        # B3: Singlet carbene correction (CCl2, CF2, etc.)
        # Carbon center with lp=1, z=2, no π: LP blocks σ channel.
        # Gate: at least one terminal must have LP (heavy atom), not H.
        # For CH₂ (both terminals = H), the LP cooperation with H is
        # already captured by the v4 seed — no attenuation needed.
        if Z_B == 6 and lp_B == 1 and n_pi_total == 0:
            max_term_lp = max(topology.lp[t] for t in terms)
            if max_term_lp > 0:
                # Heavy carbene (CCl2, CF2): LP on terminals crowds center
                # The attenuation is stronger for terminals with more LP
                # (Cl lp=3 is more crowding than F lp=3 because Cl is bigger)
                polarity = max(d_A['IE'], d_C['IE']) / max(min(d_A['IE'], d_C['IE']), 0.1)
                avg_term_lp = sum(topology.lp[t] for t in terms) / 2.0
                atten = D5 + D3 * S_HALF * polarity / P1
                # Terminal LP enhancement: more LP = more crowding
                atten *= (1.0 + (avg_term_lp - 1) * S3 / P1)
                atten = max(0.0, min(atten, S3 + D5))
                D0_AB *= (1.0 - atten)
                D0_BC *= (1.0 - atten)
                mechanism = "triatomic_carbene"

        # B4: Mixed bond-order bent (HNO, ClNO, Nitrosyl_chloride)
        # Center has LP (0 < lp_B < z_B) and bonds have different orders.
        # The LP on center creates σ-π competition: the LP blocks one
        # σ channel, and both bonds share the remaining spectral budget.
        # Attenuation: both bonds get lp_B × S₃ / z_B screening.
        elif bo_span > 0 and n_pi_total > 0 and 0 < lp_B < zc[B]:
            S_mixed = lp_B * S3 / max(zc[B], 1)
            D0_AB *= (1.0 - S_mixed)
            D0_BC *= (1.0 - S_mixed)
            # Additional screening on the weaker bond (σ-π asymmetry)
            if D0_AB < D0_BC:
                D0_AB *= (1.0 - D5)
            else:
                D0_BC *= (1.0 - D5)
            mechanism = "triatomic_mixed_bent"

        # B5: Bent π (SO2-like, SSO-like)
        elif n_pi_total >= 2 and per_B >= 3:
            Z_A_p, Z_C_p = topology.Z_list[A], topology.Z_list[C]
            if Z_A_p != Z_C_p:
                # Heterogeneous terminals (SSO-like): π-plane asymmetry
                # creates additional screening from polarity mismatch.
                ie_terms_p = [d_A['IE'], d_C['IE']]
                q_rel_p = min(ie_terms_p) / max(ie_terms_p)
                S_hetero = (1.0 - q_rel_p) * S3
                D0_AB *= (1.0 - D5 * S_HALF - S_hetero)
                D0_BC *= (1.0 - D5 * S_HALF - S_hetero)
                mechanism = "triatomic_bent_pi_hetero"
            else:
                # Symmetric terminals: mild π plane sharing
                D0_AB *= (1.0 - D5 * S_HALF)
                D0_BC *= (1.0 - D5 * S_HALF)
                mechanism = "triatomic_bent_pi"

        else:
            mechanism = "triatomic_bent"

    # ── Ghost VP attenuation + assembly ──
    bond_results: List[BondResult] = []
    D_at_total = 0.0
    D_sum_P1, D_sum_P2, D_sum_P3 = 0.0, 0.0, 0.0

    for D0_bond, sr, (ia, ib, bo_b, bi) in [
        (D0_AB, sr_AB, bond_AB), (D0_BC, sr_BC, bond_BC)
    ]:
        di_b, dj_b = atom_data[topology.Z_list[ia]], atom_data[topology.Z_list[ib]]
        per_Ab, per_Bb = di_b['per'], dj_b['per']
        n_shells = ((per_Ab - 1) + (per_Bb - 1)) / 2.0
        z_ib, z_jb = zc[ia], zc[ib]
        z_min_b = min(z_ib, z_jb)
        f_z_ghost = 1.0 + max(0, z_min_b - 1) * D3
        S_ghost = BETA_GHOST * D3 * S_HALF * max(n_shells, 1.0) * f_z_ghost
        ghost_atten = math.exp(-S_ghost * D_FULL)

        D0_final = D0_bond * ghost_atten

        # Per-face decomposition using v4 ratios
        D0_raw = sr.D0
        if D0_raw > 1e-10:
            scale_b = D0_bond / D0_raw
        else:
            scale_b = 1.0
        D_P1_b = sr.D_P1 * scale_b * ghost_atten
        D_P2_b = sr.D_P2 * scale_b * ghost_atten
        D_P3_b = sr.D_P3 * scale_b * ghost_atten

        D_sum_P1 += D_P1_b
        D_sum_P2 += D_P2_b
        D_sum_P3 += D_P3_b

        # Physical observables
        lp_ib = topology.lp[ia]
        lp_jb = topology.lp[ib]
        r_e_b = r_equilibrium(per_Ab, per_Bb, bo_b, sr.S_total,
                              z_ib, z_jb, lp_ib, lp_jb)
        w_e_b = omega_e(D0_final, r_e_b, di_b['mass'], dj_b['mass'])

        bond_results.append(BondResult(
            D0=D0_final,
            v_sigma=D_P1_b,
            v_pi=D_P2_b,
            v_ionic=D_P3_b,
            r_e=r_e_b,
            omega_e=w_e_b,
        ))
        D_at_total += D0_final

    D_at = D_at_total + D_face

    return TransferResult(
        D_at=D_at,
        D_at_P1=D_sum_P1,
        D_at_P2=D_sum_P2,
        D_at_P3=D_sum_P3,
        E_spectral=D_face,
        bonds=bond_results,
        spectrum_P1=eigvals,
        spectrum_P2=None,
        spectrum_P3=None,
        topology=topology,
        mechanism=mechanism,
    )


# ╔════════════════════════════════════════════════════════════════════╗
# ║  SECTION 10: compute_D_at_transfer — SCF T⁴ PIPELINE              ║
# ╚════════════════════════════════════════════════════════════════════╝

def compute_D_at_transfer(topology: Topology,
                          source: str = "v4") -> TransferResult:
    """Compute molecular atomization energy via SCF on T⁴.

    Pipeline:
      1. Resolve atomic data (IE, EA, T⁴ coordinates)
      2. Diatomic fast-path (2x2 analytical formula)
      3. Huckel aromatic caps (ring pi delocalization)
      4. Per-bond seeds (4-face DFT screening)
      4b. [v4] DFT cross-spectrum correction + Parseval budget
      5. Shell attenuation (hypervalent vertices)
      6. SCF iterate (Perron on T_T4)
      7. Ghost attenuation (Mertens tail p>=11)
      8. Spectral extraction (eigenvalue corrections)
      9. Assembly -> TransferResult

    Sources:
      "v4" (default): v4 spectral DFT cross-spectrum correction on
          Z/(2P₁)Z + Parseval budget for hypervalent vertices. The
          cross-spectrum captures inter-atomic spectral coherence.
          The Parseval budget distributes spectral power at vertices
          with z > P_l across their bonds.
      "v4_spectral": per-bond seeds from v4 D0_screening (diatomic
          quality) + Parseval z^(-1/6) sharing + DFT vertex coupling.
          Replaces the 4-face DFT pipeline with direct v4 calls.
      "full_pt": original engine without v4 corrections.
      "full_pt_v4": alias for "v4".

    0 adjustable parameters. All from s = 1/2.
    """
    # ── DIM 0: VERTICES — atom properties ────────────────────────────
    atom_data = _resolve_atom_data(topology, source)
    n_atoms = topology.n_atoms
    n_bonds = len(topology.bonds)

    # ══════════════════════════════════════════════════════════════════
    # DIATOMIC FAST-PATH (n_atoms=2, n_bonds=1): analytical 2x2
    # ══════════════════════════════════════════════════════════════════
    if n_atoms == 2 and n_bonds == 1:
        i, j, bo = topology.bonds[0]
        Zi, Zj = topology.Z_list[i], topology.Z_list[j]
        di, dj = atom_data[Zi], atom_data[Zj]
        lp_i = topology.lp[i]
        lp_j = topology.lp[j]

        # ── v4 screening_bond direct pathway ──
        # D0_screening already includes D_FULL, all 7 screening terms,
        # and per-face decomposition. No ghost needed (absorbed in v4).
        sr = _D0_screening_v4(Zi, Zj, bo, lp_A=lp_i, lp_B=lp_j)
        D0_final = sr.D0

        # Physical observables
        per_A, per_B = di['per'], dj['per']
        z_i = topology.z_count[i]
        z_j = topology.z_count[j]
        S_bond = sr.S_total
        r_e = r_equilibrium(per_A, per_B, bo, S_bond,
                            z_i, z_j, lp_i, lp_j)
        w_e = omega_e(D0_final, r_e, di['mass'], dj['mass'])

        bond_result = BondResult(
            D0=D0_final,
            v_sigma=sr.D_P1,
            v_pi=sr.D_P2,
            v_ionic=sr.D_P3,
            r_e=r_e,
            omega_e=w_e,
        )

        return TransferResult(
            D_at=D0_final,
            D_at_P1=sr.D_P1,
            D_at_P2=sr.D_P2,
            D_at_P3=sr.D_P3,
            E_spectral=0.0,
            bonds=[bond_result],
            spectrum_P1=None,
            spectrum_P2=None,
            spectrum_P3=None,
            topology=topology,
            mechanism=None,
        )

    # ══════════════════════════════════════════════════════════════════
    # TRIATOMIC FAST-PATH (n_atoms=3, n_bonds=2): 3x3 + v4 seeds
    # ══════════════════════════════════════════════════════════════════
    if n_atoms == 3 and n_bonds == 2 and source in ("v4", "full_pt_v4"):
        _tri_result = _triatomic_fast_path(topology, atom_data)
        if _tri_result is not None:
            return _tri_result

    # ══════════════════════════════════════════════════════════════════
    # POLYATOMIC PATH (n_atoms >= 3): SCF on T⁴
    # ══════════════════════════════════════════════════════════════════

    # Step 3: Huckel aromatic caps (ring pi delocalization)
    huckel_caps = _huckel_aromatic(topology, atom_data)

    # Step 4: Per-bond seeds
    _use_v4 = source in ("v4", "full_pt_v4")
    _use_v4_spectral = source == "v4_spectral"
    bond_seeds: Dict[int, BondSeed] = {}

    if _use_v4_spectral:
        # ── v4_spectral: v4-capped 4-face DFT seeds ──
        # The spectral blending caps overbinding DFT seeds at v4 × (1+S₃)
        # and fades the correction for n > P₃ (where SCF handles sharing).
        bond_seeds = _compute_bond_seed_spectral(topology, atom_data, huckel_caps)
    else:
        for bi, (i, j, bo) in enumerate(topology.bonds):
            if _use_v4:
                bond_seeds[bi] = _compute_bond_seed_v4(
                    i, j, bo, bi, topology, atom_data, huckel_caps
                )
            else:
                bond_seeds[bi] = _compute_bond_seed(
                    i, j, bo, bi, topology, atom_data, huckel_caps
                )

    # ── SMALL-MOLECULE SEED CALIBRATION [v4 anchor, 0 params] ─────────
    # For small molecules (n < P₃), the polyatomic bond seeds deviate
    # from the v4 diatomic values because the vertex polygon DFT was
    # tuned for large molecules. The v4 engine (MAE 0.93% on 135
    # diatomics) provides a high-quality per-bond reference.
    #
    # TRIATOMIC REGIME (n=3): the poly engine's vertex DFT shares a
    # single polygon between 2 bonds, creating large correlated errors
    # (±15-25%). For triatomics, use a STRONG blend toward v4 with a
    # 3-center coordination screening factor:
    #
    #   D0_tri = v4 × (1 - (z_c - 1) × D₃ / z_c)
    #
    # where z_c is the central vertex coordination. The factor D₃/z_c
    # captures the face density shared between bonds (Fisher dispersion
    # on Z/(2P₁)Z, divided by z neighbors).
    #
    # LARGER MOLECULES (n=4..6): blend toward v4 with C₃ factor.
    #
    # Three regimes for overbinding:
    # 1. EXCESS > (1+S₃): cap at v4 × (1+S₃)
    # 2. MILD EXCESS (ratio in (1, 1+S₃]): graduated damping D₃×S₃×f_size
    #    (only when mean ratio > 1, i.e. collective overbinding)
    # 3. DEFICIT (ratio < 1): blend toward v4 with C₃
    #
    # 0 adjustable parameters.
    if not _use_v4_spectral:
        if n_atoms < P3 and _use_v4:
            _EXCESS_THRESHOLD = 1.0 + S3  # ≈ 1.219
            _f_size = max(0.0, 1.0 - float(n_atoms - P1) / (P3 - P1))

            # For triatomics (n=3): find central vertex coordination
            _z_central = 2  # default for A-B-C
            if n_atoms == 3:
                for _v_tri in range(3):
                    if topology.z_count[_v_tri] >= 2:
                        _z_central = topology.z_count[_v_tri]
                        break
                # 3-center coordination screening factor
                # Each additional bond at the vertex costs D₃/z on the
                # shared polygon face (Fisher dispersion per neighbor)
                _f_3c = 1.0 - float(_z_central - 1) * D3 / _z_central

            # Pre-compute per-bond ratios and mean for collective overbinding detection
            _ratios_4b: Dict[int, float] = {}
            _sr_v4_cache: Dict[int, object] = {}
            for bi, (i, j, bo) in enumerate(topology.bonds):
                Zi, Zj = topology.Z_list[i], topology.Z_list[j]
                sr_v4 = _D0_screening_v4(Zi, Zj, bo,
                                         lp_A=topology.lp[i],
                                         lp_B=topology.lp[j])
                _sr_v4_cache[bi] = sr_v4
                _ratios_4b[bi] = bond_seeds[bi].D0 / max(sr_v4.D0, 1e-10)
            _mean_ratio = (sum(_ratios_4b.values()) / max(len(_ratios_4b), 1)
                           if _ratios_4b else 1.0)

            for bi, (i, j, bo) in enumerate(topology.bonds):
                sr_v4 = _sr_v4_cache[bi]
                seed = bond_seeds[bi]
                ratio = _ratios_4b[bi]

                # ── ASYMMETRIC LINEAR VERTEX CORRECTION [0 params] ───
                # At a linear vertex (z=2, lp=0) with asymmetric bond
                # orders (bo_max > bo_min), the π-modes of the multiple
                # bond are spectrally orthogonal to the σ-mode of the
                # single bond on Z/(2P₁)Z.  The vertex DFT correctly
                # captures the σ→π donation (multi-center enhancement),
                # which is absent in the isolated diatomic v4 seed.
                #
                # Each asymmetric linear endpoint adds 1/(2P₁) spectral
                # budget from the orthogonal σ-mode.  The Parseval
                # identity on Z/(2P₁)Z preserves each mode independently.
                # The threshold is raised by C₃^(-n_asym/2): the inverse
                # hexagonal face coupling constant per endpoint, since the
                # modes are uncoupled (cos²(π/3) = C₃ is the coupling for
                # NON-orthogonal modes; removing it for orthogonal ones).
                #
                # n_asym=0: standard threshold (1+S₃)
                # n_asym=1: threshold × C₃^(-½) ≈ 1.38
                # n_asym=2: threshold × C₃^(-1)  ≈ 1.56
                _n_asym_bi = 0
                for _v_al in (i, j):
                    if (topology.z_count[_v_al] == 2
                            and topology.lp[_v_al] == 0):
                        _bos_al = [topology.bonds[_bk][2]
                                   for _bk in topology.vertex_bonds.get(_v_al, [])]
                        if len(_bos_al) == 2 and min(_bos_al) < max(_bos_al):
                            _n_asym_bi += 1
                _thr_bi = _EXCESS_THRESHOLD * C3 ** (-_n_asym_bi / 2.0)

                if ratio > _thr_bi:
                    # Overbinding seed — cap at v4 × threshold
                    D0_capped = sr_v4.D0 * _thr_bi
                    _scale_cap = D0_capped / max(seed.D0, 1e-10)
                    bond_seeds[bi] = BondSeed(
                        D0=D0_capped,
                        D0_P0=seed.D0_P0 * _scale_cap,
                        D0_P1=seed.D0_P1 * _scale_cap,
                        D0_P2=seed.D0_P2 * _scale_cap,
                        D0_P3=seed.D0_P3 * _scale_cap,
                        S0=seed.S0,
                        S1=seed.S1,
                        S2=seed.S2,
                        S3=seed.S3,
                        Q_eff=seed.Q_eff,
                    )
                elif 1.0 < ratio <= _thr_bi and _f_size > 0 and _mean_ratio > 1.0:
                    # Mild overbinding dead zone — graduated dampening.
                    # Only fires when the molecule collectively overbinds
                    # (mean ratio > 1.0), preventing under-correction of
                    # molecules that have mixed over/under bonds.
                    # D₃ holonomy × f_size ensures fade at n≥7, 0 new params.
                    #
                    # C19: Skip when C16 degeneracy will suppress NLO.
                    # At per ≤ 2 z ≥ P₁ lp=0 vertices with ALL identical
                    # bonds (same bo, same partner Z), C16 sets f_degen=0,
                    # removing the NLO correction entirely.  The mild damp
                    # was designed to pre-compensate for NLO overcounting,
                    # so it double-corrects when C16 fires.
                    # 0 adjustable parameters.
                    _c16_suppresses = False
                    for _v_c19 in (i, j):
                        _z_c19 = topology.z_count[_v_c19]
                        _lp_c19 = topology.lp[_v_c19]
                        _per_c19 = atom_data[topology.Z_list[_v_c19]]['per']
                        if _z_c19 >= P1 and _lp_c19 == 0 and _per_c19 <= 2:
                            _vb_c19 = topology.vertex_bonds.get(_v_c19, [])
                            if len(_vb_c19) >= P1:
                                _bos = [topology.bonds[b][2] for b in _vb_c19]
                                _Zps = []
                                for _bk in _vb_c19:
                                    _ik, _jk, _ = topology.bonds[_bk]
                                    _pk = _jk if _ik == _v_c19 else _ik
                                    _Zps.append(topology.Z_list[_pk])
                                if (max(_bos) - min(_bos) < 0.01
                                        and len(set(_Zps)) == 1):
                                    _c16_suppresses = True
                                    break
                    if _c16_suppresses:
                        pass  # skip: C16 removes NLO, damp would double-correct
                    else:
                        _excess_frac = (ratio - 1.0) / (_thr_bi - 1.0)
                        _damp = 1.0 - _excess_frac * _f_size * D3 * S3
                        _D0_damped = seed.D0 * _damp
                        if _D0_damped < seed.D0:
                            _scale_damp = _D0_damped / max(seed.D0, 1e-10)
                            bond_seeds[bi] = BondSeed(
                                D0=_D0_damped,
                                D0_P0=seed.D0_P0 * _scale_damp,
                                D0_P1=seed.D0_P1 * _scale_damp,
                                D0_P2=seed.D0_P2 * _scale_damp,
                                D0_P3=seed.D0_P3 * _scale_damp,
                                S0=seed.S0,
                                S1=seed.S1,
                                S2=seed.S2,
                                S3=seed.S3,
                                Q_eff=seed.Q_eff,
                            )
                elif ratio < 1.0 and _f_size > 0:
                    # Underbinding seed — blend toward v4, guarded by
                    # bond-order-dependent Shannon cap.
                    _bo_bi = topology.bonds[bi][2]
                    _shannon_bo = (RY / P1
                                   + min(max(_bo_bi - 1.0, 0.0), 1.0) * RY / P2
                                   + min(max(_bo_bi - 2.0, 0.0), 1.0) * RY / P3)
                    _v4_reliable = sr_v4.D0 < _shannon_bo * (1.0 + S3)
                    if _v4_reliable:
                        _deficit = 1.0 - ratio
                        _f_boost = C3 * _deficit * _f_size
                        _D0_gap = sr_v4.D0 - seed.D0
                        _D0_new = seed.D0 + _D0_gap * _f_boost
                        if _D0_new > seed.D0:
                            _scale_boost = _D0_new / max(seed.D0, 1e-10)
                            bond_seeds[bi] = BondSeed(
                                D0=_D0_new,
                                D0_P0=seed.D0_P0 * _scale_boost,
                                D0_P1=seed.D0_P1 * _scale_boost,
                                D0_P2=seed.D0_P2 * _scale_boost,
                                D0_P3=seed.D0_P3 * _scale_boost,
                                S0=seed.S0,
                                S1=seed.S1,
                                S2=seed.S2,
                                S3=seed.S3,
                                Q_eff=seed.Q_eff,
                            )

        # Step 4b: Parseval spectral budget (multi-centre constraint)
        # Parseval distributes spectral power at vertices with z > P_l.
        # Shell attenuation handles polygon promotion (Z/6Z -> Z/10Z).
        # Both are applied: Parseval is a SPECTRAL correction (gentle),
        # shell attenuation is a GEOMETRIC correction (strong for excess bonds).
        if _use_v4:
            bond_seeds = _parseval_budget(topology, atom_data, bond_seeds)

    # v4_spectral: spectral SCF replaces static Parseval
    # The SCF iterates the Parseval constraint mode-by-mode until
    # convergence, capturing inter-bond spectral competition.
    if _use_v4_spectral:
        bond_seeds = _scf_spectral(topology, atom_data, bond_seeds)

    # Step 5: Shell attenuation (hypervalent vertices, in-place)
    seed_energies: Dict[int, float] = {bi: seed.D0 for bi, seed in bond_seeds.items()}
    _apply_shell_attenuation(topology, atom_data, seed_energies)

    # Update seeds if shell attenuation changed energies
    for bi in bond_seeds:
        if abs(seed_energies[bi] - bond_seeds[bi].D0) > 1e-12:
            old_seed = bond_seeds[bi]
            scale_shell = seed_energies[bi] / max(old_seed.D0, 1e-10)
            bond_seeds[bi] = BondSeed(
                D0=seed_energies[bi],
                D0_P0=old_seed.D0_P0 * scale_shell,
                D0_P1=old_seed.D0_P1 * scale_shell,
                D0_P2=old_seed.D0_P2 * scale_shell,
                D0_P3=old_seed.D0_P3 * scale_shell,
                S0=old_seed.S0,
                S1=old_seed.S1,
                S2=old_seed.S2,
                S3=old_seed.S3,
                Q_eff=old_seed.Q_eff,
            )

    # Step 6: SCF iterate (Perron on T_T4)
    bond_energies_scf, eigenvalues = _scf_iterate(topology, atom_data, bond_seeds)

    # ══════════════════════════════════════════════════════════════════
    # PORTED PT-DERIVED CORRECTIONS (from old engine, 0 adjustable params)
    # Applied after SCF + shell attenuation, before ghost + spectral.
    # All corrections modify bond_energies in-place.
    # ══════════════════════════════════════════════════════════════════
    bond_energies = bond_energies_scf  # alias for ported code


    # ── C1: FACE-FRACTION OVERCROWDING [polyatomic n≥3] ────────────────
    # When face_fraction = (nf + z) / (2P₁) > 1.0 at a vertex,
    # the center polygon is OVER-SATURATED. The π bonds overcounted.
    # Attenuation: exp(-S_ff × (ff-1)) per adjacent bond where S_ff = S₃/P₁.
    if topology.n_atoms >= 3:
        for v in range(topology.n_atoms):
            z_v = topology.z_count[v]
            if z_v < 2:
                continue
            d_v = atom_data[topology.Z_list[v]]
            nf_v = d_v.get('nf', 0)
            face_fraction = (nf_v + z_v) / (2.0 * P1)
            if face_fraction > 1.0:
                # Over-saturated: attenuate all adjacent bonds
                excess_ff = face_fraction - 1.0
                # Count π bonds at this vertex
                n_pi_v = sum(1 for bi, (i, j, bo) in enumerate(topology.bonds)
                             if (i == v or j == v) and bo > 1)
                # Base screening: S₃/P₁ × excess
                S_ff = S3 / P1 * excess_ff
                # π amplification: when ALL bonds are π (SO₃),
                # the π overcounting is quadratic: n_pi × (n_pi-1) / z²
                if n_pi_v >= z_v and z_v >= 2:
                    S_ff *= (1.0 + n_pi_v * (n_pi_v - 1) / (z_v * z_v))
                for bi, (i, j, bo) in enumerate(topology.bonds):
                    if i == v or j == v:
                        bond_energies[bi] *= math.exp(-S_ff)

    # ── C2: LP TERMINAL CROWDING [polyatomic n≥3] ──────────────────────
    # When total terminal LP >> z at a vertex, the LP repel each other
    # through the center polygon → additional screening ∝ (LP_excess/z) × D₃ × s.
    #
    # Super-Bohr halogens (F): LP normally excluded from crowding sum
    # because they can donate into center vacancy. However, when the
    # center LACKS d-vacancy (period-2 atoms: B, C, N, O), the LP has
    # nowhere to go and MUST be counted as crowding.
    # This resolves the CF₄/CBr₄ overbinding asymmetry: CF₄ previously
    # had zero LP crowding (all F excluded) while CBr₄ had 6.5%.
    if topology.n_atoms >= 3:
        for v in range(topology.n_atoms):
            z_v = topology.z_count[v]
            if z_v < 2:
                continue
            # Check center atom properties for super-Bohr LP inclusion gate
            d_v_center = atom_data[topology.Z_list[v]]
            per_v_center = d_v_center['per']
            l_v_center = d_v_center.get('l', 0)
            nf_v_center = d_v_center.get('nf', 0)
            _center_has_d_vacancy = per_v_center >= P1 and l_v_center >= 1
            # Include super-Bohr LP in crowding when:
            # (a) center lacks d-vacancy (LP can't be absorbed), AND
            # (b) center p-shell has > 1 electron (not electron-deficient)
            #     — exempts B (nf_p=1), Al (nf_p=1) whose empty p-orbitals
            #       genuinely absorb F LP (Lewis acid back-donation)
            # (c) center has no LP (LP=0)
            #     — when center already has LP (N in NF₃), terminal LP
            #       crowding is already captured by face fraction (C1).
            #       Including super-Bohr LP here would double-count.
            _lp_v_center = topology.lp[v]
            _include_super_bohr = (not _center_has_d_vacancy
                                   and nf_v_center > 1
                                   and _lp_v_center == 0)
            # Sum LP on adjacent terminals, weighted by orbital compactness
            lp_adj_weighted = 0.0
            lp_adj_raw = 0
            adj_bonds = []
            n_above_bohr = 0
            for bi, (i, j, bo) in enumerate(topology.bonds):
                if i == v or j == v:
                    t_idx = j if i == v else i
                    adj_bonds.append(bi)
                    lp_t = topology.lp[t_idx]
                    lp_adj_raw += lp_t
                    ie_t = atom_data[topology.Z_list[t_idx]]['IE']
                    per_t = atom_data[topology.Z_list[t_idx]]['per']
                    if ie_t > RY * (1 + S3):
                        n_above_bohr += 1
                        if _include_super_bohr:
                            # No d-vacancy + non-deficient: super-Bohr LP
                            # CANNOT be absorbed → include in crowding
                            lp_adj_weighted += lp_t * (2.0 / max(per_t, 2)) ** (1.0 / P1)
                    else:
                        # Sub-Bohr LP weighted by (2/per)^(1/P₁): compact LP crowd more
                        lp_adj_weighted += lp_t * (2.0 / max(per_t, 2)) ** (1.0 / P1)
            # LP crowding: excess weighted LP beyond z
            _S_crowd_applied = 0.0
            if lp_adj_weighted > z_v and adj_bonds:
                f_crowd = (lp_adj_weighted - z_v) / max(lp_adj_weighted, 1)
                # Strength: D₃ × s × f_crowd × (1 + lp_per_term/P₁)
                # The (1 + lp/P₁) factor makes crowding stronger when each terminal
                # has many LP (Cl LP=3 > H LP=0)
                lp_per_term = lp_adj_raw / max(z_v, 1.0)
                _S_crowd_applied = D3 * S_HALF * f_crowd * (1.0 + lp_per_term / P1)
                for bi in adj_bonds:
                    bond_energies[bi] *= math.exp(-_S_crowd_applied)

            # ── C2c: FACE SATURATION PARSEVAL SHARING [z = P_l] ──────
            # When z = P_l at vertex v, the hexagonal face Z/(2P_l)Z is
            # fully occupied by bonds. The Parseval budget constraint
            #   Σ_k |ρ̂(k)|² = 1/(2P_l)
            # means each bond gets 1/z of the total spectral power.
            # If terminals carry LP that crowd the face, the spectral
            # sharing is under-corrected by C2's mean-field.
            #
            # The face LP density:
            #   ρ_LP = lp_adj_weighted / (z × P_l)
            # measures the LP spectral occupancy per face position.
            #
            # Attenuation per bond:
            #   S_sat = (s/P_l) × (z-1)/z × ρ_LP × f_size
            #
            # s/P_l = S_HALF/P1 = 1/6: Parseval exponent on Z/(2P_l)Z.
            # (z-1)/z: sharing fraction (other bonds' contribution).
            # ρ_LP: LP spectral density (0 when no LP, e.g. NH₃).
            # f_size: fades for large molecules where SCF suffices.
            #
            # Gate: z = P_l AND lp_adj_weighted > z (C2 already fires).
            # 0 adjustable parameters.
            if z_v == P1 and lp_adj_weighted > z_v and adj_bonds:
                _P_l_v = {0: 2, 1: P1, 2: P2, 3: P3}.get(
                    d_v_center.get('l', 0), P1)
                if z_v == _P_l_v:
                    _rho_lp = lp_adj_weighted / (z_v * _P_l_v)
                    _f_size_sat = max(
                        0.0,
                        1.0 - float(topology.n_atoms - P1) / max(P3 - P1, 1))
                    _S_sat = ((S_HALF / _P_l_v)
                              * float(z_v - 1) / z_v
                              * _rho_lp
                              * _f_size_sat)
                    if _S_sat > 0:
                        for bi in adj_bonds:
                            bond_energies[bi] *= math.exp(-_S_sat)

            # ── F-donation boost: when ALL terminals are super-Bohr (F) ──
            # F LP donates INTO center vacancy → bond strengthened.
            lp_v = topology.lp[v]
            d_v = atom_data[topology.Z_list[v]]
            per_v = d_v['per']
            l_v = d_v.get('l', 0)
            has_d_vacancy = per_v >= P1 and l_v >= 1  # d-shell accessible (Si, P, S)
            has_s_vacancy = l_v == 0  # s-block (Be)
            has_p_vacancy = l_v == 1 and d_v['IE'] < RY  # sub-Bohr p-block acceptor (B, Al)
            if n_above_bohr == z_v and lp_v == 0 and (has_d_vacancy or has_s_vacancy or has_p_vacancy) and adj_bonds:
                nf_v = d_v.get('nf', 0)
                if has_d_vacancy:
                    vacancy_v = max(0.0, (2.0 * P2 - nf_v)) / (2.0 * P2)
                else:
                    vacancy_v = max(0, 2 * P1 - round(topology.sum_bo[v]) - 2 * lp_v) / (2 * P1)
                if vacancy_v > 0:
                    lp_per_term = lp_adj_raw / max(z_v, 1)
                    boost = min(lp_per_term, 3) * vacancy_v * S_HALF / P2 * (RY / P2)
                    for bi in adj_bonds:
                        bond_energies[bi] += boost

            # ── C2b: HALOGEN LP PAIR ENHANCEMENT [beyond mean-field] ──
            # C2 treats LP crowding as mean-field (linear in LP excess).
            # For n_hal ≥ 2 halogens at a vertex, LP-LP repulsion is
            # quadratic: n_hal×(n_hal−1)/2 pair interactions on Z/(2P₁)Z.
            # C2b adds the EXCESS of the pair interaction above C2's
            # mean-field: S_extra = max(0, S_pair − S_crowd_C2).
            #
            # S_pair = [n_pairs / z] × [lp/(2P₁)]² / P₁
            #
            # Suppressed when:
            # - center has d-vacancy (LP absorbed into d-shell)
            # - center is electron-deficient (nf_p ≤ 1: B, Al)
            # - center has own LP (crowding already in C1 face fraction)
            # 0 adjustable parameters.
            if (not _center_has_d_vacancy
                    and nf_v_center > 1
                    and _lp_v_center == 0
                    and adj_bonds):
                _n_hal_v = 0
                for bi in adj_bonds:
                    ii, jj, _bo = topology.bonds[bi]
                    t_idx = jj if ii == v else ii
                    Z_t = topology.Z_list[t_idx]
                    if Z_t in (9, 17, 35, 53):
                        _n_hal_v += 1
                if _n_hal_v >= 2:
                    _n_pairs = _n_hal_v * (_n_hal_v - 1) / 2.0
                    # C2b fix [P2b]: per-dependent effective LP for
                    # inter-terminal pair overlap.  Heavier halogens
                    # (larger per) have more diffuse LP clouds that
                    # overlap MORE with neighboring terminals.
                    # lp_eff = lp × (per/P₀)^(1/P₁)
                    #   F(per=2): 1.0 (baseline, unchanged)
                    #   Cl(per=3): (3/2)^(1/3) = 1.145
                    #   Br(per=4): (4/2)^(1/3) = 1.260
                    #   I(per=5):  (5/2)^(1/3) = 1.357
                    # 0 adjustable parameters: P₀, P₁ from s = 1/2.
                    _lp_hal_sum = 0.0
                    for _bi_hal in adj_bonds:
                        _ii_h, _jj_h, _ = topology.bonds[_bi_hal]
                        _t_h = _jj_h if _ii_h == v else _ii_h
                        _Z_h = topology.Z_list[_t_h]
                        if _Z_h in (9, 17, 35, 53):
                            _per_h = atom_data[_Z_h]['per']
                            _lp_h = topology.lp[_t_h]
                            _lp_hal_sum += float(_lp_h) * (float(_per_h) / P0) ** (1.0 / P1)
                    _lp_hal = _lp_hal_sum / _n_hal_v
                    _S_pair = (_n_pairs / z_v
                               * (_lp_hal / (2.0 * P1)) ** 2
                               / P1)
                    _S_extra = max(0.0, _S_pair - _S_crowd_applied)
                    if _S_extra > 0:
                        for bi in adj_bonds:
                            bond_energies[bi] *= math.exp(-_S_extra)

    # ── C3: NLO PERTURBATIVE VERTEX COUPLING [2nd order on T³] ───────
    # At LO each bond is independent. At NLO, pairs of bonds at the same
    # vertex couple through the polygon Z/(2P₁)Z.
    # E_NLO(v) = -Σᵢ<ⱼ Vᵢⱼ² / (ΔEᵢⱼ + Vᵢⱼ)     [2nd order perturbation]
    #
    # SPECTRAL SCF MODULATION: the coupling strength V is modulated by
    # the cross-spectrum competition metric from the spectral SCF.
    # High competition (bonds share exchange modes) → V amplified
    # Low competition (orthogonal modes) → V reduced
    # This replaces the uniform D₃/(z×P₁) coupling with a mode-aware one.
    _has_spectral_comp = hasattr(topology, '_spectral_competition')
    if topology.n_atoms >= 3:
        ring_bond_set = topology.ring_bonds or set()
        nlo_correction = {}  # bi → total NLO correction (negative)
        for v in range(topology.n_atoms):
            z_v = topology.z_count[v]
            if z_v < 2:
                continue
            v_bonds = topology.vertex_bonds.get(v, [])
            if len(v_bonds) < 2:
                continue
            # Linear vertices: z=2 + no LP + two multi-bonds = orthogonal π planes.
            # The π-modes are genuinely orthogonal (allene geometry: π₁⊥π₂),
            # so the π-NLO coupling vanishes.  However, both bonds share
            # the SAME σ-mode (k=0 on Z/(2P₁)Z).  The σ-channel coupling
            # is a fraction 1/bo_min of the total, since only 1 out of
            # bo_min spectral modes is the shared σ-mode.
            #
            # f_sigma_linear = 1 / bo_min  (0 adjustable parameters)
            #   bo_min=2 (cumulated doubles): f = 1/2 (half-strength NLO)
            #   bo_min=3 (cumulated triples): f = 1/3 (third-strength NLO)
            lp_v = topology.lp[v]
            per_v = atom_data[topology.Z_list[v]]['per']
            _f_sigma_linear = 1.0
            if z_v == 2 and lp_v == 0:
                bos_v = [topology.bonds[bi][2] for bi in v_bonds]
                if min(bos_v) > 1:
                    _f_sigma_linear = 1.0 / min(bos_v)

            # Spectral competition modulation factor for this vertex.
            # The NLO coupling V between bonds is amplified when the bonds
            # share exchange modes (high competition) and reduced when
            # they are spectrally orthogonal (low competition).
            #
            # f_comp = C₃ + S₃ × competition²
            # When competition = 0: f_comp = C₃ ≈ 0.781 (geometric coupling only)
            # When competition = 1: f_comp = 1.0 (full coupling)
            #
            # The square on competition is the PT 2nd order effect:
            # the coupling V enters as V² in the 2nd order perturbation,
            # so the competition metric should also be squared.
            #
            # PT derivation: C₃ = cos²(θ₃) is the base geometric coupling
            # that persists even for orthogonal modes (through the polygon
            # geometry). S₃ = sin²(θ₃) is the mode-dependent part that
            # requires spectral overlap.
            if _has_spectral_comp:
                comp_v = topology._spectral_competition[v]
                f_comp = C3 + S3 * comp_v * comp_v
            else:
                f_comp = 1.0  # no spectral data: full coupling (backward compat)

            # Collect bond energies at vertex
            v_energies = [(bi, bond_energies.get(bi, 0.0)) for bi in v_bonds]
            E_v_total = sum(e for _, e in v_energies)
            if E_v_total <= 0:
                continue
            # Pairwise NLO coupling
            E_nlo_v = 0.0
            for a in range(len(v_energies)):
                bi_a, D_a = v_energies[a]
                if D_a <= 0:
                    continue
                for b in range(a + 1, len(v_energies)):
                    bi_b, D_b = v_energies[b]
                    if D_b <= 0:
                        continue
                    # Coupling strength, modulated by spectral competition
                    # and σ-fraction for linear cumulated vertices
                    V = D3 * math.sqrt(D_a * D_b) / (z_v * P1) * f_comp * _f_sigma_linear
                    # Ring enhancement: both bonds in same ring → topological mode
                    if bi_a in ring_bond_set and bi_b in ring_bond_set:
                        V *= math.sqrt(P1)
                    delta_E = abs(D_a - D_b)
                    # Degeneracy correction [C16]: at non-ring vertices
                    # with z ≥ 3 and no LP, degenerate bonds split
                    # SYMMETRICALLY into bonding and antibonding levels.
                    # The total occupied energy is conserved (no net
                    # loss). Only non-degenerate pairs give a net
                    # 2nd-order shift.
                    #
                    # Gate: z ≥ 3 (multi-bond cooperation), lp = 0
                    # (bonding vertex), per ≤ 2 (compact sp orbitals —
                    # diffuse per ≥ 3 orbitals compete even when
                    # degenerate), NEITHER bond in a ring (ring NLO
                    # represents delocalization, not splitting).
                    #
                    # f_degen = ΔE/(ΔE + V) → 0 for degenerate,
                    #                         → 1 for non-degenerate.
                    #
                    # 0 adjustable parameters. All from s = 1/2.
                    f_degen = 1.0
                    if (z_v >= P1 and lp_v == 0
                            and per_v <= 2
                            and bi_a not in ring_bond_set
                            and bi_b not in ring_bond_set):
                        f_degen = delta_E / (delta_E + V) if V > 1e-12 else 1.0
                    E_pair = -(V * V) / (delta_E + V) * f_degen
                    E_nlo_v += E_pair
            if lp_v > 0:
                D_lp = S3 * RY / P1
                for bi_lp, D_b in v_energies:
                    if D_b <= 0:
                        continue
                    V_lp = D3 * math.sqrt(max(D_lp * D_b, 0.01)) / (z_v * P1)
                    delta_lp = abs(D_b - D_lp)
                    E_lp = -lp_v * (V_lp * V_lp) / (delta_lp + V_lp)
                    nlo_correction[bi_lp] = nlo_correction.get(bi_lp, 0.0) + E_lp
            # Distribute correction proportionally to bond energy
            if E_nlo_v < 0:
                for bi, D_bi in v_energies:
                    frac = D_bi / E_v_total
                    nlo_correction[bi] = nlo_correction.get(bi, 0.0) + E_nlo_v * frac
        # Apply NLO corrections
        for bi, corr in nlo_correction.items():
            if corr < 0:
                bond_energies[bi] = max(bond_energies[bi] + corr, 0.01)



    # ── C4: P₃ DARK CLUSTER QUENCHING [v40, 0 params] ────────────────
    # When ≥2 heavy atoms in an aromatic ring are P₃-invisible (Z%7=0),
    # the heptagonal face of T³ cannot resolve them → screening deficit.
    _P3_DARK_COUPLING = S3 * C7  # 0.219 × 0.827 = 0.181, derived
    if topology.rings and topology.n_atoms >= 4:
        ring_bond_set = topology.ring_bonds or set()
        for ring in topology.rings:
            N_ring = len(ring)
            if N_ring < 3:
                continue
            # Check for aromatic ring (any bond with bo ≈ 1.5)
            is_aro_ring = False
            ring_set = set(ring)
            ring_bi_list: List[int] = []
            for bi_r in ring_bond_set:
                i_r, j_r, bo_r = topology.bonds[bi_r]
                if i_r in ring_set and j_r in ring_set:
                    ring_bi_list.append(bi_r)
                    if abs(bo_r - 1.5) < 0.01:
                        is_aro_ring = True
            if not is_aro_ring:
                continue
            # Count P₃-invisible heavy atoms (Z % P₃ == 0, Z > 1)
            n_dark = sum(
                1 for v in ring
                if topology.Z_list[v] > 1 and topology.Z_list[v] % P3 == 0
            )
            if n_dark < 2:
                continue
            # Quench: S₃ × C₇ × (n_dark / N_ring)
            f_dark = n_dark / N_ring
            quench = _P3_DARK_COUPLING * f_dark
            for bi_r in ring_bi_list:
                i_r, j_r, bo_r = topology.bonds[bi_r]
                if abs(bo_r - 1.5) < 0.01:  # aromatic bonds only
                    old = bond_energies.get(bi_r, 0.0)
                    if old > 0:
                        bond_energies[bi_r] = old * (1.0 - quench)

    # ── C4b: HALOGEN LP DELOCALIZATION [collective, 0 params] ─────────
    # When halogens are distributed across a ring or adjacent centres,
    # their LP clouds delocalise collectively through the P₂ channel.
    # Per-vertex C2/C2b captures LOCAL LP crowding; C4b captures the
    # COLLECTIVE part that C2b misses when n_hal_v < 2 at each vertex.
    #
    # Physics: on T³ = Z/3Z × Z/5Z × Z/7Z, halogen LP on adjacent
    # vertices are phase-locked on the pentagonal face (Z mod 5Z).
    # The collective screening is:
    #   S = S₃ × (n_hal / N_ring) × min(lp_avg / P₁, 1)
    # applied as exp(-S) to all bonds at ring vertices, capped at S₃.
    #
    # For non-ring systems (C₂F₄, C₂Cl₄): when two adjacent heavy
    # atoms both carry halogen terminals, their LP clouds interfere
    # across the connecting bond → same screening formula applied to
    # that bond, with f_hal = sqrt(f_i × f_j).
    #
    # Gate: n_hal ≥ 2 total (distributed), EXCLUDES pure benzene/etc.
    # 0 adjustable parameters: S₃, P₁, D₃ all from s = 1/2.
    _HALOGENS_SET = frozenset({9, 17, 35, 53})
    _c4b_ring_vertices: set = set()  # vertices screened by ring Part 1
    if topology.n_atoms >= 4:
        # --- Part 1: RING LP delocalization ---
        if topology.rings:
            ring_bond_set = topology.ring_bonds or set()
            _ring_screened_bonds: set = set()
            for ring in topology.rings:
                N_ring = len(ring)
                if N_ring < 3:
                    continue
                ring_set = set(ring)
                # Collect ring bonds
                ring_bi_list: List[int] = []
                for bi_r in ring_bond_set:
                    i_r, j_r, bo_r = topology.bonds[bi_r]
                    if i_r in ring_set and j_r in ring_set:
                        ring_bi_list.append(bi_r)
                if not ring_bi_list:
                    continue
                # Count halogen terminals on ring vertices
                # A halogen "on" the ring = bonded to a ring vertex but NOT in the ring
                n_hal_ring = 0
                lp_hal_sum = 0.0
                for v_ring in ring:
                    for bk in topology.vertex_bonds.get(v_ring, []):
                        a_k, b_k, _ = topology.bonds[bk]
                        nb = b_k if a_k == v_ring else a_k
                        if nb not in ring_set and topology.Z_list[nb] in _HALOGENS_SET:
                            n_hal_ring += 1
                            lp_hal_sum += topology.lp[nb]
                if n_hal_ring < 2:
                    continue
                # Collective screening: S₃ × (n_hal/N) × min(lp_avg/P₁, 1)
                lp_avg = lp_hal_sum / n_hal_ring
                f_hal = n_hal_ring / N_ring
                S_ring = S3 * f_hal * min(lp_avg / P1, 1.0)
                S_ring = min(S_ring, S3)  # Shannon cap: one pass of P₁ filter
                if S_ring > 1e-12:
                    atten_ring = math.exp(-S_ring)
                    # Exocyclic attenuation: LP transport cost D₃ reduces
                    # the screening on bonds leaving the ring.
                    atten_exo = math.exp(-S_ring * D3)
                    # Apply to ring bonds (full screening)
                    for bi_r in ring_bi_list:
                        old = bond_energies.get(bi_r, 0.0)
                        if old > 0 and bi_r not in _ring_screened_bonds:
                            bond_energies[bi_r] = old * atten_ring
                            _ring_screened_bonds.add(bi_r)
                    # Apply to exocyclic C-X bonds at halogenated vertices
                    for v_ring in ring:
                        for bk in topology.vertex_bonds.get(v_ring, []):
                            if bk in _ring_screened_bonds:
                                continue
                            a_k, b_k, _ = topology.bonds[bk]
                            nb = b_k if a_k == v_ring else a_k
                            if nb not in ring_set:
                                old = bond_energies.get(bk, 0.0)
                                if old > 0:
                                    bond_energies[bk] = old * atten_exo
                                    _ring_screened_bonds.add(bk)
                    _c4b_ring_vertices.update(ring)

        # --- Part 2: CHAIN LP delocalization (non-ring) ---
        # For bonds connecting two heavy atoms that both carry halogen
        # terminals (e.g. C₂F₄: F₂C=CF₂), the LP clouds from both
        # sides create a collective screening field around BOTH vertices.
        # Screen all bonds at both vertices, not just the connector.
        #
        # Gate: the connecting bond must have π character (bo > 1).
        # For σ-only bonds (bo = 1), per-vertex C2b is sufficient.
        # LP delocalization through a π bond is efficient (π cloud
        # provides a channel for LP interference); through σ only,
        # the LP are localised and C2b handles them per-vertex.
        _chain_screened_bonds: set = set()
        for bi, (i, j, bo) in enumerate(topology.bonds):
            # Skip if both vertices already screened by ring Part 1
            if i in _c4b_ring_vertices and j in _c4b_ring_vertices:
                continue
            # π gate: LP delocalisation requires π channel
            if bo <= 1.0:
                continue
            Zi, Zj = topology.Z_list[i], topology.Z_list[j]
            # Both endpoints must be heavy (not H, not halogen themselves)
            if Zi <= 1 or Zj <= 1:
                continue
            if Zi in _HALOGENS_SET or Zj in _HALOGENS_SET:
                continue
            z_i, z_j = topology.z_count[i], topology.z_count[j]
            if z_i < 2 or z_j < 2:
                continue
            # Count halogen terminals at each side
            n_hal_i, lp_sum_i = 0, 0.0
            for bk in topology.vertex_bonds.get(i, []):
                if bk == bi:
                    continue
                a_k, b_k, _ = topology.bonds[bk]
                nb = b_k if a_k == i else a_k
                if topology.Z_list[nb] in _HALOGENS_SET:
                    n_hal_i += 1
                    lp_sum_i += topology.lp[nb]
            n_hal_j, lp_sum_j = 0, 0.0
            for bk in topology.vertex_bonds.get(j, []):
                if bk == bi:
                    continue
                a_k, b_k, _ = topology.bonds[bk]
                nb = b_k if a_k == j else a_k
                if topology.Z_list[nb] in _HALOGENS_SET:
                    n_hal_j += 1
                    lp_sum_j += topology.lp[nb]
            if n_hal_i < 1 or n_hal_j < 1:
                continue
            # Both sides have halogens → cross-bond LP delocalization
            f_i = n_hal_i / z_i
            f_j = n_hal_j / z_j
            lp_avg = (lp_sum_i + lp_sum_j) / (n_hal_i + n_hal_j)
            f_cross = math.sqrt(f_i * f_j)
            S_cross = S3 * f_cross * min(lp_avg / P1, 1.0)
            S_cross = min(S_cross, S3)  # Shannon cap
            if S_cross > 1e-12:
                atten = math.exp(-S_cross)
                # Screen all bonds at both vertices
                for v_chain in (i, j):
                    for bk in topology.vertex_bonds.get(v_chain, []):
                        if bk not in _chain_screened_bonds:
                            old = bond_energies.get(bk, 0.0)
                            if old > 0:
                                bond_energies[bk] = old * atten
                                _chain_screened_bonds.add(bk)

    # ── C5: T³ PERTURBATIVE SCREENING [Z mod {3,5,7}, 0 params] ──────
    # Cross-vertex NLO correction on T³ = Z/3Z × Z/5Z × Z/7Z.
    if topology.n_atoms >= 3:
        _crt = [(Z % P1, Z % P2, Z % P3) for Z in topology.Z_list]
        for bi, (i, j, bo) in enumerate(topology.bonds):
            D_bi = bond_energies.get(bi, 0.0)
            if D_bi <= 0.0:
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
                        occ_k = math.sin(_TWO_PI * r_nb_k / Pk) ** 2 if r_nb_k else 0.0
                        if occ_k == 0.0:
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
            # Perturbative: D₃ = one-loop attenuation on T³ hexagonal face
            delta = 1.0 - surv_total
            bond_energies[bi] = D_bi * (1.0 - delta * D3)

    # ── C6: DICKE VERTEX COHERENCE [T1 on molecular graph] ───────────
    # On T³, z identical partners at vertex v form a coherent Dicke state.
    if not topology.is_diatomic:
        for v in range(topology.n_atoms):
            z_v = topology.z_count[v]
            if z_v < 2:
                continue
            v_bonds_d = topology.vertex_bonds.get(v, [])
            if len(v_bonds_d) < 2:
                continue
            # Count same-Z clusters
            partner_Z_count: Dict[int, int] = {}
            for bk in v_bonds_d:
                ii_d, jj_d, _ = topology.bonds[bk]
                p_d = jj_d if ii_d == v else ii_d
                Zp = topology.Z_list[p_d]
                partner_Z_count[Zp] = partner_Z_count.get(Zp, 0) + 1
            # Dicke boost for each cluster of size ≥ 2
            for Zp_cl, n_cl in partner_Z_count.items():
                if n_cl < 2:
                    continue
                # Coherent fraction: D₃² = 4th cumulant on Z/3Z (Fisher²)
                f_dicke = D3 * D3 * (n_cl - 1.0) / (z_v * P1)
                # Apply boost to each bond in the cluster
                for bk in v_bonds_d:
                    ii_d, jj_d, _ = topology.bonds[bk]
                    p_d = jj_d if ii_d == v else ii_d
                    if topology.Z_list[p_d] == Zp_cl:
                        D_old = bond_energies.get(bk, 0.0)
                        if D_old > 0:
                            bond_energies[bk] = D_old * (1.0 + f_dicke)

    # ── C7: π-DELOCALIZATION COOPERATIVE [multi-π vertex mode k≥1] ───
    # Vertices with n_multi ≥ 2 π bonds: π electrons delocalize across bonds.
    if not topology.is_diatomic:
        for v in range(topology.n_atoms):
            z_v = topology.z_count[v]
            if z_v < 2:
                continue
            # Count multi-bonds at this vertex
            multi_bonds = []
            for bk in topology.vertex_bonds.get(v, []):
                _, _, bo_k = topology.bonds[bk]
                if bo_k > 1.0:
                    multi_bonds.append(bk)
            n_multi = len(multi_bonds)
            if n_multi < 2:
                continue
            lp_v = topology.lp[v]
            # Only for bent/sp² vertices (lp > 0 or radical)
            Zv = topology.Z_list[v]
            rad_v = (valence_electrons(Zv) - round(topology.sum_bo[v])) % 2 == 1
            if lp_v == 0 and not rad_v:
                continue  # linear → π in orthogonal planes, no sharing
            # Cooperative bonus per π bond: D₅² = Fisher² on pentagonal face
            f_coop = D5 * D5 * (n_multi - 1.0) / (n_multi * P2)
            for bk in multi_bonds:
                D_old = bond_energies.get(bk, 0.0)
                if D_old > 0:
                    bond_energies[bk] = D_old * (1.0 + f_coop)

    # ── C7b: PI-ORTHOGONAL SCREENING [cumulene vertices, 0 params] ────
    # At a linear vertex with 2 adjacent double bonds (cumulene centre,
    # e.g. central C in allene C=C=C), the two pi systems lie in
    # ORTHOGONAL planes.  This screens the bond energy by the geometric
    # mismatch fraction f_ortho = D3 * sin^2(pi/P1) = D3 * 3/4.
    # PT derivation (C13): 2-body geometric effect in CRT P1=3 channel.
    # sin^2(pi/P1) = sin^2(60 deg) = 3/4 is the maximal angular mismatch
    # on the hexagonal face Z/(2P1)Z.  [0 adjustable parameters]
    #
    # Gates: (1) not ring atom (Kekule alternation != cumulene),
    #        (2) z=2, lp=0, both bonds multi-bond (bo > 1),
    #        (3) skip bonds to LP-bearing partners (LP already screens),
    #        (4) skip bonds to d-block partners (back-bonding, not ortho).
    # Each bond screened at most once (tracked via _screened_bonds).
    _SIN2_PI_P1 = 0.75  # sin^2(pi/3) = 3/4, derived
    _F_PI_ORTHO = D3 * _SIN2_PI_P1  # approx 0.0873, 0 params
    if not topology.is_diatomic:
        _ring_atoms_c7b = topology.ring_atoms or set()
        _cumul_verts = set()
        for v in range(topology.n_atoms):
            if v in _ring_atoms_c7b:
                continue
            if topology.z_count[v] != 2 or topology.lp[v] > 0:
                continue
            vb = topology.vertex_bonds.get(v, [])
            if len(vb) != 2:
                continue
            if min(topology.bonds[bi][2] for bi in vb) > 1.0:
                _cumul_verts.add(v)
        _screened_bonds_c7b = set()
        for v in _cumul_verts:
            for bi_v in topology.vertex_bonds.get(v, []):
                if bi_v in _screened_bonds_c7b:
                    continue
                i_b, j_b, _ = topology.bonds[bi_v]
                p = j_b if i_b == v else i_b
                if topology.lp[p] > 0:
                    continue
                if atom_data[topology.Z_list[p]].get('is_d_block', False):
                    continue
                D_old = bond_energies.get(bi_v, 0.0)
                if D_old > 0:
                    bond_energies[bi_v] = D_old * (1.0 - _F_PI_ORTHO)
                    _screened_bonds_c7b.add(bi_v)

    # ── C10: THROUGH-BOND pi STABILISATION [polyatomic L4, 0 params] ──
    # Generalises the triatomic L4 mechanism to polyatomic chains.
    # When a triple bond (bo≥3) shares a vertex with a single bond (bo≤1),
    # the triple bond's π energy stabilises the single bond partner
    # through σ backbone confinement on Z/(2P₁)Z.
    #
    # PT derivation: the triple bond's k≥1 modes on the hexagonal face
    # create a confining potential at the shared vertex. The single bond's
    # k=0 (σ) mode is rigidified, reducing its screening.
    # The stabilisation energy uses the v4 DIATOMIC D_P1 reference:
    #   ΔD = D_P1_v4(triple) × (bo-1)/bo × S₃ × s × f_deficit
    # where f_deficit = max(0, 1 - D_current/D_v4) clamps the boost:
    # bonds already at or above their v4 diatomic value get no boost.
    # This avoids over-boosting molecules where the polyatomic pipeline
    # already captures the cooperative energy.
    #
    # Gate: vertex z=2 (chain), lp=0 (linear sp), one bond bo≥3,
    # the other bo≤1 (single bond). Both bonds at the vertex must have
    # lp=0 neighbors (no LP suppression of sigma confinement).
    # 0 adjustable parameters.
    if not topology.is_diatomic and topology.n_atoms >= 4:
        # Pre-compute v4 diatomic references for deficit gating
        _v4_diat_refs: Dict[int, float] = {}
        for bi_ref, (i_ref, j_ref, bo_ref) in enumerate(topology.bonds):
            sr_ref = _D0_screening_v4(topology.Z_list[i_ref], topology.Z_list[j_ref],
                                      bo_ref, lp_A=topology.lp[i_ref],
                                      lp_B=topology.lp[j_ref])
            _v4_diat_refs[bi_ref] = sr_ref.D0

        for v in range(topology.n_atoms):
            z_v = topology.z_count[v]
            if z_v != 2:
                continue
            lp_v = topology.lp[v]
            if lp_v > 0:
                continue
            v_bonds_l4 = topology.vertex_bonds.get(v, [])
            if len(v_bonds_l4) != 2:
                continue
            bi_a, bi_b = v_bonds_l4[0], v_bonds_l4[1]
            bo_a = topology.bonds[bi_a][2]
            bo_b = topology.bonds[bi_b][2]
            # One must be triple (bo≥3), the other single (bo≤1)
            if bo_a >= 3.0 and bo_b <= 1.0:
                bi_triple, bi_single = bi_a, bi_b
                bo_triple = bo_a
            elif bo_b >= 3.0 and bo_a <= 1.0:
                bi_triple, bi_single = bi_b, bi_a
                bo_triple = bo_b
            else:
                continue
            # Use v4 diatomic D_P1 for the triple bond
            i_t, j_t, _ = topology.bonds[bi_triple]
            Zi_t, Zj_t = topology.Z_list[i_t], topology.Z_list[j_t]
            sr_v4_triple = _D0_screening_v4(Zi_t, Zj_t, bo_triple,
                                            lp_A=topology.lp[i_t],
                                            lp_B=topology.lp[j_t])
            pi_frac = (bo_triple - 1.0) / bo_triple
            D_stab_max = sr_v4_triple.D_P1 * pi_frac * S3 * S_HALF
            # Deficit gating: only boost up to the v4 diatomic level
            D_current = bond_energies.get(bi_single, 0.0)
            D_v4_ref = _v4_diat_refs.get(bi_single, D_current)
            f_deficit = max(0.0, 1.0 - D_current / max(D_v4_ref, 1e-10))
            D_stab = D_stab_max * f_deficit
            if D_current > 0 and D_stab > 0:
                bond_energies[bi_single] = D_current + D_stab

    # ── C8: GFT P₁/P₂ ANTI-DOUBLE-COUNTING [Principle 7 on z > P₁] ─
    # When z > P₁ (hypercoordinated), excess bonds live on the P₂ face.
    if not topology.is_diatomic:
        for v in range(topology.n_atoms):
            z_v = topology.z_count[v]
            per_v = atom_data[topology.Z_list[v]]['per']
            if z_v <= P1 or per_v < P1:
                continue  # not hypercoordinated or period < 3
            n_excess = z_v - P1
            f_gft = D3 * D3 * float(n_excess) / (z_v * P1)
            for bk in topology.vertex_bonds.get(v, []):
                D_old = bond_energies.get(bk, 0.0)
                if D_old > 0:
                    bond_energies[bk] = D_old * (1.0 + f_gft)

    # ── C9: RING HOLONOMY T³ [Z mod {3,5,7} frustration] ─────────────
    # Transport parallèle sur T³ : chaque arête (i→j) du cycle accumule
    # une phase sin²(2π·Δr_k/P_k) par face k.
    E_ring_holo = 0.0
    if topology.rings and topology.n_atoms >= 4:
        ring_bond_set = topology.ring_bonds or set()
        n_cyclomatic = max(len(topology.bonds) - topology.n_atoms + 1, 1)
        _crt_holo = [(Z % P1, Z % P2, Z % P3) for Z in topology.Z_list]
        for ring in topology.rings:
            N_ring = len(ring)
            if N_ring < 3:
                continue
            is_aro_ring = False
            ring_bi_list = []
            ring_set = set(ring)
            for bi_r in ring_bond_set:
                i_r, j_r, bo_r = topology.bonds[bi_r]
                if i_r in ring_set and j_r in ring_set:
                    ring_bi_list.append(bi_r)
                    if abs(bo_r - 1.5) < 0.01:
                        is_aro_ring = True
            if is_aro_ring:
                continue
            D_ring_sum = sum(bond_energies.get(bi_r, 0.0) for bi_r in ring_bi_list)
            if D_ring_sum <= 0:
                continue
            # T³ holonomy: frustration per face
            H = [0.0, 0.0, 0.0]
            for idx_r in range(N_ring):
                a = ring[idx_r]
                b = ring[(idx_r + 1) % N_ring]
                ra, rb = _crt_holo[a], _crt_holo[b]
                for k, Pk in enumerate((P1, P2, P3)):
                    dr = abs(ra[k] - rb[k])
                    if dr > 0:
                        H[k] += math.sin(_TWO_PI * dr / Pk) ** 2
            # Two sources of ring frustration (complementary):
            # 1. Geometric: all-C ring strain from angle mismatch → sin²(π/N)
            # 2. Composition: T³ frustration from heterogeneous Z mod {3,5,7}
            n_C_ring = sum(1 for k in ring if topology.Z_list[k] == 6)
            f_homo_geom = (float(n_C_ring) / N_ring) ** 2
            _S_SUM = S3 + S5 + S7
            f_T3_comp = (S3 * H[0] + S5 * H[1] + S7 * H[2]) / (N_ring * _S_SUM)
            # Combine: geometric strain + composition frustration
            f_ring = max(f_homo_geom, f_T3_comp)
            f_residual = (N_ring - 2.0) / N_ring
            if f_ring * f_residual < 1e-12:
                continue
            holo = (
                -S3
                * math.sin(math.pi / N_ring) ** 2
                * D_ring_sum
                * f_ring
                * f_residual
                / n_cyclomatic
            )
            E_ring_holo += holo

    # ── C9c: SUB-SATURATED s¹ RING POLYGON OVERRIDE ──────────────────
    # When every ring atom carries outer ns¹ with no np valence, the
    # σ ring system is electron-deficient: only ⌊n_e/2⌋ delocalized
    # pairs are available for N edges, so the per-edge sum of
    # bond_energies over-counts a 2c-2e localized picture that does
    # not exist. The PT-pure unit is the polygon P_N itself, not the
    # edge: cycle-Hückel modes ε_k = α - 2t cos(2πk/N) carry the
    # bonding, calibrated by β = D_dimer / 2.
    #
    # Override the in-ring edge-sum by:
    #     D_polygon = D_dimer · Σ_k n_k cos(2πk/N)
    # with Aufbau filling (largest cos first), n_e = N (one s¹
    # electron per atom, neutral ring assumption).
    #
    # Gate: every ring atom is Group 1 alkali (Li/Na/K/Rb/Cs/Fr) or
    # Group 11 coinage (Cu/Ag/Au) — i.e. ns=1, np=0. Ring size 3..6.
    s1_rings_handled: set = set()
    if topology.rings and topology.n_atoms >= 3:
        ring_bond_set_c9c = topology.ring_bonds or set()
        for ring in topology.rings:
            N_ring = len(ring)
            if not (3 <= N_ring <= 6):
                continue
            Zs_r = [topology.Z_list[k] for k in ring]
            if not all(_is_s1_outer(z) for z in Zs_r):
                continue
            ring_set = set(ring)
            ring_bis = []
            for bi_r in ring_bond_set_c9c:
                i_r, j_r, _ = topology.bonds[bi_r]
                if i_r in ring_set and j_r in ring_set:
                    ring_bis.append(bi_r)
            if not ring_bis:
                continue
            edge_sum = sum(bond_energies.get(bi_r, 0.0) for bi_r in ring_bis)
            if edge_sum <= 0:
                continue
            # σ-electron count: 1 per s¹ atom, neutral ring
            n_e = N_ring
            mult = _huckel_polygon_multiplier(N_ring, n_e)
            # D_dimer reference: homonuclear → unique; heteronuclear →
            # average over the distinct (Zi,Zj) pairs of in-ring edges
            unique_Z = set(Zs_r)
            if len(unique_Z) == 1:
                zh = next(iter(unique_Z))
                D_dim = _dimer_D_cached(zh, zh)
            else:
                pairs = set()
                for bi_r in ring_bis:
                    i_r, j_r, _ = topology.bonds[bi_r]
                    pairs.add((min(topology.Z_list[i_r], topology.Z_list[j_r]),
                               max(topology.Z_list[i_r], topology.Z_list[j_r])))
                ds = [_dimer_D_cached(za, zb) for za, zb in pairs]
                D_dim = sum(ds) / len(ds) if ds else 0.0
            if D_dim <= 0:
                continue
            D_polygon = mult * D_dim
            E_ring_holo += D_polygon - edge_sum
            s1_rings_handled.add(tuple(ring))

    # ── C9b: σ-AROMATICITY [non-carbon metal rings] ──────────────────
    # Aromaticity = holonomic phase condition on a closed cycle of T³.
    # _huckel_aromatic handles π-aromaticity via SMILES bo=1.5 gate.
    # For non-C metal rings (Al₃⁻, Bi₃³⁻, Hg₄²⁻, mixed Zintl clusters,
    # actinide-capped rings…) the σ channel on Z/(2P₁)Z supports the
    # same Hückel-like delocalization — SMILES emits no bo=1.5 flag for
    # metals, so this block catches them.
    #
    # The stabilisation is modulated by the T³ Fourier coherence of the
    # ring composition, f_coh ∈ [0,1]:
    #     f_coh = (S₃·C₁ + S₅·C₂ + S₇·C₃) / (S₃+S₅+S₇)
    #     C_k   = |1/N · Σ_a exp(2π i · (Z_a mod P_k) / P_k)|²
    # f_coh = 1 for strict homonuclear (all residues aligned on every
    # T³ face); f_coh < 1 proportionally as the composition dephases.
    # No arbitrary threshold — the PT-pure modulator carries the physics.
    #
    # Gate (narrow, verified 0/806 overlap with existing bench):
    #   - ring size 3 ≤ N ≤ 6
    #   - no carbon or hydrogen in ring (C/H handled elsewhere)
    #   - every ring atom has period ≥ 3 (σ-aromaticity regime)
    #   - ring not SMILES-aromatic (bo ≠ 1.5 everywhere)
    #
    # Stabilisation per ring: +S₃ · sin²(π/N) · Σ_bonds D_bond · f_coh
    if topology.rings and topology.n_atoms >= 3:
        ring_bond_set = topology.ring_bonds or set()
        for ring in topology.rings:
            if tuple(ring) in s1_rings_handled:
                continue  # C9c already handled this ring
            N_ring = len(ring)
            if not (3 <= N_ring <= 6):
                continue
            Zs = [topology.Z_list[k] for k in ring]
            if any(z in (1, 6) for z in Zs):
                continue
            if any(period(z) < 3 for z in Zs):
                continue
            ring_set = set(ring)
            ring_bis = []
            is_aro_ring = False
            for bi_r in ring_bond_set:
                i_r, j_r, bo_r = topology.bonds[bi_r]
                if i_r in ring_set and j_r in ring_set:
                    ring_bis.append(bi_r)
                    if abs(bo_r - 1.5) < 0.01:
                        is_aro_ring = True
            if is_aro_ring:
                continue
            D_ring_sum = sum(bond_energies.get(bi_r, 0.0) for bi_r in ring_bis)
            if D_ring_sum <= 0:
                continue
            # T³ Fourier coherence across the ring (PT-pure, no threshold)
            f_coh = 0.0
            _S_SUM_COH = S3 + S5 + S7
            for k_face, Pk in ((0, P1), (1, P2), (2, P3)):
                s_re = 0.0
                s_im = 0.0
                for a in ring:
                    phase = 2.0 * math.pi * (Zs[ring.index(a)] % Pk) / Pk \
                            if False else \
                            2.0 * math.pi * (topology.Z_list[a] % Pk) / Pk
                    s_re += math.cos(phase)
                    s_im += math.sin(phase)
                Ck = (s_re * s_re + s_im * s_im) / (N_ring * N_ring)
                Sk = (S3, S5, S7)[k_face]
                f_coh += Sk * Ck
            f_coh /= _S_SUM_COH
            # ── f-block back-donation [actinide-capped rings] ──────
            # When ring atoms have f-block (l=3) exocyclic neighbors
            # (e.g. U-capped Bi₃), the 5f orbitals donate into the σ
            # ring system via the pent-hept cross-face coupling R₅₇
            # = D₅·D₇ — same mechanism as atom.py insight #75, now on
            # the ring cycle rather than on a single atom.
            #
            # Magnitude:  + R₅₇ · D_ring · (n_f_cap / N_ring)
            # where n_f_cap = number of ring atoms having ≥1 exocyclic
            # f-block neighbor.
            n_f_cap = 0
            for a in ring:
                has_f_nb = False
                for bi_nb in topology.vertex_bonds.get(a, []):
                    ii, jj, _ = topology.bonds[bi_nb]
                    nb = jj if ii == a else ii
                    if nb in ring_set:
                        continue
                    Z_b = topology.Z_list[nb]
                    if period(Z_b) >= 6 and l_of(Z_b) == 3:
                        has_f_nb = True
                        break
                if has_f_nb:
                    n_f_cap += 1
            f_cap = (D5 * D7) * (n_f_cap / N_ring) if n_f_cap else 0.0

            E_ring_holo += (
                S3
                * math.sin(math.pi / N_ring) ** 2
                * D_ring_sum
                * f_coh
                * (1.0 + f_cap)
            )

    # ── C10: PERRON SPECTRAL REDISTRIBUTION [T5 on molecular graph] ──
    # The Ruelle spectral correction D₃ × σ(δλ) is distributed PER-BOND
    # via the Perron eigenvector of T_P1 matrix.
    # Build T_P1 inline for Perron and spectral use
    N = n_atoms
    T_P1 = np.zeros((N, N))
    for k in range(N):
        Zk = topology.Z_list[k]
        T_P1[k, k] = atom_data[Zk]['IE'] / RY
    for bi, (ii, jj, bo) in enumerate(topology.bonds):
        t_ij = bond_energies.get(bi, 0.0)
        T_P1[ii, jj] = t_ij
        T_P1[jj, ii] = t_ij

    _v45_perron_corrections: Dict[int, float] = {}
    if not topology.is_diatomic and len(topology.bonds) >= 2:
        spec_P1_arr = np.sort(np.linalg.eigvalsh(T_P1))
        # Use T_P1 as proxy for T_T3 (new engine doesn't have separate T_T3)
        evals_T3, evecs_T3 = np.linalg.eigh(T_P1)
        v_perron = evecs_T3[:, -1]

        delta_spec = np.sort(evals_T3) - spec_P1_arr
        sigma_delta = float(np.std(delta_spec))
        E_T3_total = D3 * sigma_delta

        n_bonds_perron = len(topology.bonds)
        w_bonds = [abs(v_perron[i] * v_perron[j])
                    for i, j, _ in topology.bonds]
        w_sum = sum(w_bonds) or 1e-12

        for bi_w in range(n_bonds_perron):
            _v45_perron_corrections[bi_w] = (
                E_T3_total * w_bonds[bi_w] / w_sum
            )

    # Update bond_energies_scf reference (alias is the same dict)
    bond_energies_scf = bond_energies

    # ── C11: VERTEX RABI SELF-ENERGY [post-NLO, 0 params] ────────────
    # After all vertex corrections (C1-C10), small molecules (n ≤ P₂)
    # still have systematic vertex bias: the poly engine's per-bond seeds
    # over-screen bonds at shared vertices by summing face densities from
    # ALL adjacent bonds, but downstream corrections only partially undo
    # this.  The residual is bidirectional and scales with the IE polarity
    # at the shared vertex.
    #
    # For each bond at a degree-k vertex, the residual self-energy is:
    #   ΔE_i = Σ_{j≠i} D₃ × D_i × (bo_j/z) × pol_eff_j × f_size
    #
    # pol_eff_j = (IE_term_j - IE_center) / RY
    #
    # Terminal LP back-donation is NOT included because in triatomics
    # terminal LP point away from the center vertex (z=1 terminals).
    # Center LP screening is captured by the LP_center / P₁ damping.
    #
    # f_size fades the correction for larger molecules where many-body
    # averaging from the spectral SCF dilutes vertex effects.
    # C11 gate: 1.0 at n=P₁(3), fade from n=P₂(5) to n=P₃(7).
    # n=4 (tetrahedral) is excluded: the overbinding bias from
    # face-sharing at z=4 vertices is NOT a vertex self-energy effect.
    if n_atoms == P1:
        _c11_f = 1.0
    elif P2 <= n_atoms < P3:
        _c11_f = max(0.0, 1.0 - float(n_atoms - P2) / (P3 - P2))
    else:
        _c11_f = 0.0
    if _c11_f > 0:
        rabi_correction: Dict[int, float] = {}
        for v in range(topology.n_atoms):
            z_v = topology.z_count[v]
            if z_v < 2:
                continue
            v_bonds_c11 = topology.vertex_bonds.get(v, [])
            if len(v_bonds_c11) < 2:
                continue
            lp_v = topology.lp[v]
            Zv = topology.Z_list[v]
            ie_v = atom_data[Zv]['IE']
            if ie_v <= 0:
                continue
            for bi_this in v_bonds_c11:
                D_this = bond_energies_scf.get(bi_this, 0.0)
                if D_this <= 0:
                    continue
                for bi_other in v_bonds_c11:
                    if bi_other == bi_this:
                        continue
                    i_ot, j_ot, bo_ot = topology.bonds[bi_other]
                    t_other = j_ot if i_ot == v else i_ot
                    ie_t = atom_data[topology.Z_list[t_other]]['IE']
                    # IE pull: pure electronegativity-driven polarity
                    pol = (ie_t - ie_v) / RY
                    # LP on center damps the polarity (LP fills face,
                    # reducing the effective electron flow)
                    if lp_v > 0:
                        pol -= lp_v * D3 / P1
                    # Self-energy: D₃ coupling × D × (bo/z) × pol × f_size
                    V_self = D3 * D_this * (bo_ot / z_v) * pol * _c11_f
                    rabi_correction[bi_this] = (
                        rabi_correction.get(bi_this, 0.0) + V_self
                    )
        for bi, corr in rabi_correction.items():
            bond_energies_scf[bi] = max(bond_energies_scf[bi] + corr, 0.01)

    # ══════════════════════════════════════════════════════════════════
    # Multi-center corrections (PTChem_Gauge mechanisms, 0 params)
    # Applied after C1-C11, before spectral extraction.
    # These capture orbital-space competition, VSEPR resonance,
    # LP-LP anti-bonding, and LP directional Coulomb stabilization.
    #
    # Gate: C1-C11 already handle face-fraction overcrowding (C1),
    # LP terminal crowding (C2), NLO vertex coupling (C3).
    # MC corrections capture RESIDUAL multi-center effects:
    #   MC1: non-hybridized orbital competition (C3 NLO residual)
    #   MC2: VSEPR angle resonance (genuinely new)
    #   MC3: unilateral LP anti-bonding in multi-bonds (new)
    #   MC4: LP directional Coulomb stabilization (new)
    # ══════════════════════════════════════════════════════════════════
    if n_atoms >= 3:
        _IE_NONHYBRID_THRESHOLD = RY - S3  # ≈ 13.387 eV (N, O, F pass)

        # ── MC1: E_competition — orbital space competition ──────────────
        # NON-HYBRIDIZED centers only (IE ≥ Ry - S₃): each extra bond
        # costs sin²₃ of the average bond energy.
        # The hybridized branch is EXCLUDED — C1 face-fraction already
        # handles it. The non-hybridized residual is what C3 NLO
        # under-corrects because NLO is perturbative (V²/ΔE) while
        # orbital saturation is non-perturbative.
        #
        # Attenuation: D₃ factor prevents double-counting with C3 NLO.
        # The C3 NLO captures D₃/z per pair; MC1 captures the (1-D₃)
        # residual that perturbation theory misses for saturated shells.
        E_comp = 0.0
        for v in range(n_atoms):
            z_v = topology.z_count[v]
            if z_v < 2:
                continue
            Zv = topology.Z_list[v]
            ie_v = atom_data[Zv]['IE']
            if ie_v < _IE_NONHYBRID_THRESHOLD:
                continue  # hybridized → skip (C1 handles)
            v_bond_indices = topology.vertex_bonds.get(v, [])
            if not v_bond_indices:
                continue
            D_on_v = [bond_energies_scf.get(bi, 0.0) for bi in v_bond_indices]
            D_avg_v = sum(D_on_v) / max(len(D_on_v), 1)
            if D_avg_v <= 0:
                continue
            # Non-perturbative residual: (1 - D₃) × S₃ × (z-1)
            # D₃ fraction is already captured by C3 NLO
            E_comp += (z_v - 1) * S3 * (1.0 - D3) * D_avg_v

        # ── MC2: E_angle — VSEPR vibrational resonance ─────────────────
        # Bond angle stabilization from geometric optimization on S².
        # LP weighs 1+C₃ ≈ 1.78× a bond in effective coordination.
        #
        # [T1] RADICAL GEOMETRY: unpaired electrons occupy s = 1/2
        # face fraction on Z/(2P₁)Z — exactly half a LP domain.
        # PT derivation: radical = half-filled orbital on polygon vertex,
        # angular weight = s (not s×(1+C₃) because unpaired e⁻ doesn't
        # pair-expand like LP). NO₂: z_eff = 2 + 0.5 = 2.5 → θ = 131.8°
        # (exp 134.1°, error 1.7% vs 180° = 26% without).
        E_angle = 0.0
        _mc2_theta_v: Dict[int, float] = {}  # cache for T2/T4
        # T1: detect molecular radical (odd TOTAL electron count)
        _total_ve = sum(valence_electrons(Z) for Z in topology.Z_list)
        _is_molecular_radical = (_total_ve % 2 == 1)
        for v in range(n_atoms):
            z_v = topology.z_count[v]
            if z_v < 2:
                continue
            lp_v = topology.lp[v]
            # T1: radical = s face fraction (half-filled orbital)
            # Gate: only if molecule has odd total electrons (true radical,
            # not formal charge artifact from SMILES like [N-]=[N+]=O).
            _odd_v = 0
            if _is_molecular_radical:
                Zv_mc2 = topology.Z_list[v]
                _odd_v = (valence_electrons(Zv_mc2) - round(topology.sum_bo[v])) % 2
            z_eff = z_v + lp_v * (1.0 + C3) + _odd_v * S_HALF
            if z_eff <= 1.001:
                continue
            cos_theta = -1.0 / (z_eff - 1.0)
            cos_theta = max(-1.0, min(1.0, cos_theta))
            theta_deg = math.degrees(math.acos(cos_theta))
            _mc2_theta_v[v] = theta_deg
            v_bond_indices = topology.vertex_bonds.get(v, [])
            if not v_bond_indices:
                continue
            D_on_v = [bond_energies_scf.get(bi, 0.0) for bi in v_bond_indices]
            D_avg_v = sum(D_on_v) / max(len(D_on_v), 1)
            if D_avg_v <= 0:
                continue
            E_angle += S3 * abs(cos_theta) * D_avg_v * S_HALF / z_v

        # (MC2b/MC2c removed — triatomic 3-center effects are handled
        #  by the v4 seed anchor in the seed calibration section above.)

        # ── MC3: E_lp_anti — LP-LP anti-bonding repulsion ──────────────
        # UNILATERAL only: one LP atom in multi-bond at non-hybridized
        # polyatomic center. The bilateral branch is EXCLUDED — C2 LP
        # terminal crowding already handles bilateral LP-LP repulsion.
        # The unilateral case (π bond + LP on adjacent center) is new:
        # the π electrons anti-bond with the LP orbital.
        E_lp_anti = 0.0
        for bi, (i, j, bo) in enumerate(topology.bonds):
            if bo < 2:
                continue
            lp_i = topology.lp[i]
            lp_j = topology.lp[j]
            D_bond = bond_energies_scf.get(bi, 0.0)
            if D_bond <= 0:
                continue
            if max(lp_i, lp_j) == 0:
                continue
            # Only unilateral: one atom has LP, in multi-bond
            lp_one = max(lp_i, lp_j)
            if lp_i >= lp_j:
                Z_lp = topology.Z_list[i]
                z_lp = topology.z_count[i]
            else:
                Z_lp = topology.Z_list[j]
                z_lp = topology.z_count[j]
            ie_lp = atom_data[Z_lp]['IE']
            if z_lp > 1 and ie_lp >= _IE_NONHYBRID_THRESHOLD:
                anti_one = lp_one * S3 * (bo - 1) / bo
                E_lp_anti += anti_one * D_bond

        # ── MC4: E_LP_dir — LP directional Coulomb stabilization ───────
        # When a central atom has LP and partners without LP,
        # the LP creates directional Coulomb stabilization.
        # Gate relaxed: lp > 0 (was lp >= z). Sub-saturated LP (lp < z)
        # requires pyramidal geometry (lp+z >= 4) and non-ring vertex.
        # Coupling uses reduced coordination lp*z/(lp+z) — the PT
        # harmonic mean analogous to reduced mass in two-body Coulomb.
        # For saturated case (lp >= z): lp*z/(lp+z) = lp/z when lp=z,
        # so existing results are exactly preserved.
        E_LP_dir = 0.0
        for v in range(n_atoms):
            z_v = topology.z_count[v]
            if z_v < 2:
                continue
            lp_v = topology.lp[v]
            if lp_v == 0:
                continue
            # Sub-saturated LP (lp < z): directional stabilization requires
            # pyramidal geometry (lp+z >= 4, tetrahedral). Planar sp2
            # (z=2, lp=1) has in-plane LP with weak directionality.
            # Also exclude ring atoms (conjugated LP) and per >= 3
            # (diffuse LP orbital, weak directionality).
            if lp_v < z_v:
                if v in topology.ring_atoms:
                    continue
                if lp_v + z_v < 4:
                    continue
                if atom_data[topology.Z_list[v]]['per'] > 2:
                    continue
            # Reduced coordination: harmonic coupling between LP and bonds
            z_red = lp_v * z_v / (lp_v + z_v)
            # Check if any partner has no LP
            v_bond_indices = topology.vertex_bonds.get(v, [])
            has_partner_no_lp = False
            r_sum = 0.0
            r_count = 0
            for bi_lp in v_bond_indices:
                i_lp, j_lp, bo_lp = topology.bonds[bi_lp]
                partner_lp = j_lp if i_lp == v else i_lp
                if topology.lp[partner_lp] == 0:
                    has_partner_no_lp = True
                Zi_lp = topology.Z_list[i_lp]
                Zj_lp = topology.Z_list[j_lp]
                per_i_lp = atom_data[Zi_lp]['per']
                per_j_lp = atom_data[Zj_lp]['per']
                seed_lp = bond_seeds.get(bi_lp)
                S_bond_lp = seed_lp.S1 if seed_lp else 0.0
                r_bond = r_equilibrium(
                    per_i_lp, per_j_lp, bo_lp, S_bond_lp,
                    topology.z_count[i_lp], topology.z_count[j_lp],
                    topology.lp[i_lp], topology.lp[j_lp],
                )
                if r_bond > 0:
                    r_sum += r_bond
                    r_count += 1
            if not has_partner_no_lp or r_count == 0:
                continue
            r_avg = r_sum / r_count
            ie_v = atom_data[topology.Z_list[v]]['IE']
            f_comp_lp = 1.0 if ie_v >= _IE_NONHYBRID_THRESHOLD else 2.0 / 3.0
            E_LP_dir += z_red * S3 * S_HALF * COULOMB_EV_A / r_avg * f_comp_lp

        # ── MC assembly: apply corrections to D_at ─────────────────────
        # D_at = D_current - E_comp + E_angle + E_coop + E_linear_3c
        #        - E_lp_anti + E_LP_dir
        E_mc_total = -E_comp + E_angle - E_lp_anti + E_LP_dir
        D_total_scf = sum(bond_energies_scf.get(bi, 0.0) for bi in range(n_bonds))
        if abs(E_mc_total) > 1e-12 and D_total_scf > 0:
            for bi in range(n_bonds):
                D_bi = bond_energies_scf.get(bi, 0.0)
                if D_bi > 0:
                    frac = D_bi / D_total_scf
                    bond_energies_scf[bi] = max(D_bi + E_mc_total * frac, 0.01)

    # ══════════════════════════════════════════════════════════════════
    # END PORTED CORRECTIONS
    # ══════════════════════════════════════════════════════════════════

    # Step 8: Spectral correction (T_P1 already built above by C10)
    E_spectral = _spectral_correction(T_P1, topology, atom_data, bond_energies_scf)
    E_spectral += E_ring_holo  # C9 ring holonomy contribution

    # Step 9: Assembly
    # Per-face decomposition from seeds
    D_sum_P1, D_sum_P2, D_sum_P3 = 0.0, 0.0, 0.0
    bond_results: List[BondResult] = []

    for bi, (i, j, bo) in enumerate(topology.bonds):
        Zi, Zj = topology.Z_list[i], topology.Z_list[j]
        di, dj = atom_data[Zi], atom_data[Zj]

        D0_scf = bond_energies_scf[bi]
        seed = bond_seeds[bi]

        # Per-bond ghost (same model as old engine)
        per_A, per_B = di['per'], dj['per']
        n_shells = ((per_A - 1) + (per_B - 1)) / 2.0
        z_i = topology.z_count[i]
        z_j = topology.z_count[j]
        z_min_ij = min(z_i, z_j)
        f_z_ghost = 1.0 + max(0, z_min_ij - 1) * D3
        S_ghost = BETA_GHOST * D3 * S_HALF * max(n_shells, 1.0) * f_z_ghost
        ghost_atten = math.exp(-S_ghost * D_FULL)

        # v45: Perron spectral correction (per-bond, before ghost VP)
        D0_scf += _v45_perron_corrections.get(bi, 0.0)

        D0_final = D0_scf * ghost_atten

        # Scale seed components by SCF ratio
        scale = D0_scf / max(seed.D0, 1e-10)
        D_P1_bond = (seed.D0_P0 + seed.D0_P1) * scale * ghost_atten
        D_P2_bond = seed.D0_P2 * scale * ghost_atten
        D_P3_bond = seed.D0_P3 * scale * ghost_atten

        D_sum_P1 += D_P1_bond
        D_sum_P2 += D_P2_bond
        D_sum_P3 += D_P3_bond

        # Physical observables
        lp_i_v = topology.lp[i]
        lp_j_v = topology.lp[j]
        r_e = r_equilibrium(per_A, per_B, bo, seed.S1,
                            z_i, z_j, lp_i_v, lp_j_v)
        w_e = omega_e(D0_final, r_e, di['mass'], dj['mass'])

        bond_results.append(BondResult(
            D0=D0_final,
            v_sigma=D_P1_bond,
            v_pi=D_P2_bond,
            v_ionic=D_P3_bond,
            r_e=r_e,
            omega_e=w_e,
        ))

    D_at_bonds = sum(br.D0 for br in bond_results)
    D_at = D_at_bonds + E_spectral

    return TransferResult(
        D_at=D_at,
        D_at_P1=D_sum_P1,
        D_at_P2=D_sum_P2,
        D_at_P3=D_sum_P3,
        E_spectral=E_spectral,
        bonds=bond_results,
        spectrum_P1=np.sort(np.linalg.eigvalsh(T_P1)) if N >= 2 else None,
        spectrum_P2=None,
        spectrum_P3=None,
        topology=topology,
        mechanism=None,
    )
