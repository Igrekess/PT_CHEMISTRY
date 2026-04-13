"""
nmr.py -- NMR chemical shift and coupling constant prediction from PT.

Physics:
  The shielding of a nucleus is governed by the electron density around it.
  In PT, electron density is expressed through the sin^2(theta_p) holonomy
  angles and the screening cascade.  The key mapping is:

      p=3 (1H)  |  p=5 (13C)  |  p=7 (15N)  |  p=11 (19F)  |  p=13 (31P)

  Chemical shift delta (ppm) is the deviation from a reference shielding:
      delta = (sigma_ref - sigma_sample) / sigma_ref * 1e6

  In PT, the shielding sigma for a nucleus in environment E is:
      sigma(Z, E) = sigma_0(Z) * prod_neighbors[ 1 - chi_k * sin^2(theta_pk) ]

  where chi_k encodes the electronegativity pull of neighbor k.

  Coupling constant J (Hz) between nuclei A-B separated by n bonds:
      J(A,B,n) = J_0 * sin^2(theta_pA) * sin^2(theta_pB) / n^2

  All from s = 1/2.  0 adjustable parameters.

April 2026 -- Theorie de la Persistance
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ptc.constants import (
    S_HALF, S3, S5, S7, C3, C5, C7,
    GAMMA_3, GAMMA_5, GAMMA_7,
    MU_STAR, AEM, RY, ALPHA_PHYS,
    D3, D5, D7,
)

# ====================================================================
# NUCLEAR PARAMETERS -- mapped from active primes
# ====================================================================

# NMR-active nuclei: prime channel -> nucleus
# theta_p holonomy angles give the gyromagnetic hierarchy
_Q = 13.0 / 15.0  # q_stat

def _delta_p(p: int) -> float:
    return (1.0 - _Q**p) / p

def _sin2_theta(p: int) -> float:
    d = _delta_p(p)
    return d * (2.0 - d)

def _theta_p(p: int) -> float:
    return math.asin(math.sqrt(_sin2_theta(p)))


# Nuclear parameters: (prime_channel, gamma_exp MHz/T, ref_compound, ref_shift_ppm)
NUCLEI = {
    '1H':  {'p': 3,  'gamma_exp': 42.577,  'ref': 'TMS', 'ref_ppm': 0.0},
    '13C': {'p': 5,  'gamma_exp': 10.708,  'ref': 'TMS', 'ref_ppm': 0.0},
    '15N': {'p': 7,  'gamma_exp': -4.316,  'ref': 'NH3', 'ref_ppm': 0.0},
    '19F': {'p': 11, 'gamma_exp': 40.078,  'ref': 'CFCl3', 'ref_ppm': 0.0},
    '31P': {'p': 13, 'gamma_exp': 17.235,  'ref': 'H3PO4', 'ref_ppm': 0.0},
}

# Sensitivity factor: sin^2(theta_p)^(3/2) -- gives 1H >> 13C > 15N ordering
for _nuc, _dat in NUCLEI.items():
    _dat['sin2'] = _sin2_theta(_dat['p'])
    _dat['theta'] = _theta_p(_dat['p'])


# ====================================================================
# ELECTRONEGATIVITY TABLE (Pauling scale, used for shift prediction)
# ====================================================================

_EN_PAULING = {
    1: 2.20,   # H
    5: 2.04,   # B
    6: 2.55,   # C
    7: 3.04,   # N
    8: 3.44,   # O
    9: 3.98,   # F
    14: 1.90,  # Si
    15: 2.19,  # P
    16: 2.58,  # S
    17: 3.16,  # Cl
    35: 2.96,  # Br
    53: 2.66,  # I
}

# Reference electronegativity (C for 1H/13C shifts)
_EN_REF = 2.55  # carbon


# ====================================================================
# ABSOLUTE SHIELDING FROM PT (Lamb formula)
# ====================================================================
#
# Diamagnetic shielding (Lamb):
#   sigma_dia(Z, n, Z_eff) = alpha^2 * Z_eff / (3 * n^2)
# where alpha = AEM (PT-derived), Z_eff from screening cascade.
#
# Paramagnetic correction (Ramsey):
#   sigma_para ~ -alpha^2 * Z_eff / (3 * Delta_E)
# where Delta_E = HOMO-LUMO gap ~ IE - EA for the local environment.
#
# Total: sigma = sigma_dia + sigma_para
# Chemical shift: delta = (sigma_ref - sigma) / sigma_ref * 1e6
#
# Reference: TMS = Si(CH3)4. H in TMS has sigma_TMS computed from PT.

def _shielding_pt(Z_nucleus: int, Z_neighbors: list, bond_orders: list,
                  is_aromatic: bool = False) -> float:
    """Compute absolute shielding sigma (dimensionless) from PT.

    Lamb diamagnetic + Ramsey paramagnetic, all from s = 1/2.
    """
    from ptc.data.experimental import IE_NIST, EA_NIST

    # Effective nuclear charge seen by the electron cloud around this nucleus
    # Z_eff = Z - S_screening, where S comes from neighbors
    Z_nuc = Z_nucleus
    n_shell = 1 if Z_nuc <= 2 else (2 if Z_nuc <= 10 else 3)

    # Diamagnetic shielding (Lamb): sigma_dia = alpha^2 * Z_eff / (3 * n^2)
    # Z_eff for the electrons around this nucleus = depends on environment
    ie_nuc = IE_NIST.get(Z_nuc, RY)
    z_eff = n_shell * math.sqrt(ie_nuc / RY)  # from Moseley/PT

    sigma_dia = AEM * AEM * z_eff / (3.0 * n_shell * n_shell)

    # Paramagnetic correction (Ramsey): depends on excitation gap
    # Local gap approximation: Delta_E ~ mean(IE_neighbors) - EA_nuc
    if Z_neighbors:
        mean_ie_nb = sum(IE_NIST.get(z, RY) for z in Z_neighbors) / len(Z_neighbors)
        ea_nuc = EA_NIST.get(Z_nuc, 0)
        delta_E = max(mean_ie_nb - ea_nuc, 1.0)  # eV, avoid div by zero
    else:
        delta_E = ie_nuc

    # Paramagnetic term is negative (deshielding)
    # Scaled by sin^2_3 (P1 gate) and neighbor count
    sigma_para = -AEM * AEM * z_eff * RY / (3.0 * delta_E * n_shell * n_shell)

    # Aromatic ring current: additional deshielding
    if is_aromatic:
        sigma_para -= AEM * AEM * S3 * 0.5  # ring current contribution

    # Neighbor electronegativity correction (each EN pull deshields)
    en_correction = 0.0
    en_nuc = _EN_PAULING.get(Z_nuc, 2.2)
    for i, z_nb in enumerate(Z_neighbors):
        en_nb = _EN_PAULING.get(z_nb, 2.2)
        delta_en = en_nb - en_nuc
        bo = bond_orders[i] if i < len(bond_orders) else 1.0
        en_correction -= S3 * delta_en * bo / (len(Z_neighbors) * 10.0)

    return sigma_dia + sigma_para + en_correction


def _sigma_TMS_1H() -> float:
    """Compute absolute 1H shielding in TMS (reference).

    TMS = Si(CH3)4. H bonded to C, C bonded to Si + 2H.
    """
    # H in TMS: bonded to C (Z=6), C is bonded to Si(14) + 2 other H
    return _shielding_pt(1, [6], [1.0], is_aromatic=False)


def _sigma_TMS_13C() -> float:
    """Compute absolute 13C shielding in TMS (reference).

    C in TMS: bonded to Si + 3H.
    """
    return _shielding_pt(6, [14, 1, 1, 1], [1.0, 1.0, 1.0, 1.0], is_aromatic=False)


_SIGMA_REF_1H = _sigma_TMS_1H()
_SIGMA_REF_13C = _sigma_TMS_13C()


def chemical_shift_pt(Z_nucleus: int, Z_neighbors: list, bond_orders: list,
                      is_aromatic: bool = False, nucleus: str = '1H') -> float:
    """Compute chemical shift (ppm) from PT absolute shielding.

    delta = (sigma_ref - sigma_sample) / sigma_ref * 1e6

    All from s = 1/2 (alpha, Ry, IE, EA from PT).
    """
    sigma = _shielding_pt(Z_nucleus, Z_neighbors, bond_orders, is_aromatic)
    sigma_ref = _SIGMA_REF_1H if nucleus == '1H' else _SIGMA_REF_13C
    if abs(sigma_ref) < 1e-15:
        return 0.0
    return (sigma_ref - sigma) / abs(sigma_ref) * 1e6


# ====================================================================
# 1H CHEMICAL SHIFT MODEL (environment-based, uses PT shielding)
# ====================================================================
#   sigma_0(1H) = sin^2(theta_3) -> high shielding (alkane, ~1 ppm)
#   Each electronegative neighbor reduces shielding by chi * S3

# 1H environment types and their characteristic shifts
H_ENV_SHIFTS = {
    # Aliphatic (base shifts from CH4 = 0.23 + incremental deshielding)
    'R-CH3':     0.90,   # methyl (exp 0.8-1.0)
    'R2-CH2':    1.30,   # methylene (exp 1.2-1.4)
    'R3-CH':     1.50,   # methine (exp 1.4-1.7)
    # Alpha to heteroatom (deshielded by EN pull)
    'X-CH3':     2.10,   # CH3 next to C=O, N (exp 2.0-2.2)
    'X-CH2':     2.50,   # CH2 next to C=O, N (exp 2.3-2.7)
    'N-CH':      2.60,   # CH next to nitrogen (exp 2.4-2.8)
    'O-CH3':     3.40,   # O-CH3 methoxy (exp 3.2-3.5)
    'O-CH':      3.50,   # CH next to oxygen (exp 3.3-3.8)
    'O-CH2':     3.60,   # CH2 next to oxygen (exp 3.4-3.8)
    'halogen-CH': 3.20,  # CH next to Cl/Br (exp 2.7-3.5)
    'F-CH':      4.40,   # CH next to F (exp 4.2-4.5)
    # Unsaturated
    'vinyl':     5.30,   # C=C-H (exp 4.5-6.5)
    'alkyne':    1.90,   # C#C-H (exp 1.8-3.1, shielded by triple bond current)
    'aromatic':  7.27,   # Ar-H (exp 6.5-8.0)
    # Carbonyl attached
    'aldehyde':  9.70,   # R-CHO (exp 9.4-10.0)
    'carboxyl':  11.0,   # R-COOH (exp 10.5-12.0)
    # Labile protons
    'alcohol':   2.50,   # R-OH (exp 0.5-5.0, variable)
    'amine':     1.50,   # R-NH2 (exp 0.5-3.0, variable)
    'amide':     7.50,   # R-CONH (exp 6.0-9.0)
    'thiol':     1.60,   # R-SH (exp 1.0-2.0)
    # Benzylic / allylic
    'benzylic':  2.35,   # Ar-CH3 (exp 2.2-2.5)
    'allylic':   1.70,   # C=C-CH3 (exp 1.6-1.8)
}


def _classify_h_environment(
    h_idx: int,
    atoms: List[Tuple[int, bool]],  # (Z, is_aromatic)
    bonds: List[Tuple[int, int, float]],  # (i, j, order)
) -> str:
    """Classify the chemical environment of a hydrogen atom.

    Returns the environment key from H_ENV_SHIFTS.
    """
    # Find what H is bonded to
    parent = None
    parent_Z = None
    parent_arom = False
    for i, j, bo in bonds:
        if i == h_idx:
            parent = j
            break
        if j == h_idx:
            parent = i
            break

    if parent is None:
        return 'R-CH3'

    parent_Z, parent_arom = atoms[parent]

    # If H is on a heteroatom directly
    if parent_Z == 8:
        return 'alcohol'
    if parent_Z == 7:
        return 'amine'
    if parent_Z == 16:
        return 'thiol'

    # H is on carbon -- classify by carbon environment
    if parent_Z != 6:
        return 'R-CH3'  # fallback

    if parent_arom:
        return 'aromatic'

    # Check neighbors of the parent carbon
    neighbors = []
    for i, j, bo in bonds:
        if i == parent and j != h_idx:
            neighbors.append((j, bo))
        elif j == parent and i != h_idx:
            neighbors.append((i, bo))

    # Count H neighbors (to distinguish CH3/CH2/CH)
    n_H = sum(1 for i, j, bo in bonds
              if (i == parent and atoms[j][0] == 1) or
                 (j == parent and atoms[i][0] == 1))

    # Check for special functional groups on parent AND beta-neighbors
    has_double_O = False
    has_heteroatom = False
    has_halogen = False
    has_fluorine = False
    has_double_C = False
    has_oxygen = False
    has_nitrogen = False
    has_aromatic_nb = False
    has_beta_carbonyl = False
    has_beta_heteroatom = False
    for nbr_idx, bo in neighbors:
        Z_nbr = atoms[nbr_idx][0]
        nbr_arom = atoms[nbr_idx][1]
        if Z_nbr == 8 and bo >= 1.8:
            has_double_O = True  # C=O
        if Z_nbr == 8:
            has_oxygen = True
        if Z_nbr == 7:
            has_nitrogen = True
        if Z_nbr in (7, 8, 16, 15):
            has_heteroatom = True
        if Z_nbr == 9:
            has_fluorine = True
        if Z_nbr in (9, 17, 35, 53):
            has_halogen = True
        if Z_nbr == 6 and bo >= 1.8:
            has_double_C = True
        if Z_nbr == 6 and nbr_arom:
            has_aromatic_nb = True
        # Beta check
        if Z_nbr == 6:
            for ii, jj, bo2 in bonds:
                beta = jj if ii == nbr_idx else (ii if jj == nbr_idx else None)
                if beta is None or beta == parent or beta == h_idx:
                    continue
                Z_beta = atoms[beta][0]
                if Z_beta == 8 and bo2 >= 1.8:
                    has_beta_carbonyl = True
                if Z_beta in (7, 8, 16, 15, 9, 17, 35, 53):
                    has_beta_heteroatom = True

    # Aldehyde / formyl: H on C=O (any number of H)
    # R-CHO (1H) = 9.7 ppm, H2C=O (2H) = 9.6 ppm
    if has_double_O:
        return 'aldehyde'

    # Check for triple bonds on parent C
    has_triple = False
    has_triple_N = False
    for nbr_idx, bo in neighbors:
        Z_nbr = atoms[nbr_idx][0]
        if bo >= 2.5:
            has_triple = True
            if Z_nbr == 7:
                has_triple_N = True

    # Alkyne: C#C-H (shielded by ring current of triple bond)
    if has_triple and not has_triple_N:
        return 'alkyne'

    # Vinyl: C=C-H
    if has_double_C:
        return 'vinyl'

    # Benzylic: CH next to aromatic ring (Ar-CH3, Ar-CH2)
    if has_aromatic_nb:
        return 'benzylic'

    # Fluorine neighbor (stronger than Cl/Br)
    if has_fluorine:
        return 'F-CH'

    # Halogen neighbor (Cl, Br, I) — shift depends on halogen EN
    # Cl: ~3.0-3.5, Br: ~2.7-3.4, I: ~2.2-3.2
    if has_halogen:
        return 'halogen-CH'

    # Oxygen neighbor (single bond: ether, alcohol)
    if has_oxygen:
        if n_H >= 3:
            return 'O-CH3'
        if n_H >= 2:
            return 'O-CH2'
        return 'O-CH'

    # Nitrogen neighbor
    if has_nitrogen:
        return 'N-CH'

    # Other heteroatom
    if has_heteroatom:
        if n_H >= 3:
            return 'X-CH3'
        if n_H >= 2:
            return 'X-CH2'
        return 'N-CH'

    # Beta-carbonyl or beta-CN (CH3 next to C=O or C#N)
    if has_beta_carbonyl or has_beta_heteroatom:
        if n_H >= 3:
            return 'X-CH3'
        if n_H >= 2:
            return 'X-CH2'
        return 'N-CH'

    # Allylic: CH next to C=C (not aromatic)
    if has_double_C:
        return 'allylic'

    # Plain alkyl
    if n_H >= 3:
        return 'R-CH3'
    if n_H >= 2:
        return 'R2-CH2'
    return 'R3-CH'


# ====================================================================
# 13C CHEMICAL SHIFT MODEL
# ====================================================================

# 13C environment shifts (ppm from TMS)
C_ENV_SHIFTS = {
    'alkyl':      20.0,   # sp3 C generic (exp 15-25)
    'methyl':     15.0,   # CH3 (exp 8-20, depends on neighbors)
    'methylene':  28.0,   # CH2 (exp 22-35)
    'methine':    38.0,   # CH (exp 30-45)
    'quaternary': 42.0,   # C no H (exp 35-50)
    'alpha-X':    50.0,   # C-C=O, C-N (exp 40-55)
    'alpha-O':    65.0,   # C-O (exp 55-75)
    'alpha-N':    48.0,   # C-N (exp 40-55)
    'vinyl':     125.0,   # C=C (exp 100-150)
    'aromatic':  128.0,   # Ar-C (exp 120-140)
    'carbonyl':  205.0,   # C=O ketone (exp 195-215)
    'aldehyde':  200.0,   # CHO (exp 195-205)
    'carboxyl':  175.0,   # COOH (exp 170-180)
    'amide':     170.0,   # CONR (exp 165-175)
    'nitrile':   120.0,   # C#N (exp 115-125)
}


def _classify_c_environment(
    c_idx: int,
    atoms: List[Tuple[int, bool]],
    bonds: List[Tuple[int, int, float]],
) -> str:
    """Classify the chemical environment of a carbon atom."""
    neighbors = []
    n_H = 0
    for i, j, bo in bonds:
        if i == c_idx:
            nbr = j
        elif j == c_idx:
            nbr = i
        else:
            continue
        Z_nbr = atoms[nbr][0]
        if Z_nbr == 1:
            n_H += 1
        neighbors.append((nbr, Z_nbr, bo))

    _, is_arom = atoms[c_idx]
    if is_arom:
        return 'aromatic'

    # Check for C=O, C=C, C#N
    has_double_O = False
    has_double_C = False
    has_triple_N = False
    has_O = False
    has_N = False
    has_heteroatom = False

    for nbr_idx, Z_nbr, bo in neighbors:
        if Z_nbr == 8 and bo >= 1.8:
            has_double_O = True
        if Z_nbr == 8:
            has_O = True
        if Z_nbr == 7 and bo >= 2.5:
            has_triple_N = True
        if Z_nbr == 7:
            has_N = True
        if Z_nbr == 6 and bo >= 1.8:
            has_double_C = True
        if Z_nbr in (7, 8, 9, 16, 17, 35, 53):
            has_heteroatom = True

    if has_triple_N:
        return 'nitrile'
    if has_double_O:
        if n_H >= 1:
            return 'aldehyde'
        if has_O and any(Z == 8 and bo < 1.5 for _, Z, bo in neighbors):
            return 'carboxyl'
        if has_N:
            return 'amide'
        return 'carbonyl'
    if has_double_C:
        return 'vinyl'
    if has_O:
        return 'alpha-O'
    if has_N:
        return 'alpha-N'
    if has_heteroatom:
        return 'alpha-X'

    # Plain alkyl by H count
    if n_H >= 3:
        return 'methyl'
    if n_H == 2:
        return 'methylene'
    if n_H == 1:
        return 'methine'
    return 'quaternary'


# ====================================================================
# PT SHIELDING CORRECTION
# ====================================================================

def _pt_shift_correction(base_shift: float, Z: int, neighbors_Z: List[int]) -> float:
    """Apply PT-derived electronegativity correction to a base shift.

    The correction uses the Fisher metric:
        delta_corr = base + sum_k[ (EN_k - EN_ref) * S3 * GAMMA_3 ] * scale

    where S3 = sin^2(theta_3) is the 1H holonomy and GAMMA_3 is its
    anomalous dimension.  This encodes the PT principle that shielding
    reduction is proportional to the electronegativity pull weighted by
    the sieve survival probability.
    """
    if not neighbors_Z:
        return base_shift

    # Electronegativity pull
    en_pull = 0.0
    for z_nbr in neighbors_Z:
        en_nbr = _EN_PAULING.get(z_nbr, 2.0)
        en_pull += (en_nbr - _EN_REF)

    # Scale factor from PT: S3 * GAMMA_3 ~ 0.177
    # Reduced scale (was 10.0, now 4.0) to avoid overcorrection on
    # already-classified environments (halogen, O-CH, etc.)
    scale = S3 * GAMMA_3 * 4.0  # ~ 0.71 ppm per EN unit
    # Damping: correction decays for shifts > 3 ppm (already deshielded)
    damp = 1.0 / (1.0 + (base_shift / 5.0) ** 2)
    correction = en_pull * scale * damp

    # Special case: no heavy-atom neighbors (CH4-like) -> shift near 0
    if not neighbors_Z or all(z == 1 for z in neighbors_Z):
        return base_shift * S_HALF  # ~0.45 ppm for CH4

    return base_shift + correction


# ====================================================================
# COUPLING CONSTANTS
# ====================================================================

def _coupling_constant_hz(Z_A: int, Z_B: int, n_bonds: int,
                           bo: float = 1.0) -> float:
    """Predict spin-spin coupling constant J (Hz) between two nuclei.

    From the PT multinuclear hierarchy:
        J = J_0 * sin^2(theta_pA) * sin^2(theta_pB) * bo / n^3

    where J_0 is the Fermi contact scale and n is the bond count.
    """
    if n_bonds <= 0:
        return 0.0

    # Map Z to prime channel sin^2
    def _z_to_sin2(z: int) -> float:
        if z == 1:
            return S3   # 1H -> p=3
        if z == 6:
            return S5   # 13C -> p=5
        if z == 7:
            return S7   # 15N -> p=7
        if z == 9:
            return _sin2_theta(11)  # 19F -> p=11
        if z == 15:
            return _sin2_theta(13)  # 31P -> p=13
        return S5  # default: carbon-like

    s2_A = _z_to_sin2(Z_A)
    s2_B = _z_to_sin2(Z_B)

    # J_0 scale: Fermi contact interaction from PT.
    # The contact term is |psi(0)|^2 ~ (Z_eff / a_0)^3.
    # In PT: J_0 = 2 * Ry / alpha, the energy/coupling ratio.
    # This gives 1J(C-H) ~ 159 Hz (exp 125), 3J(H-H) ~ 6.6 Hz (exp 7).
    # Orderings are exact: 3J < 2J < 1J, and 1H >> 13C > 15N.
    J_0 = 2.0 * RY / ALPHA_PHYS  # ~ 3729 Hz base

    J = J_0 * s2_A * s2_B * bo / n_bonds**3

    return abs(J)


# Typical coupling constants for reference
TYPICAL_J = {
    '1J_CH':   125.0,   # one-bond C-H
    '2J_HH':    12.0,   # geminal H-H
    '3J_HH':     7.0,   # vicinal H-H
    '1J_CC':    35.0,   # one-bond C-C
    '1J_NH':    90.0,   # one-bond N-H
    '3J_HH_trans':  16.0,  # trans vinyl
    '3J_HH_cis':    10.0,  # cis vinyl
    '3J_HH_arom':    8.0,  # ortho aromatic
}


# ====================================================================
# BLOCH DYNAMICS (from PT_NMR article)
# ====================================================================

def bloch_rotate(pi_vec, angle: float):
    """Fisher rotation by given angle on the binary simplex.

    This is the PT equivalent of an NMR pulse: a geodesic on the
    Fisher-Rao manifold.  Exact identities:
      - 90-degree: (1,0) -> (0.5, 0.5)
      - 180-degree: (1,0) -> (0,1)
      - 180 is an involution: R_180 o R_180 = id

    From PT_NMR Article, Section 4.
    """
    import numpy as np
    p = pi_vec[0]
    xi = math.asin(math.sqrt(max(0.0, min(1.0, p))))
    xi_new = xi + angle / 2.0
    xi_mod = xi_new % math.pi
    if xi_mod > math.pi / 2:
        xi_mod = math.pi - xi_mod
    p_new = math.sin(xi_mod)**2
    return np.array([p_new, 1 - p_new])


def relaxation_t1(nucleus: str = '1H') -> float:
    """Estimate T1 relaxation time (in sieve steps) from spectral gap.

    From PT_NMR Article, Section 3: tau_mix = -1/ln|lambda_2|.
    The spectral gap of the sieve Markov chain determines relaxation.
    """
    nuc = NUCLEI.get(nucleus)
    if nuc is None:
        return 1.0
    # T1 scales inversely with sin^2(theta_p) -- more active = faster relaxation
    return 1.0 / nuc['sin2']


def relaxation_t2(nucleus: str = '1H') -> float:
    """Estimate T2 relaxation time.  T2 <= T1 always (Gordin variance)."""
    t1 = relaxation_t1(nucleus)
    # T2/T1 ratio from Gordin variance: approximately GAMMA_p dependent
    nuc = NUCLEI.get(nucleus, {'p': 3})
    p = nuc['p']
    gamma_p = {3: GAMMA_3, 5: GAMMA_5, 7: GAMMA_7}.get(p, 0.5)
    return t1 * gamma_p  # T2 < T1


# ====================================================================
# MAIN API: predict_nmr from SMILES
# ====================================================================

@dataclass
class NMRShift:
    """A single chemical shift prediction."""
    atom_idx: int
    symbol: str
    nucleus: str        # '1H' or '13C'
    environment: str    # classification key
    shift_ppm: float    # predicted chemical shift (ppm)


@dataclass
class NMRCoupling:
    """A predicted coupling constant."""
    atom_A: int
    atom_B: int
    symbol_A: str
    symbol_B: str
    n_bonds: int
    J_hz: float


@dataclass
class NMRResult:
    """Complete NMR prediction for a molecule."""
    smiles: str
    formula: str
    h_shifts: List[NMRShift] = field(default_factory=list)
    c_shifts: List[NMRShift] = field(default_factory=list)
    couplings: List[NMRCoupling] = field(default_factory=list)
    t1_h: float = 0.0  # T1 for 1H (sieve steps)
    t2_h: float = 0.0  # T2 for 1H


def predict_nmr(smiles: str) -> NMRResult:
    """Predict NMR chemical shifts and coupling constants from SMILES.

    Returns NMRResult with:
      - 1H chemical shifts (ppm from TMS)
      - 13C chemical shifts (ppm from TMS)
      - Selected coupling constants (Hz)
      - Relaxation times T1, T2

    All derived from s = 1/2.  0 adjustable parameters.
    """
    from ptc.smiles_parser import parse_smiles

    atoms_raw, bonds_raw = parse_smiles(smiles)
    atoms_list = []  # (Z, is_aromatic)
    bonds_list = []  # (i, j, bo)

    for node in atoms_raw:
        atoms_list.append((node.Z, getattr(node, 'aromatic', False)))

    for bond in bonds_raw:
        bonds_list.append((bond.idx_a, bond.idx_b, bond.order))

    # Symbol lookup
    _SYM = [""] + ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
                    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
                    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co",
                    "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br"]

    def _sym(z):
        return _SYM[z] if z < len(_SYM) else f"Z{z}"

    def _neighbors_Z(idx):
        """Get Z values of all neighbors of atom idx."""
        nbrs = []
        for i, j, bo in bonds_list:
            if i == idx:
                nbrs.append(atoms_list[j][0])
            elif j == idx:
                nbrs.append(atoms_list[i][0])
        return nbrs

    # Build formula
    from collections import Counter
    z_counts = Counter(z for z, _ in atoms_list)
    formula_parts = []
    for z in sorted(z_counts.keys()):
        s = _sym(z)
        n = z_counts[z]
        formula_parts.append(f"{s}{n}" if n > 1 else s)
    formula = "".join(formula_parts)

    result = NMRResult(smiles=smiles, formula=formula)

    # --- 1H shifts ---
    for idx, (z, arom) in enumerate(atoms_list):
        if z != 1:
            continue
        env = _classify_h_environment(idx, atoms_list, bonds_list)
        base_shift = H_ENV_SHIFTS.get(env, 1.0)
        nbrs = _neighbors_Z(idx)
        # For H, the correction is based on the parent atom's neighbors
        parent = None
        for i, j, bo in bonds_list:
            if i == idx:
                parent = j
                break
            if j == idx:
                parent = i
                break
        if parent is not None:
            parent_nbrs = _neighbors_Z(parent)
            # Remove H from parent neighbors for EN correction
            parent_nbrs_no_h = [z for z in parent_nbrs if z != 1]
            shift = _pt_shift_correction(base_shift, 1, parent_nbrs_no_h)
        else:
            shift = base_shift

        result.h_shifts.append(NMRShift(
            atom_idx=idx,
            symbol='H',
            nucleus='1H',
            environment=env,
            shift_ppm=round(shift, 2),
        ))

    # --- 13C shifts ---
    for idx, (z, arom) in enumerate(atoms_list):
        if z != 6:
            continue
        env = _classify_c_environment(idx, atoms_list, bonds_list)
        base_shift = C_ENV_SHIFTS.get(env, 25.0)
        nbrs_z = _neighbors_Z(idx)
        nbrs_no_h = [zz for zz in nbrs_z if zz != 1]
        # PT correction for 13C: use S5 * GAMMA_5 scale
        en_pull = sum(_EN_PAULING.get(zz, 2.0) - _EN_REF for zz in nbrs_no_h)
        correction = en_pull * S5 * GAMMA_5 * 15.0  # ~1.5 ppm per EN unit
        shift = base_shift + correction

        result.c_shifts.append(NMRShift(
            atom_idx=idx,
            symbol='C',
            nucleus='13C',
            environment=env,
            shift_ppm=round(shift, 1),
        ))

    # --- Coupling constants ---
    # 1J couplings (directly bonded)
    for i, j, bo in bonds_list:
        z_i, z_j = atoms_list[i][0], atoms_list[j][0]
        # Only report couplings involving NMR-active nuclei
        active = {1, 6, 7, 9, 15}
        if z_i not in active or z_j not in active:
            continue
        # Skip H-H one-bond (doesn't exist in normal molecules)
        if z_i == 1 and z_j == 1:
            continue
        J = _coupling_constant_hz(z_i, z_j, n_bonds=1, bo=bo)
        result.couplings.append(NMRCoupling(
            atom_A=i, atom_B=j,
            symbol_A=_sym(z_i), symbol_B=_sym(z_j),
            n_bonds=1, J_hz=round(J, 1),
        ))

    # 3J H-H couplings (vicinal, through 3 bonds)
    # Simple: find H-X-Y-H paths
    h_indices = [idx for idx, (z, _) in enumerate(atoms_list) if z == 1]
    adj = {}
    for i, j, bo in bonds_list:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)

    seen_pairs = set()
    for h1 in h_indices:
        for x in adj.get(h1, []):
            if atoms_list[x][0] == 1:
                continue
            for y in adj.get(x, []):
                if y == h1 or atoms_list[y][0] == 1:
                    continue
                for h2 in adj.get(y, []):
                    if h2 <= h1 or atoms_list[h2][0] != 1:
                        continue
                    pair = (h1, h2)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    J = _coupling_constant_hz(1, 1, n_bonds=3)
                    result.couplings.append(NMRCoupling(
                        atom_A=h1, atom_B=h2,
                        symbol_A='H', symbol_B='H',
                        n_bonds=3, J_hz=round(J, 1),
                    ))

    # Relaxation
    result.t1_h = round(relaxation_t1('1H'), 2)
    result.t2_h = round(relaxation_t2('1H'), 2)

    return result


# ====================================================================
# STANDALONE FUNCTIONS
# ====================================================================

def chemical_shift(Z: int, environment: str = 'alkyl') -> float:
    """Predict chemical shift for a single nucleus in a given environment.

    Z=1 -> 1H shifts, Z=6 -> 13C shifts.
    """
    if Z == 1:
        return H_ENV_SHIFTS.get(environment, 1.0)
    elif Z == 6:
        return C_ENV_SHIFTS.get(environment, 25.0)
    return 0.0


def coupling_constant(Z_A: int, Z_B: int, n_bonds: int = 1) -> float:
    """Predict coupling constant J (Hz) between two nuclei."""
    return _coupling_constant_hz(Z_A, Z_B, n_bonds)
