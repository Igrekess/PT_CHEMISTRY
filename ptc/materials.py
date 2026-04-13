"""
materials.py -- Module 4: Band gaps and materials properties.

PT Theory for Band Gaps
=======================
The band gap is the P4 bifurcation at crystalline scale.
The Phillips-Van Vechten spectral decomposition maps naturally to PT:

  E_gap = sqrt(E_h^2 + C^2)     [Pythagorean CRT: covalent perp ionic]

where:
  E_h  = homopolar gap (covalent channel)
       = IE * sin^2_3 * (P1/per)^2 * f_screen * f_d
  C    = heteropolar gap (ionic channel)
       tetrahedral: |dIE| * (sin^2_3 + delta_3)  [P1 vertex coupling]
       rocksalt:    |dIE| + (sin^2_3+sin^2_5)*n_val*Ry/per^2  [Madelung]

Classification follows the PT energy hierarchy:
  Metal:         E_gap < s * kT            (gap closes at thermal scale)
  Semiconductor: s * kT < E_gap < Ry/P1    (gap between thermal and Shannon)
  Insulator:     E_gap > Ry/P1             (gap exceeds Shannon cap)

Dielectric constant (PT Penn model):
  eps_electronic = 1 + [P1^2 + P1*cos^2_3 + delta_3] / E_gap^(s + delta_3)
  eps_ionic      = (C/E_gap)^2 * n_coord * s * n_charge  [polar only]

INPUT:   s = 1/2  (everything derived from this)
PARAMS:  0 adjustable
April 2026 -- Theorie de la Persistance
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ptc.constants import (
    S_HALF,
    P1, P2, P3,
    S3, S5, S7,
    C3, C5, C7,
    AEM,
    ALPHA_PHYS,
    RY,
    A_BOHR,
    D3, D5, D7,
    D_FULL,
)
from ptc.data.experimental import IE_NIST, SYMBOLS
from ptc.periodic import period

# ── Boltzmann constant (eV/K) -- unit conversion, not a parameter ──
K_BOLTZMANN_EV = 8.617333e-5  # eV/K  (CODATA exact in new SI)


# ╔════════════════════════════════════════════════════════════════════╗
# ║  CRYSTAL STRUCTURE DATA                                           ║
# ║  Coordination numbers derive from geometry, not parameters.       ║
# ╚════════════════════════════════════════════════════════════════════╝

_COORD = {
    'diamond':     4,
    'zincblende':  4,
    'wurtzite':    4,
    'rocksalt':    6,
    'fluorite':    8,
    'bcc':         8,
    'fcc':        12,
    'hcp':        12,
}

# ── Divalent cations (group 2 and group 12) ──
_DIVALENT = frozenset({4, 12, 20, 38, 56, 88})

# ── Trivalent cations (group 13: Al, Ga, In, Tl) ──
_TRIVALENT_CATIONS = frozenset({13, 31, 49, 81})


# ╔════════════════════════════════════════════════════════════════════╗
# ║  HOMOPOLAR GAP  E_h                                               ║
# ║                                                                    ║
# ║  E_h = IE * sin^2_3 * (P1/per)^2 * f_screen * f_d * f_struct     ║
# ║                                                                    ║
# ║  The covalent gap scales with atomic IE (energy scale) modulated  ║
# ║  by the CRT filter sin^2_3, hydrogen-like 1/per^2 decay, and     ║
# ║  crystal-specific screening factors -- all from s=1/2.            ║
# ╚════════════════════════════════════════════════════════════════════╝

def _homopolar_gap(Z_A: int, Z_B: int | None, n_coord: int,
                   structure: str) -> float:
    """Compute E_h, the homopolar (covalent) part of the band gap.

    E_h = IE_avg * S3 * (P1/per_avg)^2 * f_screen * f_d * f_struct

    - S3 = sin^2(theta_3): the P1 CRT filter for bonding [D07]
    - (P1/per)^2: hydrogen-like radial scaling [T3]
    - f_screen = (C3*C5)^clip(per-2, 0, 1): crystal screening beyond per=2
      C3*C5 = cos^2_3 * cos^2_5 accounts for core polarization from
      the P1 and P2 shells below the valence.  Saturates at 1 shell
      because the (P1/per)^2 factor already captures the radial decay.
    - f_d = 1 + D_FULL*S5/P1 for per>=4: d-electron back-bonding pushes
      the antibonding band UP, partially opening the gap.
    - f_struct = S3 for rocksalt: the ionic crystal environment adds
      one extra CRT pass to the covalent channel.
    """
    per_A = period(Z_A)
    if Z_B is not None:
        per_B = period(Z_B)
        per_avg = (per_A + per_B) / 2.0
        ie_avg = (IE_NIST.get(Z_A, 8.0) + IE_NIST.get(Z_B, 8.0)) / 2.0
    else:
        per_avg = float(per_A)
        ie_avg = IE_NIST.get(Z_A, 8.0)

    # Crystal screening: C3*C5 per extra period beyond per=2 (max 1 shell)
    n_extra = max(0.0, min(1.0, per_avg - 2.0))
    f_screen = (C3 * C5) ** n_extra

    # d-block back-bonding correction (per >= 4)
    has_d = per_avg >= 3.5
    f_d = (1.0 + D_FULL * S5 / P1) if has_d else 1.0

    # Structure factor: ionic environment adds one S3 gate
    f_struct = S3 if structure in ('rocksalt', 'fluorite') else 1.0

    E_h = ie_avg * S3 * (P1 / per_avg) ** 2 * f_screen * f_d * f_struct
    return E_h


# ╔════════════════════════════════════════════════════════════════════╗
# ║  HETEROPOLAR GAP  C                                               ║
# ║                                                                    ║
# ║  Tetrahedral (covalent-dominated):                                 ║
# ║    C = |dIE| * (sin^2_3 + delta_3)   [P1 vertex: filter + defect] ║
# ║                                                                    ║
# ║  Rocksalt (ionic-dominated):                                       ║
# ║    C = |dIE| + (sin^2_3+sin^2_5)*n_val*Ry/per^2  [Madelung term] ║
# ║    The first term = atomic IE polarity (unfiltered in ionic limit) ║
# ║    The second term = crystalline Madelung correction through       ║
# ║    P1+P2 channels, scaling as hydrogen with valence charge n_val. ║
# ╚════════════════════════════════════════════════════════════════════╝

def _heteropolar_gap(Z_A: int, Z_B: int | None, n_coord: int,
                     structure: str) -> float:
    """Compute C, the heteropolar (ionic) part of the band gap."""
    if Z_B is None:
        return 0.0

    ie_A = IE_NIST.get(Z_A, 8.0)
    ie_B = IE_NIST.get(Z_B, 8.0)
    dIE = abs(ie_A - ie_B)
    per_A = period(Z_A)
    per_B = period(Z_B)
    per_avg = (per_A + per_B) / 2.0

    if structure in ('diamond', 'zincblende', 'wurtzite'):
        # Covalent-dominated: P1 vertex coupling
        # S3 + D3 = full P1 channel strength (filter + quantum defect)
        return dIE * (S3 + D3)

    # Ionic-dominated (rocksalt, fluorite):
    # Bare IE difference + Madelung correction through P1+P2 channels
    # n_val = number of electrons transferred per formula unit
    n_val = _valence_charge(Z_A, Z_B)
    C_madelung = (S3 + S5) * n_val * RY / (per_avg ** 2)
    return dIE + C_madelung


def _valence_charge(Z_A: int, Z_B: int | None) -> int:
    """Effective valence charge transfer per formula unit.

    Derived from group position (periodic table structure from PT).
    """
    if Z_B is None:
        return 0
    # Divalent: Be, Mg, Ca, Sr, Ba, Ra
    if Z_A in _DIVALENT or Z_B in _DIVALENT:
        return 2
    # Trivalent cations: Al, Ga, In, Tl (group 13)
    if Z_A in _TRIVALENT_CATIONS or Z_B in _TRIVALENT_CATIONS:
        return 1  # partial ionicity in III-V
    return 1  # default monovalent


# ╔════════════════════════════════════════════════════════════════════╗
# ║  BAND GAP: Pythagorean combination (Principle 3)                  ║
# ╚════════════════════════════════════════════════════════════════════╝

def band_gap_pt(Z_A: int, Z_B: int | None = None,
                structure: str = 'diamond') -> float:
    """Compute the PT band gap in eV.

    Parameters
    ----------
    Z_A : int
        Atomic number of element A (or the only element for elemental).
    Z_B : int or None
        Atomic number of element B for binary compounds.
        None for elemental semiconductors (Si, Ge, C diamond).
    structure : str
        Crystal structure: 'diamond', 'zincblende', 'rocksalt', 'wurtzite'.

    Returns
    -------
    float
        Band gap in eV.

    Theory
    ------
    E_gap = sqrt(E_h^2 + C^2)

    The covalent and ionic channels are ORTHOGONAL in the CRT
    (Principle 3), so they combine as a Pythagorean sum.
    This is the crystalline version of the molecular bifurcation P4.
    """
    struct = structure.lower()
    n_coord = _COORD.get(struct, 4)

    E_h = _homopolar_gap(Z_A, Z_B, n_coord, struct)
    C = _heteropolar_gap(Z_A, Z_B, n_coord, struct)

    return math.sqrt(E_h ** 2 + C ** 2)


# ╔════════════════════════════════════════════════════════════════════╗
# ║  CLASSIFICATION                                                    ║
# ╚════════════════════════════════════════════════════════════════════╝

def classify_material(E_gap: float, T: float = 300.0) -> str:
    """Classify a material based on its band gap.

    Parameters
    ----------
    E_gap : float
        Band gap in eV.
    T : float
        Temperature in Kelvin (default: 300 K).

    Returns
    -------
    str
        'metal', 'semiconductor', or 'insulator'.

    Theory
    ------
    Metal:         E_gap < s * kT        (gap closes at thermal scale)
    Semiconductor: s * kT < E_gap <= Ry/P1  (between thermal and Shannon cap)
    Insulator:     E_gap > Ry/P1         (gap exceeds Shannon cap ~ 4.5 eV)

    The Shannon cap Ry/P1 ~ 4.5 eV is the information-theoretic limit
    per CRT face.  Materials with gaps above this cannot conduct
    through any single PT channel at room temperature.
    """
    kT = K_BOLTZMANN_EV * T
    thermal_threshold = S_HALF * kT           # ~ 0.013 eV at 300 K
    insulator_threshold = RY / P1             # Shannon cap ~ 4.535 eV

    if E_gap < thermal_threshold:
        return 'metal'
    elif E_gap <= insulator_threshold:
        return 'semiconductor'
    else:
        return 'insulator'


# ╔════════════════════════════════════════════════════════════════════╗
# ║  DIELECTRIC CONSTANT                                               ║
# ║                                                                    ║
# ║  PT Penn model (electronic):                                       ║
# ║    eps_e = 1 + C_Penn / E_gap^(s + delta_3)                       ║
# ║    C_Penn = P1^2 + P1*cos^2_3 + delta_3 = 11.46 [all from s=1/2] ║
# ║                                                                    ║
# ║  Ionic polarizability (polar crystals only):                       ║
# ║    eps_i = (C/E_gap)^2 * n_coord * s * n_charge                   ║
# ╚════════════════════════════════════════════════════════════════════╝

# C_Penn = P1^2 + P1*C3 + D3  [derived from s=1/2, 0 parameters]
_C_PENN = P1 ** 2 + P1 * C3 + D3

# Penn exponent = s + delta_3 = 0.5 + 0.116 = 0.616
_PENN_POWER = S_HALF + D3


def dielectric_constant_pt(Z_A: int, Z_B: int | None = None,
                           structure: str = 'diamond') -> float:
    """Compute the static dielectric constant from PT.

    Parameters
    ----------
    Z_A, Z_B, structure : as in band_gap_pt

    Returns
    -------
    float
        Static dielectric constant (dimensionless).

    Theory
    ------
    Electronic (PT Penn model):
      eps_e = 1 + C_Penn / E_gap^(s + delta_3)
      C_Penn = P1^2 + P1*cos^2_3 + delta_3

    Ionic (polar crystals):
      eps_i = (C_ion/E_gap)^2 * n_coord * s * n_charge

    Total: eps = eps_e + eps_i
    """
    struct = structure.lower()
    n_coord = _COORD.get(struct, 4)

    E_gap = band_gap_pt(Z_A, Z_B, struct)
    if E_gap < 1e-10:
        return float('inf')

    # Electronic dielectric (Penn model, PT-derived constants)
    eps_electronic = 1.0 + _C_PENN / (E_gap ** _PENN_POWER)

    # Ionic polarizability (only for polar/binary compounds)
    eps_ionic = 0.0
    if Z_B is not None:
        C_ion = _heteropolar_gap(Z_A, Z_B, n_coord, struct)
        n_charge = _valence_charge(Z_A, Z_B)
        if C_ion > 0 and E_gap > 0:
            eps_ionic = (C_ion / E_gap) ** 2 * n_coord * S_HALF * n_charge

    return eps_electronic + eps_ionic


# ╔════════════════════════════════════════════════════════════════════╗
# ║  RESULT DATACLASS                                                  ║
# ╚════════════════════════════════════════════════════════════════════╝

@dataclass
class MaterialResult:
    """Result container for PT materials calculation.

    Attributes
    ----------
    E_gap : float
        Band gap in eV.
    classification : str
        'metal', 'semiconductor', or 'insulator'.
    epsilon : float
        Static dielectric constant.
    E_h : float
        Homopolar (covalent) gap component in eV.
    C_ionic : float
        Heteropolar (ionic) gap component in eV.
    """
    E_gap: float
    classification: str
    epsilon: float
    E_h: float
    C_ionic: float


def analyze_material(Z_A: int, Z_B: int | None = None,
                     structure: str = 'diamond',
                     T: float = 300.0) -> MaterialResult:
    """Full PT analysis of a material.

    Parameters
    ----------
    Z_A : int
        Atomic number of element A.
    Z_B : int or None
        Atomic number of element B (None for elemental).
    structure : str
        Crystal structure type.
    T : float
        Temperature in Kelvin.

    Returns
    -------
    MaterialResult
        Complete PT materials analysis.
    """
    struct = structure.lower()
    n_coord = _COORD.get(struct, 4)

    E_h = _homopolar_gap(Z_A, Z_B, n_coord, struct)
    C = _heteropolar_gap(Z_A, Z_B, n_coord, struct)
    E_gap = math.sqrt(E_h ** 2 + C ** 2)
    classification = classify_material(E_gap, T)
    epsilon = dielectric_constant_pt(Z_A, Z_B, structure)

    return MaterialResult(
        E_gap=E_gap,
        classification=classification,
        epsilon=epsilon,
        E_h=E_h,
        C_ionic=C,
    )


# ╔════════════════════════════════════════════════════════════════════╗
# ║  LABEL HELPER                                                     ║
# ╚════════════════════════════════════════════════════════════════════╝

def _label(Z_A: int, Z_B: int | None) -> str:
    """Human-readable label for a material."""
    sym_A = SYMBOLS.get(Z_A, f'Z{Z_A}')
    if Z_B is None:
        return sym_A
    sym_B = SYMBOLS.get(Z_B, f'Z{Z_B}')
    return f'{sym_A}{sym_B}'
