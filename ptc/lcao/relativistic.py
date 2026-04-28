"""Per-atom valence contraction factors (PT-pure).

PT-pure radial contraction for heavy elements (Z >= 57).
All factors derived from the s = 1/2 cascade gamma_p — no fit, no empirical.

Three layered contributions:
  - lanthanide_factor(Z) : 4f imperfect screening (Z=57..71), gamma_5 ramp
  - relativistic_factor(Z) : Dirac kinematic (Z>=72), sqrt(1-(Z*alpha)^2)
  - actinide_factor(Z) : 5f imperfect screening (Z=90..103), gamma_7 ramp

Combined per-atom: valence_contraction(Z) = gamma_la * gamma_rel * gamma_an,
applied as bond-pair product r_base *= g(Z_A) * g(Z_B) in r_equilibrium.

The lanthanide and actinide ramps use the half-exponent sqrt(gamma_p) so that
the per-atom factor at saturation (Z=71 or Z=103) reproduces the empirical
post-lanthanide / post-actinide bond shrinkage when applied as a pair-product.
"""
from __future__ import annotations

import math

from ptc.constants import ALPHA_PHYS, GAMMA_5, GAMMA_7

_SQRT_GAMMA_5 = math.sqrt(GAMMA_5)
_SQRT_GAMMA_7 = math.sqrt(GAMMA_7)

LANTHANIDE_Z_MIN = 57
LANTHANIDE_Z_MAX = 71
LANTHANIDE_FILL = 15
ACTINIDE_Z_MIN = 90
ACTINIDE_Z_MAX = 103
ACTINIDE_FILL = 14
RELATIVISTIC_Z_MIN = 72


def lanthanide_factor(Z: int) -> float:
    """4f imperfect screening contraction (Z = 57..71)."""
    if Z < LANTHANIDE_Z_MIN or Z >= RELATIVISTIC_Z_MIN:
        return 1.0
    f = (Z - (LANTHANIDE_Z_MIN - 1)) / float(LANTHANIDE_FILL)
    return _SQRT_GAMMA_5 ** f


def relativistic_factor(Z: int) -> float:
    """Dirac kinematic radial contraction (Z >= 72)."""
    if Z < RELATIVISTIC_Z_MIN:
        return 1.0
    Za2 = (Z * ALPHA_PHYS) ** 2
    if Za2 >= 1.0:
        return 0.5
    return math.sqrt(1.0 - Za2)


def scalar_rel_factor(Z: int) -> float:
    """Scalar relativistic kinematic contraction for ALL Z (Phase 6.B.5).

    Returns sqrt(1 - (Z*alpha)^2) — the Dirac kinematic factor without
    the Z >= 72 threshold of ``relativistic_factor``. PT-natural with
    zero parameters: alpha = 1/137.036 from PT q_stat sin² product.

    Use this for chemical-shielding corrections on moderate-Z atoms
    (Cl Z=17 gives γ=0.992 → 0.8% contraction ; Br Z=35 → γ=0.967 ;
    I Z=53 → γ=0.911) where the conventional ``relativistic_factor``
    returns 1.0 because the threshold is set for Pyykkö-style heavy-
    element NMR work.

    For NMR shielding the kinematic contraction renormalises σ_p by
    roughly the same factor at the heavy nucleus.
    """
    if Z < 1:
        return 1.0
    Za2 = (Z * ALPHA_PHYS) ** 2
    if Za2 >= 1.0:
        return 0.5
    return math.sqrt(1.0 - Za2)


def actinide_factor(Z: int) -> float:
    """5f imperfect screening contraction (Z = 90..103)."""
    if Z < ACTINIDE_Z_MIN:
        return 1.0
    f = min(1.0, (Z - (ACTINIDE_Z_MIN - 1)) / float(ACTINIDE_FILL))
    return _SQRT_GAMMA_7 ** f


def valence_contraction(Z: int) -> float:
    """Combined per-atom valence contraction g(Z) = gamma_la * gamma_rel * gamma_an."""
    return (lanthanide_factor(Z)
            * relativistic_factor(Z)
            * actinide_factor(Z))
