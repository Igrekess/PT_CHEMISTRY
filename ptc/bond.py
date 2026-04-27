"""
bond.py — Bond observables (r_e, ω_e) and result dataclass.

The bond ENERGY is computed in molecule.py via the molecular polygon.
This module provides physical observables and the BondResult container.

March 2026 — Théorie de la Persistance
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ptc.constants import A_BOHR, GAMMA_3, GAMMA_5, D3 as _D3, S_HALF as _S_HALF


@dataclass(frozen=True)
class BondResult:
    """Result of D₀ computation for one A-B bond."""
    D0: float           # total bond dissociation energy (eV)
    v_sigma: float      # σ contribution (eV)
    v_pi: float         # π contribution (eV)
    v_ionic: float      # ionic contribution (eV)
    r_e: float          # equilibrium distance (Å)
    omega_e: float      # vibrational frequency (cm⁻¹)
    details: dict = None
    geometry: 'BondGeometry' = None


@dataclass(frozen=True)
class BondGeometry:
    """Per-bond 3D geometry emerging from screening on T³.

    r_e: equilibrium distance from exp(-S_bond)
    theta_A/B: angle at each vertex from face_fraction × solid_angle
    sin2_half_A/B: sin²(θ/2) — T6 holonomy direct
    lp_proj_A/B: LP projection on bond axis (face overflow)
    """
    r_e: float
    theta_A: float
    theta_B: float
    sin2_half_A: float
    sin2_half_B: float
    lp_proj_A: float
    lp_proj_B: float


def _Q_koide_excess(n_p: int) -> float:
    """Koide structural excess on p-orbitals."""
    if n_p <= 0 or n_p > 6:
        return 0.0
    pops = [0, 0, 0]
    for i in range(n_p):
        pops[i % 3] += 1
    sqrt_sum = sum(p ** 0.5 for p in pops)
    total = sum(pops)
    if total <= 0:
        return 0.0
    Q = sqrt_sum ** 2 / (3.0 * total)
    return max(0.0, min((Q - 2.0 / 3.0) / (1.0 - 2.0 / 3.0), 1.0))


def _gamma_rel(Z: int) -> float:
    """Dirac kinematic radial contraction γ_rel = √(1 − (Zα)²) for Z ≥ 72."""
    if Z < 72:
        return 1.0
    from ptc.constants import ALPHA_PHYS as _A
    Za2 = (Z * _A) ** 2
    if Za2 >= 1.0:
        return 0.5
    return math.sqrt(1.0 - Za2)


def r_equilibrium(per_A: int, per_B: int, bo: float,
                  S_bond: float = 0.0,
                  z_A: int = 1, z_B: int = 1,
                  lp_A: int = 0, lp_B: int = 0,
                  Z_A: int = 0, Z_B: int = 0) -> float:
    """Equilibrium bond distance from Bianchi I metric + coordination (Å)."""
    import math
    from ptc.constants import P1 as _P1, S3 as _S3
    w_s = 1.0 / bo
    w_p = min(bo - 1, 2) / bo
    gamma_eff = GAMMA_3 ** w_s * GAMMA_5 ** w_p

    per_eff = 2.0 * math.sqrt(max(per_A, 1) * max(per_B, 1))

    r_base = A_BOHR * per_eff / (2.0 * bo ** (1.0 / 3.0) * gamma_eff)

    # ── Relativistic Dirac contraction (both Z ≥ 72) ──
    if Z_A >= 72 and Z_B >= 72:
        r_base *= _gamma_rel(Z_A) * _gamma_rel(Z_B)

    # ── 4f / 5f imperfect screening (per-atom, always applied) ──
    from ptc.lcao.relativistic import lanthanide_factor, actinide_factor
    r_base *= lanthanide_factor(Z_A) * lanthanide_factor(Z_B)
    r_base *= actinide_factor(Z_A) * actinide_factor(Z_B)

    lp_w_A = lp_A * _S3 if z_A >= 2 else 0
    lp_w_B = lp_B * _S3 if z_B >= 2 else 0
    z_eff_A = max(z_A + lp_w_A, 1.0)
    z_eff_B = max(z_B + lp_w_B, 1.0)
    f_coord = (z_eff_A * z_eff_B) ** (1.0 / (_P1 * _P1))

    r_e = r_base * f_coord

    if S_bond > 0:
        r_e *= math.exp(S_bond * _D3 * _S_HALF)
    return r_e


def omega_e(D0: float, r_e: float, mass_A: float, mass_B: float) -> float:
    """Harmonic vibrational frequency (cm⁻¹)."""
    if r_e <= 0 or D0 <= 0:
        return 0.0
    mu = mass_A * mass_B / (mass_A + mass_B)
    k_force = 2.0 * D0 / (r_e ** 2)
    k_Nm = k_force * 16.022
    mu_kg = mu * 1.6605e-27
    if mu_kg <= 0:
        return 0.0
    omega_rad = math.sqrt(k_Nm / mu_kg)
    return omega_rad / (2.0 * math.pi * 2.998e10)
