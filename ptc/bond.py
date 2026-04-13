"""
bond.py — Bond observables (r_e, ω_e) and result dataclass.

The bond ENERGY is computed in molecule.py via the molecular polygon.
This module provides physical observables and the BondResult container.

March 2026 — Persistence Theory
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


def r_equilibrium(per_A: int, per_B: int, bo: float,
                  S_bond: float = 0.0,
                  z_A: int = 1, z_B: int = 1,
                  lp_A: int = 0, lp_B: int = 0) -> float:
    """Equilibrium bond distance from Bianchi I metric + coordination (Å).

    r_e = r_base × f_coord × exp(S × D₃ × s)

    r_base: A_BOHR × per_eff / (2 × bo^(1/3) × γ_eff)
    f_coord: coordination stretch — more bonds/LP at vertex → longer bond.
      f_coord = (z_eff_A × z_eff_B)^(1/(2P₁))
      where z_eff = z + lp × weight on polygon Z/(2P₁)Z.
    """
    import math
    from ptc.constants import P1 as _P1, S3 as _S3
    # Metric inflation factor from bond order [T6, D10 Bianchi I]
    w_s = 1.0 / bo
    w_p = min(bo - 1, 2) / bo
    gamma_eff = GAMMA_3 ** w_s * GAMMA_5 ** w_p

    # Period effective: geometric mean (asymmetric bonds)
    per_eff = 2.0 * math.sqrt(max(per_A, 1) * max(per_B, 1))

    r_base = A_BOHR * per_eff / (2.0 * bo ** (1.0 / 3.0) * gamma_eff)

    # Coordination stretch [polygon Z/(2P₁)Z occupancy]
    # More bonds at vertex → bond pushed outward.
    # Only BONDS stretch (not LP — LP are perpendicular to bond axis).
    # LP contribute only at polyatomic vertices (z≥2) where they compete
    # for in-plane space, with S₃ weight.
    lp_w_A = lp_A * _S3 if z_A >= 2 else 0
    lp_w_B = lp_B * _S3 if z_B >= 2 else 0
    z_eff_A = max(z_A + lp_w_A, 1.0)
    z_eff_B = max(z_B + lp_w_B, 1.0)
    # Stretch power 1/P₁²: very gentle, avoids overcorrection
    f_coord = (z_eff_A * z_eff_B) ** (1.0 / (_P1 * _P1))

    r_e = r_base * f_coord

    # ── LP-LP PAULI REPULSION for terminal diatomics ──
    # When BOTH atoms are terminal (z=1) with LP, their LP face each
    # other head-on along the bond axis. The Pauli exclusion between
    # the LP pairs on Z/(2P₁)Z stretches the bond.
    # PT mechanism: δ₃ holonomic shift per mutual LP pair, normalized
    # by the period effective radius (larger atoms → weaker repulsion).
    # Only fires for z=1 (terminal); polyatomic LP handled by f_coord.
    if z_A == 1 and z_B == 1 and lp_A > 0 and lp_B > 0:
        lp_min = min(lp_A, lp_B)
        r_e *= (1.0 + _D3 * lp_min / per_eff)

    # Screening-geometry coupling [Principe 5: holonomic rotation]
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
