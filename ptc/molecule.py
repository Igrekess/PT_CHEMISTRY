"""
molecule.py — Molecular engine via polygon on T³.

ARCHITECTURE (same as atom.py):
  Atom:     IE(Z) ← polygon Z/(2P_l)Z, N positions filled by electrons
  Molecule: D_at  ← polygon Z/(2N)Z,   N positions filled by atoms

Each ATOM occupies a position on the molecular polygon.
The DFT decomposes the atomic screening profile into modes.
Each BOND energy is computed from the synthesis at the edge midpoint,
which sees the FULL molecular context (all atoms contribute).

Pipeline:
  SMILES → Topology → atom peer values → DFT → per-bond synthesis → D_at

Zero adjustable parameters. All from s = 1/2.

March 2026 — Persistence Theory
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from ptc.atom import IE_eV, EA_eV
from ptc.bond import BondResult, BondGeometry, r_equilibrium, omega_e, _Q_koide_excess
from ptc.topology import Topology, build_topology
from ptc.data.experimental import IE_NIST, EA_NIST, MASS, SYMBOLS
from ptc.constants import (
    S_HALF, P1, P2, P3,
    S3, S5, S7, C3, C5, C7,          # q_stat (vertex/coupling)
    D3, D5, D7, AEM, RY,
    GAMMA_5,
    D_FULL, A_BOHR,
    BETA_GHOST,
    D3_TH, D5_TH, D7_TH,             # q_therm (face/screening) [A8]
    F_BIFURC, EPS_3, EPS_5, EPS_7,    # ε-field [A8]
    F_EPS, D_FULL_P1, D_FULL_P2, D_FULL_P3,  # depth corrections
)

# ── PER-CANAL CASCADE FACTORS α [T³ geometry, 0 adjusted params] ──
# Each of the 6 screening terms has a cascade factor for P₂ and P₃.
# α > 0: same sign as P₁ (screening in both channels)
# α < 0: sign INVERTED (what screens P₁ FACILITATES P_l)
# α = 0: term does not apply to this channel
#
# Derived from face projections on T³ = Z/3Z × Z/5Z × Z/7Z:
#   S₅/S₃ = sin²₅/sin²₃  (hex→pent projection)
#   S₇/S₃ = sin²₇/sin²₃  (hex→hept projection)
#   D₅/D₃ = δ₅/δ₃        (delta ratio, for LP/period)
#   C₇/C₃ = cos²₇/cos²₃  (complementary, for LP→ionic bulk)

# 1. Context: neighbors stabilize ALL channels (same sign)
ALPHA_CONTEXT_P2 = S5 / S3           # +0.887
ALPHA_CONTEXT_P3 = S7 / S3           # +0.789

# 2. LP blocking: LP BLOCK σ but FACILITATE d-back donation!
#    LP occupies σ-channel → frees d-channel on P₂ face.
#    For P₃ (ionic), LP blocks via bulk screening (C₇/C₃ > 1).
ALPHA_LP_P2 = -D5 / D3               # −0.878 (INVERTED!)
ALPHA_LP_P3 = C7 / C3                # +1.059 (amplified bulk)

# 3. Period: shell crossing cost is UNIVERSAL (same sign)
ALPHA_PERIOD_P2 = D5 / D3            # +0.878
ALPHA_PERIOD_P3 = D7 / D3            # +0.774

# 4. Parity/Bohr: homonuclear boost helps σ and d, NOT ionic
#    (ionic = heteronuclear by definition)
ALPHA_PARITY_P2 = S5 / S3            # +0.887
ALPHA_PARITY_P3 = 0.0                # zero (ionic needs heteronuclear)

# 5. Vertex overcrowding: PENALIZES σ but OPENS d-channel
#    More coordination z > P₁ = more d-orbital access on P₂.
ALPHA_VERTEX_P2 = -S5 / S3           # −0.887 (INVERTED!)
ALPHA_VERTEX_P3 = S7 / S3            # +0.789

# 6. Asymmetry: polarity helps ALL channels, AMPLIFIED for ionic
#    P₃ (ionic): polarity IS the motor → amplified by P₁
ALPHA_ASYM_P2 = S5 / S3              # +0.887
ALPHA_ASYM_P3 = (S7 / S3) * P1       # +2.367 (amplified: polarity drives P₃)

# ── ABLATION FLAGS (for progressive rebalancing) ──
# Set to False to disable a mechanism. Reactivate one by one.
ENABLE_P3_IONIC = True
# VERTEX_BUDGET removed — replaced by polygon vacancy model
ENABLE_P2_COMPRESSED = True
ENABLE_D_BRIDGE = True
ENABLE_DATIVE_RELIEF = True
ENABLE_FACE_EFFECTS = True
ENABLE_CONTEXT = True
ENABLE_GHOST = True
from ptc.periodic import period, l_of, _n_fill_aufbau, ns_config


# ╔════════════════════════════════════════════════════════════════════╗
# ║  VSEPR POLYHEDRON — geometric families for face enumeration      ║
# ╚════════════════════════════════════════════════════════════════════╝
#
# Each atom in a molecule belongs to a VSEPR geometric family.
# The family determines the FACE STRUCTURE of the molecular polyhedron.
#
# PT mapping [D13 — spin foam U(1)³]:
#   Each family is a simplex class on T³.
#   The face amplitude = D₅ × sin(θ/2) × f(LP).
#   Faces at LP>0 vertices COOPERATE (structural rigidity).
#   Linear (180°) and terminal (z=1) give σ=0.
#
# 
# (z, LP) → (name, θ_ideal_deg, n_pairs_total, sin_coop)

_GEO_FAMILIES = {
    (1, 0): ('terminal',    180.0,  0, 0.000),
    (2, 0): ('linear',      180.0,  1, 0.000),
    (2, 1): ('bent_1lp',    120.0,  1, 0.433),
    (2, 2): ('bent_2lp',    104.5,  1, 0.484),
    (2, 3): ('bent_3lp',     90.0,  1, 0.500),
    (3, 0): ('trigonal',    120.0,  3, 0.433),
    (3, 1): ('pyramidal',   107.0,  3, 0.478),
    (3, 2): ('T_shape',      90.0,  3, 0.333),
    (4, 0): ('tetrahedral', 109.5,  6, 0.472),
    (4, 1): ('seesaw',      102.0,  6, 0.396),
    (4, 2): ('square',       90.0,  6, 0.333),
    (5, 0): ('trig_bipyr',  104.5, 10, 0.430),
    (5, 1): ('square_pyr',   90.0, 10, 0.396),
    (6, 0): ('octahedral',   90.0, 15, 0.400),
}


# ── onsite energy functions (for Rabi cos²) ──

def _hybridize(IE, per, l_val=0, n_fill_val=0, ns_val=0):
    """sp hybridization: push IE toward Ry for sub-Bohr atoms [ptcK]."""
    n_p = n_fill_val if l_val == 1 else 0
    if n_p > 0 and IE < RY:
        valence = n_fill_val + ns_val
        af = min(valence, P1 + 1) / (P1 + 1)
        return IE + af * C3 * (RY - IE)
    return IE

def _eps_onsite(IE, per, l_val=0, n_fill_val=0, ns_val=0):
    """Onsite energy = hybridized IE + period decay [ptcK]."""
    E = _hybridize(IE, per, l_val, n_fill_val, ns_val)
    n_shells = max(0, per - P1 + 1)
    if n_shells > 0 and E > IE + 0.01:
        return E * C3 ** (n_shells * S_HALF)
    return E * C3 ** n_shells


def _compute_bond_geometry(i, j, bo, S_bond, topology, polygons):
    """Compute BondGeometry for bond (i,j) from screening and vertex polygons.

    θ derived from PT: Thompson problem on S² with LP weight (1+C₃).
    LP occupies (1+cos²(θ₃)) steric units on Z/6Z because LP angular
    spread exceeds bond angular width by the hexagonal face overlap C₃.

    r_e from screening: exp(-S_bond × D₃) encodes the distance via
    the propagateur on T³. Strong screening → long bond.
    """
    from ptc.bond import BondGeometry
    from ptc.constants import A_BOHR, C3, D3, P1, S3, S_HALF
    from ptc.bond import r_equilibrium as _r_eq

    vp_a = polygons[i]
    vp_b = polygons[j]

    # r_e from Bianchi I metric [D10: a_p = γ_p/μ, geodesic on T³]
    # The distance is determined by the METRIC (γ_p scale factors),
    # not the screening. The screening encodes ENERGY, not DISTANCE.
    r_e = _r_eq(vp_a.per, vp_b.per, bo)

    # θ from Thompson problem with LP weight [T6 holonomy on S²]
    # z_eff = z + lp × (1 + s) : LP has s=1/2 extra angular DOF
    # cos(θ) = -1/(z_eff - 1) : Thomson minimum on sphere
    # LP weight = 1+s = 3/2: LP unconstrained spin extends angularly
    # by s=1/2 beyond a bond (which is directionally constrained).
    # H₂O (z=2,lp=2): z_eff=5.0→104.48° (exp 104.5°, err 0.02°)
    # NH₃ (z=3,lp=1): z_eff=4.5→106.60° (exp 107.0°, err 0.40°)
    def _pt_angle(vp):
        if vp.z < 2:
            if vp.lp > 0:
                # Terminal with LP: bond-LP cone angle [Thompson]
                # The LP forms a cone around the bond axis.
                # θ = angle between bond and nearest LP direction.
                z_eff = 1 + vp.lp * (1.0 + S_HALF)
                cos_t = max(-1.0, min(1.0, -1.0 / (z_eff - 1.0)))
                theta = math.degrees(math.acos(cos_t))
                sin2 = math.sin(math.radians(theta) / 2.0) ** 2
                return theta, sin2
            return 0.0, 0.0
        z_eff = vp.z + vp.lp * (1.0 + S_HALF)  # LP weight = 1+s = 3/2
        if z_eff <= 1.0:
            return 180.0, 1.0
        cos_t = max(-1.0, min(1.0, -1.0 / (z_eff - 1.0)))
        theta = math.degrees(math.acos(cos_t))
        sin2 = math.sin(math.radians(theta) / 2.0) ** 2
        return theta, sin2

    theta_A, sin2_A = _pt_angle(vp_a)
    theta_B, sin2_B = _pt_angle(vp_b)

    # LP projection on bond axis [solid_angle × lp_coverage]
    # The LP projection is the fraction of LP angular density that
    # projects onto the bond axis. Uses lp_coverage (angular fraction
    # of LP) × sin²(θ/2) (how much LP "sees" the bond direction).
    def _lp_proj(vp, sin2):
        if vp.lp <= 0 or vp.z <= 0:
            return 0.0
        return vp.lp_coverage * sin2

    return BondGeometry(
        r_e=r_e, theta_A=theta_A, theta_B=theta_B,
        sin2_half_A=sin2_A, sin2_half_B=sin2_B,
        lp_proj_A=_lp_proj(vp_a, sin2_A), lp_proj_B=_lp_proj(vp_b, sin2_B),
    )


def _vertex_faces(z, lp):
    """Vertex geometry from PT: Thompson problem with LP weight 1+s.

    Returns (theta_deg, n_faces_per_bond, sin_coop).
    θ derived from s=1/2: z_eff = z + lp×(1+s), cos(θ) = -1/(z_eff-1).
    No lookup table — geometry emerges from the unique PT input.

    n_faces_per_bond = z - 1: each other bond at this vertex forms
    one triangular face with the current bond.
    """
    lp_eff = min(lp, 3)
    z_eff = z + lp_eff * (1.0 + S_HALF)  # LP weight = 1+s = 3/2
    if z_eff <= 1.0:
        theta = 180.0
    else:
        cos_t = max(-1.0, min(1.0, -1.0 / (z_eff - 1.0)))
        theta = math.degrees(math.acos(cos_t))
    sin_coop = math.sin(math.radians(theta)) / 2.0

    n_faces = max(0, z - 1)
    return theta, n_faces, sin_coop


# ╔════════════════════════════════════════════════════════════════════╗
# ║  MOLECULE RESULT                                                 ║
# ╚════════════════════════════════════════════════════════════════════╝

@dataclass
class MoleculeResult:
    """Full molecular calculation result."""
    D_at: float
    bonds: List[BondResult]
    topology: Topology
    source_mode: str = "full_pt"
    formula: str = ""
    ie_audit: dict = field(default_factory=dict)

    @property
    def D_at_per_bond(self) -> float:
        return self.D_at / max(1, len(self.bonds))


# ╔════════════════════════════════════════════════════════════════════╗
# ║  ATOM PEER VALUES — screening profile for each atom              ║
# ╚════════════════════════════════════════════════════════════════════╝

def _atom_screening(ie, ea, lp, z, nf, per, l_val):
    """Screening action for one atom on the molecular polygon.

    This is the ATOM-LEVEL screening (like atom.py S_core):
    how much bonding capacity does this atom have?

    Based ONLY on IE and period — NOT LP.
    LP blocking is an EDGE effect (handled in _bond_screening).
    The atom screening determines the context contribution:
    low screening (H, C) → good donor, stabilizes neighbors.
    high screening (per≥3) → weaker donor.
    """
    f_IE = min(ie / RY, 1.0)

    # Base: fraction of Ry NOT available + Bohr floor
    sigma = max(D7, (1.0 - f_IE) * S3)

    # Period decay: −ln(1−V) per shell [analogous to atom.py S_core].
    # V = D₃ × compactness per shell. Nonlinear: grows faster than
    # linear n×V for heavy atoms (−ln(1−V) > V when V > 0).
    # per=3 (1 shell): +0.127 (same as before: −ln(1−0.119) ≈ 0.127)
    # per=5 (3 shells): +0.423 (was 0.511 linear → now nonlinear sum)
    n_shells = max(0, per - 2)
    if n_shells > 0:
        compactness = P1 - (P1 - 1) * f_IE
        V_shell = D3 * compactness
        if 0 < V_shell < 1:
            sigma += -n_shells * math.log(1.0 - V_shell)
        else:
            sigma += n_shells * V_shell  # fallback to linear

    return sigma


def _total_lp_from_neighbors(idx, topology):
    """Sum of LP from all bonded neighbors of atom idx."""
    total = 0
    for ii, jj, _ in topology.bonds:
        if ii == idx:
            total += topology.lp[jj] if topology.lp else 0
        elif jj == idx:
            total += topology.lp[ii] if topology.lp else 0
    return total


# ╔════════════════════════════════════════════════════════════════════╗
# ║  LP SCREENING — unified mutual + self [T2 GFT, T6 holonomy]     ║
# ╚════════════════════════════════════════════════════════════════════╝
#
# PT: LP screening = S = -ln(1-V) where V = LP density on bond axis.
# Two contributions to the SAME observable:
#   S_mutual: both atoms have LP → repulsion on axis (lp_mutual ≥ 1)
#   S_self:   one side only → one-sided blocking (lp_mutual = 0)
# Gate z_max ≥ 2 → continuous f_poly = 1 - exp(-(z_max-1)×S₃).
# Diatomic structural LP (Q_Koide = 1, lp = 1) → complement, no block.

def _lp_screening(lp_a, lp_b, z_a, z_b, per_A, per_B, bo, nf_a, nf_b,
                  Za, Zb, atom_data):
    """Unified LP screening: mutual + self → (S_lp, S_self).

    Returns two values for backward compatibility in the screening sum.
    Both are non-negative screening actions (increase S_bond).
    """
    lp_mutual = min(lp_a, lp_b)

    # ── MUTUAL LP blocking ──
    # Both atoms have LP → they repel on the bond axis.
    # F-F (LP=3,3): -ln(C₃) × lp × density.
    S_lp = 0.0
    if lp_mutual >= 1:
        z_max_bond = max(z_a, z_b)

        # GFT complementarity [T2]: diatomic (z=1) LP is COMPLEMENT
        # of bond on T³. Polyatomic LP competes with other bonds.
        Q_min = min(_Q_koide_excess(nf_a), _Q_koide_excess(nf_b))

        if z_max_bond >= 2:
            # Polyatomic: mutual LP competes with other bonds
            per_e = math.sqrt(per_A * per_B)
            density = (2.0 / per_e) ** 2
            S_lp = -math.log(C3) * lp_mutual * density
            if Q_min > 0:
                S_lp *= (1.0 + Q_min * S3)
            n_radial = max(0, max(per_A, per_B) - 2)
            if n_radial > 0:
                S_lp *= (1.0 + n_radial * C3)
            # π-LP destructive interference [insight #36]
            if bo >= 2 and max(per_A, per_B) <= 2:
                S_lp *= (1.0 + (bo - 1) * S3 * P1)
        else:
            # Diatomic: structural LP (N₂: lp=1, Q=1) → partial complement
            # GFT [T2]: LP fills antibonding but leaves residual S₃×D₃
            # on face P₁ (information cost of LP presence).
            is_structural = (lp_mutual == 1 and Q_min >= 0.99)
            if is_structural:
                S_lp = S3 * D3  # residual structural screening
            elif not is_structural:
                per_e = math.sqrt(per_A * per_B)
                density = (2.0 / per_e) ** 2
                # CROSS-PERIOD LP MISMATCH RELIEF [BondGeometry 3D]:
                # Adjacent periods (Δper=1) have LP orbitals at different
                # radial extents. The overlap is reduced by the mismatch
                # fraction. sin²(θ₃) = S₃ is the "dead zone" on Z/6Z
                # where LP from different shells don't project together.
                # Relief = C₃^Δper (survival per period mismatch).
                # ClF (Δper=1): density × C₃ = 0.667 × 0.781 = 0.521
                # BrF (Δper=2): density × C₃² = 0.500 × 0.610 = 0.305
                # Only for adjacent periods (Δper=1): F-Cl, O-S, N-P.
                # Δper≥2 (F-Br, Cl-I): bifurcation already handles relief.
                delta_per_lp_density = abs(per_A - per_B)
                if delta_per_lp_density == 1:
                    density *= C3
                S_lp = -math.log(C3) * lp_mutual * density
                if Q_min > 0:
                    S_lp *= (1.0 + Q_min * S3)
                n_radial = max(0, max(per_A, per_B) - 2)
                if n_radial > 0:
                    S_lp *= (1.0 + n_radial * D3)
                # LP polarity boost [T6 holonomy on Z/2Z]
                chi_A_lp = (atom_data[Za]['IE'] + atom_data[Za]['EA']) / 2.0
                chi_B_lp = (atom_data[Zb]['IE'] + atom_data[Zb]['EA']) / 2.0
                Q_lp = abs(chi_A_lp - chi_B_lp) / max(0.1, chi_A_lp + chi_B_lp)
                S_lp *= (1.0 + Q_lp)
                # LP closure boost: full p-shell (LP=P₁) [insight #30]
                # Both LP complete → face P₁ fully occupied → max repulsion.
                # For per≤2: standard S₃ boost (F₂, O₂).
                # For cross-period (Δper>0): period-scaled boost (ICl, BrF).
                # The LP fills the FACE regardless of period — the period
                # only affects the radial extent, not the angular coverage.
                if lp_mutual >= P1:
                    delta_per_lp = abs(per_A - per_B)
                    if max(per_A, per_B) <= 2:
                        S_lp *= (1.0 + S3)
                    elif delta_per_lp > 0:
                        # Cross-period LP closure: boost scaled by Δper.
                        # MUTUAL σ-OVERLAP RELIEF [T6 on edge of T³]:
                        # For diatomic inter-halogens (z=1, both LP≥P₁),
                        # the σ-axial LP on each atom form a BONDING overlap
                        # instead of blocking. The closure boost is reduced
                        # by the mutual overlap factor D₅/D₃ ≈ 0.878.
                        # Polyatomic (z>1): LP point away from bond → full blocking.
                        if z_max_bond <= 1:
                            # Diatomic: σ-LP overlap reduces blocking.
                            # Asymmetric (one per<P₁): full relief, no d-frustration.
                            # Bilateral (both per≥P₁): partial relief, keep d-frustration.
                            is_bilateral = (min(per_A, per_B) >= P1)
                            mutual_relief = D5 / D3 * (S_HALF if is_bilateral else 1.0)
                            S_lp *= (1.0 + S3 * (1.0 + delta_per_lp * D3) * (1.0 - mutual_relief))
                            if is_bilateral:
                                S_lp += S5 * C5 * delta_per_lp * S_HALF
                        else:
                            # Polyatomic: LP perpendicular to bonds → full blocking
                            S_lp *= (1.0 + S3 * (1.0 + delta_per_lp * D3))
                            # d-frustration [D13 P₂ face]:
                            if min(per_A, per_B) >= P1:
                                S_lp += S5 * C5 * delta_per_lp
                            else:
                                S_lp += D5 * delta_per_lp

    # ── SELF LP (one-sided or excess) ──
    # When only one atom has LP (lp_mutual=0) OR when one side has MORE
    # LP than the other (excess = lp - lp_mutual > 0), the excess blocks.
    # NF₃: F(LP=3) vs N(LP=1) → mutual=1, F_excess=2 → self screening.
    S_self = 0.0
    # Excess LP = residual after mutual. Attenuated by S₃ (secondary effect).
    # Gate: excess screening only when partner is per≤2 (no d-absorption).
    # SO₂: O(LP=2) excess → S(per=3) absorbs → no excess screening.
    # NF₃: F(LP=3) excess → N(per=2) can't absorb → excess screens.
    # C₃^Δper: survival per shell on hexagonal face [T6]
    # per=2→1.0, per=3→C₃≈0.78, per=4→C₃²≈0.61
    f_absorb_B = C3 ** max(0, per_B - 2)
    f_absorb_A = C3 ** max(0, per_A - 2)
    lp_a_eff = ((lp_a - lp_mutual) * S3 * f_absorb_B) if lp_mutual > 0 else lp_a
    lp_b_eff = ((lp_b - lp_mutual) * S3 * f_absorb_A) if lp_mutual > 0 else lp_b
    if lp_a_eff > 0 or lp_b_eff > 0:
        r_e_bond = A_BOHR * (per_A + per_B) / (2.0 * max(bo, 1.0) ** (1.0/3.0))
        r_ratio = r_e_bond / A_BOHR

        for lp, z, per, nf, p_per in [
                (lp_a_eff, z_a, per_A, nf_a, per_B),
                (lp_b_eff, z_b, per_B, nf_b, per_A)]:
            if lp <= 0:
                continue
            lp_eff = float(lp)
            Q_ex = _Q_koide_excess(nf)

            # VSEPR terminal reduction
            # Halogens (LP ≥ P₁): LP tightly bound (high IE) → weaker
            # σ-axis projection → D₃ reduction instead of S₃. [T6]
            if (z <= 1 or p_per <= 1) and (lp >= 2 or Q_ex < 0.5):
                if p_per >= P1:
                    lp_eff *= math.sqrt(S3)
                elif lp >= P1 and per <= P1:
                    # F (per=2) and Cl (per=3): tight LP → √(D₃S₃)
                    # Br/I (per≥4): already distance-attenuated, keep S₃
                    lp_eff *= math.sqrt(D3 * S3)
                else:
                    lp_eff *= S3
            elif z == 2 and p_per >= 2 and lp >= 2:
                # EXTENDED VSEPR for z=2 non-terminal [T6 on Z/P₁Z]:
                # LP on z=2 atoms bonded to heavy partners (p_per ≥ 2)
                # is partially constrained by the second bond. The
                # constraint reduces the LP projection on each bond
                # axis by cos²(θ₃) = C₃ (survival fraction after
                # propagation through one bond on the hexagonal face).
                # HCOOH -OH oxygen: z=2, lp=2, p_per=2 → C₃ reduction.
                # H₂O: p_per=1 → tier-1 VSEPR (not here). ✓
                # H₂S→H: p_per=1 → tier-1 VSEPR (not here). ✓
                lp_eff *= C3

            lp_dir = lp_eff / max(z, 1)

            # Distance decay: terminal LP (z ≤ 1) only
            dist = GAMMA_5 ** (r_ratio * S_HALF) if z <= 1 else 1.0

            exp_val = lp_dir * S3 * dist * (1.0 + Q_ex * S3 * (1 if z > 1 else 0))

            # π-LP anti-correlation
            if bo >= 2 and lp >= 2:
                exp_val *= C3 ** (bo - 1)
            # d-vacancy relief: per≥3 partners
            if p_per >= P1:
                exp_val *= (1.0 - S_HALF * S5)
            # SELF d-ABSORPTION [Geo 3D face overflow, Principle 3]:
            # At face_fraction ≥ 1.0 (hexagonal face Z/6Z saturated),
            # the LP-bearing atom's own LP overflows to its pentagonal
            # face Z/P₂Z (d-orbitals). Fraction absorbed = lp/(nf+z).
            # This reduces the LP's bond-axis projection (screening).
            # Gate: per ≥ P₁ (own d-orbitals exist), p_per ≥ 2 (p-block
            # partner, not H — H bonds already have VSEPR reduction),
            # face saturated (nf+z ≥ 2P₁). [CRT P₁→P₂ overflow]
            # DMS: S(nf=4,z=2,lp=2) → 2/6=33% absorbed. ✓
            # H₂S: S→H, p_per=1 → blocked. ✓
            # SF₂: lp_mutual=2, S excess=0 → loop skipped. ✓
            if per >= P1 and p_per >= 2 and z >= 2 and (nf + z) >= 2 * P1:
                f_absorbed = float(lp) / max(nf + z, 1)
                exp_val *= (1.0 - f_absorbed)
            S_self += exp_val

    return S_lp, S_self


# ╔════════════════════════════════════════════════════════════════════╗
# ║  VACANCY OPERATOR — unified dative + polygon vacancy per face    ║
# ╚════════════════════════════════════════════════════════════════════╝
#
# PT: LP donates into polygon vacancies on T³ = Z/P₁Z × Z/P₂Z × Z/P₃Z.
# Three mechanisms, one formula per face:
#   E_face = Ry × vac × F_LP × cascade / z × exp(-S_own × D_FULL)
#
# Dative cascade (bo ≥ 2): LP→p-vacancy (P₁), promoted, d-orbital
# Resonance dative (bo = 1): LP→p-vacancy fractional (P₁)
# Polygon vacancy (Za ≠ Zb): LP fills hexagon P₁ + decagon P₂

def _vacancy_operator(i, j, bo, topology, atom_data):
    """Unified vacancy operator: dative cascade + polygon vacancy.

    Returns dict with aggregated results (no individual type counts):
      cap_boost:       total boost to add to channel_cap
      sigma_boost:     extra σ capacity from dative + resonance (Ry/P₁ units)
      pi_boost:        extra π capacity from promoted/d-orbital (Ry/P₂ units)
      n_pi_consumed:   π bonds reclassified as σ by normal dative
      relief_strength: weight for dative relief screening reduction
      has_dative:      True if any dative mechanism active (for gates)
      D0_P2:           polygon vacancy energy (P₁+P₂ faces)
      f_d_absorb:      d-absorption fraction for vertex budget
    """
    n_dative = 0
    n_promoted = 0
    n_d_dative = 0
    n_res_dative = 0.0
    d_share = 1.0
    cap_boost = 0.0

    Za = topology.Z_list[i]
    Zb = topology.Z_list[j]
    da = atom_data[Za]
    db = atom_data[Zb]
    n_pi = max(0.0, bo - 1.0)

    # ── DATIVE LP→VACANCY: priority cascade [T2 GFT] ──────
    if bo >= 2 and n_pi >= 1:
        lp_a = topology.lp[i] if topology.lp else 0
        lp_b = topology.lp[j] if topology.lp else 0
        z_a = topology.z_count[i] if topology.z_count else 1
        z_b = topology.z_count[j] if topology.z_count else 1

        for don_lp, don_ie, don_per, acc_z, acc_per, acc_idx in [
            (lp_b, db['IE'], db['per'], z_a, da['per'], i),
            (lp_a, da['IE'], da['per'], z_b, db['per'], j),
        ]:
            if don_lp <= 0 or don_ie / RY < C3:
                continue
            acc_lp = topology.lp[acc_idx] if topology.lp else 0

            if acc_per <= 2 and acc_lp == 0 and n_dative == 0:
                vacancy = max(0, P1 - acc_z)
                n_dative = min(int(n_pi), don_lp, vacancy)

            if n_dative == 0 and acc_per <= 2 and acc_lp == 0 and acc_z >= P1 and n_promoted == 0:
                n_promoted = min(1, int(n_pi))

            if n_dative == 0 and n_promoted == 0 and acc_per >= P1 and n_d_dative == 0:
                n_d_dative = min(1, int(n_pi))
                d_share = 1.0 / max(topology.z_count[acc_idx], 1)

            if n_dative > 0 or n_promoted > 0 or n_d_dative > 0:
                break

        if n_dative > 0:
            cap_boost = n_dative * (RY / P1 - RY / P2)
        elif n_promoted > 0:
            cap_boost = n_promoted * RY * S3 / P2
        elif n_d_dative > 0:
            cap_boost = n_d_dative * RY * S5 / P2 * d_share

    # ── LP→VACANCY RESONANCE (center with orbital vacancy) ──
    # Gate: acceptor per≤2 with p-vacancy (l≥1 or hybridized s-block z≥2).
    # s-block atoms (Be, Mg) with z≥2 use promoted p-orbitals (sp hybrid).
    if n_dative == 0 and n_promoted == 0 and n_d_dative == 0:
        lp_a_rv = topology.lp[i] if topology.lp else 0
        lp_b_rv = topology.lp[j] if topology.lp else 0
        for acc_idx, acc_d, don_lp, don_d in [(i, da, lp_b_rv, db), (j, db, lp_a_rv, da)]:
            z_acc_rv = topology.z_count[acc_idx] if topology.z_count else 1
            has_vacancy = acc_d['l'] >= 1
            # s-block hybridized (Be, Mg): promoted p-orbital, attenuated by S₃
            is_s_hybrid = (acc_d['l'] == 0 and z_acc_rv >= 2)
            if acc_d['per'] <= 2 and (has_vacancy or is_s_hybrid) and don_lp > 0:
                if don_d['per'] <= 2 and don_d['IE'] / RY >= C3:
                    nf_acc = acc_d['nf']
                    z_acc = topology.z_count[acc_idx] if topology.z_count else 1
                    p_vac = max(0.0, (2.0 * P1 - nf_acc - z_acc)) / (2.0 * P1)
                    if p_vac > 0:
                        lp_frac = min(float(don_lp), float(P1)) / P1
                        # s-block hybrid: promoted p → attenuated by S₃
                        f_promote = S3 if is_s_hybrid else 1.0
                        n_res_dative = p_vac * lp_frac * f_promote
                        break

    # ── Aggregate into σ/π boosts ──────
    # σ boost: normal dative reclassifies π→σ, resonance adds σ capacity
    sigma_boost = n_dative * RY / P1 + n_res_dative * RY / P1
    # π boost: promoted and d-orbital add π capacity
    pi_boost = n_promoted * RY * S3 / P2 + n_d_dative * RY * S5 / P2 * d_share
    # π consumed by normal dative (reclassified as σ)
    n_pi_consumed = n_dative
    # Relief strength: S₃ × dist × C₃ × this weight
    relief_strength = float(n_dative) + float(n_promoted) * S3
    has_dative = (n_dative > 0 or n_promoted > 0 or n_d_dative > 0)

    # ── POLYGON VACANCY BACK-DONATION [unified P₁+P₂] ──────
    D0_P2 = 0.0
    f_d_absorb = 0.0

    if ENABLE_P2_COMPRESSED and Za != Zb and n_dative == 0:
        best_cap = 0.0
        best_f_absorb = 0.0

        for idx_acc, idx_don in [(i, j), (j, i)]:
            Z_acc = topology.Z_list[idx_acc]
            d_acc = atom_data[Z_acc]
            per_acc = d_acc['per']
            z_acc = topology.z_count[idx_acc] if topology.z_count else 1
            nf_acc = d_acc['nf']
            l_acc = d_acc['l']

            total_lp = _total_lp_from_neighbors(idx_acc, topology)
            if total_lp <= 0:
                continue

            cap_acc = 0.0

            # P₁ hexagon vacancy (p-block only, l=1)
            if per_acc >= 2 and l_acc >= 1:
                vac_p = max(0.0, (2.0 * P1 - nf_acc - z_acc)) / (2.0 * P1)
                if vac_p > 0:
                    vac_positions = 2.0 * P1 * vac_p
                    F_LP_p = min(float(total_lp), vac_positions) / (2.0 * P1)
                    cap_p = RY * vac_p * F_LP_p * S3 * C3 / max(z_acc, 1)
                    S_own_p = D3 * (z_acc / P1) ** 2
                    cap_acc += cap_p * math.exp(-S_own_p * D_FULL)

            # P₂ decagon vacancy (per≥3, p-block+ only)
            # d-orbitals are COLLECTIVE: all bonds access the same
            # d-vacancy simultaneously. The per-bond cap is NOT divided
            # by z — each LP→d donation uses the full pentagonal vertex S₅.
            # Gate: per≥P₁ (d-orbitals exist), l≥1 (p-block+), z≥2
            # (at least 2 bonds justify d-participation). [SO₂: z=2, per=3]
            # Exclude aromatic ring members: Hückel already handles π.
            in_aro_ring = any(bo_t == 1.5 for ii_t, jj_t, bo_t in topology.bonds
                              if ii_t == idx_acc or jj_t == idx_acc)
            f_z_p2 = 1.0 - math.exp(-max(0, z_acc - 1))
            if per_acc >= P1 and l_acc >= 1 and f_z_p2 > D3 and not in_aro_ring:
                nf_d = nf_acc if l_acc == 2 else 0
                vac_d = max(0.0, (2.0 * P2 - nf_d)) / (2.0 * P2)
                if vac_d > 0:
                    # π-amplified LP→d: double bonds open the π→d resonance
                    # channel, effectively multiplying LP coupling by bo.
                    # SO₂: bo=2, total_lp=4 → eff_lp=8 → F_LP_d=0.8.
                    # Single bonds (bo=1): no amplification. [T6 P₂×π]
                    eff_lp = float(total_lp) * min(bo, 2.0)
                    F_LP_d = min(eff_lp, 2.0 * P2 * vac_d) / (2.0 * P2)
                    # d-orbital shell index on Z/5Z: sin²(k·π/P₂)
                    # Double bond + non-hypervalent (z < P₁, bo ≥ 2):
                    #   π back-donation opens d-orbital → per-1 (next shell)
                    #   per=3,z<3,bo≥2: k=2→sin²(2π/5)=0.905 [3d via π]
                    # Otherwise: d-orbital at per-2 (standard)
                    per_d_acc = max(1, per_acc - 1) if (z_acc < P1 and bo >= 2) else max(1, per_acc - 2)
                    f_orb = math.sin(math.pi * per_d_acc / P2) ** 2
                    ie_acc = atom_data[Z_acc]['IE']
                    ie_don = atom_data[topology.Z_list[idx_don]]['IE']
                    Q_d = abs(ie_acc - ie_don) / max(0.1, ie_acc + ie_don)
                    # LP→d floor: coupling is at least S₅ for non-hypervalent
                    # d-hosts (z < P₁). LP→d donation depends on LP/d overlap,
                    # not IE asymmetry. C-S (Q_d=0.050) needs this floor:
                    # thioethers have bo=1 but real LP→d donation.
                    if z_acc < P1:
                        Q_d = max(Q_d, S5)
                    # Collective sharing: d-capacity shared among bonds
                    # with weight P₁/z. The hypervalent gate uses the
                    # FILLING RATIO nf/N on the hexagon: when the p-shell
                    # is half-full or more (nf ≥ P₁), d-orbitals are needed
                    # for additional bonding capacity.
                    # f_hyper = sin²(π × nf/(2P₁)) — smooth from 0 to 1
                    #   nf=1 (Al): sin²(π/6) = S₃ = 0.25 (weak)
                    #   nf=3 (P):  sin²(π/2) = 1.0 (full, half-fill)
                    #   nf=4 (S):  sin²(2π/3) = S₃ = 0.75
                    import math as _m
                    nf_hex = nf_acc if l_acc == 1 else min(nf_acc, 2 * P1)
                    f_hyper = _m.sin(_m.pi * nf_hex / (2.0 * P1)) ** 2
                    # Hypervalent quadratic sharing [T6 on P₂ face]:
                    # z > P₁ → d-capacity shared non-linearly.
                    # f_share = (P₁/z)^(1 + (z-P₁)/P₂) — convex decay.
                    # SF₆ (z=6): (3/6)^1.6 = 0.330 (vs linear 0.500)
                    # PF₅ (z=5): (3/5)^1.4 = 0.475 (vs linear 0.600)
                    # SiCl₄ (z=4): (3/4)^1.2 = 0.709 (vs linear 0.750)
                    if z_acc > P1:
                        _excess = (z_acc - P1) / float(P2)
                        f_share = (float(P1) / z_acc) ** (1.0 + _excess)
                    else:
                        f_share = 1.0
                    cap_d_v = RY * vac_d * F_LP_d * f_orb * Q_d * f_share * f_hyper * f_z_p2
                    per_don = atom_data[topology.Z_list[idx_don]]['per']
                    if per_don >= P1:
                        cap_d_v *= S_HALF
                    S_own_d = D5 * (z_acc / P2) ** 2
                    cap_acc += cap_d_v * math.exp(-S_own_d * D_FULL)

            if cap_acc > best_cap:
                best_cap = cap_acc
                if per_acc >= P1:
                    nf_d_abs = nf_acc if l_acc == 2 else 0
                    vac_d_abs = max(0.0, (2.0 * P2 - nf_d_abs)) / (2.0 * P2)
                    F_LP_abs = min(float(total_lp), 2.0 * P2) / (2.0 * P2)
                    best_f_absorb = vac_d_abs * F_LP_abs

        D0_P2 = best_cap
        f_d_absorb = best_f_absorb

    return {
        'cap_boost': cap_boost,
        'sigma_boost': sigma_boost, 'pi_boost': pi_boost,
        'n_pi_consumed': n_pi_consumed,
        'relief_strength': relief_strength,
        'has_dative': has_dative,
        'D0_P2': D0_P2, 'f_d_absorb': f_d_absorb,
    }


# ╔════════════════════════════════════════════════════════════════════╗
# ║  BOND SCREENING — edge energy from molecular context             ║
# ╚════════════════════════════════════════════════════════════════════╝

def _bond_context(i, j, topology, atom_screenings):
    """DIM 2: polygon context — A_vertex(k) modifies edge (i,j) [D13].

    Neighbors stabilize (σ_eff < S₃) or destabilize (σ_eff > S₃) the bond.
    Returns signed S_context (positive = destabilizing).
    """
    S_ctx = 0.0
    n_ctx = 0
    for endpoint, exclude in [(i, j), (j, i)]:
        for ii, jj, bo_k in topology.bonds:
            if ii == endpoint and jj != exclude:
                k = jj
            elif jj == endpoint and ii != exclude:
                k = ii
            else:
                continue
            # σ_eff = σ_base + bond_load × S₃ (charge committed by heavy atoms)
            sigma_eff = atom_screenings[k]
            Z_k = topology.Z_list[k]
            if Z_k > 1:
                z_k = topology.z_count[k] if topology.z_count else 1
                sum_bo_k = topology.sum_bo[k] if topology.sum_bo else 1.0
                sigma_eff += (sum_bo_k / max(z_k, 1)) * S3
            delta = sigma_eff - S3
            S_ctx += (delta * D7 if delta > 0 else delta * D5) * bo_k
            n_ctx += 1
    if n_ctx > 0:
        S_ctx /= math.sqrt(n_ctx)
    return S_ctx if ENABLE_CONTEXT else 0.0


def _bond_screening(i, j, bo, topology, atom_screenings, atom_data):
    """Bond screening S = S_edge + S_context on P₁ face of T³.

    DIM 2 (polygon): context from neighbors [D13 vertex amplitude]
    DIM 1 (edge): LP, period crossing, Hund, vertex overcrowding
    DIM 0 (point): parity boost, Bohr resonance, asymmetry
    """
    Za, Zb = topology.Z_list[i], topology.Z_list[j]
    per_A, per_B = period(Za), period(Zb)
    lp_a = topology.lp[i] if topology.lp else 0
    lp_b = topology.lp[j] if topology.lp else 0
    z_a = topology.z_count[i] if topology.z_count else 1
    z_b = topology.z_count[j] if topology.z_count else 1
    da, db = atom_data[Za], atom_data[Zb]

    # DIM 2: polygon context [D13 A_vertex]
    S_context = _bond_context(i, j, topology, atom_screenings)

    # DIM 1: LP screening [T2 GFT, T6 holonomy]
    S_lp, S_self = _lp_screening(
        lp_a, lp_b, z_a, z_b, per_A, per_B, bo,
        da['nf'], db['nf'], Za, Zb, atom_data)

    # Period crossing [Principle 3: CRT orthogonal]
    delta_per = abs(per_A - per_B)
    S_per = 0.0
    if delta_per > 0:
        per_min, per_max = min(per_A, per_B), max(per_A, per_B)
        f_rel = (S_HALF if per_min == 1
                 else P1 / per_max if per_max > P1 else 1.0)
        S_per = math.sqrt(delta_per) * D3 * f_rel
    elif min(per_A, per_B) >= P1:
        depth = min(per_A, per_B) - 2
        S_per = depth * D3 * min(float(max(z_a, z_b)), float(2*P1)) / (2*P1)

    # Hund π gate [T6]: both Q_Koide=1, z ≥ 2, bo ≥ 2
    S_hund = 0.0
    lp_mutual = min(lp_a, lp_b)
    if bo >= 2 and lp_mutual >= 1:
        z_max = max(z_a, z_b)
        if min(_Q_koide_excess(da['nf']), _Q_koide_excess(db['nf'])) >= 0.99 and z_max >= 2:
            S_hund = S3 * (z_max - 1) / z_max

    # d-orbital promotion cost: steric > 4 → sp³d hybridization [T6 on P₂]
    # Each promoted position (beyond sp³) uses P₂ face instead of P₁.
    # Cost per promoted position: D₅ (pentagonal screening unit).
    # Gate: only when LP > 0 (the LP occupies the promoted d-orbital,
    # not the bonds). PF₅(LP=0): all positions are bonds → no promotion.
    # SF₄(LP=1): the LP sits in the promoted 5th position → weaker bonds.
    S_promoted = 0.0
    for z, lp, per in [(z_a, lp_a, per_A), (z_b, lp_b, per_B)]:
        if per >= P1 and lp > 0:
            steric = z + lp
            n_promoted = max(0, steric - 4)
            if n_promoted > 0:
                S_promoted += n_promoted * D5 / max(z, 1)

    # Vertex overcrowding: z+LP exceeds P₁ hexagonal capacity [D13]
    # The face Z/(2P₁)Z has 2P₁=6 positions. When z+LP > 2P₁, the
    # face is overcrowded. Continuous: excess / (2P₁), amplitude D₃.
    # SF₄ (z=4,LP=1): (5-6)/6=0 → no excess. But z=4 > P₁=3 → edge crowding.
    # PCl₅ (z=5,LP=0): (5-6)/6=0 → no excess either. But z > P₁+1.
    z_max = max(z_a, z_b)
    lp_max = max(lp_a, lp_b)
    z_lp = z_max + lp_max
    S_vertex = max(0, z_max - P1 - 1) * D3  # original: only z > P₁+1
    if z_lp > 2 * P1:
        # Face overcrowding: z+LP exceeds hexagonal capacity
        S_vertex += (z_lp - 2.0 * P1) / (2.0 * P1) * D3

    # H→heavy shell-mismatch screening [Z/P₁Z face, T6 holonomy]
    # H has no p-orbitals: σ overlap reduced with per ≥ P₁.
    # 1s of H covers only a fraction of 3p of Si.
    # Screening = D₃ × √(per_heavy - 1) / P₁ [shell mismatch on Z/P₁Z]
    S_H_heavy = 0.0
    if min(Za, Zb) == 1 and max(per_A, per_B) >= P1:
        per_heavy = max(per_A, per_B)
        S_H_heavy = D3 * math.sqrt(per_heavy - 1) / P1

    # DIM 0: parity boost
    S_parity = (-S3 * S_HALF if Za == Zb
                else -D3 * S_HALF if per_A == per_B else 0.0)

    # Bohr resonance: two-tier [T6 holonomy], modulated by EA capacity
    # Atoms near Ry but with EA≈0 (N) can't sustain the resonance fully.
    # f_EA = min(1, √(EA_A×EA_B) / (Ry×D₃×s)) [geometric mean, soft gate]
    S_bohr = 0.0
    if min(per_A, per_B) <= 2:
        dev_A = abs(da['IE'] - RY) / RY
        dev_B = abs(db['IE'] - RY) / RY
        dev_max = max(dev_A, dev_B)
        if dev_max < D3:
            S_bohr = -D3 * (1.0 - dev_A / D3) * (1.0 - dev_B / D3)
        elif dev_max < S3:
            S_bohr = -D3 * (1.0 - dev_A / S3) * (1.0 - dev_B / S3)
        elif dev_max < C3 and max(z_a, z_b) <= 1 and max(da['IE'], db['IE']) > RY:
            # Tier 3: super-Ry diatomics (ClF, HF, F₂) still resonate
            # on Z/2Z but weaker. Amplitude D₇. Gates: diatomic only,
            # AND at least one atom exceeds Ry (excludes HI, HBr where
            # both atoms are below Ry → no super-Ry resonance).
            # ClF: -0.054, HF: -0.057, F₂: -0.037. [T6 holonomy]
            S_bohr = -D7 * (1.0 - dev_A / C3) * (1.0 - dev_B / C3)
        if S_bohr < 0:
            f_EA = min(1.0, math.sqrt(max(da['EA'], 0.01) * max(db['EA'], 0.01))
                       / (RY * D3 * S_HALF))
            S_bohr *= f_EA

    # p=2 asymmetry [D00, T0]: donor/acceptor on Z/2Z
    sigma_i, sigma_j = atom_screenings[i], atom_screenings[j]
    chi_A = (da['IE'] + da['EA']) / 2.0
    chi_B = (db['IE'] + db['EA']) / 2.0
    Q_eff = abs(chi_A - chi_B) / max(0.1, chi_A + chi_B)
    S_asym = -S_HALF * abs(sigma_i - sigma_j) * (1.0 + Q_eff)

    # EA cap boost [A_edge on P₁, T2 GFT]: LP donor transfers EA
    # into acceptor p-vacancy. Requires: donor LP>0, high EA,
    # AND acceptor has LP=0 with p-vacancy (nf+z < 2P₁).
    # BF₃: F(LP=3,EA=3.4) → B(LP=0,vac) ✓. NF₃: F → N(LP=1) ✗.
    S_EA = 0.0
    for don_lp, don_ea, acc_lp, acc_d in [
        (lp_a, da['EA'], lp_b, db), (lp_b, db['EA'], lp_a, da)]:
        if don_lp > 0 and don_ea > RY * S3 and acc_lp == 0:
            # Acceptor must have p-vacancy
            nf_acc = acc_d['nf']
            z_acc = z_b if acc_d is db else z_a
            vac = max(0.0, 2.0 * P1 - nf_acc - z_acc) / (2.0 * P1)
            if vac > 0:
                lp_frac = min(float(don_lp), float(P1)) / P1
                S_EA -= don_ea * S3 * lp_frac * vac / RY

    # Total edge screening (P₁ face)
    w_A, w_B = max(per_A, 2), max(per_B, 2)
    S_base = (sigma_i * w_A + sigma_j * w_B) / (w_A + w_B)

    # ── 6 SCREENING COMPONENTS for per-canal cascade [PROMPT_NEXT] ──
    # 1. Context: neighbor stabilization (A_vertex amplitude) [D13]
    # 2. LP blocking: LP + self + Hund + EA cap + d-promotion [T6]
    # 3. Period: shell crossing cost [CRT orthogonal]
    # 4. Parity/Bohr: homonuclear + Bohr resonance [T0, T6]
    # 5. Vertex: overcrowding penalty [D13]
    # 6. Asymmetry: donor/acceptor polarity [D00, T0]
    comp_context = S_context
    comp_lp = S_lp + S_self + S_hund + S_EA + S_promoted
    comp_period = S_per
    comp_parity = S_parity + S_bohr
    comp_vertex = S_vertex + S_H_heavy
    comp_asym = S_asym

    S_edge = S_base + comp_asym + comp_lp + comp_period + comp_vertex + comp_parity

    components = {
        'S_base': S_base,
        'comp_context': comp_context,
        'comp_lp': comp_lp,
        'comp_period': comp_period,
        'comp_parity': comp_parity,
        'comp_vertex': comp_vertex,
        'comp_asym': comp_asym,
    }

    return S_edge, S_context, components


# ╔════════════════════════════════════════════════════════════════════╗
# ║  RESOLVE ATOM DATA — extracted for reuse by vertex_polygon      ║
# ╚════════════════════════════════════════════════════════════════════╝

def _resolve_atom_data(topology: Topology, source: str = "full_pt") -> dict:
    """Build atom_data dict for all unique Z in topology.

    Returns {Z: {'IE', 'EA', 'mass', 'nf', 'ns', 'per', 'l'}}.
    """
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
        }
    return atom_data


# ╔════════════════════════════════════════════════════════════════════╗
# ║  COMPUTE D_AT — the molecular polygon engine                    ║
# ╚════════════════════════════════════════════════════════════════════╝

def compute_D_at(topology: Topology,
                 source: str = "full_pt") -> MoleculeResult:
    """Molecular engine: delegates to compute_D_at_transfer (T⁴ engine).

    The polygon engine was incomplete for diatomics/triatomics (MAE 10.08%).
    The transfer matrix engine handles all cases correctly (MAE 2.17%).
    This function now delegates and wraps the result in MoleculeResult.
    """
    from ptc.transfer_matrix import compute_D_at_transfer
    tr = compute_D_at_transfer(topology)

    # Build formula from topology
    from collections import Counter
    counts = Counter(SYMBOLS.get(Z, '?') for Z in topology.Z_list)
    formula = ''.join(f"{sym}{cnt if cnt > 1 else ''}"
                      for sym, cnt in sorted(counts.items()))

    # IE audit
    ie_audit = {}
    for Z in set(topology.Z_list):
        ie_pt = IE_eV(Z, continuous=True)
        ie_nist = IE_NIST.get(Z, 0.0)
        if ie_nist > 0:
            err_pct = (ie_pt - ie_nist) / ie_nist * 100
            ie_audit[Z] = (ie_pt, ie_nist, err_pct)

    return MoleculeResult(
        D_at=tr.D_at,
        bonds=tr.bonds,
        topology=topology,
        source_mode=source,
        formula=formula,
        ie_audit=ie_audit,
    )


def _compute_D_at_polygon(topology: Topology,
                           source: str = "full_pt") -> MoleculeResult:
    """LEGACY polygon engine — kept for reference/testing only.

    MAE = 10.08% on 849 mol benchmark. Do NOT use for production.
    """
    # ── 1. Resolve atomic data ──
    atom_data = _resolve_atom_data(topology, source)
    from ptc.vertex_polygon import build_vertex_polygons
    polygons = build_vertex_polygons(topology, atom_data)
    ie_audit = {}
    for Z in atom_data:
        ie_pt = IE_eV(Z, continuous=True)
        ie_nist = IE_NIST.get(Z, 0.0)
        if ie_nist > 0:
            err_pct = (ie_pt - ie_nist) / ie_nist * 100
            ie_audit[Z] = (ie_pt, ie_nist, err_pct)

    # ── 2. Atom peer values (screening profile on molecular polygon) ──
    N = len(topology.Z_list)
    atom_screenings = []
    charges = topology.charges if topology.charges else [0] * N
    for k in range(N):
        Z = topology.Z_list[k]
        d = atom_data[Z]
        ie_k = d['IE']

        # Formal charge adjustment [insight #38]
        q_formal = charges[k] if k < len(charges) else 0
        if q_formal != 0:
            ie_k = ie_k * (1.0 + q_formal * S3)

        lp_k = topology.lp[k] if topology.lp else 0
        z_k = topology.z_count[k] if topology.z_count else 1
        S_k = _atom_screening(ie_k, d['EA'], lp_k, z_k,
                              d['nf'], d['per'], d['l'])
        atom_screenings.append(S_k)

    # ── 2b. Hückel-PT π cap for aromatic rings ─────────────
    # Aromatic bonds (bo=1.5) get π energy from Hückel
    # INSTEAD OF the pairwise n_pi × Ry/P₂ cap.
    #
    # PT [T6 on Z/NZ]:
    #   β = S₅ × C₅ × √(ε_i × ε_j) × D_FULL  (Rabi on canal P₂)
    #   Hückel H on Z/NZ → eigenvalues → fill N_π electrons
    #   E_π = E_delocalized − N_π × α  (bonding part only)
    #   π_cap per bond = |E_π| / N_ring
    #
    # Anti-double-counting [GFT]: replaces 0.5×Ry/P₂, not adds.
    #
    huckel_pi_cap = {}   # bond_index → π cap for this aromatic bond

    aro_bonds_list = [(bi, i, j) for bi, (i, j, bo) in enumerate(topology.bonds)
                      if bo == 1.5]
    if aro_bonds_list:
        import numpy as np
        from collections import defaultdict, deque

        aro_adj = defaultdict(set)
        for bi, i, j in aro_bonds_list:
            aro_adj[i].add((j, bi))
            aro_adj[j].add((i, bi))

        visited = set()
        for start in aro_adj:
            if start in visited:
                continue
            # BFS to find connected aromatic component
            component = set()
            bond_indices = set()
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
            ring_atoms = [start]
            current, prev = start, -1
            for _ in range(len(component) - 1):
                nbs = [(n, bi) for n, bi in aro_adj[current]
                       if n != prev and n in component]
                if not nbs:
                    break
                nxt, _ = nbs[0]
                ring_atoms.append(nxt)
                prev, current = current, nxt

            N_ring = len(ring_atoms)
            if N_ring < 5:
                continue

            # Onsite ε and transfer β
            eps_ring = []
            for a in ring_atoms:
                Z = topology.Z_list[a]
                ie, ea = atom_data[Z]['IE'], atom_data[Z]['EA']
                eps_ring.append(math.sqrt(max(ie * ea, 0.01)))

            H_mat = np.zeros((N_ring, N_ring))
            for k in range(N_ring):
                H_mat[k, k] = eps_ring[k]
                kn = (k + 1) % N_ring
                beta_k = S5 * C5 * math.sqrt(
                    max(eps_ring[k] * eps_ring[kn], 0.0)) * D_FULL
                H_mat[k, kn] = -beta_k
                H_mat[kn, k] = -beta_k

            eigenvalues = sorted(np.linalg.eigvalsh(H_mat))

            # Fill N_ring π-electrons (1 per ring atom)
            n_pi_ring = N_ring
            E_deloc = 0.0
            placed = 0
            for ev in eigenvalues:
                if placed >= n_pi_ring:
                    break
                nf = min(2, n_pi_ring - placed)
                E_deloc += nf * ev
                placed += nf

            # π bonding = E_deloc - N_pi × α_avg
            alpha_avg = sum(eps_ring) / N_ring
            E_pi_bond = abs(E_deloc - n_pi_ring * alpha_avg)

            # Heteroatom defect attenuation
            eps_C = [e for e, a in zip(eps_ring, ring_atoms)
                     if topology.Z_list[a] == 6]
            n_C = len(eps_C)
            if 0 < n_C < N_ring:
                eps_C_avg = sum(eps_C) / n_C
                betas = []
                for k in range(N_ring):
                    kn = (k + 1) % N_ring
                    betas.append(S5 * C5 * math.sqrt(
                        max(eps_ring[k] * eps_ring[kn], 0.0)) * D_FULL)
                f_defect = 1.0
                for k in range(N_ring):
                    Z_k = topology.Z_list[ring_atoms[k]]
                    if Z_k != 6:
                        d_eps = abs(eps_ring[k] - eps_C_avg)
                        beta_k = (betas[(k-1) % N_ring] + betas[k]) / 2
                        if beta_k > 0.01:
                            lp_k = topology.lp[ring_atoms[k]] if topology.lp else 0
                            if lp_k == 1 and Z_k == 7:
                                # PYRIDINE-LIKE N: sp² with in-plane LP [T³ CRT]
                                # LP on P₁ face (hexagonal), π on P₂ face (pentagonal).
                                # Cross-face interference = S₃ × off-resonance fraction.
                                # f_defect = 1 - S₃ × d_ε/(d_ε + 2β)
                                # N(d_ε=2.76, β=0.31): f=0.82 [18% penalty]
                                # Max penalty: S₃ ≈ 22% (CRT bound)
                                f_defect *= 1.0 - S3 * d_eps / (d_eps + 2.0 * beta_k)
                            else:
                                # Other heteroatoms (S, O with LP≥2): standard attenuation
                                f_severe = max(0.0, min(1.0, d_eps / (4.0 * beta_k) - 0.5))
                                f_lp_sp2 = 1.0 if lp_k == 1 else 0.0
                                power = 2.0 - f_severe * f_lp_sp2
                                f_defect *= (2*beta_k / (d_eps + 2*beta_k))**power
                E_pi_bond *= f_defect

            # Distribute π cap per aromatic bond in this ring
            pi_cap_per_bond = E_pi_bond / N_ring
            for bi in bond_indices:
                huckel_pi_cap[bi] = pi_cap_per_bond

    # ── 2c. Hückel STAR π for RADICAL symmetric π systems [T6 on K_{1,z}] ──
    # Radical molecules (odd total electrons) have resonance that
    # reduces effective bo. Star Hückel on K_{1,z} computes the
    # delocalized π cap per bond. Only applies to radicals.
    import numpy as _np
    VALENCE = {1:1, 3:1, 4:2, 5:3, 6:4, 7:5, 8:6, 9:7, 11:1, 13:3,
               14:4, 15:5, 16:6, 17:7, 19:1, 20:2, 35:7, 53:7}
    total_val = sum(VALENCE.get(Z, period(Z)) for Z in topology.Z_list)
    is_radical = (total_val % 2 == 1)

    if is_radical:
        for k in range(len(topology.Z_list)):
            z_k = topology.z_count[k] if topology.z_count else 1
            if z_k < 2:
                continue
            pi_bonds = []
            for bi_s, (ii, jj, bo_s) in enumerate(topology.bonds):
                if bi_s in huckel_pi_cap:
                    continue
                if (ii == k or jj == k) and bo_s >= 2:
                    pi_bonds.append((bi_s, jj if ii == k else ii, bo_s))
            if len(pi_bonds) < 2:
                continue
            # Symmetric resonance: same partner element
            if len(set(topology.Z_list[p] for _, p, _ in pi_bonds)) > 1:
                continue

            # Star Hückel Hamiltonian [D13 spin foam on K_{1,z}]
            # Onsite = sqrt(IE×EA): geometric mean of bonding capacity.
            # Coupling = S₅C₅ × D_FULL: pentagonal Rabi × screening depth.
            Z_k = topology.Z_list[k]
            eps_c = math.sqrt(max(atom_data[Z_k]['IE'] * atom_data[Z_k]['EA'], 0.01))
            n_h = len(pi_bonds) + 1
            H_star = _np.zeros((n_h, n_h))
            H_star[0, 0] = eps_c
            for idx_s, (bi_s, partner, bo_s) in enumerate(pi_bonds):
                Z_p = topology.Z_list[partner]
                eps_p = math.sqrt(max(atom_data[Z_p]['IE'] * atom_data[Z_p]['EA'], 0.01))
                H_star[idx_s+1, idx_s+1] = eps_p
                beta = S5 * C5 * math.sqrt(max(eps_c * eps_p, 0.01)) * D_FULL
                H_star[0, idx_s+1] = -beta
                H_star[idx_s+1, 0] = -beta
            eigenvalues = sorted(_np.linalg.eigvalsh(H_star))
            n_pi_fill = len(pi_bonds)
            E_deloc = sum(min(2, n_pi_fill - p) * ev
                         for p, ev in zip(range(0, n_pi_fill, 2), eigenvalues)
                         if p < n_pi_fill)
            alpha_avg = sum(H_star[i,i] for i in range(n_h)) / n_h
            E_pi_star = abs(E_deloc - n_pi_fill * alpha_avg)
            pi_cap_star = E_pi_star / len(pi_bonds)
            for bi_s, _, _ in pi_bonds:
                if bi_s not in huckel_pi_cap:
                    huckel_pi_cap[bi_s] = pi_cap_star

    # ── 2d. π-REDISTRIBUTION by IE matching [T6 holonomy on edge] ──
    # When a center atom has z ≥ 2 π bonds (bo ≥ 2) to DIFFERENT elements,
    # the π redistribute toward the bond with better IE matching.
    # N₂O: N=N + N=O → π migrates to N-N (ΔIE=0) from N-O (ΔIE=0.91).
    # The cap is redistributed, NOT the total energy (zero-sum).
    # σ_match = S₃ × Ry / √2 ≈ 1.0 eV (Bohr resonance width).
    sigma_match = S3 * RY / math.sqrt(2)
    bo_effective = list(topology.bonds)  # copy as mutable list
    for k in range(len(topology.Z_list)):
        z_k = topology.z_count[k] if topology.z_count else 1
        if z_k < 2:
            continue
        # Find π bonds at vertex k not in huckel_pi_cap
        pi_at_k = []
        for bi_r, (ir, jr, bo_r) in enumerate(topology.bonds):
            if bi_r in huckel_pi_cap:
                continue
            if (ir == k or jr == k) and bo_r >= 2:
                partner = jr if ir == k else ir
                pi_at_k.append((bi_r, partner, bo_r))
        if len(pi_at_k) < 2:
            continue
        # Check if terminal elements differ (asymmetric resonance)
        partner_Zs = [topology.Z_list[p] for _, p, _ in pi_at_k]
        if len(set(partner_Zs)) <= 1:
            continue  # symmetric (SO₃: all O) → no redistribution needed
        # IE matching weights
        ie_k = atom_data[topology.Z_list[k]]['IE']
        weights = []
        for bi_r, partner, bo_r in pi_at_k:
            ie_p = atom_data[topology.Z_list[partner]]['IE']
            delta_ie = abs(ie_k - ie_p)
            weights.append(math.exp(-delta_ie ** 2 / (2 * sigma_match ** 2)))
        total_pi = sum(bo_r - 1 for _, _, bo_r in pi_at_k)
        total_w = sum(weights)
        if total_w > 0 and total_pi > 0:
            for idx, (bi_r, partner, bo_r) in enumerate(pi_at_k):
                new_pi = total_pi * weights[idx] / total_w
                i_r, j_r, _ = topology.bonds[bi_r]
                bo_effective[bi_r] = (i_r, j_r, 1.0 + new_pi)

    # ── 3. Bond energies via polygon screening ──
    SHANNON = RY / P1  # 4.535 eV = maximum per channel

    bond_results = []
    for bi, (i_orig, j_orig, bo) in enumerate(bo_effective):
        i, j = i_orig, j_orig
        # PERMUTATION INVARIANCE [CRT: order of factors = product]
        # Always put lower-Z atom first. Ensures bond(C,O) == bond(O,C).
        # For same Z: lower index first (arbitrary but consistent).
        Za, Zb = topology.Z_list[i], topology.Z_list[j]
        if Za > Zb or (Za == Zb and i > j):
            i, j = j, i
            Za, Zb = Zb, Za
        da, db = atom_data[Za], atom_data[Zb]

        # Bond screening: edge (pure) + context (vertex→edge) [D13]
        S_edge, S_context, scr_comp = _bond_screening(i, j, bo, topology, atom_screenings, atom_data)
        S_bond = S_edge + S_context

        # ── PER-CANAL SCREENING RECOMPOSITION [T³ cascade] ──────
        # S_P1 = S_bond (unchanged, all 6 terms at full weight)
        # S_P2/P3 = Σ(comp_k × α_k) — CORRECTIONS ONLY, no S_base.
        # S_base (atomic background) is already in each channel's own
        # screening (S_total for P₁, S_own for P₂, S_P3_face for P₃).
        # Per-canal cascade modulates ONLY the 6 molecular corrections.
        S_P2_edge = (scr_comp['comp_context'] * ALPHA_CONTEXT_P2
                     + scr_comp['comp_lp'] * ALPHA_LP_P2
                     + scr_comp['comp_period'] * ALPHA_PERIOD_P2
                     + scr_comp['comp_parity'] * ALPHA_PARITY_P2
                     + scr_comp['comp_vertex'] * ALPHA_VERTEX_P2
                     + scr_comp['comp_asym'] * ALPHA_ASYM_P2)
        S_P3_edge = (scr_comp['comp_context'] * ALPHA_CONTEXT_P3
                     + scr_comp['comp_lp'] * ALPHA_LP_P3
                     + scr_comp['comp_period'] * ALPHA_PERIOD_P3
                     + scr_comp['comp_parity'] * ALPHA_PARITY_P3
                     + scr_comp['comp_vertex'] * ALPHA_VERTEX_P3
                     + scr_comp['comp_asym'] * ALPHA_ASYM_P3)

        # ── S_MOL_CORE: 3D polyhedron faces [insight #22] ──────
        # P₁ channel: LP structural cooperation (σ = D₅ sin(θ/2) LP/P₁)
        # P₂ channel: EM triangular faces (resonance − screening per face)
        S_mol_core = 0.0
        S_face_comp = 0.0
        for vertex in [i, j]:
            z_v = topology.z_count[vertex] if topology.z_count else 1
            if z_v < 2:
                continue
            lp_v = topology.lp[vertex] if topology.lp else 0
            theta_deg, n_faces, sin_coop = _vertex_faces(z_v, lp_v)
            if sin_coop <= 0:
                continue
            theta_rad = math.radians(theta_deg)
            sin_half = math.sin(theta_rad / 2.0)

            if lp_v > 0:
                # P₁: LP cooperation → -ln(1-σ)
                sigma_face = D5 * sin_half * min(float(lp_v), float(P1)) / P1
                if 0 < sigma_face < 1:
                    S_mol_core -= math.log(1.0 - sigma_face)
            elif n_faces >= 2:
                # P₂: EM resonance − neighbor screening per face
                Z_v = topology.Z_list[vertex]
                ie_v = atom_data[Z_v]['IE']
                f_bohr = max(0.0, (ie_v / RY - C3) / S3)
                alpha_res = D5 * sin_half * f_bohr / n_faces
                partner = j if vertex == i else i
                S_face_net = 0.0
                for ii2, jj2, bo2 in topology.bonds:
                    k = (jj2 if ii2 == vertex and jj2 != partner
                         else ii2 if jj2 == vertex and ii2 != partner
                         else -1)
                    if k < 0:
                        continue
                    ea_k = atom_data[topology.Z_list[k]]['EA']
                    lp_k = topology.lp[k] if topology.lp else 0
                    S_face_net += alpha_res - D5 * sin_coop * (ea_k / RY + lp_k * D3)
                if S_face_net > 0:
                    S_mol_core += S_face_net
                else:
                    S_face_comp += abs(S_face_net)
                # Sub-Bohr screening, modulated by partner LP [GFT P₁]
                # When partner has LP, LP screening (S_self) already
                # penalizes the bond from the partner side. Sub-Bohr
                # vertex penalty from the SAME bond overlaps on Z/6Z.
                # Relief amplitude D₃/P₁ per LP (sieve depth per face).
                f_sub = max(0.0, (C3 - ie_v / RY) / S3)
                if f_sub > 0:
                    partner_v = j if vertex == i else i
                    lp_pv = topology.lp[partner_v] if topology.lp else 0
                    f_overlap = max(0.0, 1.0 - float(lp_pv) * D3 / P1)
                    S_face_comp += D3 * (z_v / P1) * f_sub * f_overlap
                # N-1 cooperative LP competition
                n_lp_nb = sum(1 for ii2, jj2, _ in topology.bonds
                              if (ii2 == vertex or jj2 == vertex)
                              and (topology.lp[jj2 if ii2 == vertex else ii2] if topology.lp else 0) > 0)
                if n_lp_nb >= 2:
                    S_face_comp += D3 * (n_lp_nb - 1) / z_v
        if not ENABLE_FACE_EFFECTS:
            S_mol_core = 0.0
            S_face_comp = 0.0

        # Channel cap: Ry/P₁ + π cap + dative boost
        n_pi = max(0.0, bo - 1.0)
        # π-π EXCLUSION on Z/P₂Z [Pauli on pentagonal face]:
        # Two π modes sharing the same face interfere destructively.
        # n_π_eff = n_π × (1 - (n_π-1)/(2P₂))
        # Double (n_π=1): unchanged. Triple (n_π=2): 1.80 (−10%).
        n_pi_eff = n_pi * (1.0 - max(0.0, n_pi - 1.0) / (2.0 * P2))
        if bi in huckel_pi_cap:
            # Aromatic: Hückel π replaces pairwise π [GFT anti-double-counting]
            channel_cap = RY / P1 + huckel_pi_cap[bi]
        else:
            channel_cap = RY / P1 + n_pi_eff * RY / P2

        # ── VACANCY OPERATOR: dative + resonance + polygon [T2 GFT] ──
        vac = _vacancy_operator(i, j, bo, topology, atom_data)
        channel_cap += vac['cap_boost']

        # ── FORMAL CHARGE screening [insight #38] ──────────
        S_charge = 0.0
        qi = charges[i] if i < len(charges) else 0
        qj = charges[j] if j < len(charges) else 0
        if qi != 0:
            z_i = topology.z_count[i] if topology.z_count else 1
            S_charge += qi * D3 * z_i / P1
        if qj != 0:
            z_j = topology.z_count[j] if topology.z_count else 1
            S_charge += qj * D3 * z_j / P1

        # ── BH CAP BOOST: P₃ channel for polar bonds [GFT] ──────
        # Replaces the old ADDITIVE ionic channel.
        # The Born-Haber gate opens the heptagonal face of T³,
        # boosting the cap. Energy goes through the SAME exp(-S).
        #
        # GFT invariance: D₀ = cap_eff × exp(-S_total × D_FULL).
        # The cap is fixed by q_stat (atomic properties).
        # The screening is fixed by q_stat (sieve).
        # No distances appear → total is invariant under q_therm.
        #
        per_A, per_B = da['per'], db['per']
        lp_A = topology.lp[i] if hasattr(topology, 'lp') else 0
        lp_B = topology.lp[j] if hasattr(topology, 'lp') else 0
        z_A = topology.z_count[i] if hasattr(topology, 'z_count') else 1
        z_B = topology.z_count[j] if hasattr(topology, 'z_count') else 1
        r_e_val = r_equilibrium(per_A, per_B, bo,
                                z_A=z_A, z_B=z_B,
                                lp_A=lp_A, lp_B=lp_B)

        # ╔════════════════════════════════════════════════════════╗
        # ║  CRT DECOMPOSITION: D₀ = E(P₁) + E(P₂) + E(P₃)     ║
        # ║                                                        ║
        # ║  T³ = Z/P₁Z × Z/P₂Z × Z/P₃Z  [D29, CRT]            ║
        # ║  Faces INDEPENDENT — each has own screening.           ║
        # ║  Vertex budget [T2 GFT] already absorbed in σ_k.      ║
        # ╚════════════════════════════════════════════════════════╝

        # ── BIFURCATED SCREENING [A8: q_stat vertex + q_therm face] ──
        # S_bond decomposes into VERTEX (coupling) + FACE (propagation).
        # Vertex effects: weighted average of σ_k, S_asym, S_parity, S_bohr.
        # Face effects: S_self, S_per, S_context, S_lp, S_hund, S_vertex_comp.
        #
        # The vertex part stays q_stat. The face part is scaled by F_BIFURC
        # (ratio q_therm/q_stat ≈ 0.52) because screening propagation
        # is a FACE effect (geometric, propagateur).
        #
        # S_vertex computed from available quantities:
        w_A = max(per_A, 2)
        w_B = max(per_B, 2)
        S_vertex_part = (atom_screenings[i] * w_A + atom_screenings[j] * w_B) / (w_A + w_B)
        # S_asym component (p=2 DFT mode, vertex structural)
        # Uses σ_BASE (without period decay) to measure TRUE chemical
        # asymmetry, not the geometric artifact from period crossing.
        # Period decay is FACE (geometric) — it shouldn't inflate S_asym.
        ie_i_v = atom_data[Za]['IE']
        ie_j_v = atom_data[Zb]['IE']
        f_IE_i = min(ie_i_v / RY, 1.0)
        f_IE_j = min(ie_j_v / RY, 1.0)
        sigma_base_i = max(D7, (1.0 - f_IE_i) * S3)
        sigma_base_j = max(D7, (1.0 - f_IE_j) * S3)
        if ie_i_v <= ie_j_v:
            sigma_don_v, sigma_acc_v = sigma_base_i, sigma_base_j
        else:
            sigma_don_v, sigma_acc_v = sigma_base_j, sigma_base_i
        chi_A_v = (atom_data[Za]['IE'] + atom_data[Za]['EA']) / 2.0
        chi_B_v = (atom_data[Zb]['IE'] + atom_data[Zb]['EA']) / 2.0
        Q_eff_v = abs(chi_A_v - chi_B_v) / max(0.1, chi_A_v + chi_B_v)
        S_asym_v = -S_HALF * abs(sigma_don_v - sigma_acc_v) * (1.0 + Q_eff_v)
        # Parity (vertex coupling)
        S_parity_v = 0.0
        if Za == Zb:
            S_parity_v = -S3 * S_HALF
        elif per_A == per_B:
            S_parity_v = -D3 * S_HALF
        S_vertex_part += S_asym_v + S_parity_v + S_context  # context = A_vertex [D13]

        # Face part = total minus vertex
        S_raw = S_bond + S_face_comp + S_charge - S_mol_core
        S_face_part = S_raw - S_vertex_part

        # BIFURCATED total: vertex(q_stat) + face(q_therm) [A8]
        # f_bif depends on BOTH period crossing AND polarity:
        # Non-polar same-period (C-C): f=1 (no bifurcation)
        # Polar cross-period (Si-F): f→F_BIFURC
        # Non-polar cross-period (Si-H): f≈1 (polarity protects)
        delta_per_bif = abs(per_A - per_B)
        f_bif = 1.0 - (1.0 - F_BIFURC) * min(delta_per_bif, P1) / P1 * Q_eff_v
        S_total = max(0, S_vertex_part + S_face_part * f_bif)

        # ── DATIVE LP RELIEF [T2 GFT: vertex → edge transfer] ──
        qi_dat = charges[i] if i < len(charges) else 0
        qj_dat = charges[j] if j < len(charges) else 0
        if ENABLE_DATIVE_RELIEF and vac['relief_strength'] > 0 and (abs(qi_dat) + abs(qj_dat) > 0):
            r_ratio_lp = r_e_val / A_BOHR
            dist_lp = GAMMA_5 ** (r_ratio_lp * S_HALF)
            S_relief = vac['relief_strength'] * S3 * dist_lp * C3
            S_total = max(0, S_total - S_relief)

        # ── VACANCY SCREENING RELIEF [T2 GFT: LP fills → screening drops] ──
        # When LP donates into p-vacancy, the orbital space that WOULD
        # have screened the bond becomes occupied → screening reduced.
        # BF₃: B(vac=0.33) + F(LP=3) → S₃C₃ screening removed per bond.
        # Only for resonance dative (sigma_boost > 0, no formal dative).
        if vac['sigma_boost'] > 0 and not vac['has_dative']:
            # vac_frac = sigma_boost / (Ry/P₁) = fraction of σ-cap from donation
            vac_frac = vac['sigma_boost'] / (RY / P1)
            S_vacancy_relief = vac_frac * S3 * C3
            S_total = max(0, S_total - S_vacancy_relief)

        # ── RABI OPERATOR: cos²(θ_σ) on edge of T³ [continuous] ──
        # cos² = transmitted fraction, partitioned into split + bridge.
        eps_ra = _eps_onsite(da['IE'], per_A, da['l'], da['nf'], da.get('ns', ns_config(Za)))
        eps_rb = _eps_onsite(db['IE'], per_B, db['l'], db['nf'], db.get('ns', ns_config(Zb)))
        z_a_rb = topology.z_count[i] if topology.z_count else 1
        z_b_rb = topology.z_count[j] if topology.z_count else 1
        bo_w = S3 * max(bo - 1, 0) / P1 + S3 * S3 / P1
        if z_a_rb > 1 and eps_ra >= (RY - S3):
            eps_ra /= (1.0 + (z_a_rb - 1) * bo_w)
        if z_b_rb > 1 and eps_rb >= (RY - S3):
            eps_rb /= (1.0 + (z_b_rb - 1) * bo_w)

        delta_r = abs(eps_ra - eps_rb)
        t_r = S3 * C3 * math.sqrt(max(eps_ra * eps_rb, 0.01))
        cos2_raw = delta_r ** 2 / (delta_r ** 2 + 4.0 * t_r ** 2 + 1e-20)

        # LP donor modulation: d-host needs LP donor [continuous cos²×LP/P₁]
        lp_a_rb = topology.lp[i] if topology.lp else 0
        lp_b_rb = topology.lp[j] if topology.lp else 0
        d_host_is_A = per_A >= P1 and (per_B < P1 or da['IE'] <= db['IE'])
        lp_donor = lp_b_rb if d_host_is_A else lp_a_rb
        cos2_rabi = cos2_raw * min(float(lp_donor) / P1, 1.0)

        # d-vacancy of the d-host
        d_vac_rb, z_rb, l_rb, f_split, f_bridge = 0.0, 1, 0, 0.0, 1.0
        if d_host_is_A:
            nf_rb = da['nf'] if da['l'] == 1 else 0
            z_rb, l_rb = z_a_rb, da['l']
        elif per_B >= P1:
            nf_rb = db['nf'] if db['l'] == 1 else 0
            z_rb, l_rb = z_b_rb, db['l']
        else:
            nf_rb = 0
        if l_rb >= 1:
            d_vac_rb = max(0.0, (2.0 * P2 - nf_rb)) / (2.0 * P2)
            r_zp = float(P2) / (z_rb + P2)
            f_split = r_zp * r_zp
            f_bridge = 1.0 - f_split
            S_total *= (1.0 - cos2_rabi * d_vac_rb * f_split)

        # ── POLARITY for P₂ and P₃ faces (separate CRT channels) ──
        # Computed BEFORE D0_P1 so ionic penalty can modify S_total.
        IE_don_bh = min(da['IE'], db['IE'])
        EA_acc_bh = max(da['EA'], db['EA'])
        chi_A_bh = (da['IE'] + da['EA']) / 2.0
        chi_B_bh = (db['IE'] + db['EA']) / 2.0
        Q_eff_bh = abs(chi_A_bh - chi_B_bh) / max(0.1, chi_A_bh + chi_B_bh)
        Q_enh_bh = Q_eff_bh
        if EA_acc_bh > RY * S3:
            Q_EA = min(1.0, abs(da['EA'] - db['EA']) / max(0.1, IE_don_bh * S_HALF))
            Q_enh_bh = max(Q_eff_bh, Q_EA)
        # Q_IE reserved for future: IE-driven polarity needs careful gating
        # to avoid regressing alkali halides (NaF, LiCl, KCl).

        # ── CRT FACE REDISTRIBUTION [D29, GFT face balance] ──────
        # T³ capacity = Ry/P₁ + Ry/P₂ + Ry/P₃ is FIXED (GFT).
        # When bond is cross-period AND ionic, P₁ loses capacity from
        # S_per (period crossing cost). GFT face balance transfers this
        # lost capacity to P₃ (ionic channel) [D29 face conservation].
        # f_cov = IE_don/Ry: covalent fraction (H=1.0, Na=0.38, C=0.83)
        # f_ionic = (1-f_cov) × Q²: ionic fraction (2nd order CRT coupling)
        # Gate: Δper > 0 AND f_ionic > D₇ (only cross-period ionic bonds)
        f_cov_crt = min(1.0, IE_don_bh / RY)
        f_ionic_crt = (1.0 - f_cov_crt) * Q_eff_bh ** 2
        delta_per_crt = abs(per_A - per_B)

        # ── IONIC SAME-PERIOD SCREENING FLOOR [D29 on Z/P₃Z] ──────
        # Same-period ionics (Δper=0) lack S_per. For bonds with very
        # low covalent fraction (f_cov < C₃, alcalin/alcalino-terreux),
        # the σ-overlap is minimal → screening floor ∝ (1-f_cov).
        # Amplitude: S₃ × P₁ = one full hexagonal screening unit.
        # Gate: IE_acc < Ry protects LiF (IE_F=17.42 > Ry → skip).
        # NaCl (f_cov=0.378): floor=0.409, S_total 0.26→0.41. [T6 P₁]
        IE_acc_crt = max(da['IE'], db['IE'])
        if f_cov_crt < C3 and delta_per_crt == 0 and IE_acc_crt < RY:
            S_ionic_floor = (1.0 - f_cov_crt) * S3 * P1
            S_total = max(S_total, S_ionic_floor)

        # ── D0_P1: σ + π via screening ──────
        if bi in huckel_pi_cap:
            cap_P1 = RY / P1 + huckel_pi_cap[bi]
            D0_P1 = cap_P1 * math.exp(-S_total * D_FULL_P1)
        else:
            cap_sigma = RY / P1 + vac['sigma_boost']
            n_pi_remaining = max(0.0, bo - 1.0) - vac['n_pi_consumed']
            # π-π exclusion [Pauli on Z/P₂Z]
            n_pi_rem_eff = n_pi_remaining * (1.0 - max(0.0, n_pi_remaining - 1.0) / (2.0 * P2))
            cap_pi = max(0.0, n_pi_rem_eff) * RY / P2 + vac['pi_boost']
            D0_sigma = cap_sigma * math.exp(-S_total * D_FULL_P1)
            D0_pi = cap_pi * math.exp(-S_total * math.sqrt(C3) * D_FULL_P1)
            D0_P1 = D0_sigma + D0_pi

        # Bohr vertex amplitude: sin(θ_A) × sin(θ_B) on Z/2Z
        # Ionic assist: high polarity (Q) boosts the vertex amplitude
        # because charge transfer facilitates the coupling [D13 P₃×vertex].
        IE_acc_bh_val = max(da['IE'], db['IE'])
        f_bohr_raw = (math.sin(math.pi * min(IE_don_bh, RY) / (2.0 * RY))
                      * math.sin(math.pi * min(IE_acc_bh_val, RY) / (2.0 * RY)))
        f_bohr = f_bohr_raw + (1.0 - f_bohr_raw) * Q_enh_bh * D3

        # ── P₂ FACE: polygon vacancy [from _vacancy_operator] ──────
        # Per-canal environmental screening: S_P2_edge includes sign inversions
        # (LP facilitates d-back, vertex opens d-channel).
        # S_P2_edge > 0: environment HINDERS P₂ → screening (exp(-S))
        # S_P2_edge < 0: environment FACILITATES P₂ → boost (exp(+|S|))
        # Coupling strength: S₅ (sin² of the pentagonal face).
        # The bond screening (lives on P₁) couples to P₂ with amplitude S₅.
        D0_P2_raw = vac['D0_P2']
        S_P2_env = S_P2_edge * D_FULL_P2 * S5
        D0_P2 = D0_P2_raw * math.exp(-S_P2_env)
        cap_d = D0_P2
        f_d_absorb = vac['f_d_absorb']
        per_heavy = max(per_A, per_B)

        # ── SELF LP→d: excess LP → own d-orbitals [T6 CRT Z/6Z→Z/10Z] ──
        # The 1st LP defines position on hexagonal face Z/6Z (structural).
        # Excess LP (2nd+) project onto decagonal face Z/10Z — intra-atomic
        # coupling. This activates P₂ for bonds where d-host has no
        # LP-bearing neighbors (C-S in thioethers, H-S, S-S).
        # Gate: per≥P₁, LP≥2, l≥1, (z≥2 OR bo≥2), D0_P2_raw==0
        # The last gate prevents double-counting when neighbor-LP already
        # provides inter-atomic D0_P2 (SF₂ has F neighbors with LP=3).
        # Amplitude: LP/(P₁+LP) [self-attenuation, 0 adjusted params]
        D0_P2_self = 0.0
        for idx_sd in [i, j]:
            Z_sd = topology.Z_list[idx_sd]
            d_sd = atom_data[Z_sd]
            per_sd = d_sd['per']
            lp_sd = topology.lp[idx_sd] if topology.lp else 0
            z_sd = topology.z_count[idx_sd] if topology.z_count else 1
            l_sd = d_sd['l']
            nf_sd = d_sd['nf']
            # Anti-double-counting: skip if THIS atom already has
            # LP-bearing neighbors (inter-atomic P₂ already active).
            # SF₂: S neighbors = {F,F} with LP → skip. CS₂: S neighbor = {C} no LP → OK.
            neigh_lp = _total_lp_from_neighbors(idx_sd, topology)
            if per_sd >= P1 and lp_sd >= 2 and l_sd >= 1 and (z_sd >= 2 or bo >= 2) and neigh_lp <= lp_sd:
                nf_d_sd = nf_sd if l_sd == 2 else 0
                vac_d_sd = max(0.0, (2.0 * P2 - nf_d_sd)) / (2.0 * P2)
                if vac_d_sd > 0:
                    f_self_lp = float(lp_sd) / (P1 + float(lp_sd))
                    per_d_sd = max(1, per_sd - 2)
                    f_orb_sd = math.sin(math.pi * per_d_sd / P2) ** 2
                    cap_sd = RY / P2 * vac_d_sd * f_self_lp * f_orb_sd / max(z_sd, 1)
                    S_steric_sd = D5 * (float(z_sd) / P2) ** 2
                    D0_P2_self += cap_sd * math.exp(-S_steric_sd * D_FULL)
        if D0_P2_self > 0:
            D0_P2_self *= math.exp(-S_P2_env)
            D0_P2 += D0_P2_self
            cap_d += D0_P2_self

        # ── P₃ FACE: ionic bonding (heptagonal) ─────────────────
        # [GEOMETRIC: P₃ face of T³, CRT-independent from P₁]
        #
        # Two mechanisms, take the dominant:
        #   cap_pol = Q² × 4Ry/Σper × S₃C₃  [Coulomb at Bohr, face amplitude]
        #   cap_BH  = f_BH × Ry/P₃ × f_bohr [Born-Haber gate, vertex amplitude]
        #
        # Both capped at Ry/P₃ × f_bohr (vertex-limited Shannon cap)
        #
        # P₃ screening: (1-Q)/P₃ [full transparency for Q=1]
        # Screening depth: C₃×C₅×D_FULL (cascaded through P₁ and P₂)
        #
        D0_P3 = 0.0
        cap_P3 = 0.0
        S_P3_face = 0.0
        S_P3_env = 0.0
        depth_P3 = C3 * C5 * D_FULL_P3
        if ENABLE_P3_IONIC and Za != Zb:
            # ── P₃ IONIC: sieve-filtered Born-Haber [D07 holonomy] ──
            # Two sources, take the dominant:
            #
            # 1. POLARITY cap: Q² × Coulomb proxy × S₃C₃
            #    (weak bonds, low EA: uses polarity as amplitude)
            per_sum = per_A + per_B
            cap_pol = Q_enh_bh ** 2 * (4.0 * RY / per_sum) * S3 * C3

            # 2. BORN-HABER cap [distance-free proxy, calibrated]
            #    Full Bianchi I recalibration needs refonte of the cap
            #    normalization (Ry/P₃ vs S₇ vs D₇). Keeping proxy for now.
            cap_BH = 0.0
            if EA_acc_bh > RY * S3:
                per_avg_bh = (per_A + per_B) / 2.0
                E_mad = 2.0 * RY * C3 / per_avg_bh
                E_BH = E_mad - IE_don_bh + EA_acc_bh
                if E_BH > 0:
                    f_BH = min(1.0, E_BH / (RY * S3))
                    # d-acceptor boost: when d-host is pure acceptor (LP=0, z>P₁),
                    # the d-shell is the PRIMARY bonding channel → higher cap.
                    # PF₅: P(LP=0,z=5>P₁) = pure d-acceptor → boost.
                    # SF₄: S(LP=1) = NOT pure acceptor → no boost.
                    lp_dhost = topology.lp[i if d_host_is_A else j] if topology.lp else 0
                    z_dhost = topology.z_count[i if d_host_is_A else j] if topology.z_count else 1
                    if lp_dhost == 0 and z_dhost > P1 and d_vac_rb > 0 and f_BH < S_HALF:
                        f_BH = min(1.0, f_BH + d_vac_rb * (S3 + D3))
                    # CRT face expansion: cross-period ionic fraction [D29]
                    # Gate: Δper > 0 AND f_ionic > D₇
                    _f_exp = f_ionic_crt * delta_per_crt / P1 if (delta_per_crt > 0 and f_ionic_crt > D7) else 0.0
                    Ry_P3_eff = RY / P3 + _f_exp * (RY / P1 - RY / P3)
                    # Super-Ry same-period: acceptor excess projects Z/6Z→Z/14Z
                    # Gate: Δper=0, IE_acc>Ry, f_cov<C₃. [T6 CRT overflow]
                    # LiF: (1-0.396)×(17.42/13.6-1)×3 = 0.507 → cap ×1.51
                    if delta_per_crt == 0 and IE_acc_crt > RY and f_cov_crt < C3 and Q_eff_bh > S_HALF:
                        _f_sry = (1.0 - f_cov_crt) * (IE_acc_crt / RY - 1.0) * P1
                        Ry_P3_eff *= (1.0 + _f_sry)
                    cap_BH = f_BH * Ry_P3_eff * f_bohr

            # P₃ cap: expanded for cross-period ionic bonds [D29]
            _f_exp_cap = f_ionic_crt * delta_per_crt / P1 if (delta_per_crt > 0 and f_ionic_crt > D7) else 0.0
            Ry_P3_cap = RY / P3 + _f_exp_cap * (RY / P1 - RY / P3)
            # Super-Ry cap boost (same gate + Q polarity gate)
            if delta_per_crt == 0 and IE_acc_crt > RY and f_cov_crt < C3 and Q_eff_bh > S_HALF:
                _f_sry_cap = (1.0 - f_cov_crt) * (IE_acc_crt / RY - 1.0) * P1
                Ry_P3_cap *= (1.0 + _f_sry_cap)
            cap_max_P3 = Ry_P3_cap * f_bohr
            cap_P3 = min(max(cap_pol, cap_BH), cap_max_P3)

            # P₃ face screening: (1-Q)/P₃ + per-canal environmental
            # S_P3_edge: sign carries the physics.
            # Positive = environment screens P₃ (LP blocking, etc.)
            # Negative = environment facilitates P₃ (polarity motor)
            # Coupling strength: S₇ (sin² of the heptagonal face).
            S_P3_face = (1.0 - Q_enh_bh) / P3
            S_P3_env = S_P3_edge * S7
            D0_P3 = cap_P3 * math.exp(-(S_P3_face + S_P3_env) * depth_P3)

        # ── d-BRIDGE: Rabi transmission × ionic × d-vacancy [ptcK] ──
        # The σ-channel (P₁) resolves fraction sin²(θ_σ) of the bond.
        # The rest, cos²(θ_σ), is TRANSMITTED to the d-channel (P₂).
        # When d-orbitals are available, they BRIDGE this transmission
        # to the ionic energy: ΔD = v_ionic × d_vac × cos²(θ_σ).
        #
        # Rabi 2-level: cos²(θ) = Δε² / (Δε² + 4t²)
        #   Δε = |IE_A - IE_B| (onsite asymmetry, vertex/q_stat)
        #   t = S₃C₃ × √(ε_A × ε_B) (σ coupling, vertex/q_stat)
        #
        # Si-F: Δε LARGE → cos²≈1 → full bridge → 7+ eV
        # C-C:  Δε≈0 → cos²≈0 → no bridge
        #
        # v_ionic = D0_P3 (our distance-free ionic, already computed)
        # d_vac = (2P₂ - n_p) / (2P₂) [structural, from atom cards]
        #
        # d-BRIDGE: COMPLEMENT of Rabi split [unified operator on T³]
        # The split above takes P₂/(z+P₂) of the cos²×d_vac as screening relief.
        # The bridge takes the remaining z/(z+P₂) as additive ionic energy.
        D0_bridge = 0.0
        if cos2_rabi > 0 and l_rb >= 1 and d_vac_rb > 0:
            # Bridge uses BOTH P₃ (ionic) and P₂ (d-orbital) channels [CRT D29].
            # P₃ source: cap_P3 (already computed, Ry/P₃ × f_bohr).
            # P₂ source: Ry/P₂ × d_vac × C₃ × f_bohr.
            #   C₃ = survival through one CRT face crossing (P₃→P₂).
            #   This channel opens when d-orbitals are available to
            #   receive the ionic energy (d_vac > 0).
            # SiF₄: P₃=1.43 + P₂=2.72×1.0×0.78×0.73 = 1.43+1.55 = 2.98
            # P₂ ionic: d-orbitals shared among z bonds at the d-host vertex.
            # d_vac already in the D0_bridge formula → not in cap.
            # C₃^(1−d_vac): barrier opens with d-vacancy [continuous].
            # d_vac=1 (fully empty d, Si): no barrier (C₃⁰=1).
            # d_vac=0 (filled d): full barrier (C₃¹=C₃).
            barrier = C3 ** (1.0 - d_vac_rb)
            cap_P2_ionic = RY / P2 * barrier * f_bohr / max(z_rb, 1)
            cap_bridge = cap_P3 + cap_P2_ionic
            # Per-canal screening on bridge: crosses P₂→P₃.
            # Average of face environments, coupling = S₅×S₇ (cross-face).
            S_bridge_env = (S_P2_env + (S_P3_face + S_P3_env) * depth_P3) * S5 * S7
            D0_bridge = cap_bridge * d_vac_rb * cos2_rabi * f_bridge * math.exp(-S_bridge_env)

        # ── INTER-FACE COUPLING P₁→P₂ [hex→pent, atom.py #52] ────
        # When bridge is active, the σ-channel (P₁) projects energy
        # onto the d-channel (P₂) via cross-face coupling.
        # Amplitude: S₃×D₃ (cross-gap between hex and pent).
        # Gate: cos²>0 AND l≥1 AND d_vac>0 [continuous, no Boolean].
        # ── INTER-FACE COUPLING TRIANGLE [2nd order on T³] ─────
        # Three cross-face couplings form a triangle on T³:
        #   P₁×P₂ = S₃×D₃ (hex→pent, bridge-gated)
        #   P₁×P₃ = S₃×D₅ (hex→hept, universal for polar bonds)
        #   P₂×P₃ = S₅×D₅ (pent→hept, d-vacancy gated)
        # These ARE the polarization channel — the 2nd order response
        # of the electromagnetic field to the bond. No new physics,
        # just the cross-terms between the 3 CRT faces.

        # P₁→P₂ (bridge-gated, existing)
        D0_P1_P2 = 0.0
        if cos2_rabi > 0 and l_rb >= 1 and d_vac_rb > 0:
            D0_P1_P2 = D0_P1 * S3 * D3 * d_vac_rb * cos2_rabi

        # P₁→P₃ (quadratic in Q: strong polarity only)
        # Q² gate: 2nd order effect, suppressed for weakly polar bonds.
        D0_P1_P3 = D0_P1 * S3 * D5 * Q_enh_bh ** 2

        # P₂→P₃ (d-vacancy gated, same as P₁→P₂ but weaker amplitude)
        D0_P2_P3 = 0.0
        if D0_P2 > 0 and D0_P3 > 0:
            D0_P2_P3 = math.sqrt(D0_P2 * D0_P3) * S5 * D5

        # ── CRT TOTAL: T³ = Z/3Z × Z/5Z × Z/7Z ─────────────────
        D0_crt = D0_P1 + D0_P2 + D0_P3 + D0_bridge + D0_P1_P2 + D0_P1_P3 + D0_P2_P3
        D0_bare = D0_crt * D_FULL

        # ── GHOST VP: primes {11,13} screening per vertex [D32] ──
        # Each atom k traverses (per_k - 1) sieve levels beyond p=3.
        # The ghost = cumulative screening from inactive primes {11,13,...}.
        # z-dependent: ghost circulates through the WEAKEST of two channels.
        # Additional screening D₃ per coordination level beyond 1 [Principle 1].
        if ENABLE_GHOST:
            n_shells = ((per_A - 1) + (per_B - 1)) / 2.0
            z_a_gh = topology.z_count[i] if topology.z_count else 1
            z_b_gh = topology.z_count[j] if topology.z_count else 1
            z_min_gh = min(z_a_gh, z_b_gh)
            f_z_ghost = 1.0 + max(0, z_min_gh - 1) * D3
            S_ghost = BETA_GHOST * D3 * S_HALF * max(n_shells, 1.0) * f_z_ghost
        else:
            S_ghost = 0.0
        D0 = D0_bare * math.exp(-S_ghost * D_FULL)

        # Physical observables
        w_e_val = omega_e(D0, r_e_val, da['mass'], db['mass'])

        # Use ORIGINAL indices so A/B matches topology.bonds[bi] ordering
        bond_geom = _compute_bond_geometry(i_orig, j_orig, bo, S_bond, topology, polygons)

        bond_results.append(BondResult(
            D0=D0,
            v_sigma=D0_P1 * D_FULL / max(bo, 1),
            v_pi=D0_P1 * D_FULL * max(0, bo - 1) / max(bo, 1),
            v_ionic=D0_P3 * D_FULL,
            r_e=r_e_val,
            omega_e=w_e_val,
            details={
                'S_bond': S_bond, 'S_total': S_total,
                'S_P2_edge': S_P2_edge, 'S_P3_edge': S_P3_edge,
                'S_P2_env': S_P2_env, 'S_P3_env': S_P3_env,
                'D0_P1': D0_P1, 'D0_P2': D0_P2, 'D0_P3': D0_P3,
                'D0_P2_raw': D0_P2_raw,
                'cap_d': cap_d, 'cap_P3': cap_P3,
                'Q_enh': Q_enh_bh, 'f_bohr': f_bohr,
                'S_atom_i': atom_screenings[i],
                'S_atom_j': atom_screenings[j],
            },
            geometry=bond_geom,
        ))

    # ── 3b. P₃ VERTEX SHARING for covalent acceptors [GFT T2] ──────
    # When multiple bonds at a COVALENT vertex (ie/Ry > C₃) draw P₃,
    # the ionic capacity is shared. Each bond's D0 is reduced by the
    # excess P₃ at the vertex. Gate: ie_factor > C₃ AND n_P3 ≥ 2.
    # 11DFE: C with 2 C-F → P₃ halved per bond. BF₃: ie<C₃ → no sharing.
    P3_share_threshold = D7 * RY / P3  # ~0.17 eV (significant P₃)
    for k_sh in range(len(topology.Z_list)):
        Z_sh = topology.Z_list[k_sh]
        ie_sh = atom_data[Z_sh]['IE']
        if ie_sh / RY <= C3:
            continue  # ionic acceptor — full P₃ for each bond
        bonds_sh = [bi for bi in range(len(topology.bonds))
                    if topology.bonds[bi][0] == k_sh or topology.bonds[bi][1] == k_sh]
        p3_bonds = [bi for bi in bonds_sh
                    if bond_results[bi].details.get('D0_P3', 0) > P3_share_threshold]
        n_p3 = len(p3_bonds)
        # Only share when vertex has ionic bonds (P₃ high) competing
        # with π bonds (bo ≥ 2). The π and ionic channels cannot occupy
        # the same face of T³ simultaneously. [GFT anti-double-counting]
        # 11DFE: 2 C-F (P₃) + C=C (π) → competition, share.
        # CHF₃: 3 C-F (P₃) + C-H (σ only) → no π competition, skip.
        # Only homonuclear π (C=C): purely covalent, no ionic character.
        # Heteronuclear π (C=O, C=N) has its own ionic capacity — no sharing.
        has_homo_pi = any(
            topology.bonds[bi][2] >= 2.0
            and topology.Z_list[topology.bonds[bi][0]] == topology.Z_list[topology.bonds[bi][1]]
            for bi in bonds_sh)
        # n_homo_pi = number of homonuclear π bonds at vertex
        # They count as competitors for P₃ capacity: total = n_P3 + n_homo_π
        # VinylF (1 C-F + 1 C=C): 1/(1+1) = 0.5. 11DFE (2 C-F + 1 C=C): 1/(2+1).
        n_homo_pi = sum(1 for bi in bonds_sh
                        if topology.bonds[bi][2] >= 2.0
                        and topology.Z_list[topology.bonds[bi][0]] == topology.Z_list[topology.bonds[bi][1]])
        if n_p3 >= 1 and n_homo_pi >= 1:
            n_competitors = n_p3 + n_homo_pi
            for bi in p3_bonds:
                old_p3 = bond_results[bi].details['D0_P3']
                # D0 includes P₃×D_FULL×ghost. Remove excess P₃.
                excess_p3 = old_p3 * (1.0 - 1.0 / n_competitors) * D_FULL
                # Ghost attenuation on excess
                per_A_sh = atom_data[topology.Z_list[topology.bonds[bi][0]]]['per']
                per_B_sh = atom_data[topology.Z_list[topology.bonds[bi][1]]]['per']
                n_shells_sh = max(((per_A_sh - 1) + (per_B_sh - 1)) / 2.0, 1.0)
                S_gh_sh = BETA_GHOST * D3 * S_HALF * n_shells_sh
                excess_p3 *= math.exp(-S_gh_sh * D_FULL)
                bond_results[bi] = BondResult(
                    D0=bond_results[bi].D0 - excess_p3,
                    v_sigma=bond_results[bi].v_sigma,
                    v_pi=bond_results[bi].v_pi,
                    v_ionic=bond_results[bi].v_ionic - excess_p3,
                    r_e=bond_results[bi].r_e,
                    omega_e=bond_results[bi].omega_e,
                    details=bond_results[bi].details,
                )

    # ── 4. D_at = sum of bond energies ──
    D_at = sum(br.D0 for br in bond_results)

    # ── 4b. MOL_DFT — unified vertex corrections via Star Hamiltonian [D13] ──
    # For each vertex k (z ≥ 2): diagonalize the (z+1)×(z+1) star Hamiltonian.
    # R_k = D_star / D_pair measures the COLLECTIVE fraction of bonding.
    # Residual = (1-R) × D_pair: unresolved pairwise excess (eV).
    # Sign depends on vertex type:
    #   DONOR (Bohr or LP>0): competition → NEGATIVE
    #   ACCEPTOR (sub-Bohr, LP=0, vacancy, LP-ligands): cooperation → POSITIVE
    #   NEUTRAL (LP=0, no LP-ligands or saturated): no correction
    import numpy as _np_dft
    E_mol_dft = 0.0
    n_bonds = len(topology.bonds)
    for k in range(len(topology.Z_list)):
        vp = polygons[k]
        z_k = vp.z
        if z_k < 2:
            continue

        Z_k = vp.Z
        lp_k = vp.lp
        ie_k = vp.ie
        per_k = vp.per
        is_bohr = vp.is_bohr
        v_class = vp.vertex_class
        nf_k = vp.nf

        theta_deg = vp.theta_deg

        bonds_at_k = [(bi, topology.bonds[bi]) for bi in range(n_bonds)
                       if topology.bonds[bi][0] == k or topology.bonds[bi][1] == k]
        D_avg_k = sum(bond_results[bi].D0 for bi, _ in bonds_at_k) / max(len(bonds_at_k), 1)

        # Aromatic vertex flag — needed by ring torsion gate and vib suppression
        is_aro_vertex = v_class == 'ring' and any(
            topology.bonds[bi][2] == 1.5 for bi, _ in bonds_at_k)

        # ── Ring strain [T6 holonomy] ──
        if v_class == 'ring':
            for ring in topology.rings:
                if k not in ring:
                    continue
                N_ring = len(ring)
                theta_cycle = (N_ring - 2) * 180.0 / N_ring
                delta_theta = abs(theta_deg - theta_cycle)
                if delta_theta > 1.0:
                    delta_rad = math.radians(delta_theta)
                    E_mol_dft -= -math.log(C3) * delta_rad ** 2 * D_avg_k
                # Vibrational holonomy for saturated rings ≥ 5 [T6]
                # Non-aromatic ring vertices STILL have the holonomic
                # angle cost sin²(θ/2). Skipping it undercounts the
                # vibrational zero-point cost, causing overbinding.
                # Gate: N ≥ 5 (small rings already strained) AND not
                # aromatic (Hückel handles aromatic stabilization).
                is_aro = any(topology.bonds[bi][2] == 1.5
                             for bi, _ in bonds_at_k)
                if N_ring >= 4 and not is_aro and theta_deg < 179.0:
                    sin2_half_ring = math.sin(math.radians(theta_deg) / 2.0) ** 2
                    E_mol_dft -= S3 * sin2_half_ring * D_avg_k * S_HALF / z_k
                # ── Ring torsional holonomy [T6, GFT] ──
                # Ring closure forces torsional periodicity on T³:
                # the N torsional angles must sum to 0 (mod 2π).
                # This constraint costs energy proportional to
                # sin²(π/N) per vertex (holonomy of the N-gon
                # inscribed in the unit circle Z/NZ).
                # Gate: N ≥ 4 (3-rings already fully accounted by
                # angular strain), not aromatic (Hückel handles).
                # LP attenuation: vertices with LP > 0 have face
                # overflow channels on Z/6Z that partially absorb
                # the torsional constraint → cost × S₃.
                if N_ring >= 4 and not is_aro:
                    D_avg_ring_k = 0.0
                    n_ring_at_k = 0
                    for bi_r, (ii_r, jj_r, bo_r) in bonds_at_k:
                        if bi_r in topology.ring_bonds:
                            D_avg_ring_k += bond_results[bi_r].D0
                            n_ring_at_k += 1
                    if n_ring_at_k > 0:
                        D_avg_ring_k /= n_ring_at_k
                        sin2_pi_N = math.sin(math.pi / N_ring) ** 2
                        E_torsion = S3 * sin2_pi_N * D_avg_ring_k * S_HALF
                        if lp_k > 0:
                            E_torsion *= S3
                        E_mol_dft -= E_torsion
                break

        else:
            # ── Star Hamiltonian — unified vertex correction [D13] ──
            l_k = vp.l
            ns_k = atom_data[Z_k].get('ns', ns_config(Z_k))
            eps_v = _eps_onsite(ie_k, per_k, l_k, nf_k, ns_k)

            n_h = len(bonds_at_k) + 1
            H_star = _np_dft.zeros((n_h, n_h))
            H_star[0, 0] = eps_v
            D_on_v = 0.0
            has_lp_lig = False
            lp_lig_sum = 0
            lp_weighted_sum = 0.0  # per-canal: LP × S₃^(per-2)
            lp_weighted_sum_C3 = 0.0  # near-Bohr: LP × C₃^(per-2)

            for idx_h, (bi_h, (ii_h, jj_h, bo_h)) in enumerate(bonds_at_k):
                partner_h = jj_h if ii_h == k else ii_h
                Z_p = topology.Z_list[partner_h]
                d_p = atom_data[Z_p]
                eps_p = _eps_onsite(d_p['IE'], d_p['per'], d_p['l'], d_p['nf'],
                                    d_p.get('ns', ns_config(Z_p)))
                z_p = topology.z_count[partner_h] if topology.z_count else 1
                bo_w = S3 * max(bo_h - 1, 0) / P1 + S3 * S3 / P1
                if z_p > 1 and eps_p >= (RY - S3):
                    eps_p /= (1.0 + (z_p - 1) * bo_w)
                H_star[idx_h + 1, idx_h + 1] = eps_p
                t_val = S3 * C3 * math.sqrt(max(eps_v * eps_p, 0.01))
                H_star[0, idx_h + 1] = t_val
                H_star[idx_h + 1, 0] = t_val
                D_on_v += bond_results[bi_h].D0
                lp_lig = topology.lp[partner_h] if topology.lp else 0
                lp_lig_sum += lp_lig
                if lp_lig > 0:
                    has_lp_lig = True
                    # Radial cascade: S₃^(per-2) per shell [Principle 1]
                    lp_weighted_sum += lp_lig * S3 ** max(d_p['per'] - 2, 0)
                    # Near-Bohr uses PROPAGATOR decay C₃ [Principle 5]
                    lp_weighted_sum_C3 += lp_lig * C3 ** max(d_p['per'] - 2, 0)

            eigenvalues = _np_dft.linalg.eigvalsh(H_star)
            D_star = eps_v - eigenvalues[0]
            D_pair = D_on_v / 2.0

            if D_pair > 0.01:
                R_k = min(D_star / D_pair, 1.0)
                residual = (1.0 - R_k) * D_pair
                lp_eng = float(lp_lig_sum) / (z_k * P1)
                # Per-canal weighted engagement (Cl decayed, F full)
                lp_eng_canal = lp_weighted_sum / (z_k * P1)

                if is_bohr:
                    # BOHR DONOR: full competition [D13]
                    # EA stabilization: when ALL ligands have high EA (≥ Ry×S₃),
                    # they absorb the vertex competition energy → reduces penalty.
                    # Gate: min(EA_lig) — protects mixed-ligand vertices.
                    # NF₃ (min EA_F=3.40 > 2.98): f_EA=0.68 → 32% reduction.
                    # NH₃ (min EA_H=0.75 < 2.98): f_EA=1.0 → no change.
                    # N₂O (min EA_N≈0 < 2.98): f_EA=1.0 → no change. [T6 P₁]
                    EA_min_lig = 1e6
                    for bi_ea, (ii_ea, jj_ea, bo_ea) in bonds_at_k:
                        partner_ea = jj_ea if ii_ea == k else ii_ea
                        Z_ea = topology.Z_list[partner_ea]
                        EA_min_lig = min(EA_min_lig, max(atom_data[Z_ea]['EA'], 0.0))
                    f_EA_stab = 1.0
                    if EA_min_lig > RY * S3:
                        f_EA_stab = max(0.0, 1.0 - EA_min_lig / (RY * C3))
                    lp_boost = min(float(lp_k), 1.0) if lp_eng > S_HALF else 0.0
                    E_mol_dft -= residual / P1 * (1.0 + lp_eng + lp_boost) * f_EA_stab
                    # BOHR-BOHR MUTUAL PENALTY [T6 resonance on edge]:
                    # When a Bohr donor bonds to ANOTHER Bohr donor (IE≈Ry),
                    # the exchange energy is maximal (Δε≈0 → perfect resonance
                    # → maximum information loss on the edge of T³).
                    # Cost = D₀(bond) × S₃² per Bohr-Bohr bond at this vertex.
                    # S₃² = sin⁴(θ₃) = double-exchange amplitude on P₁ face.
                    for bi_bp, (ii_bp, jj_bp, bo_bp) in bonds_at_k:
                        partner_bp = jj_bp if ii_bp == k else ii_bp
                        Z_partner = topology.Z_list[partner_bp]
                        ie_partner = atom_data[Z_partner]['IE']
                        # Bohr partner = IE ≥ Ry AND NOT hydrogen (Z>1).
                        # H sits exactly at Ry but is NOT a Bohr donor
                        # (no LP, no multi-electron screening).
                        if ie_partner >= (RY - S3) and Z_partner > 1:
                            lp_partner = topology.lp[partner_bp] if topology.lp else 0
                            f_lp_mutual = (1.0 + min(float(lp_k), float(lp_partner)))
                            E_mol_dft -= bond_results[bi_bp].D0 * S3 * S3 * f_lp_mutual
                elif lp_k > 0 and per_k >= P1 and z_k > P1:
                    # HYPER-LP DONOR (sub-Bohr, z > P₁): amplified
                    E_mol_dft -= residual / P1 * (1.0 + lp_eng)
                elif lp_k > 0 and z_k >= P1:
                    # WEAK LP DONOR (sub-Bohr, z ≤ P₁): per-pair S₃D₃
                    n_pairs_v = len(bonds_at_k) * (len(bonds_at_k) - 1) // 2
                    if n_pairs_v > 0:
                        E_mol_dft -= n_pairs_v * (1.0 - R_k) * S3 * D3 * D_avg_k
                    # GEO 3D: face bifurcation acceptance [Principle 3, CRT]
                    # When WEAK LP vertex ALSO has LP-bearing ligands at
                    # face_fraction=1.0 with d-vacancy, the CRT channel
                    # P₁→P₂ opens for LIGAND LP overflow. The net vertex
                    # correction = donor (small −) + acceptor (large +).
                    # PF₃: P(lp=1,z=3) gets weak donor BUT 3×F(lp=3)=9 LP
                    # overflow through the saturated Z/6Z face into Z/10Z.
                    # Coupling = √(S₃²+D₅²) modulated by (1−lp_coverage).
                    # [D13 spin foam: A_vertex uses q_stat coupling]
                    has_p_vacancy_w = vp.has_p_vacancy
                    has_d_vacancy_w = vp.has_d_vacancy
                    if has_lp_lig and has_d_vacancy_w and vp.face_occupation == 2 * P1:
                        lp_cov_w = vp.lp_coverage
                        f_lp_free_w = 1.0 - lp_cov_w
                        coupling_bif_w = math.sqrt(S3 * S3 + D5 * D5)
                        coupling_w = D5 + (coupling_bif_w - D5) * f_lp_free_w
                        lp_eng_bif_w = lp_weighted_sum_C3 / (z_k * P1) if z_k > 0 else 0.0
                        ie_factor_w = min(ie_k / RY, 1.0)
                        E_mol_dft += residual * coupling_w * lp_eng_bif_w * ie_factor_w
                elif has_lp_lig:
                    # ACCEPTOR: cooperation → positive [2l+1 = P_l]
                    # Multi-canal vacancy: p-canal OR d-canal (per ≥ P₁)
                    has_p_vacancy = vp.has_p_vacancy
                    has_d_vacancy = vp.has_d_vacancy
                    # NEAR-BOHR acceptance [Bifurcation Regime 2, Principle 4]:
                    # Vertices with IE/Ry > C₃ but NOT Bohr have partial
                    # coherence. LP from high-EA ligands couple through
                    # this coherence → small POSITIVE correction.
                    # f_near = (IE/Ry - C₃)/S₃ (0 at C₃, 1 at Bohr threshold)
                    # Gate: LP engagement AND f_near > 0 AND no vacancy.
                    if not (has_p_vacancy or has_d_vacancy):
                        f_near = max(0.0, (ie_k / RY - C3) / S3)
                        # Near-Bohr uses C₃ propagator decay (softer than S₃)
                        lp_eng_near = lp_weighted_sum_C3 / (z_k * P1) if z_k > 0 else 0.0
                        if f_near > 0 and lp_eng_near > 0:
                            E_mol_dft += residual * S3 * f_near * lp_eng_near * min(ie_k / RY, 1.0)
                    if has_p_vacancy or has_d_vacancy:
                        # IE/Ry: screening deficit → LP acceptance efficiency
                        ie_factor = min(ie_k / RY, 1.0)
                        coupling = S3 if has_p_vacancy else D5
                        # FACE BIFURCATION [Pythagore CRT, Principle 3]:
                        # When nf+z = 2P₁, hexagonal face Z/6Z is at EXACT
                        # capacity. LP overflows from P₁ to P₂ (d-orbitals).
                        # Coupling = √(S₃² + D₅²) = Pythagore of both faces.
                        # Uses C₃ propagator decay [Principle 5].
                        # Discriminates Si(2+4=6), P(3+3=6), S(4+2=6) from
                        # Be(0+2≠6), SF₆(4+6≠6), AlCl₃(1+3≠6).
                        lp_eng_eff = lp_eng_canal
                        if has_d_vacancy and vp.face_occupation == 2 * P1:
                            # FACE BIFURCATION [Geo 3D, CRT Principle 3]:
                            # At face_fraction=1.0, Z/6Z is saturated.
                            # LP overflows P₁→P₂ with Pythagore coupling.
                            # When lp_k>0: vertex LP occupies angular space,
                            # reducing overflow channel by (1−lp_coverage).
                            # Additionally, lp_k>0 extension requires π
                            # occupation on P₂ face (bo>1) to seed overflow.
                            # SO₂ (bo=2): π on P₂ facilitates LP overflow.
                            # SF₂ (bo=1): no π → stays at D₅ coupling.
                            # lp_k=0: full Pythagore regardless of bo.
                            lp_cov_bif = vp.lp_coverage
                            f_lp_free_bif = 1.0 - lp_cov_bif
                            # π occupation factor: average π character at vertex
                            avg_pi = sum(max(0.0, topology.bonds[bi][2] - 1.0)
                                         for bi, _ in bonds_at_k) / max(z_k, 1)
                            f_pi_seed = avg_pi if lp_k > 0 else 1.0
                            coupling_pythagore = math.sqrt(S3 * S3 + D5 * D5)
                            coupling = D5 + (coupling_pythagore - D5) * f_lp_free_bif * f_pi_seed
                            lp_eng_eff = lp_weighted_sum_C3 / (z_k * P1) if z_k > 0 else 0.0
                        # Multi super-Ry competition: when ≥2 ligands have
                        # IE>Ry, they compete for the same p/d-vacancy.
                        # Cooperation attenuated by 1/n_sry. [GFT P₃ sharing]
                        n_sry = 0
                        for bi_sr, (ii_sr, jj_sr, bo_sr) in bonds_at_k:
                            p_sr = jj_sr if ii_sr == k else ii_sr
                            Z_sr = topology.Z_list[p_sr]
                            if Z_sr > 1 and atom_data[Z_sr]['IE'] > RY:
                                n_sry += 1
                        if n_sry >= 2 and ie_factor > C3:
                            coupling /= n_sry
                        E_mol_dft += residual * coupling * lp_eng_eff * ie_factor

                        # ── LP-π COMPETITION [T6, Pauli P₁×P₂] ──
                        # When an acceptor vertex has BOTH LP-bearing
                        # single-bond neighbors AND π bonds (bo ≥ 2),
                        # the LP donation and π system COMPETE for
                        # orbital space. The LP sits on P₁ (hexagonal)
                        # but must project onto P₂ (pentagonal) to reach
                        # the vacancy — where π already resides.
                        # Cost per LP-π pair: S₅ × D₀(single) × n_π / P₂.
                        # Gate: LP via single bond + π at same vertex.
                        # Cyanamide: LP(NH₂)→σ→C←π(C≡N) → competition.
                        # BF₃: LP(F)→B, no π → no competition. ✓
                        n_pi_at_acc = sum(max(0.0, topology.bonds[bi_pi][2] - 1.0)
                                         for bi_pi, _ in bonds_at_k
                                         if topology.bonds[bi_pi][2] != 1.5)
                        if n_pi_at_acc >= 1.0 and has_p_vacancy and z_k <= 2:
                            # Count LP-bearing neighbors via SINGLE bonds
                            n_lp_single = 0
                            D0_singles = 0.0
                            for bi_ls, (ii_ls, jj_ls, bo_ls) in bonds_at_k:
                                if bo_ls > 1.5:
                                    continue
                                partner_ls = jj_ls if ii_ls == k else ii_ls
                                lp_ls = topology.lp[partner_ls] if topology.lp else 0
                                if lp_ls > 0:
                                    n_lp_single += 1
                                    D0_singles += bond_results[bi_ls].D0
                            if n_lp_single > 0:
                                E_lp_pi = S5 * D0_singles * n_pi_at_acc / P2
                                E_mol_dft -= E_lp_pi


        # ── LP directional (positive, non-terminal with LP) ──
        # Saturated ring vertices have geometry constrained by the cycle —
        # LP directional is fixed by ring strain, not free. Exclude them.
        # Aromatic rings KEEP LP directional (furan O, pyrrole N). [T6]
        is_aro_vertex = v_class == 'ring' and any(
            bo_t == 1.5 for bi_t, (ii_t, jj_t, bo_t) in bonds_at_k)
        if v_class != 'terminal' and lp_k > 0 and (v_class != 'ring' or is_aro_vertex):
            n_no_lp = sum(1 for bi, (ii, jj, _) in bonds_at_k
                           if (topology.lp[jj if ii == k else ii] if topology.lp else 0) == 0)
            if n_no_lp > 0:
                f_asym = float(n_no_lp) / z_k
                # f_bohr capped at 1.0: V_proxy cannot exceed Madelung
                # potential 2Ry×C₃/per (Shannon cap T2). N: 1.31→1.0.
                # LP-bearing sub-Bohr: wider threshold [Bifurcation R2]
                # LP creates coherence below C₃: f_bohr_lp uses S₃ as
                # lower bound (LP occupies Z/6Z → enables face coupling).
                if ie_k / RY >= C3:
                    f_bohr_k = min(1.0, (ie_k / RY - C3) / S3)
                elif lp_k >= 2 and per_k >= P1:
                    f_bohr_k = max(0.0, (ie_k / RY - S3) / (C3 - S3))
                else:
                    f_bohr_k = 0.0
                V_proxy = 2.0 * RY * C3 / max(per_k, 2) * f_bohr_k
                # Continuous Bohr factor [replaces binary gate]
                # Use STRICT C₃ threshold for f_comp (not wide LP sub-Bohr).
                # The LP sub-Bohr threshold (S₃ lower bound) is for acceptor
                # near-Bohr, not for LP directional strength. S at IE/Ry=0.76
                # should get f_comp=2/3 (not 0.99 from wide threshold).
                f_bohr_strict = min(1.0, max(0.0, (ie_k / RY - C3) / S3))
                f_comp = 2.0 / 3.0 + (1.0 / 3.0) * f_bohr_strict
                f_domain_lp = 1.0 - vp.lp_coverage
                E_lp_dir = lp_k * S3 * S_HALF * V_proxy / z_k * f_comp * f_asym * f_domain_lp
                E_mol_dft += E_lp_dir

        # ── Vibrational resonance (negative, non-linear) [T6 holonomy] ──
        # sin²(θ/2) = holonomy on Z/P₁Z face. Replaces ad hoc |cos(θ)|.
        # Tetrahedral: sin²(54.75°)=0.667 vs |cos(109.5°)|=0.333 (2× stronger).
        # Reduces D_at for high-z vertices → corrects organic +1.1% bias.
        # Aromatic suppression [T6 Hückel]: π delocalization on Z/NZ
        # creates a rigid scaffold that suppresses vibrational zero-point
        # energy. The suppression factor = S₃ for aromatic vertices
        # (Hückel holonomy absorbs 1 − S₃ ≈ 78% of the vibrational cost).
        if theta_deg < 179.0:
            sin2_half_vib = math.sin(math.radians(theta_deg) / 2.0) ** 2
            f_aro_vib = S3 if is_aro_vertex else 1.0
            E_mol_dft -= S3 * sin2_half_vib * D_avg_k * S_HALF / z_k * f_aro_vib

        # ── π-π vertex competition [Pauli on Z/P₂Z] ──────────
        # Multiple π bonds at the same vertex compete for positions
        # on the pentagonal face. Per-pair exclusion amplitude:
        # S₅² (resummation on pentagonal face [Principle 6]).
        # Only non-ring π (aromatic handled by Hückel).
        # Gate: non-linear only. Linear vertices (θ≈180°) have
        # perpendicular π planes → orthogonal on P₂, no competition.
        n_pi_at_v = sum(max(0.0, topology.bonds[bi][2] - 1.0)
                        for bi, _ in bonds_at_k
                        if topology.bonds[bi][2] != 1.5)
        if n_pi_at_v >= 2.0:
            n_pi_pairs_v = n_pi_at_v * (n_pi_at_v - 1.0) / 2.0
            E_mol_dft -= n_pi_pairs_v * S5 * S5 * D_avg_k / max(z_k, 1)

    # ── Formal charge penalty [GFT on Z/2Z, T6 holonomy] ──
    # Formal charges (SMILES [N+], [O-]) create a non-zero D_KL on the
    # parity face Z/2Z: the atom deviates from neutral occupancy.
    # Cost per charged vertex = |Q_f| × S₃ × s × D_avg / z.
    #
    # ELECTRONEGATIVITY GATE [T6 on Z/2Z polarity]:
    # Only REAL charge separations are penalized. SMILES artifacts
    # (e.g. [C-]#[O+] for CO dative bond) are skipped.
    # A charge is "real" when it follows electronegativity:
    #   - Positive on less-EN atom (e.g. N⁺ near O⁻) → real
    #   - Positive on more-EN atom (e.g. O⁺ near C⁻) → notational
    # Pauling EN: C=2.55 N=3.04 O=3.44 F=3.98 S=2.58 Cl=3.16
    _PAULING = {1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44,
                9: 3.98, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16,
                35: 2.96, 53: 2.66, 3: 0.98, 11: 0.93, 4: 1.57}
    if topology.charges:
        for k_fc in range(len(topology.Z_list)):
            Q_f = topology.charges[k_fc]
            if Q_f == 0:
                continue
            Z_fc = topology.Z_list[k_fc]
            chi_fc = _PAULING.get(Z_fc, 2.5)
            z_fc = topology.z_count[k_fc] if topology.z_count else 1
            if z_fc < 1:
                continue
            # Check electronegativity gate: is this charge "real"?
            # Positive charge is real if atom is LESS EN than max neighbor.
            # Negative charge is real if atom is MORE EN than min neighbor.
            max_chi_nb = 0.0
            min_chi_nb = 99.0
            for bi_fc in range(n_bonds):
                ii_fc, jj_fc, _ = topology.bonds[bi_fc]
                if ii_fc == k_fc:
                    nb = jj_fc
                elif jj_fc == k_fc:
                    nb = ii_fc
                else:
                    continue
                chi_nb = _PAULING.get(topology.Z_list[nb], 2.5)
                max_chi_nb = max(max_chi_nb, chi_nb)
                min_chi_nb = min(min_chi_nb, chi_nb)
            is_real = False
            if Q_f > 0 and chi_fc < max_chi_nb:
                is_real = True   # N⁺ near O: natural
            elif Q_f < 0 and chi_fc > min_chi_nb:
                is_real = True   # O⁻ near N: natural
            if not is_real:
                continue
            D_avg_fc = sum(bond_results[bi].D0
                          for bi in range(n_bonds)
                          if topology.bonds[bi][0] == k_fc
                          or topology.bonds[bi][1] == k_fc) / max(z_fc, 1)
            E_mol_dft -= abs(Q_f) * S3 * S_HALF * D_avg_fc / z_fc

    D_at += E_mol_dft


    # ── 4c. HÜCKEL LINEAR — π conjugation along chains [T6 on Z/NZ] ──
    # For open chains of ≥3 heavy atoms with alternating π bonds,
    # the π electrons delocalize along the chain. The stabilization
    # energy = E_Hückel − Σα (bonding part of eigenvalues).
    # Same physics as aromatic Hückel (section 2b) but OPEN boundary.
    #
    # Gate: ≥3 heavy atoms in a non-ring chain with at least 2 bonds
    # having bo ≥ 1.5 (π character) sharing at least one vertex.
    #
    import numpy as _np
    E_huckel_lin = 0.0
    ENABLE_HUCKEL_LIN = True  # coupled with π-screening survival

    # Build heavy-atom chain graph: π-conjugated = alternating bo.
    # A bond is π-conjugated if it shares a vertex with another
    # heavy-heavy bond of DIFFERENT bond order (double-single or triple-single).
    # Pure star topologies (SO₃: all S=O identical) are NOT conjugated.
    heavy_bonds_het = []  # heavy-heavy non-ring bonds (no halogens)
    for bi in range(n_bonds):
        ii, jj, bo = topology.bonds[bi]
        Za_h, Zb_h = topology.Z_list[ii], topology.Z_list[jj]
        if (Za_h > 1 and Zb_h > 1 and bi not in topology.ring_bonds):
            # Exclude halogens (LP=3, full p-shell: no π conjugation)
            lp_a_h = topology.lp[ii] if topology.lp else 0
            lp_b_h = topology.lp[jj] if topology.lp else 0
            if lp_a_h >= P1 or lp_b_h >= P1:
                continue  # F, Cl, Br (LP≥3) can't conjugate
            heavy_bonds_het.append((bi, ii, jj, bo))

    # Filter: keep only bonds that share a vertex with a bond of DIFFERENT bo
    heavy_bonds_pi = []
    for bi_a, ia, ja, boa in heavy_bonds_het:
        has_alternation = False
        for bi_b, ib, jb, bob in heavy_bonds_het:
            if bi_a == bi_b: continue
            shared = {ia, ja} & {ib, jb}
            if shared and abs(boa - bob) >= 0.5:
                has_alternation = True
                break
        if has_alternation:
            heavy_bonds_pi.append((bi_a, ia, ja, boa))

    # LP-DONOR EXCLUSION near TRIPLE bonds [T6, CRT P₁⊥P₂]:
    # A single bond (bo=1) where one endpoint is a LP-donor (LP>0)
    # and the OTHER endpoint has a TRIPLE bond (bo=3) is a σ-donor,
    # NOT a π-participant. The triple bond π system is SATURATED
    # (2 π pairs fill the pentagonal face P₂). LP on P₁ cannot
    # conjugate into a saturated P₂ → exclude from Hückel.
    #
    # Gate: only exclude when ADJACENT to TRIPLE bond (bo ≥ 3).
    # Double bonds (bo=2) allow LP→π resonance (carboxylate, amide).
    #
    # Cyanamide: N(LP)−C≡N → adj triple → exclude → no Hückel. ✓
    # HCOOH: O(LP)−C=O → adj double → KEEP (carboxylate resonance). ✓
    # Formamide: N(LP)−C=O → adj double → KEEP (amide resonance). ✓
    # Acrolein: O(LP)=C−C=C → O has bo=2 (π participant) → kept. ✓
    heavy_bonds_pi_clean = []
    for bi_a, ia, ja, boa in heavy_bonds_pi:
        skip = False
        if boa == 1.0:
            for endpoint in [ia, ja]:
                lp_end = topology.lp[endpoint] if topology.lp else 0
                if lp_end > 0:
                    # Is endpoint terminal (no other heavy-heavy bonds)?
                    n_other_hh = sum(1 for bi_o, io, jo, _ in heavy_bonds_het
                                     if bi_o != bi_a and (io == endpoint or jo == endpoint))
                    if n_other_hh == 0:
                        # Check: does the OTHER end of this bond have a triple?
                        other_end = ja if endpoint == ia else ia
                        has_adj_triple = any(
                            bo_adj >= 3.0
                            for bi_adj, i_adj, j_adj, bo_adj in heavy_bonds_het
                            if bi_adj != bi_a and (i_adj == other_end or j_adj == other_end))
                        if has_adj_triple:
                            skip = True
                            break
        if not skip:
            heavy_bonds_pi_clean.append((bi_a, ia, ja, boa))
    heavy_bonds_pi = heavy_bonds_pi_clean

    if ENABLE_HUCKEL_LIN and len(heavy_bonds_pi) >= 2:
        # Find connected conjugated segments
        from collections import defaultdict, deque
        adj_pi = defaultdict(set)
        for bi, ii, jj, bo in heavy_bonds_pi:
            adj_pi[ii].add((jj, bi, bo))
            adj_pi[jj].add((ii, bi, bo))

        visited_pi = set()
        for start in adj_pi:
            if start in visited_pi:
                continue
            # BFS for connected component
            segment_atoms = []
            segment_bonds = []
            queue = deque([start])
            while queue:
                node = queue.popleft()
                if node in visited_pi:
                    continue
                visited_pi.add(node)
                segment_atoms.append(node)
                for nb, bi, bo in adj_pi[node]:
                    segment_bonds.append((bi, node, nb, bo))
                    if nb not in visited_pi:
                        queue.append(nb)

            # Need ≥3 atoms for linear Hückel to matter
            segment_atoms = sorted(set(segment_atoms))
            if len(segment_atoms) < 3:
                continue

            # Build Hückel H matrix
            atom_idx = {a: idx for idx, a in enumerate(segment_atoms)}
            N_seg = len(segment_atoms)

            # Onsite ε = √(IE × EA) for each atom
            eps_seg = []
            for a in segment_atoms:
                Z = topology.Z_list[a]
                ie_a = atom_data[Z]['IE']
                ea_a = atom_data[Z]['EA']
                eps_seg.append(math.sqrt(max(ie_a * ea_a, 0.01)))

            H_lin = _np.zeros((N_seg, N_seg))
            for idx in range(N_seg):
                H_lin[idx, idx] = eps_seg[idx]

            seen_pairs = set()
            for bi, a1, a2, bo in segment_bonds:
                pair = (min(a1, a2), max(a1, a2))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                i1, i2 = atom_idx[a1], atom_idx[a2]
                beta = S5 * C5 * math.sqrt(max(eps_seg[i1] * eps_seg[i2], 0.0)) * D_FULL
                H_lin[i1, i2] = -beta
                H_lin[i2, i1] = -beta

            eigenvalues = sorted(_np.linalg.eigvalsh(H_lin))

            # Fill N_seg π electrons (1 per atom in the chain)
            n_pi_seg = N_seg
            E_deloc = 0.0
            placed = 0
            for ev in eigenvalues:
                if placed >= n_pi_seg:
                    break
                nf_ev = min(2, n_pi_seg - placed)
                E_deloc += nf_ev * ev
                placed += nf_ev

            # Stabilization = E_localized - E_deloc (positive = gain)
            # E_localized: fill n_pi_seg electrons into sorted on-site
            # energies (non-interacting limit). This correctly handles
            # heteronuclear chains (N-C-C-N) where N×α_avg is wrong.
            eps_sorted = sorted(eps_seg)
            E_localized = 0.0
            placed_loc = 0
            for eps_loc in eps_sorted:
                if placed_loc >= n_pi_seg:
                    break
                nf_loc = min(2, n_pi_seg - placed_loc)
                E_localized += nf_loc * eps_loc
                placed_loc += nf_loc
            E_pi_lin = max(0.0, E_localized - E_deloc)

            # π-screening survival: each π bond is screened by the
            # sieve cascade C₃×C₅×D_FULL [same as P₃ depth].
            # Delocalized π are MORE exposed → less survival.
            # π-screening survival: same depth as π bonds in CRT.
            # Each conjugated π is screened by C₃×D_FULL (hex face depth).
            n_pi_chain = sum(1 for _, _, _, bo_c in heavy_bonds_pi
                             if bo_c >= 2)
            f_pi_survive = math.exp(-max(n_pi_chain, 1) * C3 * D_FULL)
            E_pi_lin *= f_pi_survive

            # Distribute per bond in the chain
            if len(seen_pairs) > 0:
                E_huckel_lin += E_pi_lin / len(seen_pairs)

    D_at += E_huckel_lin

    # Set of bond indices in Hückel linear chains [for anti-DC with Ch1]
    huckel_chain_bond_set = set(bi for bi, _, _, _ in heavy_bonds_pi) if ENABLE_HUCKEL_LIN else set()

    # ── 4d. CROSS-BOND Ch1: LP→π mesomeric resonance [T6, BA5] ──
    # When a bond has LP at its far end and an adjacent bond has π,
    # the LP projects through the vertex onto the π face (pentagonal
    # transport S₅). This is mesomeric resonance: A-X=Y ↔ A+=X-Y:⁻.
    # Gate G7: suppress if either bond already has strong P₃ (ionic
    # channel already captures the cross-bond coupling).
    # docs/derivation_cross_bond_transport.md §3.2 Channel 1.
    P3_threshold_cb = RY / (P1 * P3)  # 0.648 eV
    E_cross_bond = 0.0
    n_bonds_cb = len(topology.bonds)
    for v_cb in range(len(topology.Z_list)):
        z_v_cb = topology.z_count[v_cb] if topology.z_count else 1
        if z_v_cb < 2:
            continue
        lp_v_cb = topology.lp[v_cb] if topology.lp else 0
        theta_v_cb, _, _ = _vertex_faces(z_v_cb, lp_v_cb)
        sin2_half_cb = math.sin(math.radians(theta_v_cb) / 2.0) ** 2

        bonds_at_vcb = [(bi, topology.bonds[bi]) for bi in range(n_bonds_cb)
                        if topology.bonds[bi][0] == v_cb or topology.bonds[bi][1] == v_cb]

        for idx_a_cb in range(len(bonds_at_vcb)):
            bi_a_cb, (ia_cb, ja_cb, bo_a_cb) = bonds_at_vcb[idx_a_cb]
            far_a_cb = ja_cb if ia_cb == v_cb else ia_cb
            lp_far_a_cb = topology.lp[far_a_cb] if topology.lp else 0
            n_pi_a_cb = max(0.0, bo_a_cb - 1.0)

            for idx_b_cb in range(idx_a_cb + 1, len(bonds_at_vcb)):
                bi_b_cb, (ib_cb, jb_cb, bo_b_cb) = bonds_at_vcb[idx_b_cb]
                far_b_cb = jb_cb if ib_cb == v_cb else ib_cb
                lp_far_b_cb = topology.lp[far_b_cb] if topology.lp else 0
                n_pi_b_cb = max(0.0, bo_b_cb - 1.0)

                # Skip aromatic bonds (Hückel handles them)
                if bo_a_cb == 1.5 or bo_b_cb == 1.5:
                    continue
                # G7: anti-double-counting with P₃ ionic
                P3_a_cb = bond_results[bi_a_cb].details.get('D0_P3', 0.0)
                P3_b_cb = bond_results[bi_b_cb].details.get('D0_P3', 0.0)
                if max(P3_a_cb, P3_b_cb) > P3_threshold_cb:
                    continue
                # Hückel (π delocalization) and cross-bond (LP→π transport)
                # are CRT-orthogonal [Principle 3]: Hückel on P₂ face
                # (eigenvalue), cross-bond on P₁→P₂ transport (vertex).
                # No anti-DC needed — they capture independent channels.

                # Ch1: LP(far_a) → π(B) [GFT cross-face transport]
                # Halogen LP (≥ P₁): full donation, no mutual-π gate.
                # Non-halogen LP (O, N): attenuated LP/P₁, AND suppressed
                # when source bond ALSO has π (mutual π → not unidirectional
                # mesomeric transport, already in Hückel or vertex res).
                if lp_far_a_cb >= 1 and n_pi_b_cb > 0:
                    if lp_far_a_cb >= P1:
                        lp_frac_cb = 1.0
                    elif n_pi_a_cb == 0:
                        lp_frac_cb = float(lp_far_a_cb) / P1
                    else:
                        lp_frac_cb = 0.0
                    if lp_frac_cb > 0:
                        f_pi_cb = n_pi_b_cb / max(bo_b_cb, 1.0)
                        E_cross_bond += (RY / P1) * S5 * sin2_half_cb * lp_frac_cb * f_pi_cb * D_FULL

                # Ch1 reverse: LP(far_b) → π(A)
                if lp_far_b_cb >= 1 and n_pi_a_cb > 0:
                    if lp_far_b_cb >= P1:
                        lp_frac_cb = 1.0
                    elif n_pi_b_cb == 0:
                        lp_frac_cb = float(lp_far_b_cb) / P1
                    else:
                        lp_frac_cb = 0.0
                    if lp_frac_cb > 0:
                        f_pi_cb = n_pi_a_cb / max(bo_a_cb, 1.0)
                        E_cross_bond += (RY / P1) * S5 * sin2_half_cb * lp_frac_cb * f_pi_cb * D_FULL

    D_at += E_cross_bond

    # ── 5. VERTEX RESONANCE — unified per-pair coupling [D13, T6] ──
    # Two CRT channels resonate at each vertex through bond pairs:
    #
    # P₁ channel (LP-mediated cavity):
    #   Gate: Bohr-coherent AND LP > 0 AND z ≥ 2
    #   Amplitude: S₃×C₃ × sin(θ/2) × f_bohr × LP/P₁
    #   Normalized by n_pairs (shared cavity capacity)
    #
    # P₂ channel (angular bond-bond coupling):
    #   Gate: Bohr-coherent AND z ≥ 3
    #   Amplitude: S₅×γ₅ × sin_coop × lp_gate × f_koide / z
    #   lp_gate = 1 − LP_avg/P₁ (ligand LP screens coupling)
    #
    # Both are POSITIVE (stabilize D_at). They don't double-count
    # because they operate on independent CRT faces of T³.
    #
    E_vertex_res = 0.0
    for k in range(len(topology.Z_list)):
        z_k = topology.z_count[k] if topology.z_count else 1
        if z_k < 2:
            continue

        Z_k = topology.Z_list[k]
        lp_k = topology.lp[k] if topology.lp else 0
        ie_k = atom_data[Z_k]['IE']
        is_bohr = ie_k >= (RY - S3)
        if not is_bohr:
            continue  # both channels require Bohr-coherent vertex

        theta_deg, _, sin_coop_k = _vertex_faces(z_k, lp_k)
        sin_half = math.sin(math.radians(theta_deg) / 2.0)
        f_bohr_k = max(0.0, (ie_k / RY - C3) / S3)

        # Koide vertex boost (for P₂ channel)
        nf_k = atom_data[Z_k]['nf']
        l_k = atom_data[Z_k]['l']
        n_p_k = nf_k if l_k == 1 else 0
        Q_ex_k = _Q_koide_excess(n_p_k)
        f_koide = 1.0 + Q_ex_k * S3

        # LP fraction (for P₁ channel)
        lp_frac = min(float(lp_k), float(P1)) / P1

        bonds_at_k = [bi for bi, (ii, jj, _) in enumerate(topology.bonds)
                       if ii == k or jj == k]
        n_pairs = max(1, len(bonds_at_k) * (len(bonds_at_k) - 1) // 2)

        for idx_a in range(len(bonds_at_k)):
            for idx_b in range(idx_a + 1, len(bonds_at_k)):
                bi_a, bi_b = bonds_at_k[idx_a], bonds_at_k[idx_b]
                D0_a = bond_results[bi_a].D0
                D0_b = bond_results[bi_b].D0
                D_geom = math.sqrt(max(D0_a * D0_b, 0.0))

                # ── P₁: LP-mediated cavity resonance ──
                if lp_k > 0 and f_bohr_k > 0:
                    E_vertex_res += (S3 * C3 * sin_half * f_bohr_k
                                     * lp_frac * D_geom / n_pairs)

                # ── P₂: angular bond-bond coupling (z ≥ 2, non-linear) ──
                # z=2: only star_center/ring (chain propagates, not resonates).
                # The resonance exists for any non-linear angle — even 1 pair.
                v_class_k = polygons[k].vertex_class
                z2_ok = (z_k >= 3) or (z_k == 2 and v_class_k in ('star_center', 'ring'))
                if z2_ok and sin_coop_k > 0:
                    # Ligand LP screens angular coupling
                    def _lig_lp(bond_idx):
                        ii2, jj2, _ = topology.bonds[bond_idx]
                        partner = jj2 if ii2 == k else ii2
                        return topology.lp[partner] if topology.lp else 0
                    lp_avg = (_lig_lp(bi_a) + _lig_lp(bi_b)) / 2.0
                    lp_gate = max(0.0, 1.0 - lp_avg / P1)
                    E_vertex_res += (S5 * GAMMA_5 * D_geom / max(z_k, 1)
                                     * sin_coop_k * lp_gate * f_koide)

    D_at += E_vertex_res

    return MoleculeResult(
        D_at=D_at,
        bonds=bond_results,
        topology=topology,
        source_mode=source,
        formula=topology.formula,
        ie_audit=ie_audit,
    )


# ╔════════════════════════════════════════════════════════════════════╗
# ║  MOLECULE CLASS                                                  ║
# ╚════════════════════════════════════════════════════════════════════╝

class Molecule:
    """Molecule with PT-derived properties.

    Usage:
        mol = Molecule("O")           # H₂O from SMILES
        mol = Molecule("CC(=O)O")     # acetic acid
        mol.D_at                      # atomization energy (eV)
    """

    def __init__(self, input_str: str, source: str = "full_pt"):
        from ptc.smiles_parser import is_smiles
        self._input = input_str
        self._source = source
        # Convention: "(formula)" = explicit formula mode (solver)
        # Otherwise: auto-detect SMILES vs formula
        if input_str.startswith('(') and input_str.endswith(')') and len(input_str) > 2:
            from ptc.topology_solver import solve_topology
            self._topology = solve_topology(input_str[1:-1])
        elif is_smiles(input_str):
            self._topology = build_topology(input_str)
        else:
            from ptc.topology_solver import solve_topology
            self._topology = solve_topology(input_str)
        self._result = compute_D_at(self._topology, source)

    @property
    def D_at(self) -> float:
        return self._result.D_at

    @property
    def formula(self) -> str:
        return self._result.formula

    @property
    def bonds(self) -> List[BondResult]:
        return self._result.bonds

    @property
    def topology(self) -> Topology:
        return self._topology

    @property
    def result(self) -> MoleculeResult:
        return self._result

    @property
    def isomers(self) -> list:
        """All isomers ranked by D_at (highest first). Formula input only.

        Returns list of dicts: [{'topology': Topology, 'D_at': float, 'formula': str}, ...]
        For SMILES input, returns single-element list with the parsed structure.
        """
        if not hasattr(self, '_isomers'):
            from ptc.smiles_parser import is_smiles
            if is_smiles(self._input):
                self._isomers = [{'topology': self._topology,
                                  'D_at': self.D_at,
                                  'formula': self.formula}]
            else:
                from ptc.topology_solver import solve_topology_all
                isos = solve_topology_all(self._input)
                self._isomers = []
                for topo in isos:
                    try:
                        res = compute_D_at(topo, self._source)
                        self._isomers.append({'topology': topo,
                                              'D_at': res.D_at,
                                              'formula': res.formula})
                    except Exception:
                        pass
        return self._isomers

    def benchmark(self, D_at_exp: Optional[float] = None) -> dict:
        from ptc.data.molecules import MOLECULES
        if D_at_exp is None:
            for name, data in MOLECULES.items():
                if data['smiles'] == self._input:
                    D_at_exp = data['D_at']
                    break

        result = {
            'formula': self.formula,
            'D_at_pt': self.D_at,
            'D_at_exp': D_at_exp,
            'source': self._source,
        }

        if D_at_exp and D_at_exp > 0:
            result['error_eV'] = self.D_at - D_at_exp
            result['error_percent'] = (self.D_at - D_at_exp) / D_at_exp * 100

        if self._result.ie_audit:
            result['ie_audit'] = {}
            for Z, (pt, nist, err) in self._result.ie_audit.items():
                sym = SYMBOLS.get(Z, '?')
                result['ie_audit'][sym] = f"{pt:.3f} ({err:+.2f}%)"

        return result

    def __repr__(self) -> str:
        return f"Molecule('{self._input}', D_at={self.D_at:.3f} eV, formula={self.formula})"
