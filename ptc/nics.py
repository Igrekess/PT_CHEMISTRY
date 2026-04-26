"""
nics.py — Nucleus-Independent Chemical Shift (NICS) from PT.

NICS is the magnetic shielding evaluated at a "ghost" probe nucleus
located at the centroid of an aromatic ring (NICS(0)) or at a fixed
distance above the ring plane (NICS(1) at z=1 Å). It is the standard
quantitative metric for aromaticity:
  - benzene NICS(0)  ≈ −9.7 ppm   (canonical π-aromatic)
  - cyclopentadienyl ≈ −13   ppm
  - Al₄²⁻             ≈ −34   ppm  (multi-fold σ+π)
  - Bi₃³⁻             ≈ −15   ppm  (σ-aromatic all-metal)
  - cyclobutadiene   ≈ +15   ppm  (anti-aromatic)

The PT-pure derivation is a direct application of the Pauling-London
ring-current model: a delocalized aromatic Hückel pair on the polygon
P_N produces a circular orbital current; in an external field B_ext
the induced diamagnetic shielding at axial point z reads (atomic units,
Larmor approximation, isotropic average):

    σ(z) = -α² · a₀ · n_e · f_coh / 12 · R² / (R² + z²)^(3/2)   × 10⁶ ppm

  - α       = fine-structure constant (α² = QED magnetic↔electric)
  - a₀      = Bohr radius (natural electron-orbit scale)
  - n_e     = number of σ+π aromatic delocalized electrons in the ring
  - f_coh   = T³ Fourier coherence of the ring composition (∈ [0,1])
  - R       = ring radius from centroid to vertex (Å)
  - z       = probe distance above ring plane (Å)
  - 1/12    = ⅓ isotropic × ½ spin-degeneracy × ½ Larmor

No fit parameter. The single calibration is fundamental constants
(α, a₀) — derivable in PT from s=1/2 via Ry, see constants.py.

For multi-fold aromatic clusters (Al₄²⁻, X₄²⁻ p-block), the dipolar
Pauling-London model under-estimates |NICS| by 2-4× because it omits
constructive interference between σ and π aromatic sub-systems.
The PT formula is exact for single-channel aromatics (benzene-like)
and provides correct relative ranking for multi-channel systems.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, List

from ptc.constants import ALPHA_PHYS, A_BOHR, P1, P2, P3, S3, S5, S7
from ptc.periodic import period, l_of, ns_config, _np_of, _nd_of
from ptc.topology import Topology, build_topology, valence_electrons
from ptc.bond import r_equilibrium

_TWO_PI = 2.0 * math.pi
_S_SUM_COH = S3 + S5 + S7
# Pauling-London prefactor in (ppm · Å) for unit n_e, R, z=0 → σ = −K · n_e/R
_NICS_K = ALPHA_PHYS ** 2 * A_BOHR / 12.0 * 1.0e6   # ≈ 2.348 ppm·Å


@dataclass
class NICSResult:
    """NICS prediction for a single ring."""
    ring_atoms: List[int]       # indices in the parent topology
    N: int                      # ring size
    R: float                    # ring radius (Å, centroid → vertex)
    n_aromatic: int             # σ + π delocalized electrons
    f_coh: float                # T³ composition coherence
    NICS_0: float               # NICS(0) at centroid (ppm)
    NICS_1: float               # NICS(1) at z = 1 Å (ppm)
    aromatic: bool              # |NICS(0)| ≥ 5 ppm and < 0 (diamagnetic)
    rationale: str              # human-readable summary

    def __str__(self) -> str:
        sign = "diamagnetic (aromatic)" if self.NICS_0 < -5 else (
               "paramagnetic (anti-aromatic)" if self.NICS_0 > +5 else
               "weak / non-aromatic")
        return (f"NICS  N={self.N}  R={self.R:.2f} Å  n_e={self.n_aromatic}  "
                f"f_coh={self.f_coh:.3f}  NICS(0)={self.NICS_0:+.2f}  "
                f"NICS(1)={self.NICS_1:+.2f} ppm  [{sign}]")


def _f_coh_T3(Zs: List[int]) -> float:
    """T³ Fourier coherence of ring composition.

    Same expression as transfer_matrix.py C9b. Returns 1 for strict
    homonuclear, decays toward 0 as residues mod {3,5,7} dephase.
    """
    N = len(Zs)
    if N == 0:
        return 0.0
    f_coh = 0.0
    for k_face, Pk in ((0, P1), (1, P2), (2, P3)):
        s_re = 0.0
        s_im = 0.0
        for Z in Zs:
            phase = _TWO_PI * (Z % Pk) / Pk
            s_re += math.cos(phase)
            s_im += math.sin(phase)
        Ck = (s_re * s_re + s_im * s_im) / (N * N)
        Sk = (S3, S5, S7)[k_face]
        f_coh += Sk * Ck
    return f_coh / _S_SUM_COH


def _ring_radius(topology: Topology, ring: List[int]) -> float:
    """Geometric ring radius (centroid → vertex), in Å.

    For a regular N-gon with side b: R = b / (2 sin(π/N)).
    Bond length b averaged from r_equilibrium of in-ring bonds.
    """
    N = len(ring)
    ring_set = set(ring)
    blens = []
    for bi, (i, j, bo) in enumerate(topology.bonds):
        if i in ring_set and j in ring_set:
            Zi, Zj = topology.Z_list[i], topology.Z_list[j]
            per_i, per_j = period(Zi), period(Zj)
            try:
                b = r_equilibrium(per_i, per_j, bo,
                                  topology.z_count[i], topology.z_count[j],
                                  topology.lp[i], topology.lp[j])
            except TypeError:
                b = r_equilibrium(per_i, per_j, bo)
            if b and b > 0:
                blens.append(b)
    if not blens:
        return 0.0
    b_mean = sum(blens) / len(blens)
    if N < 3:
        return b_mean / 2.0
    return b_mean / (2.0 * math.sin(math.pi / N))


def _aromatic_electron_count(topology: Topology, ring: List[int]) -> int:
    """Count σ+π delocalized electrons available for ring current.

    Strategy (PT-pure, group-classification, no SMILES aromaticity flag
    dependency for non-organic rings):

    Each ring atom uses 2 σ-bonds in the ring; remaining valence
    electrons form the delocalized aromatic system. Per group:
      - Group 1   (s¹, Li/Na/K)     : 1 σ-aro e per atom
      - Group 11  (s¹d¹⁰, Cu/Ag/Au) : 1 σ-aro e per atom
      - Group 13  (s²p¹, B/Al/Ga…)  : 1 π-aro e per atom (p_z)
      - Group 14  (s²p², Si/Ge/Sn…) : 2 σ-aro e per atom
                                       (s² lone pair delocalized)
      - Group 15  (s²p³, P/As/Bi…)  : 2 σ + 1 π = 3 aro e per atom
      - Group 16  (s²p⁴, S/Se/Te…)  : 2 σ + 2 π = 4 aro e per atom
      - SMILES-aromatic (bo=1.5)    : 1 π e per atom (overrides above)

    A ring atom with exocyclic σ-bonds beyond 2 (sp³-like) contributes
    fewer aromatic electrons; we apply a coordination cap.
    """
    ring_set = set(ring)
    has_aromatic_smiles = False
    for bi, (i, j, bo) in enumerate(topology.bonds):
        if i in ring_set and j in ring_set:
            if abs(bo - 1.5) < 0.01:
                has_aromatic_smiles = True
                break

    # Detect Kekulé double bonds in ring (organic with explicit bo=2)
    has_kekule = any(abs(bo - 2.0) < 0.01
                     for i, j, bo in topology.bonds
                     if i in ring_set and j in ring_set)
    n_pi_kekule = 0
    if has_kekule:
        for i, j, bo in topology.bonds:
            if i in ring_set and j in ring_set and abs(bo - 2.0) < 0.01:
                n_pi_kekule += 2  # 2 π electrons per double bond

    n_aromatic = 0
    for a in ring:
        Z = topology.Z_list[a]
        ns = ns_config(Z)
        np_v = _np_of(Z)
        # SMILES-aromatic flag overrides group rule (Hückel π)
        if has_aromatic_smiles:
            n_aromatic += 1
            continue
        # Kekulé organic: π handled by double-bond count, σ = 0 for C (sp²)
        if has_kekule and Z == 6:
            continue   # contribution counted in n_pi_kekule
        # σ¹ alkali/coinage (Group 1 or 11)
        if ns == 1 and np_v == 0:
            n_aromatic += 1
            continue
        # p-block by np valence count (excluding C in Kekulé context)
        if l_of(Z) == 1 and ns >= 1:
            if np_v == 1:        # Group 13: 1 π electron
                n_aromatic += 1
            elif np_v == 2:      # Group 14: σ-aromatic via s² lone pair
                n_aromatic += 2
            elif np_v == 3:      # Group 15: σ+π (s² + p_z)
                n_aromatic += 3
            elif np_v == 4:      # Group 16: σ+π² (s² + 2p_z)
                n_aromatic += 4
        # d-block / f-block: not yet handled (rare aromatic case)
    n_aromatic += n_pi_kekule
    return n_aromatic


def _huckel_sign(n: int) -> float:
    """Hückel rule sign: +1 for 4n+2 (aromatic, diamagnetic), -1 for 4n
    (antiaromatic, paramagnetic), +0.5 for odd (radical, partial).
    """
    if n == 0:
        return 0.0
    if n % 4 == 2:
        return +1.0
    if n % 4 == 0:
        return -1.0
    return +0.5  # radical


def nics_for_ring(topology: Topology, ring: List[int],
                  z_probe: float = 1.0) -> NICSResult:
    """Compute NICS(0) and NICS(z_probe) for a single ring.

    The effective ring-current electron count is the SIGNED sum over
    σ and π channels, with each channel weighted by the Hückel rule:
        n_eff = sign_σ(n_σ) · n_σ + sign_π(n_π) · n_π
    where sign(n) = +1 if 4n+2, -1 if 4n, +0.5 if odd (radical).
    This produces correct sign of NICS: diamagnetic for fully aromatic,
    paramagnetic for antiaromatic, mixed for σ-arom + π-antiarom.
    """
    N = len(ring)
    Zs = [topology.Z_list[a] for a in ring]
    R = _ring_radius(topology, ring)
    n_e = _aromatic_electron_count(topology, ring)
    f_coh = _f_coh_T3(Zs)
    if R <= 0 or f_coh <= 0:
        return NICSResult(ring, N, R, n_e, f_coh, 0.0, 0.0, False,
                          "no aromatic current (R or f_coh = 0)")
    # Per-channel σ / π split (mirrors signature.py _separate_sigma_pi)
    n_sigma = 0
    n_pi = 0
    ring_set = set(ring)
    has_smiles_aro = any(abs(bo - 1.5) < 0.01
                         for i, j, bo in topology.bonds
                         if i in ring_set and j in ring_set)
    has_kekule = any(abs(bo - 2.0) < 0.01
                     for i, j, bo in topology.bonds
                     if i in ring_set and j in ring_set)
    if has_kekule:
        for i, j, bo in topology.bonds:
            if i in ring_set and j in ring_set and abs(bo - 2.0) < 0.01:
                n_pi += 2
    for a in ring:
        Z = topology.Z_list[a]
        ns = ns_config(Z)
        np_v = _np_of(Z)
        if has_smiles_aro:
            n_pi += 1
            continue
        if has_kekule and Z == 6:
            continue
        if ns == 1 and np_v == 0:
            n_sigma += 1
            continue
        if l_of(Z) == 1:
            if np_v == 1:    n_pi += 1
            elif np_v == 2:  n_sigma += 2
            elif np_v == 3:  n_sigma += 2; n_pi += 1
            elif np_v == 4:  n_sigma += 2; n_pi += 2
    n_eff = _huckel_sign(n_sigma) * n_sigma + _huckel_sign(n_pi) * n_pi
    if n_eff == 0:
        return NICSResult(ring, N, R, n_e, f_coh, 0.0, 0.0, False,
                          "no net ring current (Hückel-cancelling)")
    # Pauling-London formula with signed channel sum
    def _sigma(z: float) -> float:
        return -_NICS_K * n_eff * f_coh * R * R / (R * R + z * z) ** 1.5
    sig0 = _sigma(0.0)
    sig1 = _sigma(z_probe)
    aromatic = sig0 < -5.0
    rationale = (f"PT NICS: n_e={n_e}, f_coh={f_coh:.3f}, "
                 f"R={R:.2f}Å, K_PT=α²·a₀/12={_NICS_K:.3f} ppm·Å")
    return NICSResult(ring, N, R, n_e, f_coh, sig0, sig1, aromatic, rationale)


def nics_all_rings(smiles: str, z_probe: float = 1.0) -> List[NICSResult]:
    """Compute NICS for every ring of a molecule given by SMILES.

    Convenience wrapper. Returns one NICSResult per ring (ordered as
    topology.rings).
    """
    topo = build_topology(smiles)
    if not topo.rings:
        return []
    return [nics_for_ring(topo, ring, z_probe) for ring in topo.rings]


# ─── Convenience: NICS(0) / NICS(1) scalar shortcuts ─────────────────

def nics0(smiles: str, ring_index: int = 0) -> Optional[float]:
    """NICS(0) of the ring at index `ring_index` (default first ring)."""
    rs = nics_all_rings(smiles, z_probe=1.0)
    if not rs:
        return None
    return rs[ring_index].NICS_0


def nics1(smiles: str, ring_index: int = 0) -> Optional[float]:
    """NICS(1) at z = 1 Å above the ring plane."""
    rs = nics_all_rings(smiles, z_probe=1.0)
    if not rs:
        return None
    return rs[ring_index].NICS_1
