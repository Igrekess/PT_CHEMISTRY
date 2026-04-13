"""
frequencies.py -- Full vibrational frequencies via Hessian matrix.

Module 2 of the PTC universal chemistry simulator.

Computes normal-mode vibrational frequencies by:
  1. Building a molecular potential energy surface (Morse + bend + torsion)
  2. Computing the 3N x 3N Hessian by numerical finite differences
  3. Mass-weighting the Hessian
  4. Diagonalizing -> eigenvalues -> frequencies in cm^-1
  5. Projecting out 6 (or 5 for linear) zero-frequency modes

All force constants derived from s = 1/2 via PT.
0 adjustable parameters.

April 2026 -- Theorie de la Persistance
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ptc.constants import (
    S_HALF, S3, C3, P0, P1, RY,
    A_BOHR, HBAR_C_EV_A, AMU_KG, EV_J, CM1_PER_EV,
)
from ptc.data.experimental import MASS, SYMBOLS, IE_NIST
from ptc.bond import BondResult
from ptc.topology import Topology
from ptc.geometry import bond_angle_pt, bond_length_pt, period_of


# ====================================================================
# RESULT DATACLASS
# ====================================================================

@dataclass
class FrequencyResult:
    """Vibrational frequencies and normal modes from Hessian diagonalization."""
    frequencies: List[float]         # vibrational frequencies in cm^-1 (ascending)
    modes: Optional[np.ndarray]      # (n_modes, 3*N) eigenvectors, or None
    n_modes: int                     # 3N - 6 (nonlinear) or 3N - 5 (linear)
    ZPE: float                       # zero-point energy (eV)
    all_eigenvalues: Optional[List[float]] = None  # all 3N eigenvalues for debug


# ====================================================================
# INTERNAL GEOMETRY HELPERS
# ====================================================================

def _distance(x: np.ndarray, i: int, j: int) -> float:
    """Interatomic distance between atoms i and j from flat coord array."""
    ri = x[3*i:3*i+3]
    rj = x[3*j:3*j+3]
    return np.linalg.norm(ri - rj)


def _angle(x: np.ndarray, i: int, j: int, k: int) -> float:
    """Bond angle i-j-k (radians) from flat coord array. j is the center."""
    ri = x[3*i:3*i+3]
    rj = x[3*j:3*j+3]
    rk = x[3*k:3*k+3]
    v1 = ri - rj
    v2 = rk - rj
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return math.pi
    cos_a = np.dot(v1, v2) / (n1 * n2)
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.acos(cos_a)


def _dihedral(x: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    """Dihedral angle i-j-k-l (radians) from flat coord array."""
    ri = x[3*i:3*i+3]
    rj = x[3*j:3*j+3]
    rk = x[3*k:3*k+3]
    rl = x[3*l:3*l+3]
    b1 = rj - ri
    b2 = rk - rj
    b3 = rl - rk
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    nn1 = np.linalg.norm(n1)
    nn2 = np.linalg.norm(n2)
    if nn1 < 1e-12 or nn2 < 1e-12:
        return 0.0
    n1 = n1 / nn1
    n2 = n2 / nn2
    cos_d = np.dot(n1, n2)
    cos_d = max(-1.0, min(1.0, cos_d))
    return math.acos(cos_d)


# ====================================================================
# PT-DERIVED FORCE CONSTANTS
# ====================================================================

def _morse_beta(D0_eV: float, r_e_A: float, Z_A: int, Z_B: int,
                bo: float = 1.0, lp_A: int = 0, lp_B: int = 0,
                z_A: int = 1, z_B: int = 1) -> float:
    """Morse beta parameter (1/Angstrom) from PT orbital overlap decay.

    PT derivation: each atomic orbital decays as exp(-zeta * r) where
    zeta_i = sqrt(IE_i / Ry) / a_B  [Slater effective exponent from T6].

    For a covalent bond, the overlap integral decays with the average
    exponent: beta_overlap = (zeta_A + zeta_B) / 2.

    Bond order correction: multiple bonds have steeper wells.
    beta *= bo_eff^s, where s = 1/2.

    Triple bond saturation [PT: pi_perp face screening on T^3]:
    For triple bonds (bo >= 2.9), the 3rd electron pair (pi_perp)
    occupies the perpendicular face of T^3, which is screened by
    sin^2_3 from the two in-plane pairs (sigma + pi_par).

    The screening strength depends on the terminal fraction t:
      t = (number of terminal endpoints with z=1) / 2
      bo_eff = bo * (1 - t * (1 - (P0 - S3)/bo))

    Diatomic (t=1): bo_eff = P0 - S3 = 2 - sin^2_3 (full screening)
    Mixed (t=0.5):  bo_eff = bo * (1 - 0.5 * ...) (partial screening)
    Polyatomic (t=0): bo_eff = bo (no screening, adjacent bonds stabilize)

    Physical basis: terminal vertices have no adjacent bonds to
    stabilize the pi_perp face, so the screening is maximal.
    Polyatomic vertices with z >= 2 share the pi_perp face with
    adjacent bond faces, preventing the S3 screening.

    Lone pair stiffening [PT: LP face-overflow on T^3]:
    LP electrons on polyatomic vertices (z >= 2) create additional
    short-range repulsion that steepens the repulsive wall of the
    Morse potential. This is a geometric effect on T^3: LP occupy
    face area that compresses the bond face.
    beta *= (1 + n_lp_eff * S3 / P1)
    where n_lp_eff counts LP on vertices with z >= 2.

    Mutual terminal LP in period 2 [PT: compact LP head-on collision]:
    When both atoms are terminal (z=1) with LP, single bond, and
    both in period 2, the LP face each other head-on along the sigma
    axis. In period 2, LP orbitals are compact (n=2), so the Pauli
    overlap is direct — the S3 angular reduction does not apply.
    For period >= 3, LP are diffuse (larger n), and the angular
    average over the bond axis gives the S3 factor.
    Gate: z_A=z_B=1, lp_A>0, lp_B>0, bo<1.5, per_A<=2, per_B<=2.

    0 adjustable parameters. All from s = 1/2.
    """
    if r_e_A <= 0 or D0_eV <= 0:
        return 1.0

    IE_A = IE_NIST.get(Z_A, RY)
    IE_B = IE_NIST.get(Z_B, RY)

    # Orbital decay exponents (1/Angstrom)
    zeta_A = math.sqrt(IE_A / RY) / A_BOHR
    zeta_B = math.sqrt(IE_B / RY) / A_BOHR

    # Average overlap decay rate
    beta = (zeta_A + zeta_B) / 2.0

    # Bond order: multiple bonds are steeper (more electron pairs)
    # Triple bond saturation: pi_perp screened by S3 on T^3
    if bo >= 2.9:
        n_terminal = (1 if z_A == 1 else 0) + (1 if z_B == 1 else 0)
        t = n_terminal / 2.0     # terminal fraction: 0, 0.5, or 1
        reduction = 1.0 - (P0 - S3) / bo
        bo_eff = bo * (1.0 - t * reduction)
    else:
        bo_eff = bo
    beta *= bo_eff ** S_HALF

    # LP stiffening [PT: LP face-overflow on T^3]:
    # LP electrons compress the bond face, steepening the repulsive wall.
    # Polyatomic vertices (z >= 2): LP are in the molecular plane -> full effect.
    # Terminal vertices (z = 1): LP are distributed around bond axis -> reduced
    # effect by sin^2_3 (angular average of the LP projection onto bond axis).
    #
    # Exception: mutual terminal LP on single bond in period 2.
    # Compact n=2 LP face each other head-on -> full count (no S3 reduction).
    # PT: the P1 hexagonal face is fully occupied by LP on both sides,
    # and the compact orbitals have no angular averaging.
    _mutual_p2_single = (z_A == 1 and z_B == 1 and lp_A > 0 and lp_B > 0
                         and bo < 1.5
                         and period_of(Z_A) <= 2 and period_of(Z_B) <= 2)

    lp_eff = 0.0
    if z_A >= 2:
        lp_eff += lp_A
    elif lp_A > 0:
        if _mutual_p2_single:
            lp_eff += lp_A              # compact LP: full Pauli overlap
        else:
            lp_eff += lp_A * S3         # reduced for terminal atoms
    if z_B >= 2:
        lp_eff += lp_B
    elif lp_B > 0:
        if _mutual_p2_single:
            lp_eff += lp_B              # compact LP: full Pauli overlap
        else:
            lp_eff += lp_B * S3         # reduced for terminal atoms
    if lp_eff > 0:
        beta *= (1.0 + lp_eff * S3 / P1)

    return beta


def _bending_k(D0_eV: float, r_e_A: float) -> float:
    """Bending force constant (eV / rad^2).

    PT derivation: k_bend = s * D0 / (r_e * a_B)

    The bending force constant is set by the bond energy D0 and the
    electronic-nuclear area r_e * a_B. The factor s = 1/2 comes from
    the spin-1/2 projection on the angular bending face of T^3.

    a_B is the electronic length scale (angular curvature),
    r_e is the nuclear length scale (lever arm).
    """
    if r_e_A <= 0 or D0_eV <= 0:
        return 0.0
    return S_HALF * D0_eV / (r_e_A * A_BOHR)


def _torsion_V(D0_eV: float, bo: float) -> float:
    """Torsion barrier height (eV).

    PT derivation: V_n = D0 * S3^2 * C3 / P1
    Torsion is a third-order effect: two angular projections (S3^2)
    times the complementary face (C3), divided by coordination P1.
    For double bonds: barrier increases by factor P1 (restricted rotation).
    """
    V_base = D0_eV * S3 ** 2 * C3 / P1
    if bo >= 2.0:
        V_base *= P1  # restricted rotation for double bonds
    return V_base


# ====================================================================
# MOLECULAR ENERGY FUNCTION
# ====================================================================

def _build_angle_list(topo: Topology) -> List[Tuple[int, int, int]]:
    """Build list of all bond angle triples (i, center, k)."""
    n = topo.n_atoms
    # Build adjacency: atom -> list of neighbors
    neighbors = {i: [] for i in range(n)}
    for bi, (a, b, bo) in enumerate(topo.bonds):
        neighbors[a].append((b, bi))
        neighbors[b].append((a, bi))

    angles = []
    for center in range(n):
        nb_list = [nb for nb, _ in neighbors[center]]
        for ii in range(len(nb_list)):
            for jj in range(ii + 1, len(nb_list)):
                angles.append((nb_list[ii], center, nb_list[jj]))
    return angles


def _build_torsion_list(topo: Topology) -> List[Tuple[int, int, int, int, float]]:
    """Build list of all proper torsion quadruples (i, j, k, l, bo_jk)."""
    n = topo.n_atoms
    neighbors = {i: [] for i in range(n)}
    for bi, (a, b, bo) in enumerate(topo.bonds):
        neighbors[a].append((b, bo))
        neighbors[b].append((a, bo))

    torsions = []
    seen = set()
    for bi, (j, k, bo_jk) in enumerate(topo.bonds):
        # Only build torsions around bonds where both ends have z >= 2
        if topo.z_count[j] < 2 or topo.z_count[k] < 2:
            continue
        for i_atom, _ in neighbors[j]:
            if i_atom == k:
                continue
            for l_atom, _ in neighbors[k]:
                if l_atom == j or l_atom == i_atom:
                    continue
                key = (min(i_atom, l_atom), j, k, max(i_atom, l_atom))
                if key not in seen:
                    seen.add(key)
                    torsions.append((i_atom, j, k, l_atom, bo_jk))
    return torsions


def _build_energy_function(
    topo: Topology,
    bond_results: List[BondResult],
    angle_list: List[Tuple[int, int, int]],
    torsion_list: List[Tuple[int, int, int, int, float]],
    eq_angles: Dict[Tuple[int, int, int], float],
):
    """Build the total molecular potential energy function V(x).

    Returns a callable V(x) -> float, where x is a flat 3N array of
    Cartesian coordinates in Angstroms, returning energy in eV.
    """
    # Precompute per-bond Morse parameters
    bond_params = []
    for bi, (a, b, bo) in enumerate(topo.bonds):
        br = bond_results[bi]
        D0 = br.D0
        r_e = br.r_e
        Z_A = topo.Z_list[a]
        Z_B = topo.Z_list[b]
        lp_A = topo.lp[a] if a < len(topo.lp) else 0
        lp_B = topo.lp[b] if b < len(topo.lp) else 0
        z_A = topo.z_count[a] if a < len(topo.z_count) else 1
        z_B = topo.z_count[b] if b < len(topo.z_count) else 1
        beta = _morse_beta(D0, r_e, Z_A, Z_B, bo, lp_A, lp_B, z_A, z_B)
        bond_params.append((a, b, D0, r_e, beta))

    # Precompute per-angle bending parameters
    angle_params = []
    for (i, center, k) in angle_list:
        # Find average D0 and r_e for bonds at this center
        D0_sum = 0.0
        r_e_sum = 0.0
        count = 0
        for bi, (ba, bb, bo) in enumerate(topo.bonds):
            if ba == center or bb == center:
                D0_sum += bond_results[bi].D0
                r_e_sum += bond_results[bi].r_e
                count += 1
        D0_avg = D0_sum / max(count, 1)
        r_e_avg = r_e_sum / max(count, 1)
        k_bend = _bending_k(D0_avg, r_e_avg)

        theta_eq = eq_angles.get((i, center, k), math.radians(109.47))

        # Linear asymmetric correction [PT: tan^2(theta_3) holonomy]:
        # At theta_eq ~ 180 (linear center), if the two bonds forming
        # the angle have very different bond orders (e.g. H-C#N: bo=1
        # vs bo=3), the bending restoring force is dominated by the
        # weaker bond. The sigma component vanishes at linearity,
        # leaving only the pi-channel with coupling S3/C3 = tan^2_3.
        # Symmetric linear centers (CO2: bo=2, bo=2) are unaffected.
        if theta_eq > math.radians(170.0):
            # Find bo of the two specific bonds forming this angle
            bo_ic = 1.0  # bond i-center
            bo_ck = 1.0  # bond center-k
            for bi, (ba, bb, bo) in enumerate(topo.bonds):
                if (ba == i and bb == center) or (ba == center and bb == i):
                    bo_ic = bo
                if (ba == k and bb == center) or (ba == center and bb == k):
                    bo_ck = bo
            bo_min = min(bo_ic, bo_ck)
            bo_max = max(bo_ic, bo_ck)
            if bo_max > 0 and bo_min / bo_max < 0.6:
                # Asymmetric: apply tan^2(theta_3) correction
                k_bend *= S3 / C3

        angle_params.append((i, center, k, k_bend, theta_eq))

    # Precompute per-torsion parameters
    torsion_params = []
    for (i, j, k, l, bo_jk) in torsion_list:
        # Find D0 for the j-k bond
        D0_jk = 0.0
        for bi, (ba, bb, bo) in enumerate(topo.bonds):
            if (ba == j and bb == k) or (ba == k and bb == j):
                D0_jk = bond_results[bi].D0
                break
        V_n = _torsion_V(D0_jk, bo_jk)
        n_fold = 3 if bo_jk < 1.5 else 2  # 3-fold for single, 2-fold for double
        torsion_params.append((i, j, k, l, V_n, n_fold))

    def energy(x: np.ndarray) -> float:
        """Total potential energy V(x) in eV. x is flat 3N array in Angstroms."""
        V = 0.0

        # Morse stretching: V = D0 * (1 - exp(-beta*(r - r_e)))^2
        for (a, b, D0, r_e, beta) in bond_params:
            r = _distance(x, a, b)
            if r < 1e-6:
                r = 1e-6
            dr = r - r_e
            V += D0 * (1.0 - math.exp(-beta * dr)) ** 2

        # Angle bending: V = k_bend/2 * (theta - theta_eq)^2
        for (i, center, k, k_bend, theta_eq) in angle_params:
            theta = _angle(x, i, center, k)
            dtheta = theta - theta_eq
            V += 0.5 * k_bend * dtheta ** 2

        # Torsion: V = V_n/2 * (1 - cos(n*phi))
        for (i, j, k, l, V_n, n_fold) in torsion_params:
            phi = _dihedral(x, i, j, k, l)
            V += V_n / 2.0 * (1.0 - math.cos(n_fold * phi))

        return V

    return energy


# ====================================================================
# NUMERICAL HESSIAN
# ====================================================================

def compute_hessian(
    topo: Topology,
    coords: Dict[int, np.ndarray],
    bond_results: List[BondResult],
) -> np.ndarray:
    """Compute the 3N x 3N Hessian matrix by central finite differences.

    H[i,j] = (V(x+ei+ej) - V(x+ei-ej) - V(x-ei+ej) + V(x-ei-ej)) / (4*h^2)

    Parameters:
      topo : Topology object
      coords : dict atom_index -> [x, y, z] in Angstroms
      bond_results : list of BondResult per bond

    Returns:
      H : (3N, 3N) numpy array, Hessian in eV/Angstrom^2
    """
    n = topo.n_atoms
    dim = 3 * n

    # Flatten coordinates
    x0 = np.zeros(dim)
    for i in range(n):
        x0[3*i:3*i+3] = coords[i]

    # Build internal coordinate lists
    angle_list = _build_angle_list(topo)
    torsion_list = _build_torsion_list(topo)

    # Equilibrium angles from current geometry
    eq_angles = {}
    for (i, center, k) in angle_list:
        theta_eq_deg = bond_angle_pt(
            topo.z_count[center], topo.lp[center], topo.Z_list[center]
        )
        eq_angles[(i, center, k)] = math.radians(theta_eq_deg)

    # Build energy function
    V = _build_energy_function(topo, bond_results, angle_list, torsion_list, eq_angles)

    # Finite difference step (Angstroms)
    # Smaller for lighter atoms, but keep numerically stable
    h = 0.005  # 0.005 Angstroms ~ 0.01 Bohr

    # Central finite differences for Hessian
    H = np.zeros((dim, dim))

    for i in range(dim):
        for j in range(i, dim):
            if i == j:
                # Diagonal: (V(x+h) - 2*V(x) + V(x-h)) / h^2
                xp = x0.copy()
                xm = x0.copy()
                xp[i] += h
                xm[i] -= h
                H[i, i] = (V(xp) - 2.0 * V(x0) + V(xm)) / (h * h)
            else:
                # Off-diagonal: (V(x+ei+ej) - V(x+ei-ej)
                #                - V(x-ei+ej) + V(x-ei-ej)) / (4*h^2)
                xpp = x0.copy()
                xpm = x0.copy()
                xmp = x0.copy()
                xmm = x0.copy()
                xpp[i] += h; xpp[j] += h
                xpm[i] += h; xpm[j] -= h
                xmp[i] -= h; xmp[j] += h
                xmm[i] -= h; xmm[j] -= h
                H[i, j] = (V(xpp) - V(xpm) - V(xmp) + V(xmm)) / (4.0 * h * h)
                H[j, i] = H[i, j]

    return H


# ====================================================================
# MASS-WEIGHTED HESSIAN AND DIAGONALIZATION
# ====================================================================

def _is_linear(topo: Topology, coords: Dict[int, np.ndarray]) -> bool:
    """Determine if molecule is linear from actual 3D geometry.

    A molecule is linear if all atoms lie on a single line,
    i.e. the second principal moment of inertia is near zero
    (or equivalently, all bond angles are ~180 degrees).
    """
    n = topo.n_atoms
    if n <= 2:
        return True

    # Compute center of mass
    positions = np.array([coords[i] for i in range(n)])
    masses = np.array([MASS.get(topo.Z_list[i], 2.0 * topo.Z_list[i]) for i in range(n)])
    com = np.average(positions, weights=masses, axis=0)
    centered = positions - com

    # SVD to find principal axes
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # If linear, only 1 singular value is significant
    if S[0] < 1e-10:
        return True
    ratio = S[1] / S[0] if len(S) > 1 else 0.0
    return ratio < 0.02  # tolerance for near-linearity


def compute_frequencies(
    topo: Topology,
    coords: Dict[int, np.ndarray],
    bond_results: List[BondResult],
) -> FrequencyResult:
    """Compute vibrational frequencies from full Hessian diagonalization.

    Steps:
      1. Compute 3N x 3N Hessian H (eV/A^2) by finite differences
      2. Mass-weight: H_mw[i,j] = H[i,j] / sqrt(m_i * m_j)
      3. Diagonalize -> eigenvalues lambda_k
      4. freq_k = sqrt(lambda_k) / (2*pi*c) in cm^-1
      5. Separate out 6 (or 5 for linear) zero-frequency modes
      6. ZPE = sum(hbar * omega / 2)

    Parameters:
      topo : Topology object
      coords : dict atom_index -> [x, y, z] in Angstroms
      bond_results : list of BondResult per bond

    Returns:
      FrequencyResult with frequencies (cm^-1), modes, n_modes, ZPE (eV)
    """
    n = topo.n_atoms
    dim = 3 * n

    # Special case: single atom
    if n <= 1:
        return FrequencyResult(
            frequencies=[], modes=None, n_modes=0, ZPE=0.0,
        )

    # Step 1: Compute Hessian
    H = compute_hessian(topo, coords, bond_results)

    # Step 2: Mass-weight the Hessian
    # H_mw[3i+a, 3j+b] = H[3i+a, 3j+b] / sqrt(m_i * m_j)
    # Masses in amu; we need them as sqrt-inverse weights
    masses_amu = np.array([
        MASS.get(topo.Z_list[i], 2.0 * topo.Z_list[i]) for i in range(n)
    ])
    # Build mass vector of length 3N: [m0, m0, m0, m1, m1, m1, ...]
    mass_3n = np.repeat(masses_amu, 3)
    inv_sqrt_m = 1.0 / np.sqrt(mass_3n)

    H_mw = H * np.outer(inv_sqrt_m, inv_sqrt_m)

    # Step 3: Diagonalize
    eigenvalues, eigenvectors = np.linalg.eigh(H_mw)

    # Step 4: Convert eigenvalues to frequencies in cm^-1
    # eigenvalue lambda has units of eV / (amu * A^2)
    # omega = sqrt(lambda * eV_J / (amu_kg * 1e-20))  [1e-20 for A^2 -> m^2]
    # freq_cm = omega / (2 * pi * c)  where c = 2.998e10 cm/s

    conv_factor = EV_J / (AMU_KG * 1e-20)  # eV/(amu*A^2) -> s^-2
    c_cm_s = 2.998e10  # speed of light in cm/s

    all_freqs_cm = []
    for lam in eigenvalues:
        if lam > 0:
            omega = math.sqrt(lam * conv_factor)
            freq_cm = omega / (2.0 * math.pi * c_cm_s)
            all_freqs_cm.append(freq_cm)
        elif lam < -1e-6 * abs(eigenvalues[-1]):
            # Small negative eigenvalue -> imaginary frequency (report as negative)
            omega = math.sqrt(abs(lam) * conv_factor)
            freq_cm = -omega / (2.0 * math.pi * c_cm_s)
            all_freqs_cm.append(freq_cm)
        else:
            all_freqs_cm.append(0.0)

    # Step 5: Separate out translation + rotation modes
    linear = _is_linear(topo, coords)
    n_tr_rot = 5 if linear else 6
    if n <= 1:
        n_tr_rot = 3  # single atom: 3 translations only
    elif n == 2:
        n_tr_rot = 5  # diatomic: always linear

    n_modes = max(dim - n_tr_rot, 0)

    # Sort all frequencies by magnitude (ascending)
    freq_idx = sorted(range(dim), key=lambda k: abs(all_freqs_cm[k]))

    # The n_tr_rot smallest-magnitude eigenvalues are translations + rotations
    # The rest are vibrational modes
    vib_indices = freq_idx[n_tr_rot:]
    vib_freqs = sorted([all_freqs_cm[k] for k in vib_indices])

    # Filter: only positive frequencies for ZPE
    # (imaginary frequencies indicate saddle points, not equilibrium)
    positive_freqs = [f for f in vib_freqs if f > 0]

    # Step 6: ZPE = sum(h*nu/2) where h*nu = freq_cm / CM1_PER_EV
    zpe = sum(f / CM1_PER_EV / 2.0 for f in positive_freqs)

    # Extract mode vectors (transform back to Cartesian)
    # mode_k = eigvec_k / sqrt(m) (mass-weighted -> Cartesian displacement)
    modes = np.zeros((len(vib_indices), dim))
    for mi, vi in enumerate(sorted(vib_indices, key=lambda k: all_freqs_cm[k])):
        modes[mi] = eigenvectors[:, vi] * inv_sqrt_m

    return FrequencyResult(
        frequencies=vib_freqs,
        modes=modes,
        n_modes=n_modes,
        ZPE=zpe,
        all_eigenvalues=sorted(all_freqs_cm),
    )
