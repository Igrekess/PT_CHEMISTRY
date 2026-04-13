"""
geometry.py -- Molecular 3D geometry from the sieve.

All derived from s = 1/2. Zero adjustable parameters.

Bond angles : VSEPR = max entropy GFT on S^2 (Fisher metric)
  cos(theta) = -1/(z_eff - 1)  with LP compression via gamma_7

3D placement : analytical VSEPR directions per vertex,
  recursive placement for 2nd-level neighbors.

Export : XYZ format for Avogadro/VESTA/py3Dmol.

April 2026 -- Theorie de la Persistance
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np

from ptc.constants import (
    S_HALF, P1, GAMMA_5, GAMMA_7, S3, C3, A_BOHR, RY, D3,
)
from ptc.data.experimental import SYMBOLS, MASS, IE_NIST


# ====================================================================
# PERIOD TABLE (derived from Z)
# ====================================================================

_PERIOD_BOUNDS = [0, 2, 10, 18, 36, 54, 86, 118]


def period_of(Z: int) -> int:
    """Period of element Z (1-7)."""
    for per, bound in enumerate(_PERIOD_BOUNDS[1:], 1):
        if Z <= bound:
            return per
    return 7


# ====================================================================
# BOND ANGLES (VSEPR from Fisher metric, P1+P4+P7)
# ====================================================================

def bond_angle_pt(z_bonds: int, n_lp: int, Z_central: int = 0) -> float:
    """Bond angle from Fisher metric (degrees).

    Base formula (Fisher):
      cos(theta) = -1 / (z_eff - 1)
    where z_eff = z_bonds + n_lp * w_lp.

    LP domain weighting: w_lp = 1 + s = 3/2.
    PT derivation: a bond domain constrains both atoms (occupation=1).
    An LP domain constrains only the host atom, gaining extra s = 1/2
    angular freedom, so occupation = 1 + s = 3/2.

    Period >= 3 with LP: hybridization mixing
      theta = f * theta_Fisher + (1-f) * 90
      f = s^(per-1): less hybridization at higher periods.

    Examples:
      CH4 (4,0) -> 109.47  [cos = -1/3 = -1/P1, algebraic identity]
      H2O (2,2) -> 104.48  [w_lp = 3/2 -> z_eff = 5, cos = -1/4]
      NH3 (3,1) -> 106.61  [pyramidal]
      H2S (2,2,per=3) -> ~93.6 [mixing with 90 p-pure]
    """
    if z_bonds <= 0:
        return 180.0

    # LP domain weight: 1 + s = 3/2 [PT: LP has extra angular freedom s]
    w_lp = 1.0 + S_HALF  # 1.5

    # Effective steric number
    z_eff = z_bonds + n_lp * w_lp

    if z_eff <= 1.0:
        return 180.0

    # Fisher metric angle
    if z_eff >= 6.0 and n_lp == 0:
        cos_theta = 0.0  # octahedral
    else:
        cos_theta = -1.0 / (z_eff - 1.0)

    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta_fisher = math.degrees(math.acos(cos_theta))

    # Period >= 3 with LP: mix Fisher with 90 (p-pure orbital limit)
    per = period_of(Z_central) if Z_central > 0 else 2
    if per >= 3 and n_lp >= 1:
        f_hybrid = S_HALF ** (per - 1)  # 0.25 for per=3, 0.125 for per=4
        return f_hybrid * theta_fisher + (1.0 - f_hybrid) * 90.0

    return theta_fisher


def classify_geometry(n_bonds: int, n_lp: int) -> str:
    """Classify VSEPR geometry from (n_bonds, n_lp)."""
    n_domains = n_bonds + n_lp
    if n_domains <= 1:
        return "atomic"
    if n_domains == 2:
        return "linear"
    if n_domains == 3:
        if n_lp == 0:
            return "trigonal_planar"
        return "bent"
    if n_domains == 4:
        if n_lp == 0:
            return "tetrahedral"
        if n_lp == 1:
            return "pyramidal"
        return "bent"
    if n_domains == 5:
        if n_lp == 0:
            return "trigonal_bipyramidal"
        if n_lp == 1:
            return "seesaw"
        if n_lp == 2:
            return "T-shaped"
        return "linear"
    if n_domains == 6:
        if n_lp == 0:
            return "octahedral"
        if n_lp == 1:
            return "square_pyramidal"
        return "square_planar"
    return f"complex_{n_domains}"


# ====================================================================
# BOND LENGTHS (atomic radii from IE cascade, RC25)
# ====================================================================

def _atomic_radius(Z: int) -> float:
    """Covalent radius from PT cascade (Angstroms).

    r(Z) = a_B * n_eff^alpha / Z_eff^beta
    alpha = 1 + s * gamma_5 (orbital exponent, P7)
    Z_eff = n_eff * sqrt(IE/Ry)
    Special: H -> a_B * s * (1 + sin^2_3)
    """
    per = period_of(Z)
    ie = IE_NIST.get(Z, RY)

    if Z == 1:
        return A_BOHR * S_HALF * (1.0 + S3)

    if per <= 3:
        n_eff = float(per)
    else:
        n_eff = per - S_HALF * (1.0 - GAMMA_7)

    Z_eff = n_eff * math.sqrt(ie / RY)
    Z_eff = max(Z_eff, 0.5)

    alpha_r = 1.0 + S_HALF * GAMMA_5
    return A_BOHR * n_eff ** alpha_r / Z_eff


def bond_length_pt(Z_A: int, Z_B: int, n_bond: float = 1.0,
                   lp_A: int = 0, lp_B: int = 0) -> float:
    """Bond length from PT additive radii with pi contraction (Angstroms).

    d(A-B) = r(A) + r(B) - contraction(n_bond) + elongation_LP

    Contraction: (n-1) * sin^2_3 * s * d_sigma / 2

    LP-LP inter-atomic repulsion [P1 channel, single bonds only]:
    When both atoms carry lone pairs, the LP on atom A face those on
    atom B across the bond axis, creating Pauli repulsion that elongates
    the bond. Each facing LP pair (n_LP_min = min(lp_A, lp_B)) pushes
    the nuclei apart by a_B * S3 * per_avg / P1.

    PT derivation: each LP occupies 1/P1 of the hexagonal face on Z/P1Z.
    The Pauli repulsion per LP pair is proportional to sin^2_3 (holonomy
    on the P1 face). The orbital extent scales with per_avg (average
    period), giving the per_avg/P1 prefactor.

    Gate: single bonds only (bo < 1.5). For multiple bonds, the pi
    electrons occupy the inter-nuclear region on T^3, fully screening
    the LP-LP repulsion across the bond.
    """
    r_A = _atomic_radius(Z_A)
    r_B = _atomic_radius(Z_B)
    d_sigma = r_A + r_B
    contraction = (n_bond - 1) * S3 * S_HALF * d_sigma / 2.0
    d = d_sigma - contraction

    # LP-LP inter-atomic repulsion [single bonds only]
    n_LP_min = min(lp_A, lp_B)
    if n_bond < 1.5 and n_LP_min > 0:
        per_A = period_of(Z_A)
        per_B = period_of(Z_B)
        per_avg = (per_A + per_B) / 2.0
        d += A_BOHR * n_LP_min * S3 * per_avg / P1

    return max(d, 0.3)


def torsion_angle_pt(lp_A: float, lp_B: float, n_bond: float = 1.0) -> float:
    """Torsion angle from LP-LP repulsion (degrees).

    No LP -> staggered 60 = pi/P1.
    LP > 0 -> gauche via cos^2_7 attenuation.
    """
    lp_min = min(lp_A, lp_B)
    if lp_min <= 0:
        return 60.0
    cos2_7 = 1.0 - (1.0 - (1.0 - (1.0 - 13.0/15.0)**7)/7) * ((1.0 - (1.0 - 13.0/15.0)**7)/7)
    # More directly: cos^2(theta_7) for q_stat
    from ptc.constants import C7
    return 120.0 * C7 ** (lp_min * S_HALF)


# ====================================================================
# 3D PLACEMENT ENGINE
# ====================================================================

# Standard tetrahedral directions
_TETRA_DIRS = np.array([
    [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
], dtype=float)
_TETRA_DIRS /= np.linalg.norm(_TETRA_DIRS[0])


def _place_neighbors(n_bonds: int, n_lp: int, period: int,
                     distances: List[float],
                     Z_central: int = 0) -> List[np.ndarray]:
    """Place n_bonds neighbors around origin using VSEPR geometry.

    Returns list of position vectors (np.array shape (3,)).
    LP are not placed (no atom) but influence angles.
    """
    n_domains = n_bonds + n_lp
    positions = []

    if n_bonds == 0:
        return positions

    if n_bonds == 1:
        positions.append(np.array([0.0, 0.0, distances[0]]))

    elif n_domains == 2:
        # Linear
        positions.append(np.array([0.0, 0.0, distances[0]]))
        if n_bonds >= 2:
            positions.append(np.array([0.0, 0.0, -distances[1]]))

    elif n_domains == 3:
        if n_lp == 0:
            # Trigonal planar
            for i in range(n_bonds):
                theta = i * 2.0 * math.pi / 3.0
                d = distances[min(i, len(distances) - 1)]
                positions.append(d * np.array([math.sin(theta), 0.0, math.cos(theta)]))
        elif n_lp == 1 and n_bonds == 2:
            # Bent (like SO2)
            angle_rad = math.radians(bond_angle_pt(n_bonds, n_lp, Z_central))
            half = angle_rad / 2.0
            positions.append(distances[0] * np.array([math.sin(half), 0.0, math.cos(half)]))
            positions.append(distances[1] * np.array([-math.sin(half), 0.0, math.cos(half)]))
        else:
            positions.append(np.array([0.0, 0.0, distances[0]]))

    elif n_domains == 4:
        if n_lp == 0:
            # Tetrahedral
            for i in range(n_bonds):
                d = distances[min(i, len(distances) - 1)]
                positions.append(d * _TETRA_DIRS[i])
        elif n_lp == 1:
            # Pyramidal (NH3-like): 3 bonds on a cone
            angle_rad = math.radians(bond_angle_pt(n_bonds, n_lp, Z_central))
            cos_theta = math.cos(angle_rad)
            cos2_alpha = (2.0 * cos_theta + 1.0) / 3.0
            cos2_alpha = max(0.0, min(1.0, cos2_alpha))
            alpha = math.acos(math.sqrt(cos2_alpha))
            for i in range(min(n_bonds, 3)):
                phi = i * 2.0 * math.pi / 3.0
                d = distances[min(i, len(distances) - 1)]
                positions.append(d * np.array([
                    math.sin(alpha) * math.cos(phi),
                    math.sin(alpha) * math.sin(phi),
                    math.cos(alpha),
                ]))
        elif n_lp == 2:
            # Bent (H2O-like)
            angle_rad = math.radians(bond_angle_pt(n_bonds, n_lp, Z_central))
            half = angle_rad / 2.0
            positions.append(distances[0] * np.array([math.sin(half), 0.0, math.cos(half)]))
            positions.append(distances[1] * np.array([-math.sin(half), 0.0, math.cos(half)]))
        else:
            positions.append(np.array([0.0, 0.0, distances[0]]))

    elif n_domains == 5:
        # Trigonal bipyramidal
        eq_count = min(n_bonds, 3)
        for i in range(eq_count):
            theta = i * 2.0 * math.pi / 3.0
            d = distances[min(i, len(distances) - 1)]
            positions.append(d * np.array([math.cos(theta), math.sin(theta), 0.0]))
        for i in range(n_bonds - eq_count):
            sign = 1.0 if i == 0 else -1.0
            idx = eq_count + i
            d = distances[min(idx, len(distances) - 1)]
            positions.append(d * np.array([0.0, 0.0, sign]))

    elif n_domains == 6:
        # Octahedral
        dirs = [
            np.array([1, 0, 0], dtype=float), np.array([-1, 0, 0], dtype=float),
            np.array([0, 1, 0], dtype=float), np.array([0, -1, 0], dtype=float),
            np.array([0, 0, 1], dtype=float), np.array([0, 0, -1], dtype=float),
        ]
        for i in range(n_bonds):
            if i < len(dirs):
                d = distances[min(i, len(distances) - 1)]
                positions.append(d * dirs[i])

    else:
        # Fallback: golden-angle distribution on sphere
        for i in range(n_bonds):
            phi = math.pi * (1 + 5**0.5) * i
            theta = math.acos(1 - 2 * (i + 0.5) / n_bonds)
            d = distances[min(i, len(distances) - 1)]
            positions.append(d * np.array([
                math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi),
                math.cos(theta),
            ]))

    return positions


# ====================================================================
# RESULT DATACLASS
# ====================================================================

@dataclass
class Geometry3D:
    """3D molecular geometry result."""
    coords: Dict[int, np.ndarray]         # atom_index -> [x, y, z] (Angstroms)
    Z_list: List[int]                      # atomic numbers
    angles: List[Dict]                     # {'atoms': (i,j,k), 'angle': degrees}
    lengths: List[Dict]                    # {'atoms': (i,j), 'length': A, 'order': n}
    torsions: List[Dict]                   # {'atoms': (i,j,k,l), 'angle': degrees}
    vsepr_class: Dict[int, str] = field(default_factory=dict)  # atom -> geometry name

    @property
    def xyz(self) -> str:
        """Export as XYZ format string."""
        return to_xyz(self.coords, self.Z_list)

    @property
    def n_atoms(self) -> int:
        return len(self.Z_list)


# ====================================================================
# MAIN ENTRY: Topology -> 3D Geometry
# ====================================================================

def _place_ring(ring_atoms: List[int], topo, bond_lengths: dict,
                neighbors: dict) -> Dict[int, np.ndarray]:
    """Place ring atoms as a regular polygon in the xy-plane.

    Returns dict of atom_index -> np.array([x, y, z]).
    Ring is centered at origin, atoms equally spaced on circle.
    """
    n_ring = len(ring_atoms)
    if n_ring < 3:
        return {}

    # Average bond length in the ring
    edge_index = {}
    for bi, (a, b, bo) in enumerate(topo.bonds):
        edge_index[(a, b)] = bi
        edge_index[(b, a)] = bi

    ring_bond_len = []
    for k in range(n_ring):
        a, b = ring_atoms[k], ring_atoms[(k + 1) % n_ring]
        bi = edge_index.get((a, b))
        if bi is not None and bi in bond_lengths:
            ring_bond_len.append(bond_lengths[bi])
    avg_d = sum(ring_bond_len) / max(len(ring_bond_len), 1) if ring_bond_len else 1.4

    # Radius of regular polygon: R = d / (2 * sin(pi/n))
    R = avg_d / (2.0 * math.sin(math.pi / n_ring))

    coords = {}
    for k, atom_idx in enumerate(ring_atoms):
        angle = 2.0 * math.pi * k / n_ring
        coords[atom_idx] = np.array([R * math.cos(angle), R * math.sin(angle), 0.0])

    return coords


def _place_ring_substituents(atom_idx: int, ring_center: np.ndarray,
                             center_coord: np.ndarray,
                             sub_data: List[Tuple[int, float]],
                             z_count: int, n_lp: int) -> Dict[int, np.ndarray]:
    """Place substituent atoms on a ring atom.

    For sp2 (aromatic, z=3 in ring): one substituent, radially outward.
    For sp3 (saturated, z=4 in ring): two substituents, above/below plane + radial.

    sub_data: list of (neighbor_idx, distance)
    Returns dict {neighbor_idx: coord}.
    """
    radial = center_coord - ring_center
    norm_r = np.linalg.norm(radial)
    if norm_r > 1e-10:
        radial_dir = radial / norm_r
    else:
        radial_dir = np.array([1.0, 0.0, 0.0])

    # Normal to ring plane = z-axis (ring is in xy-plane)
    up = np.array([0.0, 0.0, 1.0])

    result = {}
    n_sub = len(sub_data)

    if n_sub == 1:
        # sp2: single substituent radially outward (in-plane)
        nb_idx, d = sub_data[0]
        result[nb_idx] = center_coord + d * radial_dir

    elif n_sub == 2:
        # sp3: two substituents — tetrahedral-like, above/below + radial
        # Tetrahedral angle from ring plane: ~54.75 deg (arccos(1/3) / 2 ≈ 35.26 from z)
        # Direction = radial * cos(35.26) ± up * sin(35.26)
        tilt = math.radians(35.26)
        for i, (nb_idx, d) in enumerate(sub_data):
            sign = 1.0 if i == 0 else -1.0
            direction = math.cos(tilt) * radial_dir + sign * math.sin(tilt) * up
            direction /= np.linalg.norm(direction)
            result[nb_idx] = center_coord + d * direction

    elif n_sub == 3:
        # Rare: 3 substituents on ring atom (e.g., N+ in ring)
        tilt = math.radians(35.26)
        for i, (nb_idx, d) in enumerate(sub_data):
            if i == 0:
                direction = radial_dir
            else:
                sign = 1.0 if i == 1 else -1.0
                direction = 0.5 * radial_dir + sign * math.sin(tilt) * up
                direction /= np.linalg.norm(direction)
            result[nb_idx] = center_coord + d * direction

    else:
        # Fallback: spread evenly around radial direction
        for i, (nb_idx, d) in enumerate(sub_data):
            angle = i * 2.0 * math.pi / max(n_sub, 1)
            direction = math.cos(angle) * radial_dir + math.sin(angle) * up
            direction /= np.linalg.norm(direction)
            result[nb_idx] = center_coord + d * direction

    return result


def compute_geometry_3d(topo, bond_results=None) -> Geometry3D:
    """Compute 3D molecular geometry from Topology.

    Strategy:
      1. If rings exist: place ring atoms as regular polygon, then BFS outward
      2. If no rings: root-based BFS with VSEPR placement

    Parameters:
      topo : ptc.topology.Topology (finalized)
      bond_results : optional list of ptc.bond.BondResult (for r_e values)

    Returns:
      Geometry3D with coordinates, angles, lengths, torsions.
    """
    n = len(topo.Z_list)

    # Build adjacency: atom -> [(neighbor, bond_order, bond_index)]
    neighbors: Dict[int, List[Tuple[int, float, int]]] = {i: [] for i in range(n)}
    for bi, (a, b, bo) in enumerate(topo.bonds):
        neighbors[a].append((b, bo, bi))
        neighbors[b].append((a, bo, bi))

    # Get bond lengths: prefer BondResult.r_e if available, else compute
    bond_lengths = {}
    for bi, (a, b, bo) in enumerate(topo.bonds):
        if bond_results and bi < len(bond_results) and bond_results[bi] is not None:
            bond_lengths[bi] = bond_results[bi].r_e
        else:
            bond_lengths[bi] = bond_length_pt(
                topo.Z_list[a], topo.Z_list[b], bo,
                lp_A=topo.lp[a] if a < len(topo.lp) else 0,
                lp_B=topo.lp[b] if b < len(topo.lp) else 0,
            )

    # Coordinates array
    coords: Dict[int, np.ndarray] = {i: np.zeros(3) for i in range(n)}
    placed = set()

    # VSEPR classification per atom
    vsepr_class = {}
    for i in range(n):
        z = topo.z_count[i]
        lp = topo.lp[i]
        vsepr_class[i] = classify_geometry(z, lp)

    # ── PHASE 1: Place ring atoms as regular polygons ──
    if topo.rings:
        # Use the largest ring
        largest_ring = max(topo.rings, key=len)
        ring_coords = _place_ring(largest_ring, topo, bond_lengths, neighbors)
        ring_center = np.mean(list(ring_coords.values()), axis=0)

        for atom_idx, coord in ring_coords.items():
            coords[atom_idx] = coord
            placed.add(atom_idx)

        # Place substituents on ring atoms (H, OH, CH3, etc.)
        for atom_idx in largest_ring:
            sub_data = []
            for nb, bo, bi in neighbors[atom_idx]:
                if nb not in placed and nb not in topo.ring_atoms:
                    d = bond_lengths.get(bi, 1.0)
                    sub_data.append((nb, d))
            if sub_data:
                sub_coords = _place_ring_substituents(
                    atom_idx, ring_center, coords[atom_idx],
                    sub_data, topo.z_count[atom_idx], topo.lp[atom_idx])
                for nb_idx, coord in sub_coords.items():
                    coords[nb_idx] = coord
                    placed.add(nb_idx)

    # ── PHASE 2: BFS for remaining atoms ──
    if len(placed) < n:
        # Find root among unplaced (highest coordination)
        unplaced = [i for i in range(n) if i not in placed]
        if placed:
            # Start BFS from placed atoms (ring neighbors)
            queue = [i for i in placed]
        else:
            # No rings: pick root
            root = max(unplaced, key=lambda i: (topo.z_count[i], topo.Z_list[i]))
            coords[root] = np.zeros(3)
            placed.add(root)
            queue = [root]

        while queue:
            center = queue.pop(0)
            Z_c = topo.Z_list[center]
            per_c = period_of(Z_c)
            z_c = topo.z_count[center]
            lp_c = topo.lp[center]

            # Collect unplaced neighbors
            unplaced_nb = [(nb, bo, bi) for nb, bo, bi in neighbors[center]
                           if nb not in placed]
            if not unplaced_nb:
                continue

            # Already-placed neighbors (for angle reference)
            placed_nb = [(nb, bo, bi) for nb, bo, bi in neighbors[center]
                         if nb in placed and nb != center]

            distances = [bond_lengths[bi] for _, _, bi in unplaced_nb]

            if not placed_nb:
                # Root atom: place all neighbors analytically
                positions = _place_neighbors(z_c, lp_c, per_c, distances, Z_c)
                for i, (nb, bo, bi) in enumerate(unplaced_nb):
                    if i < len(positions):
                        coords[nb] = coords[center] + positions[i]
                        placed.add(nb)
                        queue.append(nb)
            else:
                # Non-root: use existing bond direction as reference
                ref_nb = placed_nb[0][0]
                vec_ref = coords[ref_nb] - coords[center]
                norm_ref = np.linalg.norm(vec_ref)
                if norm_ref > 1e-10:
                    dir_ref = vec_ref / norm_ref
                else:
                    dir_ref = np.array([0.0, 0.0, 1.0])

                angle_deg = bond_angle_pt(z_c, lp_c, Z_c)
                angle_rad = math.radians(angle_deg)

                # Inherit plane from parent's neighbors (sp2 coplanarity)
                perp = None
                parent_nb_coords = [
                    coords[nb2] - coords[ref_nb]
                    for nb2, _, _ in neighbors[ref_nb]
                    if nb2 in placed and nb2 != center
                ]
                if parent_nb_coords:
                    v_pnb = parent_nb_coords[0]
                    norm_pnb = np.linalg.norm(v_pnb)
                    if norm_pnb > 1e-10:
                        plane_normal = np.cross(-dir_ref, v_pnb / norm_pnb)
                        norm_pn = np.linalg.norm(plane_normal)
                        if norm_pn > 1e-10:
                            plane_normal /= norm_pn
                            perp = np.cross(plane_normal, dir_ref)
                            norm_p = np.linalg.norm(perp)
                            if norm_p > 1e-10:
                                perp /= norm_p
                            else:
                                perp = None

                if perp is None:
                    if abs(dir_ref[0]) < 0.9:
                        perp = np.cross(dir_ref, np.array([1.0, 0.0, 0.0]))
                    else:
                        perp = np.cross(dir_ref, np.array([0.0, 1.0, 0.0]))
                    perp = perp / np.linalg.norm(perp)

                perp2 = np.cross(dir_ref, perp)

                for j, (nb, bo, bi) in enumerate(unplaced_nb):
                    d = bond_lengths[bi]
                    rot_angle = j * 2.0 * math.pi / max(len(unplaced_nb), 1)
                    cos_r = math.cos(rot_angle)
                    sin_r = math.sin(rot_angle)
                    rotated_perp = cos_r * perp + sin_r * perp2

                    sub_dir = (math.cos(angle_rad) * dir_ref +
                               math.sin(angle_rad) * rotated_perp)
                    norm_sd = np.linalg.norm(sub_dir)
                    if norm_sd > 1e-10:
                        sub_dir /= norm_sd

                    coords[nb] = coords[center] + d * sub_dir
                    placed.add(nb)
                    queue.append(nb)

    # Handle disconnected atoms
    for i in range(n):
        if i not in placed:
            coords[i] = np.array([float(i) * 2.0, 0.0, 0.0])

    # Compute angles from coordinates
    angles_info = []
    for center in range(n):
        nb_list = [nb for nb, _, _ in neighbors[center]]
        for i in range(len(nb_list)):
            for j in range(i + 1, len(nb_list)):
                a, b = nb_list[i], nb_list[j]
                va = coords[a] - coords[center]
                vb = coords[b] - coords[center]
                na, nb_ = np.linalg.norm(va), np.linalg.norm(vb)
                if na > 1e-10 and nb_ > 1e-10:
                    cos_ab = np.dot(va, vb) / (na * nb_)
                    cos_ab = max(-1.0, min(1.0, cos_ab))
                    angle_deg = math.degrees(math.acos(cos_ab))
                    angles_info.append({
                        'atoms': (a, center, b),
                        'angle': angle_deg,
                    })

    # Bond lengths from coordinates (actual)
    lengths_info = []
    for bi, (a, b, bo) in enumerate(topo.bonds):
        d = np.linalg.norm(coords[a] - coords[b])
        lengths_info.append({
            'atoms': (a, b),
            'length': d,
            'order': bo,
        })

    # Torsion angles
    torsions_info = []
    for bi, (a, b, bo) in enumerate(topo.bonds):
        if topo.z_count[a] >= 2 and topo.z_count[b] >= 2:
            phi = torsion_angle_pt(topo.lp[a], topo.lp[b], bo)
            torsions_info.append({
                'atoms': (a, b),
                'angle': phi,
            })

    return Geometry3D(
        coords=coords,
        Z_list=topo.Z_list,
        angles=angles_info,
        lengths=lengths_info,
        torsions=torsions_info,
        vsepr_class=vsepr_class,
    )


# ====================================================================
# EXPORT
# ====================================================================

def to_xyz(coords: Dict[int, np.ndarray], Z_list: List[int],
           comment: str = "") -> str:
    """Export coordinates to XYZ format string.

    Standard format:
      N
      comment
      Sym  x  y  z
      ...
    """
    n = len(Z_list)
    lines = [str(n)]
    lines.append(comment or "PTC geometry | 0 params | s=1/2")
    for i in range(n):
        sym = SYMBOLS.get(Z_list[i], 'X')
        x, y, z = coords[i]
        lines.append(f"{sym:>2s} {x:12.6f} {y:12.6f} {z:12.6f}")
    return '\n'.join(lines) + '\n'


def to_mol_block(coords: Dict[int, np.ndarray], Z_list: List[int],
                 bonds: List[Tuple[int, int, float]]) -> str:
    """Export to MOL V2000 format string."""
    n_atoms = len(Z_list)
    n_bonds = len(bonds)

    lines = [
        "PTC Molecule",
        "  PTC   04122026  3D",
        "",
        f"{n_atoms:>3d}{n_bonds:>3d}  0  0  0  0  0  0  0  0999 V2000",
    ]

    for i in range(n_atoms):
        sym = SYMBOLS.get(Z_list[i], 'X')
        x, y, z = coords[i]
        lines.append(f"{x:>10.4f}{y:>10.4f}{z:>10.4f} {sym:<3s} 0  0  0  0  0  0  0  0  0  0  0  0")

    # For aromatic bonds (bo=1.5): alternate single/double (Kekule rendering)
    # so 3Dmol.js visually shows the double bonds
    aromatic_toggle = True
    for a, b, bo in bonds:
        if abs(bo - 1.5) < 0.1:
            bt = 2 if aromatic_toggle else 1  # alternating double/single
            aromatic_toggle = not aromatic_toggle
        else:
            bt = min(int(round(bo)), 3)
        lines.append(f"{a+1:>3d}{b+1:>3d}{bt:>3d}  0  0  0  0")

    lines.append("M  END")
    return '\n'.join(lines) + '\n'
