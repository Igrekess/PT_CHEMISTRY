"""ptc/lcao/grid.py — Becke fuzzy-cell partition + multi-centre grid.

Becke's fuzzy-cell weighting (J. Chem. Phys. 88, 2547, 1988) for
multi-centre numerical integration. Used here for:
  - shielding_diamagnetic_tensor_GIAO (Phase 5d)
  - current_density_at_points (gauge='ipsocentric')

Only the parameter-free version (no Bragg-radius tuning) is exposed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import numpy as np


def _becke_p3(mu: np.ndarray) -> np.ndarray:
    """Becke smooth-step iterated 3 times: p(p(p(mu)))."""
    p = (3.0 * mu - mu ** 3) / 2.0
    p = (3.0 * p - p ** 3) / 2.0
    p = (3.0 * p - p ** 3) / 2.0
    return p


def becke_partition_weights(coords: np.ndarray,
                              points: np.ndarray) -> np.ndarray:
    """Becke fuzzy-cell weights w_A(r) for each grid point r against atoms A.

    Returns (N_pts, N_atoms) with each row summing to 1.
    """
    coords = np.asarray(coords, dtype=float)
    points = np.asarray(points, dtype=float)
    N_atoms = coords.shape[0]

    diff = points[:, None, :] - coords[None, :, :]
    r_atom = np.linalg.norm(diff, axis=-1)

    R_AB = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    R_AB_safe = np.where(R_AB > 1.0e-12, R_AB, 1.0)

    P = np.ones((points.shape[0], N_atoms))
    for A in range(N_atoms):
        for B in range(N_atoms):
            if A == B:
                continue
            mu = (r_atom[:, A] - r_atom[:, B]) / R_AB_safe[A, B]
            mu = np.clip(mu, -1.0, 1.0)
            s = 0.5 * (1.0 - _becke_p3(mu))
            P[:, A] *= s

    Z = P.sum(axis=1, keepdims=True)
    Z_safe = np.where(Z > 0.0, Z, 1.0)
    return P / Z_safe


@dataclass(frozen=True)
class BeckeGrid:
    points: np.ndarray
    weights: np.ndarray


def _angular_grid(lebedev_order: int = 50) -> tuple:
    """Build an angular grid (Lebedev-equivalent) on the unit sphere.

    Falls back to a spherical theta×phi product grid when ``lebedev``
    is unavailable.  Order maps roughly to angular precision.

    Returns (points, weights) on the unit sphere.
    """
    # Map lebedev_order to (n_theta, n_phi) for spherical fallback
    n_theta = max(8, min(40, int(math.sqrt(lebedev_order * 6))))
    n_phi = max(12, min(64, n_theta * 2))
    leg_u, w_u = np.polynomial.legendre.leggauss(n_theta)
    phi = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)
    phi_w = 2.0 * math.pi / n_phi
    pts = []
    ws = []
    for u, wu in zip(leg_u, w_u):
        st = math.sqrt(max(0.0, 1.0 - u * u))
        for p in phi:
            pts.append([st * math.cos(p), st * math.sin(p), u])
            ws.append(wu * phi_w)
    return np.array(pts), np.array(ws)


def build_becke_grid(coords: np.ndarray,
                       R_max: float = 8.0,
                       n_radial: int = 30,
                       lebedev_order: int = 50,
                       probe: Optional[np.ndarray] = None,
                       ) -> BeckeGrid:
    """Multi-centre Becke + angular grid for molecular integration.

    Each atom carries a Gauss-Legendre radial × angular sub-grid,
    reweighted by the Becke fuzzy-cell partition. If ``probe`` is given,
    a probe-centred sub-grid is added (treats probe as a virtual atom
    so the partition sees it as a centre with weight 1 there).
    """
    coords = np.asarray(coords, dtype=float)
    N = coords.shape[0]

    # Radial grid (per atom): Gauss-Legendre on [0, R_max], r² weight
    leg_r, w_r_raw = np.polynomial.legendre.leggauss(n_radial)
    r_nodes = R_max / 2.0 * (leg_r + 1.0)
    r_weights = R_max / 2.0 * w_r_raw * (r_nodes ** 2)

    ang_pts, ang_w = _angular_grid(lebedev_order)

    # Augment coordinate list with probe (treated as a centre for partition)
    if probe is not None:
        probe = np.asarray(probe, dtype=float)
        coords_aug = np.vstack([coords, probe[None, :]])
    else:
        coords_aug = coords
    n_centres = coords_aug.shape[0]

    all_pts = []
    all_w = []
    centre_indices = []
    for atom_idx in range(n_centres):
        centre = coords_aug[atom_idx]
        for r, wr in zip(r_nodes, r_weights):
            for ap, aw in zip(ang_pts, ang_w):
                all_pts.append(centre + r * ap)
                all_w.append(wr * aw)
                centre_indices.append(atom_idx)
    pts = np.array(all_pts)
    raw_w = np.array(all_w)
    centre_indices = np.array(centre_indices)

    # Becke partition over the augmented centre list
    w_atoms_aug = becke_partition_weights(coords_aug, pts)
    # Each grid point's effective weight = its raw quadrature weight × the
    # Becke weight assigned to the SUB-GRID's centre.
    becke_factor = np.zeros(len(pts))
    for k in range(n_centres):
        sl = (centre_indices == k)
        becke_factor[sl] = w_atoms_aug[sl, k]

    return BeckeGrid(points=pts, weights=raw_w * becke_factor)
