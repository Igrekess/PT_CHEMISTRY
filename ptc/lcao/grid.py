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


def _radial_grid_linear(R_max: float, n_radial: int):
    """Standard Gauss-Legendre on [0, R_max] (linear in r). Phase 5a default."""
    leg_r, w_r_raw = np.polynomial.legendre.leggauss(n_radial)
    r_nodes = R_max / 2.0 * (leg_r + 1.0)
    r_weights = R_max / 2.0 * w_r_raw * (r_nodes ** 2)
    return r_nodes, r_weights


def _radial_grid_log(R_atom: float, n_radial: int):
    """Treutler-Ahlrichs M4 logarithmic radial grid (Phase 6.B.7).

    From J. Chem. Phys. 102, 346 (1995). Maps Gauss-Legendre nodes
    x ∈ (-1, 1) to r ∈ (0, ∞) via::

        r(x) = (R_atom / ln 2) · (1+x)^0.6 · ln(2/(1-x))

    This puts ~half the radial nodes in the inner region r < R_atom
    (whereas the linear mapping spreads them uniformly), which is
    essential for resolving tight inner-shell orbitals (1s of Z>10
    has size ~ a₀/Z ≈ 0.06 Å for Cl, beneath the resolution of any
    feasible linear grid). Standard mapping in modern QM codes
    (NWChem, ORCA, Q-Chem default).

    Parameters
    ----------
    R_atom : per-atom Bragg-like scale (Å). Suggest 1/ζ_outer.
    n_radial : number of radial nodes.

    Returns (r_nodes, r_weights) — already includes the r² spherical
    Jacobian and the dr/dx mapping Jacobian.
    """
    x_nodes, x_weights = np.polynomial.legendre.leggauss(n_radial)
    # Avoid x = ±1 singularities
    x_safe = np.clip(x_nodes, -0.99999, 0.99999)
    one_plus_x = 1.0 + x_safe
    one_minus_x = 1.0 - x_safe
    log2 = math.log(2.0)
    log_term = np.log(2.0 / one_minus_x)
    pow_term = one_plus_x ** 0.6

    r_nodes = (R_atom / log2) * pow_term * log_term
    # dr/dx = (R_atom / ln 2) · [0.6·(1+x)^(-0.4)·ln(2/(1-x)) + (1+x)^0.6 / (1-x)]
    dr_dx = (R_atom / log2) * (
        0.6 * (one_plus_x ** (-0.4)) * log_term
        + pow_term / one_minus_x
    )
    r_weights = x_weights * dr_dx * (r_nodes ** 2)
    return r_nodes, r_weights


def build_becke_grid(coords: np.ndarray,
                       R_max: float = 8.0,
                       n_radial: int = 30,
                       lebedev_order: int = 50,
                       probe: Optional[np.ndarray] = None,
                       radial_method: str = "linear",
                       R_atom_per_atom: Optional[np.ndarray] = None,
                       ) -> BeckeGrid:
    """Multi-centre Becke + angular grid for molecular integration.

    Each atom carries a radial × angular sub-grid (per-atom radial map),
    reweighted by the Becke fuzzy-cell partition. If ``probe`` is given,
    a probe-centred sub-grid is added (treats probe as a virtual atom
    so the partition sees it as a centre with weight 1 there).

    Parameters
    ----------
    radial_method : 'linear' (default, Gauss-Legendre on [0, R_max]) or
        'log' (Treutler-Ahlrichs M4 logarithmic mapping). The log mapping
        puts ~half the radial nodes in r < R_atom, essential for
        resolving tight inner-shell orbitals (1s of Z > 10).
    R_atom_per_atom : optional (N_atoms,) array of per-atom Bragg-like
        scales for the log mapping. Defaults to 1.0 Å each (Slater-Bragg
        average) — pass per-atom values for better resolution on
        elements with very tight inner shells.
    """
    coords = np.asarray(coords, dtype=float)
    N = coords.shape[0]

    # Radial grid (per atom)
    if radial_method == "linear":
        r_nodes, r_weights = _radial_grid_linear(R_max, n_radial)
        radial_nodes_per_atom = [r_nodes] * N
        radial_weights_per_atom = [r_weights] * N
    elif radial_method == "log":
        if R_atom_per_atom is None:
            R_atom_per_atom = np.full(N, 1.0)
        else:
            R_atom_per_atom = np.asarray(R_atom_per_atom, dtype=float)
            if R_atom_per_atom.shape != (N,):
                raise ValueError(
                    f"R_atom_per_atom must have shape ({N},), got "
                    f"{R_atom_per_atom.shape}"
                )
        radial_nodes_per_atom = []
        radial_weights_per_atom = []
        for A in range(N):
            r_n, r_w = _radial_grid_log(R_atom_per_atom[A], n_radial)
            radial_nodes_per_atom.append(r_n)
            radial_weights_per_atom.append(r_w)
    else:
        raise ValueError(
            f"radial_method must be 'linear' or 'log', got {radial_method!r}"
        )

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
        # Use per-atom radial grid (probe gets the last entry, i.e. atom 0
        # if probe is appended ; for safety re-use the first atom's grid).
        a_idx = atom_idx if atom_idx < len(radial_nodes_per_atom) else 0
        r_n_atom = radial_nodes_per_atom[a_idx]
        r_w_atom = radial_weights_per_atom[a_idx]
        for r, wr in zip(r_n_atom, r_w_atom):
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
