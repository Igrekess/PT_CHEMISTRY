"""Phase C: GIAO magnetic perturbation operators and shielding tensor.

Computes the chemical-shielding tensor sigma_alpha,beta(P) at an
arbitrary 3D probe point P, starting from the closed-shell PT density
matrix produced by Phase B.

For Phase C this delivery focuses on the **diamagnetic** contribution:

    sigma^d_iso(P) = (alpha^2 / 3) * Tr[ rho * M(P) ],
    M(P)[i, j]    = < phi_i | 1/|r - P| | phi_j > .

Validation : the famous Lamb formula for the isolated H atom recovers
sigma_iso = alpha^2 / 3 ~ 17.75 ppm exactly, since rho_H = |1s><1s| with
zeta = 1/a_0 gives <1/r> = 1/a_0 in atomic units and the constants line up.

Paramagnetic shielding (sigma^p) requires Coupled-Perturbed PT (CPPT) on
the angular-momentum operators L_alpha and is left as a Phase C
continuation. London phase factors for explicit gauge invariance are
likewise pending: the present implementation uses a **common-origin**
gauge at the probe point P, which is exact for spherical atoms and
sufficient for the H/He-atom Lamb validations.

All integrals are evaluated by direct 3D Gauss quadrature on a spherical
grid centred at P (or at a user-specified gauge origin). The grid is
constructed so that the 1/|r-P| singularity is cancelled by the radial
volume element r^2 dr.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np

from ptc.constants import ALPHA_PHYS, A_BOHR
from ptc.lcao.atomic_basis import (
    PTAtomicOrbital,
    PTContractedOrbital,
    iter_primitives,
)
from ptc.lcao.density_matrix import (
    PTMolecularBasis,
    build_molecular_basis,
    density_matrix_PT,
)
from ptc.topology import Topology, build_topology


# ─────────────────────────────────────────────────────────────────────
# STO orbital evaluator (real spherical harmonics, s and p only)
# ─────────────────────────────────────────────────────────────────────


def evaluate_sto(orb,
                 points: np.ndarray,
                 atom_pos: np.ndarray) -> np.ndarray:
    """Evaluate normalised real-spherical STO at one or many 3D points.

    Parameters
    ----------
    orb       : PTAtomicOrbital (single zeta) or PTContractedOrbital
                (sum of N primitives sharing (n,l,m), each with its own
                zeta and contraction coefficient c_i).
    points    : ndarray of shape (..., 3), Angstrom.
    atom_pos  : ndarray of shape (3,), the orbital centre.

    Real spherical harmonic convention (matches atomic_basis):
        l = 0          : 1s, 2s, ... (m = 0)
        l = 1, m = -1  : p_y  (sin phi)
        l = 1, m =  0  : p_z  (cos theta)
        l = 1, m = +1  : p_x  (cos phi)

    For a contracted orbital, returns Sum_i c_i * g_i(r; zeta_i).
    """
    # Phase 6.B.8: dispatch on contracted orbitals
    if isinstance(orb, PTContractedOrbital):
        out = None
        for c_i, prim in iter_primitives(orb):
            v = evaluate_sto(prim, points, atom_pos)
            if out is None:
                out = c_i * v
            else:
                out = out + c_i * v
        return out

    rel = points - atom_pos
    r = np.linalg.norm(rel, axis=-1)

    # Radial part: N_n * r^(n-1) * exp(-zeta r)
    n, zeta = orb.n, orb.zeta
    rad_norm = (2.0 * zeta) ** n * math.sqrt(2.0 * zeta / math.factorial(2 * n))
    radial = rad_norm * (r ** (n - 1)) * np.exp(-zeta * r)

    # Angular part
    if orb.l == 0:
        angular = 1.0 / math.sqrt(4.0 * math.pi)
    elif orb.l == 1:
        ang_norm = math.sqrt(3.0 / (4.0 * math.pi))
        # safe division by r at r ~ 0 (p-orbital is 0 there anyway)
        r_safe = np.where(r > 1.0e-15, r, 1.0)
        if orb.m == -1:
            comp = rel[..., 1]
        elif orb.m == 0:
            comp = rel[..., 2]
        elif orb.m == 1:
            comp = rel[..., 0]
        else:
            raise ValueError(f"l=1 expects m in (-1, 0, +1), got {orb.m}")
        angular = ang_norm * comp / r_safe
        if isinstance(angular, np.ndarray):
            angular = np.where(r > 1.0e-15, angular, 0.0)
    elif orb.l == 2:
        # Real cubic harmonics for d-block:
        #   m=-2 d_xy        : sqrt(15/(4 pi)) * x y / r^2
        #   m=-1 d_yz        : sqrt(15/(4 pi)) * y z / r^2
        #   m= 0 d_z2        : sqrt(5/(16 pi)) * (3 z^2 - r^2) / r^2
        #   m=+1 d_xz        : sqrt(15/(4 pi)) * x z / r^2
        #   m=+2 d_x2-y2     : sqrt(15/(16 pi)) * (x^2 - y^2) / r^2
        r_safe = np.where(r > 1.0e-15, r, 1.0)
        x = rel[..., 0]; y = rel[..., 1]; z = rel[..., 2]
        if orb.m == -2:
            ang_norm = math.sqrt(15.0 / (4.0 * math.pi))
            comp = x * y
        elif orb.m == -1:
            ang_norm = math.sqrt(15.0 / (4.0 * math.pi))
            comp = y * z
        elif orb.m == 0:
            ang_norm = math.sqrt(5.0 / (16.0 * math.pi))
            comp = 3.0 * z * z - r_safe * r_safe
        elif orb.m == 1:
            ang_norm = math.sqrt(15.0 / (4.0 * math.pi))
            comp = x * z
        elif orb.m == 2:
            ang_norm = math.sqrt(15.0 / (16.0 * math.pi))
            comp = x * x - y * y
        else:
            raise ValueError(f"l=2 expects m in (-2..+2), got {orb.m}")
        angular = ang_norm * comp / (r_safe * r_safe)
        if isinstance(angular, np.ndarray):
            angular = np.where(r > 1.0e-15, angular, 0.0)
    elif orb.l == 3:
        # Real cubic harmonics for f-block (4f / 5f).
        r_safe = np.where(r > 1.0e-15, r, 1.0)
        x = rel[..., 0]; y = rel[..., 1]; z = rel[..., 2]
        rr = r_safe * r_safe
        if orb.m == -3:
            ang_norm = math.sqrt(35.0 / (32.0 * math.pi))
            comp = y * (3.0 * x * x - y * y)
        elif orb.m == -2:
            ang_norm = math.sqrt(105.0 / (4.0 * math.pi))
            comp = x * y * z
        elif orb.m == -1:
            ang_norm = math.sqrt(21.0 / (32.0 * math.pi))
            comp = y * (5.0 * z * z - rr)
        elif orb.m == 0:
            ang_norm = math.sqrt(7.0 / (16.0 * math.pi))
            comp = z * (5.0 * z * z - 3.0 * rr)
        elif orb.m == 1:
            ang_norm = math.sqrt(21.0 / (32.0 * math.pi))
            comp = x * (5.0 * z * z - rr)
        elif orb.m == 2:
            ang_norm = math.sqrt(105.0 / (16.0 * math.pi))
            comp = z * (x * x - y * y)
        elif orb.m == 3:
            ang_norm = math.sqrt(35.0 / (32.0 * math.pi))
            comp = x * (x * x - 3.0 * y * y)
        else:
            raise ValueError(f"l=3 expects m in (-3..+3), got {orb.m}")
        angular = ang_norm * comp / (r_safe * r_safe * r_safe)
        if isinstance(angular, np.ndarray):
            angular = np.where(r > 1.0e-15, angular, 0.0)
    elif orb.l == 4:
        # Real cubic harmonics for g-orbitals (DZP+ polarisation).
        r_safe = np.where(r > 1.0e-15, r, 1.0)
        x = rel[..., 0]; y = rel[..., 1]; z = rel[..., 2]
        rr = r_safe * r_safe
        rrrr = rr * rr
        if orb.m == -4:
            ang_norm = (3.0 / 4.0) * math.sqrt(35.0 / math.pi)
            comp = x * y * (x * x - y * y)
        elif orb.m == -3:
            ang_norm = (3.0 / 4.0) * math.sqrt(35.0 / (2.0 * math.pi))
            comp = y * z * (3.0 * x * x - y * y)
        elif orb.m == -2:
            ang_norm = (3.0 / 4.0) * math.sqrt(5.0 / math.pi)
            comp = 2.0 * x * y * (7.0 * z * z - rr)
        elif orb.m == -1:
            ang_norm = (3.0 / 4.0) * math.sqrt(5.0 / (2.0 * math.pi))
            comp = y * z * (7.0 * z * z - 3.0 * rr)
        elif orb.m == 0:
            ang_norm = (3.0 / 16.0) * math.sqrt(1.0 / math.pi)
            comp = 35.0 * z ** 4 - 30.0 * z * z * rr + 3.0 * rrrr
        elif orb.m == 1:
            ang_norm = (3.0 / 4.0) * math.sqrt(5.0 / (2.0 * math.pi))
            comp = x * z * (7.0 * z * z - 3.0 * rr)
        elif orb.m == 2:
            ang_norm = (3.0 / 8.0) * math.sqrt(5.0 / math.pi)
            comp = (x * x - y * y) * (7.0 * z * z - rr)
        elif orb.m == 3:
            ang_norm = (3.0 / 4.0) * math.sqrt(35.0 / (2.0 * math.pi))
            comp = x * z * (x * x - 3.0 * y * y)
        elif orb.m == 4:
            ang_norm = (3.0 / 16.0) * math.sqrt(35.0 / math.pi)
            comp = x ** 4 - 6.0 * x * x * y * y + y ** 4
        else:
            raise ValueError(f"l=4 expects m in (-4..+4), got {orb.m}")
        angular = ang_norm * comp / rrrr
        if isinstance(angular, np.ndarray):
            angular = np.where(r > 1.0e-15, angular, 0.0)
    else:
        raise NotImplementedError(
            f"evaluate_sto: l={orb.l} not implemented (s/p/d/f/g supported; "
            "h / higher pending)"
        )

    return radial * angular


# ─────────────────────────────────────────────────────────────────────
# Spherical grid (centred at P) for nuclear-attraction-style integrals
# ─────────────────────────────────────────────────────────────────────


@dataclass
class _SphericalGrid:
    """Gauss-Legendre x Gauss-Legendre x trapezoidal grid on a sphere
    centred at the origin (caller shifts to probe point).
    """
    points: np.ndarray   # (N_total, 3)
    weights: np.ndarray  # (N_total,)


def _build_spherical_grid(R_max: float,
                          n_radial: int = 60,
                          n_theta: int = 24,
                          n_phi: int = 32) -> _SphericalGrid:
    """Spherical-coordinate grid for integrating f(r) over a ball.

    Volume element  r^2 sin(theta) dr dtheta dphi  ->  carrying the
    sin(theta) into the substitution u = cos(theta) gives
        weight_grid_point = r^2 * w_r * w_u * (2*pi/n_phi)
    The 1/r factor of nuclear-attraction integrals is folded in by the
    caller (see `nuclear_attraction_matrix`).
    """
    leg_r, w_r = np.polynomial.legendre.leggauss(n_radial)
    r_nodes = R_max / 2.0 * (leg_r + 1.0)
    r_weights = R_max / 2.0 * w_r

    leg_u, w_u = np.polynomial.legendre.leggauss(n_theta)
    cos_theta = leg_u
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta ** 2))

    phi_nodes = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)
    phi_weight = 2.0 * math.pi / n_phi

    R, CT, PHI = np.meshgrid(r_nodes, cos_theta, phi_nodes, indexing='ij')
    ST = np.sqrt(np.maximum(0.0, 1.0 - CT ** 2))
    X = R * ST * np.cos(PHI)
    Y = R * ST * np.sin(PHI)
    Z = R * CT

    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    # weights = r^2 * w_r * w_u * phi_weight  (sin(theta) absorbed by u-substitution)
    # broadcast all three axes to (n_radial, n_theta, n_phi)
    phi_w = np.full(n_phi, phi_weight)
    W = (
        ((r_nodes ** 2) * r_weights)[:, None, None]
        * w_u[None, :, None]
        * phi_w[None, None, :]
    )
    return _SphericalGrid(points=pts, weights=W.flatten())


def _orbital_values_on_grid(basis: PTMolecularBasis,
                            grid_pts: np.ndarray) -> np.ndarray:
    """Return (N_orb, N_pts) array of phi_i evaluated at the global points."""
    N = basis.n_orbitals
    psi = np.zeros((N, grid_pts.shape[0]))
    for k, orb in enumerate(basis.orbitals):
        atom_pos = basis.coords[basis.atom_index[k]]
        psi[k] = evaluate_sto(orb, grid_pts, atom_pos)
    return psi


def _build_quadrature_grid(basis: PTMolecularBasis,
                            origin: np.ndarray,
                            n_radial: int,
                            n_theta: int,
                            n_phi: int,
                            use_becke: bool,
                            lebedev_order: int) -> tuple:
    """Helper: choose between spherical (centred on `origin`) and Becke
    multi-centre grids and return (pts_global, weights, rel_to_origin).

    rel_to_origin = pts_global - origin (kept for callers that need
    operators expressed relative to the gauge / probe origin).
    """
    origin = np.asarray(origin, dtype=float)
    max_dist = max(
        float(np.linalg.norm(origin - basis.coords[i]))
        for i in range(basis.n_atoms)
    )
    min_zeta = min(float(o.zeta) for o in basis.orbitals)
    R_max = max(max_dist + 8.0 / min_zeta, 12.0 / min_zeta)

    if use_becke:
        from ptc.lcao.grid import build_becke_grid
        # When the integrand has an operator centred on `origin` with a
        # 1/|r-O|^k singularity (σ_d at probe, L at gauge origin, M at
        # dipole point) the per-atom Becke grid alone undersamples the
        # singular region. Adding `origin` as a virtual atom produces a
        # probe-centred spherical sub-grid that integrates the singularity
        # properly (1/r × r² dr = r dr is smooth in spherical at probe).
        # Skip the augmentation only if the probe coincides with an atom.
        atom_dists = np.linalg.norm(basis.coords - origin[None, :], axis=-1)
        probe_on_atom = float(atom_dists.min()) < 1.0e-8
        bg = build_becke_grid(
            basis.coords,
            R_max=8.0 / min_zeta,
            n_radial=n_radial,
            lebedev_order=lebedev_order,
            probe=None if probe_on_atom else origin,
        )
        pts_global = bg.points
        weights = bg.weights
    else:
        sg = _build_spherical_grid(R_max, n_radial, n_theta, n_phi)
        pts_global = sg.points + origin[None, :]
        weights = sg.weights
    rel = pts_global - origin[None, :]
    return pts_global, weights, rel


# ─────────────────────────────────────────────────────────────────────
# Nuclear-attraction matrix M(P)[i, j] = <phi_i | 1/|r - P| | phi_j>
# ─────────────────────────────────────────────────────────────────────


def nuclear_attraction_matrix(basis: PTMolecularBasis,
                               P: np.ndarray,
                               n_radial: int = 60,
                               n_theta: int = 24,
                               n_phi: int = 32,
                               use_becke: bool = False,
                               lebedev_order: int = 50) -> np.ndarray:
    """3D quadrature of <phi_i | 1/|r - P| | phi_j> at probe point P.

    Default backend (use_becke=False)
    ---------------------------------
    Spherical grid centred on P. The 1/|r-P| factor combines with the
    radial volume r^2 to give r * dr (smooth at origin).

    Becke backend (use_becke=True)
    ------------------------------
    Multi-centre Becke + Lebedev grid. The 1/|r-P| factor is evaluated
    pointwise; no grid point sits on P (atoms are away from P in the
    NMR-probe use-case), so the singularity is handled smoothly.
    n_theta / n_phi are ignored; lebedev_order replaces them.

    Notes
    -----
    * Returned matrix is symmetric.
    * R_max chosen so that all atom-localised orbitals decay below
      ~1e-7 of their peak.
    """
    P = np.asarray(P, dtype=float)
    # Choose R_max from atom positions and orbital exponents
    max_dist = max(
        float(np.linalg.norm(P - basis.coords[i])) for i in range(basis.n_atoms)
    )
    min_zeta = min(float(o.zeta) for o in basis.orbitals)
    R_max = max(max_dist + 8.0 / min_zeta, 12.0 / min_zeta)

    if use_becke:
        from ptc.lcao.grid import build_becke_grid
        R_atom = 8.0 / min_zeta
        atom_dists = np.linalg.norm(basis.coords - P[None, :], axis=-1)
        probe_on_atom = float(atom_dists.min()) < 1.0e-8
        bg = build_becke_grid(basis.coords, R_max=R_atom,
                              n_radial=n_radial, lebedev_order=lebedev_order,
                              probe=None if probe_on_atom else P)
        pts_global = bg.points
        # 1/|r - P|; with probe-augmented Becke, the singularity at P is
        # in the probe sub-grid where 1/r × r² dr is well-conditioned.
        r_to_P = np.linalg.norm(pts_global - P[None, :], axis=-1)
        inv_r = np.where(r_to_P > 1e-15, 1.0 / r_to_P, 0.0)
        W_eff = bg.weights * inv_r
    else:
        grid = _build_spherical_grid(R_max, n_radial, n_theta, n_phi)
        # Shift to probe point
        pts_global = grid.points + P[None, :]
        # 1/|r - P| = 1/||grid.points|| (origin at P)
        r_local = np.linalg.norm(grid.points, axis=-1)
        inv_r = np.where(r_local > 1e-15, 1.0 / r_local, 0.0)
        W_eff = grid.weights * inv_r

    psi = _orbital_values_on_grid(basis, pts_global)  # (N_orb, N_pts)

    # M = psi @ diag(W_eff) @ psi.T  --> efficient
    M = (psi * W_eff[None, :]) @ psi.T
    return 0.5 * (M + M.T)  # enforce symmetry numerically


# ─────────────────────────────────────────────────────────────────────
# Diamagnetic shielding (isotropic)
# ─────────────────────────────────────────────────────────────────────


def shielding_diamagnetic_iso(rho: np.ndarray,
                                basis: PTMolecularBasis,
                                P: np.ndarray,
                                **quad_kwargs) -> float:
    """Isotropic diamagnetic shielding sigma^d_iso at probe point P (ppm).

    Common-origin gauge at P, so the formula reduces to
        sigma^d_iso(P) = (alpha^2 / 3) * Tr[ rho * M(P) ]
    where M(P)[i, j] = <phi_i | 1/|r - P| | phi_j>.

    Returns a float in *ppm* (10^-6 unit-less).
    """
    M = nuclear_attraction_matrix(basis, P, **quad_kwargs)
    # zeta in atomic_basis is in Angstrom^-1; the integrals come out in
    # 1/Angstrom. Convert to atomic-unit 1/a_0 by multiplying by a_0(A).
    M_au = M * A_BOHR
    sigma_au = (ALPHA_PHYS ** 2 / 3.0) * float(np.trace(rho @ M_au))
    return sigma_au * 1.0e6  # to ppm


# ─────────────────────────────────────────────────────────────────────
# High-level convenience: shielding at an arbitrary point of a molecule
# ─────────────────────────────────────────────────────────────────────


def shielding_at_point(topology: Topology,
                        P: np.ndarray,
                        **quad_kwargs) -> float:
    """End-to-end pipeline: build basis, density matrix, return sigma^d_iso (ppm)."""
    basis = build_molecular_basis(topology)
    rho, _, _, _ = density_matrix_PT(topology, basis=basis)
    return shielding_diamagnetic_iso(rho, basis, np.asarray(P, dtype=float),
                                      **quad_kwargs)


def H_atom_lamb_ppm(**quad_kwargs) -> float:
    """Direct H-atom diamagnetic shielding, expected ~17.75 ppm (Lamb)."""
    topo = build_topology("[H]")
    P = np.zeros(3)  # H atom placed at origin by build_molecular_basis
    return shielding_at_point(topo, P, **quad_kwargs)


# ─────────────────────────────────────────────────────────────────────
# STO gradient (analytic, closed-form for l = 0, 1, 2)
# ─────────────────────────────────────────────────────────────────────


def evaluate_sto_gradient_analytic(orb,
                                    points: np.ndarray,
                                    atom_pos: np.ndarray) -> np.ndarray:
    """Closed-form gradient of a normalised real-spherical STO.

    Phase 6.B.8 — also dispatches on PTContractedOrbital, in which
    case the gradient is the contraction-weighted sum of primitive
    gradients (linearity of the derivative).

    The STO has the factored form
        phi(r) = N_rad * Y_lm(rel) * f(r),
    where for our convention (matching evaluate_sto)
        l = 0 :  phi = (N_rad / sqrt(4 pi)) * r^(n-1) * exp(-zeta r)
        l = 1 :  phi = N_rad * ang_norm * r^(n-2) * exp(-zeta r) * x_alpha
        l = 2 :  phi = N_rad * ang_norm * r^(n-3) * exp(-zeta r) * A(x,y,z)
    with A a quadratic homogeneous polynomial fixed by m. Cartesian
    differentiation gives, with k = n - 1 - l and R(r) = r^k * exp(-zeta r),
        d phi / d x_b = N_rad * ang_norm * R(r)
                       * [ d P / d x_b
                           + P * (x_b / r) * (k/r - zeta) ],
    where P is the polynomial part (1, x_alpha, or A).

    Replaces the previous central-difference implementation: T matrix
    elements are now precise to ~1e-12 (vs ~1e-3 with eps = 1e-5).
    """
    # Phase 6.B.8: dispatch contracted orbital -> sum of primitive gradients
    if isinstance(orb, PTContractedOrbital):
        out = None
        for c_i, prim in iter_primitives(orb):
            g = evaluate_sto_gradient_analytic(prim, points, atom_pos)
            if out is None:
                out = c_i * g
            else:
                out = out + c_i * g
        return out

    rel = points - atom_pos
    r = np.linalg.norm(rel, axis=-1)
    n, l, m, zeta = orb.n, orb.l, orb.m, orb.zeta
    rad_norm = (2.0 * zeta) ** n * math.sqrt(2.0 * zeta / math.factorial(2 * n))

    eps_r = 1.0e-15
    r_safe = np.where(r > eps_r, r, 1.0)
    out_shape = rel.shape if rel.ndim > 1 else (3,)
    grad = np.zeros(out_shape)
    zero_mask = (r <= eps_r)

    if l == 0:
        ang_norm = 1.0 / math.sqrt(4.0 * math.pi)
        k = n - 1
        if k == 0:
            radial = np.exp(-zeta * r_safe)
            df_over_r = -zeta * np.ones_like(r_safe) / r_safe
        else:
            radial = (r_safe ** k) * np.exp(-zeta * r_safe)
            df_over_r = (k / r_safe - zeta) / r_safe
        coeff = rad_norm * ang_norm * radial * df_over_r
        grad = coeff[..., None] * rel if rel.ndim > 1 else (coeff * rel)
        # 1s has a cusp at the origin; gradient ill-defined but quadrature
        # weight ~ r^2 * dr -> 0 there, so safely set to 0.
        if np.any(zero_mask):
            if rel.ndim > 1:
                grad[zero_mask] = 0.0
            elif zero_mask:
                grad[:] = 0.0

    elif l == 1:
        ang_norm = math.sqrt(3.0 / (4.0 * math.pi))
        alpha = {-1: 1, 0: 2, 1: 0}.get(m)
        if alpha is None:
            raise ValueError(f"l=1 expects m in (-1, 0, +1), got {m}")
        x_alpha = rel[..., alpha]
        k = n - 2
        if k == 0:
            radial = np.exp(-zeta * r_safe)
            df_over_r = -zeta * np.ones_like(r_safe) / r_safe
        else:
            radial = (r_safe ** k) * np.exp(-zeta * r_safe)
            df_over_r = (k / r_safe - zeta) / r_safe
        prefac = rad_norm * ang_norm * radial
        # delta_{alpha,beta} term
        if rel.ndim > 1:
            grad[..., alpha] = prefac
        else:
            grad[alpha] = prefac
        # quadratic term: prefac * x_alpha * (k/r - zeta) * x_beta / r
        coeff = prefac * x_alpha * df_over_r
        grad = grad + (coeff[..., None] * rel if rel.ndim > 1 else coeff * rel)
        # for n >= 3 (k >= 1) the factor r^k vanishes at r=0 so prefac->0;
        # for n = 2 the delta term equals N_rad*ang_norm and quadratic
        # term carries x_alpha=0 at the origin, so no cleanup needed.

    elif l == 2:
        x = rel[..., 0]; y = rel[..., 1]; z = rel[..., 2]
        if m == -2:
            ang_norm = math.sqrt(15.0 / (4.0 * math.pi))
            A = x * y
            dA = (y, x, np.zeros_like(z))
        elif m == -1:
            ang_norm = math.sqrt(15.0 / (4.0 * math.pi))
            A = y * z
            dA = (np.zeros_like(x), z, y)
        elif m == 0:
            ang_norm = math.sqrt(5.0 / (16.0 * math.pi))
            A = 3.0 * z * z - (x * x + y * y + z * z)
            dA = (-2.0 * x, -2.0 * y, 4.0 * z)
        elif m == 1:
            ang_norm = math.sqrt(15.0 / (4.0 * math.pi))
            A = x * z
            dA = (z, np.zeros_like(y), x)
        elif m == 2:
            ang_norm = math.sqrt(15.0 / (16.0 * math.pi))
            A = x * x - y * y
            dA = (2.0 * x, -2.0 * y, np.zeros_like(z))
        else:
            raise ValueError(f"l=2 expects m in (-2..+2), got {m}")

        k = n - 3
        if k == 0:
            radial = np.exp(-zeta * r_safe)
            df_over_r = -zeta * np.ones_like(r_safe) / r_safe
        else:
            radial = (r_safe ** k) * np.exp(-zeta * r_safe)
            df_over_r = (k / r_safe - zeta) / r_safe
        prefac = rad_norm * ang_norm * radial
        # dP/dx_beta term
        if rel.ndim > 1:
            grad[..., 0] = prefac * dA[0]
            grad[..., 1] = prefac * dA[1]
            grad[..., 2] = prefac * dA[2]
        else:
            grad[0] = prefac * dA[0]
            grad[1] = prefac * dA[1]
            grad[2] = prefac * dA[2]
        # P * (k/r - zeta) * x_beta / r term
        coeff = prefac * A * df_over_r
        grad = grad + (coeff[..., None] * rel if rel.ndim > 1 else coeff * rel)

    elif l in (3, 4):
        # f / g orbital gradient via numerical FD (4-point central differences).
        # Analytic gradient for cubic / quartic harmonics is tractable but
        # lengthy; FD with eps=1e-5 yields ~1e-9 accuracy, sufficient for
        # CPHF response and current density.
        eps_fd = 1.0e-5
        if rel.ndim > 1:
            for d in range(3):
                shift = np.zeros(3); shift[d] = eps_fd
                psi_p = evaluate_sto(orb, points + shift, atom_pos)
                psi_m = evaluate_sto(orb, points - shift, atom_pos)
                grad[..., d] = (psi_p - psi_m) / (2.0 * eps_fd)
        else:
            for d in range(3):
                shift = np.zeros(3); shift[d] = eps_fd
                psi_p = evaluate_sto(orb, (points + shift)[None, :], atom_pos)[0]
                psi_m = evaluate_sto(orb, (points - shift)[None, :], atom_pos)[0]
                grad[d] = (psi_p - psi_m) / (2.0 * eps_fd)

    else:
        raise NotImplementedError(
            f"evaluate_sto_gradient_analytic: l={l} not implemented"
        )

    return grad


def evaluate_sto_gradient(orb: PTAtomicOrbital,
                           points: np.ndarray,
                           atom_pos: np.ndarray,
                           eps: float | None = None) -> np.ndarray:
    """3D gradient of an STO orbital at points (..., 3) — analytic.

    Returns shape (..., 3). Closed-form expressions for l = 0, 1, 2;
    accuracy ~1e-13 (machine epsilon), independent of any step size.

    The legacy `eps` keyword is accepted for backwards-compatibility but
    has no effect: the analytic path is used unconditionally.
    """
    return evaluate_sto_gradient_analytic(orb, points, atom_pos)


# ─────────────────────────────────────────────────────────────────────
# Full diamagnetic tensor sigma^d_{alpha,beta}(P)
# ─────────────────────────────────────────────────────────────────────


def shielding_diamagnetic_tensor(rho: np.ndarray,
                                  basis: PTMolecularBasis,
                                  P: np.ndarray,
                                  n_radial: int = 60,
                                  n_theta: int = 24,
                                  n_phi: int = 32,
                                  use_becke: bool = False,
                                  lebedev_order: int = 50) -> np.ndarray:
    """Full 3 x 3 diamagnetic shielding tensor at probe P (ppm).

    sigma^d_{alpha,beta}(P) = (alpha^2 / 2) * Tr[ rho * M^d_{alpha,beta}(P) ]
    M^d_{alpha,beta}(P)[i,j] = <phi_i | (r^2 delta_{a,b} - r_a r_b) / r^3 | phi_j>

    Common-origin gauge at P. The trace gives (1/3)Tr -> isotropic = the
    same result as `shielding_diamagnetic_iso`.
    """
    pts_global, weights, rel = _build_quadrature_grid(
        basis, P, n_radial, n_theta, n_phi, use_becke, lebedev_order
    )
    r = np.linalg.norm(rel, axis=-1)
    inv_r3 = np.where(r > 1.0e-15, 1.0 / r ** 3, 0.0)

    psi = _orbital_values_on_grid(basis, pts_global)

    sigma_d = np.zeros((3, 3))
    for alpha in range(3):
        for beta in range(alpha, 3):
            if alpha == beta:
                op = (r ** 2 - rel[:, alpha] ** 2) * inv_r3
            else:
                op = -rel[:, alpha] * rel[:, beta] * inv_r3
            W = weights * op
            M_ab = (psi * W[None, :]) @ psi.T
            M_ab = 0.5 * (M_ab + M_ab.T)
            # zeta in 1/Angstrom -> need 1/a_0 in operator
            M_ab *= A_BOHR
            sigma_d[alpha, beta] = (ALPHA_PHYS ** 2 / 2.0) * float(np.trace(rho @ M_ab))
            sigma_d[beta, alpha] = sigma_d[alpha, beta]
    return sigma_d * 1.0e6  # ppm


# ─────────────────────────────────────────────────────────────────────
# GIAO diamagnetic tensor (Phase 5d — London phase factors)
# ─────────────────────────────────────────────────────────────────────


def shielding_diamagnetic_tensor_GIAO(rho: np.ndarray,
                                       basis: PTMolecularBasis,
                                       P: np.ndarray,
                                       n_radial: int = 60,
                                       n_theta: int = 24,
                                       n_phi: int = 32,
                                       use_becke: bool = False,
                                       lebedev_order: int = 50) -> np.ndarray:
    """GIAO diamagnetic shielding tensor at probe P (ppm). Phase 5d.

    For London-phased orbitals chi_mu = exp(-i/(2c) (B x R_mu) . r) phi_mu,
    the diamagnetic tensor reduces to a per-orbital-pair form where the
    matrix element uses the bond-midpoint R_munu = (R_mu + R_nu)/2:

        M^d_GIAO[mu,nu]_alpha,beta =
              <phi_mu |
                  [(r-P) . (r-R_munu) delta_{alpha,beta}
                   - (r-P)_beta (r-R_munu)_alpha] / |r-P|^3
              | phi_nu>

    Reduces to the common-origin form when R_munu = P (single-atom case)
    and gives gauge-INVARIANT total shielding when paired with
    GIAO-corrected sigma^p (i.e., angular_momentum_matrices_GIAO).

    sigma^d_GIAO_{alpha,beta}(P) = (alpha^2 / 2) * Tr[ rho * M^d_GIAO ]

    Notes
    -----
    Computed via single 3D quadrature evaluating both common-origin and
    correction integrals on the same grid. The correction is
        (P - R_munu) . r^P delta_{alpha,beta} / r_P^3
        - (P - R_munu)_alpha (r^P)_beta / r_P^3
    which factorises into vector integrals
        V[mu, nu]_gamma = <phi_mu | (r-P)_gamma / |r-P|^3 | phi_nu>
        S[mu, nu]       = <phi_mu | 1 / |r-P|^3      | phi_nu>
    These can be computed on the shared Becke grid and then assembled
    algebraically.
    """
    pts_global, weights, rel = _build_quadrature_grid(
        basis, P, n_radial, n_theta, n_phi, use_becke, lebedev_order
    )
    r = np.linalg.norm(rel, axis=-1)
    inv_r3 = np.where(r > 1.0e-15, 1.0 / r ** 3, 0.0)

    psi = _orbital_values_on_grid(basis, pts_global)
    n_orb = basis.n_orbitals

    # 1) Vector integral V_gamma[mu, nu] = <phi_mu | r^P_gamma / r_P^3 | phi_nu>
    V_vec = np.zeros((3, n_orb, n_orb))
    for gamma in range(3):
        op = rel[:, gamma] * inv_r3
        W = weights * op
        Mg = (psi * W[None, :]) @ psi.T
        V_vec[gamma] = 0.5 * (Mg + Mg.T)

    # 2) Common-origin diamagnetic matrix elements (alpha, beta block)
    M_common = np.zeros((3, 3, n_orb, n_orb))
    for alpha in range(3):
        for beta in range(alpha, 3):
            if alpha == beta:
                op = (r ** 2 - rel[:, alpha] ** 2) * inv_r3
            else:
                op = -rel[:, alpha] * rel[:, beta] * inv_r3
            W = weights * op
            Mab = (psi * W[None, :]) @ psi.T
            Mab = 0.5 * (Mab + Mab.T)
            M_common[alpha, beta] = Mab
            M_common[beta, alpha] = Mab

    # 3) Per-orbital-pair midpoint shift  s[mu, nu] = P - R_munu
    R_atom = np.array(
        [basis.coords[basis.atom_index[mu]] for mu in range(n_orb)]
    )
    R_munu = 0.5 * (R_atom[:, None, :] + R_atom[None, :, :])  # (n, n, 3)
    s = P[None, None, :] - R_munu                              # (n, n, 3)

    # 4) Assemble GIAO correction. Standard derivation (Helgaker, Jorgensen,
    # Olsen 2000) of the diamagnetic shielding under London phase factors
    # gives:
    #    M^d_GIAO[mu,nu]_alpha,beta = <phi_mu | (r_O.r_P delta_alpha,beta
    #                                  - r_O_beta r_P_alpha) / r_P^3 | phi_nu>
    # with r_O = r - R_munu and r_P = r - P. Substituting r_O = r_P + s
    # where s = P - R_munu yields the correction
    #    correction_alpha,beta = (s . r_P) delta_alpha,beta / r_P^3
    #                            - s_beta r_P_alpha / r_P^3
    # i.e. the SECOND term has indices (s_beta, V_alpha) — NOT (s_alpha, V_beta).
    # This is the index ordering needed for gauge invariance with the GIAO
    # paramagnetic term.

    # delta_term[mu, nu] = sum_gamma s_gamma * V_gamma
    delta_term = sum(s[..., g] * V_vec[g] for g in range(3))   # (n, n)

    M_GIAO = np.zeros_like(M_common)
    for alpha in range(3):
        for beta in range(3):
            corr = -s[..., beta] * V_vec[alpha]
            if alpha == beta:
                corr = corr + delta_term
            M_GIAO[alpha, beta] = M_common[alpha, beta] + corr
            # Symmetrise per (alpha, beta) block
            M_GIAO[alpha, beta] = 0.5 * (M_GIAO[alpha, beta]
                                          + M_GIAO[alpha, beta].T)

    sigma_d = np.zeros((3, 3))
    for alpha in range(3):
        for beta in range(3):
            # zeta in 1/Angstrom -> 1/a0 (multiply operator by A_BOHR)
            M_ab = M_GIAO[alpha, beta] * A_BOHR
            sigma_d[alpha, beta] = (
                (ALPHA_PHYS ** 2 / 2.0) * float(np.trace(rho @ M_ab))
            )
    return sigma_d * 1.0e6   # ppm


# ─────────────────────────────────────────────────────────────────────
# Angular momentum operators L_alpha at gauge origin O
# ─────────────────────────────────────────────────────────────────────


def angular_momentum_matrices(basis: PTMolecularBasis,
                               gauge_origin: np.ndarray,
                               n_radial: int = 50,
                               n_theta: int = 20,
                               n_phi: int = 24,
                               use_becke: bool = False,
                               lebedev_order: int = 50) -> np.ndarray:
    """Real antisymmetric matrices L_imag[alpha, i, j] such that
    < phi_i | L_alpha | phi_j > = -i * L_imag[alpha, i, j].

    L_alpha = ((r - R_O) x p)_alpha with p = -i nabla, gauge origin R_O.
    For real STO basis the matrix element is purely imaginary and the
    'imaginary part' L_imag is real antisymmetric.

    Returns array shape (3, n_orb, n_orb).
    """
    pts_global, weights, rel = _build_quadrature_grid(
        basis, gauge_origin, n_radial, n_theta, n_phi, use_becke, lebedev_order
    )
    n = basis.n_orbitals
    Npts = pts_global.shape[0]

    # Evaluate orbitals + gradients at every grid point
    psi = np.zeros((n, Npts))
    grad = np.zeros((n, Npts, 3))
    for k, orb in enumerate(basis.orbitals):
        atom_pos = basis.coords[basis.atom_index[k]]
        psi[k] = evaluate_sto(orb, pts_global, atom_pos)
        grad[k] = evaluate_sto_gradient(orb, pts_global, atom_pos)

    L_imag = np.zeros((3, n, n))
    for alpha in range(3):
        beta = (alpha + 1) % 3
        gamma = (alpha + 2) % 3
        # (r x grad)_alpha = r_beta * grad_gamma - r_gamma * grad_beta
        op_phi = (
            rel[:, beta][None, :] * grad[:, :, gamma]
            - rel[:, gamma][None, :] * grad[:, :, beta]
        )  # shape (n, Npts)
        L_imag[alpha] = (psi * weights[None, :]) @ op_phi.T
        # Numerical antisymmetrisation
        L_imag[alpha] = 0.5 * (L_imag[alpha] - L_imag[alpha].T)
    return L_imag


# ─────────────────────────────────────────────────────────────────────
# Ramsey "magnetic-dipole" operator L_alpha / |r-K|^3 at probe K
# ─────────────────────────────────────────────────────────────────────


def momentum_matrix(basis: PTMolecularBasis,
                     n_radial: int = 50,
                     n_theta: int = 20,
                     n_phi: int = 24,
                     use_becke: bool = False,
                     lebedev_order: int = 50) -> np.ndarray:
    """Real antisymmetric matrices p_imag[alpha, i, j] such that
    < phi_i | p_alpha | phi_j > = -i * p_imag[alpha, i, j].

    p = -i * nabla. In real STO basis the matrix element is purely
    imaginary and p_imag is real antisymmetric (p is hermitian -> i*p
    anti-hermitian -> p_imag antisymmetric).
    """
    centroid = basis.coords.mean(axis=0)
    pts_global, weights, _ = _build_quadrature_grid(
        basis, centroid, n_radial, n_theta, n_phi, use_becke, lebedev_order
    )
    n = basis.n_orbitals

    psi = np.zeros((n, pts_global.shape[0]))
    grad = np.zeros((n, pts_global.shape[0], 3))
    for k, orb in enumerate(basis.orbitals):
        atom_pos = basis.coords[basis.atom_index[k]]
        psi[k] = evaluate_sto(orb, pts_global, atom_pos)
        grad[k] = evaluate_sto_gradient(orb, pts_global, atom_pos)

    p_imag = np.zeros((3, n, n))
    for alpha in range(3):
        # < phi_i | grad_alpha | phi_j > on grid
        p_imag[alpha] = (psi * weights[None, :]) @ grad[..., alpha].T
        p_imag[alpha] = 0.5 * (p_imag[alpha] - p_imag[alpha].T)
    return p_imag


def angular_momentum_matrices_GIAO(basis: PTMolecularBasis,
                                     gauge_origin: np.ndarray | None = None,
                                     **quad_kwargs) -> np.ndarray:
    """GIAO angular momentum matrices with London phase factor correction.

    For London-phased orbitals chi_mu = exp(-i/(2c) (B x R_mu) . r) phi_mu,
    the angular-momentum matrix element reduces to a PER-PAIR gauge
    origin at the orbital-centre midpoint M_munu = (R_mu + R_nu)/2:

        L_beta^GIAO[mu, nu] = ((r - M_munu) x p)_beta[mu, nu]

    To compute it numerically efficiently, we evaluate L_beta at a
    SINGLE reference origin O_ref (the molecular centroid) and apply
    the algebraic shift

        L_beta^M_munu[mu, nu] = L_beta^O_ref[mu, nu]
                                - ((M_munu - O_ref) x p[mu, nu])_beta

    The result is truly gauge-invariant -- it does not depend on the
    user-supplied gauge_origin parameter (kept only for backward API
    compat; internally we always integrate at the centroid for numeric
    stability). Returned shape (3, n_orb, n_orb), real antisymmetric.
    """
    # Numerically stable single integration origin: molecular centroid
    O_ref = basis.coords.mean(axis=0)

    L_imag = angular_momentum_matrices(basis, O_ref, **quad_kwargs)
    p_imag = momentum_matrix(basis, **quad_kwargs)

    n = basis.n_orbitals
    R_atom = np.array([basis.coords[basis.atom_index[mu]] for mu in range(n)])
    R_sum = R_atom[:, None, :] + R_atom[None, :, :]   # (n, n, 3)
    shift = R_sum * 0.5 - O_ref[None, None, :]        # M_munu - O_ref

    correction = np.zeros((3, n, n))
    for beta in range(3):
        gamma = (beta + 1) % 3
        delta = (beta + 2) % 3
        correction[beta] = (
            shift[..., gamma] * p_imag[delta]
            - shift[..., delta] * p_imag[gamma]
        )

    L_GIAO = L_imag - correction
    for b in range(3):
        L_GIAO[b] = 0.5 * (L_GIAO[b] - L_GIAO[b].T)
    return L_GIAO


def magnetic_dipole_matrices(basis: PTMolecularBasis,
                              K: np.ndarray,
                              n_radial: int = 60,
                              n_theta: int = 24,
                              n_phi: int = 32,
                              use_becke: bool = False,
                              lebedev_order: int = 50) -> np.ndarray:
    """Real antisymmetric matrices M_imag[alpha, i, j] such that
    < phi_i | (L_alpha^K / |r - K|^3) | phi_j > = -i * M_imag[alpha, i, j].

    L_alpha^K = ((r - K) x p)_alpha, divided by |r - K|^3 to give the
    Ramsey magnetic-dipole operator at point K.

    The 1/r^3 singularity at K is integrable (1/r^3 * r^2 dr = dr/r ->
    log-singular but vanishes for any function that is O(r) at r=0).
    Antisymmetric construction enforced numerically.
    """
    pts_global, weights, rel = _build_quadrature_grid(
        basis, K, n_radial, n_theta, n_phi, use_becke, lebedev_order
    )
    r_K = np.linalg.norm(rel, axis=-1)
    inv_r3 = np.where(r_K > 1.0e-15, 1.0 / r_K ** 3, 0.0)
    n = basis.n_orbitals

    psi = np.zeros((n, pts_global.shape[0]))
    grad = np.zeros((n, pts_global.shape[0], 3))
    for k, orb in enumerate(basis.orbitals):
        atom_pos = basis.coords[basis.atom_index[k]]
        psi[k] = evaluate_sto(orb, pts_global, atom_pos)
        grad[k] = evaluate_sto_gradient(orb, pts_global, atom_pos)

    W_eff = weights * inv_r3

    M_imag = np.zeros((3, n, n))
    for alpha in range(3):
        beta = (alpha + 1) % 3
        gamma = (alpha + 2) % 3
        op_phi = (
            rel[:, beta][None, :] * grad[:, :, gamma]
            - rel[:, gamma][None, :] * grad[:, :, beta]
        )
        M_imag[alpha] = (psi * W_eff[None, :]) @ op_phi.T
        M_imag[alpha] = 0.5 * (M_imag[alpha] - M_imag[alpha].T)
    return M_imag


# ─────────────────────────────────────────────────────────────────────
# Paramagnetic shielding via uncoupled CP-PT (closed shell)
# ─────────────────────────────────────────────────────────────────────


def paramagnetic_shielding_iso(basis: PTMolecularBasis,
                                 mo_eigvals: np.ndarray,
                                 mo_coeffs: np.ndarray,
                                 n_e_total: int,
                                 K: np.ndarray,
                                 gauge_origin: np.ndarray | None = None,
                                 use_giao: bool = True,
                                 **quad_kwargs) -> float:
    """Isotropic paramagnetic shielding sigma^p_iso(K) (ppm).

    Uncoupled-CP-PT formula for closed-shell systems:

        sigma^p_{alpha,beta}(K) =
            (4 alpha^2) Sum_{a in virt, i in occ}
                L_imag[beta, a, i] * M_imag[alpha, i, a]
                / (eps_a - eps_i)

    derivation:
      sigma^p = -(2/c^2) Re Sum_{ai} [<i|L_b^O|a><a|L_a^K/r^3|i> + c.c.]
                             / (eps_a - eps_i)
      Real basis: <i|L|a> = -i L_imag[i,a] (real)
      Product (-i)(-i) = -1
      L_imag and M_imag both antisymmetric -> the overall sign comes
      out positive with the (-2/c^2) prefactor flipped to (+4 alpha^2).

    The default gauge origin equals K (common-origin gauge). The
    valence-only minimal basis (Phase A) has no virtuals for closed-
    shell atoms (all valence MOs occupied), which trivially gives
    sigma^p = 0 -- matching the symmetry expectation for spherical
    atoms.
    """
    if gauge_origin is None:
        gauge_origin = K
    gauge_origin = np.asarray(gauge_origin, dtype=float)
    K = np.asarray(K, dtype=float)

    n_occ = n_e_total // 2
    n_virt = mo_coeffs.shape[1] - n_occ
    if n_occ == 0 or n_virt == 0:
        return 0.0

    if use_giao:
        # GIAO L: per-pair midpoint gauge -> result independent of gauge_origin
        L_imag = angular_momentum_matrices_GIAO(basis, **quad_kwargs)
    else:
        L_imag = angular_momentum_matrices(basis, gauge_origin, **quad_kwargs)
    M_imag = magnetic_dipole_matrices(basis, K, **quad_kwargs)

    # Convert to MO basis (c is S-orthonormal: c.T S c = I, so MO basis is
    # equivalent to AO basis modulo similarity transform). The integrals
    # transform as: L_MO = c.T @ L_AO @ c.
    L_mo = np.array([mo_coeffs.T @ L_imag[k] @ mo_coeffs for k in range(3)])
    M_mo = np.array([mo_coeffs.T @ M_imag[k] @ mo_coeffs for k in range(3)])

    eps_a = mo_eigvals[n_occ:]              # (n_virt,)
    eps_i = mo_eigvals[:n_occ]              # (n_occ,)
    diff = eps_a[None, :] - eps_i[:, None]  # (n_occ, n_virt)

    # zeta is in 1/Angstrom; integrals come out in units of (1/A) for L
    # and (1/A)^3 for M (since 1/r^3 in 1/A^3). Convert to atomic units:
    # multiply L by A_BOHR and M by A_BOHR^3.
    L_mo_au = L_mo * A_BOHR
    M_mo_au = M_mo * (A_BOHR ** 3)

    # Energies are in eV (from IE_eV); convert to atomic units (Hartree):
    # 1 Hartree = 27.211 eV
    HARTREE_eV = 27.211386245988
    diff_au = diff / HARTREE_eV

    sigma_p_diag = np.zeros(3)
    for alpha in range(3):
        # σ^p_αα = (4 α^2) Σ_{ia} L_imag[α, a, i] × M_imag[α, i, a] / (ε_a - ε_i)
        L_ai = L_mo_au[alpha, n_occ:, :n_occ]   # (n_virt, n_occ)
        M_ia = M_mo_au[alpha, :n_occ, n_occ:]   # (n_occ, n_virt)
        # element-wise (i,a) product
        prod = L_ai.T * M_ia                     # (n_occ, n_virt)
        sigma_p_diag[alpha] = 4.0 * (ALPHA_PHYS ** 2) * (prod / diff_au).sum()
    sigma_p_iso = sigma_p_diag.mean()
    return sigma_p_iso * 1.0e6  # ppm


# ─────────────────────────────────────────────────────────────────────
# Total isotropic shielding (dia + para)
# ─────────────────────────────────────────────────────────────────────


def shielding_total_iso(topology: Topology,
                         P: np.ndarray,
                         **quad_kwargs) -> dict:
    """End-to-end total shielding: dia + para CP-PT.

    Returns dict with keys 'sigma_d', 'sigma_p', 'sigma_total' (ppm).
    """
    basis = build_molecular_basis(topology)
    rho, S, eigvals, c = density_matrix_PT(topology, basis=basis)
    n_e = int(round(basis.total_occ))

    sigma_d = shielding_diamagnetic_iso(rho, basis, np.asarray(P, dtype=float),
                                          **quad_kwargs)
    sigma_p = paramagnetic_shielding_iso(basis, eigvals, c, n_e,
                                           np.asarray(P, dtype=float),
                                           **quad_kwargs)
    return {
        "sigma_d": sigma_d,
        "sigma_p": sigma_p,
        "sigma_total": sigma_d + sigma_p,
    }
