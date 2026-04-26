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
from ptc.lcao.atomic_basis import PTAtomicOrbital
from ptc.lcao.density_matrix import (
    PTMolecularBasis,
    build_molecular_basis,
    density_matrix_PT,
)
from ptc.topology import Topology, build_topology


# ─────────────────────────────────────────────────────────────────────
# STO orbital evaluator (real spherical harmonics, s and p only)
# ─────────────────────────────────────────────────────────────────────


def evaluate_sto(orb: PTAtomicOrbital,
                 points: np.ndarray,
                 atom_pos: np.ndarray) -> np.ndarray:
    """Evaluate normalised real-spherical STO at one or many 3D points.

    Parameters
    ----------
    orb       : PTAtomicOrbital with quantum numbers (n, l, m) and zeta
                (in Angstrom^-1, see atomic_basis.build_atom_basis).
    points    : ndarray of shape (..., 3), Angstrom.
    atom_pos  : ndarray of shape (3,), the orbital centre.

    Real spherical harmonic convention (matches atomic_basis):
        l = 0          : 1s, 2s, ... (m = 0)
        l = 1, m = -1  : p_y  (sin phi)
        l = 1, m =  0  : p_z  (cos theta)
        l = 1, m = +1  : p_x  (cos phi)
    """
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
    else:
        raise NotImplementedError(
            f"evaluate_sto: l={orb.l} not implemented (s, p, d supported; "
            "f / higher pending Phase A continuation)"
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


# ─────────────────────────────────────────────────────────────────────
# Nuclear-attraction matrix M(P)[i, j] = <phi_i | 1/|r - P| | phi_j>
# ─────────────────────────────────────────────────────────────────────


def nuclear_attraction_matrix(basis: PTMolecularBasis,
                               P: np.ndarray,
                               n_radial: int = 60,
                               n_theta: int = 24,
                               n_phi: int = 32) -> np.ndarray:
    """3D quadrature of <phi_i | 1/|r - P| | phi_j> at probe point P.

    Spherical grid centred on P. The 1/|r-P| factor combines with the
    radial volume r^2 to give r * dr (smooth at origin).

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

    grid = _build_spherical_grid(R_max, n_radial, n_theta, n_phi)

    # Shift to probe point
    pts_global = grid.points + P[None, :]

    # 1/|r - P| = 1/||grid.points|| (origin at P)
    r_local = np.linalg.norm(grid.points, axis=-1)
    inv_r = np.where(r_local > 1e-15, 1.0 / r_local, 0.0)

    # operator weight including 1/r
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
# STO gradient (finite-difference, vectorised)
# ─────────────────────────────────────────────────────────────────────


def evaluate_sto_gradient(orb: PTAtomicOrbital,
                           points: np.ndarray,
                           atom_pos: np.ndarray,
                           eps: float = 1.0e-5) -> np.ndarray:
    """Numerical 3D gradient of an STO orbital at points (..., 3).

    Returns shape (..., 3). Central differences keep the truncation
    error at O(eps^2) ~ 1e-10 with eps = 1e-5; round-off dominates.

    For our valence basis (s, p), an analytical gradient is also
    derivable but the central-difference route stays compact and
    matches `evaluate_sto` exactly in convention.
    """
    grad = np.zeros(points.shape if points.ndim > 1 else (3,))
    h = np.zeros(3)
    for i in range(3):
        h[:] = 0.0
        h[i] = eps
        f_plus = evaluate_sto(orb, points + h, atom_pos)
        f_minus = evaluate_sto(orb, points - h, atom_pos)
        grad[..., i] = (f_plus - f_minus) / (2.0 * eps)
    return grad


# ─────────────────────────────────────────────────────────────────────
# Full diamagnetic tensor sigma^d_{alpha,beta}(P)
# ─────────────────────────────────────────────────────────────────────


def shielding_diamagnetic_tensor(rho: np.ndarray,
                                  basis: PTMolecularBasis,
                                  P: np.ndarray,
                                  n_radial: int = 60,
                                  n_theta: int = 24,
                                  n_phi: int = 32) -> np.ndarray:
    """Full 3 x 3 diamagnetic shielding tensor at probe P (ppm).

    sigma^d_{alpha,beta}(P) = (alpha^2 / 2) * Tr[ rho * M^d_{alpha,beta}(P) ]
    M^d_{alpha,beta}(P)[i,j] = <phi_i | (r^2 delta_{a,b} - r_a r_b) / r^3 | phi_j>

    Common-origin gauge at P. The trace gives (1/3)Tr -> isotropic = the
    same result as `shielding_diamagnetic_iso`.
    """
    P = np.asarray(P, dtype=float)
    max_dist = max(
        float(np.linalg.norm(P - basis.coords[i])) for i in range(basis.n_atoms)
    )
    min_zeta = min(float(o.zeta) for o in basis.orbitals)
    R_max = max(max_dist + 8.0 / min_zeta, 12.0 / min_zeta)

    grid = _build_spherical_grid(R_max, n_radial, n_theta, n_phi)
    pts_global = grid.points + P[None, :]
    rel = grid.points
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
            W = grid.weights * op
            M_ab = (psi * W[None, :]) @ psi.T
            M_ab = 0.5 * (M_ab + M_ab.T)
            # zeta in 1/Angstrom -> need 1/a_0 in operator
            M_ab *= A_BOHR
            sigma_d[alpha, beta] = (ALPHA_PHYS ** 2 / 2.0) * float(np.trace(rho @ M_ab))
            sigma_d[beta, alpha] = sigma_d[alpha, beta]
    return sigma_d * 1.0e6  # ppm


# ─────────────────────────────────────────────────────────────────────
# Angular momentum operators L_alpha at gauge origin O
# ─────────────────────────────────────────────────────────────────────


def angular_momentum_matrices(basis: PTMolecularBasis,
                               gauge_origin: np.ndarray,
                               n_radial: int = 50,
                               n_theta: int = 20,
                               n_phi: int = 24) -> np.ndarray:
    """Real antisymmetric matrices L_imag[alpha, i, j] such that
    < phi_i | L_alpha | phi_j > = -i * L_imag[alpha, i, j].

    L_alpha = ((r - R_O) x p)_alpha with p = -i nabla, gauge origin R_O.
    For real STO basis the matrix element is purely imaginary and the
    'imaginary part' L_imag is real antisymmetric.

    Returns array shape (3, n_orb, n_orb).
    """
    O = np.asarray(gauge_origin, dtype=float)
    max_dist = max(
        float(np.linalg.norm(O - basis.coords[i])) for i in range(basis.n_atoms)
    )
    min_zeta = min(float(o.zeta) for o in basis.orbitals)
    R_max = max(max_dist + 8.0 / min_zeta, 12.0 / min_zeta)

    grid = _build_spherical_grid(R_max, n_radial, n_theta, n_phi)
    pts_global = grid.points + O[None, :]
    rel = grid.points  # r relative to O
    n = basis.n_orbitals
    Npts = grid.points.shape[0]

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
        W = grid.weights
        L_imag[alpha] = (psi * W[None, :]) @ op_phi.T
        # Numerical antisymmetrisation
        L_imag[alpha] = 0.5 * (L_imag[alpha] - L_imag[alpha].T)
    return L_imag


# ─────────────────────────────────────────────────────────────────────
# Ramsey "magnetic-dipole" operator L_alpha / |r-K|^3 at probe K
# ─────────────────────────────────────────────────────────────────────


def momentum_matrix(basis: PTMolecularBasis,
                     n_radial: int = 50,
                     n_theta: int = 20,
                     n_phi: int = 24) -> np.ndarray:
    """Real antisymmetric matrices p_imag[alpha, i, j] such that
    < phi_i | p_alpha | phi_j > = -i * p_imag[alpha, i, j].

    p = -i * nabla. In real STO basis the matrix element is purely
    imaginary and p_imag is real antisymmetric (p is hermitian -> i*p
    anti-hermitian -> p_imag antisymmetric).
    """
    centroid = basis.coords.mean(axis=0)
    max_dist = max(
        float(np.linalg.norm(centroid - basis.coords[i]))
        for i in range(basis.n_atoms)
    )
    min_zeta = min(float(o.zeta) for o in basis.orbitals)
    R_max = max(max_dist + 8.0 / min_zeta, 12.0 / min_zeta)

    grid = _build_spherical_grid(R_max, n_radial, n_theta, n_phi)
    pts_global = grid.points + centroid[None, :]
    n = basis.n_orbitals

    psi = np.zeros((n, grid.points.shape[0]))
    grad = np.zeros((n, grid.points.shape[0], 3))
    for k, orb in enumerate(basis.orbitals):
        atom_pos = basis.coords[basis.atom_index[k]]
        psi[k] = evaluate_sto(orb, pts_global, atom_pos)
        grad[k] = evaluate_sto_gradient(orb, pts_global, atom_pos)

    p_imag = np.zeros((3, n, n))
    for alpha in range(3):
        # < phi_i | grad_alpha | phi_j > on grid
        p_imag[alpha] = (psi * grid.weights[None, :]) @ grad[..., alpha].T
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
                              n_phi: int = 32) -> np.ndarray:
    """Real antisymmetric matrices M_imag[alpha, i, j] such that
    < phi_i | (L_alpha^K / |r - K|^3) | phi_j > = -i * M_imag[alpha, i, j].

    L_alpha^K = ((r - K) x p)_alpha, divided by |r - K|^3 to give the
    Ramsey magnetic-dipole operator at point K.

    The 1/r^3 singularity at K is integrable (1/r^3 * r^2 dr = dr/r ->
    log-singular but vanishes for any function that is O(r) at r=0).
    Antisymmetric construction enforced numerically.
    """
    K = np.asarray(K, dtype=float)
    max_dist = max(
        float(np.linalg.norm(K - basis.coords[i])) for i in range(basis.n_atoms)
    )
    min_zeta = min(float(o.zeta) for o in basis.orbitals)
    R_max = max(max_dist + 8.0 / min_zeta, 12.0 / min_zeta)

    grid = _build_spherical_grid(R_max, n_radial, n_theta, n_phi)
    pts_global = grid.points + K[None, :]
    rel = grid.points
    r_K = np.linalg.norm(rel, axis=-1)
    inv_r3 = np.where(r_K > 1.0e-15, 1.0 / r_K ** 3, 0.0)
    n = basis.n_orbitals

    psi = np.zeros((n, grid.points.shape[0]))
    grad = np.zeros((n, grid.points.shape[0], 3))
    for k, orb in enumerate(basis.orbitals):
        atom_pos = basis.coords[basis.atom_index[k]]
        psi[k] = evaluate_sto(orb, pts_global, atom_pos)
        grad[k] = evaluate_sto_gradient(orb, pts_global, atom_pos)

    W_eff = grid.weights * inv_r3

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
