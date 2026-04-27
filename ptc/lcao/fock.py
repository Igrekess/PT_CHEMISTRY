"""Phase B continuation #3 — Fock matrix with 2-electron integrals.

Limitation #3 of BACKLOG_LCAO_PRECISION.md: replace the uncoupled
core-Hamiltonian SCF (H = T + V_nuc) with self-consistent Hartree-Fock
(F = H_core + J - K/2).

Closed-shell Fock matrix:
    F[mu, nu] = H_core[mu, nu] + J[mu, nu] - (1/2) K[mu, nu]

with:
    J[mu, nu] = sum_{kl} rho[k, l] (mu nu | k l)
    K[mu, nu] = sum_{kl} rho[k, l] (mu k | nu l)

(mu nu | k l) is the chemists' / Mulliken 2-electron integral.

Implementation
==============
Direct numerical evaluation on a single shared spherical Gauss grid
centred on the molecular centroid:

  Density at each grid point:  n(g) = sum_{kl} rho[k, l] phi_k(g) phi_l(g)
  Coulomb potential at g:      V_C(g) = sum_{g'} n(g') W(g') / |g-g'|
  J matrix:                    J[mu, nu] = sum_g phi_mu(g) phi_nu(g) V_C(g) W(g)

The exchange K matrix uses a non-local kernel:
  rho_op(g, g') = sum_{kl} rho[k, l] phi_k(g) phi_l(g')
  K[mu, nu]    = sum_{g, g'} phi_mu(g) rho_op(g, g') phi_nu(g')
                              W(g) W(g') / |g-g'|

Both operations are O(N_grid^2) and dominate the cost. Memory is kept
manageable by chunking the outer-grid loop. Self-interaction of a
single grid point with itself is set to zero (small bias, vanishes
with grid refinement).
"""

from __future__ import annotations

import numpy as np

from ptc.constants import COULOMB_EV_A
from ptc.lcao.density_matrix import (
    PTMolecularBasis,
    core_hamiltonian,
    overlap_matrix,
    solve_mo,
)
from ptc.lcao.giao import _build_spherical_grid, _orbital_values_on_grid


# ─────────────────────────────────────────────────────────────────────
# Cached molecular grid + orbital values (avoid recomputing per iter)
# ─────────────────────────────────────────────────────────────────────


class _MolGrid:
    """Lightweight grid holder: absolute points + matching weights.

    Returned by `_build_molecular_grid` regardless of which backend
    (spherical vs Becke) was used. Downstream consumers (J / K matrix
    builds) only need `.points` (absolute coordinates) and `.weights`.
    """
    __slots__ = ("points", "weights")

    def __init__(self, points: np.ndarray, weights: np.ndarray):
        self.points = points
        self.weights = weights


def _build_molecular_grid(basis: PTMolecularBasis,
                            n_radial: int = 30,
                            n_theta: int = 14,
                            n_phi: int = 18,
                            use_becke: bool = False,
                            lebedev_order: int = 50):
    """Build a 3D quadrature grid + orbital values for the molecule.

    Default backend (use_becke=False)
    ---------------------------------
    Single spherical Gauss grid centred on the molecular centroid. Cheap
    but cannot resolve very tight inner-shell core orbitals (1s with
    zeta ~ 10 / Angstrom on second-row atoms) — the radial spacing of
    a centroid-centred grid is too coarse near each nucleus.

    Becke + Lebedev backend (use_becke=True)
    ----------------------------------------
    Multi-centre adaptive grid: each atom contributes its own (Gauss
    radial) x (Lebedev angular) sphere, weighted by the Becke fuzzy
    partition. Required for all-electron SCF: the radial grid near each
    atom is dense enough to resolve tight 1s orbitals. Phase 5a.

    n_theta / n_phi are ignored when use_becke=True; lebedev_order
    replaces them.
    """
    centroid = basis.coords.mean(axis=0)
    max_dist = max(
        float(np.linalg.norm(centroid - basis.coords[i]))
        for i in range(basis.n_atoms)
    )
    min_zeta = min(float(o.zeta) for o in basis.orbitals)

    if use_becke:
        from ptc.lcao.grid import build_becke_grid
        # Each atomic sphere extends 8/zeta_min — covers diffuse valence;
        # tight inner orbitals are well within this radius.
        R_atom = 8.0 / min_zeta
        bg = build_becke_grid(basis.coords, R_max=R_atom,
                              n_radial=n_radial, lebedev_order=lebedev_order)
        pts_abs = bg.points
        weights = bg.weights
    else:
        R_max = max(max_dist + 8.0 / min_zeta, 12.0 / min_zeta)
        sg = _build_spherical_grid(R_max, n_radial, n_theta, n_phi)
        pts_abs = sg.points + centroid[None, :]
        weights = sg.weights

    psi = _orbital_values_on_grid(basis, pts_abs)  # (N_orb, N_grid)
    grid = _MolGrid(points=pts_abs, weights=weights)
    return grid, psi


# ─────────────────────────────────────────────────────────────────────
# Coulomb J matrix
# ─────────────────────────────────────────────────────────────────────


def coulomb_J_matrix(rho: np.ndarray,
                      basis: PTMolecularBasis,
                      grid=None,
                      psi=None,
                      chunk_size: int = 200,
                      n_radial: int = 30,
                      n_theta: int = 14,
                      n_phi: int = 18,
                      use_becke: bool = False,
                      lebedev_order: int = 50) -> np.ndarray:
    """J matrix in eV via shared molecular grid.

        J[mu, nu] = sum_{kl} rho[k, l] (mu nu | k l)
                  = INT phi_mu(r) V_C(r) phi_nu(r) dr

    where V_C(r) = sum_{kl} rho[k, l] INT phi_k(r') phi_l(r') / |r-r'| dr'
                = INT n(r') / |r-r'| dr'   with  n(r') = sum_{kl} rho_kl phi_k(r') phi_l(r').
    """
    if grid is None or psi is None:
        grid, psi = _build_molecular_grid(basis, n_radial, n_theta, n_phi,
                                            use_becke=use_becke,
                                            lebedev_order=lebedev_order)

    n_grid = grid.points.shape[0]

    # Density at each grid point
    rho_psi = rho @ psi                              # (N_orb, N_grid)
    n_at_grid = np.sum(psi * rho_psi, axis=0)        # (N_grid,)
    nW = n_at_grid * grid.weights                    # density * volume weight

    pos = grid.points
    V = np.zeros(n_grid)
    for i_start in range(0, n_grid, chunk_size):
        i_end = min(i_start + chunk_size, n_grid)
        diff = pos[i_start:i_end, None, :] - pos[None, :, :]   # (chunk, N, 3)
        dist = np.linalg.norm(diff, axis=2)                    # (chunk, N)
        # Self contribution -> 0 (singular integrand approximation)
        rng = np.arange(i_start, i_end)
        dist[np.arange(i_end - i_start), rng] = 1.0
        contrib = nW[None, :] / dist
        contrib[np.arange(i_end - i_start), rng] = 0.0
        V[i_start:i_end] = contrib.sum(axis=1)

    # J[mu, nu] = sum_g psi_mu(g) psi_nu(g) V(g) W(g)
    weighted = psi * (V * grid.weights)[None, :]
    J = weighted @ psi.T
    J = 0.5 * (J + J.T)
    return J * COULOMB_EV_A


# ─────────────────────────────────────────────────────────────────────
# Exchange K matrix
# ─────────────────────────────────────────────────────────────────────


def exchange_K_matrix(rho: np.ndarray,
                        basis: PTMolecularBasis,
                        grid=None,
                        psi=None,
                        chunk_size: int = 100,
                        n_radial: int = 30,
                        n_theta: int = 14,
                        n_phi: int = 18,
                        symmetry: str = "auto",
                        use_becke: bool = False,
                        lebedev_order: int = 50) -> np.ndarray:
    """K matrix in eV via shared molecular grid.

        K[mu, nu] = sum_{kl} rho[k, l] (mu k | nu l)
                  = INT_INT phi_mu(r) rho_op(r, r') phi_nu(r') / |r-r'| dr dr'

    with rho_op(r, r') = sum_{kl} rho_kl phi_k(r) phi_l(r').

    Symmetry parameter
    ------------------
    'sym'    : K is symmetric (closed-shell SCF on a hermitian rho)
    'antisym': K is antisymmetric (response density rho^(1) for magnetic
               perturbation; J vanishes, only K contributes to F^(1))
    'auto'   : detect from input rho

    K of an antisymmetric rho IS antisymmetric (proof: relabel kappa <->
    lambda in the integral pattern); the projection step purifies the
    output against numerical leakage.
    """
    if grid is None or psi is None:
        grid, psi = _build_molecular_grid(basis, n_radial, n_theta, n_phi,
                                            use_becke=use_becke,
                                            lebedev_order=lebedev_order)

    n_grid = grid.points.shape[0]
    n_orb = basis.n_orbitals

    pos = grid.points
    W = grid.weights
    rho_psi = rho @ psi

    K = np.zeros((n_orb, n_orb))
    for i_start in range(0, n_grid, chunk_size):
        i_end = min(i_start + chunk_size, n_grid)
        diff = pos[i_start:i_end, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        rng = np.arange(i_start, i_end)
        dist[np.arange(i_end - i_start), rng] = 1.0
        kernel = (W[i_start:i_end, None] * W[None, :]) / dist
        kernel[np.arange(i_end - i_start), rng] = 0.0
        rho_op_chunk = rho_psi[:, i_start:i_end].T @ psi
        weighted_rho_op = rho_op_chunk * kernel
        K += psi[:, i_start:i_end] @ weighted_rho_op @ psi.T

    if symmetry == "auto":
        rho_sym = np.abs(rho + rho.T).max()
        rho_anti = np.abs(rho - rho.T).max()
        symmetry = "sym" if rho_sym >= rho_anti else "antisym"

    if symmetry == "sym":
        K = 0.5 * (K + K.T)
    elif symmetry == "antisym":
        K = 0.5 * (K - K.T)
    else:
        raise ValueError(f"symmetry must be 'sym', 'antisym', or 'auto'")
    return K * COULOMB_EV_A


# ─────────────────────────────────────────────────────────────────────
# Fock matrix and SCF
# ─────────────────────────────────────────────────────────────────────


def fock_matrix(rho: np.ndarray,
                 basis: PTMolecularBasis,
                 H_core: np.ndarray,
                 mode: str = "hf",
                 **grid_kwargs) -> np.ndarray:
    """Fock matrix.

    mode='hf'      : F = H_core + J - K/2          (full Hartree-Fock)
    mode='hartree' : F = H_core + J                (Hartree only, biased
                                                     by self-interaction)
    """
    grid, psi = _build_molecular_grid(basis, **grid_kwargs)
    J = coulomb_J_matrix(rho, basis, grid=grid, psi=psi)
    if mode == "hf":
        K = exchange_K_matrix(rho, basis, grid=grid, psi=psi)
        F = H_core + J - 0.5 * K
    elif mode == "hartree":
        F = H_core + J
    else:
        raise ValueError(f"mode must be 'hf' or 'hartree', got {mode!r}")
    return 0.5 * (F + F.T)


class DIIS:
    """Pulay's Direct Inversion in Iterative Subspace.

    Accumulates Fock matrices and error vectors, solves the constrained
    minimisation problem to extrapolate an optimal next Fock for SCF.
    Standard formulation; SCF error vector e = F P S - S P F (commutator
    in non-orthogonal basis, vanishes at convergence).
    """

    def __init__(self, max_size: int = 8):
        self.max_size = max_size
        self.F_list = []
        self.e_list = []

    def add(self, F: np.ndarray, P: np.ndarray, S: np.ndarray):
        e = F @ P @ S - S @ P @ F
        self.F_list.append(F.copy())
        self.e_list.append(e.copy())
        if len(self.F_list) > self.max_size:
            self.F_list.pop(0)
            self.e_list.pop(0)
        return float(np.abs(e).max())

    def extrapolate(self) -> np.ndarray:
        n = len(self.F_list)
        if n < 2:
            return self.F_list[-1].copy()
        B = np.zeros((n + 1, n + 1))
        for i in range(n):
            for j in range(n):
                B[i, j] = float(np.sum(self.e_list[i] * self.e_list[j]))
        B[-1, :n] = -1.0
        B[:n, -1] = -1.0
        rhs = np.zeros(n + 1)
        rhs[-1] = -1.0
        try:
            coeffs = np.linalg.solve(B, rhs)[:n]
        except np.linalg.LinAlgError:
            return self.F_list[-1].copy()
        F_diis = np.zeros_like(self.F_list[0])
        for i in range(n):
            F_diis += coeffs[i] * self.F_list[i]
        return F_diis


def density_matrix_PT_scf(
    topology,
    basis: PTMolecularBasis | None = None,
    *,
    mode: str = "hf",
    max_iter: int = 50,
    tol: float = 1.0e-5,
    damping: float = 0.5,
    diis: bool = True,
    diis_start: int = 2,
    n_radial: int = 24,
    n_theta: int = 12,
    n_phi: int = 16,
    verbose: bool = False,
    nuclear_charge: str = "actual",
    use_becke: bool = False,
    lebedev_order: int = 50,
):
    """Self-consistent-field density matrix.

    mode='hf' (default) does full Hartree-Fock with both J and K.
    mode='hartree' is Hartree-only (test path; biased by self-interaction).

    Phase 5a: when use_becke=True, the J/K integrals are evaluated on a
    multi-centre Becke + Lebedev grid (Phase 3 grid module). This is
    REQUIRED for all-electron HF: the centroid-centred spherical grid
    cannot resolve tight 1s core orbitals (zeta ~ 10/Angstrom). With
    Becke, each atom carries its own dense radial mesh.

    Returns (rho, S, eigvals, c, converged_iter, last_residual).
    """
    from ptc.lcao.density_matrix import build_molecular_basis

    if basis is None:
        basis = build_molecular_basis(topology)

    S = overlap_matrix(basis)
    H_core = core_hamiltonian(
        basis, S,
        n_radial=n_radial, n_theta=n_theta, n_phi=n_phi,
        nuclear_charge=nuclear_charge,
    )

    n_e = int(round(basis.total_occ))
    n_doubly = n_e // 2
    odd = n_e % 2 == 1

    # Initial guess from H_core diagonalisation
    eigvals, c = solve_mo(H_core, S)

    def build_rho(c_mat):
        rho_local = np.zeros_like(S)
        for k in range(n_doubly):
            rho_local += 2.0 * np.outer(c_mat[:, k], c_mat[:, k])
        if odd and n_doubly < c_mat.shape[1]:
            rho_local += 1.0 * np.outer(c_mat[:, n_doubly], c_mat[:, n_doubly])
        return rho_local

    rho = build_rho(c)

    grid, psi = _build_molecular_grid(basis, n_radial, n_theta, n_phi,
                                        use_becke=use_becke,
                                        lebedev_order=lebedev_order)

    diis_solver = DIIS(max_size=8) if diis else None

    converged_iter = -1
    last_residual = float("inf")
    for it in range(max_iter):
        J = coulomb_J_matrix(rho, basis, grid=grid, psi=psi)
        if mode == "hf":
            K = exchange_K_matrix(rho, basis, grid=grid, psi=psi, symmetry="sym")
            F = H_core + J - 0.5 * K
        else:
            F = H_core + J
        F = 0.5 * (F + F.T)

        # DIIS extrapolation
        diis_err = None
        if diis_solver is not None and it >= diis_start:
            diis_err = diis_solver.add(F, rho, S)
            F_use = diis_solver.extrapolate()
        else:
            F_use = F

        eigvals, c = solve_mo(F_use, S)
        rho_new = build_rho(c)
        last_residual = float(np.abs(rho_new - rho).max())

        if diis_solver is not None and it >= diis_start:
            # Trust DIIS extrapolation directly
            rho = rho_new
        else:
            rho = damping * rho + (1.0 - damping) * rho_new

        if verbose:
            extra = f" diis_err={diis_err:.2e}" if diis_err is not None else ""
            print(f"  iter {it}: max|drho|={last_residual:.4e}, "
                  f"HOMO={eigvals[max(n_doubly-1, 0)]:.3f}{extra}")

        if last_residual < tol:
            converged_iter = it
            break

    return rho, S, eigvals, c, converged_iter, last_residual


# ─────────────────────────────────────────────────────────────────────
# Coupled-Perturbed Hartree-Fock for magnetic perturbation
# ─────────────────────────────────────────────────────────────────────


def coupled_cphf_response(
    basis: PTMolecularBasis,
    mo_eigvals: np.ndarray,
    mo_coeffs: np.ndarray,
    n_e_total: int,
    *,
    use_giao: bool = True,
    max_iter: int = 30,
    tol: float = 1.0e-5,
    damping: float = 0.5,
    level_shift: float = 0.0,
    n_radial_grid: int = 24,
    n_theta_grid: int = 12,
    n_phi_grid: int = 16,
    n_radial_op: int = 30,
    n_theta_op: int = 14,
    n_phi_op: int = 18,
    use_becke: bool = False,
    lebedev_order: int = 50,
    verbose: bool = False,
) -> np.ndarray:
    """Solve the coupled-perturbed HF equations for magnetic perturbation
    H^(1)_beta = L_beta. Returns U_imag of shape (3, n_virt, n_occ) such
    that U_imag[beta, a, i] is the response amplitude.

    The CPHF equations for closed-shell magnetic perturbation:

        (eps_a - eps_i) U_ai + Sum_{bj} A_ai,bj U_bj = -L_beta_ai (in MO basis)

    where the orbital Hessian is A = J^(1) - (1/2) K^(1). For purely
    imaginary B*L perturbation, the response density rho^(1) is
    antisymmetric, hence J^(1) = 0. Only K^(1) couples.

    Iterative scheme:
        1. U^(0) = -L_beta_MO_ai / (eps_a - eps_i)              (uncoupled)
        2. Build rho^(1)_AO_imag from U^(k)
        3. Compute K^(1)_AO_imag = K[rho^(1)_AO_imag] (antisym mode)
        4. Transform K^(1) to MO basis
        5. Update U^(k+1)_ai = -(L_beta_ai_MO + 0.5 * K^(1)_ai_MO)
                                    / (eps_a - eps_i)            (coupled)
        6. Damp and iterate to convergence.

    Returns U_imag (3, n_virt, n_occ).
    """
    from ptc.lcao.giao import angular_momentum_matrices_GIAO

    n_orb = mo_coeffs.shape[0]
    n_occ = n_e_total // 2
    n_virt = n_orb - n_occ
    if n_occ == 0 or n_virt == 0:
        return np.zeros((3, n_virt, n_occ))

    # GIAO L_beta matrices in AO and MO bases
    L_imag_AO = angular_momentum_matrices_GIAO(
        basis,
        n_radial=n_radial_op, n_theta=n_theta_op, n_phi=n_phi_op,
        use_becke=use_becke, lebedev_order=lebedev_order,
    )
    L_imag_MO = np.array([
        mo_coeffs.T @ L_imag_AO[k] @ mo_coeffs for k in range(3)
    ])

    eps_a = mo_eigvals[n_occ:]
    eps_i = mo_eigvals[:n_occ]
    diff = eps_a[None, :] - eps_i[:, None]   # (n_occ, n_virt)
    if level_shift > 0.0:
        # Avoid 1/0 in U = -L/diff when frontier MOs are (near-)degenerate.
        diff = np.maximum(diff, level_shift)

    c_occ = mo_coeffs[:, :n_occ]
    c_virt = mo_coeffs[:, n_occ:]

    grid, psi = _build_molecular_grid(basis, n_radial_grid, n_theta_grid, n_phi_grid,
                                        use_becke=use_becke,
                                        lebedev_order=lebedev_order)

    U_imag = np.zeros((3, n_virt, n_occ))

    for beta in range(3):
        # Uncoupled initial guess
        # Convention: U_imag[a, i] s.t. rho^(1)_AO is antisym in (mu, nu)
        # rho^(1) = 2 sum_ai U (c_a c_i^T - c_i c_a^T)
        # CPHF: U_ai = -L_imag_MO[beta, a, i] / (eps_a - eps_i)
        # (negative sign because matrix elements are -i L_imag and
        #  U should produce positive contribution to ρ^(1).)
        L_ai = L_imag_MO[beta, n_occ:, :n_occ]   # (n_virt, n_occ) shape [a, i]
        # diff has shape (n_occ, n_virt) -> need diff.T = (n_virt, n_occ)
        U = -L_ai / diff.T

        for it in range(max_iter):
            # Build rho^(1)_AO (real antisymmetric) from U
            # rho^(1)[mu, nu] = 2 sum_{a, i} U[a, i] (c_virt[mu, a] c_occ[nu, i]
            #                                          - c_occ[mu, i] c_virt[nu, a])
            rho1 = 2.0 * (c_virt @ U @ c_occ.T - c_occ @ U.T @ c_virt.T)

            # K^(1) for antisym density
            K1_AO = exchange_K_matrix(
                rho1, basis, grid=grid, psi=psi, symmetry="antisym",
            )
            K1_MO = mo_coeffs.T @ K1_AO @ mo_coeffs   # (n_orb, n_orb)
            K1_ai = K1_MO[n_occ:, :n_occ]              # (n_virt, n_occ)

            # CPHF update: U_ai = -(L_ai - 0.5 K^(1)_ai) / (eps_a - eps_i)
            # Equivalent: F^(1) = -K^(1)/2, U_ai = -(L_ai + F^(1)_ai) / diff
            # = -(L_ai - 0.5 K^(1)_ai) / diff
            U_new = -(L_ai - 0.5 * K1_ai) / diff.T

            change = float(np.abs(U_new - U).max())
            U = damping * U + (1.0 - damping) * U_new

            if verbose:
                print(f"  CPHF beta={beta} iter {it}: max|dU|={change:.4e}")

            if change < tol:
                break

        U_imag[beta] = U

    return U_imag


def paramagnetic_shielding_tensor_coupled(
    basis: PTMolecularBasis,
    mo_eigvals: np.ndarray,
    mo_coeffs: np.ndarray,
    n_e_total: int,
    K_probe: np.ndarray,
    *,
    use_giao: bool = True,
    max_iter: int = 30,
    tol: float = 1.0e-5,
    n_radial: int = 30,
    n_theta: int = 14,
    n_phi: int = 18,
    use_becke: bool = False,
    lebedev_order: int = 50,
    verbose: bool = False,
) -> np.ndarray:
    """Coupled-CPHF paramagnetic shielding TENSOR sigma^p_alpha_beta(K) in ppm.

    Phase 5c. Returns the full 3 x 3 paramagnetic tensor (not just isotropic).
    The off-diagonal alpha != beta couplings are essential to recover the
    correct sign of sigma^p on aromatic ring currents: the uncoupled CP-PT
    drops them, leading to the documented benzene sign-flip artefact.

    Formula in MO basis:
        sigma^p_alpha,beta(K) = (4 alpha^2) sum_ai
                                 U_imag[beta, a, i] * M_imag[alpha, i, a]
        x A_BOHR^4 x HARTREE_eV    (unit conversions; see comment below)

    where U_imag is the CPHF response amplitude for perturbation L_beta,
    and M_imag[alpha] is the Ramsey magnetic-dipole operator at probe K.
    """
    from ptc.constants import A_BOHR, ALPHA_PHYS
    from ptc.lcao.giao import magnetic_dipole_matrices

    n_orb = mo_coeffs.shape[0]
    n_occ = n_e_total // 2
    n_virt = n_orb - n_occ
    if n_occ == 0 or n_virt == 0:
        return np.zeros((3, 3))

    K_probe = np.asarray(K_probe, dtype=float)

    # Solve CPHF for U (3, n_virt, n_occ)
    U_imag = coupled_cphf_response(
        basis, mo_eigvals, mo_coeffs, n_e_total,
        use_giao=use_giao, max_iter=max_iter, tol=tol,
        n_radial_op=n_radial, n_theta_op=n_theta, n_phi_op=n_phi,
        use_becke=use_becke, lebedev_order=lebedev_order,
        verbose=verbose,
    )

    # Magnetic-dipole operator at probe (3, n_orb, n_orb)
    M_imag_AO = magnetic_dipole_matrices(
        basis, K_probe,
        n_radial=n_radial, n_theta=n_theta, n_phi=n_phi,
        use_becke=use_becke, lebedev_order=lebedev_order,
    )
    M_imag_MO = np.array([
        mo_coeffs.T @ M_imag_AO[k] @ mo_coeffs for k in range(3)
    ])

    # Unit conversions (see paramagnetic_shielding_iso_coupled for derivation):
    # U has units 1/(A * eV); M has units 1/A^3. To atomic units:
    # multiply by A_BOHR (for L unit in U) x A_BOHR^3 (for M) = A_BOHR^4
    # times HARTREE_eV to convert eps^-1 from 1/eV to 1/Hartree.
    HARTREE_eV = 27.211386245988
    unit_factor = (A_BOHR ** 4) * HARTREE_eV

    sigma_p = np.zeros((3, 3))
    for alpha in range(3):
        M_ia = M_imag_MO[alpha, :n_occ, n_occ:]   # (n_occ, n_virt)
        for beta in range(3):
            U_ai = U_imag[beta]                    # (n_virt, n_occ)
            prod = U_ai.T * M_ia                   # element-wise (n_occ, n_virt)
            # σ^p = -4α² Σ U·M (the minus comes from U = -L/Δ; equivalent
            # to the uncoupled +4α² ΣL·M/Δ when U is the uncoupled response.)
            sigma_p[alpha, beta] = (
                -4.0 * (ALPHA_PHYS ** 2) * float(prod.sum()) * unit_factor
            )
    return sigma_p * 1.0e6   # ppm


def paramagnetic_shielding_iso_coupled(
    basis: PTMolecularBasis,
    mo_eigvals: np.ndarray,
    mo_coeffs: np.ndarray,
    n_e_total: int,
    K_probe: np.ndarray,
    *,
    use_giao: bool = True,
    max_iter: int = 30,
    tol: float = 1.0e-5,
    n_radial: int = 30,
    n_theta: int = 14,
    n_phi: int = 18,
    verbose: bool = False,
) -> float:
    """Coupled-CPHF paramagnetic isotropic shielding sigma^p_iso(K) (ppm).

    Uses the Ramsey magnetic-dipole operator at probe K_probe and the
    CPHF-converged response amplitudes U^beta. The formula in MO basis:

        sigma^p_alpha,beta(K) = -(2/c^2) Re sum_ai
                                  [<i|L_beta|a><a|L_alpha^K/r_K^3|i> + c.c.]
                                  / (eps_a - eps_i)

    With CPHF U replacing the uncoupled denominator: equivalently

        sigma^p_alpha,beta = -2 alpha^2 sum_ai
                               [U_imag^beta_ai * M_imag_alpha_ia + c.c.]

    Note the factor of 2 (vs 4 in the uncoupled formula) because we
    have already absorbed the eps difference into U.
    """
    from ptc.constants import A_BOHR, ALPHA_PHYS
    from ptc.lcao.giao import magnetic_dipole_matrices

    n_orb = mo_coeffs.shape[0]
    n_occ = n_e_total // 2
    n_virt = n_orb - n_occ
    if n_occ == 0 or n_virt == 0:
        return 0.0

    K_probe = np.asarray(K_probe, dtype=float)

    # Solve CPHF for U
    U_imag = coupled_cphf_response(
        basis, mo_eigvals, mo_coeffs, n_e_total,
        use_giao=use_giao, max_iter=max_iter, tol=tol,
        n_radial_op=n_radial, n_theta_op=n_theta, n_phi_op=n_phi,
        verbose=verbose,
    )

    # Build magnetic-dipole operator at probe
    M_imag_AO = magnetic_dipole_matrices(
        basis, K_probe,
        n_radial=n_radial, n_theta=n_theta, n_phi=n_phi,
    )
    M_imag_MO = np.array([
        mo_coeffs.T @ M_imag_AO[k] @ mo_coeffs for k in range(3)
    ])

    # Convert units: zeta in 1/Angstrom -> 1/a_0 (x A_BOHR for L, x A_BOHR^3 for M).
    # U was solved in eV-based eps differences, so its MO-energy unit is eV.
    # The factor that brings everything to atomic units:
    A_BOHR_3 = A_BOHR ** 3
    # sigma^p_alpha,beta = (4 alpha^2) sum_ai U_imag[beta, a, i] *
    #                     M_imag[alpha, i, a] * (A_BOHR^3 for M; A_BOHR for L was
    #                     implicitly included when we used MO eigvalues in eV)
    #
    # Rederivation: in atomic units the formula has factor 4 alpha^2, with
    # eps in Hartree. Here U has 1/(eps in eV). Each L_imag unit is 1/Ang.
    # Conversion: U_imag has units (1/A) / eV = 1/(A * eV).
    # M_imag has units 1/A^3. Conversion to a.u.: A^-1 * A_BOHR = dimensionless,
    # A^-3 * A_BOHR^3 = dimensionless.
    # eps differences: 1/eV -> 1/Hartree means multiply by HARTREE_eV = 27.21.
    #
    # Net: sigma_iso = (4 alpha^2 / 3) sum_alpha U_imag[alpha, a, i] M_imag[alpha, i, a]
    #                  * A_BOHR * A_BOHR^3 * HARTREE_eV
    HARTREE_eV = 27.211386245988

    sigma_p_diag = np.zeros(3)
    for alpha in range(3):
        M_ia = M_imag_MO[alpha, :n_occ, n_occ:]      # (n_occ, n_virt)
        U_ai = U_imag[alpha]                          # (n_virt, n_occ)
        prod = U_ai.T * M_ia                          # (n_occ, n_virt)
        # σ^p_αα = -4α² Σ U·M (minus = U absorbed -L/Δ: identical magnitude
        # to uncoupled +4α² ΣL·M/Δ but opposite sign in U-form, matching
        # the documented coupled formula above).
        sigma_p_diag[alpha] = (
            -4.0 * (ALPHA_PHYS ** 2)
            * float(prod.sum())
            * A_BOHR * A_BOHR_3 * HARTREE_eV
        )
    return float(sigma_p_diag.mean()) * 1.0e6   # ppm
