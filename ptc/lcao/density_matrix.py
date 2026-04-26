"""Phase B: PT-LCAO molecular density matrix.

Assembles the per-molecule overlap matrix S, builds an effective
Hueckel Hamiltonian H_eff (Mulliken / K=1 Wolfsberg-Helmholz, parameter
free), solves the generalised eigenvalue H c = epsilon S c, and
constructs the closed-shell Aufbau density matrix

    rho[i, j] = sum_k n_k c_ki c_kj

with n_k = 2 for occupied MOs (one half-occupation for radicals).

PT-purity
=========
The current Phase B commit uses the **Mulliken approximation** for the
off-diagonal elements of H_eff:

    H_ij = (H_ii + H_jj) / 2 * S_ij

This is parameter-free (the standard Wolfsberg-Helmholz constant
K = 1.75 is replaced by K = 1, an arithmetic mean -- no fit). The
diagonal H_ii uses -IE_eV(Z_host) from atom.py (PT-pure to s = 1/2).

The full T3-derived Hamiltonian (the spec's "reuse transfer_matrix
infrastructure") is reserved for a Phase B continuation. For the
validation milestones in the spec, that is unnecessary: trace and
idempotency of rho follow from MO orthonormalisation alone, regardless
of the off-diagonal Hueckel choice.

Coverage
========
Restricted to s and p valence orbitals, matching Phase A. Molecules with
d / f valence (transition metals, lanthanides, actinides) raise a clean
NotImplementedError telling the user that Phase A continuation must
land first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from ptc.atom import IE_eV
from ptc.geometry import compute_geometry_3d
from ptc.lcao.atomic_basis import (
    PTAtomBasis,
    PTAtomicOrbital,
    build_atom_basis,
    overlap_atomic,
)
from ptc.periodic import block_of
from ptc.topology import Topology


# ─────────────────────────────────────────────────────────────────────
# Molecular basis
# ─────────────────────────────────────────────────────────────────────


@dataclass
class PTMolecularBasis:
    """Flat molecular orbital basis assembled from per-atom PTAtomBasis.

    Attributes
    ----------
    atoms       : per-atom PTAtomBasis list (length = N_atoms)
    coords      : ndarray of shape (N_atoms, 3), Angstrom
    orbitals    : flat list of PTAtomicOrbital across all atoms
    atom_index  : same length as `orbitals`; gives the host-atom index
                  in `atoms` and `coords`.
    Z_list      : the atomic numbers, parallel to atoms / coords
    """

    atoms: List[PTAtomBasis]
    coords: np.ndarray
    orbitals: List[PTAtomicOrbital] = field(default_factory=list)
    atom_index: List[int] = field(default_factory=list)
    Z_list: List[int] = field(default_factory=list)

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def n_orbitals(self) -> int:
        return len(self.orbitals)

    @property
    def total_occ(self) -> float:
        return sum(o.occ for o in self.orbitals)


def build_molecular_basis(topology: Topology,
                            polarisation: bool = False) -> PTMolecularBasis:
    """Construct the LCAO basis for a molecule.

    Steps
    -----
    1. Reject f-block atoms (Phase A continuation gate).
    2. Get 3D coordinates via `geometry.compute_geometry_3d`.
    3. For each atom, assemble PTAtomBasis honouring its formal charge
       and the optional polarisation flag.
    4. Flatten into a single list of orbitals + parallel atom_index.

    Parameters
    ----------
    polarisation : bool, default False
        Pass through to build_atom_basis. When True, every atom gets one
        shell of polarisation functions (2p on H, 3d on C/N/O/F, 4d on
        Si/P/S/Cl, etc.). Polarisation orbitals are unoccupied in the
        ground state but provide variational flexibility for the GIAO
        response.
    """
    for Z in topology.Z_list:
        if block_of(Z) == "f":
            raise NotImplementedError(
                f"build_molecular_basis: Z={Z} is in the f-block. "
                "PT-LCAO f-orbital overlaps are pending Phase A continuation; "
                "currently s, p, and d valence are supported."
            )

    geom = compute_geometry_3d(topology)
    n = geom.n_atoms
    coords_arr = np.array([geom.coords[i] for i in range(n)])

    atoms: List[PTAtomBasis] = []
    flat_orbs: List[PTAtomicOrbital] = []
    atom_idx: List[int] = []

    for i, Z in enumerate(topology.Z_list):
        charge = topology.charges[i] if i < len(topology.charges) else 0
        atom_basis = build_atom_basis(Z, charge=charge, polarisation=polarisation)
        atoms.append(atom_basis)
        for orb in atom_basis.orbitals:
            flat_orbs.append(orb)
            atom_idx.append(i)

    return PTMolecularBasis(
        atoms=atoms,
        coords=coords_arr,
        orbitals=flat_orbs,
        atom_index=atom_idx,
        Z_list=list(topology.Z_list),
    )


# ─────────────────────────────────────────────────────────────────────
# Overlap matrix
# ─────────────────────────────────────────────────────────────────────


def overlap_matrix(basis: PTMolecularBasis) -> np.ndarray:
    """N x N overlap matrix S[i, j] = <phi_i | phi_j>.

    Uses `overlap_atomic` from `atomic_basis`; symmetric by construction.
    """
    n = basis.n_orbitals
    S = np.zeros((n, n))
    for i in range(n):
        S[i, i] = 1.0  # all PT atomic orbitals are individually normalised
        for j in range(i + 1, n):
            r_ij = (
                basis.coords[basis.atom_index[j]]
                - basis.coords[basis.atom_index[i]]
            )
            s_ij = overlap_atomic(basis.orbitals[i], basis.orbitals[j], r_ij)
            S[i, j] = s_ij
            S[j, i] = s_ij
    return S


# ─────────────────────────────────────────────────────────────────────
# Hueckel-like effective Hamiltonian (Mulliken, K=1)
# ─────────────────────────────────────────────────────────────────────


_HUECKEL_K = 2.0  # PT-integer (S = 1/2 -> doubling), no empirical fit
# rationale: Mulliken K = 1 collapses homonuclear two-orbital generalised
# eigenvalues to a single degenerate level (bonding = antibonding),
# which makes paramagnetic CP-PT diverge. K = 2 is the smallest
# PT-integer choice that breaks the degeneracy without introducing a
# fitted constant. K = 1 + gamma_3 = 1.808 would be the PT cascade
# alternative; we prefer the integer for simplicity.
#
# Phase B continuation (limitation #2 of BACKLOG_LCAO_PRECISION):
# the Hueckel K=2 is replaced by the rigorous one-electron core
# Hamiltonian H = T + V_nuc when hamiltonian="core" is selected.
# That uses physical kinetic-energy and PT-screened nuclear-attraction
# integrals, with NO empirical scaling factor.


def hueckel_hamiltonian(basis: PTMolecularBasis, S: np.ndarray) -> np.ndarray:
    """Effective Hueckel Hamiltonian, parameter-free.

    Diagonal:
        H_ii = -IE_eV(Z_host)  (eV, negative of valence ionisation energy)
    Off-diagonal:
        H_ij = K * (H_ii + H_jj) / 2  *  S_ij      with  K = 2
        (Wolfsberg-Helmholz with K = 2: PT-integer, no empirical fit.)

    Note: K = 1 (pure Mulliken arithmetic mean) would collapse
    homonuclear two-orbital eigenvalues to a single degenerate level,
    making paramagnetic shielding diverge in CP-PT. K = 2 is the
    smallest defensible PT-pure choice that preserves the bonding /
    antibonding splitting required for finite paramagnetic shielding.
    """
    n = basis.n_orbitals
    H = np.zeros((n, n))
    onsite = np.zeros(n)
    for i, orb in enumerate(basis.orbitals):
        onsite[i] = -float(IE_eV(orb.Z))

    for i in range(n):
        H[i, i] = onsite[i]
    for i in range(n):
        for j in range(i + 1, n):
            H[i, j] = _HUECKEL_K * 0.5 * (onsite[i] + onsite[j]) * S[i, j]
            H[j, i] = H[i, j]
    return H


# ─────────────────────────────────────────────────────────────────────
# Generalised eigenvalue solver (Lowdin orthogonalisation)
# ─────────────────────────────────────────────────────────────────────


def _lowdin_orthogonalise(S: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    """Compute S^{-1/2} via spectral decomposition with eigenvalue floor."""
    s_eigvals, s_eigvecs = np.linalg.eigh(S)
    # Defensive floor for tiny numerical negatives
    s_eigvals = np.maximum(s_eigvals, eps)
    inv_sqrt = s_eigvecs @ np.diag(1.0 / np.sqrt(s_eigvals)) @ s_eigvecs.T
    return inv_sqrt


def kinetic_matrix(basis: PTMolecularBasis,
                    n_radial: int = 50,
                    n_theta: int = 18,
                    n_phi: int = 24) -> np.ndarray:
    """Kinetic-energy matrix T[i, j] = <phi_i | -hbar^2/(2m) nabla^2 | phi_j> in eV.

    Uses Green's identity: <phi_i | -nabla^2 | phi_j> = <nabla phi_i | nabla phi_j>
    (orbitals decay -> surface term zero). Numerical 3D Gauss quadrature
    on a spherical grid centred on the molecular centroid.

    Reuses evaluate_sto_gradient and the spherical-grid infrastructure
    from giao.py.
    """
    from ptc.constants import HBAR_C_EV_A, ME_C2_EV
    from ptc.lcao.giao import _build_spherical_grid, evaluate_sto_gradient

    T_PREFAC = (HBAR_C_EV_A ** 2) / (2.0 * ME_C2_EV)   # eV * Angstrom^2

    centroid = basis.coords.mean(axis=0)
    max_dist = max(
        float(np.linalg.norm(centroid - basis.coords[i]))
        for i in range(basis.n_atoms)
    )
    min_zeta = min(float(o.zeta) for o in basis.orbitals)
    R_max = max(max_dist + 8.0 / min_zeta, 12.0 / min_zeta)

    grid = _build_spherical_grid(R_max, n_radial, n_theta, n_phi)
    pts = grid.points + centroid[None, :]
    n = basis.n_orbitals

    grad = np.zeros((n, grid.points.shape[0], 3))
    for k, orb in enumerate(basis.orbitals):
        atom_pos = basis.coords[basis.atom_index[k]]
        grad[k] = evaluate_sto_gradient(orb, pts, atom_pos)

    T = np.zeros((n, n))
    for alpha in range(3):
        T += (grad[:, :, alpha] * grid.weights[None, :]) @ grad[:, :, alpha].T
    T *= T_PREFAC
    T = 0.5 * (T + T.T)
    return T


def nuclear_attraction_total(basis: PTMolecularBasis,
                               n_radial: int = 60,
                               n_theta: int = 24,
                               n_phi: int = 32) -> np.ndarray:
    """V[i, j] = -sum_K Z_eff(K) * <phi_i | 1/|r-R_K| | phi_j>  in eV.

    Sums the Coulomb-attraction integral over all atomic centres K with
    PT-screened effective nuclear charges from atom.effective_charge.
    Reuses giao.nuclear_attraction_matrix(basis, R_K).
    """
    from ptc.atom import effective_charge
    from ptc.constants import COULOMB_EV_A
    from ptc.lcao.giao import nuclear_attraction_matrix

    n = basis.n_orbitals
    V = np.zeros((n, n))
    for K_idx in range(basis.n_atoms):
        R_K = basis.coords[K_idx]
        Z_eff_K = float(effective_charge(basis.Z_list[K_idx]))
        # nuclear_attraction_matrix returns the integral <phi|1/|r-K||phi>
        # in 1/Angstrom; multiply by COULOMB_EV_A (eV*A) for energy.
        M_K = nuclear_attraction_matrix(basis, R_K,
                                          n_radial=n_radial,
                                          n_theta=n_theta,
                                          n_phi=n_phi)
        V -= Z_eff_K * COULOMB_EV_A * M_K
    return V


def core_hamiltonian(basis: PTMolecularBasis,
                      S: np.ndarray | None = None,
                      n_radial: int = 50,
                      n_theta: int = 18,
                      n_phi: int = 24) -> np.ndarray:
    """One-electron core Hamiltonian H = T + V_nuc, in eV.

    Replaces the Hueckel K=2 approximation with rigorous physical
    integrals (kinetic energy + sum-over-nuclei Coulomb attraction
    with PT-screened Z_eff). PT-pure: NO empirical K parameter.

    Limitation #2 of BACKLOG_LCAO_PRECISION.md.
    """
    T = kinetic_matrix(basis, n_radial=n_radial, n_theta=n_theta, n_phi=n_phi)
    # nuclear_attraction grid can be denser since 1/r decays slowly
    V = nuclear_attraction_total(basis,
                                   n_radial=max(n_radial, 60),
                                   n_theta=max(n_theta, 24),
                                   n_phi=max(n_phi, 32))
    H = T + V
    return 0.5 * (H + H.T)


def solve_mo(H: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve H c = epsilon S c, returning (eigvals_sorted, c_columns).

    `c` columns are S-orthonormal: c.T @ S @ c = I.
    """
    try:
        from scipy.linalg import eigh as _eigh   # type: ignore
        eigvals, eigvecs = _eigh(H, S)
        return np.asarray(eigvals), np.asarray(eigvecs)
    except ImportError:
        S_inv_sqrt = _lowdin_orthogonalise(S)
        Hp = S_inv_sqrt @ H @ S_inv_sqrt
        eigvals, u = np.linalg.eigh(Hp)
        c = S_inv_sqrt @ u
        return eigvals, c


# ─────────────────────────────────────────────────────────────────────
# Density matrix
# ─────────────────────────────────────────────────────────────────────


def density_matrix_PT(
    topology: Topology,
    basis: PTMolecularBasis | None = None,
    hamiltonian: str = "hueckel",
    **quad_kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Closed-shell Aufbau density matrix.

    Parameters
    ----------
    hamiltonian : "hueckel" (default) or "core"
        - "hueckel": Mulliken K=2 (parameter-free but crude). Default
          for back-compat with phase-B/C tests.
        - "core": rigorous one-electron core H = T + V_nuc using
          kinetic-energy and PT-screened Coulomb attraction integrals.
          More accurate (no K parameter), at the cost of two 3D
          quadratures per call.

    Returns
    -------
    rho      : (N_orb, N_orb) density matrix with Tr(rho S) = N_electrons
    S        : overlap matrix (returned for convenience)
    mo_eigvals : MO energies (eV), sorted ascending
    mo_coeffs  : MO coefficient matrix; column k is c_k

    Properties (verified in tests):
      Tr(rho @ S) = N_electrons        (trace conservation)
      rho @ S @ rho = 2 * rho          (closed-shell idempotency)
    """
    if basis is None:
        basis = build_molecular_basis(topology)

    S = overlap_matrix(basis)
    if hamiltonian == "core":
        H = core_hamiltonian(basis, S, **quad_kwargs)
    elif hamiltonian == "hueckel":
        H = hueckel_hamiltonian(basis, S)
    else:
        raise ValueError(
            f"hamiltonian must be 'hueckel' or 'core', got {hamiltonian!r}"
        )
    eigvals, c = solve_mo(H, S)

    n_e_total = int(round(basis.total_occ))
    n_doubly = n_e_total // 2

    rho = np.zeros_like(S)
    for k in range(n_doubly):
        rho += 2.0 * np.outer(c[:, k], c[:, k])
    if n_e_total % 2 == 1 and n_doubly < c.shape[1]:
        rho += 1.0 * np.outer(c[:, n_doubly], c[:, n_doubly])

    return rho, S, eigvals, c


# ─────────────────────────────────────────────────────────────────────
# Convenience: Mulliken atomic populations
# ─────────────────────────────────────────────────────────────────────


def mulliken_populations(
    rho: np.ndarray, S: np.ndarray, basis: PTMolecularBasis,
) -> np.ndarray:
    """Mulliken atomic populations from rho and S.

    population[A] = sum_{i in A} (rho @ S)[i, i]
    Sum equals N_electrons.
    """
    rs = rho @ S
    pops = np.zeros(basis.n_atoms)
    for i, atom_i in enumerate(basis.atom_index):
        pops[atom_i] += rs[i, i]
    return pops
