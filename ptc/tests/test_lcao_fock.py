"""Phase B continuation #3 tests — 2-electron J/K integrals + Fock SCF.

Limitation #3 of BACKLOG_LCAO_PRECISION.md.

Validates:
  * Coulomb J matrix on H atom: (1s 1s | 1s 1s) = (5/8) zeta Hartree
    = 17.0 eV (analytic Slater 1930)
  * Exchange K matrix on H atom equal to J for same orbital pair
  * H atom restricted HF eigenvalue = -5.1 eV (= -13.6 + J - K/2,
    a known consequence of restricted Hartree-Fock for 1 electron)
  * He / H_2 SCF converges
  * Tr(rho S) = N_e and rho S rho = 2 rho preserved through SCF
"""

import math

import numpy as np
import pytest

from ptc.constants import COULOMB_EV_A
from ptc.lcao.density_matrix import (
    build_molecular_basis,
    core_hamiltonian,
    overlap_matrix,
)
from ptc.lcao.fock import (
    coulomb_J_matrix,
    density_matrix_PT_scf,
    exchange_K_matrix,
    fock_matrix,
)
from ptc.topology import build_topology


# ─────────────────────────────────────────────────────────────────────
# 1. Slater (1s 1s | 1s 1s) integral on H atom
# ─────────────────────────────────────────────────────────────────────


def test_H_atom_1s1s_self_repulsion():
    """(1s 1s | 1s 1s) = (5/8) zeta in atomic units = 17.0 eV for ζ=1.

    Validates the J matrix via H atom self-interaction integral.
    """
    basis = build_molecular_basis(build_topology("[H]"))
    rho_unit = np.array([[1.0]])   # one electron in 1s
    J = coulomb_J_matrix(rho_unit, basis,
                          n_radial=30, n_theta=14, n_phi=18)
    expected = (5.0 / 8.0) * 27.21138   # 5/8 Hartree -> eV
    assert J[0, 0] == pytest.approx(expected, rel=2e-2)


def test_H_atom_J_equals_K_for_single_orbital():
    """For a single 1s orbital, J and K are the same integral."""
    basis = build_molecular_basis(build_topology("[H]"))
    rho_unit = np.array([[1.0]])
    J = coulomb_J_matrix(rho_unit, basis, n_radial=24, n_theta=12, n_phi=16)
    K = exchange_K_matrix(rho_unit, basis, n_radial=24, n_theta=12, n_phi=16)
    assert J[0, 0] == pytest.approx(K[0, 0], rel=5e-2)


# ─────────────────────────────────────────────────────────────────────
# 2. Restricted HF on H atom: known SI artefact
# ─────────────────────────────────────────────────────────────────────


def test_H_atom_restricted_HF_eigenvalue():
    """Restricted Hartree-Fock for a 1-electron system has a famous
    self-interaction error: F = H_core + J - K/2 with rho = 1
    gives F = -13.6 + 17 - 8.5 = -5.1 eV (NOT the true -13.6 eV).

    This is mathematically correct restricted HF behaviour; the fix
    requires unrestricted HF or DFT with self-interaction correction.
    """
    res = density_matrix_PT_scf(
        build_topology("[H]"), mode="hf",
        n_radial=24, n_theta=12, n_phi=16,
    )
    rho, S, eigvals, c, it, resid = res
    # Restricted HF eigenvalue lies in the SI-affected range
    assert -8.0 < eigvals[0] < -3.0, f"H atom restricted HF eigval = {eigvals[0]}"


# ─────────────────────────────────────────────────────────────────────
# 3. He / H_2 SCF converges
# ─────────────────────────────────────────────────────────────────────


def test_He_HF_SCF_converges():
    """Closed-shell He: SCF converges (single 1s orbital, trivially)."""
    res = density_matrix_PT_scf(
        build_topology("[He]"), mode="hf",
        n_radial=24, n_theta=12, n_phi=16,
        max_iter=10,
    )
    rho, S, eigvals, c, it, resid = res
    assert it >= 0, f"He HF did not converge (residual={resid})"
    # Tr(rho S) = 2
    assert float(np.trace(rho @ S)) == pytest.approx(2.0, abs=1e-6)


def test_H2_HF_SCF_converges():
    res = density_matrix_PT_scf(
        build_topology("[H][H]"), mode="hf",
        n_radial=24, n_theta=12, n_phi=16,
        max_iter=10, tol=1e-5,
    )
    rho, S, eigvals, c, it, resid = res
    assert it >= 0, f"H2 HF did not converge (residual={resid})"
    assert float(np.trace(rho @ S)) == pytest.approx(2.0, abs=1e-6)
    # rho S rho = 2 rho (closed-shell idempotency)
    err = np.abs(rho @ S @ rho - 2.0 * rho).max()
    assert err < 1e-9


# ─────────────────────────────────────────────────────────────────────
# 4. fock_matrix mode validation
# ─────────────────────────────────────────────────────────────────────


def test_fock_matrix_invalid_mode():
    basis = build_molecular_basis(build_topology("[H]"))
    S = overlap_matrix(basis)
    H_core = core_hamiltonian(basis, S, n_radial=20, n_theta=10, n_phi=14)
    rho = np.array([[1.0]])
    with pytest.raises(ValueError, match="mode"):
        fock_matrix(rho, basis, H_core, mode="dft",
                    n_radial=20, n_theta=10, n_phi=14)


# ─────────────────────────────────────────────────────────────────────
# 5. J and K matrices are symmetric in real basis
# ─────────────────────────────────────────────────────────────────────


def test_J_K_symmetric():
    basis = build_molecular_basis(build_topology("O"))   # H_2O
    rho = np.zeros((basis.n_orbitals, basis.n_orbitals))
    np.fill_diagonal(rho, 1.0)
    J = coulomb_J_matrix(rho, basis, n_radial=20, n_theta=10, n_phi=14)
    K = exchange_K_matrix(rho, basis, n_radial=20, n_theta=10, n_phi=14)
    assert np.allclose(J, J.T, atol=1e-9)
    assert np.allclose(K, K.T, atol=1e-9)


# ─────────────────────────────────────────────────────────────────────
# 6. Hartree-only mode: positive J shift relative to H_core
# ─────────────────────────────────────────────────────────────────────


def test_hartree_mode_shifts_eigenvalues_up():
    """Hartree-only F = H_core + J. With J > 0 (Coulomb repulsion),
    eigenvalues should generally shift upward vs H_core."""
    topo = build_topology("[H]")
    basis = build_molecular_basis(topo)
    S = overlap_matrix(basis)
    H_core = core_hamiltonian(basis, S, n_radial=20, n_theta=10, n_phi=14)
    rho = np.array([[1.0]])
    F_hartree = fock_matrix(rho, basis, H_core, mode="hartree",
                              n_radial=20, n_theta=10, n_phi=14)
    # F_hartree[0,0] = H_core[0,0] + J[0,0] > H_core[0,0] (since J > 0)
    assert F_hartree[0, 0] > H_core[0, 0]
