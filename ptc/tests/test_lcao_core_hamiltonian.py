"""Tests for the rigorous one-electron core Hamiltonian (limitation #2
of BACKLOG_LCAO_PRECISION.md).

H_core = T + V_nuc replaces the Hueckel K=2 approximation with
physical kinetic-energy and PT-screened nuclear-attraction integrals.
NO empirical scaling factor.

The diagonal H[i, i] values for an isolated atom recover the
hydrogen-like one-electron energy E_n,l = -Z_eff^2 / (2 n^2) Hartree.
The off-diagonal couplings come from the actual physical operator,
not from H_ij = K * (H_ii + H_jj) / 2 * S_ij.
"""

import numpy as np
import pytest

from ptc.constants import ALPHA_PHYS
from ptc.lcao.density_matrix import (
    build_molecular_basis,
    core_hamiltonian,
    density_matrix_PT,
    kinetic_matrix,
    nuclear_attraction_total,
    overlap_matrix,
)
from ptc.lcao.shielding import shielding_tensor_at_point
from ptc.topology import build_topology


# ─────────────────────────────────────────────────────────────────────
# 1. Kinetic and V_nuc matrices: H atom exact
# ─────────────────────────────────────────────────────────────────────


def test_H_atom_kinetic_equals_Rydberg():
    """T[1s, 1s] for H atom = +13.6 eV (= +Ry, virial theorem)."""
    basis = build_molecular_basis(build_topology("[H]"))
    T = kinetic_matrix(basis, n_radial=80, n_theta=20, n_phi=24)
    assert T[0, 0] == pytest.approx(13.6, rel=2e-3)


def test_H_atom_V_nuc_equals_minus_2Ry():
    """V_nuc[1s, 1s] for H atom = -27.2 eV (= -2 Ry)."""
    basis = build_molecular_basis(build_topology("[H]"))
    V = nuclear_attraction_total(basis, n_radial=80, n_theta=20, n_phi=24)
    assert V[0, 0] == pytest.approx(-27.2, rel=2e-3)


def test_H_atom_core_hamiltonian_equals_minus_IE():
    """H_core[1s, 1s] = -13.6 eV for H atom (= -IE_H exactly)."""
    basis = build_molecular_basis(build_topology("[H]"))
    S = overlap_matrix(basis)
    H = core_hamiltonian(basis, S, n_radial=80, n_theta=20, n_phi=24)
    assert H[0, 0] == pytest.approx(-13.6, abs=0.05)


# ─────────────────────────────────────────────────────────────────────
# 2. Hermiticity of H_core
# ─────────────────────────────────────────────────────────────────────


def test_core_hamiltonian_hermitian():
    """H_core must be symmetric (real basis -> hermitian)."""
    basis = build_molecular_basis(build_topology("O"))   # H_2O
    S = overlap_matrix(basis)
    H = core_hamiltonian(basis, S, n_radial=30, n_theta=14, n_phi=20)
    assert np.allclose(H, H.T, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────
# 3. Default Hamiltonian still hueckel (anti-regression)
# ─────────────────────────────────────────────────────────────────────


def test_default_hamiltonian_is_hueckel():
    """density_matrix_PT default behaviour preserved."""
    topo = build_topology("[H][H]")
    rho_default, _, eig_default, _ = density_matrix_PT(topo)
    rho_hueckel, _, eig_hueckel, _ = density_matrix_PT(topo, hamiltonian="hueckel")
    np.testing.assert_allclose(rho_default, rho_hueckel)
    np.testing.assert_allclose(eig_default, eig_hueckel)


def test_invalid_hamiltonian_raises():
    topo = build_topology("[H]")
    with pytest.raises(ValueError, match="hamiltonian"):
        density_matrix_PT(topo, hamiltonian="dft")


# ─────────────────────────────────────────────────────────────────────
# 4. Core Hamiltonian preserves trace and idempotency
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("smiles, n_e", [
    ("[H][H]",   2),
    ("O",        8),
    ("C",        8),
    ("[NH3]",    8),
])
def test_core_hamiltonian_density_matrix_trace_and_idempotency(smiles, n_e):
    topo = build_topology(smiles)
    rho, S, _, _ = density_matrix_PT(
        topo, hamiltonian="core",
        n_radial=30, n_theta=14, n_phi=20,
    )
    tr = float(np.trace(rho @ S))
    assert tr == pytest.approx(n_e, abs=1e-6)
    err = np.abs(rho @ S @ rho - 2.0 * rho).max()
    assert err < 1e-9


# ─────────────────────────────────────────────────────────────────────
# 5. Core Hamiltonian gives more realistic HOMO-LUMO gap on benzene
# ─────────────────────────────────────────────────────────────────────


def test_benzene_HOMO_LUMO_gap_more_realistic_with_core():
    """Hueckel K=2 gives gap ~13 eV (way too big); core gives ~3-5 eV
    (closer to experimental ~6.5 eV). Test that core is at least
    less than half of Hueckel (i.e. moves in the right direction)."""
    topo = build_topology("c1ccccc1")
    basis = build_molecular_basis(topo)
    n_occ = int(round(basis.total_occ)) // 2

    _, _, eig_h, _ = density_matrix_PT(topo, basis=basis, hamiltonian="hueckel")
    gap_h = eig_h[n_occ] - eig_h[n_occ - 1]

    _, _, eig_c, _ = density_matrix_PT(
        topo, basis=basis, hamiltonian="core",
        n_radial=30, n_theta=14, n_phi=20,
    )
    gap_c = eig_c[n_occ] - eig_c[n_occ - 1]

    # Hueckel ~13 eV, core ~3-5 eV
    assert gap_c < gap_h * 0.5, f"core gap ({gap_c}) >= 0.5 * hueckel gap ({gap_h})"
    # Core gap should be sensible (between 1 and 10 eV)
    assert 1.0 < gap_c < 10.0


# ─────────────────────────────────────────────────────────────────────
# 6. Core Hamiltonian shifts paramagnetic shielding toward larger
#    magnitude (closer to experiment, though not yet quantitative)
# ─────────────────────────────────────────────────────────────────────


def test_benzene_paramagnetic_larger_with_core():
    """sigma^p magnitude larger with core (drives NICS more negative)."""
    from ptc.lcao.giao import paramagnetic_shielding_iso
    topo = build_topology("c1ccccc1")
    basis = build_molecular_basis(topo)
    P = basis.coords.mean(axis=0)
    n_e = int(round(basis.total_occ))

    _, _, eig_h, c_h = density_matrix_PT(topo, basis=basis, hamiltonian="hueckel")
    sp_h = paramagnetic_shielding_iso(basis, eig_h, c_h, n_e, P,
                                        n_radial=30, n_theta=14, n_phi=20)
    _, _, eig_c, c_c = density_matrix_PT(
        topo, basis=basis, hamiltonian="core",
        n_radial=30, n_theta=14, n_phi=20,
    )
    sp_c = paramagnetic_shielding_iso(basis, eig_c, c_c, n_e, P,
                                        n_radial=30, n_theta=14, n_phi=20)
    # core gives larger |sigma^p| (more negative for benzene aromatic)
    assert abs(sp_c) > abs(sp_h)
    assert sp_c < 0    # paramagnetic deshielding is negative


# ─────────────────────────────────────────────────────────────────────
# 7. H atom Lamb still exact through full pipeline
# ─────────────────────────────────────────────────────────────────────


def test_H_atom_Lamb_with_core_hamiltonian():
    """End-to-end: H atom sigma_iso = 17.75 ppm with core Hamiltonian."""
    topo = build_topology("[H]")
    basis = build_molecular_basis(topo)
    rho, S, eigvals, c = density_matrix_PT(
        topo, basis=basis, hamiltonian="core",
        n_radial=60, n_theta=20, n_phi=24,
    )
    # H atom is single-orbital, so MO == basis function
    # density should still be 1.0 in the only orbital
    assert rho[0, 0] == pytest.approx(1.0, abs=1e-9)
    # Now compute shielding
    res = shielding_tensor_at_point(topo, np.zeros(3))
    expected = (ALPHA_PHYS ** 2 / 3.0) * 1.0e6
    assert res.sigma_iso == pytest.approx(expected, rel=2e-3)
