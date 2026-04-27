"""Tests for ptc.lcao.density_matrix (Phase B).

Validation milestones from PROMPT_PT_LCAO_GIAO.md Section 3 Phase B:
  * rho_PT idempotence at half-occupation : rho S rho = (n_max) * rho
    for closed-shell electrons (test paired-electron molecules)
  * Trace(rho S) = N_electrons  +/- 1e-6 for ~20 test molecules
  * Mulliken populations sum to N_electrons; charges coherent with topology

Phase B coverage is restricted to s/p valence (Phase A scope); d/f
elements raise NotImplementedError, tested separately.
"""

import math

import numpy as np
import pytest

from ptc.lcao.density_matrix import (
    build_molecular_basis,
    density_matrix_PT,
    hueckel_hamiltonian,
    mulliken_populations,
    overlap_matrix,
    solve_mo,
)
from ptc.topology import build_topology


# Small set of s/p molecules with known electron counts
SP_TEST_MOLECULES = [
    ("[H][H]",      "H2",       2),     # H_2 : 2 valence e-
    ("O",           "H2O",      8),     # H_2O: 6 + 2 = 8
    ("[NH3]",       "NH3",      8),     # N + 3H = 5 + 3 = 8
    ("C",           "CH4",      8),     # C + 4H = 4 + 4 = 8
    ("CC",          "C2H6",     14),    # 2C + 6H = 8 + 6 = 14
    ("C=C",         "C2H4",     12),    # 2C + 4H = 8 + 4 = 12
    ("C#C",         "C2H2",     10),    # 2C + 2H = 8 + 2 = 10
    ("[F][F]",      "F2",       14),    # 2F = 14
    ("N#N",         "N2",       10),    # 2N = 10
    ("O=O",         "O2",       12),    # 2O = 12
    ("C=O",         "CH2O",     12),    # C + 2H + O = 4 + 2 + 6 = 12
    ("CO",          "CH3OH",    14),    # C + 4H + O = 4 + 4 + 6 = 14
    ("[Cl][H]",     "HCl",      8),     # Cl + H = 7 + 1 = 8
    ("S",           "H2S",      8),     # S + 2H = 6 + 2 = 8
    ("P",           "PH3",      8),     # P + 3H = 5 + 3 = 8
    ("[SiH4]",      "SiH4",     8),     # Si + 4H = 4 + 4 = 8
    ("CN",          "CH3NH2",   14),    # C + 5H + N = 4 + 5 + 5 = 14
    ("c1ccccc1",    "C6H6",     30),    # 6C + 6H = 24 + 6 = 30
    ("OC=O",        "HCOOH",    18),    # 2O + C + 2H = 12 + 4 + 2 = 18
    ("O=C=O",       "CO2",      16),    # C + 2O = 4 + 12 = 16
]


# ─────────────────────────────────────────────────────────────────────
# 1. PTMolecularBasis assembly
# ─────────────────────────────────────────────────────────────────────


def test_build_molecular_basis_H2():
    basis = build_molecular_basis(build_topology("[H][H]"))
    assert basis.n_atoms == 2
    assert basis.n_orbitals == 2
    assert basis.coords.shape == (2, 3)
    assert basis.total_occ == pytest.approx(2.0)
    assert basis.atom_index == [0, 1]


def test_build_molecular_basis_H2O():
    basis = build_molecular_basis(build_topology("O"))
    assert basis.n_atoms == 3   # O + 2H
    # O contributes 2s + 2p_xyz = 4 orbitals; each H contributes 1s = 1
    assert basis.n_orbitals == 6
    assert basis.total_occ == pytest.approx(8.0)


def test_build_molecular_basis_accepts_d_block():
    """d-block was gated in the first Phase B commit; now supported via
    3D numerical overlap fallback (Phase A continuation)."""
    basis = build_molecular_basis(build_topology("[Fe]"))
    # Fe valence: 4s + 5 * 3d = 6 orbitals
    assert basis.n_orbitals == 6


def test_build_atom_basis_accepts_f_block():
    """f-block atoms now build via l=3 cubic-harmonic STO at the atom level."""
    from ptc.lcao.atomic_basis import build_atom_basis
    basis = build_atom_basis(92)   # U
    assert basis.n_orbitals == 8


# ─────────────────────────────────────────────────────────────────────
# 2. Overlap matrix is symmetric, diag=1, positive-definite
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("smiles, name, _", SP_TEST_MOLECULES)
def test_overlap_matrix_is_SPD(smiles, name, _):
    basis = build_molecular_basis(build_topology(smiles))
    S = overlap_matrix(basis)
    assert S.shape == (basis.n_orbitals, basis.n_orbitals)
    # diagonal exactly 1
    assert np.allclose(np.diag(S), 1.0, atol=1e-12), f"{name}: diag != 1"
    # symmetric
    assert np.allclose(S, S.T, atol=1e-12), f"{name}: S not symmetric"
    # positive-definite
    eigs = np.linalg.eigvalsh(S)
    assert (eigs > 0).all(), f"{name}: S not PD; eigs.min={eigs.min():.4e}"


# ─────────────────────────────────────────────────────────────────────
# 3. MO orthonormalisation under S
# ─────────────────────────────────────────────────────────────────────


def test_mo_S_orthonormal_H2O():
    basis = build_molecular_basis(build_topology("O"))
    S = overlap_matrix(basis)
    H = hueckel_hamiltonian(basis, S)
    _, c = solve_mo(H, S)
    # c.T @ S @ c should be identity
    cTSc = c.T @ S @ c
    np.testing.assert_allclose(cTSc, np.eye(c.shape[1]), atol=1e-10)


# ─────────────────────────────────────────────────────────────────────
# 4. Trace(rho S) = N_electrons
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("smiles, name, n_e", SP_TEST_MOLECULES)
def test_trace_rho_S_equals_Ne(smiles, name, n_e):
    basis = build_molecular_basis(build_topology(smiles))
    rho, S, _, _ = density_matrix_PT(build_topology(smiles), basis=basis)
    tr = float(np.trace(rho @ S))
    assert tr == pytest.approx(n_e, abs=1e-6), (
        f"{name}: Tr(rho S) = {tr} expected {n_e}"
    )


# ─────────────────────────────────────────────────────────────────────
# 5. Closed-shell idempotency: rho S rho = 2 rho
# ─────────────────────────────────────────────────────────────────────


_CLOSED_SHELL = [
    smi for smi, name, n_e in SP_TEST_MOLECULES if n_e % 2 == 0
]


@pytest.mark.parametrize("smiles", _CLOSED_SHELL)
def test_rho_idempotency_closed_shell(smiles):
    basis = build_molecular_basis(build_topology(smiles))
    rho, S, _, _ = density_matrix_PT(build_topology(smiles), basis=basis)
    lhs = rho @ S @ rho
    rhs = 2.0 * rho
    err = np.abs(lhs - rhs).max()
    assert err < 1e-9, f"{smiles}: ||rho S rho - 2 rho|| = {err:.4e}"


# ─────────────────────────────────────────────────────────────────────
# 6. Mulliken populations sum to N_electrons
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("smiles, name, n_e", SP_TEST_MOLECULES)
def test_mulliken_sum_equals_Ne(smiles, name, n_e):
    basis = build_molecular_basis(build_topology(smiles))
    rho, S, _, _ = density_matrix_PT(build_topology(smiles), basis=basis)
    pops = mulliken_populations(rho, S, basis)
    assert pops.shape == (basis.n_atoms,)
    assert float(pops.sum()) == pytest.approx(n_e, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────
# 7. H2 sanity: bonding MO has lowest energy, occupation = 2
# ─────────────────────────────────────────────────────────────────────


def test_H2_bonding_mo_doubly_occupied():
    basis = build_molecular_basis(build_topology("[H][H]"))
    rho, S, eigvals, c = density_matrix_PT(build_topology("[H][H]"), basis=basis)
    # H2 has 2 electrons -> 1 doubly occupied MO
    # Lowest-eigenvalue MO is the bonding combination (positive in both AOs)
    assert eigvals[0] < eigvals[1]   # bonding < antibonding
    # rho ~ 2 * |bond><bond|
    cb = c[:, 0]
    rho_expected = 2.0 * np.outer(cb, cb)
    np.testing.assert_allclose(rho, rho_expected, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────
# 8. Charge handling: OH-, H3O+
# ─────────────────────────────────────────────────────────────────────


def test_OH_anion_has_one_extra_electron():
    basis = build_molecular_basis(build_topology("[OH-]"))
    # OH- : O(6) + H(1) + 1 extra = 8 valence electrons
    assert basis.total_occ == pytest.approx(8.0)
    rho, S, _, _ = density_matrix_PT(build_topology("[OH-]"), basis=basis)
    assert float(np.trace(rho @ S)) == pytest.approx(8.0, abs=1e-6)


def test_H3O_cation_has_one_fewer_electron():
    basis = build_molecular_basis(build_topology("[OH3+]"))
    # H3O+: O(6) + 3H(3) - 1 = 8 valence electrons
    assert basis.total_occ == pytest.approx(8.0)
    rho, S, _, _ = density_matrix_PT(build_topology("[OH3+]"), basis=basis)
    assert float(np.trace(rho @ S)) == pytest.approx(8.0, abs=1e-6)
