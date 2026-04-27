"""Tests for the optional polarisation flag in build_atom_basis /
build_molecular_basis (limitation #1 of BACKLOG_LCAO_PRECISION.md).

Polarisation orbitals are unoccupied (occ = 0) extra atomic orbitals
of angular momentum l_polar = l_valence + 1 at principal quantum
number n_polar = n_valence + 1. They expand the variational space
for the GIAO magnetic response without changing the ground-state
electron count.
"""

import numpy as np
import pytest

from ptc.constants import A_BOHR
from ptc.lcao.atomic_basis import build_atom_basis
from ptc.lcao.density_matrix import (
    build_molecular_basis,
    density_matrix_PT,
    overlap_matrix,
)
from ptc.topology import build_topology


# ─────────────────────────────────────────────────────────────────────
# 1. polarisation flag default off (anti-regression)
# ─────────────────────────────────────────────────────────────────────


def test_polarisation_default_off_preserves_basis():
    """Default behaviour (no polarisation) is identical to pre-extension."""
    b = build_atom_basis(6)         # default
    b_explicit = build_atom_basis(6, polarisation=False)
    assert b.n_orbitals == b_explicit.n_orbitals == 4   # 2s + 2p^3
    # default off: no l = 2 orbital appears
    assert all(o.l <= 1 for o in b.orbitals)


# ─────────────────────────────────────────────────────────────────────
# 2. opt-in polarisation adds the right orbital count
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("Z, base_orbs, expected_orbs", [
    (1,  1, 1 + 3),       # H : 1s + (2p polar)
    (3,  1, 1 + 3),       # Li
    (5,  4, 4 + 5),       # B  : 2s+2p + (3d polar)
    (6,  4, 4 + 5),       # C
    (7,  4, 4 + 5),       # N
    (8,  4, 4 + 5),       # O
    (9,  4, 4 + 5),       # F
    (14, 4, 4 + 5),       # Si: 3s+3p + (4d polar)
    (15, 4, 4 + 5),       # P
    (16, 4, 4 + 5),       # S
    (17, 4, 4 + 5),       # Cl
])
def test_polarisation_orbital_count(Z, base_orbs, expected_orbs):
    b_no = build_atom_basis(Z, polarisation=False)
    b_p = build_atom_basis(Z, polarisation=True)
    assert b_no.n_orbitals == base_orbs, f"Z={Z}: base size mismatch"
    assert b_p.n_orbitals == expected_orbs, f"Z={Z}: with-polar size mismatch"


def test_polarisation_added_for_d_block():
    """d-block atoms now receive f-polarisation (l=3) via the
    cubic-harmonic STO evaluator. Fe gains 7 f-orbitals (m=-3..+3)
    on top of its 6 valence (4s + 5*3d) shells."""
    b_no = build_atom_basis(26, polarisation=False)         # Fe
    b_p = build_atom_basis(26, polarisation=True)
    assert b_no.n_orbitals == 6
    assert b_p.n_orbitals == 6 + 7


# ─────────────────────────────────────────────────────────────────────
# 3. Polarisation orbitals are unoccupied
# ─────────────────────────────────────────────────────────────────────


def test_polarisation_orbitals_zero_occupation():
    """Total occupation = number of valence electrons (unchanged)."""
    b_no = build_atom_basis(6)
    b_p = build_atom_basis(6, polarisation=True)
    assert b_no.total_occ == pytest.approx(b_p.total_occ)
    # Specifically: every l=2 orbital has occ = 0
    for o in b_p.orbitals:
        if o.l == 2:
            assert o.occ == 0.0


# ─────────────────────────────────────────────────────────────────────
# 4. zeta of polarisation is PT-derived
# ─────────────────────────────────────────────────────────────────────


def test_polarisation_zeta_diffuse_relative_to_valence():
    """zeta_polar = Z_eff / ((n_val + 1) * a0) is more diffuse than
    valence (since we divide by n_val + 1 instead of n_val)."""
    b_p = build_atom_basis(6, polarisation=True)
    zeta_2p = next(o.zeta for o in b_p.orbitals if o.l == 1)
    zeta_3d = next(o.zeta for o in b_p.orbitals if o.l == 2)
    # n_val = 2 -> divide by 2; n_polar = 3 -> divide by 3
    # ratio should be 2/3 ~ 0.667
    assert zeta_3d == pytest.approx(zeta_2p * 2 / 3, rel=1e-12)


# ─────────────────────────────────────────────────────────────────────
# 5. Molecular basis plumbing
# ─────────────────────────────────────────────────────────────────────


def test_molecular_basis_polarisation_flag():
    """Pass-through of polarisation to per-atom builders."""
    b_no = build_molecular_basis(build_topology("C"), polarisation=False)
    b_p = build_molecular_basis(build_topology("C"), polarisation=True)
    # CH4: 1 C + 4 H
    # without polarisation: 4 + 4*1 = 8
    # with polarisation:    9 + 4*4 = 25
    assert b_no.n_orbitals == 8
    assert b_p.n_orbitals == 25
    # electron count unchanged
    assert b_no.total_occ == pytest.approx(b_p.total_occ)


# ─────────────────────────────────────────────────────────────────────
# 6. Overlap matrix with polarisation: still SPD
# ─────────────────────────────────────────────────────────────────────


def test_overlap_matrix_with_polarisation_SPD():
    """With polarisation, the overlap matrix is larger but must remain
    symmetric positive-definite. Note: heavy linear dependence may bring
    eigenvalues close to 0 (basis-set quality issue, not a bug)."""
    basis = build_molecular_basis(build_topology("[H][H]"), polarisation=True)
    S = overlap_matrix(basis)
    assert np.allclose(np.diag(S), 1.0, atol=1e-10)
    assert np.allclose(S, S.T, atol=1e-10)
    eigs = np.linalg.eigvalsh(S)
    assert (eigs > 0).all(), f"min eig = {eigs.min():.4e}"


# ─────────────────────────────────────────────────────────────────────
# 7. Density matrix with polarisation: trace and idempotency
# ─────────────────────────────────────────────────────────────────────


def test_density_matrix_with_polarisation():
    """Tr(rho S) = N_e and rho S rho = 2 rho still hold with polarisation."""
    topo = build_topology("[H][H]")
    basis = build_molecular_basis(topo, polarisation=True)
    rho, S, _, _ = density_matrix_PT(topo, basis=basis)
    n_e = int(round(basis.total_occ))
    assert float(np.trace(rho @ S)) == pytest.approx(n_e, abs=1e-6)
    err = np.abs(rho @ S @ rho - 2.0 * rho).max()
    assert err < 1e-9
