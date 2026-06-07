"""Tests for ptc.lcao.parity_violation_mp2.

These tests stay on small molecules (H2O for the symmetry / zero check,
chiral H2O2 for the sign / magnitude checks) so the full MP2 + Z-vector
relaxation pipeline runs in a few seconds on a single core. The alanine
benchmark is wired in ``PT_PROJECTS/PT_HOMOCHIRALITY/scripts/
compute_e_pv_alanine_mp2.py``.

The MP2-improved E_PV must:

* preserve the SIGN of the RHF response (correlation does not flip the
  sign of an SO/PV second-order sum);
* INCREASE the MAGNITUDE of |E_PV| relative to the RHF reference, because
  MP2 narrows the occupied-virtual gaps that sit in the denominator of
  the Bakasov-Ha-Quack 1998 perturbative formula;
* vanish for an achiral closed-shell molecule (H2O) at the same level of
  numerical precision as the RHF baseline.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ptc.lcao.atomic_basis import build_atom_basis
from ptc.lcao.density_matrix import (
    PTMolecularBasis,
    density_matrix_PT,
)
from ptc.lcao.fock import density_matrix_PT_scf
from ptc.lcao.parity_violation_mp2 import (
    EPVMP2Result,
    e_pv_so_response_mp2,
)


# Coarser grids than production — enough to register the sign / scaling
# pattern we care about, while keeping unit-test wall time bounded.
_FAST_HSO = dict(n_radial=20, n_theta=10, n_phi=14)
_FAST_MP2 = dict(n_radial=12, n_theta=8, n_phi=10, use_becke=False,
                 lebedev_order=14)
_FAST_LAG = dict(n_radial=12, n_theta=8, n_phi=10, use_becke=False,
                 lebedev_order=14)
_FAST_Z = dict(n_radial_grid=12, n_theta_grid=8, n_phi_grid=10,
               use_becke=False, lebedev_order=14,
               max_iter=8, tol=1.0e-3)
_FAST_RELAX = dict(n_radial=12, n_theta=8, n_phi=10, use_becke=False,
                   lebedev_order=14)
_FAST_SCF = dict(max_iter=20, tol=1.0e-3, n_radial=16, n_theta=10,
                 n_phi=12)


# ---------------------------------------------------------------------
# Geometry helpers (re-used from the RHF tests).
# ---------------------------------------------------------------------


def _manual_basis(Z_list, coords_A):
    atoms = []
    flat_orbs = []
    atom_idx = []
    for i, Z in enumerate(Z_list):
        ab = build_atom_basis(int(Z))
        atoms.append(ab)
        for orb in ab.orbitals:
            flat_orbs.append(orb)
            atom_idx.append(i)
    coords_arr = np.array(coords_A, dtype=float)
    return PTMolecularBasis(
        atoms=atoms,
        coords=coords_arr,
        orbitals=flat_orbs,
        atom_index=atom_idx,
        Z_list=list(Z_list),
    )


def _water_geometry_Cs():
    r = 0.96
    half = math.radians(104.5 / 2.0)
    hx = r * math.sin(half)
    hz = -r * math.cos(half)
    Z_list = [8, 1, 1]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [+hx, 0.0, hz],
        [-hx, 0.0, hz],
    ])
    return Z_list, coords


def _h2o2_geometry(dihedral_deg: float,
                   r_OH1: float = 0.95,
                   r_OH2: float = 0.95):
    r_OO = 1.475
    theta_OOH = math.radians(94.8)
    phi = math.radians(dihedral_deg / 2.0)

    O1 = np.array([-r_OO / 2.0, 0.0, 0.0])
    O2 = np.array([+r_OO / 2.0, 0.0, 0.0])

    def _h_direction(sign_x: float, sign_phi: float):
        base = np.array([
            sign_x * math.cos(math.pi - theta_OOH),
            0.0,
            math.sin(math.pi - theta_OOH),
        ])
        ang = sign_phi * phi
        c, s = math.cos(ang), math.sin(ang)
        return np.array([
            base[0],
            c * base[1] - s * base[2],
            s * base[1] + c * base[2],
        ])

    H1 = O1 + r_OH1 * _h_direction(sign_x=-1.0, sign_phi=+1.0)
    H2 = O2 + r_OH2 * _h_direction(sign_x=+1.0, sign_phi=-1.0)

    Z_list = [8, 8, 1, 1]
    coords = np.array([O1, O2, H1, H2])
    return Z_list, coords


def _run_rhf(basis: PTMolecularBasis):
    """Tiny wrapper around density_matrix_PT_scf with fast grids."""
    rho, S, eigvals, c, conv, residual = density_matrix_PT_scf(
        None, basis=basis, mode="hf", **_FAST_SCF,
    )
    n_e = int(round(basis.total_occ))
    return rho, eigvals, c, n_e


# ---------------------------------------------------------------------
# 1. Achiral H2O: MP2 response must still be (essentially) zero
# ---------------------------------------------------------------------


def test_mp2_h2o_zero():
    """Achiral symmetric water -> E_PV via SO/PV response on MP2-relaxed
    orbitals is zero to numerical precision, exactly like at the RHF
    level. Correlation can change orbital energies and densities, but it
    cannot break the underlying C2v parity symmetry.
    """
    Z_list, coords = _water_geometry_Cs()
    basis = _manual_basis(Z_list, coords)
    rho, eigvals_hf, c_hf, n_e = _run_rhf(basis)

    result = e_pv_so_response_mp2(
        basis, None, eigvals_hf, c_hf, n_e,
        hso_kwargs=_FAST_HSO,
        mp2_kwargs=_FAST_MP2,
        lagrangian_kwargs=_FAST_LAG,
        z_vector_kwargs=_FAST_Z,
        relax_kwargs=_FAST_RELAX,
    )
    # Same tolerance as the RHF C2v test: the response sum is bounded
    # well below the canonical 1e-20 Ha PV scale.
    assert abs(result.e_pv_mp2_full) < 1.0e-23, (
        f"E_PV^MP2(H2O) = {result.e_pv_mp2_full}; expected zero by C2v "
        f"symmetry"
    )
    # RHF baseline returned in the same call should also be near zero.
    assert abs(result.e_pv_rhf) < 1.0e-23


# ---------------------------------------------------------------------
# 2. Chiral H2O2: MP2 response is finite and SAME SIGN as RHF
# ---------------------------------------------------------------------


def test_mp2_response_sign_matches_rhf():
    """For chiral H2O2 at a non-zero dihedral, both the RHF baseline and
    the MP2-improved response are finite and carry the SAME SIGN. MP2
    must not flip the sign of E_PV — that would indicate an instability
    or a bug in the orbital-response machinery.
    """
    Z, coords = _h2o2_geometry(+119.8)
    basis = _manual_basis(Z, coords)
    _, eigvals_hf, c_hf, n_e = _run_rhf(basis)

    result = e_pv_so_response_mp2(
        basis, None, eigvals_hf, c_hf, n_e,
        hso_kwargs=_FAST_HSO,
        mp2_kwargs=_FAST_MP2,
        lagrangian_kwargs=_FAST_LAG,
        z_vector_kwargs=_FAST_Z,
        relax_kwargs=_FAST_RELAX,
    )
    assert abs(result.e_pv_rhf) > 1.0e-25, (
        f"RHF E_PV(H2O2) = {result.e_pv_rhf} too small to be a signal"
    )
    assert abs(result.e_pv_mp2_full) > 1.0e-25, (
        f"MP2 E_PV(H2O2) = {result.e_pv_mp2_full} too small to be a signal"
    )
    same_sign = (result.e_pv_rhf * result.e_pv_mp2_full) > 0.0
    assert same_sign, (
        f"MP2 flipped the sign of E_PV: RHF = {result.e_pv_rhf}, "
        f"MP2 full = {result.e_pv_mp2_full}"
    )


# ---------------------------------------------------------------------
# 3. Chiral H2O2: |E_PV^MP2| > |E_PV^RHF|
# ---------------------------------------------------------------------


def test_mp2_response_magnitude_larger_than_rhf():
    """MP2 orbital relaxation shrinks the occupied-virtual gaps in the
    Bakasov-Ha-Quack denominator and concentrates a small amount of
    density on the heavy nuclei. Both effects amplify |E_PV| compared to
    the bare RHF response.

    We require a strict increase here, which is consistent with all
    closed-shell molecules tested so far. If a future basis / geometry
    ever produces a tiny decrease at the few-percent level (e.g. by a
    cancellation of contributions), this test should be relaxed to a
    weaker "magnitude is within a factor of 2 of the RHF baseline"
    condition.
    """
    Z, coords = _h2o2_geometry(+119.8)
    basis = _manual_basis(Z, coords)
    _, eigvals_hf, c_hf, n_e = _run_rhf(basis)

    result = e_pv_so_response_mp2(
        basis, None, eigvals_hf, c_hf, n_e,
        hso_kwargs=_FAST_HSO,
        mp2_kwargs=_FAST_MP2,
        lagrangian_kwargs=_FAST_LAG,
        z_vector_kwargs=_FAST_Z,
        relax_kwargs=_FAST_RELAX,
    )
    a_rhf = abs(result.e_pv_rhf)
    a_mp2 = abs(result.e_pv_mp2_full)
    assert a_mp2 > a_rhf, (
        f"|E_PV^MP2| = {a_mp2} not larger than |E_PV^RHF| = {a_rhf}; "
        f"ratio = {a_mp2 / a_rhf}"
    )


# ---------------------------------------------------------------------
# 4. Result bundle structure
# ---------------------------------------------------------------------


def test_result_bundle_shapes_h2o2():
    """The EPVMP2Result bundle exposes the MP2 result, Z-vector and the
    two sets of relaxed orbitals with the expected shapes.
    """
    Z, coords = _h2o2_geometry(+90.0)
    basis = _manual_basis(Z, coords)
    _, eigvals_hf, c_hf, n_e = _run_rhf(basis)
    n_occ = n_e // 2
    n_virt = c_hf.shape[0] - n_occ

    result = e_pv_so_response_mp2(
        basis, None, eigvals_hf, c_hf, n_e,
        hso_kwargs=_FAST_HSO,
        mp2_kwargs=_FAST_MP2,
        lagrangian_kwargs=_FAST_LAG,
        z_vector_kwargs=_FAST_Z,
        relax_kwargs=_FAST_RELAX,
    )
    assert isinstance(result, EPVMP2Result)
    assert result.z_vector.shape == (n_virt, n_occ)
    assert result.eigvals_mp2_full.shape == (c_hf.shape[0],)
    assert result.c_mp2_full.shape == c_hf.shape
    # All three E_PV scalars are finite floats:
    assert np.isfinite(result.e_pv_rhf)
    assert np.isfinite(result.e_pv_mp2_lo)
    assert np.isfinite(result.e_pv_mp2_full)


# ---------------------------------------------------------------------
# 5. Skip-baseline path
# ---------------------------------------------------------------------


def test_compute_baselines_false_only_full():
    """With ``compute_baselines=False`` the driver skips the RHF and
    MP2-LO response evaluations (NaN placeholders) and returns only the
    production "MP2-full" value. Useful for fast scans over geometries.
    """
    Z, coords = _h2o2_geometry(+119.8)
    basis = _manual_basis(Z, coords)
    _, eigvals_hf, c_hf, n_e = _run_rhf(basis)

    result = e_pv_so_response_mp2(
        basis, None, eigvals_hf, c_hf, n_e,
        compute_baselines=False,
        hso_kwargs=_FAST_HSO,
        mp2_kwargs=_FAST_MP2,
        lagrangian_kwargs=_FAST_LAG,
        z_vector_kwargs=_FAST_Z,
        relax_kwargs=_FAST_RELAX,
    )
    assert math.isnan(result.e_pv_rhf)
    assert math.isnan(result.e_pv_mp2_lo)
    assert np.isfinite(result.e_pv_mp2_full)
    assert abs(result.e_pv_mp2_full) > 1.0e-25
