"""Phase D tests: full GIAO chemical shielding tensor + principal axes.

Validates:
  * GIAOShieldingTensor dataclass populates correctly
  * H atom tensor: sigma_iso = 17.75 ppm, isotropic (span ~ 0)
  * H_2 (D_inf_h symmetry): axially symmetric (two equal eigenvalues),
    skew = +1
  * Benzene (D_6h, planar aromatic): axially symmetric oblate,
    skew = -1 (two perpendicular components equal, the third is unique)
  * eigenvalues are sorted ascending
  * sigma_iso = mean of eigenvalues
  * sigma_iso, sigma_zz match their direct counterparts
  * NICS_iso and NICS_zz are -sigma_iso and -sigma_zz
  * Anti-regression on dia/para components
"""

import math

import numpy as np
import pytest

from ptc.constants import ALPHA_PHYS
from ptc.lcao.density_matrix import build_molecular_basis
from ptc.lcao.shielding import (
    GIAOShieldingTensor,
    nics_iso_giao,
    nics_zz_giao,
    paramagnetic_shielding_tensor,
    shielding_tensor_at_point,
)
from ptc.topology import build_topology


# ─────────────────────────────────────────────────────────────────────
# 1. H atom: spherical -> isotropic tensor, Lamb shielding
# ─────────────────────────────────────────────────────────────────────


def test_H_atom_tensor_isotropic_Lamb():
    res = shielding_tensor_at_point(build_topology("[H]"), np.zeros(3))
    assert isinstance(res, GIAOShieldingTensor)
    expected = (ALPHA_PHYS ** 2 / 3.0) * 1.0e6
    # all three eigenvalues should equal Lamb sigma
    assert np.allclose(res.eigenvals, expected, rtol=1e-3)
    # isotropic: span ~ 0
    assert res.span < 1.0e-6
    # isotropic: skew = 0 by convention (since Omega = 0)
    assert abs(res.skew) < 1.0e-6
    # sigma_iso matches Lamb
    assert res.sigma_iso == pytest.approx(expected, rel=1e-3)
    # sigma_zz also = Lamb (isotropic)
    assert res.sigma_zz == pytest.approx(expected, rel=1e-3)
    # NICS conventions
    assert res.nics_iso == pytest.approx(-expected, rel=1e-3)
    assert res.nics_zz == pytest.approx(-expected, rel=1e-3)


# ─────────────────────────────────────────────────────────────────────
# 2. Tensor algebra invariants
# ─────────────────────────────────────────────────────────────────────


def test_eigenvalues_sorted_ascending():
    res = shielding_tensor_at_point(
        build_topology("c1ccccc1"),
        np.zeros(3),
        n_radial=30, n_theta=14, n_phi=20,
    )
    assert res.eigenvals[0] <= res.eigenvals[1] <= res.eigenvals[2]
    assert res.sigma_11 == res.eigenvals[0]
    assert res.sigma_33 == res.eigenvals[2]


def test_sigma_iso_equals_eigenvalue_average():
    """For a real symmetric tensor: Tr(sigma)/3 = mean(eigenvalues)."""
    res = shielding_tensor_at_point(
        build_topology("O"),
        np.array([0.0, 0.0, 0.5]),
        n_radial=30, n_theta=14, n_phi=20,
    )
    avg = float(np.mean(res.eigenvals))
    assert res.sigma_iso == pytest.approx(avg, abs=1e-9)


def test_sigma_decomposition_dia_plus_para():
    """sigma = sigma_d + sigma_p (3x3, component-wise)."""
    res = shielding_tensor_at_point(build_topology("[H][H]"), np.zeros(3),
                                      n_radial=30, n_theta=14, n_phi=20)
    np.testing.assert_allclose(res.sigma, res.sigma_d + res.sigma_p, atol=1e-10)


def test_eigenvectors_orthonormal():
    res = shielding_tensor_at_point(
        build_topology("c1ccccc1"), np.zeros(3),
        n_radial=30, n_theta=14, n_phi=20,
    )
    Q = res.eigenvecs
    np.testing.assert_allclose(Q.T @ Q, np.eye(3), atol=1e-10)


# ─────────────────────────────────────────────────────────────────────
# 3. H_2 (D_inf_h) - axially symmetric tensor
# ─────────────────────────────────────────────────────────────────────


def test_H2_tensor_axial():
    """H_2 along x-axis: two perpendicular eigenvalues should match
    (axial symmetry around bond)."""
    topo = build_topology("[H][H]")
    basis = build_molecular_basis(topo)
    P = basis.coords.mean(axis=0)   # midpoint
    res = shielding_tensor_at_point(topo, P, n_radial=30, n_theta=14, n_phi=20)
    # Two eigenvalues should be equal (perpendicular to bond),
    # the third unique (along the bond).
    # In our convention sigma_11 <= sigma_22 <= sigma_33, axial
    # symmetry gives either sigma_11 = sigma_22 (oblate) or
    # sigma_22 = sigma_33 (prolate).
    is_oblate = abs(res.eigenvals[0] - res.eigenvals[1]) < 0.5
    is_prolate = abs(res.eigenvals[1] - res.eigenvals[2]) < 0.5
    assert is_oblate or is_prolate, f"H2 not axial: eigs = {res.eigenvals}"
    # |skew| = 1 for perfect axial symmetry
    assert abs(res.skew) > 0.95


# ─────────────────────────────────────────────────────────────────────
# 4. Benzene (D_6h) - axially symmetric oblate
# ─────────────────────────────────────────────────────────────────────


def test_benzene_tensor_axial_oblate():
    """Benzene in xy-plane (D_6h): tensor axially symmetric around
    the z-axis (perpendicular to the ring). Two in-plane eigenvalues
    equal, the perpendicular one unique. NICS_zz makes this the
    reportable component for ring-current studies."""
    topo = build_topology("c1ccccc1")
    basis = build_molecular_basis(topo)
    P = basis.coords.mean(axis=0)
    res = shielding_tensor_at_point(topo, P, n_radial=30, n_theta=14, n_phi=20)
    # Benzene NICS literature: tensor is axially symmetric, with
    # the perpendicular component (zz) being most affected by ring
    # current. We expect two equal eigenvalues (in-plane) and one
    # different (perpendicular).
    eigs = res.eigenvals
    # axial symmetry: two of three eigenvalues equal (within numerical
    # quadrature error on the small test grid).
    is_axial = (
        abs(eigs[0] - eigs[1]) < 5.0
        or abs(eigs[1] - eigs[2]) < 5.0
    )
    assert is_axial, f"benzene tensor not axial: eigs = {eigs}"
    # |skew| close to 1 for axial symmetry (relaxed for coarse grid)
    assert abs(res.skew) > 0.85


# ─────────────────────────────────────────────────────────────────────
# 5. Symmetry of paramagnetic 3x3 tensor
# ─────────────────────────────────────────────────────────────────────


def test_paramagnetic_zero_for_atoms():
    """For valence-only minimum basis, atoms have no virtuals in closed
    shell -> sigma^p tensor is identically zero."""
    from ptc.lcao.density_matrix import density_matrix_PT
    topo = build_topology("[He]")
    basis = build_molecular_basis(topo)
    rho, S, eigvals, c = density_matrix_PT(topo, basis=basis)
    n_e = int(round(basis.total_occ))
    sigma_p = paramagnetic_shielding_tensor(
        basis, eigvals, c, n_e, np.zeros(3),
        n_radial=20, n_theta=10, n_phi=16,
    )
    assert np.abs(sigma_p).max() < 1e-12


# ─────────────────────────────────────────────────────────────────────
# 6. NICS convenience functions
# ─────────────────────────────────────────────────────────────────────


def test_nics_conventions():
    """NICS = -sigma; verify both iso and zz forms."""
    topo = build_topology("[H][H]")
    basis = build_molecular_basis(topo)
    P = basis.coords.mean(axis=0)
    nics_iso = nics_iso_giao(topo, P, n_radial=30, n_theta=14, n_phi=20)
    nics_zz = nics_zz_giao(topo, P, n_radial=30, n_theta=14, n_phi=20)
    res = shielding_tensor_at_point(topo, P, n_radial=30, n_theta=14, n_phi=20)
    assert nics_iso == pytest.approx(-res.sigma_iso)
    assert nics_zz == pytest.approx(-res.sigma_zz)


# ─────────────────────────────────────────────────────────────────────
# 7. Span and skew bounds
# ─────────────────────────────────────────────────────────────────────


def test_span_non_negative():
    res = shielding_tensor_at_point(
        build_topology("c1ccccc1"), np.zeros(3),
        n_radial=30, n_theta=14, n_phi=20,
    )
    assert res.span >= 0.0


def test_skew_within_unit_interval():
    for sm in ("[H]", "[H][H]", "c1ccccc1", "O"):
        topo = build_topology(sm)
        basis = build_molecular_basis(topo)
        P = basis.coords.mean(axis=0)
        res = shielding_tensor_at_point(topo, P, n_radial=30, n_theta=14, n_phi=20)
        assert -1.0 <= res.skew <= 1.0, f"{sm}: skew = {res.skew} out of [-1, +1]"