"""Induced current density and ring current strength tests.

Tests for ``ptc/lcao/current.py``:
  - module-level constants (ALPHA_FS, CURRENT_TO_NA_PER_T)
  - benzene D6h symmetry, sign, magnitude (calibration anchor at -12 nA/T)
  - pyridine C2v symmetry, diamagnetic
  - alternate gauges (common, ipsocentric, giao)
"""
import numpy as np
import pytest

from ptc.lcao.current import (
    ALPHA_FS,
    CURRENT_TO_NA_PER_T,
    bond_current_strength,
    current_density_at_points,
    nics_zz_from_current,
    precompute_response,
    ring_current_strength,
)
from ptc.topology import build_topology


_FAST_CPHF_KW = dict(
    max_iter=10, tol=1.0e-4,
    n_radial_grid=16, n_theta_grid=10, n_phi_grid=12,
    n_radial_op=20, n_theta_op=10, n_phi_op=12,
)


# constants

def test_alpha_fs_value():
    assert abs(ALPHA_FS - 1.0 / 137.035999) < 1.0e-6


def test_current_to_na_per_t_positive():
    assert CURRENT_TO_NA_PER_T > 0.0


# benzene fixture

@pytest.fixture(scope="module")
def benzene_response():
    topo = build_topology("c1ccccc1")
    return precompute_response(topo, cphf_kwargs=_FAST_CPHF_KW)


def test_benzene_response_shapes(benzene_response):
    resp = benzene_response
    assert resp.basis.n_orbitals == 30
    assert resp.n_occ == 15


# current_density_at_points

def test_current_density_shape(benzene_response):
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    j, j_p, j_d, rho = current_density_at_points(benzene_response, pts, beta=2)
    assert j.shape == (2, 3)
    assert j_p.shape == (2, 3)
    assert j_d.shape == (2, 3)
    assert rho.shape == (2,)


def test_current_density_diamagnetic_at_origin_zero(benzene_response):
    pts = np.array([[0.0, 0.0, 0.0]])
    _, _, j_d, _ = current_density_at_points(benzene_response, pts, beta=2)
    assert np.allclose(j_d[0], 0.0, atol=1.0e-15)


def test_current_density_density_positive(benzene_response):
    pts = np.array([[x, 0.0, 0.0] for x in np.linspace(-2, 2, 11)])
    _, _, _, rho = current_density_at_points(benzene_response, pts, beta=2)
    assert np.all(rho >= -1.0e-12)


# bond / ring current

def test_benzene_bond_current_diamagnetic(benzene_response):
    J = bond_current_strength(benzene_response, 0, 1)
    assert J < 0.0


def test_benzene_bond_current_magnitude(benzene_response):
    J = bond_current_strength(benzene_response, 0, 1)
    assert abs(abs(J) - 12.0) < 2.0


def test_benzene_d6h_bond_equality(benzene_response):
    js = []
    for k in range(6):
        js.append(bond_current_strength(benzene_response, k, (k + 1) % 6))
    arr = np.array(js)
    assert np.std(arr) < 1.0e-3 * abs(np.mean(arr))


def test_benzene_ring_current_within_target(benzene_response):
    J_avg, _ = ring_current_strength(
        benzene_response, [0, 1, 2, 3, 4, 5],
    )
    assert abs(abs(J_avg) - 12.0) < 2.0


def test_bond_current_distinct_atoms_required(benzene_response):
    with pytest.raises(ValueError):
        bond_current_strength(benzene_response, 0, 0)


# alternate gauges

def test_gauge_common_baseline(benzene_response):
    J, _ = ring_current_strength(benzene_response, [0, 1, 2, 3, 4, 5],
                                    gauge="common")
    assert abs(abs(J) - 12.0) < 2.0


def test_gauge_giao_diamagnetic(benzene_response):
    """Full GIAO j_para correction (London-phase TERM2): J<0 + D6h."""
    J, js = ring_current_strength(
        benzene_response, [0, 1, 2, 3, 4, 5],
        n_in=31, n_out=31, gauge="giao",
    )
    assert J < 0.0
    arr = np.array(js)
    assert np.std(arr) < 1.0e-3 * abs(np.mean(arr))


def test_gauge_invalid_value_raises(benzene_response):
    with pytest.raises(ValueError, match="gauge"):
        current_density_at_points(
            benzene_response,
            grid_pts=np.array([[0.0, 0.0, 0.0]]),
            beta=2, gauge="not-a-gauge",
        )


# pyridine

@pytest.fixture(scope="module")
def pyridine_response():
    topo = build_topology("c1ccncc1")
    return precompute_response(topo, cphf_kwargs=_FAST_CPHF_KW)


def test_pyridine_aromatic_diamagnetic(pyridine_response):
    ring = [i for i, Z in enumerate(pyridine_response.basis.Z_list)
            if Z in (6, 7)]
    J_avg, _ = ring_current_strength(pyridine_response, ring)
    assert J_avg < 0.0


# nics_zz_from_current

def test_nics_zz_from_current_returns_float(benzene_response):
    pts = np.array([[x, y, z]
                     for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
                     if not (x == 0 and y == 0 and z == 0)])
    weights = np.ones(len(pts))
    sigma = nics_zz_from_current(benzene_response, np.zeros(3), pts, weights)
    assert isinstance(sigma, float)
    assert np.isfinite(sigma)
