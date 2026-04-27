"""End-to-end Bi3@U2 inverse-sandwich cluster validation.

Reproduces Ding 2026 NICS_zz = +0.08 ppm at the cluster centre via
the full PT-LCAO + GIMIC stack:
  - f-block geometry contractions in r_equilibrium
  - explicit-coordinate cluster builder
  - GIMIC PT-pure ring current density via CPHF response
  - l=3 cubic-harmonic STO evaluator (5f orbitals on U)

Stripped cluster: 3 Bi triangle + 2 U axial caps (no Cp* ligands).
NICS_zz target: in [-2, +2] ppm (cluster ligand-screened ring current).
"""
import numpy as np
import pytest

from ptc.lcao.cluster import precompute_response_explicit
from ptc.lcao.current import (
    nics_zz_from_current,
    ring_current_strength,
)
from ptc.lcao.giao import _build_spherical_grid


# Geometry (Ding 2026, X-ray)

D_BI_BI = 3.05
Z_U_CAP = 2.1
R_BI = D_BI_BI / np.sqrt(3.0)
COORDS = np.array([
    [R_BI, 0.0, 0.0],
    [-R_BI / 2.0, R_BI * np.sqrt(3) / 2.0, 0.0],
    [-R_BI / 2.0, -R_BI * np.sqrt(3) / 2.0, 0.0],
    [0.0, 0.0, +Z_U_CAP],
    [0.0, 0.0, -Z_U_CAP],
])
Z_LIST = [83, 83, 83, 92, 92]
BONDS = (
    [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]
    + [(b, u, 1.0) for u in (3, 4) for b in (0, 1, 2)]
)
RING = [0, 1, 2]

_FAST_CPHF = dict(
    max_iter=10, tol=1.0e-3,
    n_radial_grid=14, n_theta_grid=8, n_phi_grid=10,
    n_radial_op=16, n_theta_op=8, n_phi_op=10,
)


@pytest.fixture(scope="module")
def cluster_response():
    return precompute_response_explicit(
        Z_LIST, COORDS, bonds=BONDS, basis_type="SZ",
        scf_mode="hueckel", cphf_kwargs=_FAST_CPHF,
    )


# Basis size

def test_cluster_sz_size(cluster_response):
    """SZ Bi3@U2: 3 x 4 (Bi: 6s + 6p_xyz) + 2 x 8 (U: 7s + 5f_x7) = 28."""
    assert cluster_response.basis.n_orbitals == 28


# Geometry exact

def test_cluster_geometry_distances():
    d_bi_bi = float(np.linalg.norm(COORDS[0] - COORDS[1]))
    assert abs(d_bi_bi - D_BI_BI) < 1.0e-6
    d_u_bi = float(np.linalg.norm(COORDS[3] - COORDS[0]))
    expected = float(np.sqrt(R_BI ** 2 + Z_U_CAP ** 2))
    assert abs(d_u_bi - expected) < 1.0e-6


# D3h symmetry

def test_cluster_d3h_symmetry(cluster_response):
    """3 Bi-Bi bond currents within 5% of mean (D3h preserved)."""
    J, js = ring_current_strength(
        cluster_response, RING, n_in=21, n_out=21, gauge="common",
    )
    arr = np.array(js)
    spread = (max(arr) - min(arr)) / abs(np.mean(arr))
    assert spread < 0.05, f"D3h not preserved: spread={spread:.2%}"


# Gauge invariance (common vs GIAO)

def test_cluster_gauge_invariance(cluster_response):
    J_c, _ = ring_current_strength(
        cluster_response, RING, n_in=21, n_out=21, gauge="common",
    )
    J_g, _ = ring_current_strength(
        cluster_response, RING, n_in=21, n_out=21, gauge="giao",
    )
    assert abs(J_c - J_g) < 0.10 * abs(J_c)


# NICS at cluster centre — primary validation

def _nics_at_origin(resp):
    sg = _build_spherical_grid(R_max=8.0, n_radial=24, n_theta=12, n_phi=16)
    return nics_zz_from_current(resp, np.zeros(3), sg.points, sg.weights, beta=2)


def test_cluster_nics_in_target_window(cluster_response):
    """NICS_zz at centre must be in [-2, +2] ppm."""
    sigma = _nics_at_origin(cluster_response)
    nics = -sigma
    assert -2.0 < nics < 2.0, f"NICS={nics:+.3f} outside [-2, +2] ppm"


def test_cluster_nics_close_to_experiment(cluster_response):
    """SZ Hueckel NICS_zz error < 2 ppm vs Ding 2026 +0.08 ppm."""
    sigma = _nics_at_origin(cluster_response)
    nics = -sigma
    assert abs(nics - 0.08) < 2.0
