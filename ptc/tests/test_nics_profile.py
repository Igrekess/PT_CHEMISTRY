"""Tests for ptc.lcao.current.nics_profile_cluster (Chantier 8 Étape 3)."""
from __future__ import annotations

import numpy as np
import pytest

from ptc.lcao.cluster import (
    build_inverse_sandwich,
    precompute_response_explicit,
)
from ptc.lcao.current import nics_profile_cluster


def _bi3_th2_response():
    """Build a Bi3@Th2 stripped response (charged to even-electron count)."""
    Z, coords, bonds = build_inverse_sandwich(
        X_ring_Z=83, n_ring=3, M_cap_Z=90,
        d_xx=3.05, z_cap=2.10, ligands="none",
    )
    resp = precompute_response_explicit(
        Z_list=Z, coords=coords, bonds=bonds,
        basis_type="SZ", scf_mode="hueckel",
        include_f_block_d_shell=True,
        cphf_kwargs={"max_iter": 0, "level_shift": 5.0e-3},
    )
    return resp


def test_nics_profile_returns_correct_length():
    resp = _bi3_th2_response()
    profile = nics_profile_cluster(resp, z_max=3.0, n_z=7)
    assert len(profile) == 7
    for entry in profile:
        assert len(entry) == 2


def test_nics_profile_offsets_symmetric():
    resp = _bi3_th2_response()
    profile = nics_profile_cluster(resp, z_max=2.0, n_z=5)
    offsets = [z for z, _ in profile]
    np.testing.assert_allclose(offsets, [-2.0, -1.0, 0.0, 1.0, 2.0])


def test_nics_profile_axis_z_default_through_centroid():
    """At z=0 the probe coincides with the ring centroid."""
    resp = _bi3_th2_response()
    profile = nics_profile_cluster(resp, z_max=0.0, n_z=1)
    z0, sigma_z0 = profile[0]
    assert z0 == 0.0
    # sanity : finite and not absurdly large
    assert np.isfinite(sigma_z0)
    assert abs(sigma_z0) < 100.0


def test_nics_profile_invalid_axis():
    resp = _bi3_th2_response()
    with pytest.raises(ValueError, match="axis must be"):
        nics_profile_cluster(resp, axis="w")


def test_nics_profile_x_axis_traverses_ring_atom():
    """Axis='x' offset should pass through ring atom 0 (at +R, 0, 0)."""
    resp = _bi3_th2_response()
    profile = nics_profile_cluster(resp, z_max=2.0, n_z=5, axis="x")
    # central point still at centroid
    z0, _ = profile[len(profile) // 2]
    assert z0 == 0.0


def test_nics_profile_finite_for_all_offsets():
    resp = _bi3_th2_response()
    profile = nics_profile_cluster(resp, z_max=4.0, n_z=9)
    for _, sigma in profile:
        assert np.isfinite(sigma)
