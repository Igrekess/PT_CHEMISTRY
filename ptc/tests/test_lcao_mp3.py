"""Tests for Phase 6.B.10 MP3 amplitude correction and σ_p^MP3 pipeline.
"""
from __future__ import annotations

import numpy as np
import pytest

from ptc.lcao.cluster import build_explicit_cluster
from ptc.lcao.fock import density_matrix_PT_scf
from ptc.lcao.mp2 import mp2_at_hf
from ptc.lcao.mp3 import (
    _mp3_hh_ladder,
    _mp3_pp_ladder,
    _t_tilde,
    mp3_amplitudes_correction,
    mp3_at_hf,
)


# ─────────────────────────────────────────────────────────────────────
# t̃ singlet-coupled amplitudes
# ─────────────────────────────────────────────────────────────────────


def test_t_tilde_explicit_formula():
    """t̃[i,a,j,b] = 2 t[i,a,j,b] − t[j,a,i,b]"""
    t = np.random.RandomState(42).rand(2, 3, 2, 3)
    t_t = _t_tilde(t)
    for i in range(2):
        for a in range(3):
            for j in range(2):
                for b in range(3):
                    expected = 2.0 * t[i, a, j, b] - t[j, a, i, b]
                    assert t_t[i, a, j, b] == pytest.approx(expected, abs=1e-13)


def test_t_tilde_idempotent_sum():
    """t̃ + t̃.swap_ij = 3 (t + t.swap_ij) − 2 (t + t.swap_ij) = ... a closed-shell identity check.

    Actually: t̃ + t̃.swap_ij = 2t - t.swap_ij + 2 t.swap_ij - t = t + t.swap_ij
    """
    t = np.random.RandomState(7).rand(3, 4, 3, 4)
    t_t = _t_tilde(t)
    t_t_swap = t_t.transpose(2, 1, 0, 3)
    t_swap = t.transpose(2, 1, 0, 3)
    np.testing.assert_allclose(t_t + t_t_swap, t + t_swap, atol=1e-13)


# ─────────────────────────────────────────────────────────────────────
# Ladder term shapes and identities
# ─────────────────────────────────────────────────────────────────────


def test_pp_ladder_shape_and_zero_when_no_eri():
    t = np.random.RandomState(0).rand(2, 3, 2, 3)
    n_v = t.shape[1]
    eri_zero = np.zeros((n_v, n_v, n_v, n_v))
    out = _mp3_pp_ladder(t, eri_zero)
    assert out.shape == t.shape
    np.testing.assert_array_equal(out, 0.0)


def test_hh_ladder_shape_and_zero_when_no_eri():
    t = np.random.RandomState(1).rand(2, 3, 2, 3)
    n_o = t.shape[0]
    eri_zero = np.zeros((n_o, n_o, n_o, n_o))
    out = _mp3_hh_ladder(t, eri_zero)
    assert out.shape == t.shape
    np.testing.assert_array_equal(out, 0.0)


def test_pp_ladder_linear_in_t():
    """The pp-ladder contraction is linear in the input amplitudes."""
    rng = np.random.RandomState(3)
    n_o, n_v = 2, 3
    t1 = rng.rand(n_o, n_v, n_o, n_v)
    t2 = rng.rand(n_o, n_v, n_o, n_v)
    eri = rng.rand(n_v, n_v, n_v, n_v)
    out_sum = _mp3_pp_ladder(t1 + 2.0 * t2, eri)
    out1 = _mp3_pp_ladder(t1, eri)
    out2 = _mp3_pp_ladder(t2, eri)
    np.testing.assert_allclose(out_sum, out1 + 2.0 * out2, atol=1e-12)


# ─────────────────────────────────────────────────────────────────────
# Smoke test on N2 SZ : MP3 amplitude correction is non-trivial and
# improves the correlation energy vs MP2.
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_mp3_correlation_energy_increases_magnitude_on_N2_SZ():
    """MP3 correction should increase |E_corr| relative to MP2 on N2."""
    Z = [7, 7]
    coords = np.array([[0.0, 0.0, 0.0], [1.0975, 0.0, 0.0]])
    bonds = [(0, 1, 3.0)]
    basis, topo = build_explicit_cluster(
        Z_list=Z, coords=coords, bonds=bonds, basis_type="SZ",
    )
    rho, S, eigvals, c, conv, _ = density_matrix_PT_scf(
        topo, basis=basis, mode="hf", max_iter=15, tol=1e-3,
        n_radial=10, n_theta=8, n_phi=10,
    )
    n_occ = int(round(basis.total_occ)) // 2
    gk = dict(n_radial=10, n_theta=8, n_phi=10,
              use_becke=False, lebedev_order=14)

    mp2 = mp2_at_hf(basis, eigvals, c, n_occ, **gk)
    mp3 = mp3_at_hf(basis, eigvals, c, n_occ, mp2_result=mp2, **gk)

    assert mp3.t.shape == mp2.t.shape
    # MP3 amplitudes differ from MP2 amplitudes
    assert np.linalg.norm(mp3.t - mp2.t) > 1e-3
    # MP3 correlation energy is more negative (larger magnitude) than MP2
    assert mp3.e_corr < mp2.e_corr - 1e-3, (
        f"MP3 e_corr={mp3.e_corr} should be more negative than MP2 e_corr={mp2.e_corr}"
    )


def test_mp3_amplitudes_correction_pp_hh_isolated_additivity():
    """Phase 6.B.10a invariant: pp and hh contributions add linearly
    when the ring is disabled (regression test for ladder isolation)."""
    Z = [7, 7]
    coords = np.array([[0.0, 0.0, 0.0], [1.0975, 0.0, 0.0]])
    bonds = [(0, 1, 3.0)]
    basis, topo = build_explicit_cluster(
        Z_list=Z, coords=coords, bonds=bonds, basis_type="SZ",
    )
    rho, S, eigvals, c, conv, _ = density_matrix_PT_scf(
        topo, basis=basis, mode="hf", max_iter=10, tol=1e-3,
        n_radial=10, n_theta=8, n_phi=10,
    )
    n_occ = int(round(basis.total_occ)) // 2
    gk = dict(n_radial=10, n_theta=8, n_phi=10,
              use_becke=False, lebedev_order=14)
    mp2 = mp2_at_hf(basis, eigvals, c, n_occ, **gk)

    delta_pp = mp3_amplitudes_correction(
        basis, c, n_occ, eigvals, mp2,
        include_pp_ladder=True, include_hh_ladder=False,
        include_ring=False, **gk,
    )
    delta_hh = mp3_amplitudes_correction(
        basis, c, n_occ, eigvals, mp2,
        include_pp_ladder=False, include_hh_ladder=True,
        include_ring=False, **gk,
    )
    delta_pp_hh = mp3_amplitudes_correction(
        basis, c, n_occ, eigvals, mp2,
        include_pp_ladder=True, include_hh_ladder=True,
        include_ring=False, **gk,
    )

    np.testing.assert_allclose(delta_pp_hh, delta_pp + delta_hh, atol=1e-10)


def test_mp3_ring_shape_and_zero_when_no_eri():
    """The ring contribution returns same shape as t, and vanishes
    if the (jc|ka) ERI block is zero."""
    from ptc.lcao.mp3 import _mp3_ring
    rng = np.random.RandomState(11)
    n_o, n_v = 2, 3
    t = rng.rand(n_o, n_v, n_o, n_v)
    eri_zero = np.zeros((n_o, n_v, n_o, n_v))
    out = _mp3_ring(t, eri_zero)
    assert out.shape == t.shape
    np.testing.assert_array_equal(out, 0.0)


def test_mp3_ring_linear_in_t():
    """Ring contraction is linear in the input amplitudes."""
    from ptc.lcao.mp3 import _mp3_ring
    rng = np.random.RandomState(12)
    n_o, n_v = 2, 3
    t1 = rng.rand(n_o, n_v, n_o, n_v)
    t2 = rng.rand(n_o, n_v, n_o, n_v)
    eri = rng.rand(n_o, n_v, n_o, n_v)
    out_sum = _mp3_ring(t1 + 2.0 * t2, eri)
    out1 = _mp3_ring(t1, eri)
    out2 = _mp3_ring(t2, eri)
    np.testing.assert_allclose(out_sum, out1 + 2.0 * out2, atol=1e-10)


def test_mp3_amplitudes_correction_pp_hh_ring_additive():
    """With ring on, the full correction is the sum of pp + hh + ring."""
    Z = [7, 7]
    coords = np.array([[0.0, 0.0, 0.0], [1.0975, 0.0, 0.0]])
    bonds = [(0, 1, 3.0)]
    basis, topo = build_explicit_cluster(
        Z_list=Z, coords=coords, bonds=bonds, basis_type="SZ",
    )
    rho, S, eigvals, c, conv, _ = density_matrix_PT_scf(
        topo, basis=basis, mode="hf", max_iter=10, tol=1e-3,
        n_radial=10, n_theta=8, n_phi=10,
    )
    n_occ = int(round(basis.total_occ)) // 2
    gk = dict(n_radial=10, n_theta=8, n_phi=10,
              use_becke=False, lebedev_order=14)
    mp2 = mp2_at_hf(basis, eigvals, c, n_occ, **gk)

    delta_pp = mp3_amplitudes_correction(
        basis, c, n_occ, eigvals, mp2,
        include_pp_ladder=True, include_hh_ladder=False,
        include_ring=False, **gk,
    )
    delta_hh = mp3_amplitudes_correction(
        basis, c, n_occ, eigvals, mp2,
        include_pp_ladder=False, include_hh_ladder=True,
        include_ring=False, **gk,
    )
    delta_ring = mp3_amplitudes_correction(
        basis, c, n_occ, eigvals, mp2,
        include_pp_ladder=False, include_hh_ladder=False,
        include_ring=True, **gk,
    )
    delta_full = mp3_amplitudes_correction(
        basis, c, n_occ, eigvals, mp2,
        include_pp_ladder=True, include_hh_ladder=True,
        include_ring=True, **gk,
    )
    np.testing.assert_allclose(
        delta_full, delta_pp + delta_hh + delta_ring, atol=1e-10
    )


def test_mp3_ring_satisfies_pair_symmetry():
    """Phase 6.B.10c — ring contribution must be invariant under the
    simultaneous swap (i↔j, a↔b) by construction (we symmetrise
    σ + σ.transpose(2,3,0,1)).
    """
    from ptc.lcao.mp3 import _mp3_ring
    rng = np.random.RandomState(99)
    n_o, n_v = 3, 4
    t = rng.rand(n_o, n_v, n_o, n_v) - 0.5
    eri = rng.rand(n_o, n_v, n_o, n_v) - 0.5
    ring = _mp3_ring(t, eri)
    # ring[i,a,j,b] should equal ring[j,b,i,a] exactly
    np.testing.assert_allclose(ring, ring.transpose(2, 3, 0, 1), atol=1e-12)


def test_mp3_ring_compensates_pp_ladder_on_N2_SZ():
    """Phase 6.B.10c — the ring contribution must reduce |E_MP3| relative
    to pp+hh-only, matching the textbook closed-shell expectation that the
    ring term opposes the pp-ladder over-correction on covalent systems.
    """
    Z = [7, 7]
    coords = np.array([[0.0, 0.0, 0.0], [1.0975, 0.0, 0.0]])
    bonds = [(0, 1, 3.0)]
    basis, topo = build_explicit_cluster(
        Z_list=Z, coords=coords, bonds=bonds, basis_type="SZ",
    )
    rho, S, eigvals, c, conv, _ = density_matrix_PT_scf(
        topo, basis=basis, mode="hf", max_iter=10, tol=1e-3,
        n_radial=10, n_theta=8, n_phi=10,
    )
    n_occ = int(round(basis.total_occ)) // 2
    gk = dict(n_radial=10, n_theta=8, n_phi=10,
              use_becke=False, lebedev_order=14)
    mp2 = mp2_at_hf(basis, eigvals, c, n_occ, **gk)

    # MP3 with pp+hh only (default)
    delta_t_no_ring = mp3_amplitudes_correction(
        basis, c, n_occ, eigvals, mp2,
        include_pp_ladder=True, include_hh_ladder=True, include_ring=False,
        **gk,
    )
    # MP3 with pp+hh+ring
    delta_t_ring = mp3_amplitudes_correction(
        basis, c, n_occ, eigvals, mp2,
        include_pp_ladder=True, include_hh_ladder=True, include_ring=True,
        **gk,
    )

    combo = 2.0 * mp2.eri_mo_iajb - mp2.eri_mo_iajb.transpose(0, 3, 2, 1)
    e_mp2 = -float(np.sum(mp2.t * combo))
    e_mp3_ladder = -float(np.sum((mp2.t + delta_t_no_ring) * combo))
    e_mp3_full = -float(np.sum((mp2.t + delta_t_ring) * combo))

    # All three negative (covalent system, attractive correlation)
    assert e_mp2 < 0
    assert e_mp3_ladder < 0
    assert e_mp3_full < 0
    # MP3 (pp+hh) is more negative than MP2 (over-correction)
    assert e_mp3_ladder < e_mp2 - 1e-3
    # MP3 (pp+hh+ring) is BETWEEN MP2 and MP3 (pp+hh) — ring compensates
    # the over-correction of the pp ladder.
    assert e_mp2 > e_mp3_full > e_mp3_ladder, (
        f"Expected E_MP2 ({e_mp2:.4f}) > E_MP3_full ({e_mp3_full:.4f}) "
        f"> E_MP3_ladder ({e_mp3_ladder:.4f}) — ring should reduce "
        "the over-correction."
    )


def test_mp3_default_excludes_ring():
    """Phase 6.B.10b: ring is OFF by default (experimental, diverges on DZP)."""
    Z = [7, 7]
    coords = np.array([[0.0, 0.0, 0.0], [1.0975, 0.0, 0.0]])
    bonds = [(0, 1, 3.0)]
    basis, topo = build_explicit_cluster(
        Z_list=Z, coords=coords, bonds=bonds, basis_type="SZ",
    )
    rho, S, eigvals, c, conv, _ = density_matrix_PT_scf(
        topo, basis=basis, mode="hf", max_iter=10, tol=1e-3,
        n_radial=10, n_theta=8, n_phi=10,
    )
    n_occ = int(round(basis.total_occ)) // 2
    gk = dict(n_radial=10, n_theta=8, n_phi=10,
              use_becke=False, lebedev_order=14)
    mp2 = mp2_at_hf(basis, eigvals, c, n_occ, **gk)
    # default call must equal pp+hh only (ring off)
    delta_default = mp3_amplitudes_correction(basis, c, n_occ, eigvals, mp2, **gk)
    delta_pp_hh = mp3_amplitudes_correction(
        basis, c, n_occ, eigvals, mp2,
        include_pp_ladder=True, include_hh_ladder=True,
        include_ring=False, **gk,
    )
    np.testing.assert_allclose(delta_default, delta_pp_hh, atol=1e-12)
