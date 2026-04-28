"""Tests for Phase 6.B.11a Linearized CCD (LCCD) iteration.
"""
from __future__ import annotations

import numpy as np
import pytest

from ptc.lcao.ccd import (
    LCCDResult,
    _DIIS,
    _ccd_F_oo_intermediate,
    _ccd_F_vv_intermediate,
    _ccd_quadratic_F_intermediates,
    _energy_from_amplitudes,
    ccd_iterate,
    lccd_iterate,
)
from ptc.lcao.cluster import build_explicit_cluster
from ptc.lcao.fock import density_matrix_PT_scf
from ptc.lcao.mp2 import mp2_at_hf
from ptc.lcao.mp3 import mp3_at_hf


# ─────────────────────────────────────────────────────────────────────
# Energy formula sanity
# ─────────────────────────────────────────────────────────────────────


def test_diis_first_call_is_identity():
    """Single history entry → no extrapolation possible, return t_raw."""
    diis = _DIIS(max_vectors=8)
    t_prev = np.zeros((2, 3, 2, 3))
    t_raw = np.random.RandomState(0).rand(2, 3, 2, 3)
    out = diis.extrapolate(t_raw, t_prev)
    np.testing.assert_array_equal(out, t_raw)
    assert diis.n_vectors == 1


def test_diis_truncates_history_to_max_vectors():
    """After max_vectors+k pushes, history retains only the last
    max_vectors entries."""
    diis = _DIIS(max_vectors=3)
    t_prev = np.zeros((2, 2, 2, 2))
    rng = np.random.RandomState(1)
    for _ in range(10):
        diis.extrapolate(rng.rand(2, 2, 2, 2), t_prev)
    assert diis.n_vectors == 3


def test_diis_reset_clears_history():
    diis = _DIIS(max_vectors=3)
    t_prev = np.zeros((1, 1, 1, 1))
    rng = np.random.RandomState(2)
    diis.extrapolate(rng.rand(1, 1, 1, 1), t_prev)
    diis.extrapolate(rng.rand(1, 1, 1, 1), t_prev)
    assert diis.n_vectors == 2
    diis.reset()
    assert diis.n_vectors == 0


def test_diis_invalid_max_vectors():
    with pytest.raises(ValueError):
        _DIIS(max_vectors=0)


def test_diis_extrapolation_fixed_point_property():
    """If two raw updates are identical (we are AT the fixed point),
    DIIS must return that point."""
    diis = _DIIS(max_vectors=8)
    t_fixed = np.array([[[[1.0, 2.0]]]])
    # Two pushes of the same point with err=0
    diis.extrapolate(t_fixed, t_fixed)
    out = diis.extrapolate(t_fixed, t_fixed)
    np.testing.assert_allclose(out, t_fixed, atol=1e-12)


def test_energy_from_amplitudes_zero_when_t_is_zero():
    eri = np.random.RandomState(1).rand(2, 3, 2, 3)
    t = np.zeros((2, 3, 2, 3))
    assert _energy_from_amplitudes(t, eri) == 0.0


def test_energy_from_amplitudes_linear_in_t():
    rng = np.random.RandomState(2)
    eri = rng.rand(2, 3, 2, 3)
    t1 = rng.rand(2, 3, 2, 3)
    t2 = rng.rand(2, 3, 2, 3)
    e_sum = _energy_from_amplitudes(t1 + 2.0 * t2, eri)
    e1 = _energy_from_amplitudes(t1, eri)
    e2 = _energy_from_amplitudes(t2, eri)
    assert e_sum == pytest.approx(e1 + 2.0 * e2, rel=1e-12)


# ─────────────────────────────────────────────────────────────────────
# LCCD convergence on N₂ SZ
# ─────────────────────────────────────────────────────────────────────


def _setup_n2_sz():
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
    return basis, topo, eigvals, c, n_occ, gk


def test_lccd_first_iter_recovers_MP3_with_ring():
    """LCCD's iteration 0 = MP2; iteration 1 = MP3 (with ring)."""
    basis, _topo, eigvals, c, n_occ, gk = _setup_n2_sz()
    mp2 = mp2_at_hf(basis, eigvals, c, n_occ, **gk)
    mp3_full = mp3_at_hf(
        basis, eigvals, c, n_occ, mp2_result=mp2,
        include_ring=True, **gk,
    )
    # 1 LCCD iteration = MP3 (with ring) by construction
    res = lccd_iterate(
        basis, c, n_occ, eigvals, mp2_result=mp2,
        max_iter=1, tol=0.0,         # force exactly 1 iteration
        include_pp_ladder=True, include_hh_ladder=True, include_ring=True,
        **gk,
    )
    np.testing.assert_allclose(res.t, mp3_full.t, atol=1e-10)
    assert res.e_corr == pytest.approx(mp3_full.e_corr, abs=1e-10)


def test_lccd_iterates_monotonically_jacobi_only():
    """Pure Jacobi+damping (no DIIS) iteration on N₂ SZ produces a
    monotone non-increasing energy sequence."""
    basis, _topo, eigvals, c, n_occ, gk = _setup_n2_sz()
    mp2 = mp2_at_hf(basis, eigvals, c, n_occ, **gk)
    res = lccd_iterate(
        basis, c, n_occ, eigvals, mp2_result=mp2,
        max_iter=50, tol=1e-12, damp=0.3, use_diis=False, **gk,
    )
    diffs = np.diff(res.history_e)
    assert (diffs <= 1e-9).all(), (
        f"Pure-Jacobi LCCD energy must be monotone non-increasing; "
        f"diffs = {diffs}"
    )
    assert res.e_corr < 0.0
    assert res.e_corr < 1.5 * mp2.e_corr
    assert res.t.shape == mp2.t.shape


def test_lccd_diis_converges_fast_on_N2_SZ():
    """Phase 6.B.11b — Pulay DIIS extrapolation reduces the LCCD
    convergence on N₂/SZ from > 80 pure-Jacobi iterations to < 30."""
    basis, _topo, eigvals, c, n_occ, gk = _setup_n2_sz()
    mp2 = mp2_at_hf(basis, eigvals, c, n_occ, **gk)
    res = lccd_iterate(
        basis, c, n_occ, eigvals, mp2_result=mp2,
        max_iter=30, tol=1e-7, use_diis=True, **gk,
    )
    assert res.n_iter < 30, (
        f"DIIS did not converge in 30 iters: n_iter={res.n_iter}, "
        f"final E = {res.e_corr:.6f} eV"
    )
    # Final correlation energy strictly negative
    assert res.e_corr < 0.0
    # More correlation captured than MP2
    assert res.e_corr < mp2.e_corr - 1e-3


def test_lccd_diis_consistent_with_pure_jacobi_at_convergence():
    """DIIS-converged energy must match the asymptotic Jacobi+damping
    energy within numerical noise (both target the same fixed point)."""
    basis, _topo, eigvals, c, n_occ, gk = _setup_n2_sz()
    mp2 = mp2_at_hf(basis, eigvals, c, n_occ, **gk)
    res_diis = lccd_iterate(
        basis, c, n_occ, eigvals, mp2_result=mp2,
        max_iter=30, tol=1e-7, use_diis=True, **gk,
    )
    res_jac = lccd_iterate(
        basis, c, n_occ, eigvals, mp2_result=mp2,
        max_iter=200, tol=1e-12, damp=0.5, use_diis=False, **gk,
    )
    # The two methods should converge to the same fixed point.
    # Pure Jacobi may not fully converge in 200 iters; allow 1e-3
    # absolute tolerance.
    assert abs(res_diis.e_corr - res_jac.e_corr) < 1e-3, (
        f"DIIS final E ({res_diis.e_corr:.6f}) differs from Jacobi "
        f"asymptote ({res_jac.e_corr:.6f}) by more than 1e-3 eV"
    )


def test_lccd_captures_more_correlation_than_MP2():
    """At any iteration ≥ 1, LCCD captures more correlation than MP2
    on a covalent system (iteration cumulates pp+hh+ring contributions)."""
    basis, _topo, eigvals, c, n_occ, gk = _setup_n2_sz()
    mp2 = mp2_at_hf(basis, eigvals, c, n_occ, **gk)
    lccd = lccd_iterate(
        basis, c, n_occ, eigvals, mp2_result=mp2,
        max_iter=100, tol=1e-6, damp=0.5, **gk,
    )
    # Sanity: both are negative
    assert mp2.e_corr < 0
    assert lccd.e_corr < 0
    # LCCD is more negative than MP2 (more correlation captured)
    assert lccd.e_corr < mp2.e_corr - 1e-6, (
        f"LCCD ({lccd.e_corr:.4f}) must capture more than MP2 ({mp2.e_corr:.4f})"
    )


# ─────────────────────────────────────────────────────────────────────
# Phase 6.B.11c — CCD quadratic F intermediates
# ─────────────────────────────────────────────────────────────────────


def test_ccd_F_oo_shape_and_zero_at_t_zero():
    """F_oo[k,i] = ½ Σ_lcd (kl|cd) t̃_il^cd vanishes when t = 0."""
    n_o, n_v = 3, 4
    t = np.zeros((n_o, n_v, n_o, n_v))
    eri = np.random.RandomState(0).rand(n_o, n_o, n_v, n_v)
    F = _ccd_F_oo_intermediate(t, eri)
    assert F.shape == (n_o, n_o)
    np.testing.assert_array_equal(F, 0.0)


def test_ccd_F_vv_shape_and_zero_at_t_zero():
    n_o, n_v = 3, 4
    t = np.zeros((n_o, n_v, n_o, n_v))
    eri = np.random.RandomState(1).rand(n_o, n_o, n_v, n_v)
    F = _ccd_F_vv_intermediate(t, eri)
    assert F.shape == (n_v, n_v)
    np.testing.assert_array_equal(F, 0.0)


def test_ccd_quadratic_F_intermediates_pair_symmetric():
    """The quadratic contribution must be P(ij,ab)-symmetric by
    construction (we sum quad + quad.transpose(2,3,0,1))."""
    rng = np.random.RandomState(7)
    n_o, n_v = 3, 4
    t = rng.rand(n_o, n_v, n_o, n_v) - 0.5
    eri = rng.rand(n_o, n_o, n_v, n_v) - 0.5
    quad = _ccd_quadratic_F_intermediates(t, eri)
    np.testing.assert_allclose(
        quad, quad.transpose(2, 3, 0, 1), atol=1e-12
    )


def test_ccd_quadratic_zero_at_t_zero():
    n_o, n_v = 2, 3
    t = np.zeros((n_o, n_v, n_o, n_v))
    eri = np.random.RandomState(3).rand(n_o, n_o, n_v, n_v)
    quad = _ccd_quadratic_F_intermediates(t, eri)
    np.testing.assert_array_equal(quad, 0.0)


def test_ccd_iterate_default_includes_quadratic():
    """ccd_iterate is a thin wrapper that flips include_quadratic=True."""
    basis, _topo, eigvals, c, n_occ, gk = _setup_n2_sz()
    mp2 = mp2_at_hf(basis, eigvals, c, n_occ, **gk)
    res_ccd = ccd_iterate(
        basis, c, n_occ, eigvals, mp2_result=mp2,
        max_iter=20, tol=1e-7, **gk,
    )
    res_lccd = lccd_iterate(
        basis, c, n_occ, eigvals, mp2_result=mp2,
        max_iter=20, tol=1e-7, include_quadratic=False, **gk,
    )
    # CCD should differ from LCCD (quadratic terms are non-zero)
    assert abs(res_ccd.e_corr - res_lccd.e_corr) > 1e-5, (
        f"CCD e_corr = {res_ccd.e_corr}, LCCD e_corr = {res_lccd.e_corr}; "
        "expected quadratic terms to make a difference"
    )


def test_ccd_captures_more_correlation_than_LCCD_on_N2_SZ():
    """CCD's T2² closure typically tightens the correlation energy
    beyond LCCD on covalent systems."""
    basis, _topo, eigvals, c, n_occ, gk = _setup_n2_sz()
    mp2 = mp2_at_hf(basis, eigvals, c, n_occ, **gk)
    res_ccd = ccd_iterate(
        basis, c, n_occ, eigvals, mp2_result=mp2,
        max_iter=30, tol=1e-7, **gk,
    )
    res_lccd = lccd_iterate(
        basis, c, n_occ, eigvals, mp2_result=mp2,
        max_iter=30, tol=1e-7, include_quadratic=False, **gk,
    )
    # CCD more negative than LCCD (more correlation)
    assert res_ccd.e_corr < res_lccd.e_corr - 1e-6, (
        f"CCD ({res_ccd.e_corr:.6f}) should be more negative than "
        f"LCCD ({res_lccd.e_corr:.6f})"
    )
    # And both more negative than MP2
    assert res_ccd.e_corr < mp2.e_corr - 1e-3


def test_lccd_zero_iterations_returns_MP2():
    """With max_iter=0 and tol=∞ the iteration should not run; result
    should be MP2 amplitudes."""
    basis, _topo, eigvals, c, n_occ, gk = _setup_n2_sz()
    mp2 = mp2_at_hf(basis, eigvals, c, n_occ, **gk)
    res = lccd_iterate(
        basis, c, n_occ, eigvals, mp2_result=mp2,
        max_iter=0, tol=1e10, **gk,
    )
    np.testing.assert_allclose(res.t, mp2.t, atol=1e-12)
    assert res.e_corr == pytest.approx(mp2.e_corr, abs=1e-12)
    assert res.n_iter == 0
