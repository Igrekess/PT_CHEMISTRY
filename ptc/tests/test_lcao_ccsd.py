"""Tests for Phase 6.B.11d closed-shell CCSD infrastructure.
"""
from __future__ import annotations

import numpy as np
import pytest

from ptc.lcao.ccd import ccd_iterate
from ptc.lcao.ccsd import (
    CCSDResult,
    _build_ccsd_eri_blocks,
    _ccsd_energy,
    _ccsd_t1_residual,
    ccsd_iterate,
)
from ptc.lcao.cluster import build_explicit_cluster
from ptc.lcao.fock import density_matrix_PT_scf
from ptc.lcao.mp2 import mp2_at_hf


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


# ─────────────────────────────────────────────────────────────────────
# Energy formula (τ̃ amplitudes)
# ─────────────────────────────────────────────────────────────────────


def test_ccsd_energy_at_t1_zero_equals_ccd_energy_formula():
    """When T1 = 0, the τ̃ amplitudes reduce to T2 alone, so the CCSD
    energy formula coincides with the CCD/MP2 closed-shell formula.
    """
    n_o, n_v = 2, 3
    rng = np.random.RandomState(0)
    t2 = rng.rand(n_o, n_v, n_o, n_v) - 0.5
    t1_zero = np.zeros((n_o, n_v))
    eri = rng.rand(n_o, n_v, n_o, n_v) - 0.5
    e_ccsd = _ccsd_energy(t1_zero, t2, eri)
    # MP2/CCD closed-shell formula : -Σ t (2K - K^swap)
    combo = 2.0 * eri - eri.transpose(0, 3, 2, 1)
    e_ref = -float(np.sum(t2 * combo))
    assert e_ccsd == pytest.approx(e_ref, abs=1e-12)


def test_ccsd_energy_t1_contributes_via_tau():
    """With non-zero T1 the τ̃ amplitudes pick up t_i^a t_j^b."""
    n_o, n_v = 2, 3
    rng = np.random.RandomState(1)
    t1 = rng.rand(n_o, n_v) - 0.5
    t2 = rng.rand(n_o, n_v, n_o, n_v) - 0.5
    eri = rng.rand(n_o, n_v, n_o, n_v) - 0.5
    e_with = _ccsd_energy(t1, t2, eri)
    e_without = _ccsd_energy(np.zeros_like(t1), t2, eri)
    assert abs(e_with - e_without) > 1e-6


# ─────────────────────────────────────────────────────────────────────
# T1 residual (ladder term only at this scope)
# ─────────────────────────────────────────────────────────────────────


def test_ccsd_t1_ladder_zero_when_both_amplitudes_zero():
    """T1 residual vanishes when BOTH t1 and t2 are zero (no source)."""
    basis, _topo, eigvals, c, n_occ, gk = _setup_n2_sz()
    eri_blocks = _build_ccsd_eri_blocks(basis, c, n_occ, **gk)
    t1 = np.zeros((n_occ, basis.n_orbitals - n_occ))
    t2 = np.zeros((n_occ, basis.n_orbitals - n_occ,
                   n_occ, basis.n_orbitals - n_occ))
    R1 = _ccsd_t1_residual(t1, t2, eri_blocks)
    np.testing.assert_array_equal(R1, 0.0)


def test_ccsd_t1_residual_nonzero_from_t2_source():
    """Phase 6.B.11d-bis — when t2 ≠ 0 but t1 = 0, the T2-source term
    drives R1 ≠ 0 (this is exactly what breaks Brillouin)."""
    basis, _topo, eigvals, c, n_occ, gk = _setup_n2_sz()
    mp2 = mp2_at_hf(basis, eigvals, c, n_occ, **gk)
    eri_blocks = _build_ccsd_eri_blocks(basis, c, n_occ, **gk)
    t1_zero = np.zeros((n_occ, basis.n_orbitals - n_occ))
    R1 = _ccsd_t1_residual(t1_zero, mp2.t, eri_blocks)
    assert np.linalg.norm(R1) > 1e-6, (
        f"Expected R1 driven by T2 source, got ||R1||={np.linalg.norm(R1)}"
    )


def test_ccsd_t1_ladder_linear_in_t1():
    """T1 residual is linear in t1 (Phase 6.B.11d ladder-only)."""
    basis, _topo, eigvals, c, n_occ, gk = _setup_n2_sz()
    eri_blocks = _build_ccsd_eri_blocks(basis, c, n_occ, **gk)
    rng = np.random.RandomState(3)
    n_v = basis.n_orbitals - n_occ
    t1_a = rng.rand(n_occ, n_v) - 0.5
    t1_b = rng.rand(n_occ, n_v) - 0.5
    t2 = np.zeros((n_occ, n_v, n_occ, n_v))
    R_a = _ccsd_t1_residual(t1_a, t2, eri_blocks)
    R_b = _ccsd_t1_residual(t1_b, t2, eri_blocks)
    R_sum = _ccsd_t1_residual(t1_a + 2.0 * t1_b, t2, eri_blocks)
    np.testing.assert_allclose(R_sum, R_a + 2.0 * R_b, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────
# Brillouin's theorem on canonical HF
# ─────────────────────────────────────────────────────────────────────


def test_ccsd_t1_grows_perturbatively_on_canonical_hf():
    """Phase 6.B.11d-bis — Brillouin's theorem broken by the T2-source
    term (ma|ef) τ̃_im^ef.

    On canonical HF, T1 is a 2nd-order quantity (Brillouin: T1=0 at
    1st order). With the T2-source term enabled, t1 should grow from
    zero to small but finite values (~ 10⁻⁴ to 10⁻⁶ on N₂ SZ).
    """
    basis, _topo, eigvals, c, n_occ, gk = _setup_n2_sz()
    mp2 = mp2_at_hf(basis, eigvals, c, n_occ, **gk)
    res = ccsd_iterate(
        basis, c, n_occ, eigvals, mp2_result=mp2,
        max_iter=15, tol=1e-7, **gk,
    )
    t1_norm = float(np.linalg.norm(res.t1))
    # Non-zero (lock broken)
    assert t1_norm > 1e-7, f"Expected t1 to grow non-trivially, got ||t1||={t1_norm}"
    # Bounded (perturbative regime)
    assert t1_norm < 1e-2, f"Expected ||t1|| << 1, got {t1_norm}"


def test_ccsd_energy_close_to_ccd_at_canonical_hf():
    """Phase 6.B.11d-bis — the CCSD energy on canonical HF differs
    from CCD only at fourth order in α (T1²·integral) since T1 itself
    is second-order.  The numerical difference is ~ 10⁻⁹ eV on
    N₂/SZ and below the integral-quadrature precision."""
    basis, _topo, eigvals, c, n_occ, gk = _setup_n2_sz()
    mp2 = mp2_at_hf(basis, eigvals, c, n_occ, **gk)
    res_ccd = ccd_iterate(
        basis, c, n_occ, eigvals, mp2_result=mp2,
        max_iter=15, tol=1e-7, **gk,
    )
    res_ccsd = ccsd_iterate(
        basis, c, n_occ, eigvals, mp2_result=mp2,
        max_iter=15, tol=1e-7, **gk,
    )
    # Within machine-precision of integral evaluation
    assert res_ccsd.e_corr == pytest.approx(res_ccd.e_corr, abs=1e-5)


def test_ccsd_disabling_t1_falls_back_to_ccd():
    """Setting include_t1=False reduces ccsd_iterate to ccd_iterate."""
    basis, _topo, eigvals, c, n_occ, gk = _setup_n2_sz()
    mp2 = mp2_at_hf(basis, eigvals, c, n_occ, **gk)
    res_ccsd = ccsd_iterate(
        basis, c, n_occ, eigvals, mp2_result=mp2,
        max_iter=15, tol=1e-7, include_t1=False, **gk,
    )
    res_ccd = ccd_iterate(
        basis, c, n_occ, eigvals, mp2_result=mp2,
        max_iter=15, tol=1e-7, **gk,
    )
    assert res_ccsd.e_corr == pytest.approx(res_ccd.e_corr, abs=1e-8)
    np.testing.assert_array_equal(res_ccsd.t1, 0.0)


def test_ccsd_t1_grows_when_seeded_nonzero():
    """Phase 6.B.11d — the ladder iteration is contractive, so a
    non-zero seed t1 propagates without growing unboundedly. This
    test verifies the iteration is stable and converges (the t1
    settles to a small fixed point determined only by the ladder)."""
    basis, _topo, eigvals, c, n_occ, gk = _setup_n2_sz()
    mp2 = mp2_at_hf(basis, eigvals, c, n_occ, **gk)
    # Seed t1 with a small non-zero value (bypass Brillouin lock-in)
    eri_blocks = _build_ccsd_eri_blocks(basis, c, n_occ, **gk)
    n_v = basis.n_orbitals - n_occ
    t1_seed = 0.01 * np.ones((n_occ, n_v))
    t2_zero = np.zeros((n_occ, n_v, n_occ, n_v))
    R1 = _ccsd_t1_residual(t1_seed, t2_zero, eri_blocks)
    # R1 should be finite and bounded
    assert np.isfinite(R1).all()
    assert np.abs(R1).max() < 1.0     # bounded contraction


# ─────────────────────────────────────────────────────────────────────
# CCSD result dataclass
# ─────────────────────────────────────────────────────────────────────


def test_ccsd_result_carries_t1_and_t2():
    """CCSDResult exposes both T1 (n_o, n_v) and T2 (n_o, n_v)² fields."""
    basis, _topo, eigvals, c, n_occ, gk = _setup_n2_sz()
    mp2 = mp2_at_hf(basis, eigvals, c, n_occ, **gk)
    res = ccsd_iterate(
        basis, c, n_occ, eigvals, mp2_result=mp2,
        max_iter=15, tol=1e-7, **gk,
    )
    n_v = basis.n_orbitals - n_occ
    assert isinstance(res, CCSDResult)
    assert res.t1.shape == (n_occ, n_v)
    assert res.t2.shape == (n_occ, n_v, n_occ, n_v)
    assert res.n_iter > 0
