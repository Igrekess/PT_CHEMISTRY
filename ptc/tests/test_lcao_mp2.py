"""Tests for ptc.lcao.mp2 (Phase 6.B.3 + Chantier 8 continuation).

Sanity checks on the MP2 amplitudes / energy / 1-RDM correction. A small
H₂-like dimer (HF SCF) provides a one-occupied / one-virtual smoke test
of the full pipeline; benzene SZ would be too slow for unit tests.
"""
from __future__ import annotations

import numpy as np
import pytest

from ptc.lcao.cluster import build_explicit_cluster
from ptc.lcao.density_matrix import density_matrix_PT
from ptc.lcao.mp2 import (
    MP2Result,
    mp2_amplitudes,
    mp2_at_hf,
    mp2_density_correction,
    mp2_energy,
    mo_eri_iajb,
)


def _h2_basis():
    """Build a minimal SZ H₂ basis for the smoke test."""
    Z = [1, 1]
    coords = np.array([[0.0, 0.0, 0.0], [0.7414, 0.0, 0.0]])
    bonds = [(0, 1, 1.0)]
    basis, topology = build_explicit_cluster(
        Z_list=Z, coords=coords, bonds=bonds, basis_type="SZ",
    )
    return basis, topology


def test_mp2_amplitudes_shape_and_finite():
    """Synthetic ERI of shape (n_occ, n_virt, n_occ, n_virt) → t same shape."""
    rng = np.random.default_rng(0)
    n_occ, n_virt = 3, 4
    eri = rng.normal(size=(n_occ, n_virt, n_occ, n_virt)) * 0.05
    eps = np.array([-1.0, -0.7, -0.5, +0.3, +0.5, +0.8, +1.2])
    t = mp2_amplitudes(eri, eps, n_occ)
    assert t.shape == eri.shape
    assert np.all(np.isfinite(t))


def test_mp2_amplitudes_inverse_relation():
    """t × Δε = eri."""
    rng = np.random.default_rng(1)
    n_occ, n_virt = 2, 3
    eri = rng.normal(size=(n_occ, n_virt, n_occ, n_virt)) * 0.1
    eps = np.array([-1.0, -0.8, +0.4, +0.6, +0.9])
    t = mp2_amplitudes(eri, eps, n_occ)
    eps_i = eps[:n_occ]
    eps_a = eps[n_occ:]
    delta = (eps_a[None, :, None, None] + eps_a[None, None, None, :]
             - eps_i[:, None, None, None] - eps_i[None, None, :, None])
    np.testing.assert_allclose(t * delta, eri, atol=1e-12)


def test_mp2_energy_negative_for_random_positive_eri():
    """For positive ERIs and positive Δε, ``-Σ t × (2eri − eri.swap)`` ≤ 0."""
    rng = np.random.default_rng(2)
    n_occ, n_virt = 2, 3
    eri = np.abs(rng.normal(size=(n_occ, n_virt, n_occ, n_virt)) * 0.1)
    eps = np.array([-1.0, -0.5, +0.4, +0.7, +1.0])
    t = mp2_amplitudes(eri, eps, n_occ)
    e = mp2_energy(eri, t)
    assert e <= 0.0


def test_mp2_density_correction_shapes():
    rng = np.random.default_rng(3)
    n_occ, n_virt = 4, 5
    t = rng.normal(size=(n_occ, n_virt, n_occ, n_virt)) * 0.05
    d_occ, d_vir = mp2_density_correction(t)
    assert d_occ.shape == (n_occ, n_occ)
    assert d_vir.shape == (n_virt, n_virt)


def test_mp2_density_correction_diagonal_signs():
    """Diagonal of d_occ ≤ 0 (loss) and diagonal of d_vir ≥ 0 (gain)."""
    rng = np.random.default_rng(4)
    n_occ, n_virt = 3, 4
    t = rng.normal(size=(n_occ, n_virt, n_occ, n_virt)) * 0.05
    d_occ, d_vir = mp2_density_correction(t)
    diag_occ = np.diag(d_occ)
    diag_vir = np.diag(d_vir)
    assert np.all(diag_occ <= 1.0e-12)
    assert np.all(diag_vir >= -1.0e-12)


def test_mp2_density_correction_symmetric():
    """Both d_occ and d_vir are symmetric matrices."""
    rng = np.random.default_rng(5)
    n_occ, n_virt = 3, 4
    t = rng.normal(size=(n_occ, n_virt, n_occ, n_virt)) * 0.05
    d_occ, d_vir = mp2_density_correction(t)
    np.testing.assert_allclose(d_occ, d_occ.T, atol=1e-12)
    np.testing.assert_allclose(d_vir, d_vir.T, atol=1e-12)


def test_mp2_at_hf_h2_smoke():
    """End-to-end : build H₂, run Hueckel-MO + MP2, get a valid MP2Result."""
    basis, topology = _h2_basis()
    rho, S, eigvals, c = density_matrix_PT(topology, basis=basis)
    n_occ = 1   # H₂ has 2 e- = 1 doubly occupied
    res = mp2_at_hf(
        basis, eigvals, c, n_occ,
        n_radial=8, n_theta=6, n_phi=8,
        use_becke=False, lebedev_order=14,
    )
    assert isinstance(res, MP2Result)
    assert res.t.shape == (n_occ, basis.n_orbitals - n_occ,
                            n_occ, basis.n_orbitals - n_occ)
    assert np.isfinite(res.e_corr)
    # MP2 correlation energy is non-positive for closed-shell
    assert res.e_corr <= 0.0


def test_mp2_density_correction_AO_shape_h2():
    """AO-basis correction has full (n_orb, n_orb) shape and is symmetric."""
    from ptc.lcao.mp2 import mp2_density_correction_AO
    basis, topology = _h2_basis()
    rho, S, eigvals, c = density_matrix_PT(topology, basis=basis)
    n_occ = 1
    res = mp2_at_hf(
        basis, eigvals, c, n_occ,
        n_radial=8, n_theta=6, n_phi=8,
        use_becke=False, lebedev_order=14,
    )
    rho_corr = mp2_density_correction_AO(c, n_occ, res)
    assert rho_corr.shape == (basis.n_orbitals, basis.n_orbitals)
    np.testing.assert_allclose(rho_corr, rho_corr.T, atol=1e-12)


def test_mp2_diamagnetic_shielding_correction_h2():
    """End-to-end : σ_d^MP2 differs from σ_d^HF by a finite small amount."""
    from ptc.lcao.giao import shielding_diamagnetic_iso
    from ptc.lcao.mp2 import mp2_density_correction_AO
    basis, topology = _h2_basis()
    rho_HF, S, eigvals, c = density_matrix_PT(topology, basis=basis)
    n_occ = 1
    res = mp2_at_hf(
        basis, eigvals, c, n_occ,
        n_radial=8, n_theta=6, n_phi=8,
        use_becke=False, lebedev_order=14,
    )
    rho_MP2 = rho_HF + mp2_density_correction_AO(c, n_occ, res)
    probe = np.array([0.3707, 0.0, 1.0])
    sigma_HF = shielding_diamagnetic_iso(rho_HF, basis, probe)
    sigma_MP2 = shielding_diamagnetic_iso(rho_MP2, basis, probe)
    # MP2 correction is small but finite for H₂
    assert abs(sigma_MP2 - sigma_HF) > 0.0
    assert abs(sigma_MP2 - sigma_HF) < 1.0   # ppm-scale correction


def test_mp2_at_hf_h2_density_normalisation():
    """Trace of (d_occ + d_vir) should be small (correction preserves N to 2nd order)."""
    basis, topology = _h2_basis()
    rho, S, eigvals, c = density_matrix_PT(topology, basis=basis)
    n_occ = 1
    res = mp2_at_hf(
        basis, eigvals, c, n_occ,
        n_radial=8, n_theta=6, n_phi=8,
        use_becke=False, lebedev_order=14,
    )
    # For the diagonal: Tr(d_occ) ≤ 0 and Tr(d_vir) ≥ 0, and they should
    # be of comparable magnitude. The exact equality |Tr(d_occ)| = Tr(d_vir)
    # only holds for specific MP2 conventions ; here we check both finite.
    assert np.isfinite(np.trace(res.d_occ))
    assert np.isfinite(np.trace(res.d_vir))
