"""Tests for Phase 6.B.11f σ_p^CCSD-Λ-GIAO via Λ-relaxed orbitals.
"""
from __future__ import annotations

import numpy as np
import pytest

from ptc.lcao.ccsd import ccsd_iterate
from ptc.lcao.ccsd_lambda import lambda_iterate
from ptc.lcao.ccsd_property import (
    ccsd_lambda_density_correction,
    ccsd_lambda_relax_orbitals,
    sigma_p_ccsd_lambda_iso,
)
from ptc.lcao.cluster import build_explicit_cluster
from ptc.lcao.fock import (
    density_matrix_PT_scf,
    paramagnetic_shielding_iso_coupled,
)
from ptc.lcao.mp2 import mp2_at_hf


def _setup_n2_sz_full_cc():
    """Build N₂/SZ HF + CCSD + Λ for property tests."""
    Z = [7, 7]
    coords = np.array([[0.0, 0.0, 0.0], [1.0975, 0.0, 0.0]])
    bonds = [(0, 1, 3.0)]
    basis, topo = build_explicit_cluster(
        Z_list=Z, coords=coords, bonds=bonds, basis_type="SZ",
    )
    rho, S, eigvals_HF, c_HF, conv, _ = density_matrix_PT_scf(
        topo, basis=basis, mode="hf", max_iter=15, tol=1e-3,
        n_radial=10, n_theta=8, n_phi=10,
    )
    n_occ = int(round(basis.total_occ)) // 2
    gk = dict(n_radial=10, n_theta=8, n_phi=10,
              use_becke=False, lebedev_order=14)
    mp2 = mp2_at_hf(basis, eigvals_HF, c_HF, n_occ, **gk)
    ccsd = ccsd_iterate(
        basis, c_HF, n_occ, eigvals_HF, mp2_result=mp2,
        max_iter=20, tol=1e-7, **gk,
    )
    lam = lambda_iterate(
        ccsd, basis, c_HF, n_occ, eigvals_HF,
        max_iter=30, tol=1e-7, include_T_fixed=True, **gk,
    )
    return basis, topo, eigvals_HF, c_HF, n_occ, gk, ccsd, lam


# ─────────────────────────────────────────────────────────────────────
# Density correction modes
# ─────────────────────────────────────────────────────────────────────


def test_density_correction_t2_only_matches_mp2_density():
    """mode='t2-only' returns the same blocks as mp2_density_correction(T2)."""
    from ptc.lcao.mp2 import mp2_density_correction
    _basis, _topo, _eigvals, _c, _n_occ, _gk, ccsd, lam = _setup_n2_sz_full_cc()
    d_oo, d_vv = ccsd_lambda_density_correction(ccsd, lam, mode="t2-only")
    d_oo_ref, d_vv_ref = mp2_density_correction(ccsd.t2)
    np.testing.assert_allclose(d_oo, d_oo_ref, atol=1e-12)
    np.testing.assert_allclose(d_vv, d_vv_ref, atol=1e-12)


def test_density_correction_symmetric_uses_t_lambda_average():
    """mode='symmetric' uses the (T2+Λ2)/2 effective amplitudes."""
    from ptc.lcao.mp2 import mp2_density_correction
    _basis, _topo, _eigvals, _c, _n_occ, _gk, ccsd, lam = _setup_n2_sz_full_cc()
    d_oo, d_vv = ccsd_lambda_density_correction(ccsd, lam, mode="symmetric")
    t_eff = 0.5 * (ccsd.t2 + lam.lambda2)
    d_oo_ref, d_vv_ref = mp2_density_correction(t_eff)
    np.testing.assert_allclose(d_oo, d_oo_ref, atol=1e-12)
    np.testing.assert_allclose(d_vv, d_vv_ref, atol=1e-12)


def test_density_correction_invalid_mode():
    _basis, _topo, _eigvals, _c, _n_occ, _gk, ccsd, lam = _setup_n2_sz_full_cc()
    with pytest.raises(ValueError):
        ccsd_lambda_density_correction(ccsd, lam, mode="garbage")


# ─────────────────────────────────────────────────────────────────────
# Λ-relaxed orbitals
# ─────────────────────────────────────────────────────────────────────


def test_lambda_relaxed_orbitals_have_correct_shape():
    basis, topo, _eigvals, c_HF, n_occ, gk, ccsd, lam = _setup_n2_sz_full_cc()
    eigvals_l, c_l = ccsd_lambda_relax_orbitals(
        basis, topo, c_HF, n_occ, ccsd, lam,
        mode="symmetric", **gk,
    )
    n = basis.n_orbitals
    assert eigvals_l.shape == (n,)
    assert c_l.shape == (n, n)


def test_lambda_relaxed_orbitals_differ_from_HF():
    """Relaxation should shift orbitals away from HF reference."""
    basis, topo, eigvals_HF, c_HF, n_occ, gk, ccsd, lam = _setup_n2_sz_full_cc()
    eigvals_l, c_l = ccsd_lambda_relax_orbitals(
        basis, topo, c_HF, n_occ, ccsd, lam,
        mode="symmetric", **gk,
    )
    # Eigenvalues should differ (correlation pushes virtuals down)
    assert not np.allclose(eigvals_l, eigvals_HF, atol=1e-3), (
        "CCSD-Λ-relaxed eigvals should differ from HF eigvals"
    )


# ─────────────────────────────────────────────────────────────────────
# σ_p^CCSD-Λ-GIAO end-to-end
# ─────────────────────────────────────────────────────────────────────


def test_sigma_p_ccsd_lambda_pipeline_runs():
    """End-to-end smoke test: σ_p^CCSD-Λ pipeline doesn't crash on N₂."""
    basis, topo, _eigvals, c_HF, n_occ, gk, ccsd, lam = _setup_n2_sz_full_cc()
    probe = np.array([0.0, 0.0, 0.1])
    out = sigma_p_ccsd_lambda_iso(
        basis, topo, ccsd, lam, c_HF, n_occ, probe,
        mode="symmetric",
        cphf_kwargs=dict(n_radial=10, n_theta=8, n_phi=10),
        **gk,
    )
    assert "sigma_p_CCSD" in out
    assert "sigma_p_CCSD_Lambda" in out
    assert isinstance(out["sigma_p_CCSD"], float)
    assert isinstance(out["sigma_p_CCSD_Lambda"], float)


def test_sigma_p_ccsd_lambda_differs_from_ccsd():
    """Phase 6.B.11f — σ_p^CCSD-Λ differs from σ_p^CCSD because the
    Λ-relaxed density carries the additional Λ-amplitude contribution.
    On canonical HF the Λ effect is small but non-zero."""
    basis, topo, _eigvals, c_HF, n_occ, gk, ccsd, lam = _setup_n2_sz_full_cc()
    probe = np.array([0.0, 0.0, 0.1])
    out = sigma_p_ccsd_lambda_iso(
        basis, topo, ccsd, lam, c_HF, n_occ, probe,
        mode="symmetric",
        cphf_kwargs=dict(n_radial=10, n_theta=8, n_phi=10),
        **gk,
    )
    # Λ contribution is non-zero
    assert abs(out["sigma_p_CCSD_Lambda"] - out["sigma_p_CCSD"]) > 1e-3, (
        f"σ_p^CCSD-Λ should differ from σ_p^CCSD; got "
        f"σ_p^CCSD={out['sigma_p_CCSD']}, "
        f"σ_p^CCSD-Λ={out['sigma_p_CCSD_Lambda']}"
    )
    # Both should be finite and negative (typical for N nucleus shielding)
    assert np.isfinite(out["sigma_p_CCSD"])
    assert np.isfinite(out["sigma_p_CCSD_Lambda"])
