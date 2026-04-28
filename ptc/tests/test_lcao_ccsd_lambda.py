"""Tests for Phase 6.B.11e CCSD Λ-equations skeleton.
"""
from __future__ import annotations

import numpy as np
import pytest

from ptc.lcao.ccsd import ccsd_iterate
from ptc.lcao.ccsd_lambda import (
    LambdaResult,
    lambda_initialize,
    lambda_iterate,
)
from ptc.lcao.cluster import build_explicit_cluster
from ptc.lcao.fock import density_matrix_PT_scf
from ptc.lcao.mp2 import mp2_at_hf


def _setup_n2_sz_with_ccsd():
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
    ccsd = ccsd_iterate(
        basis, c, n_occ, eigvals, mp2_result=mp2,
        max_iter=15, tol=1e-7, **gk,
    )
    return basis, eigvals, c, n_occ, gk, ccsd


# ─────────────────────────────────────────────────────────────────────
# ZAO initialisation
# ─────────────────────────────────────────────────────────────────────


def test_lambda_initialize_returns_t1_t2_copies():
    """Phase 6.B.11e — λ = T at the ZAO / variational limit."""
    _basis, _eigvals, _c, _n_occ, _gk, ccsd = _setup_n2_sz_with_ccsd()
    res = lambda_initialize(ccsd)
    assert isinstance(res, LambdaResult)
    np.testing.assert_array_equal(res.lambda1, ccsd.t1)
    np.testing.assert_array_equal(res.lambda2, ccsd.t2)
    # Must be a COPY (modifying λ should not modify T)
    res.lambda1[0, 0] = 999.0
    assert ccsd.t1[0, 0] != 999.0


def test_lambda_initialize_records_norm_in_history():
    _basis, _eigvals, _c, _n_occ, _gk, ccsd = _setup_n2_sz_with_ccsd()
    res = lambda_initialize(ccsd)
    assert len(res.history_norm) == 1
    assert res.history_norm[0] == pytest.approx(np.linalg.norm(ccsd.t2))


def test_lambda_initialize_marks_converged():
    """ZAO is the trivial fixed point at this scope, so converged=True."""
    _basis, _eigvals, _c, _n_occ, _gk, ccsd = _setup_n2_sz_with_ccsd()
    res = lambda_initialize(ccsd)
    assert res.converged is True
    assert res.n_iter == 0


# ─────────────────────────────────────────────────────────────────────
# Skeleton iteration (returns ZAO at this scope)
# ─────────────────────────────────────────────────────────────────────


def test_lambda_iterate_zao_is_fixed_point():
    """Phase 6.B.11e-bis — at the linearised CCSD-Λ level (CCD residual
    structure applied to multipliers), the ZAO seed λ2 = t2 is already
    at the fixed point of the residual function (since R(t2) = 0 at
    CCSD convergence). The iteration should converge in one step with
    ||λ2 − t2|| at numerical noise."""
    basis, eigvals, c, n_occ, gk, ccsd = _setup_n2_sz_with_ccsd()
    res = lambda_iterate(
        ccsd, basis, c, n_occ, eigvals,
        max_iter=15, tol=1e-7, **gk,
    )
    np.testing.assert_array_equal(res.lambda1, ccsd.t1)
    # λ2 differs from T2 by numerical residual noise only
    drift = float(np.linalg.norm(res.lambda2 - ccsd.t2))
    assert drift < 1e-6, (
        f"linearised CCSD-Λ iteration must keep λ2 ≈ t2 to numerical "
        f"noise; got drift = {drift}"
    )
    assert res.converged is True
    assert res.n_iter <= 5


def test_lambda_iterate_accepts_full_api_kwargs():
    """API compatibility test : lambda_iterate accepts the same
    convergence-control kwargs as ccsd_iterate, even though the
    skeleton ignores them."""
    basis, eigvals, c, n_occ, gk, ccsd = _setup_n2_sz_with_ccsd()
    # Should not raise
    res = lambda_iterate(
        ccsd, basis, c, n_occ, eigvals,
        max_iter=50, tol=1e-9, use_diis=False, diis_max_vectors=4,
        verbose=False, **gk,
    )
    assert isinstance(res, LambdaResult)


# ─────────────────────────────────────────────────────────────────────
# Lambda result shapes
# ─────────────────────────────────────────────────────────────────────


def test_lambda_result_shapes_match_ccsd_amplitudes():
    basis, _eigvals, _c, n_occ, _gk, ccsd = _setup_n2_sz_with_ccsd()
    res = lambda_initialize(ccsd)
    n_v = basis.n_orbitals - n_occ
    assert res.lambda1.shape == (n_occ, n_v)
    assert res.lambda2.shape == (n_occ, n_v, n_occ, n_v)


def test_lambda_iterate_T_fixed_breaks_zao_degeneracy():
    """Phase 6.B.11e-bis-bis — with the B-diagram T-fixed source enabled,
    the Λ2 fixed point genuinely differs from T2 (canonical CCSD-Λ
    behaviour, not variational ZAO).
    """
    basis, eigvals, c, n_occ, gk, ccsd = _setup_n2_sz_with_ccsd()
    res = lambda_iterate(
        ccsd, basis, c, n_occ, eigvals,
        max_iter=30, tol=1e-7, include_T_fixed=True, **gk,
    )
    drift = float(np.linalg.norm(res.lambda2 - ccsd.t2))
    rel_drift = drift / float(np.linalg.norm(ccsd.t2))
    # Λ ≠ T at fixed point — drift should be measurable, ~ a few percent
    assert drift > 1e-4, (
        f"include_T_fixed=True must break Λ=T degeneracy; "
        f"drift = {drift:.3e}"
    )
    # But still bounded — Λ stays in the perturbative regime
    assert rel_drift < 0.5, (
        f"||λ2-t2||/||t2|| should stay < 0.5 (perturbative); "
        f"got {rel_drift:.3e}"
    )
    assert res.converged is True


def test_lambda_iterate_T_fixed_off_recovers_zao():
    """Phase 6.B.11e-bis-bis — with include_T_fixed=False the iteration
    falls back to the Phase 6.B.11e-bis ZAO behaviour (Λ = T)."""
    basis, eigvals, c, n_occ, gk, ccsd = _setup_n2_sz_with_ccsd()
    res = lambda_iterate(
        ccsd, basis, c, n_occ, eigvals,
        max_iter=15, tol=1e-7, include_T_fixed=False, **gk,
    )
    drift = float(np.linalg.norm(res.lambda2 - ccsd.t2))
    assert drift < 1e-6, (
        f"include_T_fixed=False must recover Λ ≈ T; got drift = {drift:.3e}"
    )


def test_lambda_iterate_from_zero_seed_grows_to_t2():
    """Phase 6.B.11e-bis — starting Λ from a non-ZAO seed (here λ2=0),
    the iteration should drive λ2 toward T2 (the fixed point of the
    linearised residual) within a few iterations."""
    basis, eigvals, c, n_occ, gk, ccsd = _setup_n2_sz_with_ccsd()
    # Manually seed lambda from zero
    from ptc.lcao.ccsd_lambda import _lambda2_residual
    from ptc.lcao.ccsd import _build_ccsd_eri_blocks
    eri_blocks = _build_ccsd_eri_blocks(basis, c, n_occ, **gk)
    eri_iajb = ccsd.eri_mo_iajb

    eps_i = eigvals[:n_occ]
    eps_a = eigvals[n_occ:]
    delta = (eps_a[None, :, None, None]
             + eps_a[None, None, None, :]
             - eps_i[:, None, None, None]
             - eps_i[None, None, :, None])

    n_v = basis.n_orbitals - n_occ
    lambda2 = np.zeros((n_occ, n_v, n_occ, n_v))
    for _ in range(15):
        residual = _lambda2_residual(lambda2, eri_blocks, eri_iajb)
        lambda2 = residual / delta

    # After 15 iters from zero seed, λ2 should be close to T2
    drift = float(np.linalg.norm(lambda2 - ccsd.t2))
    assert drift < 0.3, (
        f"From zero seed, λ2 should approach T2; got drift = {drift}"
    )
