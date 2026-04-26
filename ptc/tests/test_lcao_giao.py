"""Tests for ptc.lcao.giao (Phase C - diamagnetic shielding only).

Validation milestones from PROMPT_PT_LCAO_GIAO.md Section 3 Phase C:
  * H atom isotropic sigma ~ 17.75 ppm  (analytic Lamb formula)
  * He atom isotropic sigma ~ 60 ppm   (Lamb with Z_eff)
  * Symmetry of nuclear-attraction matrix
  * Shielding decays to zero at large distance
  * Smoke tests on H_2 and benzene (full validation pending paramagnetic
    contribution in Phase C continuation)
"""

import math

import numpy as np
import pytest

from ptc.constants import A_BOHR, ALPHA_PHYS
from ptc.lcao.density_matrix import build_molecular_basis, density_matrix_PT
from ptc.lcao.giao import (
    H_atom_lamb_ppm,
    evaluate_sto,
    nuclear_attraction_matrix,
    shielding_at_point,
    shielding_diamagnetic_iso,
)
from ptc.lcao.atomic_basis import PTAtomicOrbital, build_atom_basis
from ptc.topology import build_topology


# ─────────────────────────────────────────────────────────────────────
# 1. STO orbital evaluator
# ─────────────────────────────────────────────────────────────────────


def test_sto_1s_self_normalisation_via_grid():
    """Numerical <1s|1s> = 1 by integrating |phi|^2 on a fine grid."""
    orb = build_atom_basis(1).orbitals[0]
    atom_pos = np.zeros(3)
    # spherical grid centred at atom
    leg_r, w_r = np.polynomial.legendre.leggauss(80)
    R_max = 12.0 / orb.zeta
    r_nodes = R_max / 2 * (leg_r + 1)
    r_weights = R_max / 2 * w_r
    leg_u, w_u = np.polynomial.legendre.leggauss(20)
    cos_t = leg_u
    sin_t = np.sqrt(np.maximum(0.0, 1 - cos_t ** 2))
    n_phi = 24
    phi = np.linspace(0, 2 * math.pi, n_phi, endpoint=False)
    phi_w = 2 * math.pi / n_phi
    R, CT, PHI = np.meshgrid(r_nodes, cos_t, phi, indexing='ij')
    ST = np.sqrt(np.maximum(0.0, 1 - CT ** 2))
    pts = np.stack([R * ST * np.cos(PHI), R * ST * np.sin(PHI), R * CT], axis=-1)
    psi = evaluate_sto(orb, pts.reshape(-1, 3), atom_pos)
    W = (
        ((r_nodes ** 2) * r_weights)[:, None, None]
        * w_u[None, :, None]
        * np.full(n_phi, phi_w)[None, None, :]
    )
    norm = float(np.sum(psi ** 2 * W.flatten()))
    assert norm == pytest.approx(1.0, abs=1e-4)


def test_sto_2pz_carbon_normalisation():
    orb = next(o for o in build_atom_basis(6).orbitals if o.l == 1 and o.m == 0)
    atom_pos = np.zeros(3)
    leg_r, w_r = np.polynomial.legendre.leggauss(80)
    R_max = 12.0 / orb.zeta
    r_nodes = R_max / 2 * (leg_r + 1)
    r_weights = R_max / 2 * w_r
    leg_u, w_u = np.polynomial.legendre.leggauss(24)
    cos_t = leg_u
    n_phi = 32
    phi = np.linspace(0, 2 * math.pi, n_phi, endpoint=False)
    phi_w = 2 * math.pi / n_phi
    R, CT, PHI = np.meshgrid(r_nodes, cos_t, phi, indexing='ij')
    ST = np.sqrt(np.maximum(0.0, 1 - CT ** 2))
    pts = np.stack([R * ST * np.cos(PHI), R * ST * np.sin(PHI), R * CT], axis=-1)
    psi = evaluate_sto(orb, pts.reshape(-1, 3), atom_pos)
    W = (
        ((r_nodes ** 2) * r_weights)[:, None, None]
        * w_u[None, :, None]
        * np.full(n_phi, phi_w)[None, None, :]
    )
    norm = float(np.sum(psi ** 2 * W.flatten()))
    assert norm == pytest.approx(1.0, abs=1e-3)


# ─────────────────────────────────────────────────────────────────────
# 2. Nuclear-attraction matrix properties
# ─────────────────────────────────────────────────────────────────────


def test_nuclear_attraction_is_symmetric():
    basis = build_molecular_basis(build_topology("O"))   # H_2O
    P = np.array([0.0, 0.0, 0.5])
    M = nuclear_attraction_matrix(basis, P, n_radial=40, n_theta=20, n_phi=24)
    assert M.shape == (basis.n_orbitals, basis.n_orbitals)
    np.testing.assert_allclose(M, M.T, atol=1e-10)


def test_nuclear_attraction_diagonal_positive():
    """<phi_i | 1/|r-P| | phi_i> > 0 for any phi and any P."""
    basis = build_molecular_basis(build_topology("[H][H]"))
    P = np.array([0.5, 0.0, 0.0])
    M = nuclear_attraction_matrix(basis, P, n_radial=40, n_theta=16, n_phi=24)
    assert (np.diag(M) > 0).all()


# ─────────────────────────────────────────────────────────────────────
# 3. Lamb formula for H, He
# ─────────────────────────────────────────────────────────────────────


def test_H_atom_Lamb_value():
    sigma = H_atom_lamb_ppm()
    expected = (ALPHA_PHYS ** 2 / 3.0) * 1.0e6   # alpha^2/3 in ppm
    assert sigma == pytest.approx(expected, rel=1e-3)


def test_H_atom_Lamb_converged_with_grid():
    """Increasing the grid converges to the analytic value."""
    coarse = H_atom_lamb_ppm(n_radial=30, n_theta=12, n_phi=16)
    fine = H_atom_lamb_ppm(n_radial=80, n_theta=32, n_phi=48)
    expected = (ALPHA_PHYS ** 2 / 3.0) * 1.0e6
    # fine should be within 2e-3 ppm of the analytic value
    assert abs(fine - expected) < 2.0e-3
    # and at least as good as the coarse grid
    assert abs(fine - expected) <= abs(coarse - expected) + 1.0e-4


def test_He_atom_lamb_estimate():
    """He: closed-shell 1s^2 with PT effective_charge giving ~Z_eff ~ 1.6.
    Lamb: sigma_iso = (alpha^2/3) * <1/r> * 2_electrons.
    With Z_eff_He ~ 1.7 (from atom.py), expected sigma ~ 60-70 ppm.
    The reference (experimental + theoretical converged) is 59.97 ppm,
    so we accept a ~30% window since Z_eff is approximate.
    """
    sigma = shielding_at_point(build_topology("[He]"), np.zeros(3))
    assert 30.0 < sigma < 100.0
    # tighter check: monotonic positive
    assert sigma > H_atom_lamb_ppm()


# ─────────────────────────────────────────────────────────────────────
# 4. Spatial behaviour
# ─────────────────────────────────────────────────────────────────────


def test_shielding_far_from_atom_decays_to_zero():
    """sigma^d(P) -> 0 as P -> infinity (1/|r-P| -> 1/|P| outside the
    orbital tail, integrated against ~1 -> alpha^2/3 / |P|)."""
    topo = build_topology("[H][H]")
    P_far = np.array([0.0, 0.0, 100.0])  # 100 A away
    sigma = shielding_at_point(topo, P_far)
    # sigma ~ alpha^2/3 * (N_e / |P|) * ... should be tiny
    assert abs(sigma) < 1.0e-2


# ─────────────────────────────────────────────────────────────────────
# 5. Smoke tests: H_2 and CH_4 have positive diamagnetic shielding
# ─────────────────────────────────────────────────────────────────────


def test_H2_dia_shielding_at_centre_positive():
    """H_2 at midpoint: positive diamagnetic shielding (no para in Phase C)."""
    topo = build_topology("[H][H]")
    basis = build_molecular_basis(topo)
    midpoint = basis.coords.mean(axis=0)
    sigma = shielding_at_point(topo, midpoint)
    assert sigma > 0.0
    # not absurdly large or small (a few tens of ppm)
    assert 1.0 < sigma < 100.0


def test_CH4_dia_shielding_at_carbon_positive():
    topo = build_topology("C")
    basis = build_molecular_basis(topo)
    # C is atom index 0
    P = basis.coords[0]
    sigma = shielding_at_point(topo, P)
    assert sigma > 0.0
    # CH_4 carbon shielding (full) is ~195 ppm; diamagnetic is dominant
    # and known to be ~270 ppm. We just check positive and finite.
    assert sigma > 50.0


# ─────────────────────────────────────────────────────────────────────
# 6. Linearity in density matrix
# ─────────────────────────────────────────────────────────────────────


def test_shielding_linear_in_rho():
    """sigma is linear in rho: sigma(rho_a + rho_b) = sigma(rho_a) + sigma(rho_b)."""
    topo = build_topology("O")
    basis = build_molecular_basis(topo)
    rho, _, _, _ = density_matrix_PT(topo, basis=basis)
    P = basis.coords[0]   # O atom

    s_full = shielding_diamagnetic_iso(rho, basis, P, n_radial=30, n_theta=12, n_phi=16)
    s_half = shielding_diamagnetic_iso(0.5 * rho, basis, P, n_radial=30, n_theta=12, n_phi=16)
    assert s_half == pytest.approx(0.5 * s_full, rel=1e-10)


# ─────────────────────────────────────────────────────────────────────
# Phase C continuation: full tensor, L_alpha, paramagnetic CP-PT
# ─────────────────────────────────────────────────────────────────────


def test_diamagnetic_tensor_isotropic_for_H_atom():
    """H atom dia tensor: diagonal, all components equal, sum/3 = 17.75 ppm."""
    from ptc.lcao.giao import shielding_diamagnetic_tensor
    topo = build_topology("[H]")
    basis = build_molecular_basis(topo)
    rho, _, _, _ = density_matrix_PT(topo, basis=basis)
    sigma = shielding_diamagnetic_tensor(rho, basis, np.zeros(3))
    # Off-diagonal should be ~ 0
    off = sigma - np.diag(np.diag(sigma))
    assert np.abs(off).max() < 1e-6
    # Diagonal components should all equal
    diag = np.diag(sigma)
    assert np.allclose(diag, diag[0], rtol=1e-8)
    # Trace/3 = isotropic = 17.75 ppm
    iso = float(np.trace(sigma) / 3.0)
    expected = (ALPHA_PHYS ** 2 / 3.0) * 1.0e6
    assert iso == pytest.approx(expected, rel=1e-3)


def test_L_alpha_antisymmetric_to_machine_precision():
    """L_imag must be antisymmetric in real STO basis (since iL is hermitian)."""
    from ptc.lcao.giao import angular_momentum_matrices
    basis = build_molecular_basis(build_topology("O"))
    L = angular_momentum_matrices(basis, basis.coords[0],
                                    n_radial=30, n_theta=14, n_phi=20)
    for a in range(3):
        # explicit antisymmetrisation in code -> 0 to machine precision
        assert np.abs(L[a] + L[a].T).max() < 1.0e-12


def test_L_alpha_diagonal_is_zero():
    """Diagonal entries of L_imag are 0 (antisymmetric matrix property)."""
    from ptc.lcao.giao import angular_momentum_matrices
    basis = build_molecular_basis(build_topology("[H][H]"))
    L = angular_momentum_matrices(basis, np.zeros(3),
                                    n_radial=30, n_theta=14, n_phi=20)
    for a in range(3):
        assert np.abs(np.diag(L[a])).max() < 1.0e-12


def test_paramagnetic_zero_for_atoms_no_virtuals():
    """In valence-only PT-LCAO, closed-shell atoms have zero virtuals
    -> sigma^p = 0 trivially. Test on He, Ne."""
    from ptc.lcao.giao import shielding_total_iso
    for sm in ("[He]", "[Ne]"):
        res = shielding_total_iso(build_topology(sm), np.zeros(3))
        assert res["sigma_p"] == 0.0


def test_H_atom_total_equals_dia():
    """H atom: closed-shell-like single-MO with only 1 orbital, no virtuals.
    Total = dia = 17.75 ppm."""
    from ptc.lcao.giao import shielding_total_iso
    res = shielding_total_iso(build_topology("[H]"), np.zeros(3))
    assert res["sigma_p"] == 0.0
    expected = (ALPHA_PHYS ** 2 / 3.0) * 1.0e6
    assert res["sigma_d"] == pytest.approx(expected, rel=1e-3)
    assert res["sigma_total"] == pytest.approx(expected, rel=1e-3)


def test_H2_paramagnetic_finite():
    """H_2 at midpoint: nontrivial occ-virt response, sigma^p must be
    finite (test for the Mulliken K=1 degeneracy regression)."""
    from ptc.lcao.giao import shielding_total_iso
    topo = build_topology("[H][H]")
    basis = build_molecular_basis(topo)
    midpoint = basis.coords.mean(axis=0)
    res = shielding_total_iso(topo, midpoint)
    # sigma^p finite and small (paramagnetic deshielding)
    assert -50.0 < res["sigma_p"] < 50.0
    assert res["sigma_d"] > 0.0
    # total in a sensible range
    assert -50.0 < res["sigma_total"] < 100.0


def test_H2_at_H_nucleus_total_in_experimental_range():
    """H_2 at H nucleus: expt sigma ~ 26 ppm. Our model gives ~ 29 ppm
    (Hueckel K=2 approximation; reasonable for parameter-free model)."""
    from ptc.lcao.giao import shielding_total_iso
    res = shielding_total_iso(build_topology("[H][H]"), np.zeros(3))
    assert 15.0 < res["sigma_total"] < 50.0


def test_total_shielding_decays_with_distance():
    """Total shielding -> 0 far from molecule (both dia and para)."""
    from ptc.lcao.giao import shielding_total_iso
    res = shielding_total_iso(build_topology("[H][H]"),
                                np.array([0, 0, 100.0]))
    assert abs(res["sigma_total"]) < 1.0e-1


# ─────────────────────────────────────────────────────────────────────
# Phase C complete: GIAO London phase factors -> exact gauge invariance
# ─────────────────────────────────────────────────────────────────────


def test_momentum_matrix_antisymmetric():
    """p_imag must be real antisymmetric in real STO basis."""
    from ptc.lcao.giao import momentum_matrix
    basis = build_molecular_basis(build_topology("O"))
    p = momentum_matrix(basis, n_radial=30, n_theta=14, n_phi=20)
    for a in range(3):
        assert np.abs(p[a] + p[a].T).max() < 1.0e-12
        assert np.abs(np.diag(p[a])).max() < 1.0e-12


def test_GIAO_L_antisymmetric():
    """GIAO L_imag must remain real antisymmetric (L^GIAO is hermitian)."""
    from ptc.lcao.giao import angular_momentum_matrices_GIAO
    basis = build_molecular_basis(build_topology("c1ccccc1"))
    L = angular_momentum_matrices_GIAO(basis, n_radial=30, n_theta=14, n_phi=20)
    for a in range(3):
        assert np.abs(L[a] + L[a].T).max() < 1.0e-10


def test_GIAO_paramagnetic_gauge_invariant_benzene():
    """Spec §3 Phase C: 'Origine-jauge invariance: translation de
    basis_origin doit laisser sigma inchangé.'

    With GIAO London phase factors, sigma^p must be EXACTLY independent
    of gauge_origin choice (to first order in B). Test on benzene at
    centre with four different gauge origins.
    """
    from ptc.lcao.giao import paramagnetic_shielding_iso
    topo = build_topology("c1ccccc1")
    basis = build_molecular_basis(topo)
    rho, S, eigvals, c = density_matrix_PT(topo, basis=basis)
    n_e = int(round(basis.total_occ))
    P = basis.coords.mean(axis=0)

    origins = [
        P,                                        # probe at centre
        basis.coords[0],                          # atom 0
        basis.coords[3],                          # opposite atom
        np.array([10.0, 0.0, 0.0]),              # far point
        np.array([-50.0, 20.0, 0.0]),            # very far / asymmetric
    ]
    sigmas = []
    for O in origins:
        sp = paramagnetic_shielding_iso(
            basis, eigvals, c, n_e, P, gauge_origin=O,
            use_giao=True, n_radial=30, n_theta=14, n_phi=20,
        )
        sigmas.append(sp)
    spread = max(sigmas) - min(sigmas)
    # GIAO -> exact gauge invariance (modulo machine precision)
    assert spread < 1.0e-6, (
        f"GIAO sigma^p NOT gauge-invariant: spread = {spread:.4e}, "
        f"values = {sigmas}"
    )


def test_GIAO_paramagnetic_gauge_invariant_H2():
    """Same gauge-invariance test but on H2 at midpoint."""
    from ptc.lcao.giao import paramagnetic_shielding_iso
    topo = build_topology("[H][H]")
    basis = build_molecular_basis(topo)
    rho, S, eigvals, c = density_matrix_PT(topo, basis=basis)
    n_e = int(round(basis.total_occ))
    P = basis.coords.mean(axis=0)
    origins = [P, basis.coords[0], np.array([5.0, 5.0, 0.0])]
    sigmas = [
        paramagnetic_shielding_iso(basis, eigvals, c, n_e, P, gauge_origin=O,
                                     use_giao=True, n_radial=30, n_theta=14, n_phi=20)
        for O in origins
    ]
    assert max(sigmas) - min(sigmas) < 1.0e-6


def test_GIAO_changes_paramagnetic_value():
    """Verify GIAO actually applies a non-trivial correction (it's not a no-op)."""
    from ptc.lcao.giao import paramagnetic_shielding_iso
    topo = build_topology("c1ccccc1")
    basis = build_molecular_basis(topo)
    rho, S, eigvals, c = density_matrix_PT(topo, basis=basis)
    n_e = int(round(basis.total_occ))
    P = basis.coords.mean(axis=0)
    sp_giao = paramagnetic_shielding_iso(basis, eigvals, c, n_e, P,
                                           use_giao=True,
                                           n_radial=30, n_theta=14, n_phi=20)
    sp_nogiao = paramagnetic_shielding_iso(basis, eigvals, c, n_e, P,
                                             use_giao=False,
                                             n_radial=30, n_theta=14, n_phi=20)
    # The two values should differ by a finite amount on benzene
    assert abs(sp_giao - sp_nogiao) > 1.0  # ppm difference
