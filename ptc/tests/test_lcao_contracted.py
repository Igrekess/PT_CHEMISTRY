"""Tests for Phase 6.B.8 contracted-polarisation PT-pure orbitals.

Validation criteria from PROMPT_PT_LCAO_QUANT.md Chantier 3:
    * shape / dataclass invariants
    * normalisation Sum_ij c_i c_j S_ij = 1
    * analytic gradient consistent with finite differences
    * N=1 contraction equivalent to single-zeta STO
    * matrix integrals coherent with single-primitive equivalents
    * pVDZ-PT builder produces well-conditioned overlap

Constants and primitives are built from PT (GAMMA_3, GAMMA_5) so that
nothing is fitted: every coefficient traces back to s = 1/2.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ptc.constants import GAMMA_3, GAMMA_5
from ptc.lcao.atomic_basis import (
    PTAtomicOrbital,
    PTContractedOrbital,
    build_atom_basis,
    cascade_pt_contraction,
    iter_primitives,
    overlap_atomic,
)
from ptc.lcao.giao import (
    evaluate_sto,
    evaluate_sto_gradient,
    evaluate_sto_gradient_analytic,
)


# ─────────────────────────────────────────────────────────────────────
# Dataclass invariants
# ─────────────────────────────────────────────────────────────────────


def test_contracted_orbital_dataclass_basic():
    prims = ((1.5, 0.6), (1.0, 0.4))
    orb = PTContractedOrbital(Z=6, n=2, l=1, m=0, primitives=prims, occ=2.0)
    assert orb.Z == 6
    assert orb.n == 2
    assert orb.l == 1
    assert orb.m == 0
    assert orb.occ == 2.0
    assert len(orb.primitives) == 2
    # zeta property returns smallest primitive zeta (most diffuse)
    assert orb.zeta == pytest.approx(1.0)
    # frozen — assignment raises
    with pytest.raises(Exception):
        orb.Z = 7  # type: ignore[misc]


def test_contracted_orbital_label():
    orb = PTContractedOrbital(
        Z=7, n=2, l=1, m=-1,
        primitives=((1.0, 1.0), (0.7, 0.3)),
    )
    assert "Z7_2p_m-1_C2" in orb.label


def test_iter_primitives_dispatch_atomic():
    a = PTAtomicOrbital(Z=1, n=1, l=0, m=0, zeta=1.0, occ=1.0)
    pairs = list(iter_primitives(a))
    assert len(pairs) == 1
    c, prim = pairs[0]
    assert c == 1.0
    assert prim is a


def test_iter_primitives_dispatch_contracted():
    orb = PTContractedOrbital(
        Z=6, n=2, l=0, m=0,
        primitives=((1.5, 0.6), (1.0, 0.4)),
        occ=2.0,
    )
    pairs = list(iter_primitives(orb))
    assert len(pairs) == 2
    for (c, prim), (z_ref, c_ref) in zip(pairs, orb.primitives):
        assert isinstance(prim, PTAtomicOrbital)
        assert prim.Z == 6 and prim.n == 2 and prim.l == 0 and prim.m == 0
        assert prim.zeta == pytest.approx(z_ref)
        assert c == pytest.approx(c_ref)


# ─────────────────────────────────────────────────────────────────────
# Cascade PT contraction (Option B)
# ─────────────────────────────────────────────────────────────────────


def test_cascade_pt_contraction_cumulative_cascade_zetas():
    """Phase 6.B.8b: zetas follow the cumulative cascade
        z_base * {1, GAMMA_3, GAMMA_3*GAMMA_5, GAMMA_3*GAMMA_5*GAMMA_7}
    (10× span, matching Dunning cc-pVDZ exponent ratios)."""
    from ptc.constants import GAMMA_7
    z_base = 2.0
    prims = cascade_pt_contraction(z_base, n=2, n_prim=4, option="B")
    assert len(prims) == 4
    expected = [
        z_base,
        z_base * GAMMA_3,
        z_base * GAMMA_3 * GAMMA_5,
        z_base * GAMMA_3 * GAMMA_5 * GAMMA_7,
    ]
    for (zeta_k, _c), exp_z in zip(prims, expected):
        assert zeta_k == pytest.approx(exp_z)


def test_cascade_pt_contraction_normalised():
    """<chi|chi> = Sum_ij c_i c_j S_ij must equal 1 to machine epsilon."""
    prims = cascade_pt_contraction(2.0, n=2, n_prim=4, option="B")
    orb = PTContractedOrbital(
        Z=6, n=2, l=1, m=0, primitives=tuple(prims), occ=0.0,
    )
    overlap = overlap_atomic(orb, orb, 0.0)
    assert overlap == pytest.approx(1.0, abs=1e-12)


def test_cascade_pt_contraction_n1_normalises_to_unit():
    """A single primitive with c=1 (raw) must rescale to a unit STO."""
    prims = cascade_pt_contraction(2.0, n=2, n_prim=1, option="B")
    assert len(prims) == 1
    zeta_0, c_0 = prims[0]
    assert zeta_0 == pytest.approx(2.0)
    # Normalised STO has overlap 1 with itself, so c must be 1.
    assert c_0 == pytest.approx(1.0)


def test_cascade_pt_contraction_invalid_option():
    """Option C (Slater fit) is still deferred — must raise."""
    with pytest.raises(NotImplementedError):
        cascade_pt_contraction(1.0, n=2, n_prim=4, option="C")


# ─────────────────────────────────────────────────────────────────────
# Option A — hydrogenic projection
# ─────────────────────────────────────────────────────────────────────


def test_cascade_pt_contraction_option_A_normalised():
    """Option A coefficients must yield <chi|chi> = 1 to machine eps."""
    prims = cascade_pt_contraction(1.95, n=2, n_prim=4, option="A", l=0)
    orb = PTContractedOrbital(
        Z=7, n=2, l=0, m=0, primitives=tuple(prims), occ=2.0,
    )
    assert overlap_atomic(orb, orb, 0.0) == pytest.approx(1.0, abs=1e-10)


def test_cascade_pt_contraction_option_A_2p_collapses_to_single_zeta():
    """For l=1 the hydrogenic R_21 is itself a single STO at zeta = Z/n,
    so the projection on a cascade with z_base = Z/n gives c_0 = 1
    and c_{k>=1} = 0 (single-zeta limit)."""
    z_base = 1.5
    prims = cascade_pt_contraction(z_base, n=2, n_prim=4, option="A", l=1)
    # First primitive's coefficient should be ≈ 1, rest ≈ 0
    assert prims[0][1] == pytest.approx(1.0, abs=1e-10)
    for zeta_k, c_k in prims[1:]:
        assert abs(c_k) < 1e-10


def test_cascade_pt_contraction_option_A_2s_has_alternating_signs():
    """The 2s wavefunction has one radial node, so the projection on a
    monotone-zeta cascade requires alternating-sign coefficients to
    reproduce the node (multi-zeta flexibility)."""
    prims = cascade_pt_contraction(1.95, n=2, n_prim=4, option="A", l=0)
    coeffs = [c for _z, c in prims]
    # At least one sign change in the coefficient sequence
    sign_changes = sum(
        1 for i in range(1, len(coeffs))
        if coeffs[i] * coeffs[i - 1] < 0
    )
    assert sign_changes >= 1, (
        f"Option A 2s should reproduce the radial node via at least one "
        f"sign change in coefficients, got {coeffs}"
    )


def test_cascade_pt_contraction_option_A_requires_l_in_range():
    """Only (n, l) with 0 <= l < n are valid hydrogenic states."""
    with pytest.raises(ValueError):
        cascade_pt_contraction(1.0, n=2, n_prim=4, option="A", l=2)


def test_cascade_pt_contraction_option_A_high_n_unsupported():
    """The hydrogenic table only goes up to n=4; n=5 must raise."""
    with pytest.raises(NotImplementedError):
        cascade_pt_contraction(1.0, n=5, n_prim=4, option="A", l=0)


# ─────────────────────────────────────────────────────────────────────
# Evaluator dispatch — value
# ─────────────────────────────────────────────────────────────────────


def test_evaluate_sto_n1_contracted_equals_single_zeta():
    """A 1-primitive contraction must coincide with the underlying STO."""
    zeta = 1.7
    orb_single = PTAtomicOrbital(Z=6, n=2, l=1, m=1, zeta=zeta, occ=2.0)
    orb_contr = PTContractedOrbital(
        Z=6, n=2, l=1, m=1,
        primitives=((zeta, 1.0),),
        occ=2.0,
    )
    pts = np.array([[0.6, -0.4, 0.3],
                    [1.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0]])
    centre = np.array([0.0, 0.0, 0.0])
    v_single = evaluate_sto(orb_single, pts, centre)
    v_contr = evaluate_sto(orb_contr, pts, centre)
    np.testing.assert_allclose(v_contr, v_single, rtol=0, atol=0)


def test_evaluate_sto_contracted_linearity():
    """chi(r) = Sum_i c_i g_i — verify against explicit primitive sum."""
    z_base = 1.8
    prims = cascade_pt_contraction(z_base, n=2, n_prim=4, option="B")
    orb_contr = PTContractedOrbital(
        Z=7, n=2, l=0, m=0, primitives=tuple(prims), occ=2.0,
    )
    pts = np.array([[0.5, 0.3, -0.1], [1.2, 0.0, 0.0]])
    centre = np.zeros(3)
    v_contr = evaluate_sto(orb_contr, pts, centre)

    expected = np.zeros(pts.shape[0])
    for c_i, prim in iter_primitives(orb_contr):
        expected += c_i * evaluate_sto(prim, pts, centre)
    np.testing.assert_allclose(v_contr, expected, rtol=1e-15, atol=1e-15)


# ─────────────────────────────────────────────────────────────────────
# Evaluator dispatch — gradient
# ─────────────────────────────────────────────────────────────────────


def test_evaluate_sto_gradient_contracted_n1_equals_single():
    zeta = 1.4
    orb_single = PTAtomicOrbital(Z=6, n=2, l=1, m=0, zeta=zeta, occ=2.0)
    orb_contr = PTContractedOrbital(
        Z=6, n=2, l=1, m=0,
        primitives=((zeta, 1.0),),
        occ=2.0,
    )
    pts = np.array([[0.4, 0.2, 0.6], [0.9, -0.3, 0.1]])
    centre = np.zeros(3)
    g_single = evaluate_sto_gradient_analytic(orb_single, pts, centre)
    g_contr = evaluate_sto_gradient_analytic(orb_contr, pts, centre)
    np.testing.assert_allclose(g_contr, g_single, rtol=0, atol=0)


def test_evaluate_sto_gradient_contracted_matches_FD():
    """Analytic contracted gradient agrees with central-difference value."""
    z_base = 1.6
    prims = cascade_pt_contraction(z_base, n=2, n_prim=4, option="B")
    orb = PTContractedOrbital(
        Z=6, n=2, l=1, m=0, primitives=tuple(prims), occ=2.0,
    )
    pts = np.array([[0.3, 0.4, 0.5], [0.8, -0.2, 0.1], [1.2, 0.0, -0.3]])
    centre = np.zeros(3)
    g_an = evaluate_sto_gradient(orb, pts, centre)
    eps = 1.0e-5
    g_fd = np.zeros_like(g_an)
    for d in range(3):
        shift = np.zeros(3); shift[d] = eps
        v_p = evaluate_sto(orb, pts + shift, centre)
        v_m = evaluate_sto(orb, pts - shift, centre)
        g_fd[..., d] = (v_p - v_m) / (2.0 * eps)
    # Each component should agree to better than 1e-7 (the analytic
    # form is exact and FD has O(eps^2) truncation error).
    np.testing.assert_allclose(g_an, g_fd, atol=1e-7)


# ─────────────────────────────────────────────────────────────────────
# Overlap of contracted orbitals
# ─────────────────────────────────────────────────────────────────────


def test_overlap_contracted_self_at_R0_is_one():
    z_base = 2.0
    prims = cascade_pt_contraction(z_base, n=2, n_prim=4, option="B")
    orb = PTContractedOrbital(
        Z=6, n=2, l=0, m=0, primitives=tuple(prims), occ=0.0,
    )
    s = overlap_atomic(orb, orb, 0.0)
    assert s == pytest.approx(1.0, abs=1e-12)


def test_overlap_contracted_orthogonal_lm_at_R0():
    """Different (l, m) at the same centre must have zero overlap, even
    when both are contracted (angular part is orthogonal radial-by-radial).
    """
    z_base = 2.0
    prims_s = cascade_pt_contraction(z_base, n=2, n_prim=4, option="B")
    prims_p = cascade_pt_contraction(z_base, n=2, n_prim=4, option="B")
    orb_s = PTContractedOrbital(
        Z=6, n=2, l=0, m=0, primitives=tuple(prims_s),
    )
    orb_px = PTContractedOrbital(
        Z=6, n=2, l=1, m=1, primitives=tuple(prims_p),
    )
    assert overlap_atomic(orb_s, orb_px, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_overlap_contracted_atomic_mixed():
    """Overlap between a contraction and a single-zeta STO equals the
    contraction-coefficient-weighted sum of primitive overlaps."""
    zeta = 1.5
    single = PTAtomicOrbital(Z=6, n=2, l=0, m=0, zeta=zeta, occ=0.0)
    prims = cascade_pt_contraction(2.0, n=2, n_prim=3, option="B")
    contr = PTContractedOrbital(
        Z=6, n=2, l=0, m=0, primitives=tuple(prims),
    )
    R = np.array([1.2, 0.0, 0.0])
    s = overlap_atomic(contr, single, R)
    expected = sum(
        c_i * overlap_atomic(prim, single, R)
        for c_i, prim in iter_primitives(contr)
    )
    assert s == pytest.approx(expected, abs=1e-12)


# ─────────────────────────────────────────────────────────────────────
# Builder: pVDZ-PT
# ─────────────────────────────────────────────────────────────────────


def test_pVDZ_PT_basis_uses_contracted_orbitals():
    """Phase 6.B.8c — split-valence pVDZ-PT: each occupied valence
    shell becomes 2 contracted BFs per m (inner + outer)."""
    basis = build_atom_basis(7, basis_type="pVDZ-PT")
    # All orbitals should be contracted
    assert all(isinstance(o, PTContractedOrbital) for o in basis.orbitals)
    # Valence: (2s × 2 inner+outer) + (2p × 3 m × 2 inner+outer) = 8 BFs
    # Polarisation: 3d × 5 m = 5 BFs (still 1 per m)
    # Total: 13 BFs (matches DZP count for N)
    assert basis.n_orbitals == 13
    n_prim_per_orb = [len(o.primitives) for o in basis.orbitals]
    # Valence: 8 BFs × 4 primitives
    assert n_prim_per_orb.count(4) == 8
    # Polarisation: 5 BFs × 2 primitives
    assert n_prim_per_orb.count(2) == 5


def test_pVDZ_PT_split_valence_inner_outer_independent():
    """Phase 6.B.8c — the two BFs per (n,l,m) must be linearly independent
    (different coefficient profiles forward vs reversed)."""
    basis = build_atom_basis(7, basis_type="pVDZ-PT")
    # Find the two 2s BFs
    s_orbs = [o for o in basis.orbitals if o.l == 0 and o.n == 2]
    assert len(s_orbs) == 2
    inner, outer = s_orbs
    coefs_inner = [c for _z, c in inner.primitives]
    coefs_outer = [c for _z, c in outer.primitives]
    # Inner concentrated on first primitive, outer on last
    assert coefs_inner[0] > coefs_inner[-1]
    assert coefs_outer[0] < coefs_outer[-1]
    # And they are NOT proportional (lin indep): cross overlap < 0.99
    s_io = overlap_atomic(inner, outer, 0.0)
    assert abs(s_io) < 0.99, (
        f"inner and outer 2s should be lin indep, overlap={s_io}"
    )


def test_pVDZ_PT_self_overlap_unity():
    """Every orbital in a pVDZ-PT basis is normalised to <chi|chi> = 1."""
    basis = build_atom_basis(6, basis_type="pVDZ-PT")
    for orb in basis.orbitals:
        s = overlap_atomic(orb, orb, 0.0)
        assert s == pytest.approx(1.0, abs=1e-10), (
            f"orbital {orb.label} has self-overlap {s}, expected 1"
        )


def test_pVDZ_PT_unknown_atom_falls_through():
    """Hydrogen has no polarisation if l_polar > 4? — H has 1s only, so
    l_polar = 1 (still <= 4); we just check builder doesn't crash."""
    basis = build_atom_basis(1, basis_type="pVDZ-PT")
    assert basis.n_orbitals >= 1


# ─────────────────────────────────────────────────────────────────────
# Matrix-element coherence
# ─────────────────────────────────────────────────────────────────────


def test_pVTZ_PT_basis_uses_three_BFs_per_shell():
    """Phase 6.B.8e — split-valence pVTZ-PT: 3 contracted BFs per
    occupied valence shell per m (inner + middle + outer)."""
    basis = build_atom_basis(7, basis_type="pVTZ-PT")
    # All orbitals contracted
    assert all(isinstance(o, PTContractedOrbital) for o in basis.orbitals)
    # N: (2s × 3 BFs) + (2p × 3m × 3 BFs) + (3d × 5m × 1 BF) = 3+9+5 = 17
    assert basis.n_orbitals == 17
    n_prim_per_orb = [len(o.primitives) for o in basis.orbitals]
    # Valence: 12 BFs × 6 primitives
    assert n_prim_per_orb.count(6) == 12
    # Polarisation: 5 BFs × 3 primitives
    assert n_prim_per_orb.count(3) == 5


def test_pVTZ_PT_three_profiles_linearly_independent():
    """Inner / middle / outer must be lin indep (different cumulative
    cascade weight profiles forward / centered / reversed)."""
    basis = build_atom_basis(7, basis_type="pVTZ-PT")
    s_orbs = [o for o in basis.orbitals if o.l == 0 and o.n == 2]
    assert len(s_orbs) == 3
    inner, middle, outer = s_orbs
    # Inner concentrated on tightest primitive, outer on most diffuse,
    # middle peaks in between.
    inner_coefs = [c for _z, c in inner.primitives]
    middle_coefs = [c for _z, c in middle.primitives]
    outer_coefs = [c for _z, c in outer.primitives]
    assert inner_coefs[0] > inner_coefs[-1]
    assert outer_coefs[0] < outer_coefs[-1]
    # Middle has its peak in the middle two primitives
    assert max(middle_coefs) in (middle_coefs[2], middle_coefs[3])
    # All three pairwise overlaps strictly less than 1
    s_im = overlap_atomic(inner, middle, 0.0)
    s_mo = overlap_atomic(middle, outer, 0.0)
    s_io = overlap_atomic(inner, outer, 0.0)
    assert abs(s_im) < 0.99
    assert abs(s_mo) < 0.99
    assert abs(s_io) < 0.99


def test_pVTZ_PT_self_overlap_unity():
    """Every contracted BF in pVTZ-PT is normalised to 1."""
    basis = build_atom_basis(6, basis_type="pVTZ-PT")
    for orb in basis.orbitals:
        s = overlap_atomic(orb, orb, 0.0)
        assert s == pytest.approx(1.0, abs=1e-10)


def test_option_B_weight_profile_centered_symmetric():
    """The 'centered' profile must be symmetric about the middle index."""
    from ptc.lcao.atomic_basis import _option_B_weight_profile
    coefs = _option_B_weight_profile(6, profile="centered")
    # Symmetry: c[k] == c[N-1-k]
    for k in range(3):
        assert coefs[k] == pytest.approx(coefs[5 - k], rel=1e-12)


def test_overlap_matrix_pVDZ_PT_diagonal_is_unit():
    """Building a pVDZ-PT basis and computing the overlap matrix at the
    same centre yields identity on the diagonal."""
    basis = build_atom_basis(6, basis_type="pVDZ-PT")
    n = len(basis.orbitals)
    S = np.zeros((n, n))
    for i, oi in enumerate(basis.orbitals):
        for j, oj in enumerate(basis.orbitals):
            S[i, j] = overlap_atomic(oi, oj, 0.0)
    # Diagonal should be 1
    np.testing.assert_allclose(np.diag(S), np.ones(n), atol=1e-10)
    # Off-diagonal between different (n,l,m): zero
    for i, oi in enumerate(basis.orbitals):
        for j, oj in enumerate(basis.orbitals):
            if i == j:
                continue
            same_nlm = (oi.n == oj.n and oi.l == oj.l and oi.m == oj.m)
            if not same_nlm:
                assert abs(S[i, j]) < 1e-10
