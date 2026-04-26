"""Tests for ptc.lcao.atomic_basis (Phase A).

Validation milestones from PROMPT_PT_LCAO_GIAO.md §3 Phase A:
  * Self-overlap = 1.000 +/- 1e-9 for 50 representative atoms
  * H_2 overlap sigma at r_AB = 0.74 Angstrom approx 0.75 (Slater analytic)
  * No regression on D_at MAE (Phase A is read-only) -- enforced separately
    by the global benchmark, not in pytest.
"""

import math

import numpy as np
import pytest

from ptc.constants import A_BOHR
from ptc.lcao.atomic_basis import (
    PTAtomBasis,
    PTAtomicOrbital,
    Z_eff_shell,
    build_atom_basis,
    occupied_shells,
    overlap_atomic,
)
from ptc.lcao.atomic_basis import _A_n, _B_n, _overlap_1s_1s


# ─────────────────────────────────────────────────────────────────────
# 1. Auxiliary integrals
# ─────────────────────────────────────────────────────────────────────


def test_A_n_known_values():
    # A_0(1) = e^-1 / 1
    assert _A_n(0, 1.0) == pytest.approx(math.exp(-1.0))
    # A_2(1) = e^-1 (1 + 2 + 2) = 5/e
    assert _A_n(2, 1.0) == pytest.approx(5.0 * math.exp(-1.0))


def test_B_n_at_zero_via_taylor():
    # int_{-1}^1 dx = 2; int_{-1}^1 x dx = 0; int_{-1}^1 x^2 dx = 2/3
    assert _B_n(0, 0.0) == pytest.approx(2.0)
    assert _B_n(1, 0.0) == pytest.approx(0.0, abs=1e-12)
    assert _B_n(2, 0.0) == pytest.approx(2.0 / 3.0)


def test_B_n_continuity_across_zero():
    # numerical continuity between Taylor branch and sinh/cosh branch
    eps = 5.0e-2  # boundary of switch (chosen to avoid cancellation)
    for n in (0, 1, 2):
        below = _B_n(n, eps - 1e-9)
        above = _B_n(n, eps + 1e-9)
        assert abs(below - above) < 1e-9, f"discontinuity at n={n}"


# ─────────────────────────────────────────────────────────────────────
# 2. Closed-form 1s-1s overlap
# ─────────────────────────────────────────────────────────────────────


def test_1s1s_same_zeta_known_formula():
    # Standard H_2 STO with zeta = 1.0 (a0^-1) at R = 1.4 a0
    # S = (1 + rho + rho^2/3) e^-rho, rho = 1.4
    # zeta_invA = 1/a0, R_A = 1.4 * a0  -> rho = zeta * R = 1.4 * 1 = 1.4
    zeta = 1.0 / A_BOHR
    R = 1.4 * A_BOHR
    rho = 1.4
    expected = (1.0 + rho + rho * rho / 3.0) * math.exp(-rho)
    got = _overlap_1s_1s(zeta, zeta, R)
    assert got == pytest.approx(expected, rel=1e-10)


def test_1s1s_self_at_R_zero():
    zeta = 1.5 / A_BOHR
    assert _overlap_1s_1s(zeta, zeta, 0.0) == pytest.approx(1.0)


def test_1s1s_different_zeta_at_R_zero():
    # Closed-form: 8 (zA zB)^(3/2) / (zA + zB)^3
    zA = 1.0 / A_BOHR
    zB = 1.5 / A_BOHR
    expected = 8.0 * (zA * zB) ** 1.5 / (zA + zB) ** 3
    got = _overlap_1s_1s(zA, zB, 0.0)
    assert got == pytest.approx(expected, rel=1e-10)


def test_1s1s_decays_to_zero_at_infinity():
    zeta = 1.0 / A_BOHR
    R_huge = 30.0  # 30 A: e^(-zeta R) ~ e^(-56) ~ 1e-25
    got = _overlap_1s_1s(zeta, zeta, R_huge)
    assert abs(got) < 1e-15


def test_1s1s_smooth_in_R():
    # Monotone decreasing for same-zeta on the R > 0 ray
    zeta = 1.0 / A_BOHR
    Rs = np.linspace(0.1, 5.0, 30)
    Ss = [_overlap_1s_1s(zeta, zeta, R) for R in Rs]
    for i in range(1, len(Ss)):
        assert Ss[i] < Ss[i - 1]


def test_1s1s_symmetric_in_zetas():
    zA, zB = 0.7 / A_BOHR, 1.3 / A_BOHR
    R = 0.7
    s_AB = _overlap_1s_1s(zA, zB, R)
    s_BA = _overlap_1s_1s(zB, zA, R)
    assert s_AB == pytest.approx(s_BA, rel=1e-12)


# ─────────────────────────────────────────────────────────────────────
# 3. Z_eff and basis enumeration
# ─────────────────────────────────────────────────────────────────────


def test_Z_eff_returns_atom_effective_charge():
    from ptc.atom import effective_charge as ec
    for Z in (1, 6, 8, 14, 26, 47, 79):
        assert Z_eff_shell(Z, 1, 0) == pytest.approx(ec(Z))


def test_occupied_shells_hydrogen():
    sh = occupied_shells(1)
    assert sh == [(1, 0, 1)]


def test_occupied_shells_helium():
    sh = occupied_shells(2)
    assert sh == [(1, 0, 2)]


def test_occupied_shells_lithium():
    # Li: [He] 2s^1
    sh = occupied_shells(3)
    assert sh == [(2, 0, 1)]


def test_occupied_shells_carbon():
    # C: [He] 2s^2 2p^2
    sh = occupied_shells(6)
    assert (2, 0, 2) in sh
    assert (2, 1, 2) in sh


def test_occupied_shells_oxygen():
    # O: [He] 2s^2 2p^4
    sh = occupied_shells(8)
    assert (2, 0, 2) in sh
    assert (2, 1, 4) in sh


def test_occupied_shells_iron():
    # Fe: [Ar] 4s^2 3d^6
    sh = occupied_shells(26)
    assert (4, 0, 2) in sh
    assert (3, 2, 6) in sh


# ─────────────────────────────────────────────────────────────────────
# 4. PTAtomBasis assembly
# ─────────────────────────────────────────────────────────────────────


def test_basis_hydrogen_one_orbital():
    b = build_atom_basis(1)
    assert b.n_orbitals == 1
    o = b.orbitals[0]
    assert (o.n, o.l, o.m) == (1, 0, 0)
    assert o.occ == pytest.approx(1.0)
    # zeta in 1/Angstrom: Z_eff/n / a0 = 1/0.529 ~ 1.89
    assert o.zeta == pytest.approx(1.0 / A_BOHR, rel=2e-3)


def test_basis_carbon_orbitals_count():
    # C valence: 2s (1 orb) + 2p (3 orb) = 4 orbitals
    b = build_atom_basis(6)
    assert b.n_orbitals == 4
    ls = sorted(o.l for o in b.orbitals)
    assert ls == [0, 1, 1, 1]


def test_basis_total_occ_matches_valence_electron_count():
    cases = {
        1: 1,    # H 1s^1
        2: 2,    # He 1s^2
        3: 1,    # Li 2s^1
        6: 4,    # C 2s^2 2p^2
        8: 6,    # O 2s^2 2p^4
        10: 8,   # Ne 2s^2 2p^6
        26: 8,   # Fe 4s^2 3d^6
    }
    for Z, expected in cases.items():
        b = build_atom_basis(Z)
        assert b.total_occ == pytest.approx(expected), f"Z={Z}: {b.total_occ}"


def test_basis_charge_application_for_OH_anion_like():
    # O-: 2s^2 2p^5 (one extra electron on the highest-l shell)
    b = build_atom_basis(8, charge=-1)
    assert b.total_occ == pytest.approx(7.0)
    # extra electron should be on 2p
    p_total = sum(o.occ for o in b.orbitals if o.l == 1)
    assert p_total == pytest.approx(5.0)


# ─────────────────────────────────────────────────────────────────────
# 5. overlap_atomic public API
# ─────────────────────────────────────────────────────────────────────


def test_self_overlap_is_unity_50_atoms():
    # one self-overlap test per atomic basis, for Z = 1..50
    for Z in range(1, 51):
        b = build_atom_basis(Z)
        for o in b.orbitals:
            s = overlap_atomic(o, o, np.zeros(3))
            assert s == pytest.approx(1.0, abs=1e-12), (
                f"self-overlap != 1 for Z={Z}, orbital={o.label}"
            )


def test_orthogonal_orbitals_same_centre():
    # 1s and 2p_z on the same centre: orthogonal Y_lm => 0
    a = PTAtomicOrbital(Z=6, n=2, l=0, m=0, zeta=1.0, occ=2.0)
    b = PTAtomicOrbital(Z=6, n=2, l=1, m=0, zeta=1.0, occ=2.0)
    assert overlap_atomic(a, b, 0.0) == pytest.approx(0.0)


def test_H2_overlap_at_0_74_A_matches_slater_value():
    a = build_atom_basis(1).orbitals[0]
    b = build_atom_basis(1).orbitals[0]
    R = 0.74  # Angstrom
    rho = a.zeta * R  # zeta is in A^-1, R in A
    expected = (1.0 + rho + rho * rho / 3.0) * math.exp(-rho)
    got = overlap_atomic(a, b, np.array([R, 0.0, 0.0]))
    # H Z_eff is essentially 1 -> rho ~ 1.4 -> expected ~ 0.753
    assert got == pytest.approx(expected, rel=1e-10)
    assert 0.70 < got < 0.80


def test_H2_overlap_value_is_in_known_range():
    """Sanity-check vs published H_2 STO overlap at R = 1.4 a0."""
    a = build_atom_basis(1).orbitals[0]
    b = build_atom_basis(1).orbitals[0]
    R = 1.4 * A_BOHR  # 0.7407 A
    s = overlap_atomic(a, b, np.array([R, 0.0, 0.0]))
    assert 0.752 < s < 0.754


def test_overlap_decays_with_distance():
    a = build_atom_basis(1).orbitals[0]
    b = build_atom_basis(1).orbitals[0]
    s_close = overlap_atomic(a, b, np.array([0.5, 0.0, 0.0]))
    s_med = overlap_atomic(a, b, np.array([2.0, 0.0, 0.0]))
    s_far = overlap_atomic(a, b, np.array([10.0, 0.0, 0.0]))
    assert s_close > s_med > abs(s_far)
    assert abs(s_far) < 1e-3


def test_overlap_orientation_invariant():
    # Spherically symmetric 1s: overlap depends only on |r_AB|
    a = build_atom_basis(1).orbitals[0]
    b = build_atom_basis(1).orbitals[0]
    R = 1.0
    s_x = overlap_atomic(a, b, np.array([R, 0, 0]))
    s_y = overlap_atomic(a, b, np.array([0, R, 0]))
    s_diag = overlap_atomic(a, b, np.array([R / math.sqrt(3)] * 3))
    assert s_x == pytest.approx(s_y, rel=1e-12)
    assert s_x == pytest.approx(s_diag, rel=1e-12)


def test_unsupported_overlap_raises():
    # f-orbitals not yet wired (s/p analytic + d numerical 3D done)
    a = PTAtomicOrbital(Z=58, n=4, l=3, m=0, zeta=2.0, occ=1.0)
    b = PTAtomicOrbital(Z=58, n=4, l=3, m=0, zeta=2.0, occ=1.0)
    with pytest.raises(NotImplementedError):
        overlap_atomic(a, b, np.array([2.0, 0.0, 0.0]))
