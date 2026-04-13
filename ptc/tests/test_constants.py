"""Tests for PT constants — all derived from s=1/2 and q=13/15."""

from ptc.constants import (
    AEM,
    A_BOHR,
    BETA_GHOST,
    C3,
    C5,
    C7,
    COULOMB_EV_A,
    D3,
    D5,
    D7,
    D_DARK,
    D_FULL,
    D_NNLO,
    G_FISHER,
    GAMMA_11,
    GAMMA_13,
    GAMMA_3,
    GAMMA_5,
    GAMMA_7,
    LAM,
    MU_STAR,
    P1,
    P2,
    P3,
    Q_KOIDE,
    RY,
    S11,
    S13,
    S3,
    S5,
    S7,
    S_HALF,
)


def test_primes():
    assert P1 == 3
    assert P2 == 5
    assert P3 == 7
    assert MU_STAR == P1 + P2 + P3 == 15


def test_spin():
    assert S_HALF == 0.5


def test_sin2_values():
    assert abs(S3 - 0.219155040999848) < 1e-12
    assert abs(S5 - 0.193974726748742) < 1e-12
    assert abs(S7 - 0.172614219698474) < 1e-12


def test_cos2_complement():
    assert abs(C3 - (1.0 - S3)) < 1e-12
    assert abs(C5 - (1.0 - S5)) < 1e-12
    assert abs(C7 - (1.0 - S7)) < 1e-12


def test_alpha_em_bare():
    assert abs(AEM - S3 * S5 * S7) < 1e-15
    assert abs(1.0 / AEM - 136.27833445429476) < 1e-12


def test_deltas_are_ordered():
    assert D3 > D5 > D7 > 0.0


def test_d_full_construction_is_positive():
    assert D_FULL > 1.0
    assert D_NNLO > D_FULL
    assert D_DARK > 0.0


def test_koide():
    assert abs(float(Q_KOIDE) - 2.0 / 3.0) < 1e-12


def test_fisher():
    assert G_FISHER == 4


def test_rydberg_and_units():
    # PT-derived constants use alpha_phys (~19 ppm vs CODATA),
    # so Ry_PT differs from Ry_CODATA by ~38 ppm.  Tolerance 1e-3
    # accommodates this without masking real regressions.
    assert abs(RY - 13.6057) < 1e-3
    assert abs(COULOMB_EV_A - 14.3996) < 1e-3
    assert abs(A_BOHR - 0.5292) < 1e-3


def test_gammas_are_decreasing():
    assert GAMMA_3 > GAMMA_5 > GAMMA_7 > 0.0


def test_ghost_sector_is_inactive():
    assert abs(S11 - 0.13895232276847763) < 1e-12
    assert abs(S13 - 0.1256852047834586) < 1e-12
    assert abs(GAMMA_11 - 0.4257331357329715) < 1e-12
    assert abs(GAMMA_13 - 0.35624075947551403) < 1e-12
    assert GAMMA_11 < 0.5
    assert GAMMA_13 < 0.5
    assert abs(BETA_GHOST - 0.10393080089649875) < 1e-12


def test_stop_rate():
    assert abs(LAM - 2.0 / MU_STAR) < 1e-15
