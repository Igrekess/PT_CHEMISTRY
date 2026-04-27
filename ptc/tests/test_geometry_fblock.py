"""f-block geometry tests — PT-pure relativistic and lanthanide/actinide factors.

Tests for ``ptc/lcao/relativistic.py`` + ``ptc/bond.py::r_equilibrium``.
"""
import math
import pytest

from ptc.bond import r_equilibrium
from ptc.constants import GAMMA_5, GAMMA_7, ALPHA_PHYS
from ptc.lcao.relativistic import (
    lanthanide_factor,
    relativistic_factor,
    actinide_factor,
    valence_contraction,
)


# lanthanide_factor

def test_lanthanide_below_57_is_one():
    assert lanthanide_factor(1) == 1.0
    assert lanthanide_factor(56) == 1.0


def test_lanthanide_above_71_is_one():
    assert lanthanide_factor(72) == 1.0


def test_lanthanide_la_value():
    expected = math.sqrt(GAMMA_5) ** (1.0 / 15.0)
    assert abs(lanthanide_factor(57) - expected) < 1e-12


def test_lanthanide_lu_saturation():
    expected = math.sqrt(GAMMA_5)
    assert abs(lanthanide_factor(71) - expected) < 1e-12


def test_lanthanide_monotone():
    last = 1.0
    for Z in range(57, 72):
        v = lanthanide_factor(Z)
        assert v < last
        last = v


# relativistic_factor

def test_relativistic_below_72_is_one():
    assert relativistic_factor(50) == 1.0
    assert relativistic_factor(71) == 1.0


def test_relativistic_dirac_form():
    expected = math.sqrt(1.0 - (92 * ALPHA_PHYS) ** 2)
    assert abs(relativistic_factor(92) - expected) < 1e-12


# actinide_factor

def test_actinide_below_90_is_one():
    assert actinide_factor(83) == 1.0


def test_actinide_lr_saturation():
    expected = math.sqrt(GAMMA_7)
    assert abs(actinide_factor(103) - expected) < 1e-12


# valence_contraction

def test_valence_contraction_layer_separation():
    assert valence_contraction(56) == 1.0
    assert abs(valence_contraction(63) - lanthanide_factor(63)) < 1e-12
    assert abs(valence_contraction(83) - relativistic_factor(83)) < 1e-12


# r_equilibrium bench

FBLOCK_BENCH = [
    (57, 57, 1, 1, 1, 0, 0, 6, 6, 3.50, 12.0, "La-La"),
    (71, 71, 1, 1, 1, 0, 0, 6, 6, 2.92, 10.0, "Lu-Lu"),
    (72, 72, 1, 1, 1, 0, 0, 6, 6, 2.86, 5.0, "Hf-Hf"),
    (82, 82, 1, 2, 2, 1, 1, 6, 6, 2.93, 8.0, "Pb-Pb"),
    (83, 83, 1, 2, 2, 1, 1, 6, 6, 3.05, 8.0, "Bi-Bi"),
    (92, 83, 1, 4, 2, 0, 1, 7, 6, 3.05, 8.0, "U-Bi"),
    (92, 8, 3, 1, 1, 0, 0, 7, 2, 1.78, 8.0, "U=O"),
]


@pytest.mark.parametrize(
    "Z_A,Z_B,bo,z_A,z_B,lp_A,lp_B,per_A,per_B,r_exp,tol,label",
    FBLOCK_BENCH,
    ids=[c[11] for c in FBLOCK_BENCH],
)
def test_r_equilibrium_fblock_bench(
    Z_A, Z_B, bo, z_A, z_B, lp_A, lp_B, per_A, per_B, r_exp, tol, label
):
    r = r_equilibrium(
        per_A=per_A, per_B=per_B, bo=bo,
        z_A=z_A, z_B=z_B, lp_A=lp_A, lp_B=lp_B,
        Z_A=Z_A, Z_B=Z_B,
    )
    err_pct = abs(r - r_exp) / r_exp * 100.0
    assert err_pct < tol, (
        f"{label}: r_PT={r:.3f} vs exp={r_exp:.2f} ({err_pct:+.2f}% > {tol}%)"
    )


def test_fblock_bench_global_mae():
    abs_errs = []
    for row in FBLOCK_BENCH:
        Z_A, Z_B, bo, z_A, z_B, lp_A, lp_B, per_A, per_B, r_exp = row[:10]
        r = r_equilibrium(
            per_A=per_A, per_B=per_B, bo=bo,
            z_A=z_A, z_B=z_B, lp_A=lp_A, lp_B=lp_B,
            Z_A=Z_A, Z_B=Z_B,
        )
        abs_errs.append(abs(r - r_exp) / r_exp * 100.0)
    mae = sum(abs_errs) / len(abs_errs)
    assert mae < 15.0, f"f-block MAE = {mae:.2f}% (target < 15%)"


def test_bi_bi_in_range():
    r = r_equilibrium(
        per_A=6, per_B=6, bo=1, z_A=2, z_B=2, lp_A=1, lp_B=1, Z_A=83, Z_B=83
    )
    assert 2.5 < r < 3.5
