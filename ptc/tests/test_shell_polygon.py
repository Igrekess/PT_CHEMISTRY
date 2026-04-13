"""Tests for the ShellPolygon geometric shell operator."""
from __future__ import annotations

import math

import pytest

from ptc.shell_polygon import ShellPolygon, AtomicShell, build_atomic_shell
from ptc.constants import S3, S5, S7, S_HALF


# ── helpers ──────────────────────────────────────────────────────────────────

def sp(prime: int, n: int) -> ShellPolygon:
    return ShellPolygon(prime=prime, n_occupied=n)


# ── 1. capacity is 2×prime for all blocks ────────────────────────────────────

def test_capacity_is_twice_prime():
    assert sp(1, 0).capacity == 2
    assert sp(3, 0).capacity == 6
    assert sp(5, 0).capacity == 10
    assert sp(7, 0).capacity == 14


# ── 2. vacancies = capacity - n_occupied ─────────────────────────────────────

def test_vacancies_equals_capacity_minus_occupied():
    assert sp(3, 2).vacancies == 4
    assert sp(5, 7).vacancies == 3
    assert sp(7, 14).vacancies == 0
    assert sp(1, 0).vacancies == 2


# ── 3. fill_fraction correct ─────────────────────────────────────────────────

def test_fill_fraction_correct():
    assert sp(3, 3).fill_fraction == pytest.approx(0.5)
    assert sp(5, 4).fill_fraction == pytest.approx(0.4)
    assert sp(1, 1).fill_fraction == pytest.approx(0.5)
    assert sp(7, 7).fill_fraction == pytest.approx(0.5)


# ── 4. half_filled True only when n == prime ─────────────────────────────────

def test_half_filled_only_when_n_equals_prime():
    assert sp(3, 3).half_filled is True
    assert sp(5, 5).half_filled is True
    assert sp(7, 7).half_filled is True
    assert sp(1, 1).half_filled is True
    # False cases
    assert sp(3, 2).half_filled is False
    assert sp(3, 4).half_filled is False
    assert sp(5, 4).half_filled is False
    assert sp(5, 6).half_filled is False


# ── 5. hund_config: n≤p → (n,0), n>p → (p, n-p) ────────────────────────────

def test_hund_config():
    # p-block (prime=3)
    assert sp(3, 0).hund_config() == (0, 0)
    assert sp(3, 1).hund_config() == (1, 0)
    assert sp(3, 2).hund_config() == (2, 0)
    assert sp(3, 3).hund_config() == (3, 0)
    assert sp(3, 4).hund_config() == (3, 1)
    assert sp(3, 5).hund_config() == (3, 2)
    assert sp(3, 6).hund_config() == (3, 3)
    # d-block (prime=5)
    assert sp(5, 5).hund_config() == (5, 0)
    assert sp(5, 7).hund_config() == (5, 2)


# ── 6. exchange_pairs: C(n_up,2) + C(n_down,2) ───────────────────────────────

def test_exchange_pairs_combinatorics():
    # p3: n_up=3, n_down=0 → C(3,2)+C(0,2) = 3
    assert sp(3, 3).exchange_pairs() == 3
    # p2: n_up=2, n_down=0 → C(2,2) = 1
    assert sp(3, 2).exchange_pairs() == 1
    # p4: n_up=3, n_down=1 → 3+0 = 3
    assert sp(3, 4).exchange_pairs() == 3
    # p5: n_up=3, n_down=2 → 3+1 = 4
    assert sp(3, 5).exchange_pairs() == 4
    # d5 (half): n_up=5, n_down=0 → C(5,2) = 10
    assert sp(5, 5).exchange_pairs() == 10
    # d10 (full): n_up=5, n_down=5 → 10+10 = 20
    assert sp(5, 10).exchange_pairs() == 20
    # empty: 0
    assert sp(3, 0).exchange_pairs() == 0


# ── 7. closure_boost values ───────────────────────────────────────────────────

def test_closure_boost_values():
    # 0 vacancies → 0.0
    assert sp(3, 6).closure_boost() == 0.0
    # p-block (prime=3, scale=1.0): 1 vacancy → 1/s²×(1+sin²₅), 2 vac → 2.0
    from ptc.shell_polygon import _EXCHANGE_WEIGHT
    assert sp(3, 5).closure_boost() == pytest.approx(
        1.0 / S_HALF ** 2 * (1.0 + _EXCHANGE_WEIGHT[3]) * 1.0)
    assert sp(3, 4).closure_boost() == pytest.approx(1.0 / S_HALF * 1.0)
    # 3+ vacancies → 1.0
    assert sp(3, 3).closure_boost() == pytest.approx(1.0)
    assert sp(3, 0).closure_boost() == pytest.approx(1.0)
    # s-block (prime=1): pair completion via γ₇
    from ptc.constants import GAMMA_7
    assert sp(1, 1).closure_boost() == pytest.approx(GAMMA_7)
    # d-block (prime=5, scale=5/3): stronger closure
    assert sp(5, 10).closure_boost() == 0.0
    assert sp(5, 9).closure_boost() == pytest.approx(
        1.0 / S_HALF ** 2 * (1.0 + _EXCHANGE_WEIGHT[5]) * 5.0 / 3.0)
    assert sp(5, 8).closure_boost() == pytest.approx(1.0 / S_HALF * 5.0 / 3.0)


# ── 8. geometric_profile = 0 at empty and full, positive in between ──────────

def test_geometric_profile_zero_at_extremes_positive_between():
    for prime in (3, 5, 7):
        cap = 2 * prime
        assert sp(prime, 0).geometric_profile() == pytest.approx(0.0), \
            f"prime={prime} empty should be 0"
        assert sp(prime, cap).geometric_profile() == pytest.approx(0.0), \
            f"prime={prime} full should be 0"
        # At half-filling: envelope > 0, closure_boost=1.0
        assert sp(prime, prime).geometric_profile() > 0.0, \
            f"prime={prime} half-filled should be positive"


# ── 9. geometric_profile for s-block: positive only at n=1 ───────────────────

def test_geometric_profile_s_block_positive_only_at_n1():
    assert sp(1, 0).geometric_profile() == pytest.approx(0.0)
    assert sp(1, 1).geometric_profile() > 0.0
    assert sp(1, 2).geometric_profile() == pytest.approx(0.0)


# ── 10. capture_amplitude = 0 when full ──────────────────────────────────────

def test_capture_amplitude_zero_when_full():
    assert sp(3, 6).capture_amplitude() == pytest.approx(0.0)
    assert sp(5, 10).capture_amplitude() == pytest.approx(0.0)
    assert sp(7, 14).capture_amplitude() == pytest.approx(0.0)
    assert sp(1, 2).capture_amplitude() == pytest.approx(0.0)


# ── 11. ejection_amplitude = 0 when empty ────────────────────────────────────

def test_ejection_amplitude_zero_when_empty():
    assert sp(3, 0).ejection_amplitude() == pytest.approx(0.0)
    assert sp(5, 0).ejection_amplitude() == pytest.approx(0.0)
    assert sp(1, 0).ejection_amplitude() == pytest.approx(0.0)


# ── 12. capture positive for halogens (p5, 1 vacancy) ────────────────────────

def test_capture_positive_for_halogen_p5():
    halogen = sp(3, 5)  # p5: 1 vacancy
    assert halogen.capture_amplitude() > 0.0


# ── 13. ejection positive for alkali (s1) ────────────────────────────────────

def test_ejection_positive_for_alkali_s1():
    alkali = sp(1, 1)  # s1: 1 electron
    assert alkali.ejection_amplitude() > 0.0


# ── 14. capture weaker at half-filling than at p4 ────────────────────────────

def test_capture_weaker_at_half_filling_than_p4():
    # p3 (half-filled): closure_boost=1.0
    # p4: closure_boost=1/S_HALF (strong near-closure pull) → higher capture
    p3_cap = sp(3, 3).capture_amplitude()
    p4_cap = sp(3, 4).capture_amplitude()
    assert p3_cap < p4_cap


# ── 15. ejection stronger at half-filling (Hund) ─────────────────────────────

def test_ejection_stronger_at_half_filling_than_past_half():
    # N-like (p3, half-filled) should have HIGHER ejection than O-like (p4)
    # cos²(π×3/3) = cos²(π) = 1 → ej = S3 × 0.5 × 2.0 = S3
    # cos²(π×4/3) = 0.25 → ej = S3 × (4/6) × 1.25
    p3_ej = sp(3, 3).ejection_amplitude()
    p4_ej = sp(3, 4).ejection_amplitude()
    assert p3_ej > p4_ej, f"Hund: ej(p3)={p3_ej:.4f} should > ej(p4)={p4_ej:.4f}"


# ── 16. polygon_symmetry: Hund peak only (sin²(π×fill)) ──────────────────────

def test_polygon_symmetry_asymmetric():
    # GFT harmonic series: log₂(1+1/n) for Hund, -log₂(1+1/n_down) for pairing
    sym = {n: sp(3, n).polygon_symmetry() for n in range(7)}
    assert sym[0] == pytest.approx(0.0)   # empty
    assert sym[6] == pytest.approx(0.0)   # full: zero (closure)
    # Phase 1 (filling ↑): DECREASES harmonically (n=1 is strongest)
    assert sym[1] > sym[2] > sym[3] > 0
    assert sym[1] == pytest.approx(1.0, abs=0.01)  # first electron = 1 bit
    # Phase 2 (pairing ↓): NEGATIVE, peaks at first pairing
    assert sym[4] < sym[5] < 0   # O-like strongest, F-like less
    assert sym[4] == pytest.approx(-1.0, abs=0.01)  # first pairing = -1 bit


def test_polygon_symmetry_s_block():
    # s-block (prime=1): no exchange → symmetry = 0 always
    assert sp(1, 0).polygon_symmetry() == pytest.approx(0.0)
    assert sp(1, 1).polygon_symmetry() == pytest.approx(0.0)  # no Hund in pair
    assert sp(1, 2).polygon_symmetry() == pytest.approx(0.0)


# ── additional: layer_name correct ───────────────────────────────────────────

def test_layer_name():
    assert sp(1, 0).layer_name == "s"
    assert sp(3, 0).layer_name == "p"
    assert sp(5, 0).layer_name == "d"
    assert sp(7, 0).layer_name == "f"


# ── additional: geometric_profile manual check for p2 ────────────────────────

def test_geometric_profile_p2_manual():
    # p2: x=2/6=1/3, koide=0.5
    # Phase B: exchange weight = sin²₅ (next-prime), not sin²₃
    # exchange: count=2, w=sin²₅ → (1+2*sin²₅)²
    from ptc.shell_polygon import _EXCHANGE_WEIGHT
    s = sp(3, 2)
    x = 2 / 6.0
    koide = (x * x * (1.0 - x)) / (4.0 / 27.0)
    w = _EXCHANGE_WEIGHT[3]  # sin²₅ for p-block
    exchange = (1.0 + w * 2) ** 2
    closure = 1.0
    expected = koide * exchange * closure
    assert s.geometric_profile() == pytest.approx(expected, rel=1e-9)


# ── additional: coupling constants correct ────────────────────────────────────

def test_s_ejection_amplitude_manual():
    # s1 (Koide bifurcation): prime=1 → dip=0 (trivial polygon)
    # ej = 1 + sym*hund_w - koide*dip = 1 + 1.0*0.0 - 0.844*0.0 = 1.0
    s = sp(1, 1)
    assert s.ejection_amplitude() == pytest.approx(1.0)


def test_capture_d_block_positive_and_density_scaled():
    # d-block capture should be positive (pentagon birth amplitude)
    d1 = sp(5, 1)  # d1
    assert d1.capture_amplitude() > 0.0
    # d-block late-fill (d9) should capture more than early (d1)
    # because Koide envelope + closure boost peak near closure
    d9 = sp(5, 9)  # d9 (near-closure)
    assert d9.capture_amplitude() > d1.capture_amplitude()


# ── Task 4: AtomicShell tests ─────────────────────────────────────────────────

def test_build_atomic_shell_H():
    """H (Z=1): s(1), p(0), d(0), f(0); active = s polygon."""
    shell = build_atomic_shell(1)
    s, p, d, f = shell.polygons
    assert s.prime == 1 and s.n_occupied == 1
    assert p.prime == 3 and p.n_occupied == 0
    assert d.prime == 5 and d.n_occupied == 0
    assert f.prime == 7 and f.n_occupied == 0
    assert shell.active_polygon.prime == 1


def test_build_atomic_shell_C():
    """C (Z=6): s(2), p(2); active = p polygon."""
    shell = build_atomic_shell(6)
    s, p, d, f = shell.polygons
    assert s.prime == 1 and s.n_occupied == 2
    assert p.prime == 3 and p.n_occupied == 2
    assert d.n_occupied == 0
    assert f.n_occupied == 0
    assert shell.active_polygon.prime == 3


def test_build_atomic_shell_Fe():
    """Fe (Z=26): s(2), p(6), d(6); active = d polygon."""
    shell = build_atomic_shell(26)
    s, p, d, f = shell.polygons
    assert s.prime == 1 and s.n_occupied == 2
    assert p.prime == 3 and p.n_occupied == 6
    assert d.prime == 5 and d.n_occupied == 6
    assert f.n_occupied == 0
    assert shell.active_polygon.prime == 5


def test_saturated_polygons_Fe():
    """Fe: s and p are saturated below the active d shell."""
    shell = build_atomic_shell(26)
    sat = shell.saturated_polygons
    assert len(sat) == 2
    assert sat[0].prime == 1 and sat[0].n_occupied == 2   # s full
    assert sat[1].prime == 3 and sat[1].n_occupied == 6   # p full


def test_n_total():
    """n_total sums all occupied electrons."""
    assert build_atomic_shell(1).n_total == 1    # H
    assert build_atomic_shell(6).n_total == 4    # C: 2+2
    assert build_atomic_shell(26).n_total == 14  # Fe: 2+6+6


def test_stack_memory_H_is_one():
    """H has no saturated layers → stack_memory = 1.0."""
    shell = build_atomic_shell(1)
    assert shell.stack_memory() == pytest.approx(1.0)


def test_stack_memory_C_gt_one():
    """C has saturated s layer → stack_memory > 1.0."""
    shell = build_atomic_shell(6)
    assert shell.stack_memory() > 1.0


def test_stack_memory_Fe_gt_C():
    """Fe has two saturated layers (s, p) → stack_memory > C's."""
    fe = build_atomic_shell(26)
    c = build_atomic_shell(6)
    assert fe.stack_memory() > c.stack_memory()


def test_composite_figure_H():
    assert build_atomic_shell(1).composite_figure() == "point"


def test_composite_figure_C():
    assert build_atomic_shell(6).composite_figure() == "tetrahedron"


def test_composite_figure_Fe():
    assert build_atomic_shell(26).composite_figure() == "9-cell"
