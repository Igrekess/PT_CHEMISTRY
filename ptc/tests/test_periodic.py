"""Tests for PT periodic-structure helper functions."""

from ptc.periodic import (
    block_of, capacity, l_of, n_fill, _n_fill_aufbau,
    ns_config, period, period_start, _MADELUNG_PROMOTIONS,
)


def test_period_boundaries():
    assert period(1) == 1
    assert period(2) == 1
    assert period(3) == 2
    assert period(10) == 2
    assert period(11) == 3
    assert period(18) == 3
    assert period(19) == 4


def test_period_start():
    assert period_start(1) == 1
    assert period_start(2) == 3
    assert period_start(3) == 11
    assert period_start(4) == 19


def test_basic_shell_assignments():
    assert l_of(1) == 0
    assert l_of(6) == 1
    assert l_of(14) == 1
    assert l_of(24) == 2
    assert l_of(58) == 3


def test_aufbau_fill_values():
    """_n_fill_aufbau returns Aufbau filling (no promotions)."""
    assert _n_fill_aufbau(1) == 1
    assert _n_fill_aufbau(6) == 2
    assert _n_fill_aufbau(8) == 4
    assert _n_fill_aufbau(14) == 2
    assert _n_fill_aufbau(24) == 4   # Aufbau: d4
    assert _n_fill_aufbau(29) == 9   # Aufbau: d9


def test_madelung_fill_values():
    """n_fill returns Madelung filling (with d5/d10 promotions)."""
    # Non-promoted elements: same as Aufbau
    assert n_fill(1) == 1
    assert n_fill(6) == 2
    assert n_fill(8) == 4
    assert n_fill(14) == 2
    assert n_fill(26) == 6   # Fe: d6 (not promoted)
    # Promoted elements: Madelung
    assert n_fill(24) == 5   # Cr: d5 (half-fill)
    assert n_fill(29) == 10  # Cu: d10 (closure)
    assert n_fill(42) == 5   # Mo: d5 (half-fill)
    assert n_fill(46) == 10  # Pd: d10 (double closure)
    assert n_fill(47) == 10  # Ag: d10 (closure)
    assert n_fill(79) == 10  # Au: d10 (closure)


def test_ns_config_promotions():
    assert ns_config(6) == 2
    assert ns_config(24) == 1   # Cr: d5s1
    assert ns_config(29) == 1   # Cu: d10s1
    assert ns_config(46) == 0   # Pd: d10s0
    assert ns_config(79) == 1   # Au: d10s1


def test_non_promoted_unchanged():
    """Non-promoted d-block elements have n_fill == _n_fill_aufbau."""
    non_promoted_d = [Z for Z in range(21, 81) if l_of(Z) == 2 and Z not in _MADELUNG_PROMOTIONS]
    for Z in non_promoted_d:
        assert n_fill(Z) == _n_fill_aufbau(Z), f"Z={Z} should be unchanged"


def test_block_and_capacity():
    assert block_of(1) == "s"
    assert block_of(6) == "p"
    assert block_of(24) == "d"
    assert block_of(58) == "f"
    assert capacity(1) == 2
    assert capacity(6) == 6
    assert capacity(24) == 10
    assert capacity(58) == 14
