"""Tests for bundled experimental atomic data tables."""

from ptc.data.experimental import EA_NIST, IE_NIST, MASS, SYMBOLS, symbol_to_Z


def test_tables_have_core_entries():
    for z in (1, 6, 8, 17, 26, 79):
        assert z in IE_NIST
        assert z in MASS
        assert z in SYMBOLS


def test_known_atomic_values():
    assert abs(IE_NIST[1] - 13.598) < 1e-12
    assert abs(IE_NIST[8] - 13.618) < 1e-12
    assert abs(EA_NIST[9] - 3.401) < 1e-12
    assert abs(MASS[6] - 12.011) < 1e-12
    assert SYMBOLS[17] == "Cl"


def test_symbol_to_z():
    assert symbol_to_Z("H") == 1
    assert symbol_to_Z("O") == 8
    assert symbol_to_Z("Cl") == 17
    assert symbol_to_Z("Og") == 118
