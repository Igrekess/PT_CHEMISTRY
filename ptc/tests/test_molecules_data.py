"""Tests for the bundled molecular benchmark table."""

from ptc.data.molecules import MOLECULES


def test_benchmark_contains_reference_molecules():
    for name in ("H2O", "CH4", "benzene", "SiH4", "NaCl"):
        assert name in MOLECULES


def test_reference_entries_have_expected_shape():
    water = MOLECULES["H2O"]
    methane = MOLECULES["CH4"]
    benzene = MOLECULES["benzene"]

    assert water["smiles"] == "O"
    assert abs(water["D_at"] - 9.511) < 1e-12
    assert methane["smiles"] == "C"
    assert methane["source"] == "ATcT"
    assert benzene["smiles"] == "c1ccccc1"
    assert benzene["source"] == "NIST"


def test_benchmark_is_large_enough_for_phase_one():
    assert len(MOLECULES) >= 121
