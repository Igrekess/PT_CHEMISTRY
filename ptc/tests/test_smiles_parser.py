"""Tests for the standalone SMILES parser bundled in PTC phase 1."""

from ptc.smiles_parser import is_smiles, parse_smiles


def test_is_smiles_heuristic():
    assert is_smiles("O=C=O") is True
    assert is_smiles("c1ccccc1") is True
    assert is_smiles("H2O") is False
    assert is_smiles("(CH3)3N") is False


def test_parse_water_with_implicit_hydrogen():
    atoms, bonds = parse_smiles("O")
    assert len(atoms) == 3
    assert len(bonds) == 2
    assert [a.symbol for a in atoms] == ["O", "H", "H"]
    assert [b.order for b in bonds] == [1.0, 1.0]


def test_parse_ethane():
    atoms, bonds = parse_smiles("CC")
    assert len(atoms) == 8
    assert len(bonds) == 7
    assert sum(1 for a in atoms if a.symbol == "C") == 2
    assert sum(1 for a in atoms if a.symbol == "H") == 6


def test_parse_double_bond():
    atoms, bonds = parse_smiles("C=C")
    assert len(atoms) == 6
    assert len(bonds) == 5
    assert bonds[0].order == 2.0


def test_parse_aromatic_ring():
    atoms, bonds = parse_smiles("c1ccccc1")
    assert len(atoms) == 12
    assert len(bonds) == 12
    aromatic_orders = [b.order for b in bonds[:6]]
    assert aromatic_orders == [1.5] * 6


def test_parse_small_ring():
    atoms, bonds = parse_smiles("C1CC1")
    assert len(atoms) == 9
    assert len(bonds) == 9
    assert sum(1 for a in atoms if a.symbol == "C") == 3
