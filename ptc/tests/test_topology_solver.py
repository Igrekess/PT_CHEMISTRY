"""Tests for topology solver."""
import pytest
from ptc.topology import Topology
from ptc.topology_solver import parse_formula, solve_topology


class TestTopologyFromAtoms:
    def test_basic(self):
        topo = Topology.from_atoms([6, 1, 1, 1, 1])
        assert topo.Z_list[0] == 6  # C first (heavy)
        assert all(Z == 1 for Z in topo.Z_list[1:])
        assert topo.bonds == []
        assert topo.source == "solver"

    def test_sorted_by_ie(self):
        topo = Topology.from_atoms([6, 8, 1, 1, 1, 1, 1, 1])
        # O (IE~13.6) before C (IE~11.3)
        assert topo.Z_list[0] == 8
        assert topo.Z_list[1] == 6


class TestParseFormula:
    def test_ch4(self):
        assert sorted(parse_formula("CH4")) == sorted([6, 1, 1, 1, 1])

    def test_c2h6(self):
        result = parse_formula("C2H6")
        assert sorted(result) == sorted([6, 6, 1, 1, 1, 1, 1, 1])

    def test_co(self):
        assert sorted(parse_formula("CO")) == sorted([6, 8])

    def test_nacl(self):
        assert sorted(parse_formula("NaCl")) == sorted([11, 17])

    def test_h2o(self):
        assert sorted(parse_formula("H2O")) == sorted([1, 1, 8])

    def test_h2so4(self):
        result = parse_formula("H2SO4")
        assert sorted(result) == sorted([1, 1, 16, 8, 8, 8, 8])

    def test_unknown_element(self):
        with pytest.raises(ValueError, match="Unknown element"):
            parse_formula("Xx2")


class TestSolveTopology:
    def test_methane(self):
        """CH4 -> C with 4 single bonds to H."""
        topo = solve_topology("CH4")
        assert len(topo.bonds) == 4
        for i, j, bo in topo.bonds:
            Zs = {topo.Z_list[i], topo.Z_list[j]}
            assert Zs == {6, 1}, f"Unexpected bond: {Zs}"
            assert bo == 1.0

    def test_water(self):
        """H2O -> O with 2 single bonds to H."""
        topo = solve_topology("H2O")
        assert len(topo.bonds) == 2
        for i, j, bo in topo.bonds:
            Zs = {topo.Z_list[i], topo.Z_list[j]}
            assert Zs == {8, 1}

    def test_ammonia(self):
        """NH3 -> N with 3 single bonds to H."""
        topo = solve_topology("NH3")
        assert len(topo.bonds) == 3

    def test_ethane(self):
        """C2H6 -> C-C + 6 C-H."""
        topo = solve_topology("C2H6")
        assert len(topo.bonds) == 7
        cc = [(i, j, bo) for i, j, bo in topo.bonds
              if topo.Z_list[i] == 6 and topo.Z_list[j] == 6]
        assert len(cc) == 1

    def test_hf(self):
        """HF -> H-F single bond."""
        topo = solve_topology("HF")
        assert len(topo.bonds) == 1
        assert topo.bonds[0][2] == 1.0


from ptc.molecule import Molecule


class TestMoleculeAutoDetect:
    def test_smiles_still_works(self):
        mol = Molecule("C")
        assert mol.D_at > 0
        assert mol.topology.source == "smiles"

    def test_formula_triggers_solver(self):
        mol = Molecule("CH4")
        assert mol.D_at > 0
        assert mol.topology.source == "solver"

    def test_formula_matches_smiles_ch4(self):
        d_smi = Molecule("C").D_at
        d_form = Molecule("CH4").D_at
        err = abs(d_smi - d_form) / d_smi * 100
        assert err < 0.5, f"CH4 mismatch: {err:.2f}%"

    def test_formula_matches_smiles_h2o(self):
        d_smi = Molecule("O").D_at
        d_form = Molecule("H2O").D_at
        err = abs(d_smi - d_form) / d_smi * 100
        assert err < 0.5, f"H2O mismatch: {err:.2f}%"

    def test_formula_matches_smiles_nh3(self):
        d_smi = Molecule("N").D_at
        d_form = Molecule("NH3").D_at
        err = abs(d_smi - d_form) / d_smi * 100
        assert err < 0.5, f"NH3 mismatch: {err:.2f}%"

    def test_ethane_formula(self):
        d_smi = Molecule("CC").D_at
        d_form = Molecule("C2H6").D_at
        err = abs(d_smi - d_form) / d_smi * 100
        assert err < 0.5, f"C2H6 mismatch: {err:.2f}%"


class TestSolverEndToEnd:
    """Parametrized: verify solver finds correct structure for diverse formulas."""

    import pytest

    @pytest.mark.parametrize("formula,smiles,max_err", [
        ("CH4", "C", 0.5),
        ("H2O", "O", 0.5),
        ("NH3", "N", 0.5),
        ("C2H6", "CC", 0.5),
        ("HF", "[H]F", 0.5),
        ("H2S", "S", 0.5),
        ("HCl", "[H]Cl", 0.5),
    ])
    def test_formula_vs_smiles(self, formula, smiles, max_err):
        d_smi = Molecule(smiles).D_at
        d_form = Molecule(formula).D_at
        err = abs(d_smi - d_form) / d_smi * 100
        assert err < max_err, (
            f"{formula}: SMILES({smiles})={d_smi:.3f} vs "
            f"formula={d_form:.3f} ({err:.2f}%)"
        )
