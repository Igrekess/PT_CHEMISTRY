"""Tests for Atom class and PT IE/EA engines."""

from ptc.atom import Atom, EA_eV, benchmark_atom_ea_models_against_nist, compare_ea_channels


def test_atom_hydrogen():
    a = Atom(1)
    assert a.Z == 1
    assert a.symbol == 'H'
    assert a.period == 1
    assert abs(a.IE - 13.598) < 0.5


def test_atom_carbon():
    a = Atom(6)
    assert a.symbol == 'C'
    assert a.period == 2
    assert a.l == 1
    assert abs(a.IE - 11.26) < 0.5


def test_atom_nitrogen():
    a = Atom(7)
    assert abs(a.IE - 14.534) < 0.5


def test_atom_oxygen():
    a = Atom(8)
    assert abs(a.IE - 13.618) < 0.5


def test_atom_iron():
    a = Atom(26)
    assert a.symbol == 'Fe'
    assert a.period == 4
    assert a.l == 2


def test_atom_full_nist_mode():
    a = Atom(6, source="full_nist")
    assert abs(a.IE - 11.260) < 0.001
    assert abs(a.EA - 1.262) < 0.001


def test_atom_hybrid_mode_per_property():
    a = Atom(6, source="hybrid", ie_source="full_nist", ea_source="full_pt")
    assert abs(a.IE - 11.260) < 0.001
    assert abs(a.EA - a.EA_pt) < 1e-12


def test_atom_override():
    a = Atom(6, IE=99.0)
    assert a.IE == 99.0


def test_atom_mass():
    a = Atom(1)
    assert abs(a.mass - 1.008) < 0.01


def test_atom_mulliken():
    a = Atom(6)
    assert abs(a.chi_mulliken - (a.IE + a.EA) / 2) < 0.01


def test_atom_pt_ea_is_derived_from_ie():
    h = Atom(1)
    c = Atom(6)
    f = Atom(9)
    ne = Atom(10)
    assert 0.4 < h.EA_pt < 0.9
    assert 1.0 < c.EA_pt < 1.8
    assert 2.5 < f.EA_pt < 4.5
    assert ne.EA_pt == 0.0
    assert f.EA_pt > c.EA_pt > h.EA_pt > 0.0


def test_atom_pt_ea_closed_shell_s_block_is_small():
    assert Atom(2).EA_pt == 0.0           # He: no tunneling target at per=1
    assert Atom(4).EA_pt < 0.5            # Be: small tunnel EA (exp ~ 0)
    assert Atom(12).EA_pt < 0.5           # Mg: small tunnel EA (exp ~ 0)


def test_atom_screening_properties_are_exposed():
    a = Atom(8)
    assert a.S_screening > 0.0
    assert 0.0 < a.Z_eff < a.Z


def test_ea_function_matches_atom_pt_cache():
    a = Atom(17)
    assert abs(EA_eV(17, ie=a.IE_pt) - a.EA_pt) < 1e-12


def test_atom_exposes_both_ea_channels():
    a = Atom(17)
    fe = Atom(26)
    assert a.ea_model == "classic"
    assert abs(a.EA_pt - a.EA_classic_pt) < 1e-12
    assert a.EA_operator_pt > 0.0
    assert abs(fe.EA_operator_pt - fe.EA_classic_pt) > 1e-6


def test_atom_can_select_operator_ea_model():
    a = Atom(17, ea_model="operator")
    assert a.ea_model == "operator"
    assert abs(a.EA_pt - a.EA_operator_pt) < 1e-12
    assert abs(a.EA - a.EA_operator_pt) < 1e-12


def test_operator_ea_model_keeps_full_nist_resolution():
    a = Atom(17, source="full_nist", ea_model="operator")
    assert abs(a.EA - 3.613) < 0.001
    assert a.EA_operator_pt > 0.0


def test_compare_ea_channels_exposes_both_models_for_one_atom():
    row = compare_ea_channels(17)
    assert row["symbol"] == "Cl"
    assert row["block"] == "p"
    assert abs(row["EA_classic_pt"] - Atom(17, ea_model="classic").EA_classic_pt) < 1e-12
    assert abs(row["EA_operator_pt"] - Atom(17, ea_model="operator").EA_operator_pt) < 1e-12
    assert row["EA_nist"] == 3.613
    assert row["classic_error_percent"] is not None
    assert row["operator_error_percent"] is not None


def test_benchmark_atom_ea_models_against_nist_shows_operator_global_gain():
    report = benchmark_atom_ea_models_against_nist()
    assert report["count"] == 73
    # Classic (ea_geo engine) is now superior to operator
    assert report["models"]["classic"]["mae_percent"] < 3.0
    assert "p" in report["models"]["classic"]["by_block"]
    assert "f" in report["models"]["operator"]["by_block"]


def test_ie_mae_period2():
    """PT IE for period 2 should have MAE < 5%."""
    from ptc.data.experimental import IE_NIST
    errors = []
    for Z in range(3, 11):
        a = Atom(Z)
        exp = IE_NIST[Z]
        err = abs(a.IE - exp) / exp * 100
        errors.append(err)
    mae = sum(errors) / len(errors)
    assert mae < 5.0, f"Period 2 MAE = {mae:.1f}%"


def test_ie_mae_period3():
    """PT IE for period 3 should have MAE < 5%."""
    from ptc.data.experimental import IE_NIST
    errors = []
    for Z in range(11, 19):
        a = Atom(Z)
        exp = IE_NIST[Z]
        err = abs(a.IE - exp) / exp * 100
        errors.append(err)
    mae = sum(errors) / len(errors)
    assert mae < 5.0, f"Period 3 MAE = {mae:.1f}%"
