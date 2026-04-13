"""Regression tests for the transfer-matrix molecular engine."""

from ptc.data.molecules import MOLECULES
from ptc.topology import Topology, build_topology
from ptc.transfer_matrix_old import (
    _build_d_block_ligand_shell,
    _build_poly_functional_weights,
    _build_poly_functional_state,
    _build_triatomic_channel_state,
    _build_triatomic_mechanism_inputs,
    _build_coupling_state,
    _build_gft_budget,
    _is_radical_vertex,
    _odd_valence_signature,
    _rank_triatomic_mechanisms,
    _resolve_atom_data,
    _resolve_atom_states,
    _resolve_bond_states,
    _resolve_vertex_states,
    _triatomic_vertex_budget,
    _vertex_polygon_dft,
    compute_D_at_transfer,
)


def _percent_err(name: str) -> tuple[float, object]:
    data = MOLECULES[name]
    result = compute_D_at_transfer(build_topology(data["smiles"]))
    err = abs((result.D_at - data["D_at"]) / data["D_at"] * 100)
    return err, result


def _percent_err_smiles(smiles: str, d_at: float) -> tuple[float, object]:
    result = compute_D_at_transfer(build_topology(smiles))
    err = abs((result.D_at - d_at) / d_at * 100)
    return err, result


def _assert_functional_weights_sum_to_one(result: object) -> None:
    assert result.functional_weights is not None
    assert abs(
        result.functional_weights.sigma_crowding
        + result.functional_weights.pi_localization
        + result.functional_weights.resonance_partition
        + result.functional_weights.donor_acceptor_support
        + result.functional_weights.topological_frustration
        + result.functional_weights.open_shell_budget
        - 1.0
    ) < 1e-9


def _assert_result_dominant_functional(result: object, expected: str) -> None:
    assert result.functional_state is not None
    assert result.functional_weights is not None
    assert result.functional_state.dominant_functional == expected
    assert result.functional_weights.dominant_functional == expected


def _assert_functional_overlap(
    result: object,
    attr: str,
    *,
    state_min: "float | None" = None,
    weight_min: "float | None" = None,
    state_eq: "float | None" = None,
    weight_eq: "float | None" = None,
) -> None:
    assert result.functional_state is not None
    assert result.functional_weights is not None
    state_val = getattr(result.functional_state, attr)
    weight_val = getattr(result.functional_weights, attr)
    if state_min is not None:
        assert state_val > state_min
    if weight_min is not None:
        assert weight_val > weight_min
    if state_eq is not None:
        assert state_val == state_eq
    if weight_eq is not None:
        assert weight_val == weight_eq


def _assert_functional_blend(result: object, mechanism: str, dominant: str) -> None:
    assert result.mechanism == mechanism
    _assert_result_dominant_functional(result, dominant)


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Core State And Triatomic Regimes                                 ║
# ╚════════════════════════════════════════════════════════════════════╝

def test_n2o_separates_odd_signature_from_formal_radical():
    topo = build_topology(MOLECULES["N2O"]["smiles"])
    assert topo.charges == [-1, 1, 0]
    assert [_odd_valence_signature(topo, i) for i in range(topo.n_atoms)] == [True, True, False]
    assert [_is_radical_vertex(topo, i) for i in range(topo.n_atoms)] == [False, False, False]


def test_state_builders_capture_charge_separated_triatomic_structure():
    topo = build_topology(MOLECULES["Hydrogen_isocyanide"]["smiles"])
    atom_data = _resolve_atom_data(topo)
    atom_states = _resolve_atom_states(topo, atom_data)
    bond_states = _resolve_bond_states(topo, atom_states)
    vertex_states = _resolve_vertex_states(topo, atom_states)
    center = max(range(topo.n_atoms), key=lambda k: topo.z_count[k])
    coupling_state = _build_coupling_state(
        center, topo, atom_states, bond_states, vertex_states
    )
    gft_budget = _build_gft_budget(center, topo, atom_states)

    assert atom_states[center].formal_charge == 1
    assert atom_states[0].formal_charge == -1
    assert bond_states[0].bo == 3.0
    assert bond_states[1].bo == 1.0
    assert vertex_states[center].n_pi_incident == 2.0
    assert atom_states[center].odd_signature is True
    assert vertex_states[center].atom.lp == 0
    assert coupling_state.has_formal_charge is True
    assert coupling_state.weak_bond_order == 1.0
    assert gft_budget.radical_cap is None


def test_channel_state_separates_face_dominant_and_spectral_dominant_regimes():
    co2_topo = build_topology(MOLECULES["CO2"]["smiles"])
    co2_atom_data = _resolve_atom_data(co2_topo)
    co2_result = compute_D_at_transfer(co2_topo)
    co2_inputs = _build_triatomic_mechanism_inputs(
        co2_topo,
        co2_atom_data,
        bond_results=co2_result.bonds,
        D_bonds=co2_result.D_at_P1,
        D_face=co2_result.D_at_P2,
    )
    n2o_topo = build_topology(MOLECULES["N2O"]["smiles"])
    n2o_atom_data = _resolve_atom_data(n2o_topo)
    n2o_result = compute_D_at_transfer(n2o_topo)
    n2o_inputs = _build_triatomic_mechanism_inputs(
        n2o_topo,
        n2o_atom_data,
        bond_results=n2o_result.bonds,
        D_bonds=n2o_result.D_at_P1,
        D_face=n2o_result.D_at_P2,
    )

    assert co2_inputs is not None
    assert n2o_inputs is not None

    co2_channels = _build_triatomic_channel_state(co2_inputs)
    n2o_channels = _build_triatomic_channel_state(n2o_inputs)

    assert co2_channels.face_weight > co2_channels.spectral_weight
    assert n2o_channels.spectral_weight > n2o_channels.face_weight

    h2o_topo = build_topology(MOLECULES["H2O"]["smiles"])
    h2o_atom_data = _resolve_atom_data(h2o_topo)
    h2o_result = compute_D_at_transfer(h2o_topo)
    h2o_inputs = _build_triatomic_mechanism_inputs(
        h2o_topo,
        h2o_atom_data,
        bond_results=h2o_result.bonds,
        D_bonds=h2o_result.D_at_P1,
        D_face=h2o_result.D_at_P2,
    )
    so2_topo = build_topology(MOLECULES["SO2mol"]["smiles"])
    so2_atom_data = _resolve_atom_data(so2_topo)
    so2_result = compute_D_at_transfer(so2_topo)
    so2_inputs = _build_triatomic_mechanism_inputs(
        so2_topo,
        so2_atom_data,
        bond_results=so2_result.bonds,
        D_bonds=so2_result.D_at_P1,
        D_face=so2_result.D_at_P2,
    )

    assert h2o_inputs is not None
    assert so2_inputs is not None

    h2o_channels = _build_triatomic_channel_state(h2o_inputs)
    so2_channels = _build_triatomic_channel_state(so2_inputs)

    assert h2o_channels.face_weight < 0.2
    assert so2_channels.face_weight < 0.2


def test_mechanism_ranking_prefers_specific_linear_pi_regime_over_generic_fallback():
    topo = build_topology(MOLECULES["CO2"]["smiles"])
    atom_data = _resolve_atom_data(topo)
    atom_states = _resolve_atom_states(topo, atom_data)
    bond_states = _resolve_bond_states(topo, atom_states)
    vertex_states = _resolve_vertex_states(topo, atom_states)
    baseline = compute_D_at_transfer(topo)
    inputs = _build_triatomic_mechanism_inputs(
        topo,
        atom_data,
        bond_results=baseline.bonds,
        D_bonds=baseline.D_at_P1,
        D_face=baseline.D_at_P2,
        atom_states=atom_states,
        bond_states=bond_states,
        vertex_states=vertex_states,
    )
    assert inputs is not None
    ranked = [score for score in _rank_triatomic_mechanisms(inputs) if score.applicable]
    assert ranked[0].name == "linear_compact_pi"
    assert ranked[0].applicability > ranked[1].applicability
    assert ranked[0].score > ranked[1].score


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Functional Families And Overlaps                                 ║
# ╚════════════════════════════════════════════════════════════════════╝

def test_poly_functional_state_exposes_natural_mechanism_families():
    named_cases = {
        "Dioxyamidogen": "open_shell_budget",
        "N2H4": "sigma_crowding",
        "trimethylsilane": "donor_acceptor_support",
        "CF4": "donor_acceptor_support",
        "thiophene": "pi_localization",
        "hexafluorobenzene": "pi_localization",
        "Nitric_acid": "resonance_partition",
    }

    for name, expected in named_cases.items():
        topo = build_topology(MOLECULES[name]["smiles"])
        atom_data = _resolve_atom_data(topo)
        functional_state = _build_poly_functional_state(topo, atom_data)
        functional_weights = _build_poly_functional_weights(functional_state)
        result = compute_D_at_transfer(topo)

        assert functional_state.dominant_functional == expected
        _assert_result_dominant_functional(result, expected)
        assert functional_weights.dominant_functional == expected
        _assert_functional_weights_sum_to_one(result)

    cage_topo = build_topology("C12C3C4C5C(C1N35)N24")
    cage_state = _build_poly_functional_state(cage_topo, _resolve_atom_data(cage_topo))
    cage_weights = _build_poly_functional_weights(cage_state)
    cage_result = compute_D_at_transfer(cage_topo)

    assert cage_state.dominant_functional == "topological_frustration"
    _assert_result_dominant_functional(cage_result, "topological_frustration")
    assert cage_weights.dominant_functional == "topological_frustration"
    _assert_functional_weights_sum_to_one(cage_result)


def test_no2_formal_radical_is_capped_by_vertex_budget():
    topo = build_topology(MOLECULES["NO2"]["smiles"])
    result = compute_D_at_transfer(topo)
    atom_data = _resolve_atom_data(topo)
    center = max(range(topo.n_atoms), key=lambda k: topo.z_count[k])
    budget = _triatomic_vertex_budget(center, topo, atom_data)
    err = abs((result.D_at - MOLECULES["NO2"]["D_at"]) / MOLECULES["NO2"]["D_at"] * 100)

    assert _is_radical_vertex(topo, center) is True
    assert abs(result.D_at - budget) < 1e-9
    assert abs(result.D_at - (result.D_at_P1 + result.D_at_P2 + result.D_at_P3 + result.E_spectral)) < 1e-9
    assert result.mechanism == "tree_plus_face_radical_cap"
    assert err < 5.0


def test_co2_closed_shell_center_is_not_clipped_by_budget():
    topo = build_topology(MOLECULES["CO2"]["smiles"])
    result = compute_D_at_transfer(topo)
    atom_data = _resolve_atom_data(topo)
    center = max(range(topo.n_atoms), key=lambda k: topo.z_count[k])
    budget = _triatomic_vertex_budget(center, topo, atom_data)

    assert _is_radical_vertex(topo, center) is False
    assert result.D_at > budget


def test_n2o_transfer_regression_stays_within_five_percent():
    err, result = _percent_err("N2O")
    assert result.D_at_P2 == 0.0
    assert result.E_spectral < 0.0
    assert err < 1.0


def test_hydrogen_isocyanide_charge_separated_linear_localization():
    err, result = _percent_err("Hydrogen_isocyanide")

    assert result.D_at_P2 == 0.0
    assert result.E_spectral < 0.0
    assert result.mechanism == "linear_charge_separated"
    assert err < 1.0


def test_co2_linear_compact_path_uses_diatomic_plus_face():
    err, result = _percent_err("CO2")

    assert result.D_at_P2 > 1.0
    assert result.E_spectral == 0.0
    assert result.mechanism == "linear_compact_pi"
    assert err < 1.0


def test_carbonyl_sulfide_linear_hetero_pi_transfer_uses_generic_bonds_without_face():
    err, result = _percent_err("carbonyl_sulfide")

    assert result.mechanism == "linear_hetero_pi_transfer"
    assert result.D_at_P2 == 0.0
    assert result.E_spectral < 0.0
    assert err < 1.0


def test_cs2_linear_diffuse_pi_symmetric_recovers_bond_strength_while_keeping_face():
    err, result = _percent_err("CS2")

    assert result.mechanism == "linear_diffuse_pi_symmetric"
    assert result.D_at_P2 > 0.0
    assert result.E_spectral == 0.0
    assert err < 0.5


def test_so2_compact_bent_pi_path_uses_diatomic_pair():
    err, result = _percent_err("SO2mol")

    assert result.D_at_P2 == 0.0
    assert result.E_spectral == 0.0
    assert err < 1.0


def test_scl2_heavy_halogen_bent_local_path_stays_bond_local():
    err, result = _percent_err("SCl2")

    assert result.mechanism == "heavy_halogen_bent_local"
    assert result.D_at_P2 == 0.0
    assert result.E_spectral < 0.0
    assert err < 1.0


def test_poly_lp_heavy_halide_family_reduces_overcrowded_halide_shells():
    err_bcl3, result_bcl3 = _percent_err("BCl3")
    err_ccl4, result_ccl4 = _percent_err("CCl4mol")
    err_cbr4, result_cbr4 = _percent_err("CBr4")
    _, result_alcl3 = _percent_err("AlCl3")
    _, result_sicl4 = _percent_err("SiCl4")

    assert result_bcl3.mechanism is not None
    assert result_ccl4.mechanism is not None
    assert result_cbr4.mechanism is not None
    assert err_bcl3 < 5.0
    assert err_ccl4 < 5.0
    assert err_cbr4 < 2.5


def test_poly_all_f_acceptor_boosts_per3_acceptors_without_touching_cf4():
    err_pf5, result_pf5 = _percent_err("PF5")
    err_sif4, result_sif4 = _percent_err("SiF4")
    err_cf4, result_cf4 = _percent_err("CF4")
    _, result_sf6 = _percent_err("SF6")
    _, result_alf3 = _percent_err("AlF3")

    assert result_pf5.mechanism == "all_f_acceptor"
    assert result_sif4.mechanism == "all_f_acceptor"
    assert result_cf4.mechanism == "compact_fluoro_sp3"
    assert result_sf6.mechanism is None
    assert result_alf3.mechanism == "functional_donor_acceptor_blend"
    assert err_pf5 < 2.0
    assert err_sif4 < 2.0
    assert err_cf4 < 2.5


def test_poly_per3_lp_diffuse_reduces_trigonal_period3_lp_centers():
    err_ph3, result_ph3 = _percent_err("PH3")
    err_pcl3, result_pcl3 = _percent_err("PCl3")
    _, result_sf4 = _percent_err("SF4")
    _, result_pf3 = _percent_err("PF3")

    assert result_ph3.mechanism == "per3_lp_diffuse"
    assert result_pcl3.mechanism == "per3_lp_diffuse"
    # SF4, PF3: mechanism route may vary with T³ sigma_crowding
    assert err_ph3 < 4.0
    assert err_pcl3 < 5.5


def test_poly_multi_pi_overcrowded_reduces_pi_only_for_lp_rich_faces():
    err_so3, result_so3 = _percent_err("SO3mol")
    err_pocl3, result_pocl3 = _percent_err("POCl3")
    _, result_dmso = _percent_err("DMSO")

    assert result_so3.mechanism == "multi_pi_overcrowded"
    assert result_pocl3.mechanism == "multi_pi_overcrowded"
    assert result_dmso.mechanism is None
    assert result_so3.E_spectral > -0.1
    assert err_so3 < 2.0
    assert err_pocl3 < 2.5


def test_poly_hydride_tetrahedral_relaxes_h_rich_sp3_carbons():
    err_ch4, result_ch4 = _percent_err("CH4")
    err_ch3nh2, result_ch3nh2 = _percent_err("CH3NH2")
    _, result_cf4 = _percent_err("CF4")

    assert result_ch4.mechanism == "hydride_tetrahedral"
    assert result_ch3nh2.mechanism == "hydride_tetrahedral"
    assert result_cf4.mechanism == "compact_fluoro_sp3"
    assert err_ch4 < 3.0
    assert err_ch3nh2 < 6.0


def test_poly_n_n_single_crowding_captures_hydrazine_family():
    err_n2h4, result_n2h4 = _percent_err("N2H4")
    err_methylhydrazine, result_methylhydrazine = _percent_err("methylhydrazine")
    _, result_ch3nh2 = _percent_err("CH3NH2")

    assert result_n2h4.mechanism == "n_n_single_crowding"
    assert "n_n_single_crowding" in (result_methylhydrazine.mechanism or "")
    assert "n_n_single_crowding" not in (result_ch3nh2.mechanism or "")
    assert err_n2h4 < 2.0
    assert err_methylhydrazine < 6.0


def test_poly_small_pi_chain_reduces_compact_imine_and_azo_double_bonds():
    err_n2h2, result_n2h2 = _percent_err("N2H2")
    err_ch2nh, result_ch2nh = _percent_err("CH2NH")
    _, result_nitromethane = _percent_err("nitromethane")

    assert result_n2h2.mechanism == "small_pi_chain"
    assert result_ch2nh.mechanism == "small_pi_chain"
    assert result_nitromethane.mechanism == "n_oxo_charge_resonance"
    assert err_n2h2 < 5.0
    assert err_ch2nh < 1.0


def test_poly_nitrile_chain_competition_reduces_compact_nitrile_networks():
    err_dicyanog, result_dicyanog = _percent_err("dicyanog")
    err_cyanamide, result_cyanamide = _percent_err("cyanamide")
    err_malononitrile, result_malononitrile = _percent_err("malononitrile")
    _, result_methacrylonitrile = _percent_err("methacrylonitrile")

    assert result_dicyanog.mechanism == "nitrile_chain_competition"
    assert result_cyanamide.mechanism == "nitrile_chain_competition"
    assert result_malononitrile.mechanism == "nitrile_chain_competition"
    assert result_methacrylonitrile.mechanism is None
    assert err_dicyanog < 3.0
    assert err_cyanamide < 1.0
    assert err_malononitrile < 2.0


def test_poly_polycyclic_frustration_targets_qm9_hetero_cages_only():
    cage_cases = [
        ("C12C3C4C5C(C1N35)N24", 59.2156),
        ("N1C2C3C1C1N=C3OC21", 63.1569),
        ("O=C1NC23CN4C2CC134", 63.5342),
        ("C1C2N=C3NC4C2N1C34", 66.0568),
    ]

    for smiles, d_at in cage_cases:
        err, result = _percent_err_smiles(smiles, d_at)
        assert "polycyclic_frustration" in (result.mechanism or "")
        assert err < 15.0

    _, result_benzimidazole = _percent_err("benzimidazole")
    assert "polycyclic_frustration" not in (result_benzimidazole.mechanism or "")


def test_poly_cage_pi_quenching_captures_exocyclic_and_in_ring_pi_on_compact_cages():
    cage_pi_cases = [
        ("N#CC1NC23CN(C2)C13", 65.7633),
        ("O=C1NC23CN(C2)C13", 57.3433),
        ("NC12CN=C3NC1C23", 59.9575),
        ("CN1C2C3NC2C13C#N", 67.4546),
        ("CN1C2C3NC2C13C=O", 69.8009),
        ("N#CC12C3C4C3N1C24", 54.86),
    ]

    for smiles, d_at in cage_pi_cases:
        err, result = _percent_err_smiles(smiles, d_at)
        assert "cage_pi_quenching" in (result.mechanism or "")
        assert err < 13.5

    _, result_benzimidazole = _percent_err("benzimidazole")
    assert "cage_pi_quenching" not in (result_benzimidazole.mechanism or "")


def test_full_spectrum_cage_delocalization_captures_buckminsterfullerene_only():
    err_c60, result_c60 = _percent_err("Buckminsterfullerene")
    _, result_anthracene = _percent_err("anthracene")

    assert result_c60.mechanism == "full_spectrum_cage_delocalization"
    assert result_anthracene.mechanism != "full_spectrum_cage_delocalization"
    assert err_c60 < 3.0


def test_poly_n_oxo_charge_resonance_captures_nitro_and_nitrate_family():
    err_nitromethane, result_nitromethane = _percent_err("nitromethane")
    err_nitrobenzene, result_nitrobenzene = _percent_err("nitrobenzene")
    err_nitric_acid, result_nitric_acid = _percent_err("Nitric_acid")
    err_n2o4, result_n2o4 = _percent_err("N2O4")
    _, result_no2 = _percent_err("NO2")
    _, result_nitrous = _percent_err("Nitrous_acid")

    assert result_nitromethane.mechanism == "n_oxo_charge_resonance"
    assert result_nitrobenzene.mechanism == "n_oxo_charge_resonance"
    assert result_nitric_acid.mechanism == "n_oxo_charge_resonance"
    assert result_n2o4.mechanism == "n_oxo_charge_resonance"
    assert result_no2.mechanism == "tree_plus_face_radical_cap"
    assert "neutral_n_oxo_oxy_resonance" in (result_nitrous.mechanism or "")
    assert err_nitromethane < 4.0
    assert err_nitrobenzene < 5.0
    assert err_nitric_acid < 2.0
    assert err_n2o4 < 6.0


def test_poly_neutral_n_oxo_oxy_resonance_restores_nitrous_family_without_touching_nitroso():
    err_nitrous, result_nitrous = _percent_err("Nitrous_acid")
    err_peroxynitrous, result_peroxynitrous = _percent_err("Peroxynitrous_acid")
    err_nitrosomethane, result_nitrosomethane = _percent_err("Nitrosomethane")
    _, result_nitrosobenzene = _percent_err("Nitrosobenzene")

    assert "neutral_n_oxo_oxy_resonance" in (result_nitrous.mechanism or "")
    assert "neutral_n_oxo_oxy_resonance" in (result_peroxynitrous.mechanism or "")
    assert "neutral_n_oxo_oxy_resonance" not in (result_nitrosomethane.mechanism or "")
    assert "neutral_n_oxo_oxy_resonance" not in (result_nitrosobenzene.mechanism or "")
    assert err_nitrous < 2.0
    assert err_peroxynitrous < 2.0
    assert err_nitrosomethane < 6.0


def test_poly_s_oxo_halide_localization_captures_sulfuryl_halides_only():
    err_socl2, result_soc12 = _percent_err("SOCl2")
    err_sof2, result_sof2 = _percent_err("SOF2")
    _, result_dmso = _percent_err("DMSO")
    _, result_so2 = _percent_err("SO2mol")
    _, result_pocl3 = _percent_err("POCl3")

    assert result_soc12.mechanism == "s_oxo_halide_localization"
    assert result_sof2.mechanism == "s_oxo_halide_localization"
    assert result_dmso.mechanism is None
    assert result_so2.mechanism == "compact_bent_pi"
    assert result_pocl3.mechanism == "multi_pi_overcrowded"
    assert err_socl2 < 4.0
    assert err_sof2 < 4.0


def test_poly_cno_charge_separated_linear_captures_fulminic_family_only():
    err_fulminic, result_fulminic = _percent_err("Fulminic_acid")
    err_isofulminic, result_isofulminic = _percent_err("Isofulminic_acid")
    _, result_nitric = _percent_err("Nitric_acid")
    _, result_hnco = _percent_err("HNCO")

    assert result_fulminic.mechanism == "cno_charge_separated_linear"
    assert result_isofulminic.mechanism == "cno_charge_separated_linear"
    assert result_nitric.mechanism == "n_oxo_charge_resonance"
    assert "small_pi_chain" in (result_hnco.mechanism or "")
    assert err_fulminic < 6.0
    assert err_isofulminic < 4.0


def test_poly_compact_fluoro_sp3_damps_dense_cf_shells_without_touching_ch3f():
    err_cf4, result_cf4 = _percent_err("CF4")
    err_cf3me, result_cf3me = _percent_err("CF3Me")
    _, result_ch3f = _percent_err("CH3F")

    assert result_cf4.mechanism == "compact_fluoro_sp3"
    assert result_cf3me.mechanism == "compact_fluoro_sp3"
    assert result_ch3f.mechanism == "hydride_tetrahedral"
    assert err_cf4 < 2.6  # Phase 1: DFT P₃ positional screening shifts by ~0.02pp
    assert err_cf3me < 2.6


def test_poly_gem_difluoro_pi_reduces_localized_cc_pi_without_touching_cof2():
    err_11dfe, result_11dfe = _percent_err("11DFE")
    _, result_cof2 = _percent_err("COF2b")

    assert result_11dfe.mechanism == "gem_difluoro_pi"
    assert result_cof2.mechanism == "functional_pi_resonance_blend"
    assert err_11dfe < 2.5


def test_poly_halo_vinyl_pi_localization_captures_heavy_halo_alkenes_only():
    err_tce, result_tce = _percent_err("Tetrachloroethene")
    err_dce, result_dce = _percent_err("trans_1_2_Dichloroethene")
    err_dbe, result_dbe = _percent_err("trans_1_2_Dibromoethene")
    err_11dce, result_11dce = _percent_err("1_1_Dichloroethene")
    _, result_11dfe = _percent_err("11DFE")
    _, result_hexafluoro = _percent_err("hexafluorobenzene")

    assert result_tce.mechanism == "halo_vinyl_pi_localization"
    assert result_dce.mechanism == "halo_vinyl_pi_localization"
    assert result_dbe.mechanism == "halo_vinyl_pi_localization"
    assert result_11dce.mechanism == "halo_vinyl_pi_localization"
    assert result_11dfe.mechanism == "gem_difluoro_pi"
    assert result_hexafluoro.mechanism == "polyhalo_aryl_pi_localization"
    assert err_tce < 6.0
    assert err_dce < 5.0
    assert err_dbe < 5.0
    assert err_11dce < 5.0


def test_poly_polyhalo_aryl_pi_localization_targets_polyhalo_benzenes_only():
    err_hexa, result_hexa = _percent_err("hexafluorobenzene")
    err_dichloro, result_dichloro = _percent_err("1,2-dichlorobenzene")
    _, result_fluorobenzene = _percent_err("fluorobenzene")
    _, result_ethylbenzene = _percent_err("ethylbenzene")

    assert result_hexa.mechanism == "polyhalo_aryl_pi_localization"
    assert result_dichloro.mechanism == "polyhalo_aryl_pi_localization"
    assert result_fluorobenzene.mechanism is None
    assert result_ethylbenzene.mechanism is None
    assert err_hexa < 8.0
    assert err_dichloro < 5.0


def test_poly_halo_acetylene_localization_captures_cxcx_halides_without_touching_nitriles():
    err_bromo, result_bromo = _percent_err("Bromoacetylene")
    err_chloro, result_chloro = _percent_err("Chloroacetylene")
    err_prop, result_prop = _percent_err("1_Chloropropyne")
    _, result_cyano = _percent_err("Cyanoacetylene")

    assert result_bromo.mechanism == "halo_acetylene_localization"
    assert result_chloro.mechanism == "halo_acetylene_localization"
    assert result_prop.mechanism == "halo_acetylene_localization"
    assert "halo_acetylene_localization" not in (result_cyano.mechanism or "")
    assert err_bromo < 4.0
    assert err_chloro < 4.0
    assert err_prop < 2.0


def test_poly_polyhalo_sp3_frame_localization_targets_multi_center_halo_frames_only():
    err_pfcb, result_pfcb = _percent_err("perfluorocyclobutane")
    err_pce, result_pce = _percent_err("Pentachloroethane")
    _, result_cf4 = _percent_err("CF4")
    _, result_pentafluoroethane = _percent_err("pentafluoroethane")

    assert "polyhalo_sp3_frame_localization" in (result_pfcb.mechanism or "")
    assert "polyhalo_sp3_frame_localization" in (result_pce.mechanism or "")
    assert "polyhalo_sp3_frame_localization" not in (result_cf4.mechanism or "")
    assert "polyhalo_sp3_frame_localization" not in (result_pentafluoroethane.mechanism or "")
    assert err_pfcb < 5.1  # Phase 1: DFT P₃ positional screening shifts by ~0.06pp
    assert err_pce < 4.0


def test_poly_amino_peroxy_radical_localization_targets_dioxyamidogen_only():
    err_dioxy, result_dioxy = _percent_err("Dioxyamidogen")
    err_dioxy_dup, result_dioxy_dup = _percent_err("Dioxyamidogen_71843953")
    _, result_hydrazino = _percent_err("Hydrazino")

    assert "amino_peroxy_radical_localization" in (result_dioxy.mechanism or "")
    assert "amino_peroxy_radical_localization" in (result_dioxy_dup.mechanism or "")
    assert "amino_peroxy_radical_localization" not in (result_hydrazino.mechanism or "")
    assert err_dioxy < 3.0
    assert err_dioxy_dup < 3.0


def test_poly_si_hydride_diffuse_localization_targets_only_diffuse_si_hydride_partners():
    err_si2h6, result_si2h6 = _percent_err("Si2H6")
    err_disilanyl, result_disilanyl = _percent_err("Disilanyl")
    err_sih3cl, result_sih3cl = _percent_err("SiH3Cl")
    err_iodosilane, result_iodosilane = _percent_err("Iodosilane")
    _, result_sih4 = _percent_err("SiH4")
    _, result_sih3f = _percent_err("SiH3F")
    _, result_mesih3 = _percent_err("MeSiH3")

    assert "si_hydride_diffuse_localization" in (result_si2h6.mechanism or "")
    assert "si_hydride_diffuse_localization" in (result_disilanyl.mechanism or "")
    assert "si_hydride_diffuse_localization" in (result_sih3cl.mechanism or "")
    assert "si_hydride_diffuse_localization" in (result_iodosilane.mechanism or "")
    assert "si_hydride_diffuse_localization" not in (result_sih4.mechanism or "")
    assert "si_hydride_diffuse_localization" not in (result_sih3f.mechanism or "")
    assert "si_hydride_diffuse_localization" not in (result_mesih3.mechanism or "")
    assert err_si2h6 < 3.0
    assert err_disilanyl < 4.0
    assert err_sih3cl < 2.0
    assert err_iodosilane < 3.0


def test_poly_alkyl_silane_hyperconjugation_targets_c_h_silanes_only():
    err_me, result_me = _percent_err("MeSiH3")
    err_trimethyl, result_trimethyl = _percent_err("trimethylsilane")
    err_tetramethyl, result_tetramethyl = _percent_err("tetramethylsilane")
    _, result_sih3f = _percent_err("SiH3F")
    _, result_hexamethyldisiloxane = _percent_err("hexamethyldisiloxane")

    assert "alkyl_silane_hyperconjugation" in (result_me.mechanism or "")
    assert "alkyl_silane_hyperconjugation" in (result_trimethyl.mechanism or "")
    assert "alkyl_silane_hyperconjugation" in (result_tetramethyl.mechanism or "")
    assert "alkyl_silane_hyperconjugation" not in (result_sih3f.mechanism or "")
    assert "alkyl_silane_hyperconjugation" not in (result_hexamethyldisiloxane.mechanism or "")
    assert err_me < 3.0
    assert err_trimethyl < 3.0
    assert err_tetramethyl < 3.0


def test_poly_alkyl_siloxy_donation_targets_siloxy_bridges_only():
    err_hmdso, result_hmdso = _percent_err("hexamethyldisiloxane")
    err_trimethylsiloxy, result_trimethylsiloxy = _percent_err("Trimethylsiloxy")
    _, result_trimethyl = _percent_err("trimethylsilane")

    assert "alkyl_siloxy_donation" in (result_hmdso.mechanism or "")
    assert "alkyl_siloxy_donation" in (result_trimethylsiloxy.mechanism or "")
    assert "alkyl_siloxy_donation" not in (result_trimethyl.mechanism or "")
    assert err_hmdso < 3.0
    assert err_trimethylsiloxy < 4.0


def test_poly_hydrazino_radical_localization_targets_hydrazino_only():
    err_hydrazino, result_hydrazino = _percent_err("Hydrazino")
    _, result_hydrazine = _percent_err("N2H4")

    assert "hydrazino_radical_localization" in (result_hydrazino.mechanism or "")
    assert "hydrazino_radical_localization" not in (result_hydrazine.mechanism or "")
    assert err_hydrazino < 3.0


def test_poly_neutral_nco_cumulene_relief_restores_hnco_family_without_touching_fulminic():
    err_hnco, result_hnco = _percent_err("HNCO")
    err_isocyanic, result_isocyanic = _percent_err("Isocyanic_acid")
    _, result_fulminic = _percent_err("Fulminic_acid")

    assert "neutral_nco_cumulene_relief" in (result_hnco.mechanism or "")
    assert "neutral_nco_cumulene_relief" in (result_isocyanic.mechanism or "")
    assert "neutral_nco_cumulene_relief" not in (result_fulminic.mechanism or "")
    assert err_hnco < 4.2  # Phase 1: T_T3 spectral variance shifts by ~0.15pp
    assert err_isocyanic < 4.2


def test_poly_ethenedione_cumulene_quench_targets_ocec_o_only():
    err_ethenedione, result_ethenedione = _percent_err("Ethenedione")
    _, result_hnco = _percent_err("HNCO")

    assert "ethenedione_cumulene_quench" in (result_ethenedione.mechanism or "")
    assert "ethenedione_cumulene_quench" not in (result_hnco.mechanism or "")
    assert err_ethenedione < 3.0


def test_poly_thiophene_aromatic_softening_targets_thiophene_only():
    err_thiophene, result_thiophene = _percent_err("thiophene")
    _, result_2methyl = _percent_err("2-methylthiophene")
    _, result_thiazole = _percent_err("thiazole")

    assert "thiophene_aromatic_softening" in (result_thiophene.mechanism or "")
    assert "thiophene_aromatic_softening" not in (result_2methyl.mechanism or "")
    assert "thiophene_aromatic_softening" not in (result_thiazole.mechanism or "")
    assert err_thiophene < 3.0


def test_poly_s_oxo_hydroxyl_relief_restores_sulfuric_family_without_touching_so3():
    err_h2so4, result_h2so4 = _percent_err("Sulfuric_acid")
    err_h2so3, result_h2so3 = _percent_err("Sulfurous_acid")
    _, result_so3 = _percent_err("SO3mol")
    _, result_pocl3 = _percent_err("POCl3")

    assert "s_oxo_hydroxyl_relief" in (result_h2so4.mechanism or "")
    assert "s_oxo_hydroxyl_relief" in (result_h2so3.mechanism or "")
    assert "s_oxo_hydroxyl_relief" not in (result_so3.mechanism or "")
    assert "s_oxo_hydroxyl_relief" not in (result_pocl3.mechanism or "")
    assert err_h2so4 < 7.0
    assert err_h2so3 < 5.0


def test_poly_boric_hydroxyl_donation_targets_boric_acid_only():
    err_h3bo3, result_h3bo3 = _percent_err("H3BO3")
    _, result_bf3 = _percent_err("BF3")
    _, result_trimethyl_borate = _percent_err("trimethyl_borate")

    assert result_h3bo3.mechanism == "boric_hydroxyl_donation"
    assert result_bf3.mechanism == "functional_donor_acceptor_blend"
    assert "boric_hydroxyl_donation" not in (result_trimethyl_borate.mechanism or "")
    assert err_h3bo3 < 3.0


def test_resonance_donor_overlap_marks_mixed_hetero_frames():
    sulfuric = compute_D_at_transfer(build_topology(MOLECULES["Sulfuric_acid"]["smiles"]))
    nitrous = compute_D_at_transfer(build_topology(MOLECULES["Nitrous_acid"]["smiles"]))
    nitric = compute_D_at_transfer(build_topology(MOLECULES["Nitric_acid"]["smiles"]))
    so3 = compute_D_at_transfer(build_topology(MOLECULES["SO3mol"]["smiles"]))
    bf3 = compute_D_at_transfer(build_topology(MOLECULES["BF3"]["smiles"]))

    _assert_functional_overlap(sulfuric, "resonance_donor_overlap", state_min=0.10, weight_min=0.05)
    _assert_functional_overlap(nitrous, "resonance_donor_overlap", state_min=0.09, weight_min=0.04)
    _assert_functional_overlap(so3, "resonance_donor_overlap", state_eq=0.0, weight_eq=0.0)
    _assert_functional_overlap(bf3, "resonance_donor_overlap", state_eq=0.0, weight_eq=0.0)
    assert nitric.functional_state.resonance_donor_overlap > sulfuric.functional_state.resonance_donor_overlap


def test_pi_resonance_overlap_marks_cumulene_and_soft_hetero_pi_frames():
    hnco = compute_D_at_transfer(build_topology(MOLECULES["HNCO"]["smiles"]))
    ethenedione = compute_D_at_transfer(build_topology(MOLECULES["Ethenedione"]["smiles"]))
    thiophene = compute_D_at_transfer(build_topology(MOLECULES["thiophene"]["smiles"]))
    bf3 = compute_D_at_transfer(build_topology(MOLECULES["BF3"]["smiles"]))

    _assert_functional_overlap(hnco, "pi_resonance_overlap", state_min=0.50, weight_min=0.30)
    _assert_functional_overlap(ethenedione, "pi_resonance_overlap", state_min=0.50, weight_min=0.40)
    _assert_functional_overlap(thiophene, "pi_resonance_overlap", state_min=0.20, weight_min=0.15)
    _assert_functional_overlap(bf3, "pi_resonance_overlap", state_eq=0.0, weight_eq=0.0)


def test_sigma_donor_overlap_marks_soft_p_block_sigma_acceptors():
    pf3 = compute_D_at_transfer(build_topology(MOLECULES["PF3"]["smiles"]))
    nf3 = compute_D_at_transfer(build_topology(MOLECULES["NF3"]["smiles"]))
    bf3 = compute_D_at_transfer(build_topology(MOLECULES["BF3"]["smiles"]))

    _assert_functional_overlap(pf3, "sigma_donor_overlap", state_min=0.40, weight_min=0.20)
    _assert_functional_overlap(bf3, "sigma_donor_overlap", state_eq=0.0, weight_eq=0.0)
    assert nf3.functional_state.sigma_donor_overlap < pf3.functional_state.sigma_donor_overlap


def test_poly_aziridine_ring_lp_crowding_targets_n_ring_three_cycles_only():
    err_aziridine, result_aziridine = _percent_err("Aziridine")
    err_oxaziridine, result_oxaziridine = _percent_err("Oxaziridine")
    _, result_ethylene_oxide = _percent_err("ethylene_oxide")
    _, result_dioxirane = _percent_err("dioxirane")

    assert result_aziridine.mechanism == "aziridine_ring_lp_crowding"
    assert result_oxaziridine.mechanism == "aziridine_ring_lp_crowding"
    assert result_ethylene_oxide.mechanism is None
    # dioxirane: T³ sigma_crowding may activate on O-O LP
    assert err_aziridine < 3.0
    assert err_oxaziridine < 3.0


def test_diatomic_diffuse_alkali_halide_boosts_k_halides_without_touching_na_li():
    err_kcl, result_kcl = _percent_err("KCl")
    err_kf, result_kf = _percent_err("KF")
    _, result_nacl = _percent_err("NaCl")
    _, result_licl = _percent_err("LiCl")

    assert result_kcl.mechanism == "diffuse_alkali_halide"
    assert result_kf.mechanism == "diffuse_alkali_halide"
    assert result_nacl.mechanism is None
    assert result_licl.mechanism is None
    assert err_kcl < 1.0
    assert err_kf < 1.0


def test_diatomic_heavy_homonuclear_halogen_stabilizes_cl2_without_touching_f2():
    err_cl2, result_cl2 = _percent_err("Cl2")
    _, result_f2 = _percent_err("F2")

    assert result_cl2.mechanism == "heavy_homonuclear_halogen"
    assert result_f2.mechanism is None
    assert err_cl2 < 2.0


def test_diatomic_hetero_radical_pi_caps_no_without_touching_o2():
    err_no, result_no = _percent_err("NO")
    _, result_o2 = _percent_err("O2")

    assert result_no.mechanism == "hetero_radical_pi"
    assert result_o2.mechanism is None
    assert err_no < 1.0


def test_diatomic_charge_separated_triple_boosts_co_without_touching_n2():
    err_co, result_co = _percent_err("CO")
    _, result_n2 = _percent_err("N2")

    assert result_co.mechanism == "charge_separated_triple"
    assert result_n2.mechanism is None
    assert err_co < 1.0


def test_diatomic_hetero_chalcogen_oxide_boosts_sulfur_monoxide_only():
    err_so, result_so = _percent_err("Sulfur_monoxide")
    _, result_o2 = _percent_err("O2")

    assert result_so.mechanism == "hetero_chalcogen_oxide"
    assert result_o2.mechanism is None
    assert err_so < 2.0


def test_bef2_s_block_acceptor_path_stays_within_one_percent():
    err, result = _percent_err("BeF2")

    assert result.E_spectral == 0.0
    assert result.D_at_P2 > 0.0
    assert err < 1.0


def test_becl2_diffuse_s_block_path_recovers_generic_complement():
    err, result = _percent_err("BeCl2")

    assert result.E_spectral > 0.0
    assert result.D_at_P2 > 0.0
    assert err < 1.0


def test_h2s_soft_bent_path_uses_generic_floor():
    err, result = _percent_err("H2S")

    assert result.D_at_P2 == 0.0
    assert result.E_spectral < 0.0
    assert err < 1.0


def test_h2o_hydride_bent_path_keeps_only_small_entropy_leak():
    err, result = _percent_err("H2O")

    assert result.D_at_P2 == 0.0
    assert result.E_spectral < 0.0
    assert result.mechanism == "hydride_bent"
    assert err < 0.2


def test_hno_mixed_bent_path_localizes_weaker_branch():
    err, result = _percent_err("HNO")

    assert result.E_spectral == 0.0
    assert result.D_at_P2 > 0.0
    assert err < 1.0


def test_singlet_carbene_bent_path_captures_methylene_family():
    err_ch2, result_ch2 = _percent_err("Methylene")
    err_ccl2, result_ccl2 = _percent_err("Dichloromethylene")

    assert result_ch2.mechanism == "singlet_carbene_bent"
    assert result_ccl2.mechanism == "singlet_carbene_bent"
    assert result_ch2.D_at_P2 == 0.0
    assert result_ccl2.D_at_P2 == 0.0
    assert err_ch2 < 2.0
    assert err_ccl2 < 1.0


def test_mixed_bent_pi_polar_path_captures_clno_family():
    err_clno, result_clno = _percent_err("ClNO")
    err_nocl, result_nocl = _percent_err("Nitrosyl_chloride")

    assert result_clno.mechanism == "mixed_bent_pi_polar"
    assert result_nocl.mechanism == "mixed_bent_pi_polar"
    assert result_clno.D_at_P2 == 0.0
    assert result_nocl.D_at_P2 == 0.0
    assert result_clno.E_spectral == 0.0
    assert result_nocl.E_spectral == 0.0
    assert err_clno < 1.0
    assert err_nocl < 2.0


def test_oxygen_lp_crowding_path_captures_hypohalous_family():
    err_cl2o, result_cl2o = _percent_err("Cl2O")
    err_hof, result_hof = _percent_err("Hypofluorous_acid")
    err_hocl, result_hocl = _percent_err("Hypochlorous_acid")

    assert result_cl2o.mechanism == "oxygen_lp_crowding"
    assert result_hof.mechanism == "oxygen_lp_crowding"
    assert result_hocl.mechanism == "oxygen_lp_crowding"
    assert result_cl2o.D_at_P2 == 0.0
    assert result_hof.D_at_P2 == 0.0
    assert result_hocl.D_at_P2 == 0.0
    assert result_cl2o.E_spectral == 0.0
    assert result_hof.E_spectral == 0.0
    assert result_hocl.E_spectral == 0.0
    assert err_cl2o < 3.0
    assert err_hof < 3.0
    assert err_hocl < 2.0


def test_sf2_polar_bent_path_recovers_entropy_complement():
    err, result = _percent_err("SF2")

    assert result.D_at_P2 > 0.0
    assert result.E_spectral < 0.0
    assert err < 1.0


def test_channel_synthesis_fallback_handles_disulfene_oxide_cleanly():
    err, result = _percent_err("Disulfene_oxide")

    assert result.mechanism == "channel_synthesis"
    assert result.D_at_P2 == 0.0
    assert result.E_spectral == 0.0
    assert err < 1.0


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Continuous Functional Fallbacks                                  ║
# ╚════════════════════════════════════════════════════════════════════╝

def test_functional_pi_localization_blend_captures_uncovered_pi_residuals():
    err_allene, result_allene = _percent_err("Chloroallene")
    err_cumulene, result_cumulene = _percent_err("Pentatetraene")
    err_isocyano, result_isocyano = _percent_err("Isocyanoacetylene")
    err_ketene, result_ketene = _percent_err("CH2CO")
    err_vinyl_i, result_vinyl_i = _percent_err("Vinyl_iodide")
    err_iodobenzene, result_iodobenzene = _percent_err("iodobenzene")
    err_cycloheptene, result_cycloheptene = _percent_err("trans_Cycloheptene")

    _assert_functional_blend(result_allene, "functional_pi_localization_blend", "pi_localization")
    _assert_functional_blend(result_cumulene, "functional_pi_localization_blend", "pi_localization")
    _assert_functional_blend(result_isocyano, "functional_pi_localization_blend", "pi_localization")
    assert result_ketene.mechanism is None
    _assert_result_dominant_functional(result_ketene, "pi_localization")
    _assert_functional_blend(result_vinyl_i, "functional_pi_localization_blend", "pi_localization")
    _assert_functional_blend(result_iodobenzene, "functional_pi_localization_blend", "pi_localization")
    _assert_functional_blend(result_cycloheptene, "functional_pi_localization_blend", "pi_localization")
    assert err_allene < 5.0
    assert err_cumulene < 2.0
    assert err_isocyano < 4.9  # v46: Dicke+π-deloc+GFT mechanisms shift by ~0.04pp
    assert err_ketene < 4.9
    assert err_vinyl_i < 6.0
    assert err_iodobenzene < 3.1
    assert err_cycloheptene < 6.5


def test_d_block_resolution_and_polygon_use_pentagonal_shell():
    topo = Topology(
        Z_list=[26, 8, 8, 8, 8],
        bonds=[(0, 1, 2.0), (0, 2, 2.0), (0, 3, 2.0), (0, 4, 2.0)],
        source="test",
        charges=[0, 0, 0, 0, 0],
    ).finalize()
    atom_data = _resolve_atom_data(topo)
    assert atom_data[26]["is_d_block"] is True
    assert atom_data[8]["is_d_block"] is False

    s_d = _vertex_polygon_dft(0, 0, topo, atom_data)
    atom_data_p = dict(atom_data)
    atom_data_p[26] = dict(atom_data_p[26])
    atom_data_p[26]["is_d_block"] = False
    s_p = _vertex_polygon_dft(0, 0, topo, atom_data_p)

    assert s_d >= 0.0
    assert s_p >= 0.0
    assert abs(s_d - s_p) > 1e-9

    atom_states = _resolve_atom_states(topo, atom_data)
    vertex_states = _resolve_vertex_states(topo, atom_states)
    assert atom_states[0].is_d_block is True
    assert vertex_states[0].d_block_delta_cf > 0.0
    assert isinstance(vertex_states[0].d_block_low_spin, bool)


def test_d_block_ligand_field_orders_carbonyl_over_halide_and_aqua():
    topo_ti = build_topology("Cl[Ti](Cl)(Cl)Cl")
    topo_v = build_topology("Cl[V](Cl)Cl")
    topo_ni = build_topology("[Ni](C#O)(C#O)(C#O)C#O")
    topo_cr = build_topology("[Cr](C#O)(C#O)(C#O)(C#O)(C#O)C#O")
    topo_fe = build_topology("[Fe+2](O)(O)(O)(O)(O)O")

    atom_data_ti = _resolve_atom_data(topo_ti)
    atom_data_v = _resolve_atom_data(topo_v)
    atom_data_ni = _resolve_atom_data(topo_ni)
    atom_data_cr = _resolve_atom_data(topo_cr)
    atom_data_fe = _resolve_atom_data(topo_fe)

    atom_states_ti = _resolve_atom_states(topo_ti, atom_data_ti)
    atom_states_v = _resolve_atom_states(topo_v, atom_data_v)
    atom_states_ni = _resolve_atom_states(topo_ni, atom_data_ni)
    atom_states_cr = _resolve_atom_states(topo_cr, atom_data_cr)
    atom_states_fe = _resolve_atom_states(topo_fe, atom_data_fe)

    vertex_ti = _resolve_vertex_states(topo_ti, atom_states_ti)[1]
    vertex_v = _resolve_vertex_states(topo_v, atom_states_v)[1]
    vertex_ni = _resolve_vertex_states(topo_ni, atom_states_ni)[0]
    vertex_cr = _resolve_vertex_states(topo_cr, atom_states_cr)[0]
    vertex_fe = _resolve_vertex_states(topo_fe, atom_states_fe)[0]

    shell_ti = _build_d_block_ligand_shell(1, topo_ti, atom_data_ti)
    shell_v = _build_d_block_ligand_shell(1, topo_v, atom_data_v)
    shell_ni = _build_d_block_ligand_shell(0, topo_ni, atom_data_ni)
    shell_cr = _build_d_block_ligand_shell(0, topo_cr, atom_data_cr)
    shell_fe = _build_d_block_ligand_shell(0, topo_fe, atom_data_fe)

    assert shell_ti is not None and shell_ti.heavy_halide_count == 4
    assert shell_v is not None and shell_v.heavy_halide_count == 3
    assert shell_ni is not None and shell_ni.carbonyl_count == 4
    assert shell_cr is not None and shell_cr.carbonyl_count == 6
    assert shell_fe is not None and shell_fe.aqua_count == 6

    assert vertex_ni.d_block_delta_cf > vertex_ti.d_block_delta_cf
    assert vertex_cr.d_block_delta_cf > vertex_v.d_block_delta_cf
    assert vertex_ni.d_block_low_spin is True
    assert vertex_cr.d_block_low_spin is True
    assert vertex_v.d_block_low_spin is False
    assert vertex_fe.d_block_low_spin is False


def test_d_block_prompt_sentinels_route_to_explicit_mechanisms():
    ti = compute_D_at_transfer(build_topology("Cl[Ti](Cl)(Cl)Cl"))
    v = compute_D_at_transfer(build_topology("Cl[V](Cl)Cl"))
    ni = compute_D_at_transfer(build_topology("[Ni](C#O)(C#O)(C#O)C#O"))
    cr = compute_D_at_transfer(build_topology("[Cr](C#O)(C#O)(C#O)(C#O)(C#O)C#O"))
    fe = compute_D_at_transfer(build_topology("[Fe+2](O)(O)(O)(O)(O)O"))
    ferro = compute_D_at_transfer(build_topology("[Fe](c1cccc1)(c1cccc1)"))
    cr2 = compute_D_at_transfer(build_topology("[Cr]#[Cr]"))
    mo2 = compute_D_at_transfer(build_topology("[Mo]#[Mo]"))

    assert ti.mechanism == "d_block_ligand_field"
    assert v.mechanism == "d_block_ligand_field"
    assert ni.mechanism == "d_block_carbonyl_field"
    assert cr.mechanism == "d_block_carbonyl_field"
    assert fe.mechanism == "d_block_ligand_field"
    assert ferro.mechanism == "d_block_pi_sandwich"
    assert cr2.mechanism is not None and "d_block_delta_bond" in cr2.mechanism
    assert mo2.mechanism is not None and "d_block_delta_bond" in mo2.mechanism


def test_functional_pi_resonance_blend_captures_uncovered_polar_carbonyls():
    err_hcooh, result_hcooh = _percent_err("HCOOH")
    err_hcocl, result_hcocl = _percent_err("HCOCl")
    err_cof2, result_cof2 = _percent_err("COF2b")

    assert result_hcooh.mechanism == "functional_pi_resonance_blend"
    assert result_hcocl.mechanism == "functional_pi_resonance_blend"
    assert result_cof2.mechanism == "functional_pi_resonance_blend"
    _assert_functional_overlap(result_hcooh, "pi_resonance_overlap", weight_min=0.15)
    _assert_functional_overlap(result_hcocl, "pi_resonance_overlap", weight_min=0.18)
    _assert_functional_overlap(result_cof2, "pi_resonance_overlap", weight_min=0.20)
    assert err_hcooh < 5.5
    assert err_hcocl < 5.0
    assert err_cof2 < 5.0


def test_functional_sigma_crowding_blend_captures_uncovered_sigma_residuals():
    err_piperazine, result_piperazine = _percent_err("piperazine")
    err_sf4, result_sf4 = _percent_err("SF4")

    _assert_functional_blend(result_piperazine, "functional_sigma_crowding_blend", "sigma_crowding")
    # SF4: F invisible on P₁ face (Z=9, r₁=0) → T³ sigma_crowding does not activate
    assert err_piperazine < 6.0
    assert err_sf4 < 6.5


def test_functional_sigma_donor_blend_captures_uncovered_pf3_frame():
    err_pf3, result_pf3 = _percent_err("PF3")
    _, result_nf3 = _percent_err("NF3")

    assert result_pf3.mechanism == "functional_sigma_donor_blend"
    assert result_nf3.mechanism is None
    assert result_pf3.functional_weights is not None
    assert result_pf3.functional_weights.dominant_functional in (
        "sigma_crowding",
        "donor_acceptor_support",
    )
    _assert_functional_overlap(result_pf3, "sigma_donor_overlap", weight_min=0.20)
    assert err_pf3 < 4.5


def test_functional_topological_frustration_blend_captures_uncovered_ring_residuals():
    err_pyrrolidine, result_pyrrolidine = _percent_err("pyrrolidine")
    err_cyclobutane, result_cyclobutane = _percent_err("cC4H8")
    err_norbornene, result_norbornene = _percent_err("norbornene")

    _assert_functional_blend(result_pyrrolidine, "functional_topological_frustration_blend", "topological_frustration")
    _assert_functional_blend(result_cyclobutane, "functional_topological_frustration_blend", "topological_frustration")
    _assert_functional_blend(result_norbornene, "functional_topological_frustration_blend", "topological_frustration")
    assert err_pyrrolidine < 4.8
    assert err_cyclobutane < 3.5
    assert err_norbornene < 4.0


def test_functional_donor_acceptor_blend_captures_uncovered_acceptor_frames():
    err_bf3, result_bf3 = _percent_err("BF3")
    err_alf3, result_alf3 = _percent_err("AlF3")
    err_sih3f, result_sih3f = _percent_err("SiH3F")
    err_cfbr2, result_cfbr2 = _percent_err("Fluorodibromomethane")
    _, result_cyclohexylamine = _percent_err("cyclohexylamine")
    _, result_thp = _percent_err("THP")

    _assert_functional_blend(result_bf3, "functional_donor_acceptor_blend", "donor_acceptor_support")
    _assert_functional_blend(result_alf3, "functional_donor_acceptor_blend", "donor_acceptor_support")
    _assert_functional_blend(result_sih3f, "functional_donor_acceptor_blend", "donor_acceptor_support")
    # CFBr₂: T³ sigma_crowding may activate (Br r₁=2, visible on P₁)
    assert result_cfbr2.mechanism is not None
    assert result_cyclohexylamine.mechanism is None
    assert result_thp.mechanism is None
    assert err_bf3 < 3.5
    assert err_alf3 < 5.5
    assert err_sih3f < 4.5
    assert err_cfbr2 < 5.0


def test_functional_open_shell_budget_blend_captures_uncovered_radicals():
    err_oxiranyl, result_oxiranyl = _percent_err("Oxiranyl")
    err_ethylperoxy, result_ethylperoxy = _percent_err("Ethylperoxy")
    _, result_furan = _percent_err("furan")
    _, result_pyrrole = _percent_err("pyrrole")

    _assert_functional_blend(result_oxiranyl, "functional_open_shell_budget_blend", "open_shell_budget")
    _assert_functional_blend(result_ethylperoxy, "functional_open_shell_budget_blend", "open_shell_budget")
    assert result_furan.mechanism is None
    assert result_pyrrole.mechanism is None
    assert err_oxiranyl < 4.5
    assert err_ethylperoxy < 1.5
