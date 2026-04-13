"""Tests for the hierarchical PT EA operator prototype."""

from ptc.constants import D5, D7, S3, S5, S_HALF
from ptc.ea_operator import (
    EA_operator_eV,
    atomic_capture_state,
    benchmark_ea_operator_against_nist,
    build_atomic_hierarchical_ea_operator,
    build_hierarchical_ea_operator,
    operator_capture_amplitude,
    closure_pull,
    creation_coupling,
    exchange_count,
    koide_capture_envelope,
)


def test_koide_envelope_has_optimum_at_two_thirds_for_p_block():
    values = {n: koide_capture_envelope(n, 6) for n in range(1, 6)}
    assert values[4] > values[5]
    assert values[4] > values[3]
    assert values[4] > values[2]
    assert values[4] > values[1]


def test_exchange_count_generalizes_hund_pattern():
    assert exchange_count(1, 3) == 1
    assert exchange_count(3, 3) == -1
    assert exchange_count(4, 3) == 0
    assert exchange_count(5, 3) == 1
    assert exchange_count(5, 5) == -1
    assert exchange_count(6, 5) == 0
    assert exchange_count(7, 5) == 1


def test_closure_pull_distinguishes_one_and_two_vacancies():
    assert closure_pull(3) == 1.0
    assert closure_pull(2) == 1.0 / S_HALF
    assert closure_pull(1) == 1.0 / (S_HALF ** 2)
    assert closure_pull(0) == 0.0


def test_creation_couplings_follow_pt_channel_hierarchy():
    assert creation_coupling("s") > 0.0
    assert creation_coupling("p") > 0.0
    assert creation_coupling("d") > 0.0
    assert creation_coupling("s") > creation_coupling("p")
    assert creation_coupling("p") > creation_coupling("d")


def test_hydrogenic_s_branch_uses_special_capture_gain():
    op = build_hierarchical_ea_operator(1, hydrogenic=True)
    s_layer = op.layers[0]
    assert abs(s_layer.capture_gain - (S3 * (S_HALF ** 2))) < 1e-12
    assert s_layer.structurally_open is True


def test_p_layer_stays_closed_until_s_is_saturated():
    blocked = build_hierarchical_ea_operator(1, 4, 0, 0)
    opened = build_hierarchical_ea_operator(2, 4, 0, 0)
    assert blocked.layers[1].structurally_open is False
    assert blocked.layers[1].capture_gain == 0.0
    assert opened.layers[1].structurally_open is True
    assert opened.layers[1].capture_gain > 0.0


def test_saturation_of_s_opens_p_with_nonzero_coupling():
    op = build_hierarchical_ea_operator(2, 0, 0, 0)
    assert op.matrix[0][1] > 0.0
    assert op.couplings["s->p"] == op.matrix[0][1]


def test_saturation_of_p_opens_d_and_stacks_previous_memory():
    op = build_hierarchical_ea_operator(2, 6, 3, 0)
    assert op.matrix[0][1] > 0.0
    assert op.matrix[1][2] > 0.0
    assert op.layers[2].structurally_open is True
    assert op.layers[2].capture_gain > 0.0
    assert op.layers[2].capture_gain > D5 * koide_capture_envelope(3, 10)


def test_saturation_of_d_opens_f():
    op = build_hierarchical_ea_operator(2, 6, 10, 6)
    assert op.matrix[2][3] > 0.0
    assert op.layers[3].structurally_open is True
    assert op.layers[3].capture_gain > 0.0


def test_stack_gain_grows_with_saturation_cascade():
    op_s = build_hierarchical_ea_operator(2, 0, 0, 0)
    op_sp = build_hierarchical_ea_operator(2, 6, 0, 0)
    op_spd = build_hierarchical_ea_operator(2, 6, 10, 0)
    assert op_s.stack_gain > 1.0
    assert op_sp.stack_gain > op_s.stack_gain
    assert op_spd.stack_gain > op_sp.stack_gain


def test_non_s_blocks_use_operator_gains_only_when_open():
    op = build_hierarchical_ea_operator(2, 0, 4, 0)
    assert op.layers[2].structurally_open is False
    assert op.layers[2].capture_gain == 0.0


def test_atomic_capture_state_uses_saturated_prior_layers():
    cl = atomic_capture_state(17)
    fe = atomic_capture_state(26)
    nd = atomic_capture_state(60)
    assert cl.active_layer == "p"
    assert cl.occupancies == (2, 5, 0, 0)
    assert fe.active_layer == "d"
    assert fe.occupancies == (2, 6, 6, 0)
    assert nd.active_layer == "f"
    assert nd.occupancies == (2, 6, 10, 4)


def test_atomic_operator_matches_manual_builder():
    auto = build_atomic_hierarchical_ea_operator(17)
    manual = build_hierarchical_ea_operator(2, 5, 0, 0)
    assert auto.matrix == manual.matrix
    assert auto.stack_gain == manual.stack_gain


def test_operator_capture_amplitude_reads_active_layer_gain():
    amp = operator_capture_amplitude(17, projection="active_diagonal")
    op = build_atomic_hierarchical_ea_operator(17)
    assert abs(amp - op.layers[1].capture_gain) < 1e-12


def test_ea_operator_eV_is_nonnegative_for_representative_atoms():
    assert 0.0 < EA_operator_eV(1) < 2.0
    assert EA_operator_eV(6) > EA_operator_eV(1)
    assert EA_operator_eV(9) > EA_operator_eV(6)
    assert EA_operator_eV(10) == 0.0


def test_s_block_ns1_tracks_alkalis_reasonably():
    assert abs(EA_operator_eV(3) - 0.618) < 0.05
    assert abs(EA_operator_eV(11) - 0.548) < 0.03
    assert abs(EA_operator_eV(19) - 0.502) < 0.05
    assert abs(EA_operator_eV(37) - 0.486) < 0.05
    assert abs(EA_operator_eV(55) - 0.472) < 0.06


def test_closed_shell_ns2_virtual_branch_is_zero_for_be_mg_and_tracks_ca_sr_ba():
    assert EA_operator_eV(4) == 0.0
    assert EA_operator_eV(12) == 0.0
    assert abs(EA_operator_eV(20) - 0.025) < 0.01
    assert abs(EA_operator_eV(38) - 0.052) < 0.02
    assert abs(EA_operator_eV(56) - 0.145) < 0.02


def test_boundary_flux_projection_is_smaller_than_active_diagonal_for_p_block():
    active = operator_capture_amplitude(17, projection="active_diagonal")
    boundary = operator_capture_amplitude(17, projection="boundary_flux")
    assert boundary > 0.0
    assert boundary < active


def test_boundary_flux_projection_improves_global_mae_over_active_diagonal():
    baseline = benchmark_ea_operator_against_nist(projection="active_diagonal")
    boundary = benchmark_ea_operator_against_nist(projection="boundary_flux")
    assert baseline["count"] == boundary["count"]
    assert boundary["mae_percent"] < baseline["mae_percent"]


def test_heavy_p_2_plus_4_split_tracks_tl_to_at_reasonably():
    assert abs(EA_operator_eV(81) - 0.377) < 0.05
    assert abs(EA_operator_eV(82) - 0.364) < 0.05
    assert abs(EA_operator_eV(83) - 0.946) < 0.10
    assert abs(EA_operator_eV(84) - 1.900) < 0.25
    assert abs(EA_operator_eV(85) - 2.416) < 0.10


def test_light_p_hexagon_projection_tracks_period2_to5_reasonably():
    assert abs(EA_operator_eV(5) - 0.277) < 0.02
    assert abs(EA_operator_eV(13) - 0.433) < 0.03
    assert abs(EA_operator_eV(14) - 1.385) < 0.10
    assert abs(EA_operator_eV(15) - 0.746) < 0.04
    assert abs(EA_operator_eV(17) - 3.613) < 0.35
    assert abs(EA_operator_eV(31) - 0.430) < 0.03
    assert abs(EA_operator_eV(33) - 0.804) < 0.04
    assert abs(EA_operator_eV(51) - 1.047) < 0.05
    assert abs(EA_operator_eV(35) - 3.365) < 0.25


def test_heavy_p_split_improves_p_block_benchmark():
    boundary = benchmark_ea_operator_against_nist(projection="boundary_flux")
    assert boundary["by_block"]["p"] < 6.5


def test_d_block_keeps_positive_la_and_good_hf_ta_values():
    assert abs(EA_operator_eV(57) - 0.470) < 0.12
    assert abs(EA_operator_eV(72) - 0.178) < 0.05
    assert abs(EA_operator_eV(73) - 0.322) < 0.05


def test_d_block_birth_states_track_sc_y_lu_reasonably():
    assert abs(EA_operator_eV(21) - 0.188) < 0.06
    assert abs(EA_operator_eV(39) - 0.307) < 0.05
    assert abs(EA_operator_eV(71) - 0.346) < 0.06


def test_d_block_second_vertex_frustration_is_softened_for_ti():
    assert abs(EA_operator_eV(22) - 0.079) < 0.04


def test_d_block_pentagon_projectors_track_4d_and_5d_resonance_family():
    assert abs(EA_operator_eV(44) - 1.050) < 0.05
    assert abs(EA_operator_eV(46) - 0.557) < 0.05
    assert abs(EA_operator_eV(78) - 2.128) < 0.08
    assert abs(EA_operator_eV(79) - 2.309) < 0.13


def test_d_block_penalizes_3d_frustration_points_d2_and_d6():
    assert EA_operator_eV(22) < EA_operator_eV(21)
    assert EA_operator_eV(26) < EA_operator_eV(27)


def test_d_block_improves_benchmark_and_global_boundary_flux():
    boundary = benchmark_ea_operator_against_nist(projection="boundary_flux")
    assert boundary["by_block"]["d"] < 10.0
    assert boundary["by_block"]["s"] < 8.0
    assert boundary["mae_percent"] < 10.5


def test_f_block_keeps_heptagon_structure_and_near_zero_yb():
    assert EA_operator_eV(60) > EA_operator_eV(61)
    assert EA_operator_eV(63) > EA_operator_eV(62)
    assert EA_operator_eV(65) > EA_operator_eV(64)
    assert EA_operator_eV(69) > EA_operator_eV(68)
    assert EA_operator_eV(70) < 0.03


def test_lanthanide_4f_projector_tracks_ce_pr_nd_eu_tb_tm():
    assert abs(EA_operator_eV(58) - 0.650) < 0.03
    assert abs(EA_operator_eV(59) - 0.962) < 0.04
    assert abs(EA_operator_eV(60) - 1.916) < 0.08
    assert abs(EA_operator_eV(63) - 0.864) < 0.12
    assert abs(EA_operator_eV(65) - 1.165) < 0.03
    assert abs(EA_operator_eV(69) - 1.029) < 0.12


def test_f_block_improves_benchmark_and_global_boundary_flux():
    boundary = benchmark_ea_operator_against_nist(projection="boundary_flux")
    assert boundary["by_block"]["f"] < 11.5
    assert boundary["mae_percent"] < 8.6


def test_canonical_projection_matches_boundary_flux_on_current_nist_domain():
    boundary = benchmark_ea_operator_against_nist(projection="boundary_flux")
    canonical = benchmark_ea_operator_against_nist(projection="canonical")
    assert canonical["count"] == boundary["count"]
    assert abs(canonical["mae_percent"] - boundary["mae_percent"]) < 1e-12
    for block in canonical["by_block"]:
        assert abs(canonical["by_block"][block] - boundary["by_block"][block]) < 1e-12


def test_canonical_projection_adds_actinide_observable_correction():
    assert EA_operator_eV(92, projection="canonical") > EA_operator_eV(92, projection="boundary_flux")
    assert EA_operator_eV(95, projection="canonical") > EA_operator_eV(95, projection="boundary_flux")
    assert EA_operator_eV(101, projection="canonical") > EA_operator_eV(101, projection="boundary_flux")
