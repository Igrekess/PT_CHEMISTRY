from ptc.atomic_precision_frontier import (
    apply_candidate,
    benchmark_residuals,
    depth_operator_minus_stack,
    frontier_candidates,
    residual_rows,
    top_residuals,
)


def test_current_atomic_precision_scores_are_locked():
    ie = benchmark_residuals("IE")
    ea = benchmark_residuals("EA")

    assert ie["count"] == 118
    assert ie["mae_percent"] < 0.046
    assert ea["count"] == 73
    assert ea["mae_percent"] < 0.99


def test_top_residuals_expose_expected_frontier_families():
    ie_top = top_residuals("IE", limit=4)
    ea_top = top_residuals("EA", limit=8)

    assert any(row.symbol == "Pm" and row.block == "f" for row in ie_top)
    assert any(row.symbol == "Pr" and row.block == "f" for row in ea_top)
    assert any(row.symbol == "Re" and row.block == "d" for row in ea_top)


def test_capture_coordinate_stays_inside_channel_for_closed_shells():
    yb = next(row for row in residual_rows("EA") if row.symbol == "Yb")

    assert yb.n == yb.capacity
    assert 0.0 < yb.capture_coordinate <= 1.0


def test_frontier_candidates_are_diagnostic_not_canonical():
    candidates = frontier_candidates()
    names = {candidate.name for candidate in candidates}

    assert "f_contact_center" in names
    assert "d6_second_harmonic_depth" in names
    assert "ea_contact_projector_stack" in names
    assert "ea_contact_depth_operator" in names
    assert all(
        candidate.status in {"DIAG", "DER-PHYS"} for candidate in candidates
    )


def test_remaining_d4_probe_is_only_subcanonical_after_activation():
    ea = benchmark_residuals("EA")
    candidate = next(
        item for item in frontier_candidates() if item.name == "d4_compact_core_phase"
    )
    report = apply_candidate("EA", candidate)

    assert report["mae_percent"] < ea["mae_percent"]
    assert ea["mae_percent"] - report["mae_percent"] < 0.001


def test_contact_projector_stack_is_already_in_canonical_engine():
    ea = benchmark_residuals("EA")
    candidate = next(
        item
        for item in frontier_candidates()
        if item.name == "ea_contact_projector_stack"
    )
    report = apply_candidate("EA", candidate)

    assert ea["mae_percent"] < 1.0
    assert report["mae_percent"] > ea["mae_percent"]


def test_depth_operator_samples_same_field_as_projector_stack():
    for row in residual_rows("EA"):
        assert abs(depth_operator_minus_stack(row)) < 1e-15


def test_depth_operator_is_already_in_canonical_engine():
    ea = benchmark_residuals("EA")
    candidate = next(
        item
        for item in frontier_candidates()
        if item.name == "ea_contact_depth_operator"
    )
    report = apply_candidate("EA", candidate)

    assert ea["mae_percent"] < 1.0
    assert report["mae_percent"] > ea["mae_percent"]
