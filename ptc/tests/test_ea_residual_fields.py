"""Tests for exploratory EA residual fields."""
from __future__ import annotations

from ptc.ea_cpr_capture import benchmark_ea_cpr_capture
from ptc.ea_residual_fields import (
    benchmark_ea_residual_fields,
    compare_ea_residual_fields,
    continuous_channel_lambda,
    d_core_polarization_lambda,
    d_second_harmonic_lambda,
    f_edge_leak_lambda,
    light_p_center_lambda,
    p_center_pressure_support,
    p_completed_d_closure_depth,
    p_closure_polarization_lambda,
    p_threshold_lambda,
)


def test_p_threshold_field_is_p_local():
    assert p_threshold_lambda(53) != 0.0
    assert p_threshold_lambda(79) == 0.0


def test_p_closure_field_splits_p4_and_period5_p5():
    assert p_closure_polarization_lambda(52) != 0.0  # Te p4
    assert p_closure_polarization_lambda(53) != 0.0  # I p5, d10 core
    assert p_closure_polarization_lambda(35) == 0.0  # Br p5
    assert p_closure_polarization_lambda(85) == 0.0  # At p5


def test_d_core_field_is_d_local():
    assert d_core_polarization_lambda(79) != 0.0
    assert d_core_polarization_lambda(53) == 0.0


def test_continuous_channel_terms_are_channel_local():
    assert d_second_harmonic_lambda(75) != 0.0  # Re, d shell
    assert d_second_harmonic_lambda(53) == 0.0
    assert f_edge_leak_lambda(70) != 0.0  # Yb, f shell
    assert f_edge_leak_lambda(79) == 0.0
    assert light_p_center_lambda(15) != 0.0  # P, light p shell
    assert light_p_center_lambda(53) == 0.0  # I, heavier p shell


def test_p_center_pressure_support_stops_at_double_d_closure():
    assert p_completed_d_closure_depth(17) == 0  # Cl, no completed d closure
    assert p_completed_d_closure_depth(35) == 1  # Br, first d10 closure
    assert p_completed_d_closure_depth(53) == 2  # I, double d10 stack
    assert p_center_pressure_support(35) == 1.0
    assert p_center_pressure_support(53) == 0.0
    assert p_center_pressure_support(75) == 0.0


def test_continuous_channel_field_combines_fixed_pt_terms():
    assert continuous_channel_lambda(75) == d_second_harmonic_lambda(75)
    assert continuous_channel_lambda(70) == f_edge_leak_lambda(70)
    assert continuous_channel_lambda(15) == light_p_center_lambda(15)


def test_residual_fields_improve_boundary_continuum_mae():
    baseline = benchmark_ea_cpr_capture(
        mode="boundary_continuum",
        strength=5.0,
    )
    residual = benchmark_ea_residual_fields(mode="continuous_channel")
    contact = benchmark_ea_residual_fields(mode="contact_depth_operator")
    assert residual["mae_percent"] < baseline["mae_percent"]
    assert residual["mae_percent"] < 1.13
    assert contact["mae_percent"] < 0.99


def test_compare_ea_residual_fields_orders_variants():
    report = compare_ea_residual_fields()
    variants = report["variants"]
    assert len(variants) == 7
    assert variants[0]["mae_percent"] <= variants[-1]["mae_percent"]
    assert variants[0]["mode"] == "contact_depth_operator"
