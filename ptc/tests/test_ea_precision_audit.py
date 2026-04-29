"""Tests for the canonical EA precision audit helpers."""
from __future__ import annotations

from ptc.ea_precision_audit import (
    benchmark_ea_precision,
    precision_ladder,
    small_atom_precision_rows,
)


def test_canonical_ea_precision_global_score_is_locked():
    report = benchmark_ea_precision("canonical")
    assert report["count"] == 73
    assert report["mae_percent"] < 0.99
    assert report["mae_eV"] < 0.0069
    assert report["max_percent"] < 3.54


def test_canonical_alias_matches_contact_depth_operator():
    canonical = benchmark_ea_precision("canonical")
    contact = benchmark_ea_precision("contact_depth_operator")
    continuous = benchmark_ea_precision("continuous_channel")
    assert canonical["mae_percent"] == contact["mae_percent"]
    assert canonical["mae_eV"] == contact["mae_eV"]
    assert canonical["mae_percent"] < continuous["mae_percent"]


def test_small_atom_precision_rows_cover_first_two_periods():
    rows = small_atom_precision_rows()
    assert [row.symbol for row in rows] == [
        "H",
        "Li",
        "B",
        "C",
        "O",
        "F",
        "Na",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
    ]
    # The residual small-atom precision target is now in the tens of meV,
    # which is exactly where finite-mass/relativistic/QED layers become visible.
    assert max(row.abs_error_eV for row in rows) < 0.026


def test_precision_ladder_tracks_ab_initio_layers():
    layers = {item["layer"] for item in precision_ladder()}
    assert {
        "correlation",
        "finite_mass",
        "relativistic_scalar",
        "fine_structure",
        "qed",
    } <= layers
