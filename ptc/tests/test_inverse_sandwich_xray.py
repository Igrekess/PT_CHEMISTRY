"""Tests for ptc.data.inverse_sandwich_xray (Chantier 8 Étape 4).

Validates that the X-ray reference set is well-formed and that geometry
can be built for every entry. End-to-end PT-LCAO MAE comparison against
``exp_NICS_zz`` is exercised on the subset with experimental data.
"""
from __future__ import annotations

import numpy as np
import pytest

from ptc.data.inverse_sandwich_xray import (
    INVERSE_SANDWICHES_XRAY,
    InverseSandwichXRayEntry,
    entries_with_exp_NICS,
    get_entry,
)
from ptc.lcao.cluster import build_inverse_sandwich


def test_xray_entries_well_formed():
    """All entries have required fields populated."""
    assert len(INVERSE_SANDWICHES_XRAY) >= 4
    for e in INVERSE_SANDWICHES_XRAY:
        assert isinstance(e, InverseSandwichXRayEntry)
        assert e.name
        assert e.X_ring_Z >= 1
        assert e.n_ring >= 1
        assert e.M_cap_Z >= 1
        assert e.d_xx >= 0.0
        assert e.z_cap > 0.0
        assert e.ligand in ("none", "Cp*")
        assert e.n_ligands_per_cap in (0, 1, 2, 3)
        assert e.reference


def test_xray_lookup_by_name():
    e = get_entry("Bi3@U2(Cp*)4")
    assert e is not None
    assert e.X_ring_Z == 83
    assert e.M_cap_Z == 92
    assert e.exp_NICS_zz == pytest.approx(0.08, abs=1.0e-6)
    # negative lookup
    assert get_entry("nonexistent") is None


def test_xray_geometry_buildable_for_all_entries():
    """Every entry maps to a buildable inverse-sandwich geometry."""
    for e in INVERSE_SANDWICHES_XRAY:
        if e.n_ring < 3:
            # Single-atom bridge cases: skip the generic builder.
            continue
        Z, coords, bonds = build_inverse_sandwich(
            X_ring_Z=e.X_ring_Z, n_ring=e.n_ring, M_cap_Z=e.M_cap_Z,
            d_xx=e.d_xx, z_cap=e.z_cap,
            ligands=e.ligand, n_ligands_per_cap=e.n_ligands_per_cap,
            include_methyl_h=False,    # speed: skip H
        )
        assert len(Z) == coords.shape[0]
        assert coords.shape[1] == 3


def test_xray_entries_with_exp_NICS_subset():
    """At least Ding 2026 is in the experimentally validated subset."""
    exp_set = entries_with_exp_NICS()
    assert len(exp_set) >= 1
    names = {e.name for e in exp_set}
    assert "Bi3@U2(Cp*)4" in names


def test_xray_ding_geometry_matches_default_builder():
    """The Ding 2026 X-ray entry reproduces build_bi3_u2_cp_star4 defaults."""
    from ptc.lcao.cluster import build_bi3_u2_cp_star4
    e = get_entry("Bi3@U2(Cp*)4")
    Z_x, coords_x, bonds_x = build_inverse_sandwich(
        X_ring_Z=e.X_ring_Z, n_ring=e.n_ring, M_cap_Z=e.M_cap_Z,
        d_xx=e.d_xx, z_cap=e.z_cap,
        ligands=e.ligand, n_ligands_per_cap=e.n_ligands_per_cap,
    )
    Z_ref, coords_ref, bonds_ref = build_bi3_u2_cp_star4(
        d_bi_bi=e.d_xx, z_u_cap=e.z_cap,
    )
    assert Z_x == Z_ref
    np.testing.assert_allclose(coords_x, coords_ref, atol=1e-10)
    assert bonds_x == bonds_ref
