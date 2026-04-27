"""Tests for ptc.lcao.cluster.build_inverse_sandwich (Chantier 8 Étape 1).

The generic builder must reproduce ``build_bi3_u2_cp_star4`` for the
canonical (X=Bi, n_ring=3, M=U, ligands='Cp*') case and accept new
combinations (Pb3@Th2, As4@U2 stripped, COT@U2 cas limite).
"""
from __future__ import annotations

import numpy as np
import pytest

from ptc.lcao.cluster import (
    build_bi3_u2_cp_star4,
    build_inverse_sandwich,
    build_explicit_cluster,
)


def test_inverse_sandwich_reproduces_bi3_u2_cp_star4():
    Z_ref, coords_ref, bonds_ref = build_bi3_u2_cp_star4()
    Z, coords, bonds = build_inverse_sandwich(
        X_ring_Z=83, n_ring=3, M_cap_Z=92,
        d_xx=3.05, z_cap=2.1,
        ligands="Cp*", n_ligands_per_cap=2,
        cp_cp_angle_deg=134.0, d_m_cp_centroid=2.5,
        include_methyl_h=True,
    )
    assert Z == Z_ref
    np.testing.assert_allclose(coords, coords_ref, atol=1e-10)
    assert bonds == bonds_ref


def test_inverse_sandwich_pb3_th2_cp_star4():
    Z, coords, bonds = build_inverse_sandwich(
        X_ring_Z=82, n_ring=3, M_cap_Z=90,
        ligands="Cp*", n_ligands_per_cap=2,
    )
    # 3 ring + 2 caps + 4 Cp*(15 atoms each: 5 ring C + 5 methyl C + 15 H)
    assert len(Z) == 3 + 2 + 4 * 25
    assert sum(z == 82 for z in Z) == 3
    assert sum(z == 90 for z in Z) == 2
    # D3h ring: equilateral triangle in xy plane
    ring = coords[:3]
    np.testing.assert_allclose(ring[:, 2], 0.0, atol=1e-10)
    sides = [
        np.linalg.norm(ring[i] - ring[(i + 1) % 3]) for i in range(3)
    ]
    assert max(sides) - min(sides) < 1e-10  # equilateral
    # caps on z-axis, symmetric
    np.testing.assert_allclose(coords[3, :2], 0.0, atol=1e-10)
    np.testing.assert_allclose(coords[4, :2], 0.0, atol=1e-10)
    assert coords[3, 2] == -coords[4, 2]


def test_inverse_sandwich_as4_u2_stripped():
    """4-ring, no ligand: D4h skeletal cluster (4 + 2 = 6 atoms)."""
    Z, coords, bonds = build_inverse_sandwich(
        X_ring_Z=33, n_ring=4, M_cap_Z=92,
        ligands="none",
    )
    assert len(Z) == 6
    assert Z[:4] == [33, 33, 33, 33]
    assert Z[4:] == [92, 92]
    # ring is a square in xy plane
    ring = coords[:4]
    np.testing.assert_allclose(ring[:, 2], 0.0, atol=1e-10)
    sides = [
        np.linalg.norm(ring[i] - ring[(i + 1) % 4]) for i in range(4)
    ]
    assert max(sides) - min(sides) < 1e-10
    diags = [
        np.linalg.norm(ring[0] - ring[2]),
        np.linalg.norm(ring[1] - ring[3]),
    ]
    assert abs(diags[0] - diags[1]) < 1e-10
    assert diags[0] > sides[0]  # diagonal > side
    # bonds: 4 ring + 2 caps × 4 ring = 12
    assert len(bonds) == 4 + 2 * 4


def test_inverse_sandwich_cot_u2_eight_ring():
    """COT-like 8-ring cas limite, no ligand (10 atoms total)."""
    Z, coords, bonds = build_inverse_sandwich(
        X_ring_Z=6, n_ring=8, M_cap_Z=92,
        d_xx=1.40,           # aromatic C-C
        z_cap=2.0,
        ligands="none",
    )
    assert len(Z) == 10
    assert Z[:8] == [6] * 8
    assert Z[8:] == [92, 92]
    # circumradius for octagon: d / (2 sin(pi/8))
    expected_R = 1.40 / (2.0 * np.sin(np.pi / 8))
    R_actual = np.linalg.norm(coords[0, :2])
    assert abs(R_actual - expected_R) < 1e-10


def test_inverse_sandwich_d_xx_default_uses_r_equilibrium():
    """When d_xx=None the ring distance comes from r_equilibrium."""
    from ptc.bond import r_equilibrium
    from ptc.periodic import period
    expected = r_equilibrium(period(15), period(15), bo=1.0,
                              Z_A=15, Z_B=15)
    Z, coords, bonds = build_inverse_sandwich(
        X_ring_Z=15, n_ring=3, M_cap_Z=92,
        d_xx=None, ligands="none",
    )
    side = np.linalg.norm(coords[0] - coords[1])
    assert abs(side - expected) < 1e-9


def test_inverse_sandwich_axial_half_sandwich():
    """n_ligands_per_cap=1 → one Cp* per cap on z-axis."""
    Z, coords, bonds = build_inverse_sandwich(
        X_ring_Z=83, n_ring=3, M_cap_Z=92,
        ligands="Cp*", n_ligands_per_cap=1,
        include_methyl_h=False,
    )
    # 3 ring + 2 caps + 2 Cp* (10 atoms each w/o H: 5 ring C + 5 methyl C)
    assert len(Z) == 3 + 2 + 2 * 10
    assert sum(z == 6 for z in Z) == 2 * 10


def test_inverse_sandwich_invalid_n_ring():
    with pytest.raises(ValueError, match="n_ring must be >= 3"):
        build_inverse_sandwich(X_ring_Z=83, n_ring=2, M_cap_Z=92)


def test_inverse_sandwich_invalid_ligand():
    with pytest.raises(ValueError, match="ligands must be"):
        build_inverse_sandwich(
            X_ring_Z=83, n_ring=3, M_cap_Z=92,
            ligands="ferrocene",
        )


def test_inverse_sandwich_invalid_n_ligands():
    with pytest.raises(ValueError, match="n_ligands_per_cap"):
        build_inverse_sandwich(
            X_ring_Z=83, n_ring=3, M_cap_Z=92,
            ligands="Cp*", n_ligands_per_cap=3,
        )


def test_inverse_sandwich_orbital_count_pb3_th2():
    """Sanity check: ground-state basis for Pb3@Th2(Cp*)4 SZ."""
    Z, coords, bonds = build_inverse_sandwich(
        X_ring_Z=82, n_ring=3, M_cap_Z=90,
        ligands="Cp*", n_ligands_per_cap=2,
    )
    basis, topo = build_explicit_cluster(
        Z_list=Z, coords=coords, bonds=bonds,
        basis_type="SZ", include_f_block_d_shell=True,
    )
    assert basis.n_orbitals > 200  # 105 atoms with at least 2 orb each


def test_inverse_sandwich_ring_rotation_offset():
    Z0, coords0, _ = build_inverse_sandwich(
        X_ring_Z=83, n_ring=3, M_cap_Z=92,
        d_xx=3.0, ligands="none", ring_rotation_offset_deg=0.0,
    )
    Z1, coords1, _ = build_inverse_sandwich(
        X_ring_Z=83, n_ring=3, M_cap_Z=92,
        d_xx=3.0, ligands="none", ring_rotation_offset_deg=60.0,
    )
    # ring coords rotated, caps unchanged
    np.testing.assert_allclose(coords0[3:], coords1[3:], atol=1e-10)
    assert not np.allclose(coords0[:3], coords1[:3])
