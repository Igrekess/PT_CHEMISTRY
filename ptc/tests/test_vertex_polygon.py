"""Tests for the VertexPolygon dataclass and builder."""
from __future__ import annotations

import pytest

from ptc.vertex_polygon import VertexPolygon, build_vertex_polygons


# ── helper ──────────────────────────────────────────────────────────────────

def _get_polygons(smiles: str):
    from ptc.molecule import _resolve_atom_data
    from ptc.topology import build_topology
    topo = build_topology(smiles)
    atom_data = _resolve_atom_data(topo)
    return build_vertex_polygons(topo, atom_data), topo


# ── 1. CH4: carbon has z=4, lp=0, nf=2, face_fraction=1.0, polygon_type=6 ──

def test_C_in_CH4():
    polys, topo = _get_polygons("C")
    # Find the carbon (Z=6)
    c_idx = [i for i, Z in enumerate(topo.Z_list) if Z == 6]
    assert len(c_idx) == 1
    c = polys[c_idx[0]]
    assert c.Z == 6
    assert c.symbol == 'C'
    assert c.z == 4
    assert c.lp == 0
    assert c.nf == 2
    assert c.polygon_type == 6
    assert c.face_capacity == 6
    assert c.face_occupation == 6       # nf(2) + z(4) = 6
    assert c.face_fraction == pytest.approx(1.0)
    assert c.overflow == pytest.approx(0.0)
    assert c.has_p_vacancy is False      # face_occ = 6 = 2*P1


# ── 2. DMS (CSC): sulfur has z=2, lp=2, nf=4, lp_coverage=0.5, has_d_vacancy ──

def test_S_in_DMS():
    polys, topo = _get_polygons("CSC")
    s_idx = [i for i, Z in enumerate(topo.Z_list) if Z == 16]
    assert len(s_idx) == 1
    s = polys[s_idx[0]]
    assert s.Z == 16
    assert s.z == 2
    assert s.lp == 2
    assert s.nf == 4
    assert s.polygon_type == 6           # l=1 -> 6
    assert s.face_fraction == pytest.approx(1.0)  # (4+2)/6 = 1.0
    assert s.lp_coverage == pytest.approx(0.5)  # 2/(2+2)
    assert s.has_d_vacancy is True       # per=3 >= P1=3, l=1, not has_p_vacancy


# ── 3. SiCl4: Si has z=4, lp=0, face_fraction=1.0, has_d_vacancy ──

def test_Si_in_SiCl4():
    polys, topo = _get_polygons("Cl[Si](Cl)(Cl)Cl")
    si_idx = [i for i, Z in enumerate(topo.Z_list) if Z == 14]
    assert len(si_idx) == 1
    si = polys[si_idx[0]]
    assert si.Z == 14
    assert si.symbol == 'Si'
    assert si.z == 4
    assert si.lp == 0
    assert si.polygon_type == 6           # l=1 -> 6
    assert si.face_fraction == pytest.approx(1.0)  # (2+4)/6 = 1.0
    assert si.has_d_vacancy is True       # per=3 >= P1=3, l=1, not p_vac


# ── 4. PF3: P has z=3, lp=1, lp_coverage=0.25, has_d_vacancy ──

def test_P_in_PF3():
    polys, topo = _get_polygons("FP(F)F")
    p_idx = [i for i, Z in enumerate(topo.Z_list) if Z == 15]
    assert len(p_idx) == 1
    p = polys[p_idx[0]]
    assert p.Z == 15
    assert p.z == 3
    assert p.lp == 1
    assert p.polygon_type == 6            # l=1 -> 6
    assert p.face_fraction == pytest.approx(1.0)  # (3+3)/6 = 1.0
    assert p.lp_coverage == pytest.approx(0.25)  # 1/(3+1) = 0.25
    assert p.has_d_vacancy is True        # per=3 >= P1=3, l=1


# ── 5. BeCl2: Be has z=2, lp=0, nf=2, has_p_vacancy ──
# Be is s-block (l=0), polygon_type=2, face_occ=nf+z=4, overflow=1.0

def test_Be_in_BeCl2():
    polys, topo = _get_polygons("Cl[Be]Cl")
    be_idx = [i for i, Z in enumerate(topo.Z_list) if Z == 4]
    assert len(be_idx) == 1
    be = polys[be_idx[0]]
    assert be.Z == 4
    assert be.z == 2
    assert be.lp == 0
    assert be.nf == 2
    assert be.polygon_type == 2            # s-block: l=0 -> 2
    assert be.face_capacity == 2
    assert be.face_occupation == 4         # nf(2) + z(2) = 4
    assert be.face_fraction == pytest.approx(2.0)
    assert be.overflow == pytest.approx(1.0)
    assert be.has_p_vacancy is True        # face_occ(4) < 2*P1(6)


# ── 6. H terminal ──

def test_H_terminal():
    polys, topo = _get_polygons("C")
    h_idx = [i for i, Z in enumerate(topo.Z_list) if Z == 1]
    assert len(h_idx) >= 1
    h = polys[h_idx[0]]
    assert h.Z == 1
    assert h.z == 1
    assert h.vertex_class == 'terminal'


# ── 7. Count: len(polygons) == len(topo.Z_list) ──

def test_polygon_count_matches_atoms():
    for smiles in ["C", "CSC", "Cl[Si](Cl)(Cl)Cl", "FP(F)F", "O=O"]:
        polys, topo = _get_polygons(smiles)
        assert len(polys) == len(topo.Z_list), f"Mismatch for {smiles}"


# ── 8. Frozen: cannot mutate VertexPolygon ──

def test_frozen():
    polys, _ = _get_polygons("C")
    with pytest.raises(AttributeError):
        polys[0].z = 99
