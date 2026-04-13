"""
vertex_polygon.py — VertexPolygon dataclass for molecular vertices.

Each atom in a molecule occupies a VERTEX of the molecular polygon.
This dataclass captures the full geometric identity of that vertex:
  - Hexagonal face (polygon type, capacity, occupation, overflow)
  - Lone pair coverage and VSEPR solid angle
  - Coordination and period/block
  - Vacancy and Bohr classification

Zero adjustable parameters. All from s = 1/2.

March 2026 — Theorie de la Persistance
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

from ptc.constants import P1, RY, S3, C3
from ptc.data.experimental import SYMBOLS
from ptc.periodic import period, l_of, _n_fill_aufbau


# ── polygon_type: 2(2l+1) = number of positions on the face ──
_POLYGON_TYPE = {0: 2, 1: 6, 2: 10, 3: 14}

# ── PT-derived angle: Thompson problem with LP weight (1+s) ──
# NO lookup table. The angle EMERGES from s=1/2:
#   z_eff = z + lp × (1+s) = z + lp × 3/2
#   cos(θ) = -1/(z_eff - 1)   [Thomson minimum on S²]
#
# LP has one extra angular DOF (unconstrained spin) vs bond
# (directionally constrained by partner). Weight = 1+s = 3/2.
#
# Results (0 adjustable parameters):
#   H₂O (z=2,lp=2): 104.48° (exp 104.5°, err 0.02°)
#   NH₃ (z=3,lp=1): 106.60° (exp 107.0°, err 0.40°)
#   CH₄ (z=4,lp=0): 109.47° (exact)

def _pt_angle(z: int, lp: int) -> float:
    """Molecular angle from PT: Thompson problem with LP weight 1+s.

    Derived from s=1/2 (unique PT input). No lookup table.
    """
    from ptc.constants import S_HALF
    z_eff = z + lp * (1.0 + S_HALF)  # LP weight = 1+s = 3/2
    if z_eff <= 1.0:
        return 180.0
    cos_theta = -1.0 / (z_eff - 1.0)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


@dataclass(frozen=True)
class VertexPolygon:
    """Geometric identity of a molecular vertex.

    Each atom in a molecule is a vertex on the molecular polygon T^3.
    This dataclass unifies the ~15 inline boolean gates in molecule.py
    into a single geometric object.
    """
    # ── Identity ──
    idx: int
    Z: int
    symbol: str

    # ── Hexagonal face ──
    polygon_type: int          # 2(2l+1): 2, 6, 10, or 14
    face_capacity: int         # 2 * P_l (same as polygon_type)
    face_occupation: int       # nf + z (electrons + bonds filling the face)
    face_fraction: float       # occupation / capacity
    overflow: float            # max(0, face_fraction - 1)

    # ── Lone pairs ──
    lp: int
    lp_coverage: float         # lp / (z + lp)
    solid_angle: float         # z / (2 * (z + lp))
    theta_deg: float           # VSEPR angle

    # ── Coordination ──
    z: int                     # coordination number (number of bonds)
    per: int                   # period
    l: int                     # angular momentum quantum number
    nf: int                    # Aufbau filling of valence sub-shell

    # ── Capacities ──
    has_p_vacancy: bool        # face_occupation < 2*P1
    has_d_vacancy: bool        # per >= P1 and l <= 1 and not has_p_vacancy
    is_bohr: bool              # ie >= RY - S3
    ie: float                  # ionization energy (eV)

    # ── Classification ──
    vertex_class: str          # terminal, star_center, chain, branch, ring


def build_vertex_polygons(
    topology,
    atom_data: Dict[int, dict],
) -> List[VertexPolygon]:
    """Build one VertexPolygon per atom in the topology.

    Parameters
    ----------
    topology : Topology
        Molecular topology (must be finalized).
    atom_data : dict
        Per-Z data: {'IE', 'nf', 'per', 'l', ...}.

    Returns
    -------
    List[VertexPolygon]
        One polygon per atom, in topology.Z_list order.
    """
    polygons: List[VertexPolygon] = []
    n = len(topology.Z_list)

    for k in range(n):
        Z = topology.Z_list[k]
        d = atom_data[Z]

        sym = SYMBOLS.get(Z, f"Z{Z}")
        per = d['per']
        l = d['l']
        nf = d['nf']
        ie = d['IE']

        z_k = topology.z_count[k] if topology.z_count else 1
        lp_k = topology.lp[k] if topology.lp else 0
        vc = topology.vertex_class[k] if topology.vertex_class else 'terminal'

        # ── Hexagonal face ──
        poly_type = _POLYGON_TYPE.get(l, 6)
        face_cap = poly_type
        face_occ = nf + z_k
        face_frac = face_occ / face_cap if face_cap > 0 else 0.0
        ovf = max(0.0, face_frac - 1.0)

        # ── Lone pairs ──
        total = z_k + lp_k
        lp_cov = lp_k / total if total > 0 else 0.0
        s_angle = z_k / (2.0 * total) if total > 0 else 0.5
        theta = _pt_angle(z_k, lp_k)

        # ── Capacities ──
        has_p_vac = face_occ < 2 * P1
        has_d_vac = per >= P1 and l <= 1 and not has_p_vac
        is_bohr = ie >= RY - S3

        vp = VertexPolygon(
            idx=k,
            Z=Z,
            symbol=sym,
            polygon_type=poly_type,
            face_capacity=face_cap,
            face_occupation=face_occ,
            face_fraction=face_frac,
            overflow=ovf,
            lp=lp_k,
            lp_coverage=lp_cov,
            solid_angle=s_angle,
            theta_deg=theta,
            z=z_k,
            per=per,
            l=l,
            nf=nf,
            has_p_vacancy=has_p_vac,
            has_d_vacancy=has_d_vac,
            is_bohr=is_bohr,
            ie=ie,
            vertex_class=vc,
        )
        polygons.append(vp)

    return polygons
