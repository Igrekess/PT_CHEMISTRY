"""ptc/lcao/cluster.py — Explicit-coordinate cluster builder.

Most LCAO entry points start from a SMILES via ``build_topology`` and let
``compute_geometry_3d`` derive Cartesian coordinates from PT bond lengths.
For multi-cap actinide clusters (Bi3@U2, U-O frameworks, sandwich
complexes) the SMILES grammar is too narrow and the auto-geometry can
mis-place caps. This module provides a thin façade that takes explicit
``(Z_list, coords)`` plus an optional bond list, builds a finalised
``Topology`` with custom geometry, and returns a ready ``PTMolecularBasis``.

Usage
=====
>>> import numpy as np
>>> from ptc.lcao.cluster import build_explicit_cluster
>>> coords = np.array([
...     [ 1.50,  0.00, 0.0],
...     [-0.75,  1.30, 0.0],
...     [-0.75, -1.30, 0.0],
... ])
>>> basis, topology = build_explicit_cluster(
...     Z_list=[83, 83, 83],
...     coords=coords,
...     bonds=[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)],
... )
>>> basis.n_orbitals
12
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ptc.lcao.atomic_basis import PTAtomicOrbital, build_atom_basis
from ptc.lcao.density_matrix import PTMolecularBasis
from ptc.topology import Topology


def build_explicit_cluster(
    Z_list: Sequence[int],
    coords: np.ndarray,
    bonds: Optional[Sequence[Tuple[int, int, float]]] = None,
    charges: Optional[Sequence[int]] = None,
    basis_type: str = "SZ",   # accepted, only 'SZ' currently wired
    polarisation: bool = False,
    include_f_block_d_shell: bool = False,
) -> Tuple[PTMolecularBasis, Topology]:
    """Build a PTMolecularBasis from explicit (Z, coords, bonds).

    Parameters
    ----------
    Z_list   : sequence of N atomic numbers.
    coords   : (N, 3) array, Angstroms.
    bonds    : optional list of (i, j, bond_order). If None, no bonds.
    charges  : per-atom formal charges; default zeros.
    basis_type, include_core, zeta_method, polarisation,
    include_f_block_d_shell : see ``build_atom_basis``.

    Returns
    -------
    basis    : PTMolecularBasis with the explicit ``coords`` baked in.
    topology : finalised Topology stub (Z_list, bonds, charges).
    """
    Z_list = list(Z_list)
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (len(Z_list), 3):
        raise ValueError(
            f"coords shape {coords.shape} does not match {len(Z_list)} atoms"
        )
    if charges is None:
        charges = [0] * len(Z_list)
    charges = list(charges)
    if bonds is None:
        bonds = []
    bonds_list = [(int(i), int(j), float(bo)) for i, j, bo in bonds]

    topology = Topology(
        Z_list=Z_list,
        bonds=bonds_list,
        source="explicit",
    )
    topology.charges = charges
    topology.finalize()

    atoms = []
    flat_orbs: List[PTAtomicOrbital] = []
    atom_idx: List[int] = []
    for i, Z in enumerate(Z_list):
        ab = build_atom_basis(
            Z, charge=charges[i],
            polarisation=polarisation,
            basis_type=basis_type,
            include_f_block_d_shell=include_f_block_d_shell,
        )
        atoms.append(ab)
        for orb in ab.orbitals:
            flat_orbs.append(orb)
            atom_idx.append(i)

    basis = PTMolecularBasis(
        atoms=atoms,
        coords=coords.copy(),
        orbitals=flat_orbs,
        atom_index=atom_idx,
        Z_list=Z_list,
    )
    return basis, topology


def precompute_response_explicit(
    Z_list: Sequence[int],
    coords: np.ndarray,
    bonds: Optional[Sequence[Tuple[int, int, float]]] = None,
    charges: Optional[Sequence[int]] = None,
    basis_type: str = "SZ",
    scf_mode: str = "hueckel",
    scf_kwargs: Optional[dict] = None,
    cphf_kwargs: Optional[dict] = None,
    include_f_block_d_shell: bool = False,
):
    """Build basis + ground-state density + CPHF response on explicit coords.

    ``scf_mode`` : 'hueckel' (Mulliken K=2 fast) or 'hartree' / 'hf'
    (full SCF via density_matrix_PT_scf with DIIS).
    """
    from ptc.lcao.current import _ResponseData
    from ptc.lcao.density_matrix import density_matrix_PT
    from ptc.lcao.fock import coupled_cphf_response, density_matrix_PT_scf

    basis, topology = build_explicit_cluster(
        Z_list, coords, bonds=bonds, charges=charges,
        basis_type=basis_type,
        include_f_block_d_shell=include_f_block_d_shell,
    )

    if scf_mode == "hueckel":
        rho, _S, eigvals, c = density_matrix_PT(topology, basis=basis)
    elif scf_mode in ("hartree", "hf"):
        scf_kw = dict(scf_kwargs) if scf_kwargs else {}
        rho, _S, eigvals, c, _conv, _resid = density_matrix_PT_scf(
            topology, basis=basis, mode=scf_mode, **scf_kw,
        )
    else:
        raise ValueError(
            f"scf_mode must be 'hueckel'/'hartree'/'hf', got {scf_mode!r}"
        )

    n_e = int(round(basis.total_occ))
    n_occ = n_e // 2
    cphf_kw = dict(cphf_kwargs) if cphf_kwargs else {}
    U_imag = coupled_cphf_response(basis, eigvals, c, n_e, **cphf_kw)
    return _ResponseData(
        basis=basis, rho=rho, mo_coeffs=c, U_imag=U_imag, n_occ=n_occ,
        eigvals=eigvals,
    )


# ─────────────────────────────────────────────────────────────────────
# Cp* ligand geometry (pentamethylcyclopentadienyl)
# ─────────────────────────────────────────────────────────────────────


_CP_RING_R = 1.42 / (2.0 * np.sin(np.radians(36.0)))
_CP_METHYL_C_R = _CP_RING_R + 1.50
_CH_DISTANCE = 1.10


def cp_star_atoms(centroid: np.ndarray,
                    normal: np.ndarray,
                    rotation_deg: float = 0.0,
                    include_h: bool = True,
                    ) -> Tuple[list, np.ndarray, list, list]:
    """Build a Cp* (pentamethylcyclopentadienyl, C5(CH3)5) ring.

    Returns (Z_local, coords (N, 3), bonds list, ring_indices list).
    """
    centroid = np.asarray(centroid, dtype=float)
    n_hat = np.asarray(normal, dtype=float)
    n_hat = n_hat / float(np.linalg.norm(n_hat))

    if abs(n_hat[2]) < 0.95:
        u_hat = np.cross(np.array([0.0, 0.0, 1.0]), n_hat)
    else:
        u_hat = np.cross(np.array([1.0, 0.0, 0.0]), n_hat)
    u_hat = u_hat / float(np.linalg.norm(u_hat))
    v_hat = np.cross(n_hat, u_hat)

    Z_local: list = []
    coord_list: list = []
    bonds: list = []
    ring_indices: list = []

    rot_rad = np.radians(rotation_deg)

    # 5 ring C atoms (indices 0..4)
    for k in range(5):
        angle = 2.0 * np.pi * k / 5.0 + rot_rad
        pos = centroid + _CP_RING_R * (np.cos(angle) * u_hat
                                          + np.sin(angle) * v_hat)
        Z_local.append(6)
        coord_list.append(pos)
        ring_indices.append(k)

    # 5 methyl C atoms (indices 5..9)
    for k in range(5):
        angle = 2.0 * np.pi * k / 5.0 + rot_rad
        pos = centroid + _CP_METHYL_C_R * (np.cos(angle) * u_hat
                                              + np.sin(angle) * v_hat)
        Z_local.append(6)
        coord_list.append(pos)

    for k in range(5):
        bonds.append((k, (k + 1) % 5, 1.5))
    for k in range(5):
        bonds.append((k, k + 5, 1.0))

    if include_h:
        for k in range(5):
            angle = 2.0 * np.pi * k / 5.0 + rot_rad
            radial_hat = np.cos(angle) * u_hat + np.sin(angle) * v_hat
            methyl_pos = coord_list[5 + k]
            h_a = methyl_pos + _CH_DISTANCE * radial_hat
            tilt = np.radians(30.0)
            h_b = methyl_pos + _CH_DISTANCE * (
                np.cos(tilt) * radial_hat + np.sin(tilt) * n_hat
            )
            h_c = methyl_pos + _CH_DISTANCE * (
                np.cos(tilt) * radial_hat - np.sin(tilt) * n_hat
            )
            base_idx = 10 + 3 * k
            for hidx, hpos in enumerate([h_a, h_b, h_c]):
                Z_local.append(1)
                coord_list.append(hpos)
                bonds.append((5 + k, base_idx + hidx, 1.0))

    return Z_local, np.array(coord_list), bonds, ring_indices


def build_inverse_sandwich(
    X_ring_Z: int,
    n_ring: int,
    M_cap_Z: int,
    d_xx: Optional[float] = None,
    z_cap: float = 2.1,
    ligands: str = "Cp*",
    n_ligands_per_cap: int = 2,
    cp_cp_angle_deg: float = 134.0,
    d_m_cp_centroid: float = 2.5,
    include_methyl_h: bool = True,
    rotation_deg_per_cp: Optional[Sequence[float]] = None,
    ring_rotation_offset_deg: float = 0.0,
) -> Tuple[list, np.ndarray, list]:
    """Generic inverse sandwich (X_n)@(M_2)(L_p) cluster geometry.

    Reproduces ``build_bi3_u2_cp_star4`` for X=Bi, n_ring=3, M=U,
    ligands="Cp*", n_ligands_per_cap=2.

    Parameters
    ----------
    X_ring_Z : Z of the homonuclear ring atoms.
    n_ring : ring size, ≥ 3.
    M_cap_Z : Z of the two axial cap atoms (typically lanthanide / actinide).
    d_xx : ring X-X distance in Å. If None, derived from ``r_equilibrium``.
    z_cap : axial height of the caps above the ring plane (Å).
    ligands : ``"none"`` or ``"Cp*"``.
    n_ligands_per_cap : 0, 1 (axial half-sandwich) or 2 (bent metallocene).
    cp_cp_angle_deg : Cp-M-Cp bend (only for n_ligands_per_cap=2).
    d_m_cp_centroid : M-Cp* centroid distance.
    include_methyl_h : include CH3 hydrogens on Cp*.
    rotation_deg_per_cp : rotation in degrees for each ligand (length =
        n_ligands_per_cap × 2). Defaults to staggered (0, 36, 0, 36)
        for matching ``build_bi3_u2_cp_star4``.
    ring_rotation_offset_deg : rigid in-plane rotation of the ring.

    Returns
    -------
    (Z_list, coords (N x 3), bonds list)
    """
    if n_ring < 3:
        raise ValueError(f"n_ring must be >= 3, got {n_ring}")
    if ligands not in ("none", "Cp*"):
        raise ValueError(
            f"ligands must be 'none' or 'Cp*', got {ligands!r}"
        )
    if n_ligands_per_cap not in (0, 1, 2):
        raise ValueError(
            f"n_ligands_per_cap must be 0/1/2, got {n_ligands_per_cap}"
        )

    if d_xx is None:
        from ptc.bond import r_equilibrium
        from ptc.periodic import period
        per_x = period(X_ring_Z)
        d_xx = r_equilibrium(
            per_x, per_x, bo=1.0, Z_A=X_ring_Z, Z_B=X_ring_Z,
        )

    R_x = float(d_xx) / (2.0 * np.sin(np.pi / n_ring))
    rot0 = np.radians(ring_rotation_offset_deg)
    ring_coords = []
    for k in range(n_ring):
        angle = rot0 + 2.0 * np.pi * k / n_ring
        ring_coords.append(
            [R_x * np.cos(angle), R_x * np.sin(angle), 0.0]
        )

    Z_list: list = [int(X_ring_Z)] * n_ring + [int(M_cap_Z), int(M_cap_Z)]
    all_coords: list = list(np.asarray(ring_coords)) + [
        np.array([0.0, 0.0, float(z_cap)]),
        np.array([0.0, 0.0, -float(z_cap)]),
    ]
    bonds: list = [(k, (k + 1) % n_ring, 1.0) for k in range(n_ring)]
    cap_top_idx = n_ring
    cap_bot_idx = n_ring + 1
    for cap in (cap_top_idx, cap_bot_idx):
        for ring in range(n_ring):
            bonds.append((ring, cap, 1.0))

    if ligands == "none" or n_ligands_per_cap == 0:
        return Z_list, np.array(all_coords), bonds

    # Cp* attachment
    if n_ligands_per_cap == 2:
        if rotation_deg_per_cp is None:
            rotation_deg_per_cp = (0.0, 36.0, 0.0, 36.0)
        if len(rotation_deg_per_cp) != 4:
            raise ValueError(
                "rotation_deg_per_cp must have 4 entries for "
                "n_ligands_per_cap=2"
            )
        half_angle = np.radians(cp_cp_angle_deg / 2.0)
        cp_orientations = [
            (cap_top_idx, +1, +1),
            (cap_top_idx, -1, +1),
            (cap_bot_idx, +1, -1),
            (cap_bot_idx, -1, -1),
        ]
        for cp_idx, (cap_atom, sx, sz) in enumerate(cp_orientations):
            cap_pos = all_coords[cap_atom]
            z_outer_hat = np.array([0.0, 0.0, float(sz)])
            direction = (
                sx * np.sin(half_angle) * np.array([1.0, 0.0, 0.0])
                + np.cos(half_angle) * z_outer_hat
            )
            cp_centroid = cap_pos + d_m_cp_centroid * direction
            Z_cp, coords_cp, bonds_cp, ring_idx = cp_star_atoms(
                cp_centroid, direction,
                rotation_deg=rotation_deg_per_cp[cp_idx],
                include_h=include_methyl_h,
            )
            offset = len(Z_list)
            Z_list.extend(Z_cp)
            all_coords.extend(coords_cp)
            for (a, b, bo) in bonds_cp:
                bonds.append((a + offset, b + offset, bo))
            for r in ring_idx:
                bonds.append((cap_atom, offset + r, 0.5))
    else:  # n_ligands_per_cap == 1 (axial half-sandwich)
        if rotation_deg_per_cp is None:
            rotation_deg_per_cp = (0.0, 0.0)
        if len(rotation_deg_per_cp) != 2:
            raise ValueError(
                "rotation_deg_per_cp must have 2 entries for "
                "n_ligands_per_cap=1"
            )
        for cp_idx, (cap_atom, sz) in enumerate(
            [(cap_top_idx, +1), (cap_bot_idx, -1)]
        ):
            cap_pos = all_coords[cap_atom]
            direction = np.array([0.0, 0.0, float(sz)])
            cp_centroid = cap_pos + d_m_cp_centroid * direction
            Z_cp, coords_cp, bonds_cp, ring_idx = cp_star_atoms(
                cp_centroid, direction,
                rotation_deg=rotation_deg_per_cp[cp_idx],
                include_h=include_methyl_h,
            )
            offset = len(Z_list)
            Z_list.extend(Z_cp)
            all_coords.extend(coords_cp)
            for (a, b, bo) in bonds_cp:
                bonds.append((a + offset, b + offset, bo))
            for r in ring_idx:
                bonds.append((cap_atom, offset + r, 0.5))

    return Z_list, np.array(all_coords), bonds


def build_bi3_u2_cp_star4(
    d_bi_bi: float = 3.05,
    z_u_cap: float = 2.1,
    d_u_cp_centroid: float = 2.5,
    cp_cp_angle_deg: float = 134.0,
    include_methyl_h: bool = True,
    rotation_deg_per_cp: Tuple[float, float, float, float] = (
        0.0, 36.0, 0.0, 36.0,
    ),
) -> Tuple[list, np.ndarray, list]:
    """Assemble the full Bi3@U2(Cp*)4 cluster (Ding 2026 inverse sandwich).

    Each U carries a bent-metallocene pair of Cp* with
    ``cp_cp_angle_deg`` between them on the U-outer side.

    Returns (Z_list, coords (N x 3), bonds).
    """
    half_angle = np.radians(cp_cp_angle_deg / 2.0)

    R_bi = d_bi_bi / np.sqrt(3.0)
    bi_coords = np.array([
        [R_bi,            0.0,                        0.0],
        [-R_bi / 2.0,     R_bi * np.sqrt(3) / 2.0,    0.0],
        [-R_bi / 2.0,    -R_bi * np.sqrt(3) / 2.0,    0.0],
    ])

    u_top = np.array([0.0, 0.0, z_u_cap])
    u_bot = np.array([0.0, 0.0, -z_u_cap])

    Z_list: list = [83, 83, 83, 92, 92]
    all_coords: list = list(bi_coords) + [u_top, u_bot]
    bonds: list = [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]
    for u_idx in (3, 4):
        for bi in (0, 1, 2):
            bonds.append((bi, u_idx, 1.0))

    cp_orientations = [
        (3, +1, +1),
        (3, -1, +1),
        (4, +1, -1),
        (4, -1, -1),
    ]

    for cp_idx, (u_atom, sx, sz) in enumerate(cp_orientations):
        u_pos = all_coords[u_atom]
        z_outer_hat = np.array([0.0, 0.0, float(sz)])
        direction = (
            sx * np.sin(half_angle) * np.array([1.0, 0.0, 0.0])
            + np.cos(half_angle) * z_outer_hat
        )
        cp_centroid = u_pos + d_u_cp_centroid * direction
        cp_normal = direction
        rotation = rotation_deg_per_cp[cp_idx]
        Z_cp, coords_cp, bonds_cp, ring_idx = cp_star_atoms(
            cp_centroid, cp_normal, rotation_deg=rotation,
            include_h=include_methyl_h,
        )
        offset = len(Z_list)
        Z_list.extend(Z_cp)
        all_coords.extend(coords_cp)
        for (a, b, bo) in bonds_cp:
            bonds.append((a + offset, b + offset, bo))
        for r in ring_idx:
            bonds.append((u_atom, offset + r, 0.5))

    return Z_list, np.array(all_coords), bonds
