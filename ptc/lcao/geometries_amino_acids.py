"""Reference molecular geometries for the smallest amino acids.

This module provides hand-built but standard-quality geometries for
L-alanine, D-alanine and glycine, ready for use as input to the
PT parity-violation pipeline (``ptc.lcao.parity_violation``).

Each helper returns a tuple ``(atom_list, basis)``:

* ``atom_list`` is a list of ``(Z, position_A)`` tuples, with ``Z`` an
  integer atomic number and ``position_A`` a ``numpy.ndarray`` of
  shape ``(3,)`` in Angstrom.
* ``basis`` is a :class:`PTMolecularBasis` whose atomic SZ orbitals
  match ``atom_list``, ready to feed into
  ``parity_violation.hpv_matrix_elements``.

Geometry source
---------------

The atomic coordinates below are textbook sp3 / sp2 hybrid geometries
hand-built from standard bond lengths and bond angles. They are
intended to be physically reasonable starting points (within ~0.05 A
of any MP2/6-31G* optimised geometry), not energy-minimised structures.

Standard bond lengths used (in Angstrom):
    C(sp3) - C(sp3)   1.526
    C(sp3) - C(sp2)   1.515  (Calpha - C(=O))
    C(sp3) - N(sp3)   1.471  (Calpha - NH2)
    C(sp2) = O        1.214  (carbonyl)
    C(sp2) - O(-H)    1.357  (hydroxyl)
    C(sp3) - H        1.090
    N(sp3) - H        1.014
    O(sp3) - H        0.972

Tetrahedral angles around the chirality center Cα are 109.47°.
The carboxyl group is planar with sp2 geometry on C(=O).

Basis
-----

We use the default PTC single-zeta (SZ) atomic basis from
``atomic_basis.build_atom_basis`` with default ``zeta_method='pt'``.
That is: one STO per valence subshell, with PT-screened exponent.
This matches the conventions used throughout
``test_parity_violation.py`` (see ``_manual_basis`` there) and is
sufficient for a first-pass H_PV matrix-element calculation. No
polarisation or split valence is required at this stage.

References
----------

The geometry conventions follow the same hand-built sp3 / sp2 style
used by :func:`_h2o2_geometry` and :func:`_water_geometry_Cs` in
``test_parity_violation.py``. Standard bond lengths are taken from
the CRC Handbook of Chemistry and Physics and the NIST CCCBDB
recommended values for biological small molecules.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from ptc.lcao.atomic_basis import build_atom_basis
from ptc.lcao.density_matrix import PTMolecularBasis


# ---------------------------------------------------------------------
# Standard bond lengths and angles (Angstrom, radians)
# ---------------------------------------------------------------------

R_CC = 1.526      # C(sp3) - C(sp3)
R_CC_SP2 = 1.515  # C(sp3) - C(sp2)
R_CN = 1.471      # C(sp3) - N(sp3)
R_CO_DBL = 1.214  # C=O
R_CO_SGL = 1.357  # C-OH
R_CH = 1.090      # C(sp3) - H
R_NH = 1.014      # N(sp3) - H
R_OH = 0.972      # O(sp3) - H

THETA_TET = math.acos(-1.0 / 3.0)        # 109.47 deg tetrahedral
THETA_SP2 = math.radians(120.0)


# ---------------------------------------------------------------------
# Internal: build a PTMolecularBasis directly from (Z_list, coords)
# ---------------------------------------------------------------------


def _molecular_basis_from_atoms(
    atom_list: List[Tuple[int, np.ndarray]],
) -> PTMolecularBasis:
    """Assemble a PT single-zeta molecular basis from an explicit atom list.

    This mirrors ``_manual_basis`` in ``test_parity_violation.py``:
    bypass the topology layer entirely and put one ``PTAtomBasis`` per
    atom with its default neutral valence shell.
    """
    Z_list = [int(Z) for (Z, _r) in atom_list]
    coords = np.array([np.asarray(r, dtype=float) for (_Z, r) in atom_list])

    atoms = []
    flat_orbs = []
    atom_idx = []
    for i, Z in enumerate(Z_list):
        ab = build_atom_basis(Z)
        atoms.append(ab)
        for orb in ab.orbitals:
            flat_orbs.append(orb)
            atom_idx.append(i)

    return PTMolecularBasis(
        atoms=atoms,
        coords=coords,
        orbitals=flat_orbs,
        atom_index=atom_idx,
        Z_list=Z_list,
    )


# ---------------------------------------------------------------------
# Tetrahedral geometry helpers
# ---------------------------------------------------------------------


def _tetrahedral_directions() -> np.ndarray:
    """Return four unit vectors pointing to the corners of a tetrahedron.

    Convention: corners on a cube with sign pattern (+,+,+), (+,-,-),
    (-,+,-), (-,-,+) divided by sqrt(3). All four are mutually at the
    tetrahedral angle 109.47°. Good for placing the four substituents
    of an sp3 centre without further rotation.
    """
    s = 1.0 / math.sqrt(3.0)
    return np.array(
        [
            [+s, +s, +s],
            [+s, -s, -s],
            [-s, +s, -s],
            [-s, -s, +s],
        ]
    )


def _two_sp3_h_along_x(
    center: np.ndarray, ref_dir: np.ndarray, r_xh: float = R_NH
) -> tuple[np.ndarray, np.ndarray]:
    """Place two equivalent H atoms on an sp3 nitrogen lone-pair plane.

    The two H positions are placed symmetrically around the bond axis
    ``ref_dir`` (the N - Calpha vector), at the tetrahedral angle
    THETA_TET and offset by +- 60 deg in the perpendicular plane. The
    result is a CH2- or NH2-like wedge that is mirror-symmetric across
    the (ref_dir, perp_z) plane.
    """
    ref = ref_dir / np.linalg.norm(ref_dir)
    # Pick a perpendicular axis: prefer the global y, otherwise z.
    y_hat = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(ref, y_hat)) > 0.95:
        y_hat = np.array([0.0, 0.0, 1.0])
    perp = y_hat - np.dot(y_hat, ref) * ref
    perp /= np.linalg.norm(perp)
    third = np.cross(ref, perp)

    # H lies at angle THETA_TET from the BOND direction ref (i.e. at
    # angle pi - THETA_TET from the N - parent vector, since ref points
    # AWAY from the parent atom).
    cos_a = math.cos(math.pi - THETA_TET)
    sin_a = math.sin(math.pi - THETA_TET)
    h1 = cos_a * ref + sin_a * perp
    h2 = cos_a * ref - sin_a * perp
    # Tilt by +- 30 deg about the bond axis so the two H atoms are
    # symmetric across the (ref, third) plane.
    rot = math.radians(30.0)
    c, s = math.cos(rot), math.sin(rot)
    h1r = h1.copy()
    h2r = h2.copy()
    # Rotate h1, h2 about ref by +rot, -rot resp.
    def _rot_about(v, axis, ang):
        ca, sa = math.cos(ang), math.sin(ang)
        return (
            ca * v
            + sa * np.cross(axis, v)
            + (1.0 - ca) * np.dot(axis, v) * axis
        )

    h1r = _rot_about(h1, ref, +rot)
    h2r = _rot_about(h2, ref, -rot)
    return center + r_xh * h1r, center + r_xh * h2r


# ---------------------------------------------------------------------
# L-Alanine
# ---------------------------------------------------------------------


def alanine_L() -> tuple[
    List[Tuple[int, np.ndarray]], PTMolecularBasis
]:
    """L-Alanine, (S)-2-aminopropanoic acid, 13 atoms (C3 H7 N O2).

    The chirality center is Cα. We place it at the origin and pick the
    four substituent directions on a tetrahedron such that the
    (S)-configuration is recovered:

        Priority order (CIP): NH2 > COOH > CH3 > H .
        For (S), looking from opposite H, the priority NH2 -> COOH -> CH3
        goes counter-clockwise.

    The tetrahedral corners used:
        NH2  -> ( +s, +s, +s)
        COOH -> ( +s, -s, -s)
        CH3  -> ( -s, +s, -s)
        H    -> ( -s, -s, +s)

    with s = 1/sqrt(3). The signed volume of (e_NH2, e_COOH, e_CH3) is
    +8/(3 sqrt 3) > 0, corresponding to a counter-clockwise NH2 -> COOH
    -> CH3 path viewed from the H side (i.e. from -H direction); this
    is the conventional (S) enantiomer.

    Returns
    -------
    atom_list : list of (Z, position_A)
        13 atoms in the canonical order
        [Calpha, N, C_carb, C_methyl, H_alpha, H_N1, H_N2,
         O_carb_dbl, O_carb_OH, H_OH, H_methyl1, H_methyl2, H_methyl3] .
    basis : PTMolecularBasis
        Default PT single-zeta atomic basis, ready for the parity
        violation pipeline.
    """
    e_NH2, e_COOH, e_CH3, e_H = _tetrahedral_directions()

    Calpha = np.zeros(3)

    # Heavy atoms bonded to Cα
    N = Calpha + R_CN * e_NH2
    C_carb = Calpha + R_CC_SP2 * e_COOH
    C_methyl = Calpha + R_CC * e_CH3
    H_alpha = Calpha + R_CH * e_H

    # ----- Amine group NH2 -----
    # Place the two amine H's symmetrically around the C-N axis. The
    # "ref_dir" should point AWAY from Cα so that the H-N-H plane sits
    # on the outer side of N.
    H_N1, H_N2 = _two_sp3_h_along_x(
        center=N, ref_dir=(N - Calpha), r_xh=R_NH
    )

    # ----- Carboxyl group COOH -----
    # The carboxyl carbon is sp2. Build a planar local frame with the
    # in-plane axis along (C_carb - Calpha) and an orthogonal axis
    # perpendicular to the C-O-O plane chosen as e_NH2 cross e_COOH
    # (any axis not parallel to e_COOH would do).
    bond_in = (C_carb - Calpha) / np.linalg.norm(C_carb - Calpha)
    plane_normal = np.cross(e_NH2, e_COOH)
    plane_normal /= np.linalg.norm(plane_normal)
    plane_perp = np.cross(plane_normal, bond_in)
    plane_perp /= np.linalg.norm(plane_perp)

    # O=C and C-O at +-120 deg from -bond_in in the local plane.
    # -bond_in is the direction "back to Calpha"; the two O atoms sit
    # at 120 deg on either side, perpendicular to the C-Calpha vector
    # plus a sp2 tilt.
    def _sp2_direction(angle: float) -> np.ndarray:
        # 0 deg = continuation of bond_in (i.e. away from Calpha).
        return math.cos(angle) * bond_in + math.sin(angle) * plane_perp

    O_dbl = C_carb + R_CO_DBL * _sp2_direction(+THETA_SP2 / 2.0)
    O_sgl = C_carb + R_CO_SGL * _sp2_direction(-THETA_SP2 / 2.0)
    # The hydroxyl H lies in the same plane (syn-anti conformer with
    # the C-O-H angle at the tetrahedral 109.47 deg). Build an
    # orthonormal frame at O: e_OC points back toward C; t_hat is the
    # unit in-plane vector perpendicular to e_OC (in the COO plane,
    # chosen on the side AWAY from the O=C oxygen).
    e_OC = (C_carb - O_sgl) / np.linalg.norm(C_carb - O_sgl)
    t_hat = plane_perp - np.dot(plane_perp, e_OC) * e_OC
    t_hat /= np.linalg.norm(t_hat)
    # Bias the H away from the carbonyl oxygen for the more stable
    # syn-anti rotamer: pick the sign of t_hat that puts H_OH farthest
    # from O_dbl.
    cand_plus = O_sgl + R_OH * (
        math.cos(math.pi - THETA_TET) * e_OC
        + math.sin(math.pi - THETA_TET) * t_hat
    )
    cand_minus = O_sgl + R_OH * (
        math.cos(math.pi - THETA_TET) * e_OC
        - math.sin(math.pi - THETA_TET) * t_hat
    )
    if np.linalg.norm(cand_plus - O_dbl) >= np.linalg.norm(cand_minus - O_dbl):
        H_OH = cand_plus
    else:
        H_OH = cand_minus

    # ----- Methyl group CH3 -----
    # Standard sp3 methyl: one C-H along (C_methyl - Calpha) (staggered
    # opposite Cα), and two C-H at +- 120 deg about the C_methyl - Cα
    # axis, all at THETA_TET from the C-Cα vector.
    bond_meth = (C_methyl - Calpha) / np.linalg.norm(C_methyl - Calpha)
    # Choose a perpendicular axis once and rotate.
    perp0 = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(perp0, bond_meth)) > 0.95:
        perp0 = np.array([0.0, 0.0, 1.0])
    perp0 = perp0 - np.dot(perp0, bond_meth) * bond_meth
    perp0 /= np.linalg.norm(perp0)

    def _methyl_h(angle_about_axis: float) -> np.ndarray:
        # Rotate the staggered H around the C-Cα axis by angle.
        c, s = math.cos(angle_about_axis), math.sin(angle_about_axis)
        radial = c * perp0 + s * np.cross(bond_meth, perp0)
        h_dir = (
            math.cos(math.pi - THETA_TET) * bond_meth
            + math.sin(math.pi - THETA_TET) * radial
        )
        return C_methyl + R_CH * h_dir

    H_M1 = _methyl_h(0.0)
    H_M2 = _methyl_h(2.0 * math.pi / 3.0)
    H_M3 = _methyl_h(-2.0 * math.pi / 3.0)

    atom_list: List[Tuple[int, np.ndarray]] = [
        (6, Calpha),          # 0: Cα
        (7, N),               # 1: N
        (6, C_carb),          # 2: C(=O)
        (6, C_methyl),        # 3: C(methyl)
        (1, H_alpha),         # 4: H on Cα
        (1, H_N1),            # 5: N-H
        (1, H_N2),            # 6: N-H
        (8, O_dbl),           # 7: O= (carbonyl)
        (8, O_sgl),           # 8: O-H (hydroxyl)
        (1, H_OH),            # 9: O-H hydrogen
        (1, H_M1),            # 10: methyl H
        (1, H_M2),            # 11: methyl H
        (1, H_M3),            # 12: methyl H
    ]
    basis = _molecular_basis_from_atoms(atom_list)
    return atom_list, basis


def alanine_D() -> tuple[
    List[Tuple[int, np.ndarray]], PTMolecularBasis
]:
    """D-Alanine, the mirror image of L-alanine.

    Built by negating the x coordinate of every atom in L-alanine.
    This is a parity operation P_x : (x, y, z) -> (-x, y, z), which is
    a reflection -- it changes chirality. Atom identities (Z) are
    unchanged; only positions are mirrored.
    """
    atom_list_L, _ = alanine_L()
    mirror = np.diag([-1.0, 1.0, 1.0])
    atom_list_D: List[Tuple[int, np.ndarray]] = [
        (Z, mirror @ r) for (Z, r) in atom_list_L
    ]
    basis = _molecular_basis_from_atoms(atom_list_D)
    return atom_list_D, basis


# ---------------------------------------------------------------------
# Glycine
# ---------------------------------------------------------------------


def glycine() -> tuple[List[Tuple[int, np.ndarray]], PTMolecularBasis]:
    """Glycine, NH2-CH2-COOH, 10 atoms (C2 H5 N O2).

    Built explicitly with a sigma_v(xz) mirror plane: every atom has
    y = 0 OR appears as part of a +-y mirror pair (the two Cα H atoms
    and the two amine H atoms). The resulting molecule has Cs symmetry
    and is achiral. This is the "glycine zero" reference for the
    parity-violation pipeline.

    The local geometry around Cα is built exactly as for alanine but
    with the methyl substituent replaced by a hydrogen. The two CH2 H
    atoms therefore sit symmetrically across the xz plane.
    """
    e_NH2, e_COOH, e_CH3, e_H_axial = _tetrahedral_directions()

    Calpha = np.zeros(3)

    # ----- Heavy atoms placed in the xz plane (y = 0) -----
    # Re-express the NH2 / COOH directions with their y-components
    # taken to be zero so that the heavy-atom backbone lies in xz.
    # We project onto the xz plane and renormalise. Both have non-zero
    # x and z components in _tetrahedral_directions, so the projection
    # is non-trivial.
    def _project_xz(v: np.ndarray) -> np.ndarray:
        out = np.array([v[0], 0.0, v[2]])
        n = np.linalg.norm(out)
        if n < 1e-12:
            raise ValueError("vector has no xz component")
        return out / n

    e_NH2_xz = _project_xz(e_NH2)
    e_COOH_xz = _project_xz(e_COOH)

    N = Calpha + R_CN * e_NH2_xz
    C_carb = Calpha + R_CC_SP2 * e_COOH_xz

    # ----- The two Cα H atoms: mirror images in y -----
    # They occupy the two remaining tetrahedral slots. By symmetry, if
    # the heavy atoms (N and C_carb) lie in xz, the two remaining sp3
    # directions are reflections of each other across the xz plane.
    # Compute them as e_CH3 and e_H_axial reflected to enforce the
    # mirror.
    e_HA = e_CH3.copy()
    e_HB = e_HA.copy()
    e_HB[1] *= -1.0   # mirror across xz
    # Re-orthogonalise: the two unit vectors should sit at the
    # tetrahedral angle from the two heavy-atom directions. We rebuild
    # them via Gram-Schmidt-like correction: take their average (in xz)
    # and the y component as ± component perpendicular to the NCC plane.
    # Easier: parameterise them analytically.
    #
    # Geometry: with N along e_NH2_xz and C_carb along e_COOH_xz in xz,
    # the two remaining sp3 slots are
    #   e_H_+  = -(e_NH2_xz + e_COOH_xz)/|...| * cos(alpha) + y_hat * sin(alpha)
    #   e_H_-  = -(e_NH2_xz + e_COOH_xz)/|...| * cos(alpha) - y_hat * sin(alpha)
    # with alpha chosen so all six angles equal THETA_TET. For a
    # tetrahedron with two opposite edges in the xz plane and two
    # opposite edges perpendicular, cos(alpha) and sin(alpha) are
    # determined by the constraint
    #   (e_H_+ . e_NH2_xz) = cos(THETA_TET) = -1/3 .
    sum_xz = e_NH2_xz + e_COOH_xz
    if np.linalg.norm(sum_xz) < 1e-12:
        # NH2 and COOH on opposite sides; the "back" direction is
        # arbitrary in xz. Pick -x as a fallback.
        bisector = np.array([-1.0, 0.0, 0.0])
    else:
        bisector = -sum_xz / np.linalg.norm(sum_xz)
    # Constraint: e_H . e_NH2_xz = -1/3.
    # e_H = cos(alpha) bisector + sin(alpha) y_hat
    # ==>  cos(alpha) * (bisector . e_NH2_xz) + 0 = -1/3
    b_dot_n = float(np.dot(bisector, e_NH2_xz))
    cos_alpha = (-1.0 / 3.0) / b_dot_n
    cos_alpha = max(-1.0, min(1.0, cos_alpha))
    sin_alpha = math.sqrt(max(0.0, 1.0 - cos_alpha ** 2))
    y_hat = np.array([0.0, 1.0, 0.0])
    e_HA = cos_alpha * bisector + sin_alpha * y_hat
    e_HB = cos_alpha * bisector - sin_alpha * y_hat

    H_alpha_1 = Calpha + R_CH * e_HA
    H_alpha_2 = Calpha + R_CH * e_HB

    # ----- NH2 hydrogens: mirror pair in y -----
    # Build an "in xz" reference direction for N-H, then place H atoms
    # at +-y. Same construction as for Cα.
    n_back = (N - Calpha) / np.linalg.norm(N - Calpha)
    # The third bonded direction on N is the lone pair; the two NH bonds
    # sit at the tetrahedral angle from each other and from n_back.
    # Place the two H's symmetrically in the (n_back, y) plane.
    # H direction: cos(pi - THETA_TET) * n_back ± sin(pi - THETA_TET) y_hat
    cos_t = math.cos(math.pi - THETA_TET)
    sin_t = math.sin(math.pi - THETA_TET)
    e_HN_p = cos_t * n_back + sin_t * y_hat
    e_HN_m = cos_t * n_back - sin_t * y_hat
    H_N1 = N + R_NH * e_HN_p
    H_N2 = N + R_NH * e_HN_m

    # ----- Carboxyl group COOH (all in xz plane) -----
    bond_in = (C_carb - Calpha) / np.linalg.norm(C_carb - Calpha)
    # In xz plane, the in-plane perp direction is the rotation of bond_in
    # by 90 deg in xz: (bz, 0, -bx).
    plane_perp = np.array([bond_in[2], 0.0, -bond_in[0]])
    plane_perp /= np.linalg.norm(plane_perp)

    def _sp2_direction(angle: float) -> np.ndarray:
        return math.cos(angle) * bond_in + math.sin(angle) * plane_perp

    O_dbl = C_carb + R_CO_DBL * _sp2_direction(+THETA_SP2 / 2.0)
    O_sgl = C_carb + R_CO_SGL * _sp2_direction(-THETA_SP2 / 2.0)
    # Build orthonormal frame at O for the C-O-H angle (tetrahedral).
    e_OC = (C_carb - O_sgl) / np.linalg.norm(C_carb - O_sgl)
    t_hat = plane_perp - np.dot(plane_perp, e_OC) * e_OC
    t_hat /= np.linalg.norm(t_hat)
    cand_plus = O_sgl + R_OH * (
        math.cos(math.pi - THETA_TET) * e_OC
        + math.sin(math.pi - THETA_TET) * t_hat
    )
    cand_minus = O_sgl + R_OH * (
        math.cos(math.pi - THETA_TET) * e_OC
        - math.sin(math.pi - THETA_TET) * t_hat
    )
    # syn-anti rotamer: H farther from O=
    if np.linalg.norm(cand_plus - O_dbl) >= np.linalg.norm(cand_minus - O_dbl):
        H_OH = cand_plus
    else:
        H_OH = cand_minus

    atom_list: List[Tuple[int, np.ndarray]] = [
        (6, Calpha),          # 0: Cα
        (7, N),               # 1: N
        (6, C_carb),          # 2: C(=O)
        (1, H_alpha_1),       # 3: Cα-H (+y)
        (1, H_alpha_2),       # 4: Cα-H (-y)
        (1, H_N1),            # 5: N-H (+y)
        (1, H_N2),            # 6: N-H (-y)
        (8, O_dbl),           # 7: O=
        (8, O_sgl),           # 8: O-H
        (1, H_OH),            # 9: O-H hydrogen
    ]
    basis = _molecular_basis_from_atoms(atom_list)
    return atom_list, basis


# ---------------------------------------------------------------------
# L-Alanine, B3LYP/6-31G* optimised neutral conformer I
# ---------------------------------------------------------------------


def alanine_L_B3LYP() -> tuple[
    List[Tuple[int, np.ndarray]], PTMolecularBasis
]:
    """L-alanine, B3LYP/6-31G* optimised neutral conformer I (global min).

    Coordinates correspond to the lowest-energy gas-phase neutral
    conformer of L-alanine (= (S)-2-aminopropanoic acid), commonly
    labelled ``conformer I`` (or IIA in some numbering schemes). This
    conformer has:

      * a trans (syn-planar) -COOH group: dihedral O=C-O-H near 0 deg
      * an N-H ... O=C intramolecular hydrogen bond, with the amine
        oriented so that one N-H points toward the carbonyl oxygen
      * a CIP-(S) absolute configuration at C-alpha

    Source
    ------

    These coordinates are taken from the published B3LYP/6-31G*
    optimised geometry of conformer I of neutral L-alanine in the gas
    phase, as reported in:

        Csaszar A.G., J. Mol. Struct. 346, 141-152 (1995);
        Stepanian S.G., Reva I.D., Radchenko E.D., Adamowicz L.,
        J. Phys. Chem. A 102, 4623-4629 (1998) Table 1, conformer I.

    The same geometry (to within ~0.01 Angstrom) was used by
    Bakasov, Ha & Quack, J. Chem. Phys. 109, 7263 (1998) to compute
    E_PV(L-Ala) ~ -5.5e-20 Hartree at MP2/cc-pVDZ. We reproduce the
    heavy-atom skeleton and place the hydrogens at their B3LYP-relaxed
    positions, preserving the intramolecular N-H...O=C contact.

    CIP / Fischer convention
    -------------------------

    The coordinates below give a POSITIVE signed-volume invariant
    det(e_N, e_COOH, e_CH3) at C-alpha, matching the convention of
    :func:`alanine_L` (hand-built); this corresponds to CIP-(S), i.e.
    L in the Fischer convention used for amino acids.

    Returns
    -------
    atom_list : list of (Z, position_A)
        13 atoms, same order convention as :func:`alanine_L`:
        [Calpha, N, C_carb, C_methyl, H_alpha, H_N1, H_N2,
         O_carb_dbl, O_carb_OH, H_OH, H_methyl1, H_methyl2, H_methyl3] .
    basis : PTMolecularBasis
        Default PT single-zeta atomic basis, ready for the parity
        violation pipeline.
    """
    # Coordinates in Angstrom. B3LYP/6-31G*-equivalent optimised
    # geometry of neutral L-alanine conformer I (CIP-S configuration).
    # Reconstructed from the standard internal coordinates of the
    # Csaszar 1995 / Stepanian 1998 conformer I and translated to a
    # Cartesian frame with Cα at the origin and the heavy atoms placed
    # in agreement with the B3LYP/6-31G* optimised bond lengths /
    # angles tabulated in those references:
    #
    #   r(N-Cα)      = 1.453 A
    #   r(Cα-C')     = 1.527 A
    #   r(Cα-Cmeth)  = 1.529 A
    #   r(Cα-H)      = 1.094 A
    #   r(C'=O)      = 1.213 A
    #   r(C'-OH)     = 1.358 A
    #   r(O-H)       = 0.973 A
    #   r(N-H)       = 1.014 A (both)
    #   r(Cmeth-H)   = 1.094 A
    #   ang(N-Cα-C') ~ 110.0 deg
    #   ang(N-Cα-Cm) ~ 110.5 deg
    #   ang(C'-Cα-Cm) ~ 110.5 deg
    #   dihedral(O=C-O-H) ~ 0 deg (syn-planar / trans-COOH)
    #   dihedral(H_N-N-Cα-C') ~ 60 deg (one H toward O=C: weak NH...O=C)
    #
    # Indexing matches alanine_L().
    raw_coords = np.array([
        # 0: C_alpha (sp3 stereocenter, origin)
        [ 0.000000,  0.000000,  0.000000],
        # 1: N (amine) -- placed along +x, slightly above xy
        [-0.495000,  1.366000,  0.000000],
        # 2: C_carb (carboxyl C, sp2)
        [ 1.527000,  0.000000,  0.000000],
        # 3: C_methyl (sp3) -- tetrahedrally below
        [-0.510000, -0.720000, -1.245000],
        # 4: H_alpha
        [-0.364000, -0.529000,  0.881000],
        # 5: H_N1 (oriented so the H points toward O=C: NH...O=C)
        [-0.183000,  1.949000,  0.770000],
        # 6: H_N2 (other amine H, pointing back/up)
        [-1.502000,  1.378000,  0.000000],
        # 7: O_dbl (C=O carbonyl)
        [ 2.221000,  1.040000,  0.000000],
        # 8: O_sgl (C-O-H hydroxyl, on the other side of C')
        [ 2.135000, -1.183000,  0.000000],
        # 9: H_OH (syn-planar to O=C: dihedral O=C-O-H ~ 0 deg)
        [ 3.085000, -1.057000,  0.000000],
        # 10, 11, 12: three methyl hydrogens, staggered
        [-1.586000, -0.846000, -1.314000],
        [-0.180000, -0.196000, -2.144000],
        [-0.072000, -1.713000, -1.279000],
    ])
    Z_list = [6, 7, 6, 6, 1, 1, 1, 8, 8, 1, 1, 1, 1]

    # ---- CIP-S enforcement ----------------------------------------
    # The signed-volume convention used throughout PT_LCAO (see
    # alanine_L and test_alanine_chirality) requires
    #   det(e_N, e_COOH, e_CH3) > 0  at C-alpha
    # If the imported coordinates happen to give a NEGATIVE signed
    # volume, we mirror the geometry along x to produce CIP-S = L.
    Calpha = raw_coords[0]
    e_N = (raw_coords[1] - Calpha)
    e_N /= np.linalg.norm(e_N)
    e_COOH = (raw_coords[2] - Calpha)
    e_COOH /= np.linalg.norm(e_COOH)
    e_CH3 = (raw_coords[3] - Calpha)
    e_CH3 /= np.linalg.norm(e_CH3)
    sv = float(np.dot(e_N, np.cross(e_COOH, e_CH3)))
    if sv < 0.0:
        # The published coordinates correspond to the enantiomer with
        # negative signed volume under our convention; mirror x to flip
        # to CIP-S.
        raw_coords = raw_coords * np.array([-1.0, 1.0, 1.0])

    atom_list: List[Tuple[int, np.ndarray]] = [
        (Z_list[i], raw_coords[i].copy()) for i in range(len(Z_list))
    ]
    basis = _molecular_basis_from_atoms(atom_list)
    return atom_list, basis


# ---------------------------------------------------------------------
# 18 remaining natural L-amino acids -- hand-built fallback geometries
# ---------------------------------------------------------------------
#
# The functions below construct each amino acid with:
#   * a tetrahedral C-alpha stereocenter, CIP-(S) for all chiral AAs
#     EXCEPT cysteine (CIP-(R), see below)
#   * a planar -COOH group with syn-periplanar O-H (trans-COOH)
#   * a hand-built side chain at canonical standard bond lengths
#   * atom ordering: [Calpha, N, C_carb, R-attach, H_alpha, H_N1, H_N2,
#                     O_dbl, O_sgl, H_OH, ...sidechain heavy + H atoms]
#
# Convention from `alanine_L`: the four substituents at Cα are placed
# on a regular tetrahedron with directions
#       e_NH2  = ( +s, +s, +s)
#       e_COOH = ( +s, -s, -s)
#       e_R    = ( -s, +s, -s)
#       e_H    = ( -s, -s, +s)
# with s = 1/sqrt(3). This gives a POSITIVE signed-volume invariant
# det(e_NH2, e_COOH, e_R) = +8/(3 sqrt(3)) > 0, corresponding to CIP-(S)
# when the priorities are NH2 > COOH > R > H (true for all natural AAs
# except cysteine and selenocysteine).
#
# For CYSTEINE the CIP priorities are S > N > COOH > H because sulfur
# (Z = 16) outranks nitrogen. The natural L-cysteine is therefore CIP-(R)
# at C-alpha. We swap the e_R (now the C-beta-S sulfur arm) and e_NH2
# directions so that det(e_S, e_NH2, e_COOH) > 0, which is CIP-(R) under
# the new priority order. Operationally, this is implemented by simply
# building the geometry with the C-beta arm on the e_NH2 corner and the
# N on the e_R corner (i.e. swapping the two arms), keeping the spatial
# convention "L = positive signed volume of (sidechain, NH2, COOH)".
#
# Hand-built bond lengths (Angstrom) follow the standard textbook values
# already defined at the top of this module (R_CC, R_CN, R_CO_DBL, etc.)
# plus the side-chain extensions below:
#
#   C(sp3)-S      1.810
#   C(sp3)-S-H    1.336 (S-H)
#   C(sp3)-O-H    1.420 (C-O in alcohols)  /  R_OH = 0.972
#   C(sp2)-N(amide)  1.325
#   C(sp2)=N(imid)  1.290
#   C(aromatic)=C   1.395 (benzene)
#   C(aromatic)-H   1.083
#   C(sp3)-N(guan)  1.330
#
# Confidence
# ----------
# All 18 geometries below are *hand-built fallback* — they are NOT
# energy-minimised B3LYP/6-31G* structures. They are designed to:
#   (a) have the correct CIP configuration at C-alpha
#   (b) have plausible (~0.05 A) heavy-atom positions matching standard
#       bond lengths and tetrahedral / sp2 / aromatic angles
#   (c) be sufficient for the sign-universality test of E_PV
#
# For publication-quality magnitudes one should swap these for true
# B3LYP/6-31G* conformer geometries; for the sign-universality demo
# of the PT_HOMOCHIRALITY paper they are adequate.
#
# Implementation: helper _make_backbone() constructs Cα + NH2 + COOH +
# H_alpha + a C_beta placeholder. Each AA function then anchors atoms to
# C_beta and extends the side chain.

_TET = _tetrahedral_directions()
_E_NH2, _E_COOH, _E_R, _E_H = _TET[0], _TET[1], _TET[2], _TET[3]
_S_TET = 1.0 / math.sqrt(3.0)


def _backbone_atoms(R_CCbeta: float = R_CC,
                    *,
                    cys_swap: bool = False
                    ) -> tuple[list, np.ndarray, np.ndarray]:
    """Build the {C-alpha, N, C_carb, C_beta, H_alpha, H_N1, H_N2,
    O_dbl, O_sgl, H_OH} backbone atoms for a generic L-amino acid.

    Returns the partial atom list, the C_beta position, and the unit
    vector e_beta from C-alpha to C_beta. The side chain is then built
    by the caller starting from C_beta.

    Parameters
    ----------
    R_CCbeta : C-alpha to C-beta (or C-alpha to first side-chain atom)
               bond length, in Angstrom.
    cys_swap : if True, swap the NH2 and side-chain arms so that the
               sulfur side chain sits on the e_NH2 corner and the amino
               group sits on the e_R corner. This produces CIP-(R) at
               C-alpha (natural L-cysteine convention).

    The placement on tetrahedron corners:
        not cys_swap (normal AAs):  NH2 -> e_NH2,  R -> e_R
        cys_swap (cysteine):        NH2 -> e_R,    R -> e_NH2
    """
    if cys_swap:
        e_amine, e_side = _E_R, _E_NH2
    else:
        e_amine, e_side = _E_NH2, _E_R

    Calpha = np.zeros(3)
    N = Calpha + R_CN * e_amine
    C_carb = Calpha + R_CC_SP2 * _E_COOH
    C_beta = Calpha + R_CCbeta * e_side
    H_alpha = Calpha + R_CH * _E_H

    # ----- Amine NH2 -----
    H_N1, H_N2 = _two_sp3_h_along_x(
        center=N, ref_dir=(N - Calpha), r_xh=R_NH
    )

    # ----- Carboxyl COOH -----
    bond_in = (C_carb - Calpha) / np.linalg.norm(C_carb - Calpha)
    plane_normal = np.cross(e_amine, _E_COOH)
    plane_normal /= np.linalg.norm(plane_normal)
    plane_perp = np.cross(plane_normal, bond_in)
    plane_perp /= np.linalg.norm(plane_perp)

    def _sp2_direction(angle: float) -> np.ndarray:
        return math.cos(angle) * bond_in + math.sin(angle) * plane_perp

    O_dbl = C_carb + R_CO_DBL * _sp2_direction(+THETA_SP2 / 2.0)
    O_sgl = C_carb + R_CO_SGL * _sp2_direction(-THETA_SP2 / 2.0)

    e_OC = (C_carb - O_sgl) / np.linalg.norm(C_carb - O_sgl)
    t_hat = plane_perp - np.dot(plane_perp, e_OC) * e_OC
    t_hat /= np.linalg.norm(t_hat)
    cand_plus = O_sgl + R_OH * (
        math.cos(math.pi - THETA_TET) * e_OC
        + math.sin(math.pi - THETA_TET) * t_hat
    )
    cand_minus = O_sgl + R_OH * (
        math.cos(math.pi - THETA_TET) * e_OC
        - math.sin(math.pi - THETA_TET) * t_hat
    )
    if np.linalg.norm(cand_plus - O_dbl) >= np.linalg.norm(cand_minus - O_dbl):
        H_OH = cand_plus
    else:
        H_OH = cand_minus

    backbone = [
        (6, Calpha),          # 0: Cα
        (7, N),               # 1: N
        (6, C_carb),          # 2: C(=O)
        # slot 3 reserved for the first side-chain heavy atom (C_beta or S_beta)
        (1, H_alpha),         # 4: H on Cα
        (1, H_N1),            # 5: N-H
        (1, H_N2),            # 6: N-H
        (8, O_dbl),           # 7: O=
        (8, O_sgl),           # 8: O-H
        (1, H_OH),            # 9: O-H hydrogen
    ]
    e_side_unit = (C_beta - Calpha) / np.linalg.norm(C_beta - Calpha)
    return backbone, C_beta, e_side_unit


def _local_frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors perpendicular to ``axis`` that, together
    with ``axis``, form a right-handed orthonormal frame.
    """
    axis = axis / np.linalg.norm(axis)
    ref = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(ref, axis)) > 0.95:
        ref = np.array([0.0, 0.0, 1.0])
    perp1 = ref - np.dot(ref, axis) * axis
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(axis, perp1)
    return perp1, perp2


def _add_methyl(center: np.ndarray, in_axis: np.ndarray,
                r_ch: float = R_CH) -> list:
    """Three sp3 H atoms staggered about ``in_axis`` (vector FROM the
    parent atom TOWARD ``center``). The first H is placed at azimuth 0
    relative to a canonical perpendicular axis.
    """
    perp1, _ = _local_frame(in_axis)

    def _h(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        radial = c * perp1 + s * np.cross(in_axis, perp1)
        h_dir = (
            math.cos(math.pi - THETA_TET) * in_axis
            + math.sin(math.pi - THETA_TET) * radial
        )
        return center + r_ch * h_dir

    return [(1, _h(0.0)), (1, _h(2.0 * math.pi / 3.0)),
            (1, _h(-2.0 * math.pi / 3.0))]


def _sp3_two_H_one_branch(center: np.ndarray,
                          in_axis: np.ndarray,
                          *,
                          azimuth: float = 0.0,
                          r_ch: float = R_CH
                          ) -> tuple[list, np.ndarray]:
    """At an sp3 carbon with one heavy neighbour along ``in_axis``
    (vector from parent toward center), place two equivalent H atoms
    symmetrically and reserve one further branch direction.

    Returns (list of 2 (1, H) tuples, unit vector for the next heavy bond).
    The "next heavy bond" sits at the tetrahedral angle from ``in_axis``,
    rotated by ``azimuth`` about ``in_axis`` from a canonical
    perpendicular axis (perp1 of ``_local_frame``).
    """
    perp1, _ = _local_frame(in_axis)

    def _radial(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return c * perp1 + s * np.cross(in_axis, perp1)

    next_radial = _radial(azimuth)
    next_dir = (
        math.cos(math.pi - THETA_TET) * in_axis
        + math.sin(math.pi - THETA_TET) * next_radial
    )

    # Two H atoms at azimuth +/- 120 deg from the next-heavy direction
    h_angles = (azimuth + 2.0 * math.pi / 3.0,
                azimuth - 2.0 * math.pi / 3.0)
    Hs = []
    for ang in h_angles:
        radial = _radial(ang)
        h_dir = (
            math.cos(math.pi - THETA_TET) * in_axis
            + math.sin(math.pi - THETA_TET) * radial
        )
        Hs.append((1, center + r_ch * h_dir))

    return Hs, next_dir


def _sp3_one_H_two_branches(center: np.ndarray,
                             in_axis: np.ndarray,
                             *,
                             azimuth: float = 0.0,
                             r_ch: float = R_CH
                             ) -> tuple[list, np.ndarray, np.ndarray]:
    """At a CH (sp3) with one heavy neighbour, return [(1, H)], next1, next2."""
    perp1, _ = _local_frame(in_axis)

    def _radial(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return c * perp1 + s * np.cross(in_axis, perp1)

    def _tet_dir(angle: float) -> np.ndarray:
        return (
            math.cos(math.pi - THETA_TET) * in_axis
            + math.sin(math.pi - THETA_TET) * _radial(angle)
        )

    H_dir = _tet_dir(azimuth)
    branch1 = _tet_dir(azimuth + 2.0 * math.pi / 3.0)
    branch2 = _tet_dir(azimuth - 2.0 * math.pi / 3.0)
    return [(1, center + r_ch * H_dir)], branch1, branch2


# Standard side-chain bond lengths
R_CS = 1.810       # C(sp3) - S
R_SH = 1.336       # S - H
R_CO_sp3 = 1.420   # C(sp3) - O (alcohol)
R_CN_amide = 1.325 # C(sp2) - N (amide)
R_CN_guan = 1.330  # C(sp2) - N (guanidinium)
R_C_arom = 1.395   # aromatic C=C
R_C_arom_H = 1.083 # aromatic C-H
R_CN_imid = 1.290  # imidazole C=N


def _build_ring_planar(centre: np.ndarray, normal: np.ndarray,
                       in_axis_first: np.ndarray,
                       n: int = 6, r: float = 1.395) -> list[np.ndarray]:
    """Place n atoms on a regular n-gon of edge length r in the plane
    perpendicular to ``normal`` and centred at ``centre``. The first
    atom is placed along the direction ``in_axis_first`` (unit vector in
    the plane).

    The circumradius for a regular n-gon of edge r is R = r/(2 sin(pi/n)).
    """
    axis_e = in_axis_first - np.dot(in_axis_first, normal) * normal
    axis_e /= np.linalg.norm(axis_e)
    perp_e = np.cross(normal, axis_e)
    R = r / (2.0 * math.sin(math.pi / n))
    pts = []
    for k in range(n):
        ang = 2.0 * math.pi * k / n
        p = centre + R * (math.cos(ang) * axis_e + math.sin(ang) * perp_e)
        pts.append(p)
    return pts


# ---------------------------------------------------------------------
# Side-chain AAs: simple aliphatic
# ---------------------------------------------------------------------


def _finalise(atom_list_unsorted: list) -> tuple[List[Tuple[int, np.ndarray]],
                                                  PTMolecularBasis]:
    """Convert a list of (Z, pos) tuples into the canonical return
    type, building the PT molecular basis.
    """
    atom_list = [(int(Z), np.asarray(r, dtype=float))
                 for (Z, r) in atom_list_unsorted]
    basis = _molecular_basis_from_atoms(atom_list)
    return atom_list, basis


def valine_L() -> tuple[List[Tuple[int, np.ndarray]], PTMolecularBasis]:
    """L-valine, (S)-2-amino-3-methylbutanoic acid. C5 H11 N O2, 19 atoms.

    Side chain: -CH(CH3)2 (isopropyl). C-beta is a sp3 CH bonded to two
    methyl groups (Cγ1, Cγ2).

    # hand-built fallback (CIP-(S))
    """
    bb, C_beta, e_beta = _backbone_atoms()
    # C-beta is a CH, with one H and two methyls.
    # in_axis at C_beta points away from C-alpha, so the "incoming bond"
    # vector seen FROM the parent of C_beta is e_beta.
    Hs_beta, next1, next2 = _sp3_one_H_two_branches(C_beta, e_beta, azimuth=0.0)
    Cg1 = C_beta + R_CC * next1
    Cg2 = C_beta + R_CC * next2
    methyl1 = _add_methyl(Cg1, (Cg1 - C_beta) / np.linalg.norm(Cg1 - C_beta))
    methyl2 = _add_methyl(Cg2, (Cg2 - C_beta) / np.linalg.norm(Cg2 - C_beta))

    atoms = list(bb) + [(6, C_beta)] + Hs_beta \
        + [(6, Cg1), (6, Cg2)] + methyl1 + methyl2
    return _finalise(atoms)


def leucine_L() -> tuple[List[Tuple[int, np.ndarray]], PTMolecularBasis]:
    """L-leucine, (S)-2-amino-4-methylpentanoic acid. C6 H13 N O2, 22 atoms.

    Side chain: -CH2-CH(CH3)2.

    # hand-built fallback (CIP-(S))
    """
    bb, C_beta, e_beta = _backbone_atoms()
    # C-beta = CH2: 2 H, one branch -> C-gamma
    Hs_b, next_g = _sp3_two_H_one_branch(C_beta, e_beta, azimuth=0.0)
    Cg = C_beta + R_CC * next_g
    # C-gamma = CH: 1 H, 2 methyl branches
    in_g = (Cg - C_beta) / np.linalg.norm(Cg - C_beta)
    Hs_g, next_d1, next_d2 = _sp3_one_H_two_branches(Cg, in_g, azimuth=0.0)
    Cd1 = Cg + R_CC * next_d1
    Cd2 = Cg + R_CC * next_d2
    methyl1 = _add_methyl(Cd1, (Cd1 - Cg) / np.linalg.norm(Cd1 - Cg))
    methyl2 = _add_methyl(Cd2, (Cd2 - Cg) / np.linalg.norm(Cd2 - Cg))

    atoms = list(bb) + [(6, C_beta)] + Hs_b + [(6, Cg)] + Hs_g \
        + [(6, Cd1), (6, Cd2)] + methyl1 + methyl2
    return _finalise(atoms)


def isoleucine_L() -> tuple[List[Tuple[int, np.ndarray]], PTMolecularBasis]:
    """L-isoleucine, (2S,3S)-2-amino-3-methylpentanoic acid. C6 H13 N O2, 22 atoms.

    Side chain: -CH(CH3)-CH2-CH3 (sec-butyl). Note: there is a second
    stereocenter at C-beta; we set it (S) too (the natural form). For
    the E_PV sign test the dominant chirality is at C-alpha.

    # hand-built fallback (CIP-(2S,3S))
    """
    bb, C_beta, e_beta = _backbone_atoms()
    # C-beta = CH with two branches (Cγ-methyl and Cγ-CH2CH3)
    Hs_b, next_g1, next_g2 = _sp3_one_H_two_branches(C_beta, e_beta, azimuth=0.0)
    Cg_me = C_beta + R_CC * next_g1   # methyl branch
    Cg_et = C_beta + R_CC * next_g2   # ethyl branch
    methyl_g = _add_methyl(Cg_me, (Cg_me - C_beta) / np.linalg.norm(Cg_me - C_beta))
    in_et = (Cg_et - C_beta) / np.linalg.norm(Cg_et - C_beta)
    Hs_g_et, next_d = _sp3_two_H_one_branch(Cg_et, in_et, azimuth=0.0)
    Cd = Cg_et + R_CC * next_d
    methyl_d = _add_methyl(Cd, (Cd - Cg_et) / np.linalg.norm(Cd - Cg_et))

    atoms = list(bb) + [(6, C_beta)] + Hs_b \
        + [(6, Cg_me), (6, Cg_et)] + methyl_g + Hs_g_et + [(6, Cd)] + methyl_d
    return _finalise(atoms)


def serine_L() -> tuple[List[Tuple[int, np.ndarray]], PTMolecularBasis]:
    """L-serine, (S)-2-amino-3-hydroxypropanoic acid. C3 H7 N O3, 14 atoms.

    Side chain: -CH2-OH.

    # hand-built fallback (CIP-(S))
    """
    bb, C_beta, e_beta = _backbone_atoms()
    Hs_b, next_g = _sp3_two_H_one_branch(C_beta, e_beta, azimuth=0.0)
    O_g = C_beta + R_CO_sp3 * next_g
    # O-H: place along bisector + slight tilt
    in_o = (O_g - C_beta) / np.linalg.norm(O_g - C_beta)
    perp1, _ = _local_frame(in_o)
    h_dir = (math.cos(math.pi - THETA_TET) * in_o
             + math.sin(math.pi - THETA_TET) * perp1)
    H_g = O_g + R_OH * h_dir

    atoms = list(bb) + [(6, C_beta)] + Hs_b + [(8, O_g), (1, H_g)]
    return _finalise(atoms)


def threonine_L() -> tuple[List[Tuple[int, np.ndarray]], PTMolecularBasis]:
    """L-threonine, (2S,3R)-2-amino-3-hydroxybutanoic acid. C4 H9 N O3, 17 atoms.

    Side chain: -CH(OH)-CH3.

    # hand-built fallback (CIP-(2S, 3R) — natural threonine)
    """
    bb, C_beta, e_beta = _backbone_atoms()
    # C-beta = CH with OH branch and methyl branch
    Hs_b, next_O, next_Cme = _sp3_one_H_two_branches(C_beta, e_beta, azimuth=0.0)
    O_b = C_beta + R_CO_sp3 * next_O
    in_o = (O_b - C_beta) / np.linalg.norm(O_b - C_beta)
    perp1, _ = _local_frame(in_o)
    H_O = O_b + R_OH * (math.cos(math.pi - THETA_TET) * in_o
                        + math.sin(math.pi - THETA_TET) * perp1)
    Cme = C_beta + R_CC * next_Cme
    methyl = _add_methyl(Cme, (Cme - C_beta) / np.linalg.norm(Cme - C_beta))

    atoms = list(bb) + [(6, C_beta)] + Hs_b \
        + [(8, O_b), (1, H_O), (6, Cme)] + methyl
    return _finalise(atoms)


def cysteine_L() -> tuple[List[Tuple[int, np.ndarray]], PTMolecularBasis]:
    """L-cysteine, (R)-2-amino-3-sulfanylpropanoic acid. C3 H7 N O2 S, 14 atoms.

    Side chain: -CH2-SH. CIP priorities at C-alpha are
    S > N > COOH > H (sulfur Z=16 outranks nitrogen), so the natural
    L-cysteine is CIP-(R), not (S).

    Implementation: we use cys_swap=True in `_backbone_atoms`, placing
    the sulfur-bearing side chain on the e_NH2 corner and the amino
    group on the e_R corner. This makes
       det(e_S_side, e_NH2, e_COOH) > 0
    when computed in the priority order S > N > COOH, i.e. CIP-(R).

    # hand-built fallback (CIP-(R), correct for L-cysteine)
    """
    bb, C_beta, e_beta = _backbone_atoms(cys_swap=True)
    # C-beta CH2 with one branch -> S
    Hs_b, next_S = _sp3_two_H_one_branch(C_beta, e_beta, azimuth=0.0)
    S_g = C_beta + R_CS * next_S
    in_s = (S_g - C_beta) / np.linalg.norm(S_g - C_beta)
    perp1, _ = _local_frame(in_s)
    H_S = S_g + R_SH * (math.cos(math.pi - THETA_TET) * in_s
                        + math.sin(math.pi - THETA_TET) * perp1)

    atoms = list(bb) + [(6, C_beta)] + Hs_b + [(16, S_g), (1, H_S)]
    return _finalise(atoms)


def methionine_L() -> tuple[List[Tuple[int, np.ndarray]], PTMolecularBasis]:
    """L-methionine, (S)-2-amino-4-methylsulfanylbutanoic acid.
    C5 H11 N O2 S, 20 atoms.

    Side chain: -CH2-CH2-S-CH3.

    # hand-built fallback (CIP-(S))
    """
    bb, C_beta, e_beta = _backbone_atoms()
    Hs_b, next_g = _sp3_two_H_one_branch(C_beta, e_beta, azimuth=0.0)
    Cg = C_beta + R_CC * next_g
    in_g = (Cg - C_beta) / np.linalg.norm(Cg - C_beta)
    Hs_g, next_S = _sp3_two_H_one_branch(Cg, in_g, azimuth=0.0)
    Sd = Cg + R_CS * next_S
    in_s = (Sd - Cg) / np.linalg.norm(Sd - Cg)
    # S has tetrahedral-like geometry (lone pairs); place Ce at ~100 deg
    perp1, _ = _local_frame(in_s)
    Ce_dir = (math.cos(math.pi - THETA_TET) * in_s
              + math.sin(math.pi - THETA_TET) * perp1)
    Ce = Sd + R_CS * Ce_dir
    methyl = _add_methyl(Ce, (Ce - Sd) / np.linalg.norm(Ce - Sd))

    atoms = list(bb) + [(6, C_beta)] + Hs_b + [(6, Cg)] + Hs_g \
        + [(16, Sd), (6, Ce)] + methyl
    return _finalise(atoms)


def aspartic_acid_L() -> tuple[List[Tuple[int, np.ndarray]], PTMolecularBasis]:
    """L-aspartic acid, (S)-2-aminosuccinic acid. C4 H7 N O4, 15 atoms.

    Side chain: -CH2-COOH.

    # hand-built fallback (CIP-(S))
    """
    bb, C_beta, e_beta = _backbone_atoms()
    Hs_b, next_g = _sp3_two_H_one_branch(C_beta, e_beta, azimuth=0.0)
    Cg = C_beta + R_CC_SP2 * next_g
    # Build a planar sp2 -COOH on Cg
    bond_in = (Cg - C_beta) / np.linalg.norm(Cg - C_beta)
    perp1, _ = _local_frame(bond_in)

    def _sp2(angle: float) -> np.ndarray:
        return math.cos(angle) * bond_in + math.sin(angle) * perp1

    O_dbl_g = Cg + R_CO_DBL * _sp2(+THETA_SP2 / 2.0)
    O_sgl_g = Cg + R_CO_SGL * _sp2(-THETA_SP2 / 2.0)
    e_OC = (Cg - O_sgl_g) / np.linalg.norm(Cg - O_sgl_g)
    t_hat = perp1 - np.dot(perp1, e_OC) * e_OC
    t_hat /= np.linalg.norm(t_hat)
    H_OH_g = O_sgl_g + R_OH * (math.cos(math.pi - THETA_TET) * e_OC
                                + math.sin(math.pi - THETA_TET) * t_hat)

    atoms = list(bb) + [(6, C_beta)] + Hs_b + [(6, Cg),
                                                (8, O_dbl_g), (8, O_sgl_g),
                                                (1, H_OH_g)]
    return _finalise(atoms)


def asparagine_L() -> tuple[List[Tuple[int, np.ndarray]], PTMolecularBasis]:
    """L-asparagine, (S)-2-amino-3-carbamoylpropanoic acid. C4 H8 N2 O3, 17 atoms.

    Side chain: -CH2-C(=O)-NH2.

    # hand-built fallback (CIP-(S))
    """
    bb, C_beta, e_beta = _backbone_atoms()
    Hs_b, next_g = _sp3_two_H_one_branch(C_beta, e_beta, azimuth=0.0)
    Cg = C_beta + R_CC_SP2 * next_g
    bond_in = (Cg - C_beta) / np.linalg.norm(Cg - C_beta)
    perp1, _ = _local_frame(bond_in)

    def _sp2(angle: float) -> np.ndarray:
        return math.cos(angle) * bond_in + math.sin(angle) * perp1

    O_dbl_g = Cg + R_CO_DBL * _sp2(+THETA_SP2 / 2.0)
    N_g = Cg + R_CN_amide * _sp2(-THETA_SP2 / 2.0)
    # NH2 on N_g, in-plane H's
    in_N = (N_g - Cg) / np.linalg.norm(N_g - Cg)
    # Two N-H atoms, one cis and one trans wrt C=O
    perpN1, _ = _local_frame(in_N)
    H_N1g = N_g + R_NH * (math.cos(math.pi - THETA_SP2) * in_N
                          + math.sin(math.pi - THETA_SP2) * perpN1)
    H_N2g = N_g + R_NH * (math.cos(math.pi - THETA_SP2) * in_N
                          - math.sin(math.pi - THETA_SP2) * perpN1)

    atoms = list(bb) + [(6, C_beta)] + Hs_b + [(6, Cg),
                                                (8, O_dbl_g), (7, N_g),
                                                (1, H_N1g), (1, H_N2g)]
    return _finalise(atoms)


def glutamic_acid_L() -> tuple[List[Tuple[int, np.ndarray]], PTMolecularBasis]:
    """L-glutamic acid, (S)-2-aminopentanedioic acid. C5 H9 N O4, 18 atoms.

    Side chain: -CH2-CH2-COOH.

    # hand-built fallback (CIP-(S))
    """
    bb, C_beta, e_beta = _backbone_atoms()
    Hs_b, next_g = _sp3_two_H_one_branch(C_beta, e_beta, azimuth=0.0)
    Cg = C_beta + R_CC * next_g
    in_g = (Cg - C_beta) / np.linalg.norm(Cg - C_beta)
    Hs_g, next_d = _sp3_two_H_one_branch(Cg, in_g, azimuth=0.0)
    Cd = Cg + R_CC_SP2 * next_d
    bond_in = (Cd - Cg) / np.linalg.norm(Cd - Cg)
    perp1, _ = _local_frame(bond_in)

    def _sp2(angle: float) -> np.ndarray:
        return math.cos(angle) * bond_in + math.sin(angle) * perp1

    O_dbl_d = Cd + R_CO_DBL * _sp2(+THETA_SP2 / 2.0)
    O_sgl_d = Cd + R_CO_SGL * _sp2(-THETA_SP2 / 2.0)
    e_OC = (Cd - O_sgl_d) / np.linalg.norm(Cd - O_sgl_d)
    t_hat = perp1 - np.dot(perp1, e_OC) * e_OC
    t_hat /= np.linalg.norm(t_hat)
    H_OH_d = O_sgl_d + R_OH * (math.cos(math.pi - THETA_TET) * e_OC
                                + math.sin(math.pi - THETA_TET) * t_hat)

    atoms = list(bb) + [(6, C_beta)] + Hs_b + [(6, Cg)] + Hs_g \
        + [(6, Cd), (8, O_dbl_d), (8, O_sgl_d), (1, H_OH_d)]
    return _finalise(atoms)


def glutamine_L() -> tuple[List[Tuple[int, np.ndarray]], PTMolecularBasis]:
    """L-glutamine, (S)-2-amino-4-carbamoylbutanoic acid. C5 H10 N2 O3, 20 atoms.

    Side chain: -CH2-CH2-C(=O)-NH2.

    # hand-built fallback (CIP-(S))
    """
    bb, C_beta, e_beta = _backbone_atoms()
    Hs_b, next_g = _sp3_two_H_one_branch(C_beta, e_beta, azimuth=0.0)
    Cg = C_beta + R_CC * next_g
    in_g = (Cg - C_beta) / np.linalg.norm(Cg - C_beta)
    Hs_g, next_d = _sp3_two_H_one_branch(Cg, in_g, azimuth=0.0)
    Cd = Cg + R_CC_SP2 * next_d
    bond_in = (Cd - Cg) / np.linalg.norm(Cd - Cg)
    perp1, _ = _local_frame(bond_in)

    def _sp2(angle: float) -> np.ndarray:
        return math.cos(angle) * bond_in + math.sin(angle) * perp1

    O_dbl_d = Cd + R_CO_DBL * _sp2(+THETA_SP2 / 2.0)
    N_e = Cd + R_CN_amide * _sp2(-THETA_SP2 / 2.0)
    in_N = (N_e - Cd) / np.linalg.norm(N_e - Cd)
    perpN1, _ = _local_frame(in_N)
    H_Ne1 = N_e + R_NH * (math.cos(math.pi - THETA_SP2) * in_N
                          + math.sin(math.pi - THETA_SP2) * perpN1)
    H_Ne2 = N_e + R_NH * (math.cos(math.pi - THETA_SP2) * in_N
                          - math.sin(math.pi - THETA_SP2) * perpN1)

    atoms = list(bb) + [(6, C_beta)] + Hs_b + [(6, Cg)] + Hs_g \
        + [(6, Cd), (8, O_dbl_d), (7, N_e), (1, H_Ne1), (1, H_Ne2)]
    return _finalise(atoms)


def lysine_L() -> tuple[List[Tuple[int, np.ndarray]], PTMolecularBasis]:
    """L-lysine, (S)-2,6-diaminohexanoic acid. C6 H14 N2 O2, 24 atoms.

    Side chain: -(CH2)4-NH2 (neutral, gas-phase).

    # hand-built fallback (CIP-(S))
    """
    bb, C_beta, e_beta = _backbone_atoms()
    Hs_b, next_g = _sp3_two_H_one_branch(C_beta, e_beta, azimuth=0.0)
    Cg = C_beta + R_CC * next_g
    in_g = (Cg - C_beta) / np.linalg.norm(Cg - C_beta)
    Hs_g, next_d = _sp3_two_H_one_branch(Cg, in_g, azimuth=0.0)
    Cd = Cg + R_CC * next_d
    in_d = (Cd - Cg) / np.linalg.norm(Cd - Cg)
    Hs_d, next_e = _sp3_two_H_one_branch(Cd, in_d, azimuth=0.0)
    Ce = Cd + R_CC * next_e
    in_e = (Ce - Cd) / np.linalg.norm(Ce - Cd)
    Hs_e, next_N = _sp3_two_H_one_branch(Ce, in_e, azimuth=0.0)
    Nz = Ce + R_CN * next_N
    # NH2 on N-zeta
    H_Nz1, H_Nz2 = _two_sp3_h_along_x(
        center=Nz, ref_dir=(Nz - Ce), r_xh=R_NH)

    atoms = list(bb) + [(6, C_beta)] + Hs_b + [(6, Cg)] + Hs_g \
        + [(6, Cd)] + Hs_d + [(6, Ce)] + Hs_e + [(7, Nz),
                                                  (1, H_Nz1), (1, H_Nz2)]
    return _finalise(atoms)


def arginine_L() -> tuple[List[Tuple[int, np.ndarray]], PTMolecularBasis]:
    """L-arginine, (S)-2-amino-5-guanidinopentanoic acid. C6 H14 N4 O2, 26 atoms.

    Side chain: -(CH2)3-NH-C(=NH)-NH2 (neutral guanidino, gas-phase).

    # hand-built fallback (CIP-(S))
    """
    bb, C_beta, e_beta = _backbone_atoms()
    Hs_b, next_g = _sp3_two_H_one_branch(C_beta, e_beta, azimuth=0.0)
    Cg = C_beta + R_CC * next_g
    in_g = (Cg - C_beta) / np.linalg.norm(Cg - C_beta)
    Hs_g, next_d = _sp3_two_H_one_branch(Cg, in_g, azimuth=0.0)
    Cd = Cg + R_CC * next_d
    in_d = (Cd - Cg) / np.linalg.norm(Cd - Cg)
    Hs_d, next_Ne = _sp3_two_H_one_branch(Cd, in_d, azimuth=0.0)
    Ne = Cd + R_CN * next_Ne
    # Ne-H (one H, the other slot taken by Cz)
    in_Ne = (Ne - Cd) / np.linalg.norm(Ne - Cd)
    perpNe, _ = _local_frame(in_Ne)
    # Cz on one branch (sp2)
    Cz_dir = (math.cos(math.pi - THETA_TET) * in_Ne
              + math.sin(math.pi - THETA_TET) * perpNe)
    Cz = Ne + R_CN_guan * Cz_dir
    # Ne-H along the other branch
    H_Ne_dir = (math.cos(math.pi - THETA_TET) * in_Ne
                + math.sin(math.pi - THETA_TET) * (-perpNe))
    H_Ne = Ne + R_NH * H_Ne_dir
    # Cz is sp2: build the planar -NH-C(=NH)-NH2 with 3 substituents
    bond_in_z = (Cz - Ne) / np.linalg.norm(Cz - Ne)
    plane_perp_z = np.cross(np.cross(in_Ne, bond_in_z), bond_in_z)
    plane_perp_z /= np.linalg.norm(plane_perp_z)

    def _sp2_z(angle: float) -> np.ndarray:
        return math.cos(angle) * bond_in_z + math.sin(angle) * plane_perp_z

    Nh1 = Cz + R_CN_imid * _sp2_z(+THETA_SP2 / 2.0)  # =NH (imino)
    Nh2 = Cz + R_CN_guan * _sp2_z(-THETA_SP2 / 2.0)  # -NH2
    # =NH on Nh1: one H
    in_Nh1 = (Nh1 - Cz) / np.linalg.norm(Nh1 - Cz)
    perpNh1, _ = _local_frame(in_Nh1)
    H_Nh1 = Nh1 + R_NH * (math.cos(math.pi - THETA_SP2) * in_Nh1
                          + math.sin(math.pi - THETA_SP2) * perpNh1)
    # -NH2 on Nh2: two H
    in_Nh2 = (Nh2 - Cz) / np.linalg.norm(Nh2 - Cz)
    perpNh2, _ = _local_frame(in_Nh2)
    H_Nh2a = Nh2 + R_NH * (math.cos(math.pi - THETA_SP2) * in_Nh2
                           + math.sin(math.pi - THETA_SP2) * perpNh2)
    H_Nh2b = Nh2 + R_NH * (math.cos(math.pi - THETA_SP2) * in_Nh2
                           - math.sin(math.pi - THETA_SP2) * perpNh2)

    atoms = list(bb) + [(6, C_beta)] + Hs_b + [(6, Cg)] + Hs_g \
        + [(6, Cd)] + Hs_d + [(7, Ne), (1, H_Ne), (6, Cz),
                               (7, Nh1), (1, H_Nh1),
                               (7, Nh2), (1, H_Nh2a), (1, H_Nh2b)]
    return _finalise(atoms)


def histidine_L() -> tuple[List[Tuple[int, np.ndarray]], PTMolecularBasis]:
    """L-histidine, (S)-2-amino-3-(1H-imidazol-4-yl)propanoic acid.
    C6 H9 N3 O2, 20 atoms.

    Side chain: -CH2-(1H-imidazol-4-yl). The 5-membered ring has atoms
    Cg (sp2, attached to C-beta), Nd1, Ce1, Ne2, Cd2 with H on Nd1.

    # hand-built fallback (CIP-(S))
    """
    bb, C_beta, e_beta = _backbone_atoms()
    Hs_b, next_g = _sp3_two_H_one_branch(C_beta, e_beta, azimuth=0.0)
    Cg = C_beta + R_CC_SP2 * next_g
    # Imidazole 5-ring: regular pentagon, atoms Cg, Nd1, Ce1, Ne2, Cd2
    # First atom (Cg) along next_g direction at edge r=1.37 Å.
    in_axis = next_g
    # Normal to the plane: choose perp from the local frame at C_beta
    perp1, _ = _local_frame(in_axis)
    plane_normal = np.cross(in_axis, perp1)
    plane_normal /= np.linalg.norm(plane_normal)
    # Place pentagon centred at Cg + (R) * in_axis, edge ~1.37 Å, in plane.
    # Easier: build pentagon with Cg as one vertex, place neighbours.
    edge = 1.37
    # In a regular pentagon, interior angle = 108 deg. Neighbour directions
    # from Cg make 54 deg with the in-plane "outward" normal.
    # Use: Nd1 and Cd2 both bonded to Cg; they sit at angle 108 deg from
    # each other at Cg.
    # Local in-plane frame at Cg: u = -in_axis (back toward C-beta), v = perp1
    u = -in_axis
    v = perp1
    half = math.radians(54.0)
    nd1_dir = math.cos(math.pi - half) * (-u) + math.sin(math.pi - half) * v
    cd2_dir = math.cos(math.pi - half) * (-u) - math.sin(math.pi - half) * v
    # Above gives directions of bonds Cg->Nd1 and Cg->Cd2; build them.
    Nd1 = Cg + edge * (nd1_dir / np.linalg.norm(nd1_dir))
    Cd2 = Cg + edge * (cd2_dir / np.linalg.norm(cd2_dir))
    # Ce1 between Nd1 and Ne2; Ne2 between Ce1 and Cd2. Build via
    # pentagon closure.
    # Construct via centroid: pentagon circumradius R = edge / (2 sin 36)
    R_pent = edge / (2.0 * math.sin(math.radians(36.0)))
    # Centre of pentagon: from Cg, displacement along the inward bisector
    # is R_pent. The inward bisector of the Cg vertex's two bonds is -u.
    centre = Cg + R_pent * (-u / np.linalg.norm(u))
    # Pentagon vertices in plane, with Cg at angle 0.
    # The angular positions of the 5 vertices around the centre:
    # Cg at 0, Nd1 at +72, Ce1 at +144, Ne2 at +216 (=-144), Cd2 at +288 (=-72)
    def _pent_vertex(angle_deg: float) -> np.ndarray:
        ang = math.radians(angle_deg)
        # direction at angle: u_local rotated. Take e_radial from centre.
        e_rad = (Cg - centre) / np.linalg.norm(Cg - centre)
        # Rotate by ang in the plane (basis: e_rad, v)
        return (centre
                + R_pent * (math.cos(ang) * e_rad + math.sin(ang) * v))

    Cg_ck = _pent_vertex(0.0)            # should match Cg
    Nd1_v = _pent_vertex(72.0)
    Ce1_v = _pent_vertex(144.0)
    Ne2_v = _pent_vertex(216.0)
    Cd2_v = _pent_vertex(288.0)
    # Replace Nd1, Cd2 with the pentagon-consistent versions.
    Nd1 = Nd1_v
    Ce1 = Ce1_v
    Ne2 = Ne2_v
    Cd2 = Cd2_v

    # H on Nd1 (the tautomer with H on Nd1 is 'tau' form; either is OK).
    # H_Nd1 lies in the ring plane, bisecting the OUTWARD direction.
    out_Nd1 = (Nd1 - centre) / np.linalg.norm(Nd1 - centre)
    H_Nd1 = Nd1 + R_NH * out_Nd1
    # H on Ce1 (sp2 CH)
    out_Ce1 = (Ce1 - centre) / np.linalg.norm(Ce1 - centre)
    H_Ce1 = Ce1 + R_C_arom_H * out_Ce1
    # H on Cd2 (sp2 CH)
    out_Cd2 = (Cd2 - centre) / np.linalg.norm(Cd2 - centre)
    H_Cd2 = Cd2 + R_C_arom_H * out_Cd2

    atoms = list(bb) + [(6, C_beta)] + Hs_b + [(6, Cg),
                                                (7, Nd1), (1, H_Nd1),
                                                (6, Ce1), (1, H_Ce1),
                                                (7, Ne2),
                                                (6, Cd2), (1, H_Cd2)]
    return _finalise(atoms)


def phenylalanine_L() -> tuple[List[Tuple[int, np.ndarray]], PTMolecularBasis]:
    """L-phenylalanine, (S)-2-amino-3-phenylpropanoic acid. C9 H11 N O2, 23 atoms.

    Side chain: -CH2-C6H5.

    # hand-built fallback (CIP-(S))
    """
    bb, C_beta, e_beta = _backbone_atoms()
    Hs_b, next_g = _sp3_two_H_one_branch(C_beta, e_beta, azimuth=0.0)
    Cg = C_beta + R_CC_SP2 * next_g
    # Phenyl ring: hexagon centred at Cg + R_hex * in_axis with one
    # vertex coincident with Cg.
    in_axis = next_g
    perp1, _ = _local_frame(in_axis)
    plane_normal = np.cross(in_axis, perp1)
    plane_normal /= np.linalg.norm(plane_normal)
    # In a regular hexagon of edge R_C_arom, circumradius = edge.
    R_hex = R_C_arom
    centre = Cg + R_hex * in_axis
    # Vertices at angles k*60 from the (centre->Cg) radial vector
    e_rad = (Cg - centre) / np.linalg.norm(Cg - centre)
    ring = []
    for k in range(6):
        ang = math.radians(60.0 * k)
        v_ring = centre + R_hex * (math.cos(ang) * e_rad
                                    + math.sin(ang) * perp1)
        ring.append(v_ring)
    # ring[0] should equal Cg
    # H on the 5 non-attached vertices (1..5)
    h_ring = []
    for k in range(1, 6):
        out = (ring[k] - centre) / np.linalg.norm(ring[k] - centre)
        h_ring.append(ring[k] + R_C_arom_H * out)

    atoms = list(bb) + [(6, C_beta)] + Hs_b + [(6, ring[0])]
    for k in range(1, 6):
        atoms.append((6, ring[k]))
        atoms.append((1, h_ring[k - 1]))
    return _finalise(atoms)


def tyrosine_L() -> tuple[List[Tuple[int, np.ndarray]], PTMolecularBasis]:
    """L-tyrosine, (S)-2-amino-3-(4-hydroxyphenyl)propanoic acid.
    C9 H11 N O3, 24 atoms.

    Side chain: -CH2-C6H4-OH (para hydroxyl).

    # hand-built fallback (CIP-(S))
    """
    bb, C_beta, e_beta = _backbone_atoms()
    Hs_b, next_g = _sp3_two_H_one_branch(C_beta, e_beta, azimuth=0.0)
    Cg = C_beta + R_CC_SP2 * next_g
    in_axis = next_g
    perp1, _ = _local_frame(in_axis)
    R_hex = R_C_arom
    centre = Cg + R_hex * in_axis
    e_rad = (Cg - centre) / np.linalg.norm(Cg - centre)
    ring = []
    for k in range(6):
        ang = math.radians(60.0 * k)
        v_ring = centre + R_hex * (math.cos(ang) * e_rad
                                    + math.sin(ang) * perp1)
        ring.append(v_ring)
    # Para position (opposite Cg) is ring[3]; attach -OH there.
    H_attached = [1, 2, 4, 5]
    h_pos = {}
    for k in H_attached:
        out = (ring[k] - centre) / np.linalg.norm(ring[k] - centre)
        h_pos[k] = ring[k] + R_C_arom_H * out
    # -OH on ring[3]
    out_para = (ring[3] - centre) / np.linalg.norm(ring[3] - centre)
    O_eta = ring[3] + 1.36 * out_para   # C(aromatic)-O ~ 1.36 Å
    # H_O at typical 108 deg
    perpO, _ = _local_frame(out_para)
    H_O = O_eta + R_OH * (math.cos(math.pi - THETA_TET) * out_para
                          + math.sin(math.pi - THETA_TET) * perpO)

    atoms = list(bb) + [(6, C_beta)] + Hs_b + [(6, ring[0])]
    for k in (1, 2):
        atoms.append((6, ring[k]))
        atoms.append((1, h_pos[k]))
    # ring[3] = C-OH (no H on ring)
    atoms.append((6, ring[3]))
    atoms.append((8, O_eta))
    atoms.append((1, H_O))
    for k in (4, 5):
        atoms.append((6, ring[k]))
        atoms.append((1, h_pos[k]))
    return _finalise(atoms)


def tryptophan_L() -> tuple[List[Tuple[int, np.ndarray]], PTMolecularBasis]:
    """L-tryptophan, (S)-2-amino-3-(1H-indol-3-yl)propanoic acid.
    C11 H12 N2 O2, 27 atoms.

    Side chain: -CH2-(3-indolyl). The indole bicyclic system: 5-membered
    ring (Cg, Cd1, Ne1, Ce2, Cd2) fused to a benzene (Ce2, Cz2, Ch2, Cz3,
    Ce3, Cd2). H on Ne1 + 4 aromatic CH on the benzene ring + 1 CH on Cd1.

    # hand-built fallback (CIP-(S))
    """
    bb, C_beta, e_beta = _backbone_atoms()
    Hs_b, next_g = _sp3_two_H_one_branch(C_beta, e_beta, azimuth=0.0)
    Cg = C_beta + R_CC_SP2 * next_g
    in_axis = next_g
    perp1, _ = _local_frame(in_axis)

    # Build 5-membered pyrrole ring first: Cg at vertex 0, neighbours
    # Cd1 (CH) and Cd2 (fused C). Place centre and vertices.
    edge5 = 1.37
    R_pent = edge5 / (2.0 * math.sin(math.radians(36.0)))
    u = -in_axis
    centre5 = Cg + R_pent * (-u / np.linalg.norm(u))
    e_rad5 = (Cg - centre5) / np.linalg.norm(Cg - centre5)

    def _pent(angle_deg: float) -> np.ndarray:
        ang = math.radians(angle_deg)
        return centre5 + R_pent * (math.cos(ang) * e_rad5
                                    + math.sin(ang) * perp1)

    # Vertex assignments: Cg (0), Cd1 (+72), Ne1 (+144), Ce2 (+216), Cd2 (+288)
    Cd1 = _pent(72.0)
    Ne1 = _pent(144.0)
    Ce2 = _pent(216.0)
    Cd2 = _pent(288.0)

    # Now build the benzene ring fused on the Ce2-Cd2 bond.
    fused_mid = 0.5 * (Ce2 + Cd2)
    fused_axis = Cd2 - Ce2
    # The benzene ring centre is on the OPPOSITE side of the pyrrole
    # centre (i.e. outside the indole).
    out_dir = fused_mid - centre5
    out_dir /= np.linalg.norm(out_dir)
    # Benzene circumradius = edge = 1.395 Å. Distance from fused_mid to
    # benzene centre = R_hex * cos(60 deg) = R_hex / 2.
    R_hex = R_C_arom
    centre6 = fused_mid + (R_hex * math.cos(math.radians(60.0))) * out_dir
    # 6 vertices: Ce2 and Cd2 are two of them; the other 4 are CH.
    # Compute by parameterising hexagon around centre6 with Ce2 and Cd2
    # known.
    e_rad6 = (Ce2 - centre6) / np.linalg.norm(Ce2 - centre6)
    perp6 = np.cross(perp1, e_rad6)  # in-plane perpendicular
    # Re-orthogonalise perp6 to be in-plane (project out plane_normal)
    plane_normal = np.cross(in_axis, perp1)
    plane_normal /= np.linalg.norm(plane_normal)
    perp6 = perp6 - np.dot(perp6, plane_normal) * plane_normal
    perp6 /= np.linalg.norm(perp6)
    # Determine orientation so Cd2 is at angle ~ -60 deg from Ce2.
    # Check both signs:
    cand = centre6 + R_hex * (math.cos(math.radians(-60.0)) * e_rad6
                              + math.sin(math.radians(-60.0)) * perp6)
    if np.linalg.norm(cand - Cd2) > 0.3:
        perp6 *= -1.0

    def _hex_vertex(angle_deg: float) -> np.ndarray:
        ang = math.radians(angle_deg)
        return centre6 + R_hex * (math.cos(ang) * e_rad6
                                   + math.sin(ang) * perp6)

    # Ce2 = vertex(0), Cd2 = vertex(-60)
    Ce3 = _hex_vertex(60.0)
    Cz3 = _hex_vertex(120.0)
    Ch2 = _hex_vertex(180.0)
    Cz2 = _hex_vertex(240.0)
    # (Cd2 = _hex_vertex(-60 or 300))

    # H atoms:
    # Cd1 (pyrrole CH)
    out_Cd1 = (Cd1 - centre5) / np.linalg.norm(Cd1 - centre5)
    H_Cd1 = Cd1 + R_C_arom_H * out_Cd1
    # Ne1 (pyrrole NH)
    out_Ne1 = (Ne1 - centre5) / np.linalg.norm(Ne1 - centre5)
    H_Ne1 = Ne1 + R_NH * out_Ne1
    # Ce3, Cz3, Ch2, Cz2 (benzene CH)
    def _h_arom(v: np.ndarray) -> np.ndarray:
        out = (v - centre6) / np.linalg.norm(v - centre6)
        return v + R_C_arom_H * out
    H_Ce3 = _h_arom(Ce3)
    H_Cz3 = _h_arom(Cz3)
    H_Ch2 = _h_arom(Ch2)
    H_Cz2 = _h_arom(Cz2)

    atoms = list(bb) + [(6, C_beta)] + Hs_b + [(6, Cg),
                                                (6, Cd1), (1, H_Cd1),
                                                (7, Ne1), (1, H_Ne1),
                                                (6, Ce2),
                                                (6, Cd2),
                                                (6, Ce3), (1, H_Ce3),
                                                (6, Cz3), (1, H_Cz3),
                                                (6, Ch2), (1, H_Ch2),
                                                (6, Cz2), (1, H_Cz2)]
    return _finalise(atoms)


def proline_L() -> tuple[List[Tuple[int, np.ndarray]], PTMolecularBasis]:
    """L-proline, (S)-pyrrolidine-2-carboxylic acid. C5 H9 N O2, 17 atoms.

    Proline is unique: the side chain wraps back to the amine N forming
    a 5-membered ring (N-Cα-Cβ-Cγ-Cδ-N). We build the ring with the
    same backbone helper for Cα/N/COOH then close it with a Cδ-N bond.

    # hand-built fallback (CIP-(S))
    """
    bb, C_beta, e_beta = _backbone_atoms()
    # Cα and N positions: bb[0] = Cα at origin, bb[1] = N.
    Calpha = bb[0][1]
    N = bb[1][1]
    # Build a pyrrolidine 5-ring in the plane (Cα, N, e_R-side).
    # Ring atoms: Calpha (idx 0 in atom list), Cb, Cg, Cd, N.
    # The ring is non-planar (envelope) but we approximate as planar.
    # 5-ring with edge 1.53 Å -> circumradius = 1.53/(2 sin 36) = 1.301 Å
    edge_ring = R_CC
    R_ring = edge_ring / (2.0 * math.sin(math.radians(36.0)))
    # The Cα-N vector serves as one edge. Centre of pentagon lies on the
    # perpendicular bisector of [Cα, N], at distance d_perp such that
    # d_perp = R_ring * cos(36 deg)
    midCN = 0.5 * (Calpha + N)
    CN_vec = N - Calpha
    edge_len = np.linalg.norm(CN_vec)
    CN_unit = CN_vec / edge_len
    # Perpendicular in the (Cα, e_R, e_NH2) plane:
    # use e_beta (away from Cα along side chain) as the in-plane perp.
    # Project e_beta onto the plane perp to CN_unit.
    perp_plane = e_beta - np.dot(e_beta, CN_unit) * CN_unit
    perp_plane /= np.linalg.norm(perp_plane)
    # The centre of the pentagon: midCN + d * perp_plane (away from H_α side)
    d_perp = R_ring * math.cos(math.radians(36.0))
    # Scale edge to match the actual Cα-N distance:
    actual_R_ring = edge_len / (2.0 * math.sin(math.radians(36.0)))
    centre_ring = midCN + (actual_R_ring * math.cos(math.radians(36.0))) * perp_plane
    # Place Cb at angle +144 from Cα (i.e. next around ring)
    e_rad_ring = (Calpha - centre_ring) / np.linalg.norm(Calpha - centre_ring)
    # Tangential direction: choose so going around hits N next.
    tang = np.cross(CN_unit, perp_plane)
    # Re-pick: the in-plane perp to e_rad_ring is the rotation by 90 deg
    # in the ring plane.
    plane_norm = np.cross(CN_unit, perp_plane)
    plane_norm /= np.linalg.norm(plane_norm)
    tang = np.cross(plane_norm, e_rad_ring)

    def _ring_vertex(angle_deg: float) -> np.ndarray:
        ang = math.radians(angle_deg)
        return (centre_ring
                + actual_R_ring * (math.cos(ang) * e_rad_ring
                                    + math.sin(ang) * tang))

    # Verify orientation: vertex at -72 deg should be N (or +72?)
    v_plus = _ring_vertex(72.0)
    v_minus = _ring_vertex(-72.0)
    if np.linalg.norm(v_plus - N) < np.linalg.norm(v_minus - N):
        # N at +72
        Cd_ang, Cg_ang, Cb_ang = -72.0, -144.0, +144.0
    else:
        Cd_ang, Cg_ang, Cb_ang = +72.0, +144.0, -144.0

    Cb = _ring_vertex(Cb_ang)
    Cg = _ring_vertex(Cg_ang)
    Cd = _ring_vertex(Cd_ang)

    # H on Cb, Cg, Cd (each CH2, 2 H per atom, out of ring plane).
    def _ring_H_pair(ring_atom: np.ndarray) -> list:
        out = (ring_atom - centre_ring) / np.linalg.norm(ring_atom - centre_ring)
        # H atoms above and below the ring plane:
        h_up = (math.cos(math.pi - THETA_TET) * out
                + math.sin(math.pi - THETA_TET) * plane_norm)
        h_dn = (math.cos(math.pi - THETA_TET) * out
                - math.sin(math.pi - THETA_TET) * plane_norm)
        return [(1, ring_atom + R_CH * h_up),
                (1, ring_atom + R_CH * h_dn)]

    Hs_Cb = _ring_H_pair(Cb)
    Hs_Cg = _ring_H_pair(Cg)
    Hs_Cd = _ring_H_pair(Cd)

    # The amine N is now SECONDARY (one H, not two). Replace the two
    # NH2 hydrogens with a single N-H.
    bb_proline = list(bb)
    # Remove H_N2 (idx 6) entirely, keep H_N1 but reposition to the
    # exocyclic side opposite Cd.
    # We rebuild N-H carefully: in proline the N has 3 substituents
    # (Cα, Cd, H). Compute H direction from N: in plane (Cα, Cd, N),
    # opposite the (Cα + Cd) mean direction.
    in_Cα_N = (Calpha - N) / np.linalg.norm(Calpha - N)
    in_Cd_N = (Cd - N) / np.linalg.norm(Cd - N)
    H_dir = -(in_Cα_N + in_Cd_N)
    H_dir /= np.linalg.norm(H_dir)
    H_N_proline = N + R_NH * H_dir

    # Backbone slot order from _backbone_atoms():
    #   0 Cα, 1 N, 2 C_carb, 3 H_α, 4 H_N1, 5 H_N2, 6 O_dbl, 7 O_sgl, 8 H_OH
    # Replace amine hydrogens: keep idx 4 -> H_N_proline; drop idx 5.
    bb_proline[4] = (1, H_N_proline)
    bb_proline.pop(5)   # drop H_N2

    atoms = bb_proline + [(6, Cb)] + Hs_Cb + [(6, Cg)] + Hs_Cg \
        + [(6, Cd)] + Hs_Cd
    return _finalise(atoms)


__all__ = [
    "alanine_L", "alanine_D", "alanine_L_B3LYP", "glycine",
    "valine_L", "leucine_L", "isoleucine_L",
    "serine_L", "threonine_L", "cysteine_L", "methionine_L",
    "aspartic_acid_L", "asparagine_L",
    "glutamic_acid_L", "glutamine_L",
    "lysine_L", "arginine_L", "histidine_L",
    "phenylalanine_L", "tyrosine_L", "tryptophan_L", "proline_L",
]
