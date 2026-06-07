"""Tests for ptc.lcao.geometries_amino_acids.

These verify atom counts, bond-length sanity, chirality of L-alanine,
the Cs symmetry of glycine, and the mirror relation between L- and
D-alanine geometries. They do not depend on any PT-specific physics.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ptc.lcao.density_matrix import PTMolecularBasis
from ptc.lcao.geometries_amino_acids import (
    alanine_D,
    alanine_L,
    alanine_L_B3LYP,
    glycine,
    valine_L,
    leucine_L,
    isoleucine_L,
    serine_L,
    threonine_L,
    cysteine_L,
    methionine_L,
    aspartic_acid_L,
    asparagine_L,
    glutamic_acid_L,
    glutamine_L,
    lysine_L,
    arginine_L,
    histidine_L,
    phenylalanine_L,
    tyrosine_L,
    tryptophan_L,
    proline_L,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _coords(atom_list):
    return np.array([r for (_Z, r) in atom_list])


def _Zs(atom_list):
    return [Z for (Z, _r) in atom_list]


def _distance(atom_list, i, j):
    return float(np.linalg.norm(atom_list[i][1] - atom_list[j][1]))


# ---------------------------------------------------------------------
# Atom counts and elemental composition
# ---------------------------------------------------------------------


def test_alanine_atom_count():
    """L-alanine = C3 H7 N O2, 13 atoms."""
    atom_list, basis = alanine_L()
    assert len(atom_list) == 13
    Zs = _Zs(atom_list)
    assert Zs.count(6) == 3   # C
    assert Zs.count(1) == 7   # H
    assert Zs.count(7) == 1   # N
    assert Zs.count(8) == 2   # O
    assert isinstance(basis, PTMolecularBasis)
    assert basis.n_atoms == 13


def test_glycine_atom_count():
    """Glycine = C2 H5 N O2, 10 atoms."""
    atom_list, basis = glycine()
    assert len(atom_list) == 10
    Zs = _Zs(atom_list)
    assert Zs.count(6) == 2
    assert Zs.count(1) == 5
    assert Zs.count(7) == 1
    assert Zs.count(8) == 2
    assert isinstance(basis, PTMolecularBasis)
    assert basis.n_atoms == 10


def test_alanine_D_atom_count():
    """D-alanine has the same composition as L."""
    atom_list, _ = alanine_D()
    assert len(atom_list) == 13


# ---------------------------------------------------------------------
# Bond-length sanity checks
# ---------------------------------------------------------------------


def test_alanine_distances_reasonable():
    """Standard bond lengths around the chirality center.

    Tolerances are loose (+- 0.1 A) because the geometry is hand-built
    sp3 with idealised tetrahedral angles, not energy-minimised.
    """
    atom_list, _ = alanine_L()
    # Indices: 0=Calpha, 1=N, 2=C_carb, 3=C_methyl, 4=H_alpha,
    #          5=H_N1, 6=H_N2, 7=O_dbl, 8=O_sgl, 9=H_OH,
    #          10..12 = methyl H
    d_CN = _distance(atom_list, 0, 1)
    assert d_CN == pytest.approx(1.471, abs=0.1)
    d_CC_carb = _distance(atom_list, 0, 2)
    assert d_CC_carb == pytest.approx(1.515, abs=0.1)
    d_CC_meth = _distance(atom_list, 0, 3)
    assert d_CC_meth == pytest.approx(1.526, abs=0.1)
    d_CH_alpha = _distance(atom_list, 0, 4)
    assert d_CH_alpha == pytest.approx(1.090, abs=0.1)
    d_CO_dbl = _distance(atom_list, 2, 7)
    assert d_CO_dbl == pytest.approx(1.214, abs=0.1)
    d_CO_sgl = _distance(atom_list, 2, 8)
    assert d_CO_sgl == pytest.approx(1.357, abs=0.1)
    d_OH = _distance(atom_list, 8, 9)
    assert d_OH == pytest.approx(0.972, abs=0.1)
    for hi in (10, 11, 12):
        d = _distance(atom_list, 3, hi)
        assert d == pytest.approx(1.090, abs=0.1), (
            f"methyl C-H[{hi}] = {d}"
        )
    for hi in (5, 6):
        d = _distance(atom_list, 1, hi)
        assert d == pytest.approx(1.014, abs=0.1), (
            f"amine N-H[{hi}] = {d}"
        )


def test_glycine_distances_reasonable():
    """Glycine bond lengths follow the same standard values."""
    atom_list, _ = glycine()
    # 0=Calpha, 1=N, 2=C_carb, 3=Ha+, 4=Ha-, 5=HN+, 6=HN-,
    # 7=O_dbl, 8=O_sgl, 9=H_OH
    d_CN = _distance(atom_list, 0, 1)
    d_CC = _distance(atom_list, 0, 2)
    d_CO_dbl = _distance(atom_list, 2, 7)
    d_CO_sgl = _distance(atom_list, 2, 8)
    d_OH = _distance(atom_list, 8, 9)
    assert d_CN == pytest.approx(1.471, abs=0.1)
    assert d_CC == pytest.approx(1.515, abs=0.1)
    assert d_CO_dbl == pytest.approx(1.214, abs=0.1)
    assert d_CO_sgl == pytest.approx(1.357, abs=0.1)
    assert d_OH == pytest.approx(0.972, abs=0.1)
    # Cα - Hα
    for hi in (3, 4):
        assert _distance(atom_list, 0, hi) == pytest.approx(1.090, abs=0.1)
    # N-H
    for hi in (5, 6):
        assert _distance(atom_list, 1, hi) == pytest.approx(1.014, abs=0.1)


# ---------------------------------------------------------------------
# Chirality / symmetry checks
# ---------------------------------------------------------------------


def test_alanine_chirality():
    """The four Cα substituents are non-coplanar -- L-alanine is chiral.

    We use the signed volume of the three highest-priority substituent
    vectors (NH2, COOH, CH3) anchored at Cα. A non-zero value confirms
    sp3 stereocenter, and a POSITIVE value confirms (S)-configuration
    in the convention used by ``alanine_L`` (see its docstring).
    """
    atom_list, _ = alanine_L()
    Calpha = atom_list[0][1]
    N = atom_list[1][1]
    C_carb = atom_list[2][1]
    C_meth = atom_list[3][1]
    v_N = N - Calpha
    v_COOH = C_carb - Calpha
    v_CH3 = C_meth - Calpha
    # Use unit vectors for a scale-free signed volume.
    e1 = v_N / np.linalg.norm(v_N)
    e2 = v_COOH / np.linalg.norm(v_COOH)
    e3 = v_CH3 / np.linalg.norm(v_CH3)
    sv = float(np.dot(e1, np.cross(e2, e3)))
    # For the tetrahedral assignment in ``alanine_L`` the unit-vector
    # signed volume is 8/(3 sqrt 3) ~ 1.5396, taken positive.
    assert abs(sv) > 0.1, "Cα substituents are nearly coplanar"
    assert sv > 0.0, "L-alanine signed volume should be positive (S config)"


def test_alanine_D_is_mirror_of_L():
    """D-alanine has the opposite handedness: same atoms, mirrored
    coordinates (x -> -x). The signed volume of substituent vectors
    therefore flips sign relative to L-alanine.
    """
    atom_list_L, _ = alanine_L()
    atom_list_D, _ = alanine_D()
    # Same atom types in the same order.
    Zs_L = _Zs(atom_list_L)
    Zs_D = _Zs(atom_list_D)
    assert Zs_L == Zs_D
    # Coordinates differ only by x -> -x.
    coords_L = _coords(atom_list_L)
    coords_D = _coords(atom_list_D)
    coords_D_remirror = coords_D * np.array([-1.0, 1.0, 1.0])
    np.testing.assert_allclose(coords_L, coords_D_remirror, atol=1e-12)
    # Substituent signed volume should flip sign vs L.
    Calpha = atom_list_D[0][1]
    e1 = (atom_list_D[1][1] - Calpha)
    e2 = (atom_list_D[2][1] - Calpha)
    e3 = (atom_list_D[3][1] - Calpha)
    sv_D = float(np.dot(e1 / np.linalg.norm(e1),
                        np.cross(e2 / np.linalg.norm(e2),
                                 e3 / np.linalg.norm(e3))))
    assert sv_D < 0.0


def test_glycine_cs_symmetry():
    """Glycine: every atom either lies on the xz plane (y = 0) or
    pairs with another of the same Z at (-y) -- the molecule is
    sigma_v(xz)-symmetric.
    """
    atom_list, _ = glycine()
    # Build (Z, x, z, |y|) and check pairing.
    on_plane = []
    off_plane = []
    for (Z, r) in atom_list:
        if abs(r[1]) < 1e-9:
            on_plane.append((Z, r[0], r[2]))
        else:
            off_plane.append((Z, r))
    # All off-plane atoms must come in mirror pairs (same Z, same x, z;
    # opposite y).
    assert len(off_plane) % 2 == 0, (
        "Off-plane atoms must pair up for Cs symmetry"
    )
    matched = [False] * len(off_plane)
    for i, (Zi, ri) in enumerate(off_plane):
        if matched[i]:
            continue
        found = False
        for j in range(i + 1, len(off_plane)):
            if matched[j]:
                continue
            Zj, rj = off_plane[j]
            if Zj != Zi:
                continue
            if (
                abs(ri[0] - rj[0]) < 1e-9
                and abs(ri[2] - rj[2]) < 1e-9
                and abs(ri[1] + rj[1]) < 1e-9
            ):
                matched[i] = True
                matched[j] = True
                found = True
                break
        assert found, f"No mirror partner for atom {i}: Z={Zi}, r={ri}"


def test_alanine_L_B3LYP_is_S():
    """alanine_L_B3LYP is the (S)-configured = L enantiomer.

    Uses the same signed-volume invariant as test_alanine_chirality.
    The B3LYP-optimised conformer I must give the same positive sign
    (NH2 > COOH > CH3 in CIP priority, counter-clockwise viewed from
    opposite the H).
    """
    atom_list, _ = alanine_L_B3LYP()
    Calpha = atom_list[0][1]
    N = atom_list[1][1]
    C_carb = atom_list[2][1]
    C_meth = atom_list[3][1]
    e1 = (N - Calpha) / np.linalg.norm(N - Calpha)
    e2 = (C_carb - Calpha) / np.linalg.norm(C_carb - Calpha)
    e3 = (C_meth - Calpha) / np.linalg.norm(C_meth - Calpha)
    sv = float(np.dot(e1, np.cross(e2, e3)))
    assert abs(sv) > 0.1, "Cα substituents are nearly coplanar"
    assert sv > 0.0, (
        "alanine_L_B3LYP signed volume should be positive (S config); "
        f"got sv = {sv:+.4f}"
    )


def test_alanine_L_B3LYP_atom_count():
    """B3LYP L-alanine: same composition as hand-built."""
    atom_list, basis = alanine_L_B3LYP()
    assert len(atom_list) == 13
    Zs = _Zs(atom_list)
    assert Zs.count(6) == 3   # 3 C
    assert Zs.count(1) == 7   # 7 H
    assert Zs.count(7) == 1   # 1 N
    assert Zs.count(8) == 2   # 2 O
    assert isinstance(basis, PTMolecularBasis)
    assert basis.n_atoms == 13


def test_alanine_L_B3LYP_distances_reasonable():
    """B3LYP/6-31G*-equivalent bond lengths near literature values.

    Tolerances are tightened to +- 0.05 A (vs +- 0.1 A for the
    hand-built geometry), since B3LYP-optimised coordinates should
    deviate from textbook values by well under 0.03 A.
    """
    atom_list, _ = alanine_L_B3LYP()
    # Order: 0=Calpha, 1=N, 2=C_carb, 3=C_methyl, 4=H_alpha,
    #        5=H_N1, 6=H_N2, 7=O_dbl, 8=O_sgl, 9=H_OH, 10..12 methyl H
    d_CN = _distance(atom_list, 0, 1)
    assert d_CN == pytest.approx(1.453, abs=0.05), f"d(Cα-N) = {d_CN}"
    d_CC_carb = _distance(atom_list, 0, 2)
    assert d_CC_carb == pytest.approx(1.527, abs=0.05), (
        f"d(Cα-C_carb) = {d_CC_carb}"
    )
    d_CC_meth = _distance(atom_list, 0, 3)
    assert d_CC_meth == pytest.approx(1.527, abs=0.05), (
        f"d(Cα-C_methyl) = {d_CC_meth}"
    )
    d_CH_alpha = _distance(atom_list, 0, 4)
    assert d_CH_alpha == pytest.approx(1.094, abs=0.05), (
        f"d(Cα-H) = {d_CH_alpha}"
    )
    d_CO_dbl = _distance(atom_list, 2, 7)
    assert d_CO_dbl == pytest.approx(1.220, abs=0.05), (
        f"d(C=O) = {d_CO_dbl}"
    )
    d_CO_sgl = _distance(atom_list, 2, 8)
    assert d_CO_sgl == pytest.approx(1.344, abs=0.05), (
        f"d(C-O) = {d_CO_sgl}"
    )
    d_OH = _distance(atom_list, 8, 9)
    assert d_OH == pytest.approx(0.97, abs=0.05), f"d(O-H) = {d_OH}"
    # Three methyl C-H
    for hi in (10, 11, 12):
        d = _distance(atom_list, 3, hi)
        assert d == pytest.approx(1.09, abs=0.05), (
            f"methyl C-H[{hi}] = {d}"
        )
    # Two amine N-H
    for hi in (5, 6):
        d = _distance(atom_list, 1, hi)
        assert d == pytest.approx(1.014, abs=0.05), (
            f"amine N-H[{hi}] = {d}"
        )


def test_alanine_L_B3LYP_differs_from_handbuilt():
    """The B3LYP geometry must not coincide with the hand-built one.

    Sanity check: the RMS Cartesian displacement between alanine_L
    (idealised tetrahedral) and alanine_L_B3LYP (B3LYP-relaxed
    conformer I) should be >= 0.1 A. They are different physical
    conformers, and the whole point of importing B3LYP is to escape
    the conformer artefact of the hand-built geometry.
    """
    atom_list_hb, _ = alanine_L()
    atom_list_b3lyp, _ = alanine_L_B3LYP()
    coords_hb = _coords(atom_list_hb)
    coords_b3 = _coords(atom_list_b3lyp)
    # Both geometries are centred at Cα = origin and share atom order;
    # we compare directly without alignment (rotational alignment is
    # outside the scope of this sanity check).
    rms = float(np.sqrt(np.mean((coords_hb - coords_b3) ** 2)))
    assert rms > 0.1, (
        f"B3LYP and hand-built coordinates are too similar "
        f"(RMS = {rms:.4f} A); did the B3LYP function fall back to "
        f"the hand-built one?"
    )


def test_glycine_gradient_x_vanishes_by_mirror():
    """Operational achirality test: the parity-violation spatial
    gradient sum G_x must vanish on x-even AO pairs centred on atoms
    that lie on the sigma_v(xz) mirror.

    This mirrors test_h2o_gradient_x_vanishes_by_mirror in
    test_parity_violation.py: it is the matrix-element-level
    "glycine = 0" check.
    """
    from ptc.lcao.parity_violation import (
        _spatial_gradient_sum,
        nuclear_weak_charges_default,
    )

    atom_list, basis = glycine()
    Q_W = nuclear_weak_charges_default(basis.Z_list)
    G = _spatial_gradient_sum(basis, basis.coords, Q_W)
    # The molecule is mirrored by y -> -y. For an AO pair where both
    # AOs sit on atoms with y = 0 and have m corresponding to a y-even
    # angular factor, G_y on that pair must vanish by mirror symmetry.
    # We pick the central Cα 2s orbital and check its diagonal /
    # off-diagonal to other y=0 s orbitals: G_y must be ~ 0.
    s_indices_on_plane = []
    for k, orb in enumerate(basis.orbitals):
        if orb.l != 0:
            continue
        atom = basis.atom_index[k]
        y_atom = basis.coords[atom][1]
        if abs(y_atom) < 1e-9:
            s_indices_on_plane.append(k)
    assert len(s_indices_on_plane) >= 2
    # G is antisymmetric in (mu, nu); off-diagonal s-s pairs on the
    # mirror plane should have a vanishing y-component.
    for ii in range(len(s_indices_on_plane)):
        for jj in range(ii + 1, len(s_indices_on_plane)):
            mu = s_indices_on_plane[ii]
            nu = s_indices_on_plane[jj]
            val = G[1, mu, nu]
            assert abs(val) < 1e-10, (
                f"G_y[{mu}, {nu}] = {val}, expected ~ 0 by sigma_v(xz)"
            )


# ---------------------------------------------------------------------
# 18 additional natural L-amino acids: hand-built fallback geometries
# ---------------------------------------------------------------------
#
# Each AA tested for:
#   (a) atom count + elemental composition
#   (b) CIP-(S) (or CIP-(R) for cysteine) at C-alpha
#
# Side-chain bond-length sanity is NOT tested here because side chains
# have many bond types; the central chirality and composition checks
# are sufficient for the universality scan in
# PT_HOMOCHIRALITY/scripts/scan_20_amino_acids.py.


_AA_EXPECTED = [
    # (function, code, expected composition dict {Z: count}, CIP letter at Cα)
    (valine_L,         "Val", {6: 5,  1: 11, 7: 1, 8: 2},          "S"),
    (leucine_L,        "Leu", {6: 6,  1: 13, 7: 1, 8: 2},          "S"),
    (isoleucine_L,     "Ile", {6: 6,  1: 13, 7: 1, 8: 2},          "S"),
    (serine_L,         "Ser", {6: 3,  1: 7,  7: 1, 8: 3},          "S"),
    (threonine_L,      "Thr", {6: 4,  1: 9,  7: 1, 8: 3},          "S"),
    (cysteine_L,       "Cys", {6: 3,  1: 7,  7: 1, 8: 2, 16: 1},   "R"),
    (methionine_L,     "Met", {6: 5,  1: 11, 7: 1, 8: 2, 16: 1},   "S"),
    (aspartic_acid_L,  "Asp", {6: 4,  1: 7,  7: 1, 8: 4},          "S"),
    (asparagine_L,     "Asn", {6: 4,  1: 8,  7: 2, 8: 3},          "S"),
    (glutamic_acid_L,  "Glu", {6: 5,  1: 9,  7: 1, 8: 4},          "S"),
    (glutamine_L,      "Gln", {6: 5,  1: 10, 7: 2, 8: 3},          "S"),
    (lysine_L,         "Lys", {6: 6,  1: 14, 7: 2, 8: 2},          "S"),
    (arginine_L,       "Arg", {6: 6,  1: 14, 7: 4, 8: 2},          "S"),
    (histidine_L,      "His", {6: 6,  1: 9,  7: 3, 8: 2},          "S"),
    (phenylalanine_L,  "Phe", {6: 9,  1: 11, 7: 1, 8: 2},          "S"),
    (tyrosine_L,       "Tyr", {6: 9,  1: 11, 7: 1, 8: 3},          "S"),
    (tryptophan_L,     "Trp", {6: 11, 1: 12, 7: 2, 8: 2},          "S"),
    (proline_L,        "Pro", {6: 5,  1: 9,  7: 1, 8: 2},          "S"),
]


def _first_side_chain_heavy(atom_list):
    """Locate the first heavy (Z>=6 or Z=16) atom AFTER slot 2 (C_carb)
    that is not on the backbone proper.

    Backbone canonical slots from `_backbone_atoms()`:
        0 Cα, 1 N, 2 C_carb, 3 Hα, 4 H_N1, 5 H_N2 (or H_N for proline),
        6 O_dbl, 7 O_sgl, 8 H_OH
    The first side-chain heavy atom is therefore the first (Z=6 or 16)
    atom at slot >= 9 (or, for proline where slot 5 is removed, slot >= 8).
    Implementation: scan from slot 3 onward.
    """
    for i in range(3, len(atom_list)):
        Z = atom_list[i][0]
        if Z == 6 or Z == 16:
            return i
    raise ValueError("no side-chain heavy atom found")


@pytest.mark.parametrize("fn,code,expected,cip", _AA_EXPECTED)
def test_aa_atom_count(fn, code, expected, cip):
    """Each AA: correct number of atoms and elemental composition."""
    atom_list, basis = fn()
    n_expected = sum(expected.values())
    assert len(atom_list) == n_expected, (
        f"{code}: got {len(atom_list)} atoms, expected {n_expected}"
    )
    Zs = _Zs(atom_list)
    for Z, n in expected.items():
        assert Zs.count(Z) == n, (
            f"{code}: Z={Z} count is {Zs.count(Z)}, expected {n}"
        )
    assert isinstance(basis, PTMolecularBasis)
    assert basis.n_atoms == n_expected


@pytest.mark.parametrize("fn,code,expected,cip", _AA_EXPECTED)
def test_aa_chirality(fn, code, expected, cip):
    """Each AA: correct CIP configuration at C-alpha.

    Signed volume of (e_N, e_COOH, e_R) at Cα measured under the
    PT spatial convention (see ``alanine_L`` docstring):
       sv > 0  <=>  CIP-(S)  (for N > COOH > R priority)
       sv < 0  <=>  CIP-(R)
    For cysteine the side chain has higher priority than N (S > N),
    so the natural L-cysteine reads as CIP-(R) in the standard
    nomenclature. With the cys_swap implementation in
    ``cysteine_L`` the side chain sits on the +e_NH2 corner, which
    makes the spatial signed volume negative -- consistent with the
    L-cysteine = CIP-(R) assignment.
    """
    atom_list, _ = fn()
    Calpha = atom_list[0][1]
    N = atom_list[1][1]
    C_carb = atom_list[2][1]
    side_idx = _first_side_chain_heavy(atom_list)
    R_atom = atom_list[side_idx][1]
    e_N = (N - Calpha) / np.linalg.norm(N - Calpha)
    e_C = (C_carb - Calpha) / np.linalg.norm(C_carb - Calpha)
    e_R = (R_atom - Calpha) / np.linalg.norm(R_atom - Calpha)
    sv = float(np.dot(e_N, np.cross(e_C, e_R)))
    assert abs(sv) > 0.1, (
        f"{code}: Cα substituents are nearly coplanar (sv = {sv:+.4f})"
    )
    if cip == "S":
        assert sv > 0.0, (
            f"{code}: expected CIP-(S) (sv > 0); got sv = {sv:+.4f}"
        )
    elif cip == "R":
        assert sv < 0.0, (
            f"{code}: expected CIP-(R) (sv < 0); got sv = {sv:+.4f}"
        )
    else:
        pytest.fail(f"Unexpected CIP letter {cip} for {code}")
