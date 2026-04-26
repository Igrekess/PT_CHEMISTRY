"""Tests for ptc.lcao.sto_overlap (Phase A general STO overlap library).

Validates:
  * Numerical 2D quadrature reproduces analytic 1s-1s (1e-10 agreement)
  * Brute-force 3D integration matches the 2D SK integrals for sp/pp
    (independent ground truth)
  * Slater-Koster geometric rotation reproduces the axial integrals when
    R_AB is along +z, and rotates correctly when R_AB is along +x or +y
  * Overlap matrix is symmetric and positive-definite for H_2 and CH_4
  * H_2 sigma/pi values are physically reasonable
"""

import math

import numpy as np
import pytest

from ptc.constants import A_BOHR
from ptc.lcao.atomic_basis import (
    PTAtomicOrbital,
    build_atom_basis,
    overlap_atomic,
)
from ptc.lcao.atomic_basis import _overlap_1s_1s
from ptc.lcao.sto_overlap import (
    overlap_sp_general,
    slater_koster_axial,
)


# ─────────────────────────────────────────────────────────────────────
# 1. Numerical SK == analytic for 1s-1s (1e-10 agreement)
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("R_au, zA_au, zB_au", [
    (1.4, 1.0, 1.0),
    (1.4, 1.0, 1.5),
    (2.0, 0.8, 1.2),
    (3.0, 1.5, 1.5),
    (0.5, 1.0, 1.0),
])
def test_axial_quadrature_matches_1s_1s_analytic(R_au, zA_au, zB_au):
    R = R_au * A_BOHR
    zA = zA_au / A_BOHR
    zB = zB_au / A_BOHR
    analytic = _overlap_1s_1s(zA, zB, R)
    numerical = slater_koster_axial(1, 0, 1, 0, 0, zA, zB, R)
    assert numerical == pytest.approx(analytic, rel=1e-10, abs=1e-12)


# ─────────────────────────────────────────────────────────────────────
# 2. Brute-force 3D integration as ground truth for sp / pp
# ─────────────────────────────────────────────────────────────────────


def _brute_force_pp_sigma_or_pi(zeta_au, R_au, kind: str, N=81, L=12.0):
    """Direct 3D integration of <p_z|p_z> ('sigma') or <p_x|p_x> ('pi')
    in atomic units. Cubic grid (-L, +L)^3 with N nodes per axis."""
    xs = np.linspace(-L, L, N)
    dx = xs[1] - xs[0]
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing='ij')
    Nrad = (2 * zeta_au) ** 2 * math.sqrt(2 * zeta_au / 24.0)
    Nang = math.sqrt(3.0 / (4.0 * math.pi))

    def stoval(comp, x0, y0, z0):
        XS, YS, ZS = X - x0, Y - y0, Z - z0
        r = np.sqrt(XS * XS + YS * YS + ZS * ZS) + 1e-30
        c = {'pz': ZS, 'px': XS}[comp]
        return Nrad * Nang * c * np.exp(-zeta_au * r)

    if kind == 'sigma':
        a, b = stoval('pz', 0, 0, 0), stoval('pz', 0, 0, R_au)
    elif kind == 'pi':
        a, b = stoval('px', 0, 0, 0), stoval('px', 0, 0, R_au)
    else:
        raise ValueError(kind)
    return float((a * b).sum() * dx ** 3)


def test_pp_sigma_matches_brute_force():
    z = 1.0
    R = 1.4
    bf = _brute_force_pp_sigma_or_pi(z, R, 'sigma')
    sk = slater_koster_axial(2, 1, 2, 1, 0, z / A_BOHR, z / A_BOHR, R * A_BOHR)
    assert sk == pytest.approx(bf, abs=2e-4)


def test_pp_pi_matches_brute_force():
    z = 1.0
    R = 1.4
    bf = _brute_force_pp_sigma_or_pi(z, R, 'pi')
    sk = slater_koster_axial(2, 1, 2, 1, 1, z / A_BOHR, z / A_BOHR, R * A_BOHR)
    assert sk == pytest.approx(bf, abs=2e-4)


def test_pp_sigma_changes_sign_with_R():
    """In our convention (both p_z aligned along global +z), ppσ is
    POSITIVE at short R (overlap dominated by tail with same lobe sign)
    and NEGATIVE at large R (bonding region's antiparallel +/- lobes
    dominate). ppπ stays positive throughout. Cross-check vs brute force
    is in test_pp_sigma_matches_brute_force."""
    z = 1.0 / A_BOHR
    short = slater_koster_axial(2, 1, 2, 1, 0, z, z, 0.5)
    long_ = slater_koster_axial(2, 1, 2, 1, 0, z, z, 2.0)
    assert short > 0.0
    assert long_ < 0.0


def test_pp_pi_is_positive():
    z = 1.0 / A_BOHR
    for R in (0.4, 0.8, 1.5, 2.5):
        pi = slater_koster_axial(2, 1, 2, 1, 1, z, z, R)
        assert pi > 0.0, f"ppπ < 0 at R={R}"


# ─────────────────────────────────────────────────────────────────────
# 3. Slater-Koster geometric rotation
# ─────────────────────────────────────────────────────────────────────


def _make_sp_pair(n_a, l_a, m_a, n_b, l_b, m_b, zeta=1.5/A_BOHR):
    a = PTAtomicOrbital(Z=6, n=n_a, l=l_a, m=m_a, zeta=zeta, occ=1.0)
    b = PTAtomicOrbital(Z=6, n=n_b, l=l_b, m=m_b, zeta=zeta, occ=1.0)
    return a, b


def test_sk_rotation_along_z_reduces_to_axial():
    a, b = _make_sp_pair(2, 1, 0, 2, 1, 0)  # both p_z
    R = 1.5
    direct = overlap_sp_general(a, b, np.array([0, 0, R]))
    axial = slater_koster_axial(2, 1, 2, 1, 0, a.zeta, b.zeta, R)
    assert direct == pytest.approx(axial, rel=1e-12)


def test_sk_rotation_x_axis_pp_xx():
    a, b = _make_sp_pair(2, 1, 1, 2, 1, 1)  # both p_x
    R = 1.5
    # along +x axis: direction cosines (1, 0, 0)
    # <p_x|p_x>(R along x) = 1^2 ppσ + 0 ppπ = ppσ
    direct = overlap_sp_general(a, b, np.array([R, 0, 0]))
    axial_sigma = slater_koster_axial(2, 1, 2, 1, 0, a.zeta, b.zeta, R)
    assert direct == pytest.approx(axial_sigma, rel=1e-12)


def test_sk_rotation_x_axis_pp_yy():
    a, b = _make_sp_pair(2, 1, -1, 2, 1, -1)  # both p_y
    R = 1.5
    # along +x: <p_y|p_y> = 0^2 ppσ + 1 × ppπ = ppπ
    direct = overlap_sp_general(a, b, np.array([R, 0, 0]))
    axial_pi = slater_koster_axial(2, 1, 2, 1, 1, a.zeta, b.zeta, R)
    assert direct == pytest.approx(axial_pi, rel=1e-12)


def test_sk_rotation_x_axis_pp_xy_zero_by_symmetry():
    a, b = _make_sp_pair(2, 1, 1, 2, 1, -1)  # p_x and p_y
    R = 1.5
    # along +x: <p_x|p_y> = 1·0·(ppσ - ppπ) + 0 = 0
    direct = overlap_sp_general(a, b, np.array([R, 0, 0]))
    assert direct == pytest.approx(0.0, abs=1e-12)


def test_sk_rotation_diagonal_axis_pp_xx():
    """Bond along (1,1,1)/sqrt(3): <p_x|p_x> = lx^2 ppσ + (1-lx^2) ppπ
    with lx = 1/sqrt(3)."""
    a, b = _make_sp_pair(2, 1, 1, 2, 1, 1)
    L = 1.5
    R_vec = np.array([L, L, L]) / math.sqrt(3.0)
    direct = overlap_sp_general(a, b, R_vec)
    pps = slater_koster_axial(2, 1, 2, 1, 0, a.zeta, b.zeta, L)
    ppp = slater_koster_axial(2, 1, 2, 1, 1, a.zeta, b.zeta, L)
    expected = (1.0 / 3.0) * pps + (2.0 / 3.0) * ppp
    assert direct == pytest.approx(expected, rel=1e-12)


def test_sk_overlap_symmetric_in_arguments():
    """<orb_A|orb_B> (with R_AB) = <orb_B|orb_A> (with -R_AB), since real basis."""
    a, b = _make_sp_pair(2, 0, 0, 2, 1, 1)  # 2s and 2p_x
    R_vec = np.array([1.2, -0.8, 0.6])
    s_AB = overlap_sp_general(a, b, R_vec)
    s_BA = overlap_sp_general(b, a, -R_vec)
    assert s_AB == pytest.approx(s_BA, rel=1e-12)


# ─────────────────────────────────────────────────────────────────────
# 4. Whole-molecule overlap matrices (H_2, CH_4)
# ─────────────────────────────────────────────────────────────────────


def _build_overlap_matrix(orbitals, coords):
    """Construct N x N overlap matrix from a list of (orb, coord)."""
    n = len(orbitals)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            r_ij = coords[j] - coords[i]
            S[i, j] = overlap_atomic(orbitals[i], orbitals[j], r_ij)
    return S


def test_H2_overlap_matrix():
    R = 0.74  # Angstrom
    a, b = build_atom_basis(1).orbitals[0], build_atom_basis(1).orbitals[0]
    coords = [np.array([0, 0, 0]), np.array([R, 0, 0])]
    S = _build_overlap_matrix([a, b], coords)
    assert S.shape == (2, 2)
    # diagonal is 1
    assert np.allclose(np.diag(S), [1.0, 1.0], atol=1e-12)
    # symmetric
    assert np.allclose(S, S.T, atol=1e-12)
    # off-diagonal = H_2 STO overlap ~ 0.75
    assert 0.70 < S[0, 1] < 0.80
    # positive-definite (eigenvalues > 0)
    eigs = np.linalg.eigvalsh(S)
    assert (eigs > 0).all(), f"non-PD eigenvalues {eigs}"


def test_CH4_overlap_matrix_is_PD():
    """CH_4: 1 C with valence 2s + 2p_{x,y,z} + 4 H around tetrahedron.
    Bond length 1.087 A (B3LYP). Overlap matrix must be SPD."""
    R_CH = 1.087
    # Tetrahedron vertices
    h_dirs = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ], dtype=float)
    h_dirs /= np.linalg.norm(h_dirs, axis=1, keepdims=True)
    h_coords = h_dirs * R_CH
    c_coord = np.array([0, 0, 0])

    C_orbs = build_atom_basis(6).orbitals    # 4 orbitals: 2s, 2p_x, 2p_y, 2p_z
    H_orbs = [build_atom_basis(1).orbitals[0] for _ in range(4)]

    orbs = list(C_orbs) + list(H_orbs)
    coords = [c_coord] * len(C_orbs) + list(h_coords)
    S = _build_overlap_matrix(orbs, coords)

    # symmetric and 1 on diagonal
    assert np.allclose(np.diag(S), 1.0, atol=1e-12)
    assert np.allclose(S, S.T, atol=1e-12), "S not symmetric"
    eigs = np.linalg.eigvalsh(S)
    assert (eigs > 0).all(), f"S not PD; min eig = {eigs.min():.4e}"


def test_NH3_overlap_matrix_is_PD():
    """NH_3 with experimental-like geometry."""
    R_NH = 1.012
    angle = math.radians(106.7)
    # N at origin; H atoms in pyramidal arrangement
    h_dirs = []
    for k in range(3):
        phi = 2 * math.pi * k / 3
        sa = math.sin(angle)
        h_dirs.append([sa * math.cos(phi), sa * math.sin(phi), -math.cos(angle)])
    h_coords = [np.array(d) * R_NH for d in h_dirs]

    N_orbs = build_atom_basis(7).orbitals     # 2s + 2p_xyz
    H_orbs = [build_atom_basis(1).orbitals[0] for _ in range(3)]

    orbs = list(N_orbs) + list(H_orbs)
    coords = [np.array([0, 0, 0])] * len(N_orbs) + h_coords
    S = _build_overlap_matrix(orbs, coords)

    assert np.allclose(np.diag(S), 1.0, atol=1e-12)
    assert np.allclose(S, S.T, atol=1e-12), "S not symmetric"
    eigs = np.linalg.eigvalsh(S)
    assert (eigs > 0).all(), f"S not PD; min eig = {eigs.min():.4e}"


# ─────────────────────────────────────────────────────────────────────
# 5. Orthogonality on same atom
# ─────────────────────────────────────────────────────────────────────


def test_p_orbitals_same_atom_orthogonal():
    px = PTAtomicOrbital(Z=6, n=2, l=1, m=1, zeta=2.0, occ=1.0)
    py = PTAtomicOrbital(Z=6, n=2, l=1, m=-1, zeta=2.0, occ=1.0)
    pz = PTAtomicOrbital(Z=6, n=2, l=1, m=0, zeta=2.0, occ=1.0)
    s2 = PTAtomicOrbital(Z=6, n=2, l=0, m=0, zeta=2.0, occ=1.0)
    zero = np.zeros(3)
    assert overlap_atomic(px, py, zero) == pytest.approx(0.0, abs=1e-12)
    assert overlap_atomic(px, pz, zero) == pytest.approx(0.0, abs=1e-12)
    assert overlap_atomic(py, pz, zero) == pytest.approx(0.0, abs=1e-12)
    assert overlap_atomic(s2, px, zero) == pytest.approx(0.0, abs=1e-12)
