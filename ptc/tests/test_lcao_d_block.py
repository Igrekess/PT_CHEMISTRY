"""Phase A continuation tests: d-orbital coverage for transition metals.

Validates:
  * STO d-orbital evaluator (5 cubic harmonics) is correctly normalised
  * 3D numerical overlap matches analytic 1s-1s within ~1e-5
  * Self-overlap = 1 for d-orbitals
  * d-block atom basis assembled correctly (1 ns + 5 (n-1)d)
  * Transition metal molecule (FeH, Cu2) overlap matrix is SPD
  * Density matrix on Cu2: Tr(rho S) = N_electrons, rho S rho = 2 rho
  * f-block remains gated (NotImplementedError) until next continuation
"""

import math

import numpy as np
import pytest

from ptc.constants import A_BOHR
from ptc.lcao.atomic_basis import (
    PTAtomicOrbital,
    _overlap_1s_1s,
    build_atom_basis,
    overlap_atomic,
)
from ptc.lcao.density_matrix import (
    build_molecular_basis,
    density_matrix_PT,
    overlap_matrix,
)
from ptc.lcao.giao import evaluate_sto
from ptc.lcao.sto_overlap import overlap_3d_numerical
from ptc.topology import build_topology


# ─────────────────────────────────────────────────────────────────────
# 1. d-orbital STO evaluator
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("m, comp_str", [
    (-2, "d_xy"),
    (-1, "d_yz"),
    ( 0, "d_z2"),
    ( 1, "d_xz"),
    ( 2, "d_x2-y2"),
])
def test_d_orbital_normalisation(m, comp_str):
    """Numerical <d_lm | d_lm> = 1 on a fine grid."""
    orb = PTAtomicOrbital(Z=26, n=3, l=2, m=m, zeta=2.0, occ=1.0)
    leg_r, w_r = np.polynomial.legendre.leggauss(80)
    R_max = 12.0 / orb.zeta
    r_nodes = R_max / 2 * (leg_r + 1)
    r_weights = R_max / 2 * w_r
    leg_u, w_u = np.polynomial.legendre.leggauss(24)
    cos_t = leg_u
    n_phi = 32
    phi = np.linspace(0, 2 * math.pi, n_phi, endpoint=False)
    phi_w = 2 * math.pi / n_phi
    R, CT, PHI = np.meshgrid(r_nodes, cos_t, phi, indexing='ij')
    ST = np.sqrt(np.maximum(0.0, 1 - CT ** 2))
    pts = np.stack([R * ST * np.cos(PHI), R * ST * np.sin(PHI), R * CT], axis=-1)
    psi = evaluate_sto(orb, pts.reshape(-1, 3), np.zeros(3))
    W = (
        ((r_nodes ** 2) * r_weights)[:, None, None]
        * w_u[None, :, None]
        * np.full(n_phi, phi_w)[None, None, :]
    )
    norm = float(np.sum(psi ** 2 * W.flatten()))
    assert norm == pytest.approx(1.0, abs=2e-3), f"{comp_str}: norm = {norm}"


def test_d_orbitals_orthogonal_same_atom():
    """5 d-orbitals on same atom are mutually orthogonal."""
    base = PTAtomicOrbital(Z=26, n=3, l=2, m=0, zeta=2.0, occ=1.0)
    ms = [-2, -1, 0, 1, 2]
    for ma in ms:
        for mb in ms:
            if ma == mb:
                continue
            a = PTAtomicOrbital(Z=26, n=3, l=2, m=ma, zeta=2.0, occ=1.0)
            b = PTAtomicOrbital(Z=26, n=3, l=2, m=mb, zeta=2.0, occ=1.0)
            s = overlap_atomic(a, b, np.zeros(3))
            assert s == pytest.approx(0.0, abs=1e-10), (
                f"d_{ma} | d_{mb} = {s}, expected 0"
            )


# ─────────────────────────────────────────────────────────────────────
# 2. 3D numerical overlap consistency
# ─────────────────────────────────────────────────────────────────────


def test_3d_numerical_matches_analytic_1s_1s():
    """3D quadrature reproduces analytic 1s-1s closed form within ~1e-5."""
    zeta = 1.0 / A_BOHR
    R = 1.4 * A_BOHR
    a = PTAtomicOrbital(Z=1, n=1, l=0, m=0, zeta=zeta, occ=1.0)
    b = PTAtomicOrbital(Z=1, n=1, l=0, m=0, zeta=zeta, occ=1.0)
    analytic = _overlap_1s_1s(zeta, zeta, R)
    numeric = overlap_3d_numerical(a, b, np.array([R, 0, 0]))
    assert numeric == pytest.approx(analytic, abs=1e-4)


# ─────────────────────────────────────────────────────────────────────
# 3. Transition metal atom basis
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("Z, name, expected_orb", [
    (21, "Sc",  6),    # 4s + 5×3d (n_s=2, nd=1 -> still 5 m sub-states)
    (26, "Fe",  6),
    (29, "Cu",  6),    # Madelung: 4s^1 3d^10
    (47, "Ag",  6),    # Madelung: 5s^1 4d^10
    (78, "Pt",  6),
])
def test_d_block_basis_size(Z, name, expected_orb):
    basis = build_atom_basis(Z)
    assert basis.n_orbitals == expected_orb, (
        f"{name}: {basis.n_orbitals} orbitals, expected {expected_orb}"
    )


def test_d_block_atom_self_overlap():
    """All d-orbitals across transition row have self-overlap = 1."""
    for Z in (21, 26, 29, 39, 47):
        basis = build_atom_basis(Z)
        for o in basis.orbitals:
            s = overlap_atomic(o, o, np.zeros(3))
            assert s == pytest.approx(1.0, abs=1e-10), (
                f"Z={Z} {o.label}: self-overlap = {s}"
            )


# ─────────────────────────────────────────────────────────────────────
# 4. Transition metal molecule overlap matrices
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("smi, name", [
    ("[FeH]",   "FeH"),
    ("[Cu][Cu]", "Cu2"),
])
def test_transition_metal_overlap_matrix_SPD(smi, name):
    topo = build_topology(smi)
    basis = build_molecular_basis(topo)
    S = overlap_matrix(basis)
    assert np.allclose(np.diag(S), 1.0, atol=1e-12), f"{name}: diag != 1"
    assert np.allclose(S, S.T, atol=1e-10), f"{name}: not symmetric"
    eigs = np.linalg.eigvalsh(S)
    assert (eigs > 0).all(), f"{name}: not PD; min eig = {eigs.min():.4e}"


# ─────────────────────────────────────────────────────────────────────
# 5. Density matrix on Cu2 (closed shell)
# ─────────────────────────────────────────────────────────────────────


def test_Cu2_density_matrix_trace_and_idempotency():
    topo = build_topology("[Cu][Cu]")
    basis = build_molecular_basis(topo)
    rho, S, eigvals, c = density_matrix_PT(topo, basis=basis)
    n_e = int(round(basis.total_occ))
    tr = float(np.trace(rho @ S))
    assert tr == pytest.approx(n_e, abs=1e-6)
    err = np.abs(rho @ S @ rho - 2.0 * rho).max()
    assert err < 1e-9, f"||rho S rho - 2 rho|| = {err:.4e}"


# ─────────────────────────────────────────────────────────────────────
# 6. f-block remains gated
# ─────────────────────────────────────────────────────────────────────


def test_f_block_still_gated():
    """Cerium (Z=58) is f-block: should still raise NotImplementedError
    until the next Phase A continuation."""
    with pytest.raises(NotImplementedError, match="f-block"):
        build_molecular_basis(build_topology("[Ce]"))
