"""Phase 6.B.3/4 — MP2 amplitudes, 1-RDM correction, and Z-vector.

Møller-Plesset 2nd-order quantities from a converged HF SCF, reusing
the existing PT-LCAO molecular grid for two-electron integrals.

Phase 6.B.4 (this addition) brings full Stanton-Gauss MP2-GIAO closure
via the orbital Z-vector ``A·Z = -L``, completing the relaxed 1-RDM
``D_MP2 = D_HF + d_occ ⊕ d_vir + 2·Z`` used by σ_p^MP2.

API
===

Phase 6.B.3 (existing)
  * ``mo_eri_iajb(basis, c, n_occ, ...)`` : (ia|jb) integrals in MO basis.
  * ``mp2_amplitudes(eri, eps, n_occ)`` : t_ij^ab = (ia|jb)/Δε.
  * ``mp2_energy(eri, t)`` : closed-shell MP2 correlation energy.
  * ``mp2_density_correction(t)`` : occ-occ and vir-vir 1-RDM blocks.
  * ``mp2_at_hf(basis, eps, c, n_occ, ...)`` : one-shot MP2 wrapper.
  * ``mp2_relax_orbitals(...)``: occupation-shift orbital rebuild.

Phase 6.B.4 (this module)
  * ``mo_eri_block(basis, c_p, c_q, c_r, c_s, ...)``: general (pq|rs)
    ERI builder over arbitrary MO-coefficient slices.
  * ``mp2_lagrangian(basis, c, n_occ, mp2_result, ...)``: closed-shell
    MP2 orbital Lagrangian L_ai (vir, occ) in chemist convention.
  * ``mp2_lagrangian_fd(basis, eps, c, n_occ, ...)``: finite-difference
    reference for L_ai used as gold-standard validation of the analytic
    formula in unit tests.
  * ``solve_z_vector(basis, eigvals, c, n_occ, lagrangian, ...)``:
    iterative solver reusing the CPHF orbital-Hessian machinery, but
    with a SYMMETRIC ρ^(1) and J^(1) coupling (real Z, real RHS).
  * ``mp2_density_relaxed_AO(c, n_occ, mp2_result, z_vector)``: full
    relaxed MP2 1-RDM in AO basis, including the off-diagonal Z-block.
  * ``mp2_paramagnetic_shielding(...)``: end-to-end pipeline producing
    σ_p^MP2-GIAO at a probe point given an SCF reference.

References
----------
* Szabo & Ostlund, *Modern Quantum Chemistry*, §6.5 (basic MP2).
* Stanton & Gauss, J. Chem. Phys. 97 (1992) 6602 (Z-vector).
* Helgaker & Jørgensen, *Molecular Electronic-Structure Theory* §13.7.
* Aikens & Gordon, J. Phys. Chem. A 107 (2003) 11399.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ptc.constants import COULOMB_EV_A
from ptc.lcao.density_matrix import PTMolecularBasis
from ptc.lcao.fock import _build_molecular_grid


@dataclass
class MP2Result:
    """Bundle of MP2 quantities at a single SCF point."""
    t: np.ndarray            # (n_occ, n_virt, n_occ, n_virt)
    eri_mo_iajb: np.ndarray  # (n_occ, n_virt, n_occ, n_virt)
    e_corr: float            # correlation energy in eV
    d_occ: np.ndarray        # (n_occ, n_occ)
    d_vir: np.ndarray        # (n_virt, n_virt)


def _coulomb_potential_on_grid_batch(n_grid_batch: np.ndarray,
                                        grid_pts: np.ndarray,
                                        grid_weights: np.ndarray,
                                        chunk_size: int = 200) -> np.ndarray:
    """Batched Coulomb potential of multiple densities on a shared grid.

    Computes ``V[r, s, g] = Σ_{g'} n[r, s, g'] · w[g'] / |g - g'|`` for a
    *batch* of MO-pair densities indexed by (r, s), with the diagonal
    (g = g') masked to zero — same convention as the per-pair builder
    ``_coulomb_potential_on_grid`` but a single BLAS matmul replaces the
    per-pair loop.

    Speedup vs. the per-pair builder is roughly proportional to
    ``n_r * n_s`` because the chunked distance evaluation is amortised
    across all pairs instead of being recomputed for each.

    Parameters
    ----------
    n_grid_batch : (n_r, n_s, n_grid) array of densities evaluated on the
        grid : ``n[r, s, g] = ψ_r(g) · ψ_s(g)``.
    grid_pts, grid_weights : same as the per-pair builder.
    chunk_size : grid block size for the outer loop ; controls peak
        memory of the (chunk × n_grid) distance kernel.

    Returns
    -------
    V : (n_r, n_s, n_grid) array of Coulomb potentials in eV·Å.
    """
    n_r, n_s, n_grid = n_grid_batch.shape
    if n_grid_batch.size == 0:
        return np.zeros_like(n_grid_batch)

    # Pre-multiply by weights and flatten (r, s) into one axis for BLAS:
    # n_w[g, rs_flat] = w[g] · n[r, s, g]
    n_w_flat = (n_grid_batch * grid_weights[None, None, :]) \
        .reshape(n_r * n_s, n_grid).T   # (n_grid, n_r*n_s) — Fortran-friendly

    V_flat = np.zeros((n_grid, n_r * n_s), dtype=float)

    # scipy.cdist is the BLAS-bound shortcut: ~2-3× faster than the
    # numpy broadcast + np.linalg.norm pattern, with 3× lower peak
    # memory (no explicit (chunk, N, 3) `diff` array).
    from scipy.spatial.distance import cdist

    for i_start in range(0, n_grid, chunk_size):
        i_end = min(i_start + chunk_size, n_grid)
        n_chunk = i_end - i_start
        dist = cdist(grid_pts[i_start:i_end], grid_pts)   # (chunk, n_grid)
        rng = np.arange(i_start, i_end)
        loc = np.arange(n_chunk)
        dist[loc, rng] = 1.0
        inv_dist = 1.0 / dist
        inv_dist[loc, rng] = 0.0                          # self-mask

        # V_chunk[g_chunk, rs] = Σ_{g'} inv_dist[g_chunk, g'] · n_w_flat[g', rs]
        V_flat[i_start:i_end, :] = inv_dist @ n_w_flat

    return (V_flat.T.reshape(n_r, n_s, n_grid)) * COULOMB_EV_A


def _coulomb_potential_on_grid(rho_pq: np.ndarray,
                                grid_pts: np.ndarray,
                                grid_weights: np.ndarray,
                                psi: np.ndarray,
                                chunk_size: int = 200) -> np.ndarray:
    """Coulomb potential of an AO density on the same molecular grid.

    V(r) = ∫ ρ(r')/|r-r'| dr' with ρ(r') = Σ_pq ρ_pq φ_p(r') φ_q(r').

    Returns (n_grid,) array in eV·Å.  The diagonal (r=r') is masked
    with zero weight to avoid the 1/0 singularity.
    """
    n_grid = grid_pts.shape[0]
    rho_psi = rho_pq @ psi                            # (n_orb, n_grid)
    n_at_grid = np.einsum("pg,pg->g", psi, rho_psi)
    nW = n_at_grid * grid_weights                     # (n_grid,)

    V = np.zeros(n_grid)
    for i_start in range(0, n_grid, chunk_size):
        i_end = min(i_start + chunk_size, n_grid)
        diff = grid_pts[i_start:i_end, None, :] - grid_pts[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        rng = np.arange(i_start, i_end)
        local_idx = np.arange(i_end - i_start)
        dist[local_idx, rng] = 1.0     # mask self
        contrib = nW[None, :] / dist
        contrib[local_idx, rng] = 0.0
        V[i_start:i_end] = contrib.sum(axis=1)
    return V * COULOMB_EV_A


def mo_eri_iajb(basis: PTMolecularBasis,
                  c: np.ndarray,
                  n_occ: int,
                  n_radial: int = 30,
                  n_theta: int = 14,
                  n_phi: int = 18,
                  use_becke: bool = True,
                  lebedev_order: int = 50) -> np.ndarray:
    """(ia|jb) ERIs in the MO basis from a molecular grid Coulomb potential.

    Strategy
    --------
    For each occupied-virtual pair (j, b), build a rank-1 AO density
    ρ^{jb}_pq = c_pj c_qb, evaluate the Coulomb potential V_{jb}(r) on
    the grid, then contract::

        (ia|jb) = ∫ χ_i(r) χ_a(r) V_{jb}(r) dr
                ≈ Σ_g w_g χ_i(g) χ_a(g) V_{jb}(g)

    Cost: O(n_occ · n_virt) potentials × O(N_grid²) per potential, then
    O(n_occ² · n_virt² · N_grid) contraction. Tractable for benzene SZ.

    Returns an array of shape (n_occ, n_virt, n_occ, n_virt) in eV.

    Phase 6.B.4d note
    -----------------
    Implementation now delegates to the vectorised :func:`mo_eri_block`
    with the canonical (occ, virt, occ, virt) slicing — same numerics,
    BLAS-bound performance, no triple Python loop.
    """
    c_occ = c[:, :n_occ]
    c_vir = c[:, n_occ:]
    return mo_eri_block(
        basis, c_occ, c_vir, c_occ, c_vir,
        n_radial=n_radial, n_theta=n_theta, n_phi=n_phi,
        use_becke=use_becke, lebedev_order=lebedev_order,
    )


def mp2_amplitudes(eri_iajb: np.ndarray,
                     eps: np.ndarray,
                     n_occ: int) -> np.ndarray:
    """t_ij^ab = (ia|jb) / (ε_a + ε_b − ε_i − ε_j) (closed-shell).

    Layout follows ``eri_iajb`` : t[i, a, j, b].
    """
    eps_i = eps[:n_occ]
    eps_a = eps[n_occ:]
    delta = (eps_a[None, :, None, None]
             + eps_a[None, None, None, :]
             - eps_i[:, None, None, None]
             - eps_i[None, None, :, None])
    return eri_iajb / delta


def mp2_energy(eri_iajb: np.ndarray, t: np.ndarray) -> float:
    """Closed-shell MP2 correlation energy in eV.

    E_MP2 = -Σ_ij^ab t[i,a,j,b] · [2(ia|jb) - (ib|ja)]

    The minus sign is from the convention t = (ia|jb)/(+Δε) used here.
    Standard MP2 with (ε_i + ε_j − ε_a − ε_b) gives positive energy
    contributions; this convention flips the sign and amplifies it
    correctly via ``-Σ``.

    ``(ib|ja) = eri_iajb.transpose(0, 3, 2, 1)``.
    """
    combo = 2.0 * eri_iajb - eri_iajb.transpose(0, 3, 2, 1)
    return -float(np.sum(t * combo))


def mp2_density_correction(t: np.ndarray):
    """Occupied and virtual blocks of the MP2 1-RDM correction.

    Closed-shell MP2 1-RDM correction (chemist-convention ERIs):

        ρ^(2)_ij = -2 Σ_k Σ_a Σ_b t[i,a,k,b] t[j,a,k,b]
        ρ^(2)_ab = +2 Σ_i Σ_j Σ_c t[i,a,j,c] t[i,b,j,c]

    Diagonal of these blocks shifts MO occupations away from
    {2, 0}.  Used by σ_d^MP2 = Tr(D · ρ^(2)).
    """
    d_occ = -2.0 * np.einsum("iakb,jakb->ij", t, t)
    d_vir = +2.0 * np.einsum("iajc,ibjc->ab", t, t)
    return d_occ, d_vir


def mp2_density_correction_AO(c: np.ndarray,
                                 n_occ: int,
                                 mp2_result: "MP2Result") -> np.ndarray:
    """Build the MP2 1-RDM correction in AO basis from the MO-basis blocks.

    The closed-shell HF density in AO basis is ρ_HF = 2 c_occ c_occ^T.
    The 2nd-order correction in MO basis has only diagonal blocks
    (d_occ in occ-occ, d_vir in vir-vir, zero elsewhere) ; the AO-basis
    image is::

        ρ^(2)_AO = c_occ · d_occ · c_occ^T + c_vir · d_vir · c_vir^T

    so that the MP2 density satisfies ``ρ_MP2 = ρ_HF + ρ^(2)_AO`` and
    feeds directly into ``shielding_diamagnetic_iso(ρ_MP2, basis, P)``.
    """
    c_occ = c[:, :n_occ]
    c_vir = c[:, n_occ:]
    return (c_occ @ mp2_result.d_occ @ c_occ.T
            + c_vir @ mp2_result.d_vir @ c_vir.T)


def mp2_relax_orbitals(basis: PTMolecularBasis,
                         topology,
                         c_HF: np.ndarray,
                         n_occ: int,
                         mp2_result: "MP2Result",
                         *,
                         z_vector: np.ndarray | None = None,
                         n_radial: int = 16,
                         n_theta: int = 10,
                         n_phi: int = 12,
                         use_becke: bool = False,
                         lebedev_order: int = 26,
                         nuclear_charge: str = "actual",
                         ) -> tuple:
    """MP2 orbital relaxation — re-diagonalise F[ρ_MP2].

    Pipeline
    --------
    1. ρ_MP2 = ρ_HF + correction_AO
       - if ``z_vector`` is None: correction = d_occ ⊕ d_vir
         (leading-order occupation-shift, Phase 6.B.3 behaviour)
       - if ``z_vector`` provided: correction = d_occ ⊕ d_vir + 2·Z
         (full Stanton-Gauss MP2 relaxation, Phase 6.B.4)
    2. F_MP2 = H_core + J[ρ_MP2] - ½ K[ρ_MP2]   (HF Fock built from ρ_MP2)
    3. Diagonalise F_MP2 in AO basis → (eigvals_MP2, c_MP2)

    With ``z_vector=None`` the output orbitals carry only the
    occupation-relaxation effect of MP2 (d_occ/d_vir 1-RDM blocks).
    With ``z_vector`` from ``solve_z_vector(...)`` they include the full
    off-diagonal correlation relaxation, completing the Stanton-Gauss
    MP2-GIAO closure for σ_p coupling.

    Parameters
    ----------
    basis, topology : usual PT-LCAO inputs.
    c_HF            : HF MO coefficients (n_orb, n_orb).
    n_occ           : number of doubly-occupied orbitals.
    mp2_result      : output of ``mp2_at_hf`` for this (basis, c_HF).
    z_vector        : optional Z-vector (n_virt, n_occ). When given, the
                       relaxed density includes the off-diagonal block.
    quadrature args : forwarded to the J / K builders.

    Returns
    -------
    (eigvals_MP2, c_MP2) : MO energies and coefficients after one round
    of orbital relaxation.
    """
    from ptc.lcao.density_matrix import (
        core_hamiltonian, overlap_matrix, solve_mo,
    )
    from ptc.lcao.fock import (
        coulomb_J_matrix, exchange_K_matrix, _build_molecular_grid,
    )

    # 1. AO-basis MP2 density (with or without Z-block)
    if z_vector is None:
        rho_corr = mp2_density_correction_AO(c_HF, n_occ, mp2_result)
    else:
        rho_corr = mp2_density_relaxed_AO(c_HF, n_occ, mp2_result, z_vector)
    rho_HF = 2.0 * c_HF[:, :n_occ] @ c_HF[:, :n_occ].T
    rho_MP2 = rho_HF + rho_corr

    # 2. H_core + J - K/2 with rho_MP2
    S = overlap_matrix(basis)
    H_core = core_hamiltonian(
        basis, S,
        n_radial=n_radial, n_theta=n_theta, n_phi=n_phi,
        nuclear_charge=nuclear_charge,
    )
    grid, psi = _build_molecular_grid(
        basis, n_radial, n_theta, n_phi,
        use_becke=use_becke, lebedev_order=lebedev_order,
    )
    J = coulomb_J_matrix(rho_MP2, basis, grid=grid, psi=psi)
    K = exchange_K_matrix(
        rho_MP2, basis, grid=grid, psi=psi, symmetry="sym",
    )
    F_MP2 = H_core + J - 0.5 * K
    F_MP2 = 0.5 * (F_MP2 + F_MP2.T)

    # 3. Diagonalise
    eigvals_MP2, c_MP2 = solve_mo(F_MP2, S)
    return eigvals_MP2, c_MP2


def mp2_at_hf(basis: PTMolecularBasis,
                eps: np.ndarray,
                c: np.ndarray,
                n_occ: int,
                **grid_kwargs) -> MP2Result:
    """One-shot MP2 from a converged HF SCF.

    Parameters
    ----------
    basis : built PTMolecularBasis
    eps   : MO eigenvalues (a.u. or eV — must match the unit of the
            integrals so Δε has the same unit ; eV is the convention
            used elsewhere in PTC).
    c     : MO coefficients in AO basis, shape (n_orb, n_orb).
    n_occ : number of doubly-occupied orbitals.
    grid_kwargs : forwarded to ``_build_molecular_grid``.

    Returns
    -------
    MP2Result with amplitudes, ERIs, correlation energy and the two
    1-RDM correction blocks.
    """
    eri = mo_eri_iajb(basis, c, n_occ, **grid_kwargs)
    t = mp2_amplitudes(eri, eps, n_occ)
    e_corr = mp2_energy(eri, t)
    d_occ, d_vir = mp2_density_correction(t)
    return MP2Result(
        t=t, eri_mo_iajb=eri, e_corr=e_corr,
        d_occ=d_occ, d_vir=d_vir,
    )


# ─────────────────────────────────────────────────────────────────────
# Phase 6.B.4 — Z-vector / Stanton-Gauss MP2-GIAO closure
# ─────────────────────────────────────────────────────────────────────


def mo_eri_block(basis: PTMolecularBasis,
                  c_p: np.ndarray,
                  c_q: np.ndarray,
                  c_r: np.ndarray,
                  c_s: np.ndarray,
                  *,
                  n_radial: int = 30,
                  n_theta: int = 14,
                  n_phi: int = 18,
                  use_becke: bool = True,
                  lebedev_order: int = 50,
                  grid=None,
                  psi=None) -> np.ndarray:
    """Generalised MO-basis (pq|rs) ERI block for arbitrary MO slices.

    Parameters
    ----------
    basis : PTMolecularBasis
    c_p, c_q, c_r, c_s : (n_orb, n_x) AO→MO coefficient slices for the
        four indices p, q, r, s of the chemist integral (pq|rs).
    grid, psi : optional pre-built molecular grid + AO values to reuse
        across multiple block calls (recommended for the Lagrangian).

    Strategy (vectorised — Phase 6.B.4d)
    ------------------------------------
    Project all MO sets onto the grid once, build the FULL batch of
    Coulomb potentials ``V[r, s, g]`` in a single BLAS matmul via
    ``_coulomb_potential_on_grid_batch``, then contract ``ψ_p ψ_q · w · V``
    in a single ``einsum`` (or chunked matmul). Replaces the
    pre-vectorisation triple Python loop and gives ~10-50× speedup,
    making DZP+ basis tractable.

    Mathematically::

        (pq|rs) = ∫ χ_p(r) χ_q(r) V_{rs}(r) dr
                ≈ Σ_g w_g · ψ_p(g) · ψ_q(g) · V_{rs}(g)

    Returns array of shape ``(c_p.shape[1], c_q.shape[1], c_r.shape[1],
    c_s.shape[1])`` in eV.
    """
    if grid is None or psi is None:
        grid, psi = _build_molecular_grid(
            basis, n_radial, n_theta, n_phi,
            use_becke=use_becke, lebedev_order=lebedev_order,
        )
    pts = grid.points
    w = grid.weights

    # Project each MO set onto the grid : ψ_x(g) = c_x.T @ AO(g)
    pg = c_p.T @ psi          # (n_p, n_grid)
    qg = c_q.T @ psi          # (n_q, n_grid)
    rg = c_r.T @ psi          # (n_r, n_grid)
    sg = c_s.T @ psi          # (n_s, n_grid)

    n_p = pg.shape[0]
    n_q = qg.shape[0]
    n_r = rg.shape[0]
    n_s = sg.shape[0]

    # Batch density on the grid: n_rs[r, s, g] = ψ_r(g) · ψ_s(g)
    # (broadcasting outer product, O(n_r·n_s·n_grid) memory)
    n_rs_g = rg[:, None, :] * sg[None, :, :]    # (n_r, n_s, n_grid)

    # Batched Coulomb potential V[r, s, g] in one BLAS matmul over chunks
    V_rs = _coulomb_potential_on_grid_batch(n_rs_g, pts, w)
    weighted_V = V_rs * w[None, None, :]        # absorb integration weight

    # Final contraction (pq|rs) = Σ_g pg[p,g] qg[q,g] weighted_V[r,s,g]
    # The matmul-based reshape avoids pathological einsum paths and
    # gives optimal BLAS throughput for moderate sizes. Memory of the
    # intermediate phi_pq_g is n_p·n_q·n_grid floats, which we keep in
    # bounds by the basis sizes used in PT-LCAO unit tests and DZP+.
    phi_pq_g = (pg[:, None, :] * qg[None, :, :]).reshape(n_p * n_q, -1)
    weighted_V_flat = weighted_V.transpose(2, 0, 1).reshape(-1, n_r * n_s)
    eri = (phi_pq_g @ weighted_V_flat).reshape(n_p, n_q, n_r, n_s)
    return eri


def _t_tilde(t: np.ndarray) -> np.ndarray:
    """Closed-shell antisymmetrised amplitudes T̃[i,a,j,b] = 2t - t.swap.

    Convention: ``t.swap[i,a,j,b] = t[i,b,j,a]`` (swap of the two
    virtual indices a ↔ b at fixed (i, j)). This is the standard
    closed-shell "spin-summed" tilde combination that arises in MP2
    energy and gradient expressions.
    """
    return 2.0 * t - t.transpose(0, 3, 2, 1)


def mp2_lagrangian(basis: PTMolecularBasis,
                    c: np.ndarray,
                    n_occ: int,
                    mp2_result: "MP2Result",
                    *,
                    grid=None,
                    psi=None,
                    n_radial: int = 30,
                    n_theta: int = 14,
                    n_phi: int = 18,
                    use_becke: bool = True,
                    lebedev_order: int = 50) -> np.ndarray:
    """Closed-shell MP2 orbital Lagrangian L_ai (chemist convention).

    Returns an array of shape (n_virt, n_occ) such that the Z-vector
    equation ``A·Z = -L`` produces the orbital relaxation completing the
    MP2 1-RDM. A is the same orbital Hessian as in CPHF.

    Formula (direct chain-rule derivation; matches Helgaker §13.7 up to
    sign convention). For closed-shell MP2 with chemist (pq|rs) integrals
    and amplitudes t[i,a,j,b] = (ia|jb)/Δε, define T̃ = 2t - t.swap_ab.
    Then differentiating ``E_MP2 = -Σ T̃_t × ERI`` with respect to the
    occupied-virtual orbital rotation U_{a,i} (with ε held fixed by
    Brillouin) yields::

        L_ai = -4 Σ_jbc T̃[i,b,j,c] (ab|jc)
              + 4 Σ_jkb T̃[j,a,k,b] (ki|jb)

    The first term is the virtual-virtual contraction with ERI block
    (vv|ov) — captures correlation density relaxing into the virtual
    space. The second is the occupied-occupied contraction with (oo|ov)
    — captures correlation density depleting the occupied space. The
    overall sign is fixed so that L_ai matches ∂E_MP2/∂U_ai (validated
    by the finite-difference reference in the test suite).

    Cost
    ----
    Two general MO ERI block builds:
      - (ab|jc) of shape (n_virt, n_virt, n_occ, n_virt)
      - (ki|jb) of shape (n_occ, n_occ, n_occ, n_virt)
    Plus two ``einsum`` contractions. Tractable for SZ benzene and
    smaller. For larger systems the (vv|ov) block dominates.
    """
    if grid is None or psi is None:
        grid, psi = _build_molecular_grid(
            basis, n_radial, n_theta, n_phi,
            use_becke=use_becke, lebedev_order=lebedev_order,
        )

    c_occ = c[:, :n_occ]
    c_vir = c[:, n_occ:]

    t = mp2_result.t
    T = _t_tilde(t)          # (n_occ, n_virt, n_occ, n_virt)

    # (ab|jc) — virt, virt, occ, virt
    eri_vvov = mo_eri_block(basis, c_vir, c_vir, c_occ, c_vir,
                              grid=grid, psi=psi)
    # (ki|jb) — occ, occ, occ, virt
    eri_ooov = mo_eri_block(basis, c_occ, c_occ, c_occ, c_vir,
                              grid=grid, psi=psi)

    # First term  : -4 Σ_jbc T̃[i,b,j,c] (ab|jc) → indexed by (a, i)
    L_term1 = -4.0 * np.einsum("ibjc,abjc->ai", T, eri_vvov)
    # Second term : +4 Σ_jkb T̃[j,a,k,b] (ki|jb) → indexed by (a, i)
    L_term2 = 4.0 * np.einsum("jakb,kijb->ai", T, eri_ooov)

    return L_term1 + L_term2


def mp2_lagrangian_fd(basis: PTMolecularBasis,
                       eps: np.ndarray,
                       c: np.ndarray,
                       n_occ: int,
                       *,
                       step: float = 1.0e-3,
                       grid_kwargs: dict | None = None) -> np.ndarray:
    """Finite-difference reference for L_ai (gold-standard validation).

    Computes ``L_ai = ∂E_MP2/∂U_ai`` at U=0 by central differencing of
    a small orbital rotation that mixes occupied i with virtual a:

        c'(U_ai) = c · exp(K),   K[a,i] = -K[i,a] = U,
        E_MP2(U) = MP2 energy at the rotated orbitals (with FIXED ε).

    The Brillouin theorem at HF reference guarantees that ε_p does not
    change at first order in U, so using fixed eps is consistent with
    the analytic Lagrangian.

    Cost: ``O(n_occ × n_virt)`` MP2 energy evaluations. Use only for
    small systems (H₂, LiH) as a unit-test reference.
    """
    if grid_kwargs is None:
        grid_kwargs = dict(n_radial=8, n_theta=6, n_phi=8,
                              use_becke=False, lebedev_order=14)

    n_orb = c.shape[0]
    n_virt = n_orb - n_occ

    def E_at(c_rot: np.ndarray) -> float:
        eri = mo_eri_iajb(basis, c_rot, n_occ, **grid_kwargs)
        t = mp2_amplitudes(eri, eps, n_occ)
        return mp2_energy(eri, t)

    L_fd = np.zeros((n_virt, n_occ))
    for a in range(n_virt):
        for i in range(n_occ):
            K = np.zeros((n_orb, n_orb))
            # Off-diagonal anti-Hermitian element: K[a,i] = +U, K[i,a] = -U
            # (using MO indices: column n_occ+a is virtual a, column i is occ)
            K[n_occ + a, i] = +step
            K[i, n_occ + a] = -step
            R_plus = _expm_skew_small(K)
            c_plus = c @ R_plus
            E_plus = E_at(c_plus)

            R_minus = _expm_skew_small(-K)
            c_minus = c @ R_minus
            E_minus = E_at(c_minus)

            L_fd[a, i] = (E_plus - E_minus) / (2.0 * step)
    return L_fd


def _expm_skew_small(K: np.ndarray) -> np.ndarray:
    """Tiny exp(K) for skew-Hermitian K with sparse single-pair support.

    For unit tests we only ever rotate one (a, i) pair, so K has at
    most two non-zero entries (K[a,i] = -K[i,a] = U). This admits a
    closed-form 2×2 rotation; for safety we fall back on a Taylor
    series for general K. Strictly K = -K.T, so exp(K) is orthogonal.
    """
    # Generic small-K Taylor expansion (4 terms is enough for U ~ 1e-3)
    n = K.shape[0]
    R = np.eye(n)
    Kp = np.eye(n)
    fact = 1.0
    for k in range(1, 8):
        Kp = Kp @ K
        fact *= k
        R = R + Kp / fact
    return R


def solve_z_vector(basis: PTMolecularBasis,
                     mo_eigvals: np.ndarray,
                     mo_coeffs: np.ndarray,
                     n_occ: int,
                     lagrangian: np.ndarray,
                     *,
                     max_iter: int = 30,
                     tol: float = 1.0e-5,
                     damping: float = 0.5,
                     level_shift: float = 0.0,
                     n_radial_grid: int = 24,
                     n_theta_grid: int = 12,
                     n_phi_grid: int = 16,
                     use_becke: bool = False,
                     lebedev_order: int = 50,
                     verbose: bool = False) -> np.ndarray:
    """Solve the closed-shell Z-vector equation A·Z = -L.

    The orbital Hessian A is the SAME as in CPHF (closed-shell, real
    perturbation), but the response density ρ^(1) is now SYMMETRIC
    (since L is real and Z is real) — both J^(1) and K^(1) contribute,
    in contrast to the magnetic-perturbation CPHF where ρ^(1) is
    antisymmetric and J^(1) vanishes.

    Iterative scheme (matches ``coupled_cphf_response``)::

        ρ^(1)_AO = 2 Σ_ai Z[a,i] (c_a c_i^T + c_i c_a^T)        (sym, factor 2 = spin)
        F^(1)_AO = J[ρ^(1)] - (1/2) K[ρ^(1)]                     (closed-shell HF)
        F^(1)_ai = c_a^T F^(1)_AO c_i
        Z_new[a,i] = -(L[a,i] + F^(1)_ai) / (ε_a - ε_i)

    Expanding gives the standard MO-basis orbital Hessian
    A_ai,bj = (ε_a-ε_i) δ + 4(ai|bj) - (ab|ij) - (aj|ib),
    consistent with Helgaker §13.7 / Aikens-Gordon eq. 14.

    Damping: Z = damp · Z + (1 - damp) · Z_new.
    Convergence: max |Z_new - Z| < tol.

    Parameters
    ----------
    lagrangian : (n_virt, n_occ) RHS L_ai. Computed via ``mp2_lagrangian``.

    Returns
    -------
    Z : (n_virt, n_occ) Z-vector amplitudes (real, dimensionless ish —
        same units as L / (ε_a - ε_i), in practice eV/eV → 1).
    """
    from ptc.lcao.fock import (
        _build_molecular_grid as _build_grid,
        coulomb_J_matrix,
        exchange_K_matrix,
    )

    n_orb = mo_coeffs.shape[0]
    n_virt = n_orb - n_occ
    if n_virt == 0 or n_occ == 0:
        return np.zeros((n_virt, n_occ))

    eps_a = mo_eigvals[n_occ:]
    eps_i = mo_eigvals[:n_occ]
    diff = eps_a[:, None] - eps_i[None, :]    # (n_virt, n_occ)
    if level_shift > 0.0:
        diff = np.maximum(diff, level_shift)

    c_occ = mo_coeffs[:, :n_occ]
    c_vir = mo_coeffs[:, n_occ:]

    grid, psi = _build_grid(
        basis, n_radial_grid, n_theta_grid, n_phi_grid,
        use_becke=use_becke, lebedev_order=lebedev_order,
    )

    # Uncoupled initial guess
    Z = -lagrangian / diff

    for it in range(max_iter):
        # Symmetric AO density from current Z
        rho1 = 2.0 * (c_vir @ Z @ c_occ.T + c_occ @ Z.T @ c_vir.T)

        # Two-electron response Fock for closed-shell sym density.
        # Convention: ρ^(1) carries the factor of 2 from spin summation,
        # so the standard HF combination J - K/2 reproduces the orbital
        # Hessian A_ai,bj = 4(ai|bj) - (ab|ij) - (aj|ib).
        J1_AO = coulomb_J_matrix(rho1, basis, grid=grid, psi=psi)
        K1_AO = exchange_K_matrix(rho1, basis, grid=grid, psi=psi,
                                     symmetry="sym")
        F1_AO = J1_AO - 0.5 * K1_AO

        # MO projection
        F1_MO = mo_coeffs.T @ F1_AO @ mo_coeffs
        F1_ai = F1_MO[n_occ:, :n_occ]      # (n_virt, n_occ)

        # CPHF-style update
        Z_new = -(lagrangian + F1_ai) / diff
        change = float(np.abs(Z_new - Z).max())
        Z = damping * Z + (1.0 - damping) * Z_new

        if verbose:
            print(f"  Z-vector iter {it}: max|dZ|={change:.4e}")

        if change < tol:
            break

    return Z


def mp2_density_relaxed_AO(c: np.ndarray,
                             n_occ: int,
                             mp2_result: "MP2Result",
                             z_vector: np.ndarray) -> np.ndarray:
    """Full orbital-relaxed MP2 1-RDM in AO basis (with Z-vector).

    Builds the relaxed MP2 density::

        D^MP2 = D^HF + D^(2)_AO

    where D^(2) in MO basis has the closed-shell block structure::

        D^(2)_MO[i, j] = d_occ[i, j]              (occ-occ)
        D^(2)_MO[a, b] = d_vir[a, b]              (vir-vir)
        D^(2)_MO[a, i] = D^(2)_MO[i, a] = 2 Z[a,i]   (off-diagonal)

    The factor of 2 in the off-diagonal block comes from closed-shell
    spin summation (``Z`` is the spatial-orbital Z-vector).

    Returns ``(n_orb, n_orb)`` symmetric AO matrix that, added to ``D^HF
    = 2 c_occ c_occ^T``, gives the relaxed MP2 density.
    """
    c_occ = c[:, :n_occ]
    c_vir = c[:, n_occ:]

    # MO-basis correction (without HF reference: D_HF is added separately)
    # Build via AO contraction directly to avoid an explicit MO matrix.
    rho_corr_AO = (
        c_occ @ mp2_result.d_occ @ c_occ.T
        + c_vir @ mp2_result.d_vir @ c_vir.T
        + 2.0 * c_vir @ z_vector @ c_occ.T
        + 2.0 * c_occ @ z_vector.T @ c_vir.T
    )
    return 0.5 * (rho_corr_AO + rho_corr_AO.T)


def mp2_paramagnetic_shielding(basis: PTMolecularBasis,
                                  mo_eigvals: np.ndarray,
                                  mo_coeffs: np.ndarray,
                                  n_occ: int,
                                  probe: np.ndarray,
                                  *,
                                  mp2_result: "MP2Result | None" = None,
                                  z_vector: np.ndarray | None = None,
                                  use_z_vector: bool = True,
                                  mp2_kwargs: dict | None = None,
                                  lagrangian_kwargs: dict | None = None,
                                  z_vector_kwargs: dict | None = None,
                                  ) -> dict:
    """End-to-end MP2-GIAO paramagnetic shielding at ``probe``.

    Computes the diamagnetic (1-RDM-only) part of σ from the relaxed
    MP2 1-RDM. The full paramagnetic σ_p^MP2 with response U^(1) is the
    next continuation; this wrapper exposes the relaxed-density piece
    that drops directly into ``shielding_diamagnetic_iso``.

    Parameters
    ----------
    basis, mo_eigvals, mo_coeffs : converged HF SCF outputs.
    n_occ : doubly-occupied count.
    probe : (3,) Cartesian probe point in Å.
    mp2_result : optional pre-computed MP2Result (skips MP2 redo).
    z_vector : optional pre-computed Z-vector (skips solver).
    use_z_vector : if False, sets Z = 0 (Phase 6.B.3 pre-Z behaviour).
    mp2_kwargs / lagrangian_kwargs / z_vector_kwargs : forwarded to the
        respective sub-builds.

    Returns a dict with all intermediate objects (mp2_result, z_vector,
    rho_corr_AO, rho_MP2_AO, sigma_HF, sigma_MP2_unrelaxed,
    sigma_MP2_relaxed, delta_sigma_HF_to_MP2).
    """
    from ptc.lcao.giao import shielding_diamagnetic_iso

    mp2_kwargs = mp2_kwargs or {}
    lagrangian_kwargs = lagrangian_kwargs or {}
    z_vector_kwargs = z_vector_kwargs or {}

    if mp2_result is None:
        mp2_result = mp2_at_hf(basis, mo_eigvals, mo_coeffs, n_occ,
                                  **mp2_kwargs)

    rho_HF = 2.0 * mo_coeffs[:, :n_occ] @ mo_coeffs[:, :n_occ].T

    # Pre-Z (occupation-shift only) MP2 density
    rho_corr_unrelax = mp2_density_correction_AO(mo_coeffs, n_occ, mp2_result)
    rho_MP2_unrelax = rho_HF + rho_corr_unrelax

    if use_z_vector:
        if z_vector is None:
            L = mp2_lagrangian(basis, mo_coeffs, n_occ, mp2_result,
                                **lagrangian_kwargs)
            z_vector = solve_z_vector(
                basis, mo_eigvals, mo_coeffs, n_occ, L,
                **z_vector_kwargs,
            )
        rho_corr_relax = mp2_density_relaxed_AO(
            mo_coeffs, n_occ, mp2_result, z_vector,
        )
    else:
        z_vector = np.zeros((mo_coeffs.shape[0] - n_occ, n_occ))
        rho_corr_relax = rho_corr_unrelax

    rho_MP2_relax = rho_HF + rho_corr_relax

    sigma_HF = shielding_diamagnetic_iso(rho_HF, basis, probe)
    sigma_unrelax = shielding_diamagnetic_iso(rho_MP2_unrelax, basis, probe)
    sigma_relax = shielding_diamagnetic_iso(rho_MP2_relax, basis, probe)

    return {
        "mp2_result": mp2_result,
        "z_vector": z_vector,
        "rho_HF": rho_HF,
        "rho_corr_unrelax": rho_corr_unrelax,
        "rho_corr_relax": rho_corr_relax,
        "rho_MP2_unrelax": rho_MP2_unrelax,
        "rho_MP2_relax": rho_MP2_relax,
        "sigma_HF": sigma_HF,
        "sigma_MP2_unrelaxed": sigma_unrelax,
        "sigma_MP2_relaxed": sigma_relax,
        "delta_sigma_HF_to_MP2": sigma_relax - sigma_HF,
    }


def mp2_paramagnetic_shielding_coupled(basis: PTMolecularBasis,
                                          topology,
                                          mo_eigvals_HF: np.ndarray,
                                          mo_coeffs_HF: np.ndarray,
                                          n_occ: int,
                                          K_probe: np.ndarray,
                                          *,
                                          mp2_result: "MP2Result | None" = None,
                                          z_vector: np.ndarray | None = None,
                                          relax_kwargs: dict | None = None,
                                          cphf_kwargs: dict | None = None,
                                          mp2_kwargs: dict | None = None,
                                          lagrangian_kwargs: dict | None = None,
                                          z_vector_kwargs: dict | None = None,
                                          isotropic: bool = True,
                                          ) -> dict:
    """Full Stanton-Gauss σ_p MP2-GIAO via CPHF on Z-relaxed orbitals.

    Pipeline
    --------
    1. MP2 amplitudes from HF reference (skipped if ``mp2_result`` given)
    2. Lagrangien L_ai (skipped if ``z_vector`` given)
    3. Z-vector A·Z = -L (skipped if ``z_vector`` given)
    4. Full MP2 relaxation: re-diagonalise F[ρ_HF + d_oo ⊕ d_vv + 2·Z]
       → (eigvals_MP2_full, c_MP2_full)
    5. coupled_cphf_response on (eigvals_MP2_full, c_MP2_full) → U_imag
    6. paramagnetic_shielding_iso_coupled (or _tensor) at K_probe → σ_p

    Comparison points (returned in dict):
      - σ_p^HF         : standard CPHF on HF orbitals (Phase 5c baseline)
      - σ_p^MP2_LO     : CPHF on occupation-shift-relaxed orbitals
                         (Phase 6.B.3 leading-order, no Z block)
      - σ_p^MP2_full   : CPHF on Z-vector-relaxed orbitals
                         (Phase 6.B.4 full Stanton-Gauss closure)

    Parameters
    ----------
    basis, topology : built PT_LCAO inputs.
    mo_eigvals_HF, mo_coeffs_HF : converged HF SCF outputs.
    n_occ : doubly-occupied count (n_e_total // 2).
    K_probe : (3,) probe point in Å.
    mp2_result, z_vector : optional pre-computed objects (skip recomputation).
    relax_kwargs, cphf_kwargs, mp2_kwargs, lagrangian_kwargs,
        z_vector_kwargs : grid / iteration parameters.
    isotropic : if True, return scalar σ_iso ; if False, the full 3×3 tensor.

    Returns
    -------
    dict with σ_p^HF, σ_p^MP2_LO, σ_p^MP2_full, the three sets of MO orbitals,
    and the Z-vector / mp2_result objects. Sigmas are in ppm.
    """
    from ptc.lcao.fock import (
        paramagnetic_shielding_iso_coupled,
        paramagnetic_shielding_tensor_coupled,
    )

    relax_kwargs = relax_kwargs or {}
    cphf_kwargs = cphf_kwargs or {}
    mp2_kwargs = mp2_kwargs or {}
    lagrangian_kwargs = lagrangian_kwargs or {}
    z_vector_kwargs = z_vector_kwargs or {}

    K_probe = np.asarray(K_probe, dtype=float)
    n_e_total = 2 * n_occ

    # Step 1-3: MP2 + Lagrangian + Z-vector
    if mp2_result is None:
        mp2_result = mp2_at_hf(
            basis, mo_eigvals_HF, mo_coeffs_HF, n_occ, **mp2_kwargs,
        )
    if z_vector is None:
        L = mp2_lagrangian(
            basis, mo_coeffs_HF, n_occ, mp2_result, **lagrangian_kwargs,
        )
        z_vector = solve_z_vector(
            basis, mo_eigvals_HF, mo_coeffs_HF, n_occ, L,
            **z_vector_kwargs,
        )

    # Step 4a: leading-order relaxed orbitals (occupation-shift only)
    eigvals_LO, c_LO = mp2_relax_orbitals(
        basis, topology, mo_coeffs_HF, n_occ, mp2_result,
        z_vector=None, **relax_kwargs,
    )
    # Step 4b: full Z-vector-relaxed orbitals
    eigvals_full, c_full = mp2_relax_orbitals(
        basis, topology, mo_coeffs_HF, n_occ, mp2_result,
        z_vector=z_vector, **relax_kwargs,
    )

    # The iso version does not accept ``use_becke`` / ``lebedev_order`` ;
    # filter them transparently so callers can pass a unified cphf_kwargs.
    if isotropic:
        iso_kwargs = {k: v for k, v in cphf_kwargs.items()
                       if k not in ("use_becke", "lebedev_order")}
        sigma_HF = paramagnetic_shielding_iso_coupled(
            basis, mo_eigvals_HF, mo_coeffs_HF, n_e_total, K_probe,
            **iso_kwargs,
        )
        sigma_LO = paramagnetic_shielding_iso_coupled(
            basis, eigvals_LO, c_LO, n_e_total, K_probe, **iso_kwargs,
        )
        sigma_full = paramagnetic_shielding_iso_coupled(
            basis, eigvals_full, c_full, n_e_total, K_probe, **iso_kwargs,
        )
    else:
        sigma_HF = paramagnetic_shielding_tensor_coupled(
            basis, mo_eigvals_HF, mo_coeffs_HF, n_e_total, K_probe,
            **cphf_kwargs,
        )
        sigma_LO = paramagnetic_shielding_tensor_coupled(
            basis, eigvals_LO, c_LO, n_e_total, K_probe, **cphf_kwargs,
        )
        sigma_full = paramagnetic_shielding_tensor_coupled(
            basis, eigvals_full, c_full, n_e_total, K_probe, **cphf_kwargs,
        )

    return {
        "mp2_result": mp2_result,
        "z_vector": z_vector,
        "eigvals_HF": mo_eigvals_HF,
        "c_HF": mo_coeffs_HF,
        "eigvals_MP2_LO": eigvals_LO,
        "c_MP2_LO": c_LO,
        "eigvals_MP2_full": eigvals_full,
        "c_MP2_full": c_full,
        "sigma_p_HF": sigma_HF,
        "sigma_p_MP2_LO": sigma_LO,
        "sigma_p_MP2_full": sigma_full,
        "delta_LO_minus_HF": (np.array(sigma_LO) - np.array(sigma_HF)).tolist()
            if not isotropic else float(sigma_LO - sigma_HF),
        "delta_full_minus_LO": (np.array(sigma_full) - np.array(sigma_LO)).tolist()
            if not isotropic else float(sigma_full - sigma_LO),
    }
