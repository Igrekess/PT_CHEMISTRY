"""Phase 6.B.3 — MP2 amplitudes and 1-RDM correction.

Møller-Plesset 2nd-order quantities from a converged HF SCF, reusing
the existing PT-LCAO molecular grid for two-electron integrals.

API
===
* ``mo_eri_iajb(basis, c, n_occ, ...)`` : (ia|jb) integrals in MO basis
  built from a shared molecular grid Coulomb potential.
* ``mp2_amplitudes(eri, eps, n_occ)`` : t_ij^ab = (ia|jb)/Δε.
* ``mp2_energy(eri, t)`` : closed-shell MP2 correlation energy
  E_MP2 = -Σ t × [2(ia|jb) − (ib|ja)].
* ``mp2_density_correction(t)`` : occupied and virtual blocks of the
  1-RDM second-order correction ρ^(2). Used downstream by
  σ_d^MP2 = Tr(D · ρ^(2)) for chemical-shielding.
* ``mp2_at_hf(basis, eps, c, n_occ, ...)`` : one-shot MP2 wrapper.

Reference: Szabo & Ostlund, *Modern Quantum Chemistry*, §6.5.
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
    """
    grid, psi = _build_molecular_grid(
        basis, n_radial, n_theta, n_phi,
        use_becke=use_becke, lebedev_order=lebedev_order,
    )
    pts = grid.points
    w = grid.weights
    n_orb = c.shape[0]
    n_virt = n_orb - n_occ

    mo_grid = c.T @ psi                # (n_orb_MO, n_grid)
    occ_g = mo_grid[:n_occ]            # (n_occ, n_grid)
    vir_g = mo_grid[n_occ:]            # (n_virt, n_grid)

    eri = np.zeros((n_occ, n_virt, n_occ, n_virt))
    for j in range(n_occ):
        for b in range(n_virt):
            rho_jb = np.outer(c[:, j], c[:, n_occ + b])
            rho_jb = 0.5 * (rho_jb + rho_jb.T)
            V_jb = _coulomb_potential_on_grid(rho_jb, pts, w, psi)
            wV = w * V_jb
            for i in range(n_occ):
                for a in range(n_virt):
                    eri[i, a, j, b] = float(
                        np.sum(wV * occ_g[i] * vir_g[a])
                    )
    return eri


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
