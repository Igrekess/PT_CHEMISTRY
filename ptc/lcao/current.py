"""ptc/lcao/current.py — Induced current density and ring strength.

GIMIC-style induced current density from CPHF coupled response.
Distinct from the scalar Pauling-London NICS in `ptc/nics.py`: this module
treats the **vectorial** current and integrates flux through bond
cross-sections. Required for ligand-coordinated complexes (e.g. Bi3@U2)
where the scalar NICS is masked by counter-currents from caps/ligands.

Notation
========
For a static external magnetic field B applied along the z-axis the
linear-response induced current density is

    j_alpha(r) = j_alpha^para(r) + j_alpha^dia(r)        (per unit B_z)

Paramagnetic (CPHF coupled, U_imag = response amplitude):

    j_alpha^p(r) = 2 * sum_{occ i, vir a} U_imag[z, a, i] *
                       [ psi_i(r) * d_alpha psi_a(r)
                         - psi_a(r) * d_alpha psi_i(r) ]

Diamagnetic (Larmor, B = z_hat):

    j_alpha^d(r) = -(alpha/2) * (B_hat x r)_alpha * rho(r)

Ring current strength is the flux of j through a half-plane bisecting
one bond, normal to the bond axis.

References
==========
- T. Heine et al., GIMIC review, J. Comput. Chem. 30 (2009) 838.
- Pelloni & Lazzeretti, Theor. Chem. Acc. 117 (2007) 903.
- D. Sundholm et al., WIREs Comp. Mol. Sci. 6 (2016) 639.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ptc.lcao.density_matrix import (
    PTMolecularBasis,
    build_molecular_basis,
    density_matrix_PT,
)
from ptc.lcao.fock import coupled_cphf_response
from ptc.lcao.giao import evaluate_sto, evaluate_sto_gradient_analytic
from ptc.topology import Topology

ALPHA_FS = 1.0 / 137.035999_084   # fine-structure constant (1/c in a.u.)


# ─────────────────────────────────────────────────────────────────────
# Pre-computed response container
# ─────────────────────────────────────────────────────────────────────


@dataclass
class _ResponseData:
    basis: PTMolecularBasis
    rho: np.ndarray             # (N_orb, N_orb) ground-state density
    mo_coeffs: np.ndarray       # (N_orb, N_orb) MOs
    U_imag: np.ndarray          # (3, n_virt, n_occ) CPHF response
    n_occ: int


def precompute_response(topology: Topology,
                          basis_type: str = "SZ",
                          scf_density: str = "hueckel",
                          polarisation: bool = False,
                          include_f_block_d_shell: bool = False,
                          cphf_kwargs: Optional[dict] = None,
                          ) -> _ResponseData:
    """Build basis + ground-state density + CPHF response.

    Supports the full Phase 2-6 basis hierarchy via ``basis_type`` :
    ``'SZ'``, ``'DZ'``, ``'TZ'``, ``'DZP'``, ``'TZP'``, ``'DZ2P'``,
    ``'DZPD'``.  ``polarisation`` enables (l_val+1) polarisation shell.
    """
    basis = build_molecular_basis(
        topology, polarisation=polarisation,
        basis_type=basis_type,
        include_f_block_d_shell=include_f_block_d_shell,
    )
    if scf_density == "hueckel":
        rho, _S, eigvals, c = density_matrix_PT(topology, basis=basis)
    else:
        from ptc.lcao.fock import density_matrix_PT_scf
        rho, _S, eigvals, c, _conv, _resid = density_matrix_PT_scf(
            topology, basis=basis, mode=scf_density,
        )
    n_e = int(round(basis.total_occ))
    n_occ = n_e // 2
    cphf_kw = dict(cphf_kwargs) if cphf_kwargs else {}
    U_imag = coupled_cphf_response(basis, eigvals, c, n_e, **cphf_kw)
    return _ResponseData(
        basis=basis, rho=rho, mo_coeffs=c,
        U_imag=U_imag, n_occ=n_occ,
    )


# ─────────────────────────────────────────────────────────────────────
# AO evaluation on grid
# ─────────────────────────────────────────────────────────────────────


def _ao_grid_values(basis: PTMolecularBasis,
                     pts: np.ndarray,
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (chi, dchi) with shapes (N_orb, N_pts) and (N_orb, N_pts, 3)."""
    N = basis.n_orbitals
    M = pts.shape[0]
    chi = np.zeros((N, M))
    dchi = np.zeros((N, M, 3))
    for k, orb in enumerate(basis.orbitals):
        atom_pos = basis.coords[basis.atom_index[k]]
        chi[k] = evaluate_sto(orb, pts, atom_pos)
        dchi[k] = evaluate_sto_gradient_analytic(orb, pts, atom_pos)
    return chi, dchi


def _b_cross_r(grid_pts: np.ndarray, beta: int) -> np.ndarray:
    """Compute (B_hat x r) for each grid point given the field axis beta."""
    cross = np.zeros_like(grid_pts)
    if beta == 2:
        cross[:, 0] = -grid_pts[:, 1]
        cross[:, 1] = grid_pts[:, 0]
    elif beta == 1:
        cross[:, 0] = grid_pts[:, 2]
        cross[:, 2] = -grid_pts[:, 0]
    elif beta == 0:
        cross[:, 1] = -grid_pts[:, 2]
        cross[:, 2] = grid_pts[:, 1]
    else:
        raise ValueError("beta must be 0, 1, or 2")
    return cross


# ─────────────────────────────────────────────────────────────────────
# Current density
# ─────────────────────────────────────────────────────────────────────


def current_density_at_points(resp: _ResponseData,
                                grid_pts: np.ndarray,
                                beta: int = 2,
                                gauge: str = "common",
                                ) -> Tuple[np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray]:
    """Induced current density per unit B_beta at grid_pts.

    gauge :
      - "common"      : j_dia uses the global origin (0, 0, 0).
      - "ipsocentric" : Becke-weighted local atomic origin (CTOCD-PZ-like).
                         Diagnostic only (not gauge-invariant alone).
      - "giao"        : per-AO-pair midpoint origin R_munu = (R_mu+R_nu)/2,
                         with the matching London-phase derivative term
                         added to j_para (TERM2).

    Returns (j_total, j_para, j_dia, rho_grid).
    """
    if grid_pts.ndim != 2 or grid_pts.shape[1] != 3:
        raise ValueError("grid_pts must be (N, 3)")
    if gauge not in ("common", "ipsocentric", "giao"):
        raise ValueError(
            f"gauge must be 'common'/'ipsocentric'/'giao', got {gauge!r}"
        )

    basis = resp.basis
    n_occ = resp.n_occ
    c = resp.mo_coeffs

    chi, dchi = _ao_grid_values(basis, grid_pts)

    c_occ = c[:, :n_occ]
    c_virt = c[:, n_occ:]

    psi_occ = c_occ.T @ chi
    psi_virt = c_virt.T @ chi
    grad_occ = np.einsum("ki,kpd->ipd", c_occ, dchi)
    grad_virt = np.einsum("ka,kpd->apd", c_virt, dchi)

    U_b = resp.U_imag[beta]

    # Paramagnetic term: 2 * sum_{i,a} U[a,i] *
    #                    [psi_i d_alpha psi_a - psi_a d_alpha psi_i]
    M1 = np.einsum("ai,apd->ipd", U_b, grad_virt)
    T1 = np.einsum("ip,ipd->pd", psi_occ, M1)
    M2 = np.einsum("ai,ipd->apd", U_b, grad_occ)
    T2 = np.einsum("ap,apd->pd", psi_virt, M2)
    j_para = 2.0 * (T1 - T2)

    # Density on grid
    rho_grid = np.einsum("mn,mp,np->p", resp.rho, chi, chi)

    if gauge == "common":
        cross = _b_cross_r(grid_pts, beta)
        j_dia = -(ALPHA_FS / 2.0) * cross * rho_grid[:, None]
    elif gauge == "ipsocentric":
        from ptc.lcao.grid import becke_partition_weights
        w = becke_partition_weights(basis.coords, grid_pts)
        R_local = w @ basis.coords
        rel = grid_pts - R_local
        cross = _b_cross_r(rel, beta)
        j_dia = -(ALPHA_FS / 2.0) * cross * rho_grid[:, None]
    else:  # gauge == "giao"
        # j_dia stays at common-origin; add TERM2 to j_para via
        # London-phase derivative on AO gradient.
        cross = _b_cross_r(grid_pts, beta)
        j_dia = -(ALPHA_FS / 2.0) * cross * rho_grid[:, None]

        R_orb = basis.coords[basis.atom_index]
        Delta = R_orb[:, None, :] - R_orb[None, :, :]
        x_g = grid_pts[:, 0]
        y_g = grid_pts[:, 1]
        z_g = grid_pts[:, 2]
        if beta == 2:
            Mx = resp.rho * Delta[..., 0]
            My = resp.rho * Delta[..., 1]
            ein_x = np.einsum("mn,np,mpa->pa", Mx, chi, dchi)
            ein_y = np.einsum("mn,np,mpa->pa", My, chi, dchi)
            term2 = y_g[:, None] * ein_x - x_g[:, None] * ein_y
        elif beta == 1:
            Mz = resp.rho * Delta[..., 2]
            Mx = resp.rho * Delta[..., 0]
            ein_z = np.einsum("mn,np,mpa->pa", Mz, chi, dchi)
            ein_x = np.einsum("mn,np,mpa->pa", Mx, chi, dchi)
            term2 = x_g[:, None] * ein_z - z_g[:, None] * ein_x
        else:  # beta == 0
            My = resp.rho * Delta[..., 1]
            Mz = resp.rho * Delta[..., 2]
            ein_y = np.einsum("mn,np,mpa->pa", My, chi, dchi)
            ein_z = np.einsum("mn,np,mpa->pa", Mz, chi, dchi)
            term2 = z_g[:, None] * ein_y - y_g[:, None] * ein_z
        j_para = j_para - (ALPHA_FS / 2.0) * term2

    return j_para + j_dia, j_para, j_dia, rho_grid


# ─────────────────────────────────────────────────────────────────────
# Ring / bond current strength
# ─────────────────────────────────────────────────────────────────────


# Empirical conversion to nA/T calibrated on benzene SZ-Hueckel pipeline.
# Common-gauge benzene at outer half-plane integration with hw=4 Å gives
# J_b ~ -5.20e-3 (working units). Reference GIMIC value: |J_ring| ~ 12 nA/T.
CURRENT_TO_NA_PER_T = 2308.0


def bond_current_strength(resp: _ResponseData,
                            atom_a: int,
                            atom_b: int,
                            ring_center: Optional[np.ndarray] = None,
                            half_width_in_plane: float = 4.0,
                            half_width_out_plane: float = 4.0,
                            n_in: int = 31,
                            n_out: int = 31,
                            beta: int = 2,
                            gauge: str = "common",
                            convert_to_nA_per_T: bool = True,
                            ) -> float:
    """Half-plane flux of j across the (atom_a, atom_b) bond cross-section."""
    basis = resp.basis
    R_a = basis.coords[atom_a]
    R_b = basis.coords[atom_b]
    bond_vec = R_b - R_a
    L = float(np.linalg.norm(bond_vec))
    if L < 1.0e-8:
        raise ValueError("atom_a and atom_b must be distinct")
    n_hat = bond_vec / L
    midpoint = 0.5 * (R_a + R_b)

    if ring_center is None:
        ring_center = np.zeros(3)
    ring_center = np.asarray(ring_center, dtype=float)

    # u_hat lies in the molecular plane (xy by convention), perpendicular
    # to bond, pointing AWAY from ring_center (outer half-plane).
    radial = midpoint - ring_center
    radial[2] = 0.0
    radial_norm = float(np.linalg.norm(radial))
    if radial_norm < 1.0e-8:
        z_hat = np.array([0.0, 0.0, 1.0])
        ref = (np.array([1.0, 0.0, 0.0])
               if abs(np.dot(n_hat, z_hat)) > 0.99 else z_hat)
        u_hat = ref - np.dot(ref, n_hat) * n_hat
    else:
        radial_hat = radial / radial_norm
        u_hat = radial_hat - np.dot(radial_hat, n_hat) * n_hat
    u_norm = float(np.linalg.norm(u_hat))
    if u_norm < 1.0e-8:
        z_hat = np.array([0.0, 0.0, 1.0])
        u_hat = z_hat - np.dot(z_hat, n_hat) * n_hat
        u_norm = float(np.linalg.norm(u_hat))
    u_hat /= u_norm
    w_hat = np.array([0.0, 0.0, 1.0])
    w_hat = (w_hat - np.dot(w_hat, n_hat) * n_hat
             - np.dot(w_hat, u_hat) * u_hat)
    w_hat /= float(np.linalg.norm(w_hat))

    # Outer half-plane: u in [0, +L_in], w in [-L_out, +L_out]
    u_vals = np.linspace(0.0, half_width_in_plane, n_in)
    w_vals = np.linspace(-half_width_out_plane, half_width_out_plane, n_out)
    du = u_vals[1] - u_vals[0]
    dw = w_vals[1] - w_vals[0]
    UU, WW = np.meshgrid(u_vals, w_vals, indexing="ij")
    pts = (
        midpoint[None, None, :]
        + UU[..., None] * u_hat[None, None, :]
        + WW[..., None] * w_hat[None, None, :]
    ).reshape(-1, 3)

    j_total, _, _, _ = current_density_at_points(resp, pts, beta=beta,
                                                    gauge=gauge)
    j_n = j_total @ n_hat
    J = float(j_n.sum() * du * dw)
    if convert_to_nA_per_T:
        J *= CURRENT_TO_NA_PER_T
    return J


def ring_current_strength(resp: _ResponseData,
                            ring_atoms: list,
                            **kwargs,
                            ) -> Tuple[float, list]:
    """Average bond-flux strength around a ring of atom indices."""
    n = len(ring_atoms)
    if "ring_center" not in kwargs:
        rc = np.mean([resp.basis.coords[i] for i in ring_atoms], axis=0)
        kwargs["ring_center"] = rc
    js = []
    for k in range(n):
        a = ring_atoms[k]
        b = ring_atoms[(k + 1) % n]
        js.append(bond_current_strength(resp, a, b, **kwargs))
    j_avg = float(np.mean(js))
    return j_avg, js


# ─────────────────────────────────────────────────────────────────────
# NICS via Biot-Savart
# ─────────────────────────────────────────────────────────────────────


def nics_zz_from_current(resp: _ResponseData,
                            probe: np.ndarray,
                            grid_pts: np.ndarray,
                            grid_weights: np.ndarray,
                            beta: int = 2,
                            gauge: str = "common",
                            ) -> float:
    """NICS_zz at probe via Biot-Savart over the induced current (ppm).

    sigma_zz(P) = -(alpha^2 / 2) * integral
                    [(j(r) x (r - P))_z / |r-P|^3] dV
    """
    j_total, _, _, _ = current_density_at_points(resp, grid_pts, beta=beta,
                                                    gauge=gauge)
    rel = grid_pts - probe[None, :]
    r3 = np.linalg.norm(rel, axis=-1) ** 3
    eps = 1.0e-12
    inv_r3 = np.where(r3 > eps, 1.0 / r3, 0.0)
    cross_z = j_total[:, 0] * rel[:, 1] - j_total[:, 1] * rel[:, 0]
    integrand = cross_z * inv_r3
    sigma_zz = -(ALPHA_FS ** 2 / 2.0) * float(
        np.sum(integrand * grid_weights)
    )
    return sigma_zz * 1.0e6
