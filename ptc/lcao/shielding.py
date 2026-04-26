"""Phase D: full GIAO chemical shielding tensor analysis.

Builds on the diamagnetic tensor and paramagnetic CP-PT shielding
already implemented in `giao.py`, completing the spec milestones:

  Phase D delivers:
    * sigma_alpha_beta(P) full 3x3 tensor at any 3D probe point
    * principal-axis decomposition (eigenvalues + eigenvectors)
    * isotropic shielding sigma_iso
    * sigma_zz (lab-frame z-component)
    * span Omega = sigma_max - sigma_min
    * skew kappa = 3 (sigma_22 - sigma_iso) / Omega    in [-1, +1]
    * NICS_iso(P) = -sigma_iso(P)
    * NICS_zz(P)  = -sigma_zz(P)

  Convention: principal eigenvalues sorted sigma_11 <= sigma_22 <= sigma_33.

The chemical-shielding tensor is the symmetric part of sigma_alpha_beta;
the antisymmetric part is unphysical and discarded for principal-axis
analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ptc.constants import A_BOHR, ALPHA_PHYS
from ptc.lcao.density_matrix import (
    PTMolecularBasis,
    build_molecular_basis,
    density_matrix_PT,
)
from ptc.lcao.giao import (
    angular_momentum_matrices_GIAO,
    magnetic_dipole_matrices,
    shielding_diamagnetic_tensor,
)
from ptc.topology import Topology


# Conversion: PT energies are in eV; shielding formula needs Hartree.
_HARTREE_eV = 27.211386245988


@dataclass
class GIAOShieldingTensor:
    """Full chemical-shielding tensor analysis at a given probe point.

    Attributes
    ----------
    sigma       : (3, 3) total tensor in ppm
    sigma_d     : (3, 3) diamagnetic part in ppm
    sigma_p     : (3, 3) paramagnetic part (CP-PT) in ppm
    sigma_iso   : isotropic shielding (Tr sigma / 3) in ppm
    sigma_zz    : lab-frame z-component sigma[2, 2] in ppm
    eigenvals   : principal values sigma_11 <= sigma_22 <= sigma_33 (ppm)
    eigenvecs   : principal axes as columns of a 3x3 orthogonal matrix
    span        : Omega = sigma_33 - sigma_11 in ppm (>= 0)
    skew        : kappa = 3 (sigma_22 - sigma_iso) / Omega in [-1, +1]
                  (kappa = 0 if Omega = 0, by convention)
    """
    sigma: np.ndarray
    sigma_d: np.ndarray
    sigma_p: np.ndarray
    sigma_iso: float
    sigma_zz: float
    eigenvals: np.ndarray
    eigenvecs: np.ndarray
    span: float
    skew: float

    @property
    def sigma_11(self) -> float:
        return float(self.eigenvals[0])

    @property
    def sigma_22(self) -> float:
        return float(self.eigenvals[1])

    @property
    def sigma_33(self) -> float:
        return float(self.eigenvals[2])

    @property
    def nics_iso(self) -> float:
        """NICS isotropic = -sigma_iso (chemical-shift convention)."""
        return -self.sigma_iso

    @property
    def nics_zz(self) -> float:
        """NICS_zz = -sigma_zz."""
        return -self.sigma_zz


# ─────────────────────────────────────────────────────────────────────
# Paramagnetic 3x3 tensor (extends paramagnetic_shielding_iso)
# ─────────────────────────────────────────────────────────────────────


def paramagnetic_shielding_tensor(basis: PTMolecularBasis,
                                    mo_eigvals: np.ndarray,
                                    mo_coeffs: np.ndarray,
                                    n_e_total: int,
                                    K: np.ndarray,
                                    use_giao: bool = True,
                                    **quad_kwargs) -> np.ndarray:
    """Full 3 x 3 paramagnetic shielding tensor sigma^p_alpha_beta(K) (ppm).

    Uncoupled CP-PT formula:
        sigma^p_{alpha,beta}(K) = (4 alpha^2) Sum_{a in virt, i in occ}
            L_imag^GIAO[beta, a, i] * M_imag[alpha, i, a]
            / (eps_a - eps_i)

    With use_giao=True (default), L is the GIAO London-phase variant
    (gauge-invariant). The K argument fixes the Ramsey magnetic-dipole
    operator at the probe point; it is gauge-fixed by construction.
    """
    K = np.asarray(K, dtype=float)
    n_occ = n_e_total // 2
    n_virt = mo_coeffs.shape[1] - n_occ
    if n_occ == 0 or n_virt == 0:
        return np.zeros((3, 3))

    if use_giao:
        from ptc.lcao.giao import angular_momentum_matrices_GIAO
        L_imag = angular_momentum_matrices_GIAO(basis, **quad_kwargs)
    else:
        from ptc.lcao.giao import angular_momentum_matrices
        L_imag = angular_momentum_matrices(basis, K, **quad_kwargs)

    M_imag = magnetic_dipole_matrices(basis, K, **quad_kwargs)

    L_mo = np.array([mo_coeffs.T @ L_imag[k] @ mo_coeffs for k in range(3)])
    M_mo = np.array([mo_coeffs.T @ M_imag[k] @ mo_coeffs for k in range(3)])

    eps_a = mo_eigvals[n_occ:]
    eps_i = mo_eigvals[:n_occ]
    diff = eps_a[None, :] - eps_i[:, None]   # (n_occ, n_virt)

    # zeta in 1/Angstrom -> atomic units: x A_BOHR for L, x A_BOHR^3 for M
    L_au = L_mo * A_BOHR
    M_au = M_mo * (A_BOHR ** 3)
    diff_au = diff / _HARTREE_eV

    sigma_p = np.zeros((3, 3))
    for alpha in range(3):
        for beta in range(3):
            L_b_ai = L_au[beta, n_occ:, :n_occ]   # (n_virt, n_occ)
            M_a_ia = M_au[alpha, :n_occ, n_occ:]  # (n_occ, n_virt)
            prod = L_b_ai.T * M_a_ia               # element-wise (i, a)
            sigma_p[alpha, beta] = 4.0 * (ALPHA_PHYS ** 2) * (prod / diff_au).sum()

    return sigma_p * 1.0e6   # ppm


# ─────────────────────────────────────────────────────────────────────
# Tensor analysis: principal axes, span, skew
# ─────────────────────────────────────────────────────────────────────


def _tensor_analysis(sigma: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray, float, float]:
    """Symmetric part eigenanalysis. Returns
        (sigma_iso, sigma_zz, eigenvals_sorted, eigenvecs, span, skew).

    The chemical-shielding tensor is conventionally the symmetric part
    of sigma; antisymmetric part is unphysical to first order.
    """
    sigma_sym = 0.5 * (sigma + sigma.T)
    sigma_iso = float(np.trace(sigma_sym) / 3.0)
    sigma_zz = float(sigma[2, 2])

    eigvals, eigvecs = np.linalg.eigh(sigma_sym)   # ascending
    span = float(eigvals[2] - eigvals[0])
    if abs(span) < 1.0e-12:
        skew = 0.0
    else:
        skew = float(3.0 * (eigvals[1] - sigma_iso) / span)
        # Numerical clamp to [-1, +1]
        skew = max(-1.0, min(1.0, skew))

    return sigma_iso, sigma_zz, eigvals, eigvecs, span, skew


# ─────────────────────────────────────────────────────────────────────
# High-level pipelines
# ─────────────────────────────────────────────────────────────────────


def shielding_tensor_at_point(topology: Topology,
                                P: np.ndarray,
                                use_giao: bool = True,
                                **quad_kwargs) -> GIAOShieldingTensor:
    """End-to-end GIAO shielding tensor at probe P.

    Returns
    -------
    GIAOShieldingTensor with full sigma_d, sigma_p, sigma, eigenvalues
    and eigenvectors of the symmetric part, span and skew.
    """
    P = np.asarray(P, dtype=float)
    basis = build_molecular_basis(topology)
    rho, S, eigvals, c = density_matrix_PT(topology, basis=basis)
    n_e = int(round(basis.total_occ))

    sigma_d = shielding_diamagnetic_tensor(rho, basis, P, **quad_kwargs)
    sigma_p = paramagnetic_shielding_tensor(
        basis, eigvals, c, n_e, P,
        use_giao=use_giao, **quad_kwargs,
    )
    sigma = sigma_d + sigma_p

    sigma_iso, sigma_zz, ev, evec, span, skew = _tensor_analysis(sigma)

    return GIAOShieldingTensor(
        sigma=sigma,
        sigma_d=sigma_d,
        sigma_p=sigma_p,
        sigma_iso=sigma_iso,
        sigma_zz=sigma_zz,
        eigenvals=ev,
        eigenvecs=evec,
        span=span,
        skew=skew,
    )


def nics_iso_giao(topology: Topology,
                   P: np.ndarray,
                   **quad_kwargs) -> float:
    """Convenience: NICS_iso(P) = -sigma_iso(P) in ppm."""
    return float(shielding_tensor_at_point(topology, P, **quad_kwargs).nics_iso)


def nics_zz_giao(topology: Topology,
                  P: np.ndarray,
                  **quad_kwargs) -> float:
    """Convenience: NICS_zz(P) = -sigma_zz(P) in ppm."""
    return float(shielding_tensor_at_point(topology, P, **quad_kwargs).nics_zz)
