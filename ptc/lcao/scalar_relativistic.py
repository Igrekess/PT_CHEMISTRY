"""Phase 6.B.6 — Cowan-Griffin scalar relativistic corrections to H_core.

Two operator additions to the one-electron Hamiltonian beyond the simple
γ_rel rescaling of Z_eff (Phase 6.B.5b) :

* Mass-velocity ::  V_MV = -(α²/8) p⁴ ≈ -(α²/2) T² (a.u.)
  Per matrix element in non-orthogonal AO basis :
      ⟨χ_μ|V_MV|χ_ν⟩ ≈ -(α²/(2·HARTREE_eV)) · (T·S⁻¹·T)_μν   [eV]

* Darwin   ::      V_DW = (π α²/2) Σ_K Z_K δ³(r - R_K) (a.u.)
  Per matrix element :
      ⟨χ_μ|V_DW|χ_ν⟩ = (π α²/2)·HARTREE_eV·a₀³ · Σ_K Z_K χ_μ(R_K)·χ_ν(R_K)
  (the a₀³ factor converts Å⁻³ orbital values to atomic units of charge
   density before applying the prefactor).

Both terms are PT-natural (α derives from the q_stat product at μ*=15 ;
zero parameters). The Cowan-Griffin Hamiltonian (Cowan-Griffin 1976) is
the reference scalar-relativistic operator behind, e.g., the cc-pVDZ-DK
Dunning bases and ZORA / DKH transformations. Effects scale ~Z⁴ for
inner shells, becoming dominant for chemical-shielding analysis on
elements with Z > 17 (Cl-35 σ shifts of ~10 ppm, Br-79 of ~80 ppm).
"""
from __future__ import annotations

import numpy as np

from ptc.constants import A_BOHR, ALPHA_PHYS
from ptc.lcao.density_matrix import PTMolecularBasis
from ptc.lcao.giao import _orbital_values_on_grid

# PT-derived alpha = 1/(1/AEM + ...). Same constant used everywhere.
_ALPHA_PHYS = ALPHA_PHYS
_HARTREE_eV = 27.211386245988


def mass_velocity_matrix(T_matrix: np.ndarray,
                            S_matrix: np.ndarray) -> np.ndarray:
    """Mass-velocity correction matrix in eV.

    V_MV ≈ -(α²/2) T² in atomic units. In a non-orthogonal AO basis the
    matrix realisation of T² is ``T·S⁻¹·T`` ; the eV-to-Hartree factor
    appears once in the prefactor (T already in eV).

    Parameters
    ----------
    T_matrix : (n_orb, n_orb) kinetic matrix in eV.
    S_matrix : (n_orb, n_orb) overlap matrix.

    Returns
    -------
    V_MV : (n_orb, n_orb) symmetric matrix in eV.
    """
    S_inv = np.linalg.inv(S_matrix)
    T2 = T_matrix @ S_inv @ T_matrix
    prefactor = -0.5 * (_ALPHA_PHYS ** 2) / _HARTREE_eV
    V_MV = prefactor * T2
    return 0.5 * (V_MV + V_MV.T)


def darwin_matrix(basis: PTMolecularBasis,
                    nuclear_charge_method: str = "actual") -> np.ndarray:
    """Darwin term matrix in eV.

    V_DW = (π α²/2) Σ_K Z_K δ³(r - R_K)  (a.u.)
    ⟨χ_μ|V_DW|χ_ν⟩ = (π α²/2) HARTREE_eV a₀³ Σ_K Z_K χ_μ(R_K) χ_ν(R_K)

    The a₀³ factor converts our STO normalisation (Å⁻³⁄₂ amplitude →
    Å⁻³ density at probe point) to the atomic-unit charge density that
    the operator prefactor expects.

    Parameters
    ----------
    basis : built molecular basis.
    nuclear_charge_method : 'actual' uses Z directly ; 'pt' uses
        ``effective_charge`` from atomic_basis (consistent with the
        Hueckel core-Hamiltonian convention if needed).

    Returns
    -------
    V_DW : (n_orb, n_orb) symmetric matrix in eV.
    """
    n_orb = basis.n_orbitals
    coords = np.asarray(basis.coords, dtype=float)
    Z_list = list(basis.Z_list)

    # Evaluate every AO at every nucleus position
    psi_at_nuclei = _orbital_values_on_grid(basis, coords)  # (n_orb, n_atoms)

    if nuclear_charge_method == "actual":
        Z_eff = [float(Z) for Z in Z_list]
    elif nuclear_charge_method == "pt":
        from ptc.atom import effective_charge
        Z_eff = [float(effective_charge(Z)) for Z in Z_list]
    else:
        raise ValueError(f"unknown nuclear_charge_method {nuclear_charge_method!r}")

    prefactor = 0.5 * np.pi * (_ALPHA_PHYS ** 2) * _HARTREE_eV * (A_BOHR ** 3)

    V_DW = np.zeros((n_orb, n_orb), dtype=float)
    for K, ZK in enumerate(Z_eff):
        psi_K = psi_at_nuclei[:, K]   # (n_orb,)
        V_DW += prefactor * ZK * np.outer(psi_K, psi_K)
    return 0.5 * (V_DW + V_DW.T)


def scalar_relativistic_h_core(T_matrix: np.ndarray,
                                  V_nuc_matrix: np.ndarray,
                                  S_matrix: np.ndarray,
                                  basis: PTMolecularBasis,
                                  include_mass_velocity: bool = True,
                                  include_darwin: bool = True,
                                  nuclear_charge_method: str = "actual",
                                  ) -> np.ndarray:
    """Cowan-Griffin H_core = T + V_nuc + V_MV + V_DW (eV).

    Returns the standard non-relativistic core Hamiltonian augmented
    with the two scalar-relativistic Cowan-Griffin terms (mass-velocity
    and Darwin). Returns the original H_core if both flags are False.
    """
    H = T_matrix + V_nuc_matrix
    if include_mass_velocity:
        H = H + mass_velocity_matrix(T_matrix, S_matrix)
    if include_darwin:
        H = H + darwin_matrix(basis, nuclear_charge_method=nuclear_charge_method)
    return 0.5 * (H + H.T)
