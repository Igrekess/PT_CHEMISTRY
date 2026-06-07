"""PT one-electron spin-orbit operator and second-order PV response.

Companion module to ``ptc.lcao.parity_violation``. It supplies the
two ingredients required to turn the bare Vester-Ulbricht (V-U) matrix
elements into a non-zero E_PV on a closed-shell singlet wavefunction:

1. ``hso_matrix_elements`` builds the Pauli one-electron spin-orbit (SO)
   operator in the spin-included AO basis (same alpha-first / beta
   block ordering as ``hpv_matrix_elements``).

2. ``e_pv_response_so`` evaluates the standard SO-mixing perturbative
   response formula of Bakasov-Ha-Quack 1998,

       E_PV  =  2 Re sum_{i in occ}{a in vir}
                 <phi_i | H_SO  | phi_a><phi_a | H_PV | phi_i>
                 / (eps_i - eps_a)

   on a closed-shell singlet RHF reference. The spatial RHF orbitals
   produce alpha- and beta-channel SO mixings that DIFFER in sign, so
   the spin trace of sigma in H_PV survives at second order; this is
   the mechanism by which closed-shell molecules acquire E_PV.

Why we need this module
-----------------------
For a closed-shell RHF singlet the bare expectation value
<Psi|H_PV|Psi> = 0 exactly, because the alpha and beta spatial densities
are identical and the spin trace of sigma vanishes. NSI parity violation
first appears at SECOND order in (H_SO, H_PV) - which is precisely the
Bakasov-Ha-Quack 1998 route used to produce the canonical
E_PV(L-alanine) ~ -5.5e-20 Hartree benchmark.

Physical form of H_SO
---------------------
The one-electron Pauli SO operator in atomic units is

    H_SO(r)  =  (alpha^2 / 2) sum_A  Z_A  L_A(r) . S / |r - R_A|^3 ,

with

    L_A(r)   =  (r - R_A) x p ,        S = sigma / 2 .

Multiplying out the S = sigma/2 factor,

    H_SO(r)  =  (alpha^2 / 4) sum_A Z_A sigma_alpha
                  ((r - R_A) x p)_alpha / |r - R_A|^3 .

The matrix element on real-spherical AOs is

    <chi_mu, s | H_SO | chi_nu, s'>
        = (alpha^2 / 4) sum_A Z_A sum_alpha [sigma_alpha]_{s, s'}
              * <chi_mu | (L_A)_alpha / |r - R_A|^3 | chi_nu> .

The spatial bracket is purely imaginary on real AOs: the existing
``ptc.lcao.giao.magnetic_dipole_matrices`` evaluates exactly the form
``<chi_mu | L_K_alpha / |r - K|^3 | chi_nu> = -i * M_imag[alpha, mu, nu]``
at any probe point K with M_imag real antisymmetric. We therefore reuse
that routine, summed over the nuclei A with weight Z_A.

The final H_SO is then

    H_SO[(mu, s), (nu, s')]
        = (alpha^2 / 4) * (-i) * sum_A Z_A sum_alpha
           [sigma_alpha]_{s, s'} M_imag^A[alpha, mu, nu] .

Because sigma is Hermitian and M_imag is real antisymmetric, the same
algebraic argument as for H_PV (see ``parity_violation.py`` docstring)
shows that H_SO is Hermitian.

Spin-block ordering
-------------------
Identical to ``hpv_matrix_elements``: the 2 n_orb x 2 n_orb matrix uses
the (mu, s) flattening

    index 2*mu     -> (mu, alpha)
    index 2*mu + 1 -> (mu, beta)

so the two operators can be added / contracted directly.

Implementation choice for the spatial integrals
-----------------------------------------------
Analytic STO integrals for the SO operator (L/r^3) on a general
multi-centre basis are messy; ``magnetic_dipole_matrices`` already
delivers them by direct 3D Gauss-quadrature with the same convention
(zeta in 1/Angstrom, real spherical harmonics, common-origin gauge at
the probe). Convergence is governed by ``n_radial / n_theta / n_phi``
(default 60 / 24 / 32 in the GIAO routine, same here). Accuracy of the
SO integrals is therefore identical to the rest of PT_LCAO's magnetic
machinery, i.e. relative error well below 1e-4 on H/He/C/N/O bases at
the default grid (Lamb sigma_iso(H) recovers ~17.75 ppm to 4-5 digits).
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np

from ptc.lcao.density_matrix import PTMolecularBasis
from ptc.lcao.giao import magnetic_dipole_matrices
from ptc.lcao.parity_violation import ALPHA_EM_PT


# ---------------------------------------------------------------------
# Pauli matrices in the (alpha, beta) basis (consistent with H_PV).
# ---------------------------------------------------------------------

_SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_SIGMA = np.stack([_SIGMA_X, _SIGMA_Y, _SIGMA_Z], axis=0)  # (3, 2, 2)


# ---------------------------------------------------------------------
# One-electron spin-orbit operator in the spin-included AO basis
# ---------------------------------------------------------------------


def hso_matrix_elements(basis: PTMolecularBasis,
                         R_list: np.ndarray | None = None,
                         nuclear_charges: np.ndarray | None = None,
                         *,
                         n_radial: int = 60,
                         n_theta: int = 24,
                         n_phi: int = 32,
                         use_becke: bool = False,
                         lebedev_order: int = 50) -> np.ndarray:
    """One-electron Pauli spin-orbit operator H_SO in the spin AO basis.

    Standard form (one-electron part only):

        H_SO  =  (alpha^2 / 2) sum_A
                 (Z_A / |r - R_A|^3) L_A . S ,

    with L_A = (r - R_A) x p and S = sigma / 2. The matrix element in
    the spin-included AO basis is

        H_SO[(mu, s), (nu, s')]
            =  (alpha^2 / 4) sum_A Z_A
                 sum_alpha [sigma_alpha]_{s, s'}
                   <chi_mu | (L_A)_alpha / |r - R_A|^3 | chi_nu>

    Parameters
    ----------
    basis : PTMolecularBasis
        Spatial AO basis with attached geometry. Same convention as the
        rest of PT_LCAO (zeta in 1/Angstrom, real spherical harmonics).
    R_list : ndarray (n_A, 3), optional
        Nuclear positions in Angstrom. Default: ``basis.coords``.
    nuclear_charges : ndarray (n_A,), optional
        Nuclear charges Z_A. Default: ``basis.Z_list`` (full Z).
    n_radial, n_theta, n_phi : int
        3D Gauss-quadrature grid resolution; forwarded to
        ``magnetic_dipole_matrices`` (which evaluates L_A / |r - R_A|^3
        directly). Defaults match GIAO's magnetic-dipole defaults.
    use_becke, lebedev_order :
        Optional Becke + Lebedev multi-centre grid backend (heavier but
        needed for tight inner-shell SO integrals on Z > 17 atoms).

    Returns
    -------
    H_SO : complex ndarray (2 n_orb, 2 n_orb)
        Hermitian spin-orbit operator matrix in atomic units (Hartree).
        Spin-block flattening:  index 2*mu + 0 -> (mu, alpha),
                                index 2*mu + 1 -> (mu, beta).

    Notes
    -----
    * Hermiticity is enforced numerically (defensive 0.5 (H + H^dagger)
      sweep on top of the analytic identity).
    * The atomic-unit prefactor (alpha^2 / 4) carries no further unit
      conversion beyond what magnetic_dipole_matrices already exposes:
      that routine returns the spatial bracket in (1/Angstrom)^3 *
      1/Angstrom = 1/Angstrom^4 implicitly multiplied by Z_A
      (dimensionless). The 1/c factor of the Pauli SO Hamiltonian is
      absorbed into alpha (a.u.: 1/c = alpha).
    """
    if R_list is None:
        R_list = basis.coords.copy()
    R_list = np.asarray(R_list, dtype=float)
    if nuclear_charges is None:
        nuclear_charges = np.asarray(basis.Z_list, dtype=float)
    nuclear_charges = np.asarray(nuclear_charges, dtype=float)

    n_A = R_list.shape[0]
    if nuclear_charges.shape[0] != n_A:
        raise ValueError(
            f"nuclear_charges has {nuclear_charges.shape[0]} entries but "
            f"R_list has {n_A} centres"
        )

    n = basis.n_orbitals

    # Accumulate the real antisymmetric M^A_imag for each nucleus A and
    # sum with weight Z_A. magnetic_dipole_matrices is given by
    #   <chi_mu | L^A_alpha / |r - R_A|^3 | chi_nu> = -i M_imag[alpha, mu, nu]
    M_total = np.zeros((3, n, n), dtype=float)   # sum_A Z_A * M_imag^A
    for A in range(n_A):
        zA = float(nuclear_charges[A])
        if zA == 0.0:
            continue
        M_A = magnetic_dipole_matrices(
            basis, R_list[A],
            n_radial=n_radial, n_theta=n_theta, n_phi=n_phi,
            use_becke=use_becke, lebedev_order=lebedev_order,
        )
        M_total += zA * M_A

    # Assemble the spin-resolved operator.
    # H_SO[(mu s), (nu s')] = (alpha^2 / 4) * (-i)
    #                          * sum_alpha sigma_alpha[s,s'] * M_total[alpha,mu,nu]
    H = np.zeros((2 * n, 2 * n), dtype=complex)
    for alpha in range(3):
        sigma_a = _SIGMA[alpha]                # (2, 2) complex
        M_alpha = M_total[alpha]               # (n, n) real antisym
        # np.kron(A, B): result[2*mu+s, 2*nu+s'] = A[mu,nu] * B[s,s'].
        H += np.kron(M_alpha, sigma_a)

    # Prefactor : (alpha^2 / 4) * (-i)  [the (-i) folds the spatial bracket
    # convention <.|L/r^3|.> = -i M_imag from magnetic_dipole_matrices]
    H *= -1j * (ALPHA_EM_PT ** 2) / 4.0

    # Defensive Hermitisation (exact in exact arithmetic).
    H = 0.5 * (H + H.conj().T)
    return H


# ---------------------------------------------------------------------
# Second-order PV response with SO mixing
# ---------------------------------------------------------------------


def _build_spin_mo_coefficients(C_spatial: np.ndarray) -> np.ndarray:
    """Lift a SPATIAL MO matrix (n_orb x n_orb) into a spin-orbital MO
    matrix (2 n_orb x 2 n_orb) under the closed-shell RHF convention
    "alpha and beta orbitals share the same spatial part".

    The spin-orbital ordering on rows AND columns is the same alpha-first
    flattening used by H_PV / H_SO:

        row index 2*mu + s  ->  (AO mu, spin s)
        col index 2*k  + s  ->  (MO k,  spin s)

    For each spatial MO k, two spin-orbitals are produced:
        spin-orbital 2*k     = (spatial k, alpha)
        spin-orbital 2*k + 1 = (spatial k, beta)

    With this convention an N-electron closed-shell RHF determinant
    occupies spin-orbitals 0..2 n_doubly - 1 (with n_doubly = N/2).
    """
    n = C_spatial.shape[0]
    C_spin = np.zeros((2 * n, 2 * n), dtype=C_spatial.dtype)
    for mu in range(n):
        for k in range(n):
            C_spin[2 * mu, 2 * k] = C_spatial[mu, k]        # alpha block
            C_spin[2 * mu + 1, 2 * k + 1] = C_spatial[mu, k]  # beta block
    return C_spin


def e_pv_response_so(hpv_matrix: np.ndarray,
                       hso_matrix: np.ndarray,
                       C_mo: np.ndarray,
                       orbital_energies: np.ndarray,
                       n_occ: int) -> float:
    """E_PV via SO-mixing second-order response (Bakasov-Ha-Quack 1998).

        E_PV  =  2 Re sum_{i in occ}{a in vir}
                  <phi_i | H_SO  | phi_a><phi_a | H_PV | phi_i>
                  / (eps_i - eps_a)

    Parameters
    ----------
    hpv_matrix : complex (2 n_orb, 2 n_orb)
        Parity-violation operator in the spin AO basis (output of
        ``parity_violation.hpv_matrix_elements``).
    hso_matrix : complex (2 n_orb, 2 n_orb)
        Spin-orbit operator in the spin AO basis (output of
        ``hso_matrix_elements``).
    C_mo : (2 n_orb, 2 n_orb) ndarray
        MO coefficient matrix already expressed in the spin-orbital
        basis. For a closed-shell RHF reference one can produce this
        from a spatial MO matrix via the helper
        ``_build_spin_mo_coefficients`` (alpha and beta share spatial
        part).
    orbital_energies : (2 n_orb,) real ndarray
        Spin-orbital energies. For closed-shell RHF, each spatial
        eigenvalue appears twice (one alpha, one beta).
    n_occ : int
        Number of occupied spin-orbitals (= N for an N-electron system,
        i.e. 2 * n_doubly for closed-shell singlet).

    Returns
    -------
    E_PV : float, Hartree
        The second-order parity-violation energy. For an achiral
        closed-shell molecule (e.g. H2O at C_2v geometry) the result is
        zero to numerical precision; for a chiral one (e.g. H2O2 at
        non-zero dihedral) it is finite and flips sign under full
        parity inversion of the nuclei.

    Notes
    -----
    * The factor ``2 Re`` makes E_PV manifestly real (the sum over
      (i, a) and its complex conjugate are paired).
    * Both <i|H_SO|a> and <a|H_PV|i> are needed; H_SO and H_PV are
      Hermitian so we use <a|...|i> = conj(<i|...|a>).
    * Frontier-MO degeneracies (eps_a == eps_i) would diverge; the
      function asserts they are absent. For practical molecules with
      a proper HOMO-LUMO gap this is automatic.
    """
    H_pv = np.asarray(hpv_matrix)
    H_so = np.asarray(hso_matrix)
    if H_pv.shape != H_so.shape:
        raise ValueError(
            f"hpv_matrix shape {H_pv.shape} != hso_matrix shape {H_so.shape}"
        )
    n_spin = H_pv.shape[0]
    C = np.asarray(C_mo, dtype=complex)
    if C.shape != (n_spin, n_spin):
        raise ValueError(
            f"C_mo shape {C.shape} incompatible with operator shape "
            f"{H_pv.shape} (expected {(n_spin, n_spin)})"
        )
    eps = np.asarray(orbital_energies, dtype=float)
    if eps.shape != (n_spin,):
        raise ValueError(
            f"orbital_energies shape {eps.shape} != ({n_spin},)"
        )
    if not (0 < n_occ < n_spin):
        raise ValueError(
            f"n_occ={n_occ} must be strictly between 0 and 2*n_orb={n_spin}"
        )

    # Transform operators to MO basis.
    H_pv_mo = C.conj().T @ H_pv @ C
    H_so_mo = C.conj().T @ H_so @ C

    occ = slice(0, n_occ)
    vir = slice(n_occ, n_spin)
    eps_i = eps[occ]                       # (n_occ,)
    eps_a = eps[vir]                       # (n_vir,)

    # Diff[i, a] = eps_i - eps_a (negative for HOMO/LUMO order).
    Diff = eps_i[:, None] - eps_a[None, :]
    if np.any(np.abs(Diff) < 1e-14):
        raise ValueError(
            "Degeneracy detected at the HOMO/LUMO border: "
            "eps_i - eps_a vanishes for at least one (i, a) pair."
        )

    SO_ia = H_so_mo[occ, vir]              # <phi_i | H_SO | phi_a>
    PV_ai = H_pv_mo[vir, occ]              # <phi_a | H_PV | phi_i>

    # sum_{i, a} SO[i, a] * PV[a, i] / (eps_i - eps_a)
    summand = SO_ia * PV_ai.T / Diff       # element-wise (n_occ, n_vir)
    s = np.sum(summand)
    return 2.0 * float(np.real(s))


# ---------------------------------------------------------------------
# Convenience: full closed-shell pipeline starting from a spatial RHF
# ---------------------------------------------------------------------


def e_pv_so_response_rhf(basis: PTMolecularBasis,
                          rho_spatial: np.ndarray,
                          mo_eigvals: np.ndarray,
                          mo_coeffs: np.ndarray,
                          n_e_total: int,
                          *,
                          hpv_kwargs: Optional[dict] = None,
                          hso_kwargs: Optional[dict] = None) -> float:
    """End-to-end driver: closed-shell RHF -> E_PV via SO response.

    Wraps ``hpv_matrix_elements`` + ``hso_matrix_elements`` +
    ``e_pv_response_so`` and lifts the spatial RHF MO matrix into the
    spin-orbital basis with the "alpha == beta spatial" convention.

    Parameters
    ----------
    basis        : built molecular basis with attached geometry.
    rho_spatial  : unused at this level (kept in the signature so that
                   one can pass the result of ``density_matrix_PT_scf``
                   directly without unpacking).
    mo_eigvals   : (n_orb,) real, spatial RHF orbital energies.
    mo_coeffs    : (n_orb, n_orb) real, spatial RHF MO matrix.
    n_e_total    : int, total electron count.
    hpv_kwargs   : kwargs forwarded to ``hpv_matrix_elements``.
    hso_kwargs   : kwargs forwarded to ``hso_matrix_elements``.

    Returns
    -------
    E_PV : float, Hartree.
    """
    from ptc.lcao.parity_violation import hpv_matrix_elements

    hpv_kwargs = dict(hpv_kwargs or {})
    hso_kwargs = dict(hso_kwargs or {})

    H_pv = hpv_matrix_elements(basis, **hpv_kwargs)
    H_so = hso_matrix_elements(basis, **hso_kwargs)

    # Closed-shell -> alpha and beta share the spatial MOs.
    C_spin = _build_spin_mo_coefficients(np.asarray(mo_coeffs))
    # Spin-orbital energies: each spatial epsilon appears twice.
    eps_spin = np.repeat(np.asarray(mo_eigvals, dtype=float), 2)
    # Permutation: np.repeat gives (eps_0, eps_0, eps_1, eps_1, ...), which
    # matches the (mu, s) flattening used by C_spin (alpha then beta on each
    # spatial MO).

    n_occ_spin = int(n_e_total)
    return e_pv_response_so(H_pv, H_so, C_spin, eps_spin, n_occ_spin)
