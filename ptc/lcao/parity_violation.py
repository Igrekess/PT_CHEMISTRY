"""PT parity-violation Hamiltonian H_PV (Vester-Ulbricht, NSI part).

PT-pure implementation of the nuclear-spin-independent (NSI) parity-violating
one-electron operator in the LCAO atomic-orbital basis.

The physics
-----------
For a chiral molecule, the leading P-odd weak interaction between an electron
and a point nucleus A is the Bouchiat-Bouchiat (1974) NSI operator

    H_PV  =  (G_F / (2 sqrt(2) m_e c))  sum_A  Q_W^A
             { sigma_e . p_e ,  delta^3(r_e - R_A) }_+

where {.,.}_+ is the anti-commutator and Q_W^A = Z_A (1 - 4 sin^2 theta_W) - N_A
is the weak nuclear charge. In atomic units (hbar = m_e = e = 1, c = 1/alpha_EM)
this becomes

    H_PV  =  (G_F alpha_EM / (2 sqrt(2)))  sum_A  Q_W^A
             { sigma_e . p_e ,  delta^3(r_e - R_A) }_+ .

The expectation value E_PV = <Psi|H_PV|Psi> is the parity-violation energy of
one enantiomer; |Delta E_PV| = 2 |E_PV| separates the two mirror-image
geometries. For chiral biomolecules this lies in the 10^-19 to 10^-20 Hartree
range (Bakasov-Ha-Quack 1998).

PT-derived inputs (zero adjustable parameters)
----------------------------------------------
All four electroweak inputs trace back to s = 1/2 through the PT cascade:

    sin^2 theta_W^PT  = gamma_7^2 / sum_p gamma_p^2 + NNLO  =  0.23119
                                            [DER], monograph ch. 11
    1 - 4 sin^2 theta_W^PT                                  =  0.07524
    alpha_EM^PT       = prod_{p in {3,5,7}} sin^2 theta_p (q_stat) ... = 1/137.036
                                            [DER], monograph ch. 11
    G_F^PT            = 1 / (sqrt(2) v_PT^2),  v_PT from R51
                                            [DER], monograph R51

See PT_PROJECTS/PT_HOMOCHIRALITY/notes/02_vester_ulbricht_pt.md for the full
derivation chain and discussion of magnitude / sign predictions.

Analytical reduction of the matrix elements
-------------------------------------------
For two real spatial AOs chi_mu and chi_nu, the action of delta^3(r - R_A)
collapses the matrix element to a point-evaluation. Using the identity

    {p_i, delta^3(r - R)}_+  =  p_i delta + delta p_i

and integrating by parts in the spatial integral (boundary terms vanish for
square-integrable orbitals), one obtains

    <chi_mu | {p_i, delta^3(r - R)}_+ | chi_nu>
       =  i [ (partial_i chi_mu)(R) chi_nu(R)
            - chi_mu(R) (partial_i chi_nu)(R) ]
       =  i  Im_{mu, nu, i}^A  .

This expression is ANTI-symmetric in (mu, nu) and explicitly Hermitian
(complex-conjugating swaps mu and nu, picks up an extra minus from the
factor of i, and the antisymmetry cancels it). The naive `-i d_i[chi_mu
chi_nu](R)` form sometimes quoted in the parity-violation literature is the
SYMMETRIC alternative; that form is anti-Hermitian and is not the matrix
element of the anti-commutator. The correct expression follows from a clean
integration by parts.

The full matrix element including electron spin is then a 2 x 2 block in the
(s, s') alpha/beta basis:

    M^A_{(mu s), (nu s')}  =  prefactor * Q_W^A * sum_i [sigma_i]_{s,s'}
                              * Im_{mu, nu, i}^A

with
    Im_{mu, nu, i}^A  =  i [ (d_i chi_mu)(R_A) chi_nu(R_A)
                          - chi_mu(R_A) (d_i chi_nu)(R_A) ]

and prefactor = G_F alpha_EM / (2 sqrt(2)) ~ 5.73e-17 Hartree * a0^3 in atomic
units. The matrix H_PV is Hermitian by construction.

Its expectation value vanishes identically for any wavefunction that is
closed-shell singlet without spin-orbit coupling — the spin trace of sigma
kills the contribution. This is physically correct: NSI parity violation in
closed-shell molecules first appears via spin-orbit mixing.

Sign convention
---------------
sigma_x, sigma_y, sigma_z are the standard Pauli matrices with sigma_z
diagonal in (alpha, beta). Spatial AOs chi_mu are real (real-spherical
harmonics), so the only source of i in M is the explicit -i factor above.
H_PV is Hermitian: H = H^dagger.

Conventions of the AO basis
---------------------------
This module reuses the existing PT-LCAO basis (atomic_basis.py +
density_matrix.py) and the closed-form STO gradient (giao.evaluate_sto_gradient).
zeta is carried in 1/Angstrom; in the analytical identity above we therefore
report the spatial-gradient piece g^A_i in 1/Angstrom * Angstrom^-3 = 1/A^4.
The H_PV prefactor is given in atomic units (Hartree, a0). Tests use SI-free
numerical relations only.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from ptc.constants import A_BOHR
from ptc.lcao.atomic_basis import (
    PTAtomicOrbital,
    PTContractedOrbital,
)
from ptc.lcao.density_matrix import PTMolecularBasis
from ptc.lcao.giao import evaluate_sto, evaluate_sto_gradient


# ---------------------------------------------------------------------
# PT-derived electroweak constants
# ---------------------------------------------------------------------

SIN2_THETA_W_PT: float = 0.23119
"""sin^2 theta_W^PT from monograph chapter 11 (gamma_7^2 / sum gamma_p^2 + NNLO).
0.010 % below the CODATA value 0.23121.  [DER]"""

ONE_MINUS_4SIN2: float = 1.0 - 4.0 * SIN2_THETA_W_PT
"""1 - 4 sin^2 theta_W^PT = 0.07524 (signs of Z and N in the weak charge)."""

# Fermi constant in atomic units. R51 gives G_F = 1/(sqrt(2) v^2) with v from
# the PT scalar field VEV; converting to atomic units (Hartree^-1 a0^-3 in
# natural form, but the literature usage in this context is Hartree^-2 once
# folded with c = 1/alpha) the standard tabulated value is 2.222e-14 Hartree^-2.
# The accompanying note 02_vester_ulbricht_pt.md gives the prefactor
# G_F alpha_EM / (2 sqrt 2) = 5.73e-17 Hartree^-1 . a0^3 directly.
G_F_AU: float = 2.222e-14
"""G_F^PT in atomic units (Hartree^-2). PT-derived from R51."""

ALPHA_EM_PT: float = 1.0 / 137.036
"""alpha_EM^PT, product of sin^2 theta_p (q_stat) at p = 3, 5, 7 with
echo-prime dressing.  [DER], monograph ch. 11."""

PREFACTOR_AU: float = G_F_AU * ALPHA_EM_PT / (2.0 * math.sqrt(2.0))
"""Global prefactor G_F alpha_EM / (2 sqrt 2) in atomic units.

Numerical value ~ 5.73e-17 (Hartree^-1 a0^3 in the operator's natural units;
multiplying by Q_W (dimensionless) and the gradient piece in a0^-4 gives
the matrix element in Hartree)."""


# ---------------------------------------------------------------------
# Weak nuclear charge Q_W^A
# ---------------------------------------------------------------------


def weak_charge(Z: int, N: int) -> float:
    """PT-derived weak nuclear charge of a nucleus (Z, N).

    Q_W^A = Z * (1 - 4 sin^2 theta_W^PT) - N

    with sin^2 theta_W^PT = 0.23119 from the PT cascade (monograph ch. 11).
    This differs from the SM CODATA Q_W by ~ 0.01 % (purely through the
    PT-vs-CODATA shift of sin^2 theta_W).

    Parameters
    ----------
    Z : atomic number (proton count)
    N : neutron count (use the most abundant isotope when in doubt)

    Returns
    -------
    Q_W : float, dimensionless

    Examples
    --------
    >>> abs(weak_charge(1, 0) - 0.07524) < 5e-5
    True
    >>> abs(weak_charge(6, 6) - (-5.5486)) < 5e-4
    True
    """
    return ONE_MINUS_4SIN2 * float(Z) - float(N)


# ---------------------------------------------------------------------
# Most-abundant-isotope neutron count (helper)
# ---------------------------------------------------------------------

# Z -> most-abundant neutron count. Covers the biologically relevant atoms;
# the table is intentionally short — callers should pass N explicitly when
# the isotope matters.
_ABUNDANT_N: dict[int, int] = {
    1: 0,    # 1H
    6: 6,    # 12C
    7: 7,    # 14N
    8: 8,    # 16O
    9: 10,   # 19F
    14: 14,  # 28Si
    15: 16,  # 31P
    16: 16,  # 32S
    17: 18,  # 35Cl  (most abundant)
}


def default_neutron_count(Z: int) -> int:
    """Return the most-abundant-isotope neutron count for the listed atoms.

    Raises KeyError for atoms not in the small biology-centred table; the
    caller is then expected to pass N explicitly to `weak_charge`.
    """
    return _ABUNDANT_N[Z]


def nuclear_weak_charges_default(Z_list: Sequence[int]) -> np.ndarray:
    """Vector of Q_W^A for a list of atomic numbers, using the abundant
    isotope for each. Intended for quick benchmark calls.
    """
    return np.array(
        [weak_charge(int(Z), default_neutron_count(int(Z))) for Z in Z_list],
        dtype=float,
    )


# ---------------------------------------------------------------------
# Spatial gradient of chi_mu(R) chi_nu(R) at a nuclear position
# ---------------------------------------------------------------------


def _ao_value_and_gradient_at(orb,
                              atom_pos: np.ndarray,
                              R: np.ndarray) -> tuple[float, np.ndarray]:
    """Return (chi(R), grad chi(R)) for a single AO centred at atom_pos.

    Uses the same evaluators as the GIAO machinery, so the convention
    (zeta in 1/Angstrom, real-spherical harmonics) matches the rest of
    PT_LCAO. R and atom_pos are 3D points in Angstrom.
    """
    pts = R.reshape(1, 3)
    val = float(evaluate_sto(orb, pts, atom_pos)[0])
    grad = evaluate_sto_gradient(orb, pts, atom_pos)[0]
    return val, np.asarray(grad, dtype=float)


def gradient_of_ao_product_at_point(orb_mu,
                                    pos_mu: np.ndarray,
                                    orb_nu,
                                    pos_nu: np.ndarray,
                                    R: np.ndarray) -> np.ndarray:
    """Closed-form spatial gradient of the AO product at a point.

    Returns the 3-vector

        d/dr_i [ chi_mu(r) chi_nu(r) ]  evaluated at  r = R
            = (d_i chi_mu)(R) chi_nu(R) + chi_mu(R) (d_i chi_nu)(R) .

    This is a SYMMETRIC quantity in (mu, nu). Used internally by the
    `current_density`-style operators. Note that the parity-violation
    matrix element of {p_i, delta^3}_+ involves the ANTI-SYMMETRIC
    combination
       (d_i chi_mu)(R) chi_nu(R) - chi_mu(R) (d_i chi_nu)(R)
    (see module docstring) — see also `_antisym_gradient_sum` below.

    R is the nuclear position (the support of the delta-function); pos_mu
    and pos_nu are the atomic centres of the two AOs. All vectors are
    arrays of shape (3,) in Angstrom.
    """
    chi_mu, grad_mu = _ao_value_and_gradient_at(orb_mu, pos_mu, R)
    chi_nu, grad_nu = _ao_value_and_gradient_at(orb_nu, pos_nu, R)
    return grad_mu * chi_nu + chi_mu * grad_nu


def antisym_gradient_of_ao_product_at_point(orb_mu,
                                            pos_mu: np.ndarray,
                                            orb_nu,
                                            pos_nu: np.ndarray,
                                            R: np.ndarray) -> np.ndarray:
    """Anti-symmetric gradient combination for H_PV matrix elements.

    Returns
        (d_i chi_mu)(R) chi_nu(R) - chi_mu(R) (d_i chi_nu)(R) .

    This is the antisymmetric piece that appears in <mu | {p_i, delta}_+ | nu>
    after integration by parts (see module docstring).
    """
    chi_mu, grad_mu = _ao_value_and_gradient_at(orb_mu, pos_mu, R)
    chi_nu, grad_nu = _ao_value_and_gradient_at(orb_nu, pos_nu, R)
    return grad_mu * chi_nu - chi_mu * grad_nu


# ---------------------------------------------------------------------
# H_PV matrix in the spin-included AO basis
# ---------------------------------------------------------------------


# Pauli matrices in the (alpha, beta) basis.
_SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_SIGMA = np.stack([_SIGMA_X, _SIGMA_Y, _SIGMA_Z], axis=0)  # (3, 2, 2)


def _spatial_gradient_sum(basis: PTMolecularBasis,
                          R_list: np.ndarray,
                          Q_W: np.ndarray) -> np.ndarray:
    """Compute G_i[mu, nu] = sum_A Q_W^A * Im_{mu, nu, i}^A with

        Im_{mu, nu, i}^A = (d_i chi_mu)(R_A) chi_nu(R_A)
                          - chi_mu(R_A) (d_i chi_nu)(R_A) .

    This is the ANTI-SYMMETRIC (in mu, nu) spatial kernel that arises
    from <mu | {p_i, delta^3(r - R_A)}_+ | nu> after integration by
    parts. The factor of `i` and the spin Pauli matrices are NOT applied
    here — they are folded in by `hpv_matrix_elements`.

    Returned array shape (3, n_orb, n_orb), real-valued, antisymmetric
    in (mu, nu).
    """
    n = basis.n_orbitals
    n_A = R_list.shape[0]
    if Q_W.shape[0] != n_A:
        raise ValueError(
            f"Q_W has {Q_W.shape[0]} entries but R_list has {n_A} centres"
        )

    # Pre-evaluate chi_mu(R_A) and (grad chi_mu)(R_A) once per (mu, A).
    chi = np.zeros((n, n_A))
    grad = np.zeros((n, n_A, 3))
    for mu in range(n):
        orb = basis.orbitals[mu]
        pos = basis.coords[basis.atom_index[mu]]
        chi[mu] = evaluate_sto(orb, R_list, pos)
        grad[mu] = evaluate_sto_gradient(orb, R_list, pos)

    G = np.zeros((3, n, n))
    for A in range(n_A):
        qA = float(Q_W[A])
        if qA == 0.0:
            continue
        for i in range(3):
            # antisymmetric combination
            term = (
                np.outer(grad[:, A, i], chi[:, A])
                - np.outer(chi[:, A], grad[:, A, i])
            )
            G[i] += qA * term
    # Enforce exact antisymmetry against floating-point noise.
    for i in range(3):
        G[i] = 0.5 * (G[i] - G[i].T)
    return G


def hpv_matrix_elements(basis: PTMolecularBasis,
                        R_list: np.ndarray | None = None,
                        Q_W: np.ndarray | None = None) -> np.ndarray:
    """Build the H_PV matrix in the spin-included AO basis.

    The result is the complex (2 n_orb) x (2 n_orb) matrix

        H_PV[(mu, s), (nu, s')]
            = prefactor * sum_A Q_W^A * i
              * sum_i [sigma_i]_{s, s'} * G_i^A[mu, nu] ,

    with the antisymmetric (in mu, nu) spatial kernel

        G_i^A[mu, nu] = (d_i chi_mu)(R_A) chi_nu(R_A)
                       - chi_mu(R_A) (d_i chi_nu)(R_A)

    and prefactor = G_F^PT alpha_EM^PT / (2 sqrt(2)) (atomic units).

    Hermiticity
    -----------
    sigma_i is Hermitian; G_i^A is real and antisymmetric in (mu, nu).
    Therefore (i * sigma_i (x) G_i)^dagger = (-i) * sigma_i (x) G_i^T
    = (-i) * sigma_i (x) (-G_i) = (i) * sigma_i (x) G_i. The whole H_PV
    is therefore exactly Hermitian.

    Spin-block ordering
    -------------------
    The (mu, s) flattening is alpha-first then beta:
        index 2*mu     -> (mu, alpha)
        index 2*mu + 1 -> (mu, beta)

    Parameters
    ----------
    basis : PTMolecularBasis
        Spatial AO basis with attached geometry (coords, atom_index).
    R_list : ndarray of shape (n_A, 3), optional
        Nuclear positions in Angstrom. Default: basis.coords (one centre
        per atom in the basis).
    Q_W : ndarray of shape (n_A,), optional
        Weak nuclear charges per centre. Default: most-abundant-isotope
        Q_W^PT for each Z in basis.Z_list.

    Returns
    -------
    H_PV : ndarray of complex shape (2 n_orb, 2 n_orb), Hermitian.
    """
    if R_list is None:
        R_list = basis.coords.copy()
    R_list = np.asarray(R_list, dtype=float)
    if Q_W is None:
        Q_W = nuclear_weak_charges_default(basis.Z_list)
    Q_W = np.asarray(Q_W, dtype=float)

    G = _spatial_gradient_sum(basis, R_list, Q_W)   # (3, n, n), antisymmetric

    n = basis.n_orbitals
    H = np.zeros((2 * n, 2 * n), dtype=complex)
    # H_(mu s)(nu s') = prefactor * i * sum_i sigma_i[s,s'] * G_i[mu,nu]
    for i in range(3):
        sigma_i = _SIGMA[i]            # (2, 2), complex
        Gi = G[i]                      # (n, n), real antisymmetric
        # np.kron(A, B): result[2*mu+s, 2*nu+s'] = A[mu,nu] * B[s,s'].
        H += np.kron(Gi, sigma_i)

    H *= 1j * PREFACTOR_AU

    # Defensive Hermitisation against floating-point noise (exact in
    # exact arithmetic).
    H = 0.5 * (H + H.conj().T)
    return H


# ---------------------------------------------------------------------
# Expectation value E_PV
# ---------------------------------------------------------------------


def _expand_to_spin_basis(D_spatial: np.ndarray) -> np.ndarray:
    """Expand a closed-shell SPATIAL density matrix (n x n) to the spin
    basis (2n x 2n) under the convention alpha = beta.

    For closed-shell RHF, rho_spin = (1/2) rho_spatial (x) I_2. With this
    expansion, sum over electrons recovers Tr(rho_spatial S) electrons.
    """
    D_spatial = np.asarray(D_spatial)
    I2 = np.eye(2, dtype=D_spatial.dtype)
    return 0.5 * np.kron(D_spatial, I2)


def E_PV(density_matrix: np.ndarray,
         hpv_matrix: np.ndarray) -> float:
    """Parity-violation energy E_PV = Tr[D . H_PV] in Hartree.

    Two input conventions are accepted automatically based on the shape:

    * Spatial-only (n_orb x n_orb): interpreted as a closed-shell RHF
      density. It is silently expanded to (1/2) D (x) I_2 in the spin
      basis. With this convention E_PV vanishes identically for any
      closed-shell singlet (the spin trace of sigma is zero).

    * Spin-resolved (2 n_orb x 2 n_orb): used as-is.

    A small numerical imaginary part is ignored (H_PV is Hermitian, D is
    real on the spatial side, so the trace is real to machine precision).
    """
    D = np.asarray(density_matrix)
    n_h = hpv_matrix.shape[0]
    if D.shape == (n_h, n_h):
        D_spin = D
    elif 2 * D.shape[0] == n_h and D.shape[0] == D.shape[1]:
        D_spin = _expand_to_spin_basis(D)
    else:
        raise ValueError(
            f"density_matrix shape {D.shape} incompatible with "
            f"hpv_matrix shape {hpv_matrix.shape}"
        )
    val = np.trace(D_spin @ hpv_matrix)
    return float(np.real(val))


# ---------------------------------------------------------------------
# Convenience: explicit-spin density from an MO matrix
# ---------------------------------------------------------------------


def spin_density_from_mos(C_alpha: np.ndarray,
                          C_beta: np.ndarray,
                          n_alpha: int,
                          n_beta: int) -> np.ndarray:
    """Build a spin-resolved (2n x 2n) one-particle density matrix from
    alpha / beta MO coefficient matrices and occupation counts.

    For each spin channel s in {alpha, beta},
        D_s[mu, nu] = sum_{k < n_s} C_s[mu, k] C_s[nu, k] .

    The returned matrix is block-diagonal in the alpha/beta basis with the
    (mu, s) flattening used by `hpv_matrix_elements`:

        D_spin[2 mu + s, 2 nu + s'] = D_s[mu, nu] * delta_{s, s'}.

    Tracing against H_PV gives a generally non-zero E_PV when the alpha
    and beta densities differ — i.e. for an open-shell or
    spin-orbit-mixed wavefunction.
    """
    n = C_alpha.shape[0]
    if C_beta.shape[0] != n:
        raise ValueError("C_alpha and C_beta must have the same row count")

    D_alpha = C_alpha[:, :n_alpha] @ C_alpha[:, :n_alpha].T if n_alpha > 0 \
        else np.zeros((n, n))
    D_beta = C_beta[:, :n_beta] @ C_beta[:, :n_beta].T if n_beta > 0 \
        else np.zeros((n, n))

    D_spin = np.zeros((2 * n, 2 * n), dtype=float)
    for mu in range(n):
        for nu in range(n):
            D_spin[2 * mu, 2 * nu] = D_alpha[mu, nu]
            D_spin[2 * mu + 1, 2 * nu + 1] = D_beta[mu, nu]
    return D_spin
