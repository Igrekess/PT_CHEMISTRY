"""PT atomic-orbital basis (Phase A).

PT-LCAO Phase A: per-atom Slater-type-orbital (STO) basis built directly
from the existing PT screening machinery in `ptc.atom`. No fit, no
external basis set.

Scope of Phase A
================
Valence-shell minimum basis (matches PTC's overall valence-only philosophy
in `transfer_matrix.py`):
    s-block valence  : n_v s          (1 orbital, 2-fold occ)
    p-block valence  : n_v s + n_v p  (4 orbitals, up to 8-fold occ)
    d-block valence  : n_v s + (n_v-1) d (6 orbitals)
    f-block valence  : n_v s + (n_v-1) d + (n_v-2) f (13 orbitals)

The orbital exponent is uniformly
    zeta(Z) = Z_eff(Z) / n_v   with   Z_eff(Z) = effective_charge(Z)

This keeps every constant traceable to s = 1/2 via screening_action(Z),
which is itself a closed expression in S_3, S_5, S_7 and the PT cascade
(see ptc/atom.py::screening_action).

Higher principal quantum numbers (i.e. inner-shell core orbitals) are
NOT enumerated in Phase A; this matches the philosophy that the PT
density on T^3 is a valence-orbital construction. They will be
re-introduced in Phase B if and only if they are required to close the
density-matrix idempotency check on multi-electron heavy atoms.

Overlaps implemented in Phase A
===============================
* 1s-1s same-zeta (analytic, closed form):
    S(R) = (1 + rho + rho^2/3) * exp(-rho),  rho = zeta R
* 1s-1s different-zeta (analytic, prolate spheroidal coords):
    S = (zA*zB)**1.5 * R^3 / 4 * [A2(p) B0(q) - A0(p) B2(q)]
    p = R*(zA+zB)/2,  q = R*(zA-zB)/2
* Same-orbital identity (R = 0): returns 1 exactly.
* All other (l_A, l_B) combinations: NotImplementedError (Phase B work).

Higher-order overlaps (2s-2s, 2p-2p sigma/pi, 1s-2s etc.) come in
subsequent commits inside Phase A; Phase A is "complete" only when the
overlap matrix S of every benchmark molecule has positive eigenvalues
(checked in Phase B via density_matrix.overlap_matrix).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np

from ptc.atom import effective_charge
from ptc.constants import A_BOHR, S_HALF
from ptc.periodic import (
    block_of,
    capacity,
    l_of,
    n_fill,
    ns_config,
    period,
)


# ─────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PTAtomicOrbital:
    """A single atomic-orbital slot in the PT-LCAO valence basis.

    Attributes
    ----------
    Z      : atomic number of the host atom
    n      : principal quantum number
    l      : angular momentum (0=s, 1=p, 2=d, 3=f)
    m      : magnetic quantum number (-l .. +l), real-spherical convention
    zeta   : Slater orbital exponent in Angstrom^-1, zeta = Z_eff(Z)/(n * a0)
    occ    : fractional occupation in the neutral / charged ground state
             (sum of |m| sub-states is the shell occupation)
    """

    Z: int
    n: int
    l: int
    m: int
    zeta: float
    occ: float

    @property
    def label(self) -> str:
        sym = "spdfgh"[self.l] if self.l < 6 else f"l{self.l}"
        return f"Z{self.Z}_{self.n}{sym}_m{self.m:+d}"


@dataclass
class PTAtomBasis:
    """Per-atom valence STO basis."""

    Z: int
    charge: int
    orbitals: List[PTAtomicOrbital] = field(default_factory=list)

    @property
    def n_orbitals(self) -> int:
        return len(self.orbitals)

    @property
    def total_occ(self) -> float:
        return sum(o.occ for o in self.orbitals)


# ─────────────────────────────────────────────────────────────────────
# Z_eff per shell
# ─────────────────────────────────────────────────────────────────────


def Z_eff_shell(Z: int, n: int, l: int) -> float:
    """PT effective nuclear charge for shell (n, l) of atom Z.

    Phase A: every valence-shell orbital uses the same atomic
    effective_charge(Z) from `ptc.atom`. This is consistent with PTC's
    valence-only philosophy across atom/bond/transfer_matrix:
    screening_action(Z) is the *valence* screening, and ζ = Z_eff/n
    with n = period(Z) reproduces ε_val = -Ry (Z_eff/n)^2 = -IE.

    Inner-shell support (n < period(Z)) is reserved for Phase B; here
    we accept any (n, l) but always return the same valence Z_eff to
    keep the formula closed-form and PT-pure (no Slater-rule constants
    introduced).
    """
    return float(effective_charge(Z))


# ─────────────────────────────────────────────────────────────────────
# Valence-shell occupation enumeration
# ─────────────────────────────────────────────────────────────────────


def occupied_shells(Z: int, charge: int = 0) -> List[tuple]:
    """Enumerate occupied valence shells of atom Z with optional charge.

    Returns
    -------
    List of (n, l, n_electrons) tuples for each occupied valence
    sub-shell. n is the principal quantum number, l the orbital angular
    momentum, and n_electrons the integer count (0 < n_electrons <=
    2*(2l+1)) shared evenly across the (2l+1) magnetic sub-states.

    The ground-state filling follows Aufbau + Madelung promotions
    already encoded in `ptc.periodic`.
    """
    per = period(Z)
    block = block_of(Z)
    l_val = l_of(Z)
    cap_val = capacity(Z)
    n_val_block = n_fill(Z)
    n_s = ns_config(Z)

    shells: List[tuple] = []

    # principal n for s-shell
    n_s_principal = per

    if block == "s":
        if n_val_block > 0:
            shells.append((n_s_principal, 0, n_val_block))

    elif block == "p":
        if n_s > 0:
            shells.append((n_s_principal, 0, n_s))
        if n_val_block > 0:
            shells.append((n_s_principal, 1, n_val_block))

    elif block == "d":
        # (n-1)d valence + ns
        if n_s > 0:
            shells.append((n_s_principal, 0, n_s))
        if n_val_block > 0:
            shells.append((n_s_principal - 1, 2, n_val_block))

    elif block == "f":
        # (n-2)f, plus possible (n-1)d entry, plus ns
        if n_s > 0:
            shells.append((n_s_principal, 0, n_s))
        if n_val_block > 0:
            shells.append((n_s_principal - 2, 3, n_val_block))

    # apply charge by removing/adding electrons from the highest-l shell
    if charge != 0:
        shells = _apply_charge(shells, charge, l_val, cap_val)

    return [s for s in shells if s[2] > 0]


def _apply_charge(shells: List[tuple], charge: int, l_val: int, cap_val: int) -> List[tuple]:
    """Add (negative charge) or remove (positive charge) electrons from
    the highest-angular-momentum valence shell. Closed-shell s-block
    falls back to the s-shell when the spec is empty for that l."""
    if not shells:
        return shells
    out = list(shells)
    # find the shell with the largest l first, then largest n
    idx = max(range(len(out)), key=lambda i: (out[i][1], out[i][0]))
    n, l, ne = out[idx]
    new_ne = ne - charge  # charge>0 means cation -> lose electrons
    if new_ne < 0:
        # underflow: just clamp to 0 (Phase A approximation)
        new_ne = 0
    out[idx] = (n, l, new_ne)
    return out


# ─────────────────────────────────────────────────────────────────────
# build_atom_basis
# ─────────────────────────────────────────────────────────────────────


def build_atom_basis(Z: int, charge: int = 0,
                      polarisation: bool = False) -> PTAtomBasis:
    """Construct PT-LCAO valence basis for atom Z.

    For each occupied valence sub-shell (n, l, n_electrons), one
    PTAtomicOrbital is generated per magnetic quantum number m in
    {-l, ..., +l}, each carrying the equally distributed fractional
    occupation n_electrons / (2l+1). The orbital exponent is
        zeta = Z_eff_shell(Z, n, l) / n

    Parameters
    ----------
    Z, charge : standard
    polarisation : bool, default False
        If True, add ONE shell of polarisation functions with angular
        momentum  l_polar = l_valence + 1, principal quantum number
        n_polar = n_valence + 1, and orbital exponent
            zeta_polar = Z_eff(Z) / (n_polar * a0)
        which is one shell more diffuse than the valence (PT-pure rule:
        no fitted constant). Polarisation orbitals carry zero ground-
        state occupation; they enter the calculation only via the
        coupled-perturbed magnetic response (CP-PT virtual space).
        Standard chemistry analogue: 2p polarisation on H, 3d on C/N/O,
        4d on Si/P/S/Cl, etc.
    """
    if Z < 1:
        raise ValueError(f"Z must be >= 1, got {Z}")

    basis = PTAtomBasis(Z=Z, charge=charge, orbitals=[])
    for n, l, ne in occupied_shells(Z, charge=charge):
        zeta = Z_eff_shell(Z, n, l) / (float(n) * A_BOHR)
        per_m_occ = ne / float(2 * l + 1)
        for m in range(-l, l + 1):
            basis.orbitals.append(
                PTAtomicOrbital(Z=Z, n=n, l=l, m=m, zeta=zeta, occ=per_m_occ)
            )

    if polarisation:
        # Polarisation: one shell of (l_val + 1) orbitals at n = n_val + 1.
        # Use the highest-l valence shell as reference.
        l_val = l_of(Z)
        n_val = period(Z)
        l_polar = l_val + 1
        if l_polar > 2:
            # f-polarisation needs l=3 STO evaluator (pending Phase A
            # continuation for f-block); skip silently for d-block atoms.
            return basis
        n_polar = n_val + 1
        zeta_polar = Z_eff_shell(Z, n_polar, l_polar) / (float(n_polar) * A_BOHR)
        for m in range(-l_polar, l_polar + 1):
            basis.orbitals.append(
                PTAtomicOrbital(
                    Z=Z, n=n_polar, l=l_polar, m=m,
                    zeta=zeta_polar, occ=0.0,
                )
            )
    return basis


# ─────────────────────────────────────────────────────────────────────
# Overlap integrals (analytic for 1s-1s)
# ─────────────────────────────────────────────────────────────────────


def _A_n(n: int, p: float) -> float:
    """Auxiliary integral A_n(p) = int_1^inf x^n exp(-p x) dx, p > 0."""
    if p <= 0.0:
        raise ValueError(f"A_n requires p > 0, got {p}")
    e = math.exp(-p)
    if n == 0:
        return e / p
    if n == 1:
        return e * (p + 1.0) / (p * p)
    if n == 2:
        return e * (p * p + 2.0 * p + 2.0) / (p ** 3)
    raise NotImplementedError(f"A_n only implemented for n in 0..2, got {n}")


def _B_n(n: int, q: float) -> float:
    """Auxiliary integral B_n(q) = int_{-1}^{1} x^n exp(-q x) dx.

    Stable around q = 0 via Taylor series (avoids 0/0 in sinh(q)/q etc.)
    """
    # Use Taylor expansion for small |q| to avoid catastrophic
    # cancellation in (q^2 sinh - 2q cosh + 2 sinh)/q^3 etc. The
    # next neglected term is O(q^4 / 5040) ~ 2e-13 at q = 0.05.
    if abs(q) < 5.0e-2:
        if n == 0:
            return 2.0 + (q ** 2) / 3.0 + (q ** 4) / 60.0
        if n == 1:
            # Taylor: -2q/3 - q^3/15 - q^5/210 - ...
            return -2.0 * q / 3.0 - (q ** 3) / 15.0 - (q ** 5) / 210.0
        if n == 2:
            return 2.0 / 3.0 + (q ** 2) / 5.0 + (q ** 4) / 84.0
    s = math.sinh(q)
    c = math.cosh(q)
    if n == 0:
        return 2.0 * s / q
    if n == 1:
        # int_{-1}^{1} x exp(-q x) dx = -2 cosh(q)/q + 2 sinh(q)/q^2
        return -2.0 * c / q + 2.0 * s / (q * q)
    if n == 2:
        # int_{-1}^{1} x^2 exp(-q x) dx
        # = (2/q^3)[q^2 sinh(q) - 2 q cosh(q) + 2 sinh(q)]
        return 2.0 * (q * q * s - 2.0 * q * c + 2.0 * s) / (q ** 3)
    raise NotImplementedError(f"B_n only implemented for n in 0..2, got {n}")


def _overlap_1s_1s(zeta_a: float, zeta_b: float, R: float) -> float:
    """Analytic 1s-1s STO overlap.

    Closed form via prolate spheroidal coordinates:
        S = (zA zB)^(3/2) (R^3 / 4) [A2(p) B0(q) - A0(p) B2(q)]
        p = R(zA + zB)/2,   q = R(zA - zB)/2
    Reduces to (1 + rho + rho^2/3) exp(-rho) for zA = zB = z, rho = zR.
    """
    if R <= 0.0:
        # same centre: orthonormal basis -> 1 if same orbital, else 0
        # This branch is hit only when called with R == 0 by mistake;
        # callers should use overlap_atomic which short-circuits.
        if abs(zeta_a - zeta_b) < 1.0e-12:
            return 1.0
        # different exponents on same centre: NOT zero in general.
        # Closed form for 1s(zA) | 1s(zB) at R=0:
        #   S = 8 (zA zB)^(3/2) / (zA + zB)^3
        return 8.0 * (zeta_a * zeta_b) ** 1.5 / (zeta_a + zeta_b) ** 3
    p = R * (zeta_a + zeta_b) / 2.0
    q = R * (zeta_a - zeta_b) / 2.0
    pre = (zeta_a * zeta_b) ** 1.5 * (R ** 3) / 4.0
    bracket = _A_n(2, p) * _B_n(0, q) - _A_n(0, p) * _B_n(2, q)
    return pre * bracket


def overlap_atomic(orb_A: PTAtomicOrbital,
                   orb_B: PTAtomicOrbital,
                   r_AB: np.ndarray | float = 0.0) -> float:
    """Real overlap < phi_A | phi_B > between two PT atomic orbitals.

    Parameters
    ----------
    orb_A, orb_B : PTAtomicOrbital
        Each carries (Z, n, l, m, zeta).
    r_AB : np.ndarray of shape (3,) OR scalar
        Vector from centre A to centre B (Angstrom-independent: zeta is
        in inverse-length units consistent with R).

    Phase A coverage
    ----------------
    Implemented:
        - same orbital, R = 0  ->  1.0 exactly
        - 1s-1s (l=0, n=1)     ->  closed form
    Not yet implemented (raise NotImplementedError):
        - any (l, n) combination beyond 1s-1s
    The tests in test_lcao_atomic_basis only exercise the implemented
    cases; full coverage lands in Phase A continuation commits.
    """
    if isinstance(r_AB, np.ndarray):
        R = float(np.linalg.norm(r_AB))
    else:
        R = float(r_AB)

    same_centre = R < 1.0e-12
    same_orbital = (
        same_centre
        and orb_A.Z == orb_B.Z
        and orb_A.n == orb_B.n
        and orb_A.l == orb_B.l
        and orb_A.m == orb_B.m
        and abs(orb_A.zeta - orb_B.zeta) < 1.0e-12
    )
    if same_orbital:
        return 1.0

    # Same-centre, different angular momentum (orthogonal Y_lm) -> 0
    if same_centre and (orb_A.l != orb_B.l or orb_A.m != orb_B.m):
        return 0.0

    # 1s-1s (n=1, l=0, m=0) on different centres: keep the analytic
    # closed form as the reference path; the general numerical
    # quadrature in sto_overlap is validated to agree at 1e-10.
    if (
        orb_A.n == 1 and orb_B.n == 1
        and orb_A.l == 0 and orb_B.l == 0
        and orb_A.m == 0 and orb_B.m == 0
    ):
        return _overlap_1s_1s(orb_A.zeta, orb_B.zeta, R)

    # Build the actual r_AB vector
    if isinstance(r_AB, np.ndarray):
        r_vec = r_AB.astype(float)
    else:
        r_vec = np.array([0.0, 0.0, float(r_AB)])

    # Dispatch s and p (l <= 1) to the analytic Slater-Koster path
    # (fast prolate spheroidal 2D quadrature + geometric SK rotation).
    if orb_A.l <= 1 and orb_B.l <= 1:
        from ptc.lcao.sto_overlap import overlap_sp_general
        return overlap_sp_general(orb_A, orb_B, r_vec)

    # d-orbital (l = 2) and any pair involving it: fallback to 3D
    # Gauss-quadrature overlap. Slower but universal; validated against
    # analytic 1s-1s when both l = 0. Supports s-d, p-d, d-d.
    if orb_A.l <= 2 and orb_B.l <= 2:
        from ptc.lcao.sto_overlap import overlap_3d_numerical
        return overlap_3d_numerical(orb_A, orb_B, r_vec)

    raise NotImplementedError(
        f"overlap_atomic: (n_A,l_A,m_A,n_B,l_B,m_B) = "
        f"({orb_A.n},{orb_A.l},{orb_A.m},{orb_B.n},{orb_B.l},{orb_B.m}) "
        f"not yet supported. s/p/d done; f-orbital pending Phase A continuation."
    )
