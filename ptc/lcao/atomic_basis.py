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
from ptc.constants import A_BOHR, GAMMA_3, GAMMA_5, GAMMA_7, S_HALF
from ptc.periodic import (
    block_of,
    capacity,
    l_of,
    n_fill,
    ns_config,
    period,
)


# Phase 2 (Axis 1) — PT-pure split-valence factors. Each occupied valence
# shell of a DZ basis is split into two STOs: an inner "contracted" zeta
# scaled by GAMMA_3 (PT cascade at p=3) and an outer "diffuse" zeta scaled
# by GAMMA_5 (p=5). DZPD adds a very-diffuse function with GAMMA_7 (p=7).
# Phase 3: DZ2P adds a second polarisation shell at l_val + 1, n_val + 2
# (e.g. 4d on top of 3d for C/N/O). Nothing is fitted: 0.808/0.696/0.595
# come straight from the PT cascade.
_BASIS_TYPES = ("SZ", "DZ", "TZ", "DZP", "TZP", "DZ2P", "DZPD")


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


def Z_eff_shell(Z: int, n: int, l: int, method: str = "pt") -> float:
    """PT effective nuclear charge for shell (n, l) of atom Z.

    The three available methods probe Z_eff at three different scales:

    Phase A (``method='pt'``, default) : every valence-shell orbital
    uses the atomic ``effective_charge(Z)`` from ``ptc.atom``, which is
    calibrated to the **binding-energy scale** — i.e. ``ε_val = -Ry
    (Z_eff/n)^2 = -IE``. This is correct for energetic observables but
    leaves the orbital tail too diffuse for the chemical-shielding
    integral ⟨1/r³⟩.

    Phase 6.B.1 (``method='slater'``) returns the textbook Slater Z_eff
    (Slater 1930, *Phys. Rev.* 36, 57) for benchmarking. Slater's rules
    give about twice the effective charge of PT for second-row valence
    (e.g. C 2p: 3.25 vs 1.82) because they target the **orbital-tail
    scale**. Used as a non-PT reference.

    Phase 6.C (``method='pt-shielding'``) returns a PT-pure
    orbital-scale Z_eff that reproduces Slater within ~3% from first
    principles, with NO empirical fit. The two screening coefficients
    are derived from the cascade::

        β_intra = γ₃ · γ₅ · γ₇            ≈ 0.335   (≈ Slater 0.35)
        β_inner = exp(-S_HALF / P₁)       ≈ 0.846   (≈ Slater 0.85)
        β_deep  = 1.0  (full screening for shells ≥ n - 2)

    γ_p are the PT cascade anomalous dimensions at μ* = 15 (PT chap. 6
    holonomy); S_HALF = 1/2 is the unique input; P₁ = 3 is the smallest
    active prime. The 1s intrashell coefficient gets an extra γ₃ for the
    innermost-circle:  β_1s_intra = γ₃ · β_intra ≈ 0.270 (≈ Slater 0.30).

    This recovers the orbital-scale screening from the same s = 1/2
    starting point that gives the binding-energy screening, but applied
    to the wavefunction tail rather than to the eigenvalue.
    """
    if method == "slater":
        return _slater_z_eff(Z, n, l)
    if method == "pt-shielding":
        return _pt_shielding_z_eff(Z, n, l)
    if method != "pt":
        raise ValueError(
            f"method must be 'pt' / 'slater' / 'pt-shielding', got {method!r}"
        )
    return float(effective_charge(Z))


# ─────────────────────────────────────────────────────────────────────
# PT-shielding Z_eff (orbital-scale, Phase 6.C)
# ─────────────────────────────────────────────────────────────────────


_BETA_INTRA = GAMMA_3 * GAMMA_5 * GAMMA_7        # ≈ 0.3349 (≈ Slater 0.35)
_BETA_INNER = math.exp(-S_HALF / 3.0)            # ≈ 0.8465 (≈ Slater 0.85)
                                                  # P₁ = 3 hard-coded to avoid
                                                  # circular import.
_BETA_DEEP = 1.0                                  # full screening (n ≤ n_target-2)
_BETA_1S_INTRA = GAMMA_3 * _BETA_INTRA           # ≈ 0.2704 (≈ Slater 0.30)


def _pt_shielding_z_eff(Z: int, n: int, l: int) -> float:
    """PT-pure orbital-scale Z_eff at shell (n, l).

    Algorithm — exactly the Slater shell partition, but with PT-derived
    coefficients (γ₃·γ₅·γ₇, exp(-S_HALF/P₁), 1, γ₃·γ₃·γ₅·γ₇) instead of
    the empirical Slater (0.35, 0.85, 1.00, 0.30). All four constants
    derive from the cascade at μ* = 15; nothing is fitted.
    """
    occ = _slater_shell_occupations(Z)
    group = _slater_group_label(n, l)

    if group == "1s":
        n_1s = occ.get("1s", 1)
        sigma = _BETA_1S_INTRA * max(n_1s - 1, 0)
        return float(Z) - sigma

    if group.endswith("sp"):
        n_target = int(group[0])
        n_in_group = occ.get(group, 0)
        sigma = _BETA_INTRA * max(n_in_group - 1, 0)
        for label, ne in occ.items():
            if label == group:
                continue
            n_other = int(label[0])
            if n_other == n_target - 1:
                sigma += _BETA_INNER * ne
            elif n_other < n_target - 1:
                sigma += _BETA_DEEP * ne
        return float(Z) - sigma

    # d / f shells: in-shell at γ₃·γ₅·γ₇, all lower shells fully screen.
    n_target = int(group[0])
    n_in_group = occ.get(group, 0)
    sigma = _BETA_INTRA * max(n_in_group - 1, 0)
    for label, ne in occ.items():
        if label == group:
            continue
        n_other = int(label[0])
        l_other = label[1:]
        if n_other < n_target:
            sigma += _BETA_DEEP * ne
        elif n_other == n_target and l_other in ("sp",):
            sigma += _BETA_DEEP * ne
    return float(Z) - sigma


def _slater_shell_occupations(Z: int) -> dict:
    """Slater shell occupations for atom Z.

    Returns a dict keyed on the Slater group label
    ('1s', '2sp', '3sp', '3d', '4sp', '4d', '4f', '5sp', '5d', ...) →
    integer electron count in that group. This follows Slater's
    grouping convention where (ns, np) share a screening group and
    (nd) is its own group.
    """
    # Walk Aufbau order until total = Z, populating each Slater group.
    aufbau = [
        ("1s", 2), ("2sp", 8), ("3sp", 8), ("3d", 10),
        ("4sp", 8), ("4d", 10), ("4f", 14),
        ("5sp", 8), ("5d", 10), ("5f", 14),
        ("6sp", 8), ("6d", 10), ("7sp", 8),
    ]
    out: dict = {}
    remaining = Z
    for label, cap in aufbau:
        if remaining <= 0:
            break
        ne = min(remaining, cap)
        out[label] = ne
        remaining -= ne
    return out


def _slater_group_label(n: int, l: int) -> str:
    """Map (n, l) to the Slater screening group label."""
    if n == 1 and l == 0:
        return "1s"
    if l <= 1:
        return f"{n}sp"
    if l == 2:
        return f"{n}d"
    if l == 3:
        return f"{n}f"
    raise ValueError(f"unsupported (n, l) = ({n}, {l})")


def _slater_z_eff(Z: int, n: int, l: int) -> float:
    """Slater's screening Z_eff for shell (n, l) of atom Z.

    Rules (Slater 1930):
      - 1s: σ = 0.30 × (N(1s) - 1)
      - ns / np target: σ = 0.35 × (N(nsp) - 1)
                          + 0.85 × N(shell_at_n-1)
                          + 1.00 × sum(N(shell_at_n-2_or_below))
      - nd / nf target: σ = 0.35 × (N(group) - 1)
                          + 1.00 × sum(N(all_lower_groups))

    For C (Z=6, n=2, l=1): N(2sp)=4, N(1s)=2
        σ = 0.35×3 + 0.85×2 = 1.05 + 1.70 = 2.75
        Z_eff = 6 - 2.75 = 3.25  → ζ = 3.25 / 2 = 1.625 a.u. ✓
    """
    occ = _slater_shell_occupations(Z)
    group = _slater_group_label(n, l)

    if group == "1s":
        n_1s = occ.get("1s", 1)
        sigma = 0.30 * max(n_1s - 1, 0)
        return float(Z) - sigma

    if group.endswith("sp"):
        n_target = int(group[0])
        n_in_group = occ.get(group, 0)
        sigma = 0.35 * max(n_in_group - 1, 0)
        # 0.85 from each electron one shell below
        for label, ne in occ.items():
            if label == group:
                continue
            n_other = int(label[0])
            if n_other == n_target - 1:
                sigma += 0.85 * ne
            elif n_other < n_target - 1:
                sigma += 1.00 * ne
        return float(Z) - sigma

    # d or f shells
    n_target = int(group[0])
    n_in_group = occ.get(group, 0)
    sigma = 0.35 * max(n_in_group - 1, 0)
    for label, ne in occ.items():
        if label == group:
            continue
        n_other = int(label[0])
        l_other = label[1:]
        # All electrons in groups *below* this d/f group screen by 1.0.
        # Slater treats lower-n shells AND same-n s/p as fully screening.
        if n_other < n_target:
            sigma += 1.00 * ne
        elif n_other == n_target and l_other in ("sp",):
            sigma += 1.00 * ne
    return float(Z) - sigma


# ─────────────────────────────────────────────────────────────────────
# Valence-shell occupation enumeration
# ─────────────────────────────────────────────────────────────────────


def core_shells(Z: int) -> List[tuple]:
    """Enumerate INNER (noble-gas-core) shells for atom Z.

    Returns a list of (n, l, n_electrons) for each filled core shell
    below the valence period. Used by `build_atom_basis(include_core=True)`
    to build an all-electron basis (Phase 4).

    Currently supports periods 2-4 (cores up to [Ar]):
        period 2 (Li-Ne): [He] = 1s²
        period 3 (Na-Ar): [Ne] = 1s² 2s² 2p⁶
        period 4 (K-Kr) : [Ar] = 1s² 2s² 2p⁶ 3s² 3p⁶
    """
    per = period(Z)
    cores: List[tuple] = []
    if per >= 2:
        cores.append((1, 0, 2))                # 1s²
    if per >= 3:
        cores.append((2, 0, 2))                # 2s²
        cores.append((2, 1, 6))                # 2p⁶
    if per >= 4:
        cores.append((3, 0, 2))                # 3s²
        cores.append((3, 1, 6))                # 3p⁶
    if per >= 5:
        cores.append((3, 2, 10))               # 3d¹⁰ (filled in [Ar] core for Z>=29)
        cores.append((4, 0, 2))                # 4s²
        cores.append((4, 1, 6))                # 4p⁶
    # Higher periods deliberately unsupported in Phase 4 — focus on
    # second-row atoms (C, N, O, F) where benzene NICS lives.
    return cores


def _zeta_core_slater(Z: int, n: int, l: int) -> float:
    """Slater-style exponent for an INNER-CORE shell.

    Uses simplified Slater screening rules:
      - 1s: zeta = (Z - 0.30) / n / a₀
      - 2s, 2p: zeta = (Z - 2*0.85) / n / a₀
      - 3s, 3p: zeta = (Z - 2*1.0 - 8*0.85) / n / a₀
      - 3d (in core only when present): same n=3 prescription as 3s/3p
    These are the standard chemistry-textbook values; they make the core
    1s very tight (ζ ~ 5.7 / a₀ for C) and ensure the SCF pulls valence
    electrons out of core states.
    """
    if n == 1 and l == 0:
        screening = 0.30
    elif n == 2 and l <= 1:
        screening = 2.0 * 0.85
    elif n == 3 and l <= 2:
        screening = 2.0 * 1.0 + 8.0 * 0.85
    else:
        screening = 0.0  # fallback: bare nuclear charge
    z_eff = max(Z - screening, 0.5 * Z)
    return float(z_eff) / (float(n) * A_BOHR)


def occupied_shells(Z: int, charge: int = 0,
                     include_f_block_d_shell: bool = False) -> List[tuple]:
    """Enumerate occupied valence shells of atom Z with optional charge.

    Parameters
    ----------
    include_f_block_d_shell : if True, allocate the (n-1)d shell on the
        seven f-block atoms with NIST-confirmed d^1/d^2 promotion
        (Ce, Gd, Th, Pa, U, Np, Cm). Default False to keep the Hueckel
        path well-conditioned.
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
        # ns + (n-1)d? + (n-2)f
        # NIST anomalous (n-1)d^1 promotion (opt-in only, Hueckel-safe default).
        _F_BLOCK_D_OCCUPATION = {
            58: 1, 64: 1, 90: 2, 91: 1, 92: 1, 93: 1, 96: 1,
        }
        d_count = (
            _F_BLOCK_D_OCCUPATION.get(Z, 0) if include_f_block_d_shell else 0
        )
        if n_s > 0:
            shells.append((n_s_principal, 0, n_s))
        if d_count > 0:
            shells.append((n_s_principal - 1, 2, d_count))
        f_count = max(0, n_val_block - d_count)
        if f_count > 0:
            shells.append((n_s_principal - 2, 3, f_count))

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
                      polarisation: bool = False,
                      basis_type: str = "SZ",
                      include_core: bool = False,
                      zeta_method: str = "pt",
                      include_f_block_d_shell: bool = False) -> PTAtomBasis:
    """Construct PT-LCAO valence basis for atom Z.

    For each occupied valence sub-shell (n, l, n_electrons), one
    PTAtomicOrbital is generated per magnetic quantum number m in
    {-l, ..., +l}, each carrying the equally distributed fractional
    occupation n_electrons / (2l+1). The orbital exponent for the
    single-zeta (SZ) basis is
        zeta = Z_eff_shell(Z, n, l, method=zeta_method) / n

    Parameters
    ----------
    Z, charge : standard
    polarisation : bool, default False
        Legacy boolean flag: equivalent to selecting basis_type='SZP' /
        'DZP' depending on whether basis_type is 'SZ' or 'DZ'. If
        basis_type already encodes polarisation ('DZP', 'DZPD') this
        flag is redundant and ignored.

        When polarisation is added, ONE shell of (l_val + 1) orbitals at
        n = n_val + 1 is appended with exponent
            zeta_polar = Z_eff(Z) / (n_polar * a0)
        carrying zero ground-state occupation (virtual space).
    basis_type : str, default 'SZ'
        - 'SZ'   : single zeta per shell (legacy, SZ behaviour).
        - 'DZ'   : split valence — each occupied valence shell becomes
                   TWO STOs with PT-pure exponents
                       zeta_inner = (Z_eff / n_a0) * GAMMA_3   (= 0.808 zeta)
                       zeta_outer = (Z_eff / n_a0) * GAMMA_5   (= 0.696 zeta)
                   The shell occupation is split equally between the two:
                   ne / (2l+1) / 2 each.
        - 'DZP'  : DZ + one polarisation shell at l_val + 1, n_val + 1.
        - 'TZ'   : triple-zeta valence — three STOs per occupied shell
                   with exponents scaled by GAMMA_3 / GAMMA_5 / GAMMA_7.
                   Each occupation per (n, l, m) is split equally in
                   thirds (Phase 6.B.2).
        - 'TZP'  : TZ + one polarisation shell at l_val + 1 (GAMMA_7
                   scaled exponent like DZP).
        - 'DZPD' : DZP + one very-diffuse shell at l_val, n_val + 1 with
                   exponent
                       zeta_diffuse = (Z_eff / ((n+1)*a0)) * GAMMA_7
                                    (= 0.595 / (n+1) shell-spread)
                   Useful for anions and lone-pair-rich molecules.

        All gamma factors (GAMMA_3, GAMMA_5, GAMMA_7) come from the PT
        cascade at p = 3, 5, 7 (see ptc.constants). Nothing fitted.

    zeta_method : str, default 'pt'
        How to compute Z_eff for each valence shell.
        'pt'    : PT-pure effective_charge(Z) (Phase A through 5).
        'slater': textbook Slater rules (Phase 6.B.1 benchmark) — gives
                  much tighter STOs (e.g. C 2p: ζ_Slater = 1.625 vs
                  ζ_PT ≈ 0.91 a.u.). Used to test if PT density is too
                  diffuse for chemical-shielding work.
    """
    if Z < 1:
        raise ValueError(f"Z must be >= 1, got {Z}")
    if basis_type not in _BASIS_TYPES:
        raise ValueError(
            f"basis_type must be one of {_BASIS_TYPES}, got {basis_type!r}"
        )

    basis = PTAtomBasis(Z=Z, charge=charge, orbitals=[])

    # Phase 4: optionally include inner-shell (noble-gas) core orbitals.
    # These are NEVER split for DZ — the valence-DZ split prescription is
    # for the active outer shells only. Core uses Slater screening.
    if include_core:
        for n, l, ne in core_shells(Z):
            zeta_core = _zeta_core_slater(Z, n, l)
            per_m_occ = ne / float(2 * l + 1)
            for m in range(-l, l + 1):
                basis.orbitals.append(
                    PTAtomicOrbital(Z=Z, n=n, l=l, m=m,
                                    zeta=zeta_core, occ=per_m_occ)
                )

    # Occupied valence shells
    for n, l, ne in occupied_shells(
        Z, charge=charge,
        include_f_block_d_shell=include_f_block_d_shell,
    ):
        z_base = Z_eff_shell(Z, n, l, method=zeta_method) / (float(n) * A_BOHR)
        per_m_occ = ne / float(2 * l + 1)

        if basis_type == "SZ":
            for m in range(-l, l + 1):
                basis.orbitals.append(
                    PTAtomicOrbital(Z=Z, n=n, l=l, m=m, zeta=z_base, occ=per_m_occ)
                )
        elif basis_type in ("TZ", "TZP"):
            # Triple-zeta valence: three STOs per (n,l,m) at exponents
            # gamma_3, gamma_5, gamma_7 × z_base. Occupation split in
            # thirds. Phase 6.B.2: third zeta widens the radial flexibility
            # for chemical-shielding response.
            zeta_inner = z_base * GAMMA_3
            zeta_middle = z_base * GAMMA_5
            zeta_outer = z_base * GAMMA_7
            third_occ = per_m_occ / 3.0
            for m in range(-l, l + 1):
                basis.orbitals.append(
                    PTAtomicOrbital(Z=Z, n=n, l=l, m=m,
                                    zeta=zeta_inner, occ=third_occ)
                )
                basis.orbitals.append(
                    PTAtomicOrbital(Z=Z, n=n, l=l, m=m,
                                    zeta=zeta_middle, occ=third_occ)
                )
                basis.orbitals.append(
                    PTAtomicOrbital(Z=Z, n=n, l=l, m=m,
                                    zeta=zeta_outer, occ=third_occ)
                )
        else:
            # DZ family: split each shell into inner (gamma_3) + outer (gamma_5).
            # Occupation is split symmetrically — the SCF / density matrix sees
            # the same total ne but with two basis functions per (n,l,m).
            zeta_inner = z_base * GAMMA_3
            zeta_outer = z_base * GAMMA_5
            half_occ = 0.5 * per_m_occ
            for m in range(-l, l + 1):
                basis.orbitals.append(
                    PTAtomicOrbital(Z=Z, n=n, l=l, m=m,
                                    zeta=zeta_inner, occ=half_occ)
                )
                basis.orbitals.append(
                    PTAtomicOrbital(Z=Z, n=n, l=l, m=m,
                                    zeta=zeta_outer, occ=half_occ)
                )

    # Polarisation shell(s) at l = l_val + 1, principal quantum n_val + 1
    # and (DZ2P only) also n_val + 2.
    # Phase 2 axis 1: scale the (n+1) polarisation exponent by GAMMA_7
    # (PT cascade at p=7 — the third active prime, naturally associated
    # with the second-next-shell anomalous dimension).
    # Phase 3: DZ2P adds a second polarisation shell at n_val + 2 with
    # exponent scaled by GAMMA_7^2 (cumulative diffusion); this gives a
    # 4d-on-C set well separated from the 3d set.
    # The legacy SZ-with-polarisation path (basis_type='SZ' and
    # polarisation=True) keeps the unscaled exponent for backwards
    # compatibility with existing test_lcao_polarisation expectations.
    needs_polar = polarisation or basis_type in ("DZP", "TZP", "DZ2P", "DZPD")
    if needs_polar:
        l_val = l_of(Z)
        n_val = period(Z)
        l_polar = l_val + 1
        if l_polar <= 4:
            # First polarisation shell
            n_polar = n_val + 1
            zeta_polar = (
                Z_eff_shell(Z, n_polar, l_polar)
                / (float(n_polar) * A_BOHR)
            )
            if basis_type in ("DZP", "TZP", "DZ2P", "DZPD"):
                # Without diffusing the polarisation, the (n+1) shell sits
                # as tight as the valence outer DZ exponent, producing
                # near-linear dependence and a sign flip in sigma^p of
                # aromatic ring currents. GAMMA_7 (= 0.595) puts the
                # polarisation l = l_val + 1 shell in the natural region
                # of an angularly promoted virtual orbital.
                zeta_polar *= GAMMA_7
            for m in range(-l_polar, l_polar + 1):
                basis.orbitals.append(
                    PTAtomicOrbital(
                        Z=Z, n=n_polar, l=l_polar, m=m,
                        zeta=zeta_polar, occ=0.0,
                    )
                )
            # Second polarisation shell (DZ2P only): n_val + 2 at l_val + 1
            # with cumulative GAMMA_7 spread (a second cascade step).
            if basis_type == "DZ2P":
                n_polar2 = n_val + 2
                zeta_polar2 = (
                    Z_eff_shell(Z, n_polar2, l_polar)
                    / (float(n_polar2) * A_BOHR)
                ) * GAMMA_7 * GAMMA_7
                for m in range(-l_polar, l_polar + 1):
                    basis.orbitals.append(
                        PTAtomicOrbital(
                            Z=Z, n=n_polar2, l=l_polar, m=m,
                            zeta=zeta_polar2, occ=0.0,
                        )
                    )
        # f-polarisation requires l=3 STO evaluator (pending continuation);
        # silently omitted for d-block atoms, matching legacy SZ behaviour.

    # Very-diffuse shell at l_val, n+1 with GAMMA_7 spread
    if basis_type == "DZPD":
        l_val = l_of(Z)
        n_val = period(Z)
        n_diff = n_val + 1
        zeta_diff = Z_eff_shell(Z, n_diff, l_val) / (float(n_diff) * A_BOHR) * GAMMA_7
        if l_val <= 2:
            for m in range(-l_val, l_val + 1):
                basis.orbitals.append(
                    PTAtomicOrbital(
                        Z=Z, n=n_diff, l=l_val, m=m,
                        zeta=zeta_diff, occ=0.0,
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

    # Same-centre, same (l, m) but different (n, zeta) — purely radial
    # overlap via < r^(n_a-1) | r^(n_b-1) > with normalised STO weights.
    # This is the case hit by DZ split-valence (same nlm, two zetas).
    if same_centre:
        n_a, n_b = orb_A.n, orb_B.n
        z_a, z_b = orb_A.zeta, orb_B.zeta
        # N = (2 zeta)^n sqrt(2 zeta / (2n)!)
        Na = (2.0 * z_a) ** n_a * math.sqrt(2.0 * z_a / math.factorial(2 * n_a))
        Nb = (2.0 * z_b) ** n_b * math.sqrt(2.0 * z_b / math.factorial(2 * n_b))
        # int_0^inf r^(n_a + n_b) exp(-(z_a + z_b) r) dr = (n_a+n_b)! / (z_a+z_b)^(n_a+n_b+1)
        rad_integral = (
            math.factorial(n_a + n_b)
            / ((z_a + z_b) ** (n_a + n_b + 1))
        )
        return Na * Nb * rad_integral

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

    # d / f / g orbitals: 3D Gauss-quadrature overlap. Universal,
    # dispatches to evaluate_sto for any (n, l, m). Supports s/p/d/f/g.
    if orb_A.l <= 4 and orb_B.l <= 4:
        from ptc.lcao.sto_overlap import overlap_3d_numerical
        return overlap_3d_numerical(orb_A, orb_B, r_vec)

    raise NotImplementedError(
        f"overlap_atomic: (n_A,l_A,m_A,n_B,l_B,m_B) = "
        f"({orb_A.n},{orb_A.l},{orb_A.m},{orb_B.n},{orb_B.l},{orb_B.m}) "
        f"not yet supported. s/p/d/f/g done; h / higher pending."
    )
