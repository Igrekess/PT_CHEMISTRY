"""Phase 6.B.11a/b/c — Coupled-Cluster Doubles (CCD) iteration.

Phase 6.B.11a — LCCD (linearized CCD) base machinery
Phase 6.B.11b — Pulay DIIS extrapolation for fast convergence
Phase 6.B.11c — quadratic T2² terms via F_oo / F_vv intermediates,
              upgrading LCCD to full closed-shell CCD.

Iterates the closed-shell T2 amplitudes using ONLY the linear residual
terms identified and validated in Phase 6.B.10:

    R_ij^ab(t) = (ia|jb)
               + ½ Σ_cd (ac|bd) t̃_ij^cd          (pp ladder)
               + ½ Σ_kl (ki|lj) t̃_kl^ab          (hh ladder)
               + ring contribution                 (P(ij,ab)-symmetric)

Update:  t_ij^ab ← R_ij^ab / Δε_ij^ab.

This is "infinite-order MP3" : at iteration 0 we recover MP2; at
iteration 1 we recover MP3-with-ring (Phase 6.B.10c). At convergence
we get LCCD energy, which is bounded between MP3 and full CCD.

What LCCD is NOT
  * no T1 amplitudes (added in Phase 6.B.11c)
  * no quadratic T2×T2 contractions (Phase 6.B.11b)
  * no Fock-block terms (vanish at canonical HF orbitals → fine)
  * no Λ-equations (Phase 6.B.11d)

LCCD is sufficient as a stepping stone for the iteration infrastructure
(convergence check, ERI cache, residual factorisation) that the full
CCSD pipeline will need.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from ptc.lcao.density_matrix import PTMolecularBasis
from ptc.lcao.mp2 import (
    MP2Result,
    mo_eri_block,
    mp2_amplitudes,
    mp2_at_hf,
    mp2_density_correction,
)
from ptc.lcao.mp3 import _mp3_hh_ladder, _mp3_pp_ladder, _mp3_ring, _t_tilde


# ─────────────────────────────────────────────────────────────────────
# Quadratic T2² intermediates (Phase 6.B.11c)
# ─────────────────────────────────────────────────────────────────────


def _ccd_F_oo_intermediate(t: np.ndarray,
                             eri_klcd: np.ndarray) -> np.ndarray:
    """Occupied-occupied T2-renormalized Fock intermediate.

        F_oo[k, i] = ½ Σ_lcd (kl|cd) t̃_{il}^{cd}

    Layout: eri[k, l, c, d] ≡ (kl|cd), t[i, c, l, d] ≡ t_{il}^{cd}.
    Returns array shape (n_occ, n_occ).
    """
    t_t = _t_tilde(t)
    return 0.5 * np.einsum("klcd,icld->ki", eri_klcd, t_t, optimize=True)


def _ccd_F_vv_intermediate(t: np.ndarray,
                             eri_klcd: np.ndarray) -> np.ndarray:
    """Virtual-virtual T2-renormalized Fock intermediate.

        F_vv[a, c] = ½ Σ_kld (kl|cd) t̃_{kl}^{ad}

    Returns array shape (n_virt, n_virt).
    """
    t_t = _t_tilde(t)
    return 0.5 * np.einsum("klcd,kald->ac", eri_klcd, t_t, optimize=True)


def _ccd_quadratic_F_intermediates(t: np.ndarray,
                                     eri_klcd: np.ndarray) -> np.ndarray:
    """Quadratic T2² contribution to the CCD T2 residual via the
    F_oo / F_vv Fock-like intermediates.

        R_quad[i,a,j,b] = - Σ_k F_oo[k,i] t_{kj}^{ab}
                          + Σ_c F_vv[a,c] t_{ij}^{cb}
                          + P(ij,ab) image (i↔j, a↔b)

    The P(ij,ab) symmetrisation is applied explicitly via
    `quad + quad.transpose(2, 3, 0, 1)`. The result is the full
    quadratic contribution (not divided by 2) as defined in the
    standard closed-shell CCD residual.
    """
    F_oo = _ccd_F_oo_intermediate(t, eri_klcd)
    F_vv = _ccd_F_vv_intermediate(t, eri_klcd)

    # Unsymmetrised contributions
    quad = (
        -np.einsum("ki,kajb->iajb", F_oo, t, optimize=True)
        + np.einsum("ac,icjb->iajb", F_vv, t, optimize=True)
    )
    # P(ij,ab) symmetrisation : ring closes under (i↔j, a↔b)
    return quad + quad.transpose(2, 3, 0, 1)


@dataclass
class LCCDResult:
    """Bundle of LCCD quantities at convergence."""
    t: np.ndarray                  # converged T2 amplitudes
    eri_mo_iajb: np.ndarray        # (ia|jb) — same as MP2
    e_corr: float                  # LCCD correlation energy in eV
    d_occ: np.ndarray              # MP2-style 1-RDM occ block
    d_vir: np.ndarray              # MP2-style 1-RDM vir block
    n_iter: int                    # number of iterations used
    history_e: List[float] = field(default_factory=list)  # E per iteration


def _energy_from_amplitudes(t: np.ndarray, eri_iajb: np.ndarray) -> float:
    """Closed-shell correlation energy : E = -Σ t_ij^ab × (2 K - K^swap)."""
    combo = 2.0 * eri_iajb - eri_iajb.transpose(0, 3, 2, 1)
    return -float(np.sum(t * combo))


class _DIIS:
    """Pulay DIIS extrapolation for CC amplitude iterations.

    Standard formulation (Pulay 1980): keep the last ``max_vectors``
    amplitude/error pairs (t_k, e_k) with e_k = t_{k+1}^{raw} - t_k.
    The extrapolated amplitude minimises the effective residual

        || Σ_k c_k e_k ||²    subject to    Σ_k c_k = 1

    via the Lagrange-multiplier system

        | B  -1 | | c |   | 0 |
        |       | |   | = |   |        with B_{ij} = <e_i, e_j>.
        | -1  0 | | λ |   |-1 |

    The extrapolated amplitude is then ``t_ext = Σ c_k t_k`` (using the
    *raw-update* amplitudes t_{k+1}^{raw}, not the previous t_k —
    standard DIIS convention).

    Falls back to identity (no extrapolation) when fewer than two
    history entries are available, or when the linear system is
    singular.
    """

    def __init__(self, max_vectors: int = 8):
        if max_vectors < 1:
            raise ValueError("DIIS max_vectors must be >= 1")
        self.max_vectors = int(max_vectors)
        self._history_t: List[np.ndarray] = []
        self._history_e: List[np.ndarray] = []

    def reset(self) -> None:
        """Clear history (e.g. when restarting from a different guess)."""
        self._history_t = []
        self._history_e = []

    @property
    def n_vectors(self) -> int:
        return len(self._history_t)

    def extrapolate(self,
                     t_raw: np.ndarray,
                     t_prev: np.ndarray) -> np.ndarray:
        """Update history with (t_raw, e=t_raw-t_prev) and extrapolate.

        Returns the next-iteration amplitude. With a single history
        entry, this is just ``t_raw`` (identity). With ≥ 2 entries,
        applies the Pulay DIIS combination.
        """
        # Build error vector and append to history
        err = (t_raw - t_prev).reshape(-1)
        self._history_t.append(t_raw.copy())
        self._history_e.append(err.copy())

        # Truncate to max_vectors
        if len(self._history_t) > self.max_vectors:
            self._history_t = self._history_t[-self.max_vectors:]
            self._history_e = self._history_e[-self.max_vectors:]

        n = len(self._history_t)
        if n < 2:
            return t_raw

        # Build B matrix (overlap of error vectors) + Lagrange row/column
        B = np.zeros((n + 1, n + 1))
        for i in range(n):
            for j in range(i, n):
                B[i, j] = float(self._history_e[i] @ self._history_e[j])
                B[j, i] = B[i, j]
        B[-1, :-1] = -1.0
        B[:-1, -1] = -1.0
        B[-1, -1] = 0.0

        rhs = np.zeros(n + 1)
        rhs[-1] = -1.0

        try:
            c = np.linalg.solve(B, rhs)[:-1]
        except np.linalg.LinAlgError:
            # Singular matrix : drop oldest entry and retry next iter
            self._history_t = self._history_t[1:]
            self._history_e = self._history_e[1:]
            return t_raw

        # Linear combination of raw-update amplitudes
        t_ext = np.zeros_like(t_raw)
        for k, t_k in enumerate(self._history_t):
            t_ext = t_ext + c[k] * t_k
        return t_ext


def lccd_iterate(basis: PTMolecularBasis,
                  c: np.ndarray,
                  n_occ: int,
                  eps: np.ndarray,
                  mp2_result: MP2Result | None = None,
                  *,
                  max_iter: int = 30,
                  tol: float = 1.0e-7,
                  damp: float = 0.0,
                  use_diis: bool = True,
                  diis_start: int = 2,
                  diis_max_vectors: int = 8,
                  include_pp_ladder: bool = True,
                  include_hh_ladder: bool = True,
                  include_ring: bool = True,
                  include_quadratic: bool = False,
                  verbose: bool = False,
                  **grid_kwargs) -> LCCDResult:
    """Iterate T2 amplitudes to LCCD convergence on canonical HF orbitals.

    Parameters
    ----------
    basis, c, n_occ, eps : standard PT-LCAO post-SCF inputs.
    mp2_result : optional pre-computed MP2 driver output (skip recomputation).
    max_iter, tol : convergence controls. ``tol`` applies to the L²
        norm of the residual ``|R(t)/D − t|``, in units of t.
    damp : optional simple amplitude damping in [0, 1).  ``t_new = (1−damp)
        × t_update + damp × t_old``.  Useful when the linear system is
        near-singular (rich virtual spaces).  Default 0 (no damping).
    use_diis : enable Pulay DIIS extrapolation (default True). On
        near-singular linear systems (small basis, rich virtuals)
        DIIS reduces N₂/SZ convergence from > 80 iterations of pure
        Jacobi to ~ 10 iterations.
    diis_start : iteration index from which DIIS extrapolation kicks
        in (the first ``diis_start - 1`` iterations are pure Jacobi
        with damping). Default 2.
    diis_max_vectors : DIIS history length. Default 8 (standard).
    include_pp_ladder, include_hh_ladder, include_ring : toggle the
        linear residual contributions; the defaults (all True)
        reproduce the full LCCD residual (Phase 6.B.10/11a/b).
    include_quadratic : when True, add the closed-shell CCD quadratic
        T2² contribution via the F_oo / F_vv Fock-like intermediates
        (Phase 6.B.11c).  Default False keeps backward-compatible LCCD
        behaviour; the public ``ccd_iterate`` thin wrapper flips this
        to True.
    grid_kwargs : forwarded to mp2_at_hf and mo_eri_block.

    Returns
    -------
    LCCDResult with the converged amplitudes, energy, and 1-RDM blocks.
    The amplitudes can be substituted for ``mp2_result.t`` in any
    Stanton-Gauss-style downstream pipeline (with the Phase 6.B.10b
    caveat that the MP2-derived Lagrangian formula is not consistent
    with non-MP2 amplitudes).
    """
    if mp2_result is None:
        mp2_result = mp2_at_hf(basis, eps, c, n_occ, **grid_kwargs)

    c_occ = c[:, :n_occ]
    c_vir = c[:, n_occ:]

    # Build ERI blocks ONCE — they are independent of t.
    if include_pp_ladder:
        eri_abcd = mo_eri_block(
            basis, c_vir, c_vir, c_vir, c_vir, **grid_kwargs,
        )
    else:
        eri_abcd = None
    if include_hh_ladder:
        eri_ijkl = mo_eri_block(
            basis, c_occ, c_occ, c_occ, c_occ, **grid_kwargs,
        )
    else:
        eri_ijkl = None
    if include_ring:
        eri_jcka = mo_eri_block(
            basis, c_occ, c_vir, c_occ, c_vir, **grid_kwargs,
        )
    else:
        eri_jcka = None
    if include_quadratic:
        # (kl|cd) — occ/occ/vir/vir block needed for F_oo, F_vv intermediates
        eri_klcd = mo_eri_block(
            basis, c_occ, c_occ, c_vir, c_vir, **grid_kwargs,
        )
    else:
        eri_klcd = None

    eri_iajb = mp2_result.eri_mo_iajb

    # Orbital-energy denominator
    eps_i = eps[:n_occ]
    eps_a = eps[n_occ:]
    delta = (eps_a[None, :, None, None]
             + eps_a[None, None, None, :]
             - eps_i[:, None, None, None]
             - eps_i[None, None, :, None])

    # Initial guess: MP2 amplitudes
    t = mp2_result.t.copy()
    history_e = [_energy_from_amplitudes(t, eri_iajb)]
    diis = _DIIS(max_vectors=diis_max_vectors) if use_diis else None

    if verbose:
        diis_label = f", DIIS (start={diis_start}, vec={diis_max_vectors})" \
                     if use_diis else ", no DIIS"
        print(f"  LCCD iter  0  E = {history_e[0]:+.6e} eV  (MP2 init"
              f"{diis_label})")

    for it in range(1, max_iter + 1):
        # Linear residual: R(t) = (ia|jb) + ladder/ring contributions
        residual = eri_iajb.copy()
        if include_pp_ladder:
            residual = residual + _mp3_pp_ladder(t, eri_abcd)
        if include_hh_ladder:
            residual = residual + _mp3_hh_ladder(t, eri_ijkl)
        if include_ring:
            residual = residual + _mp3_ring(t, eri_jcka)
        if include_quadratic:
            residual = residual + _ccd_quadratic_F_intermediates(t, eri_klcd)

        t_raw = residual / delta
        if damp > 0.0:
            t_raw = (1.0 - damp) * t_raw + damp * t

        if use_diis and it >= diis_start:
            t_new = diis.extrapolate(t_raw, t)
        else:
            t_new = t_raw

        diff = float(np.linalg.norm(t_new - t)) / max(np.linalg.norm(t), 1e-30)
        t = t_new

        e_iter = _energy_from_amplitudes(t, eri_iajb)
        history_e.append(e_iter)

        if verbose:
            de = e_iter - history_e[-2]
            tag = "DIIS" if (use_diis and it >= diis_start) else "Jac "
            print(f"  LCCD iter {it:2d} [{tag}] E = {e_iter:+.6e} eV  "
                  f"ΔE = {de:+.2e}  ||δt||/||t|| = {diff:.2e}")

        if diff < tol:
            break
    else:
        if verbose:
            print(f"  LCCD did not converge after {max_iter} iterations "
                  f"(||δt||/||t|| = {diff:.2e}, tol = {tol:.0e})")

    d_occ, d_vir = mp2_density_correction(t)
    return LCCDResult(
        t=t,
        eri_mo_iajb=eri_iajb,
        e_corr=_energy_from_amplitudes(t, eri_iajb),
        d_occ=d_occ,
        d_vir=d_vir,
        n_iter=len(history_e) - 1,
        history_e=history_e,
    )


# ─────────────────────────────────────────────────────────────────────
# CCD wrapper — full quadratic closed-shell CCD (Phase 6.B.11c)
# ─────────────────────────────────────────────────────────────────────


def ccd_iterate(basis: PTMolecularBasis,
                  c: np.ndarray,
                  n_occ: int,
                  eps: np.ndarray,
                  mp2_result: MP2Result | None = None,
                  **kwargs) -> LCCDResult:
    """Closed-shell CCD iteration.

    Thin wrapper over :func:`lccd_iterate` that defaults to
    ``include_quadratic=True``, adding the closed-shell T2² Fock-like
    contribution

        R_quad[i,a,j,b] = - Σ_k F_oo[k,i] t_{kj}^{ab}
                          + Σ_c F_vv[a,c] t_{ij}^{cb}
                          + P(ij,ab) image,

    where F_oo / F_vv are the T2-renormalized Fock-like intermediates
    (see :func:`_ccd_F_oo_intermediate`). Together with the linear
    pp + hh + ring residual terms (Phase 6.B.10) and DIIS extrapolation
    (Phase 6.B.11b), this delivers full closed-shell CCD on canonical
    HF orbitals (no T1 amplitudes — Phase 6.B.11d adds CCSD).

    Returns the same `LCCDResult` dataclass; ``e_corr`` then carries
    the CCD correlation energy (typically more negative than LCCD by
    a few % to ~10 % on covalent systems).
    """
    kwargs.setdefault("include_quadratic", True)
    return lccd_iterate(
        basis, c, n_occ, eps, mp2_result=mp2_result, **kwargs,
    )
