"""Phase 6.B.11e/e-bis/e-bis-bis — CCSD Λ-equations.

The Λ-equations are the LEFT eigenvalue equation of the similarity-
transformed Hamiltonian H̄ = e^{-T} H e^T :

    ⟨0| (1 + Λ) (H̄ - E_CC) = 0

For closed-shell CCSD, Λ has singles λ_i^a and doubles λ_{ij}^{ab}
multipliers, with the same shapes as T1 and T2 amplitudes.  Once
Λ is converged, the property gradient (e.g. paramagnetic shielding
σ_p^CCSD-Λ-GIAO) is computed as

    P^pert = ⟨0| (1 + Λ) [H̄, X^pert] |0⟩

where X^pert is the perturbation operator (angular momentum, dipole,
etc.).  This is Phase 6.B.11f — beyond this scope.

Phase 6.B.11e scope (this commit)
  * `LambdaResult` dataclass mirroring `CCSDResult` for symmetry
  * `lambda_initialize(ccsd_result)` returning the
    leading-order multipliers λ1=0, λ2=T2 (the so-called "ZAO"
    approximation, equivalent to the variational CC limit where
    Λ = T).
  * `lambda_iterate(...)` SKELETON that exposes the iteration API
    (max_iter, tol, DIIS, ...) but currently returns the
    initialization unchanged — the full closed-shell Λ-equation
    residuals are deferred to Phase 6.B.11e-bis.
  * Validation tests verifying the structural correctness of the
    skeleton: dataclass shapes, fall-back to ZAO, basic invariants.

What 6.B.11e is NOT (yet)
  * The full Λ residuals R_{λ}^{1}, R_{λ}^{2} (with all CCSD-Λ
    contractions and T1-renormalised intermediates).  These are
    Phase 6.B.11e-bis ; the skeleton documents the API they
    will plug into.
  * The σ_p^CCSD-Λ-GIAO computation via ⟨Λ | [H̄, X] | 0⟩.
    Phase 6.B.11f.

Even at the skeleton level, the ZAO multipliers λ2 = T2 give a
non-trivial first-order property gradient that is exact for
variational CCSD-like methods (e.g. orbital-optimised CC).  This
serves as a baseline for the iterative refinement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from ptc.lcao.ccd import (
    _DIIS,
    _ccd_quadratic_F_intermediates,
)
from ptc.lcao.ccsd import CCSDResult, _build_ccsd_eri_blocks
from ptc.lcao.density_matrix import PTMolecularBasis
from ptc.lcao.mp3 import _mp3_hh_ladder, _mp3_pp_ladder, _mp3_ring


# ─────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────


@dataclass
class LambdaResult:
    """Bundle of Λ-multipliers for CCSD response theory.

    Mirrors the (t1, t2) shapes from CCSDResult so that downstream
    σ_p^CCSD-Λ-GIAO code can use Λ and T interchangeably modulo the
    sign / weight conventions of the property gradient.
    """
    lambda1: np.ndarray              # (n_occ, n_virt)
    lambda2: np.ndarray              # (n_occ, n_virt, n_occ, n_virt)
    n_iter: int
    history_norm: List[float] = field(default_factory=list)
    converged: bool = True


# ─────────────────────────────────────────────────────────────────────
# Initialisation : ZAO (Λ = T at convergence)
# ─────────────────────────────────────────────────────────────────────


def lambda_initialize(ccsd_result: CCSDResult) -> LambdaResult:
    """Leading-order Λ = T initialisation (the ZAO / variational limit).

    For variational coupled-cluster (e.g. orbital-optimised OCCSD)
    this is the EXACT answer; for canonical CCSD it is a useful
    starting guess that already gives sensible first-order property
    gradients.  Phase 6.B.11e-bis upgrades this to the iterative
    Λ-equation solution.
    """
    lambda1 = ccsd_result.t1.copy()
    lambda2 = ccsd_result.t2.copy()
    return LambdaResult(
        lambda1=lambda1,
        lambda2=lambda2,
        n_iter=0,
        history_norm=[float(np.linalg.norm(lambda2))],
        converged=True,
    )


# ─────────────────────────────────────────────────────────────────────
# Iteration skeleton — Phase 6.B.11e
# ─────────────────────────────────────────────────────────────────────


def _lambda_B_diagram_T_fixed(lambda2: np.ndarray,
                                t2_fixed: np.ndarray,
                                eri_klcd: np.ndarray) -> np.ndarray:
    """Phase 6.B.11e-bis-bis : B-diagram λ-T cross-coupling.

    Closed-shell B-diagram from canonical CCSD-Λ residual :

        ΔR_λ2[i,a,j,b] = ½ Σ_klcd (kl|cd) λ̃_{kl}^{ab} t̃_{ij}^{cd}
                       + P(ij,ab)   (i↔j, a↔b symmetrise)

    where λ̃, t̃ are the singlet-coupled amplitudes 2t - t.swap_ij.
    This term is LINEAR in λ (single λ factor) but depends on the
    converged T2 amplitudes (t_fixed), so it acts as a T-fixed
    inhomogeneity that pushes λ ≠ T at the fixed point — distinguishing
    canonical CCSD-Λ from the variational ZAO limit (Phase 6.B.11e-bis).

    The contraction order avoids the full O(n_o²·n_v²·n_o²·n_v²)
    intermediate by going through the (cd,ab) Y intermediate of
    shape (n_v, n_v, n_v, n_v).  For PTC bases this is comfortably
    in RAM.
    """
    lambda_t = 2.0 * lambda2 - lambda2.transpose(2, 1, 0, 3)
    t_t = 2.0 * t2_fixed - t2_fixed.transpose(2, 1, 0, 3)
    # Y[c,d,a,b] = Σ_kl eri[k,l,c,d] × λ̃[k,a,l,b]
    Y = np.einsum("klcd,kalb->cdab", eri_klcd, lambda_t, optimize=True)
    # contrib[i,a,j,b] = ½ Σ_cd Y[c,d,a,b] × t̃[i,c,j,d]
    contrib = 0.5 * np.einsum("cdab,icjd->iajb", Y, t_t, optimize=True)
    return contrib + contrib.transpose(2, 3, 0, 1)


def _lambda2_residual(lambda2: np.ndarray,
                        eri_blocks: dict,
                        eri_iajb: np.ndarray,
                        t2_fixed: np.ndarray | None = None) -> np.ndarray:
    """Compute the closed-shell Λ2 residual.

    Phase 6.B.11e-bis (default) : CCD residual structure on λ
        R_λ2 = (ia|jb)
             + ½ Σ_cd (ac|bd) λ̃_ij^cd          (pp ladder)
             + ½ Σ_kl (ki|lj) λ̃_kl^ab          (hh ladder)
             + ring(λ)                          (P(ij,ab) symmetric)
             + F-intermediate quadratic on λ    (Phase 6.B.11c)

    Phase 6.B.11e-bis-bis (when ``t2_fixed`` is provided) :
        + B-diagram T-fixed cross-coupling (see
          :func:`_lambda_B_diagram_T_fixed`)

    The Phase 6.B.11e-bis-bis additional term breaks the trivial
    Λ = T fixed point and gives genuine canonical CCSD-Λ multipliers.
    Full closed-shell CCSD-Λ has more T-fixed sources (W_oooo and
    W_vvvv renormalisations) — those are deferred to a follow-up.
    """
    R = eri_iajb.copy()
    R = R + _mp3_pp_ladder(lambda2, eri_blocks["abcd"])
    R = R + _mp3_hh_ladder(lambda2, eri_blocks["ijkl"])
    R = R + _mp3_ring(lambda2, eri_blocks["jcka"])
    R = R + _ccd_quadratic_F_intermediates(lambda2, eri_blocks["klcd"])
    if t2_fixed is not None:
        R = R + _lambda_B_diagram_T_fixed(
            lambda2, t2_fixed, eri_blocks["klcd"]
        )
    return R


def lambda_iterate(ccsd_result: CCSDResult,
                     basis: PTMolecularBasis,
                     c: np.ndarray,
                     n_occ: int,
                     eps: np.ndarray,
                     *,
                     max_iter: int = 30,
                     tol: float = 1.0e-7,
                     damp: float = 0.0,
                     use_diis: bool = True,
                     diis_start: int = 2,
                     diis_max_vectors: int = 8,
                     include_T_fixed: bool = False,
                     verbose: bool = False,
                     **grid_kwargs) -> LambdaResult:
    """Solve the closed-shell CCSD Λ2 equations (linearised, T fixed).

    Iterates the Λ2 multipliers using the CCD residual structure
    (Phase 6.B.11c with λ in place of T). The Λ1 multipliers stay
    at their ZAO value λ1 = t1 throughout.

    Parameters
    ----------
    include_T_fixed : Phase 6.B.11e-bis-bis flag. When True, adds
        the B-diagram λ-T cross-coupling

            ½ Σ_klcd (kl|cd) λ̃_kl^ab t̃_ij^cd + P(ij,ab)

        to the Λ2 residual.  This T-fixed inhomogeneity breaks the
        trivial Λ = T fixed point of the CCD-on-λ iteration and
        delivers a canonical-CCSD-Λ approximation.  Default False
        keeps the Phase 6.B.11e-bis variational/ZAO limit.

    The iteration starts from λ2 = T2 (ZAO seed) and uses DIIS
    extrapolation on λ2.  Convergence is typically reached in
    O(10) iterations on covalent systems thanks to the well-
    conditioned linear structure (λ-equations are linear since T
    is fixed).
    """
    eri_blocks = _build_ccsd_eri_blocks(basis, c, n_occ, **grid_kwargs)
    eri_iajb = ccsd_result.eri_mo_iajb

    eps_i = eps[:n_occ]
    eps_a = eps[n_occ:]
    delta2 = (eps_a[None, :, None, None]
              + eps_a[None, None, None, :]
              - eps_i[:, None, None, None]
              - eps_i[None, None, :, None])

    # Initialisation : ZAO seed
    lambda1 = ccsd_result.t1.copy()
    lambda2 = ccsd_result.t2.copy()
    history_norm = [float(np.linalg.norm(lambda2))]
    diis = _DIIS(max_vectors=diis_max_vectors) if use_diis else None
    converged = False

    if verbose:
        print(f"  Λ-iter 0  ||λ2|| = {history_norm[0]:.6e}  (ZAO seed)")

    t2_fixed = ccsd_result.t2 if include_T_fixed else None

    for it in range(1, max_iter + 1):
        residual = _lambda2_residual(lambda2, eri_blocks, eri_iajb, t2_fixed)
        l2_raw = residual / delta2
        if damp > 0.0:
            l2_raw = (1.0 - damp) * l2_raw + damp * lambda2

        if use_diis and it >= diis_start:
            l2_new = diis.extrapolate(l2_raw, lambda2)
        else:
            l2_new = l2_raw

        diff = float(np.linalg.norm(l2_new - lambda2)) \
            / max(np.linalg.norm(lambda2), 1e-30)
        lambda2 = l2_new
        history_norm.append(float(np.linalg.norm(lambda2)))

        if verbose:
            tag = "DIIS" if (use_diis and it >= diis_start) else "Jac "
            print(f"  Λ-iter {it:2d} [{tag}] ||λ2|| = {history_norm[-1]:.6e}  "
                  f"||δλ||/||λ|| = {diff:.2e}")

        if diff < tol:
            converged = True
            break

    return LambdaResult(
        lambda1=lambda1,
        lambda2=lambda2,
        n_iter=len(history_norm) - 1,
        history_norm=history_norm,
        converged=converged,
    )
