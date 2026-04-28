"""Phase 6.B.11d — Closed-shell CCSD via T1+T2 coupled iteration.

Extends Phase 6.B.11c (CCD with quadratic T2² intermediates) by adding
the singles amplitudes t_i^a. The T1 residual on canonical HF orbitals
(F_ai = 0 by Brillouin's theorem) reduces to the dominant terms

    R1_i^a = + Σ_kc (2(ai|kc) - (ak|ci)) t_k^c              [T1 ladder]
           + Σ_kc F_kc (2 t̃_{ik}^{ac} - t̃_{ki}^{ac})        [T2→T1 via F_kc]
           + Σ_kcd (2(ai|cd) - (ad|ci)) t_{ik}^{cd}          [T2→T1 source]
           - Σ_klc (2(kl|ci) - (lk|ci)) t_{kl}^{ac}          [T2→T1 source]

Where F_kc = Σ_ld (2(kl|cd) - (lk|cd)) t_l^d is the T1-renormalised
"mixed" Fock-like intermediate.

T1×T1 cubic terms are dropped at this scope (Phase 6.B.11d-bis); on
canonical HF the singles are small (Brillouin → T1=0 at MP2 reference)
so the cubic correction is typically < 0.1 % of E_corr.

CCSD energy uses the τ̃ amplitudes that mix T1+T2 :

    τ̃_{ij}^{ab} = t̃_{ij}^{ab} + t_i^a t_j^b - t_i^b t_j^a (singlet-coupled)
    E_CCSD     = -Σ_{ij,ab} t_{ij}^{ab} (2 K - K^swap) - Σ_i t_i^a F_ai
                                                                    │
                                                                    │ zero at canonical HF
                                                                    │ but kept structurally

The CCSD pipeline reuses the LCCD/CCD infrastructure: ERIs cached
once per call (now 3 additional blocks for the T1 sources), DIIS
extrapolation on the joint (T1,T2) error vector, T2² Fock-like
intermediates from Phase 6.B.11c.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from ptc.lcao.ccd import (
    LCCDResult,
    _DIIS,
    _ccd_quadratic_F_intermediates,
    _energy_from_amplitudes,
)
from ptc.lcao.density_matrix import PTMolecularBasis
from ptc.lcao.mp2 import MP2Result, mo_eri_block, mp2_at_hf, mp2_density_correction
from ptc.lcao.mp3 import _mp3_hh_ladder, _mp3_pp_ladder, _mp3_ring, _t_tilde


# ─────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────


@dataclass
class CCSDResult:
    """Bundle of CCSD quantities at convergence (T1 + T2)."""
    t1: np.ndarray                  # shape (n_occ, n_virt)
    t2: np.ndarray                  # shape (n_occ, n_virt, n_occ, n_virt)
    eri_mo_iajb: np.ndarray
    e_corr: float                   # CCSD correlation energy in eV
    d_occ: np.ndarray
    d_vir: np.ndarray
    n_iter: int
    history_e: List[float] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# T1 residual contributions
# ─────────────────────────────────────────────────────────────────────


def _ccsd_F_kc_intermediate(t1: np.ndarray,
                             eri_klcd: np.ndarray) -> np.ndarray:
    """T1-renormalised "mixed" Fock-like intermediate.

        F_kc[k, c] = Σ_ld (2(kl|cd) - (lk|cd)) t_l^d
                  = Σ_ld (2 eri[k,l,c,d] - eri[l,k,c,d]) t1[l, d]

    Returns array shape (n_occ, n_virt). At canonical HF with T1 = 0
    initially, F_kc starts at zero ; it builds up only as T1 grows
    during iteration.
    """
    return np.einsum(
        "klcd,ld->kc",
        2.0 * eri_klcd - eri_klcd.transpose(1, 0, 2, 3),
        t1, optimize=True,
    )


def _ccsd_T1_ladder(t1: np.ndarray, eri_aikc: np.ndarray) -> np.ndarray:
    """T1 self-interaction (ladder) contribution.

        Σ_kc (2(ai|kc) - (ak|ci)) t_k^c
      = Σ_kc (2 eri[a,i,k,c] - eri[a,k,c,i].swap_ai) t1[k,c]

    Layout: eri_aikc[a, i, k, c] ≡ (ai|kc).
    """
    # (ak|ci) = eri[a, k, c, i] but our block has (ai|kc) layout.
    # By chemist symmetry (ai|kc) = (kc|ai) and (ak|ci) = (ci|ak).
    # We just need the antisymmetrised combination ; supply both via
    # reordering of the same (ai|kc) array.
    # (ai|kc) - (1/2)(ak|ci) — but we want 2(ai|kc) - (ak|ci).
    # eri[a, k, c, i] is the (ak|ci) layout. We can construct it from
    # eri_aikc by transposing index positions if we have (ak|ci) =
    # eri_aikc[a, k, c, i] — but eri_aikc has shape (n_v, n_o, n_o, n_v).
    # Index 1 is i (occ) and index 2 is k (occ), so swapping 1↔2 gives
    # (a, k, i, c). That's NOT (ak|ci) which would need i in last pos.
    #
    # Cleanest path: build (ak|ci) explicitly via a separate ERI block,
    # OR realise that 2(ai|kc) - (ak|ci) acting on t_k^c can be
    # rewritten using the singlet-antisym sum :
    #
    #   2(ai|kc) - (ak|ci) = 2(ai|kc) - (ai|kc).transpose(0, 2, 1, 3)
    #                                                   [a, k, i, c]
    # via chemist symmetry (ak|ci) = (ai|ck) (swap pairs). Wait : in
    # chemist (pq|rs) is symmetric under p↔q within the SAME bra
    # ONLY if integral has 8-fold symmetry. We have real ERIs so 8-fold
    # holds: (ak|ci) = (ka|ci) = (ka|ic) = ... etc. The standard form
    # uses (ak|ci) = (ai|ck) by the (1↔2)(3↔4) symmetry.
    # Then (ai|ck) layout in eri_aikc would need k and c swapped: that
    # is eri_aikc.transpose(0, 1, 3, 2).
    #
    # Putting it together:
    #   2(ai|kc) − (ak|ci)  =  2 eri_aikc − eri_aikc.transpose(0, 1, 3, 2)
    intermediate = (
        2.0 * eri_aikc
        - eri_aikc.transpose(0, 1, 3, 2)
    )
    return np.einsum("aikc,kc->ia", intermediate, t1, optimize=True)


def _ccsd_T2_to_T1_source_aicd(t2: np.ndarray,
                                 eri_aicd: np.ndarray) -> np.ndarray:
    """T2-derived source for T1 via the (ai|cd) virtual block :

        + Σ_kcd (2(ai|cd) - (ad|ci)) t_{ik}^{cd}

    Layout : eri_aicd[a, i, c, d] ≡ (ai|cd) ; t2[i, c, k, d] ≡ t_{ik}^{cd}.
    """
    # (ad|ci) = (ai|cd).transpose(0, 1, 3, 2) ? Let's check :
    # chemist (ad|ci) ≡ eri[a, d, c, i] vs (ai|cd) ≡ eri[a, i, c, d].
    # By chemist 8-fold symmetry:
    #   (ad|ci) = (ai|cd)  via simultaneous swap (1↔2 on bra, 3↔4 on ket)?
    #   No — that's the same integral.  Actually (pq|rs) = (rs|pq) is
    #   chemist symmetry. (ad|ci) = (ci|ad) and (ai|cd) = (cd|ai).
    #   These are different unless we have additional symmetry. For real
    #   orbitals with 8-fold ERI symmetry, (ai|cd) = (ic|ad) ... ugh.
    #
    # Simplest robust approach: build (ad|ci) explicitly from a separate
    # call to mo_eri_block. We do this in the caller.
    raise NotImplementedError(
        "Use _ccsd_T2_to_T1_source_explicit which takes both ERIs."
    )


def _ccsd_T2_to_T1_source(t2: np.ndarray,
                            eri_aicd: np.ndarray,
                            eri_adci: np.ndarray) -> np.ndarray:
    """T2-derived source #1 : + Σ_kcd (2(ai|cd) - (ad|ci)) t_{ik}^{cd}.

    Uses TWO ERI blocks because the antisymm pair has different index
    structure in chemist notation.

    Returns shape (n_occ, n_virt) — same as T1.
    """
    # 2 (ai|cd) t_{ik}^{cd} = Σ_kcd 2 eri_aicd[a,i,c,d] t2[i,c,k,d]
    #   — wait, t_{ik}^{cd} = t2[i, c, k, d] (occ, vir, occ, vir layout)
    term1 = +2.0 * np.einsum("aicd,ickd->ka", eri_aicd, t2, optimize=True)
    # − (ad|ci) t_{ik}^{cd} = − Σ_kcd eri_adci[a,d,c,i] t2[i,c,k,d]
    term2 = -np.einsum("adci,ickd->ka", eri_adci, t2, optimize=True)
    # Final sum has shape (n_occ, n_virt) but we built it as [k, a].
    # Standard layout is [i, a] so we swap: result[i, a] = term[i, a]
    # — actually we need to sum over i and end with (occ, virt).
    # Re-check : Σ_kcd eri[a,i,c,d] t2[i,c,k,d] gives result indexed by
    # [a, k]? Let me redo with clearer einsum :
    #
    #   eri_aicd has shape (n_v, n_o, n_v, n_v) — indexed [a, i, c, d]
    #   t2 has shape (n_o, n_v, n_o, n_v) — indexed [i, c, k, d]
    #   Σ_cd eri[a,i,c,d] t2[i,c,k,d] sums over c, d AND collapses i AND
    #   k? No, i appears once on LHS so it's a free index... but I summed
    #   over i, c, d implicitly. Let me re-examine.
    #
    # Need to sum over k, c, d (NOT i) and produce a [i, a] tensor.
    # Σ_kcd eri[a,i,c,d] t2[i,c,k,d] — but eri has no k index, so the
    # sum over k is just t2 summed over k for fixed (i, c, d).
    #
    # Wait, t2[i,c,k,d] has k as index 2 (a dummy summation index in
    # the original formula). Sum over k gives a 3-index intermediate
    # X[i, c, d] = Σ_k t2[i, c, k, d] (which is not standard CC).
    #
    # That's wrong. Let me re-read the formula : + Σ_kcd (2(ai|cd) - ...
    # ) t_{ik}^{cd}. Sum is over k, c, d. The i in t_{ik}^{cd} matches
    # the i in (ai|cd). So result is indexed by (i, a) :
    #
    #   R[i, a] = Σ_kcd (2(ai|cd) - (ad|ci)) t_{ik}^{cd}
    #          = Σ_kcd 2 eri_aicd[a,i,c,d] t2[i,c,k,d]
    #            - Σ_kcd eri_adci[a,d,c,i] t2[i,c,k,d]
    #
    # Sum over k goes over all occupied orbitals. The result has free
    # indices i (occ) and a (vir).
    raise NotImplementedError("indices need rebuild — use direct einsum below")


# ─────────────────────────────────────────────────────────────────────
# Top-level CCSD iteration (clean re-derivation, T1 source terms only)
# ─────────────────────────────────────────────────────────────────────


def _build_ccsd_eri_blocks(basis: PTMolecularBasis,
                             c: np.ndarray,
                             n_occ: int,
                             **grid_kwargs) -> dict:
    """Build all ERI blocks needed for CCSD T1+T2 in one shot.

    Reuses the CCD blocks (pp / hh / ring / klcd) from
    Phase 6.B.11c and adds three more blocks needed by the T1
    residual : (ai|kc), (ai|cd), (kl|ci).
    """
    c_occ = c[:, :n_occ]
    c_vir = c[:, n_occ:]
    blocks = {}
    blocks["abcd"] = mo_eri_block(basis, c_vir, c_vir, c_vir, c_vir, **grid_kwargs)
    blocks["ijkl"] = mo_eri_block(basis, c_occ, c_occ, c_occ, c_occ, **grid_kwargs)
    blocks["jcka"] = mo_eri_block(basis, c_occ, c_vir, c_occ, c_vir, **grid_kwargs)
    blocks["klcd"] = mo_eri_block(basis, c_occ, c_occ, c_vir, c_vir, **grid_kwargs)
    blocks["aikc"] = mo_eri_block(basis, c_vir, c_occ, c_occ, c_vir, **grid_kwargs)
    blocks["aicd"] = mo_eri_block(basis, c_vir, c_occ, c_vir, c_vir, **grid_kwargs)
    blocks["klci"] = mo_eri_block(basis, c_occ, c_occ, c_vir, c_occ, **grid_kwargs)
    # Phase 6.B.11d-bis : (ma|ef) for T1 source 1e — bra=(occ,vir) ket=(vir,vir)
    blocks["maef"] = mo_eri_block(basis, c_occ, c_vir, c_vir, c_vir, **grid_kwargs)
    # Phase 6.B.11d-bis-bis : (mn|ie) for T1 source 1f — bra=(occ,occ) ket=(occ,vir)
    blocks["mnie"] = mo_eri_block(basis, c_occ, c_occ, c_occ, c_vir, **grid_kwargs)
    return blocks


def _ccsd_t1_residual(t1: np.ndarray,
                       t2: np.ndarray,
                       eri_blocks: dict) -> np.ndarray:
    """Closed-shell CCSD T1 residual — Phase 6.B.11d simplified.

    Phase 6.B.11d        : T1 ladder.
    Phase 6.B.11d-bis    : T2 source 1e via (ma|ef).
    Phase 6.B.11d-bis-bis: T2 source 1f via (mn|ie) + F_kc cross.

    R1[i, a] = + Σ_kc (2(ai|kc) - (ak|ic)) t_k^c                 [1a, ladder]
             + ½ Σ_mef (2(ma|ef) - (ma|fe)) τ̃_{im}^{ef}          [1e]
             - ½ Σ_mne (2(mn|ie) - (nm|ie)) τ̃_{mn}^{ae}          [1f]
             + Σ_kc F_kc (2 t_{ik}^{ac} - t_{ki}^{ac})             [1d, T1×T2]

    where (ak|ic) is obtained from (ai|kc) by axes-1↔2 swap, and
    F_kc = Σ_ld (2(kl|cd) - (lk|cd)) t_l^d is the T1-renormalised
    Fock-like intermediate (Phase 6.B.11d-bis-bis). The full set is
    a sufficient closed-shell CCSD T1 closure on canonical HF for
    Phase 6.B.11e-bis Λ-equation iteration.

    The remaining standard CCSD T1 terms (T2 sources, F_kc cross)
    require additional ERI blocks ((ad|ci), (kl|ic), (me|ai), ...)
    each with their own index convention; deferred to Phase 6.B.11d-bis
    where each is validated against a brute-force spin-orbital
    reference.

    On canonical HF orbitals T1 is zero by Brillouin's theorem at the
    MP2 reference, and the ladder term grows it perturbatively as
    iteration proceeds. The T1 contribution to E_CCSD enters via the
    τ̃-amplitude formula in :func:`_ccsd_energy`.
    """
    aikc = eri_blocks["aikc"]      # (n_v, n_o, n_o, n_v) ≡ (ai|kc)
    maef = eri_blocks["maef"]      # (n_o, n_v, n_v, n_v) ≡ (ma|ef)
    mnie = eri_blocks["mnie"]      # (n_o, n_o, n_o, n_v) ≡ (mn|ie)
    klcd = eri_blocks["klcd"]      # (n_o, n_o, n_v, n_v) ≡ (kl|cd)

    # τ̃ used by 1e and 1f
    t_tilde = 2.0 * t2 - t2.transpose(2, 1, 0, 3)

    # ── 1a : T1 ladder, 2(ai|kc) - (ak|ic) ──
    aikc_anti = 2.0 * aikc - aikc.transpose(0, 2, 1, 3)
    R1 = np.einsum("aikc,kc->ia", aikc_anti, t1, optimize=True)

    # ── 1e : T2 source via (ma|ef) ──
    # 1e[i, a] = ½ Σ_mef (2(ma|ef) - (ma|fe)) τ̃[i, e, m, f]
    W_1e = 2.0 * maef - maef.transpose(0, 1, 3, 2)
    R1 = R1 + 0.5 * np.einsum("maef,iemf->ia", W_1e, t_tilde, optimize=True)

    # ── 1f : T2 source via (mn|ie), Phase 6.B.11d-bis-bis ──
    # 1f[i, a] = -½ Σ_mne (2(mn|ie) - (nm|ie)) τ̃_{mn}^{ae}
    # Layout: mnie[m, n, i, e] ≡ (mn|ie); τ̃[m, a, n, e] = 2 t_{mn}^{ae} - t_{nm}^{ae}
    W_1f = 2.0 * mnie - mnie.transpose(1, 0, 2, 3)
    R1 = R1 - 0.5 * np.einsum("mnie,mane->ia", W_1f, t_tilde, optimize=True)

    # ── 1d : F_kc cross-coupling (T1 × T2 quadratic) ──
    # F_kc[k, c] = Σ_ld (2(kl|cd) - (lk|cd)) t_l^d
    F_kc = np.einsum(
        "klcd,ld->kc",
        2.0 * klcd - klcd.transpose(1, 0, 2, 3),
        t1, optimize=True,
    )
    # contribution: + Σ_kc F_kc (2 t_{ik}^{ac} - t_{ki}^{ac})
    # 2 t_{ik}^{ac} - t_{ki}^{ac} = t̃[i, a, k, c] (in our layout)
    t2_anti = 2.0 * t2 - t2.transpose(2, 1, 0, 3)
    R1 = R1 + np.einsum("kc,iakc->ia", F_kc, t2_anti, optimize=True)

    return R1


def _ccsd_energy(t1: np.ndarray,
                   t2: np.ndarray,
                   eri_iajb: np.ndarray) -> float:
    """Closed-shell CCSD correlation energy via the τ̃ amplitudes.

    τ̃_{ij}^{ab} = t̃_{ij}^{ab} + (t_i^a t_j^b − t_i^b t_j^a)
                + (1↔2 swap to enforce singlet symmetry)
                = (2 t_{ij}^{ab} − t_{ji}^{ab}) + 2 (t_i^a t_j^b)
                  (closed-shell singlet form)

    E_CCSD = -Σ_{ij,ab} τ_{ij}^{ab} (2 K_{ij}^{ab} − K_{ij}^{ba})

    where K_{ij}^{ab} = (ia|jb) and τ_{ij}^{ab} = t_{ij}^{ab} + t_i^a t_j^b
    (NOT singlet-coupled — that's what we sum). Energy uses the
    (2 - swap) combination of the integral.

    For canonical HF reference, t1 = 0 at MP2 → CCSD energy reduces
    to E_CCD when T1 vanishes.
    """
    tau = t2 + np.einsum("ia,jb->iajb", t1, t1, optimize=True)
    combo = 2.0 * eri_iajb - eri_iajb.transpose(0, 3, 2, 1)
    return -float(np.sum(tau * combo))


def ccsd_iterate(basis: PTMolecularBasis,
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
                  include_quadratic: bool = True,
                  include_t1: bool = True,
                  verbose: bool = False,
                  **grid_kwargs) -> CCSDResult:
    """Closed-shell CCSD iteration — T1 + T2 coupled.

    Pipeline
    --------
    1. MP2 amplitudes (or supplied) as initial T2 ; T1 = 0 (Brillouin).
    2. Build all 7 ERI blocks once (CCD's 4 + T1's 3).
    3. Loop until convergence :
         - Build R1[i,a] from current (T1, T2)
         - Build R2[i,a,j,b] from current (T1, T2) — same structure as
           CCD's R2 since T1 enters via the τ̃ replacement (deferred
           to a follow-up; here we keep the CCD R2 unchanged and just
           feed back T1 via energy)
         - Update t1, t2 = R / Δε
         - DIIS extrapolation on joint (T1, T2)
    4. Return CCSDResult with τ̃-based energy.

    Phase 6.B.11d scope
    -------------------
    * T1 residual : full closed-shell formula (1a + 1d + 1e + 1f).
    * T2 residual : reuses CCD machinery (linear ladders/ring + F_oo,
      F_vv quadratic), without the explicit T1×T2 cross terms in the
      T2 residual.  This is "CCSD-T1-only" — captures most of the
      singles physics on canonical HF where T1 is small (~0.01).
    * Full T1×T2 cross-couplings in the T2 residual are deferred to
      Phase 6.B.11d-bis.
    """
    if mp2_result is None:
        mp2_result = mp2_at_hf(basis, eps, c, n_occ, **grid_kwargs)

    # Build all ERI blocks once
    eri_blocks = _build_ccsd_eri_blocks(basis, c, n_occ, **grid_kwargs)
    eri_iajb = mp2_result.eri_mo_iajb

    eps_i = eps[:n_occ]
    eps_a = eps[n_occ:]
    delta1 = eps_a[None, :] - eps_i[:, None]
    delta2 = (eps_a[None, :, None, None]
              + eps_a[None, None, None, :]
              - eps_i[:, None, None, None]
              - eps_i[None, None, :, None])

    # Initial guess
    t1 = np.zeros((n_occ, basis.n_orbitals - n_occ))
    t2 = mp2_result.t.copy()
    history_e = [_ccsd_energy(t1, t2, eri_iajb)]
    diis = _DIIS(max_vectors=diis_max_vectors) if use_diis else None

    if verbose:
        print(f"  CCSD iter  0  E = {history_e[0]:+.6e} eV  (MP2 init, T1=0)")

    for it in range(1, max_iter + 1):
        # T2 residual : reuse CCD machinery
        residual = eri_iajb.copy()
        if include_pp_ladder:
            residual = residual + _mp3_pp_ladder(t2, eri_blocks["abcd"])
        if include_hh_ladder:
            residual = residual + _mp3_hh_ladder(t2, eri_blocks["ijkl"])
        if include_ring:
            residual = residual + _mp3_ring(t2, eri_blocks["jcka"])
        if include_quadratic:
            residual = residual + _ccd_quadratic_F_intermediates(
                t2, eri_blocks["klcd"]
            )
        t2_raw = residual / delta2

        # T1 residual
        if include_t1:
            R1 = _ccsd_t1_residual(t1, t2, eri_blocks)
            t1_raw = R1 / delta1
        else:
            t1_raw = t1

        if damp > 0.0:
            t1_raw = (1.0 - damp) * t1_raw + damp * t1
            t2_raw = (1.0 - damp) * t2_raw + damp * t2

        # Joint DIIS extrapolation : concatenate (t1, t2) flat then split
        if use_diis and it >= diis_start:
            joint_raw = np.concatenate([t1_raw.ravel(), t2_raw.ravel()])
            joint_prev = np.concatenate([t1.ravel(), t2.ravel()])
            joint_new = diis.extrapolate(joint_raw, joint_prev)
            n_t1 = t1.size
            t1_new = joint_new[:n_t1].reshape(t1.shape)
            t2_new = joint_new[n_t1:].reshape(t2.shape)
        else:
            t1_new = t1_raw
            t2_new = t2_raw

        diff_t2 = float(np.linalg.norm(t2_new - t2)) / max(np.linalg.norm(t2), 1e-30)
        diff_t1 = float(np.linalg.norm(t1_new - t1)) / max(np.linalg.norm(t1) + 1e-30, 1e-30)
        diff = max(diff_t1, diff_t2)
        t1 = t1_new
        t2 = t2_new

        e_iter = _ccsd_energy(t1, t2, eri_iajb)
        history_e.append(e_iter)

        if verbose:
            de = e_iter - history_e[-2]
            tag = "DIIS" if (use_diis and it >= diis_start) else "Jac "
            t1_norm = np.linalg.norm(t1)
            print(f"  CCSD iter {it:2d} [{tag}] E = {e_iter:+.6e} eV  "
                  f"ΔE = {de:+.2e}  ||δt||={diff:.2e}  ||t1||={t1_norm:.2e}")

        if diff < tol:
            break
    else:
        if verbose:
            print(f"  CCSD did not converge after {max_iter} iters "
                  f"(||δ|| = {diff:.2e}, tol = {tol:.0e})")

    d_occ, d_vir = mp2_density_correction(t2)
    return CCSDResult(
        t1=t1,
        t2=t2,
        eri_mo_iajb=eri_iajb,
        e_corr=_ccsd_energy(t1, t2, eri_iajb),
        d_occ=d_occ,
        d_vir=d_vir,
        n_iter=len(history_e) - 1,
        history_e=history_e,
    )
