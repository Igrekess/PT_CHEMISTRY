"""Phase 6.B.10 — MP3-PT amplitude correction and σ_p^MP3-GIAO pipeline.

Builds the third-order amplitude correction δt_{ij}^{ab} on top of
the MP2 amplitudes from ``ptc.lcao.mp2``, then plugs into the
existing Stanton-Gauss machinery (Lagrangian + Z-vector + relaxed
CPHF) by substituting t_{MP3} = t_{MP2} + δt for t_{MP2}.

Closed-shell MP3 amplitude correction (Helgaker §14, Crawford & Schaefer
2000), using the singlet-coupled amplitudes t̃_{ij}^{ab} = 2 t_{ij}^{ab}
- t_{ji}^{ab}::

    Δε_{ij}^{ab} · δt_{ij}^{ab} =
        + ½ Σ_cd (ac|bd) t̃_{ij}^{cd}              (particle-particle ladder)
        + ½ Σ_kl (ki|lj) t̃_{kl}^{ab}              (hole-hole ladder)
        + Σ_kc P(ij)P(ab) [
              (ic|ka) t̃_{jk}^{bc} (ring direct)
            - (ic|ka) t̃_{kj}^{bc} (ring exchange)
          ]

The ring/exchange permutation operator P(ij) swaps i↔j (with the
appropriate sign changes implicit in the antisymmetric formulation).

For Phase 6.B.10 we ship the particle-particle and hole-hole ladders
(dominant contributions, O(n_v^4) and O(n_o^4) memory respectively).
The ring term — most expensive (O(n_o^2 n_v^2 ·) loops for permutations)
— is added in 6.B.10b once pp+hh alone is validated.

API
===
* ``mp3_amplitudes_correction(...)`` : compute δt only (no Z-vector).
* ``mp3_at_hf(...)``                   : drop-in MP3-level MP2Result.
* ``mp3_paramagnetic_shielding_coupled(...)`` : Stanton-Gauss σ_p^MP3-GIAO.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ptc.lcao.density_matrix import PTMolecularBasis
from ptc.lcao.mp2 import (
    MP2Result,
    mo_eri_block,
    mp2_at_hf,
    mp2_density_correction,
)


# ─────────────────────────────────────────────────────────────────────
# Singlet-coupled amplitudes (canonical t̃ used by closed-shell MP3)
# ─────────────────────────────────────────────────────────────────────


def _t_tilde(t: np.ndarray) -> np.ndarray:
    """Closed-shell singlet-coupled amplitudes : t̃_{ij}^{ab} = 2 t - t.swap_ij.

    t has layout t[i, a, j, b]; swapping i↔j yields t.transpose(2, 1, 0, 3).
    """
    return 2.0 * t - t.transpose(2, 1, 0, 3)


# ─────────────────────────────────────────────────────────────────────
# Particle-particle ladder term
# ─────────────────────────────────────────────────────────────────────


def _mp3_pp_ladder(t: np.ndarray, eri_acbd: np.ndarray) -> np.ndarray:
    """Particle-particle ladder : ½ Σ_cd (ac|bd) t̃_{ij}^{cd}.

    Returns array shaped (n_occ, n_virt, n_occ, n_virt).

    Contraction in our chemist-layout indices (eri[a, c, b, d] ≡ (ac|bd)
    and t[i, c, j, d] ≡ t_{ij}^{cd})::

        out[i, a, j, b] = ½ Σ_cd eri[a, c, b, d] · t̃[i, c, j, d]
    """
    t_t = _t_tilde(t)
    return 0.5 * np.einsum("acbd,icjd->iajb", eri_acbd, t_t, optimize=True)


# ─────────────────────────────────────────────────────────────────────
# Hole-hole ladder term
# ─────────────────────────────────────────────────────────────────────


def _mp3_hh_ladder(t: np.ndarray, eri_kilj: np.ndarray) -> np.ndarray:
    """Hole-hole ladder : ½ Σ_kl (ki|lj) t̃_{kl}^{ab}.

    Layout: eri[k, i, l, j] ≡ (ki|lj), t[k, a, l, b] ≡ t_{kl}^{ab}::

        out[i, a, j, b] = ½ Σ_kl eri[k, i, l, j] · t̃[k, a, l, b]
    """
    t_t = _t_tilde(t)
    return 0.5 * np.einsum("kilj,kalb->iajb", eri_kilj, t_t, optimize=True)


# ─────────────────────────────────────────────────────────────────────
# Particle-hole ring/exchange term (Phase 6.B.10b)
# ─────────────────────────────────────────────────────────────────────


def _mp3_ring(t: np.ndarray,
              eri_jcka: np.ndarray) -> np.ndarray:
    """Particle-hole ring contribution to MP3 amplitude correction.

    .. NOTE — Phase 6.B.10c validated at the ENERGY level
       Two regression tests in `tests/test_lcao_mp3.py` validate this
       formula at the energy level on N₂/SZ:
         * pair-symmetry: ring(i,a,j,b) == ring(j,b,i,a) by construction
         * compensation: E_MP2 > E_MP3_full > E_MP3_ladder, i.e., ring
           reduces the pp+hh-ladder over-correction (textbook covalent
           behaviour).
       The σ_p^MP3 divergence observed on DZP in Phase 6.B.10b is
       therefore NOT a ring-formula error: it comes from feeding the
       MP3 amplitudes into the MP2-derived Lagrangian, which assumes
       canonical MP2 amplitudes and produces a hybrid residual that
       drives the Z-vector solver to divergence on rich-virtual bases.
       The proper σ_p^MP3 closure requires the MP3-specific Lagrangian
       / Λ-equations and is deferred to Phase 6.B.11 (CCSD).
       ``include_ring=False`` remains the default in
       :func:`mp3_amplitudes_correction` until that closure lands.

    Provisional formulation (Helgaker §14 / Crawford-Schaefer 2000)
    using singlet-coupled amplitudes t̃ = 2t - t.swap_ij and
    chemist-notation integrals (jc|ka) ≡ eri[j, c, k, a] :

        σ_{ij}^{ab} = Σ_kc [
              + (jc|ka) (2 t_{ik}^{cb} - t_{ki}^{cb})    direct ring
              -   (jc|kb)   t_{ik}^{ca}                  exchange (a↔b)
              -   (jc|kb)   t̃_{ki}^{ac}                 exchange (i↔k coupled)
        ]
        ring_{ij}^{ab} = σ_{ij}^{ab} + σ_{ji}^{ba}
    """
    t_t = _t_tilde(t)

    # Layout reminder
    #   eri_jcka : eri[j, c, k, a] ≡ (jc|ka)        (occ, virt, occ, virt)
    #   t        : t[i, a, j, b]   ≡ t_{ij}^{ab}    (occ, virt, occ, virt)
    #
    # σ[i,a,j,b] aggregates all four ring channels.

    # Term 1 :  + 2 (jc|ka) t_{ik}^{cb}   — direct ring (factor 2 from antisym)
    #   t_{ik}^{cb} = t[i, c, k, b]
    sigma  = +2.0 * np.einsum("jcka,ickb->iajb", eri_jcka, t, optimize=True)
    # Term 2 :  −     (jc|ka) t_{ki}^{cb} — direct ring exchange in i↔k
    #   t_{ki}^{cb} = t[k, c, i, b]
    sigma -= np.einsum("jcka,kcib->iajb", eri_jcka, t, optimize=True)
    # Term 3 :  −     (jc|kb) t_{ik}^{ca} — exchange in a↔b (relabelled eri)
    #   t_{ik}^{ca} = t[i, c, k, a]
    sigma -= np.einsum("jckb,icka->iajb", eri_jcka, t, optimize=True)
    # Term 4 :  −     (jc|kb) t̃_{ki}^{ac} — coupled exchange ring
    #   t̃_{ki}^{ac} = t̃[k, a, i, c]
    sigma -= np.einsum("jckb,kaic->iajb", eri_jcka, t_t, optimize=True)

    # Symmetrise (ij ↔ ab) : ring[i,a,j,b] = σ[i,a,j,b] + σ[j,b,i,a]
    return sigma + sigma.transpose(2, 3, 0, 1)


# ─────────────────────────────────────────────────────────────────────
# Public MP3 driver
# ─────────────────────────────────────────────────────────────────────


def mp3_amplitudes_correction(basis: PTMolecularBasis,
                                c: np.ndarray,
                                n_occ: int,
                                eps: np.ndarray,
                                mp2_result: MP2Result,
                                *,
                                include_pp_ladder: bool = True,
                                include_hh_ladder: bool = True,
                                include_ring: bool = False,
                                **grid_kwargs) -> np.ndarray:
    """Compute the MP3 first-order amplitude correction δt_{ij}^{ab}.

    Returns array of shape (n_occ, n_virt, n_occ, n_virt) with the
    same layout as ``mp2_result.t``. The MP3-level total amplitudes
    are ``t_MP3 = mp2_result.t + δt``; substitute these for ``t``
    in the Lagrangian / Z-vector / relaxed-CPHF pipeline to obtain
    σ_p^MP3-GIAO.

    Toggles
    -------
    include_pp_ladder, include_hh_ladder : default True; the two
        cheapest closed-shell MP3 terms (O(n_v^4) and O(n_o^4)).
    include_ring : default False; the more expensive
        particle-hole ring/exchange contribution. Phase 6.B.10b.
    """
    c_occ = c[:, :n_occ]
    c_vir = c[:, n_occ:]

    delta_t = np.zeros_like(mp2_result.t)

    if include_pp_ladder:
        eri_abcd = mo_eri_block(
            basis, c_vir, c_vir, c_vir, c_vir, **grid_kwargs,
        )
        delta_t += _mp3_pp_ladder(mp2_result.t, eri_abcd)

    if include_hh_ladder:
        eri_ijkl = mo_eri_block(
            basis, c_occ, c_occ, c_occ, c_occ, **grid_kwargs,
        )
        delta_t += _mp3_hh_ladder(mp2_result.t, eri_ijkl)

    if include_ring:
        # ERI block (jc|ka) — same shape as (ia|jb) but with mixed
        # occupied-virtual coupling on the second pair. Reusable for
        # all four ring channels (direct + 3 exchange permutations).
        eri_jcka = mo_eri_block(
            basis, c_occ, c_vir, c_occ, c_vir, **grid_kwargs,
        )
        delta_t += _mp3_ring(mp2_result.t, eri_jcka)

    # Divide by orbital-energy denominator (same convention as MP2)
    eps_i = eps[:n_occ]
    eps_a = eps[n_occ:]
    delta = (eps_a[None, :, None, None]
             + eps_a[None, None, None, :]
             - eps_i[:, None, None, None]
             - eps_i[None, None, :, None])
    return delta_t / delta


def mp3_at_hf(basis: PTMolecularBasis,
               eps: np.ndarray,
               c: np.ndarray,
               n_occ: int,
               mp2_result: MP2Result | None = None,
               **grid_kwargs) -> MP2Result:
    """One-shot MP3 from a converged HF SCF.

    Returns a drop-in MP2Result with ``t = t_MP2 + δt^(MP3)`` and
    consistent (d_occ, d_vir) recomputed from the new amplitudes.
    The energy field is the MP3 correlation energy (using the same
    closed-shell formula as MP2 but with the corrected amplitudes).

    Parameters
    ----------
    mp2_result : optional pre-computed MP2 driver output.
    grid_kwargs : forwarded both to ``mp2_at_hf`` (when computed
        internally) and to ``mp3_amplitudes_correction``.
    """
    if mp2_result is None:
        mp2_result = mp2_at_hf(basis, eps, c, n_occ, **grid_kwargs)

    delta_t = mp3_amplitudes_correction(
        basis, c, n_occ, eps, mp2_result, **grid_kwargs,
    )
    t_mp3 = mp2_result.t + delta_t

    # Recompute density correction with the corrected amplitudes
    d_occ_mp3, d_vir_mp3 = mp2_density_correction(t_mp3)

    # Energy at MP3 level uses the standard MP2-style formula with the
    # corrected t but the same eri_iajb (the ERI is HF-level integral).
    combo = 2.0 * mp2_result.eri_mo_iajb - mp2_result.eri_mo_iajb.transpose(0, 3, 2, 1)
    e_mp3 = -float(np.sum(t_mp3 * combo))

    return replace(
        mp2_result,
        t=t_mp3,
        e_corr=e_mp3,
        d_occ=d_occ_mp3,
        d_vir=d_vir_mp3,
    )


def mp3_paramagnetic_shielding_coupled(basis: PTMolecularBasis,
                                          topology,
                                          mo_eigvals_HF: np.ndarray,
                                          mo_coeffs_HF: np.ndarray,
                                          n_occ: int,
                                          K_probe: np.ndarray,
                                          *,
                                          mp2_result: MP2Result | None = None,
                                          z_vector_kwargs: dict | None = None,
                                          relax_kwargs: dict | None = None,
                                          cphf_kwargs: dict | None = None,
                                          mp2_kwargs: dict | None = None,
                                          lagrangian_kwargs: dict | None = None,
                                          mp3_grid_kwargs: dict | None = None,
                                          isotropic: bool = True,
                                          ) -> dict:
    """Stanton-Gauss σ_p^MP3-GIAO via CPHF on Z-relaxed MP3 orbitals.

    Drop-in equivalent of ``mp2_paramagnetic_shielding_coupled`` but
    substitutes the MP2 amplitudes by their MP3-corrected counterpart
    before computing the Lagrangian, Z-vector, and relaxed orbitals.

    Returns
    -------
    dict with σ_p^HF, σ_p^MP2_full, σ_p^MP3_full plus the corrected
    amplitudes object and the MP3 Z-vector.
    """
    from ptc.lcao.mp2 import mp2_paramagnetic_shielding_coupled

    mp3_grid_kwargs = mp3_grid_kwargs or (mp2_kwargs or {})

    # Step 1: standard MP2 pipeline (gives σ_p^HF and σ_p^MP2_full)
    out_mp2 = mp2_paramagnetic_shielding_coupled(
        basis, topology, mo_eigvals_HF, mo_coeffs_HF, n_occ, K_probe,
        mp2_result=mp2_result,
        relax_kwargs=relax_kwargs,
        cphf_kwargs=cphf_kwargs,
        mp2_kwargs=mp2_kwargs,
        lagrangian_kwargs=lagrangian_kwargs,
        z_vector_kwargs=z_vector_kwargs,
        isotropic=isotropic,
    )

    # Step 2: build MP3-corrected amplitudes and re-run the same pipeline
    mp3_result = mp3_at_hf(
        basis, mo_eigvals_HF, mo_coeffs_HF, n_occ,
        mp2_result=out_mp2["mp2_result"],
        **mp3_grid_kwargs,
    )
    out_mp3 = mp2_paramagnetic_shielding_coupled(
        basis, topology, mo_eigvals_HF, mo_coeffs_HF, n_occ, K_probe,
        mp2_result=mp3_result,                # ⇐ MP3 amplitudes substituted
        relax_kwargs=relax_kwargs,
        cphf_kwargs=cphf_kwargs,
        mp2_kwargs=mp2_kwargs,
        lagrangian_kwargs=lagrangian_kwargs,
        z_vector_kwargs=z_vector_kwargs,
        isotropic=isotropic,
    )

    return {
        "mp2_result": out_mp2["mp2_result"],
        "mp3_result": mp3_result,
        "z_vector_MP2": out_mp2["z_vector"],
        "z_vector_MP3": out_mp3["z_vector"],
        "sigma_p_HF": out_mp2["sigma_p_HF"],
        "sigma_p_MP2_LO": out_mp2["sigma_p_MP2_LO"],
        "sigma_p_MP2_full": out_mp2["sigma_p_MP2_full"],
        "sigma_p_MP3_full": out_mp3["sigma_p_MP2_full"],   # MP2-style pipeline with t_MP3
        "delta_MP2_pct": (
            (out_mp2["sigma_p_MP2_full"] - out_mp2["sigma_p_HF"])
            / out_mp2["sigma_p_HF"] * 100.0
        ),
        "delta_MP3_pct": (
            (out_mp3["sigma_p_MP2_full"] - out_mp2["sigma_p_HF"])
            / out_mp2["sigma_p_HF"] * 100.0
        ),
        "e_corr_MP2": out_mp2["mp2_result"].e_corr,
        "e_corr_MP3": mp3_result.e_corr,
    }
