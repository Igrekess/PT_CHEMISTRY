"""Wavefunction NMR panel — post-HF wavefunction-method NMR pipeline.

Distinguished from DFT-based shielding methods, this tab exposes the
explicit-wavefunction post-HF cascade
    HF → MP2 → MP3 → LCCD → CCD → CCSD → Λ → σ_p^CCSD-Λ-GIAO
with the correlation-energy hierarchy and the paramagnetic shielding
at a probe point.  Designed for didactic / research use — bases are
kept small (SZ / DZ) so a session completes in seconds on a laptop CPU.

PT-pure : every contraction derived from s = 1/2, no fitted parameters.
Phases 6.B.4 → 6.B.11f.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from ptc_app.i18n import t


# ── Curated small molecules (fast end-to-end on SZ/DZ) ─────────────


_PRESETS = {
    "H₂": {
        "Z": [1, 1],
        "coords": np.array([[0.0, 0.0, 0.0], [0.7414, 0.0, 0.0]]),
        "bonds": [(0, 1, 1.0)],
        "label": "H–H (r=0.74 Å)",
    },
    "N₂": {
        "Z": [7, 7],
        "coords": np.array([[0.0, 0.0, 0.0], [1.0975, 0.0, 0.0]]),
        "bonds": [(0, 1, 3.0)],
        "label": "N≡N (r=1.10 Å)",
    },
    "CO": {
        "Z": [6, 8],
        "coords": np.array([[0.0, 0.0, 0.0], [1.128, 0.0, 0.0]]),
        "bonds": [(0, 1, 3.0)],
        "label": "C≡O (r=1.13 Å)",
    },
    "HF": {
        "Z": [1, 9],
        "coords": np.array([[0.0, 0.0, 0.0], [0.917, 0.0, 0.0]]),
        "bonds": [(0, 1, 1.0)],
        "label": "H–F (r=0.92 Å)",
    },
    "F₂": {
        "Z": [9, 9],
        "coords": np.array([[0.0, 0.0, 0.0], [1.412, 0.0, 0.0]]),
        "bonds": [(0, 1, 1.0)],
        "label": "F–F (r=1.41 Å)",
    },
}


# ── Cached cascade runner ──────────────────────────────────────────


@st.cache_data(ttl=3600, show_spinner=False)
def _run_cascade(mol_key: str, basis_type: str, n_radial: int):
    """End-to-end cascade for the chosen molecule + basis.

    Returns a flat dict of energies, T1 norm, σ_p values, and timings.
    Cached for the session because each call costs O(few seconds).
    """
    import time
    from ptc.lcao.ccd import ccd_iterate, lccd_iterate
    from ptc.lcao.ccsd import ccsd_iterate
    from ptc.lcao.ccsd_lambda import lambda_iterate
    from ptc.lcao.ccsd_property import sigma_p_ccsd_lambda_iso
    from ptc.lcao.cluster import build_explicit_cluster
    from ptc.lcao.fock import (
        density_matrix_PT_scf,
        paramagnetic_shielding_iso_coupled,
    )
    from ptc.lcao.mp2 import mp2_at_hf
    from ptc.lcao.mp3 import mp3_at_hf

    preset = _PRESETS[mol_key]
    Z, coords, bonds = preset["Z"], preset["coords"], preset["bonds"]

    grid = dict(n_radial=n_radial, n_theta=8, n_phi=10,
                use_becke=False, lebedev_order=14)

    timings = {}

    t0 = time.time()
    basis, topo = build_explicit_cluster(
        Z_list=Z, coords=coords, bonds=bonds, basis_type=basis_type,
    )
    rho, S, eigvals, c, conv, _ = density_matrix_PT_scf(
        topo, basis=basis, mode="hf", max_iter=15, tol=1e-3, **grid,
    )
    n_occ = int(round(basis.total_occ)) // 2
    timings["HF"] = time.time() - t0

    t0 = time.time()
    mp2 = mp2_at_hf(basis, eigvals, c, n_occ, **grid)
    timings["MP2"] = time.time() - t0

    t0 = time.time()
    mp3 = mp3_at_hf(basis, eigvals, c, n_occ, mp2_result=mp2,
                       include_ring=True, **grid)
    timings["MP3"] = time.time() - t0

    t0 = time.time()
    lccd = lccd_iterate(basis, c, n_occ, eigvals, mp2_result=mp2,
                          max_iter=30, tol=1e-7, use_diis=True, **grid)
    timings["LCCD"] = time.time() - t0

    t0 = time.time()
    ccd = ccd_iterate(basis, c, n_occ, eigvals, mp2_result=mp2,
                        max_iter=30, tol=1e-7, **grid)
    timings["CCD"] = time.time() - t0

    t0 = time.time()
    ccsd = ccsd_iterate(basis, c, n_occ, eigvals, mp2_result=mp2,
                          max_iter=30, tol=1e-7, **grid)
    timings["CCSD"] = time.time() - t0

    t0 = time.time()
    lam = lambda_iterate(ccsd, basis, c, n_occ, eigvals,
                            max_iter=30, tol=1e-7,
                            include_T_fixed=True, **grid)
    timings["Λ"] = time.time() - t0

    # σ_p at probe just off the first atom (avoid 1/r³ singularity)
    probe = coords[0] + np.array([0.0, 0.0, 0.1])
    n_e = 2 * n_occ
    iso_grid = {k: v for k, v in grid.items()
                if k not in ("use_becke", "lebedev_order")}
    t0 = time.time()
    sigma_HF = float(paramagnetic_shielding_iso_coupled(
        basis, eigvals, c, n_e, probe, **iso_grid,
    ))
    timings["σ_p^HF"] = time.time() - t0

    t0 = time.time()
    out = sigma_p_ccsd_lambda_iso(
        basis, topo, ccsd, lam, c, n_occ, probe,
        mode="symmetric",
        cphf_kwargs=iso_grid,
        **grid,
    )
    timings["σ_p^CCSD-Λ"] = time.time() - t0

    return {
        "n_orb": basis.n_orbitals,
        "n_occ": n_occ,
        "n_virt": basis.n_orbitals - n_occ,
        "scf_converged": bool(conv),
        "energies": {
            "MP2": mp2.e_corr,
            "MP3": mp3.e_corr,
            "LCCD": lccd.e_corr,
            "CCD": ccd.e_corr,
            "CCSD": ccsd.e_corr,
        },
        "lccd_iters": lccd.n_iter,
        "ccd_iters": ccd.n_iter,
        "ccsd_iters": ccsd.n_iter,
        "ccsd_t1_norm": float(np.linalg.norm(ccsd.t1)),
        "lambda_iters": lam.n_iter,
        "lambda_T_drift": float(np.linalg.norm(lam.lambda2 - ccsd.t2)),
        "sigma_p_HF": sigma_HF,
        "sigma_p_CCSD": out["sigma_p_CCSD"],
        "sigma_p_CCSD_Lambda": out["sigma_p_CCSD_Lambda"],
        "timings": timings,
    }


# ── Public render function ─────────────────────────────────────────


def render_wavefunction_nmr_tab():
    """Streamlit panel for the post-HF correlation cascade."""
    st.markdown(f"### {t('wfn_title')}")
    st.caption(t("wfn_subtitle"))

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        mol_key = st.selectbox(
            t("wfn_molecule"),
            list(_PRESETS.keys()),
            index=1,  # N₂ default
            help=t("wfn_molecule_help"),
        )
        st.caption(_PRESETS[mol_key]["label"])
    with col2:
        basis_type = st.selectbox(
            t("wfn_basis"),
            ["SZ", "DZ", "DZP"],
            index=0,
            help=t("wfn_basis_help"),
        )
    with col3:
        n_radial = st.select_slider(
            t("wfn_grid"),
            options=[8, 10, 12, 16],
            value=10,
            help=t("wfn_grid_help"),
        )

    if not st.button(t("wfn_run"), type="primary"):
        st.info(t("wfn_press_run"))
        return

    with st.spinner(t("wfn_running")):
        result = _run_cascade(mol_key, basis_type, n_radial)

    # ── Header summary ──
    st.success(t("wfn_done"))
    a, b, c, d = st.columns(4)
    a.metric(t("wfn_n_orb"), result["n_orb"])
    b.metric(t("wfn_n_occ"), result["n_occ"])
    c.metric(t("wfn_n_virt"), result["n_virt"])
    d.metric(t("wfn_t1_norm"), f"{result['ccsd_t1_norm']:.2e}")

    # ── Correlation energy hierarchy ──
    st.markdown(f"#### {t('wfn_energies')}")
    e = result["energies"]
    e_mp2 = e["MP2"]
    rows = [
        ("MP2", e["MP2"], 0.0, "—"),
        ("MP3 (pp+hh+ring)", e["MP3"],
         (e["MP3"] - e_mp2) / e_mp2 * 100.0,
         f"{result['timings']['MP3']:.1f}s"),
        ("LCCD (∞ linear)", e["LCCD"],
         (e["LCCD"] - e_mp2) / e_mp2 * 100.0,
         f"{result['lccd_iters']} iters"),
        ("CCD (T2² renorm)", e["CCD"],
         (e["CCD"] - e_mp2) / e_mp2 * 100.0,
         f"{result['ccd_iters']} iters"),
        ("CCSD (T1+T2)", e["CCSD"],
         (e["CCSD"] - e_mp2) / e_mp2 * 100.0,
         f"{result['ccsd_iters']} iters"),
    ]
    import pandas as pd
    df = pd.DataFrame(
        rows,
        columns=[t("wfn_method"), t("wfn_e_corr"),
                 t("wfn_delta_pct"), t("wfn_detail")],
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Λ diagnostic ──
    st.markdown(f"#### {t('wfn_lambda')}")
    la, lb = st.columns(2)
    la.metric(t("wfn_lambda_iters"), result["lambda_iters"])
    lb.metric(t("wfn_lambda_drift"),
              f"{result['lambda_T_drift']:.2e}",
              help=t("wfn_lambda_drift_help"))

    # ── σ_p shielding ──
    st.markdown(f"#### {t('wfn_sigma_p')}")
    sp1, sp2, sp3 = st.columns(3)
    sp1.metric("σ_p^HF",          f"{result['sigma_p_HF']:+.2f} ppm")
    sp2.metric("σ_p^CCSD",        f"{result['sigma_p_CCSD']:+.2f} ppm")
    sp3.metric("σ_p^CCSD-Λ",      f"{result['sigma_p_CCSD_Lambda']:+.2f} ppm")
    delta = (result["sigma_p_CCSD_Lambda"] - result["sigma_p_HF"])
    st.caption(
        f"{t('wfn_sigma_p_delta')} : "
        f"σ_p^CCSD-Λ − σ_p^HF = {delta:+.2f} ppm"
    )

    # ── Timings (collapsible) ──
    with st.expander(t("wfn_timings")):
        for step, dt in result["timings"].items():
            st.text(f"  {step:<14} {dt:6.1f} s")

    # ── Theory note ──
    with st.expander(t("wfn_theory_note")):
        st.markdown(t("wfn_theory_body"))
