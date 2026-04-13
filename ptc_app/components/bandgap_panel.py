"""Band-gap tab -- Phillips-Van Vechten decomposition via PT."""

from __future__ import annotations

import math

import streamlit as st
import plotly.graph_objects as go

from ptc.materials import analyze_material, MaterialResult
from ptc.data.experimental import SYMBOLS
from ptc_app.components.gauges import error_gauge

# ── Material catalogue: (Z_A, Z_B, structure, exp_gap_eV) ──────────
MATERIALS: dict[str, tuple[int, int | None, str, float]] = {
    # Elementaires (diamond) — MAE ~1%
    "C (diamant)":         (6,  None, "diamond",     5.47),
    "Si (silicium)":       (14, None, "diamond",     1.12),
    "Ge (germanium)":      (32, None, "diamond",     0.66),
    # III-V (zincblende/wurtzite) — MAE ~3%
    "GaAs":                (31, 33,   "zincblende",  1.42),
    "GaN":                 (31, 7,    "wurtzite",    3.40),
    "SiC":                 (14, 6,    "zincblende",  2.36),
    # Ioniques (rocksalt) — MAE ~3%
    "NaCl":                (11, 17,   "rocksalt",    8.50),
    "MgO":                 (12, 8,    "rocksalt",    7.80),
    "KCl":                 (19, 17,   "rocksalt",    8.40),
    "KBr":                 (19, 35,   "rocksalt",    7.40),
    "LiF":                 (3,  9,    "rocksalt",   13.60),
}

STRUCTURES = ["diamond", "zincblende", "rocksalt", "wurtzite"]


def _bar_chart(E_h: float, C_ionic: float, E_gap: float) -> go.Figure:
    """Plotly bar chart: covalent / ionic / total."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["E_h (covalent)"],  y=[E_h],
        marker_color="#28a745", name="E_h",
        text=[f"{E_h:.3f}"], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=["C (ionic)"],      y=[C_ionic],
        marker_color="#fd7e14", name="C",
        text=[f"{C_ionic:.3f}"], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=["E_gap (total)"],  y=[E_gap],
        marker_color="#007bff", name="E_gap",
        text=[f"{E_gap:.3f}"], textposition="outside",
    ))
    fig.update_layout(
        yaxis_title="eV",
        showlegend=False,
        height=350,
        margin=dict(t=20, b=40),
    )
    return fig


def _display_result(res: MaterialResult, exp_gap: float | None,
                    structure: str) -> None:
    """Display analysis results."""
    col_left, col_right = st.columns(2)

    with col_left:
        error_gauge("E_gap", res.E_gap, exp_gap, "eV", ".3f")
        st.markdown(f"**Classification:** `{res.classification}`")
        st.markdown(f"**Constante dielectrique:** `{res.epsilon:.2f}`")
        st.markdown(f"**Structure:** `{structure}`")

    with col_right:
        fig = _bar_chart(res.E_h, res.C_ionic, res.E_gap)
        st.plotly_chart(fig, use_container_width=True)

    check = math.sqrt(res.E_h ** 2 + res.C_ionic ** 2)
    st.markdown(
        f"*Pythagore CRT (P3): E_gap = sqrt(E_h² + C²) "
        f"= sqrt({res.E_h:.3f}² + {res.C_ionic:.3f}²) "
        f"= {check:.3f} eV*"
    )


def render_bandgap_tab() -> None:
    """Render the band-gap analysis tab in Streamlit."""
    st.subheader("Band Gap — Phillips-Van Vechten (PT)")
    st.caption(
        "🟢 **Grade A** | MAE 2.5% | 11 materiaux | Ref: CRC Handbook / Kittel | 0 param\n\n"
        "Grades evalues sur mesures experimentales (Si, Ge, GaAs, NaCl, MgO, etc.)"
    )

    mode = st.radio("Mode", ["Presets", "Personnalise"], horizontal=True, key="bg_mode")

    if mode == "Presets":
        name = st.selectbox("Materiau", list(MATERIALS.keys()), key="bg_mat")
        Z_A, Z_B, structure, exp_gap = MATERIALS[name]

        if st.button("Calculer", key="bg_calc", type="primary"):
            from ptc_app.benchmark_data import ptc_timer
            with st.spinner("Calcul PTC..."), ptc_timer():
                res = analyze_material(Z_A, Z_B, structure)
            _display_result(res, exp_gap, structure)

    else:
        st.markdown("##### Materiau personnalise")
        col1, col2, col3 = st.columns(3)
        with col1:
            Z_A = st.number_input("Z element A", 1, 118, 14, key="bg_za")
            sym_a = SYMBOLS.get(Z_A, "?")
            st.caption(f"→ {sym_a}")
        with col2:
            use_binary = st.checkbox("Compose binaire (A-B)", value=False, key="bg_binary")
            Z_B = None
            if use_binary:
                Z_B = st.number_input("Z element B", 1, 118, 33, key="bg_zb")
                sym_b = SYMBOLS.get(Z_B, "?")
                st.caption(f"→ {sym_b}")
        with col3:
            structure = st.selectbox("Structure", STRUCTURES, key="bg_struct")

        exp_input = st.text_input("E_gap exp (eV, optionnel)", value="", key="bg_exp")
        exp_gap = None
        if exp_input.strip():
            try:
                exp_gap = float(exp_input)
            except ValueError:
                pass

        if st.button("Calculer", key="bg_calc_custom", type="primary"):
            from ptc_app.benchmark_data import ptc_timer
            with st.spinner("Calcul PTC..."), ptc_timer():
                try:
                    res = analyze_material(Z_A, Z_B, structure)
                    label = f"{sym_a}-{SYMBOLS.get(Z_B, '')}" if Z_B else sym_a
                    st.markdown(f"### {label} ({structure})")
                    _display_result(res, exp_gap, structure)
                except Exception as e:
                    st.error(f"Erreur: {e}")
