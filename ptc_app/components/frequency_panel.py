"""Frequency analysis panel — simulated IR spectrum + mode gauges."""
import os
os.system('find . -name "*.pyc" -delete 2>/dev/null; true')

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from ptc.api import Molecule
from ptc_app.components.gauges import error_gauge

# Experimental reference frequencies (sorted descending for comparison)
EXP_FREQ = {
    "[H]O[H]":  [1595, 3657, 3756],
    "O=C=O":    [667, 667, 1388, 2349],
    "C":        [1306, 1306, 1306, 1534, 1534, 2917, 3019, 3019, 3019],
}


def _lorentzian_spectrum(freqs: list[float], gamma: float = 30.0,
                         n_points: int = 2000):
    """Build Lorentzian-broadened IR spectrum from stick frequencies."""
    if not freqs:
        return np.array([]), np.array([])
    f_max = max(freqs) * 1.3
    x = np.linspace(200, f_max, n_points)
    y = np.zeros_like(x)
    for f in freqs:
        y += gamma / ((x - f) ** 2 + gamma ** 2)
    # Normalise to peak = 1
    y_max = y.max()
    if y_max > 0:
        y /= y_max
    return x, y


def render_freq_tab():
    """Main entry point for the frequency tab."""
    from ptc_app.benchmark_data import grade_header
    st.caption(grade_header('ir', 'raman'))

    _FREQ_PRESETS = {
        "H₂O (eau)": "[H]O[H]",
        "CO₂": "O=C=O",
        "CH₄ (methane)": "C",
        "NH₃ (ammoniac)": "N",
        "HCl": "[H]Cl",
        "HF": "[H]F",
        "C₂H₄ (ethylene)": "C=C",
        "C₆H₆ (benzene)": "c1ccccc1",
        "CH₃OH (methanol)": "CO",
        "HCHO (formaldehyde)": "C=O",
        "N₂O (protoxyde)": "[N-]=[N+]=O",
        "SO₂": "O=S=O",
        "Personnalise...": "",
    }
    col_preset, col_smi = st.columns([1, 2])
    with col_preset:
        preset = st.selectbox("Molecule", list(_FREQ_PRESETS.keys()), key="freq_preset")
    with col_smi:
        default_smi = _FREQ_PRESETS[preset]
        smiles = st.text_input("SMILES", value=default_smi, key="freq_smiles")

    run = st.button("Calculer", key="freq_run", type="primary")

    if not run:
        st.info("Entrez un SMILES et cliquez sur Calculer.")
        return

    try:
        from ptc_app.benchmark_data import ptc_timer
        with ptc_timer():
            mol = Molecule(smiles)
            res = mol.frequencies
    except Exception as exc:
        st.error(f"Erreur PTC : {exc}")
        return

    freqs = res.frequencies  # list of cm-1
    n_modes = res.n_modes
    zpe = res.ZPE  # eV

    # --- Layout: 2/3 spectrum | 1/3 details ---
    col_spec, col_info = st.columns([2, 1])

    # --- Left: simulated IR spectrum ---
    with col_spec:
        st.subheader("Spectre IR simule")
        if freqs:
            x, y = _lorentzian_spectrum(freqs)
            fig = go.Figure()
            # Filled Lorentzian envelope
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines",
                fill="tozeroy",
                fillcolor="rgba(31,119,180,0.3)",
                line=dict(color="rgb(31,119,180)", width=1.5),
                name="IR envelope",
            ))
            # Stick lines at each frequency
            for f in freqs:
                fig.add_trace(go.Scatter(
                    x=[f, f], y=[0, 1], mode="lines",
                    line=dict(color="red", width=1),
                    showlegend=False,
                ))
            fig.update_layout(
                xaxis_title="Frequence (cm⁻¹)",
                yaxis_title="Intensite (norm.)",
                xaxis=dict(range=[200, max(freqs) * 1.3]),
                yaxis=dict(range=[0, 1.05]),
                height=420,
                margin=dict(l=50, r=20, t=30, b=50),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucune frequence calculee.")

    # --- Right: mode count, ZPE, gauges ---
    with col_info:
        st.subheader("Modes vibrationnels")
        st.metric("Nombre de modes", n_modes)
        st.metric("ZPE", f"{zpe:.4f} eV")

        # Sorted descending, skip < 10 cm-1
        calc_sorted = sorted([f for f in freqs if f >= 10.0], reverse=True)
        exp_list = EXP_FREQ.get(smiles)
        exp_sorted = sorted(exp_list, reverse=True) if exp_list else None

        st.markdown("---")
        st.markdown("**Frequences individuelles**")
        for i, f_calc in enumerate(calc_sorted):
            f_exp = exp_sorted[i] if (exp_sorted and i < len(exp_sorted)) else None
            error_gauge(
                label=f"Mode {i + 1}",
                calc=f_calc,
                exp=f_exp,
                unit="cm⁻¹",
                fmt=".1f",
            )


# Alias for app.py compatibility
render_frequency_tab = render_freq_tab
