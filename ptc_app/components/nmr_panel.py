"""NMR prediction panel -- 1H and 13C chemical shifts + coupling constants."""
import os
os.system('find . -name "*.pyc" -delete 2>/dev/null; true')

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from ptc.nmr import (
    predict_nmr, NMRResult, NMRShift,
    NUCLEI, H_ENV_SHIFTS, C_ENV_SHIFTS,
    bloch_rotate, relaxation_t1, relaxation_t2,
    TYPICAL_J,
)


# ====================================================================
# SPECTRUM PLOT
# ====================================================================

def _lorentzian_stick(shifts: list[float], gamma: float = 0.05,
                      n_points: int = 2000,
                      x_min: float = -1.0, x_max: float = 14.0):
    """Build Lorentzian-broadened NMR spectrum (ppm, reversed x-axis)."""
    if not shifts:
        return np.array([]), np.array([])
    x = np.linspace(x_min, x_max, n_points)
    y = np.zeros_like(x)
    for s in shifts:
        y += gamma / ((x - s)**2 + gamma**2)
    y_max = y.max()
    if y_max > 0:
        y /= y_max
    return x, y


def _plot_nmr_spectrum(shifts: list[NMRShift], nucleus: str = '1H'):
    """Create a Plotly NMR spectrum figure."""
    if not shifts:
        return None

    ppm_values = [s.shift_ppm for s in shifts]

    if nucleus == '1H':
        x_min, x_max = -0.5, 13.0
        gamma = 0.03
    else:
        x_min, x_max = -5.0, 220.0
        gamma = 1.0

    x, y = _lorentzian_stick(ppm_values, gamma=gamma,
                              x_min=x_min, x_max=x_max)

    fig = go.Figure()

    # Envelope
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines',
        fill='tozeroy',
        fillcolor='rgba(31,119,180,0.25)',
        line=dict(color='rgb(31,119,180)', width=1.5),
        name=f'{nucleus} envelope',
        hoverinfo='skip',
    ))

    # Stick lines
    for s in shifts:
        fig.add_trace(go.Scatter(
            x=[s.shift_ppm, s.shift_ppm], y=[0, 0.95],
            mode='lines',
            line=dict(color='red', width=1.5),
            showlegend=False,
            hovertext=f"{s.environment}: {s.shift_ppm:.2f} ppm",
            hoverinfo='text',
        ))

    fig.update_layout(
        xaxis_title=f"delta ({nucleus}) / ppm",
        yaxis_title="Intensite (norm.)",
        xaxis=dict(
            range=[x_max, x_min],  # NMR convention: reversed
            dtick=1.0 if nucleus == '1H' else 20.0,
        ),
        yaxis=dict(range=[0, 1.1], showticklabels=False),
        height=380,
        margin=dict(l=50, r=20, t=30, b=50),
        showlegend=False,
    )

    return fig


# ====================================================================
# BLOCH SPHERE VISUALIZATION
# ====================================================================

def _plot_bloch_pulse_sequence():
    """Visualize 90-degree and 180-degree Fisher rotations."""
    import numpy as np

    state_0 = np.array([1.0, 0.0])

    # Trace the trajectory of a 90-degree pulse
    angles_90 = np.linspace(0, np.pi / 2, 50)
    angles_180 = np.linspace(0, np.pi, 100)

    traj_90 = [bloch_rotate(state_0, a) for a in angles_90]
    traj_180 = [bloch_rotate(state_0, a) for a in angles_180]

    # Convert to magnetization M_z = 2*p - 1
    mz_90 = [2 * s[0] - 1 for s in traj_90]
    mz_180 = [2 * s[0] - 1 for s in traj_180]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.degrees(angles_90), y=mz_90,
        mode='lines+markers',
        name='90-degree pulse',
        line=dict(color='blue', width=2),
        marker=dict(size=3),
    ))
    fig.add_trace(go.Scatter(
        x=np.degrees(angles_180), y=mz_180,
        mode='lines',
        name='180-degree pulse',
        line=dict(color='red', width=2, dash='dash'),
    ))

    # Key points
    fig.add_trace(go.Scatter(
        x=[0, 90, 180],
        y=[1.0, 0.0, -1.0],
        mode='markers+text',
        text=['Equilibre', 'M_z=0', 'Inversion'],
        textposition='top center',
        marker=dict(size=10, color=['green', 'blue', 'red']),
        showlegend=False,
    ))

    fig.update_layout(
        xaxis_title="Angle de rotation (degres)",
        yaxis_title="Magnetisation M_z",
        yaxis=dict(range=[-1.2, 1.3]),
        height=300,
        margin=dict(l=50, r=20, t=30, b=50),
    )
    return fig


# ====================================================================
# MAIN RENDER
# ====================================================================

def render_nmr_tab():
    """Main entry point for the RMN tab."""
    st.markdown("Prediction RMN depuis la Theorie de la Persistance.")
    from ptc_app.benchmark_data import grade_header
    st.caption(grade_header('nmr'))

    col_input, col_info = st.columns([2, 1])

    with col_input:
        smiles = st.text_input(
            "SMILES", value="CCO",
            key="nmr_smiles",
            help="Entrez un SMILES: CCO (ethanol), c1ccccc1 (benzene), CC=O (acetaldehyde)",
        )
        examples = {
            "Ethanol": "CCO",
            "Benzene": "c1ccccc1",
            "Acetaldehyde": "CC=O",
            "Acide acetique": "CC(=O)O",
            "Methylamine": "CN",
            "Chloroforme": "ClC(Cl)Cl",
            "Toluene": "Cc1ccccc1",
            "Acetone": "CC(=O)C",
            "Dimethylether": "COC",
            "Aniline": "Nc1ccccc1",
        }
        preset = st.selectbox(
            "Exemples",
            ["(personnalise)"] + list(examples.keys()),
            key="nmr_preset",
        )
        if preset != "(personnalise)":
            smiles = examples[preset]

    with col_info:
        st.markdown("**Noyaux actifs PT**")
        for nuc, dat in list(NUCLEI.items())[:3]:
            st.markdown(
                f"- **{nuc}**: p={dat['p']}, "
                f"sin^2={dat['sin2']:.4f}, "
                f"theta={np.degrees(dat['theta']):.1f} deg"
            )

    run = st.button("Calculer RMN", key="nmr_run", type="primary")

    if not run:
        st.info("Entrez un SMILES et cliquez sur Calculer RMN.")
        return

    # --- Compute ---
    try:
        from ptc_app.benchmark_data import ptc_timer
        with ptc_timer():
            result = predict_nmr(smiles)
    except Exception as exc:
        st.error(f"Erreur PTC: {exc}")
        return

    st.success(f"**{result.formula}** -- "
               f"{len(result.h_shifts)} shifts 1H, "
               f"{len(result.c_shifts)} shifts 13C, "
               f"{len(result.couplings)} couplages")

    # --- 1H Spectrum ---
    st.subheader("Spectre 1H RMN")
    col_spec, col_table = st.columns([2, 1])

    with col_spec:
        fig_h = _plot_nmr_spectrum(result.h_shifts, '1H')
        if fig_h:
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.info("Pas de protons dans cette molecule.")

    with col_table:
        if result.h_shifts:
            st.markdown("**Deplacements 1H (ppm)**")
            rows = []
            for s in sorted(result.h_shifts, key=lambda x: x.shift_ppm, reverse=True):
                rows.append({
                    "Idx": s.atom_idx,
                    "Env.": s.environment,
                    "delta (ppm)": f"{s.shift_ppm:.2f}",
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

    # --- 13C Spectrum ---
    if result.c_shifts:
        st.subheader("Spectre 13C RMN")
        col_spec_c, col_table_c = st.columns([2, 1])

        with col_spec_c:
            fig_c = _plot_nmr_spectrum(result.c_shifts, '13C')
            if fig_c:
                st.plotly_chart(fig_c, use_container_width=True)

        with col_table_c:
            st.markdown("**Deplacements 13C (ppm)**")
            rows_c = []
            for s in sorted(result.c_shifts, key=lambda x: x.shift_ppm, reverse=True):
                rows_c.append({
                    "Idx": s.atom_idx,
                    "Env.": s.environment,
                    "delta (ppm)": f"{s.shift_ppm:.1f}",
                })
            st.dataframe(rows_c, use_container_width=True, hide_index=True)

    # --- Coupling Constants ---
    if result.couplings:
        st.subheader("Constantes de couplage J")
        col_coup, col_ref = st.columns([2, 1])

        with col_coup:
            rows_j = []
            for c in sorted(result.couplings, key=lambda x: -x.J_hz):
                rows_j.append({
                    "A": f"{c.symbol_A}({c.atom_A})",
                    "B": f"{c.symbol_B}({c.atom_B})",
                    "n bonds": c.n_bonds,
                    "J (Hz)": f"{c.J_hz:.1f}",
                })
            st.dataframe(rows_j, use_container_width=True, hide_index=True)

        with col_ref:
            st.markdown("**Valeurs typiques (ref.)**")
            for name, val in TYPICAL_J.items():
                st.markdown(f"- {name}: {val:.0f} Hz")

    # --- Relaxation & Bloch dynamics ---
    st.subheader("Dynamique de Bloch (PT)")
    col_relax, col_bloch = st.columns([1, 2])

    with col_relax:
        st.metric("T1 (1H)", f"{result.t1_h:.2f} (u. crible)")
        st.metric("T2 (1H)", f"{result.t2_h:.2f} (u. crible)")
        st.markdown(
            "T1, T2 en unites de pas du crible.  \n"
            "T2 <= T1 (inegalite de Gordin)."
        )

    with col_bloch:
        fig_bloch = _plot_bloch_pulse_sequence()
        st.plotly_chart(fig_bloch, use_container_width=True)

    # --- Environment reference ---
    with st.expander("Reference: environnements 1H"):
        cols = st.columns(3)
        items = list(H_ENV_SHIFTS.items())
        n = len(items)
        chunk = (n + 2) // 3
        for col_idx, col in enumerate(cols):
            with col:
                for env, shift in items[col_idx * chunk:(col_idx + 1) * chunk]:
                    st.markdown(f"**{env}**: {shift:.1f} ppm")

    with st.expander("Reference: environnements 13C"):
        cols = st.columns(3)
        items = list(C_ENV_SHIFTS.items())
        n = len(items)
        chunk = (n + 2) // 3
        for col_idx, col in enumerate(cols):
            with col:
                for env, shift in items[col_idx * chunk:(col_idx + 1) * chunk]:
                    st.markdown(f"**{env}**: {shift:.1f} ppm")
