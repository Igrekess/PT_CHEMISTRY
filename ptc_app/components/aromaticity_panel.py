"""Aromaticity panel — NICS, σ/π split, full PT experimental signature."""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from ptc.api import Molecule
from ptc_app.i18n import t


# Curated benchmark molecules with known NICS values
NICS_BENCHMARKS = {
    "benzene":              ("c1ccccc1",          -9.7,  "π-aromatic (canonical)"),
    "cyclobutadiene":       ("C1=CC=C1",          +27.0, "π-antiaromatic"),
    "pyridine":             ("c1ccncc1",          -9.0,  "heterocyclic π-aromatic"),
    "thiophene":            ("c1ccsc1",          -13.6,  "5-ring π-aromatic"),
    "furan":                ("c1ccoc1",          -12.3,  "5-ring π-aromatic"),
    "S₃ (cyclic)":          ("[S]1[S][S]1",       None, "double σ⊕π aromatic, predicted"),
    "Bi₃ (cyclic)":         ("[Bi]1[Bi][Bi]1",   -15.0, "σ-aromatic all-metal"),
    "Al₄ (cyclic)":         ("[Al]1[Al][Al][Al]1", -34.0, "multi-fold σ⊕π anion"),
    "Si₆ (cyclic)":         ("[Si]1[Si][Si][Si][Si][Si]1", -10.0, "all-Si hexagon"),
    "U@S₃":                 ("[U][S]1[S][S]1",     None, "actinide-capped σ⊕π aromatic"),
    "Th@S₃":                ("[Th][S]1[S][S]1",    None, "thorium-capped — easier synthesis"),
    "B@S₃":                 ("[B][S]1[S][S]1",     None, "boron-capped — easiest synthesis"),
    "C@S₃":                 ("[C][S]1[S][S]1",     None, "carbon-capped — pure organic"),
    "P₃ (cyclic)":          ("[P]1[P][P]1",        None, "σ-aromatic + π-radical"),
    "Cu₃ (cyclic)":         ("[Cu]1[Cu][Cu]1",     None, "coinage σ-aromatic (1e/atom)"),
    "Ag₃ (cyclic)":         ("[Ag]1[Ag][Ag]1",     None, "coinage σ-aromatic"),
}


def _plot_NICS_profile(prof, R: float):
    """Plot NICS(z) vs z, with Biot-Savart reference curve."""
    zs = [p[0] for p in prof]
    nics = [p[1] for p in prof]

    fig = go.Figure()

    # Diamagnetic / paramagnetic shading
    if any(n < 0 for n in nics):
        fig.add_hrect(y0=-1e3, y1=0, fillcolor="rgba(31,119,180,0.08)",
                      line_width=0, layer="below")
        fig.add_annotation(x=max(zs)*0.85, y=min(nics)*0.6,
                           text="<b>diamagnetic</b><br>(aromatic)",
                           showarrow=False, font=dict(size=10, color="#1f77b4"))
    if any(n > 0 for n in nics):
        fig.add_hrect(y0=0, y1=1e3, fillcolor="rgba(214,39,40,0.08)",
                      line_width=0, layer="below")
        fig.add_annotation(x=max(zs)*0.85, y=max(nics)*0.6,
                           text="<b>paramagnetic</b><br>(antiaromatic)",
                           showarrow=False, font=dict(size=10, color="#d62728"))

    # NICS curve
    fig.add_trace(go.Scatter(
        x=zs, y=nics, mode='lines+markers',
        marker=dict(size=10, color='#2ca02c'),
        line=dict(width=2.5, color='#2ca02c'),
        name='NICS(z) PT',
        hovertemplate='z = %{x:.2f} Å<br>NICS = %{y:+.2f} ppm<extra></extra>',
    ))

    # Biot-Savart reference curve (continuous)
    z_dense = np.linspace(0, max(zs), 100)
    bs_ref = nics[0] * (R*R) / (R*R + z_dense*z_dense)**1.5 * R if R > 0 else None
    # Actually direct PT formula
    if R > 0 and nics[0] != 0:
        bs_dense = [nics[0] * (R*R) / (R*R + z*z)**1.5 / 1.0 for z in z_dense]
        fig.add_trace(go.Scatter(
            x=z_dense, y=bs_dense, mode='lines',
            line=dict(dash='dot', color='gray', width=1.5),
            name='Biot-Savart (PT analytical)',
            hoverinfo='skip',
        ))

    fig.update_layout(
        xaxis=dict(title='probe distance z (Å)', zeroline=True),
        yaxis=dict(title='NICS (ppm)', zeroline=True, zerolinecolor='black',
                   zerolinewidth=2),
        height=350, margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True, hovermode='x unified',
    )
    return fig


def _format_huckel_status(n: int, label: str) -> str:
    if n == 0:
        return f"{label} = 0"
    if n % 4 == 2:
        return f"**{label} = {n}** (4n+2 with n={(n-2)//4} ✓ aromatic)"
    if n % 4 == 0:
        return f"**{label} = {n}** (4n with n={n//4} — antiaromatic)"
    return f"**{label} = {n}** (odd — radical / open-shell)"


def render_aromaticity_tab():
    """Aromaticity diagnostic tab."""
    st.markdown("### NICS — Nucleus-Independent Chemical Shift")
    st.markdown(
        "Diagnostic d'aromaticité PT-pur (Pauling-London + règle de Hückel signée). "
        "K = α²·a₀/12 = 2.348 ppm·Å (préfacteur dérivé des constantes PT, sans fit)."
    )

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("#### Sélection")
        choice = st.selectbox(
            "Molécule de référence",
            options=list(NICS_BENCHMARKS.keys()),
            index=0,
            key="aro_choice",
        )
        smi_default = NICS_BENCHMARKS[choice][0]
        nics_exp = NICS_BENCHMARKS[choice][1]
        descriptor = NICS_BENCHMARKS[choice][2]

        smi = st.text_input(
            "SMILES (modifiable)",
            value=smi_default,
            key="aro_smiles",
        )
        st.caption(f"_{descriptor}_")
        if nics_exp is not None:
            st.metric("NICS(0) expérimental", f"{nics_exp:+.1f} ppm")

    with col_right:
        try:
            mol = Molecule(smi)
            aro = mol.aromaticity()
            if aro is None:
                st.warning("Aucun cycle détecté dans la molécule.")
                return
            sig = mol.signature()

            st.markdown(f"### {sig.formula}  —  {sig.aromatic_class}")
            st.caption(f"SMILES: `{smi}`")

            cols = st.columns(4)
            cols[0].metric("NICS(0) PT", f"{aro.NICS_0:+.2f} ppm",
                           delta=(f"{aro.NICS_0 - nics_exp:+.1f} vs exp"
                                  if nics_exp else None))
            cols[1].metric("NICS(1) PT", f"{aro.NICS_1:+.2f} ppm")
            ratio = aro.NICS_0 / aro.NICS_1 if aro.NICS_1 != 0 else 0
            cols[2].metric("Ratio NICS(0)/NICS(1)", f"{ratio:.2f}",
                           help=">1.5 indique σ-dominant ou double aromatic")
            cols[3].metric("Ring radius R", f"{aro.R:.2f} Å")

            st.markdown("#### Compte d'électrons délocalisés (par canal)")
            cc1, cc2 = st.columns(2)
            cc1.markdown(_format_huckel_status(sig.n_aromatic_sigma, "σ-aromatic"))
            cc2.markdown(_format_huckel_status(sig.n_aromatic_pi, "π-aromatic"))
            st.markdown(
                f"**T³ Fourier coherence f_coh = {sig.f_coh:.3f}** "
                f"(1.0 = homonucléaire pur)"
            )

            st.markdown("#### Profil NICS(z)")
            fig = _plot_NICS_profile(sig.NICS_profile, aro.R)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Erreur de calcul: {type(e).__name__}: {e}")
            return

    # ── Detail expanders ───────────────────────────────────────────
    with st.expander("📊 Energetics + fragmentation channels", expanded=False):
        col_e1, col_e2 = st.columns(2)
        col_e1.metric("D_at total", f"{sig.D_at:.3f} eV",
                      delta=f"{sig.D_per_atom:.2f} eV/atom")
        if sig.cap_binding > 0:
            col_e2.metric("Cap binding", f"{sig.cap_binding:.2f} eV",
                          delta="energy to remove cap M")
        else:
            col_e2.metric("Lowest fragmentation",
                          f"{sig.lowest_decomposition_eV:.2f} eV")

        st.markdown("**Canaux de fragmentation** (ΔE en eV) :")
        for c in sig.fragmentation:
            mark = " ← *plus bas*" if c.is_lowest else ""
            st.markdown(f"- ΔE = **{c.delta_E:+.2f}** : "
                        f"{' + '.join(c.products)} ({c.note}){mark}")

    with st.expander("🌈 Vibrational signature (Morse-calibrated)"):
        if sig.vib_modes:
            import pandas as pd
            df = pd.DataFrame([{
                'bond': v.label,
                'ω_PT raw (cm⁻¹)': f"{v.omega_raw:.0f}",
                'Morse factor': f"{v.morse_factor:.2f}",
                'ω calibrated (cm⁻¹)': f"{v.omega_calibrated:.0f}",
            } for v in sig.vib_modes])
            st.dataframe(df, use_container_width=True, hide_index=True)
        if sig.ir_breathing_cm1 > 0:
            st.success(f"**Mode breathing (totally symmetric, A₁ in C_nv): "
                       f"≈ {sig.ir_breathing_cm1:.0f} cm⁻¹** (IR-actif)")

    with st.expander("⚛️ Electronic structure (Koopmans + cycle-Hückel)"):
        c1, c2, c3 = st.columns(3)
        c1.metric("IE verticale", f"{sig.IE_PT_eV:.2f} eV")
        c2.metric("EA verticale", f"{sig.EA_PT_eV:.2f} eV")
        c3.metric("HOMO–LUMO gap", f"{sig.HL_gap_eV:.2f} eV")
        st.caption("EA est une estimation (½ × max EA atomique des atomes "
                   "membres). HOMO-LUMO gap σ peut être 0 si les modes "
                   "cycle-Hückel σ sont tous remplis (cas des doubly aromatic).")

    with st.expander("🔬 Geometry"):
        for label, r in sig.bond_lengths:
            st.markdown(f"- **{label}**: r_e = {r:.3f} Å")
        if sig.cap_height > 0:
            st.markdown(f"- **Cap height** above ring plane: {sig.cap_height:.3f} Å")

    # ── Theoretical reminder ──────────────────────────────────────
    with st.expander("📖 Formule PT (rappel théorique)"):
        st.markdown(r"""
        **Pauling-London PT NICS — formule signée Hückel :**

        $$
        \sigma(z) = -\frac{\alpha^2 a_0}{12}
        \cdot \frac{n_\text{eff}^\text{signed} \cdot f_\text{coh} \cdot R^2}
        {(R^2 + z^2)^{3/2}} \times 10^6 \text{ ppm}
        $$

        avec
        $$
        n_\text{eff}^\text{signed} = \text{sign}_\text{Hückel}(n_\sigma) \cdot n_\sigma
        + \text{sign}_\text{Hückel}(n_\pi) \cdot n_\pi
        $$

        et $\text{sign}_\text{Hückel}(n) = +1$ si $n = 4k+2$ (aromatic),
        $-1$ si $n = 4k$ (anti-aromatic), $+0.5$ si radical.

        **Aucun paramètre ajusté.** Les comptes σ/π viennent de la
        classification de groupe : G1/G11 → 1 σ, G13 → 1 π,
        G14 → 2 σ, G15 → 3 (σ+π), G16 → 4 (σ+2π).
        """)
