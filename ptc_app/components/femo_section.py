"""FeMo-cofactor — dedicated tab with 5 sections."""
import streamlit as st
from ptc_app.i18n import t


def render_femo_section():
    """Render FeMo-cofactor tab (S=3/2, χT, Mössbauer, N₂ fixation)."""
    st.markdown("## Clusters metalliques biologiques")
    from ptc_app.benchmark_data import grade_header
    st.caption(grade_header('femo'))

    # Show PT constants
    from ptc.femo import get_pt_constants_summary
    c = get_pt_constants_summary()

    col1, col2, col3 = st.columns(3)
    col1.metric("sin^2(theta_3)", f"{c['sin2_3']:.4f}")
    col2.metric("sin^2(theta_5)", f"{c['sin2_5']:.4f}")
    col3.metric("sin^2(theta_7)", f"{c['sin2_7']:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("J_0", f"{c['J0_meV']:.2f} meV")
    col5.metric("D(Fe II)", f"{c['D_FeII_cm1']:.1f} cm-1")
    col6.metric("dim(H)", f"{c['dim_H']:,}")

    st.markdown("""
**Structure Fe7MoS9C** : 2 cubanes (3 Fe(III) + 4 Fe(II))

| Facteur CRT | III-III | III-II | II-II |
|------------|---------|--------|-------|
| F_5 (mod 5) | {f55:.4f} | {f54:.4f} | {f44:.4f} |
""".format(f55=c['F5_III_III'], f54=c['F5_III_II'], f44=c['F5_II_II']))

    # Run computation
    if st.button("Diagonaliser H (135,000 x 135,000)", type="primary"):
        with st.spinner(t('computing')):
            try:
                from ptc.femo import compute_femo_ground_state
                r = compute_femo_ground_state(scenario='uniform', with_anisotropy=True)

                if r['verdict'] == 'PASS':
                    st.success(f"S_ground = {r['S_ground']:.1f} = N_c * s = 3/2  **{r['verdict']}**")
                else:
                    st.error(f"S_ground = {r['S_ground']:.1f}  **{r['verdict']}**")

                c1, c2, c3 = st.columns(3)
                c1.metric("S_ground", f"{r['S_ground']}")
                c2.metric("E_ground (meV)", f"{r['E_ground']:.2f}")
                c3.metric("Gap (cm-1)", f"{r['gap_cm1']:.1f}")

                st.markdown(f"""
- **PT** : S = 3/2 = N_c * s (0 parametre)
- **Mossbauer exp** : S = 3/2
- CRT mod 5 est **NECESSAIRE** : mod 7 seul donne S = 1/2 (sur-frustre)
""")
            except Exception as e:
                st.error(f"Erreur: {e}")
    else:
        st.info("Cliquer pour lancer la diagonalisation exacte (~15s)")
        st.markdown("""
**Prediction PT** : S = 3/2 = N_c * s

- sin^2 -> J_ij (couplages d'echange)
- J_ij -> H (hamiltonien de Heisenberg)
- H -> diag exacte -> S_ground = 3/2

**Mossbauer exp** : S = 3/2
""")

    # ── Magnetic susceptibility χ(T) ──
    st.divider()
    st.markdown("### Susceptibilite magnetique χ(T)")
    st.caption("Van Vleck sur 40 niveaux du spectre Heisenberg. 0 parametre.")

    if st.button("Calculer χ(T)", key="femo_chi"):
        with st.spinner("Diagonalisation + Boltzmann (~20s)..."):
            try:
                from ptc.femo import compute_magnetic_susceptibility
                from ptc_app.benchmark_data import ptc_timer
                import plotly.graph_objects as go

                with ptc_timer():
                    result = compute_magnetic_susceptibility(
                        T_range=(2.0, 300.0), n_points=150, n_levels=40,
                    )

                T = result['T']
                chiT = result['chiT']

                # χT vs T plot (standard magnetism representation)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=T, y=chiT,
                    mode='lines', name='PTC (0 param)',
                    line=dict(color='#FF4B4B', width=2),
                ))
                # Curie law for S=3/2: χT = C = 0.12505 × 3/2 × 5/2 = 0.469
                C_curie = 0.12505 * 1.5 * 2.5
                fig.add_hline(y=C_curie, line_dash="dash", line_color="gray",
                              annotation_text=f"Curie S=3/2: {C_curie:.3f}")
                fig.update_layout(
                    xaxis_title="T (K)",
                    yaxis_title="χT (emu·K/mol)",
                    height=350,
                    margin=dict(l=60, r=20, t=30, b=50),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Energy levels table
                st.markdown("#### Low-level spectrum")
                import pandas as pd
                rows = []
                for i in range(min(15, len(result['E_levels']))):
                    rows.append({
                        'Niveau': i,
                        'E (meV)': f"{result['E_levels'][i]:.3f}",
                        'E (cm⁻¹)': f"{result['E_levels'][i]*8.066:.1f}",
                        'S': f"{result['S_levels'][i]:.1f}",
                        '2S+1': int(2 * result['S_levels'][i] + 1),
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("S fondamental", f"{result['S_ground']:.1f}")
                c2.metric("Gap", f"{result['gap_cm1']:.1f} cm⁻¹")
                c3.metric("χT(2K)", f"{chiT[0]:.3f} emu·K/mol")

            except Exception as e:
                st.error(f"Erreur: {e}")
    else:
        st.info("Cliquer pour calculer la courbe χT vs T (Van Vleck, ~20s)")

    # ── Mössbauer parameters ──
    st.divider()
    st.markdown("### Parametres Mossbauer")
    st.caption("δ (isomer shift) et ΔEQ (quadrupole splitting) par site Fe. 0 parametre.")

    try:
        from ptc.femo import compute_mossbauer
        import pandas as pd

        mb = compute_mossbauer()
        rows = []
        for r in mb:
            rows.append({
                'Site': r['label'],
                'δ PT (mm/s)': f"{r['delta_PT']:.2f}",
                'δ exp (mm/s)': f"{r['delta_exp']:.2f}" if r['delta_exp'] else "—",
                'ΔEQ PT (mm/s)': f"{r['deltaEQ_PT']:.2f}",
                'ΔEQ exp (mm/s)': f"{r['deltaEQ_exp']:.2f}" if r['deltaEQ_exp'] else "—",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        import statistics
        d_errs = [r['err_delta'] for r in mb if r['err_delta'] is not None]
        q_errs = [r['err_deltaEQ'] for r in mb if r['err_deltaEQ'] is not None]
        c1, c2, c3 = st.columns(3)
        c1.metric("δ MAE", f"{statistics.mean(d_errs):.3f} mm/s")
        c2.metric("ΔEQ MAE", f"{statistics.mean(q_errs):.2f} mm/s")
        c3.metric("Observables", f"{len(d_errs) + len(q_errs)}")

        st.caption("Ref: Münck et al., Yoo 2000, Bjornsson 2017. "
                   "δ ∝ IE × sin²₃ (densite s au noyau). "
                   "ΔEQ ∝ sin²₅ × |n_d - P₂| (asymetrie d).")
    except Exception as e:
        st.error(f"Erreur: {e}")

    # ── Nitrogenase N₂ fixation cycle ──
    st.divider()
    st.markdown("### Cycle catalytique N₂ → 2NH₃")
    st.caption("Lowe-Thorneley 8 etapes. Adsorption via D_KL. 0 parametre.")

    col_ph, col_e = st.columns(2)
    with col_ph:
        n2_ph = st.slider("pH", 0.0, 14.0, 7.0, 0.5, key="n2_ph")
    with col_e:
        n2_e = st.slider("E_donor (V vs SHE)", -0.6, 0.0, -0.31, 0.01, key="n2_e")

    if st.button("Calculer le cycle", key="n2_cycle"):
        try:
            from ptc.nitrogenase import compute_nitrogenase_cycle
            from ptc_app.benchmark_data import ptc_timer
            import plotly.graph_objects as go
            import pandas as pd

            with ptc_timer():
                result = compute_nitrogenase_cycle(pH=n2_ph, E_donor=n2_e)

            # Energy profile plot
            prof = result['energy_profile']
            labels = ['ref'] + [s['step'] for s in result['steps']]
            colors = ['gray'] + ['#FF4B4B' if s['dG'] > 0 else '#28a745' for s in result['steps']]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=labels, y=prof,
                mode='lines+markers',
                marker=dict(size=10, color=colors),
                line=dict(color='#4B8BFF', width=2),
                text=[f"{e:+.3f} eV" for e in prof],
                hovertemplate='%{x}: %{text}<extra></extra>',
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                yaxis_title="ΔG cumulatif (eV)",
                xaxis_title="Etape",
                height=350,
                margin=dict(l=60, r=20, t=30, b=50),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Step table
            rows = []
            for s in result['steps']:
                rows.append({
                    'Etape': s['step'],
                    'ΔG (eV)': f"{s['dG']:+.3f}",
                    'E cumul (eV)': f"{s['E_cumul']:+.3f}",
                    'Type': s['type'],
                    'Reaction': s['desc'],
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("E_ads(N₂)", f"{result['E_ads_N2']:+.3f} eV")
            c2.metric("Etape limitante", result['barrier_step'])
            c3.metric("ΔG total", f"{result['overall_dG']:+.2f} eV")

        except Exception as e:
            st.error(f"Erreur: {e}")
    else:
        st.info("N₂ + 8H⁺ + 8e⁻ → 2NH₃ + H₂ (Lowe-Thorneley). Cliquer pour calculer.")

    # ── Biological cluster magnetism ──
    st.divider()
    st.markdown("### Clusters biologiques")
    st.caption("8/8 corrects. Heisenberg + Anderson + DE + JT-DE exclusion. 0 parametre.")

    try:
        from ptc.cluster_magnetism import compute_preset, list_presets
        from ptc_app.benchmark_data import ptc_timer
        import pandas as pd

        presets = list_presets()
        preset_names = [p['name'] for p in presets]
        selected = st.selectbox("Cluster", preset_names, key="cluster_select")

        if st.button("Diagonaliser", key="cluster_run", type="primary"):
            with st.spinner("Calcul..."), ptc_timer():
                r = compute_preset(selected, n_levels=10)

            p_info = next(p for p in presets if p['name'] == selected)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("S fondamental", f"{r.S_ground:.1f}")
            c2.metric("S attendu", f"{p_info['expected_S']}")
            c3.metric("Gap", f"{r.gap_cm1:.1f} cm⁻¹")
            verdict_color = "success" if r.verdict == "PASS" else "error"
            getattr(c4, verdict_color)(r.verdict)

            st.caption(p_info['description'])

            # Energy levels
            if r.E_levels:
                rows = []
                for i in range(min(8, len(r.E_levels))):
                    rows.append({
                        'Niveau': i,
                        'E (cm⁻¹)': f"{r.E_levels[i]*8.066:.1f}",
                        'S': f"{r.S_levels[i]:.1f}",
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            st.caption(f"dim(H) = {r.dim_H:,} | {r.n_sites} sites")
        else:
            # Summary table of all presets
            rows = []
            for p in presets:
                r = compute_preset(p['name'], n_levels=4)
                rows.append({
                    'Cluster': p['name'],
                    'S PT': f"{r.S_ground:.1f}",
                    'S exp': f"{p['expected_S']}",
                    'Verdict': r.verdict,
                    'Sites': p['n_sites'],
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur clusters: {e}")

    # ── Custom cluster builder ──
    st.divider()
    st.markdown("### Cluster personnalise")
    st.caption("Definissez vos propres sites metalliques et topologie. 0 parametre.")

    try:
        from ptc.cluster_magnetism import compute_cluster, MetalSite
        from ptc.atom import IE_eV
        from ptc_app.benchmark_data import ptc_timer
        import pandas as pd

        _METALS = {
            'Fe(III) S=5/2 d⁵': ('Fe', 26, 2.5, 'III', 5),
            'Fe(II) S=2 d⁶': ('Fe', 26, 2.0, 'II', 6),
            'Mn(IV) S=3/2 d³': ('Mn', 25, 1.5, 'IV', 3),
            'Mn(III) S=2 d⁴': ('Mn', 25, 2.0, 'III', 4),
            'Mn(II) S=5/2 d⁵': ('Mn', 25, 2.5, 'II', 5),
            'Co(II) S=3/2 d⁷': ('Co', 27, 1.5, 'II', 7),
            'Ni(II) S=1 d⁸': ('Ni', 28, 1.0, 'II', 8),
            'Cu(II) S=1/2 d⁹': ('Cu', 29, 0.5, 'II', 9),
            'Cr(III) S=3/2 d³': ('Cr', 24, 1.5, 'III', 3),
            'V(III) S=1 d²': ('V', 23, 1.0, 'III', 2),
        }
        _BRIDGE_TYPES = {
            'sulfido (S²⁻)': 'bridge',
            'intra-cubane (S²⁻)': 'intra',
            'oxo (O²⁻)': 'oxo',
            'di-oxo (2×O²⁻)': 'di_oxo',
            'inter-cluster': 'inter',
        }

        n_sites = st.slider("Nombre de sites", 2, 6, 2, key="cc_n")
        metal_opts = list(_METALS.keys())

        cols = st.columns(min(n_sites, 4))
        site_sels = []
        for i in range(n_sites):
            with cols[i % len(cols)]:
                s = st.selectbox(f"Site {i+1}", metal_opts,
                                 index=0 if i % 2 == 0 else 1, key=f"cc_s{i}")
                site_sels.append(s)

        bridge_sel = st.selectbox("Type de pont", list(_BRIDGE_TYPES.keys()), key="cc_br")

        if st.button("Calculer S fondamental", key="cc_go", type="primary"):
            sites = []
            for sel in site_sels:
                sym, Z, spin, ox, n_d = _METALS[sel]
                sites.append(MetalSite(sym, Z, spin, ox, n_d, IE_eV(Z)))

            btype = _BRIDGE_TYPES[bridge_sel]
            bonds = [(i, j, btype) for i in range(n_sites) for j in range(i+1, n_sites)]

            with st.spinner("Diagonalisation..."), ptc_timer():
                r = compute_cluster(sites, bonds, n_levels=10)

            c1, c2, c3 = st.columns(3)
            c1.metric("S fondamental", f"{r.S_ground:.1f}")
            c2.metric("Gap", f"{r.gap_cm1:.1f} cm⁻¹")
            c3.metric("dim(H)", f"{r.dim_H:,}")

            desc = " — ".join(f"{s.symbol}({s.ox_state})" for s in sites)
            st.markdown(f"**{desc}** | {len(bonds)} liens ({bridge_sel})")

            if r.E_levels:
                rows = []
                for i in range(min(8, len(r.E_levels))):
                    rows.append({'n': i, 'E (cm⁻¹)': f"{r.E_levels[i]*8.066:.1f}",
                                 'S': f"{r.S_levels[i]:.1f}"})
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur custom: {e}")
