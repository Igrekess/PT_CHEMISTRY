"""Catalysis tab — heterogeneous and enzymatic catalysis from PT."""
import streamlit as st
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Preset reactions for catalysis
# ---------------------------------------------------------------------------
PRESET_REACTIONS_CAT = {
    "N2 + 3 H2 >> 2 NH3 (Haber-Bosch)": "N#N + 3 [H][H] >> 2 [H]N([H])[H]",
    "H2 >> 2 H (dissociation)": "[H][H] >> [H].[H]",
    "2 H2 + O2 >> 2 H2O (combustion)": "2 [H][H] + O=O >> 2 [H]O[H]",
    "CO + H2O >> CO2 + H2 (WGS)": "[C-]#[O+] + [H]O[H] >> O=C=O + [H][H]",
    "C2H4 + H2 >> C2H6 (hydrogenation)": "C=C + [H][H] >> CC",
    "H2 + Cl2 >> 2 HCl": "[H][H] + ClCl >> 2 Cl[H]",
}


def render_catalysis_tab():
    """Render the catalysis tab with two sub-sections."""
    from ptc_app.benchmark_data import grade_header
    st.caption(grade_header('catalysis'))

    mode = st.radio(
        "Type de catalyse",
        ["Catalyse heterogene", "Catalyse enzymatique"],
        horizontal=True,
        key="cat_mode",
    )

    if mode == "Catalyse heterogene":
        _render_heterogeneous()
    else:
        _render_enzymatic()


# ---------------------------------------------------------------------------
# Heterogeneous catalysis
# ---------------------------------------------------------------------------

def _render_heterogeneous():
    from ptc.catalysis import (
        SUPPORTED_METALS,
        catalyzed_barrier,
        volcano_plot,
    )

    st.markdown("### Catalyse heterogene (surface metallique)")
    st.caption("PT Sabatier: Ea_cat = Ea_uncat x (1 - f_sabatier x max_reduction). Structures cristallines : donnees experimentales.")

    input_mode = st.radio(
        "Input", ["Prediction (reactifs seuls)", "Presets", "Equation"],
        horizontal=True, key="cat_het_mode",
    )

    col_in, col_out = st.columns([1.2, 2])

    with col_in:
        reaction_eq = None

        if input_mode == "Presets":
            preset_name = st.selectbox(
                "Reaction preset",
                list(PRESET_REACTIONS_CAT.keys()),
                key="cat_het_preset",
            )
            reaction_eq = PRESET_REACTIONS_CAT[preset_name]
            st.code(reaction_eq, language="text")

        elif input_mode == "Equation":
            reaction_eq = st.text_input(
                "Equation (SMILES ou formule, >> separateur)",
                value="N#N + 3 [H][H] >> 2 [H]N([H])[H]",
                key="cat_het_eq",
            )

        else:  # Prediction
            st.markdown("**Entrez les reactifs — PTC predit la reaction**")
            pr1 = st.text_input("Reactif 1 (SMILES ou formule)", value="H2", key="cat_pr1")
            pr2 = st.text_input("Reactif 2 (SMILES ou formule)", value="N2", key="cat_pr2")

        metal_mode = st.radio("Catalyseur", ["d-block (30 metaux)", "Personnalise"],
                              horizontal=True, key="cat_metal_mode")
        if metal_mode == "d-block (30 metaux)":
            metal = st.selectbox(
                "Metal catalyseur",
                SUPPORTED_METALS,
                index=SUPPORTED_METALS.index("Pt") if "Pt" in SUPPORTED_METALS else 0,
                key="cat_het_metal",
            )
        else:
            metal = st.text_input(
                "Symbole du metal (ex: Pd, Ru, W, Ti...)",
                value="Pd",
                key="cat_het_metal_custom",
            ).strip()

        calc_btn = st.button("Calculer (1 metal)", type="primary", key="cat_het_calc")
        best_btn = st.button("Trouver le meilleur (30 metaux d-block)", key="cat_het_best")

    # If prediction mode: first predict, then use the equation
    if input_mode == "Prediction (reactifs seuls)" and (calc_btn or best_btn):
        react_list = []
        if pr1.strip():
            react_list.append(pr1.strip())
        if pr2.strip():
            react_list.append(pr2.strip())
        if react_list:
            with st.spinner("Prediction de la reaction..."):
                from ptc.reaction_predictor import predict_reaction, _KNOWN_MOLECULES
                from ptc.reaction_predictor import _parse_smiles_atoms, _atom_sig
                from collections import Counter

                pred = predict_reaction(react_list)

                # In catalysis mode, accept even kinetically blocked reactions
                # (that's the whole point of a catalyst: lower the barrier!)
                has_products = pred.product_info and any(
                    p.get('smiles') for p in pred.product_info
                )

                if has_products and pred.delta_G < 5.0:  # accept even mildly endothermic
                    r_parts = []
                    r_counts = Counter(ri['smiles'] for ri in pred.reactant_info)
                    for smi, cnt in r_counts.items():
                        r_parts.append(f"{cnt} {smi}" if cnt > 1 else smi)

                    p_parts = []
                    for p in pred.product_info:
                        if p.get('smiles'):
                            p_parts.append(p['smiles'])

                    if p_parts:
                        reaction_eq = " + ".join(r_parts) + " >> " + " + ".join(p_parts)
                        if pred.kinetically_blocked:
                            st.warning(f"Reaction predite (bloquee sans catalyseur) : {reaction_eq}")
                        else:
                            st.success(f"Reaction predite : {reaction_eq}")
                    else:
                        st.warning("Produits non identifies")
                        reaction_eq = None
                else:
                    st.info("Pas de reaction favorable entre ces reactifs, meme avec catalyseur")
                    reaction_eq = None

    with col_out:
        if calc_btn and reaction_eq and reaction_eq.strip():
            with st.spinner("Calcul catalyse..."):
                try:
                    result = catalyzed_barrier(reaction_eq.strip(), metal)
                    st.session_state["cat_het_result"] = result
                    st.session_state["cat_het_error"] = None
                except Exception as exc:
                    st.session_state["cat_het_error"] = str(exc)
                    st.session_state.pop("cat_het_result", None)

        # -- Display single-metal result --
        err = st.session_state.get("cat_het_error")
        res = st.session_state.get("cat_het_result")
        if err:
            st.error(f"Erreur: {err}")
        elif res is not None:
            st.markdown(f"**Reaction** : `{res.reaction}`")
            st.markdown(f"**Metal** : {res.metal}")

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Ea (sans catalyseur)", f"{res.Ea_uncatalyzed:.4f} eV")
                st.metric("Ea (catalyse)", f"{res.Ea_catalyzed:.4f} eV")
                st.metric("Ea (kJ/mol)", f"{res.Ea_cat_kJ:.1f}")
            with c2:
                _rate_str = f"{res.rate_enhancement:.2e}" if res.rate_enhancement > 1e3 else f"{res.rate_enhancement:.1f}"
                st.metric("Acceleration k_cat/k", _rate_str)
                st.metric("Score Sabatier", f"{res.sabatier_score:.3f}")
                st.metric("Position volcan", res.volcano_position)

            with st.expander("Details adsorption"):
                st.markdown(f"- E_ads (reactif) : {res.E_ads_reactant:+.4f} eV")
                st.markdown(f"- E_ads (TS)      : {res.E_ads_TS:+.4f} eV")
                st.markdown(f"- E_ads (produit) : {res.E_ads_product:+.4f} eV")

        # -- Best catalyst search --
        if best_btn and reaction_eq and reaction_eq.strip():
            with st.spinner("Test des 30 metaux d-block..."):
                try:
                    all_results = volcano_plot(reaction_eq.strip())
                    # Sort by lowest Ea (best catalyst first)
                    ranked = sorted(all_results, key=lambda r: r.Ea_catalyzed)
                    best = ranked[0]

                    st.markdown(
                        f"""<div style="
                            background:#d4edda;border-left:5px solid #28a745;
                            padding:12px 16px;border-radius:4px;margin:8px 0;
                        ">
                            <span style="font-size:1.4em;">🏆</span>
                            <span style="background:#28a745;color:white;padding:4px 12px;
                                border-radius:4px;font-weight:700;font-size:1.1em;margin-left:8px;">
                                MEILLEUR CATALYSEUR : {best.metal}</span>
                            <br><span style="color:#155724;margin-top:4px;display:inline-block;">
                                Ea = {best.Ea_catalyzed:.3f} eV ({best.Ea_cat_kJ:.0f} kJ/mol)
                                | Acceleration x{best.rate_enhancement:.0e}
                                | Sabatier = {best.sabatier_score:.2f} ({best.volcano_position})
                            </span>
                        </div>""",
                        unsafe_allow_html=True,
                    )

                    # Ranking table
                    st.markdown("**Classement des 18 catalyseurs :**")
                    import pandas as pd
                    rows = []
                    for i, r in enumerate(ranked):
                        rows.append({
                            '#': i + 1,
                            'Metal': r.metal,
                            'Ea cat (eV)': round(r.Ea_catalyzed, 4),
                            'Ea cat (kJ/mol)': round(r.Ea_cat_kJ, 1),
                            'Acceleration': f"{r.rate_enhancement:.1e}",
                            'Sabatier': round(r.sabatier_score, 3),
                            'Position': r.volcano_position,
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                    # Also show volcano plot
                    _draw_volcano(all_results)

                except Exception as exc:
                    st.error(f"Erreur: {exc}")

        # (volcano plot is shown inside "best" button handler above)


def _draw_volcano(results):
    """Draw a bar chart of catalyzed Ea by metal (volcano plot)."""
    metals = [r.metal for r in results]
    ea_vals = [r.Ea_catalyzed for r in results]
    scores = [r.sabatier_score for r in results]
    positions = [r.volcano_position for r in results]

    # Color by volcano position
    color_map = {'weak': '#3498db', 'optimal': '#2ecc71', 'strong': '#e74c3c'}
    colors = [color_map.get(p, '#95a5a6') for p in positions]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metals,
        y=ea_vals,
        marker_color=colors,
        text=[f"S={s:.2f}" for s in scores],
        textposition='outside',
        name="Ea catalyse (eV)",
    ))
    fig.update_layout(
        title=f"Volcano plot: {results[0].reaction}",
        xaxis_title="Metal",
        yaxis_title="Ea catalyse (eV)",
        height=400,
        showlegend=False,
    )
    # Add legend for colors
    for pos, color in color_map.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color=color),
            name=pos,
            showlegend=True,
        ))
    fig.update_layout(showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # Best catalyst callout
    best = results[0]
    st.success(
        f"Meilleur catalyseur : **{best.metal}** "
        f"(Ea = {best.Ea_cat_kJ:.1f} kJ/mol, "
        f"x{best.rate_enhancement:.1e} acceleration)"
    )


# ---------------------------------------------------------------------------
# Enzymatic catalysis
# ---------------------------------------------------------------------------

def _render_enzymatic():
    from ptc.catalysis import (
        ENZYME_TYPES,
        _ENZYME_PROFILES,
        _INTERACTION_ENERGIES,
        enzyme_catalysis,
    )

    st.markdown("### Catalyse enzymatique")
    st.caption("PT: E_stab = S3 x Sum(n_i x E_i) / P1. L'enzyme stabilise le TS par complementarite. Tout depuis s = 1/2.")

    col_in, col_out = st.columns([1.2, 2])

    with col_in:
        rxn_mode_enz = st.radio(
            "Reaction", ["Presets", "Equation custom", "Prediction (reactifs seuls)"],
            horizontal=True, key="cat_enz_rxn_mode",
        )

        reaction_eq = None
        if rxn_mode_enz == "Presets":
            preset_name = st.selectbox(
                "Reaction preset",
                list(PRESET_REACTIONS_CAT.keys()),
                key="cat_enz_preset",
            )
            reaction_eq = PRESET_REACTIONS_CAT[preset_name]
            st.code(reaction_eq, language="text")
        elif rxn_mode_enz == "Equation custom":
            reaction_eq = st.text_input(
                "Equation (SMILES ou formule, >> separateur)",
                value="N#N + 3 [H][H] >> 2 [H]N([H])[H]",
                key="cat_enz_eq",
            )
        else:
            st.markdown("**Entrez les reactifs — PTC predit la reaction**")
            enz_pr1 = st.text_input("Reactif 1", value="H2O", key="cat_enz_pr1")
            enz_pr2 = st.text_input("Reactif 2", value="CO2", key="cat_enz_pr2")

        enz_mode = st.radio("Enzyme", ["Presets (15 classes)", "Personnalise"],
                            horizontal=True, key="cat_enz_mode")

        custom_contacts = None
        if enz_mode == "Presets (5 classes)":
            enzyme = st.selectbox(
                "Type d'enzyme",
                ENZYME_TYPES,
                format_func=lambda e: f"{e} — {_ENZYME_PROFILES[e]['description']}",
                key="cat_enz_type",
            )
        else:
            enzyme = "custom"
            st.markdown("**Definissez le profil d'interactions de votre enzyme :**")
            custom_contacts = {}
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                custom_contacts['H_bond'] = st.number_input(
                    "Liaisons H (H-bond)", 0, 20, 3, key="enz_hb")
                custom_contacts['ionic'] = st.number_input(
                    "Ponts salins (ionique)", 0, 10, 1, key="enz_ion")
                custom_contacts['hydrophobic'] = st.number_input(
                    "Contacts hydrophobes", 0, 10, 2, key="enz_hyd")
                custom_contacts['covalent_cat'] = st.number_input(
                    "Catalyse covalente", 0, 5, 0, key="enz_cov")
            with col_c2:
                custom_contacts['metal_ion'] = st.number_input(
                    "Ions metalliques (Zn, Mg...)", 0, 5, 0, key="enz_met")
                custom_contacts['pi_stack'] = st.number_input(
                    "pi-stacking (Phe, Trp, Tyr)", 0, 10, 0, key="enz_pi")
                custom_contacts['vdw'] = st.number_input(
                    "Van der Waals", 0, 20, 0, key="enz_vdw")
            # Remove zero contacts
            custom_contacts = {k: v for k, v in custom_contacts.items() if v > 0}

        calc_btn = st.button("Calculer", type="primary", key="cat_enz_calc")

        # Show interaction energy table
        with st.expander("Energies d'interaction PT (derivees de s = 1/2)"):
            for name, (E, desc) in _INTERACTION_ENERGIES.items():
                st.markdown(f"- **{name}** : {E:.4f} eV ({E*96.485:.1f} kJ/mol) — {desc}")

    # Prediction mode: predict reaction from reactants
    if rxn_mode_enz == "Prediction (reactifs seuls)" and calc_btn:
        react_list = []
        if enz_pr1.strip():
            react_list.append(enz_pr1.strip())
        if enz_pr2.strip():
            react_list.append(enz_pr2.strip())
        if react_list:
            with st.spinner("Prediction de la reaction..."):
                from ptc.reaction_predictor import predict_reaction
                from collections import Counter
                pred = predict_reaction(react_list)
                has_products = pred.product_info and any(
                    p.get('smiles') for p in pred.product_info)
                if has_products and pred.delta_G < 5.0:
                    r_parts = []
                    r_counts = Counter(ri['smiles'] for ri in pred.reactant_info)
                    for smi, cnt in r_counts.items():
                        r_parts.append(f"{cnt} {smi}" if cnt > 1 else smi)
                    p_parts = [p['smiles'] for p in pred.product_info if p.get('smiles')]
                    if p_parts:
                        reaction_eq = " + ".join(r_parts) + " >> " + " + ".join(p_parts)
                        if pred.kinetically_blocked:
                            st.warning(f"Reaction predite (bloquee sans enzyme) : {reaction_eq}")
                        else:
                            st.success(f"Reaction predite : {reaction_eq}")
                    else:
                        st.warning("Produits non identifies")
                else:
                    st.info("Pas de reaction favorable")

    with col_out:
        if calc_btn and reaction_eq and reaction_eq.strip():
            with st.spinner("Calcul catalyse enzymatique..."):
                try:
                    result = enzyme_catalysis(
                        reaction_eq.strip(),
                        enzyme if enzyme != "custom" else "generic",
                        custom_contacts=custom_contacts,
                    )
                    st.session_state["cat_enz_result"] = result
                    st.session_state["cat_enz_error"] = None
                except Exception as exc:
                    st.session_state["cat_enz_error"] = str(exc)
                    st.session_state.pop("cat_enz_result", None)

        err = st.session_state.get("cat_enz_error")
        res = st.session_state.get("cat_enz_result")
        if err:
            st.error(f"Erreur: {err}")
        elif res is not None:
            desc = _ENZYME_PROFILES.get(res.enzyme_type, {}).get('description', res.enzyme_type)
            st.markdown(f"**Reaction** : `{res.reaction}`")
            st.markdown(f"**Enzyme** : {desc}")

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Ea (sans enzyme)", f"{res.Ea_uncatalyzed:.4f} eV")
                st.metric("Ea (enzymatique)", f"{res.Ea_enzymatic:.4f} eV")
                st.metric("Ea (kJ/mol)", f"{res.Ea_enz_kJ:.1f}")
            with c2:
                _rate_str = f"{res.rate_enhancement:.2e}" if res.rate_enhancement > 1e3 else f"{res.rate_enhancement:.1f}"
                st.metric("Acceleration k_cat/k", _rate_str)
                st.metric("Stabilisation TS", f"{res.E_stabilization:.4f} eV")
                st.metric("Contacts", f"{res.n_contacts} ({res.E_per_contact:.4f} eV/contact)")
