"""Reaction tab — input reactants & products separately, compute thermochemistry."""
import streamlit as st
import plotly.graph_objects as go

from ptc.reactions import compute_reaction
from ptc_app.components.gauges import error_gauge
from ptc_app.components.mol_viewer import render_viewer

# ---------------------------------------------------------------------------
# Preset reactions: display name -> (reactants, products, exp ΔH kJ/mol)
# Each reactant/product is (coeff, SMILES, display_name)
# ---------------------------------------------------------------------------
PRESET_REACTIONS = {
    # ── Explosives / tres exothermiques ──
    "💥 H2 + F2 -> 2 HF (explosif)": {
        "reactants": [(1, "[H][H]", "H2"), (1, "FF", "F2")],
        "products":  [(2, "[H]F", "HF")],
        "exp_dH_kJ": -546.0,
    },
    "💥 2 H2 + O2 -> 2 H2O (combustion)": {
        "reactants": [(2, "[H][H]", "H2"), (1, "O=O", "O2")],
        "products":  [(2, "[H]O[H]", "H2O")],
        "exp_dH_kJ": -483.6,
    },
    "🔥 CH4 + 2 O2 -> CO2 + 2 H2O": {
        "reactants": [(1, "C", "CH4"), (2, "O=O", "O2")],
        "products":  [(1, "O=C=O", "CO2"), (2, "[H]O[H]", "H2O")],
        "exp_dH_kJ": -890.4,
    },
    # ── Exothermiques classiques ──
    "↗ H2 + Cl2 -> 2 HCl": {
        "reactants": [(1, "[H][H]", "H2"), (1, "ClCl", "Cl2")],
        "products":  [(2, "Cl[H]", "HCl")],
        "exp_dH_kJ": -184.6,
    },
    "↗ 2 Na + Cl2 -> 2 NaCl": {
        "reactants": [(2, "[Na]", "Na"), (1, "ClCl", "Cl2")],
        "products":  [(2, "[Na]Cl", "NaCl")],
        "exp_dH_kJ": -822.0,
    },
    # ── Besoin de temperature / pression ──
    "🌡 N2 + 3 H2 -> 2 NH3 (Haber, 500°C)": {
        "reactants": [(1, "N#N", "N2"), (3, "[H][H]", "H2")],
        "products":  [(2, "N", "NH3")],
        "exp_dH_kJ": -91.8,
    },
    "🌡 N2 + O2 -> 2 NO (foudre, >2000K)": {
        "reactants": [(1, "N#N", "N2"), (1, "O=O", "O2")],
        "products":  [(2, "[N]=O", "NO")],
        "exp_dH_kJ": +180.5,
    },
    # ── Endothermiques ──
    "❄ 2 H2O -> 2 H2 + O2 (electrolyse)": {
        "reactants": [(2, "[H]O[H]", "H2O")],
        "products":  [(2, "[H][H]", "H2"), (1, "O=O", "O2")],
        "exp_dH_kJ": +483.6,
    },
    # ── Does not react at room temperature ──
    "⛔ N2 + O2 (inerte a 298K)": {
        "reactants": [(1, "N#N", "N2"), (1, "O=O", "O2")],
        "products":  [(2, "[N]=O", "NO")],
        "exp_dH_kJ": +180.5,
    },
    # ── Organiques ──
    "↗ C2H4 + H2 -> C2H6 (hydrogenation)": {
        "reactants": [(1, "C=C", "C2H4"), (1, "[H][H]", "H2")],
        "products":  [(1, "CC", "C2H6")],
        "exp_dH_kJ": -137.0,
    },
    "🔥 2 C2H6 + 7 O2 -> 4 CO2 + 6 H2O": {
        "reactants": [(2, "CC", "C2H6"), (7, "O=O", "O2")],
        "products":  [(4, "O=C=O", "CO2"), (6, "[H]O[H]", "H2O")],
        "exp_dH_kJ": -3120.0,
    },
    "↗ 2 CH3OH + 3 O2 -> 2 CO2 + 4 H2O": {
        "reactants": [(2, "CO", "CH3OH"), (3, "O=O", "O2")],
        "products":  [(2, "O=C=O", "CO2"), (4, "[H]O[H]", "H2O")],
        "exp_dH_kJ": -1452.0,
    },
}


def _build_equation(reactants, products):
    """Build equation string from lists of (coeff, smiles)."""
    parts_r = []
    for coeff, smi, *_ in reactants:
        if coeff > 1:
            parts_r.append(f"{coeff} {smi}")
        else:
            parts_r.append(smi)
    parts_p = []
    for coeff, smi, *_ in products:
        if coeff > 1:
            parts_p.append(f"{coeff} {smi}")
        else:
            parts_p.append(smi)
    return " + ".join(parts_r) + " >> " + " + ".join(parts_p)


def render_reaction_tab():
    """Main entry point rendered inside a Streamlit tab."""

    st.subheader("Reactions chimiques")
    from ptc_app.benchmark_data import grade_header
    st.caption(grade_header('reactions'))

    mode = st.radio("Mode", ["Prediction (reactifs seuls)", "Presets", "Personnalise"],
                     horizontal=True, key="rxn_mode")

    if mode == "Presets":
        choice = st.selectbox("Reaction", list(PRESET_REACTIONS.keys()), key="rxn_preset")
        preset = PRESET_REACTIONS[choice]
        reactants = preset["reactants"]
        products = preset["products"]
        exp_dH = preset["exp_dH_kJ"]

        # Display nicely
        r_str = " + ".join(f"{c} {n}" if c > 1 else n for c, _, n in reactants)
        p_str = " + ".join(f"{c} {n}" if c > 1 else n for c, _, n in products)
        st.markdown(f"**{r_str}  →  {p_str}**")

    elif mode == "Prediction (reactifs seuls)":
        # ── PREDICTION MODE: only reactants, PT predicts products ──
        st.markdown("##### Enter reactants (SMILES or formula) — PTC predicts products")

        # Presets for quick testing
        PRED_PRESETS = {
            "(saisie libre)": ("", ""),
            "H2 + O2 (combustion)": ("H2", "O2"),
            "H2 + F2 (explosif)": ("H2", "F2"),
            "CH4 + O2 (combustion)": ("CH4", "O2"),
            "H2 + Cl2": ("H2", "Cl2"),
            "N2 + O2 (inerte ?)": ("N2", "O2"),
            "N2 + H2 (Haber ?)": ("N2", "H2"),
            "Na + Cl2": ("Na", "Cl2"),
            "C2H4 + H2 (hydrogenation)": ("C2H4", "H2"),
            "Fe + O2 (rouille)": ("Fe", "O2"),
            "H2 + H2 (rien ?)": ("H2", "H2"),
        }
        pred_preset = st.selectbox("Essai rapide", list(PRED_PRESETS.keys()),
                                   key="pred_preset")
        default_r1, default_r2 = PRED_PRESETS[pred_preset]

        col_pr1, col_pr2, col_pr3 = st.columns(3)
        with col_pr1:
            pr1 = st.text_input("Reactif 1 (SMILES ou formule)",
                                value=default_r1 or "H2", key="pred_r1")
            pr1_n = st.number_input("Nombre", 1, 10, 1, key="pred_r1n")
        with col_pr2:
            pr2 = st.text_input("Reactif 2 (SMILES ou formule)",
                                value=default_r2 or "O2", key="pred_r2")
            pr2_n = st.number_input("Nombre", 1, 10, 1, key="pred_r2n")
        with col_pr3:
            pr3 = st.text_input("Reactif 3 (optionnel)", value="", key="pred_r3")
            pr3_n = st.number_input("Nombre", 1, 10, 1, key="pred_r3n")

        st.caption("Accepte SMILES ([H][H], O=O, c1ccccc1) ou formules (H2, O2, CH4, CO2, NH3, H2O)")

        if st.button("Predire", type="primary", key="pred_btn"):
            react_list = []
            for smi, n in [(pr1, pr1_n), (pr2, pr2_n), (pr3, pr3_n)]:
                if smi.strip():
                    react_list.extend([smi.strip()] * n)

            if not react_list:
                st.warning("Entrez au moins 1 reactif.")
            else:
                with st.spinner("Prediction PT en cours..."):
                    from ptc.reaction_predictor import predict_reaction
                    pred = predict_reaction(react_list)

                # 3D viewers for reactants
                from ptc.api import Molecule as _PredMol
                st.markdown("##### Reactifs")
                r_cols = st.columns(len(react_list) if len(react_list) <= 6 else 6)
                for i, smi in enumerate(react_list[:6]):
                    with r_cols[i]:
                        try:
                            m = _PredMol(smi)
                            st.caption(m.formula)
                            render_viewer(m.mol_block, height=160, style="ballstick", fmt="sdf")
                        except Exception:
                            st.caption(smi)

                # Result badge
                if pred.reacts:
                    if pred.kinetically_blocked:
                        badge = ("⚠️", "#fd7e14", "BLOQUE CINETIQUEMENT",
                                 f"Thermodynamiquement favorable mais Ea = {pred.Ea_kJ:.0f} kJ/mol trop eleve")
                    elif pred.delta_G_kJ < -300:
                        badge = ("💥", "#dc3545", "REACTION EXPLOSIVE",
                                 f"ΔG = {pred.delta_G_kJ:+.0f} kJ/mol")
                    elif pred.delta_G_kJ < -100:
                        badge = ("🔥", "#fd7e14", "REACTION EXOTHERMIQUE",
                                 f"ΔG = {pred.delta_G_kJ:+.0f} kJ/mol")
                    else:
                        badge = ("↗", "#28a745", "REACTION FAVORABLE",
                                 f"ΔG = {pred.delta_G_kJ:+.0f} kJ/mol")
                else:
                    badge = ("⛔", "#6c757d", "PAS DE REACTION",
                             "Les reactifs sont deja dans leur etat le plus stable")

                st.markdown(
                    f"""<div style="
                        background:{badge[1]}22;border-left:5px solid {badge[1]};
                        padding:12px 16px;border-radius:4px;margin:12px 0;
                    ">
                        <span style="font-size:1.5em;">{badge[0]}</span>
                        <span style="background:{badge[1]};color:white;padding:4px 12px;
                            border-radius:4px;font-weight:700;font-size:1.1em;margin-left:8px;">
                            {badge[2]}</span>
                        <br><span style="color:#555;margin-top:4px;display:inline-block;">
                            {badge[3]}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

                # Predicted products with 3D viewers
                if pred.reacts and pred.product_info:
                    st.markdown("##### Produits predits")
                    p_cols = st.columns(len(pred.product_info) if len(pred.product_info) <= 6 else 6)
                    for i, prod in enumerate(pred.product_info[:6]):
                        with p_cols[i]:
                            st.markdown(f"**{prod['formula']}**")
                            if prod.get('smiles'):
                                try:
                                    m = _PredMol(prod['smiles'])
                                    render_viewer(m.mol_block, height=160, style="ballstick", fmt="sdf")
                                except Exception:
                                    pass
                            st.caption(f"D_at = {prod['D_at']:.2f} eV")

                # Details
                col_l, col_r = st.columns(2)
                with col_l:
                    st.markdown(f"**ΔG** = `{pred.delta_G:.4f}` eV / `{pred.delta_G_kJ:+.1f}` kJ/mol")
                    st.markdown(f"**Ea** = `{pred.Ea:.4f}` eV / `{pred.Ea_kJ:.1f}` kJ/mol")
                    st.markdown(f"**D_at reactifs** = `{pred.reactant_D_at:.3f}` eV")
                    st.markdown(f"**D_at produits** = `{pred.product_D_at:.3f}` eV")
                with col_r:
                    if pred.alternatives and len(pred.alternatives) > 1:
                        st.markdown("**Alternatives :**")
                        for alt in pred.alternatives[1:4]:
                            prods = ' + '.join(p['formula'] for p in alt['products'])
                            st.markdown(f"- {prods} (ΔG = {alt['delta_G_kJ']:+.0f} kJ)")

        return  # Don't show the classic reaction UI below

    else:
        exp_dH = None
        st.markdown("##### Reactifs")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            r1_smi = st.text_input("Reactif 1 (SMILES)", value="[H][H]", key="r1")
            r1_coeff = st.number_input("Coeff", 1, 10, 1, key="r1c")
        with col_r2:
            r2_smi = st.text_input("Reactif 2 (SMILES)", value="ClCl", key="r2")
            r2_coeff = st.number_input("Coeff", 1, 10, 1, key="r2c")

        st.markdown("##### Produits")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            p1_smi = st.text_input("Produit 1 (SMILES)", value="Cl[H]", key="p1")
            p1_coeff = st.number_input("Coeff", 1, 10, 2, key="p1c")
        with col_p2:
            p2_smi = st.text_input("Produit 2 (SMILES, optionnel)", value="", key="p2")
            p2_coeff = st.number_input("Coeff", 1, 10, 1, key="p2c")

        exp_input = st.text_input("ΔH exp (kJ/mol, optionnel)", value="", key="rxn_exp")
        if exp_input.strip():
            try:
                exp_dH = float(exp_input)
            except ValueError:
                pass

        reactants = []
        if r1_smi.strip():
            reactants.append((r1_coeff, r1_smi.strip()))
        if r2_smi.strip():
            reactants.append((r2_coeff, r2_smi.strip()))

        products = []
        if p1_smi.strip():
            products.append((p1_coeff, p1_smi.strip()))
        if p2_smi.strip():
            products.append((p2_coeff, p2_smi.strip()))

    col_T, col_P = st.columns(2)
    with col_T:
        T = st.slider("Temperature (K)", 200, 1000, 298, step=1, key="rxn_T")
    with col_P:
        import math as _math
        # Log-scale pressure slider: exponent from -2 to 3 (0.01 to 1000 atm)
        log_P = st.slider("Pression (atm, echelle log)",
                           min_value=-2.0, max_value=3.0, value=0.0, step=0.1,
                           format="%.1f", key="rxn_logP",
                           help="log10(P). Ex: 0 = 1 atm, 2 = 100 atm, -1 = 0.1 atm")
        P = 10.0 ** log_P
        st.caption(f"P = {P:.3g} atm")

    # ---- compute ---------------------------------------------------------
    if st.button("Calculer", type="primary", key="rxn_calc"):
        if not reactants or not products:
            st.warning("Il faut au moins 1 reactif et 1 produit.")
            return

        equation = _build_equation(reactants, products)
        st.code(equation, language="text")

        with st.spinner("Calcul PTC en cours..."):
            try:
                r = compute_reaction(equation, T=float(T), P=float(P))
            except Exception as e:
                st.error(f"Erreur: {e}")
                return

        # ---- 3D viewers FIRST: reactants | → | products -----------------
        # Each molecule repeated coeff times (2 H2 = 2 viewers)
        from ptc.api import Molecule as _RxnMol

        # Expand: [(2, "H2O")] -> [("H2O", mol), ("H2O", mol)]
        react_expanded = []
        for entry in reactants:
            coeff, smi = entry[0], entry[1]
            try:
                m = _RxnMol(smi)
                for _ in range(coeff):
                    react_expanded.append(m)
            except Exception:
                react_expanded.append(None)

        prod_expanded = []
        for entry in products:
            coeff, smi = entry[0], entry[1]
            try:
                m = _RxnMol(smi)
                for _ in range(coeff):
                    prod_expanded.append(m)
            except Exception:
                prod_expanded.append(None)

        n_r = len(react_expanded)
        n_p = len(prod_expanded)
        n_cols = n_r + 1 + n_p
        cols_3d = st.columns(n_cols)

        for i, m in enumerate(react_expanded):
            with cols_3d[i]:
                if m is not None:
                    st.caption(m.formula)
                    render_viewer(m.mol_block, height=200, style="ballstick", fmt="sdf")

        with cols_3d[n_r]:
            st.markdown(
                "<div style='display:flex;align-items:center;justify-content:center;"
                "height:200px;font-size:3em;color:#666;'>→</div>",
                unsafe_allow_html=True,
            )

        for i, m in enumerate(prod_expanded):
            with cols_3d[n_r + 1 + i]:
                if m is not None:
                    st.caption(m.formula)
                    render_viewer(m.mol_block, height=200, style="ballstick", fmt="sdf")

        st.divider()

        # ---- Classification badge ----------------------------------------
        dH = r.delta_H_kJ
        Ea = r.Ea_kJ
        if dH < -300 and Ea < 50:
            badge_text = "POTENTIELLEMENT EXPLOSIF"
            badge_color = "#dc3545"
            badge_icon = "💥"
            badge_desc = "Tres exothermique + faible barriere d'activation"
        elif dH < -200:
            badge_text = "TRES EXOTHERMIQUE"
            badge_color = "#fd7e14"
            badge_icon = "🔥"
            badge_desc = "Forte liberation d'energie"
        elif dH < 0:
            badge_text = "EXOTHERMIQUE"
            badge_color = "#28a745"
            badge_icon = "↗"
            badge_desc = "Reaction favorable (libere de l'energie)"
        elif dH > 200:
            badge_text = "TRES ENDOTHERMIQUE"
            badge_color = "#6f42c1"
            badge_icon = "❄"
            badge_desc = "Forte absorption d'energie"
        elif dH > 0:
            badge_text = "ENDOTHERMIQUE"
            badge_color = "#2196F3"
            badge_icon = "↘"
            badge_desc = "Reaction defavorable (absorbe de l'energie)"
        else:
            badge_text = "NEUTRE"
            badge_color = "#6c757d"
            badge_icon = "="
            badge_desc = "Pas de variation d'enthalpie"

        # Spontaneity
        if r.delta_G_kJ < -50:
            spont = "Spontanee"
        elif r.delta_G_kJ < 0:
            spont = "Faiblement spontanee"
        else:
            spont = "Non spontanee (necessite un apport d'energie)"

        st.markdown(
            f"""<div style="
                background:{badge_color}22;
                border-left:5px solid {badge_color};
                padding:10px 16px;
                border-radius:4px;
                margin-bottom:12px;
            ">
                <span style="font-size:1.4em;">{badge_icon}</span>
                <span style="
                    background:{badge_color};color:white;
                    padding:3px 10px;border-radius:4px;
                    font-weight:700;font-size:0.95em;
                    margin-left:8px;
                ">{badge_text}</span>
                <span style="color:#555;margin-left:10px;">{badge_desc}</span>
                <br><span style="color:#666;font-size:0.85em;margin-top:4px;display:inline-block;">
                    {spont} (ΔG = {r.delta_G_kJ:+.1f} kJ/mol, P = {r.P:.3g} atm, Δn = {r.delta_n:+d})
                </span>
            </div>""",
            unsafe_allow_html=True,
        )

        # ---- Thermochemistry details -------------------------------------
        col_left, col_right = st.columns(2)

        with col_left:
            if exp_dH is not None:
                error_gauge("ΔH", r.delta_H_kJ, exp_dH, "kJ/mol", ".1f")
            else:
                st.markdown(f"**ΔH** = `{r.delta_H_kJ:+.1f}` kJ/mol")

            st.markdown(f"**ΔH** = `{r.delta_H_eV:+.4f}` eV / `{r.delta_H_kJ:+.1f}` kJ/mol")
            st.markdown(f"**ΔG** = `{r.delta_G_eV:+.4f}` eV / `{r.delta_G_kJ:+.1f}` kJ/mol")
            st.markdown(f"**Ea** = `{r.Ea_eV:.4f}` eV / `{r.Ea_kJ:.1f}` kJ/mol")
            st.markdown(f"**Δn** = `{r.delta_n:+d}` (variation moles de gaz)")
            st.markdown(f"**P** = `{r.P:.3g}` atm")
            st.markdown(f"**k(T={T} K)** = `{r.k_rate:.4e}` s⁻¹")
            if r.K_eq < 1e30:
                st.markdown(f"**K_eq** = `{r.K_eq:.4e}`")
            else:
                st.markdown(f"**K_eq** = ∞ (irreversible)")

        with col_right:
            D_react_total = sum(r.D_at_reactants.values())
            D_prod_total = sum(r.D_at_products.values())

            fig = go.Figure(data=[
                go.Bar(
                    x=["Reactifs", "Produits"],
                    y=[D_react_total, D_prod_total],
                    marker_color=["#636EFA", "#EF553B"],
                    text=[f"{D_react_total:.2f} eV", f"{D_prod_total:.2f} eV"],
                    textposition="outside",
                ),
            ])
            fig.update_layout(
                title="Energie d'atomisation totale",
                yaxis_title="D_at (eV)",
                height=300,
                margin=dict(t=50, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)
