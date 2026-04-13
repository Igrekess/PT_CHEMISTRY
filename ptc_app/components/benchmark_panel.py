"""Benchmark panel — PTC accuracy across all molecules."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_all_molecules():
    """Load the fixed 1000-molecule benchmark set.

    Uses ptc/data/benchmark_1000.json — a frozen reference list.
    No dynamic sorting or filtering. Same 1000 for every run.
    """
    import json, os
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'ptc', 'data', 'benchmark_1000.json')
    with open(path) as f:
        mols = json.load(f)
    return [
        {"name": m["name"], "smiles": m["smiles"],
         "D_at_exp": m["D_exp"], "category": m["category"]}
        for m in mols
    ]


# ---------------------------------------------------------------------------
# Heavy compute — cached
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _compute_all_benchmark():
    """Compute D_at for every benchmark molecule. Returns a DataFrame."""
    from ptc.topology import build_topology
    from ptc.cascade import compute_D_at_cascade

    rows = _load_all_molecules()
    total = len(rows)
    progress = st.progress(0, text="Calcul PTC benchmark...")

    results = []
    for i, row in enumerate(rows):
        progress.progress((i + 1) / total, text=f"Molecule {i + 1}/{total}: {row['name']}")
        try:
            topo = build_topology(row["smiles"])
            res = compute_D_at_cascade(topo)
            calc = res.D_at
        except Exception:
            calc = np.nan
        results.append({
            **row,
            "D_at_PTC": calc,
        })

    progress.empty()

    df = pd.DataFrame(results)
    df["error_pct"] = ((df["D_at_PTC"] - df["D_at_exp"]) / df["D_at_exp"] * 100)
    df["abs_error_pct"] = df["error_pct"].abs()

    # Fixed 1000 molecules — no sorting/filtering needed
    df_valid = df.dropna(subset=["D_at_PTC"]).copy()

    return df_valid


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_benchmark_tab():
    """Main entry point called from app.py."""

    df_all = _compute_all_benchmark()

    # ── Category filter ───────────────────────────────────────────────
    cat = st.radio(
        "Categorie",
        ["Toutes", "Organique", "Inorganique", "d-block"],
        horizontal=True,
    )
    df = df_all if cat == "Toutes" else df_all[df_all["category"] == cat]
    df_valid = df.dropna(subset=["D_at_PTC"])

    # ── Header stats ──────────────────────────────────────────────────
    n_mol = len(df_valid)
    mae = df_valid["abs_error_pct"].mean()
    median_err = df_valid["abs_error_pct"].median()

    c1, c2, c3 = st.columns(3)
    c1.metric("Molecules", f"{n_mol}")
    c2.metric("MAE", f"{mae:.2f} %")
    c3.metric("Median |err|", f"{median_err:.2f} %")

    # ── Scatter plot ──────────────────────────────────────────────────
    color_map = {"Organique": "#3b82f6", "Inorganique": "#f97316", "d-block": "#22c55e"}

    fig_scatter = go.Figure()
    for cat_name, color in color_map.items():
        sub = df_valid[df_valid["category"] == cat_name]
        if sub.empty:
            continue
        fig_scatter.add_trace(go.Scatter(
            x=sub["D_at_exp"],
            y=sub["D_at_PTC"],
            mode="markers",
            marker=dict(color=color, size=5, opacity=0.7),
            name=cat_name,
            customdata=np.stack([sub["name"], sub["error_pct"].round(2)], axis=-1),
            hovertemplate="%{customdata[0]}<br>err = %{customdata[1]}%<extra></extra>",
        ))

    # Perfect y=x line
    lo = min(df_valid["D_at_exp"].min(), df_valid["D_at_PTC"].min()) * 0.95
    hi = max(df_valid["D_at_exp"].max(), df_valid["D_at_PTC"].max()) * 1.05
    fig_scatter.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines",
        line=dict(color="black", dash="dash", width=1),
        showlegend=False,
    ))

    fig_scatter.update_layout(
        title=f"PTC vs Experimental ({n_mol} molecules, MAE {mae:.2f}%)",
        xaxis_title="D_at experimental (eV)",
        yaxis_title="D_at PTC (eV)",
        height=500,
        template="plotly_white",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Error histogram ───────────────────────────────────────────────
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1000]
    labels = ["0-1%", "1-2%", "2-3%", "3-4%", "4-5%", "5-6%", "6-7%", "7-8%", "8-9%", "9-10%", ">10%"]
    df_valid = df_valid.copy()
    df_valid["bin"] = pd.cut(df_valid["abs_error_pct"], bins=bins, labels=labels, right=True)
    counts = df_valid["bin"].value_counts().reindex(labels, fill_value=0)

    bar_colors = []
    for lab in labels:
        if lab == "0-1%":
            bar_colors.append("#22c55e")
        elif lab in ("1-2%", "2-3%", "3-4%", "4-5%"):
            bar_colors.append("#eab308")
        else:
            bar_colors.append("#ef4444")

    fig_hist = go.Figure(go.Bar(
        x=labels,
        y=counts.values,
        marker_color=bar_colors,
    ))
    fig_hist.update_layout(
        title="Distribution des erreurs |err%|",
        xaxis_title="|erreur| %",
        yaxis_title="Nombre de molecules",
        height=350,
        template="plotly_white",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Dataframe table ───────────────────────────────────────────────
    st.markdown("#### Detail per molecule")
    display_df = df_valid[["name", "smiles", "category", "D_at_exp", "D_at_PTC", "error_pct", "abs_error_pct"]].copy()
    display_df.columns = ["Nom", "SMILES", "Categorie", "D_at exp (eV)", "D_at PTC (eV)", "Erreur %", "|Erreur| %"]
    display_df = display_df.sort_values("|Erreur| %", ascending=False).reset_index(drop=True)

    st.dataframe(
        display_df,
        use_container_width=True,
        height=500,
        column_config={
            "D_at exp (eV)": st.column_config.NumberColumn(format="%.3f"),
            "D_at PTC (eV)": st.column_config.NumberColumn(format="%.3f"),
            "Erreur %": st.column_config.NumberColumn(format="%.2f"),
            "|Erreur| %": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    # ══════════════════════════════════════════════════════════════════
    #  CUSTOM BENCHMARK — user uploads their own molecules
    # ══════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Benchmark personnalise")
    st.markdown(
        "Soumettez votre propre jeu de donnees pour verifier PTC "
        "sur vos molecules de reference."
    )

    # Template download
    template_csv = "nom,smiles,D_at_exp_eV,incertitude_eV\nwater,[H]O[H],9.511,0.01\nmethane,C,17.045,0.02\n"
    st.download_button(
        "Telecharger le template CSV",
        data=template_csv,
        file_name="ptc_benchmark_template.csv",
        mime="text/csv",
    )

    st.markdown(
        "**Format CSV** : colonnes `nom`, `smiles`, `D_at_exp_eV`, "
        "`incertitude_eV` (optionnelle).  \n"
        "Accepte SMILES (`[H]O[H]`) ou formules (`H2O`)."
    )

    uploaded = st.file_uploader(
        "Deposer votre fichier CSV",
        type=["csv"],
        key="custom_bench",
    )

    if uploaded is not None:
        try:
            user_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Erreur de lecture CSV : {e}")
            return

        # Validate columns
        required = {"smiles", "D_at_exp_eV"}
        cols_lower = {c.lower().strip(): c for c in user_df.columns}
        missing = required - set(cols_lower.keys())
        if missing:
            # Try alternate column names
            alt_map = {
                "smiles": ["smiles", "smi", "molecule", "mol"],
                "d_at_exp_ev": ["d_at_exp_ev", "d_at_exp", "d_at", "exp", "experimental"],
            }
            for req_col in list(missing):
                for alt in alt_map.get(req_col, []):
                    if alt in cols_lower:
                        missing.discard(req_col)
                        break
            if missing:
                st.error(f"Colonnes manquantes : {missing}. "
                         f"Colonnes trouvees : {list(user_df.columns)}")
                return

        # Normalize column names
        col_map = {}
        for c in user_df.columns:
            cl = c.lower().strip()
            if cl in ("nom", "name"):
                col_map[c] = "nom"
            elif cl in ("smiles", "smi", "molecule", "mol"):
                col_map[c] = "smiles"
            elif cl in ("d_at_exp_ev", "d_at_exp", "d_at", "exp", "experimental"):
                col_map[c] = "D_at_exp"
            elif cl in ("incertitude_ev", "incertitude", "uncertainty", "error"):
                col_map[c] = "incertitude"
        user_df = user_df.rename(columns=col_map)

        if "nom" not in user_df.columns:
            user_df["nom"] = [f"mol_{i}" for i in range(len(user_df))]
        if "incertitude" not in user_df.columns:
            user_df["incertitude"] = 0.0

        n_user = len(user_df)
        st.markdown(f"**{n_user} molecules chargees.** Calcul PTC en cours...")

        # Compute D_at for each user molecule
        from ptc.cascade import compute_D_at_cascade
        from ptc.topology import build_topology

        results = []
        progress = st.progress(0)
        for i, row in user_df.iterrows():
            smi = str(row["smiles"]).strip()
            exp = float(row["D_at_exp"])
            unc = float(row.get("incertitude", 0))
            nom = str(row.get("nom", f"mol_{i}"))

            try:
                topo = build_topology(smi)
                r = compute_D_at_cascade(topo)
                calc = r.D_at
                err = (calc - exp) / exp * 100 if exp != 0 else 0
            except Exception:
                calc = None
                err = None

            results.append({
                "Nom": nom,
                "SMILES": smi,
                "D_at exp (eV)": exp,
                "Incertitude (eV)": unc,
                "D_at PTC (eV)": calc,
                "Erreur %": err,
                "|Erreur| %": abs(err) if err is not None else None,
            })
            progress.progress((i + 1) / n_user)

        progress.empty()
        res_df = pd.DataFrame(results)
        valid = res_df.dropna(subset=["D_at PTC (eV)"])

        if valid.empty:
            st.warning("Aucune molecule n'a pu etre calculee.")
            return

        # Stats
        user_mae = valid["|Erreur| %"].mean()
        user_median = valid["|Erreur| %"].median()
        n_ok = len(valid)
        n_within_unc = 0
        for _, r in valid.iterrows():
            unc = r["Incertitude (eV)"]
            if unc > 0 and abs(r["D_at PTC (eV)"] - r["D_at exp (eV)"]) <= unc:
                n_within_unc += 1

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Molecules", f"{n_ok} / {n_user}")
        c2.metric("MAE", f"{user_mae:.2f} %")
        c3.metric("Mediane", f"{user_median:.2f} %")
        if any(valid["Incertitude (eV)"] > 0):
            c4.metric("Dans incertitude exp.", f"{n_within_unc} / {n_ok}")

        # Scatter
        fig_user = go.Figure()
        fig_user.add_trace(go.Scatter(
            x=valid["D_at exp (eV)"],
            y=valid["D_at PTC (eV)"],
            mode="markers",
            marker=dict(color="#8b5cf6", size=8),
            text=valid["Nom"],
            hovertemplate="%{text}<br>err = %{customdata:.1f}%<extra></extra>",
            customdata=valid["Erreur %"],
        ))

        # Error bars from uncertainty
        if any(valid["Incertitude (eV)"] > 0):
            fig_user.add_trace(go.Scatter(
                x=valid["D_at exp (eV)"],
                y=valid["D_at PTC (eV)"],
                mode="markers",
                marker=dict(color="rgba(0,0,0,0)"),
                error_x=dict(
                    type="data",
                    array=valid["Incertitude (eV)"],
                    visible=True,
                    color="#ccc",
                ),
                showlegend=False,
                hoverinfo="skip",
            ))

        lo = min(valid["D_at exp (eV)"].min(), valid["D_at PTC (eV)"].min()) * 0.9
        hi = max(valid["D_at exp (eV)"].max(), valid["D_at PTC (eV)"].max()) * 1.1
        fig_user.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi],
            mode="lines", line=dict(color="black", dash="dash", width=1),
            showlegend=False,
        ))
        fig_user.update_layout(
            title=f"Votre benchmark ({n_ok} mol, MAE {user_mae:.2f}%)",
            xaxis_title="D_at exp (eV)",
            yaxis_title="D_at PTC (eV)",
            height=450,
            template="plotly_white",
            showlegend=False,
        )
        st.plotly_chart(fig_user, use_container_width=True)

        # Table
        st.dataframe(
            res_df.sort_values("|Erreur| %", ascending=False).reset_index(drop=True),
            use_container_width=True,
            column_config={
                "D_at exp (eV)": st.column_config.NumberColumn(format="%.3f"),
                "D_at PTC (eV)": st.column_config.NumberColumn(format="%.3f"),
                "Incertitude (eV)": st.column_config.NumberColumn(format="%.3f"),
                "Erreur %": st.column_config.NumberColumn(format="%.2f"),
                "|Erreur| %": st.column_config.NumberColumn(format="%.2f"),
            },
        )
