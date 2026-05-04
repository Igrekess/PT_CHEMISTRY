"""Benchmark panel — PTC accuracy across all molecules."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_all_molecules():
    """Load the fixed 1000-molecule benchmark set + cross-check audit fields.

    Reads :
      - ptc/data/benchmark_1000_verified.json (canonical dataset, source/CAS/URL)
      - benchmarkb3lyp/audit_dexp_2026-05-03.json (reliability audit, optional)

    Merges audit_status / audit_diff_pct / audit_convention onto each row so
    the reliability badge can be computed downstream.
    """
    import json, os
    here = os.path.dirname(__file__)
    base = os.path.join(here, '..', '..', 'ptc', 'data')
    verified = os.path.join(base, 'benchmark_1000_verified.json')
    fallback = os.path.join(base, 'benchmark_1000.json')
    path = verified if os.path.exists(verified) else fallback
    with open(path) as f:
        mols = json.load(f)

    # Load audits : CCCBDB (legacy) and combined (CCCBDB + Burcat)
    audit_idx, combined_idx = {}, {}
    audit_path = os.path.join(here, '..', '..',
                              'benchmarkb3lyp', 'audit_dexp_2026-05-03.json')
    if os.path.exists(audit_path):
        with open(audit_path) as f:
            audit = json.load(f)
        for r in audit.get('results', []):
            audit_idx[(r['name'], r['smiles'])] = r
    combined_path = os.path.join(here, '..', '..',
                                 'benchmarkb3lyp', 'audit_combined_2026-05-04.json')
    if os.path.exists(combined_path):
        with open(combined_path) as f:
            comb = json.load(f)
        for r in comb.get('results', []):
            combined_idx[(r['name'], r['smiles'])] = r

    out = []
    for m in mols:
        row = {
            "name":     m["name"],
            "smiles":   m["smiles"],
            "D_at_exp": m["D_exp"],
            "category": m["category"],
        }
        for fld in ("source", "cas", "unc_eV",
                    "url_primary", "url_pubchem", "url_nist_webbook",
                    "url_cccbdb", "url_source_master"):
            if fld in m:
                row[fld] = m[fld]
        # Merge legacy CCCBDB audit fields
        ar = audit_idx.get((m["name"], m["smiles"]))
        if ar:
            row["audit_status"]     = ar.get("status")
            row["audit_diff_pct"]   = ar.get("best_diff_pct")
            row["audit_convention"] = ar.get("best_convention")
            row["audit_reason"]     = ar.get("reason") or ""
        else:
            row["audit_status"]     = "n/a"
            row["audit_diff_pct"]   = None
            row["audit_convention"] = None
            row["audit_reason"]     = "no_audit"
        # Merge combined audit fields
        cr = combined_idx.get((m["name"], m["smiles"]))
        if cr:
            row["combined_status"]    = cr.get("combined_status")
            row["combined_best_diff"] = cr.get("best_diff_pct")
            row["cccbdb_status"]      = cr.get("cccbdb_status")
            row["burcat_status"]      = cr.get("burcat_status")
        out.append(row)
    return out


_SOURCE_TIER_A = {"ATcT_0K", "ATcT"}
_SOURCE_TIER_B = {"NIST", "JANAF", "HH", "CRC", "Gurvich", "CCCBDB",
                  "NIST-derived", "CRC/Luo"}
_SOURCE_TIER_C = {"3dMLBE20", "Luo2007", "Morse2019", "Bao2021",
                  "Gong2014", "Casey1993", "Peterson1997"}


def _source_tier(source: str | None) -> str:
    """Quality tier for sources that couldn't be auto-verified.

      A : ATcT-grade — réseau thermochimique pondéré, error bars souvent
          <0.01 eV (haute confiance malgré l'impossibilité de cross-check)
      B : Compendia canoniques — NIST, JANAF, Huber-Herzberg, CRC,
          Gurvich. Compilations de référence depuis ~50 ans.
      C : Papiers individuels — qualité dépendante du papier mais
          clairement attribuée (3dMLBE20, Luo2007, Morse2019, …)
      ? : Source non reconnue ou absente
    """
    if not source:
        return "?"
    if source in _SOURCE_TIER_A: return "A"
    if source in _SOURCE_TIER_B: return "B"
    if source in _SOURCE_TIER_C: return "C"
    return "?"


def _reliability_label(row: dict) -> str:
    """Reliability badge using the combined CCCBDB + Burcat audit when available.

    Combined statuses (when ``combined_status`` is set) :
      🟢🟢 ok_consensus      both sources agree <2 %
      🟢   ok_one_source     one source <2 %, other has no data
      🟡   one_source_fail   one source <2 %, other flags >2 % (revision drift)
      🟠   single_fail       only one source has data and flags >2 %
      🔴   conflict          both sources flag >2 % (real divergence)
      ⚪{tier} n/a           neither source has the molecule, labelled by source tier

    Falls back to CCCBDB-only logic when combined audit is absent.
    """
    cs = row.get("combined_status")
    diff = row.get("combined_best_diff")
    if cs == "ok_consensus":
        return f"🟢🟢 {abs(diff):.1f}%" if diff is not None else "🟢🟢"
    if cs == "ok_one_source":
        return f"🟢 {abs(diff):.1f}%" if diff is not None else "🟢"
    if cs == "one_source_fail":
        return f"🟡 {abs(diff):.1f}%" if diff is not None else "🟡"
    if cs == "single_fail":
        a = abs(diff) if diff is not None else 0
        if a < 5:   return f"🟠 {a:.1f}%"
        if a < 15:  return f"🟠 {a:.1f}%"
        return f"🔴 {a:.0f}%"
    if cs == "conflict":
        a = abs(diff) if diff is not None else 0
        return f"🔴 {a:.1f}%"
    if cs == "n/a":
        return f"⚪ {_source_tier(row.get('source'))}"

    # Fallback : single-source CCCBDB audit
    s = row.get("audit_status") or "n/a"
    diff = row.get("audit_diff_pct")
    if s in ("ok_<1%", "warn_1-2%"):
        return f"🟢 {abs(diff):.1f}%" if diff is not None else "🟢"
    if s == "fail_>2%":
        if diff is None:
            return "🟠 fail"
        a = abs(diff)
        if a < 5:   return f"🟡 {a:.1f}%"
        if a < 15:  return f"🟠 {a:.1f}%"
        return f"🔴 {a:.0f}%"
    return f"⚪ {_source_tier(row.get('source'))}"


def _best_source_url(row) -> str:
    """Pick most authoritative source URL, landing close to the D_at value.

    Priority chain :
      1. url_cccbdb (CCCBDB exp atomization energy page, when CAS present)
      2. NIST WebBook by CAS with Mask matching compound type
      3. NIST WebBook by name with Mask matching compound type

    NIST Mask values :
      - Mask=400 (hex) = Constants of diatomic molecules (D0 visible directly)
      - Mask=1   (hex) = Thermochemistry (heats of formation; D_at derived)

    SMILES-based PubChem URLs avoided (break on 27 % of bracketed SMILES).
    """
    import urllib.parse
    if row.get("url_cccbdb"):
        return row["url_cccbdb"]
    n_atoms = row.get("n_atoms")
    mask = "400" if n_atoms == 2 else "1"
    cas = row.get("cas")
    if cas:
        return (
            f"https://webbook.nist.gov/cgi/cbook.cgi?"
            f"ID={urllib.parse.quote(str(cas))}&Mask={mask}&Units=SI"
        )
    name = row.get("name")
    if name:
        n = urllib.parse.quote(str(name))
        return (
            f"https://webbook.nist.gov/cgi/cbook.cgi?"
            f"Name={n}&Mask={mask}&Units=SI"
        )
    return ""


# ---------------------------------------------------------------------------
# Heavy compute — cached
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _compute_all_benchmark():
    """Compute D_at for every benchmark molecule. Returns a DataFrame."""
    from ptc.topology import build_topology
    from ptc.cascade_v5 import compute_D_at_cascade

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
            "Source URL": _best_source_url(row),
            "Fiabilite": _reliability_label(row),
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

    # ── Header: what PTC computes (full observable coverage) ─────────
    with st.expander("ℹ️ Ce que PTC calcule — toutes les observables, 0 variable d'ajustement", expanded=False):
        st.markdown(
            """
PTC n'est pas spécialisé sur l'énergie d'atomisation. C'est un calculateur
ab initio PT qui dérive **toutes les observables moléculaires** depuis
l'unique input **s = 1/2** + l'ossature des premiers actifs **{2, 3, 5, 7, 11, 13}**.

| Niveau | Observable | Précision actuelle | Statut |
|---|---|---|---|
| **Atomique** (Z = 1 → 118) | IE — énergie d'ionisation | MAE **0,046 %** | ✅ production · 118 éléments |
| | EA — affinité électronique | MAE **0,98 %** | ✅ production · 73 atomes neutres |
| **Liaison** (BondResult) | D₀ — énergie de dissociation | cf. D_at moléculaire | ✅ production |
| | r_e — distance d'équilibre | métrique Bianchi I, sans rayons covalents | ✅ production |
| | ω_e — fréquence vibrationnelle | ~10 % typique sur diatomiques | ✅ production |
| | θ — angles de liaison | fraction de face × angle solide, sans VSEPR | ✅ production |
| **Moléculaire** (TransferResult) | **D_at — atomisation totale** | MAE **3,22 %** (ATcT 994) · **6,55 %** (Burcat 650) | ✅ production · *objet de ce panel* |
| | Décomposition per-face P₁/P₂/P₃ + spectral | σ+π / d-back / ionique par liaison | ✅ production |
| **Aromaticité** (nics.py) | NICS — Pauling-London PT | ~3 ppm sur benzène | ✅ production |
| | Classification σ/π — Hückel signé | aromatique / antiaromatique / radical | ✅ production |
| **Magnétique** (lcao/, GIAO) | σ — tenseur de blindage RMN | ¹H MAE 1,4 ppm (set de Haan) | 🟡 partiel · ¹H ✓ |
| | Cascade post-HF (MP2/CCD/CCSD/Λ-CCSD) | σ_p^CCSD-Λ-GIAO | 🟡 partiel |
| | Densité de courant induite (GIMIC PT-pur) | flux par liaison · NICS Biot-Savart | 🟠 recherche |
| **Topologie** | SMILES → topologie | parser RDKit + reconstruction PT | ✅ production |
| | Formule → topologie (crible variationnel) | choisit l'isomère minimum sans SMILES | ✅ production · base test Burcat |

**Toutes** ces précisions sont mesurées avec exactement la même contrainte :
**0 variable d'ajustement, aucun degré de liberté**, l'unique input s = 1/2
et l'ossature des premiers, avec les principes PT (cycle de phase, bilan
informationnel, cascade, bifurcation q₊/q₋, anti-double-comptage).

### Comparaison à d'autres calculateurs

| Méthode | D_at typique | Temps / mol | Paramètres ajustés |
|---|---|---|---|
| **PTC** | **0,73 eV** (994 ATcT) | **~0,1 s** | **0** (s = 1/2 + premiers) |
| B3LYP/6-31G* | ~0,93 eV (mesuré) | ~minutes | 3 (calibrés G2) |
| B3LYP/def2-TZVP | ~0,15–0,3 eV | ~30 min – heures | 3 |
| DFT moderne (M06-2X, ωB97X) | ~0,1–0,2 eV | ~min – heures | 10–50 |
| Hartree-Fock | ~2 eV (pas de corrélation) | ~minutes | 0 (base param.) |
| MP2 / CCSD / CCSD(T) | 0,02–0,2 eV (gold std) | ~heures – jours | 0 (base param.) |
| G4 / W1 (composites) | ~0,01 eV | ~heures | ~10 extrapolations |
| Champs de force (UFF, MMFF) | non applicable | ~ms | 100–1 000 |
| Potentiels ML (ANI, MACE) | ~0,04–0,1 eV (in-distrib.) | ~ms – s | 10⁵–10⁸ poids |

PTC est, à notre connaissance, le seul calculateur produisant cette
**gamme complète d'observables** (D_at, r_e, ω_e, IE, EA, NICS, σ_RMN)
sans paramètre ajusté. HF et CCSD(T) atteignent aussi techniquement
"0 fit" mais reposent sur le choix d'une base atomique (objet
paramétré). DFT et ML carry des coefficients calibrés sur des données.
            """
        )

    df_all = _compute_all_benchmark()

    # ── Filters : category + verified core ───────────────────────────
    fcol1, fcol2 = st.columns([3, 2])
    with fcol1:
        cat = st.radio(
            "Catégorie",
            ["Toutes", "Organique", "Inorganique", "d-block"],
            horizontal=True,
        )
    with fcol2:
        verified_only = st.checkbox(
            "Verified core uniquement (cross-validé <2 % CCCBDB)",
            value=False,
            help=(
                "Affiche uniquement les molécules dont le D_exp est "
                "cross-validé à <2 % avec CCCBDB. Utile pour des stats "
                "headline propres ; perd les ~30 TM-diatomics canoniques HH "
                "et les 207 mol non auto-vérifiables."
            ),
        )

    df = df_all if cat == "Toutes" else df_all[df_all["category"] == cat]
    if verified_only:
        # Prefer the combined CCCBDB+Burcat status when available
        if "combined_status" in df.columns:
            df = df[df["combined_status"].isin(["ok_consensus", "ok_one_source"])]
        else:
            df = df[df["audit_status"].isin(["ok_<1%", "warn_1-2%"])]
    df_valid = df.dropna(subset=["D_at_PTC"])

    # ── Header stats ──────────────────────────────────────────────────
    n_mol = len(df_valid)
    mae = df_valid["abs_error_pct"].mean() if n_mol else 0.0
    median_err = df_valid["abs_error_pct"].median() if n_mol else 0.0

    c1, c2, c3 = st.columns(3)
    label = "Molécules (verified core)" if verified_only else "Molécules"
    c1.metric(label, f"{n_mol}")
    c2.metric("MAE", f"{mae:.2f} %")
    c3.metric("Médiane |err|", f"{median_err:.2f} %")

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
    st.caption(
        "Le point isolé en haut à droite est le buckminsterfullerène C₆₀ "
        "(D_exp ≈ 416 eV) : zone de stress connue pour PTC sur les systèmes π "
        "étendus à holonomie fermée — l'erreur y atteint +5,7 %, contre <1 % "
        "sur les alcanes voisins. Conservé par honnêteté."
    )

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
    st.markdown("#### Détail par molécule")
    st.caption(
        "Chaque ligne est traçable. La colonne « Source » indique la base d'origine "
        "de la valeur expérimentale (NIST WebBook, ATcT v1.130, JANAF, Huber-Herzberg…). "
        "La colonne « Lien » pointe vers la fiche officielle."
    )

    with st.expander("Comment vérifier une valeur de D_at en cliquant sur un lien"):
        st.markdown(
            "**Aucune base ne publie D_at directement** — toutes publient l'enthalpie "
            "de formation ΔH_f. Les deux sont reliées par "
            "`D_at(M) = Σ ΔH_f(atomes) − ΔH_f(M)`. Selon la base sur laquelle on tombe :\n\n"
            "- **CCCBDB** (cccbdb.nist.gov) — page « Experimental data » qui affiche "
            "  ΔfH°(0K) directement, donc le calcul de D_at se réduit à soustraire les "
            "  ΔfH° atomiques. Liens activés quand notre base contient un CAS validé.\n"
            "- **NIST WebBook** (webbook.nist.gov) — affiche ΔH_f de formation et, pour "
            "  les diatomiques, la « Constants of diatomic molecules » Huber-Herzberg "
            "  qui contient D₀ direct.\n"
            "- **PubChem** (pubchem.ncbi.nlm.nih.gov) — confirme l'identité de la "
            "  molécule (structure, formule). Section « Chemical and Physical "
            "  Properties » contient parfois ΔH_f, sinon cross-link vers NIST WebBook."
        )

    with st.expander("Code couleur de fiabilité — comment chaque D_exp a été vérifié"):
        st.markdown(
            "Chaque valeur expérimentale a été cross-vérifiée contre **deux** "
            "sources thermochimiques indépendantes :\n\n"
            "1. **CCCBDB** (NIST) — page Experimental data, ΔfH°(0K) et "
            "ΔfH°(298K). Audit dual-convention auto-détectant le référentiel.\n"
            "2. **Burcat** (Goos/Burcat/Ruscic, Third Millennium Database) — "
            "fichier flat-file avec ~3000 espèces gas-phase, NASA polynomials, "
            "ΔfH°(298K) souvent annotés ATcT/CODATA/Gurvich.\n\n"
            "**Code couleur combiné** :\n"
            "- 🟢🟢 **Both sources <2 %** : 310 mol (consensus, confiance maximale)\n"
            "- 🟢 **One source <2 %, other n/a** : 236 mol\n"
            "- 🟡 **One <2 %, other >2 %** : 32 mol (drift de révision)\n"
            "- 🟠 **Single source >2 %** : 46 mol\n"
            "- 🔴 **Both sources >2 %** : 16 mol (vrai conflit, à curer)\n"
            "- ⚪ A/B/C **Aucune source** : 359 mol, libellées par tier de la "
            "source originelle (ATcT-grade A / compendium B / paper C)\n\n"
            "**Total cross-validés <2 %** : 546 mol (54.7 %).\n\n"
            "**Pipeline** : enrichissement CAS PubChem → purge des CAS "
            "incorrects → audit CCCBDB dual-convention (avec safeguard "
            "silent-default) → audit Burcat offline → combinaison.\n\n"
            "Détail complet : "
            "[AUDIT_DEXP_REPORT.md](https://github.com/Igrekess/PT_CHEMISTRY/blob/main/benchmarkb3lyp/AUDIT_DEXP_REPORT.md), "
            "CSV per-mol : `benchmarkb3lyp/audit_combined_2026-05-04.csv`."
        )

    cols = ["name", "smiles", "category", "D_at_exp", "D_at_PTC",
            "error_pct", "abs_error_pct"]
    if "source" in df_valid.columns:
        cols += ["source"]
    if "Fiabilite" in df_valid.columns:
        cols += ["Fiabilite"]
    if "Source URL" in df_valid.columns:
        cols += ["Source URL"]
    display_df = df_valid[cols].copy()

    rename_map = {
        "name": "Nom", "smiles": "SMILES", "category": "Catégorie",
        "D_at_exp": "D_at exp (eV)", "D_at_PTC": "D_at PTC (eV)",
        "error_pct": "Erreur %", "abs_error_pct": "|Erreur| %",
        "source": "Source", "Fiabilite": "Fiabilité", "Source URL": "Lien",
    }
    display_df = display_df.rename(columns=rename_map)
    display_df = display_df.sort_values("|Erreur| %", ascending=False).reset_index(drop=True)

    column_config = {
        "D_at exp (eV)": st.column_config.NumberColumn(format="%.3f"),
        "D_at PTC (eV)": st.column_config.NumberColumn(format="%.3f"),
        "Erreur %":      st.column_config.NumberColumn(format="%.2f"),
        "|Erreur| %":    st.column_config.NumberColumn(format="%.2f"),
    }
    if "Lien" in display_df.columns:
        column_config["Lien"] = st.column_config.LinkColumn(
            "Lien", display_text="↗ ouvrir",
            help="Lien vers la référence expérimentale (CCCBDB, NIST WebBook, ou PubChem)",
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        height=500,
        column_config=column_config,
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
        from ptc.cascade_v5 import compute_D_at_cascade
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
