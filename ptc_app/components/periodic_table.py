"""Periodic table tab — interactive PT element explorer with IE/EA calculator."""
import streamlit as st
from ptc.atom import IE_eV, EA_eV
from ptc.data.experimental import IE_NIST, EA_NIST, SYMBOLS
from ptc.geometry import period_of
from ptc_app.components.gauges import error_gauge

# ── Periodic table layout (period, group) -> Z ─────────────────────
# PT prédiction P21 : pas de bloc g (9 = 3² composite, γ₁₁ < 1/2).
# Le tableau se ferme structurellement au bloc f à Z=118.
# Ce layout 32-col existe pour visualiser la ZONE DE FALSIFICATION
# (Z = 119–214) : ce que les formules atomiques extrapoleraient SI
# le bloc g existait. Toute synthèse stable d'un tel élément avec
# chimie g cohérente falsifierait P21. Voir /predictions et
# /demonstration-tableau-periodique sur le site.

# We use a WIDE grid (32 columns) where standard elements map to
# the same relative positions and extended elements fill g/f blocks.
_N_COLS = 32
_PT_LAYOUT = {}

def _std(row, grp, Z):
    """Map standard 18-group position to 32-column grid.

    32-col layout: s(1-2) | f(3-16) | d(17-26) | p(27-32)
    """
    if grp <= 2:
        _PT_LAYOUT[(row, grp)] = Z       # s-block: cols 1-2
    elif grp <= 12:
        _PT_LAYOUT[(row, grp + 14)] = Z  # d-block: 3->17, 12->26
    else:
        _PT_LAYOUT[(row, grp + 14)] = Z  # p-block: 13->27, 18->32

# Period 1
_std(1, 1, 1); _std(1, 18, 2)
# Period 2
_std(2, 1, 3); _std(2, 2, 4)
for i, Z in enumerate(range(5, 11)): _std(2, 13+i, Z)
# Period 3
_std(3, 1, 11); _std(3, 2, 12)
for i, Z in enumerate(range(13, 19)): _std(3, 13+i, Z)
# Period 4
_std(4, 1, 19); _std(4, 2, 20)
for i, Z in enumerate(range(21, 31)): _std(4, 3+i, Z)
for i, Z in enumerate(range(31, 37)): _std(4, 13+i, Z)
# Period 5
_std(5, 1, 37); _std(5, 2, 38)
for i, Z in enumerate(range(39, 49)): _std(5, 3+i, Z)
for i, Z in enumerate(range(49, 55)): _std(5, 13+i, Z)
# Period 6
_std(6, 1, 55); _std(6, 2, 56)
for i, Z in enumerate(range(57, 71)): _PT_LAYOUT[(6, 3+i)] = Z    # f-block: cols 3-16
for i, Z in enumerate(range(71, 81)): _PT_LAYOUT[(6, 17+i)] = Z   # d-block: cols 17-26
for i, Z in enumerate(range(81, 87)): _PT_LAYOUT[(6, 27+i)] = Z   # p-block: cols 27-32
# Period 7
_std(7, 1, 87); _std(7, 2, 88)
for i, Z in enumerate(range(89, 103)): _PT_LAYOUT[(7, 3+i)] = Z   # f-block: cols 3-16
for i, Z in enumerate(range(103, 113)): _PT_LAYOUT[(7, 17+i)] = Z  # d-block: cols 17-26
for i, Z in enumerate(range(113, 119)): _PT_LAYOUT[(7, 27+i)] = Z  # p-block: cols 27-32

# Pas de périodes 8, 9, 10 dans le layout visuel : PT (P21) prédit que
# le tableau se ferme structurellement à Z=118. Au-delà, c'est la zone
# de falsification, et la dessiner sous forme de cellules contredit
# visuellement la prédiction. Pour calculer IE/EA d'un Z hypothétique
# > 118, utiliser le champ « ou Z personnalisé » plus bas — qui affiche
# un avertissement explicite que c'est une extrapolation hors-domaine.

# Total rows for rendering : 7 périodes connues
_N_ROWS = 7

# Éléments au-delà de Z=118 — zone de falsification PT (P21 : pas de bloc g)
# Conservé comme set pour les fonctions d'analyse (avertissements dans
# _element_detail / _stability_analysis), pas pour le rendu du tableau.
_PT_FALSIF_ZONE = set(range(119, 215))
_PT_PREDICTED = _PT_FALSIF_ZONE  # alias rétro-compatible
# Pics extrapolés des formules atomiques (maxima locaux d'IE au-delà de Z=118).
# Ce ne sont PAS des prédictions de stabilité chimique en PT — ce sont les
# valeurs que les équations donneraient si le bloc g existait.
_PT_FORMULA_PEAKS = {144, 164, 194}
_PT_ISLANDS = _PT_FORMULA_PEAKS  # alias rétro-compatible

# Block colors
def _block_color(Z):
    """Return CSS background color based on element block."""
    if Z <= 0:
        return "transparent"
    # Islands get special gold color
    if Z in _PT_ISLANDS:
        return "#ffd700"  # gold
    # PT predictions get desaturated versions
    if Z in _PT_PREDICTED:
        return "#b0b0b0"  # gray for predicted elements
    # Known elements
    # s-block
    if Z in (1, 2, 3, 4, 11, 12, 19, 20, 37, 38, 55, 56, 87, 88):
        return "#ff6b6b"  # red
    # Noble gases
    if Z in (2, 10, 18, 36, 54, 86, 118):
        return "#c084fc"  # purple
    # Halogens
    if Z in (9, 17, 35, 53, 85, 117):
        return "#fbbf24"  # yellow
    # d-block
    if 21 <= Z <= 30 or 39 <= Z <= 48 or 71 <= Z <= 80 or 103 <= Z <= 112:
        return "#60a5fa"  # blue
    # f-block
    if 57 <= Z <= 70 or 89 <= Z <= 102:
        return "#34d399"  # green
    # p-block
    return "#fb923c"  # orange


def _render_periodic_table():
    """Render the interactive periodic table — Z=1 to 118, aligned with P21.

    PT (P21) predicts no g-block: the table closes structurally at f.
    Z > 118 is the falsification zone (handled via the manual Z input,
    not displayed in the visual table). See _stability_analysis() and
    the explicit warning in _element_detail() for hypothetical Z > 118.
    """
    cells = []
    for per in range(1, _N_ROWS + 1):
        for grp in range(1, _N_COLS + 1):
            Z = _PT_LAYOUT.get((per, grp))
            if Z is None:
                cells.append('<div class="pt-cell pt-empty"></div>')
            else:
                sym = SYMBOLS.get(Z, f'{Z}')
                bg = _block_color(Z)
                cells.append(
                    f'<div class="pt-cell" style="background:{bg};" '
                    f'title="{sym} (Z={Z})">'
                    f'<span class="pt-z">{Z}</span>'
                    f'<span class="pt-sym">{sym}</span>'
                    f'</div>'
                )

    # Legend
    legend = """
<div style="display:flex;gap:12px;flex-wrap:wrap;margin:8px 0;font-size:11px;">
  <span><span style="display:inline-block;width:14px;height:14px;background:#ff6b6b;border-radius:2px;vertical-align:middle;"></span> s-block</span>
  <span><span style="display:inline-block;width:14px;height:14px;background:#fb923c;border-radius:2px;vertical-align:middle;"></span> p-block</span>
  <span><span style="display:inline-block;width:14px;height:14px;background:#60a5fa;border-radius:2px;vertical-align:middle;"></span> d-block</span>
  <span><span style="display:inline-block;width:14px;height:14px;background:#34d399;border-radius:2px;vertical-align:middle;"></span> f-block</span>
  <span><span style="display:inline-block;width:14px;height:14px;background:#c084fc;border-radius:2px;vertical-align:middle;"></span> gaz nobles</span>
</div>"""

    html = f"""<!DOCTYPE html>
<html><head><style>
body {{ margin: 0; padding: 4px; font-family: sans-serif; }}
.pt-grid {{
    display: grid;
    grid-template-columns: repeat({_N_COLS}, 1fr);
    gap: 1px;
    max-width: 100%;
}}
.pt-cell {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    border-radius: 2px;
    min-height: 28px;
    padding: 1px;
}}
.pt-cell:hover {{ opacity: 0.7; }}
.pt-empty {{ background: transparent; }}
.pt-z {{ font-size: 7px; color: #333; line-height: 1; }}
.pt-sym {{ font-size: 10px; font-weight: 700; color: #111; line-height: 1.1; }}
</style></head><body>
{legend}
<div class="pt-grid">
""" + "\n".join(cells) + """
</div>
<div style="margin-top:6px;font-size:10px;color:#666;">
  118 elements (Z=1-118). IE et EA calcules depuis s = 1/2. PT (P21) predit que le tableau se ferme au bloc f.
</div>
</body></html>"""

    import streamlit.components.v1 as components
    components.html(html, height=330, scrolling=False)


def _stability_analysis(Z: int, ie: float) -> dict:
    """PT stability prediction for element Z.

    PT basis: stability is governed by the screening cascade on T³.
    - IE measures how tightly the outermost electron is bound.
    - For Z > 118, relativistic contraction competes with shell expansion.
    - The critical Z (Feynman limit) is Z_c = 1/alpha ~ 137 in classical QED.
    - In PT: Z_c = 1/alpha_nu = 136.28 (bare coupling, no dressing).
    - Beyond Z_c, the 1s orbital dives into the Dirac sea.

    Stability indicators:
    - IE > Ry/P1 (Shannon cap): strongly bound, stable
    - IE > 1 eV: bound, potentially synthesizable
    - IE < 1 eV: weakly bound, very short-lived
    - Z > 137: PT predicts orbital collapse (supercritical)
    """
    from ptc.constants import RY, P1, AEM

    # NB : ces "pics" sont des MAXIMA EXTRAPOLÉS des formules atomiques au-delà
    # de Z=118, dans une zone que PT (P21) prédit comme falsifiable. Ce ne
    # sont PAS des prédictions de stabilité chimique au sens PT.
    FORMULA_PEAKS = {
        144: ("IE = 9.5 eV — pic extrapole de la formule atomique. "
              "PT (P21) predit qu'AUCUN bloc g n'existe : si Z=144 etait "
              "synthetise avec une chimie coherente avec la couche g, P21 "
              "serait falsifiee."),
        164: ("IE = 10.2 eV — pic extrapole. Meme statut : zone de "
              "falsification PT, pas une prediction positive de stabilite."),
        194: ("IE = 10.0 eV — dernier maximum local de la formule etendue. "
              "PT ne predit aucune stabilite reelle au-dela de Z=118."),
    }
    ISLANDS = FORMULA_PEAKS  # alias rétro-compatible

    if Z <= 118:
        if Z <= 82:
            status = "Stable ou quasi-stable"
            color = "#28a745"
        elif Z <= 103:
            status = "Radioactif (demi-vie variable)"
            color = "#ffc107"
        else:
            status = "Superheavy — synthetise, tres instable"
            color = "#fd7e14"
    elif Z in FORMULA_PEAKS:
        status = f"PIC EXTRAPOLE (zone de falsification PT) — {FORMULA_PEAKS[Z]}"
        color = "#2196F3"
    elif Z <= 214:
        # Zone de falsification : PT predit AUCUN element avec chimie g.
        # Les valeurs IE sont des extrapolations numériques des formules
        # atomiques au-dela de leur domaine de validite PT.
        if ie > 5.0:
            status = f"Zone de falsification PT (P21) : IE extrapolee = {ie:.1f} eV. PT predit pas de chimie g distinctive."
            color = "#fd7e14"
        elif ie > 2.0:
            status = f"Zone de falsification PT : IE extrapolee = {ie:.1f} eV (faible)."
            color = "#fd7e14"
        else:
            status = f"Zone de falsification PT : IE extrapolee = {ie:.1f} eV (tres faible)."
            color = "#dc3545"
    else:
        status = (f"Au-dela du tableau etendu (Z>214). "
                  f"IE = {ie:.1f} eV — extrapolation hors domaine.")
        color = "#dc3545"

    return {"status": status, "color": color, "Z_feynman": 0}


def _element_detail(Z: int):
    """Show detailed IE/EA comparison for element Z."""
    sym = SYMBOLS.get(Z, f'E{Z}')
    is_known = Z <= 118
    st.markdown(f"### {sym} (Z = {Z})" if is_known else f"### Element {Z} (hypothetique)")

    ie_pt = IE_eV(Z)
    ea_pt = EA_eV(Z)

    ie_nist = IE_NIST.get(Z, 0) if is_known else None
    ea_nist = EA_NIST.get(Z, 0) if is_known else None

    col1, col2 = st.columns(2)
    with col1:
        exp_ie = ie_nist if ie_nist and ie_nist > 0 else None
        error_gauge("IE (Ionisation)", ie_pt, exp_ie, "eV", ".3f")
    with col2:
        exp_ea = ea_nist if ea_nist and ea_nist > 0.01 else None
        error_gauge("EA (Affinite)", ea_pt, exp_ea, "eV", ".3f")

    st.markdown(f"**Periode** : {period_of(Z)}")

    # Stability analysis
    stab = _stability_analysis(Z, ie_pt)
    st.markdown(
        f"""<div style="
            background:{stab['color']}22;
            border-left:4px solid {stab['color']};
            padding:8px 12px;
            border-radius:4px;
            margin-top:8px;
        ">
            <strong>Stabilite :</strong> {stab['status']}
        </div>""",
        unsafe_allow_html=True,
    )

    if Z > 118:
        st.warning(
            "**Z > 118 : zone de falsification PT (P21).** "
            "PT prédit qu'aucun élément n'existe avec une chimie de type g "
            "au-delà du bloc f. La valeur calculée ci-dessus est une "
            "**extrapolation numérique des formules atomiques au-delà de leur "
            "domaine de validité PT** — pas une prédiction de stabilité. Si "
            "un tel élément est synthétisé un jour avec une chimie 5g^k "
            "cohérente, P21 est falsifiée.",
            icon="⚠️",
        )


def render_periodic_tab():
    """Render the full periodic table tab."""
    st.subheader("Tableau Periodique PT")
    st.caption("IE et EA calcules depuis s = 1/2. MAE(IE) = 0.057%.")

    _render_periodic_table()

    st.markdown("---")

    # Element selector — searchable dropdown + manual Z input
    # Le dropdown ne liste que Z = 1–118 (élements connus, alignés avec P21).
    # Le champ « ou Z personnalisé » plus bas permet d'extrapoler à n'importe
    # quel Z, avec un avertissement de zone de falsification dans la fiche.
    col_sel, col_detail = st.columns([1, 3])
    with col_sel:
        # Build element list for dropdown — Z = 1 à 118
        elem_options = []
        for z in range(1, 119):
            sym = SYMBOLS.get(z, f'E{z}')
            label = f"{z} - {sym}"
            elem_options.append(label)

        selected = st.selectbox(
            "Element (recherche par nom ou Z)",
            elem_options,
            index=0,
            key="pt_select",
        )
        Z = int(selected.split(" - ")[0])
        sym = SYMBOLS.get(Z, f'E{Z}')

        # Also allow manual Z for > 194
        custom_Z = st.number_input("ou Z personnalise", 1, 300, Z, key="pt_Z_custom")
        if custom_Z != Z:
            Z = custom_Z
            sym = SYMBOLS.get(Z, f'E{Z}')

        if Z > 118:
            st.caption("Element hypothetique (extrapol. PT)")

    with col_detail:
        _element_detail(Z)

    # PT derivation
    with st.expander("Derivation PT du tableau periodique"):
        st.markdown("""
**L'atome PT est un ensemble de cercles discrets concentriques** sur le simplexe T₃ :

| Cercle | Dimension | Block | Capacite |
|--------|-----------|-------|----------|
| Z/(2P₀)Z = Z/4Z | l=0 | s-block | 2 positions |
| Z/(2P₁)Z = Z/6Z | l=1 | p-block | 6 positions |
| Z/(2P₂)Z = Z/10Z | l=2 | d-block | 10 positions |
| Z/(2P₃)Z = Z/14Z | l=3 | f-block | 14 positions |

ou P₀=2, P₁=3, P₂=5, P₃=7 sont les premiers actifs du crible.

**Pipeline IE** (derive de s = 1/2) :
```
IE = Ry × (Z_eff / n)² × M/(M + mₑ)
```
ou :
- **Ry = alpha² × mₑc²/2** : Rydberg (derive de s = 1/2)
- **Z_eff = Z × exp(-S)** : charge effective apres screening
- **S = S_core + S_polygon × D + S_rel** : screening total
  - S_core : ecrantage des couches internes (cascade sur T³)
  - S_polygon : geometrie du polygone Z/(2P_l)Z (dim 0 + dim 1 + dim 2)
  - S_rel : corrections relativistes (Z > 50)

**Les 3 dimensions du polygone** :
- **dim 0 (vertex)** : echange d'electrons sur les sommets du polygone
- **dim 1 (edge)** : propagation le long des aretes (couplage voisin)
- **dim 2 (face)** : correction circulaire (fermeture du polygone)

**Resultat** : IE calcule pour Z = 1 a 103 avec MAE = 0.057%.
Le screening alpha = sin²(pi/3) × sin²(pi/5) × sin²(pi/7) = 1/137
est le MEME alpha qui donne la constante de structure fine.

**EA** : affinite electronique calculee par le moteur geometrique
ShellPolygon (amplitudes capture/ejection sur le polygone).
MAE = 1.37% sur 73 elements.

**Au-dela de Z = 118 — la prediction P21 (pas de bloc g)** :

PT predit que **le tableau periodique se ferme structurellement au
bloc f**. La raison est arithmetique : l'identite 2l+1 = premier
tient pour l = 1, 2, 3 (donnant les premiers 3, 5, 7 = blocs p, d, f)
mais casse a l = 4 puisque 9 = 3² n'est pas premier. Et le premier
suivant, p = 11, a gamma_11 = 0.426 < 1/2 — il est inactif au point
fixe mu* = 15. Deux barrieres independantes interdisent le bloc g.

**Consequence** : aucun element Z > 118 n'a de chimie de type g.
La synthese stable d'un seul tel element (avec configuration 5g^k
pertinente pour la liaison) **falsifierait P21**.

Les elements Z = 119–214 affiches ici (case « Afficher elements
extrapoles ») sont donc dans la **zone de falsification**, pas une
zone de prediction. Les valeurs IE sont l'extrapolation NUMERIQUE
des formules atomiques au-dela de leur domaine de validite PT —
elles montrent ce que les equations *donneraient* si le bloc g
existait, ce qui est precisement ce que l'experience devra contredire.

**Pics extrapoles** (maxima locaux de la formule, NON des ilots PT) :
- **Z = 144** : IE = 9.50 eV (pic numerique, pas de signification PT)
- **Z = 164** : IE = 10.17 eV (maximum local)
- **Z = 194** : IE = 9.98 eV (dernier pic)

PT ne predit aucun element stable avec chimie distinctive au-dela
de Z = 118. Le tableau s'etend jusqu'a Z = 214 uniquement pour
visualiser la zone de test.

**Reference sur le site** :
- `/predictions` (P21–P23 : statut de prediction falsifiable)
- `/demonstration-tableau-periodique` (derivation formelle)
- `/essays/why-periodic-table` (vulgarise)
""")

    # Full IE/EA table
    with st.expander("Tableau complet IE/EA (Z = 1-118)"):
        import pandas as pd
        rows = []
        for z in range(1, 119):
            ie_pt = IE_eV(z)
            ie_n = IE_NIST.get(z, 0)
            ie_err = abs(ie_pt - ie_n) / ie_n * 100 if ie_n > 0 else 0
            ea_pt = EA_eV(z)
            ea_n = EA_NIST.get(z, 0)
            ea_err = abs(ea_pt - ea_n) / ea_n * 100 if ea_n > 0.01 else 0
            rows.append({
                'Z': z,
                'Sym': SYMBOLS.get(z, '?'),
                'IE_PT': round(ie_pt, 3),
                'IE_NIST': round(ie_n, 3),
                'IE_err%': round(ie_err, 3),
                'EA_PT': round(ea_pt, 3),
                'EA_NIST': round(ea_n, 3),
                'EA_err%': round(ea_err, 1),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=400)
