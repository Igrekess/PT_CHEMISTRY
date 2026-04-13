"""
i18n — Bilingual FR/EN support for PTC GUI.

Usage:
    from ptc_app.i18n import t, lang_selector

    lang_selector()          # sidebar toggle
    st.write(t("title"))     # translated string
"""
import streamlit as st

# ── Language dictionary ──────────────────────────────────────────────

STRINGS = {
    # ── App-level ──
    "app_title": {
        "fr": "PTC — Simulateur de Chimie Universel",
        "en": "PTC — Universal Chemistry Simulator",
    },
    "app_subtitle": {
        "fr": "*Derivation de la chimie depuis la Theorie de la Persistance.*  \n**Tout depuis s = 1/2.**",
        "en": "*Chemistry derived from Persistence Theory.*  \n**Everything from s = 1/2.**",
    },
    "footer": {
        "fr": "PTC v0.1 beta — Theorie de la Persistance 2026 | par Yan Senez",
        "en": "PTC v0.1 beta — Persistence Theory 2026 | by Yan Senez",
    },

    # ── Tab names ──
    "tab_periodic": {"fr": "Tableau Periodique", "en": "Periodic Table"},
    "tab_atom": {"fr": "Atome", "en": "Atom"},
    "tab_molecule": {"fr": "Molecule", "en": "Molecule"},
    "tab_reactions": {"fr": "Reactions", "en": "Reactions"},
    "tab_catalysis": {"fr": "Catalyse", "en": "Catalysis"},
    "tab_bandgap": {"fr": "Band Gaps", "en": "Band Gaps"},
    "tab_freq": {"fr": "Frequences", "en": "Frequencies"},
    "tab_nmr": {"fr": "RMN", "en": "NMR"},
    "tab_benchmark": {"fr": "Benchmark", "en": "Benchmark"},
    "tab_why": {"fr": "Pourquoi ?", "en": "Why?"},
    "tab_explorer": {"fr": "Explorateur", "en": "Explorer"},
    "tab_solvation": {"fr": "Solution", "en": "Solvation"},
    "tab_electro": {"fr": "Electrochimie", "en": "Electrochemistry"},
    "tab_dft": {"fr": "PTC vs DFT", "en": "PTC vs DFT"},

    # ── Molecule panel ──
    "category": {"fr": "Categorie", "en": "Category"},
    "all_categories": {"fr": "Toutes", "en": "All"},
    "molecule": {"fr": "Molecule", "en": "Molecule"},
    "available": {"fr": "disponibles", "en": "available"},
    "custom_smiles": {"fr": "SMILES personnalise", "en": "Custom SMILES"},
    "compute": {"fr": "Calculer", "en": "Compute"},
    "error_ptc": {"fr": "Erreur PTC", "en": "PTC Error"},
    "details_pt": {"fr": "Details PT", "en": "PT Details"},
    "freq_top3": {"fr": "Frequences (top 3)", "en": "Frequencies (top 3)"},
    "formula": {"fr": "Formule", "en": "Formula"},
    "atoms": {"fr": "Atomes", "en": "Atoms"},
    "bonds": {"fr": "Liaisons", "en": "Bonds"},
    "atom_label": {"fr": "Atome", "en": "Atom"},
    "bond_label": {"fr": "Liaison", "en": "Bond"},

    # ── Explorer panel ──
    "explorer_title": {"fr": "Explorateur de Molecules", "en": "Molecule Explorer"},
    "explorer_desc": {
        "fr": "Explorez l'espace chimique en temps reel. PTC calcule en millisecondes.",
        "en": "Explore chemical space in real time. PTC computes in milliseconds.",
    },
    "elements": {"fr": "Elements", "en": "Elements"},
    "max_atoms": {"fr": "Atomes max", "en": "Max atoms"},
    "target_property": {"fr": "Propriete cible", "en": "Target property"},
    "max_dat": {"fr": "Max D_at (stabilite)", "en": "Max D_at (stability)"},
    "min_dat": {"fr": "Min D_at", "en": "Min D_at"},
    "max_dat_per_atom": {"fr": "Max D_at/atome", "en": "Max D_at/atom"},
    "explore_btn": {"fr": "Explorer", "en": "Explore"},
    "formulas_found": {"fr": "formules trouvees", "en": "formulas found"},
    "rank": {"fr": "Rang", "en": "Rank"},
    "best_isomer": {"fr": "Meilleur isomere", "en": "Best isomer"},

    # ── Solvation panel ──
    "solvation_title": {"fr": "Chimie en Solution", "en": "Solution Chemistry"},
    "solvent": {"fr": "Solvant", "en": "Solvent"},
    "charge": {"fr": "Charge", "en": "Charge"},
    "dielectric": {"fr": "Constante dielectrique", "en": "Dielectric constant"},
    "solvation_energy": {"fr": "Energie de solvatation", "en": "Solvation energy"},
    "cavity_radius": {"fr": "Rayon de cavite", "en": "Cavity radius"},
    "pka_title": {"fr": "pKa des acides", "en": "Acid pKa"},
    "acid": {"fr": "Acide", "en": "Acid"},

    # ── Electrochemistry panel ──
    "electro_title": {"fr": "Electrochimie", "en": "Electrochemistry"},
    "std_potential": {"fr": "Potentiel standard E (SHE)", "en": "Standard potential E (SHE)"},
    "redox_reaction": {"fr": "Reaction redox", "en": "Redox reaction"},
    "spontaneous": {"fr": "Spontanee", "en": "Spontaneous"},
    "activity_series": {"fr": "Serie d'activite", "en": "Activity series"},
    "reductant": {"fr": "Reducteur", "en": "Reductant"},
    "oxidant": {"fr": "Oxydant", "en": "Oxidant"},
    "custom_metal": {"fr": "Metal personnalise", "en": "Custom metal"},

    # ── FeMo panel ──
    "femo_title": {"fr": "FeMo-cofacteur", "en": "FeMo-cofactor"},
    "femo_desc": {
        "fr": "Derivation de S = 3/2 depuis le crible (sin^2 -> J -> H -> S)",
        "en": "Derivation of S = 3/2 from the sieve (sin^2 -> J -> H -> S)",
    },
    "computing": {"fr": "Calcul en cours...", "en": "Computing..."},

    # ── DFT comparison ──
    "dft_title": {"fr": "PTC vs DFT", "en": "PTC vs DFT"},
    "dft_desc": {
        "fr": "Comparaison avec B3LYP/6-311G(d,p) et CCSD(T)/CBS",
        "en": "Comparison with B3LYP/6-311G(d,p) and CCSD(T)/CBS",
    },
    "computation_time": {"fr": "Temps de calcul", "en": "Computation time"},

    # ── Spectroscopy ──
    "uv_title": {"fr": "Spectroscopie UV-Vis", "en": "UV-Vis Spectroscopy"},
    "homo": {"fr": "HOMO", "en": "HOMO"},
    "lumo": {"fr": "LUMO", "en": "LUMO"},
    "gap": {"fr": "Gap", "en": "Gap"},
    "wavelength": {"fr": "Longueur d'onde", "en": "Wavelength"},
    "ir_spectrum": {"fr": "Spectre IR", "en": "IR Spectrum"},
    "raman_spectrum": {"fr": "Spectre Raman", "en": "Raman Spectrum"},

    # ── Common ──
    "exp": {"fr": "Exp.", "en": "Exp."},
    "error": {"fr": "Erreur", "en": "Error"},
    "yes": {"fr": "OUI", "en": "YES"},
    "no": {"fr": "NON", "en": "NO"},
    "coming_soon": {"fr": "A venir", "en": "Coming soon"},
    "modes": {"fr": "modes", "en": "modes"},
}


def lang_selector():
    """Render language toggle in sidebar. Returns current language."""
    lang = st.sidebar.radio(
        "Langue / Language",
        ["Francais", "English"],
        index=0,
        key="ptc_lang",
        horizontal=True,
    )
    return "fr" if lang == "Francais" else "en"


def get_lang() -> str:
    """Get current language from session state."""
    sel = st.session_state.get("ptc_lang", "Francais")
    return "fr" if sel == "Francais" else "en"


def t(key: str) -> str:
    """Translate a string key to the current language."""
    lang = get_lang()
    entry = STRINGS.get(key)
    if entry is None:
        return key
    return entry.get(lang, entry.get("fr", key))
