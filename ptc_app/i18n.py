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
    "tab_aromaticity": {"fr": "Aromaticité", "en": "Aromaticity"},
    "tab_benchmark": {"fr": "Benchmark", "en": "Benchmark"},
    "tab_why": {"fr": "Pourquoi ?", "en": "Why?"},
    "tab_explorer": {"fr": "Explorateur", "en": "Explorer"},
    "tab_solvation": {"fr": "Solution", "en": "Solvation"},
    "tab_electro": {"fr": "Electrochimie", "en": "Electrochemistry"},
    "tab_dft": {"fr": "PTC vs DFT", "en": "PTC vs DFT"},
    "tab_wfn_nmr": {"fr": "RMN ondulatoire", "en": "Wavefunction NMR"},

    # ── Wavefunction NMR panel (Phase 6.B.10/11 cascade) ──
    "wfn_title": {
        "fr": "RMN ondulatoire — cascade post-HF",
        "en": "Wavefunction NMR — post-HF cascade",
    },
    "wfn_subtitle": {
        "fr": "Pipeline HF → MP2 → MP3 → LCCD → CCD → CCSD → Λ → σ_p^CCSD-Λ-GIAO. "
              "Tout dérivé de s = 1/2, 0 paramètre ajusté.",
        "en": "Pipeline HF → MP2 → MP3 → LCCD → CCD → CCSD → Λ → σ_p^CCSD-Λ-GIAO. "
              "All derived from s = 1/2, 0 fitted parameters.",
    },
    "wfn_molecule": {"fr": "Molécule", "en": "Molecule"},
    "wfn_molecule_help": {
        "fr": "Diatomiques sélectionnées pour la cascade complète en quelques secondes.",
        "en": "Diatomics chosen so the full cascade completes in seconds.",
    },
    "wfn_basis": {"fr": "Base", "en": "Basis"},
    "wfn_basis_help": {
        "fr": "SZ : minimal STO PT-pur (rapide). DZ/DZP : split-valence + polarisation.",
        "en": "SZ : minimal PT-pure STO (fast). DZ/DZP : split-valence + polarisation.",
    },
    "wfn_grid": {"fr": "Grille radiale", "en": "Radial grid"},
    "wfn_grid_help": {
        "fr": "Nombre de points radiaux pour les intégrales 3D (plus = plus précis, plus lent).",
        "en": "Number of radial points for 3D integrals (more = more precise, slower).",
    },
    "wfn_run": {"fr": "Lancer la cascade", "en": "Run cascade"},
    "wfn_press_run": {
        "fr": "Choisis molécule + base + grille puis clique sur **Lancer la cascade**.",
        "en": "Pick molecule + basis + grid, then click **Run cascade**.",
    },
    "wfn_running": {
        "fr": "Cascade en cours (HF → MP2 → MP3 → CCSD → Λ → σ_p)…",
        "en": "Cascade running (HF → MP2 → MP3 → CCSD → Λ → σ_p)…",
    },
    "wfn_done": {"fr": "Cascade complète ✓", "en": "Cascade complete ✓"},
    "wfn_n_orb":  {"fr": "n orbitales", "en": "n orbitals"},
    "wfn_n_occ":  {"fr": "n occupées",  "en": "n occupied"},
    "wfn_n_virt": {"fr": "n virtuelles", "en": "n virtual"},
    "wfn_t1_norm":{"fr": "‖T1‖ CCSD",   "en": "‖T1‖ CCSD"},
    "wfn_energies": {
        "fr": "Hiérarchie d'énergie de corrélation",
        "en": "Correlation-energy hierarchy",
    },
    "wfn_method":   {"fr": "Méthode",  "en": "Method"},
    "wfn_e_corr":   {"fr": "E_corr (eV)", "en": "E_corr (eV)"},
    "wfn_delta_pct":{"fr": "Δ vs MP2 (%)", "en": "Δ vs MP2 (%)"},
    "wfn_detail":   {"fr": "Détail",   "en": "Detail"},
    "wfn_lambda":   {"fr": "Λ-équations (multiplicateurs CCSD)",
                     "en": "Λ-equations (CCSD multipliers)"},
    "wfn_lambda_iters":  {"fr": "Itérations Λ",   "en": "Λ iterations"},
    "wfn_lambda_drift":  {"fr": "‖Λ − T‖",        "en": "‖Λ − T‖"},
    "wfn_lambda_drift_help": {
        "fr": "Mesure de la non-trivialité Λ ≠ T (B-diagram λ-T cross-coupling).",
        "en": "Measures the Λ ≠ T non-triviality (B-diagram λ-T cross-coupling).",
    },
    "wfn_sigma_p":  {"fr": "Blindage paramagnétique σ_p (sonde 0.1 Å au-dessus du 1er noyau)",
                     "en": "Paramagnetic shielding σ_p (probe 0.1 Å above 1st nucleus)"},
    "wfn_sigma_p_delta": {"fr": "Effet de corrélation total",
                          "en": "Total correlation effect"},
    "wfn_timings":  {"fr": "Temps de calcul (étape par étape)",
                     "en": "Timings (step by step)"},
    "wfn_theory_note": {"fr": "Note théorique — cascade PT post-HF",
                        "en": "Theory note — post-HF PT cascade"},
    "wfn_theory_body": {
        "fr": (
            "**MP2** : 2nd ordre Møller-Plesset, amplitudes "
            "$t_{ij}^{ab} = (ia|jb)/\\Delta\\varepsilon$.\n\n"
            "**MP3** : ajoute la correction d'ordre 3 — pp ladder, "
            "hh ladder, et le ring (Phase 6.B.10).\n\n"
            "**LCCD** : itération infinie linéaire des termes ladder + ring "
            "(Phase 6.B.11a/b avec DIIS).\n\n"
            "**CCD** : ajoute les termes T2² (intermédiaires Fock-like F_oo, F_vv, "
            "Phase 6.B.11c) — renormalise l'over-correlation de LCCD.\n\n"
            "**CCSD** : ajoute T1 (singles) avec sources 1a/1e/1f/F_kc "
            "(Phase 6.B.11d-d-bis-bis). Sur HF canonique, ‖T1‖ ~ 10⁻⁴ "
            "(théorème de Brillouin → singles d'ordre 2 en α).\n\n"
            "**Λ** : multiplicateurs de réponse pour les propriétés. "
            "À canonical CCSD, Λ ≠ T via le B-diagram λ-T (Phase 6.B.11e-bis-bis).\n\n"
            "**σ_p^CCSD-Λ-GIAO** : blindage paramagnétique calculé sur orbitales "
            "Λ-relaxées (Phase 6.B.11f). Approximation Λ-uncoupled : "
            "diagonale T-Λ symétrique, off-diagonal Z-block deferred."
        ),
        "en": (
            "**MP2** : 2nd-order Møller-Plesset, amplitudes "
            "$t_{ij}^{ab} = (ia|jb)/\\Delta\\varepsilon$.\n\n"
            "**MP3** : adds the 3rd-order amplitude correction — pp ladder, "
            "hh ladder, ring (Phase 6.B.10).\n\n"
            "**LCCD** : infinite-order linear iteration of ladder + ring "
            "(Phase 6.B.11a/b with Pulay DIIS).\n\n"
            "**CCD** : adds the T2² terms (Fock-like intermediates F_oo, F_vv, "
            "Phase 6.B.11c) — renormalises the LCCD over-correlation.\n\n"
            "**CCSD** : adds T1 (singles) with sources 1a/1e/1f/F_kc "
            "(Phase 6.B.11d-d-bis-bis). On canonical HF, ‖T1‖ ~ 10⁻⁴ "
            "(Brillouin's theorem → singles emerge at 2nd order in α).\n\n"
            "**Λ** : response multipliers for properties. At canonical CCSD, "
            "Λ ≠ T via the B-diagram λ-T cross-coupling (Phase 6.B.11e-bis-bis).\n\n"
            "**σ_p^CCSD-Λ-GIAO** : paramagnetic shielding computed on "
            "Λ-relaxed orbitals (Phase 6.B.11f). Λ-uncoupled approximation : "
            "T-Λ symmetric diagonal blocks, off-diagonal Z-block deferred."
        ),
    },

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
