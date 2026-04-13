"""PTC — Universal Chemistry Simulator GUI."""
import streamlit as st
import sys
import os

# ── path setup ──────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _PROJECT_ROOT)

st.set_page_config(
    page_title="PTC — Persistence Theory Chemistry",
    page_icon="\u269b",
    layout="wide",
)

# ── i18n setup ──────────────────────────────────────────────────────
from ptc_app.i18n import lang_selector, t, get_lang
lang_selector()

st.markdown(
    f"## {t('app_title')}\n"
    f"{t('app_subtitle')}"
)

tab_pt, tab_atom, tab_mol, tab_rxn, tab_cat, tab_femo, tab_gap, tab_freq, tab_nmr, tab_explore, tab_solv, tab_electro, tab_dft, tab_bench, tab_pq = st.tabs(
    [t('tab_periodic'), t('tab_atom'), t('tab_molecule'), t('tab_reactions'),
     t('tab_catalysis'), 'Clusters Fe/Mn', t('tab_bandgap'), t('tab_freq'), t('tab_nmr'),
     t('tab_explorer'), t('tab_solvation'), t('tab_electro'), t('tab_dft'),
     t('tab_benchmark'), t('tab_why')]
)

# ── Load curated best molecules (pre-filtered by lowest error) ──────
@st.cache_data
def _load_curated_molecules():
    """Load curated 500 best molecules (250 organic + 250 inorganic).

    Pre-filtered: organic MAE 0.32%, inorganic MAE 2.57%.
    """
    import json
    path = os.path.join(os.path.dirname(__file__), "curated_molecules.json")
    with open(path) as f:
        return json.load(f)


ALL_MOLS = _load_curated_molecules()

# Experimental angles for common molecules
EXP_ANGLE = {
    "[H]O[H]": 104.5, "[H]N([H])[H]": 107.8, "C": 109.47,
    "C=C": 121.3, "c1ccccc1": 120.0, "O=C=O": 180.0,
}

# ── Atom tab ───────────────────────────────────────────────────────
with tab_atom:
    try:
        from ptc_app.components.atom_panel import render_atom_tab
        render_atom_tab()
    except ImportError:
        st.info(t('coming_soon'))

# ── Molecule tab ────────────────────────────────────────────────────
with tab_mol:
    from ptc_app.benchmark_data import grade_header
    st.caption(grade_header('explorer'))

    left, center, right = st.columns([1.2, 3, 1.5])

    # -- Left: input controls --
    with left:
        # Category filter
        categories = [t('all_categories')] + sorted(set(m['category'] for m in ALL_MOLS.values()))
        cat_filter = st.selectbox(t('category'), categories, key="mol_cat")

        # Filter molecules by category
        if cat_filter == t('all_categories'):
            filtered = ALL_MOLS
        else:
            filtered = {k: v for k, v in ALL_MOLS.items() if v['category'] == cat_filter}

        mol_names = sorted(filtered.keys())
        preset = st.selectbox(
            f"{t('molecule')} ({len(mol_names)} {t('available')})",
            mol_names,
            index=0,
        )
        custom = st.text_input(t('custom_smiles'), value="")

        if custom.strip():
            smiles = custom.strip()
            exp_dat_val = None
        else:
            smiles = filtered[preset]['smiles']
            exp_dat_val = filtered[preset]['D_at']

        calc_btn = st.button(t('compute'), type="primary", use_container_width=True)

    # -- Compute / cache molecule --
    need_compute = calc_btn or "mol" not in st.session_state
    if need_compute or st.session_state.get("_last_smiles") != smiles:
        with st.spinner("PTC..."):
            try:
                import time as _time
                from ptc.api import Molecule
                _t0 = _time.perf_counter()
                mol = Molecule(smiles)
                _dt = (_time.perf_counter() - _t0) * 1000  # ms
                st.session_state["mol"] = mol
                st.session_state["_last_smiles"] = smiles
                st.session_state["_mol_error"] = None
                st.session_state["_mol_time_ms"] = _dt
            except Exception as exc:
                st.session_state["_mol_error"] = str(exc)
                st.session_state.pop("mol", None)

    mol = st.session_state.get("mol")
    mol_err = st.session_state.get("_mol_error")

    if mol_err:
        center.error(f"{t('error_ptc')}: {mol_err}")
    elif mol is not None:
        # -- Center: 3D viewer + 2D structure --
        with center:
            from ptc_app.components.mol_viewer import render_viewer
            render_viewer(mol.mol_block, height=400, style="ballstick", fmt="sdf")

            # 2D skeletal diagram via RDKit
            try:
                from rdkit import Chem
                from rdkit.Chem import Draw
                from io import BytesIO

                rd_mol = Chem.MolFromSmiles(smiles)
                if rd_mol is None:
                    # Try interpreting as formula — build SMILES from topology
                    rd_mol = Chem.MolFromMolBlock(mol.mol_block)
                if rd_mol is not None:
                    img = Draw.MolToImage(rd_mol, size=(250, 140))
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    st.image(buf.getvalue(), caption=mol.formula, width=250)
            except Exception:
                pass  # RDKit not available or molecule can't be drawn

        # -- Right: gauges + info --
        with right:
            from ptc_app.components.gauges import error_gauge

            # D_at gauge — exp value from dataset (all 1114 molecules)
            error_gauge("D_at", mol.D_at, exp_dat_val, unit="eV", fmt=".3f")

            # Angle gauge (first angle if available)
            geom = mol.geometry
            if geom.angles:
                calc_angle = geom.angles[0]["angle"]
                exp_angle = EXP_ANGLE.get(smiles)
                error_gauge("Angle", calc_angle, exp_angle, unit="deg", fmt=".1f")

            # r_e gauge (first bond length)
            if geom.lengths:
                calc_re = geom.lengths[0]["length"]
                error_gauge("r_e", calc_re, None, unit="A", fmt=".4f")

            st.divider()

            # Top 3 frequencies
            freq = mol.frequencies
            if freq and freq.frequencies:
                st.markdown(f"**{t('freq_top3')}**")
                sorted_f = sorted(freq.frequencies, reverse=True)
                for i, f in enumerate(sorted_f[:3]):
                    st.markdown(f"{i+1}. `{f:.0f}` cm-1")
                st.caption(f"ZPE = {freq.ZPE:.4f} eV | {freq.n_modes} {t('modes')}")

            st.divider()

            # Summary info
            st.markdown(f"**{t('formula')}**: {mol.formula}")
            st.markdown(f"**{t('atoms')}**: {geom.n_atoms}")
            n_bonds = len(mol.topology.bonds) if mol.topology else 0
            st.markdown(f"**{t('bonds')}**: {n_bonds}")
            if geom.vsepr_class:
                st.markdown(f"**VSEPR**: {geom.vsepr_class}")

            # Computation time
            dt_ms = st.session_state.get("_mol_time_ms", 0)
            if dt_ms > 0:
                st.caption(f"PTC: {dt_ms:.0f} ms | 0 param")

        # -- Below: detail panel (always visible, two columns) --
        st.markdown(f"#### {t('details_pt')}")
        from ptc_app.components.drawer import atom_detail_drawer, bond_detail_drawer
        topo = mol.topology

        col_atom, col_bond = st.columns(2)

        with col_atom:
            max_idx = geom.n_atoms - 1
            idx_a = st.number_input(
                t('atom_label'), min_value=0, max_value=max_idx,
                value=0, step=1, key="atom_idx",
            )
            atom_detail_drawer(mol, idx_a)

        with col_bond:
            max_b = max(len(topo.bonds) - 1, 0) if topo else 0
            idx_b = st.number_input(
                t('bond_label'), min_value=0, max_value=max_b,
                value=0, step=1, key="bond_idx",
            )
            bond_detail_drawer(mol, idx_b)

# ── Reactions tab ───────────────────────────────────────────────────
with tab_rxn:
    try:
        from ptc_app.components.reaction_panel import render_reaction_tab
        render_reaction_tab()
    except ImportError:
        st.info(t('coming_soon'))

# ── Catalyse tab ───────────────────────────────────────────────────
with tab_cat:
    try:
        from ptc_app.components.catalysis_panel import render_catalysis_tab
        render_catalysis_tab()
    except ImportError:
        st.info(t('coming_soon'))

# ── FeMo-cofactor tab ─────────────────────────────────────────────
with tab_femo:
    try:
        from ptc_app.components.femo_section import render_femo_section
        render_femo_section()
    except ImportError as e:
        st.info(f"{t('coming_soon')} ({e})")

# ── Band Gaps tab ───────────────────────────────────────────────────
with tab_gap:
    try:
        from ptc_app.components.bandgap_panel import render_bandgap_tab
        render_bandgap_tab()
    except ImportError:
        st.info(t('coming_soon'))

# ── Frequences tab ──────────────────────────────────────────────────
with tab_freq:
    try:
        from ptc_app.components.frequency_panel import render_frequency_tab
        render_frequency_tab()
    except ImportError:
        st.info(t('coming_soon'))

    # UV-Vis / Spectroscopy section
    st.divider()
    _smiles_freq = st.session_state.get("_last_smiles", "[H]O[H]")
    try:
        from ptc_app.components.spectro_panel import render_spectro_section
        render_spectro_section(_smiles_freq)
    except ImportError:
        pass

# ── Periodic Table tab ─────────────────────────────────────────────
with tab_pt:
    try:
        from ptc_app.components.periodic_table import render_periodic_tab
        render_periodic_tab()
    except ImportError:
        st.info(t('coming_soon'))

# ── RMN tab ────────────────────────────────────────────────────────
with tab_nmr:
    try:
        from ptc_app.components.nmr_panel import render_nmr_tab
        render_nmr_tab()
    except ImportError:
        st.info(t('coming_soon'))

# ── Explorer tab ───────────────────────────────────────────────────
with tab_explore:
    try:
        from ptc_app.components.explorer_panel import render_explorer_tab
        render_explorer_tab()
    except ImportError:
        st.info(t('coming_soon'))

# ── Solvation tab ──────────────────────────────────────────────────
with tab_solv:
    try:
        from ptc_app.components.solvation_panel import render_solvation_tab
        render_solvation_tab()
    except ImportError as e:
        st.info(f"{t('coming_soon')} ({e})")

# ── Electrochemistry tab ──────────────────────────────────────────
with tab_electro:
    try:
        from ptc_app.components.electro_panel import render_electro_tab
        render_electro_tab()
    except ImportError as e:
        st.info(f"{t('coming_soon')} ({e})")

# ── DFT Comparison tab ────────────────────────────────────────────
with tab_dft:
    try:
        from ptc_app.components.dft_panel import render_dft_tab
        render_dft_tab()
    except ImportError as e:
        st.info(f"{t('coming_soon')} ({e})")

# ── Benchmark tab ──────────────────────────────────────────────────
with tab_bench:
    try:
        from ptc_app.components.benchmark_panel import render_benchmark_tab
        render_benchmark_tab()
    except ImportError:
        st.info(t('coming_soon'))

# ── Pourquoi ? tab ─────────────────────────────────────────────────
with tab_pq:
    try:
        from ptc_app.components.pourquoi_panel import render_pourquoi_tab
        render_pourquoi_tab()
    except ImportError:
        st.info(t('coming_soon'))

# ── Footer ──────────────────────────────────────────────────────────
st.divider()
st.caption(t('footer'))
