"""Explorer panel — discover molecules by scanning chemical space."""
import streamlit as st
from ptc_app.i18n import t


def render_explorer_tab():
    st.markdown(f"### {t('explorer_title')}")
    from ptc_app.benchmark_data import grade_header
    st.caption(f"{t('explorer_desc')} | {grade_header('explorer')}")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        elements_input = st.text_input(
            t('elements'),
            value="C, H, O",
            help="Ex: C, H, O, N",
        )
        elements = [e.strip() for e in elements_input.split(",") if e.strip()]

    with col2:
        max_atoms = st.slider(t('max_atoms'), 2, 10, 6)

    with col3:
        target_options = {
            t('max_dat'): 'max_D_at',
            t('min_dat'): 'min_D_at',
            t('max_dat_per_atom'): 'max_D_at_per_atom',
        }
        target_label = st.selectbox(t('target_property'), list(target_options.keys()))
        target = target_options[target_label]

    if st.button(t('explore_btn'), type="primary", use_container_width=True):
        with st.spinner(t('computing')):
            try:
                from ptc.explorer import explore
                results = explore(
                    elements=elements,
                    max_atoms=max_atoms,
                    target=target,
                    max_formulas=200,
                    timeout_s=30.0,
                )

                if not results:
                    st.warning("Aucun resultat / No results")
                    return

                st.success(f"{len(results)} {t('formulas_found')}")

                # Results table
                import pandas as pd
                rows = []
                for i, r in enumerate(results[:20]):
                    rows.append({
                        t('rank'): i + 1,
                        t('formula'): r['formula'],
                        'D_at (eV)': f"{r['D_at']:.3f}",
                        'D_at/atom': f"{r['D_at_per_atom']:.3f}",
                        t('atoms'): r['n_atoms'],
                        t('bonds'): r['n_bonds'],
                        'Isomeres': r['n_isomers'],
                        'ms': f"{r['time_ms']:.0f}",
                    })
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Best molecule 3D view
                if results:
                    st.markdown(f"**{t('best_isomer')}: {results[0]['formula']}**")
                    best = results[0]
                    try:
                        from ptc.api import Molecule
                        mol = Molecule(best['formula'])
                        from ptc_app.components.mol_viewer import render_viewer
                        render_viewer(mol.mol_block, height=350, style="ballstick", fmt="sdf")
                    except Exception:
                        st.info(f"D_at = {best['D_at']:.3f} eV")

            except Exception as e:
                st.error(f"{t('error')}: {e}")
