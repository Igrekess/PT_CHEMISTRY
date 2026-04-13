"""Solvation panel — solution chemistry from PT."""
import streamlit as st
from ptc_app.i18n import t


def render_solvation_tab():
    st.markdown(f"### {t('solvation_title')}")
    from ptc_app.benchmark_data import grade_header
    st.caption(grade_header('solvation', 'dielectric', 'pka'))

    # Precision badges
    from ptc_app.benchmark_data import precision_card
    with st.expander("Precision / Reliability", expanded=False):
        st.markdown(precision_card('dielectric'))
        st.divider()
        st.markdown(precision_card('solvation'))
        st.divider()
        st.markdown(precision_card('pka'))

    # ── Dielectric constants ──
    st.markdown(f"#### {t('dielectric')}")
    from ptc.solvation import dielectric_constant

    solvents = ['water', 'methanol', 'ethanol', 'dmso', 'acetone', 'hexane']
    exp_eps = {'water': 78.36, 'methanol': 32.7, 'ethanol': 24.5,
               'dmso': 46.7, 'acetone': 20.7, 'hexane': 1.88}

    cols = st.columns(len(solvents))
    for i, solv in enumerate(solvents):
        eps = dielectric_constant(solv)
        exp = exp_eps.get(solv, 0)
        err = abs(eps - exp) / exp * 100 if exp > 0 else 0
        with cols[i]:
            st.metric(solv.capitalize(), f"{eps:.1f}", f"{err:.1f}% err")

    st.divider()

    # ── Ion solvation ──
    st.markdown(f"#### {t('solvation_energy')}")
    col1, col2, col3 = st.columns(3)

    with col1:
        ion_sym = st.selectbox(
            "Ion",
            ['Li', 'Na', 'K', 'Mg', 'Ca', 'Al', 'F', 'Cl', 'Br', 'I',
             'Fe', 'Cu', 'Zn', 'Ag'],
        )
    with col2:
        charge = st.number_input(t('charge'), min_value=-3, max_value=3, value=1)
    with col3:
        solvent = st.selectbox(t('solvent'), solvents, index=0)

    from ptc.solvation import solvation_energy
    from ptc.data.experimental import SYMBOLS

    _SYM_Z = {sym: z for z, sym in SYMBOLS.items()}
    Z = _SYM_Z.get(ion_sym, 11)

    from ptc_app.benchmark_data import ptc_timer
    with ptc_timer():
        r = solvation_energy(Z, charge, solvent)
    c1, c2, c3 = st.columns(3)
    c1.metric("delta_G (eV)", f"{r['delta_G']:+.3f}")
    c2.metric("delta_G (kJ/mol)", f"{r['delta_G_kJ']:+.1f}")
    c3.metric(f"{t('cavity_radius')} (A)", f"{r['r_cavity']:.3f}")

    st.divider()

    # ── pKa (v2: sin²₃ LFER, 25+ acids) ──
    st.markdown(f"#### {t('pka_title')}")
    st.caption("sin²₃ LFER: gas→solution screening = P₁ channel throughput. MAE 0.16 pKa (organic).")

    from ptc.solvation import pka_v2, _ACID_DB_V2

    # Show organic acids (drug design), then inorganic
    organic_keys = ['HCOOH', 'CH3COOH', 'C2H5COOH', 'ClCH2COOH', 'Cl2CHCOOH',
                    'Cl3CCOOH', 'CF3COOH', 'C6H5COOH', 'ASPIRINE', 'IBUPROFENE',
                    'NAPROXENE', 'GLYCINE', 'ALANINE', 'PHENOL', 'PARACETAMOL',
                    'p-NO2-PHENOL', 'NH4+']
    inorganic_keys = ['HF', 'HCl', 'HBr', 'HI', 'H2O', 'H2S', 'HCN', 'HNO2', 'HNO3']

    import pandas as pd

    for label, keys in [("Organic (sin²₃ LFER)", organic_keys),
                        ("Inorganic (Born-PT)", inorganic_keys)]:
        rows = []
        for acid in keys:
            if acid not in _ACID_DB_V2:
                continue
            try:
                r = pka_v2(acid)
                err = abs(r['pKa'] - r['pKa_exp'])
                rows.append({
                    t('acid'): acid,
                    'pKa (PT)': f"{r['pKa']:.2f}",
                    'pKa (exp)': f"{r['pKa_exp']:.2f}",
                    '|Δ|': f"{err:.2f}",
                })
            except Exception:
                pass

        if rows:
            st.markdown(f"**{label}**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
