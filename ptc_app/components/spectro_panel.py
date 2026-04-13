"""UV-Vis / IR / Raman spectroscopy panel."""
import streamlit as st
from ptc_app.i18n import t


def render_spectro_section(smiles: str):
    """Render spectroscopy section within the Frequencies tab."""
    st.markdown(f"#### {t('uv_title')}")

    from ptc_app.benchmark_data import grade_header
    st.caption(grade_header('uv_vis'))

    try:
        from ptc.spectroscopy import uv_spectrum, ir_spectrum, raman_spectrum, fluorescence

        # UV-Vis
        uv = uv_spectrum(smiles)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(t('homo'), f"{uv.homo:.2f} eV")
        c2.metric(t('lumo'), f"{uv.lumo:.2f} eV")
        c3.metric(t('gap'), f"{uv.gap_eV:.2f} eV")
        c4.metric(t('wavelength'), f"{uv.wavelength_nm:.0f} nm")

        # Fluorescence
        fl = fluorescence(smiles)
        st.caption(f"Fluorescence: {fl.wavelength_nm:.0f} nm (Stokes shift)")

        st.divider()

        # IR spectrum
        st.markdown(f"#### {t('ir_spectrum')}")
        ir = ir_spectrum(smiles)
        if ir.peaks:
            import pandas as pd
            ir_rows = []
            for p in ir.peaks:
                if p.intensity > 0.01:
                    ir_rows.append({
                        'nu (cm-1)': f"{p.frequency:.0f}",
                        'I (rel)': f"{p.intensity:.3f}",
                        'Mode': p.mode,
                        'Assignment': p.assignment,
                    })
            if ir_rows:
                st.dataframe(pd.DataFrame(ir_rows), use_container_width=True, hide_index=True)

                # Bar chart
                freqs = [p.frequency for p in ir.peaks if p.intensity > 0.01]
                intens = [p.intensity for p in ir.peaks if p.intensity > 0.01]
                import altair as alt
                chart_data = pd.DataFrame({'Frequency (cm-1)': freqs, 'Intensity': intens})
                chart = alt.Chart(chart_data).mark_bar(width=3).encode(
                    x='Frequency (cm-1):Q',
                    y='Intensity:Q',
                ).properties(height=200)
                st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No IR-active modes")

        st.divider()

        # Raman spectrum
        st.markdown(f"#### {t('raman_spectrum')}")
        ram = raman_spectrum(smiles)
        if ram.peaks:
            import pandas as pd
            ram_rows = []
            for p in ram.peaks:
                if p.intensity > 0.01:
                    ram_rows.append({
                        'nu (cm-1)': f"{p.frequency:.0f}",
                        'I (rel)': f"{p.intensity:.3f}",
                        'Assignment': p.assignment,
                    })
            if ram_rows:
                st.dataframe(pd.DataFrame(ram_rows), use_container_width=True, hide_index=True)

    except Exception as e:
        st.warning(f"Spectroscopy: {e}")
