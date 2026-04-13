"""Electrochemistry panel — standard potentials and redox reactions."""
import streamlit as st
from ptc_app.i18n import t


def render_electro_tab():
    st.markdown(f"### {t('electro_title')}")
    st.caption("E(SHE) = -(IE_eff - IE_ref) × f_proj | Cross-channel P₁×P₂ | 0 parametre")

    from ptc_app.benchmark_data import precision_card
    with st.expander("Precision / Reliability", expanded=False):
        st.markdown(precision_card('electrochemistry'))

    from ptc.electrochemistry import (
        standard_potential_SHE, predict_redox, activity_series, compute_potential,
    )
    from ptc.data.experimental import SYMBOLS

    tab1, tab2, tab3 = st.tabs([
        t('activity_series'),
        t('redox_reaction'),
        t('custom_metal'),
    ])

    # ── Tab 1: Activity series ──
    with tab1:
        import pandas as pd

        col_show, _ = st.columns([1, 2])
        with col_show:
            show_all = st.toggle("30 metaux / 30 metals", value=True)

        series = activity_series(include_all=show_all)

        rows = []
        for item in series:
            sym = item['symbol']
            e_pt = item['E_standard']
            e_exp = item.get('E_exp')
            err = abs(e_pt - e_exp) if e_exp is not None else None
            rows.append({
                'Metal': f"{sym}^{item['n_charge']}+/{sym}",
                'E PT (V)': f"{e_pt:+.3f}",
                'E exp (V)': f"{e_exp:+.2f}" if e_exp is not None else "—",
                '|err| (V)': f"{err:.3f}" if err is not None else "—",
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True, height=450)

        # Summary statistics
        errs = [abs(item['E_standard'] - item['E_exp'])
                for item in series if item.get('E_exp') is not None]
        if errs:
            import statistics
            mae = statistics.mean(errs)
            n_good = sum(1 for e in errs if e < 0.2)
            st.caption(
                f"MAE = {mae:.3f} V  |  "
                f"< 0.2V: {n_good}/{len(errs)}  |  "
                f"Max: {max(errs):.3f} V  |  "
                f"0 parametre ajuste"
            )

    # ── Tab 2: Redox reactions ──
    with tab2:
        _SYM_Z = {sym: z for z, sym in SYMBOLS.items()}
        metal_list = sorted(
            {item['symbol'] for item in activity_series(include_all=True)},
            key=lambda s: _SYM_Z.get(s, 999),
        )

        col1, col2 = st.columns(2)
        with col1:
            red_sym = st.selectbox(t('reductant'), metal_list,
                                   index=metal_list.index('Zn') if 'Zn' in metal_list else 0)
            n_red = st.number_input("n (red)", min_value=1, max_value=4, value=2, key="n_red")

        with col2:
            ox_sym = st.selectbox(t('oxidant'), metal_list,
                                  index=metal_list.index('Cu') if 'Cu' in metal_list else 2)
            n_ox = st.number_input("n (ox)", min_value=1, max_value=4, value=2, key="n_ox")

        Z_red = _SYM_Z.get(red_sym, 30)
        Z_ox = _SYM_Z.get(ox_sym, 29)

        r = predict_redox(Z_red, n_red, Z_ox, n_ox)

        c1, c2, c3 = st.columns(3)
        c1.metric("E_cell (V)", f"{r['E_cell']:+.3f}")
        c2.metric("ΔG (eV)", f"{r['Delta_G']:+.2f}")
        spont = t('yes') if r['spontaneous'] else t('no')
        c3.metric(t('spontaneous'), spont)

        if r['spontaneous']:
            st.success(f"{red_sym} + {ox_sym}^{n_ox}+ → {red_sym}^{n_red}+ + {ox_sym}")
        else:
            st.warning(f"Reaction non spontanee / Not spontaneous")

    # ── Tab 3: Custom metal ──
    with tab3:
        st.markdown("Entrez un element et une charge pour calculer E°(SHE).")
        st.markdown("Enter any element and charge to compute E°(SHE).")

        col_z, col_n = st.columns(2)
        with col_z:
            z_input = st.number_input("Z (numero atomique)", min_value=1, max_value=103,
                                      value=26, key="custom_z")
        with col_n:
            n_input = st.number_input("Charge n+", min_value=1, max_value=6,
                                      value=2, key="custom_n")

        from ptc_app.benchmark_data import ptc_timer
        with ptc_timer():
            result = compute_potential(z_input, n_input)

        st.markdown(f"### {result['symbol']}^{result['n_charge']}+ / {result['symbol']}")
        st.markdown(f"**{result['half_reaction']}**")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("E° PT (V)", f"{result['E_standard']:+.3f}")
        if result['E_exp'] is not None:
            c2.metric("E° exp (V)", f"{result['E_exp']:+.3f}")
            c3.metric("|erreur| (V)", f"{result['error']:.3f}")
        else:
            c2.metric("E° exp (V)", "—")
            c3.metric("|erreur|", "—")
        c4.metric("Bloc / Period", f"{result['block']}-block, per {result['period']}")

        # Comparison bar
        if result['E_exp'] is not None:
            import pandas as pd
            chart_df = pd.DataFrame({
                'Source': ['PT', 'Exp'],
                'E° (V)': [result['E_standard'], result['E_exp']],
            })
            st.bar_chart(chart_df.set_index('Source'), height=200)
