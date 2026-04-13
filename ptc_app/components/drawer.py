"""PT detail drawer — shows cascade decomposition for clicked atom/bond."""
import streamlit as st
from ptc.data.experimental import SYMBOLS, IE_NIST, EA_NIST
from ptc.geometry import period_of, classify_geometry


def atom_detail_drawer(mol, atom_idx: int):
    """Show PT details for a clicked atom."""
    topo = mol.topology
    Z = topo.Z_list[atom_idx]
    sym = SYMBOLS.get(Z, "?")

    st.markdown(f"### Atome {sym} (#{atom_idx})")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Z** = {Z}")
        st.markdown(f"**Periode** = {period_of(Z)}")
        st.markdown(f"**Coordination** = {topo.z_count[atom_idx]}")
        st.markdown(f"**Paires libres** = {topo.lp[atom_idx]}")
    with col2:
        ie = IE_NIST.get(Z, 0)
        ea = EA_NIST.get(Z, 0)
        st.markdown(f"**IE** = {ie:.3f} eV (NIST)")
        st.markdown(f"**EA** = {ea:.3f} eV (NIST)")
        vclass = topo.vertex_class[atom_idx] if atom_idx < len(topo.vertex_class) else "?"
        st.markdown(f"**Vertex class** = {vclass}")
        geom = classify_geometry(topo.z_count[atom_idx], topo.lp[atom_idx])
        st.markdown(f"**VSEPR** = {geom}")


def bond_detail_drawer(mol, bond_idx: int):
    """Show PT details for a clicked bond."""
    topo = mol.topology
    if bond_idx >= len(topo.bonds):
        st.warning(f"Bond index {bond_idx} out of range")
        return

    a, b, bo = topo.bonds[bond_idx]
    sym_a = SYMBOLS.get(topo.Z_list[a], "?")
    sym_b = SYMBOLS.get(topo.Z_list[b], "?")

    st.markdown(f"### Liaison {sym_a}-{sym_b} (#{bond_idx})")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Ordre** = {bo}")
        st.markdown(f"**Atomes** = {a} ({sym_a}) — {b} ({sym_b})")

    bonds = mol.bonds
    if bond_idx < len(bonds) and bonds[bond_idx] is not None:
        br = bonds[bond_idx]
        with col2:
            st.markdown(f"**D0** = {br.D0:.3f} eV")
            st.markdown(f"**r_e** = {br.r_e:.4f} A")
            st.markdown(f"**omega_e** = {br.omega_e:.0f} cm-1")

        if br.v_sigma > 0 or br.v_pi > 0 or br.v_ionic > 0:
            import plotly.graph_objects as go
            fig = go.Figure(go.Bar(
                x=["sigma", "pi", "ionique"],
                y=[br.v_sigma, br.v_pi, br.v_ionic],
                marker_color=["#4CAF50", "#2196F3", "#FF9800"],
            ))
            fig.update_layout(
                title="Decomposition energetique",
                yaxis_title="eV",
                height=250,
                margin=dict(l=40, r=20, t=40, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)
