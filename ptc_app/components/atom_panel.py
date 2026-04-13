"""Atom tab — PT-native atom explorer with 3D electron shell visualization."""
import math
import streamlit as st
import streamlit.components.v1 as components

from ptc.atom import IE_eV, EA_eV
from ptc.constants import S_HALF, P1, P2, P3, S3, S5, S7, GAMMA_3, GAMMA_5, GAMMA_7, RY, A_BOHR
from ptc.data.experimental import SYMBOLS, IE_NIST, EA_NIST
from ptc.geometry import period_of
from ptc_app.components.gauges import error_gauge


# ── Electron configuration from PT ──────────────────────────────────

# Shell capacities: 2*P_l positions per circle Z/(2P_l)Z
_SHELL_CAPACITY = {
    's': 2,    # Z/4Z, P0=2
    'p': 6,    # Z/6Z, P1=3
    'd': 10,   # Z/10Z, P2=5
    'f': 14,   # Z/14Z, P3=7
}

# Aufbau filling order (n, l_name)
_AUFBAU = [
    (1, 's'), (2, 's'), (2, 'p'), (3, 's'), (3, 'p'), (4, 's'), (3, 'd'),
    (4, 'p'), (5, 's'), (4, 'd'), (5, 'p'), (6, 's'), (4, 'f'), (5, 'd'),
    (6, 'p'), (7, 's'), (5, 'f'), (6, 'd'), (7, 'p'),
]

_BLOCK_COLORS = {'s': '#ff6b6b', 'p': '#fb923c', 'd': '#60a5fa', 'f': '#34d399'}
_BLOCK_PRIMES = {'s': 2, 'p': 3, 'd': 5, 'f': 7}


def _electron_config(Z: int) -> list:
    """Compute electron configuration via Aufbau.

    Returns list of (n, l_name, n_electrons).
    """
    remaining = Z
    config = []
    for n, l_name in _AUFBAU:
        if remaining <= 0:
            break
        cap = _SHELL_CAPACITY[l_name]
        occ = min(remaining, cap)
        config.append((n, l_name, occ))
        remaining -= occ
    return config


def _config_string(config: list) -> str:
    """Format electron configuration as string: 1s² 2s² 2p⁶ ..."""
    superscripts = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    parts = []
    for n, l, occ in config:
        parts.append(f"{n}{l}{str(occ).translate(superscripts)}")
    return " ".join(parts)


# ── 3D Atom Visualization ───────────────────────────────────────────

def _render_atom_3d(Z: int, config: list, height: int = 400):
    """Render PT atom model: nucleus + concentric polygon shells.

    PT model: atom = concentric circles Z/(2P_l)Z on simplex T₃.
    Each shell is a regular polygon with electrons at vertices.
    """
    # Build shell data: list of (radius, n_electrons, color, label, P_l)
    shells = []
    r = 0.8  # starting radius
    for n, l_name, occ in config:
        color = _BLOCK_COLORS[l_name]
        P_l = _BLOCK_PRIMES[l_name]
        cap = _SHELL_CAPACITY[l_name]
        label = f"{n}{l_name}{occ}"
        shells.append({
            'r': r,
            'n_elec': occ,
            'capacity': cap,
            'color': color,
            'label': label,
            'P_l': P_l,
        })
        r += 0.55

    # Generate JS for 3D visualization using Three.js-like approach
    # Actually use HTML5 Canvas for simplicity (2D projection of concentric circles)
    sym = SYMBOLS.get(Z, f'E{Z}')
    total_r = r

    # Build SVG — full width
    cx, cy = 450, 240  # center
    scale = min(200 / max(total_r, 1), 45)

    svg_parts = []

    # Nucleus
    nuc_r = max(8, min(Z * 0.3, 25))
    svg_parts.append(
        f'<circle cx="{cx}" cy="{cy}" r="{nuc_r}" fill="#333" />'
        f'<text x="{cx}" y="{cy+4}" text-anchor="middle" fill="white" '
        f'font-size="11" font-weight="bold">{sym}</text>'
    )

    # Shells as dashed circles with electron dots
    for sh in shells:
        sr = sh['r'] * scale + nuc_r + 5
        color = sh['color']

        # Shell circle (dashed)
        svg_parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{sr}" fill="none" '
            f'stroke="{color}" stroke-width="1" stroke-dasharray="4,3" opacity="0.5" />'
        )

        # Electrons as dots on the circle (polygon vertices)
        n_e = sh['n_elec']
        cap = sh['capacity']
        for i in range(n_e):
            angle = 2 * math.pi * i / cap - math.pi / 2
            ex = cx + sr * math.cos(angle)
            ey = cy + sr * math.sin(angle)
            svg_parts.append(
                f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="5" fill="{color}" '
                f'stroke="#333" stroke-width="0.5" />'
            )

        # Empty positions (unfilled)
        for i in range(n_e, cap):
            angle = 2 * math.pi * i / cap - math.pi / 2
            ex = cx + sr * math.cos(angle)
            ey = cy + sr * math.sin(angle)
            svg_parts.append(
                f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="4" fill="none" '
                f'stroke="{color}" stroke-width="0.5" stroke-dasharray="2,2" opacity="0.3" />'
            )

        # Label on the vertical axis, to the left — stacked top to bottom
        # Each label at (cx - offset, cy - sr) = top of its circle
        lx = cx - 30
        ly = cy - sr  # top intersection of this shell with the vertical axis
        svg_parts.append(
            f'<text x="{lx:.0f}" y="{ly + 3:.0f}" font-size="8" fill="{color}" '
            f'font-weight="700" text-anchor="end">{sh["label"]}</text>'
        )

    svg_content = "\n".join(svg_parts)

    # Legend
    legend_items = []
    for l_name, color in _BLOCK_COLORS.items():
        P = _BLOCK_PRIMES[l_name]
        legend_items.append(
            f'<span style="color:{color};font-weight:600;margin-right:12px;">'
            f'● {l_name}-block (Z/{2*P}Z, P={P})</span>'
        )
    legend_html = " ".join(legend_items)

    html = f"""<!DOCTYPE html>
<html><head><style>
body {{ margin:0; padding:0; font-family:sans-serif; background:white; overflow:hidden; }}
#container {{ width:100%; height:{height - 50}px; cursor:grab; }}
#container:active {{ cursor:grabbing; }}
svg {{ display:block; }}
</style></head><body>
<div id="container">
<svg id="atomsvg" width="100%" height="{height - 50}" viewBox="0 0 900 {height - 50}">
{svg_content}
</svg>
</div>
<div style="font-size:10px;text-align:center;padding:2px 0;">
{legend_html}
&nbsp; | &nbsp; <span style="color:#999;font-size:0.9em;">Molette = zoom, Clic+glisser = deplacer</span>
</div>
<script>
(function() {{
    var svg = document.getElementById('atomsvg');
    var vb = svg.viewBox.baseVal;
    var isPanning = false, startX, startY;

    // Zoom with mouse wheel
    svg.addEventListener('wheel', function(e) {{
        e.preventDefault();
        var scale = e.deltaY > 0 ? 1.12 : 0.89;
        var pt = svg.createSVGPoint();
        pt.x = e.clientX; pt.y = e.clientY;
        var svgP = pt.matrixTransform(svg.getScreenCTM().inverse());
        vb.x = svgP.x + (vb.x - svgP.x) * scale;
        vb.y = svgP.y + (vb.y - svgP.y) * scale;
        vb.width *= scale;
        vb.height *= scale;
    }}, {{passive: false}});

    // Pan with mouse drag
    svg.addEventListener('mousedown', function(e) {{
        isPanning = true;
        startX = e.clientX; startY = e.clientY;
    }});
    window.addEventListener('mousemove', function(e) {{
        if (!isPanning) return;
        var dx = (e.clientX - startX) * (vb.width / svg.clientWidth);
        var dy = (e.clientY - startY) * (vb.height / svg.clientHeight);
        vb.x -= dx; vb.y -= dy;
        startX = e.clientX; startY = e.clientY;
    }});
    window.addEventListener('mouseup', function() {{ isPanning = false; }});
}})();
</script>
</body></html>"""

    components.html(html, height=height, scrolling=False)


# ── Main Tab ────────────────────────────────────────────────────────

def render_atom_tab():
    """Render the Atom explorer tab."""
    st.subheader("Atome PT")
    st.caption("🟢 **Grade A+** | IE MAE 0.056% (Z=1-103) | EA MAE 1.37% (73 elem) | Ref: NIST | 0 param\n\nGrades evalues sur mesures experimentales (NIST Atomic Spectra Database)")

    # Selector row
    col_z, col_info = st.columns([1, 3])
    with col_z:
        Z = st.number_input("Z", 1, 118, 6, key="atom_Z")
    with col_info:
        sym = SYMBOLS.get(Z, f'E{Z}')
        config = _electron_config(Z)
        last_block = config[-1][1] if config else 's'
        st.markdown(
            f"### {sym} (Z = {Z}) &nbsp; | &nbsp; "
            f"Periode {period_of(Z)} &nbsp; | &nbsp; "
            f"Block {last_block} (Z/{2*_BLOCK_PRIMES[last_block]}Z) &nbsp; | &nbsp; "
            f"{_config_string(config)}"
        )

    # Full-width 3D visualization
    _render_atom_3d(Z, config, height=500)

    # Properties below in 3 columns
    col_ie, col_ea, col_shells = st.columns(3)

    ie_pt = IE_eV(Z)
    ie_nist = IE_NIST.get(Z, 0)
    ea_pt = EA_eV(Z)
    ea_nist = EA_NIST.get(Z, 0)

    with col_ie:
        error_gauge("IE (ionisation)", ie_pt,
                    ie_nist if ie_nist > 0 else None, "eV", ".3f")

    with col_ea:
        error_gauge("EA (affinite)", ea_pt,
                    ea_nist if ea_nist > 0.01 else None, "eV", ".3f")

    with col_shells:
        st.markdown("**Couches**")
        for n, l, occ in config:
            P = _BLOCK_PRIMES[l]
            cap = _SHELL_CAPACITY[l]
            pct = occ / cap
            color = _BLOCK_COLORS[l]
            st.markdown(
                f"<span style='color:{color};font-weight:600'>{n}{l}</span> "
                f"{occ}/{cap} "
                f"<span style='color:#999;font-size:0.8em'>(Z/{2*P}Z)</span>",
                unsafe_allow_html=True,
            )

    # Screening details
    with st.expander("Details screening PT"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Ry** = {RY:.6f} eV")
            st.markdown(f"**a₀** = {A_BOHR:.6f} A")
            st.markdown(f"**sin²₃** = {S3:.6f} (P₁=3)")
            st.markdown(f"**sin²₅** = {S5:.6f} (P₂=5)")
        with c2:
            st.markdown(f"**sin²₇** = {S7:.6f} (P₃=7)")
            st.markdown(f"**γ₃** = {GAMMA_3:.6f}")
            st.markdown(f"**γ₅** = {GAMMA_5:.6f}")
            st.markdown(f"**γ₇** = {GAMMA_7:.6f}")
