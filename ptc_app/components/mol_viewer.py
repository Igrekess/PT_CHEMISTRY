"""3Dmol.js viewer component for Streamlit.

Supports SDF format (with bond orders for double/triple bonds)
and XYZ format (fallback, all single bonds).
"""
import os
import streamlit.components.v1 as components


_STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
_3DMOL_JS = None


def _read_3dmol_js() -> str:
    path = os.path.join(_STATIC_DIR, "3dmol-min.js")
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return ""


def render_viewer(mol_data: str, height: int = 500,
                  style: str = "ballstick",
                  fmt: str = "sdf") -> None:
    """Render a 3Dmol.js viewer.

    Args:
        mol_data: molecule data string (SDF or XYZ format)
        height: viewer height in pixels
        style: "stick", "sphere", "ballstick"
        fmt: "sdf" (with bond orders) or "xyz" (positions only)
    """
    global _3DMOL_JS
    if _3DMOL_JS is None:
        _3DMOL_JS = _read_3dmol_js()

    if style == "ballstick":
        style_js = "{stick:{radius:0.12}, sphere:{scale:0.25}}"
    elif style == "sphere":
        style_js = "{sphere:{scale:0.4}}"
    else:
        style_js = "{stick:{radius:0.15}, sphere:{scale:0.2}}"

    data_escaped = (mol_data
                    .replace("\\", "\\\\")
                    .replace("`", "\\`")
                    .replace("$", "\\$"))

    click_js = """
    viewer.setClickable({}, true, function(atom, viewer, event, container) {
        var data = JSON.stringify({
            type: 'atom_click',
            index: atom.index,
            elem: atom.elem,
            x: atom.x, y: atom.y, z: atom.z
        });
        var el = document.getElementById('click_data');
        if (el) el.textContent = data;
    });
    """

    if _3DMOL_JS:
        script_tag = f"<script>{_3DMOL_JS}</script>"
    else:
        script_tag = '<script src="https://3dmol.org/build/3Dmol-min.js"></script>'

    html = f"""<!DOCTYPE html>
<html>
<head>{script_tag}</head>
<body style="margin:0;padding:0;">
<div id="viewer" style="width:100%;height:{height}px;position:relative;"></div>
<div id="click_data" style="display:none;"></div>
<script>
    var viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "white"}});
    var moldata = `{data_escaped}`;
    viewer.addModel(moldata, "{fmt}");
    viewer.setStyle({{}}, {style_js});
    {click_js}
    viewer.zoomTo();

    // Zoom in closer (scale factor > 1 = more zoomed)
    viewer.zoom(1.4);

    // Rotate 45 deg around X and Y for a nice 3/4 perspective
    viewer.rotate(45, {{x:1, y:0, z:0}});
    viewer.rotate(45, {{x:0, y:1, z:0}});

    viewer.render();

    // Invert mouse wheel zoom direction
    var container = document.getElementById("viewer");
    container.addEventListener("wheel", function(e) {{
        e.preventDefault();
        e.stopPropagation();
        // Invert: scroll up = zoom in (negative delta = zoom in)
        var factor = e.deltaY > 0 ? 0.95 : 1.05;
        viewer.zoom(factor);
        viewer.render();
    }}, {{passive: false, capture: true}});
</script>
</body>
</html>"""
    components.html(html, height=height + 10, scrolling=False)
