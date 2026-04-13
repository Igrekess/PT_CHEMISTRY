"""
viz.py -- Molecular visualization for PTC.

Supports multiple backends:
  - plotly: 3D interactive scatter (default)
  - matplotlib: static 3D plot (fallback)
  - py3dmol: Jupyter-native 3D viewer
  - text: ASCII representation (always available)

April 2026 -- Theorie de la Persistance
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

from ptc.data.experimental import SYMBOLS

if TYPE_CHECKING:
    from ptc.api import Molecule

# ====================================================================
# ATOM COLORS (Jmol/CPK scheme)
# ====================================================================

_COLORS = {
    1: '#FFFFFF', 2: '#D9FFFF', 3: '#CC80FF', 4: '#C2FF00',
    5: '#FFB5B5', 6: '#909090', 7: '#3050F8', 8: '#FF0D0D',
    9: '#90E050', 10: '#B3E3F5', 11: '#AB5CF2', 12: '#8AFF00',
    13: '#BFA6A6', 14: '#F0C8A0', 15: '#FF8000', 16: '#FFFF30',
    17: '#1FF01F', 18: '#80D1E3', 19: '#8F40D4', 20: '#3DFF00',
    26: '#E06633', 29: '#C88033', 30: '#7D80B0', 35: '#A62929',
    53: '#940094',
}

_RADII = {
    1: 0.31, 2: 0.28, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76,
    7: 0.71, 8: 0.66, 9: 0.57, 10: 0.58, 11: 1.66, 12: 1.41,
    13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 18: 1.06,
    19: 2.03, 20: 1.76, 26: 1.32, 29: 1.32, 35: 1.20, 53: 1.39,
}


def _get_color(Z: int) -> str:
    return _COLORS.get(Z, '#808080')


def _get_radius(Z: int) -> float:
    return _RADII.get(Z, 1.0)


# ====================================================================
# PLOTLY BACKEND
# ====================================================================

def _plot_plotly(mol: 'Molecule'):
    """Interactive 3D visualization using plotly."""
    import plotly.graph_objects as go

    geom = mol.geometry
    Z_list = geom.Z_list
    coords = geom.coords
    n = len(Z_list)

    # Atom positions
    xs = [coords[i][0] for i in range(n)]
    ys = [coords[i][1] for i in range(n)]
    zs = [coords[i][2] for i in range(n)]

    # Colors and sizes
    colors = [_get_color(Z_list[i]) for i in range(n)]
    sizes = [_get_radius(Z_list[i]) * 25 + 10 for i in range(n)]
    labels = [SYMBOLS.get(Z_list[i], '?') for i in range(n)]

    fig = go.Figure()

    # Bonds as lines
    for a, b, bo in mol.topology.bonds:
        xa, ya, za = coords[a]
        xb, yb, zb = coords[b]
        fig.add_trace(go.Scatter3d(
            x=[xa, xb], y=[ya, yb], z=[za, zb],
            mode='lines',
            line=dict(color='#404040', width=4 * bo),
            showlegend=False,
            hoverinfo='skip',
        ))

    # Atoms as spheres
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers+text',
        marker=dict(size=sizes, color=colors, line=dict(width=1, color='#333')),
        text=labels,
        textposition='top center',
        textfont=dict(size=12, color='black'),
        hovertext=[f"{labels[i]} ({Z_list[i]})" for i in range(n)],
        showlegend=False,
    ))

    fig.update_layout(
        title=f"PTC | {mol.formula} | D_at = {mol.D_at:.3f} eV | 0 params",
        scene=dict(
            xaxis_title='x (A)',
            yaxis_title='y (A)',
            zaxis_title='z (A)',
            aspectmode='data',
        ),
        width=700, height=600,
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.show()
    return fig


# ====================================================================
# MATPLOTLIB BACKEND
# ====================================================================

def _plot_matplotlib(mol: 'Molecule'):
    """Static 3D visualization using matplotlib."""
    import matplotlib.pyplot as plt

    geom = mol.geometry
    Z_list = geom.Z_list
    coords = geom.coords
    n = len(Z_list)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Bonds
    for a, b, bo in mol.topology.bonds:
        xa, ya, za = coords[a]
        xb, yb, zb = coords[b]
        ax.plot([xa, xb], [ya, yb], [za, zb], 'k-', linewidth=1.5 * bo)

    # Atoms
    for i in range(n):
        x, y, z = coords[i]
        c = _get_color(Z_list[i])
        s = _get_radius(Z_list[i]) * 200 + 50
        ax.scatter(x, y, z, c=c, s=s, edgecolors='k', linewidth=0.5, zorder=5)
        ax.text(x, y, z + 0.3, SYMBOLS.get(Z_list[i], '?'),
                fontsize=9, ha='center', va='bottom')

    ax.set_xlabel('x (A)')
    ax.set_ylabel('y (A)')
    ax.set_zlabel('z (A)')
    ax.set_title(f"PTC | {mol.formula} | D_at = {mol.D_at:.3f} eV")
    plt.tight_layout()
    plt.show()
    return fig


# ====================================================================
# PY3DMOL BACKEND (Jupyter)
# ====================================================================

def _plot_py3dmol(mol: 'Molecule'):
    """Interactive Jupyter viewer using py3Dmol."""
    import py3Dmol

    viewer = py3Dmol.view(width=600, height=400)
    viewer.addModel(mol.xyz, 'xyz')
    viewer.setStyle({'stick': {'radius': 0.12}, 'sphere': {'scale': 0.25}})
    viewer.zoomTo()
    viewer.show()
    return viewer


# ====================================================================
# TEXT BACKEND (always available)
# ====================================================================

def _plot_text(mol: 'Molecule') -> str:
    """ASCII representation of the molecule."""
    geom = mol.geometry
    lines = []
    lines.append(f"=== {mol.formula} ===")
    lines.append(f"D_at = {mol.D_at:.3f} eV | {len(geom.Z_list)} atoms | 0 params")
    lines.append("")

    # Atoms
    lines.append("Atoms:")
    for i, Z in enumerate(geom.Z_list):
        sym = SYMBOLS.get(Z, '?')
        x, y, z = geom.coords[i]
        vsepr = geom.vsepr_class.get(i, '')
        lines.append(f"  {i:3d}  {sym:>2s}  ({x:7.3f}, {y:7.3f}, {z:7.3f})  {vsepr}")

    # Bonds
    lines.append("\nBonds:")
    for info in geom.lengths:
        a, b = info['atoms']
        sa = SYMBOLS.get(geom.Z_list[a], '?')
        sb = SYMBOLS.get(geom.Z_list[b], '?')
        d = info['length']
        bo = info['order']
        bond_char = '=' if bo >= 2 else ('#' if bo >= 3 else '-')
        lines.append(f"  {sa}{bond_char}{sb}  r = {d:.3f} A  (order {bo})")

    # Angles
    if geom.angles:
        lines.append("\nAngles:")
        for info in geom.angles:
            a, c, b = info['atoms']
            sa = SYMBOLS.get(geom.Z_list[a], '?')
            sc = SYMBOLS.get(geom.Z_list[c], '?')
            sb = SYMBOLS.get(geom.Z_list[b], '?')
            lines.append(f"  {sa}-{sc}-{sb}  {info['angle']:.1f} deg")

    output = '\n'.join(lines)
    print(output)
    return output


# ====================================================================
# MAIN ENTRY POINT
# ====================================================================

def plot_molecule_3d(mol: 'Molecule', backend: str = "auto"):
    """3D molecular visualization.

    backend:
      "auto"       - tries plotly -> matplotlib -> text
      "plotly"     - interactive (requires plotly)
      "matplotlib" - static (requires matplotlib)
      "py3dmol"    - Jupyter native (requires py3Dmol)
      "text"       - ASCII (always available)
    """
    if backend == "auto":
        for try_backend in ["plotly", "matplotlib", "text"]:
            try:
                return plot_molecule_3d(mol, backend=try_backend)
            except ImportError:
                continue
        return _plot_text(mol)

    if backend == "plotly":
        return _plot_plotly(mol)
    elif backend == "matplotlib":
        return _plot_matplotlib(mol)
    elif backend == "py3dmol":
        return _plot_py3dmol(mol)
    elif backend == "text":
        return _plot_text(mol)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ====================================================================
# ENERGY DIAGRAM
# ====================================================================

def plot_energy_diagram(molecules: List['Molecule'], labels: Optional[List[str]] = None):
    """Plot energy level diagram for a set of molecules.

    Shows D_at as horizontal bars, useful for comparing isomers
    or reaction pathways.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for energy diagrams")
        return None

    n = len(molecules)
    if labels is None:
        labels = [m.formula for m in molecules]

    energies = [m.D_at for m in molecules]

    fig, ax = plt.subplots(figsize=(max(6, n * 1.5), 5))

    for i, (e, label) in enumerate(zip(energies, labels)):
        ax.barh(i, e, height=0.5, color='steelblue', edgecolor='navy')
        ax.text(e + 0.1, i, f"{e:.3f} eV", va='center', fontsize=10)

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    ax.set_xlabel('D_at (eV)')
    ax.set_title('PTC Energy Diagram | 0 adjustable parameters')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()
    return fig
