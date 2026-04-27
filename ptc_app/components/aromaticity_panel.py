"""Aromaticity panel — NICS, σ/π split, full PT experimental signature."""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from ptc.api import Molecule
from ptc_app.i18n import t


# Curated benchmark molecules with known NICS values
# A SMILES of "@@cluster:<key>" is a sentinel for an explicit-coordinate
# cluster that is built via ``ptc/lcao/cluster.py`` instead of the
# SMILES → Topology pipeline.  These exercise the full GIMIC PT-pure
# stack (cluster builder + GIMIC) for systems that SMILES cannot represent
# (η5/η3 coordination, inverse sandwiches, etc.).
NICS_BENCHMARKS = {
    "benzene":              ("c1ccccc1",          -9.7,  "π-aromatic (canonical)"),
    "cyclobutadiene":       ("C1=CC=C1",          +27.0, "π-antiaromatic"),
    "pyridine":             ("c1ccncc1",          -9.0,  "heterocyclic π-aromatic"),
    "thiophene":            ("c1ccsc1",          -13.6,  "5-ring π-aromatic"),
    "furan":                ("c1ccoc1",          -12.3,  "5-ring π-aromatic"),
    "S₃ (cyclic)":          ("[S]1[S][S]1",       None, "double σ⊕π aromatic, predicted"),
    "Bi₃ (cyclic)":         ("[Bi]1[Bi][Bi]1",   -15.0, "σ-aromatic all-metal"),
    "Al₄ (cyclic)":         ("[Al]1[Al][Al][Al]1", -34.0, "multi-fold σ⊕π anion"),
    "Si₆ (cyclic)":         ("[Si]1[Si][Si][Si][Si][Si]1", -10.0, "all-Si hexagon"),
    "U@S₃":                 ("[U][S]1[S][S]1",     None, "actinide-capped σ⊕π aromatic"),
    "Th@S₃":                ("[Th][S]1[S][S]1",    None, "thorium-capped — easier synthesis"),
    "B@S₃":                 ("[B][S]1[S][S]1",     None, "boron-capped — easiest synthesis"),
    "C@S₃":                 ("[C][S]1[S][S]1",     None, "carbon-capped — pure organic"),
    "P₃ (cyclic)":          ("[P]1[P][P]1",        None, "σ-aromatic + π-radical"),
    "Cu₃ (cyclic)":         ("[Cu]1[Cu][Cu]1",     None, "coinage σ-aromatic (1e/atom)"),
    "Ag₃ (cyclic)":         ("[Ag]1[Ag][Ag]1",     None, "coinage σ-aromatic"),
    # Inverse-sandwich actinide clusters (Ding 2026) — explicit coords
    "Bi₃@U₂ stripped":      ("@@cluster:bi3_u2_stripped",   +0.08,
                              "Ding 2026 cluster sans ligands Cp* (3 Bi + 2 U axial)"),
    "Bi₃@U₂(Cp*)₄ full":    ("@@cluster:bi3_u2_cp_star4",   +0.08,
                              "Ding 2026 inverse sandwich complet (105 atomes)"),
    # User-supplied XYZ + bonds (interactive)
    "Custom cluster (XYZ)": ("@@cluster:custom",            None,
                              "Entrer ses propres atomes/coordonnées et liaisons"),
}


_DEFAULT_CUSTOM_XYZ = """5
Bi3@U2 stripped (default — modifiable)
Bi   1.7610   0.0000   0.0000
Bi  -0.8805   1.5251   0.0000
Bi  -0.8805  -1.5251   0.0000
U    0.0000   0.0000   2.1000
U    0.0000   0.0000  -2.1000
"""

_DEFAULT_CUSTOM_BONDS = "0-1, 1-2, 2-0, 0-3, 1-3, 2-3, 0-4, 1-4, 2-4"
_DEFAULT_CUSTOM_RING = "0, 1, 2"


# Atomic symbol -> Z (covers Z=1..103)
_PT_SYMBOLS = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57,
    "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64,
    "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
    "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85,
    "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92,
    "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99,
    "Fm": 100, "Md": 101, "No": 102, "Lr": 103,
}


def _parse_xyz_block(text: str):
    """Parse a relaxed XYZ block.  Returns (Z_list, coords ndarray)."""
    import numpy as np
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        raise ValueError("XYZ vide.")
    # Header: optional "<N>" and optional comment
    rows = lines
    if lines[0].strip().isdigit():
        rows = lines[2:] if len(lines) > 1 else []
    Z_list = []
    coords = []
    for k, ln in enumerate(rows):
        parts = ln.split()
        if len(parts) < 4:
            raise ValueError(f"Ligne {k+1} : format attendu 'Symbol  x  y  z', got {ln!r}")
        sym = parts[0]
        Z = _PT_SYMBOLS.get(sym)
        if Z is None:
            try:
                Z = int(sym)
            except ValueError:
                raise ValueError(f"Symbole inconnu : {sym!r}")
        try:
            x, y, z = (float(parts[1]), float(parts[2]), float(parts[3]))
        except ValueError as e:
            raise ValueError(f"Ligne {k+1} : coordonnées invalides ({e})")
        Z_list.append(Z)
        coords.append([x, y, z])
    return Z_list, np.array(coords)


def _parse_bond_list(text: str, n_atoms: int):
    """Parse 'i-j[:bo], k-l, ...' into [(i, j, bo), ...]."""
    bonds = []
    if not text.strip():
        return bonds
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        bo = 1.0
        if ":" in chunk:
            edge, bo_str = chunk.split(":", 1)
            try:
                bo = float(bo_str.strip())
            except ValueError:
                raise ValueError(f"Bond order invalide dans {chunk!r}")
        else:
            edge = chunk
        if "-" not in edge:
            raise ValueError(f"Bond {chunk!r} : format 'i-j' requis")
        i, j = edge.split("-", 1)
        try:
            i = int(i.strip())
            j = int(j.strip())
        except ValueError:
            raise ValueError(f"Indices invalides dans {chunk!r}")
        if not (0 <= i < n_atoms and 0 <= j < n_atoms):
            raise ValueError(
                f"Index hors plage dans {chunk!r} (n_atoms={n_atoms})"
            )
        if i == j:
            raise ValueError(f"Boucle propre interdite : {chunk!r}")
        bonds.append((i, j, bo))
    return bonds


# ─────────────────────────────────────────────────────────────────────
# Predictive aromaticity scan (adapted from scripts/scan_aromatic_rings.py)
# ─────────────────────────────────────────────────────────────────────


# Element groups for scan enumeration
_SCAN_S1 = [3, 11, 19, 29, 47, 79]               # Li Na K Cu Ag Au
_SCAN_G13 = [5, 13, 31, 49, 81]                  # B Al Ga In Tl
_SCAN_G14 = [6, 14, 32, 50, 82]                  # C Si Ge Sn Pb
_SCAN_G15 = [15, 33, 51, 83]                     # P As Sb Bi
_SCAN_G16 = [16, 34, 52, 84]                     # S Se Te Po
_SCAN_F = [57, 58, 59, 60, 90, 91, 92, 93, 94]   # La Ce Pr Nd Th Pa U Np Pu

_SCAN_GROUPS = {
    "Group 1 (alkali, coinage)": _SCAN_S1,
    "Group 13 (B Al Ga In Tl)": _SCAN_G13,
    "Group 14 (C Si Ge Sn Pb)": _SCAN_G14,
    "Group 15 (P As Sb Bi)": _SCAN_G15,
    "Group 16 (S Se Te Po)": _SCAN_G16,
    "f-block (La Ce…U Pu)": _SCAN_F,
}


def _scan_ring_smiles(Zs):
    from ptc.smiles_parser import _SYM
    syms = [_SYM[z] for z in Zs]
    return f"[{syms[0]}]1" + "".join(f"[{s}]" for s in syms[1:]) + "1"


def _scan_capped_smiles(Z_cap, Zs_ring):
    from ptc.smiles_parser import _SYM
    cap = _SYM[Z_cap]
    inner = "".join(f"[{_SYM[z]}]" for z in Zs_ring)
    return f"[{cap}]{inner[0:4]}1{inner[4:]}1"


def _scan_evaluate(name: str, smi: str, n_ring: int):
    """Build Molecule, run signature, return scoreable record dict.
    Returns None on parse / compute errors (silently skipped)."""
    try:
        mol = Molecule(smi)
        aro = mol.aromaticity()
        if aro is None:
            return None
        sig = mol.signature()
        return {
            "name": name,
            "smiles": smi,
            "n_atoms": n_ring,
            "n_sigma": sig.n_aromatic_sigma,
            "n_pi": sig.n_aromatic_pi,
            "NICS_0": float(aro.NICS_0),
            "NICS_1": float(aro.NICS_1),
            "R": float(aro.R),
            "f_coh": float(sig.f_coh),
            "class": sig.aromatic_class,
            "score": max(0.0, -float(aro.NICS_0)),  # diamagnetic only counts
        }
    except Exception:
        return None


def _scan_homonuclear_rings(allowed_Z, sizes):
    """Enumerate homonuclear A_n rings for n in sizes."""
    out = []
    for Z in allowed_Z:
        for n in sizes:
            smi = _scan_ring_smiles((Z,) * n)
            if not smi:
                continue
            from ptc.smiles_parser import _SYM
            sym = _SYM[Z]
            row = _scan_evaluate(f"{sym}_{n}", smi, n)
            if row is not None:
                out.append(row)
    return out


def _scan_heteronuclear_bridges(group_A_Z, group_B_Z):
    """B-A-B-A-B 5-cycle and B-A-B 3-cycle bridges (mixed)."""
    out = []
    for Za in group_A_Z:
        for Zb in group_B_Z:
            if Za == Zb:
                continue
            for ring in [(Zb, Za, Zb), (Zb, Za, Zb, Za, Zb)]:
                smi = _scan_ring_smiles(ring)
                if not smi:
                    continue
                from ptc.smiles_parser import _SYM
                tag = "".join(_SYM[z] for z in ring)
                row = _scan_evaluate(f"{tag}", smi, len(ring))
                if row is not None:
                    out.append(row)
    return out


def _scan_capped_X3(cap_Z_list, X_group_Z):
    """Cap@X₃ (3-atom ring with axial cap)."""
    out = []
    from ptc.smiles_parser import _SYM
    for Zc in cap_Z_list:
        for Zx in X_group_Z:
            smi = f"[{_SYM[Zc]}][{_SYM[Zx]}]1[{_SYM[Zx]}][{_SYM[Zx]}]1"
            row = _scan_evaluate(f"{_SYM[Zc]}@{_SYM[Zx]}_3", smi, 3)
            if row is not None:
                out.append(row)
    return out


def _run_aromatic_scan(scan_classes, top_n=15):
    """Run the requested scan classes, return top_n records by score."""
    all_rows = []
    if "Triangles homonucléaires" in scan_classes:
        all_rows += _scan_homonuclear_rings(
            _SCAN_S1 + _SCAN_G13 + _SCAN_G14 + _SCAN_G15 + _SCAN_G16, [3])
    if "Tetragones homonucléaires" in scan_classes:
        all_rows += _scan_homonuclear_rings(
            _SCAN_G13 + _SCAN_G14 + _SCAN_G15, [4])
    if "Hexagones homonucléaires" in scan_classes:
        all_rows += _scan_homonuclear_rings(_SCAN_G14 + _SCAN_G15 + _SCAN_G16, [6])
    if "Bridges hétéro (B-A-B)" in scan_classes:
        all_rows += _scan_heteronuclear_bridges(_SCAN_G13 + _SCAN_G14, _SCAN_G16)
    if "Capped X₃ (cap-G16)" in scan_classes:
        all_rows += _scan_capped_X3(_SCAN_G13 + _SCAN_G14 + _SCAN_F, _SCAN_G16)
    if "Capped X₃ (cap-G15)" in scan_classes:
        all_rows += _scan_capped_X3(_SCAN_G13 + _SCAN_G14 + _SCAN_F, _SCAN_G15)
    # Filter to diamagnetic (score > 0) and sort
    diam = [r for r in all_rows if r["score"] > 0]
    diam.sort(key=lambda r: r["score"], reverse=True)
    return diam[:top_n], len(all_rows)


def _parse_ring_atoms(text: str, n_atoms: int):
    """Parse '0, 1, 2' → list of int indices."""
    ring = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            k = int(chunk)
        except ValueError:
            raise ValueError(f"Index ring invalide : {chunk!r}")
        if not (0 <= k < n_atoms):
            raise ValueError(f"Index ring hors plage : {k}")
        ring.append(k)
    if len(ring) < 3:
        raise ValueError("Un ring nécessite au moins 3 atomes.")
    return ring


# ─────────────────────────────────────────────────────────────────────
# Cluster-mode dispatch (explicit-coordinate pipeline)
# ─────────────────────────────────────────────────────────────────────


def _compute_cluster_nics(cluster_key: str,
                            custom_data: dict = None):
    """Build the chosen explicit-coord cluster, run Hueckel + CPHF, and
    return (nics_zz, J_avg, n_atoms, n_orb, ring_indices, basis_coords).

    For ``cluster_key=='custom'`` the caller passes ``custom_data`` with
    keys ``Z_list``, ``coords``, ``bonds``, ``ring`` (results are cached
    by content hash to avoid recomputing identical inputs).

    For built-in cluster keys (``bi3_u2_stripped``, ``bi3_u2_cp_star4``)
    the geometry is hard-coded.

    Caches under ``st.session_state`` so re-renders don't recompute.
    """
    if cluster_key == "custom" and custom_data is not None:
        # Hash content for cache key
        sig = (
            tuple(custom_data["Z_list"]),
            tuple(map(tuple, custom_data["coords"].tolist())),
            tuple(custom_data["bonds"]),
            tuple(custom_data["ring"]),
        )
        cache_key = f"_cluster_resp::custom::{hash(sig)}"
    else:
        cache_key = f"_cluster_resp::{cluster_key}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    from ptc.lcao.cluster import (
        build_bi3_u2_cp_star4,
        precompute_response_explicit,
    )
    from ptc.lcao.current import (
        nics_zz_from_current,
        ring_current_strength,
    )
    from ptc.lcao.giao import _build_quadrature_grid

    if cluster_key == "bi3_u2_stripped":
        # Bare Bi3 + 2 U axial caps (no Cp* ligands)
        d_bi_bi = 3.05
        z_u_cap = 2.1
        R_bi = d_bi_bi / np.sqrt(3.0)
        coords = np.array([
            [R_bi, 0.0, 0.0],
            [-R_bi / 2.0, R_bi * np.sqrt(3) / 2.0, 0.0],
            [-R_bi / 2.0, -R_bi * np.sqrt(3) / 2.0, 0.0],
            [0.0, 0.0, +z_u_cap],
            [0.0, 0.0, -z_u_cap],
        ])
        Z_list = [83, 83, 83, 92, 92]
        bonds = (
            [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]
            + [(b, u, 1.0) for u in (3, 4) for b in (0, 1, 2)]
        )
        ring = [0, 1, 2]
    elif cluster_key == "bi3_u2_cp_star4":
        Z_list, coords, bonds = build_bi3_u2_cp_star4(include_methyl_h=True)
        ring = [0, 1, 2]   # Bi3 ring is the first 3 atoms by construction
    elif cluster_key == "custom":
        if custom_data is None:
            raise ValueError("custom cluster requires custom_data")
        Z_list = list(custom_data["Z_list"])
        coords = np.asarray(custom_data["coords"], dtype=float)
        bonds = list(custom_data["bonds"])
        ring = list(custom_data["ring"])
    else:
        raise ValueError(f"Unknown cluster key {cluster_key!r}")

    cphf_kw = dict(
        max_iter=10, tol=1.0e-3,
        n_radial_grid=12, n_theta_grid=8, n_phi_grid=10,
        n_radial_op=14, n_theta_op=8, n_phi_op=10,
    )
    resp = precompute_response_explicit(
        Z_list, coords, bonds=bonds, basis_type="SZ",
        scf_mode="hueckel", cphf_kwargs=cphf_kw,
    )
    pts, w, _ = _build_quadrature_grid(
        resp.basis, np.zeros(3),
        n_radial=20, n_theta=10, n_phi=14,
        use_becke=False, lebedev_order=26,
    )
    sigma = nics_zz_from_current(resp, np.zeros(3), pts, w, beta=2)
    nics = -sigma
    J_avg, _ = ring_current_strength(
        resp, ring,
        half_width_in_plane=4.0, half_width_out_plane=4.0,
        n_in=21, n_out=21, gauge="common",
    )

    result = {
        "nics_zz": nics,
        "J_avg": J_avg,
        "n_atoms": len(Z_list),
        "n_orb": resp.basis.n_orbitals,
        "Z_list": list(Z_list),
        "ring": ring,
        "coords": coords,
    }
    st.session_state[cache_key] = result
    return result


def _render_custom_cluster_inputs():
    """Sidebar inputs for the 'Custom cluster (XYZ)' preset.

    Returns ``custom_data`` dict (Z_list, coords, bonds, ring) or
    ``None`` if the user has not yet pressed *Compute*.
    """
    st.caption("_Entrer un cluster arbitraire : géométrie XYZ + liste de "
                "liaisons + indices du cycle pour le calcul du J_ring._")

    # ─── Help expander (open by default the first time) ──────────────
    with st.expander("📖 Comment utiliser le mode custom ?", expanded=False):
        st.markdown(
            "Le **mode custom** permet de tester n'importe quel cluster en "
            "fournissant directement la géométrie cartésienne. Pratique pour "
            "les coordinations η5/η3 (sandwiches, métallocènes) que le SMILES "
            "ne sait pas représenter.\n\n"
            "**Trois entrées sont nécessaires** :"
        )
        st.markdown(
            "**1. Format XYZ** — un atome par ligne, séparateurs blanc, "
            "coordonnées en **Ångström** :\n"
            "```text\n"
            "Symbol     x        y        z\n"
            "C        0.0000   1.4034   0.0000\n"
            "C        1.2154   0.7017   0.0000\n"
            "...\n"
            "```\n"
            "- Header `<N>` + ligne commentaire **optionnels** (parser tolérant)\n"
            "- Symboles : tous les éléments Z=1..103 (Bi, U, Th, Cp* via C+H, …)\n"
            "- Numéro atomique direct accepté (`92` ≡ `U`)"
        )
        st.markdown(
            "**2. Liaisons** — liste `i-j` séparée par virgules, indices "
            "**zero-based** (l'ordre de l'XYZ) :\n"
            "```\n"
            "0-1, 1-2, 2-0, 0-3, 1-3, 2-3\n"
            "```\n"
            "- Ordre de liaison optionnel : `0-1:1.5` pour BO=1.5 (aromatique)\n"
            "- BO par défaut = 1.0 si non spécifié\n"
            "- Inclure les liaisons aux capping atoms (η³/η⁵) et ligands"
        )
        st.markdown(
            "**3. Atomes du cycle** — indices des atomes formant le ring "
            "dont on calcule le courant induit :\n"
            "```\n"
            "0, 1, 2\n"
            "```\n"
            "- ≥ 3 atomes\n"
            "- C'est sur cette boucle que `J_avg` est calculé via flux "
            "demi-plan ; le `NICS_zz` au centroid du cycle est rapporté"
        )
        st.markdown(
            "**Exemples ready-to-paste** ⤵️"
        )
        ex_tabs = st.tabs(["Bi₃@U₂ (défaut)", "S₃ triangle",
                             "Benzène", "Al₄²⁻ tetragone"])
        with ex_tabs[0]:
            st.code("5\nBi3@U2 stripped\n"
                    "Bi   1.7610   0.0000   0.0000\n"
                    "Bi  -0.8805   1.5251   0.0000\n"
                    "Bi  -0.8805  -1.5251   0.0000\n"
                    "U    0.0000   0.0000   2.1000\n"
                    "U    0.0000   0.0000  -2.1000",
                    language="text")
            st.code("0-1, 1-2, 2-0, 0-3, 1-3, 2-3, 0-4, 1-4, 2-4",
                    language="text")
            st.code("0, 1, 2", language="text")
        with ex_tabs[1]:
            st.code("3\nS3 cyclic (PT prediction)\n"
                    "S    1.0500   0.0000   0.0000\n"
                    "S   -0.5250   0.9093   0.0000\n"
                    "S   -0.5250  -0.9093   0.0000",
                    language="text")
            st.code("0-1, 1-2, 2-0", language="text")
            st.code("0, 1, 2", language="text")
        with ex_tabs[2]:
            st.code("6\nBenzene C ring (no H)\n"
                    "C    1.4030   0.0000   0.0000\n"
                    "C    0.7015   1.2152   0.0000\n"
                    "C   -0.7015   1.2152   0.0000\n"
                    "C   -1.4030   0.0000   0.0000\n"
                    "C   -0.7015  -1.2152   0.0000\n"
                    "C    0.7015  -1.2152   0.0000",
                    language="text")
            st.code("0-1:1.5, 1-2:1.5, 2-3:1.5, 3-4:1.5, 4-5:1.5, 5-0:1.5",
                    language="text")
            st.code("0, 1, 2, 3, 4, 5", language="text")
        with ex_tabs[3]:
            st.code("4\nAl4 square (Boldyrev anion)\n"
                    "Al   1.2500   1.2500   0.0000\n"
                    "Al  -1.2500   1.2500   0.0000\n"
                    "Al  -1.2500  -1.2500   0.0000\n"
                    "Al   1.2500  -1.2500   0.0000",
                    language="text")
            st.code("0-1, 1-2, 2-3, 3-0", language="text")
            st.code("0, 1, 2, 3", language="text")
        st.info(
            "💡 **Astuces** :\n"
            "- Plus le cluster est grand, plus le calcul est long (Bi₃@U₂ "
            "stripped ≈ 2 s ; Bi₃@U₂(Cp*)₄ full ≈ 15 s)\n"
            "- Le résultat est mis en cache : modifier l'entrée déclenche "
            "un nouveau calcul, identique → résultat instantané\n"
            "- Critère cible **|NICS_zz| < 2 ppm** : cluster ligandé "
            "non-aromatique au centre, comme observé par Ding 2026 sur Bi₃@U₂\n"
            "- Erreurs de format : le parser indique la ligne / le terme "
            "exact en cause"
        )

    xyz = st.text_area(
        "XYZ (1 atome par ligne : `Symbol  x  y  z` en Å, header optionnel)",
        value=st.session_state.get("aro_custom_xyz", _DEFAULT_CUSTOM_XYZ),
        key="aro_custom_xyz", height=170,
    )
    bonds_str = st.text_input(
        "Liaisons (`i-j[:bo], k-l, …` ; bo défaut = 1.0)",
        value=st.session_state.get("aro_custom_bonds", _DEFAULT_CUSTOM_BONDS),
        key="aro_custom_bonds",
    )
    ring_str = st.text_input(
        "Atomes du cycle (`0, 1, 2, …` ; ≥ 3)",
        value=st.session_state.get("aro_custom_ring", _DEFAULT_CUSTOM_RING),
        key="aro_custom_ring",
    )
    submit = st.button("⚙️ Calculer NICS_zz custom", key="aro_custom_submit")
    if not submit:
        return None
    try:
        Z_list, coords = _parse_xyz_block(xyz)
        bonds = _parse_bond_list(bonds_str, len(Z_list))
        ring = _parse_ring_atoms(ring_str, len(Z_list))
    except Exception as e:
        st.error(f"Erreur de parsing : {type(e).__name__}: {e}")
        return None
    return {"Z_list": Z_list, "coords": coords, "bonds": bonds, "ring": ring}


def _render_cluster_panel(cluster_key: str, nics_exp, descriptor: str,
                            custom_data: dict = None):
    """Rendering branch for explicit-coord clusters (Bi3@U2 family +
    user-supplied custom)."""
    if cluster_key == "bi3_u2_stripped":
        label = "stripped"
    elif cluster_key == "bi3_u2_cp_star4":
        label = "(Cp*)₄ full"
    else:
        label = "custom"
    st.caption(f"_{descriptor}_")
    if nics_exp is not None:
        st.metric("NICS_zz expérimental (Ding 2026)", f"{nics_exp:+.2f} ppm")

    with st.spinner(f"Building cluster {label} (Hueckel + CPHF + GIMIC)…"):
        try:
            res = _compute_cluster_nics(cluster_key, custom_data=custom_data)
        except Exception as e:
            st.error(f"Erreur cluster: {type(e).__name__}: {e}")
            return

    title_prefix = "Cluster custom" if cluster_key == "custom" else f"Bi₃@U₂ {label}"
    st.markdown(f"### {title_prefix} — vectorial GIMIC PT-pure")
    cols = st.columns(4)
    cols[0].metric("NICS_zz centre",
                    f"{res['nics_zz']:+.3f} ppm",
                    delta=(f"{res['nics_zz'] - nics_exp:+.2f} vs exp"
                           if nics_exp is not None else None))
    cols[1].metric("J_avg (Bi₃ ring)", f"{res['J_avg']:+.2f} nA/T")
    cols[2].metric("n_atoms", str(res['n_atoms']))
    cols[3].metric("n_orbitals (SZ)", str(res['n_orb']))

    in_window = abs(res['nics_zz']) < 2.0
    st.success(
        f"**Cible |NICS_zz| < 2 ppm** : "
        f"{'✓ atteint' if in_window else '✗ hors fenêtre'}"
    )

    with st.expander("📐 Géométrie du cluster"):
        sym_map = {83: "Bi", 92: "U", 6: "C", 1: "H"}
        st.markdown(f"- **Composition** : {res['n_atoms']} atomes  "
                    f"(Bi₃ + 2 U axiaux"
                    + (" + 4 × Cp*" if cluster_key.endswith("cp_star4") else "")
                    + ")")
        st.markdown(f"- **Bi-Bi distance**: 3.05 Å (Ding 2026 X-ray)")
        st.markdown(f"- **U axial cap**: ±2.1 Å vs Bi₃ plane")
        if cluster_key.endswith("cp_star4"):
            st.markdown(f"- **Cp* angle Cp*-U-Cp***: 134° (bent metallocene)")
            st.markdown(f"- **U-Cp* centroid**: 2.5 Å")

    with st.expander("📖 Pipeline PT-pure utilisé"):
        st.markdown(
            "- **Géométrie f-block** : `ptc/lcao/relativistic.py` — facteurs "
            "γ_la (4f) et γ_an (5f) PT-purs dérivés de γ₅ et γ₇\n"
            "- **Densité de courant induit** : `ptc/lcao/current.py` — "
            "j_para CPHF coupled + j_dia common-origin + correction "
            "London-phase GIAO (TERM2)\n"
            "- **Harmoniques cubiques l=3** : `ptc/lcao/giao.py` — 7 réelles "
            "f-orbitales pour 5f des actinides\n"
            "- **Builder cluster explicite** : "
            "`ptc/lcao/cluster.py::build_explicit_cluster` — coordonnées "
            "Cartésiennes arbitraires, sans passage par SMILES\n"
            "- **Cp* + d-shell + g-polar** : "
            "`build_bi3_u2_cp_star4`, occupation (n-1)d opt-in (Ce/Gd/U/...), "
            "harmoniques cubiques l=4 disponibles pour DZP étendu"
        )
        st.code(
            "from ptc.lcao.cluster import build_bi3_u2_cp_star4, precompute_response_explicit\n"
            "from ptc.lcao.current import nics_zz_from_current\n\n"
            "Z, coords, bonds = build_bi3_u2_cp_star4(include_methyl_h=True)\n"
            "resp = precompute_response_explicit(Z, coords, bonds=bonds,\n"
            "    basis_type='SZ', scf_mode='hueckel')\n"
            "nics = -nics_zz_from_current(resp, [0,0,0], pts, w, beta=2)",
            language="python",
        )


def _plot_NICS_profile(prof, R: float):
    """Plot NICS(z) vs z, with Biot-Savart reference curve."""
    zs = [p[0] for p in prof]
    nics = [p[1] for p in prof]

    fig = go.Figure()

    # Diamagnetic / paramagnetic shading
    if any(n < 0 for n in nics):
        fig.add_hrect(y0=-1e3, y1=0, fillcolor="rgba(31,119,180,0.08)",
                      line_width=0, layer="below")
        fig.add_annotation(x=max(zs)*0.85, y=min(nics)*0.6,
                           text="<b>diamagnetic</b><br>(aromatic)",
                           showarrow=False, font=dict(size=10, color="#1f77b4"))
    if any(n > 0 for n in nics):
        fig.add_hrect(y0=0, y1=1e3, fillcolor="rgba(214,39,40,0.08)",
                      line_width=0, layer="below")
        fig.add_annotation(x=max(zs)*0.85, y=max(nics)*0.6,
                           text="<b>paramagnetic</b><br>(antiaromatic)",
                           showarrow=False, font=dict(size=10, color="#d62728"))

    # NICS curve
    fig.add_trace(go.Scatter(
        x=zs, y=nics, mode='lines+markers',
        marker=dict(size=10, color='#2ca02c'),
        line=dict(width=2.5, color='#2ca02c'),
        name='NICS(z) PT',
        hovertemplate='z = %{x:.2f} Å<br>NICS = %{y:+.2f} ppm<extra></extra>',
    ))

    # Biot-Savart reference curve (continuous)
    z_dense = np.linspace(0, max(zs), 100)
    bs_ref = nics[0] * (R*R) / (R*R + z_dense*z_dense)**1.5 * R if R > 0 else None
    # Actually direct PT formula
    if R > 0 and nics[0] != 0:
        bs_dense = [nics[0] * (R*R) / (R*R + z*z)**1.5 / 1.0 for z in z_dense]
        fig.add_trace(go.Scatter(
            x=z_dense, y=bs_dense, mode='lines',
            line=dict(dash='dot', color='gray', width=1.5),
            name='Biot-Savart (PT analytical)',
            hoverinfo='skip',
        ))

    fig.update_layout(
        xaxis=dict(title='probe distance z (Å)', zeroline=True),
        yaxis=dict(title='NICS (ppm)', zeroline=True, zerolinecolor='black',
                   zerolinewidth=2),
        height=350, margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True, hovermode='x unified',
    )
    return fig


def _format_huckel_status(n: int, label: str) -> str:
    if n == 0:
        return f"{label} = 0"
    if n % 4 == 2:
        return f"**{label} = {n}** (4n+2 with n={(n-2)//4} ✓ aromatic)"
    if n % 4 == 0:
        return f"**{label} = {n}** (4n with n={n//4} — antiaromatic)"
    return f"**{label} = {n}** (odd — radical / open-shell)"


def render_aromaticity_tab():
    """Aromaticity diagnostic tab."""
    st.markdown("### NICS — Nucleus-Independent Chemical Shift")
    st.markdown(
        "Diagnostic d'aromaticité PT-pur (Pauling-London + règle de Hückel signée). "
        "K = α²·a₀/12 = 2.348 ppm·Å (préfacteur dérivé des constantes PT, sans fit)."
    )

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("#### Sélection")
        choice = st.selectbox(
            "Molécule de référence",
            options=list(NICS_BENCHMARKS.keys()),
            index=0,
            key="aro_choice",
        )
        smi_default = NICS_BENCHMARKS[choice][0]
        nics_exp = NICS_BENCHMARKS[choice][1]
        descriptor = NICS_BENCHMARKS[choice][2]

        # Cluster mode (explicit Cartesian coords) — bypass SMILES input
        is_cluster = isinstance(smi_default, str) and smi_default.startswith("@@cluster:")
        custom_data = None
        if is_cluster:
            cluster_key = smi_default.split(":", 1)[1]
            st.info("**Mode cluster** : géométrie explicite (XYZ) — "
                    "pipeline `ptc/lcao/cluster.py` + GIMIC PT-pur.")
            smi = smi_default   # opaque sentinel, not editable
            if cluster_key == "custom":
                custom_data = _render_custom_cluster_inputs()
        else:
            smi = st.text_input(
                "SMILES (modifiable)",
                value=smi_default,
                key="aro_smiles",
            )
            st.caption(f"_{descriptor}_")
            if nics_exp is not None:
                st.metric("NICS(0) expérimental", f"{nics_exp:+.1f} ppm")

    with col_right:
        if is_cluster:
            if cluster_key == "custom":
                if custom_data is None:
                    st.info("👈 Entrer XYZ + liaisons + ring puis cliquer "
                            "**Calculer NICS_zz custom** pour lancer le calcul.")
                    return
                _render_cluster_panel(cluster_key, nics_exp, descriptor,
                                       custom_data=custom_data)
            else:
                _render_cluster_panel(cluster_key, nics_exp, descriptor)
            return

        try:
            mol = Molecule(smi)
            aro = mol.aromaticity()
            if aro is None:
                st.warning("Aucun cycle détecté dans la molécule.")
                return
            sig = mol.signature()

            st.markdown(f"### {sig.formula}  —  {sig.aromatic_class}")
            st.caption(f"SMILES: `{smi}`")

            cols = st.columns(4)
            cols[0].metric("NICS(0) PT", f"{aro.NICS_0:+.2f} ppm",
                           delta=(f"{aro.NICS_0 - nics_exp:+.1f} vs exp"
                                  if nics_exp else None))
            cols[1].metric("NICS(1) PT", f"{aro.NICS_1:+.2f} ppm")
            ratio = aro.NICS_0 / aro.NICS_1 if aro.NICS_1 != 0 else 0
            cols[2].metric("Ratio NICS(0)/NICS(1)", f"{ratio:.2f}",
                           help=">1.5 indique σ-dominant ou double aromatic")
            cols[3].metric("Ring radius R", f"{aro.R:.2f} Å")

            st.markdown("#### Compte d'électrons délocalisés (par canal)")
            cc1, cc2 = st.columns(2)
            cc1.markdown(_format_huckel_status(sig.n_aromatic_sigma, "σ-aromatic"))
            cc2.markdown(_format_huckel_status(sig.n_aromatic_pi, "π-aromatic"))
            st.markdown(
                f"**T³ Fourier coherence f_coh = {sig.f_coh:.3f}** "
                f"(1.0 = homonucléaire pur)"
            )

            st.markdown("#### Profil NICS(z)")
            fig = _plot_NICS_profile(sig.NICS_profile, aro.R)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Erreur de calcul: {type(e).__name__}: {e}")
            return

    # ── Detail expanders ───────────────────────────────────────────
    with st.expander("📊 Energetics + fragmentation channels", expanded=False):
        col_e1, col_e2 = st.columns(2)
        col_e1.metric("D_at total", f"{sig.D_at:.3f} eV",
                      delta=f"{sig.D_per_atom:.2f} eV/atom")
        if sig.cap_binding > 0:
            col_e2.metric("Cap binding", f"{sig.cap_binding:.2f} eV",
                          delta="energy to remove cap M")
        else:
            col_e2.metric("Lowest fragmentation",
                          f"{sig.lowest_decomposition_eV:.2f} eV")

        st.markdown("**Canaux de fragmentation** (ΔE en eV) :")
        for c in sig.fragmentation:
            mark = " ← *plus bas*" if c.is_lowest else ""
            st.markdown(f"- ΔE = **{c.delta_E:+.2f}** : "
                        f"{' + '.join(c.products)} ({c.note}){mark}")

    with st.expander("🌈 Vibrational signature (Morse-calibrated)"):
        if sig.vib_modes:
            import pandas as pd
            df = pd.DataFrame([{
                'bond': v.label,
                'ω_PT raw (cm⁻¹)': f"{v.omega_raw:.0f}",
                'Morse factor': f"{v.morse_factor:.2f}",
                'ω calibrated (cm⁻¹)': f"{v.omega_calibrated:.0f}",
            } for v in sig.vib_modes])
            st.dataframe(df, use_container_width=True, hide_index=True)
        if sig.ir_breathing_cm1 > 0:
            st.success(f"**Mode breathing (totally symmetric, A₁ in C_nv): "
                       f"≈ {sig.ir_breathing_cm1:.0f} cm⁻¹** (IR-actif)")

    with st.expander("⚛️ Electronic structure (Koopmans + cycle-Hückel)"):
        c1, c2, c3 = st.columns(3)
        c1.metric("IE verticale", f"{sig.IE_PT_eV:.2f} eV")
        c2.metric("EA verticale", f"{sig.EA_PT_eV:.2f} eV")
        c3.metric("HOMO–LUMO gap", f"{sig.HL_gap_eV:.2f} eV")
        st.caption("EA est une estimation (½ × max EA atomique des atomes "
                   "membres). HOMO-LUMO gap σ peut être 0 si les modes "
                   "cycle-Hückel σ sont tous remplis (cas des doubly aromatic).")

    with st.expander("🔬 Geometry"):
        for label, r in sig.bond_lengths:
            st.markdown(f"- **{label}**: r_e = {r:.3f} Å")
        if sig.cap_height > 0:
            st.markdown(f"- **Cap height** above ring plane: {sig.cap_height:.3f} Å")

    # ── Predictive aromaticity scan ────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Scan prédictif d'aromaticité PT")
    st.caption(
        "Énumère des familles de cycles et capping atoms ; classe par "
        "score diamagnétique = max(0, −NICS₀). Les antiaromatiques "
        "(score 0) sont écartés. Le scan reproduit la pipeline ayant "
        "produit les candidats P₃, S₃@U/Th, Cu₃, Ag₃ de l'article "
        "PT_AROMATICITY."
    )

    sc_col_left, sc_col_right = st.columns([1, 2])
    with sc_col_left:
        scan_classes = st.multiselect(
            "Classes à scanner",
            options=[
                "Triangles homonucléaires",
                "Tetragones homonucléaires",
                "Hexagones homonucléaires",
                "Bridges hétéro (B-A-B)",
                "Capped X₃ (cap-G16)",
                "Capped X₃ (cap-G15)",
            ],
            default=[
                "Triangles homonucléaires",
                "Hexagones homonucléaires",
                "Bridges hétéro (B-A-B)",
            ],
            key="aro_scan_classes",
            help="Les classes 'Capped X₃' énumèrent des systèmes "
                  "atome+anneau avec cap_binding pour la classification "
                  "synthétique ; le NICS du ring lui-même est identique "
                  "pour tous les caps de même ring (cap = info "
                  "synthèse, pas info NICS).",
        )
        top_n = st.slider("Top N à afficher", 5, 30, 15, key="aro_scan_topn")
        run_scan = st.button("🚀 Lancer le scan", key="aro_scan_run")
        st.caption(
            "_~50 ms par candidat. Cocher une seule classe pour aller "
            "plus vite._"
        )

    with sc_col_right:
        if run_scan:
            if not scan_classes:
                st.warning("Sélectionner au moins une classe.")
            else:
                with st.spinner(f"Scan en cours sur {len(scan_classes)} classe(s)…"):
                    try:
                        top, n_total = _run_aromatic_scan(scan_classes, top_n=top_n)
                    except Exception as e:
                        st.error(f"Erreur scan : {type(e).__name__}: {e}")
                        top, n_total = [], 0
                if not top:
                    st.warning(f"Aucun candidat diamagnétique trouvé "
                                f"(sur {n_total} évalués).")
                else:
                    st.success(
                        f"**{len(top)} candidats diamagnétiques** "
                        f"(top {top_n}, sur {n_total} évalués)."
                    )
                    import pandas as pd
                    df = pd.DataFrame([{
                        "Rank": i + 1,
                        "SMILES": r["smiles"],
                        "Name": r["name"],
                        "n_atoms": r["n_atoms"],
                        "n_σ": r["n_sigma"],
                        "n_π": r["n_pi"],
                        "NICS(0) ppm": f"{r['NICS_0']:+.2f}",
                        "NICS(1) ppm": f"{r['NICS_1']:+.2f}",
                        "R (Å)": f"{r['R']:.2f}",
                        "f_coh": f"{r['f_coh']:.3f}",
                        "Class": r["class"],
                        "Score": f"{r['score']:.2f}",
                    } for i, r in enumerate(top)])
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    # Quick re-pick: copy a SMILES into the main input
                    pick = st.selectbox(
                        "Charger un candidat dans le panel principal :",
                        options=["—"] + [r["smiles"] for r in top],
                        key="aro_scan_pick",
                    )
                    if pick != "—":
                        st.session_state["aro_smiles"] = pick
                        st.info(
                            f"SMILES `{pick}` chargé dans le champ "
                            "principal. Sélectionner une référence "
                            "**autre que** *Custom cluster* dans le menu, "
                            "puis revenir pour rafraîchir."
                        )
        else:
            st.info(
                "👈 Choisir les classes + cliquer **Lancer le scan**. "
                "Les meilleurs candidats par score diamagnétique "
                "apparaîtront ici, classés."
            )

    # ── Theoretical reminder ──────────────────────────────────────
    with st.expander("📖 Formule PT (rappel théorique)"):
        st.markdown(r"""
        **Pauling-London PT NICS — formule signée Hückel :**

        $$
        \sigma(z) = -\frac{\alpha^2 a_0}{12}
        \cdot \frac{n_\text{eff}^\text{signed} \cdot f_\text{coh} \cdot R^2}
        {(R^2 + z^2)^{3/2}} \times 10^6 \text{ ppm}
        $$

        avec
        $$
        n_\text{eff}^\text{signed} = \text{sign}_\text{Hückel}(n_\sigma) \cdot n_\sigma
        + \text{sign}_\text{Hückel}(n_\pi) \cdot n_\pi
        $$

        et $\text{sign}_\text{Hückel}(n) = +1$ si $n = 4k+2$ (aromatic),
        $-1$ si $n = 4k$ (anti-aromatic), $+0.5$ si radical.

        **Aucun paramètre ajusté.** Les comptes σ/π viennent de la
        classification de groupe : G1/G11 → 1 σ, G13 → 1 π,
        G14 → 2 σ, G15 → 3 (σ+π), G16 → 4 (σ+2π).
        """)
