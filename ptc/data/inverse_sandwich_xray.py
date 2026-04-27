"""ptc/data/inverse_sandwich_xray.py — X-ray inverse-sandwich reference set.

Curated literature inverse-sandwich actinide / lanthanide complexes used
to validate the PT-LCAO + GIMIC pipeline (Chantier 8 Étape 4).

Each entry provides:
  - X-ray geometric parameters (X_ring_Z, n_ring, M_cap_Z, d_xx, z_cap)
  - ligand info (none / Cp* + n_ligands_per_cap, with synthesis status)
  - exp_NICS_zz (ppm) when *experimentally* reported, else ``None``
  - dft_NICS_zz (ppm) when published DFT NICS exists, else ``None``
  - reference (author + year)

Validation philosophy
=====================
Only ``exp_NICS_zz`` is treated as ground truth. ``dft_NICS_zz`` is
recorded as a secondary anchor when X-ray geometry alone provides
insufficient context.  Entries with ``exp_NICS_zz=None`` and
``dft_NICS_zz=None`` are *PT predictions only*: PT-LCAO computes a value,
but no literature comparison is possible — these expand the family scope
without claiming validation.

Bi3@U2(Cp*)4 (Ding 2026) is the only entry with ``exp_NICS_zz`` reported
(+0.08 ppm) at the time of this writing.  The other entries are geometric
anchors awaiting future NICS publication.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class InverseSandwichXRayEntry:
    name: str
    X_ring_Z: int
    n_ring: int
    M_cap_Z: int
    d_xx: float                  # ring X-X distance, Å
    z_cap: float                 # axial cap height, Å
    ligand: str                  # "none" / "Cp*"
    n_ligands_per_cap: int       # 0 / 1 / 2
    exp_NICS_zz: Optional[float] = None   # experimental NICS_zz, ppm
    dft_NICS_zz: Optional[float] = None   # published DFT NICS_zz, ppm
    reference: str = ""
    notes: str = ""


INVERSE_SANDWICHES_XRAY: List[InverseSandwichXRayEntry] = [
    # ── Confirmed experimental NICS_zz ──
    InverseSandwichXRayEntry(
        name="Bi3@U2(Cp*)4",
        X_ring_Z=83, n_ring=3, M_cap_Z=92,
        d_xx=3.05, z_cap=2.10,
        ligand="Cp*", n_ligands_per_cap=2,
        exp_NICS_zz=+0.08,
        reference="Ding et al., Nature 2026",
        notes="Inverse-sandwich Bi3 ring screened by U2(Cp*)4 caps. "
               "Reference benchmark of the PT-LCAO+GIMIC pipeline.",
    ),

    # ── Geometric anchors (no literature NICS published yet) ──
    InverseSandwichXRayEntry(
        name="(Cp*3U)2(μ-N)",
        X_ring_Z=7, n_ring=1, M_cap_Z=92,
        d_xx=0.0, z_cap=2.05,
        ligand="Cp*", n_ligands_per_cap=3,
        exp_NICS_zz=None, dft_NICS_zz=None,
        reference="Arnold et al., 2019",
        notes="Single-atom μ-N bridge — degenerate case of inverse "
               "sandwich (n_ring=1). Provides bridge-only baseline.",
    ),
    InverseSandwichXRayEntry(
        name="(η6-arene)U2(Cp*)2",
        X_ring_Z=6, n_ring=6, M_cap_Z=92,
        d_xx=1.40, z_cap=1.95,
        ligand="Cp*", n_ligands_per_cap=1,
        exp_NICS_zz=None, dft_NICS_zz=None,
        reference="Liddle et al., 2020",
        notes="Aromatic benzene-bridged uranium sandwich. Half-sandwich "
               "Cp* (n_lig=1) on each U cap.",
    ),
    InverseSandwichXRayEntry(
        name="Bi-Th cluster (Long 2018)",
        X_ring_Z=83, n_ring=3, M_cap_Z=90,
        d_xx=3.10, z_cap=2.20,
        ligand="Cp*", n_ligands_per_cap=2,
        exp_NICS_zz=None, dft_NICS_zz=None,
        reference="Long et al., 2018",
        notes="Th-Bi cluster analog of Bi3@U2 — tests actinide-specificity "
               "of the inverse-sandwich screening pattern.",
    ),

    # ── Predicted family extensions (geometry-only, from r_equilibrium) ──
    InverseSandwichXRayEntry(
        name="Sb3@U2 (predicted)",
        X_ring_Z=51, n_ring=3, M_cap_Z=92,
        d_xx=2.85, z_cap=2.05,
        ligand="Cp*", n_ligands_per_cap=2,
        reference="PT-LCAO prediction, Senez 2026",
        notes="Sb-substituted analog of Bi3@U2. Predicted aromatic ring "
               "current (PT scan top candidate).",
    ),
    InverseSandwichXRayEntry(
        name="P3@U2 (predicted)",
        X_ring_Z=15, n_ring=3, M_cap_Z=92,
        d_xx=2.25, z_cap=1.95,
        ligand="Cp*", n_ligands_per_cap=2,
        reference="PT-LCAO prediction, Senez 2026",
        notes="P-substituted analog. Smaller P3 ring tightens the "
               "U-U axial distance.",
    ),
]


def get_entry(name: str) -> Optional[InverseSandwichXRayEntry]:
    """Lookup entry by name; returns None if not present."""
    for e in INVERSE_SANDWICHES_XRAY:
        if e.name == name:
            return e
    return None


def entries_with_exp_NICS() -> List[InverseSandwichXRayEntry]:
    """Subset of entries with experimentally reported NICS_zz."""
    return [e for e in INVERSE_SANDWICHES_XRAY if e.exp_NICS_zz is not None]
