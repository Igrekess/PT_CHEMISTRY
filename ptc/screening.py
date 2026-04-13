"""
PTC Shared Screening Functions — Interface Module.

This module defines the shared interface between the Phase 1 transfer-matrix
engine and the Phase 2 cascade engine.  All functions here are pure PT
(0 adjustable parameters, derived from s = 1/2) and can be used by
either engine.

Currently re-exported from transfer_matrix.py.  The architectural goal
is to gradually move implementations here, making transfer_matrix.py
contain ONLY Phase 1-specific code.

Shared functions (14):

  Data & types:
    _resolve_atom_data   — Load atomic IE/EA/per/l data
    BondSeed             — Per-bond 4-face seed data class

  Face capacities:
    _cap_P0              — P₀ (Z/2Z) parity capacity
    _cap_P2              — P₂ (Z/10Z) d-π capacity
    _cap_P3              — P₃ (Z/14Z) ionic capacity

  Screening (P₁ components):
    _dim3_overcrowding   — 3-dim vertex coordination screening
    _dim2_lp_mutual      — 2-dim cross-polygon LP coupling
    _dim1_exchange        — 1-dim orbital exchange penalty
    _screening_P0         — P₀ parity screening
    _dim2_vacancy_boost   — Vacancy-driven P₂ boost

  Post-seed corrections:
    _huckel_aromatic        — Hückel π delocalization (aromatic rings)
    _apply_shell_attenuation — Hypervalent shell closing

  SCF / Matrix:
    _build_T4               — T⁴ transfer matrix assembly
    _scf_iterate            — SCF power iteration (Perron eigenvector)

Note: Vertex DFT P₂/P₃ are now NATIVE in vertex_dft.py (not shared).
"""

# Re-export from transfer_matrix (canonical source, Phase 1)
from ptc.transfer_matrix import (
    # Data & types
    _resolve_atom_data,
    BondSeed,
    # Face capacities
    _cap_P0,
    _cap_P2,
    _cap_P3,
    # Screening components
    _dim3_overcrowding,
    _dim2_lp_mutual,
    _dim1_exchange,
    _screening_P0,
    _dim2_vacancy_boost,
    # Post-seed corrections
    _huckel_aromatic,
    _apply_shell_attenuation,
    # SCF / Matrix
    _build_T4,
    _scf_iterate,
)

__all__ = [
    '_resolve_atom_data', 'BondSeed',
    '_cap_P0', '_cap_P2', '_cap_P3',
    '_dim3_overcrowding', '_dim2_lp_mutual', '_dim1_exchange',
    '_screening_P0', '_dim2_vacancy_boost',
    '_huckel_aromatic', '_apply_shell_attenuation',
    '_build_T4', '_scf_iterate',
]
