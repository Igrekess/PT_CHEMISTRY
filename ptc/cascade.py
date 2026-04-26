"""
PTC CASCADE — Unified molecular D_at engine on T³.

Architecture bifurquée (T4, convergence spectrale moléculaire) :

  PHASE 1 — Perron regime (n ≤ P₃ = 7, graphes denses)
  ═══════
  La matrice de transfert T⁴ capture TOUTES les corrélations inter-bonds
  en une diagonalisation.  Le vecteur propre de Perron λ₀ donne le D_at
  exact pour les graphes moléculaires denses (α_v > s au vertex central).
  Analogue de T4 Phase 1 (k ≤ 6, α < 1/3, Q > 0 automatiquement).

  Post-corrections Phase 1 (mécanismes absents de T⁴) :
    • LP→σ back-donation [k=0 reference mode on Z/(2P₁)Z]
    • Formal charge modulation [±Ry/P₁ shift]
    • VSEPR axial attenuation [bifurcation steric > P₁]

  PHASE 2 — Cascade regime (n > P₃, graphes sparses)
  ═══════
  Cascade per-bond P₀→P₁→P₂→P₃ avec 17 corrections NLO :
    cooperative screening, Dicke coherence, T³ perturbatif,
    LP→π cross-channel, ring strain, halogen π-drain, etc.
  Les corrections NLO convergent (Mertens moléculaire) car il y a
  assez de voisins pour que la somme soit complète.
  Analogue de T4 Phase 2 (k ≥ 7, induction convergente).

  La bifurcation n = P₃ est structurelle [T4, D08] :
    n ≤ P₃ : les corrections NLO n'ont PAS convergé → Perron nécessaire
    n > P₃ : les corrections NLO ONT convergé → cascade auto-calibrée
  Aucune correction locale ne peut remplacer la diagonalisation globale
  de T⁴ (théorème de Perron-Frobenius).

  n = 2 : bilatéral exact (diatomiques)
  n = 3 : solveur triatomique (seeds Phase 1 + coopération 3×3)

0 paramètre ajusté.  Tout depuis s = 1/2.

Avril 2026 — Théorie de la Persistance
"""

from ptc.cascade_phase2 import (
    CascadeResult,
    compute_D_at_cascade,
    # Re-export internal for tools that need phase-specific access
    compute_D_at_cascade as compute_D_at_phase2,
)

# Phase 1 engine (Perron eigenvalue)
from ptc.transfer_matrix import compute_D_at_transfer as compute_D_at_phase1

__all__ = [
    'CascadeResult',
    'compute_D_at_cascade',
    'compute_D_at_phase1',
    'compute_D_at_phase2',
]
