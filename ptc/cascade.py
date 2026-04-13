"""
PTC CASCADE — Unified molecular D_at engine on T³.

Bifurcated architecture (T4, molecular spectral convergence):

  PHASE 1 — Perron regime (n <= P3 = 7, dense graphs)
  =======
  The transfer matrix T4 captures ALL inter-bond correlations
  in a single diagonalisation.  The Perron eigenvector lambda_0
  gives the exact D_at for dense molecular graphs (alpha_v > s
  at the central vertex).
  Analogous to T4 Phase 1 (k <= 6, alpha < 1/3, Q > 0 automatically).

  Phase 1 post-corrections (mechanisms absent from T4):
    - LP->sigma back-donation [k=0 reference mode on Z/(2P1)Z]
    - Formal charge modulation [+/-Ry/P1 shift]
    - VSEPR axial attenuation [bifurcation steric > P1]

  PHASE 2 — Cascade regime (n > P3, sparse graphs)
  =======
  Per-bond cascade P0->P1->P2->P3 with 17 NLO corrections:
    cooperative screening, Dicke coherence, T3 perturbative,
    LP->pi cross-channel, ring strain, halogen pi-drain, etc.
  NLO corrections converge (molecular Mertens) because there
  are enough neighbours for the sum to be complete.
  Analogous to T4 Phase 2 (k >= 7, convergent induction).

  The bifurcation at n = P3 is structural [T4, D08]:
    n <= P3: NLO corrections have NOT converged -> Perron required
    n >  P3: NLO corrections HAVE converged -> self-calibrated cascade
  No local correction can replace the global diagonalisation
  of T4 (Perron-Frobenius theorem).

  n = 2: exact bilateral (diatomics)
  n = 3: triatomic solver (Phase 1 seeds + 3x3 cooperation)

0 adjustable parameters.  All from s = 1/2.

April 2026 — Persistence Theory
"""

from ptc.cascade_phase2 import (
    CascadeResult,
    compute_D_at_cascade,
    # Re-export for tools that need phase-specific access
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
