"""ea_continuous.py — Electron Affinity from continuous screening profiles.

Derives EA from the SAME screening engine as IE (atom.py), evaluated
at filling nd+1 instead of nd. The EA is the binding energy difference:

  EA(Z) = E(Z, nd) - E(Z, nd+1)

where E(Z, nd) = Ry × (Z × exp(-S_total(nd)) / per)² is the total
energy with nd electrons in the active shell.

The continuous screening S(nd) from Li_γ polylogarithm gives a SMOOTH
profile that can be evaluated at nd+1, unlike the discrete DFT which
is defined only at integer positions.

March 2026 — Persistence Theory
"""
from __future__ import annotations

import math

from ptc.constants import (
    S_HALF, P1, P2, P3,
    S3, S5, S7, C3, C5, C7,
    D3, D5, D7, AEM, RY,
    D_FULL,
)
from ptc.periodic import period, l_of, _n_fill_aufbau, ns_config


def _S_polygon_at(l: int, nd: int, per: int) -> float:
    """Screening at filling nd using continuous profile.

    Same pipeline as atom.py S_polygon but evaluable at any nd.
    Uses continuous peer values + DFT + corrections.
    """
    from ptc.atom import (
        _get_polygon_dft, _dft_synthesis, _polygon_bifurcation,
        _polygon_curvature, _BLOCK, S_core,
    )

    if nd <= 0:
        return 0.0

    B = _BLOCK[l]
    N_l = B['N']
    if nd > N_l:
        return 0.0

    # Use continuous peer values for DFT
    coeffs = _get_polygon_dft(l, per, continuous=True)
    S_dim2 = _dft_synthesis(coeffs, N_l, nd)

    # Same bifurcation as atom.py
    ns_val = 2  # assume ns=2 for capture (most common)
    S_dim2 = _polygon_bifurcation(l, S_dim2, nd, per, ns_val)

    # Same curvature
    S_dim2 += _polygon_curvature(l, nd, per)

    return S_dim2


def EA_continuous_eV(Z: int) -> float:
    """Electron affinity from continuous screening difference.

    EA = IE_at_nd - IE_at_(nd+1)

    where IE_at_nd = Ry × (Z_eff(nd) / per)² uses the screening
    at filling nd on the continuous Li_γ profile.

    The DIFFERENCE captures the binding energy of the (nd+1)-th
    electron: how much does the screening change when one more
    electron is added to the active polygon?
    """
    from ptc.atom import IE_eV, S_core, S_rel, screening_action

    per = period(Z)
    l = l_of(Z)
    nd = _n_fill_aufbau(Z)
    ns = ns_config(Z)

    # s-block: same direct formula (no polygon to evaluate)
    if l == 0:
        ie = IE_eV(Z, continuous=True)
        if ns == 1:
            return ie * S3 * S5
        return 0.0

    # Block parameters
    _PARAMS = {1: 2 * P1, 2: 2 * P2, 3: 2 * P3}
    N = _PARAMS.get(l, 6)

    if nd <= 0 or nd >= N:
        return 0.0

    # Screening at current filling and next filling
    S_nd = _S_polygon_at(l, nd, per)
    S_ndp1 = _S_polygon_at(l, nd + 1, per)

    # Full screening action (core + polygon + relativistic)
    S_c = S_core(per)
    S_r = S_rel(Z)

    S_total_nd = S_c + S_nd * D_FULL + S_r
    S_total_ndp1 = S_c + S_ndp1 * D_FULL + S_r

    # IE at each filling: Ry × (Z × exp(-S) / per)²
    from ptc.data.experimental import MASS
    m_nuc = MASS.get(Z, 2.0 * Z)
    _ME_AMU = 5.4858e-4

    Z_eff_nd = Z * math.exp(-S_total_nd)
    Z_eff_ndp1 = Z * math.exp(-S_total_ndp1)

    IE_nd = RY * (Z_eff_nd / per) ** 2 * m_nuc / (m_nuc + _ME_AMU)
    IE_ndp1 = RY * (Z_eff_ndp1 / per) ** 2 * m_nuc / (m_nuc + _ME_AMU)

    # EA = energy difference: how much binding energy does
    # the (nd+1)-th electron add?
    # If S_ndp1 > S_nd (more screening = weaker binding):
    #   IE_ndp1 < IE_nd → EA = IE_nd - IE_ndp1 > 0
    ea = IE_nd - IE_ndp1

    return max(0.0, ea)
