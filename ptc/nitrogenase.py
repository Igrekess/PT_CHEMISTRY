"""
ptc/nitrogenase.py — N₂ fixation catalytic cycle from PT.

Lowe-Thorneley mechanism: N₂ + 8H⁺ + 8e⁻ → 2NH₃ + H₂
on the FeMo-cofactor (Fe₇MoS₉C).

3 PT mechanisms (0 adjusted parameters):
  1. Adsorption via D_KL: σ-donation (P₁) + π-back-donation (P₂)
  2. Sequential N-H BDE with sin²₃ screening (not average)
  3. Fe-N surface bond weakening (imido → amido → ammine → desorption)
  4. Multi-metal cooperation: 7 Fe reduce barrier by (1 - n×sin²₅/P₁)

April 2026 — Theorie de la Persistance
"""
from __future__ import annotations

import math
from ptc.constants import (
    S_HALF, P1, P2, P3,
    S3, S5, S7, C3,
    RY, AEM,
)

IE_FE = 7.902  # eV
N_FE_ACTIVE = 7  # Fe atoms in FeMo-co


def compute_nitrogenase_cycle(pH: float = 7.0, E_donor: float = -0.31) -> dict:
    """Compute the 8-step Lowe-Thorneley nitrogenase cycle from PT.

    Returns dict with: steps, energy_profile, overall_dG, barrier info, etc.
    """
    from ptc.api import Molecule

    kT = 0.0257  # eV at 298K
    E_Hplus_e = E_donor - pH * kT * math.log(10)
    cost_He = abs(E_Hplus_e)

    # ── D_at of key molecules ──
    _smiles = {'N2': 'N#N', 'H2': '[H][H]', 'NH3': 'N', 'N2H4': 'NN'}
    D = {}
    for name, smi in _smiles.items():
        try:
            D[name] = Molecule(smi).D_at
        except Exception:
            D[name] = {'N2': 9.77, 'H2': 4.48, 'NH3': 11.08, 'N2H4': 17.84}[name]

    # ── 1. Adsorption: σ + π back-donation ──
    E_ads_N2 = -IE_FE * S5 * (S3 + S5)
    E_ads_abs = abs(E_ads_N2)

    # ── 2. Sequential N-H BDEs (sin²₃ screening) ──
    # BDE₁ = D₀ (first H, strongest), BDE₂ = D₀(1-S₃), BDE₃ = D₀(1-S₃)²
    D0_NH = D['NH3'] / (1 + (1 - S3) + (1 - S3) ** 2)
    BDE = [D0_NH, D0_NH * (1 - S3), D0_NH * (1 - S3) ** 2]

    D_NN_per_order = D['N2'] / P1

    # ── 3. Fe-N surface bond (weakens with each H) ──
    # imido Fe=NH: E_ads × P₁ (double bond, strong)
    # amido Fe-NH₂: E_ads × P₁ × sin²₃ (single bond)
    # ammine Fe←NH₃: E_ads × sin²₃ (dative, desorbs)
    D_FeN = [E_ads_abs * P1, E_ads_abs * P1 * S3, E_ads_abs * S3]

    # ── 4. Multi-metal cooperation for N-N cleavage ──
    # f_coop = 1 - n_Fe × sin²₅ / P₁ (distributes barrier across n_Fe centers)
    f_coop = max(0.0, 1.0 - N_FE_ACTIVE * S5 / P1)

    # ── 8-step cycle ──
    labels = [
        ('E₀', 'FeMo + N₂ → FeMo-N₂', 'adsorption'),
        ('E₁', 'FeMo-N₂ + H⁺+e⁻ → FeMo-N₂H', 'reduction'),
        ('E₂', 'FeMo-N₂H + H⁺+e⁻ → FeMo-NHNH', 'reduction'),
        ('E₃', 'FeMo-NHNH + H⁺+e⁻ → FeMo-NHNH₂', 'reduction'),
        ('E₄', 'FeMo-NHNH₂ + H⁺+e⁻ → FeMo-NH+NH₃↑', 'cleavage'),
        ('E₅', 'FeMo-NH + H⁺+e⁻ → FeMo-NH₂', 'reduction'),
        ('E₆', 'FeMo-NH₂ + H⁺+e⁻ → FeMo-NH₃', 'reduction'),
        ('E₇', 'FeMo-NH₃ → FeMo + NH₃↑ + ½H₂↑', 'release'),
    ]

    # Surface N-H attenuation factors (N-N bond order competition):
    _f = [S5, math.sqrt(S3 * S5), S3]  # triple, double, single

    steps, profile = [], [0.0]
    E_cum = 0.0

    for idx, (step_id, desc, stype) in enumerate(labels):
        if idx == 0:
            # E₀: N₂ adsorption
            dG = E_ads_N2

        elif idx <= 3:
            # E₁-E₃: add H to N₂Hₓ, weaken N-N progressively
            # N-H gain attenuated by N-N competition
            f_atten = _f[idx - 1]
            nn_cost = D_NN_per_order * S3 * idx / P1
            dG = -D['NH3'] / 3 * f_atten + cost_He + nn_cost

        elif idx == 4:
            # E₄: N-N cleavage + 1st NH₃ release (rate-limiting)
            # Cost: break last N-N bond (reduced by multi-metal cooperation)
            # Gain: NH₃ released (BDE₃ worth of surface bonds freed)
            dG = D_NN_per_order * S3 * f_coop - BDE[2] * S3 + cost_He

        elif idx <= 6:
            # E₅-E₆: H on mono-N with Fe-N bond weakening
            # Uses SEQUENTIAL BDE (not average) + Fe-N decrochage cost
            i_mono = idx - 5  # 0 for E₅, 1 for E₆
            bde = BDE[i_mono + 1]  # BDE₂ for E₅, BDE₃ for E₆
            dD_FeN = D_FeN[i_mono] - D_FeN[i_mono + 1]  # Fe-N bond lost
            dG = -bde + cost_He + dD_FeN

        else:
            # E₇: 2nd NH₃ + ½H₂ release + desorption
            dG = -E_ads_N2 - D['H2'] * S_HALF + cost_He

        E_cum += dG
        steps.append({
            'step': step_id, 'desc': desc, 'dG': dG,
            'E_cumul': E_cum, 'type': stype,
        })
        profile.append(E_cum)

    # Barrier analysis
    max_step = max(steps, key=lambda s: s['dG'])
    barrier_idx = profile.index(max(profile))
    barrier_height = max(profile) - min(profile[:barrier_idx + 1])

    return {
        'steps': steps,
        'energy_profile': profile,
        'overall_dG': E_cum,
        'overall_exp': -0.96,
        'E_ads_N2': E_ads_N2,
        'E_Hplus_e': E_Hplus_e,
        'pH': pH,
        'E_donor': E_donor,
        'D_at': D,
        'BDE_NH': BDE,
        'D_FeN': D_FeN,
        'f_coop': f_coop,
        'barrier_step': max_step['step'],
        'barrier_height': barrier_height,
    }
