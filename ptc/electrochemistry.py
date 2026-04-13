"""
electrochemistry.py — Electrochemical potentials from PT first principles.

Computes standard reduction potentials E(M^n+/M) vs SHE using:
  - PT ionization energies (atom.IE_eV)
  - Born solvation model with PT screening constants
  - Relativistic corrections from sin^2/cos^2 filters
  - d-block IE2 from constants.IE2_DBLOCK
  - Cross-channel P1×P2 corrections (April 2026):
      * Metallic cohesion via n_bond_d (P2 bonding d-electrons)
      * CFSE in octahedral water field (P2 crystal field)
      * Inert pair s2 stabilization (P1 bifurcation lock)
      * d10s2 metallic weakness penalty (Group 12)

April 2026 — Theorie de la Persistance
"""
from __future__ import annotations

import math

from ptc.constants import (
    S_HALF, P1, P2, P3, MU_STAR,
    S3, S5, C3, D3,
    RY, ALPHA_PHYS, COULOMB_EV_A, A_BOHR,
    IE2_DBLOCK,
    HALOGENS, CHALCOGENS, ALKALI, NS1,
)
from ptc.atom import IE_eV, EA_eV
from ptc.periodic import period as period_of, block_of
from ptc.data.experimental import SYMBOLS, IE_NIST


# ╔════════════════════════════════════════════════════════════════════╗
# ║  IE tables — NIST reference data for higher ionizations          ║
# ╚════════════════════════════════════════════════════════════════════╝

# IE2 for p-block multivalent ions (NIST)
_IE2_PBLOCK = {13: 18.83, 50: 14.63, 82: 15.03, 31: 20.51, 49: 18.87}

# IE3 for p-block trivalent ions (NIST)
_IE3_PBLOCK = {13: 28.45, 31: 30.71, 49: 28.03, 79: 20.6}

# IE3 for d-block trivalent ions (NIST)
_IE3_DBLOCK = {
    22: 27.49,  # Ti
    23: 29.31,  # V
    24: 30.96,  # Cr
    25: 33.67,  # Mn
    26: 30.65,  # Fe
    27: 33.50,  # Co
}

# IE2 for alkaline earth (NIST)
_IE2_AE = {4: 18.21, 12: 15.04, 20: 11.87, 38: 11.03, 56: 10.00}


# ╔════════════════════════════════════════════════════════════════════╗
# ║  IE_total — sum of n successive ionization energies              ║
# ╚════════════════════════════════════════════════════════════════════╝

def ie_total(Z: int, n_electrons: int) -> float:
    """Sum of n successive ionization energies for element Z (eV).

    Uses PT IE_eV for IE1, NIST tables for IE2/IE3 when available
    (d-block, p-block, alkaline earth), and geometric fallback otherwise:
        IE_k ~ IE1 * k * (1 + (k-1) * S3 / 2)
    """
    if n_electrons <= 0:
        return 0.0

    ie1 = IE_eV(Z)

    if n_electrons == 1:
        return ie1

    # IE2: check d-block, p-block, alkaline earth tables, then fallback
    ie2 = IE2_DBLOCK.get(Z,
           _IE2_PBLOCK.get(Z,
           _IE2_AE.get(Z,
           ie1 * 2.0 * (1.0 + S3 / 2.0))))

    if n_electrons == 2:
        return ie1 + ie2

    # IE3: check p-block table, then geometric fallback
    ie3 = _IE3_PBLOCK.get(Z, ie1 * 3.0 * (1.0 + S3))

    if n_electrons == 3:
        return ie1 + ie2 + ie3

    # Higher IEs: geometric scaling from IE1
    total = ie1 + ie2 + ie3
    for k in range(4, n_electrons + 1):
        total += ie1 * k * (1.0 + (k - 1) * S3 / 2.0)
    return total


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Born solvation radius (kept for backwards compatibility)       ║
# ╚════════════════════════════════════════════════════════════════════╝

def _born_radius(Z: int, n_charge: int) -> float:
    """Effective Born solvation radius (Angstroms)."""
    per = period_of(Z)
    return A_BOHR * per / (n_charge + S3)


def _solvation_energy(Z: int, n_charge: int) -> float:
    """Born solvation energy (eV), always negative for cations."""
    if n_charge == 0:
        return 0.0
    r_born = _born_radius(Z, abs(n_charge))
    eps_factor = 1.0 - 1.0 / 78.4
    return -COULOMB_EV_A * n_charge**2 / (2.0 * r_born) * eps_factor


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Cross-channel P1×P2 corrections for d-block divalent/trivalent ║
# ╚════════════════════════════════════════════════════════════════════╝

# ── CFSE factors for high-spin octahedral d^n (Principle 2, P2 polygon) ──
# The P2 polygon Z/10Z splits into t2g (6 positions) and eg (4 positions).
# CFSE = f × Δ_oct where Δ_oct = sin²₅ × Ry × s (PT-derived).
# High-spin: electrons fill t2g first, then eg.
_CFSE_FACTOR = {
    0:  0.0,    # d0: no splitting
    1: -0.4,    # d1: 1 e in t2g
    2: -0.8,    # d2
    3: -1.2,    # d3
    4: -0.6,    # d4 HS: 3 t2g + 1 eg
    5:  0.0,    # d5 HS: half-filled
    6: -0.4,    # d6 HS
    7: -0.8,    # d7
    8: -1.2,    # d8
    9: -0.6,    # d9
    10: 0.0,    # d10: no splitting
}

# Δ_oct in PT: sin²₅ × Ry × s
# = 0.194 × 13.6 × 0.5 = 1.32 eV (cf. exp ~10400 cm⁻¹ = 1.29 eV for Fe²⁺)
_DELTA_OCT_PT = S5 * RY * S_HALF

# Δ_band (metallic d-band width): sin²₅ × Ry / P₂
# = 0.194 × 13.6 / 5 = 0.528 eV (cf. exp ~0.5 eV typical d-bandwidth per electron)
# This stabilizes the METAL (not the ion) via d-band filling.
_DELTA_BAND_PT = S5 * RY / P2

# ── Bonding d-electrons for metallic cohesion (P2 channel) ──
# n_bond_d measures delocalized d-electrons available for metallic bonding.
# At d5 half-fill: ALL spins aligned by Hund exchange → electrons are
# localized on-site → ZERO metallic d-bonding (exchange-locked).
# At d0/d10: no unpaired d-electrons → zero bonding.
# Maximum near d3/d8: enough unpaired electrons to form bands.
def _n_bond_d(n_d: int) -> int:
    """Bonding d-electrons: zero at d0/d10, max at d5."""
    return min(n_d, 2 * P2 - n_d)

# Metallic cohesion energy per bonding d-electron:
# E_coh_per_d = sin²₅ × Ry × s / P₂ = 0.194 × 13.6 × 0.5 / 5 = 0.264 eV
_E_COH_PER_D = S5 * RY * S_HALF / P2

# ── Element sets ──
_D10_METALS = {29, 47, 79}       # Cu, Ag, Au (d10 s1)
_D5_METALS  = {25, 43, 75}       # Mn, Tc, Re (d5)
_D10S2 = {30, 48, 80}            # Zn, Cd, Hg (d10 s2, Group 12)
_INERT_PAIR = {50, 82, 32}       # Sn, Pb, Ge (s2p2 inert pair)

# SHE reference energy
_IE_REF = 13.598  # eV, hydrogen IE (Rydberg limit)


def _d_electron_count(Z: int) -> int:
    """Number of d-electrons for element Z (neutral atom)."""
    from ptc.periodic import _nd_of
    return _nd_of(Z)


def standard_potential_SHE(Z: int, n_charge: int, T: float = 298.15) -> float:
    """Standard reduction potential E(M^n+/M) vs SHE (Volts).

    PT formula with cross-channel P1×P2 (0 adjusted parameters):

    Base: E = -(IE_ref - IE_eff) × f_proj
      f_proj = sin²₃ + sin²₅ × exp(-(IE_eff - IE_ref)²/Ry²)

    Cross-channel P2 corrections for multivalent:
      1. Metallic cohesion: n_bond_d × sin²₅ × Ry × s / P₂
         → stabilizes metal → E more positive
      2. CFSE in water: f_CFSE(n_d_ion) × sin²₅ × Ry × s
         → stabilizes aqueous ion → E more negative
      3. d10s2 penalty: Group 12 has no d-band bonding
         → destabilizes metal → E more negative
      4. Inert pair: Sn²⁺/Pb²⁺ s² pair locked by P1 bifurcation
         → modifies effective IE

    Parameters
    ----------
    Z : int
        Atomic number.
    n_charge : int
        Oxidation state (positive for cations, e.g. 2 for M^2+).
    T : float
        Temperature in Kelvin (default 298.15 K).

    Returns
    -------
    float
        E in Volts vs SHE.
    """
    if n_charge == 0:
        return 0.0

    n = abs(n_charge)
    za2 = (Z * ALPHA_PHYS) ** 2
    ie1 = IE_eV(Z)
    blk = block_of(Z)
    per = period_of(Z)

    # ── Relativistic correction (P1+P2) ──
    f_s = 1.0 + za2 * S3
    f_d = 1.0 + za2 * S5 if blk == 'd' else 1.0
    f_rel = f_s * f_d

    # ── Effective IE (P1) ──
    # Order matters: most specific conditions first.
    if Z in _D10_METALS:
        ie2 = IE2_DBLOCK.get(Z, ie1 * 2.0)
        ie_eff = (1.0 - S_HALF) * ie1 + S_HALF * ie2
    elif n >= 3 and blk == 'p':
        # Trivalent p-block: geometric-weighted average (P1 cascade).
        # w_k = sin²₃^(k-1) : each successive IE is filtered by P1.
        # This emphasizes IE1 (most relevant for metal dissolution).
        ie_vals = [ie1,
                   _IE2_PBLOCK.get(Z, ie1 * 2.0 * (1.0 + S3 / 2.0)),
                   _IE3_PBLOCK.get(Z, ie1 * 3.0 * (1.0 + S3))]
        ie_sum_w, w_sum = 0.0, 0.0
        for k, ie_k in enumerate(ie_vals):
            w = S3 ** k
            ie_sum_w += ie_k * w
            w_sum += w
        ie_eff = ie_sum_w / w_sum
    elif n >= 2 and blk == 'p' and Z in _IE2_PBLOCK:
        ie_eff = (ie1 + _IE2_PBLOCK[Z]) / 2.0
    elif n >= 2 and blk == 'd' and Z in IE2_DBLOCK:
        ie2 = IE2_DBLOCK[Z]
        ie_eff = (ie1 + ie2) / 2.0
    else:
        ie_eff = ie1

    ie_eff *= f_rel

    # ── Base potential with continuous projection (Principle 4, q_+/q_-) ──
    x = (ie_eff - _IE_REF) / RY
    f_proj = S3 + S5 * math.exp(-x ** 2)
    E = -(_IE_REF - ie_eff) * f_proj

    # ── Solvation stabilization for multivalent (Born differential) ──
    # For n >= 2: extra solvation of M^n+ vs M^+ stabilizes the ion.
    # SKIP for d10s1 metals: the d10 bonus already captures this physics.
    # MODERATED by sin^2_3/n to prevent over-correction.
    # For heavy d-block (per >= 5): relativistic attenuation via 1/(1+za²).
    # Larger Z → stronger metallic bonding absorbs the solvation differential.
    if n >= 2 and Z not in _D10_METALS:
        dG_n = _solvation_energy(Z, n)
        dG_1 = _solvation_energy(Z, 1)
        delta = abs(dG_n) - abs(dG_1)
        delta = min(delta, ie1 * S_HALF)  # cap at IE1/2
        # Relativistic attenuation for heavy d-block (per >= 5)
        if blk == 'd' and per >= 5:
            delta /= (1.0 + za2)
        E -= delta * S3 / n

    # ── d10 nobility bonus (P3) ──
    # d5 half-fill: NO separate bonus (exchange-locking already handled
    # by n_bond_d = 0 in cross-channel; adding bonus = double-counting).
    if Z in _D10_METALS:
        bonus = S3 * S5 * ie1 * f_rel
        if Z > 36:
            bonus *= (1.0 + n / P1 * S5)
        # ── MONOVALENT d10 enhancement ──
        # For n=1: only the s-electron is removed, leaving a PERFECT d10
        # shell.  The d10 → d10 stability is the highest per-electron
        # contribution because zero d-antibonding states are created.
        # Enhancement: (1 + S₃ × P₁) for n=1 captures the fact that
        # ALL P₁ d-bonding channels remain intact (no d-hole created).
        # For n≥2: at least one d-electron is removed → weaker bonus.
        if n == 1:
            bonus *= (1.0 + S3 * P1)
        E += bonus * S3

    # ══════════════════════════════════════════════════════════════════
    # CROSS-CHANNEL P1×P2: d-block electronic configuration corrections
    # Applies to d-block EXCEPT d10s1 (Cu, Ag, Au) which have d10 bonus.
    # ══════════════════════════════════════════════════════════════════
    if n >= 2 and blk == 'd' and Z not in _D10_METALS:
        n_d = _d_electron_count(Z)

        # d-electron count in the ION:
        # M²⁺ removes all s-electrons first, then d if needed.
        # n_s = s-electrons in neutral atom (1 for Cr d5s1, 2 for most)
        from ptc.periodic import ns_config
        ns = ns_config(Z)
        n_d_ion = n_d - max(0, n - ns)
        n_d_ion = max(0, min(n_d_ion, 2 * P2))

        # ── 1. Metallic cohesion (P2 d-band bonding) ──
        # Uses NEUTRAL atom n_d (metal state).
        # Special case: d5s2 (Mn) = all d-spins aligned by Hund exchange
        # + s² paired → d-electrons exchange-locked → zero metallic d-bonding.
        # d5s1 (Cr) retains bonding: single s allows d-s hybridization.
        nbd = _n_bond_d(n_d)
        # d5 exchange locking:
        #   d5s2 (Mn, Tc, Re): ALL d-spins aligned + s² paired → nbd=0.
        #   d5s1 (Cr, Mo): half-filled d + single s.
        #     3d (Cr, per=4): compact d, strong intra-atomic exchange
        #       → P₁ d-electrons exchange-locked, only P₁ remain bonding.
        #     4d/5d (Mo, W-like): wider d-band → exchange weaker → nbd=5.
        if n_d == P2 and ns == 1 and per <= 4:  # 3d d5s1 (Cr)
            nbd = P1
        if n_d == P2 and ns == 2:  # d5s2: exchange-locked
            nbd = 0
        E_cohesion = nbd * _E_COH_PER_D
        # Period scaling: 4d/5d metals have wider d-bands.
        # f_per = 1 + (per-4) × sin²₃ (heavier metals → stronger cohesion)
        if per > 4:
            E_cohesion *= 1.0 + (per - 4) * S3
        E += E_cohesion / n

        # ── 2. CFSE in octahedral water field ──
        # Uses ION n_d_ion (aqueous state).
        # Attenuated by sin²₃ (P1×P2 cross-channel throughput).
        f_cfse = _CFSE_FACTOR.get(n_d_ion, 0.0)
        E_cfse = f_cfse * _DELTA_OCT_PT * S3
        E += E_cfse / n

        # ── 2a. TRIVALENT d-block extra metallic cohesion ──
        # For n >= 3: removing 3 electrons from a d-metal is costly.
        # The IE3 cost (typically 27-34 eV) far exceeds IE1+IE2.
        # The base E° formula uses ie_eff = (IE1+IE2)/2 which underestimates
        # the dissolution cost for trivalent. Add the IE3 penalty projected
        # through P₂ cross-channel:
        # E_IE3 = (IE3 - IE2) × sin²₅ × sin²₃ / n
        # This represents the EXTRA cost of extracting the 3rd electron,
        # filtered through P₂ (d-channel) and P₁ (spatial projection).
        if n >= 3 and Z in _IE3_DBLOCK:
            ie2 = IE2_DBLOCK.get(Z, ie1 * 2.0)
            ie3 = _IE3_DBLOCK[Z]
            # IE3 penalty for dissolution of M³⁺ — the high cost of
            # removing the 3rd electron stabilizes the metal.
            # Projected through P₁ channel: (IE3 - IE2) × sin²₃ / n
            # Attenuated by ion CFSE (d³ in water gains strong CFSE,
            # partially offsetting the IE3 cost).
            cfse_atten = (1.0 + abs(_CFSE_FACTOR.get(n_d_ion, 0.0))) ** 2
            E_ie3_penalty = (ie3 - ie2) * S3 / (n * cfse_atten)
            E += E_ie3_penalty / n

        # ── 2c. HUND EXCHANGE stabilization for trivalent d-block ──
        # When M→M³⁺ gains exchange pairs (e.g., d6→d5: +4 pairs),
        # the ion is stabilized in aqueous solution → E more positive.
        # Exchange pairs: n_d_ion*(n_d_ion-1)/2 vs n_d*(n_d-1)/2
        # (only count majority-spin for HS, i.e., min(n_d, P2) pairs).
        # J_exch = sin²₅ × Ry / (P₂ × P₁) (exchange integral from PT).
        if n >= 3:
            n_up_ion = min(n_d_ion, P2)
            n_up_metal = min(n_d, P2)
            pairs_ion = n_up_ion * (n_up_ion - 1) // 2
            pairs_metal = n_up_metal * (n_up_metal - 1) // 2
            delta_pairs = pairs_ion - pairs_metal
            if delta_pairs > 0:
                J_exch = S5 * RY / (P2 * P1)
                E_exch = delta_pairs * J_exch
                E += E_exch / n

        # ── 2b. CFSE METALLIC (P₂ d-band stabilization of solid metal) ──
        # Same sin²(n_d×π/P₂) dependence as aqueous, but uses Δ_band
        # instead of Δ_oct. STABILISES the metal → E more POSITIVE.
        # Uses NEUTRAL n_d (metal state), attenuated by sin²₅ (P₂ channel).
        f_cfse_metal = _CFSE_FACTOR.get(n_d % (2 * P2 + 1), 0.0)
        E_cfse_metal = -f_cfse_metal * _DELTA_BAND_PT * S5
        # Period scaling: 4d/5d have wider d-bands → stronger CFSE
        if per > 4:
            E_cfse_metal *= 1.0 + (per - 4) * S3
        E += E_cfse_metal / n

        # ── 3. d10s2 metallic weakness penalty (Group 12: Zn, Cd, Hg) ──
        # Scaled by 4/per: heaviest d10s2 (Hg) has WEAKER penalty because
        # relativistic s-contraction partially restores metallic bonding.
        if Z in _D10S2:
            # Penalty: original 4/per scaling (Zn=1.0, Cd=0.8, Hg=0.667)
            avg_cohesion = P2 * _E_COH_PER_D * S_HALF * (4.0 / per)
            E -= avg_cohesion / n
            # ── 3b. d10s2 RESIDUAL s-band cohesion (per >= 5) ──
            # 4d/5d d10s2 metals retain s-band metallic bonding from wider
            # s-orbitals. E_s_coh = sin²₃ × IE₁ × s × (per - P₁) / P₂
            # Uses (per - P₁) to capture that 4d (per=5) and 5d (per=6)
            # have progressively wider s-bands beyond the 3d baseline.
            if per >= 5:
                E_s_coh = S3 * ie1 * S_HALF * (per - P1) / P2
                E += E_s_coh / n
            # ── 3c. RELATIVISTIC 6s CONTRACTION for d10s2 (per >= 6) ──
            # Hg: the 6s² pair contracts relativistically.
            # Moderate: za² × sin²₅² × IE₁ / P₂
            # Double sin²₅ filter (P₂² channel) to avoid overcorrection.
            if per >= 6:
                E_rel_s = za2 * S5 * S5 * ie1 / P2
                E += E_rel_s / n

        # ── 4. RELATIVISTIC 5d CONTRACTION (per >= 6 d-block) ──
        # For 5d metals, the relativistic contraction za² = (Z×α)²
        # widens the d-band, increasing cohesion. Two channels:
        # a) Bandwidth enhancement: za² × Ry × sin²₅ × n_d / P₂
        #    All n_d electrons benefit from the wider band.
        # b) s-d hybridization enhancement: za² × sin²₃ × IE₁ × s / P₁
        #    The relativistic s-contraction strengthens s-d mixing.
        if per >= 6 and Z not in _D10S2:
            # Relativistic 5d contraction for 3rd-row transition metals.
            # Three PT channels:
            # a) s-d hybridization: za² × IE₁ × sin²₃ / P₁
            E_rel_sd = za2 * ie1 * S3 / P1
            # b) d-band promotion bonus (for s→d promoted configs: ns=1):
            #    The promotion creates an extra d-bonding electron,
            #    with energy gain za² × IE₁ × sin²₅
            E_rel_promo = za2 * ie1 * S5 if ns == 1 else 0.0
            # c) d-band widening: za² × sin²₅ × Ry × nbd / P₂
            E_rel_d = za2 * S5 * RY * nbd / P2
            E += (E_rel_sd + E_rel_promo + E_rel_d) / n

    # ══════════════════════════════════════════════════════════════════
    # d-block s-BAND metallic bonding — BIFURCATION at n_d = P₂
    # ══════════════════════════════════════════════════════════════════
    # PT Principle 4 bifurcation: at n_d > P₂ (beyond half-fill), the
    # s-band becomes a SEPARATE metallic channel (P₁×P₂ cross).
    # For n_d ≤ P₂, the s-d hybridization is captured by IE.
    # For n_d > P₂, the minority-spin d-band drives s-band separation.
    #
    # E_s = n_s × sin²₃ × IE₁ × s / P₁ × (n_d - P₂) / P₂
    # The (n_d-P₂)/P₂ factor smoothly ramps from 0 at d⁵ to 1 at d¹⁰.
    #
    # Skip d10s1 (Cu, Ag, Au) and d10s2 (Zn, Cd, Hg) — handled separately.
    _nd_raw = _d_electron_count(Z) if blk == 'd' else 0
    if (n >= 2 and blk == 'd' and _nd_raw > P2
            and Z not in _D10_METALS and Z not in _D10S2):
        f_late = (n_d - P2) / P2  # 0.2 for d6, 0.4 for d7, etc.
        E_s_metal = ns * S3 * ie1 * S_HALF / P1 * f_late
        E += E_s_metal / n

    # ══════════════════════════════════════════════════════════════════
    # s-block metallic cohesion (Na-Cs, Mg-Ba — per >= P₁)
    # ══════════════════════════════════════════════════════════════════
    # Bifurcation at per = P₁ = 3:
    #   per < P₁ (Li, Be): compact s-band, cohesion implicit in IE-based E°
    #   per >= P₁ (Na-Cs, Mg-Ba): extended s-band, explicit cohesion needed
    #
    # E_coh = sin²₃ × IE × s / P₁
    #   sin²₃ : P₁ channel throughput for s-band bonding
    #   IE    : atomic binding energy (proportional to bond strength)
    #   s     : half-occupation factor (1 s-electron per atom)
    #   /P₁   : shared among P₁ nearest-neighbor bonds (BCC effective CN)
    # ── s-block BIFURCATION BOUNDARY cohesion (per = P₁) ──────────
    # At per = P₁ = 3, the s-band transitions from compact (per<P₁) to
    # extended (per>P₁). This boundary atom (Mg, Al-group) has BOTH:
    #   - compact s-orbital (residual sp-hybridization from per=2 overlap)
    #   - extended metallic bonding (onset of wide s-band)
    # The compact sp-contribution is D₃ × IE × s / P₁ (sub-harmonic
    # coupling of the δ₃ holonomic correction on Z/(2P₁)Z).
    # This adds to the regular cohesion (below) without double-counting.
    if blk == 's' and per == P1 and n >= 2:
        E_coh_bifurc = D3 * ie1 * S_HALF / P1
        E += E_coh_bifurc / n

    if blk == 's' and per >= P1:
        # Period-enhanced cohesion: heavier s-metals have wider s-bands.
        # E_coh = sin²₃ × IE × s / P₁ × (1 + (per - P₁) × sin²₅)
        f_per_s = 1.0 + (per - P1) * S5
        E_coh_s = S3 * ie1 * S_HALF / P1 * f_per_s
        E += E_coh_s / n

    # ══════════════════════════════════════════════════════════════════
    # s-block COMPACT lattice cohesion (Li, Be — per < P₁)
    # ══════════════════════════════════════════════════════════════════
    # For per < P₁: compact atom, strong sp-hybridized metallic bonding.
    # Be (Z=4): HCP close-packed, cohesion = 3.32 eV/atom (one of the
    # highest for s-block metals). The compact 2s shell hybridizes with 2p.
    # E_coh = sin²₃ × IE × s / per
    # Only for n >= 2 (divalent) — Li+ is monovalent and well-captured.
    if blk == 's' and per < P1 and n >= 2:
        E_coh_compact = S3 * ie1 * S_HALF / per
        E += E_coh_compact / n

    # ══════════════════════════════════════════════════════════════════
    # p-block trivalent metallic cohesion (Al³⁺, Ga³⁺, In³⁺)
    # ══════════════════════════════════════════════════════════════════
    # For n >= 3 p-block metals: strong sp-hybridized metallic bond
    # stabilizes the metal, pushing E° more positive.
    # E_coh = sin²₃ × Ry / per (P1 channel, one bond per period shell)
    if n >= 3 and blk == 'p' and Z not in _INERT_PAIR:
        E_coh_p3 = S3 * RY / per
        E += E_coh_p3 / n

    # ══════════════════════════════════════════════════════════════════
    # INERT PAIR correction (Sn²⁺, Pb²⁺, Ge²⁺ — s²p² configuration)
    # ══════════════════════════════════════════════════════════════════
    # The s² pair is locked by P1 bifurcation (only p-electrons removed).
    # The metal has strong sp-hybridized cohesion. The metallic bond
    # energy is the total ionization energy PROJECTED through P1:
    #   E_coh = sin²₃ × (IE1 + IE2) / 2
    # This captures that p-electrons participate in metallic bonding
    # but are NOT ionized in the +2 state (s² intact).
    if Z in _INERT_PAIR and n == 2 and blk == 'p':
        ie2_val = _IE2_PBLOCK.get(Z, ie1 * 2.0 * (1.0 + S3 / 2.0))
        ie_avg = (ie1 + ie2_val) / 2.0
        # Period scaling: (P₂/per)² — heavier inert-pair metals have
        # weaker sp-metallic bonding due to relativistic orbital expansion.
        per_scale = (P2 / per) ** 2
        E_coh_inert = S3 * ie_avg * per_scale
        E += E_coh_inert / n

    # ══════════════════════════════════════════════════════════════════
    # RELATIVISTIC 6s² NOBILITY for heavy p-block (per >= 6)
    # ══════════════════════════════════════════════════════════════════
    # For Z > 70 p-block, the 6s² pair contracts relativistically via
    # za² = (Z×alpha)². This makes the metal MUCH more cohesive and
    # noble than IE alone predicts (errors of -2 V without correction).
    #
    # Physics: the squared relativistic factor za²² filters IE through
    # P1 (3 spatial dimensions). Valence p-electrons screen the 6s²
    # effect via the complementary channel cos²₃ (persistence duality).
    #
    # E_rel_noble = za²² × IE₁ × P₁ / (1 + (n_p - 1) × cos²₃)
    # For n >= 3: additional inert-pair breaking cost za² × sin²₃ × IE₁
    #
    # Skip elements already handled by the inert-pair correction (Pb²⁺).
    if blk == 'p' and per >= 6 and Z not in _INERT_PAIR:
        # Valence p-electron count: Z offset within the p-block
        from ptc.periodic import period_start
        _p_start = period_start(per + 1) - 6  # first p-element in period
        n_p = Z - _p_start + 1

        # Base: squared relativistic filter × IE × P1
        E_rel_base = za2 ** 2 * ie1 * P1

        # p-electron screening via cos²₃ duality channel
        p_screen = 1.0 / (1.0 + (n_p - 1) * C3)

        # Extra cost for n >= 3: breaking the 6s² pair through P1 channel
        E_pair_break = za2 * S3 * ie1 if n >= 3 else 0.0

        E_rel_noble = E_rel_base * p_screen + E_pair_break
        E += E_rel_noble

    return E


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Redox reaction predictor                                       ║
# ╚════════════════════════════════════════════════════════════════════╝

def predict_redox(
    Z_red: int, n_red: int,
    Z_ox: int, n_ox: int,
) -> dict:
    """Predict whether a redox reaction is spontaneous.

    The half-reactions are:
      Oxidant:   M_ox^n+ + n e- -> M_ox   (reduction, cathode)
      Reductant: M_red -> M_red^n+ + n e-  (oxidation, anode)

    Parameters
    ----------
    Z_red : int
        Atomic number of the reducing agent.
    n_red : int
        Charge of the reductant cation produced.
    Z_ox : int
        Atomic number of the oxidising agent (being reduced).
    n_ox : int
        Charge of the oxidant cation consumed.

    Returns
    -------
    dict with keys:
        E_cathode (V), E_anode (V), E_cell (V),
        delta_G (eV), spontaneous (bool),
        symbol_red, symbol_ox
    """
    e_cathode = standard_potential_SHE(Z_ox, n_ox)
    e_anode = standard_potential_SHE(Z_red, n_red)

    e_cell = e_cathode - e_anode

    # Electron transfer count: LCM-based
    n_transfer = _lcm(n_red, n_ox)

    # delta_G = -n * F * E_cell  (in eV, F = 1 eV/V per electron)
    delta_g = -n_transfer * e_cell

    sym_red = SYMBOLS.get(Z_red, f"Z{Z_red}")
    sym_ox = SYMBOLS.get(Z_ox, f"Z{Z_ox}")

    return {
        "E_cathode": e_cathode,
        "E_anode": e_anode,
        "E_cell": e_cell,
        "n_electrons": n_transfer,
        "Delta_G": delta_g,
        "spontaneous": e_cell > 0,
        "symbol_red": sym_red,
        "symbol_ox": sym_ox,
    }


def _lcm(a: int, b: int) -> int:
    """Least common multiple."""
    return abs(a * b) // math.gcd(a, b)


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Activity series                                                ║
# ╚════════════════════════════════════════════════════════════════════╝

# Extended activity series — 30 common metals with standard oxidation states
_ACTIVITY_METALS = [
    # Alkali (M+)
    (3, 1),    # Li+
    (11, 1),   # Na+
    (19, 1),   # K+
    (37, 1),   # Rb+
    (55, 1),   # Cs+
    # Alkaline earth (M2+)
    (4, 2),    # Be2+
    (12, 2),   # Mg2+
    (20, 2),   # Ca2+
    (38, 2),   # Sr2+
    (56, 2),   # Ba2+
    # p-block
    (13, 3),   # Al3+
    (50, 2),   # Sn2+
    (82, 2),   # Pb2+
    (81, 1),   # Tl+
    (81, 3),   # Tl3+
    (83, 3),   # Bi3+
    # d-block 1st row
    (22, 2),   # Ti2+
    (24, 3),   # Cr3+
    (25, 2),   # Mn2+
    (26, 2),   # Fe2+
    (27, 2),   # Co2+
    (28, 2),   # Ni2+
    (29, 2),   # Cu2+
    (30, 2),   # Zn2+
    # d-block 2nd/3rd row
    (47, 1),   # Ag+
    (48, 2),   # Cd2+
    (78, 2),   # Pt2+
    (79, 3),   # Au3+
    (80, 2),   # Hg2+
    # Reference
    (1, 1),    # H+
    # Fe3+/Fe and Cu+/Cu (alternative oxidation states)
    (26, 3),   # Fe3+
    (29, 1),   # Cu+
]

# Experimental E° vs SHE (NIST, Bard-Faulkner) for benchmark
_EXP_E0: dict[tuple[int, int], float] = {
    (3, 1): -3.04,  (11, 1): -2.71, (19, 1): -2.93,
    (37, 1): -2.98, (55, 1): -3.03,
    (4, 2): -1.85,  (12, 2): -2.37, (20, 2): -2.87,
    (38, 2): -2.90, (56, 2): -2.91,
    (13, 3): -1.66, (50, 2): -0.14, (82, 2): -0.13,
    (81, 1): -0.34, (81, 3): 0.72,  (83, 3): 0.32,
    (22, 2): -1.63, (24, 3): -0.74, (25, 2): -1.18,
    (26, 2): -0.44, (27, 2): -0.28, (28, 2): -0.26,
    (29, 2): 0.34,  (30, 2): -0.76,
    (47, 1): 0.80,  (48, 2): -0.40, (78, 2): 1.18,
    (79, 3): 1.50,  (80, 2): 0.85,
    (1, 1): 0.00,
    (26, 3): -0.04, (29, 1): 0.52,
}


def activity_series(include_all: bool = True) -> list[dict]:
    """Return the electrochemical activity series sorted by E ascending.

    Parameters
    ----------
    include_all : bool
        If True, include all 30 metals. If False, only the classic 14.

    Each entry: dict with Z, symbol, E_standard (V), n_charge, E_exp (V or None).
    Most reducing (most negative E) first.
    """
    metals = _ACTIVITY_METALS if include_all else [
        m for m in _ACTIVITY_METALS
        if m in {(3,1),(11,1),(19,1),(20,2),(12,2),(13,3),(30,2),
                 (26,2),(28,2),(50,2),(1,1),(29,2),(47,1),(79,3)}
    ]
    rows = []
    for Z, n in metals:
        sym = SYMBOLS.get(Z, f"Z{Z}")
        e_std = standard_potential_SHE(Z, n)
        e_exp = _EXP_E0.get((Z, n))
        rows.append({
            'Z': Z,
            'symbol': sym,
            'E_standard': e_std,
            'n_charge': n,
            'E_exp': e_exp,
        })

    rows.sort(key=lambda r: r['E_standard'])
    return rows


def compute_potential(Z: int, n_charge: int) -> dict:
    """Compute E° for any element and charge — user-facing API.

    Returns a detailed dict with E°, experimental reference (if known),
    error, and metadata.
    """
    sym = SYMBOLS.get(Z, f"Z{Z}")
    e_pt = standard_potential_SHE(Z, n_charge)
    e_exp = _EXP_E0.get((Z, n_charge))
    blk = block_of(Z)
    per = period_of(Z)

    result = {
        'Z': Z,
        'symbol': sym,
        'n_charge': n_charge,
        'E_standard': e_pt,
        'E_exp': e_exp,
        'error': abs(e_pt - e_exp) if e_exp is not None else None,
        'block': blk,
        'period': per,
        'half_reaction': f"{sym}^{n_charge}+ + {n_charge}e⁻ → {sym}",
    }
    return result
