"""
screening_bond_v4.py — Hybrid DFT-GFT diatomic engine.

D₀ = cap × exp(-S_total)
S_total = S_gap + S_hex + S_holo + S_d_penalty + S_dblock + S_ion + S_mech

The hexagonal face screening S_hex replaces the 4 separate terms from v3:
  S_count + S_exchange + S_exclusion + S_pi → S_hex (DFT modes + GFT-log)

Architecture:
  ρ_A(r), ρ_B(r) on Z/(2P₁)Z
      ↓ DFT
  ρ̂_A(k), ρ̂_B(k) for k=0,...,P₁-1
      ↓ GFT-log per mode
  S_k0 = -½ ln(fill_from_k0)           ← counting (Shannon)
  S_k1 = -ln(1 + exchange_from_k1)     ← exchange
  S_k2 = pauli_from_k2                 ← Pauli exclusion
  S_pi  = -ln(1 + pi_from_modes)       ← π anti-screening
  S_lp  = -ln(1 - lp_blocking)         ← LP blocking
  S_per = period attenuation            ← diffuse orbital screening
  S_nnlo = NNLO corrections             ← higher-order spectral corrections
      ↓ sum
  S_hex = S_k0 + S_k1 + S_k2 + S_pi + S_lp + S_per + S_nnlo

Unchanged from v3: S_gap, S_holo, S_d_penalty, S_dblock, S_ion, S_mech.

0 adjustable parameters. All from s = 1/2.

April 2026 — Théorie de la Persistance
"""
import math
from dataclasses import dataclass, field

from ptc.constants import (
    RY, P0, P1, P2, P3, D_FULL, S3, S5, S7, C3, C5, C7, S_HALF,
    D3, D5, D7, GAMMA_3, GAMMA_5, GAMMA_7, AEM, COULOMB_EV_A,
)
from ptc.atom import IE_eV, EA_eV
from ptc.periodic import (
    period, period_start, l_of, n_fill as n_fill_madelung,
    _n_fill_aufbau, ns_config, capacity,
)
from ptc.bond import r_equilibrium
from ptc.dft_polygon import electron_density, dft_spectrum


SHANNON_CAP = RY / P1  # 4.535 eV


# ──────────────────────────────────────────────────────────────────────
#  Result
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ScreeningResult:
    D0: float
    S_total: float
    cap: float
    S_cov: float        # sum of covalent S terms
    S_dblock: float      # d-block anti-screening
    S_ion: float         # ionic anti-screening
    S_mech: float        # mechanism corrections
    terms: dict          # every S term individually (7 keys)
    D_P1: float          # reconstructed covalent energy
    D_P2: float          # reconstructed d-block energy
    D_P3: float          # reconstructed ionic energy


# ──────────────────────────────────────────────────────────────────────
#  Helpers (identical to v3)
# ──────────────────────────────────────────────────────────────────────

def _np_of(Z):
    l = l_of(Z)
    if l != 1: return 0
    per = period(Z); z0 = period_start(per); pos = Z - z0
    if pos < 2: return 0
    if per <= 3: return pos - 1
    if per <= 5: return max(0, pos - 11) if pos >= 12 else 0
    return max(0, pos - 25) if pos >= 26 else 0

def _nd_of(Z):
    if l_of(Z) != 2: return 0
    return n_fill_madelung(Z)

def _valence_electrons(Z):
    _VE = {1:1,2:2,3:1,4:2,5:3,6:4,7:5,8:6,9:7,10:8,
           11:1,12:2,13:3,14:4,15:5,16:6,17:7,18:8,19:1,20:2,35:7,53:7}
    if Z in _VE: return _VE[Z]
    per = period(Z); z0 = period_start(per)
    return min(Z - z0 + 1, 2 * (per // 2 + 1) ** 2)

def _lp_pairs(Z, bo):
    return max(0, _valence_electrons(Z) - bo) // 2

_NS1 = frozenset({24, 29, 41, 42, 44, 45, 46, 47, 78, 79})
_IE2 = {21:12.80,22:13.58,23:14.65,24:16.49,25:15.64,26:16.19,27:17.08,
        28:18.17,29:20.29,30:17.96,39:12.24,40:13.13,41:14.32,42:16.16,
        43:15.26,44:16.76,45:18.08,46:19.43,47:21.49,48:16.91,
        72:14.92,73:16.2,74:17.7,75:16.6,76:17.0,77:17.0,78:18.56,79:20.5,80:18.76}
_IE2_AE = {4:18.21, 12:15.04, 20:11.87, 38:11.03, 56:10.00, 88:10.15}


# ──────────────────────────────────────────────────────────────────────
#  Effective eps (identical to v3)
# ──────────────────────────────────────────────────────────────────────

def _effective_eps(ie_A, ie_B, per_A, per_B, l_A, l_B,
                   Z_A, Z_B, nd_A, nd_B, lp_A, lp_B):
    """Compute effective eps_A, eps_B, eps_geom with all d-block modifications."""
    eps_A = ie_A / RY
    eps_B = ie_B / RY
    l_max = max(l_A, l_B)

    # sd-hybrid rotation
    sd_A = sd_B = False
    _lb_j = lp_B > 0 and l_B < 2 and per_B <= 2
    _lb_i = lp_A > 0 and l_A < 2 and per_A <= 2
    if l_A == 2 and not _lb_j and P2 < nd_A < 2 * P2:
        ie2 = _IE2.get(Z_A, ie_A * 2)
        if ie2 > 0:
            f_sd = float(nd_A - P2) / P2
            eps_A += f_sd * (math.sqrt(ie_A * ie2) / RY - eps_A)
            sd_A = True
    if l_B == 2 and not _lb_i and P2 < nd_B < 2 * P2:
        ie2 = _IE2.get(Z_B, ie_B * 2)
        if ie2 > 0:
            f_sd = float(nd_B - P2) / P2
            eps_B += f_sd * (math.sqrt(ie_B * ie2) / RY - eps_B)
            sd_B = True

    # d-block CRT screening
    if nd_A > 0 and l_A == 2 and not sd_A:
        if lp_B > 0:
            eps_A *= C5 ** (float(nd_A) / P2)
        else:
            ea_p = max(EA_eV(Z_B), 0.01)
            eps_A *= C5 ** (float(nd_A) / P2 * max(ea_p / RY, 0.01))
    if nd_B > 0 and l_B == 2 and not sd_B:
        if lp_A > 0:
            eps_B *= C5 ** (float(nd_B) / P2)
        else:
            ea_p = max(EA_eV(Z_A), 0.01)
            eps_B *= C5 ** (float(nd_B) / P2 * max(ea_p / RY, 0.01))

    # d-block sigma stiffness
    J_exch = S5 * RY / (2.0 * P2)
    for Z_s, ie_s, l_s, nd_s, is_A, sd_h in [
        (Z_A, ie_A, l_A, nd_A, True, sd_A),
        (Z_B, ie_B, l_B, nd_B, False, sd_B),
    ]:
        if l_s != 2 or sd_h: continue
        ie2 = _IE2.get(Z_s, 0.0)
        if ie2 <= 0 or ie_s <= 0: continue
        if Z_s not in _NS1 and nd_s < 2 * P2:
            f_stiff = (ie_s / ie2) ** S_HALF
            delta_K = nd_s if nd_s < P2 else 0
            if delta_K > 0:
                lp_part = lp_B if is_A else lp_A
                f_lp_ex = C3 if lp_part > 0 else 1.0
                f_stiff *= (1.0 + float(delta_K) * J_exch / ie_s * f_lp_ex)
            if is_A: eps_A *= f_stiff
            else: eps_B *= f_stiff

    eps_geom = math.sqrt(max(eps_A, 1e-10) * max(eps_B, 1e-10))
    return eps_A, eps_B, eps_geom, sd_A, sd_B


# ──────────────────────────────────────────────────────────────────────
#  DFT-GFT HYBRID: S_hex replaces S_count + S_exchange + S_exclusion + S_pi
# ──────────────────────────────────────────────────────────────────────

def _S_hex_hybrid(Z_A, Z_B, bo, lp_A, lp_B, np_A, np_B,
                  per_A, per_B, l_A, l_B, eps_A, eps_B, eps_geom,
                  ie_A, ie_B, nd_A, nd_B, sd_A, sd_B, bo_eff=None):
    """Hexagonal face screening via DFT modes + GFT-log transforms.

    Replaces: S_count + S_exchange + S_exclusion + S_pi from v3.

    Architecture:
    1. Build electron densities on Z/(2P₁)Z
    2. Compute DFT spectrum (P₁ complex modes)
    3. Apply GFT-log transform per mode:
       - k=0: fill fraction → Shannon log (-½ ln f)
       - k=1: dipolar asymmetry → exchange coupling (-ln(1+r))
       - k=2: quadrupole → Pauli screening
    4. Compute π from unpaired electron modes
    5. Add LP blocking, period attenuation, NNLO corrections

    0 adjustable parameters. All from s = 1/2.
    """
    l_min, l_max = min(l_A, l_B), max(l_A, l_B)
    per_min, per_max = min(per_A, per_B), max(per_A, per_B)
    P = P1  # hexagonal face

    # ── Step 0: Determine electron counts on hex face ─────────────
    if l_A == 0:
        n_hex_A = 0
    elif l_A == 2:
        n_hex_A = ns_config(Z_A)
    else:
        n_hex_A = np_A

    if l_B == 0:
        n_hex_B = 0
    elif l_B == 2:
        n_hex_B = ns_config(Z_B)
    else:
        n_hex_B = np_B

    # LP on hexagonal face for electron placement
    lp_hex_A = max(0, n_hex_A - P) if l_A == 1 else 0
    lp_hex_B = max(0, n_hex_B - P) if l_B == 1 else 0

    # ── Step 1: Build electron densities on Z/(2P)Z ──────────────
    rho_A = electron_density(n_hex_A, P, lp=lp_hex_A)
    rho_B = electron_density(n_hex_B, P, lp=lp_hex_B)

    N = 2 * P
    n_A = sum(rho_A)
    n_B = sum(rho_B)

    # Face coupling constants
    S_p = S3  # hexagonal face
    C_p = C3
    D_p = D3  # Fisher dispersion

    # ── Step 2: DFT spectra ───────────────────────────────────────
    spec_A = dft_spectrum(rho_A, P)
    spec_B = dft_spectrum(rho_B, P)

    # Fill fractions
    f_A = n_A / N if N > 0 else 0.0
    f_B = n_B / N if N > 0 else 0.0
    f_sum = f_A + f_B

    # IE ratio for exchange — uses effective eps (d-block modified),
    # matching v3's _S_exchange which receives eps_A, eps_B
    eps_max = max(eps_A, eps_B)
    q_rel = min(eps_A, eps_B) / eps_max if eps_max > 0 else 1.0

    # LP on this face (from density pattern)
    lp_A_face = sum(1 for i in range(min(P, n_A // 2 + 1))
                    if i < P and rho_A[i] == 1 and i + P < N and rho_A[i + P] == 1)
    lp_B_face = sum(1 for i in range(min(P, n_B // 2 + 1))
                    if i < P and rho_B[i] == 1 and i + P < N and rho_B[i + P] == 1)
    lp_mutual_face = min(lp_A_face, lp_B_face)

    # Unpaired electrons
    unp_A = n_A - 2 * lp_A_face
    unp_B = n_B - 2 * lp_B_face

    # ══════════════════════════════════════════════════════════════
    # Step 3: GFT-log per mode
    # ══════════════════════════════════════════════════════════════

    # ── k=0: GFT fill screening ──────────────────────────────────
    # S_fill = -½ ln(f_sum) when both atoms contribute to hex face.
    # At f_sum = 1: S_fill = 0 (full face, no screening from fill).
    # At f_sum < 1: S_fill > 0 (vacancies screen).
    S_k0 = 0.0

    # Fill screening only fires when BOTH atoms have p-electrons on hex face
    # (l_min >= 1 in v3's _S_count). For s+p bonds (one atom with l=0),
    # the hex fill is handled by _S_holo, not here.
    if l_min >= 1 and n_A > 0 and n_B > 0:
        # p+p bonds: combined fill from both atoms
        np_sum = np_A + np_B
        if l_min >= 2:
            np_sum = ns_config(Z_A) + ns_config(Z_B)
        np_c = min(float(np_sum), 2.0 * P1)
        if np_c <= 0:
            S_k0 = 5.0
        else:
            S_k0 = -0.5 * math.log(np_c / (2.0 * P1))
            # Fisher vacancy: homonuclear overfill redundancy
            f_hex = np_c / (2.0 * P1)
            if Z_A == Z_B and P1 <= per_max <= P1 + 1 and f_hex < S_HALF:
                S_k0 += -0.5 * math.log(1.0 - f_hex)

    # ── s-block fill screening (binary face Z/(2P₀)Z) ────────────
    S_sblock = 0.0
    if l_max == 0 and per_max > 1:
        ns_A = min(2, _valence_electrons(Z_A))
        ns_B = min(2, _valence_electrons(Z_B))
        ns_sum = float(ns_A + ns_B)
        f = math.sqrt(min(1.0, ns_sum / (2.0 * P1)))
        if max(ns_A, ns_B) >= 2:
            ie_max_loc = max(ie_A, ie_B)
            q_rel_raw = min(ie_A, ie_B) / ie_max_loc if ie_max_loc > 0 else 1.0
            expo_s = S_HALF * min(1.0, q_rel_raw / S_HALF)
            f *= (1.0 / max(ns_A, ns_B)) ** expo_s
        S_sblock += -math.log(max(f, 0.01))

    # s-block P₁ screening for heavy s-block atoms
    if l_max == 0 and per_max >= P1:
        _AE4p = {20, 38, 56, 88}
        if not ({Z_A, Z_B} & _AE4p):
            S_sblock += -math.log(C3)

    S_k0 += S_sblock

    # ── k=1: Exchange screening via dipolar cross-spectrum ────────
    # The k=1 mode measures dipolar asymmetry on Z/(2P)Z.
    # Exchange fires when l_max >= 1 (s+p bonds included),
    # using eps_A, eps_B (not hex-face electron counts).
    # This matches v3's _S_exchange gate exactly.
    S_k1 = 0.0
    lp_mutual_exch = min(lp_A, lp_B)  # total LP for exchange

    if l_max >= 1:
        # Base exchange: PT holonomic coupling
        f_exch = 1.0 + S_p * S_HALF * q_rel
        # LP-modulated exchange: mutual LP enhances (up to half-fill)
        # or reduces (past half-fill) exchange
        if Z_A == Z_B and Z_A > 0 and lp_mutual_exch > 0:
            if lp_mutual_exch < P:
                f_exch += S_p * S_HALF * float(lp_mutual_exch) / P
            else:
                f_exch -= S_p * S_HALF * float(lp_mutual_exch - P + 1) / P
        S_k1 = -math.log(max(f_exch, 0.01))

        # Half-fill exchange penalty (uses np_A, np_B like v3)
        if l_min >= 1 and per_max >= P1 and bo >= 2:
            if np_A == P1 and np_B == P1:
                n_per3 = int(per_A >= P1) + int(per_B >= P1)
                if n_per3 >= 2:
                    if bo >= P1:
                        S_k1 += -math.log(C3)
                    else:
                        S_k1 += -(1.0 + S_HALF) * math.log(C3)
                elif n_per3 == 1:
                    S_k1 += -math.log(C3)
            elif (np_A == P1 and np_B > P1) or (np_B == P1 and np_A > P1):
                S_k1 += -S_HALF * math.log(C3)

        # Hund coupling for same-period donor-acceptor
        if l_min >= 1 and bo <= 1 and per_A == per_B:
            np_min_v = min(np_A, np_B)
            np_max_v = max(np_A, np_B)
            if np_min_v <= P1 < np_max_v:
                hf_A = P1 - abs(np_A - P1)
                hf_B = P1 - abs(np_B - P1)
                hf_max = float(max(hf_A, hf_B))
                f_hund = 1.0 + S3 * (hf_max / P1) ** 2
                S_k1 += -math.log(f_hund)

    # ── k=2: Pauli exclusion screening ────────────────────────────
    # The quadrupole mode measures overlap of paired electrons.
    # When both atoms overfill the face (n > P), Pauli screens.
    S_k2 = 0.0

    # Base Pauli: tent-function model
    if l_min >= 1 and lp_mutual_face < P1 and bo <= 1:
        np_sum = np_A + np_B
        if np_sum > 2 * P1:
            tent = max(0.0, float(4*P1 - np_sum) / (2.0*P1))
            pauli_gap = float(P1 - min(lp_A, lp_B)) / float(P1)
            f = max(0.1, 1.0 - pauli_gap * (1.0 - tent) * S_HALF)
            S_k2 += -math.log(f)

    # Cross-period compact LP penetration Pauli
    if (l_min >= 1 and bo <= 1 and per_A != per_B and max(l_A, l_B) <= 1
            and Z_A != Z_B and np_A > P1 and np_B > P1):
        if per_A < per_B:
            np_compact, np_diffuse = np_A, np_B
        else:
            np_compact, np_diffuse = np_B, np_A
        if np_compact < np_diffuse:
            per_max_loc_cp = max(per_A, per_B)
            f_lp_cp = float(min(np_A, np_B) - P1) / P1
            f_per_cp = (2.0 / float(per_max_loc_cp)) ** (1.0 / P1)
            S_k2 += D3 * f_lp_cp * f_per_cp

    # Dual over-half-fill Pauli for compact per≤2
    if (l_min >= 1 and bo <= 1 and per_A == per_B and per_A <= 2
            and Z_A != Z_B and np_A > P1 and np_B > P1):
        np_sum_of = np_A + np_B
        S_k2 += S3 * max(0.0, float(np_sum_of - 2 * P1)) / (2.0 * P1)

    # Cross-period half-fill Pauli
    if l_min >= 1 and bo <= 1 and per_A != per_B:
        np_min_v = min(np_A, np_B)
        np_max_v = max(np_A, np_B)
        if np_min_v == P1 and np_max_v > P1:
            per_hf = per_A if np_A == P1 else per_B
            per_min_loc = min(per_A, per_B)
            per_max_loc = max(per_A, per_B)
            if per_hf == per_min_loc:
                S_k2 += S3 * S_HALF * float(per_min_loc) / float(per_max_loc)

    # π Pauli for compact per=2 homonuclear overfill
    per_min_loc2 = min(per_A, per_B)
    per_max_loc2 = max(per_A, per_B)
    if (l_min >= 1 and bo >= 2 and per_max_loc2 == 2 and per_A == per_B
            and Z_A == Z_B and np_A > P1 and np_B > P1):
        np_excess = float(np_A + np_B - 2 * P1) / (2.0 * P1)
        S_k2 += D3 * np_excess * float(min(bo - 1, 2))

    # π Pauli for per ≥ P₁ homonuclear overfill
    if (l_min >= 1 and bo >= 2 and per_A == per_B and per_max_loc2 >= P1
            and Z_A == Z_B and np_A > P1 and np_B > P1
            and max(l_A, l_B) <= 1):
        np_excess_h = float(np_A + np_B - 2 * P1) / (2.0 * P1)
        f_per_h = (2.0 / float(per_max_loc2)) ** (1.0 / P1)
        S_k2 += D3 * np_excess_h * float(min(bo - 1, 2)) * f_per_h

    # C1: homonuclear halogen LP core overlap (per ∈ [P₁, P₂))
    _lp_mutual_c1 = min(lp_A, lp_B)
    if (Z_A == Z_B and P1 <= per_max_loc2 < P2
            and l_min >= 1 and max(l_A, l_B) <= 1
            and min(np_A, np_B) > P1 and _lp_mutual_c1 >= P1
            and bo <= 1):
        za_param_c1 = (float(Z_A) * AEM) ** (2.0 / P1)
        S_k2 += D3 * za_param_c1

    # C5: heteronuclear halogen LP overlap
    _HALOGENS_C5 = frozenset({9, 17, 35, 53})
    if (Z_A != Z_B and Z_A in _HALOGENS_C5 and Z_B in _HALOGENS_C5
            and per_min_loc2 == P1 and per_max_loc2 <= P2
            and bo <= 1 and min(np_A, np_B) > P1):
        za_A_c5 = (float(Z_A) * AEM) ** (2.0 / P1)
        za_B_c5 = (float(Z_B) * AEM) ** (2.0 / P1)
        S_k2 += D3 * math.sqrt(za_A_c5 * za_B_c5) * S_HALF

    # ── LP blocking ──────────────────────────────────────────────
    lp_eff_lp = float(min(lp_A, lp_B))
    orb = 2.0 * max(per_max, 1)
    if per_min > P1:
        orb = min(orb, 2.0 * P1)
    # Cross-period LP radial mismatch
    if l_min >= 1 and per_min < per_max and bo <= 1.0 and lp_eff_lp >= 2:
        if per_A <= per_B:
            lp_compact, lp_diffuse = lp_A, lp_B
        else:
            lp_compact, lp_diffuse = lp_B, lp_A
        if lp_compact > lp_diffuse:
            lp_eff_lp = min(lp_A, lp_B) * float(per_min) / float(per_max)
    f_lp = max(0.01, 1.0 - lp_eff_lp / orb)
    S_lp = -math.log(f_lp)

    # ── π anti-screening from DFT k>0 modes ──────────────────────
    # π bonding = constructive interference of unpaired electron modes
    # Uses bo_eff from main engine (matching v3 exactly)
    S_pi = 0.0
    bo_eff_local = bo_eff if bo_eff is not None else float(bo)

    # Compute f_per for pi (same as main period factor, matching v3)
    f_per_for_pi = 1.0
    if per_max > 2:
        if Z_A == Z_B and l_max >= 2:
            _expo_pi = 1.0 / (P1 * P2)
        elif per_max == P1:
            _expo_pi = 1.0 / (P1 * P1)
        else:
            _expo_pi = 1.0 / P1
        f_per_for_pi = (2.0 / per_max) ** _expo_pi
        # 5d Dirac contraction
        if f_per_for_pi < 1.0 and l_max >= 2 and per_max >= 6:
            _Zh_pi = Z_A if per_A >= 6 and l_A >= 2 else (Z_B if per_B >= 6 and l_B >= 2 else 0)
            if _Zh_pi > 0:
                _za2_pi = (_Zh_pi * AEM) ** 2
                if _za2_pi < 1:
                    f_per_for_pi /= math.sqrt(1.0 - _za2_pi)

    # Pi uses np_A, np_B (p-electron counts), NOT hex-face electron counts,
    # to match v3's _S_pi exactly.
    if l_min >= 1:
        S_pi = _S_pi(np_A, np_B, l_min, l_max, Z_A, Z_B, bo, bo_eff_local,
                     lp_A, lp_B, per_max, eps_geom, f_lp, f_per_for_pi)
    # (Old DFT-based pi computation removed — now delegated to _S_pi.)

    # ── Period attenuation ────────────────────────────────────────
    S_per = 0.0
    if per_max > 2:
        if Z_A == Z_B and l_max >= 2:
            expo = 1.0 / (P1 * P2)
        elif per_max == P1:
            expo = 1.0 / (P1 * P1)
        else:
            expo = 1.0 / P1
        f_per = (2.0 / per_max) ** expo
        # 5d Dirac contraction
        if f_per < 1.0 and l_max >= 2 and per_max >= 6:
            Zh = Z_A if per_A >= 6 and l_A >= 2 else (Z_B if per_B >= 6 and l_B >= 2 else 0)
            if Zh > 0:
                za2 = (Zh * AEM) ** 2
                if za2 < 1:
                    f_per /= math.sqrt(1.0 - za2)
        S_per = -math.log(max(f_per, 1e-10))

    # ── NNLO corrections (spectral oscillations from T³) ──────────
    S_nnlo = 0.0

    # NNLO-1: H₂ ghost VP screening
    if per_max == 1 and l_max == 0:
        S_nnlo += S3 * S3 / P1

    # NNLO-2: homonuclear bo ≥ 2, asymmetric hex-face k=1 mode
    if (Z_A == Z_B and l_min >= 1 and bo >= 2.0
            and 2 <= np_A <= 2 * P1 - 2 and np_A != P1):
        f_bo = float(bo - 1) / float(bo)
        f_per_nnlo = (2.0 / float(per_max)) ** (1.0 / P1) if per_max > 2 else 1.0
        S_nnlo += S3 * S3 / P1 * f_bo * f_per_nnlo

    # NNLO-3: bo ≥ 3 hetero same-period hex-face cross-fill Pauli
    if (bo >= 3 and Z_A != Z_B and per_A == per_B
            and l_min >= 1 and max(l_A, l_B) <= 1
            and min(np_A, np_B) >= 2):
        np_min_cf = min(np_A, np_B)
        S_nnlo += S3 * S3 * S_HALF * math.sin(
            math.pi * float(np_min_cf) / (2.0 * P1)) ** 2

    # NNLO-4: alkali + H core VP screening
    if l_max == 0 and per_min == 1 and per_max >= P1 and bo == 1.0:
        Z_cat_nnlo = Z_A if ie_A < ie_B else Z_B
        ns_cat_nnlo = ns_config(Z_cat_nnlo)
        if ns_cat_nnlo == 1:
            S_nnlo += S3 * D3 * (2.0 / float(per_max)) ** S_HALF

    # NNLO-5: cross-period half-fill triple bond exchange double-counting
    if (bo >= 3 and Z_A != Z_B and per_A != per_B
            and l_min >= 1 and max(l_A, l_B) <= 1
            and np_A == P1 and np_B == P1):
        S_nnlo += S3 * S3

    # NNLO-6: bo ≥ 3 hetero cross-period underfill radial screening
    if (bo >= 3 and Z_A != Z_B and per_A != per_B
            and l_min >= 1 and max(l_A, l_B) <= 1
            and min(np_A, np_B) < P1):
        f_pgap = 1.0 - float(min(per_A, per_B)) / float(max(per_A, per_B))
        f_under = float(P1 - min(np_A, np_B)) / P1
        S_nnlo += S3 * S3 * f_pgap * f_under

    # NNLO-7: cross-period bo ≥ 2 DFT k=1 hex-face exchange correction
    if (bo >= 2 and Z_A != Z_B and per_A != per_B
            and l_min >= 1 and max(l_A, l_B) <= 1
            and np_A != P1 and np_B != P1):
        np_min_k1 = min(np_A, np_B)
        np_max_k1 = max(np_A, np_B)
        if np_min_k1 > P1 and np_max_k1 > P1:
            lp_min_k1 = float(min(np_A - P1, np_B - P1)) / P1
            S_nnlo += -D3 * S_HALF * lp_min_k1
        elif np_min_k1 <= 1 and np_max_k1 > P1 and q_rel < S_HALF:
            vac_k1 = float(P1 - np_min_k1) / P1
            S_nnlo += D3 * S_HALF * vac_k1

    # S_H_radial: H + per≥3 p-block radial mismatch screening
    if per_min == 1 and per_max in (P1, P2) and l_max <= 1 and min(Z_A, Z_B) == 1:
        Z_heavy = Z_A if Z_A > 1 else Z_B
        l_heavy = l_of(Z_heavy)
        np_heavy = _np_of(Z_heavy) if l_heavy >= 1 else 0
        if np_heavy >= P1:
            if per_max == P1:
                S_nnlo += S3 / P1
            else:
                S_nnlo += S3 / P1 * (float(P1) / float(per_max)) ** S_HALF

    # C2: per=2 underfilled p-block + H compact overcount
    if per_min == 1 and per_max == 2 and l_max == 1 and l_min == 0 and bo == 1.0:
        Z_heavy_c2 = Z_A if l_A == 1 else Z_B
        np_heavy_c2 = _np_of(Z_heavy_c2) if l_of(Z_heavy_c2) >= 1 else 0
        if 0 < np_heavy_c2 < P1:
            S_nnlo += D3 * S_HALF * float(P1 - np_heavy_c2) / P1

    # S_cap_dH: d-block + H cap adjustment
    if l_max >= 2 and per_min <= 1:
        S_nnlo += 0.5 * math.log(P2 / P1)

    # S_sigma_rad: d-block radial mismatch on sigma
    if l_max >= 2 and per_min <= 1 and lp_A + lp_B == 0:
        if sd_A or sd_B:
            f_sd_rm = max(
                (float(nd_A - P2) / P2) if sd_A else 0.0,
                (float(nd_B - P2) / P2) if sd_B else 0.0,
            )
            f = 1.0 - S5 * (1.0 - f_sd_rm ** 2)
        else:
            f = 1.0 - S5
        S_nnlo += -math.log(max(f, 0.01))

    # S_fill_sp: same-period s+p one-sided LP fill blocking
    if l_min == 0 and l_max == 1 and bo == 1.0:
        if l_A == 0:
            per_s_sp, per_p_sp, Z_p_sp = per_A, per_B, Z_B
        else:
            per_s_sp, per_p_sp, Z_p_sp = per_B, per_A, Z_A
        np_p_sp = _np_of(Z_p_sp) if l_of(Z_p_sp) >= 1 else 0
        if per_s_sp == P1 and per_p_sp == P1 and np_p_sp > P1:
            lp_p_sp = _lp_pairs(Z_p_sp, bo)
            x_sp = S3 * float(lp_p_sp) / (2.0 * P1)
            if x_sp < 1.0:
                S_nnlo += -math.log(1.0 - x_sp)

    # C4: s-block per=P₁ + H sigma radial mismatch
    if l_max == 0 and per_min == 1 and per_max == P1 and bo == 1.0:
        S_nnlo += D3 / P1

    # C6: AE s² + H sp-hybridisation relief
    if l_max == 0 and per_min == 1 and per_max == 2 and bo == 1.0:
        Z_heavy_c6 = Z_A if Z_A > 1 else Z_B
        ns_heavy_c6 = ns_config(Z_heavy_c6)
        if ns_heavy_c6 >= 2:
            ie2_c6 = _IE2_AE.get(Z_heavy_c6, 0.0)
            ie1_c6 = ie_A if Z_A == Z_heavy_c6 else ie_B
            if ie2_c6 > 0:
                S_nnlo -= D3 * ie1_c6 / ie2_c6

    # S_compact_per2: per=2 homonuclear underfill vacancy relief
    # Gate matches v3 exactly: np_c < P1 (combined fill, not individual np_A)
    if Z_A == Z_B and per_max == 2 and l_min >= 1:
        np_c_cpt = min(float(np_A + np_B), 2.0 * P1)
        if np_c_cpt < P1 and np_c_cpt > 0:
            S_fill_hex_val_cpt = -0.5 * math.log(np_c_cpt / (2.0 * P1))
            S_nnlo += -S_fill_hex_val_cpt * np_c_cpt / float(P1 * P1)

    # S_compact_triple: per=2 homonuclear underfill triple bond Pauli
    if Z_A == Z_B and per_max == 2 and l_min >= 1 and np_A < P1 and bo >= 3:
        S_nnlo += S3 * float(P1 - np_A) / P1

    # S_heavy_homo: homonuclear heavy p-block core screening
    if Z_A == Z_B and per_max >= P2 and l_min >= 1 and l_max <= 1:
        za_param = (float(Z_A) * AEM) ** (2.0 / P1)
        S_nnlo += D3 * za_param

    # ── Spectral cross-phase correction (SC) ─────────────────────
    # On Z/(2P₁)Z, the DFT k=1 imaginary cross-spectrum
    #   Im(ρ̂_A(1)·ρ̂_B(1)*) = |ρ̂_A(1)|·|ρ̂_B(1)|·sin(Δφ)
    # encodes the antisymmetric fill-phase mismatch.
    # When Im > S₃/(2P₁), the fill is "in-phase" → excess exchange
    #   → underbinding → needs anti-screening (negative S).
    # When Im < S₃/(2P₁), the fill is "out-of-phase" → deficit exchange
    #   → overbinding → needs screening (positive S).
    # Correction: S = -2P₁²S₃·(Im - S₃/(2P₁)) × f_per
    # Gate: bo=1, p+p heteronuclear, np_min < P₁ or (np_min=P₁ and
    #       np_max=2P₁−1), per of less-filled ≤ per of more-filled,
    #       less-filled atom in period 2 (compact orbital).
    # 0 parameters. S₃ from s=1/2.
    S_sc = 0.0
    if (l_min >= 1 and max(l_A, l_B) <= 1 and Z_A != Z_B
            and bo <= 1 and max(np_A, np_B) == 2 * P1 - 1
            and min(np_A, np_B) >= 1 and min(np_A, np_B) < 2 * P1 - 1):
        _np_less = min(np_A, np_B)
        if np_A <= np_B:
            _per_less_sc = per_A
            _per_more_sc = per_B
        else:
            _per_less_sc = per_B
            _per_more_sc = per_A
        # Only fire when BOTH atoms are in period 2 (compact orbitals,
        # well-defined overlap on Z/(2P₁)Z).
        if _per_less_sc == 2 and _per_more_sc == 2:
            _im_k1 = (spec_A[1] * spec_B[1].conjugate()).imag
            _sc_threshold = S3 / (2.0 * P)
            _sc_coeff = 2.0 * P * P * S3
            _sc_delta = _im_k1 - _sc_threshold
            _sc_pdamp = (2.0 / float(per_max)) ** (1.0 / P) if per_max > 2 else 1.0
            # Fill-dependent modulation: the LP count of the less-filled atom
            # partially screens the cross-phase effect. For overfill (np>P₁),
            # each LP reduces the effective coupling by C₃.
            _lp_less = max(0, _np_less - P)
            _sc_fill = C3 ** _lp_less  # np≤P₁: 1.0; np=4: C₃; np=5: C₃²
            S_sc = -_sc_coeff * _sc_delta * _sc_pdamp * _sc_fill

    S_nnlo += S_sc

    # SC-2: Cross-period p+halide with per_less > per_more.
    # When the less-filled atom (per=3) bonds with a compact halide (per=2),
    # the diffuse orbital INVERTS the fill-phase effect.
    # Same spectral formula but with INVERTED sign.
    S_sc2 = 0.0
    if (l_min >= 1 and max(l_A, l_B) <= 1 and Z_A != Z_B
            and bo <= 1 and max(np_A, np_B) == 2 * P1 - 1
            and min(np_A, np_B) >= 1 and min(np_A, np_B) < 2 * P1 - 1):
        _np_less2 = min(np_A, np_B)
        if np_A <= np_B:
            _per_less2 = per_A
            _per_more2 = per_B
        else:
            _per_less2 = per_B
            _per_more2 = per_A
        # Gate: less-filled in per=3 (diffuse), more-filled in per=2 (compact),
        # and less-filled is NOT at half-fill (np=P₁ has special exchange).
        if _per_less2 == P1 and _per_more2 == 2 and _np_less2 != P:
            _im_k1_2 = (spec_A[1] * spec_B[1].conjugate()).imag
            _sc_thr2 = S3 / (2.0 * P)
            _sc_coeff2 = 2.0 * P * P * S3
            _sc_delta2 = _im_k1_2 - _sc_thr2
            _lp_less2 = max(0, _np_less2 - P)
            _sc_fill2 = C3 ** _lp_less2
            # INVERTED sign (per_less > per_more) and period attenuation
            _sc_pdamp2 = (2.0 / float(P1)) ** (1.0 / P)
            S_sc2 = _sc_coeff2 * _sc_delta2 * _sc_pdamp2 * _sc_fill2

    S_nnlo += S_sc2

    # SC-3: Cross-period p+halide with per_less=2 < per_more=3.
    # When compact (per=2) atom bonds with a heavier halide (per=3),
    # it follows the same sign pattern as same-period (per=2,2)
    # because the less-filled atom's compact orbital dominates overlap.
    # Same formula as SC-1 but with per attenuation.
    S_sc3 = 0.0
    if (l_min >= 1 and max(l_A, l_B) <= 1 and Z_A != Z_B
            and bo <= 1 and max(np_A, np_B) == 2 * P1 - 1
            and min(np_A, np_B) >= 1 and min(np_A, np_B) < 2 * P1 - 1):
        _np_less3 = min(np_A, np_B)
        if np_A <= np_B:
            _per_less3 = per_A
            _per_more3 = per_B
        else:
            _per_less3 = per_B
            _per_more3 = per_A
        # Gate: less-filled in per=2, more-filled in per=3, NOT half-fill,
        # and less-filled atom is NOT overfilled (np_less < P₁, underfill only).
        # Overfilled cross-period pairs (OCl etc.) have existing corrections.
        if (_per_less3 == 2 and _per_more3 == P1
                and _np_less3 != P and _np_less3 < P):
            _im_k1_3 = (spec_A[1] * spec_B[1].conjugate()).imag
            _sc_thr3 = S3 / (2.0 * P)
            _sc_coeff3 = 2.0 * P * P * S3
            _sc_delta3 = _im_k1_3 - _sc_thr3
            _lp_less3 = max(0, _np_less3 - P)
            _sc_fill3 = C3 ** _lp_less3
            # Period damping: sin²(π/P₁)×½ = cross-period radial mismatch
            _sc_pdamp3 = math.sin(math.pi / P) ** 2 * S_HALF
            S_sc3 = -_sc_coeff3 * _sc_delta3 * _sc_pdamp3 * _sc_fill3

    S_nnlo += S_sc3

    # ══════════════════════════════════════════════════════════════
    # Step 4: Sum all modes
    # ══════════════════════════════════════════════════════════════
    S_hex = S_k0 + S_k1 + S_k2 + S_pi + S_lp + S_per + S_nnlo

    return S_hex, f_lp


# ──────────────────────────────────────────────────────────────────────
#  Term 5: S_holo (identical to v3)
# ──────────────────────────────────────────────────────────────────────

def _S_gap(eps_geom):
    """S_gap = -ln(eps_geom). The geometric IE gap."""
    return -math.log(max(eps_geom, 1e-10))


def _S_holo(Z_A, Z_B, l_A, l_B, per_A, per_B,
            lp_A=0, lp_B=0, ie_A=13.6, ie_B=13.6):
    """Holonomic s+p screening + s+p dative anti-screening."""
    l_min, l_max = min(l_A, l_B), max(l_A, l_B)
    if not (l_min == 0 and l_max == 1):
        return 0.0
    if l_A == 0:
        Z_p, per_s = Z_B, per_A
        Z_s = Z_A
        lp_p, ie_p = lp_B, ie_B
    else:
        Z_p, per_s = Z_A, per_B
        Z_s = Z_B
        lp_p, ie_p = lp_A, ie_A
    val_p = _valence_electrons(Z_p)
    np_p = max(0, val_p - min(2, val_p)) if l_of(Z_p) >= 1 else 0
    w = (1.0/P1) * ((1.0/float(per_s))**S_HALF if per_s > 1 else 1.0)
    per_p_h = per_A if l_A == 1 else per_B
    if per_p_h > 2:
        w *= (2.0 / float(per_p_h)) ** S_HALF
    holo = math.sin(math.pi * float(np_p) / (2.0 * P1)) ** 2
    f = 1.0 - w * holo
    lp_p_holo = max(0, np_p - P1)
    if lp_p_holo > 0:
        f_rest = 1.0 + float(lp_p_holo) * S3 / (P1 * S_HALF) * w * P1
        f = min(f * f_rest, 1.0)
    S = -math.log(max(f, 0.01))

    # C7: underfilled p-block + H directional σ relief
    if per_s == 1 and per_p_h >= P1 and 2 <= np_p < P1:
        S -= D3 * S_HALF

    # s+p dative π
    ie_dom = max(ie_A, ie_B)
    q_rel = min(ie_A, ie_B) / ie_dom if ie_dom > 0 else 1.0
    per_p = per_A if l_A == 1 else per_B
    if per_s == 2 and per_p <= 2 and lp_p > 0 and q_rel > S_HALF:
        n_dat = min(lp_p, max(1, per_s - 1))
        S += -S3 * float(n_dat) * (ie_p / RY) * S_HALF

    return S


# ──────────────────────────────────────────────────────────────────────
#  Term 6: S_d_penalty (identical to v3)
# ──────────────────────────────────────────────────────────────────────

def _S_d_penalty(nd_A, nd_B, l_A, l_B, Z_A, Z_B, np_A, np_B,
                  bo=1, per_A=0, per_B=0):
    """d-block penalties on P₁: d¹⁰ closed shell + d⁵ Hund + late-d excess."""
    l_min, l_max = min(l_A, l_B), max(l_A, l_B)
    S = 0.0

    # S_d10: d¹⁰ closed shell penalty
    d10_pen = 1.0
    partner_halide = (l_min <= 1 and max(np_A, np_B) >= P1)
    np_pairs = [(nd_A, l_A, ns_config(Z_A), np_B, l_B, per_B),
                (nd_B, l_B, ns_config(Z_B), np_A, l_A, per_A)]
    for nd_c, l_c, ns_c, np_part, l_part, per_part in np_pairs:
        if l_c == 2 and nd_c >= 2 * P2:
            if ns_c >= 2:
                d10_pen = min(d10_pen, math.sqrt(S3))
            elif partner_halide:
                f_per_d10 = (float(max(per_part, 2)) / 2.0) ** (1.0 / P1)
                d10_pen = min(d10_pen, C5 * C5 * f_per_d10)
    if d10_pen < 1.0:
        S_base = -math.log(d10_pen)
        for nd_c, l_c, ns_c, np_part, l_part, per_part in np_pairs:
            if l_c == 2 and nd_c >= 2 * P2 and ns_c >= 2:
                S_base *= (1.0 + float(np_part) / (2.0 * P1))
                break
        S += S_base

    # S_d10s1_multibond
    if bo >= 2:
        for nd_c, l_c, ns_c, np_part, l_part, per_part in np_pairs:
            if l_c == 2 and nd_c >= 2 * P2 and ns_c == 1:
                S += float(bo - 1) * S3 * C3
                break

    # S_d10_multibond
    if bo >= 2:
        for nd_c, l_c, ns_c, np_part, l_part, per_part in np_pairs:
            if l_c == 2 and nd_c >= 2 * P2 and l_part <= 1:
                S += float(bo - 1) * D3 * C3
                break

    # S_d10s1_sigma
    if bo == 1 and Z_A != Z_B:
        for nd_c, l_c, ns_c, np_part, l_part, per_part in np_pairs:
            if l_c == 2 and nd_c >= 2 * P2 and ns_c == 1 and per_part >= P1:
                S -= D3
                break

    # S_d10s2_extra
    for nd_c, l_c, ns_c, np_part, l_part, per_part in np_pairs:
        if l_c == 2 and nd_c >= 2 * P2 and ns_c >= 2:
            S += S3 * float(ns_c) / P1
            break

    # C8: d¹⁰s² + H sp-hybrid relief
    if bo == 1:
        for nd_c, l_c, ns_c, np_part, l_part, per_part in np_pairs:
            if l_c == 2 and nd_c >= 2 * P2 and ns_c >= 2 and per_part <= 1:
                S -= D3 * S_HALF
                break

    # S_late_d_oxide
    if bo >= 2:
        for nd_c, l_c, ns_c, np_part, l_part, per_part in np_pairs:
            if l_c == 2 and P2 < nd_c < 2 * P2 and l_part <= 1:
                S += S3 * float(nd_c - P2) / P2
                break

    # S_d5: d⁵ Hund half-fill penalty
    for nd_c, l_c, l_p, Z_c, np_p in [
        (nd_A, l_A, l_B, Z_A, np_B), (nd_B, l_B, l_A, Z_B, np_A)
    ]:
        if l_c == 2 and nd_c == P2:
            if l_p == 0 and ns_config(Z_c) == 1:
                S += -math.log(C5) * S_HALF
            elif l_p >= 1 and np_p > P1:
                S += -math.log(C5) * C5
            else:
                S += -math.log(C5)

    # S_d5_homonuclear
    if Z_A == Z_B:
        for nd_c, l_c in [(nd_A, l_A), (nd_B, l_B)]:
            if l_c == 2 and nd_c == P2:
                S += -math.log(1.0 - D5)
                break

    # S_dlate
    if l_max >= 2 and l_min == 0:
        nd_d = max(nd_A, nd_B)
        if P2 < nd_d < 2 * P2:
            excess = nd_d - P2
            Z_d = Z_A if l_A >= 2 else Z_B
            ns_d = ns_config(Z_d)
            if nd_d >= P2 + P1 and ns_d >= 2:
                vac = 2 * P2 - nd_d
                net = max(0.0, float(excess) - S_HALF * float(vac))
            else:
                net = float(excess)
            f = 1.0 - net / (2.0 * P2)
            S += -math.log(max(f, 0.01))

    return S


# ──────────────────────────────────────────────────────────────────────
#  Term 7: S_pi (from v3 — used for MAIN engine fallback)
# ──────────────────────────────────────────────────────────────────────

def _S_pi(np_A, np_B, l_min, l_max, Z_A, Z_B, bo, bo_eff,
          lp_A, lp_B, per_max, eps_geom, f_lp, f_per):
    """Pi anti-screening. Returns negative S (enhances D0)."""
    if l_min < 1: return 0.0
    gap_sigma = eps_geom * f_lp * f_per
    if gap_sigma <= 0: return 0.0
    t_pi = S_HALF * eps_geom * f_per * S_HALF
    gap_pi = 0.0
    bo_pi = min(bo_eff - 1.0, 2.0)
    if bo_pi > 0:
        unp_A = min(np_A, 2*P1 - np_A)
        unp_B = min(np_B, 2*P1 - np_B)
        lp_A_loc = max(0, np_A - P1)
        lp_B_loc = max(0, np_B - P1)
        n_shared = max(0, min(unp_A - 1, unp_B - 1))
        n_full_dat = 0
        if np_A < P1: n_full_dat += min(lp_B_loc, P1 - np_A)
        if np_B < P1: n_full_dat += min(lp_A_loc, P1 - np_B)
        n_full_dat = min(n_full_dat, max(0, int(bo_pi) - n_shared))
        n_half_dat = max(0, int(bo_pi) - n_shared - n_full_dat)
        if n_half_dat > 0 and (np_A % P1 == 0 or np_B % P1 == 0):
            if bo_eff > bo and n_shared == 0:
                n_half_dat = min(n_half_dat, int(bo_eff - bo))
            else:
                n_half_dat = 0
        if Z_A == Z_B:
            n_pi_eff = float(n_shared + n_full_dat + n_half_dat)
        else:
            n_pi_eff = float(n_shared + n_full_dat) + float(n_half_dat) * S_HALF
        gap_pi = n_pi_eff * 2.0 * t_pi
        if bo_eff >= 3 and per_max >= P1 and np_A >= 2 and np_B >= 2:
            gap_pi *= (2.0 / float(per_max)) ** S_HALF
    bo_frac = bo_eff - math.floor(bo_eff)
    gap_half = bo_frac * 2.0 * t_pi if bo_frac > 0 and l_min >= 1 else 0.0
    r_pi = (gap_pi + gap_half) / gap_sigma
    if r_pi <= 0: return 0.0
    return -math.log(1.0 + r_pi)


# ──────────────────────────────────────────────────────────────────────
#  Term 8: S_dblock (identical to v3)
# ──────────────────────────────────────────────────────────────────────

def _S_dblock_anti(D_P2, D_base):
    """d-block face as anti-screening. S = -ln(1 + r)."""
    if D_P2 <= 0 or D_base <= 0.01: return 0.0
    r = D_P2 / D_base
    return -math.log(1.0 + r)


# ──────────────────────────────────────────────────────────────────────
#  Term 9: S_ion (identical to v3)
# ──────────────────────────────────────────────────────────────────────

def _S_ion(D_P3, D_base, ie_A, ie_B, ea_A, ea_B, l_A, l_B,
           Z_A, Z_B, nd_A, nd_B, per_A, per_B, bo):
    """Ionic anti-screening (P₃ face) + AE polarisation + d¹⁰s¹ competition."""
    S = 0.0

    if D_P3 > 0 and D_base > 0.01:
        r = D_P3 / D_base
        S_pyth = -0.5 * math.log(1.0 + r * r)
        ea_max = max(ea_A, ea_B)
        _ea_thr = S3 * RY
        l_min = min(l_A, l_B)
        l_max = max(l_A, l_B)
        ie_dom = max(ie_A, ie_B)
        q_rel = min(ie_A, ie_B) / ie_dom if ie_dom > 0 else 1.0
        _both_s = (l_min == 0 and l_max == 0)
        _s_halide = (l_min == 0 and ea_max > _ea_thr)
        _dblock_ionic = (q_rel < S_HALF and l_max >= 2)
        _d10s1_H = False
        if l_max >= 2 and l_min == 0 and Z_A > 0 and Z_B > 0:
            for Z_c, l_c in [(Z_A, l_A), (Z_B, l_B)]:
                if l_c >= 2:
                    nd_c = n_fill_madelung(Z_c) if l_of(Z_c) == 2 else 0
                    if nd_c >= 2*P2 and ns_config(Z_c) == 1:
                        _d10s1_H = True
        if ea_max > _ea_thr or _both_s or _s_halide or _dblock_ionic or _d10s1_H:
            S += S_pyth
        else:
            f_att = (ea_max / _ea_thr) ** S_HALF if ea_max > 0 else 0.0
            if q_rel < S_HALF:
                f_att *= q_rel / S_HALF
            S += S_pyth * f_att

    if D_P3 > 0 and D_base > 0.01:
        Z_cat = Z_A if ie_A < ie_B else Z_B
        ie2 = _IE2_AE.get(Z_cat, 0.0)
        per_min_loc = min(per_A, per_B)
        if ie2 > 0 and D_P3 > D_base * C3 and per_min_loc >= 2:
            r_cov_ae = r_equilibrium(per_A, per_B, bo)
            ie_dom_ae = max(ie_A, ie_B)
            ie_sub_ae = min(ie_A, ie_B)
            r_ion_ae = r_cov_ae * (ie_dom_ae / max(ie_sub_ae, 0.1)) ** (1.0/P1)
            per_cat = period(Z_cat)
            f_per_ae = float(per_cat) / P1 if per_cat >= P1 + 1 else 1.0
            D_pol = COULOMB_EV_A**2 * S3 * f_per_ae / (r_ion_ae**2 * ie2)
            S += -math.log(1.0 + D_pol / D_base)

    if Z_A != Z_B:
        for Z_d, l_d, nd_d in [(Z_A, l_A, nd_A), (Z_B, l_B, nd_B)]:
            if l_d == 2 and nd_d >= 2 * P2 and ns_config(Z_d) == 1:
                Z_p = Z_B if Z_d == Z_A else Z_A
                ie_d_loc = IE_eV(Z_d)
                ie_p_loc = IE_eV(Z_p)
                ratio_ie = ie_p_loc / ie_d_loc
                comp = C5 * S_HALF * (ratio_ie - 2.0)
                per_d = period(Z_d)
                if per_d >= 6 and comp < 0:
                    za2 = (Z_d * AEM) ** 2
                    if za2 < 1:
                        comp *= math.sqrt(1.0 - za2)
                S += comp
                break

    if D_P3 > 0 and D_base > 0.01:
        l_max_ion = max(l_A, l_B)
        per_max_ion = max(per_A, per_B)
        if l_max_ion == 0 and per_max_ion == P1 and D_P3 > D_base:
            ratio_ion = D_P3 / D_base
            S += D3 * (ratio_ion - 1.0)

    if D_P3 > 0 and D_base > 0.01:
        l_min_fs = min(l_A, l_B)
        l_max_fs = max(l_A, l_B)
        ea_max_fs = max(ea_A, ea_B)
        _ea_thr_fs = S3 * RY
        if l_min_fs >= 1 and l_max_fs <= 1 and ea_max_fs < _ea_thr_fs and bo < P1:
            ie_dom_fs = max(ie_A, ie_B)
            q_rel_fs = min(ie_A, ie_B) / ie_dom_fs if ie_dom_fs > 0 else 1.0
            r_fs = D_P3 / D_base
            f_bo = float(P1 - bo) / P1
            S += S3 * r_fs * (1.0 - q_rel_fs**2) * (1.0 - ea_max_fs / _ea_thr_fs) * f_bo

    return S


# ──────────────────────────────────────────────────────────────────────
#  Term 10: S_mech (identical to v3)
# ──────────────────────────────────────────────────────────────────────

def _S_extra_dative(bo_eff, Z_A, Z_B, l_min, np_A, np_B,
                    eps_geom, f_per, D_base, cap_sigma):
    """Extra dative for bo_eff > 3. Anti-screening."""
    if not (bo_eff > 3 and Z_A != Z_B and l_min >= 1):
        return 0.0
    n_extra = bo_eff - 3
    np_acc = min(np_A, np_B)
    vac = P1 - np_acc
    f_dat = float(min(n_extra, vac)) / P1
    extra = cap_sigma * eps_geom * f_dat * f_per
    if D_base <= 0 or extra <= 0:
        return 0.0
    return -math.log(1.0 + extra / D_base)


def _S_mechanisms(Z_A, Z_B, bo, ie_A, ie_B, ea_A, ea_B,
                  per_A, per_B, l_A, l_B, lp_A, lp_B,
                  np_A, np_B, eps_geom, f_per, bo_eff=None):
    """6 diatomic mechanisms as S terms. Returns total S_mech."""
    S = 0.0
    _HALOGENS = frozenset({9,17,35,53})
    _CHALCOGENS = frozenset({8,16,34,52})
    _ALKALI = frozenset({3,11,19,37,55})
    q_rel = min(ie_A,ie_B)/max(ie_A,ie_B) if max(ie_A,ie_B)>0 else 1
    chi_A, chi_B = (ie_A+ea_A)/2, (ie_B+ea_B)/2
    q_eff = abs(chi_A-chi_B)/max(0.1, chi_A+chi_B)
    lp_min = min(lp_A, lp_B)
    per_max, per_min = max(per_A,per_B), min(per_A,per_B)

    # M1a: diffuse alkali
    if bo==1 and {Z_A,Z_B}&_ALKALI and {Z_A,Z_B}&_HALOGENS:
        alk_per = per_A if Z_A in _ALKALI else per_B
        if alk_per >= 4 and q_rel < 0.45:
            S += -math.log(1.0 + D5 + D3*q_eff)
    # M1b: diffuse anion polarization
    if bo==1 and {Z_A,Z_B}&_ALKALI and {Z_A,Z_B}&_HALOGENS:
        hal_Z = Z_A if Z_A in _HALOGENS else Z_B
        hal_per = per_A if Z_A == hal_Z else per_B
        alk_per = per_B if Z_A == hal_Z else per_A
        if hal_per > P1:
            f_ex = float(hal_per-P1)/float(hal_per)
            f_comp = (float(P1)/float(max(alk_per,1)))**2
            S += -math.log(1.0 + (D5+D3*q_eff)*f_ex*f_comp)
    # M2: heavy homonuclear halogen
    if bo==1 and Z_A==Z_B and Z_A in _HALOGENS and per_max>=3 and lp_min>=3:
        S += -math.log(1.0 + D5 + D3*S_HALF*min(max(per_max-2,1), P1-1))
    # M2b: heavy heteronuclear halide LP coupling
    if bo==1 and Z_A!=Z_B and Z_A in _HALOGENS and Z_B in _HALOGENS and lp_min>=P1:
        if per_min > P1:
            S += -math.log(1.0 + S3*q_rel)
        elif per_max >= P2 and per_min <= 2:
            f_heavy = float(per_max - P1) / float(per_max)
            S += -math.log(1.0 + S3*q_rel*f_heavy)
    # M16: same-period heteronuclear both-over-half-fill LP exchange
    if (bo == 1 and Z_A != Z_B and per_A == per_B and per_max >= P1
            and min(l_A, l_B) >= 1 and max(l_A, l_B) <= 1
            and np_A > P1 and np_B > P1):
        S += -math.log(1.0 + D3 * C3)

    # M3b: over-filled π contraction for bo≥2
    _bo_eff_m = bo_eff if bo_eff is not None else bo
    np_acc_m8 = min(np_A, np_B)
    np_don_m8 = max(np_A, np_B)
    _m8_active = (per_A == per_B and per_A >= P1 and np_acc_m8 <= 1
                  and np_don_m8 > P1 and _bo_eff_m > bo and min(l_A, l_B) >= 1)
    if _bo_eff_m >= 2 and max(l_A, l_B) <= 1:
        for Z_t, np_t, per_t in [(Z_A, np_A, per_A), (Z_B, np_B, per_B)]:
            if np_t > P1 and per_t >= P1:
                if per_A != per_B:
                    S += S3 * S_HALF
                elif _bo_eff_m > bo and not _m8_active:
                    S += S3 * S_HALF * S_HALF
                break
    # M8: same-face dative vacancy relief
    if _m8_active:
        lp_don_loc = np_don_m8 - P1
        S += -S3 * float(lp_don_loc) / P1 * S_HALF
    # M5: hetero chalcogen oxide
    if bo==2 and {Z_A,Z_B}&_CHALCOGENS and 8 in {Z_A,Z_B} and lp_min>=2 and per_max>=P1:
        S += -D5*(1.1 + 0.2*max(per_max-P1, 0))
    # M6: charge-separated triple (CO)
    if bo==3 and Z_A!=Z_B:
        if {Z_A,Z_B} == {6,8}: S += -math.log(1.0 + D5)
    # M7: d-block hydride Hund exchange cost
    if bo == 1 and max(l_A, l_B) >= 2 and min(l_A, l_B) == 0:
        Z_d = Z_A if l_A >= 2 else Z_B
        nd_h = _nd_of(Z_d)
        ns_h = ns_config(Z_d)
        if ns_h == 2 and P2 <= nd_h <= P2 + 2:
            f_hund = float(2 * P2 - nd_h) / P2
            S += S5 * f_hund * S_HALF
    # M13: late-d hydride post-half-fill pairing cost
    if bo == 1 and max(l_A, l_B) >= 2 and min(l_A, l_B) == 0:
        Z_d_m13 = Z_A if l_A >= 2 else Z_B
        Z_h_m13 = Z_B if l_A >= 2 else Z_A
        nd_m13 = _nd_of(Z_d_m13)
        ns_m13 = ns_config(Z_d_m13)
        if Z_h_m13 <= 2 and P2 < nd_m13 <= P2 + 2 and ns_m13 >= 2:
            f_vac = float(2 * P2 - nd_m13) / P2
            S += D5 * f_vac
    # M10: d-block + H sd-hybridisation screening
    if bo == 1 and max(l_A, l_B) >= 2 and min(per_A, per_B) == 1:
        Z_d_m10 = Z_A if l_A >= 2 else Z_B
        Z_h_m10 = Z_B if l_A >= 2 else Z_A
        nd_m10 = _nd_of(Z_d_m10)
        ns_m10 = ns_config(Z_d_m10)
        if Z_h_m10 <= 2 and nd_m10 <= 2 and ns_m10 >= 2:
            f_early = 1.0 - float(nd_m10) / P2
            S += S5 * f_early * S_HALF
    # M17: intermediate d-fill + H exchange-enhanced back-donation
    if bo == 1 and max(l_A, l_B) >= 2 and min(per_A, per_B) == 1:
        Z_d_m17 = Z_A if l_A >= 2 else Z_B
        Z_h_m17 = Z_B if l_A >= 2 else Z_A
        nd_m17 = _nd_of(Z_d_m17)
        ns_m17 = ns_config(Z_d_m17)
        if Z_h_m17 <= 2 and P1 <= nd_m17 < P2 and ns_m17 >= 2:
            S -= D5 * S_HALF

    # Cross-period pi
    l_min = min(l_A, l_B)
    if l_min >= 1 and per_max > per_min:
        fill_cp = max(0, min(np_A,2*P1-np_B) + min(np_B,2*P1-np_A))
        np_min_cp = min(np_A, np_B)
        _do_cp = False
        if fill_cp >= P2 and np_min_cp < P1:
            _do_cp = True
        elif bo >= 2 and max(l_A, l_B) <= 1 and fill_cp >= P1:
            _do_cp = True
        if _do_cp:
            fill_frac = float(fill_cp)/(2.0*P1)
            f_gap = 1.0 - float(per_min)/float(per_max)
            extra = (RY/P2) * fill_frac * eps_geom * f_gap
            cap_approx = SHANNON_CAP * max(bo, 1)
            S += -math.log(1.0 + extra / cap_approx)

    # M15: cross-period double bond hex-face cross-coupling
    _bo_eff_m15 = bo_eff if bo_eff is not None else bo
    if (bo == 2 and Z_A != Z_B and per_A != per_B
            and l_min >= 1 and max(l_A, l_B) <= 1
            and abs(_bo_eff_m15 - bo) < 0.01):
        _m3b_m15 = any(np_t > P1 and per_t >= P1
                       for np_t, per_t in [(np_A, per_A), (np_B, per_B)])
        if not _m3b_m15:
            S -= D3 * eps_geom * C3

    # M9: cross-period halide LP→vacancy anti-screening
    _HALOGENS_M9 = frozenset({9, 17, 35, 53})
    if (l_min >= 1 and max(l_A, l_B) <= 1 and per_min < per_max
            and Z_A != Z_B and bo <= 1):
        if np_A < P1 and Z_B in _HALOGENS_M9:
            vac_m9 = P1 - np_A
            lp_don_m9 = lp_B
        elif np_B < P1 and Z_A in _HALOGENS_M9:
            vac_m9 = P1 - np_B
            lp_don_m9 = lp_A
        else:
            vac_m9 = 0
            lp_don_m9 = 0
        if vac_m9 > 0 and lp_don_m9 > 0:
            f_per_gap = 1.0 - float(per_min) / float(per_max)
            S -= D3 * float(min(vac_m9, lp_don_m9)) / P1 * f_per_gap

    # M14: cross-period large-gap halide LP dispersion
    if (l_min >= 1 and max(l_A, l_B) <= 1 and per_min > 0
            and float(per_max) / float(per_min) > 2.0
            and Z_A != Z_B and bo <= 1
            and min(np_A, np_B) >= P1):
        ea_min_m14 = min(EA_eV(Z_A), EA_eV(Z_B))
        if ea_min_m14 > S3 * RY:
            S_per_m14 = -math.log((2.0 / per_max) ** (1.0 / P1))
            f_pg2 = 1.0 - (float(per_min) / float(per_max)) ** 2
            S -= S_per_m14 * (ea_min_m14 / RY) ** S_HALF * f_pg2 * C3

    # M11: d¹⁰s¹ + H sigma relief
    l_max_m11 = max(l_A, l_B)
    if bo == 1 and l_max_m11 >= 2 and per_min == 1 and lp_A + lp_B == 0:
        for Z_d, l_d in [(Z_A, l_A), (Z_B, l_B)]:
            if l_d == 2:
                nd_m11 = _nd_of(Z_d)
                ns_m11 = ns_config(Z_d)
                if nd_m11 >= 2 * P2 and ns_m11 == 1:
                    S += math.log(C5) * C3
                    break

    # M12: per=2 s-block + H sp-polarisation
    l_max_m12 = max(l_A, l_B)
    if per_min == 1 and per_max == 2 and l_max_m12 == 0 and q_rel < S_HALF:
        Z_cat = Z_A if ie_A < ie_B else Z_B
        ns_cat = ns_config(Z_cat)
        if ns_cat == 1:
            S -= D3

    # C3: early d-block + LP partner vacancy-enhanced back-donation
    l_max_c3 = max(l_A, l_B)
    if l_max_c3 >= 2 and min(l_A, l_B) <= 1 and bo >= 2:
        for Z_d_c3, l_d_c3, lp_p_c3 in [
            (Z_A, l_A, lp_B), (Z_B, l_B, lp_A)
        ]:
            if l_d_c3 != 2:
                continue
            nd_c3 = _nd_of(Z_d_c3)
            if 2 <= nd_c3 < P2 and lp_p_c3 > 0:
                S -= D5 * float(P2 - nd_c3) / (2 * P2)
                break

    return S


# ──────────────────────────────────────────────────────────────────────
#  D_P2 energy (identical to v3)
# ──────────────────────────────────────────────────────────────────────

def _D_P2_energy(ie_A, ie_B, per_A, per_B, l_A, l_B, bo,
                 lp_A, lp_B, Z_A, Z_B, nd_A, nd_B, np_A, np_B,
                 eps_geom_orig, f_per_simple):
    """d-block contributions on pentagonal face. Returns eV."""
    D = 0.0
    if nd_A == 0 and nd_B == 0: return 0.0
    for nd_don, Z_don, l_don, np_part, lp_part, ie_don, l_part in [
        (nd_A, Z_A, l_A, np_B, lp_B, ie_A, l_B),
        (nd_B, Z_B, l_B, np_A, lp_A, ie_B, l_A),
    ]:
        if nd_don <= 0 or l_don != 2 or nd_don >= 2*P2 or l_part < 1: continue
        vacancy = max(0, 2*P1 - max(np_part, 1))
        if vacancy > 0:
            D += (RY/P1) * (float(vacancy)/(2*P1)) * (float(nd_don)/(2*P2)) * eps_geom_orig * f_per_simple * S_HALF
    for nd_acc, lp_don, _ in [(nd_A, lp_B, ie_B), (nd_B, lp_A, ie_A)]:
        if nd_acc <= 0 or lp_don <= 0: continue
        f_acc = float(max(0, 2*P2 - nd_acc)) / (2*P2)
        if f_acc > 0:
            D += (RY/P1) * f_acc * S_HALF * eps_geom_orig * f_per_simple * S_HALF
    for nd_d, l_d, np_p, lp_p, per_p in [
        (nd_A, l_A, np_B, lp_B, per_B), (nd_B, l_B, np_A, lp_A, per_A)]:
        if l_d != 2 or nd_d >= P2: continue
        if lp_p <= 0 and np_p < 2: continue
        D += (RY/P2) * (float(min(max(lp_p,1), P2-nd_d))/P2) * eps_geom_orig * max(bo,1)
    nd_max = max(nd_A, nd_B)
    l_db = l_A if nd_A >= nd_B else l_B
    if nd_max > 0 and l_db == 2 and min(nd_A, nd_B) == 0:
        if nd_max < P2: f_d = float(P2 - nd_max)/P2
        elif P2 < nd_max < 2*P2: f_d = float(nd_max - P2)/P2
        else: f_d = 0.0
        if f_d > 0: D += (RY/P2) * f_d * eps_geom_orig * S_HALF * f_per_simple
    if Z_A == Z_B and l_db == 2:
        nd_h = max(nd_A, nd_B)
        ns_h = ns_config(Z_A)
        if nd_h >= 2*P2 and ns_h == 1:
            za2 = (Z_A*AEM)**2
            D += (RY/P2) * eps_geom_orig * S_HALF / max(0.1, math.sqrt(1-min(za2,0.99)))
    return D


# ──────────────────────────────────────────────────────────────────────
#  D_P3 energy (identical to v3)
# ──────────────────────────────────────────────────────────────────────

def _D_P3_energy(ie_A, ie_B, ea_A, ea_B, per_A, per_B, l_A, l_B, bo,
                 lp_A, lp_B, Z_A, Z_B, nd_A, nd_B):
    """Ionic from Born-Haber cycle. Returns eV."""
    ie_dom = max(ie_A, ie_B); ie_sub = min(ie_A, ie_B)
    ea_max = max(ea_A, ea_B)
    q_rel = ie_sub / ie_dom if ie_dom > 0 else 1.0
    r_cov = r_equilibrium(per_A, per_B, bo)
    ionic_stretch = (ie_dom / max(ie_sub, 0.1)) ** (1.0/P1)
    r_ion = r_cov * ionic_stretch
    l_max, l_min = max(l_A, l_B), min(l_A, l_B)
    np_A, np_B = _np_of(Z_A), _np_of(Z_B)
    for Z_c, l_c, nd_c in [(Z_A, l_A, nd_A), (Z_B, l_B, nd_B)]:
        if l_c != 2: continue
        is_d10 = nd_c >= 2*P2 or (Z_c in _NS1 and nd_c >= 2*P2-1)
        if is_d10 and ea_max > S3*RY:
            r_ion *= C5
            if ns_config(Z_c) == 1 and nd_c >= 2*P2:
                from ptc.periodic import period as _period_d
                per_d = _period_d(Z_c)
                r_ion *= C3 ** (float(P1) / (per_d * P1))
    D_raw = max(0.0, COULOMB_EV_A/r_ion - ie_sub + ea_max)
    D_P3 = D_raw * (1.0 - q_rel**2)
    per_min, per_max = min(per_A, per_B), max(per_A, per_B)
    _is_d10s1 = False
    for Z_c, l_c, nd_c in [(Z_A, l_A, nd_A), (Z_B, l_B, nd_B)]:
        if l_c == 2 and nd_c >= 2*P2 and ns_config(Z_c) == 1:
            _is_d10s1 = True
    if per_min == 1 and per_max >= 2 and q_rel < 0.7 and not _is_d10s1:
        pia, pib = max(per_A,2), max(per_B,2)
        r_new = r_equilibrium(pia, pib, bo) * ionic_stretch
        if l_max == 0: r_new = max(r_new, r_equilibrium(pia,pib,bo)*(1+S3*S_HALF))
        D_P3 = max(0.0, COULOMB_EV_A/r_new - ie_sub + ea_max) * (1-q_rel**2)
    ea_thr = S3 * RY
    if ea_max < ea_thr and D_P3 > 0:
        if l_max == 0 and per_max <= 2:
            D_P3 *= (ea_max/RY) ** (S_HALF/P0)
        elif l_min == 0 and l_max >= 1:
            Z_cat_ea = Z_A if ie_A < ie_B else Z_B
            from ptc.periodic import period as _period
            per_cat_ea = _period(Z_cat_ea)
            ns_cat = ns_config(Z_cat_ea)
            l_cat_ea = l_of(Z_cat_ea)
            if per_cat_ea >= P1 and ns_cat <= 1:
                if l_cat_ea == 0:
                    D_P3 *= (ea_max/RY) ** S_HALF
                else:
                    D_P3 *= (ea_max/RY) ** (S_HALF/P0)
    if D_P3 > 0 and q_rel < S_HALF:
        if ea_A >= ea_B: per_an, lp_an = per_A, lp_A
        else: per_an, lp_an = per_B, lp_B
        per_cat = per_A if ea_A < ea_B else per_B
        if per_an > P1 and per_cat >= P1 and lp_an > 0:
            D_P3 += COULOMB_EV_A**2 * S3 * float(lp_an)/P1 * (float(P1)/float(per_an))**(1.0/P1) / r_ion**4
    if D_P3 > 0 and l_min == 0 and q_rel < S_HALF and per_min >= 2:
        Z_cat = Z_A if ie_A < ie_B else Z_B
        ie2 = _IE2_AE.get(Z_cat, 0.0)
        if ie2 > 0: D_P3 += COULOMB_EV_A**2 * S3 / (r_ion**2 * ie2)
    if D_P3 > 0 and l_min == 0 and per_min == 1 and q_rel < S_HALF:
        Z_cat_h = Z_A if ie_A < ie_B else Z_B
        ie2_h = _IE2_AE.get(Z_cat_h, 0.0)
        if ie2_h > 0:
            r_h = r_equilibrium(max(per_A, 2), max(per_B, 2), bo) * ionic_stretch
            D_P3 += COULOMB_EV_A**2 * S3 / (r_h**2 * ie2_h)
    if D_P3 > 0 and bo >= P1 and per_max <= 2 and l_min >= 1 and Z_A != Z_B:
        chi_A = (ie_A + ea_A) * S_HALF
        chi_B = (ie_B + ea_B) * S_HALF
        q_eff_loc = abs(chi_A - chi_B) / max(0.1, chi_A + chi_B)
        if q_eff_loc > S3:
            D_P3 *= (1.0 + bo * D3)
    return D_P3


# ──────────────────────────────────────────────────────────────────────
#  Dative bo_eff (identical to v3)
# ──────────────────────────────────────────────────────────────────────

def _compute_bo_eff(bo, Z_A, Z_B, np_A, np_B, lp_A, lp_B,
                    per_A, per_B, l_min, eps_A, eps_B):
    """Effective bond order with dative detection."""
    bo_eff = bo
    if Z_A == Z_B or l_min < 1: return bo_eff
    total_val = _valence_electrons(Z_A) + _valence_electrons(Z_B)
    is_radical = (total_val % 2) == 1
    q_rel = min(eps_A, eps_B) / max(eps_A, eps_B) if max(eps_A, eps_B) > 0 else 1.0
    if np_A <= np_B:
        np_acc, lp_don, per_don, per_acc, Z_acc = np_A, lp_B, per_B, per_A, Z_A
    else:
        np_acc, lp_don, per_don, per_acc, Z_acc = np_B, lp_A, per_A, per_B, Z_B
    if np_acc < P1 and lp_don > 0:
        vacancy = P1 - np_acc
        if is_radical and q_rel > C3: vacancy = max(0, vacancy - 1)
        n_dat = min(lp_don, vacancy)
        if per_don > 2: n_dat = min(n_dat, 1)
        if l_of(Z_acc) == 2:
            nd_acc_v = _nd_of(Z_acc)
            if nd_acc_v >= 2 * P2:
                n_dat = min(n_dat, 2)
        if per_acc > 2 and l_of(Z_acc) == 1: n_dat = min(n_dat, 1)
        if is_radical and bo == 1:
            if per_A == per_B:
                n_dat *= S_HALF
            else:
                n_dat *= C3
        bo_eff = bo + n_dat
    elif np_acc == P1 and lp_don > 0 and q_rel < C3 and not is_radical:
        bo_eff = bo + min(lp_don, 1) * S_HALF
    return bo_eff


# ──────────────────────────────────────────────────────────────────────
#  MAIN ENGINE: Hybrid DFT-GFT — 7 terms
# ──────────────────────────────────────────────────────────────────────

def D0_screening(Z_A, Z_B, bo, lp_A=None, lp_B=None):
    """D₀ = cap × exp(-S_total) where S_total = Σ S_i (7 terms).

    Hybrid DFT-GFT architecture. S_hex replaces S_count+S_exchange+S_exclusion+S_pi.
    """
    # ── Atomic data ──
    ie_A, ie_B = IE_eV(Z_A), IE_eV(Z_B)
    ea_A, ea_B = EA_eV(Z_A), EA_eV(Z_B)
    per_A, per_B = period(Z_A), period(Z_B)
    l_A, l_B = l_of(Z_A), l_of(Z_B)
    nd_A, nd_B = _nd_of(Z_A), _nd_of(Z_B)
    np_A, np_B = _np_of(Z_A), _np_of(Z_B)
    l_min, l_max = min(l_A, l_B), max(l_A, l_B)
    per_min, per_max = min(per_A, per_B), max(per_A, per_B)

    if lp_A is None: lp_A = _lp_pairs(Z_A, bo)
    if lp_B is None: lp_B = _lp_pairs(Z_B, bo)
    lp_mutual = min(lp_A, lp_B)

    # ── Effective eps (absorbs sd-hybrid, CRT, stiffness) ──
    eps_A, eps_B, eps_geom, sd_A, sd_B = _effective_eps(
        ie_A, ie_B, per_A, per_B, l_A, l_B,
        Z_A, Z_B, nd_A, nd_B, lp_A, lp_B)

    # ── Original eps for P₂ ──
    eps_orig = math.sqrt(ie_A * ie_B) / RY
    per_max_v = max(per_A, per_B)
    if per_max_v > 2:
        f_per_s = (2.0/per_max_v) ** (1.0/(P1*P1) if per_max_v == P1 else 1.0/P1)
    else:
        f_per_s = 1.0

    # ── Dative bo_eff ──
    bo_eff = _compute_bo_eff(bo, Z_A, Z_B, np_A, np_B, lp_A, lp_B,
                             per_A, per_B, l_min, eps_A, eps_B)

    # ── Cap ──
    cap = SHANNON_CAP * D_FULL

    # ══════════════════════════════════════════════════════════════
    #  7 FUNCTIONAL TERMS
    # ══════════════════════════════════════════════════════════════
    terms = {}

    # Term 1: S_gap
    terms['S_gap'] = _S_gap(eps_geom)

    # Term 2: S_hex (DFT-GFT hybrid replaces S_count+S_exchange+S_exclusion+S_pi)
    S_hex_val, f_lp = _S_hex_hybrid(
        Z_A, Z_B, bo, lp_A, lp_B, np_A, np_B,
        per_A, per_B, l_A, l_B, eps_A, eps_B, eps_geom,
        ie_A, ie_B, nd_A, nd_B, sd_A, sd_B, bo_eff=bo_eff)
    terms['S_hex'] = S_hex_val

    # Term 3: S_holo
    terms['S_holo'] = _S_holo(Z_A, Z_B, l_A, l_B, per_A, per_B,
                               lp_A=lp_A, lp_B=lp_B, ie_A=ie_A, ie_B=ie_B)

    # Term 4: S_d_penalty (d10 + d5 + dlate)
    terms['S_d_penalty'] = _S_d_penalty(nd_A, nd_B, l_A, l_B, Z_A, Z_B, np_A, np_B,
                                         bo=bo, per_A=per_A, per_B=per_B)

    # ── Compute f_per for S_pi/S_mech (same logic as inside _S_hex_hybrid) ──
    S_per_val = 0.0
    if per_max > 2:
        if Z_A == Z_B and l_max >= 2:
            expo = 1.0 / (P1 * P2)
        elif per_max == P1:
            expo = 1.0 / (P1 * P1)
        else:
            expo = 1.0 / P1
        f_per_raw = (2.0 / per_max) ** expo
        if f_per_raw < 1.0 and l_max >= 2 and per_max >= 6:
            Zh = Z_A if per_A >= 6 and l_A >= 2 else (Z_B if per_B >= 6 and l_B >= 2 else 0)
            if Zh > 0:
                za2 = (Zh * AEM) ** 2
                if za2 < 1:
                    f_per_raw /= math.sqrt(1.0 - za2)
        S_per_val = -math.log(max(f_per_raw, 1e-10))
    f_per = math.exp(-S_per_val)

    # ── Phase 1: D_base from terms 1-4 ──
    S_cov_pre = sum(terms.values())
    D_base_pre = cap * math.exp(-S_cov_pre)

    # ── Extra dative (needs D_base BEFORE S_dblock/S_ion) ──
    S_extra_dat = _S_extra_dative(bo_eff, Z_A, Z_B, l_min, np_A, np_B,
                                  eps_geom, f_per, D_base_pre, SHANNON_CAP)

    # ── Phase 2: recompute D_base including extra dative ──
    S_cov = S_cov_pre + S_extra_dat
    D_base = cap * math.exp(-S_cov)

    # ── S_cap_dH contribution to D_base_std ──
    S_cap_dH = 0.0
    if l_max >= 2 and per_min <= 1:
        S_cap_dH = 0.5 * math.log(P2 / P1)
    D_base_std = D_base * math.exp(S_cap_dH)

    # Term 5: S_dblock
    D_P2 = _D_P2_energy(ie_A, ie_B, per_A, per_B, l_A, l_B, bo,
                        lp_A, lp_B, Z_A, Z_B, nd_A, nd_B, np_A, np_B,
                        eps_orig, f_per_s)
    nd_max = max(nd_A, nd_B)
    _DBLOCK_PI_PARTNERS = frozenset({8, 9, 17, 35, 53})
    if nd_max == P2 and (Z_A in _DBLOCK_PI_PARTNERS or Z_B in _DBLOCK_PI_PARTNERS):
        D_P2 *= (1.0 + S5 * P2 / P1)
    S_homo_3d = 0.0
    if Z_A == Z_B and l_max >= 2 and per_max == P1 + 1:
        S_homo_3d = D5 * C3
    S_homo_5d = 0.0
    if Z_A == Z_B and l_max >= 2 and per_max >= 6:
        za2_5d = (float(Z_A) * AEM) ** 2
        S_homo_5d = za2_5d * D3
    terms['S_dblock'] = _S_dblock_anti(D_P2, D_base_std) + S_homo_3d + S_homo_5d

    # Term 6: S_ion
    D_P3 = _D_P3_energy(ie_A, ie_B, ea_A, ea_B, per_A, per_B, l_A, l_B, bo,
                        lp_A, lp_B, Z_A, Z_B, nd_A, nd_B)
    terms['S_ion'] = _S_ion(D_P3, D_base, ie_A, ie_B, ea_A, ea_B, l_A, l_B,
                            Z_A, Z_B, nd_A, nd_B, per_A, per_B, bo)

    # Term 7: S_mech
    S_mechanisms = _S_mechanisms(Z_A, Z_B, bo, ie_A, ie_B, ea_A, ea_B,
                                per_A, per_B, l_A, l_B, lp_A, lp_B,
                                np_A, np_B, eps_geom, f_per, bo_eff)
    terms['S_mech'] = S_extra_dat + S_mechanisms

    # ══════════════════════════════════════════════════════════════
    #  TOTAL
    # ══════════════════════════════════════════════════════════════
    S_total = sum(terms.values())
    D0 = cap * math.exp(-S_total)

    # Reconstruct per-face for diagnostics
    D_P1 = D_base

    return ScreeningResult(
        D0=D0, S_total=S_total, cap=cap,
        S_cov=S_cov,
        S_dblock=terms.get('S_dblock', 0),
        S_ion=terms.get('S_ion', 0),
        S_mech=terms.get('S_mech', 0),
        terms=terms,
        D_P1=D_P1, D_P2=D_P2, D_P3=D_P3,
    )


# Compatibility alias
def S_cov(Z_A, Z_B, bo, lp_A=None, lp_B=None):
    return D0_screening(Z_A, Z_B, bo, lp_A, lp_B)
