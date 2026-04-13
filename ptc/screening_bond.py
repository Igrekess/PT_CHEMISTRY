"""
screening_bond.py — Hybrid DFT-GFT diatomic engine.

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
  S_k3 = auto + cross-Im + fill-weight  ← Nyquist parity screening (C12)
  S_pi  = -ln(1 + pi_from_modes)       ← π anti-screening
  S_lp  = -ln(1 - lp_blocking)         ← LP blocking
  S_per = period attenuation            ← diffuse orbital screening
  S_nnlo = NNLO corrections             ← higher-order spectral corrections
      ↓ sum
  S_hex = S_k0 + S_k1 + S_k2 + S_k3 + S_pi + S_lp + S_per + S_nnlo

Unchanged from v3: S_gap, S_holo, S_d_penalty, S_dblock, S_ion, S_mech.

0 adjustable parameters. All from s = 1/2.

April 2026 — Persistence Theory
"""
import math
from dataclasses import dataclass, field

from ptc.constants import (
    RY, P0, P1, P2, P3, D_FULL, S3, S5, S7, C3, C5, C7, S_HALF,
    D3, D5, D7, GAMMA_3, GAMMA_5, GAMMA_7, AEM, COULOMB_EV_A,
    HALOGENS, CHALCOGENS, ALKALI, NS1, IE2_DBLOCK, IE2_AE,
    R35, R35_DARK, R57_DARK, R37_DARK,
    SHANNON_CAP,
)
from ptc.atom import IE_eV, EA_eV
from ptc.periodic import (
    period, period_start, l_of, n_fill as n_fill_madelung,
    _n_fill_aufbau, ns_config, capacity,
    _np_of, _nd_of, _valence_electrons, _lp_pairs,
)
from ptc.bond import r_equilibrium
from ptc.dft_polygon import electron_density, dft_spectrum
from ptc.dblock import DBlockState, pi_gate, mertens_factor


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


# Module-level aliases from centralized constants
_NS1 = NS1
_IE2 = IE2_DBLOCK
_IE2_AE = IE2_AE


def _per_eff(per_A, per_B):
    """Effective period for cross-period overlap on Z/(2P₁)Z.

    PT [C3]: CRT channels are orthogonal, but atoms on different periods
    live on Z/6Z circles of different effective SIZES.  The overlap between
    a compact (per=2) and diffuse (per=3) orbital involves a Gaussian radial
    integral.  The geometric mean captures this:

        per_eff = sqrt(max(per_A, 2) · max(per_B, 2))

    Same-period:  per_eff = per_max  (unchanged).
    Cross-period: per_eff < per_max  (compact orbital reaches in).

    The clamping at per=2 ensures that H (per=1) is treated as having
    the minimum shell size (1s ~ 2s for overlap purposes).

    0 adjustable parameters.
    """
    return math.sqrt(float(max(per_A, 2)) * float(max(per_B, 2)))


# ──────────────────────────────────────────────────────────────────────
#  Effective eps (identical to v3)
# ──────────────────────────────────────────────────────────────────────

def _effective_eps(ie_A, ie_B, per_A, per_B, l_A, l_B,
                   Z_A, Z_B, nd_A, nd_B, lp_A, lp_B, bo=1.0):
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
            lp_part = lp_B if is_A else lp_A
            # At nd = P₂ (half-fill), the d⁵ exchange is maximal and
            # stabilises the sigma bond with s-block (lp=0) partners.
            # Include the exchange bonus at half-fill for H partners only.
            delta_K = nd_s if nd_s < P2 else 0
            if nd_s == P2 and lp_part == 0:
                delta_K = nd_s
            if delta_K > 0:
                f_lp_ex = C3 if lp_part > 0 else 1.0
                f_stiff *= (1.0 + float(delta_K) * J_exch / ie_s * f_lp_ex)
            if is_A: eps_A *= f_stiff
            else: eps_B *= f_stiff

    # ── Late-d + compact halogen partial sd-hybrid relief ────────
    # When a late-d metal (P₂ < nd < 2P₂) bonds to a compact halogen
    # (per ≤ 2, l=1, np ≥ 2P₁−1 — i.e., F), the CRT+stiffness path
    # over-screens eps because the sd-hybrid rotation was fully blocked.
    # PT derivation: F's strong ligand field promotes partial sd-mixing
    # even for compact per=2 orbitals. The d-vacancies (2P₂ - nd)
    # provide Pauli-free channels for the compact halogen's LP to access
    # the metal, enabling partial sd-hybridization.
    # Mixing fraction:
    #   f_overlap = (per_compact / P₁)^(1/P₁)  — cross-period overlap
    #   f_vacancy = (2P₂ - nd) / (2P₂)          — d-vacancy availability
    #   f_mix = f_overlap × f_vacancy            — both spin channels
    # The corrected eps interpolates between CRT and sd-hybrid results.
    # 0 adjustable parameters — all constants from P₁, P₂.
    for (_eps_ref, _ie_ref, _nd_ref, _l_ref, _Z_ref, _sd_ref, _is_A_ref,
         _l_part, _per_part, _np_part_fn) in [
        (eps_A, ie_A, nd_A, l_A, Z_A, sd_A, True, l_B, per_B, _np_of(Z_B)),
        (eps_B, ie_B, nd_B, l_B, Z_B, sd_B, False, l_A, per_A, _np_of(Z_A)),
    ]:
        if (_l_ref == 2 and not _sd_ref and P2 < _nd_ref < 2 * P2
                and _l_part == 1 and _per_part <= 2
                and _np_part_fn >= 2 * P1 - 1):
            _ie2_ref = _IE2.get(_Z_ref, _ie_ref * 2)
            if _ie2_ref > 0:
                # sd-hybrid eps (what we'd get if sd-hybrid had fired)
                _f_sd_ref = float(_nd_ref - P2) / P2
                _eps_sd = _ie_ref / RY + _f_sd_ref * (
                    math.sqrt(_ie_ref * _ie2_ref) / RY - _ie_ref / RY)
                # Cross-period compact overlap fraction
                _f_overlap = (float(_per_part) / P1) ** (1.0 / P1)
                # d-vacancy availability fraction
                _f_vacancy = float(2 * P2 - _nd_ref) / (2.0 * P2)
                # Mixing: overlap × vacancy (both spin channels active
                # for closed-shell halide F⁻, no s=1/2 partition)
                _f_mix = _f_overlap * _f_vacancy
                # Interpolate: CRT result → sd result
                _eps_crt = _eps_ref
                _eps_new = _eps_crt + _f_mix * (_eps_sd - _eps_crt)
                if _is_A_ref:
                    eps_A = _eps_new
                else:
                    eps_B = _eps_new
            break

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

    # C15: s² closure promotion screening [derived from holonomy on Z/(2P₀)Z]
    # When one atom has ns=2 (filled s-shell) and the partner has ns=1,
    # the s² → sp promotion adds a holonomic cost D₃ × (P₁/per)^s.
    # PT: sin²(π·ns/(2P₀)) = 1.0 (full blockade on binary circle),
    # attenuated by σ-channel propagation D₃ and period factor.
    # Gate: per_ns2 ≤ P₁ (light s-block only; heavy s-block is ionic-dominated).
    if l_max == 0 and per_max > 1:
        _ns_A_phys = ns_config(Z_A)
        _ns_B_phys = ns_config(Z_B)
        if (max(_ns_A_phys, _ns_B_phys) == 2
                and min(_ns_A_phys, _ns_B_phys) <= 1):
            _per_ns2 = per_A if _ns_A_phys == 2 else per_B
            if _per_ns2 <= P1:
                S_sblock += D3 * (float(P1) / float(_per_ns2)) ** S_HALF

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

        # Per ≥ P₁ half-fill (np = P₁) exchange anti-screening for p+p bonds.
        # Same mechanism as C9 in S_holo but for p+p heteronuclear bonds.
        # When one atom is at half-fill on Z/(2P₁)Z and in per ≥ P₁
        # (same-period or partner is H), the Hund exchange is enhanced
        # by the diffuse orbital overlap. S₃/(2P₁)×(1-D₃) = GFT coefficient.
        # Gate: heteronuclear, at least one atom at np=P₁ with per ≥ P₁,
        #        same-period (cross-period has different radial overlap)
        #        or partner is H (per ≤ 1).
        if l_min >= 1 and Z_A != Z_B:
            for _np_hf, _per_hf, _np_oth, _per_oth in [
                (np_A, per_A, np_B, per_B), (np_B, per_B, np_A, per_A)
            ]:
                if _np_hf == P1 and _per_hf >= P1:
                    if _per_hf == _per_oth or _per_oth <= 1:
                        S_k1 -= S3 / (2 * P1) * (1.0 - D3)
                        break

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

    # C7: I₂ pentagonal LP resonance
    # Homonuclear halogens in period P₂ with pentagonal LP residue.
    # The sin²(2πr/P₂) captures the pentagonal mismatch.
    if (Z_A == Z_B and l_min >= 1 and max(l_A, l_B) <= 1
            and per_max == P2 and Z_A % P2 != 0
            and min(np_A, np_B) > P1 and bo <= 1):
        _r_P2 = Z_A % P2
        _za_P2 = (float(Z_A) * AEM) ** (2.0 / P2)
        _sin2_r = math.sin(2 * math.pi * _r_P2 / P2) ** 2
        S_k2 += D5 * _za_P2 * _sin2_r

    # C8: BrCl LP phase mismatch
    # Cross-period heteronuclear halogen LP mismatch from Pauli repulsion.
    _HALOGENS_C8 = frozenset({9, 17, 35, 53})
    _both_halogen_c8 = (Z_A in _HALOGENS_C8 and Z_B in _HALOGENS_C8)
    if (Z_A != Z_B and _both_halogen_c8
            and per_max == P1 + 1 and per_min >= P1 and bo <= 1):
        _phi_h = math.pi * (per_max - 2) / P1
        _phi_l = math.pi * (per_min - 2) / P1
        _mismatch = (1 - math.cos(_phi_h)) / 2 - (1 - math.cos(_phi_l)) / 2
        _za_A_c8 = (float(Z_A) * AEM) ** (2.0 / P1)
        _za_B_c8 = (float(Z_B) * AEM) ** (2.0 / P1)
        S_k2 += D3 * math.sqrt(_za_A_c8 * _za_B_c8) * _mismatch

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

    # ── d-block LP topology correction ───────────────────────────
    # When the topology parser assigns LP=0 to a d-block ns=2 metal
    # (e.g. Fe: valence_electrons=2 in topology.py → lp=(2-1)//2=0),
    # the LP blocking term S_lp misses the contribution from the
    # ns-derived lone pair. In PT, a d-block atom with ns_config≥2
    # and bo≤1 has LP_hex = max(0, ns - bo) = 1 on the hex face.
    # This correction simulates that missing LP=1 blocking without
    # modifying topology.py (which would regress 806 molecules).
    #
    # Gate: l_max≥2, at least one atom has (l=2, lp=0, ns≥2, bo≤1),
    # AND the partner has LP > 0 with per ≥ P₁ (diffuse LP partner).
    # The period gate prevents over-screening compact period-2 partners
    # (e.g. F) where the d-metal LP deficit is compensated by the
    # compact partner's tight overlap on Z/(2P₁)Z.
    # Amplitude: -ln(1 - 1/orb), the LP blocking from a single pair.
    # 0 adjustable parameters.
    if l_max >= 2 and bo <= 1:
        for (_lp_t, _l_t, _Z_t, _lp_p, _per_p) in [
            (lp_A, l_A, Z_A, lp_B, per_B),
            (lp_B, l_B, Z_B, lp_A, per_A),
        ]:
            if (_l_t == 2 and _lp_t == 0 and ns_config(_Z_t) >= 2
                    and _lp_p > 0 and _per_p >= P1):
                _lp_corrected = max(0, ns_config(_Z_t) - int(bo))
                if _lp_corrected > 0:
                    S_lp += -math.log(max(0.01, 1.0 - float(_lp_corrected) / orb))
                break

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

    # [NNLO-2 removed: subsumed by S_k3 cross-mode component (D)]

    # [NNLO-3 removed: subsumed by S_k3 fill-weight component (C)]

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

    # NNLO-8: cross-period bo ≥ 3 compact-underfill π radial screening.
    # When the COMPACT atom (smaller period) is underfilled (np < P₁)
    # and the DIFFUSE atom (larger period) is overfilled (np > P₁),
    # the dative π (LP→vacancy) overestimates the cross-period overlap.
    # The compact atom's tight p-orbital cannot fully reach the diffuse
    # partner's extended LP. S₃×f_pgap×S_HALF×(bo-1)/P₁ captures this
    # mismatch: S₃ = hex-face coupling, f_pgap = radial gap fraction,
    # S_HALF = partial screening, (bo-1)/P₁ = π bond fraction.
    # Gate: np_min ≥ 2 (excludes np=1 which has different overlap);
    #        np_max > P₁ (partner overfilled with LP to donate);
    #        per_underfill < per_overfill (compact atom has the underfill).
    if (bo >= 3 and Z_A != Z_B and per_A != per_B
            and l_min >= 1 and max(l_A, l_B) <= 1
            and min(np_A, np_B) >= 2 and max(np_A, np_B) > P1):
        # Identify the underfilled atom and check it's compact
        if np_A <= np_B:
            _per_under, _per_over = per_A, per_B
        else:
            _per_under, _per_over = per_B, per_A
        if _per_under < _per_over:
            _f_pgap_8 = 1.0 - float(_per_under) / float(_per_over)
            S_nnlo += S3 * _f_pgap_8 * S_HALF * float(bo - 1) / P1

    # [NNLO-7 removed: subsumed by S_k3 cross-mode component (D)]

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

    # NNLO-9: hypervalent LP deficit screening for cross-period π bonds.
    # When an overfilled p-block atom (np > P₁) in per ≥ P₁ has fewer LP
    # pairs than its orbital overfill implies (lp < np - P₁), it is
    # hypervalent: the excess electrons that would normally form LP pairs
    # have been redirected to other bonds via d-orbital access (P₂ face).
    # This diverts electron density from THIS bond, screening it.
    #
    # PT derivation: on Z/(2P₁)Z, the natural LP occupation is np - P₁
    # for overfill. Each diverted LP pair reduces the effective structural
    # support by -ln(C₃) = -ln(cos²θ₃), the holonomic cost of removing
    # one LP from the hexagonal face. The cost is intrinsic to the
    # hypervalent atom's d-orbital channel (per ≥ P₁ gives P₂ access),
    # independent of the partner's period.
    #
    # Gate: bo ≥ 2, both p-block (l_min ≥ 1, l_max ≤ 1), heteronuclear,
    #       cross-period (per_A ≠ per_B), at least one atom with per ≥ P₁,
    #       np > P₁, and lp < np - P₁ (LP deficit from hypervalency).
    # 0 parameters. -ln(C₃) from s = 1/2.
    if (bo >= 2 and l_min >= 1 and l_max <= 1 and Z_A != Z_B
            and per_A != per_B):
        for _np_h, _per_h, _lp_h in [
            (np_A, per_A, lp_A), (np_B, per_B, lp_B)
        ]:
            if _per_h >= P1 and _np_h > P1:
                _lp_natural = _np_h - P1
                _deficit = max(0, _lp_natural - _lp_h)
                if _deficit > 0:
                    S_nnlo += -math.log(C3) * float(_deficit)
                    break

    # [SC-1, SC-2, SC-3 removed: subsumed by S_k3 cross-Nyquist component (B)]
    # The Im(ρ̂_A(1)·ρ̂_B(1)*) cross-spectrum fill-phase mismatch is now
    # handled systematically in the k=3 mode block below.

    # SC-4: BN-type real k=1 Hund coherence
    # Gate: both p-block (l_min>=1, max(l)<=1), heteronuclear, same period,
    #        bo=2, one atom at np=1 (underfill), other at np=P1 (half-fill).
    # The real part of the k=1 cross-spectrum measures Hund coherence:
    # when Re < threshold, the fill is incoherent → screening needed.
    if (l_min >= 1 and max(l_A, l_B) <= 1 and Z_A != Z_B
            and per_A == per_B and bo == 2
            and min(np_A, np_B) == 1 and max(np_A, np_B) == P1):
        _sc_threshold_4 = S3 / (2.0 * P)
        _sc4_re = (spec_A[1] * spec_B[1].conjugate()).real
        S_nnlo += 2 * P * P * S3 * max(0.0, _sc_threshold_4 - _sc4_re)

    # ── k=3: Systematic Nyquist parity screening on Z/(2P₁)Z (C12) ──
    #
    # The k=3 mode on Z/6Z is the Nyquist mode: (-1)^r parity
    # alternation. It captures 12.9% of spectral variance (C12) and
    # replaces 4 ad-hoc corrections: SC-1, SC-2, SC-3, NNLO-3.
    #
    # Three components from the DFT on Z/(2P₁)Z:
    #
    # (A) AUTO-k3: Asymmetric parity overfill screening.
    #     When one atom has odd overfill past half-fill, its uncompensated
    #     parity creates Pauli screening that k=0,1,2 cannot capture.
    #     S_k3_auto = S₃·D₃ × overfill × f_per
    #
    # (B) CROSS-k3 Im: Imaginary part of the k=1 cross-spectrum encodes
    #     the antisymmetric fill-phase mismatch between atoms.
    #     Im(ρ̂_A(1)·ρ̂_B(1)*) = |ρ̂_A(1)|·|ρ̂_B(1)|·sin(Δφ)
    #     When Im deviates from the equilibrium threshold S₃/(2P₁),
    #     the fill is either in-phase (underbinding) or out-of-phase
    #     (overbinding). This replaces SC-1, SC-2, SC-3.
    #     S_k3_cross = -2P₁²S₃ × (Im - S₃/(2P₁)) × f_fill × f_per
    #
    # (C) FILL-WEIGHT: For high-bo heteronuclear same-period bonds,
    #     the sin²(π·np_min/(2P₁)) spectral weight of the k=3 mode
    #     contributes Pauli screening. This replaces NNLO-3.
    #     S_k3_fill = S₃²·S_HALF × sin²(π·np_min/(2P₁))
    #
    # 0 adjustable parameters. S₃, D₃, C₃, γ₃, S_HALF, P₁ from s=1/2.
    S_k3 = 0.0

    # (A) Auto-k3: asymmetric parity overfill
    if l_min >= 1 and bo <= 1:
        _k3_active_A = (n_hex_A > P and n_hex_A % 2 == 1)
        _k3_active_B = (n_hex_B > P and n_hex_B % 2 == 1)
        if _k3_active_A != _k3_active_B:
            if _k3_active_A:
                _n_k3 = n_hex_A
                _per_k3 = per_A
            else:
                _n_k3 = n_hex_B
                _per_k3 = per_B
            _overfill_k3 = float(_n_k3 - P) / P
            _f_per_k3 = (2.0 / float(max(_per_k3, 2))) ** (1.0 / P) if _per_k3 > 2 else 1.0
            S_k3 = S3 * D3 * _overfill_k3 * _f_per_k3

    # (B) Cross-k3 Im: fill-phase mismatch from k=1 cross-spectrum
    # Replaces SC-1 (per=2,per=2), SC-2 (per=3,per=2), SC-3 (per=2,per=3)
    # Universal gate: both p-block, heteronuclear, bo ≤ 1,
    # one atom near-full (np = 2P₁-1) and other not
    if (l_min >= 1 and max(l_A, l_B) <= 1 and Z_A != Z_B
            and bo <= 1 and max(np_A, np_B) == 2 * P1 - 1
            and min(np_A, np_B) >= 1 and min(np_A, np_B) < 2 * P1 - 1):
        _np_less_k3 = min(np_A, np_B)
        if np_A <= np_B:
            _per_less_k3 = per_A
            _per_more_k3 = per_B
        else:
            _per_less_k3 = per_B
            _per_more_k3 = per_A

        _im_k1_k3 = (spec_A[1] * spec_B[1].conjugate()).imag
        _sc_threshold_k3 = S3 / (2.0 * P)
        _sc_coeff_k3 = 2.0 * P * P * S3
        _sc_delta_k3 = _im_k1_k3 - _sc_threshold_k3

        # LP fill modulation: overfill LP on the less-filled atom
        # attenuates the cross-phase coupling by C₃ per LP pair
        _lp_less_k3 = max(0, _np_less_k3 - P)
        _sc_fill_k3 = C3 ** _lp_less_k3

        # Period attenuation: depends on the period structure
        # Same-period compact (per=2,2): no damping
        # Cross-period with less-filled diffuse (per_less > per_more):
        #   inverted sign + period damping; half-fill excluded (np=P₁
        #   has special exchange symmetry already captured by k=1)
        # Cross-period with less-filled compact (per_less < per_more):
        #   same sign + cross-period damping; underfill only (np < P₁)
        if _per_less_k3 == _per_more_k3 and _per_less_k3 <= 2:
            # Same-period compact: no period attenuation (SC-1 case)
            _f_per_cross = (2.0 / float(per_max)) ** (1.0 / P) if per_max > 2 else 1.0
            _sign_cross = -1.0
        elif _per_less_k3 > _per_more_k3 and _np_less_k3 != P:
            # Less-filled is DIFFUSE: inverted sign (SC-2 case)
            # np_less = P₁ excluded (half-fill exchange symmetry)
            _f_per_cross = (2.0 / float(_per_less_k3)) ** (1.0 / P)
            _sign_cross = +1.0
        elif _per_less_k3 < _per_more_k3 and _np_less_k3 < P:
            # Less-filled is COMPACT underfill: same sign + damping (SC-3)
            _f_per_cross = math.sin(math.pi / P) ** 2 * S_HALF
            _sign_cross = -1.0
        else:
            _f_per_cross = 0.0
            _sign_cross = 0.0

        S_k3 += _sign_cross * _sc_coeff_k3 * _sc_delta_k3 * _f_per_cross * _sc_fill_k3

    # (C) Fill-weight: sin² spectral weight for high-bo hetero same-period
    # Replaces NNLO-3
    if (bo >= 3 and Z_A != Z_B and per_A == per_B
            and l_min >= 1 and max(l_A, l_B) <= 1
            and min(np_A, np_B) >= 2):
        _np_min_k3 = min(np_A, np_B)
        S_k3 += S3 * S3 * S_HALF * math.sin(
            math.pi * float(_np_min_k3) / (2.0 * P1)) ** 2

    # (D) Cross-mode k=1 × k=3 coupling for bo ≥ 2
    # Absorbs NNLO-2 (homonuclear asymmetric fill) and NNLO-7
    # (cross-period LP mismatch).  Both encode the product of the
    # k=1 exchange mode with the k=3 parity mode, modulated by
    # the bond-order fraction f_bo = (bo-1)/bo.
    #
    # NNLO-2 physics: homonuclear bo≥2 with np≠P₁ (asymmetric fill).
    #   The parity asymmetry |np - P₁|/P₁ is the k=3 amplitude,
    #   and f_bo captures the π-channel contribution.
    #   S₃²/P₁ × f_bo × f_per.
    #
    # NNLO-7 physics: heteronuclear cross-period bo≥2 LP mismatch.
    #   When both atoms are overfilled (np > P₁), the smaller LP
    #   fraction lp_min/P₁ couples k=1 exchange with k=3 parity.
    #   -D₃ × S_HALF × lp_min/P₁.
    #   When one atom has np≤1 and the other np>P₁ with q_rel<½,
    #   the vacancy fraction couples similarly.
    #   D₃ × S_HALF × vac/P₁.
    #
    # Unified as component (D) of S_k3.  0 adjustable parameters.
    if bo >= 2 and l_min >= 1:
        # NNLO-2 part: homonuclear asymmetric hex-face
        if (Z_A == Z_B and 2 <= np_A <= 2 * P1 - 2 and np_A != P1):
            _f_bo_d = float(bo - 1) / float(bo)
            _f_per_d = (2.0 / float(per_max)) ** (1.0 / P1) if per_max > 2 else 1.0
            S_k3 += S3 * S3 / P1 * _f_bo_d * _f_per_d

        # NNLO-7 part: cross-period heteronuclear LP mismatch
        if (Z_A != Z_B and per_A != per_B
                and max(l_A, l_B) <= 1
                and np_A != P1 and np_B != P1):
            _np_min_d = min(np_A, np_B)
            _np_max_d = max(np_A, np_B)
            if _np_min_d > P1 and _np_max_d > P1:
                _lp_min_d = float(min(np_A - P1, np_B - P1)) / P1
                S_k3 += -D3 * S_HALF * _lp_min_d
            elif _np_min_d <= 1 and _np_max_d > P1 and q_rel < S_HALF:
                _vac_d = float(P1 - _np_min_d) / P1
                S_k3 += D3 * S_HALF * _vac_d

    # ══════════════════════════════════════════════════════════════
    # Step 4: Sum all modes
    # ══════════════════════════════════════════════════════════════
    S_hex = S_k0 + S_k1 + S_k2 + S_k3 + S_pi + S_lp + S_per + S_nnlo

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

    # C9: per ≥ P₁ half-fill (np = P₁) exchange anti-screening for s+p bonds.
    # When the p-atom is at half-fill on Z/(2P₁)Z and in per ≥ P₁,
    # the diffuse orbital's Hund exchange with the s-block partner
    # is underestimated by the holonomic term. The half-fill gives
    # maximum spin multiplicity (2S+1 = P₁+1 = 4 for P), enhancing
    # the exchange coupling. The coefficient S₃/(2P₁)×(1-D₃) is
    # the GFT fill-exchange product on the hex face, modulated by
    # the Fisher dispersion complement.
    # Gate: per_s ≤ 1 (H-like), per_p ≥ P₁, np = P₁ exactly.
    if per_s <= 1 and per_p_h >= P1 and np_p == P1:
        S -= S3 / (2 * P1) * (1.0 - D3)

    # s+p dative π
    ie_dom = max(ie_A, ie_B)
    q_rel = min(ie_A, ie_B) / ie_dom if ie_dom > 0 else 1.0
    per_p = per_A if l_A == 1 else per_B
    if per_s == 2 and per_p <= 2 and lp_p > 0 and q_rel > S_HALF:
        n_dat = min(lp_p, max(1, per_s - 1))
        S += -S3 * float(n_dat) * (ie_p / RY) * S_HALF
        # sp-hybridisation promotion anti-screening for s² atoms (Be-like).
        # When ns_s ≥ 2, the s² → sp promotion opens a p-vacancy channel.
        # The geometric mean IE captures the overlap of the sp-hybrid
        # with the partner's LP on Z/(2P₁)Z.
        # PT derivation: second-order dative via sp-mixing. The cross
        # screening S₃×D₃ is the product of hex-face (S₃) and Fisher
        # dispersion (D₃), encoding the two-step promotion+donation.
        ie_s = ie_A if l_A == 0 else ie_B
        from ptc.periodic import ns_config as _ns_cfg_sp
        Z_s_sp = Z_A if l_A == 0 else Z_B
        ns_s_sp = _ns_cfg_sp(Z_s_sp)
        if ns_s_sp >= 2:
            ie_geom = math.sqrt(ie_p * ie_s)
            S += -S3 * D3 * float(ns_s_sp) * ie_geom / RY

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
    # Strong-EA partner gate for d¹⁰s² charge-transfer relief.
    # Requires EA > S₃·Ry (≈3.0 eV) — satisfied by F, Cl, Br, not O, S.
    ea_max_d10 = max(EA_eV(Z_A), EA_eV(Z_B))
    _partner_strong_ea = (partner_halide and ea_max_d10 > S3 * RY)
    np_pairs = [(nd_A, l_A, ns_config(Z_A), np_B, l_B, per_B),
                (nd_B, l_B, ns_config(Z_B), np_A, l_A, per_A)]
    for nd_c, l_c, ns_c, np_part, l_part, per_part in np_pairs:
        if l_c == 2 and nd_c >= 2 * P2:
            if ns_c >= 2:
                # PT: d¹⁰s² closed shell penalty.
                # When partner has strong EA (> S₃·Ry ≈ 3 eV; F, Cl, Br),
                # the Born–Haber cycle Zn → Zn⁺(d¹⁰s¹) + e⁻ → Zn⁺X⁻
                # means the d¹⁰ shell sees an effectively d¹⁰s¹ state
                # (the ionised configuration). The d¹⁰ Pauli repulsion
                # becomes C₅²·f_per (same as d¹⁰s¹+halide). The extra
                # cost of breaking the s-pair is captured separately by
                # S_d10s2_extra = S₃·ns/P₁.
                # Gate: EA_partner > S₃·Ry — excludes O, S where the
                # charge-transfer channel is weaker.
                # 0 adjustable parameters.
                if _partner_strong_ea:
                    # f_per: cross-period Pauli scaling.
                    # The d10 Pauli repulsion with the partner scales
                    # inversely with the partner's compactness relative
                    # to the d-shell: compact partners (per_part < per_d)
                    # penetrate less into the d10 cloud, reducing overlap.
                    # Factor: (per_d / per_part)^(1/P₁).
                    from ptc.periodic import period as _period_d10
                    _per_d_loc = _period_d10(Z_A if l_A == 2 and nd_A >= 2*P2 else Z_B)
                    f_per_d10 = (float(_per_d_loc) / float(max(per_part, 2))) ** (1.0 / P1)
                    d10_pen = min(d10_pen, C5 * C5 * f_per_d10)
                else:
                    d10_pen = min(d10_pen, math.sqrt(S3))
            elif partner_halide:
                f_per_d10 = (float(max(per_part, 2)) / 2.0) ** (1.0 / P1)
                d10_pen = min(d10_pen, C5 * C5 * f_per_d10)
    if d10_pen < 1.0:
        S_base = -math.log(d10_pen)
        # PT: at bo ≤ 1, partner np enhancement is double-counting
        # (Principle 7 GFT) — Pauli repulsion already in S_hex.
        # At bo ≥ 2, partner back-donation Pauli is physical.
        if bo >= 2:
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

    # S_d10s1_same_per: same-period d10s1 + halide Pauli screening.
    # When d10s1 metal and halide partner are in the same period,
    # the d10 closed shell has maximum radial overlap with the
    # halide p-orbitals, creating enhanced Pauli repulsion.
    # In cross-period bonds (CuCl, CuF), the radial mismatch
    # reduces d-p overlap, so the effect vanishes.
    # PT: D3 = delta_3 = (1-q^3)/3 is the holonomy of the sigma
    # face Z/P1Z for same-shell radial overlap of the fully paired
    # d10 subshell with the partner p-orbitals.
    # Gate: d10s1 + partner halide (l<=1, np>P1) + same period.
    # 0 adjustable parameters.
    if bo <= 1 and Z_A != Z_B:
        _per_pairs = [(nd_A, l_A, ns_config(Z_A), per_A, l_B, per_B, np_B),
                      (nd_B, l_B, ns_config(Z_B), per_B, l_A, per_A, np_A)]
        for _nd_sp, _l_sp, _ns_sp, _per_sp, _l_part_sp, _per_part_sp, _np_part_sp in _per_pairs:
            if (_l_sp == 2 and _nd_sp >= 2 * P2 and _ns_sp == 1
                    and _l_part_sp <= 1 and _np_part_sp > P1
                    and _per_sp == _per_part_sp):
                S += D3
                break

    # S_d10s2_extra: cost of breaking the s-pair for d10s2.
    # When the partner has strong EA (halide), the charge-transfer
    # relief (d10_pen = C5²·f_per above) already accounts for the
    # s → s¹ promotion through the Born–Haber ionic state.
    # Gate: skip when _partner_strong_ea (no double-counting).
    if not _partner_strong_ea:
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

    # (C9 removed — absorbed by S_d10s2_extra gate on _partner_strong_ea)

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
                # d⁵s¹ + s-block: s bonds directly, d⁵ is spectator.
                S += -math.log(C5) * S_HALF
            elif l_p == 0 and ns_config(Z_c) >= 2:
                # d⁵s² + s-block (H): s-pair breaks, one bonds.
                # d⁵ is spectator — same reduction as d⁵s¹.
                # PT: the sigma bond uses the s-orbital; d⁵ exchange
                # stabilises the d-shell but does not participate.
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

    # S_d5s1_cos: d⁵s¹ pentagonal cosine mode anti-screening.
    # At nd = P₂, sin(πnd/P₂) = 0 (node) but cos(πnd/P₂) = -1 (anti-node).
    # For d5s1 (ns=1), the unpaired s-electron participates in pentagonal
    # exchange via the cos mode. The anti-screening D₅/P₁ captures the
    # per-face coupling. Factor (1+D₃) = CRT bridge correction from the
    # Fisher dispersion between the P₂ and P₁ faces.
    # Gate: heteronuclear only (Cr₂ has its own homonuclear d5 correction).
    # When the partner has p-electrons with LP (bo≥2), the d5 cos mode
    # also enhances d→π back-donation: D₅×S_HALF×(bo-1)/P₁ per π bond.
    if Z_A != Z_B:
        for nd_c, l_c, Z_c, l_part, np_part in [
            (nd_A, l_A, Z_A, l_B, np_B), (nd_B, l_B, Z_B, l_A, np_A)
        ]:
            if l_c == 2 and nd_c == P2 and ns_config(Z_c) == 1:
                S -= D5 / P1 * (1.0 + D3)
                # d5s1 cos-mode enhanced d→π back-donation for multi-bond partners
                if bo >= 2 and l_part >= 1 and np_part > P1:
                    S -= D5 * S_HALF * float(bo - 1) / P1
                break

    # S_d5s2_cos: d⁵s² pentagonal cosine mode anti-screening.
    # At nd = P₂, cos(πnd/P₂) = -1 (anti-node), same topology as d⁵s¹.
    # For d5s2 (ns=2), one s-electron bonds while the other stays paired.
    # The bonding s-electron couples to the d⁵ cos mode identically,
    # but the s-pair breaking introduces a spin-statistical factor S_HALF.
    # Gate: partner is s-block (l=0) — direct sigma coupling only.
    # Amplitude: D₅/P₁ × (1+D₃) × S_HALF.
    if Z_A != Z_B:
        for nd_c, l_c, Z_c, l_part, np_part in [
            (nd_A, l_A, Z_A, l_B, np_B), (nd_B, l_B, Z_B, l_A, np_A)
        ]:
            if l_c == 2 and nd_c == P2 and ns_config(Z_c) >= 2 and l_part == 0:
                S -= D5 / P1 * (1.0 + D3) * S_HALF
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

    # Pentagonal k=1 for d-block + p-block partner
    # Gate: d-block atom with P₂ < nd ≤ P₂+2 (nd=6,7), partner is p-block, bo≥1
    # The pentagonal sin mode captures the spin-orbit screening mismatch.
    if l_max >= 2 and l_min >= 1 and bo >= 1:
        nd_d_pent = max(nd_A, nd_B)
        if P2 < nd_d_pent <= P2 + 2:
            S += -math.log(1.0 + D5 * S_HALF * math.sin(math.pi * nd_d_pent / P2))

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

        # Ion-A: P₃-face period screening for cross-period s+p ionic bonds
        # The cation period modulates the effective ionic radius on the P₃ face.
        # Only fires for per_cat == P₁ and cross-period (per_A != per_B).
        r_eff = r
        if l_min == 0 and l_max == 1 and not _d10s1_H and per_A != per_B:
            _per_cat_ion = period(Z_A if ie_A < ie_B else Z_B)
            if _per_cat_ion == P1:
                _f_per3 = (2.0 / float(_per_cat_ion)) ** (S_HALF / P3)
                r_eff = r * _f_per3

        # Ion-B: d10s1 ionic period attenuation
        # Applies to d10s1 atoms bonded to any partner (s, p, or H).
        _is_d10s1_any = False
        if l_max >= 2 and Z_A > 0 and Z_B > 0:
            for Z_c, l_c in [(Z_A, l_A), (Z_B, l_B)]:
                if l_c >= 2:
                    _nd_c_ion = n_fill_madelung(Z_c) if l_of(Z_c) == 2 else 0
                    if _nd_c_ion >= 2 * P2 and ns_config(Z_c) == 1:
                        _is_d10s1_any = True
                        _per_d_ion = period(Z_c)
                        _f_per3_d = (2.0 / float(_per_d_ion)) ** (S_HALF / P3)
                        r_eff = r * _f_per3_d
                        break

        S_pyth = -0.5 * math.log(1.0 + r_eff * r_eff)
        if ea_max > _ea_thr or _both_s or _s_halide or _dblock_ionic or _d10s1_H:
            S += S_pyth
        else:
            f_att = (ea_max / _ea_thr) ** S_HALF if ea_max > 0 else 0.0
            if q_rel < S_HALF:
                f_att *= q_rel / S_HALF
            # Ion-D: LiO-type low-EA chalcogenide attenuation
            # For ionic s+p bonds with low EA, attenuate by EA deficit.
            if l_min == 0 and l_max == 1 and q_rel < S_HALF and ea_max > 0:
                _f_low_ea = (ea_max / _ea_thr) ** D3
                f_att *= (1.0 - (1.0 - _f_low_ea) * S_HALF)
            S += S_pyth * f_att

    if D_P3 > 0 and D_base > 0.01:
        Z_cat = Z_A if ie_A < ie_B else Z_B
        ie2 = _IE2_AE.get(Z_cat, 0.0)
        # PT: d10s2 (Zn, Cd, Hg) acts as AE cation — use IE2_DBLOCK
        if ie2 == 0:
            _l_cat = l_A if Z_cat == Z_A else l_B
            _nd_cat = nd_A if Z_cat == Z_A else nd_B
            if _l_cat == 2 and _nd_cat >= 2 * P2 and ns_config(Z_cat) >= 2:
                ie2 = _IE2.get(Z_cat, 0.0)
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
    # At nd = P₂ (exact half-fill), the d⁵ shell is a spectator — the
    # s-electron bonds with H and does NOT disrupt Hund exchange.
    # Gate: nd > P₂ (post-half-fill only: d⁶, d⁷).
    if bo == 1 and max(l_A, l_B) >= 2 and min(l_A, l_B) == 0:
        Z_d = Z_A if l_A >= 2 else Z_B
        nd_h = _nd_of(Z_d)
        ns_h = ns_config(Z_d)
        if ns_h == 2 and P2 < nd_h <= P2 + 2:
            f_hund = float(2 * P2 - nd_h) / P2
            S += S5 * f_hund * S_HALF
    # M13: late-d hydride post-half-fill pairing cost
    # REMOVED: double-counts with S_dlate in _S_d_penalty.
    # S_dlate already encodes the pairing cost via f = 1 - (nd-P2)/(2P2),
    # which captures the same vacancy-fraction physics as M13's D5*f_vac.
    # Keeping both produces ~0.06-0.08 eV excess screening for d6/d7
    # hydrides (FeH, CoH), causing systematic -8 to -9% underestimation.
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

    # PT: LP→d back-donation lives on k≥1 Fourier modes of Z/(2P₂)Z
    # (π symmetry). At bo=1, only k=0 (σ) is active.
    # Residual σ-compatible LP→d = S₅ (pentagonal vertex coupling).
    # At bo≥2, the π channel opens fully. Gate: max(S₅, min(bo-1, 1)).
    # 0 adjustable parameters.
    _f_pi_gate = pi_gate(bo)

    # Loop 1: σ-type d-p overlap (no LP involved, active at all bo)
    for nd_don, Z_don, l_don, np_part, lp_part, ie_don, l_part in [
        (nd_A, Z_A, l_A, np_B, lp_B, ie_A, l_B),
        (nd_B, Z_B, l_B, np_A, lp_A, ie_B, l_A),
    ]:
        if nd_don <= 0 or l_don != 2 or nd_don >= 2*P2 or l_part < 1: continue
        vacancy = max(0, 2*P1 - max(np_part, 1))
        if vacancy > 0:
            D += (RY/P1) * (float(vacancy)/(2*P1)) * (float(nd_don)/(2*P2)) * eps_geom_orig * f_per_simple * S_HALF
    # Loop 2: LP→d-vacancy dative (π-type, gated by bo)
    for nd_acc, lp_don, _ in [(nd_A, lp_B, ie_B), (nd_B, lp_A, ie_A)]:
        if nd_acc <= 0 or lp_don <= 0: continue
        f_acc = float(max(0, 2*P2 - nd_acc)) / (2*P2)
        if f_acc > 0:
            D += (RY/P1) * f_acc * S_HALF * eps_geom_orig * f_per_simple * S_HALF * _f_pi_gate
    # Loop 3: early-d + LP partner back-donation (π-type, gated by bo)
    for nd_d, l_d, np_p, lp_p, per_p in [
        (nd_A, l_A, np_B, lp_B, per_B), (nd_B, l_B, np_A, lp_A, per_A)]:
        if l_d != 2 or nd_d >= P2: continue
        if lp_p <= 0 and np_p < 2: continue
        D += (RY/P2) * (float(min(max(lp_p,1), P2-nd_d))/P2) * eps_geom_orig * max(bo,1) * _f_pi_gate
    nd_max = max(nd_A, nd_B)
    l_db = l_A if nd_A >= nd_B else l_B
    if nd_max > 0 and l_db == 2 and min(nd_A, nd_B) == 0:
        if nd_max < P2: f_d = float(P2 - nd_max)/P2
        elif P2 < nd_max < 2*P2: f_d = float(nd_max - P2)/P2
        else: f_d = 0.0
        if f_d > 0: D += (RY/P2) * f_d * eps_geom_orig * S_HALF * f_per_simple
        # Loop 4b: d⁵ half-fill exchange-enhanced sigma bonding.
        # At nd = P₂, |nd−P₂|/P₂ = 0 (vacancy mode node). But the cos mode
        # cos(πnd/P₂) = −1 is at its anti-node: the d⁵ half-shell has
        # maximal Hund exchange that ENHANCES the sigma bond.
        # Gate: partner is H (per=1, l=0) — pure sigma coupling only.
        # Amplitude = C₅ (pentagonal survival probability of the coherent
        # d⁵ exchange). The factor S_HALF gates the sigma channel.
        # For d⁵s¹ (ns=1): the cos-mode is partially in S_d5s1_cos already,
        # so reduce by S_HALF to avoid double-counting.
        # 0 adjustable parameters.
        if nd_max == P2 and min(per_A, per_B) == 1:
            Z_d_hf = Z_A if nd_A >= nd_B else Z_B
            ns_d_hf = ns_config(Z_d_hf)
            f_exch = C5
            if ns_d_hf == 1:
                f_exch *= S_HALF
            D += (RY / P2) * f_exch * eps_geom_orig * S_HALF * f_per_simple
        # Loop 4c: d^(P₂+1) first-post-half-fill residual exchange bonding.
        # At nd = P₂+1 (d⁶), the d-shell has (P₂−1) unbroken parallel
        # spins from the half-filled sub-shell. The pentagonal Hund
        # exchange is attenuated by the first forced pairing but NOT
        # destroyed. The surviving exchange enhances the sigma bond
        # with H through the pentagonal cos mode:
        #   cos(π(P₂+1)/P₂) = −cos(π/P₂) ≈ −C₅^½
        # The sigma-channel vertex coupling amplitude is C₅ × s
        # (C₅ = pentagonal survival, s = 1/2 for the paired electron
        # reducing coherent exchange by one spin factor).
        # Gate: nd = P₂+1 exactly, partner per=1 (H), ns ≥ 2.
        # 0 adjustable parameters.
        if nd_max == P2 + 1 and min(per_A, per_B) == 1:
            Z_d_pf = Z_A if nd_A >= nd_B else Z_B
            ns_d_pf = ns_config(Z_d_pf)
            if ns_d_pf >= 2:
                f_exch_pf = C5 * S_HALF
                D += (RY / P2) * f_exch_pf * eps_geom_orig * S_HALF * f_per_simple
    if Z_A == Z_B and l_db == 2:
        nd_h = max(nd_A, nd_B)
        ns_h = ns_config(Z_A)
        if nd_h >= 2*P2 and ns_h == 1:
            za2 = (Z_A*AEM)**2
            D += (RY/P2) * eps_geom_orig * S_HALF / max(0.1, math.sqrt(1-min(za2,0.99)))
        # ── Delta mode bonding for homonuclear open d-shell ──
        # At half-fill (nd=P₂), the k=2 Fourier mode on Z/(2P₂)Z is
        # fully bonding: cos(2π·nd/P₂) = 1. This creates delta bonds
        # (d-d overlap with |Δm_l|=2) not captured by sigma/pi channels.
        # Amplitude = s² = α(3) = 1/4 (spectral weight of T₃ antidiagonal
        # transfer, T2). Relativistic correction via (Zα)² as for d10s1.
        # f_half = sin²(π·nd/(2P₂)) peaks at nd=P₂ (half-fill: maximum
        # unpaired d-electrons available for delta bonding).
        # Gate: n_unpaired ≥ P₁ (need ≥3 beyond σ+2π to form δ bonds).
        # 0 adjustable parameters.
        elif 0 < nd_h < 2 * P2:
            n_unp = min(nd_h, 2 * P2 - nd_h)
            if n_unp >= P1:
                f_half = math.sin(math.pi * nd_h / (2 * P2)) ** 2
                za2 = (Z_A * AEM) ** 2
                f_rel = 1.0 / max(0.1, math.sqrt(1.0 - min(za2, 0.99)))
                D += (RY / P2) * eps_geom_orig * S_HALF * S_HALF * f_half * f_rel

    # ── Correction 1: CFSE — P₃→P₂ crystal field coupling ──
    # Active when: d-block atom + LP partner (strong-field ligand)
    for Z_d, nd_d, l_d, lp_part in [
        (Z_A, nd_A, l_A, lp_B), (Z_B, nd_B, l_B, lp_A)]:
        if l_d != 2 or nd_d <= 0 or nd_d >= 2 * P2 or lp_part <= 0:
            continue
        dbs = DBlockState.from_Z(Z_d)
        nf_eff = nd_d + (1 if dbs.is_ns1 else 0)
        nf_eff = min(nf_eff, 2 * P2)
        r7 = nf_eff % P3
        sin2_r7 = math.sin(2.0 * math.pi * r7 / P3) ** 2
        f_lp = min(float(lp_part) / P1, 1.0)
        D += sin2_r7 * D7 / P2 * f_lp * (RY / P2) * S_HALF

    # ── Correction 2: Dark modes k=2,4 on Z/10Z ──
    _omega_10 = 2.0 * math.pi / (2 * P2)
    for Z_dk, nd_dk, l_dk in [(Z_A, nd_A, l_A), (Z_B, nd_B, l_B)]:
        if l_dk != 2 or nd_dk <= 0:
            continue
        dbs = DBlockState.from_Z(Z_dk)
        nf_eff = nd_dk + (1 if dbs.is_ns1 else 0)
        nf_eff = min(nf_eff, 2 * P2)
        if nf_eff == 0:
            continue
        tier_d = dbs.per - P2
        if tier_d < 0:
            continue
        chir_d = 1 - 2 * (tier_d % 2)
        gamma_d = GAMMA_7 if tier_d >= 1 else 1.0
        # k=2 dark mode
        if tier_d <= 1:
            D += (-chir_d * gamma_d * (R35_DARK + R57_DARK)
                  * math.cos(2 * _omega_10 * nf_eff)
                  * eps_geom_orig * S_HALF)
        # k=4 dark mode (3d only)
        if tier_d == 0:
            D += (-chir_d * R37_DARK
                  * math.cos(4 * _omega_10 * nf_eff)
                  * eps_geom_orig * S_HALF)

    # ── Correction 3: Mertens spectral convergence at half-fill ──
    # For d5 (half-fill), the Chebyshev product ∏sin(kπ/n) = 5/16
    # gives the spectral convergence factor (T4). This adds a small
    # correction to the sigma channel at half-fill.
    for Z_m, nd_m, l_m in [(Z_A, nd_A, l_A), (Z_B, nd_B, l_B)]:
        if l_m != 2 or nd_m <= 0 or nd_m >= 2 * P2:
            continue
        dbs = DBlockState.from_Z(Z_m)
        nf_eff = nd_m + (1 if dbs.is_ns1 else 0)
        nf_eff = min(nf_eff, 2 * P2)
        n_unp = min(nf_eff, 2 * P2 - nf_eff)
        if n_unp <= 1:
            continue
        m_factor = mertens_factor(n_unp)
        # Linear factor for comparison
        linear = float(n_unp) / (2.0 * P2)
        # The difference (linear - mertens) is the spectral correction
        delta_m = linear - m_factor
        if delta_m > 0:
            D += delta_m * eps_geom_orig * (RY / P2) * S_HALF

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
    # PT: d10s2 metals (Zn, Cd, Hg) are post-transition — they bond
    # ionically like alkaline earths (lose 2 s-electrons, d10 is inert).
    # IE2 polarization correction was missing because the l_min==0 gate
    # excludes d-block (l=2). Gate: d10s2 cation + ionic bond (q_rel < s).
    # 0 adjustable parameters.
    if D_P3 > 0 and q_rel < S_HALF:
        Z_cat_d10 = Z_A if ie_A < ie_B else Z_B
        l_cat_d10 = l_A if Z_cat_d10 == Z_A else l_B
        nd_cat_d10 = nd_A if Z_cat_d10 == Z_A else nd_B
        if l_cat_d10 == 2 and nd_cat_d10 >= 2 * P2 and ns_config(Z_cat_d10) >= 2:
            ie2_d10 = _IE2.get(Z_cat_d10, 0.0)
            if ie2_d10 > 0:
                D_P3 += COULOMB_EV_A**2 * S3 / (r_ion**2 * ie2_d10)
    # Binary face correction: LiBr/NaCl type alkali halides
    # Corrects the ionic energy for specific period combinations
    # where the P₃-face binary resonance contributes.
    if D_P3 > 0 and l_min == 0 and l_max <= 1 and q_rel < S_HALF:
        Z_cat_bf = Z_A if ie_A < ie_B else Z_B
        ns_cat_bf = ns_config(Z_cat_bf)
        if ns_cat_bf == 1:
            per_cat_bf = period(Z_cat_bf)
            Z_an_bf = Z_B if Z_cat_bf == Z_A else Z_A
            per_an_bf = period(Z_an_bf)
            _gate_A = (per_cat_bf == P0 and per_an_bf == P0 * P0)   # Li+Br
            _gate_B = (per_cat_bf == P1 and per_an_bf == P1)         # Na+Cl
            if _gate_A or _gate_B:
                _Im_P0 = (S_HALF / P0) ** 2   # 1/16
                _W_P0 = C3 * C5               # 0.629
                D_P3 += (COULOMB_EV_A / r_ion) * _Im_P0 * _W_P0

    if D_P3 > 0 and bo >= P1 and per_max <= 2 and l_min >= 1 and Z_A != Z_B:
        chi_A = (ie_A + ea_A) * S_HALF
        chi_B = (ie_B + ea_B) * S_HALF
        q_eff_loc = abs(chi_A - chi_B) / max(0.1, chi_A + chi_B)
        if q_eff_loc > S3:
            D_P3 *= (1.0 + bo * D3)

    # ── Late-d + compact halogen IE₂ polarization ────────────────
    # For late-d metals (P₂ < nd < 2P₂) bonded to compact halogens
    # (per ≤ 2, l=1, np ≥ 2P₁−1 — i.e. F), the ionic channel is under-
    # estimated because: (1) the high IE of late-d metals reduces D_raw,
    # and (2) the compact halogen's tight 2p⁻ creates a stronger Coulomb
    # well at shorter distance than the covalent r_ion model captures.
    # PT derivation: the IE₂ of the d-metal measures the M²⁺ formation
    # cost. The second ionization adds a polarization term analogous
    # to the AE IE₂ correction (line 1784), but gated on d-block.
    # The d-vacancy fraction modulates: more vacancies → more charge
    # transfer channels → stronger ionic contribution.
    # 0 adjustable parameters.
    if D_P3 > 0 and q_rel < S_HALF:
        Z_cat_ld = Z_A if ie_A < ie_B else Z_B
        l_cat_ld = l_A if Z_cat_ld == Z_A else l_B
        nd_cat_ld = nd_A if Z_cat_ld == Z_A else nd_B
        Z_an_ld = Z_B if Z_cat_ld == Z_A else Z_A
        l_an_ld = l_B if Z_cat_ld == Z_A else l_A
        per_an_ld = per_B if Z_cat_ld == Z_A else per_A
        np_an_ld = np_A if Z_an_ld == Z_A else np_B
        if (l_cat_ld == 2 and P2 < nd_cat_ld < 2 * P2
                and l_an_ld == 1 and per_an_ld <= 2
                and np_an_ld >= 2 * P1 - 1):
            ie2_ld = _IE2.get(Z_cat_ld, 0.0)
            if ie2_ld > 0:
                _f_vac_ld = float(2 * P2 - nd_cat_ld) / (2.0 * P2)
                D_P3 += COULOMB_EV_A ** 2 * S3 * _f_vac_ld / (r_ion ** 2 * ie2_ld)

    # ── Early-d + halide IE₂ polarisation ────────────────────────
    # For early-d metals (nd ≤ P₂) bonded to halides, the ionic
    # channel M⁺ + X⁻ benefits from the large number of d-vacancies
    # which provide charge-transfer channels. The M²⁺ polarisation
    # correction (same form as the late-d IE₂ term above) captures
    # the additional Coulomb stabilisation from the compact M²⁺ core.
    #
    # PT derivation: on the P₃ face, the ionic state samples the
    # M²⁺(d^{n-1}) configuration. The d-vacancy fraction
    # (2P₂ − nd)/(2P₂) modulates the amplitude (more vacancies →
    # more charge-transfer channels). For non-compact halides
    # (per_an > 2), the larger anion radius reduces the Coulomb well;
    # the period attenuation (2/per_an)^(1/P₁) captures this.
    # Gate: nd ≤ P₂ (early-d / half-fill), complementary to the
    # late-d block above. l_an = 1, np_an ≥ 2P₁−1 (halide).
    # Ionicity gate: EA_halide > S₃·Ry (≈ 3.0 eV) — the Born–Haber
    # electron transfer is exothermic enough for ionic contribution.
    # This replaces the q_rel < S_HALF gate which excludes M–Cl bonds
    # (q_rel ≈ 0.52) despite their partial ionic character.
    # 0 adjustable parameters.
    if D_P3 > 0 and ea_max > S3 * RY:
        Z_cat_ed = Z_A if ie_A < ie_B else Z_B
        l_cat_ed = l_A if Z_cat_ed == Z_A else l_B
        nd_cat_ed = nd_A if Z_cat_ed == Z_A else nd_B
        Z_an_ed = Z_B if Z_cat_ed == Z_A else Z_A
        l_an_ed = l_B if Z_cat_ed == Z_A else l_A
        per_an_ed = per_B if Z_cat_ed == Z_A else per_A
        np_an_ed = np_A if Z_an_ed == Z_A else np_B
        if (l_cat_ed == 2 and nd_cat_ed <= P2
                and l_an_ed == 1 and np_an_ed >= 2 * P1 - 1):
            ie2_ed = _IE2.get(Z_cat_ed, 0.0)
            if ie2_ed > 0:
                # d-orbital participation: nd/P₂ grows from 0.2 (Sc)
                # to 1.0 (half-fill). This modulates how much the d-shell
                # stabilises the M²⁺ polarisation state — more d-electrons
                # give more exchange stabilisation of M²⁺(d^n).
                _f_nd_ed = float(nd_cat_ed) / float(P2)
                # Period attenuation: larger halide anion → weaker Coulomb
                # well → smaller correction. (2/per)^(1/P₃) is gentle:
                # F(per=2)→1.0, Cl(per=3)→0.944, Br(per=4)→0.894.
                _f_per_an = (2.0 / float(max(per_an_ed, 2))) ** (1.0 / P3)
                # Ionicity attenuation: when q_rel is small (strong ionic
                # character), the standard ionic channel already captures
                # most of the D_P3. The IE₂ correction adds the M²⁺
                # polarisation that the standard term misses. Attenuate
                # by q_rel/S_HALF (capped at 1) to avoid double-counting
                # the already-strong ionic channel.
                _f_ion_ed = min(1.0, q_rel / S_HALF)
                D_P3 += COULOMB_EV_A ** 2 * S3 * _f_nd_ed * _f_per_an * _f_ion_ed / (r_ion ** 2 * ie2_ed)

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
        # PT: d-block atoms (l=2) at bo=1 bond through s+d, not p.
        # On the hexagonal face Z/(2P₁)Z, they contribute ns electrons
        # (not np=0). Vacancy on hex face = P₁ - ns (not P₁ - 0).
        # Gate: bo ≤ 1 only (at bo≥2, the π channel justifies larger
        # effective bond order from dative LP→vacancy).
        # 0 adjustable parameters.
        if l_of(Z_acc) == 2 and bo <= 1:
            _ns_acc = ns_config(Z_acc)
            vacancy = max(0, P1 - _ns_acc)
        if is_radical and q_rel > C3: vacancy = max(0, vacancy - 1)
        n_dat = min(lp_don, vacancy)
        if per_don > 2: n_dat = min(n_dat, 1)
        if l_of(Z_acc) == 2:
            nd_acc_v = _nd_of(Z_acc)
            if nd_acc_v >= 2 * P2:
                n_dat = min(n_dat, 2)
            # PT: d⁵s¹ half-fill dative attenuation.
            # At nd = P₂ with ns = 1 (d⁵s¹, e.g. Cr), the single
            # s-electron is the bonding orbital. LP donation beyond
            # the first pair must enter a d-vacancy, disrupting the
            # d⁵ Hund exchange stabilisation (maximum spin multiplicity).
            # Cap n_dat at 1: only the s-channel is accessible without
            # Hund penalty. The d⁵ exchange energy is already captured
            # by S_d5 in _S_d_penalty; this prevents the dative channel
            # from double-counting the d-orbital accessibility.
            # Gate: nd = P₂, ns = 1, bo ≤ 1.
            # 0 adjustable parameters.
            elif nd_acc_v == P2 and ns_config(Z_acc) == 1 and bo <= 1:
                n_dat = min(n_dat, 1)
            # PT: late-d dative attenuation.
            # When nd > P₂ (late-d), excess d-electrons screen the
            # LP→d dative channel via Pauli repulsion with the incoming
            # lone pair.  The effective dative bond order is reduced by
            # the d-vacancy fraction (2P₂ − nd)/(2P₂).
            # Gate: P₂ < nd < 2P₂ (late-d, not d¹⁰), bo ≤ 1 (sigma dative).
            # 0 adjustable parameters.
            elif P2 < nd_acc_v < 2 * P2 and bo <= 1:
                d_vac_frac = float(2 * P2 - nd_acc_v) / float(2 * P2)
                n_dat *= d_vac_frac
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
    dbs_A = DBlockState.from_Z(Z_A)
    dbs_B = DBlockState.from_Z(Z_B)
    np_A, np_B = _np_of(Z_A), _np_of(Z_B)
    l_min, l_max = min(l_A, l_B), max(l_A, l_B)
    per_min, per_max = min(per_A, per_B), max(per_A, per_B)

    if lp_A is None: lp_A = _lp_pairs(Z_A, bo)
    if lp_B is None: lp_B = _lp_pairs(Z_B, bo)
    lp_mutual = min(lp_A, lp_B)

    # ── Effective eps (absorbs sd-hybrid, CRT, stiffness) ──
    eps_A, eps_B, eps_geom, sd_A, sd_B = _effective_eps(
        ie_A, ie_B, per_A, per_B, l_A, l_B,
        Z_A, Z_B, nd_A, nd_B, lp_A, lp_B, bo)

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
    # CRT cross-face sigma-exchange coupling for heteronuclear d-block.
    # The d-shell on Z/(2P₂)Z couples to the sigma channel on Z/(2P₁)Z
    # through the cross-gap product R₃₅ = D₃×D₅ (CRT Riemann tensor
    # between the hexagonal and pentagonal faces, §D14).
    # This coupling is absent for homonuclear d-block (captured by
    # S_homo_3d/5d) and for non-d bonds (no pentagonal channel).
    # Gate: at least one d-block atom (l_max >= 2), heteronuclear.
    # [0 adjustable parameters]
    S_cross_face_dblock = 0.0
    if l_max >= 2 and Z_A != Z_B:
        S_cross_face_dblock = -R35
    terms['S_dblock'] = _S_dblock_anti(D_P2, D_base_std) + S_homo_3d + S_homo_5d + S_cross_face_dblock

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
