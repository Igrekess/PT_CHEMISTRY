"""
dft_polygon.py — Discrete Fourier Transform engine on Z/(2P)Z polygons.

Replaces the ad-hoc S_count + S_exchange + S_exclusion + S_pi terms
with a PRINCIPLED DFT computation on the three discrete circles:

  - Z/(2P₀)Z = Z/4Z   (binary face, s-electrons)
  - Z/(2P₁)Z = Z/6Z   (hexagonal face, p-electrons)
  - Z/(2P₂)Z = Z/10Z  (pentagonal face, d-electrons)

Core identity (Parseval):
  Σ_k |ρ̂(k)|² = (1/2P) Σ_r |ρ(r)|² = n_e/(2P) = fill fraction

GFT identity on each face:
  log₂(2P) = D_KL(ρ || U) + H(ρ)

Cross-spectrum = bond coupling:
  S_bond(k) = Re[ρ̂_A(k) × conj(ρ̂_B(k))]

April 2026 — Theorie de la Persistance
"""
import math
import cmath
from dataclasses import dataclass
from typing import List, Tuple, Optional

from ptc.constants import (
    S3, S5, S7, C3, C5, C7,
    D3, D5, D7,
    P0, P1, P2, P3,
    S_HALF, RY, D_FULL,
    GAMMA_3, GAMMA_5, GAMMA_7,
    AEM,
)

# ──────────────────────────────────────────────────────────────────────
#  PT weights for each DFT mode
# ──────────────────────────────────────────────────────────────────────

# Mode weights w(k) for bond coupling, indexed by (P, k).
# w(0) = 1 always (DC = counting mode, Shannon).
# w(k>0) derives from holonomic sin²(πk/P) attenuated by S_p cascade.
#
# PT derivation:
#   - k=0 measures total fill fraction (counting, information mode).
#   - k=1 measures dipolar asymmetry relative to half-fill (exchange).
#   - k=2 measures quadrupolar structure (Pauli).
#   - Higher k: attenuated by S_p^k (cascade filtering).

def _holonomic_weight(P: int, k: int) -> float:
    """Holonomic weight for mode k on polygon Z/(2P)Z.

    w(k) = sin²(πk/P) for k > 0  [T6 holonomy]

    This connects the DFT mode to the geometric angle on the polygon.
    sin²(πk/P) = 0 at k=0 (counting mode, handled separately)
    sin²(πk/P) = 1 at k=P/2 (maximum asymmetry for even P)
    """
    if k == 0:
        return 1.0
    return math.sin(math.pi * k / P) ** 2


def _mode_weights(P: int) -> List[float]:
    """Compute all mode weights for polygon Z/(2P)Z.

    Returns array of length P with weights w(k) for k = 0, ..., P-1.

    The weight encodes the PT coupling strength:
    - k=0: w=1 (Shannon counting)
    - k>0: sin²(πk/P) × S_p (holonomic × cascade attenuation)

    The cascade attenuation S_p ensures that higher modes are
    filtered by the prime associated with this face.
    """
    S_p = {P0: S3, P1: S3, P2: S5, P3: S7}.get(P, S3)
    weights = []
    for k in range(P):
        if k == 0:
            weights.append(1.0)
        else:
            w_holo = math.sin(math.pi * k / P) ** 2
            # Cascade attenuation: each mode beyond k=0 is filtered
            # by S_p^(floor((k-1)/2)+1) — alternating between sin and cos
            # branches of the holonomic angle.
            cascade_order = (k - 1) // 2 + 1
            w_cascade = S_p ** cascade_order
            weights.append(w_holo * w_cascade)
    return weights


def _gft_mode_weights(P: int) -> List[float]:
    """GFT-calibrated mode weights for bond screening on Z/(2P)Z.

    These weights convert the DFT cross-spectrum into SCREENING units
    compatible with the current engine's log-based S terms.

    PT derivation for each mode:

    w(0) = 1  (DC mode, processed through GFT: -½ ln(f_sum))
        The k=0 mode measures total fill fraction.  The GFT identity
        log₂(2P) = D_KL + H maps fill information to screening via
        the Shannon log.  Weight is 1 because the log transform
        carries the physical scale.

    w(1) = S_p / (C_p × D_p)  (dipole/exchange mode)
        The k=1 mode measures dipolar asymmetry on the polygon.
        Exchange screening scales as S_p (the face coupling constant)
        divided by C_p × D_p (the complement × dispersion product).
        PT: this is the inverse Fisher information per mode:
        w(1) = sin²θ_p / [cos²θ_p × δ_p(2-δ_p)] = S_p/(C_p×D_p).
        For P₁=3: w(1) = S₃/(C₃×D₃) = 0.219/(0.781×0.172) = 1.63.

    w(2) = 1 / (C_p × S_HALF)  (quadrupole/Pauli mode)
        The k=2 mode measures quadrupolar structure (Pauli exclusion).
        The Pauli principle operates at the spin vertex (×S_HALF)
        on the complement face (×C_p).  Weight inverts this product.
        For P₁=3: w(2) = 1/(C₃×s) = 1/(0.781×0.5) = 2.56.

    w(k>2) = w(1) × S_p^(k-1)  (higher harmonics, cascade-filtered)
        Higher modes are geometrically attenuated by S_p per order.

    These weights are used in the GFT bond screening function
    bond_screening_gft(), which processes each mode through the
    appropriate logarithmic transformation.
    """
    S_p = {P0: S3, P1: S3, P2: S5, P3: S7}.get(P, S3)
    C_p = 1.0 - S_p
    D_p = S_p * C_p  # Fisher dispersion = S_p(1-S_p)

    weights = []
    for k in range(P):
        if k == 0:
            # DC mode: weight = 1 (GFT log transform carries the scale)
            weights.append(1.0)
        elif k == 1:
            # Dipole/exchange: S_p / (C_p × D_p)
            weights.append(S_p / max(C_p * D_p, 1e-10))
        elif k == 2:
            # Quadrupole/Pauli: 1 / (C_p × S_HALF)
            weights.append(1.0 / max(C_p * S_HALF, 1e-10))
        else:
            # Higher harmonics: w(1) × S_p^(k-1) cascade
            w1 = S_p / max(C_p * D_p, 1e-10)
            weights.append(w1 * S_p ** (k - 1))
    return weights


# ──────────────────────────────────────────────────────────────────────
#  Electron density on Z/(2P)Z
# ──────────────────────────────────────────────────────────────────────

def electron_density(n_e: int, P: int, lp: int = 0) -> List[int]:
    """Place n_e electrons on circle Z/(2P)Z.

    Returns array of length 2P with occupation 0 or 1.

    Filling follows Aufbau on the polygon:
    - Positions 0..P-1: spin-up (Hund's rule, fill sequentially)
    - Positions P..2P-1: spin-down (pairing, fill sequentially)

    LP electrons are placed first (positions 0..lp-1 AND P..P+lp-1),
    because lone pairs occupy BOTH spin positions of an orbital.

    Parameters
    ----------
    n_e : int
        Number of electrons to place (0 to 2P).
    P : int
        The prime: P=2 (s), P=3 (p), P=5 (d), P=7 (f).
    lp : int
        Number of lone pairs. Each LP occupies one orbital doubly.
        LP electrons are placed first in the lowest orbitals.

    Returns
    -------
    rho : list of int
        Occupation array of length 2P. rho[r] = 0 or 1.
    """
    N = 2 * P
    rho = [0] * N
    n_e = max(0, min(int(n_e), N))
    lp = max(0, min(int(lp), P))

    # Place LP electrons first: each LP fills both spin-up and spin-down
    # at the same orbital position (positions i and i+P are the same orbital).
    placed = 0
    for i in range(lp):
        if placed >= n_e:
            break
        rho[i] = 1       # spin-up
        placed += 1
        if placed >= n_e:
            break
        rho[i + P] = 1   # spin-down (paired)
        placed += 1

    # Fill remaining spin-up positions (Hund)
    for i in range(lp, P):
        if placed >= n_e:
            break
        rho[i] = 1
        placed += 1

    # Fill remaining spin-down positions (pairing)
    for i in range(lp, P):
        if placed >= n_e:
            break
        rho[i + P] = 1
        placed += 1

    return rho


# ──────────────────────────────────────────────────────────────────────
#  DFT spectrum on Z/(2P)Z
# ──────────────────────────────────────────────────────────────────────

def dft_spectrum(rho: List[int], P: int) -> List[complex]:
    """Compute DFT of density on Z/(2P)Z.

    Returns complex array rho_hat(k) for k = 0, ..., P-1.

    rho_hat(k) = (1/2P) sum_r rho(r) exp(-2*pi*i*k*r / (2P))

    Key modes:
    - k=0: DC component = n_e/(2P) = fill fraction
    - k=1: dipole = asymmetry relative to half-fill
    - k=2: quadrupole (for P >= 3)

    We only compute P modes (not 2P) because the density has a
    spin-symmetry that makes modes P..2P-1 redundant:
    the physics lives in the first P harmonics.
    """
    N = 2 * P
    spectrum = []
    for k in range(P):
        val = complex(0, 0)
        for r in range(N):
            angle = -2.0 * math.pi * k * r / N
            val += rho[r] * cmath.exp(complex(0, angle))
        val /= N
        spectrum.append(val)
    return spectrum


def spectral_power(spectrum: List[complex]) -> List[float]:
    """Compute |rho_hat(k)|^2 for each mode.

    Returns array of length P with power at each mode.
    """
    return [abs(c) ** 2 for c in spectrum]


# ──────────────────────────────────────────────────────────────────────
#  Parseval verification
# ──────────────────────────────────────────────────────────────────────

def parseval_check(rho: List[int], spectrum: List[complex], P: int) -> Tuple[float, float]:
    """Verify Parseval identity on Z/(2P)Z.

    Parseval: (1/2P) sum_r |rho(r)|^2 = sum_k |rho_hat(k)|^2

    Since rho(r) in {0,1}, |rho(r)|^2 = rho(r), so:
    LHS = n_e / (2P) = fill fraction

    Returns (lhs, rhs) for comparison.
    """
    N = 2 * P
    lhs = sum(r ** 2 for r in rho) / N  # since rho in {0,1}, r^2 = r
    rhs = sum(abs(c) ** 2 for c in spectrum)
    return lhs, rhs


# ──────────────────────────────────────────────────────────────────────
#  GFT identity
# ──────────────────────────────────────────────────────────────────────

def gft_decomposition(rho: List[int], P: int) -> Tuple[float, float, float]:
    """Compute GFT identity: log2(2P) = D_KL(rho || U) + H(rho).

    D_KL = sum_r rho_norm(r) * log2(rho_norm(r) / (1/(2P)))
    H    = -sum_r rho_norm(r) * log2(rho_norm(r))

    where rho_norm = rho / sum(rho) is the normalised density.

    Returns (log2_2P, D_KL, H_entropy).
    """
    N = 2 * P
    n_e = sum(rho)
    log2_N = math.log2(N)

    if n_e == 0 or n_e == N:
        # Edge cases: empty or full shell
        if n_e == 0:
            return log2_N, 0.0, 0.0
        else:
            # Full: uniform, D_KL = 0, H = log2(N)
            return log2_N, 0.0, log2_N

    # Normalise to probability distribution
    p_r = [float(x) / n_e for x in rho]

    D_KL = 0.0
    H = 0.0
    u = 1.0 / N  # uniform distribution

    for p in p_r:
        if p > 0:
            D_KL += p * math.log2(p / u)
            H -= p * math.log2(p)

    return log2_N, D_KL, H


# ──────────────────────────────────────────────────────────────────────
#  Bond coupling from cross-spectrum
# ──────────────────────────────────────────────────────────────────────

def bond_coupling(rho_A: List[int], rho_B: List[int], P: int) -> Tuple[float, List[float]]:
    """Compute bond screening from cross-spectrum.

    S_DFT = -sum_k Re[rho_hat_A(k) * conj(rho_hat_B(k))] * w(k)

    Positive S_DFT = screening (reduces D0).
    Negative S_DFT = anti-screening (enhances D0).

    Weights w(k) from PT:
    - w(0) = 1 (counting mode, Shannon)
    - w(k>0) = sin^2(pi*k/P) * S_p^order (holonomic * cascade)

    Parameters
    ----------
    rho_A, rho_B : list of int
        Electron densities on Z/(2P)Z for atoms A and B.
    P : int
        The polygon prime.

    Returns
    -------
    S_face : float
        Total screening for this face.
    S_per_mode : list of float
        Screening contribution from each mode k.
    """
    spec_A = dft_spectrum(rho_A, P)
    spec_B = dft_spectrum(rho_B, P)
    weights = _mode_weights(P)

    S_per_mode = []
    for k in range(P):
        cross = (spec_A[k] * spec_B[k].conjugate()).real
        S_k = -cross * weights[k]
        S_per_mode.append(S_k)

    S_face = sum(S_per_mode)
    return S_face, S_per_mode


def bond_coupling_normalized(rho_A: List[int], rho_B: List[int], P: int) -> Tuple[float, List[float]]:
    """Bond coupling normalized by Parseval constraint.

    Ensures total spectral power = s = 1/2 (the PT fundamental).

    The raw cross-spectrum is scaled so that the auto-spectrum
    satisfies Parseval: sum |rho_hat|^2 = fill_fraction.
    The cross terms are then normalized by geometric mean of fill fractions.
    """
    N = 2 * P
    n_A = sum(rho_A)
    n_B = sum(rho_B)
    f_A = n_A / N
    f_B = n_B / N

    if f_A <= 0 or f_B <= 0:
        return 0.0, [0.0] * P

    norm = math.sqrt(f_A * f_B)

    spec_A = dft_spectrum(rho_A, P)
    spec_B = dft_spectrum(rho_B, P)
    weights = _mode_weights(P)

    S_per_mode = []
    for k in range(P):
        cross = (spec_A[k] * spec_B[k].conjugate()).real
        S_k = -cross * weights[k] / norm
        S_per_mode.append(S_k)

    S_face = sum(S_per_mode)
    return S_face, S_per_mode


# ──────────────────────────────────────────────────────────────────────
#  GFT bond screening — logarithmic processing of DFT modes
# ──────────────────────────────────────────────────────────────────────

def bond_screening_gft(rho_A: List[int], rho_B: List[int], P: int,
                       bo: float = 1.0,
                       ie_A: float = 13.6, ie_B: float = 13.6,
                       Z_A: int = 0, Z_B: int = 0,
                       lp_total_A: int = -1, lp_total_B: int = -1) -> Tuple[float, dict]:
    """GFT-principled bond screening from DFT spectral decomposition.

    Processes each DFT mode through the appropriate GFT (logarithmic)
    transformation, giving screening values compatible with the current
    engine's S = -ln(f) structure.

    Architecture:
      S_total = S_fill + S_exchange + S_exclusion + S_pi + S_LP + S_per

    where each term derives from the spectral decomposition:
      S_fill      ← k=0 mode (fill fraction → GFT Shannon log)
      S_exchange  ← k=1 mode (dipolar asymmetry → exchange coupling)
      S_exclusion ← k=2 mode + overfill check (quadrupole → Pauli)
      S_pi        ← k>0 constructive interference (π bonding)
      S_LP        ← LP electron placement (fill blocking)
      S_per       ← period attenuation from T³ coordinates

    All weights from PT constants (S₃, D₃, C₃, P₁, s=½).
    0 adjustable parameters.

    Parameters
    ----------
    rho_A, rho_B : list of int
        Electron densities on Z/(2P)Z.
    P : int
        The polygon prime.
    bo : float
        Formal bond order.
    ie_A, ie_B : float
        Ionisation energies in eV (for exchange q_rel).
    Z_A, Z_B : int
        Atomic numbers (for period/block gates).

    Returns
    -------
    S_total : float
        Total screening from GFT processing.
    terms : dict
        Individual screening terms for diagnostics.
    """
    from ptc.atom import IE_eV

    N = 2 * P
    n_A = sum(rho_A)
    n_B = sum(rho_B)

    # If neither atom has electrons on this face, return zero screening.
    # The screening for this bond lives on a different polygon face.
    # Also return zero if only ONE atom has electrons (s+p bonds):
    # the screening is handled by the holonomic s+p coupling.
    _zero = {'S_fill': 0, 'S_exchange': 0, 'S_exclusion': 0,
             'S_pi': 0, 'S_LP': 0, 'S_per': 0,
             'f_A': 0, 'f_B': 0, 'f_sum': 0,
             'lp_A': 0, 'lp_B': 0, 'unp_A': 0, 'unp_B': 0,
             'q_rel': 1.0}
    if n_A == 0 and n_B == 0:
        return 0.0, _zero
    if n_A == 0 or n_B == 0:
        # One-sided: only LP blocking matters (if the occupied atom has LP).
        # Fill/exchange/pi are handled by holonomic in the caller.
        return 0.0, _zero

    f_A = n_A / N  # fill fraction A
    f_B = n_B / N  # fill fraction B
    f_sum = f_A + f_B  # total fill fraction on this face

    spec_A = dft_spectrum(rho_A, P)
    spec_B = dft_spectrum(rho_B, P)

    # Effective IE for exchange
    if Z_A > 0:
        ie_A = IE_eV(Z_A)
    if Z_B > 0:
        ie_B = IE_eV(Z_B)
    eps_A = ie_A / (S3 * 13.606 / S3)  # normalise to Ry scale
    eps_B = ie_B / (S3 * 13.606 / S3)
    # Simpler: just use ratio
    ie_max = max(ie_A, ie_B)
    q_rel = min(ie_A, ie_B) / ie_max if ie_max > 0 else 1.0

    # Face coupling constant
    S_p = {P0: S3, P1: S3, P2: S5, P3: S7}.get(P, S3)
    C_p = 1.0 - S_p
    D_p = S_p * C_p  # Fisher dispersion

    # Period data
    per_A = period(Z_A) if Z_A > 0 else 1
    per_B = period(Z_B) if Z_B > 0 else 1
    per_max = max(per_A, per_B)
    per_min = min(per_A, per_B)

    # l-block data
    l_A = l_of(Z_A) if Z_A > 0 else 0
    l_B = l_of(Z_B) if Z_B > 0 else 0
    l_min = min(l_A, l_B)
    l_max = max(l_A, l_B)

    # LP on this face
    lp_A_face = sum(1 for i in range(min(P, n_A // 2 + 1))
                    if i < P and rho_A[i] == 1 and i + P < N and rho_A[i + P] == 1)
    lp_B_face = sum(1 for i in range(min(P, n_B // 2 + 1))
                    if i < P and rho_B[i] == 1 and i + P < N and rho_B[i + P] == 1)
    lp_mutual = min(lp_A_face, lp_B_face)

    # Unpaired electrons (Hund orbitals)
    unp_A = n_A - 2 * lp_A_face
    unp_B = n_B - 2 * lp_B_face

    # ── k=0: GFT fill screening ──────────────────────────────────────
    # S_fill = -½ ln(min(f_sum, 1))
    # This is the Shannon information content of the combined fill.
    # At f_sum = 1 (exactly full): S_fill = 0 (no screening from fill).
    # At f_sum < 1 (underfilled): S_fill > 0 (screening from vacancies).
    # At f_sum > 1 (overfilled): capped at 0, vacancy handled separately.
    # When one atom has 0 electrons on this face (s+p bond), the fill
    # fraction is just the p-atom's contribution.
    S_fill = 0.0
    if f_sum <= 0:
        S_fill = 0.0  # empty face: no fill screening (handled elsewhere)
    elif f_sum < 1.0:
        S_fill = -S_HALF * math.log(f_sum)
    else:
        S_fill = 0.0
    # Fisher vacancy: for homonuclear overfill, the excess electrons
    # create redundant density.  Information = -½ ln(1-f) for f < s.
    f_c = min(float(n_A + n_B), float(N)) / float(N)
    if Z_A == Z_B and Z_A > 0 and P <= per_max <= P + 1 and f_c < S_HALF:
        S_fill += -S_HALF * math.log(1.0 - f_c)

    # ── k=1: Exchange screening ──────────────────────────────────────
    # Exchange from spectral dipole coupling, modulated by q_rel.
    # The k=1 cross-spectrum measures how well the dipolar structures
    # of A and B align on the polygon.
    S_exchange = 0.0
    # Use total LP for exchange modulation (not face LP)
    lp_A_exch = lp_total_A if lp_total_A >= 0 else lp_A_face
    lp_B_exch = lp_total_B if lp_total_B >= 0 else lp_B_face
    lp_mutual_exch = min(lp_A_exch, lp_B_exch)
    if n_A > 0 and n_B > 0:
        # Base exchange: PT holonomic coupling
        f_exch = 1.0 + S_p * S_HALF * q_rel
        # LP-modulated exchange: mutual LP enhances (up to half-fill)
        # or reduces (past half-fill) exchange
        if Z_A == Z_B and Z_A > 0 and lp_mutual_exch > 0:
            if lp_mutual_exch < P:
                f_exch += S_p * S_HALF * float(lp_mutual_exch) / P
            else:
                f_exch -= S_p * S_HALF * float(lp_mutual_exch - P + 1) / P
        S_exchange = -math.log(max(f_exch, 0.01))

        # Half-fill exchange penalty (k=P Nyquist mode)
        # At half-fill (n=P), atomic exchange stabilisation opposes bonding.
        if n_A == P and n_B == P and per_max >= P and bo >= 2:
            n_per3 = int(per_A >= P) + int(per_B >= P)
            if n_per3 >= 2:
                if bo >= P:
                    S_exchange += -math.log(C_p)
                else:
                    S_exchange += -(1.0 + S_HALF) * math.log(C_p)
            elif n_per3 == 1:
                S_exchange += -math.log(C_p)
        elif ((n_A == P and n_B > P) or (n_B == P and n_A > P)) and per_max >= P:
            S_exchange += -S_HALF * math.log(C_p)

        # Hund coupling for same-period donor-acceptor
        if bo <= 1 and per_A == per_B and l_min >= 1:
            np_min_v = min(n_A, n_B)
            np_max_v = max(n_A, n_B)
            if np_min_v <= P < np_max_v:
                hf_A = P - abs(n_A - P)
                hf_B = P - abs(n_B - P)
                hf_max = float(max(hf_A, hf_B))
                f_hund = 1.0 + S_p * (hf_max / P) ** 2
                S_exchange += -math.log(f_hund)

    # ── k=2: Pauli exclusion screening ──────────────────────────────
    # The quadrupole mode k=2 measures the overlap of paired electrons
    # from both atoms.  When both atoms overfill the face (n > P),
    # Pauli exclusion screens the bond.
    S_exclusion = 0.0
    if n_A > 0 and n_B > 0 and n_A + n_B > 2 * P and lp_mutual_exch < P and bo <= 1:
        np_sum = n_A + n_B
        tent = max(0.0, float(4 * P - np_sum) / (2.0 * P))
        pauli_gap = float(P - lp_mutual_exch) / float(P)
        f = max(0.1, 1.0 - pauli_gap * (1.0 - tent) * S_HALF)
        S_exclusion = -math.log(f)

    # ── LP blocking (compute f_lp early for use in S_pi) ──────────────
    lp_A_total = lp_total_A if lp_total_A >= 0 else lp_A_face
    lp_B_total = lp_total_B if lp_total_B >= 0 else lp_B_face
    lp_eff_lp = float(min(lp_A_total, lp_B_total))
    if l_min >= 1 and per_min < per_max and bo <= 1.0 and lp_eff_lp >= 2:
        if per_A <= per_B:
            lp_compact_lp, lp_diffuse_lp = lp_A_total, lp_B_total
        else:
            lp_compact_lp, lp_diffuse_lp = lp_B_total, lp_A_total
        if lp_compact_lp > lp_diffuse_lp:
            lp_eff_lp = min(lp_A_total, lp_B_total) * float(per_min) / float(per_max)
    orb_lp = 2.0 * max(per_max, 1)
    if per_min > P:
        orb_lp = min(orb_lp, 2.0 * P)
    f_lp = max(0.01, 1.0 - lp_eff_lp / orb_lp)

    # ── Multi-k: π anti-screening ────────────────────────────────────
    # π bonding = constructive interference of unpaired electron modes.
    # The DFT k>0 modes of unpaired electrons add constructively for
    # atoms with compatible orbital structures.
    # S_pi = -ln(1 + n_pi × 2t_pi / gap_sigma)
    # where t_pi = s × eps_geom × f_per × s (spin × IE × period × spin)
    #
    # Effective bond order includes dative channels:
    # bo_eff = bo + dative contributions from LP→vacancy
    S_pi = 0.0
    bo_eff = float(bo)
    if n_A > 0 and n_B > 0 and l_min >= 1:
        # Compute effective bond order (dative LP→vacancy)
        # For formal bo=1 single bonds, dative pi is handled by S_mech
        # (Term 10, not replaced by DFT). Only add dative for bo >= 2.
        vac_A = max(0, P - n_A)
        vac_B = max(0, P - n_B)
        lp_A_loc = max(0, n_A - P)
        lp_B_loc = max(0, n_B - P)
        if bo >= 2:
            n_dat_AB = min(vac_A, lp_B_loc)
            n_dat_BA = min(vac_B, lp_A_loc)
            bo_eff = float(bo) + float(n_dat_AB + n_dat_BA)

    if bo_eff > 1 and n_A > 0 and n_B > 0 and l_min >= 1:
        eps_geom = math.sqrt(max(ie_A * ie_B, 0.01)) / (S3 * 13.606 / S3)
        # Period attenuation for pi
        f_per_pi = 1.0
        if per_max >= P and bo_eff >= P:
            f_per_pi = (2.0 / float(per_max)) ** S_HALF
        # Pi orbital count from unpaired electrons
        bo_pi = min(bo_eff - 1.0, 2.0)
        if bo_pi > 0:
            n_shared = max(0, min(unp_A - 1, unp_B - 1))
            lp_A_loc = max(0, n_A - P)
            lp_B_loc = max(0, n_B - P)
            n_full_dat = 0
            if n_A < P:
                n_full_dat += min(lp_B_loc, P - n_A)
            if n_B < P:
                n_full_dat += min(lp_A_loc, P - n_B)
            n_full_dat = min(n_full_dat, max(0, int(bo_pi) - n_shared))
            n_half_dat = max(0, int(bo_pi) - n_shared - n_full_dat)
            # Half-fill gate: at np=P (half-fill), half-dative blocked
            if n_half_dat > 0 and (n_A % P == 0 or n_B % P == 0):
                if bo_eff > bo and n_shared == 0:
                    n_half_dat = min(n_half_dat, int(bo_eff - bo))
                else:
                    n_half_dat = 0
            if Z_A == Z_B:
                n_pi_eff = float(n_shared + n_full_dat + n_half_dat)
            else:
                n_pi_eff = float(n_shared + n_full_dat) + float(n_half_dat) * S_HALF
            # t_pi = s × eps_geom × f_per × s
            eps_A_loc = ie_A / 13.606
            eps_B_loc = ie_B / 13.606
            eps_geom_loc = math.sqrt(max(eps_A_loc * eps_B_loc, 1e-10))
            t_pi = S_HALF * eps_geom_loc * f_per_pi * S_HALF
            gap_pi = n_pi_eff * 2.0 * t_pi
            # Fractional dative: bo_eff may have non-integer part
            bo_frac = bo_eff - math.floor(bo_eff)
            if bo_frac > 0 and l_min >= 1:
                gap_pi += bo_frac * 2.0 * t_pi
            # bo_eff >= 3 period attenuation for heavy atoms
            if bo_eff >= 3 and per_max >= P and n_A >= 2 and n_B >= 2:
                gap_pi *= (2.0 / float(per_max)) ** S_HALF
            # gap_sigma = eps_geom × f_lp × f_per
            # The LP blocking reduces the sigma channel, making pi
            # relatively more important (larger r_pi).
            # f_per comes from period attenuation (computed in S_per section)
            f_per_loc = 1.0
            if per_max > 2:
                if Z_A == Z_B and l_max >= 2:
                    expo_f = 1.0 / (P * P2)
                elif per_max == P:
                    expo_f = 1.0 / (P * P)
                else:
                    expo_f = 1.0 / P
                f_per_loc = (2.0 / per_max) ** expo_f
            gap_sigma = eps_geom_loc * f_lp * f_per_loc
            r_pi = gap_pi / max(gap_sigma, 1e-10)
            if r_pi > 0:
                S_pi = -math.log(1.0 + r_pi)

    # ── LP blocking (S_lp from f_lp computed above) ────────────────
    S_lp = -math.log(f_lp)

    # ── Period attenuation ───────────────────────────────────────────
    # Higher periods have more diffuse orbitals → less screening.
    S_per = 0.0
    if per_max > 2 and (n_A > 0 or n_B > 0):
        if Z_A == Z_B and l_max >= 2:
            expo = 1.0 / (P * P2)
        elif per_max == P:
            expo = 1.0 / (P * P)
        else:
            expo = 1.0 / P
        f_per = (2.0 / per_max) ** expo
        if f_per < 1.0:
            S_per = -math.log(max(f_per, 1e-10))

    S_total = S_fill + S_exchange + S_exclusion + S_pi + S_lp + S_per

    terms = {
        'S_fill': S_fill,
        'S_exchange': S_exchange,
        'S_exclusion': S_exclusion,
        'S_pi': S_pi,
        'S_LP': S_lp,
        'S_per': S_per,
        'f_A': f_A,
        'f_B': f_B,
        'f_sum': f_sum,
        'lp_A': lp_A_face,
        'lp_B': lp_B_face,
        'unp_A': unp_A,
        'unp_B': unp_B,
        'q_rel': q_rel,
    }
    return S_total, terms


# ──────────────────────────────────────────────────────────────────────
#  DFT diagnostic dataclass
# ──────────────────────────────────────────────────────────────────────

@dataclass
class DFTResult:
    """Full DFT diagnostic for one face of one bond."""
    P: int                      # polygon prime
    face_name: str              # "binary", "hex", "pent"
    rho_A: List[int]            # density array atom A
    rho_B: List[int]            # density array atom B
    spectrum_A: List[complex]   # DFT spectrum atom A
    spectrum_B: List[complex]   # DFT spectrum atom B
    S_face: float               # total screening this face
    S_per_mode: List[float]     # per-mode screening
    parseval_A: Tuple[float, float]  # (lhs, rhs) Parseval check A
    parseval_B: Tuple[float, float]  # (lhs, rhs) Parseval check B
    gft_A: Tuple[float, float, float]  # (log2_N, D_KL, H) for A
    gft_B: Tuple[float, float, float]  # (log2_N, D_KL, H) for B


# ──────────────────────────────────────────────────────────────────────
#  Atomic data helpers (from screening_bond_v3)
# ──────────────────────────────────────────────────────────────────────

from ptc.periodic import period, l_of, ns_config, n_fill as n_fill_madelung


def _np_of(Z: int) -> int:
    """Number of p-electrons in valence shell."""
    l = l_of(Z)
    if l != 1:
        return 0
    per = period(Z)
    from ptc.periodic import period_start
    z0 = period_start(per)
    pos = Z - z0
    if pos < 2:
        return 0
    if per <= 3:
        return pos - 1
    if per <= 5:
        return max(0, pos - 11) if pos >= 12 else 0
    return max(0, pos - 25) if pos >= 26 else 0


def _nd_of(Z: int) -> int:
    """Number of d-electrons."""
    if l_of(Z) != 2:
        return 0
    return n_fill_madelung(Z)


def _valence_electrons(Z: int) -> int:
    """Total valence electrons."""
    _VE = {1:1,2:2,3:1,4:2,5:3,6:4,7:5,8:6,9:7,10:8,
           11:1,12:2,13:3,14:4,15:5,16:6,17:7,18:8,19:1,20:2,35:7,53:7}
    if Z in _VE:
        return _VE[Z]
    per = period(Z)
    from ptc.periodic import period_start
    z0 = period_start(per)
    return min(Z - z0 + 1, 2 * (per // 2 + 1) ** 2)


def _lp_pairs(Z: int, bo: float) -> int:
    """Number of lone pairs."""
    return max(0, _valence_electrons(Z) - bo) // 2


# ──────────────────────────────────────────────────────────────────────
#  Face-level DFT screening functions
# ──────────────────────────────────────────────────────────────────────

def _compute_face_dft(n_e_A: int, n_e_B: int, P: int,
                      lp_A: int = 0, lp_B: int = 0,
                      face_name: str = "") -> DFTResult:
    """Compute full DFT diagnostic for one face.

    Places electrons on Z/(2P)Z, computes DFT, cross-spectrum,
    Parseval check, and GFT decomposition.
    """
    rho_A = electron_density(n_e_A, P, lp=lp_A)
    rho_B = electron_density(n_e_B, P, lp=lp_B)

    spec_A = dft_spectrum(rho_A, P)
    spec_B = dft_spectrum(rho_B, P)

    S_face, S_per_mode = bond_coupling(rho_A, rho_B, P)

    pars_A = parseval_check(rho_A, spec_A, P)
    pars_B = parseval_check(rho_B, spec_B, P)

    gft_A = gft_decomposition(rho_A, P)
    gft_B = gft_decomposition(rho_B, P)

    return DFTResult(
        P=P, face_name=face_name,
        rho_A=rho_A, rho_B=rho_B,
        spectrum_A=spec_A, spectrum_B=spec_B,
        S_face=S_face, S_per_mode=S_per_mode,
        parseval_A=pars_A, parseval_B=pars_B,
        gft_A=gft_A, gft_B=gft_B,
    )


def S_binary_dft(Z_A: int, Z_B: int, bo: float = 1.0) -> DFTResult:
    """Binary face screening via DFT on Z/4Z (P₀ = 2).

    The s-electron face.  Circle has 4 positions (2 orbitals x 2 spins).

    Each atom contributes its s-electrons:
    - n_s = min(ns_config(Z), 2) for s-block
    - n_s = min(2, valence - p_electrons) otherwise
    """
    # s-electrons: always 1 or 2 from ns configuration
    ns_A = ns_config(Z_A)
    ns_B = ns_config(Z_B)

    # For s-block (l=0), use ns directly
    # For p-block (l=1), s-subshell is full (2 electrons)
    # For d-block (l=2), use ns_config
    l_A = l_of(Z_A)
    l_B = l_of(Z_B)

    if l_A >= 1:
        ns_A = 2  # s-subshell always filled for p/d/f-block
    if l_B >= 1:
        ns_B = 2

    # LP on the binary face = 0 for s1, 1 for s2
    lp_s_A = 1 if ns_A >= 2 else 0
    lp_s_B = 1 if ns_B >= 2 else 0

    return _compute_face_dft(ns_A, ns_B, P0, lp_A=lp_s_A, lp_B=lp_s_B,
                              face_name="binary")


def S_hex_dft(Z_A: int, Z_B: int, bo: float = 1.0,
              lp_A: int = 0, lp_B: int = 0) -> DFTResult:
    """Hexagonal face screening via DFT on Z/6Z (P₁ = 3).

    Replaces: S_fill_hex + S_lp + S_exch + S_halffill + S_pauli + S_pi

    Steps:
    1. Build rho_A and rho_B on Z/6Z from np_A, np_B, lp_A, lp_B
    2. Compute DFT spectra
    3. Compute bond coupling with PT weights
    4. Add Parseval normalisation: sum |rho_hat|^2 = fill_fraction

    For p-block atoms, np = number of p-electrons (1-6).
    For s-block atoms bonding to p-block, np = 0 (no p-electrons).
    For d-block atoms, ns_config gives the s-electron count.
    """
    np_A = _np_of(Z_A)
    np_B = _np_of(Z_B)
    l_A = l_of(Z_A)
    l_B = l_of(Z_B)

    # For s-block: no p-electrons, but s-electrons project onto hex face
    # via s-p hybridisation.  Map: ns -> 0 on hex face (they live on binary).
    # For d-block: ns electrons project weakly onto hex face.
    if l_A == 0:
        np_A = 0
    elif l_A == 2:
        np_A = ns_config(Z_A)  # s-electrons that leak onto hex face
    # (l_A == 1: np_A already set by _np_of)

    if l_B == 0:
        np_B = 0
    elif l_B == 2:
        np_B = ns_config(Z_B)

    # LP on the hexagonal face
    lp_hex_A = max(0, np_A - P1) if l_A == 1 else 0
    lp_hex_B = max(0, np_B - P1) if l_B == 1 else 0

    # Override with explicit LP if provided
    if lp_A > 0 and l_A == 1:
        lp_hex_A = min(lp_A, P1)
    if lp_B > 0 and l_B == 1:
        lp_hex_B = min(lp_B, P1)

    return _compute_face_dft(np_A, np_B, P1, lp_A=lp_hex_A, lp_B=lp_hex_B,
                              face_name="hex")


def S_pent_dft(Z_A: int, Z_B: int, bo: float = 1.0,
               nd_A: int = 0, nd_B: int = 0) -> DFTResult:
    """Pentagonal face screening via DFT on Z/10Z (P₂ = 5).

    The d-electron face.  Circle has 10 positions.

    For d-block atoms, nd = number of d-electrons (1-10).
    For non-d-block atoms, nd = 0.
    """
    if nd_A == 0:
        nd_A = _nd_of(Z_A)
    if nd_B == 0:
        nd_B = _nd_of(Z_B)

    # LP on pentagonal face = max(0, nd - P2) for half-fill
    lp_d_A = max(0, nd_A - P2)
    lp_d_B = max(0, nd_B - P2)

    return _compute_face_dft(nd_A, nd_B, P2, lp_A=lp_d_A, lp_B=lp_d_B,
                              face_name="pent")


# ──────────────────────────────────────────────────────────────────────
#  Full 3-face DFT screening
# ──────────────────────────────────────────────────────────────────────

@dataclass
class FullDFTResult:
    """Complete 3-face DFT screening for one bond."""
    binary: DFTResult       # Z/4Z face
    hex: DFTResult          # Z/6Z face
    pent: DFTResult         # Z/10Z face
    S_total: float          # total screening from all 3 faces
    S_binary: float         # binary face contribution
    S_hex: float            # hexagonal face contribution
    S_pent: float           # pentagonal face contribution


def full_dft_screening(Z_A: int, Z_B: int, bo: float = 1.0,
                       lp_A: int = 0, lp_B: int = 0) -> FullDFTResult:
    """Compute all three DFT face screenings for a bond.

    This gives the COMPLETE DFT-derived screening that would replace
    S_count + S_exchange + S_exclusion + S_pi in the current engine.

    The three faces are CRT-orthogonal (Principle 3: Pythagore):
    - Binary  Z/4Z:  s-electron structure
    - Hexagonal Z/6Z: p-electron structure (dominant for most bonds)
    - Pentagonal Z/10Z: d-electron structure

    Parameters
    ----------
    Z_A, Z_B : int
        Atomic numbers.
    bo : float
        Formal bond order.
    lp_A, lp_B : int
        Lone pairs (if not specified, computed from valence electrons).

    Returns
    -------
    FullDFTResult with all three faces and total screening.
    """
    if lp_A == 0:
        lp_A = _lp_pairs(Z_A, bo)
    if lp_B == 0:
        lp_B = _lp_pairs(Z_B, bo)

    res_bin = S_binary_dft(Z_A, Z_B, bo)
    res_hex = S_hex_dft(Z_A, Z_B, bo, lp_A=lp_A, lp_B=lp_B)
    res_pent = S_pent_dft(Z_A, Z_B, bo)

    S_total = res_bin.S_face + res_hex.S_face + res_pent.S_face

    return FullDFTResult(
        binary=res_bin, hex=res_hex, pent=res_pent,
        S_total=S_total,
        S_binary=res_bin.S_face,
        S_hex=res_hex.S_face,
        S_pent=res_pent.S_face,
    )


def S_hex_gft(Z_A: int, Z_B: int, bo: float = 1.0,
              lp_A: int = 0, lp_B: int = 0) -> float:
    """GFT-calibrated hexagonal face screening.

    Replaces S_count + S_exchange + S_exclusion + S_pi from the
    current engine with a unified GFT computation on Z/(2P₁)Z.

    Uses bond_screening_gft() with PT-derived mode weights.
    Includes s-block screening (which lives on the binary face
    but contributes to the total hex-equivalent screening via
    the GFT fill information on Z/(2P₀)Z).

    Returns total screening as a single float (positive = screening,
    negative = anti-screening).
    """
    l_A = l_of(Z_A)
    l_B = l_of(Z_B)
    l_min = min(l_A, l_B)
    l_max = max(l_A, l_B)
    per_A = period(Z_A)
    per_B = period(Z_B)
    per_max = max(per_A, per_B)
    per_min = min(per_A, per_B)

    # ── Compute electron counts on hexagonal face Z/(2P₁)Z ──────────
    np_A = _np_of(Z_A)
    np_B = _np_of(Z_B)

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

    # ── LP on hexagonal face (for electron placement on Z/6Z) ────────
    # The hex face LP is determined by the FILL FRACTION on Z/(2P₁)Z:
    # lp_hex = max(0, n_p - P₁).  At half-fill (n=P₁), lp_hex = 0.
    # The molecular LP (lp_A, lp_B) is used for LP blocking (S_lp)
    # but NOT for electron placement, because molecular LP includes
    # electrons that conceptually don't pair on the hex face (e.g.,
    # N in N₂ has 1 molecular LP but 0 hex face pairs at half-fill).
    lp_hex_A = max(0, n_hex_A - P1) if l_A == 1 else 0
    lp_hex_B = max(0, n_hex_B - P1) if l_B == 1 else 0

    # ── Build electron densities ─────────────────────────────────────
    rho_A = electron_density(n_hex_A, P1, lp=lp_hex_A)
    rho_B = electron_density(n_hex_B, P1, lp=lp_hex_B)

    # ── Total LP for bond blocking ───────────────────────────────────
    lp_total_A = _lp_pairs(Z_A, bo) if lp_A == 0 else lp_A
    lp_total_B = _lp_pairs(Z_B, bo) if lp_B == 0 else lp_B

    # ── Compute GFT screening on hex face ────────────────────────────
    S_hex, terms = bond_screening_gft(
        rho_A, rho_B, P1,
        bo=bo, Z_A=Z_A, Z_B=Z_B,
        lp_total_A=lp_total_A, lp_total_B=lp_total_B,
    )

    # ── s-block fill screening ───────────────────────────────────────
    # When both atoms are s-block (l=0), the hex face is empty.
    # The screening comes from the s-electron fill fraction on Z/(2P₀)Z,
    # processed through the GFT fill term.
    S_sblock = 0.0
    if l_max == 0 and per_max > 1:
        ns_A = min(2, _valence_electrons(Z_A))
        ns_B = min(2, _valence_electrons(Z_B))
        ns_sum = float(ns_A + ns_B)
        f_s = math.sqrt(min(1.0, ns_sum / (2.0 * P1)))
        if max(ns_A, ns_B) >= 2:
            from ptc.atom import IE_eV
            ie_max_s = max(IE_eV(Z_A), IE_eV(Z_B))
            q_rel_s = min(IE_eV(Z_A), IE_eV(Z_B)) / ie_max_s if ie_max_s > 0 else 1.0
            expo_s = S_HALF * min(1.0, q_rel_s / S_HALF)
            f_s *= (1.0 / max(ns_A, ns_B)) ** expo_s
        S_sblock = -math.log(max(f_s, 0.01))

    # s-block P₁ screening for heavy s-block atoms
    if l_max == 0 and per_max >= P1:
        _AE4p = {20, 38, 56, 88}
        if not ({Z_A, Z_B} & _AE4p):
            S_sblock += -math.log(C3)

    # ── Period attenuation (for s-block and s+p where hex face is empty)
    # bond_screening_gft includes S_per when hex face has electrons,
    # but for s-block (both atoms have 0 p-electrons) or s+p (one atom
    # has 0), the period attenuation needs to be added here.
    S_per_extra = 0.0
    if n_hex_A == 0 or n_hex_B == 0:
        if per_max > 2:
            if Z_A == Z_B and l_max >= 2:
                expo_per = 1.0 / (P1 * P2)
            elif per_max == P1:
                expo_per = 1.0 / (P1 * P1)
            else:
                expo_per = 1.0 / P1
            f_per_val = (2.0 / per_max) ** expo_per
            # 5d Dirac contraction
            if f_per_val < 1.0 and l_max >= 2 and per_max >= 6:
                Zh = Z_A if per_A >= 6 and l_A >= 2 else (Z_B if per_B >= 6 and l_B >= 2 else 0)
                if Zh > 0:
                    za2 = (Zh * AEM) ** 2
                    if za2 < 1:
                        f_per_val /= math.sqrt(1.0 - za2)
            S_per_extra = -math.log(max(f_per_val, 1e-10))

    # ── s+p holonomic screening ──────────────────────────────────────
    # NOT included here: S_holo is Term 5 in the current engine,
    # computed separately in screening_bond_v3._S_holo().
    # S_hex_gft replaces only Terms 2+3+4+7 (S_count+S_exchange+S_exclusion+S_pi).
    S_holo_sp = 0.0

    # ── NNLO corrections from T³ spectral structure ──────────────────
    # These are the higher-order corrections that the current engine
    # distributes across S_count, S_exchange, S_exclusion.
    S_nnlo = 0.0

    # H₂ ghost VP screening
    if per_max == 1 and l_max == 0:
        S_nnlo += S3 * S3 / P1

    # Homonuclear bo ≥ 2 k=1 mode
    if (Z_A == Z_B and l_min >= 1 and bo >= 2.0
            and 2 <= np_A <= 2 * P1 - 2 and np_A != P1):
        f_bo = float(bo - 1) / float(bo)
        f_per_nnlo = (2.0 / float(per_max)) ** (1.0 / P1) if per_max > 2 else 1.0
        S_nnlo += S3 * S3 / P1 * f_bo * f_per_nnlo

    # Cross-period half-fill triple bond screening
    if (bo >= 3 and Z_A != Z_B and per_A != per_B
            and l_min >= 1 and l_max <= 1
            and np_A == P1 and np_B == P1):
        S_nnlo += S3 * S3

    # H radial mismatch screening
    if per_min == 1 and per_max in (P1, P2) and l_max <= 1 and min(Z_A, Z_B) == 1:
        Z_heavy = Z_A if Z_A > 1 else Z_B
        l_heavy = l_of(Z_heavy)
        np_heavy = _np_of(Z_heavy) if l_heavy >= 1 else 0
        if np_heavy >= P1:
            if per_max == P1:
                S_nnlo += S3 / P1
            else:
                S_nnlo += S3 / P1 * (float(P1) / float(per_max)) ** S_HALF

    # Compact per=2 homonuclear underfill vacancy relief
    if Z_A == Z_B and per_max == 2 and l_min >= 1 and np_A < P1:
        np_c = min(float(np_A + np_B), 2.0 * P1)
        S_fill_hex_val = -0.5 * math.log(np_c / (2.0 * P1))
        S_nnlo -= S_fill_hex_val * np_c / float(P1 * P1)

    # Per=2 homonuclear triple bond compact Pauli
    if Z_A == Z_B and per_max == 2 and np_A < P1 and bo >= 3 and l_min >= 1:
        S_nnlo += S3 * float(P1 - np_A) / P1

    # Heavy homonuclear p-block core screening
    if Z_A == Z_B and per_max >= P2 and l_min >= 1 and l_max <= 1:
        za_param = (float(Z_A) * AEM) ** (2.0 / P1)
        S_nnlo += D3 * za_param

    # Cross-period LP penetration Pauli
    if (l_min >= 1 and bo <= 1 and per_A != per_B and l_max <= 1
            and Z_A != Z_B and np_A > P1 and np_B > P1):
        if per_A < per_B:
            np_compact, np_diffuse = np_A, np_B
        else:
            np_compact, np_diffuse = np_B, np_A
        if np_compact < np_diffuse:
            per_max_loc_cp = max(per_A, per_B)
            f_lp_cp = float(min(np_A, np_B) - P1) / P1
            f_per_cp = (2.0 / float(per_max_loc_cp)) ** (1.0 / P1)
            S_nnlo += D3 * f_lp_cp * f_per_cp

    # Dual over-half-fill Pauli for compact per=2
    if (l_min >= 1 and bo <= 1 and per_A == per_B and per_max <= 2
            and Z_A != Z_B and np_A > P1 and np_B > P1):
        np_sum_of = np_A + np_B
        S_nnlo += S3 * max(0.0, float(np_sum_of - 2 * P1)) / (2.0 * P1)

    # Cross-period half-fill Pauli
    if l_min >= 1 and bo <= 1 and per_A != per_B:
        np_min_v = min(np_A, np_B)
        np_max_v = max(np_A, np_B)
        if np_min_v == P1 and np_max_v > P1:
            per_hf = per_A if np_A == P1 else per_B
            per_min_loc = min(per_A, per_B)
            per_max_loc = max(per_A, per_B)
            if per_hf == per_min_loc:
                S_nnlo += S3 * S_HALF * float(per_min_loc) / float(per_max_loc)

    # Homonuclear halogen LP core overlap
    if (Z_A == Z_B and P1 <= per_max < P2
            and l_min >= 1 and l_max <= 1
            and min(np_A, np_B) > P1 and min(lp_A, lp_B) >= P1
            and bo <= 1):
        za_param_c1 = (float(Z_A) * AEM) ** (2.0 / P1)
        S_nnlo += D3 * za_param_c1

    # Heteronuclear halogen LP overlap
    _HALOGENS_C5 = frozenset({9, 17, 35, 53})
    if (Z_A != Z_B and Z_A in _HALOGENS_C5 and Z_B in _HALOGENS_C5
            and per_min == P1 and per_max <= P2
            and bo <= 1 and min(np_A, np_B) > P1):
        za_A_c5 = (float(Z_A) * AEM) ** (2.0 / P1)
        za_B_c5 = (float(Z_B) * AEM) ** (2.0 / P1)
        S_nnlo += D3 * math.sqrt(za_A_c5 * za_B_c5) * S_HALF

    # Pi Pauli for per=2 homonuclear overfill
    if (l_min >= 1 and bo >= 2 and per_max == 2 and per_A == per_B
            and Z_A == Z_B and np_A > P1 and np_B > P1):
        np_excess = float(np_A + np_B - 2 * P1) / (2.0 * P1)
        S_nnlo += D3 * np_excess * float(min(bo - 1, 2))

    # Pi Pauli for per >= P1 homonuclear overfill
    if (l_min >= 1 and bo >= 2 and per_A == per_B and per_max >= P1
            and Z_A == Z_B and np_A > P1 and np_B > P1
            and l_max <= 1):
        np_excess_h = float(np_A + np_B - 2 * P1) / (2.0 * P1)
        f_per_h = (2.0 / float(per_max)) ** (1.0 / P1)
        S_nnlo += D3 * np_excess_h * float(min(bo - 1, 2)) * f_per_h

    # Cross-period NNLO corrections for bo >= 2
    if (bo >= 2 and Z_A != Z_B and per_A != per_B
            and l_min >= 1 and l_max <= 1
            and np_A != P1 and np_B != P1):
        np_min_k1 = min(np_A, np_B)
        np_max_k1 = max(np_A, np_B)
        if np_min_k1 > P1 and np_max_k1 > P1:
            lp_min_k1 = float(min(np_A - P1, np_B - P1)) / P1
            S_nnlo -= D3 * S_HALF * lp_min_k1
        elif np_min_k1 <= 1 and np_max_k1 > P1:
            ie_A_loc = IE_eV(Z_A) if Z_A > 0 else 13.6
            ie_B_loc = IE_eV(Z_B) if Z_B > 0 else 13.6
            q_rel_k1 = min(ie_A_loc, ie_B_loc) / max(ie_A_loc, ie_B_loc) if max(ie_A_loc, ie_B_loc) > 0 else 1.0
            if q_rel_k1 < S_HALF:
                vac_k1 = float(P1 - np_min_k1) / P1
                S_nnlo += D3 * S_HALF * vac_k1

    # Cross-period underfill radial screening for bo >= 3
    if (bo >= 3 and Z_A != Z_B and per_A != per_B
            and l_min >= 1 and l_max <= 1
            and min(np_A, np_B) < P1):
        f_pgap = 1.0 - float(min(per_A, per_B)) / float(max(per_A, per_B))
        f_under = float(P1 - min(np_A, np_B)) / P1
        S_nnlo += S3 * S3 * f_pgap * f_under

    # Same-period triple bond cross-fill Pauli
    if (bo >= 3 and Z_A != Z_B and per_A == per_B
            and l_min >= 1 and l_max <= 1
            and min(np_A, np_B) >= 2):
        np_min_cf = min(np_A, np_B)
        S_nnlo += S3 * S3 * S_HALF * math.sin(
            math.pi * float(np_min_cf) / (2.0 * P1)) ** 2

    # s+p dative from S_holo — NOT included: part of Term 5 (S_holo)
    from ptc.atom import IE_eV

    # s+p fill blocking for per=3
    if l_min == 0 and l_max == 1 and bo == 1.0:
        if l_A == 0:
            per_s_sp2, per_p_sp2, Z_p_sp2 = per_A, per_B, Z_B
        else:
            per_s_sp2, per_p_sp2, Z_p_sp2 = per_B, per_A, Z_A
        np_p_sp2 = _np_of(Z_p_sp2) if l_of(Z_p_sp2) >= 1 else 0
        if per_s_sp2 == P1 and per_p_sp2 == P1 and np_p_sp2 > P1:
            lp_p_sp2 = _lp_pairs(Z_p_sp2, bo)
            x_sp2 = S3 * float(lp_p_sp2) / (2.0 * P1)
            if x_sp2 < 1.0:
                S_nnlo += -math.log(1.0 - x_sp2)

    # C2: per=2 underfilled p-block + H compact overcount
    if per_min == 1 and per_max == 2 and l_max == 1 and l_min == 0 and bo == 1.0:
        Z_heavy_c2 = Z_A if l_A == 1 else Z_B
        np_heavy_c2 = _np_of(Z_heavy_c2) if l_of(Z_heavy_c2) >= 1 else 0
        if 0 < np_heavy_c2 < P1:
            S_nnlo += D3 * S_HALF * float(P1 - np_heavy_c2) / P1

    # C4: s-block per=P₁ + H sigma radial mismatch
    if l_max == 0 and per_min == 1 and per_max == P1 and bo == 1.0:
        S_nnlo += D3 / P1

    # C6: AE s² + H sp-hybridisation relief
    if l_max == 0 and per_min == 1 and per_max == 2 and bo == 1.0:
        Z_heavy_c6 = Z_A if Z_A > 1 else Z_B
        ns_heavy_c6 = ns_config(Z_heavy_c6)
        if ns_heavy_c6 >= 2:
            from ptc.atom import IE_eV as _IE_eV
            _IE2_AE = {4: 18.21, 12: 15.04, 20: 11.87, 38: 11.03, 56: 10.00, 88: 10.15}
            ie2_c6 = _IE2_AE.get(Z_heavy_c6, 0.0)
            ie1_c6 = IE_eV(Z_A) if Z_A == Z_heavy_c6 else IE_eV(Z_B)
            if ie2_c6 > 0:
                S_nnlo -= D3 * ie1_c6 / ie2_c6

    # NNLO-4: alkali + H core VP screening
    if l_max == 0 and per_min == 1 and per_max >= P1 and bo == 1.0:
        ie_A_loc = IE_eV(Z_A) if Z_A > 0 else 13.6
        ie_B_loc = IE_eV(Z_B) if Z_B > 0 else 13.6
        Z_cat_nnlo = Z_A if ie_A_loc < ie_B_loc else Z_B
        ns_cat_nnlo = ns_config(Z_cat_nnlo)
        if ns_cat_nnlo == 1:
            S_nnlo += S3 * D3 * (2.0 / float(per_max)) ** S_HALF

    # C7: part of S_holo (Term 5) — NOT included here

    # d-block cap adjustment
    if l_max >= 2 and per_min <= 1:
        S_nnlo += 0.5 * math.log(P2 / P1)

    # d-block sigma radial mismatch
    from ptc.periodic import _n_fill_aufbau
    nd_A = _nd_of(Z_A)
    nd_B = _nd_of(Z_B)
    sd_A = l_A == 2 and P2 < nd_A < 2 * P2
    sd_B = l_B == 2 and P2 < nd_B < 2 * P2
    if l_max >= 2 and per_min <= 1 and lp_A + lp_B == 0:
        if sd_A or sd_B:
            f_sd_rm = max(
                (float(nd_A - P2) / P2) if sd_A else 0.0,
                (float(nd_B - P2) / P2) if sd_B else 0.0,
            )
            f_srm = 1.0 - S5 * (1.0 - f_sd_rm ** 2)
        else:
            f_srm = 1.0 - S5
        S_nnlo += -math.log(max(f_srm, 0.01))

    # Same-period het LP exchange — M16, part of S_mech (Term 10) → NOT included

    return S_hex + S_sblock + S_per_extra + S_nnlo


# ──────────────────────────────────────────────────────────────────────
#  S_DFT -> S_screening conversion
# ──────────────────────────────────────────────────────────────────────
#
#  The DFT cross-spectrum gives a RAW screening value.  To connect it
#  to the exponential D₀ = cap × exp(-S), we need to map the DFT
#  output onto the screening scale used by the current engine.
#
#  The mapping uses the GFT identity:
#    S_fill = D_KL(rho || U) = KL divergence from uniform
#    S_exch = -sum_{k>0} Re[cross(k)] * w(k)
#
#  The total DFT screening per face is:
#    S_face_DFT = S_fill_A + S_fill_B + S_cross
#
#  where S_fill captures the COUNTING information (how full is the shell)
#  and S_cross captures the CORRELATION information (how do A and B relate).

def dft_screening_to_S(dft_result: DFTResult) -> dict:
    """Convert DFT result to screening decomposition.

    Returns dict with:
    - S_fill: fill-based screening (D_KL component)
    - S_cross: cross-correlation screening
    - S_total: sum
    - D_KL_A, D_KL_B: individual KL divergences
    - H_A, H_B: individual entropies
    """
    _, D_KL_A, H_A = dft_result.gft_A
    _, D_KL_B, H_B = dft_result.gft_B

    # Fill screening = geometric mean of D_KL values
    # PT: the bond "sees" both atoms' fill fractions
    S_fill = 0.5 * (D_KL_A + D_KL_B) * S_HALF  # scaled to S units

    # Cross screening = sum of per-mode cross terms (excluding k=0)
    S_cross = sum(dft_result.S_per_mode[1:])  # k>0 modes only

    return {
        'S_fill': S_fill,
        'S_cross': S_cross,
        'S_total': S_fill + S_cross,
        'D_KL_A': D_KL_A,
        'D_KL_B': D_KL_B,
        'H_A': H_A,
        'H_B': H_B,
    }


# ──────────────────────────────────────────────────────────────────────
#  Diagnostic printing
# ──────────────────────────────────────────────────────────────────────

def print_dft_diagnostic(Z_A: int, Z_B: int, bo: float = 1.0,
                         lp_A: int = 0, lp_B: int = 0) -> FullDFTResult:
    """Compute and print full DFT diagnostic for a bond.

    Useful for interactive exploration and debugging.
    """
    from ptc.data.experimental import SYMBOLS
    sym_A = SYMBOLS.get(Z_A, f"Z{Z_A}")
    sym_B = SYMBOLS.get(Z_B, f"Z{Z_B}")

    result = full_dft_screening(Z_A, Z_B, bo, lp_A, lp_B)

    print(f"\n{'='*60}")
    print(f"  DFT Polygon Diagnostic: {sym_A}-{sym_B} (bo={bo})")
    print(f"{'='*60}")

    for face in [result.binary, result.hex, result.pent]:
        print(f"\n--- {face.face_name.upper()} face (P={face.P}, Z/{2*face.P}Z) ---")
        print(f"  rho_A = {face.rho_A}  (n_e = {sum(face.rho_A)})")
        print(f"  rho_B = {face.rho_B}  (n_e = {sum(face.rho_B)})")
        print(f"  Spectrum A: {['({:.4f},{:.4f})'.format(c.real, c.imag) for c in face.spectrum_A]}")
        print(f"  Spectrum B: {['({:.4f},{:.4f})'.format(c.real, c.imag) for c in face.spectrum_B]}")
        print(f"  Power A:    {['{:.6f}'.format(abs(c)**2) for c in face.spectrum_A]}")
        print(f"  Power B:    {['{:.6f}'.format(abs(c)**2) for c in face.spectrum_B]}")
        print(f"  Parseval A: LHS={face.parseval_A[0]:.6f}, RHS={face.parseval_A[1]:.6f}")
        print(f"  Parseval B: LHS={face.parseval_B[0]:.6f}, RHS={face.parseval_B[1]:.6f}")
        print(f"  GFT A: log2({2*face.P})={face.gft_A[0]:.4f}, D_KL={face.gft_A[1]:.4f}, H={face.gft_A[2]:.4f}")
        print(f"  GFT B: log2({2*face.P})={face.gft_B[0]:.4f}, D_KL={face.gft_B[1]:.4f}, H={face.gft_B[2]:.4f}")
        print(f"  S_per_mode: {['{:.6f}'.format(s) for s in face.S_per_mode]}")
        print(f"  S_face = {face.S_face:.6f}")

    print(f"\n--- TOTAL ---")
    print(f"  S_binary = {result.S_binary:.6f}")
    print(f"  S_hex    = {result.S_hex:.6f}")
    print(f"  S_pent   = {result.S_pent:.6f}")
    print(f"  S_total  = {result.S_total:.6f}")
    print(f"{'='*60}\n")

    return result


# ── Z/30Z stubs (CRT composite torus, used by transfer_matrix.py) ──

def electron_density_Z30(n_hex, n_pent, lp_hex=0, lp_pent=0):
    """Electron density on Z/30Z = Z/2Z x Z/3Z x Z/5Z."""
    rho = [0] * 30
    for i in range(min(n_hex, 6)):
        rho[(i * 5) % 30] += 1
    for i in range(min(n_pent, 10)):
        rho[(i * 3) % 30] += 1
    return rho


def dft_spectrum_Z30(rho):
    """DFT on Z/30Z."""
    import cmath
    N = len(rho)
    return [sum(rho[r] * cmath.exp(-2j * cmath.pi * k * r / N) for r in range(N)) / N for k in range(N)]


def vertex_cross_spectrum_Z30(rho_A, rho_B, **kw):
    """Cross-spectrum on Z/30Z."""
    sA, sB = dft_spectrum_Z30(rho_A), dft_spectrum_Z30(rho_B)
    return [a * b.conjugate() for a, b in zip(sA, sB)]
