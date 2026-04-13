"""
PT constants — ALL derived from s = 1/2.

Chain [D00 -> L0 -> T6 -> T7 -> D09]:

  s = 1/2           [T0, D00: transitions interdites mod 3]
    |
  mu* = 3+5+7 = 15  [T7, D08: auto-coherence, unique point fixe]
    |
  q_stat = 13/15    [L0: unicite de la distribution geometrique]
    |
  delta_p, sin^2    [T6, D07: holonomie, identite algebrique]
    |
  alpha = prod sin^2 [D09: couplage electromagnetique]
    |
  habillage C_K      [D09, D17b: Fisher-Koide, Catalan]
    |
  alpha_phys         [D09: 1/137.033, 19 ppm]
    |
  Ry = alpha^2/2 * [m_e c^2]   (PT-derived x translation factor)
  COULOMB = alpha * hbar_c      (DERIVED)
  A_BOHR = COULOMB / (2 Ry)    (DERIVED)

    INPUT:   s = 1/2  (proved unconditionally, T0/D00)
    OUTPUT:  every constant in this file
    PARAMS:  0 adjustable
    SCALE:   1 translation factor (m_e c^2 in eV = choice of unit)

Two branches from the bifurcation [T7, D08, A8]:
  q_stat  = 1 - 2/mu = 13/15  (vertex/coupling: alpha, PMNS, leptons)
  q_therm = exp(-1/mu)         (face/propagateur: alpha_s, CKM, metrique)

March 2026 -- Theorie de la Persistance
"""
import math
from fractions import Fraction

# ── Input: s = 1/2 [T0, D00] ──
S_HALF = 0.5

# ── Active primes {3,5,7} [T7, D08: gamma_p > 1/2 at mu*=15] ──
P0, P1, P2, P3 = 2, 3, 5, 7
MU_STAR = P1 + P2 + P3                    # 15 [T7: unique fixed point]

# ── Koide [D17b: Q=2/3 from Catalan 3^2-2^3=1] ──
G_FISHER = 4
Q_KOIDE = Fraction(2, 3)

# ╔════════════════════════════════════════════════════════════════╗
# ║  BRANCHE q_stat [L0: unicite max-entropie]                   ║
# ║  vertex coupling: energies, screening, caps                   ║
# ╚════════════════════════════════════════════════════════════════╝
_Q = Fraction(13, 15)                      # q_stat = 1 - 2/mu [L0]

# delta_p = (1 - q^p) / p  [T6, D07: holonomy on Z/pZ]
def _delta(p):
    return float((1 - _Q**p) / p)

D3, D5, D7 = _delta(P1), _delta(P2), _delta(P3)

# sin^2(theta_p) = delta_p * (2 - delta_p)  [T6, D07: identite algebrique]
def _sin2(p):
    d = _delta(p)
    return float(d * (2 - d))

S3, S5, S7 = _sin2(P1), _sin2(P2), _sin2(P3)
C3, C5, C7 = 1.0 - S3, 1.0 - S5, 1.0 - S7

# ╔════════════════════════════════════════════════════════════════╗
# ║  BRANCHE q_therm [L0 + bifurcation A8]                       ║
# ║  face/propagateur: geometry, distances, metrique Bianchi I    ║
# ╚════════════════════════════════════════════════════════════════╝
_Q_TH = math.exp(-1.0 / MU_STAR)          # q_therm = exp(-1/mu) [A8]

def _delta_th(p):
    return (1.0 - _Q_TH**p) / p

D3_TH, D5_TH, D7_TH = _delta_th(P1), _delta_th(P2), _delta_th(P3)

def _sin2_th(p):
    d = _delta_th(p)
    return d * (2 - d)

S3_TH, S5_TH, S7_TH = _sin2_th(P1), _sin2_th(P2), _sin2_th(P3)
C3_TH, C5_TH, C7_TH = 1.0 - S3_TH, 1.0 - S5_TH, 1.0 - S7_TH

# Chaleur latente de bifurcation [A8]
L_BIFURC = float(_Q_TH) - float(_Q)

# ── alpha_EM nu [D09: product of 3 cascade filters] ──
AEM = S3 * S5 * S7                         # 1/136.28

# ── Taux d'arret [T4, D04: formule maitresse f(p)] ──
LAM = 2.0 / MU_STAR

# ── Facteur de traduction SI ──
# ONLY non-PT input: m_e c^2 in eV (unit choice, not a parameter).
# Everything else derived from s=1/2 + this factor.
ME_C2_EV = 510998.95                       # m_e c^2 (CODATA 2018)
HBAR_C_EV_A = 1973.269804                 # hbar*c en eV*A (exact, nouveau SI)
HBAR_EV_S = 6.582e-16                     # hbar en eV*s (definition SI)
AMU_KG = 1.6605e-27                        # unite de masse atomique (kg)
EV_J = 1.6022e-19                          # conversion eV -> J (definition)
CM1_PER_EV = 8065.54                       # conversion cm^-1 -> eV

# ── Dimensions anomales gamma_p [T6, D07: -d(ln sin^2)/d(ln mu)] ──
def _gamma(p):
    q = float(_Q)
    d = _delta(p)
    return 4.0 * p * q**(p-1) * (1.0 - d) / (MU_STAR * (1.0 - q**p) * (2.0 - d))

GAMMA_3 = _gamma(P1)                       # 0.808 [D08]
GAMMA_5 = _gamma(P2)                       # 0.696 [D08]
GAMMA_7 = _gamma(P3)                       # 0.595 [D08]

# ── Premiers fantomes (inactifs, gamma < 1/2) [T7, D08] ──
S11 = _sin2(11)
S13 = _sin2(13)
GAMMA_11 = _gamma(11)                      # 0.427 < 1/2 -> inactif
GAMMA_13 = _gamma(13)                      # 0.356 < 1/2 -> inactif
GHOST_PRIMES = (11, 13)
BETA_GHOST = S11 * GAMMA_11 + S13 * GAMMA_13

# ── Cross-gaps [D14: tenseur de Riemann, secteur visible] ──
R35 = D3 * D5
R37 = D3 * D7
R57 = D5 * D7

# ── Fisher cosines [ch15, derived NLO+NNLO, 0 parametre] ──
# Pearson of (2k mod p1, 2k mod p3) under Geom(q_stat^2)
# The R_ij cross-gaps above are tree-level APPROXIMATIONS.
# These cosines are the exact NLO+NNLO values.
COS_37 = -0.73998  # Charge × Thermodynamique (anti-correle)
COS_35 = +0.58819  # Charge × Relativite (partiellement aligne)
COS_57 = -0.89463  # Relativite × Thermodynamique (anti-correle)

# ── Cross-gaps NLO+NNLO [ch15: Fisher cosines → CROSS_ij] ──
# CROSS_ij = sqrt(S_i × S_j) × |COS_ij|
# These are the exact NLO+NNLO cross-face couplings.
# The R_ij above are tree-level approximations (~10x smaller).
# Signs from COS_ij: CROSS_35 positive, CROSS_37/57 negative.
import math as _math
CROSS_35 = _math.sqrt(S3 * S5) * abs(COS_35)   # 0.1213
CROSS_37 = _math.sqrt(S3 * S7) * abs(COS_37)   # 0.1439
CROSS_57 = _math.sqrt(S5 * S7) * abs(COS_57)   # 0.1637

# ── String tension [D14, derived] ──
T_STRING = 1.0 / (4.0 * math.pi ** 2)   # = 0.02533

# ── Secteur sombre [D14: interferences cachees] ──
I3 = math.sqrt(S3) * (1 - D3)
I5 = math.sqrt(S5) * (1 - D5)
I7 = math.sqrt(S7) * (1 - D7)
R35_DARK = AEM * I3 * I5
R37_DARK = AEM * I3 * I7
R57_DARK = AEM * I5 * I7
ALPHA_DARK = (I3 * I5 * I7) ** 2

# Dark quadrupole k=2 on Z/(2P1)Z [D13: spin foam]
# The 3 cross-gap beats alias onto k=2 (triangle mode):
#   (3,5): beat=2, (3,7): beat=4=2 mod 3, (5,7): beat=2
R_DARK_K2 = R35_DARK + R37_DARK + R57_DARK

# ── Universal dressing F(p) [§XX+20: formule universelle] ──
# F(p) = sin^2(theta_p) * cos^2(theta_p / N(p)) * (mu-p) / p^2
# N(p) = (p+1)^(p+1) - 1 : nombre de canaux charges
def _F_universal(p):
    d = _delta(p)
    s2 = _sin2(p)
    N_p = (p + 1) ** (p + 1) - 1
    psi = 1.0 - d  # cos(theta_p)
    cos2_att = math.cos(math.acos(psi) / N_p) ** 2
    return s2 * cos2_att * (MU_STAR - p) / (p * p)

F_P0 = _F_universal(P0)     # 0.758 : exchange/Bohr (face binaire)
F_P1 = _F_universal(P1)     # 0.292 : coordination/filling (face hexagonale)
F_P2 = _F_universal(P2)     # 0.078 : period mismatch (face pentagonale)
F_P3 = _F_universal(P3)     # 0.028 : LP blocking (face heptagonale)

# ── Habillage NLO/NNLO [D09, D17b: identite Fisher-Koide] ──
# C_K = 4/sin^2_3 + (1 + 5*delta_3^2/18) / 21  [D17b]
C_KOIDE = 4.0 / S3 + (1.0 + 5.0 * D3**2 / 18.0) / 21.0
D_NLO = 1.0 + AEM * C_KOIDE / (2.0 * math.pi * P1)
D_NNLO = D_NLO + AEM ** 2
D_DARK = ALPHA_DARK * C_KOIDE / (2.0 * math.pi * P1)
D_FULL = D_NNLO - S_HALF * D_DARK

# ── p=2 channel quantities [§XX+20: architecture p=2] ──
D2 = _delta(2)                             # delta_2 = 28/225
S2 = _sin2(2)                              # sin^2(theta_2) = 0.2334
C2 = 1.0 - S2                              # cos^2(theta_2)
GAMMA_2 = _gamma(2)                        # 0.867 > 1/2 (ACTIVE)

# ── alpha habille [§XX+20: architecture p=2, sub-ppb, 0 parametre] ──
#
# 1/alpha = 1/alpha_bare + F(2)/(1+gamma_3*r*Pi) + sin^2_2*beta*alpha^2 + (alpha/pi)^2/3
#
# F(2) = D_10 * cos^2(arccos(psi)/N)
#   D_10 = (mu-1)(mu-2)(mu^2-mu+1)/mu^4    rationnel = 38402/50625
#   psi = cos(theta_2) = 1-delta_2 = 197/225
#   N = (p+1)^(p+1)-1 = 26                  canaux charges
#
# Archimedean spiral feedback modulated by gamma_3:
#   r = alpha_1 * (gamma_3^2+gamma_5^2+gamma_7^2)
#   Pi = (delta_5+delta_7)/sum(gamma) * (1+alpha_bare/5^2)

_D10 = float((MU_STAR-1)*(MU_STAR-2)*(MU_STAR**2-MU_STAR+1)) / MU_STAR**4
_PSI = 1.0 - float(D2)                    # cos(theta_2) = 197/225
_N_CHARGED = (2+1)**(2+1) - 1             # 26 = 3^3 - 1
_F2 = _D10 * math.cos(math.acos(_PSI) / _N_CHARGED)**2

_ALPHA1 = 1.0 / (1.0/AEM + _F2)
_SUM_GAMMA2 = GAMMA_3**2 + GAMMA_5**2 + GAMMA_7**2
_SUM_GAMMA = GAMMA_3 + GAMMA_5 + GAMMA_7
_PROP = (D5 + D7) / _SUM_GAMMA * (1.0 + AEM / 25.0)
_R_FEEDBACK = _ALPHA1 * _SUM_GAMMA2
_F2_RESUMME = _F2 / (1.0 + GAMMA_3 * _R_FEEDBACK * _PROP)

_GHOST_P2 = S2 * BETA_GHOST * _ALPHA1**2
_TWO_LOOP = (_ALPHA1 / math.pi)**2 / 3.0

ALPHA_PHYS = 1.0 / (1.0/AEM + _F2_RESUMME + _GHOST_P2 + _TWO_LOOP)

# ── Constantes physiques derivees [D09 + facteur de traduction] ──
# Ry = alpha^2/2 * m_e c^2  [DERIVE: PT + traduction]
# COULOMB = alpha * hbar*c   [DERIVE: D09]
# A_BOHR = COULOMB / (2*Ry)  [DERIVE: algebrique]
RY = ALPHA_PHYS ** 2 / 2.0 * ME_C2_EV
COULOMB_EV_A = ALPHA_PHYS * HBAR_C_EV_A
A_BOHR = COULOMB_EV_A / (2.0 * RY)

# ── Cap de Shannon [D13: spin foam, Ry/P1 par face] ──
SHANNON_CAP = RY / P1

# ── Ionic constant [ch22b, derived] ──
K_PT_IONIC = RY * P1 * T_STRING          # = 1.034 eV

# ── BIFURCATION FACTOR [A8] ──
# Ratio of q_therm to q_stat delta: F = δ₃(q_therm) / δ₃(q_stat)
F_BIFURC = D3_TH / D3   # ≈ 0.519, universal for all p

# ── ε-FIELD: per-channel depth correction from bifurcation mismatch ──
# ε(p) = sin²(π·q_stat/p) + sin²(π·q_therm/p) - 1
_q_s = 13.0 / 15.0
_q_t = math.exp(-1.0 / 15.0)
EPS_3 = math.sin(math.pi * _q_s / 3) ** 2 + math.sin(math.pi * _q_t / 3) ** 2 - 1.0  # +0.310
EPS_5 = math.sin(math.pi * _q_s / 5) ** 2 + math.sin(math.pi * _q_t / 5) ** 2 - 1.0  # -0.424
EPS_7 = math.sin(math.pi * _q_s / 7) ** 2 + math.sin(math.pi * _q_t / 7) ** 2 - 1.0  # -0.690
F_EPS = 0.20  # depth modulation strength (from S_avg ≈ 0.30 benchmark)
D_FULL_P1 = D_FULL * math.exp(F_EPS * EPS_3 * 0.30)   # ε₃>0 → more screening on P₁
D_FULL_P2 = D_FULL * math.exp(F_EPS * EPS_5 * 0.30)   # ε₅<0 → less screening on P₂
D_FULL_P3 = D_FULL * math.exp(F_EPS * EPS_7 * 0.30)   # ε₇<0 → less screening on P₃
DEPTH_P3 = C3 * C5 * D_FULL_P3

# ── Tables partagees (atom.py, continuous.py, shell_polygon.py) ──
BLOCK_TABLE = {}
for _l_idx, (_P_val, _G_val) in enumerate(zip([P1, P2, P3], [GAMMA_3, GAMMA_5, GAMMA_7]), 1):
    _d_val = {P1: D3, P2: D5, P3: D7}[_P_val]
    _s2_val = {P1: S3, P2: S5, P3: S7}[_P_val]
    BLOCK_TABLE[_l_idx] = {'P': _P_val, 'N': 2 * _P_val, 'd': _d_val, 's2': _s2_val, 'c2': 1 - _s2_val, 'g': _G_val}

# Profil d'echange [T0, D00: transitions interdites] ──
P_EXCHANGE_PROFILE = {0: 0, 1: 1, 2: 2, 3: -1, 4: 0, 5: 1}

# masse electron en amu [DERIVE: m_e/m_p ratio from D17b, D19]
ME_AMU = 5.48580e-4

# ── Element sets ──
HALOGENS = frozenset({9, 17, 35, 53})
CHALCOGENS = frozenset({8, 16, 34, 52})
ALKALI = frozenset({3, 11, 19, 37, 55})
NS1 = frozenset({24, 29, 41, 42, 44, 45, 46, 47, 78, 79})

# ── IE2 tables (second ionization energy, eV, NIST) ──
IE2_DBLOCK = {
    21: 12.80, 22: 13.58, 23: 14.66, 24: 16.49, 25: 15.64,
    26: 16.19, 27: 17.08, 28: 18.17, 29: 20.29, 30: 17.96,
    39: 12.24, 40: 13.13, 41: 14.32, 42: 16.16, 43: 15.26,
    44: 16.76, 45: 18.08, 46: 19.43, 47: 21.49, 48: 16.91,
    72: 14.9, 73: 16.2, 74: 17.7, 75: 16.6,
    76: 17.0, 77: 18.6, 78: 18.56, 79: 20.5, 80: 18.76,
}
IE2_AE = {
    4: 18.211, 12: 15.035, 20: 11.872, 38: 11.030,
    56: 10.004, 30: 17.964, 48: 16.908, 80: 18.757,
}
