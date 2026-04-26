"""General STO-STO overlaps via prolate spheroidal quadrature + Slater-Koster.

Phase A continuation of `atomic_basis.py`.

Strategy
========
Two pieces:

1. AXIAL Slater-Koster integrals (sigma, pi, delta) computed by 2D
   Gauss-Laguerre x Gauss-Legendre quadrature in prolate spheroidal
   coordinates. These depend only on (n_a, l_a, n_b, l_b, |m|, zeta_a,
   zeta_b, R) and are deterministic, parameter-free numerical integrals
   (no fit). For l = 0, validated to better than 1e-10 against the
   analytic 1s-1s closed form already in `atomic_basis._overlap_1s_1s`.

2. GEOMETRIC Slater-Koster rotation: real-spherical-harmonic orbitals
   in the molecular frame are projected onto the (sigma, pi, delta)
   basis defined w.r.t. the AB-axis using direction cosines. This
   reduces every overlap to a linear combination of the AXIAL integrals.
   Implemented for s and p (l <= 1) here; d-d / s-d / p-d combinations
   follow the standard Slater-Koster Table I (1954) and are stubbed
   with NotImplementedError until Phase A continuation commits.

Conventions
===========
* Real spherical harmonics, Condon-Shortley phase absorbed into
  associated Legendre `P_l^|m|(x)`.
* p-orbital encoding: m = -1 -> p_y (sin phi), m = 0 -> p_z (cos theta),
  m = +1 -> p_x (cos phi).
* Orbital exponent zeta in Angstrom^-1; internuclear distance R in
  Angstrom (consistent with `atomic_basis.build_atom_basis`).
* The axial frame uses the GLOBAL +z axis at both atoms (no flip);
  the SK integrals therefore contain whatever sign comes out of the
  raw integral, which is consistent with `_overlap_1s_1s` for l = 0.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Quadrature caches (Gauss-Laguerre for xi in [1, inf), Gauss-Legendre
# for eta in [-1, 1]). Orders chosen to converge 1s-1s closed form to
# better than 1e-12.
# ─────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=8)
def _laguerre_nodes(n: int) -> Tuple[np.ndarray, np.ndarray]:
    return tuple(map(np.asarray, np.polynomial.laguerre.laggauss(n)))


@lru_cache(maxsize=8)
def _legendre_nodes(n: int) -> Tuple[np.ndarray, np.ndarray]:
    return tuple(map(np.asarray, np.polynomial.legendre.leggauss(n)))


_DEFAULT_LAG = 40   # order for xi direction; converges exp decays for p > 0
_DEFAULT_LEG = 32   # order for eta direction


# ─────────────────────────────────────────────────────────────────────
# Radial STO and associated Legendre helpers
# ─────────────────────────────────────────────────────────────────────


def _sto_radial_norm(n: int, zeta: float) -> float:
    """N_n(zeta) = (2 zeta)^n sqrt(2 zeta / (2n)!).

    R_n(r) = N_n(zeta) r^(n-1) e^(-zeta r)  is normalised so that
    integral_0^inf R^2 r^2 dr = 1 (radial part only).
    """
    return (2.0 * zeta) ** n * math.sqrt(2.0 * zeta / math.factorial(2 * n))


def _associated_legendre(l: int, m: int, x: float) -> float:
    """P_l^|m|(x), Condon-Shortley convention (matches scipy.special.lpmv).

    Implemented via the two-term recurrence; |m| <= l <= 4 is enough for
    the s/p/d/f valence basis used in PTC.
    """
    m = abs(m)
    if m > l:
        return 0.0
    # P_m^m
    pmm = 1.0
    if m > 0:
        somx2 = math.sqrt(max(0.0, 1.0 - x * x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm *= -fact * somx2
            fact += 2.0
    if l == m:
        return pmm
    # P_{m+1}^m
    pmmp1 = x * (2 * m + 1) * pmm
    if l == m + 1:
        return pmmp1
    # upward recurrence in l
    pll = 0.0
    for ll in range(m + 2, l + 1):
        pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll


def _spherical_norm(l: int, m: int) -> float:
    """Real-spherical-harmonic prefactor (no phi part).

    For m = 0:  N = sqrt((2l+1) / (4 pi))
    For m != 0: N = sqrt((2l+1) / (2 pi) * (l-|m|)! / (l+|m|)!)
    """
    m = abs(m)
    if m == 0:
        return math.sqrt((2 * l + 1) / (4.0 * math.pi))
    return math.sqrt(
        (2 * l + 1) / (2.0 * math.pi)
        * math.factorial(l - m) / math.factorial(l + m)
    )


# ─────────────────────────────────────────────────────────────────────
# AXIAL Slater-Koster integral (sigma, pi, delta)
# ─────────────────────────────────────────────────────────────────────


def slater_koster_axial(
    n_a: int, l_a: int,
    n_b: int, l_b: int,
    abs_m: int,
    zeta_a: float, zeta_b: float,
    R: float,
    *,
    n_lag: int = _DEFAULT_LAG,
    n_leg: int = _DEFAULT_LEG,
) -> float:
    """AXIAL Slater-Koster integral SS / SP_sigma / PP_sigma / PP_pi / ...

    Computes < phi_A | phi_B > between two STOs with both magnetic
    quantum numbers equal to abs_m (positive), placed at A = origin and
    B = (0, 0, R), with both orbital angular momenta defined w.r.t. the
    GLOBAL +z axis.

    Method
    ------
    Prolate spheroidal coordinates centred on AB:
        xi  in [1, inf)
        eta in [-1, 1]
        phi in [0, 2 pi)
    with
        r_a = R(xi+eta)/2,   r_b = R(xi-eta)/2,
        cos theta_a = (1+xi*eta)/(xi+eta),
        cos theta_b = (xi*eta-1)/(xi-eta),
        dV          = (R/2)^3 (xi^2 - eta^2) dxi deta dphi.
    The phi integration is analytic:
        integral cos(m phi) cos(m phi) dphi = pi (m != 0)
        integral 1 dphi                    = 2 pi (m = 0)
    The (xi, eta) part is done by Gauss-Laguerre x Gauss-Legendre.
    """
    if R <= 0.0:
        raise ValueError("slater_koster_axial requires R > 0.")
    if abs_m < 0:
        raise ValueError("abs_m must be >= 0.")
    if abs_m > l_a or abs_m > l_b:
        return 0.0

    p = R * (zeta_a + zeta_b) / 2.0
    q = R * (zeta_a - zeta_b) / 2.0

    # Gauss-Laguerre transforms ∫_1^inf exp(-p xi) g(xi) dxi
    # -> (e^-p / p) ∫_0^inf exp(-t) g(1 + t/p) dt
    lag_t, lag_w = _laguerre_nodes(n_lag)
    xi = 1.0 + lag_t / p
    pre_xi = math.exp(-p) / p

    leg_x, leg_w = _legendre_nodes(n_leg)
    eta = leg_x

    # Radial functions
    Na = _sto_radial_norm(n_a, zeta_a)
    Nb = _sto_radial_norm(n_b, zeta_b)
    R_half = R / 2.0
    r_a = R_half * (xi[:, None] + eta[None, :])           # shape (Nlag, Nleg)
    r_b = R_half * (xi[:, None] - eta[None, :])

    # cos theta
    denom_a = xi[:, None] + eta[None, :]
    denom_b = xi[:, None] - eta[None, :]
    cos_a = (1.0 + xi[:, None] * eta[None, :]) / denom_a
    cos_b = (xi[:, None] * eta[None, :] - 1.0) / denom_b

    # numerical safety on cos
    cos_a = np.clip(cos_a, -1.0, 1.0)
    cos_b = np.clip(cos_b, -1.0, 1.0)

    # Associated Legendre vectorised via numpy
    P_a = np.vectorize(lambda x: _associated_legendre(l_a, abs_m, float(x)))(cos_a)
    P_b = np.vectorize(lambda x: _associated_legendre(l_b, abs_m, float(x)))(cos_b)

    # Radial polynomial part (the e^{-p xi - q eta} comes out separately)
    radial_poly = (
        Na * Nb
        * (r_a ** (n_a - 1)) * (r_b ** (n_b - 1))
    )

    # eta-dependent exp(-q eta): keep inside the eta integral
    eta_exp = np.exp(-q * eta)

    # spheroidal volume factor
    vol_factor = (xi[:, None] ** 2 - eta[None, :] ** 2)

    # The xi-direction has its exp(-p xi) extracted by Gauss-Laguerre
    # substitution; the integrand we evaluate is everything BUT exp(-p xi).
    # = vol_factor * radial_poly * P_a * P_b * eta_exp
    integrand = vol_factor * radial_poly * P_a * P_b * eta_exp[None, :]

    # Apply quadrature weights
    # Outer product of Laguerre weights (for xi via substitution) and
    # Legendre weights (for eta direct)
    quad_eta = (integrand * leg_w[None, :]).sum(axis=1)         # over eta
    quad_xi  = (quad_eta * lag_w).sum() * pre_xi                # over xi

    # spheroidal Jacobian (R/2)^3
    integral_2d = (R_half ** 3) * quad_xi

    # phi-integral and angular norms
    phi_factor = (2.0 * math.pi) if abs_m == 0 else math.pi
    norm = _spherical_norm(l_a, abs_m) * _spherical_norm(l_b, abs_m)

    return phi_factor * norm * integral_2d


# Cache by all parameters so a molecule re-uses unique pairs
@lru_cache(maxsize=4096)
def _cached_axial(
    n_a: int, l_a: int, n_b: int, l_b: int, abs_m: int,
    zeta_a: float, zeta_b: float, R: float,
) -> float:
    return slater_koster_axial(n_a, l_a, n_b, l_b, abs_m,
                                zeta_a, zeta_b, R)


# ─────────────────────────────────────────────────────────────────────
# GEOMETRIC Slater-Koster rotation for s and p (l <= 1)
# ─────────────────────────────────────────────────────────────────────
#
# Real-spherical-harmonic ordering for p:
#   m = -1 : p_y  (sin phi component)
#   m =  0 : p_z  (cos theta component)
#   m = +1 : p_x  (cos phi component)
#
# Direction cosines of the unit vector AB (from A to B):
#   l_x = R_AB[0] / |R_AB|,  l_y = R_AB[1] / |R_AB|,
#   l_z = R_AB[2] / |R_AB|.


def _direction_cosine(orb_l: int, orb_m: int, lx: float, ly: float, lz: float) -> float:
    """Coefficient projecting global real-spherical orbital onto the
    AXIAL z'-axis (along AB). For l = 1 this returns the projection
    onto p_sigma; for l = 0 it returns 1."""
    if orb_l == 0:
        return 1.0
    if orb_l == 1:
        if orb_m == 0:
            return lz
        if orb_m == 1:
            return lx
        if orb_m == -1:
            return ly
        raise ValueError(f"l=1 expects m in -1,0,+1, got {orb_m}")
    raise NotImplementedError(f"_direction_cosine for l={orb_l} pending")


def overlap_sp_general(
    orb_A, orb_B, r_AB: np.ndarray,
) -> float:
    """Slater-Koster overlap for s and p orbitals (l_A, l_B <= 1).

    Decomposes < orb_A | orb_B > into AXIAL SK integrals weighted by
    direction cosines. Same-centre (R = 0) is handled by the caller.
    """
    R = float(np.linalg.norm(r_AB))
    if R < 1.0e-12:
        # Should not be called this path; same-orbital handled upstream.
        return 0.0

    lx, ly, lz = (r_AB / R).tolist()
    la, lb = orb_A.l, orb_B.l
    ma, mb = orb_A.m, orb_B.m

    # SS
    if la == 0 and lb == 0:
        return _cached_axial(orb_A.n, 0, orb_B.n, 0, 0,
                              orb_A.zeta, orb_B.zeta, R)

    # SP_sigma: < s_A | p_B^global >
    # In axial frame: < s | p_z' > = SP_sigma(axial)
    # Project p_B^global onto p_z' using direction cosines of AB.
    if la == 0 and lb == 1:
        sp_sigma = _cached_axial(orb_A.n, 0, orb_B.n, 1, 0,
                                  orb_A.zeta, orb_B.zeta, R)
        return _direction_cosine(1, mb, lx, ly, lz) * sp_sigma

    # PS_sigma: < p_A | s_B >  (axially, the p projects onto p_z'_at_A)
    if la == 1 and lb == 0:
        ps_sigma = _cached_axial(orb_A.n, 1, orb_B.n, 0, 0,
                                  orb_A.zeta, orb_B.zeta, R)
        return _direction_cosine(1, ma, lx, ly, lz) * ps_sigma

    # PP: combines pp_sigma and pp_pi
    if la == 1 and lb == 1:
        pp_sigma = _cached_axial(orb_A.n, 1, orb_B.n, 1, 0,
                                  orb_A.zeta, orb_B.zeta, R)
        pp_pi    = _cached_axial(orb_A.n, 1, orb_B.n, 1, 1,
                                  orb_A.zeta, orb_B.zeta, R)
        # Slater-Koster Table I, p-p block:
        # < p_x | p_x > = lx^2 ppσ + (1-lx^2) ppπ
        # < p_x | p_y > = lx ly (ppσ - ppπ)
        # etc., and our (m=-1, 0, +1) <-> (p_y, p_z, p_x).
        cosines = {
            -1: ly,   # p_y
             0: lz,   # p_z
            +1: lx,   # p_x
        }
        ca, cb = cosines[ma], cosines[mb]
        # Unified Slater-Koster pp formula (Table I, 1954):
        #   <p_alpha | p_beta> = c_a c_b (pp_sigma - pp_pi)
        #                       + delta_{alpha beta} pp_pi
        delta_ab = 1.0 if ma == mb else 0.0
        return ca * cb * (pp_sigma - pp_pi) + delta_ab * pp_pi

    raise NotImplementedError(
        f"overlap_sp_general: (l_A, l_B) = ({la}, {lb}) not supported "
        "(s and p only in this commit; d/f pending)."
    )


def overlap_3d_numerical(orb_A, orb_B, r_AB: np.ndarray,
                          n_radial: int = 50,
                          n_theta: int = 18,
                          n_phi: int = 24) -> float:
    """3D Gauss quadrature overlap on a spherical grid centred at the
    bond midpoint. Universal: works for any (n, l, m) STO pair, including
    d/f orbitals.

    Slower than the analytic 1s-1s and Slater-Koster sp paths, but a
    clean fallback for higher-l orbitals while the SK table is being
    extended.

    Validated against analytic 1s-1s for s-s pairs (1e-6 with default
    grid; 1e-10 with refined grid).
    """
    from ptc.lcao.giao import evaluate_sto

    R = float(np.linalg.norm(r_AB))
    if R < 1.0e-12:
        # same-centre: orthonormal basis
        if (orb_A.n == orb_B.n and orb_A.l == orb_B.l
                and orb_A.m == orb_B.m
                and abs(orb_A.zeta - orb_B.zeta) < 1.0e-12):
            return 1.0
        return 0.0

    midpoint = 0.5 * np.asarray(r_AB, dtype=float)
    R_max = max(
        R / 2.0 + 8.0 / orb_A.zeta,
        R / 2.0 + 8.0 / orb_B.zeta,
        12.0 / min(orb_A.zeta, orb_B.zeta),
    )

    # spherical grid (matches density_matrix style)
    leg_r, w_r = np.polynomial.legendre.leggauss(n_radial)
    r_nodes = R_max / 2.0 * (leg_r + 1.0)
    r_weights = R_max / 2.0 * w_r

    leg_u, w_u = np.polynomial.legendre.leggauss(n_theta)
    cos_t = leg_u

    phi_nodes = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)
    phi_weight = 2.0 * math.pi / n_phi

    R_g, CT, PHI = np.meshgrid(r_nodes, cos_t, phi_nodes, indexing='ij')
    ST = np.sqrt(np.maximum(0.0, 1.0 - CT ** 2))
    X = R_g * ST * np.cos(PHI)
    Y = R_g * ST * np.sin(PHI)
    Z = R_g * CT
    grid_pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3) + midpoint

    W = (
        ((r_nodes ** 2) * r_weights)[:, None, None]
        * w_u[None, :, None]
        * np.full(n_phi, phi_weight)[None, None, :]
    )
    W_flat = W.flatten()

    psi_A = evaluate_sto(orb_A, grid_pts, np.zeros(3))
    psi_B = evaluate_sto(orb_B, grid_pts, np.asarray(r_AB, dtype=float))
    return float(np.sum(psi_A * psi_B * W_flat))
