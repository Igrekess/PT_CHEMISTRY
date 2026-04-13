"""Geometric Shell Operator — ShellPolygon and AtomicShell dataclasses.

Models a single subshell as a polygon in PT space, providing geometric
amplitudes for electron capture and ejection based on fill fraction,
exchange symmetry, and closure boost.

Primes encode the shell type:
  prime=1 → s  (capacity 2)
  prime=3 → p  (capacity 6)
  prime=5 → d  (capacity 10)
  prime=7 → f  (capacity 14)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from ptc.constants import (
    S3, S5, S7, S11, S_HALF, P1, P2, P3, MU_STAR,
    D3 as _D3, D5 as _D5, D7 as _D7, C3 as _C3, C7 as _C7,
    P_EXCHANGE_PROFILE,
)

# ── Phase B: constants derived from polygon geometry at μ*=15 ──
# Each polygon Z/(2p)Z derives its own coupling from the sieve:
#   sin²(θ_p) = δ_p(2-δ_p)  where δ_p = (1-q^p)/p, q = 1-2/μ*

_Q = 1.0 - 2.0 / MU_STAR  # 13/15

# Coupling = sin²(θ_prime) — the polygon's OWN mixing angle
_COUPLING: dict[int, float] = {
    1: S3,   # s-block: uses P₁ coupling (entry into sieve)
    3: S3,   # p-block: sin²₃
    5: S5,   # d-block: sin²₅
    7: S7,   # f-block: sin²₇
}

# Exchange weight = sin²(θ_{next_prime}) — Hund mediated by NEXT channel
# PT: exchange on Z/(2p)Z is 2nd-order, goes through the next active prime.
_EXCHANGE_WEIGHT: dict[int, float] = {
    1: 0.0,   # s-block: trivial polygon, no exchange
    3: S5,    # p-block Hund via d-channel: sin²₅ ≈ 0.194
    5: S7,    # d-block Hund via f-channel: sin²₇ ≈ 0.173
    7: S11,   # f-block Hund via ghost:     sin²₁₁ ≈ 0.139
}

# Layer name mapping
_LAYER_NAME: dict[int, str] = {
    1: "s",
    3: "p",
    5: "d",
    7: "f",
}

# ── Hexagonal Z/6Z constants for p-block capture ────────────────────────────
# _D3, _D5, _D7, _C3, _C7 imported from constants.py (Fraction-derived)

# Hexagonal exchange profile on Z/6Z (intra-hexagonal, uses S₃ coupling).
# Maps n_p → exchange count for the 4 irreducible pieces of the hexagon.
_P_HEX_EXCHANGE = P_EXCHANGE_PROFILE

# ── GFT anti-double-counting: k=1 Fourier projection constants ────────
# The polygon_symmetry harmonic profile has a dominant k=1 mode that is
# ALSO in the screening DFT (peer_values → synthesis).  GFT Principle 7
# forbids double-counting: log₂(m) = D_KL + H is an identity.
# We precompute the k=1 (a₁,b₁) coefficients for each polygon type and
# subtract them from sym before the ejection uses it.
# These depend ONLY on the prime (not per or nf).

def _compute_k1_coeffs(prime: int) -> tuple:
    """Precompute k=1 Fourier coefficients of polygon_symmetry on Z/(2p)Z."""
    N = 2 * prime
    omega = 2.0 * _math.pi / N
    # Build the profile (using a temporary per=4, doesn't matter for sym)
    profile = []
    for n in range(1, N + 1):
        if n <= 0 or n >= N:
            profile.append(0.0)
        elif n <= prime:
            profile.append(_math.log2(1.0 + 1.0 / n))
        else:
            n_down = n - prime
            profile.append(-_math.log2(1.0 + 1.0 / n_down))
    a1 = 2.0 / N * sum(v * _math.cos(omega * n) for n, v in enumerate(profile, 1))
    b1 = 2.0 / N * sum(v * _math.sin(omega * n) for n, v in enumerate(profile, 1))
    return (a1, b1)

import math as _math
_K1_COEFFS: dict[int, tuple] = {
    P1: _compute_k1_coeffs(P1),  # p-block: (0.0, 0.915)
    P2: _compute_k1_coeffs(P2),  # d-block: (0.188, 0.691)
    P3: _compute_k1_coeffs(P3),  # f-block: (0.236, 0.546)
}


@dataclass(frozen=True)
class ShellPolygon:
    """Frozen dataclass representing a subshell occupancy as a PT polygon.

    Parameters
    ----------
    prime:
        Polygon prime (1=s, 3=p, 5=d, 7=f).
    n_occupied:
        Number of electrons in this subshell (0 to 2*prime).
    per:
        Period (1-7). Determines radial support and simplex passage.
    """

    prime: int
    n_occupied: int
    per: int = 2        # default for backward compatibility
    f_buried: bool = False  # True when f14 is buried under d-shell (per≥6 post-f)

    # ── Basic properties ──────────────────────────────────────────────────

    @property
    def capacity(self) -> int:
        """Maximum number of electrons: 2 × prime."""
        return 2 * self.prime

    @property
    def vacancies(self) -> int:
        """Number of unoccupied slots."""
        return self.capacity - self.n_occupied

    @property
    def fill_fraction(self) -> float:
        """Occupation fraction n / capacity."""
        return self.n_occupied / float(self.capacity)

    @property
    def half_filled(self) -> bool:
        """True when n_occupied equals prime (Hund half-filling)."""
        return self.n_occupied == self.prime

    @property
    def layer_name(self) -> str:
        """Shell label: 's', 'p', 'd', or 'f'."""
        return _LAYER_NAME[self.prime]

    # ── Hund configuration ───────────────────────────────────────────────

    def hund_config(self) -> tuple[int, int]:
        """Return (n_up, n_down) under Hund's rule.

        n_up  = min(n, prime)
        n_down = max(0, n - prime)
        """
        n = self.n_occupied
        p = self.prime
        n_up = min(n, p)
        n_down = max(0, n - p)
        return (n_up, n_down)

    # ── Geometric methods ─────────────────────────────────────────────────

    def exchange_pairs(self) -> int:
        """Count same-spin pairs: C(n_up, 2) + C(n_down, 2)."""
        n_up, n_down = self.hund_config()
        c_up = n_up * (n_up - 1) // 2
        c_down = n_down * (n_down - 1) // 2
        return c_up + c_down

    def closure_boost(self) -> float:
        """Discrete boost toward shell closure, scaled by polygon size.

        PT: the closure pull is proportional to the circumference of the
        circle Z/(2p)Z. Larger polygons (d, f) pull harder at closure
        than smaller ones (s, p).

        Scaling: base_boost × (prime / P₁) where P₁=3 normalizes to p-block.
          s (prime=1): factor 1/3 — mild closure
          p (prime=3): factor 1.0 — reference
          d (prime=5): factor 5/3 — stronger
          f (prime=7): factor 7/3 — strongest

        vacancies=0  → 0.0
        vacancies=1  → (1/s²) × (prime/P₁)
        vacancies=2  → (1/s) × (prime/P₁)
        vacancies≥3  → 1.0
        """
        v = self.vacancies
        if v <= 0:
            return 0.0
        if v >= 3:
            return 1.0
        if self.prime == 1:
            # Pair: completion via softest channel (γ₇).
            from ptc.constants import GAMMA_7
            return GAMMA_7 if v == 1 else 1.0
        # Scale by polygon circumference relative to p-block
        scale = self.prime / 3.0  # P₁ = 3 is the reference
        if v == 1:
            # PT: last vacancy pulled by σ-channel (1/s²) PLUS next-prime
            # channel coupling. The next prime opens for the final capture.
            next_sin2 = _EXCHANGE_WEIGHT.get(self.prime, S3)  # sin²_{next}
            return (1.0 / (S_HALF ** 2)) * (1.0 + next_sin2) * scale
        if v == 2:
            return (1.0 / S_HALF) * scale
        return 1.0

    def polygon_symmetry(self) -> float:
        """Asymmetric Hund profile from the double-cover structure.

        PT: the factor 2 creates two copies of the p-gon (spin ↑ and ↓).
        The ejection cost is ASYMMETRIC because removing from ↑ (phase 1)
        costs exchange pairs, while removing from ↓ (phase 2) liberates
        pairing repulsion.

        GFT harmonic series: each electron adds log₂(1+1/n) bits of
        persistence to the polygon (Hund phase), or removes log₂(1+1/n_down)
        bits (pairing phase).  This is the D_KL increment from the
        Gap-Fluctuation Theorem applied to Z/(2p)Z.

        Result: peaks at n=1 (first electron = 1 full bit), decreases
        harmonically through Hund, switches sign at post-Hund (pairing
        destabilises), returns toward zero at closure.
        """
        if self.n_occupied <= 0 or self.n_occupied >= self.capacity:
            return 0.0
        if self.prime <= 1:
            return 0.0  # s-pair: no exchange, no Hund asymmetry

        n = self.n_occupied
        p = self.prime

        # GFT harmonic series: ΔD_KL(n) on Z/(2p)Z
        if n <= p:
            # Hund phase: each same-spin electron ADDS persistence
            raw = math.log2(1.0 + 1.0 / n)
        else:
            # Pairing phase: each cross-spin electron REMOVES persistence
            n_down = n - p
            raw = -math.log2(1.0 + 1.0 / n_down)

        return raw

    def _koide_envelope(self) -> float:
        """Generalized Koide envelope for a p-gon: x^(p-1) × (1-x).

        PT: the Koide point is NOT universal at 2/3. For a polygon with
        prime p, the optimal capture is at (p-1)/p:
          p=3 (p-block): peak at 2/3 = 0.667 → standard Koide Q=2/3
          p=5 (d-block): peak at 4/5 = 0.800 → Ni(d8), not Fe(d6)
          p=7 (f-block): peak at 6/7 = 0.857 → late lanthanide

        The envelope x^(p-1)(1-x) naturally gives higher-power suppression
        for larger polygons at low filling (x << 1), matching the physical
        observation that d-block early filling has very low EA.

        For s-block (prime=1): x^0 × (1-x) = (1-x). Special: S_HALF at n=1.
        """
        n = self.n_occupied
        if n <= 0 or n >= self.capacity:
            return 0.0

        if self.prime <= 1:
            return S_HALF if n == 1 else 0.0

        p = self.prime
        x = self.fill_fraction
        raw = x ** (p - 1) * (1.0 - x)
        # Normalize: peak at x = (p-1)/p
        x_peak = (p - 1) / p
        peak = x_peak ** (p - 1) * (1.0 / p)
        return raw / peak if peak > 0 else 0.0

    def _exchange_factor(self) -> float:
        """Exchange modulation factor (1 + w × count)².

        For s-block: always 1.0 (w=0).
        exchange_count: n<p → n, n==p → -1, n>p → n-p-1
        """
        if self.prime == 1:
            return 1.0

        n = self.n_occupied
        p = self.prime
        w = _EXCHANGE_WEIGHT[self.prime]

        # exchange_count
        if n <= 0 or n >= 2 * p:
            count = 0
        elif n < p:
            count = n
        elif n == p:
            count = -1
        else:
            count = n - p - 1

        return (1.0 + w * count) ** 2

    def geometric_profile(self) -> float:
        """Full geometric amplitude profile.

        profile = koide_envelope × exchange_factor × closure_boost
        """
        return self._koide_envelope() * self._exchange_factor() * self.closure_boost()

    # ── Koide bifurcation ─────────────────────────────────────────────

    def _koide_normalized(self) -> float:
        """Koide normalized for capture/ejection bifurcation.

        For s-block (prime=1): uses P₁=3 Koide (x²(1-x)/(4/27)) because
        the s-pair captures via the first active channel, not its own.
        For p/d/f: uses the generalized x^(p-1)(1-x) envelope.
        """
        if self.n_occupied <= 0 or self.n_occupied >= self.capacity:
            return 0.0
        if self.prime <= 1:
            # s-block: Koide through P₁=3 channel
            x = self.fill_fraction
            return (x * x * (1.0 - x)) / (4.0 / 27.0)
        return self._koide_envelope()

    # ── Hexagonal p-block capture (Z/6Z) ────────────────────────────────

    def _p_hex_capture(self) -> float:
        """p-block capture via hexagonal Z/6Z completion functional.

        PT reading: the p-shell is a hexagon Z/(2×3)Z = Z/6Z. The observable
        is a hexagonal completion functional with 4 irreducible pieces:
          1. shell completion (n/cap, linear on the hexagon)
          2. Hund/Pauli exchange profile (hex-specific, uses S₃)
          3. degeneracy lift (2-loop, before quasi-closure)
          4. period factor f_per (radial contraction/expansion)

        Ported from ea_operator._p_light_hexagon_amplitude, adapted to the
        geometric shell framework.  Zero adjustable parameters.
        """
        n = self.n_occupied
        cap = self.capacity          # 6
        per = self.per

        if n <= 0 or n >= cap:
            return 0.0

        # Heavy p-block: j-j relativistic splitting (per ≥ 6)
        if per >= 6:
            return self._p_heavy_capture()

        # ── Piece 1: Shell completion (linear on hexagon) ──
        completion = n / float(cap)

        # ── Piece 2: Exchange profile (intra-hexagonal, S₃) ──
        exchange = (1.0 + S3 * _P_HEX_EXCHANGE.get(n, 0)) ** 2

        # ── Piece 3: Degeneracy lift (2-loop) ──
        degeneracy = 0.0
        if n < cap - 1:
            degeneracy = S3 ** 2 * math.log(cap - n) / math.log(cap)

        # ── Piece 4: Small corrections ──
        c_pd = math.sqrt(1.0 + _D5) - 1.0      # p→d creation coupling

        # Boundary term at n = P₁+1, per ≥ 3
        boundary = S3 * S_HALF if (n == P1 + 1 and per >= 3) else 0.0
        # Hund depth at half-fill
        hund_depth = S3 * _C3 * (2.0 / per) ** 2 if n == P1 else 0.0
        # Embryonic penalty: first electron in contracted 2p shell
        embryonic_penalty = S_HALF * S5 if (per == 2 and n == 1) else 0.0
        # Half-shell bridge at half-fill
        half_shell_bridge = 0.0
        if n == P1:
            if per in (3, 4):
                half_shell_bridge = _C3 * c_pd
            elif per == 5:
                half_shell_bridge = S5

        # ── Additive total ──
        f_total = (
            completion * exchange
            + degeneracy
            + boundary
            + half_shell_bridge
            - hund_depth
            - embryonic_penalty
        )

        # ── Period factor ──
        # per=2: contracted (C₃ = cos²₃ = 0.781)
        # per≥3: expanded (√(1+S₃) = 1.104)
        f_per = _C3 if per == 2 else math.sqrt(1.0 + S3)

        amp = max(0.0, S3 * f_total * f_per)

        # Pairing contraction for post-Hund states.
        # Per=2: contracted shell → pairing per spin channel (C₃^(n_down/prime)).
        # Near-closure (n=cap-1): per-dependent propagator.
        #   per≤3: hexagonal C₃ propagator (no d-shell)
        #   per=4: pentagonal C₅ propagator (d-shell opens at per=4)
        #   per=5: 2-loop cross-face (1−R₅₇), tiny correction (4d diffuse)
        if per == 2 and n > P1 and n < cap - 1:
            n_down = n - P1
            amp *= _C3 ** (n_down / float(P1))
        elif n == cap - 1 and n > P1:
            n_down = n - P1
            if per <= 3:
                amp *= _C3 ** (n_down / float(cap))
            elif per == 4:
                _c5 = 1.0 - S5                         # cos²₅ (pent propagator)
                amp *= _c5 ** (n_down / float(cap))
            else:  # per == 5
                amp *= (1.0 - _D5 * _D7)               # 1 − R₅₇ (cross-face 2-loop)

        # n=2 exchange contraction: the exchange at profile[2]=2 (strongest)
        # overshoots for capture.  At per≠3, costs δ₃ (one sieve level).
        # At per=3 (natural hexagonal period, no d-shell yet), the correction
        # goes through the virtual d-channel δ₅, spin-attenuated by s.
        # At per≥5 (4d complete), add cross-face f→d correction sin²₇/P₂.
        if n == 2:
            if per == 3:
                amp *= (1.0 - _D5 * S_HALF)
            elif per >= 5:
                amp *= (1.0 - _D3) * (1.0 - S7 / P2)  # cross-face f/d at per≥5
            else:
                amp *= (1.0 - _D3)

        # n=P₁+1 spin-cascade boost (insight #55, replaces per-specific d-channel):
        # The captured electron at position n+1=P₁+2 enters via a spin cascade.
        # Amplitude per spin level = S₃×s/P₁ (hex vertex × spin × filling⁻¹).
        # Each period beyond the native hex period P₁=3 adds one cascade level
        # (1/s = 2), so the factor doubles per period: 2^(per−P₁).
        # Unifies per=3 (no d-shell, direct hex), per=4 (virtual d), per=5 (real d)
        # under a single PT law. Excludes per<P₁ (no cascade yet).
        if n == P1 + 1 and per >= P1:
            _cascade_amp = S3 * S_HALF / P1             # S₃/(2P₁) ≈ 0.0365
            amp *= (1.0 + _cascade_amp * float(2 ** (per - P1)))

        return amp

    def _p_heavy_capture(self) -> float:
        """Heavy p-block capture (per ≥ 6): j-j relativistic 2+4 splitting.

        PT reading: at period ≥ 6 the spin-orbit coupling is strong enough
        to split the hexagon into p₁/₂ (capacity 2) + p₃/₂ (capacity 4).
        The p₁/₂ pair fills first as a relativistic inner spinor; once closed,
        the p₃/₂ quartet becomes the active capture sector.
        """
        n = self.n_occupied
        cap = self.capacity                  # 6

        n_half = min(n, 2)                   # p₁/₂ occupancy
        n_three_half = max(0, n - 2)         # p₃/₂ occupancy

        c_sp = math.sqrt(1.0 + S3) - 1.0    # s→p creation coupling

        if n_half == 1 and n_three_half == 0:
            # Tl-like: direct closure of the split p₁/₂ pair
            result = S_HALF * math.sqrt(c_sp * S3 * _C3)

        elif n_three_half <= 0:
            # Pb-like baseline: closed p₁/₂ with relativistic attenuation
            result = S_HALF * c_sp * math.sqrt(1.0 - _D5)

        else:
            # Active p₃/₂ manifold: total Koide envelope on the full hexagon
            baseline = S_HALF * c_sp * math.sqrt(1.0 - _D5)
            x = n / 6.0
            total_envelope = (x * x * (1.0 - x)) / (4.0 / 27.0)

            # Closure pull on the p₃/₂ quartet
            vac_34 = 4 - n_three_half
            if vac_34 <= 0:
                cl_34 = 0.0
            elif vac_34 == 1:
                cl_34 = 1.0 / (S_HALF ** 2)
            elif vac_34 == 2:
                cl_34 = 1.0 / S_HALF
            else:
                cl_34 = 1.0

            rel_spinor = math.sqrt(1.0 / S_HALF) * (1.0 + S3 * n_three_half)
            # f14 screens the quartet half-fill (n₃/₂=2): the heptagonal
            # gap δ₇ attenuates the closure pull at this balanced state.
            if n_three_half == 2:
                rel_spinor *= (1.0 - _D7)
            active = S3 * total_envelope * cl_34 * rel_spinor
            extra = S_HALF * math.sqrt(c_sp * active)
            result = baseline + extra

        # n=4 d-channel cross-coupling (same as hex, 2-loop cascade):
        # completed d-shell at per≥5 boosts the zero-exchange n=P₁+1 state.
        if n == P1 + 1:
            result *= (1.0 + S7)

        # j-j unpaired exchange deficit: odd-n elements (Tl, Bi) have an
        # unpaired electron in the relativistic sub-shell, costing sin²₃/P₁
        # of hexagonal exchange per vertex.  Not for near-closure (At).
        if n % 2 == 1 and n < cap - 1:
            result *= (1.0 - S3 / P1)

        return result

    # ── Heptagonal f-block capture (Z/14Z) ──────────────────────────────

    def _f_hept_capture(self) -> float:
        """f-block capture via heptagonal Z/14Z amplitude (7+7 split).

        PT reading: the f-shell is two heptagons on Z/(2×7)Z = Z/14Z.
        The f-orbitals are BURIED (contracted under the valence surface),
        so the observable is a compressed-density projector, not a pure
        boundary flux.

        Lanthanides (per=6, 4f): buried-shell with discrete pair/parity
        projector on each heptagon.
        Actinides (per=7, 5f): boundary-flux amplitude with actinide
        projector (5f is less buried than 4f).

        Ported from ea_operator._lanthanide_f_amplitude and
        _f_boundary_flux_amplitude + _actinide_f_projector.
        Zero adjustable parameters.
        """
        n = self.n_occupied
        per = self.per
        p = P3  # 7

        if n <= 0 or n >= 2 * p:
            return 0.0

        c_df = math.sqrt(1.0 + _D7) - 1.0  # d→f creation coupling
        n_first = min(n, p)                  # first heptagon occupancy
        n_second = max(0, n - p)             # second heptagon occupancy

        def _env7(occ: int) -> float:
            """Koide x²(1-x)/(4/27) on the heptagon (7 states)."""
            if occ <= 0 or occ >= p:
                return 0.0
            x = occ / float(p)
            return (x * x * (1.0 - x)) / (4.0 / 27.0)

        def _cl7(vac: int) -> float:
            """Closure pull on heptagonal scale."""
            if vac <= 0:
                return 0.0
            if vac == 1:
                return 1.0 / (S_HALF ** 2)
            if vac == 2:
                return 1.0 / S_HALF
            return 1.0

        # Ghost dictionaries — per-state modulation for each heptagon
        # f6 ghost boosted by (1+S₇): approaching half-fill stability
        # lifts the pre-half-fill penalty.
        g1 = {1: 1.0, 2: 1.0 + S7, 3: 1.0 / S_HALF,
              4: 1.0 / (S_HALF ** 2), 5: _D7,
              6: _D7 * (1.0 + S7),
              7: 1.0 + S7 / S_HALF}
        g2 = {1: _D7, 2: 1.0 / S_HALF, 3: _C7,
              4: _C7, 5: _C7, 6: 1.0 / S_HALF, 7: 0.0}

        if per == 6:
            # ── Lanthanide 4f: buried-shell projector ──
            root = math.sqrt(1.0 / S_HALF)
            boost_val = 1.0 / (S_HALF * math.sqrt(S_HALF))
            # f7 projector attenuated by C₇: the half-fill is a critical
            # point where the heptagonal channel coupling changes sign.
            proj_1 = {2: boost_val, 3: root, 4: 1.0,
                      5: root, 6: 1.0, 7: boost_val * _C7}
            # proj_2 uses C₇² at heptagonal nodes (n_second=1,5):
            # the memory-to-active transition crosses TWO cos²₇ boundaries.
            proj_2 = {1: _C7 ** 2, 2: (1.0 / S_HALF) + root, 3: 1.0,
                      4: 1.0, 5: _C7 ** 2, 6: root, 7: 1.0}

            if n_second == 0:
                if n == p:  # half-filled f7
                    result = c_df * (1.0 + S7 / S_HALF) * proj_1[7]
                else:
                    active = (_D7 * _env7(n_first) * _cl7(p - n_first)
                              * g1[n_first])
                    result = active * proj_1.get(n_first, 1.0)
            else:
                memory = S_HALF * c_df * (1.0 + S7 / S_HALF)
                if n_second == p:  # f14 = full
                    result = memory * _D7
                else:
                    active = (_D7 * _env7(n_second) * _cl7(p - n_second)
                              * g2[n_second])
                    flux = memory + S_HALF * math.sqrt(c_df * active)
                    result = flux * proj_2.get(n_second, 1.0)

            # ── Heptagonal gap correction (insight #41) ──────────
            # PT: the heptagonal gap δ₇ provides a bidirectional coupling
            # between the buried f-shell and the valence surface.
            # The Koide ratio depends on the polygon geometry (Z/14Z):
            #   Near closure (vac ≤ 2): δ₇ ASSISTS capture (1st order)
            #   Half-fill / pre-half-fill: δ₇ × C₇ or δ₇ × s (sieve passage)
            #   Birth / post-peak: δ₇ × s ATTENUATES (2nd order, spin)
            if n_second == 0:
                if n_first == 7:
                    result *= (1.0 + _D7 * _C7)     # half-fill sieve passage
                elif n_first == 6:
                    result *= (1.0 + _D7 * S_HALF)   # pre-half-fill assist
            elif n_second < p:
                vac = p - n_second
                if vac == 1:
                    result *= (1.0 + _D7)             # near-closure 1st order
                elif vac == 2:
                    result *= (1.0 + _D7 * S_HALF)   # approaching closure
                elif n_second == 3:
                    result *= (1.0 + _D7)             # mid-fill exchange
                elif n_second in (1, 4):
                    result *= (1.0 - _D7 * S_HALF)   # birth/post-peak atten.

            return result

        # ── Actinide 5f: boundary flux + actinide projector ──
        if n <= p:
            act_ghost = g1.get(n_first, 1.0)
        else:
            act_ghost = g2.get(n_second, 1.0)
        act_proj = math.sqrt(1.0 + c_df * act_ghost)

        if n_second == 0:
            if n == p:
                return S_HALF * c_df * (1.0 + S7 / S_HALF) * act_proj
            active = (_D7 * _env7(n_first) * _cl7(p - n_first)
                      * g1[n_first])
            return S_HALF * math.sqrt(c_df * active) * act_proj

        memory = S_HALF * c_df * (1.0 + S7 / S_HALF)
        if n_second == p:
            return memory * _D7 * act_proj

        active = (_D7 * _env7(n_second) * _cl7(p - n_second)
                  * g2[n_second])
        return (memory + S_HALF * math.sqrt(c_df * active)) * act_proj

    # ── Pentagonal d-block capture (Z/10Z) ──────────────────────────────

    def _d_pent_capture(self) -> float:
        """d-block capture via pentagonal Z/10Z amplitude.

        PT reading: the d-shell is two pentagons (5+5) on Z/(2×5)Z = Z/10Z.
        Pentagon birth states (d1, d6) use the p→d creation coupling directly.
        Standard states (d2-d5, d7-d9) use Koide x²(1-x) (universal Q=2/3)
        plus exchange and closure.  All states are modulated by a
        period-specific pentagon projector.

        Ported from ea_operator._d_boundary_flux_amplitude.
        Zero adjustable parameters.
        """
        n = self.n_occupied
        cap = self.capacity          # 10
        per = self.per
        p = self.prime               # 5 = P2

        if n <= 0 or n >= cap:
            return 0.0

        c_pd = math.sqrt(1.0 + _D5) - 1.0       # p→d creation coupling
        c_df = math.sqrt(1.0 + _D7) - 1.0       # d→f creation coupling
        base_birth = S_HALF * math.sqrt(c_pd * _D5)

        # ── Pentagon birth: d1 (first pentagon) ──
        if n == 1:
            if per == 4:
                amp = base_birth * _C3  # 3d contraction (like embryonic penalty)
            elif per == 5:
                amp = base_birth * math.sqrt(1.0 / S_HALF)
            else:
                amp = base_birth * (1.0 / S_HALF)
            return amp * self._d_pentagon_proj()

        # ── Pentagon birth: d6 (second pentagon) ──
        if n == 6:
            if per == 4:
                # d6→d7 exchange correlation: 3 new ↑ pairs minus 1 pairing
                # cost = net 2 effective exchange pairs.
                amp = base_birth * S5 * (1.0 + 2 * S5)
            elif per == 5:
                amp = (S_HALF * c_pd) + base_birth * math.sqrt(1.0 / S_HALF) * S5
            else:
                birth = base_birth * (1.0 / S_HALF)
                if self.f_buried:
                    birth *= (1.0 - _D7)  # f14 screens pentagon birth
                amp = birth + (S_HALF * c_pd)
            return amp * self._d_pentagon_proj()

        # ── Standard: d2-d5, d7-d9 — Koide x²(1-x) + exchange + closure ──
        x = n / float(cap)
        envelope = (x * x * (1.0 - x)) / (4.0 / 27.0)

        # Exchange count (same profile as generic _exchange_factor)
        if n < p:
            count = n
        elif n == p:
            count = -1
        else:
            count = n - p - 1
        exchange = (1.0 + S5 * count) ** 2

        # Closure pull
        vac = cap - n
        if vac == 1:
            cl = 1.0 / (S_HALF ** 2)
        elif vac == 2:
            cl = 1.0 / S_HALF
        else:
            cl = 1.0

        active = _D5 * envelope * exchange * cl
        amplitude = S_HALF * math.sqrt(c_pd * active)

        # Period boost
        if per == 4:
            boost = math.sqrt(1.0 + c_pd)
        elif per == 5:
            boost = math.sqrt(1.0 / S_HALF)
        else:
            if n == 4:
                boost = math.sqrt(1.0 / S_HALF)
            elif n == 5:
                boost = 1.0
            elif n >= 7:
                boost = math.sqrt(1.0 / S_HALF) * math.sqrt(1.0 + c_df)
            else:
                boost = 1.0

        # d2 frustration: at per=4 mediated by p-channel (3d within 4s-4p)
        # → δ₃.  At per≥6 the 4f14 buried shell screens via heptagonal
        # gap → (1-δ₇)^(1/2).  d5 Hund correction below.
        if n == 2:
            if per == 4:
                boost *= math.sqrt(_D3)
            elif per >= 6 and self.f_buried:
                boost *= math.sqrt(1.0 - _D7)
        if n == 5:
            boost *= _C3

        # Late-d memory term (saturated first pentagon feeds second)
        memory = S_HALF * c_pd if n >= 7 else 0.0

        proj = self._d_pentagon_proj()
        return (memory + amplitude * boost) * proj

    def _d_pentagon_proj(self) -> float:
        """Pentagon projector — discrete per-period modulation for d-shell.

        PT reading: the d-shell is NOT a smooth decagon. Each transition-metal
        series carries its own pentagonal parity structure:
          per=4 (3d): weak radial depth, moderate projector
          per=5 (4d): maximal s-d resonance, strongest modulation
          per=6 (5d): heavy second pentagon, f-latent reinforcement
        """
        per = self.per
        n = self.n_occupied

        root = math.sqrt(1.0 / S_HALF)           # √2
        boost = 1.0 / (S_HALF * math.sqrt(S_HALF))  # 2√2

        if per == 4:
            return {
                2: root, 3: 1.0 / S_HALF, 4: 1.0 / S_HALF,
                6: 1.0 / S_HALF,
                7: 1.0 + S5,  # pentagon pairing frustration (4↑+3↓)
                8: root * (1.0 + _D3),  # p→d back-flow at near-closure
                9: root,
            }.get(n, 1.0)

        if per == 5:
            # 4d series: Hund projectors (d3,d5,d7) damped by (1-δ₃)
            # — the 2nd d-series loses δ₃ amplitude per sieve level.
            hd = 1.0 - _D3
            return {
                2: 1.0 / S_HALF, 3: boost * hd, 4: root,
                5: boost * hd, 6: 1.0 / (S_HALF ** 2),
                7: (1.0 / S_HALF) * hd, 8: _C3 ** 2, 9: 1.0 + S5,
            }.get(n, 1.0)

        if per >= 6:
            # f-shell awareness: pre-f (La/Ac, f_buried=False) vs
            # post-f (Lu-Hg, f_buried=True) have different d1 projectors.
            # Pre-f: f-latent channel OPEN → (1+S₅) boost.
            # Post-f: f14 SCREENS d → √C₃ contraction.
            d1_proj = math.sqrt(_C3) if self.f_buried else (1.0 + S5)
            hd6 = 1.0 - _D3  # Hund damping for 3rd+ d-series
            return {
                1: d1_proj, 2: 1.0 + S5, 3: 1.0 + S5, 4: root,
                5: hd6,  # d5 half-fill: δ₃ sieve attenuation (same as per=5)
                6: root, 7: 1.0 / S_HALF,
                8: 1.0 / S_HALF, 9: (1.0 / S_HALF) - S5,
            }.get(n, 1.0)

        return 1.0

    # ── Capture / ejection amplitudes ────────────────────────────────────

    def capture_amplitude(self) -> float:
        """EA-like amplitude: receptivity at vertex n+1.

        Capture is PROPORTIONAL to the Koide envelope — maximal at x=2/3,
        zero at extremes. Modulated by exchange, closure, and radial support.

        PT: heavier periods have deeper radial support (more diffuse orbitals)
        → stronger capture. The support factor √(per/2) accounts for this.
        Period 2 (compact) is the reference; period 5 captures ~1.6× better.
        """
        if self.vacancies <= 0:
            return 0.0

        # p-block: hexagonal Z/6Z completion functional (insight #13)
        if self.prime == P1:
            return self._p_hex_capture()

        # d-block: pentagonal Z/10Z amplitude (insight #14)
        if self.prime == P2:
            return self._d_pent_capture()

        # f-block: heptagonal Z/14Z amplitude (insight #15)
        if self.prime == P3:
            return self._f_hept_capture()

        coupling = _COUPLING[self.prime]
        k = self._koide_normalized()
        ex = self._exchange_factor()
        cl = self.closure_boost()
        vac_frac = self.vacancies / float(self.capacity) if self.capacity > 0 else 0.0
        cl_norm = cl * vac_frac
        # Radial support × polygon density.
        # PT: heavier periods have more radial support (√(per/2)), but
        # sparser polygons (d, f) capture less per electron.
        # density_scale = (sin²_p / sin²₃) × (P₁ / prime)
        #   p: 1.0 (reference), d: 0.53, f: 0.34, s: special
        if self.prime <= 1:
            radial = 1.0  # s-block: pair handled by γ₇ closure
        else:
            # Density: coupling per state (sparser polygons capture less)
            _ref = _COUPLING[P1] / P1  # p-block density (reference)
            _self = coupling / self.prime
            density_scale = _self / _ref if _ref > 0 else 1.0
            # Circle radius with first-appearance contraction.
            # PT: the first shell of each block type has no radial node
            # → maximally contracted. Subsequent shells extend by √per.
            # Effective period = per - first_per_of_block + 2.
            _FIRST_PER = {1: 1, P1: 2, P2: 4, P3: 6}
            first_per = _FIRST_PER.get(self.prime, 2)
            per_eff = max(self.per - first_per + 2, 2)
            circle_radius = math.sqrt(per_eff / 2.0)
            radial = circle_radius * density_scale

        # Post-capture stability: the captured electron enters a state
        # whose stability depends on pairing. Each spin-down electron in
        # the captured state (n+1) costs cos²_p of stability.
        # At closure (n+1 = cap): stability = 1.0 (maximum, octet complete).
        # Pre-Hund (n+1 ≤ prime): stability = 1.0 (all same spin).
        # Post-Hund (n+1 > prime): stability = C_p^(n_down_after/prime).
        pairing_damp = 1.0
        if self.prime > 1:
            n_after = self.n_occupied + 1
            if n_after < self.capacity:  # not closure
                n_down_after = max(0, n_after - self.prime)
                if n_down_after > 0:
                    c_p = 1.0 - _COUPLING.get(self.prime, S3)
                    pairing_damp = c_p ** (n_down_after / self.prime)

        result = coupling * k * ex * cl_norm * radial * pairing_damp

        # s-block spin-doubling (insight #17): the s-pair capture from
        # a core-screened atom has 2 degenerate spin channels (↑ and ↓).
        # Factor 1/s = 2.  For H (per=1, bare nucleus), only 1 channel
        # is open — the spin is already fixed by the nuclear field.
        if self.prime <= 1 and self.n_occupied == 1 and self.per >= 2:
            result /= S_HALF
            # Per-dependent core modulation (insights #17, #20, #42, #46):
            # per=2: 1s² template assists 2s-pair capture → (1+δ₃×s)
            # per=3: 3-loop hexagonal vertex barrier → (1-δ₃/P₁)
            #        (reduced from old (1-D5×s) — the 3-loop screening
            #        now handles most of the radial coupling)
            # per≥4: core polarisation → (1+(δ₃×s²+δ₅²×s)×(per-3))
            #        (2-loop + 3-loop radial coupling to inner d-shells)
            if self.per == 2:
                result *= (1.0 + _D3 * S_HALF)
            elif self.per == 3:
                result *= (1.0 - _D3 / P1)
            elif self.per >= 4:
                result *= (1.0 + (_D3 * S_HALF ** 2
                                  + _D5 * _D5 * S_HALF) * (self.per - 3))

        return result

    def ejection_amplitude(self) -> float:
        """IE-like amplitude: cost of leaving vertex n.

        Unified PT formula (insight #30):

            ejection = 1 + sym_mod × A
            A = δ²s × (I_Fisher - I_propagator) × cos²₃^tier

        Three PT quantities compose the coefficient:
            I_Fisher      = s / sin²_p    (Fisher information per sieve channel)
            I_propagator  = 2 × |sym_raw| (self-energy back-reaction on raw D_KL)
            cos²₃^tier    = C₃^((per-2)/2) (propagator attenuation per sieve level)

        The sym entering the product is AC-only (DC removed, pairing attenuated):
            sym_mod = polygon_symmetry - DC, then × sin²_self if negative

        The sym entering I_propagator is the RAW polygon_symmetry (pairing-
        attenuated but NOT DC-removed), because the self-energy acts on the
        full D_KL profile, not the screened AC residual.

        δ²s = δ_p² × s is the universal expansion parameter (gap² × spin).
        """
        if self.n_occupied <= 0:
            return 0.0
        if self.prime <= 1:
            return 1.0  # s-block: no exchange structure

        sym_raw = self.polygon_symmetry()

        # ── GFT k=1 Koide projection (anti-double-counting, insight #33) ──
        # The k=1 Fourier mode of polygon_symmetry is partially in the
        # screening DFT (peer_values → synthesis).  The GFT partition
        # D_KL + H = log₂(m) splits as Q_Koide = 2/3 (screening/geometry)
        # and 1-Q = 1/3 (ejection/coupling).  Only 2/3 of k=1 is in the
        # screening; we subtract that fraction to avoid double-counting.
        # Applied only to p-block (P₁=3) where phases are aligned.
        p = self.prime
        N = 2 * p
        if p == P1 and p in _K1_COEFFS:
            a1, b1 = _K1_COEFFS[p]
            omega = 2.0 * _math.pi / N
            k1_at_nf = a1 * _math.cos(omega * self.n_occupied) + b1 * _math.sin(omega * self.n_occupied)
            sym_raw -= (2.0 / 3.0) * k1_at_nf  # Q_Koide = 2/3

        coupling_self = _COUPLING.get(self.prime, S3)
        coupling_next = _EXCHANGE_WEIGHT.get(self.prime, S3)
        w_pair = coupling_self / coupling_next if coupling_next > 0 else 1.0
        dc = (self.prime - 1) * (1.0 - w_pair / 2.0) / (2 * self.prime - 1)

        # ── sym_mod: AC-only + pairing attenuation (enters the product) ──
        sym_mod = sym_raw - dc
        if sym_mod < 0:
            sym_mod *= coupling_self

        # ── sym_prop: raw + pairing (enters I_propagator) ──
        sym_prop = sym_raw
        if sym_prop < 0:
            sym_prop *= coupling_self

        # ── Unified coefficient ──
        delta = (1.0 - _Q ** p) / p
        sin2 = delta * (2.0 - delta)
        d2s = delta ** 2 * S_HALF              # expansion parameter
        I_Fisher = S_HALF / sin2               # Fisher information per channel
        I_prop = 2.0 * abs(sym_prop)           # propagator self-energy
        tier = max(0.0, (self.per - 2)) / 2.0
        damp = _C3 ** tier if tier > 0.0 else 1.0  # sieve propagator

        A = d2s * (I_Fisher - I_prop) * damp

        return 1.0 + sym_mod * A


# ── Inter-layer coupling strengths ───────────────────────────────────────────

def _creation_coupling(layer_idx: int) -> float:
    """Inter-layer coupling for layer transitions.

    s→p (idx=0): sqrt(1+S3) - 1
    p→d (idx=1): sqrt(1+S5) - 1
    d→f (idx=2): sqrt(1+S7) - 1
    f→  (idx=3): 0.0
    """
    if layer_idx == 0:
        return math.sqrt(1.0 + S3) - 1.0
    if layer_idx == 1:
        return math.sqrt(1.0 + S5) - 1.0
    if layer_idx == 2:
        return math.sqrt(1.0 + S7) - 1.0
    return 0.0


# ── Composite figure names ────────────────────────────────────────────────────

_FIGURE_NAMES: dict[int, str] = {
    1: "point",
    4: "tetrahedron",
    9: "9-cell",
    16: "tesseract",
}


@dataclass(frozen=True)
class AtomicShell:
    """Frozen dataclass representing the complete shell structure of atom Z.

    Parameters
    ----------
    Z:
        Atomic number.
    polygons:
        Four ShellPolygon instances for s, p, d, f subshells.
    """

    Z: int
    polygons: Tuple[ShellPolygon, ShellPolygon, ShellPolygon, ShellPolygon]

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def active_polygon(self) -> ShellPolygon:
        """The first polygon with 0 < n_occupied < capacity.

        If all occupied polygons are saturated, returns the last occupied one.
        """
        last_occupied = self.polygons[0]
        for poly in self.polygons:
            if poly.n_occupied > 0:
                last_occupied = poly
            if 0 < poly.n_occupied < poly.capacity:
                return poly
        return last_occupied

    @property
    def saturated_polygons(self) -> Tuple[ShellPolygon, ...]:
        """Tuple of fully-closed polygons below the active polygon."""
        active = self.active_polygon
        result = []
        for poly in self.polygons:
            if poly is active:
                break
            if poly.n_occupied == poly.capacity:
                result.append(poly)
        return tuple(result)

    @property
    def n_total(self) -> int:
        """Total number of electrons across all subshells."""
        return sum(p.n_occupied for p in self.polygons)

    # ── Methods ───────────────────────────────────────────────────────────────

    def creation_coupling(self, layer_idx: int) -> float:
        """Inter-layer coupling for layer at layer_idx."""
        return _creation_coupling(layer_idx)

    def cross_gap_correction(self) -> float:
        """Cross-gap intercouche correction for the active polygon.

        PT: the cross-gaps R_ij = δ_i × δ_j couple non-adjacent layers.
        These modulate the ejection of the active layer by providing
        additional binding from deeper shells.

        R₃₅ = δ₃×δ₅ : p↔d coupling
        R₃₇ = δ₃×δ₇ : p↔f coupling
        R₅₇ = δ₅×δ₇ : d↔f coupling

        The correction is the sum of cross-gaps between the active layer
        and ALL saturated layers (not just adjacent). Each cross-gap
        INCREASES the binding → correction > 0 → IE increases.

        Returns a multiplicative factor ≥ 1.0.
        """
        from ptc.constants import D3, D5, D7

        # Map layer index to its gap δ_p
        _DELTA = {0: D3, 1: D3, 2: D5, 3: D7}  # s uses D3 (entry into sieve)

        active = self.active_polygon
        active_idx = None
        for i, poly in enumerate(self.polygons):
            if poly is active:
                active_idx = i
                break
        if active_idx is None or active_idx == 0:
            return 1.0

        delta_active = _DELTA.get(active_idx, D3)
        cross_sum = 0.0
        for i, poly in enumerate(self.saturated_polygons):
            delta_sat = _DELTA.get(i, D3)
            # Cross-gap = product of gaps × spin factor × layer attenuation.
            # Each intermediate layer between sat and active attenuates by C₃.
            dist = max(0, active_idx - i - 1)
            atten = (1.0 - _COUPLING[P1]) ** dist  # C₃^dist
            cross_sum += delta_active * delta_sat * S_HALF * atten

        return 1.0 + cross_sum

    def stack_memory(self) -> float:
        """Product of (1 + creation_coupling(i)) for each saturated layer."""
        result = 1.0
        for i, poly in enumerate(self.polygons):
            if poly in self.saturated_polygons:
                result *= 1.0 + _creation_coupling(i)
        return result

    def composite_figure(self) -> str:
        """Name based on total orbital modes (n_layers²).

        Counts how many subshells have at least one electron, squares it,
        and maps to the geometric figure name.
        """
        n_layers = sum(1 for p in self.polygons if p.n_occupied > 0)
        n_modes = n_layers * n_layers
        return _FIGURE_NAMES.get(n_modes, f"{n_modes}-cell")

    def composite_polygon(self) -> ShellPolygon:
        """Weighted composite polygon — layers attenuated by sieve depth.

        PT: deeper layers contribute less to the active electron's geometry.
        Each layer of distance from the active layer is attenuated by sin²_p
        (transmission through one sieve level):
          active layer:  weight = 1.0
          1 layer below: weight = sin²₃ ≈ 0.22
          2 layers:      weight = sin²₃ × sin²₅ ≈ 0.043
          3 layers:      weight = sin²₃ × sin²₅ × sin²₇ ≈ 0.007

        The composite uses the ACTIVE polygon's prime (shape preserved)
        with a weighted fill fraction that reflects multi-layer context.
        """
        # Find active layer index
        active_idx = 0
        for i, poly in enumerate(self.polygons):
            if poly.n_occupied > 0:
                active_idx = i
            if 0 < poly.n_occupied < poly.capacity:
                active_idx = i
                break

        # Attenuation per layer of distance: sin²_p cascade
        _ATTEN = [S3, S5, S7]  # attenuation constants per step

        total_occ = 0.0
        total_cap = 0.0
        for i, poly in enumerate(self.polygons):
            if poly.capacity <= 0:
                continue
            dist = abs(i - active_idx)
            w = 1.0
            for d in range(dist):
                w *= _ATTEN[min(d, len(_ATTEN) - 1)]
            total_occ += w * poly.n_occupied
            total_cap += w * poly.capacity

        if total_cap <= 0:
            return self.polygons[active_idx]

        # Use active polygon's prime for shape, weighted fill for content
        active = self.polygons[active_idx]
        fill_eff = total_occ / total_cap
        n_eff = max(0, min(round(fill_eff * active.capacity), active.capacity))
        return ShellPolygon(prime=active.prime, n_occupied=n_eff)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_atomic_shell(Z: int) -> AtomicShell:
    """Build an AtomicShell for atomic number Z.

    Uses block_of, l_of, n_fill, ns_config from ptc.periodic to determine
    the occupation of each subshell (s, p, d, f).
    """
    from ptc.periodic import l_of, n_fill, ns_config, period as _period

    l = l_of(Z)
    nf = n_fill(Z)
    n_s = ns_config(Z)
    per = _period(Z)

    _f_buried = False
    if l == 0:
        n_p = 0; n_d = 0; n_f = 0
    elif l == 1:
        n_s = 2; n_p = nf; n_d = 0; n_f = 0
    elif l == 2:
        n_s = n_s; n_p = 2 * P1; n_d = max(nf, 1); n_f = 0
        # Madelung: ns_config returns 0 (Pd) or 1 (Cr/Cu/Au) for promoted
        # elements.  The s-polygon reflects the ACTUAL s-occupancy.
        # La/Ac anomaly: n_fill returns 0 for the first d-element of
        # per≥6, but La=[Xe]5d1 6s2 and Ac=[Rn]6d1 7s2 have 1 d-electron.
        # f-shell awareness: post-f-block d-elements (nf>0, per≥6) have
        # f14 buried beneath the d-shell.  Pre-f (La, Ac) have nf=0.
        _f_buried = (per >= 6 and nf > 0)
    else:
        n_s = 2; n_p = 6; n_d = 2 * P2; n_f = nf
        _f_buried = False

    polygons = (
        ShellPolygon(1, n_s, per=per),
        ShellPolygon(P1, n_p, per=per),
        ShellPolygon(P2, n_d, per=per, f_buried=_f_buried),
        ShellPolygon(P3, n_f, per=per),
    )
    return AtomicShell(Z=Z, polygons=polygons)
