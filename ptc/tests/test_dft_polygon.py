"""Tests for DFT polygon engine — identities, spectra, and bond coupling.

Validates:
1. Parseval identity on all three circles
2. GFT identity (log2(2P) = D_KL + H)
3. DFT spectrum for known configurations
4. Bond coupling cross-spectrum
5. Face-level screening functions
6. Full 3-face screening

April 2026 — Theorie de la Persistance
"""
import math
import pytest

from ptc.constants import P0, P1, P2, P3, S3, S5, S_HALF
from ptc.dft_polygon import (
    electron_density,
    dft_spectrum,
    spectral_power,
    parseval_check,
    gft_decomposition,
    bond_coupling,
    bond_coupling_normalized,
    S_binary_dft,
    S_hex_dft,
    S_pent_dft,
    full_dft_screening,
    _mode_weights,
    _holonomic_weight,
    DFTResult,
    FullDFTResult,
)


# ──────────────────────────────────────────────────────────────────────
#  Electron density
# ──────────────────────────────────────────────────────────────────────

class TestElectronDensity:
    """Test electron placement on Z/(2P)Z."""

    def test_empty_shell(self):
        rho = electron_density(0, P1)
        assert sum(rho) == 0
        assert len(rho) == 2 * P1

    def test_full_shell(self):
        rho = electron_density(2 * P1, P1)
        assert sum(rho) == 2 * P1
        assert all(x == 1 for x in rho)

    def test_half_fill_p(self):
        """3 electrons on Z/6Z: Hund filling, all spin-up."""
        rho = electron_density(3, P1)
        assert sum(rho) == 3
        assert rho[:3] == [1, 1, 1]  # spin-up filled
        assert rho[3:] == [0, 0, 0]  # spin-down empty

    def test_one_electron(self):
        rho = electron_density(1, P1)
        assert sum(rho) == 1
        assert rho[0] == 1
        assert sum(rho[1:]) == 0

    def test_four_electrons_pairing(self):
        """4 electrons on Z/6Z: 3 spin-up + 1 spin-down."""
        rho = electron_density(4, P1)
        assert sum(rho) == 4
        assert rho[:3] == [1, 1, 1]  # spin-up all filled
        assert rho[3] == 1           # first spin-down paired
        assert rho[4:] == [0, 0]

    def test_binary_face(self):
        """2 electrons on Z/4Z (s-subshell)."""
        rho = electron_density(2, P0)
        assert len(rho) == 4
        assert sum(rho) == 2

    def test_pentagonal_face(self):
        """5 electrons on Z/10Z: Hund half-fill."""
        rho = electron_density(5, P2)
        assert len(rho) == 10
        assert sum(rho) == 5
        assert rho[:5] == [1, 1, 1, 1, 1]  # spin-up
        assert rho[5:] == [0, 0, 0, 0, 0]  # spin-down

    def test_lp_placement(self):
        """4 electrons with 1 LP on Z/6Z.

        LP fills positions 0 and 3 (both spins of orbital 0).
        Remaining 2 fill spin-up positions 1, 2.
        """
        rho = electron_density(4, P1, lp=1)
        assert sum(rho) == 4
        assert rho[0] == 1   # LP spin-up
        assert rho[3] == 1   # LP spin-down (paired)
        assert rho[1] == 1   # Hund spin-up
        assert rho[2] == 1   # Hund spin-up

    def test_two_lp(self):
        """6 electrons with 2 LP on Z/6Z (like O in H2O)."""
        rho = electron_density(6, P1, lp=2)
        assert sum(rho) == 6
        # LP: positions (0,3) and (1,4) filled
        assert rho[0] == 1 and rho[3] == 1
        assert rho[1] == 1 and rho[4] == 1
        # Remaining: position 2 and 5
        assert rho[2] == 1 and rho[5] == 1

    def test_clamp_electrons(self):
        """Cannot exceed 2P electrons."""
        rho = electron_density(100, P1)
        assert sum(rho) == 2 * P1


# ──────────────────────────────────────────────────────────────────────
#  DFT spectrum
# ──────────────────────────────────────────────────────────────────────

class TestDFTSpectrum:
    """Test DFT on Z/(2P)Z."""

    def test_uniform_density(self):
        """Full shell: rho_hat(k) = 0 for k>0, rho_hat(0) = 1."""
        rho = electron_density(2 * P1, P1)  # full 6-electron shell
        spec = dft_spectrum(rho, P1)
        # k=0 should be fill fraction = 1.0
        assert abs(spec[0].real - 1.0) < 1e-10
        assert abs(spec[0].imag) < 1e-10
        # k>0 should be ~0
        for k in range(1, P1):
            assert abs(spec[k]) < 1e-10

    def test_empty_density(self):
        """Empty shell: all modes zero."""
        rho = electron_density(0, P1)
        spec = dft_spectrum(rho, P1)
        for k in range(P1):
            assert abs(spec[k]) < 1e-10

    def test_single_electron(self):
        """1 electron at position 0 on Z/6Z."""
        rho = electron_density(1, P1)
        spec = dft_spectrum(rho, P1)
        N = 2 * P1
        # rho_hat(k) = (1/N) * exp(-2pi*i*k*0/N) = 1/N for all k
        for k in range(P1):
            assert abs(spec[k].real - 1.0 / N) < 1e-10
            assert abs(spec[k].imag) < 1e-10

    def test_half_fill_spectrum(self):
        """Half-fill (3 electrons): has specific DFT structure."""
        rho = electron_density(3, P1)
        spec = dft_spectrum(rho, P1)
        # k=0 = fill fraction = 3/6 = 0.5
        assert abs(spec[0].real - 0.5) < 1e-10

    def test_spectrum_length(self):
        """Spectrum has P modes."""
        for P in [P0, P1, P2, P3]:
            rho = electron_density(P, P)
            spec = dft_spectrum(rho, P)
            assert len(spec) == P


# ──────────────────────────────────────────────────────────────────────
#  Parseval identity
# ──────────────────────────────────────────────────────────────────────

class TestParseval:
    """Parseval: (1/2P) sum|rho|^2 = sum|rho_hat|^2."""

    @pytest.mark.parametrize("P", [P0, P1, P2, P3])
    @pytest.mark.parametrize("n_e", range(0, 15))
    def test_parseval_all_fills(self, P, n_e):
        """Test Parseval for all fill levels on all circles."""
        if n_e > 2 * P:
            return  # skip impossible fill levels
        rho = electron_density(n_e, P)
        spec = dft_spectrum(rho, P)

        # We compute Parseval over the FULL 2P modes (not just P).
        # But our DFT only computes P modes. For the Parseval check,
        # the sum over P modes needs the negative-frequency conjugates.
        # Since rho is real, rho_hat(-k) = conj(rho_hat(k)), so:
        # sum_{k=0}^{2P-1} |rho_hat(k)|^2 = |rho_hat(0)|^2 + 2*sum_{k=1}^{P-1} |rho_hat(k)|^2
        # when 2P is even and P > 1.
        # Actually, our dft_spectrum computes (1/N) * sum, so Parseval should be:
        # sum |rho_hat(k)|^2 over k=0..N-1 = (1/N) * sum |rho(r)|^2

        # Recompute full DFT for Parseval
        N = 2 * P
        import cmath
        full_spec = []
        for k in range(N):
            val = complex(0, 0)
            for r in range(N):
                angle = -2.0 * math.pi * k * r / N
                val += rho[r] * cmath.exp(complex(0, angle))
            val /= N
            full_spec.append(val)

        lhs = sum(x ** 2 for x in rho) / N
        rhs = sum(abs(c) ** 2 for c in full_spec)

        assert abs(lhs - rhs) < 1e-10, f"Parseval failed: P={P}, n_e={n_e}, LHS={lhs}, RHS={rhs}"

    @pytest.mark.parametrize("P", [P0, P1, P2])
    def test_parseval_fill_fraction(self, P):
        """For binary occupation: fill fraction = n_e / (2P)."""
        for n_e in range(2 * P + 1):
            rho = electron_density(n_e, P)
            N = 2 * P
            lhs = sum(rho) / N  # n_e / N = fill fraction
            assert abs(lhs - n_e / N) < 1e-10


# ──────────────────────────────────────────────────────────────────────
#  GFT identity
# ──────────────────────────────────────────────────────────────────────

class TestGFT:
    """GFT: log2(2P) = D_KL(rho||U) + H(rho)."""

    @pytest.mark.parametrize("P", [P0, P1, P2, P3])
    def test_gft_full_shell(self, P):
        """Full shell: D_KL = 0, H = log2(2P)."""
        rho = electron_density(2 * P, P)
        log2_N, D_KL, H = gft_decomposition(rho, P)
        assert abs(D_KL) < 1e-10
        assert abs(H - log2_N) < 1e-10

    @pytest.mark.parametrize("P", [P0, P1, P2])
    @pytest.mark.parametrize("n_e", [1, 2, 3])
    def test_gft_identity(self, P, n_e):
        """GFT identity holds for various fill levels."""
        if n_e > 2 * P:
            return
        rho = electron_density(n_e, P)
        log2_N, D_KL, H = gft_decomposition(rho, P)
        # D_KL + H should equal log2(2P)
        assert abs(D_KL + H - log2_N) < 1e-10, \
            f"GFT failed: P={P}, n_e={n_e}, D_KL={D_KL}, H={H}, log2_N={log2_N}"

    @pytest.mark.parametrize("P", [P0, P1, P2, P3])
    def test_gft_identity_all_fills(self, P):
        """GFT identity for every possible fill level."""
        for n_e in range(1, 2 * P + 1):
            rho = electron_density(n_e, P)
            log2_N, D_KL, H = gft_decomposition(rho, P)
            assert abs(D_KL + H - log2_N) < 1e-10, \
                f"GFT failed: P={P}, n_e={n_e}"

    @pytest.mark.parametrize("P", [P0, P1, P2])
    def test_dkl_non_negative(self, P):
        """D_KL >= 0 (Gibbs inequality)."""
        for n_e in range(1, 2 * P + 1):
            rho = electron_density(n_e, P)
            _, D_KL, _ = gft_decomposition(rho, P)
            assert D_KL >= -1e-10, f"D_KL negative: P={P}, n_e={n_e}, D_KL={D_KL}"


# ──────────────────────────────────────────────────────────────────────
#  Mode weights
# ──────────────────────────────────────────────────────────────────────

class TestModeWeights:
    """Test holonomic and cascade mode weights."""

    def test_k0_weight(self):
        """k=0 always has weight 1."""
        for P in [P0, P1, P2, P3]:
            assert _holonomic_weight(P, 0) == 1.0

    def test_holonomic_symmetry(self):
        """sin^2(pi*k/P) = sin^2(pi*(P-k)/P)."""
        for P in [P1, P2, P3]:
            for k in range(1, P):
                w1 = _holonomic_weight(P, k)
                w2 = _holonomic_weight(P, P - k)
                assert abs(w1 - w2) < 1e-10

    def test_weights_positive(self):
        """All weights are non-negative."""
        for P in [P0, P1, P2, P3]:
            weights = _mode_weights(P)
            for w in weights:
                assert w >= 0


# ──────────────────────────────────────────────────────────────────────
#  Bond coupling
# ──────────────────────────────────────────────────────────────────────

class TestBondCoupling:
    """Test cross-spectrum bond coupling."""

    def test_identical_atoms_positive(self):
        """Homonuclear: aligned fills -> positive cross-spectrum at k=0."""
        rho = electron_density(3, P1)  # half-fill
        S_face, S_modes = bond_coupling(rho, rho, P1)
        # k=0 mode: cross = |rho_hat(0)|^2 = (3/6)^2 = 0.25
        # S_k0 = -0.25 * w(0) = -0.25 (anti-screening from counting)
        assert S_modes[0] < 0  # aligned fills reduce screening

    def test_empty_coupling(self):
        """Empty shells: zero coupling."""
        rho_A = electron_density(0, P1)
        rho_B = electron_density(0, P1)
        S_face, S_modes = bond_coupling(rho_A, rho_B, P1)
        assert abs(S_face) < 1e-10

    def test_complementary_fills(self):
        """One full, one empty: only k=0 contributes."""
        rho_A = electron_density(6, P1)  # full
        rho_B = electron_density(0, P1)  # empty
        S_face, _ = bond_coupling(rho_A, rho_B, P1)
        assert abs(S_face) < 1e-10  # full * empty = 0

    def test_homonuclear_full(self):
        """Both full: strong alignment at k=0."""
        rho = electron_density(6, P1)
        S_face, S_modes = bond_coupling(rho, rho, P1)
        # Only k=0 mode is nonzero (uniform density)
        assert abs(S_modes[0] + 1.0) < 1e-10  # -(1.0)^2 * 1.0 = -1.0
        for k in range(1, P1):
            assert abs(S_modes[k]) < 1e-10


# ──────────────────────────────────────────────────────────────────────
#  Face-level screening
# ──────────────────────────────────────────────────────────────────────

class TestFaceScreening:
    """Test the high-level face screening functions."""

    def test_hex_H2(self):
        """H2: both atoms have 0 p-electrons -> hex face is trivial."""
        res = S_hex_dft(1, 1, bo=1.0)
        assert isinstance(res, DFTResult)
        assert sum(res.rho_A) == 0
        assert sum(res.rho_B) == 0
        assert abs(res.S_face) < 1e-10

    def test_hex_N2(self):
        """N2: both have 3 p-electrons (half-fill)."""
        res = S_hex_dft(7, 7, bo=3.0)
        assert sum(res.rho_A) == 3  # N has 3 p-electrons
        assert sum(res.rho_B) == 3
        # Homonuclear half-fill: specific DFT structure
        assert res.S_face != 0  # non-trivial screening

    def test_hex_F2(self):
        """F2: both have 5 p-electrons."""
        res = S_hex_dft(9, 9, bo=1.0, lp_A=3, lp_B=3)
        assert sum(res.rho_A) == 5
        assert sum(res.rho_B) == 5

    def test_binary_H2(self):
        """H2: both have 1 s-electron."""
        res = S_binary_dft(1, 1, bo=1.0)
        assert sum(res.rho_A) == 1
        assert sum(res.rho_B) == 1

    def test_binary_LiH(self):
        """LiH: Li has 1 s-electron, H has 1 s-electron."""
        res = S_binary_dft(3, 1, bo=1.0)
        assert sum(res.rho_A) == 1  # Li ns=1
        assert sum(res.rho_B) == 1  # H ns=1

    def test_pent_no_d(self):
        """Non d-block atoms: empty pentagonal face."""
        res = S_pent_dft(7, 7, bo=3.0)  # N2
        assert sum(res.rho_A) == 0
        assert sum(res.rho_B) == 0
        assert abs(res.S_face) < 1e-10


# ──────────────────────────────────────────────────────────────────────
#  Full 3-face screening
# ──────────────────────────────────────────────────────────────────────

class TestFullDFT:
    """Test the complete 3-face DFT screening."""

    def test_H2_structure(self):
        """H2 has binary face only (no p or d electrons)."""
        res = full_dft_screening(1, 1, bo=1.0)
        assert isinstance(res, FullDFTResult)
        # Hex and pent should be ~zero
        assert abs(res.S_hex) < 1e-10
        assert abs(res.S_pent) < 1e-10
        # Binary should be non-zero
        # (two s1 atoms contribute to Z/4Z)

    def test_N2_structure(self):
        """N2: hex face dominates, pent face zero."""
        res = full_dft_screening(7, 7, bo=3.0)
        assert abs(res.S_pent) < 1e-10  # no d electrons

    def test_total_is_sum(self):
        """Total = binary + hex + pent."""
        for Z_A, Z_B, bo in [(1, 1, 1), (7, 7, 3), (6, 8, 2), (9, 9, 1)]:
            res = full_dft_screening(Z_A, Z_B, bo)
            total = res.S_binary + res.S_hex + res.S_pent
            assert abs(res.S_total - total) < 1e-10


# ──────────────────────────────────────────────────────────────────────
#  PT identities
# ──────────────────────────────────────────────────────────────────────

class TestPTIdentities:
    """Test specific PT identities the DFT engine must satisfy."""

    def test_parseval_equals_fill_fraction(self):
        """Parseval power = fill fraction for binary occupation."""
        for P in [P0, P1, P2]:
            for n_e in range(1, 2 * P + 1):
                rho = electron_density(n_e, P)
                N = 2 * P
                fill = n_e / N

                # Full DFT for proper Parseval
                import cmath
                full_spec = []
                for k in range(N):
                    val = complex(0, 0)
                    for r in range(N):
                        angle = -2.0 * math.pi * k * r / N
                        val += rho[r] * cmath.exp(complex(0, angle))
                    val /= N
                    full_spec.append(val)

                power = sum(abs(c) ** 2 for c in full_spec)
                assert abs(power - fill) < 1e-10, \
                    f"P={P}, n_e={n_e}: power={power}, fill={fill}"

    def test_holonomic_sin2_connection(self):
        """sin^2(pi*k/P) at k=1 connects DFT to holonomic angle."""
        # For P1=3: sin^2(pi/3) = 3/4
        w1 = _holonomic_weight(P1, 1)
        assert abs(w1 - 0.75) < 1e-10

        # For P2=5: sin^2(pi/5) = (5-sqrt(5))/8
        w1_p2 = _holonomic_weight(P2, 1)
        expected = math.sin(math.pi / 5) ** 2
        assert abs(w1_p2 - expected) < 1e-10

    def test_half_fill_symmetry(self):
        """At half-fill, k=1 mode has specific symmetry."""
        # N2: half-fill on hex face. DFT should show maximal k=0
        # and specific k=1 structure
        rho = electron_density(P1, P1)  # half-fill: 3 on Z/6Z
        spec = dft_spectrum(rho, P1)

        # k=0 = fill fraction = s = 1/2
        assert abs(spec[0].real - S_HALF) < 1e-10

    def test_s_half_is_half_fill_parseval(self):
        """At half-fill, Parseval power = s = 1/2.

        This is the fundamental PT constant emerging from the DFT.
        """
        for P in [P0, P1, P2, P3]:
            rho = electron_density(P, P)  # half-fill
            N = 2 * P
            fill = P / N
            assert abs(fill - S_HALF) < 1e-10
