"""Unit tests for ptc.dblock — shared d-block primitives."""
import pytest
from ptc.dblock import (
    DBlockState,
    pi_gate, vacancy_fraction, mertens_factor, d_crt_screening,
    sd_hybrid_eps, d10_ionic_contraction, cfse_energy, dark_modes,
    half_fill_exchange, d10s2_ie2_correction,
)


class TestDBlockState:
    def test_hydrogen_not_dblock(self):
        s = DBlockState.from_Z(1)
        assert s.is_d is False
        assert s.nd == 0
        assert s.ns == 1

    def test_carbon_not_dblock(self):
        s = DBlockState.from_Z(6)
        assert s.is_d is False
        assert s.l == 1

    def test_scandium_early_d(self):
        s = DBlockState.from_Z(21)
        assert s.is_d is True
        assert s.nd == 1
        assert s.ns == 2
        assert s.per == 4
        assert s.is_early is True
        assert s.is_d5 is False
        assert s.is_d10 is False
        assert s.is_ns1 is False

    def test_chromium_d5s1(self):
        s = DBlockState.from_Z(24)
        assert s.is_d is True
        assert s.nd == 5
        assert s.ns == 1
        assert s.is_d5 is True
        assert s.is_d5s1 is True
        assert s.is_d5s2 is False
        assert s.is_ns1 is True

    def test_manganese_d5s2(self):
        s = DBlockState.from_Z(25)
        assert s.is_d is True
        assert s.nd == 5
        assert s.ns == 2
        assert s.is_d5 is True
        assert s.is_d5s1 is False
        assert s.is_d5s2 is True
        assert s.is_ns1 is False

    def test_iron_late_d(self):
        s = DBlockState.from_Z(26)
        assert s.is_d is True
        assert s.nd == 6
        assert s.is_late is True
        assert s.is_early is False

    def test_copper_d10s1(self):
        s = DBlockState.from_Z(29)
        assert s.is_d is True
        assert s.nd == 10
        assert s.ns == 1
        assert s.is_d10 is True
        assert s.is_d10s2 is False
        assert s.is_ns1 is True

    def test_zinc_d10s2(self):
        s = DBlockState.from_Z(30)
        assert s.is_d is True
        assert s.nd == 10
        assert s.ns == 2
        assert s.is_d10 is True
        assert s.is_d10s2 is True
        assert s.ie2 == pytest.approx(17.96, abs=0.01)

    def test_palladium_d10s0(self):
        s = DBlockState.from_Z(46)
        assert s.is_d is True
        assert s.nd == 10
        assert s.ns == 0
        assert s.is_d10 is True
        assert s.is_d10s2 is False
        assert s.is_ns1 is True  # Pd IS in NS1 (s→d promotion set)

    def test_gold_5d(self):
        s = DBlockState.from_Z(79)
        assert s.is_d is True
        assert s.nd == 10
        assert s.ns == 1
        assert s.per == 6
        assert s.is_ns1 is True
        assert s.ie2 == pytest.approx(20.5, abs=0.1)

    def test_cached_identity(self):
        """Same Z returns same object (frozen, cached)."""
        a = DBlockState.from_Z(26)
        b = DBlockState.from_Z(26)
        assert a is b


class TestSimplePrimitives:
    def test_pi_gate_bo_1(self):
        from ptc.constants import S5
        assert pi_gate(1.0) == pytest.approx(S5)

    def test_pi_gate_bo_2(self):
        assert pi_gate(2.0) == pytest.approx(1.0)

    def test_pi_gate_bo_1_5(self):
        assert pi_gate(1.5) == pytest.approx(0.5)

    def test_vacancy_fraction_scandium(self):
        assert vacancy_fraction(DBlockState.from_Z(21)) == pytest.approx(0.9)

    def test_vacancy_fraction_zinc(self):
        assert vacancy_fraction(DBlockState.from_Z(30)) == pytest.approx(0.0)

    def test_vacancy_fraction_iron(self):
        assert vacancy_fraction(DBlockState.from_Z(26)) == pytest.approx(0.4)

    def test_vacancy_fraction_non_dblock(self):
        assert vacancy_fraction(DBlockState.from_Z(6)) == pytest.approx(0.0)

    def test_mertens_factor_5(self):
        assert mertens_factor(5) == pytest.approx(5.0 / 16.0)

    def test_mertens_factor_1(self):
        assert mertens_factor(1) == pytest.approx(1.0)

    def test_mertens_factor_0(self):
        assert mertens_factor(0) == pytest.approx(0.0)

    def test_mertens_factor_3(self):
        assert mertens_factor(3) == pytest.approx(3.0 / 4.0)

    def test_d_crt_screening_scandium(self):
        from ptc.constants import C5, P2
        expected = C5 ** (1.0 / P2)
        assert d_crt_screening(DBlockState.from_Z(21)) == pytest.approx(expected, rel=1e-10)

    def test_d_crt_screening_zinc(self):
        from ptc.constants import C5, P2
        expected = C5 ** (10.0 / P2)
        assert d_crt_screening(DBlockState.from_Z(30)) == pytest.approx(expected, rel=1e-10)

    def test_d_crt_screening_non_dblock(self):
        assert d_crt_screening(DBlockState.from_Z(6)) == pytest.approx(1.0)


class TestDuplicatedPrimitives:
    def test_sd_hybrid_iron(self):
        dbs = DBlockState.from_Z(26)
        result = sd_hybrid_eps(dbs, 7.9024)
        assert result != pytest.approx(7.9024)  # rotation changes eps

    def test_sd_hybrid_scandium(self):
        dbs = DBlockState.from_Z(21)
        assert sd_hybrid_eps(dbs, 6.56) == pytest.approx(6.56)

    def test_sd_hybrid_zinc(self):
        dbs = DBlockState.from_Z(30)
        assert sd_hybrid_eps(dbs, 9.39) == pytest.approx(9.39)

    def test_sd_hybrid_non_dblock(self):
        dbs = DBlockState.from_Z(6)
        assert sd_hybrid_eps(dbs, 11.26) == pytest.approx(11.26)

    def test_d10_ionic_contraction_zinc(self):
        from ptc.constants import C5
        assert d10_ionic_contraction(DBlockState.from_Z(30)) == pytest.approx(C5)

    def test_d10_ionic_contraction_copper(self):
        from ptc.constants import C5
        assert d10_ionic_contraction(DBlockState.from_Z(29)) == pytest.approx(C5)

    def test_d10_ionic_contraction_iron(self):
        assert d10_ionic_contraction(DBlockState.from_Z(26)) == pytest.approx(1.0)

    def test_d10_ionic_contraction_non_dblock(self):
        assert d10_ionic_contraction(DBlockState.from_Z(8)) == pytest.approx(1.0)


class TestCFSEAndDark:
    def test_cfse_iron(self):
        result = cfse_energy(DBlockState.from_Z(26), f_lp=1.0)
        assert result > 0

    def test_cfse_zinc(self):
        assert cfse_energy(DBlockState.from_Z(30), f_lp=1.0) == pytest.approx(0.0)

    def test_cfse_non_dblock(self):
        assert cfse_energy(DBlockState.from_Z(8), f_lp=1.0) == pytest.approx(0.0)

    def test_dark_modes_iron_iron(self):
        dbs = DBlockState.from_Z(26)
        assert isinstance(dark_modes(dbs, dbs), float)

    def test_dark_modes_non_dblock(self):
        dbs = DBlockState.from_Z(1)
        assert dark_modes(dbs, dbs) == pytest.approx(0.0)


class TestExchangeAndIE2:
    def test_half_fill_exchange_d5_H(self):
        result = half_fill_exchange(DBlockState.from_Z(25), DBlockState.from_Z(1), bo=1)
        assert result > 0

    def test_half_fill_exchange_non_d5(self):
        assert half_fill_exchange(DBlockState.from_Z(26), DBlockState.from_Z(1), bo=1) == pytest.approx(0.0)

    def test_d10s2_ie2_zinc(self):
        result = d10s2_ie2_correction(DBlockState.from_Z(30), q_rel=0.3, r_ion=2.0)
        assert result > 0

    def test_d10s2_ie2_zinc_high_q(self):
        assert d10s2_ie2_correction(DBlockState.from_Z(30), q_rel=0.6, r_ion=2.0) == pytest.approx(0.0)

    def test_d10s2_ie2_iron(self):
        assert d10s2_ie2_correction(DBlockState.from_Z(26), q_rel=0.3, r_ion=2.0) == pytest.approx(0.0)
