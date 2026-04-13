"""Tests for ea_geo — EA from polygon geometry."""
from __future__ import annotations

import pytest

from ptc.ea_geo import EA_geo_eV, benchmark_ea_geo


# ── Noble gases: EA must be zero ─────────────────────────────────────────────

def test_noble_gas_EA_zero_He():
    """He (full s-shell) has no capture → EA = 0."""
    assert EA_geo_eV(2) == pytest.approx(0.0)


def test_noble_gas_EA_zero_Ne():
    """Ne (full p-shell) has no capture → EA = 0."""
    assert EA_geo_eV(10) == pytest.approx(0.0)


def test_noble_gas_EA_zero_Ar():
    """Ar (full p-shell) has no capture → EA = 0."""
    assert EA_geo_eV(18) == pytest.approx(0.0)


# ── Halogens: EA must be positive ─────────────────────────────────────────────

def test_halogen_EA_positive_F():
    """F (p5, 1 vacancy) should have positive EA."""
    assert EA_geo_eV(9) > 0.0


def test_halogen_EA_positive_Cl():
    """Cl (p5, 1 vacancy) should have positive EA."""
    assert EA_geo_eV(17) > 0.0


# ── Hund: N EA < O EA ────────────────────────────────────────────────────────

def test_N_EA_less_than_O_EA():
    """N (half-filled p³, low capture drive) has lower EA than O (p⁴)."""
    ea_n = EA_geo_eV(7)
    ea_o = EA_geo_eV(8)
    assert ea_n < ea_o, (
        f"EA(N)={ea_n:.4f} should be < EA(O)={ea_o:.4f} (Hund half-fill)"
    )


# ── Non-negativity ─────────────────────────────────────────────────────────────

def test_EA_nonnegative_Z1_to_36():
    """EA must be non-negative for all Z = 1 to 36."""
    for Z in range(1, 37):
        ea = EA_geo_eV(Z)
        assert ea >= 0.0, f"EA({Z}) = {ea} is negative"


# ── Benchmark ─────────────────────────────────────────────────────────────────

def test_benchmark_ea_geo_returns_valid_dict():
    """benchmark_ea_geo returns a dict with expected keys and valid values."""
    result = benchmark_ea_geo()
    assert isinstance(result, dict)
    assert "count" in result
    assert "mae_percent" in result
    assert "by_block" in result
    assert "rows" in result
    assert result["count"] > 0
    assert result["mae_percent"] >= 0.0
    assert isinstance(result["by_block"], dict)
    assert isinstance(result["rows"], list)
    assert len(result["rows"]) == result["count"]
