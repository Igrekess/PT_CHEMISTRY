"""Tests for ie_geo — IE from polygon geometry."""
from __future__ import annotations

import pytest

from ptc.ie_geo import IE_geo_eV, benchmark_ie_geo


# ── Basic physical constraints ────────────────────────────────────────────────

def test_IE_H_approx_13_6():
    """H ionization energy must be close to 13.6 eV (within 0.1 eV)."""
    ie_h = IE_geo_eV(1)
    assert abs(ie_h - 13.6) < 0.1, f"IE(H) = {ie_h:.3f} eV, expected ~13.6 eV"


def test_IE_He_gt_H():
    """He has higher IE than H (noble gas vs alkali period)."""
    assert IE_geo_eV(2) > IE_geo_eV(1)


def test_IE_Ne_gt_F():
    """Ne (noble gas) has higher IE than F within period 2."""
    assert IE_geo_eV(10) > IE_geo_eV(9)


def test_IE_N_gt_O_hund():
    """N IE > O IE due to Hund half-fill stability (p³ vs p⁴)."""
    assert IE_geo_eV(7) > IE_geo_eV(8), (
        f"IE(N)={IE_geo_eV(7):.3f} should > IE(O)={IE_geo_eV(8):.3f}"
    )


def test_IE_positive_Z1_to_36():
    """IE must be strictly positive for all Z = 1 to 36."""
    for Z in range(1, 37):
        ie = IE_geo_eV(Z)
        assert ie > 0.0, f"IE({Z}) = {ie} is not positive"


# ── Benchmark ─────────────────────────────────────────────────────────────────

def test_benchmark_ie_geo_returns_valid_dict():
    """benchmark_ie_geo returns a dict with expected keys and valid values."""
    result = benchmark_ie_geo()
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
