"""Tests for base infrastructure: AtomProvider, Result, BaseCalculator."""

import pytest

from ptc.base import AtomProvider, BaseCalculator, Result


def test_atom_provider_default_source():
    p = AtomProvider()
    assert p.source == "full_pt"


def test_atom_provider_full_nist_source():
    p = AtomProvider(source="full_nist")
    assert p.source == "full_nist"


def test_atom_provider_hybrid_source():
    p = AtomProvider(source="hybrid")
    assert p.source == "hybrid"


def test_atom_provider_accepts_legacy_aliases():
    assert AtomProvider(source="pt").source == "full_pt"
    assert AtomProvider(source="nist").source == "full_nist"
    assert AtomProvider(source="exp").source == "full_nist"


def test_atom_provider_overrides():
    p = AtomProvider(overrides={6: {"IE": 99.0}})
    assert p.overrides[6]["IE"] == 99.0


def test_atom_provider_resolve_ie_override():
    p = AtomProvider(overrides={6: {"IE": 99.0}})
    ie = p.resolve("IE", Z=6, pt_value=11.26, exp_value=11.260)
    assert ie == 99.0


def test_atom_provider_resolve_pt():
    p = AtomProvider(source="full_pt")
    ie = p.resolve("IE", Z=6, pt_value=11.26, exp_value=11.260)
    assert ie == 11.26


def test_atom_provider_resolve_full_nist():
    p = AtomProvider(source="full_nist")
    ie = p.resolve("IE", Z=6, pt_value=11.26, exp_value=11.260)
    assert ie == 11.260


def test_atom_provider_hybrid_defaults_to_pt():
    p = AtomProvider(source="hybrid")
    ie = p.resolve("IE", Z=6, pt_value=11.26, exp_value=11.260)
    ea = p.resolve("EA", Z=6, pt_value=1.26, exp_value=1.262)
    assert ie == 11.26
    assert ea == 1.26


def test_atom_provider_per_property():
    p = AtomProvider(source="full_pt", ie_source="full_nist")
    ie = p.resolve("IE", Z=6, pt_value=11.26, exp_value=11.260)
    ea = p.resolve("EA", Z=6, pt_value=1.26, exp_value=1.262)
    assert ie == 11.260
    assert ea == 1.26


def test_atom_provider_per_property_aliases():
    p = AtomProvider(source="hybrid", ie_source="nist", ea_source="pt")
    assert p.property_source("IE") == "full_nist"
    assert p.property_source("EA") == "full_pt"


def test_atom_provider_mass_property():
    p = AtomProvider(source="full_pt", mass_source="full_nist")
    mass = p.resolve("MASS", Z=6, pt_value=12.0, exp_value=12.011)
    assert mass == 12.011


def test_atom_provider_unknown_source_raises():
    with pytest.raises(ValueError):
        AtomProvider(source="mystery")


def test_atom_provider_unknown_property_raises():
    p = AtomProvider()
    with pytest.raises(ValueError):
        p.resolve("SPIN", Z=6, pt_value=0.0, exp_value=0.0)


def test_result():
    r = Result(value=9.511, unit="eV", label="D_at")
    assert r.value == 9.511
    assert r.unit == "eV"
    assert r.label == "D_at"
    assert r.details == {}


def test_base_calculator_uses_atom_provider():
    class DummyCalculator(BaseCalculator):
        def compute(self, **kwargs):
            return Result(1.0, label="dummy")

        def benchmark(self):
            return {"ok": True}

    calc = DummyCalculator(source="full_nist", ie_source="pt")
    assert calc.atom_provider.source == "full_nist"
    assert calc.atom_provider.property_source("IE") == "full_pt"
