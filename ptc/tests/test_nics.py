"""
test_nics.py — regression tests for ptc/nics.py and aromaticity API.

Validates:
  - Pauling-London PT formula prefactor (α²·a₀/12 ≈ 2.348 ppm·Å)
  - Hückel sign rule (4n+2 = diamagnetic, 4n = paramagnetic)
  - Group-classification electron count (σ vs π per element group)
  - NICS(z) profile decays as Biot-Savart (R²/(R²+z²)^{3/2})
  - Mol.aromaticity() / .nics() / .nics_profile() public API

Reference values are from the PT formula (no fit), with sanity-check
comparison to literature where available (benzene NICS exp ≈ −9.7).
"""

import pytest
from ptc.api import Molecule
from ptc.nics import (
    _NICS_K, _huckel_sign, _aromatic_electron_count, _f_coh_T3,
    nics_for_ring, NICSResult,
)
from ptc.constants import ALPHA_PHYS, A_BOHR


# ─── PT prefactor ──────────────────────────────────────────────────

def test_NICS_prefactor_PT_pure():
    """K_NICS = α² · a₀ / 12 in ppm·Å, derived from PT constants only."""
    expected = ALPHA_PHYS ** 2 * A_BOHR / 12.0 * 1.0e6
    assert abs(_NICS_K - expected) < 1e-9
    # numerical sanity: ~2.348 ppm·Å
    assert 2.34 < _NICS_K < 2.36


# ─── Hückel sign rule ──────────────────────────────────────────────

def test_huckel_sign_4nplus2_aromatic():
    assert _huckel_sign(2) == +1.0    # n=0
    assert _huckel_sign(6) == +1.0    # n=1, benzene
    assert _huckel_sign(10) == +1.0   # n=2
    assert _huckel_sign(14) == +1.0   # n=3


def test_huckel_sign_4n_antiaromatic():
    assert _huckel_sign(4) == -1.0    # n=1, cyclobutadiene
    assert _huckel_sign(8) == -1.0    # n=2
    assert _huckel_sign(12) == -1.0   # n=3


def test_huckel_sign_radical_partial():
    assert _huckel_sign(3) == +0.5
    assert _huckel_sign(5) == +0.5


def test_huckel_sign_zero():
    assert _huckel_sign(0) == 0.0


# ─── NICS values for canonical aromatic molecules ──────────────────

class TestAromaticBenchmarks:
    """PT-pure NICS predictions on textbook aromatics."""

    def test_benzene_diamagnetic(self):
        nics0 = Molecule("c1ccccc1").nics(z=0.0)
        # PT prediction: −8.71 ppm; exp: −9.7
        assert -10.0 < nics0 < -8.0

    def test_benzene_NICS_decays_with_z(self):
        prof = Molecule("c1ccccc1").nics_profile(zs=[0, 0.5, 1.0, 2.0])
        assert all(v < 0 for _, v in prof)         # all diamagnetic
        # Strict decay |NICS(z)| decreasing with z
        magnitudes = [abs(v) for _, v in prof]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_cyclobutadiene_paramagnetic(self):
        """4n=4 antiaromatic gives positive NICS (signed Hückel)."""
        nics0 = Molecule("C1=CC=C1").nics(z=0.0)
        assert nics0 > +5.0   # PT: +8.1, exp: +27 (model under-shoots magnitude)

    def test_S3_double_aromatic(self):
        """Cyclic-S₃ has σ=6 + π=6 (both 4n+2): strong diamagnetic."""
        aro = Molecule("[S]1[S][S]1").aromaticity()
        assert aro is not None
        assert aro.NICS_0 < -15.0  # PT: −18.5

    def test_U_S3_inherits_S3_NICS(self):
        """Cap M is exocyclic → NICS preserved from bare S₃ ring."""
        nics_bare = Molecule("[S]1[S][S]1").nics(z=0.0)
        nics_capped = Molecule("[U][S]1[S][S]1").nics(z=0.0)
        # Capping should NOT change ring NICS by more than 5%
        assert abs(nics_capped - nics_bare) / abs(nics_bare) < 0.05


# ─── Group-classification electron count ──────────────────────────

class TestElectronCount:
    """σ + π count per group rule."""

    def test_S3_count_12(self):
        """Group 16 atoms × 3 in ring = 4·3 = 12 delocalized e."""
        mol = Molecule("[S]1[S][S]1")
        assert _aromatic_electron_count(mol.topology, mol.topology.rings[0]) == 12

    def test_Al3_count_3(self):
        """Group 13 × 3 = 1·3 = 3 π electrons."""
        mol = Molecule("[Al]1[Al][Al]1")
        assert _aromatic_electron_count(mol.topology, mol.topology.rings[0]) == 3

    def test_Cu3_count_3(self):
        """Coinage s¹ × 3 = 3 σ-aromatic electrons."""
        mol = Molecule("[Cu]1[Cu][Cu]1")
        assert _aromatic_electron_count(mol.topology, mol.topology.rings[0]) == 3

    def test_Si3_count_6(self):
        """Group 14 × 3 = 2·3 = 6 σ-aromatic electrons."""
        mol = Molecule("[Si]1[Si][Si]1")
        assert _aromatic_electron_count(mol.topology, mol.topology.rings[0]) == 6


# ─── T³ Fourier coherence ──────────────────────────────────────────

class TestCoherence:
    def test_homonuclear_f_coh_unity(self):
        """All same Z → coherence = 1."""
        Zs = [16, 16, 16]
        assert abs(_f_coh_T3(Zs) - 1.0) < 1e-9

    def test_heteronuclear_below_unity(self):
        Zs = [13, 16, 13]   # Al-S-Al
        f = _f_coh_T3(Zs)
        assert 0.0 < f < 1.0


# ─── Public API on Molecule ────────────────────────────────────────

class TestMoleculeAromaticityAPI:
    def test_no_ring_returns_None(self):
        """Linear molecule should give None."""
        assert Molecule("CCO").aromaticity() is None
        assert Molecule("CCO").nics(z=0.0) is None
        assert Molecule("CCO").nics_profile() == []

    def test_aromaticity_returns_NICSResult(self):
        aro = Molecule("c1ccccc1").aromaticity()
        assert isinstance(aro, NICSResult)
        assert aro.N == 6
        assert aro.R > 1.0           # benzene radius ≈ 1.4 Å
        assert aro.f_coh == pytest.approx(1.0, abs=1e-6)

    def test_nics_profile_axial_decay(self):
        """NICS(z) should follow Biot-Savart: R²/(R²+z²)^{3/2}."""
        prof = Molecule("c1ccccc1").nics_profile(zs=[0.0, 1.0, 5.0])
        # at z=5Å far from ring, NICS should be very weak
        assert abs(prof[2][1]) < abs(prof[0][1]) / 5.0


# ─── Signature integration ─────────────────────────────────────────

class TestSignatureAPI:
    def test_signature_returns_FullSignature(self):
        from ptc.signature import FullSignature
        sig = Molecule("[S]1[S][S]1").signature()
        assert isinstance(sig, FullSignature)
        assert sig.D_at > 0
        assert sig.aromatic_class.startswith("double aromatic")
        assert sig.n_aromatic_sigma == 6
        assert sig.n_aromatic_pi == 6

    def test_capped_signature_auto_detects_cap(self):
        sig = Molecule("[U][S]1[S][S]1").signature()
        # cap_idx auto-detected = 0 (U non-ring atom)
        assert sig.cap_binding > 0
        # cap binding around 4-5 eV (η¹ over-estimate but order OK)
        assert 3.0 < sig.cap_binding < 6.0

    def test_signature_NICS_matches_aromaticity(self):
        """signature.NICS_profile should match mol.nics_profile."""
        mol = Molecule("[S]1[S][S]1")
        sig = mol.signature()
        prof_api = mol.nics_profile(zs=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
        for (z_a, v_a), (z_b, v_b) in zip(sig.NICS_profile, prof_api):
            assert abs(z_a - z_b) < 1e-6
            assert abs(v_a - v_b) < 1e-3


# ─── Cap-on-S3 universality (the big PT prediction) ───────────────

@pytest.mark.parametrize("cap_smiles", [
    "[B]",  "[C]",  "[Al]", "[Si]", "[Ga]", "[Ge]",
    "[Sn]", "[Pb]", "[Th]", "[U]",
])
def test_cap_on_S3_inherits_double_aromaticity(cap_smiles):
    """PT prediction: any cap M on S₃ inherits NICS = -18.8 (±5%)."""
    smiles = f"{cap_smiles}[S]1[S][S]1"
    nics_capped = Molecule(smiles).nics(z=0.0)
    nics_bare = Molecule("[S]1[S][S]1").nics(z=0.0)
    assert nics_capped < -15.0
    assert abs(nics_capped - nics_bare) / abs(nics_bare) < 0.05
