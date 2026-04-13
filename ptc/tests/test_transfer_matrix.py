"""Precision tests for PTC transfer matrix engine.

Each test asserts D_at is within tolerance of experimental value.
No mechanism names or internal decomposition tested — only physics.
These tests define the CONTRACT the new SCF engine must satisfy.
"""
import pytest
import statistics
from ptc.data.molecules import MOLECULES
from ptc.data.molecules_extended import MOLECULES_EXT
from ptc.data.molecules_atct import MOLECULES_ATCT
from ptc.topology import build_topology
from ptc.transfer_matrix import compute_D_at_transfer


ALL_MOLS = {**MOLECULES, **MOLECULES_EXT, **MOLECULES_ATCT}


def _percent_err(name: str):
    d = ALL_MOLS[name]
    topo = build_topology(d['smiles'])
    result = compute_D_at_transfer(topo)
    return abs((result.D_at - d['D_at']) / d['D_at'] * 100), result


# ── Global benchmark ──

def test_global_mae_below_2_percent():
    """MAE across all 700 molecules must be below 2%."""
    errors = []
    for name, d in ALL_MOLS.items():
        topo = build_topology(d['smiles'])
        r = compute_D_at_transfer(topo)
        errors.append(abs((r.D_at - d['D_at']) / d['D_at'] * 100))
    mae = statistics.mean(errors)
    assert mae < 2.0, f"MAE {mae:.3f}% exceeds 2% threshold"


def test_no_molecule_above_10_percent():
    """No molecule should have error > 10%."""
    for name, d in ALL_MOLS.items():
        topo = build_topology(d['smiles'])
        r = compute_D_at_transfer(topo)
        err = abs((r.D_at - d['D_at']) / d['D_at'] * 100)
        assert err < 10.0, f"{name}: {err:.2f}% exceeds 10%"


def test_at_least_250_below_1_percent():
    """At least 250 molecules should be below 1% error."""
    count = 0
    for name, d in ALL_MOLS.items():
        topo = build_topology(d['smiles'])
        r = compute_D_at_transfer(topo)
        err = abs((r.D_at - d['D_at']) / d['D_at'] * 100)
        if err < 1.0:
            count += 1
    assert count >= 250, f"Only {count} molecules below 1%"


# ── Key molecule precision tests ──

@pytest.mark.parametrize("name,max_err", [
    # Diatomics
    ("H2", 3.0),
    ("HF", 3.0),
    ("HCl", 3.0),
    ("HBr", 3.0),
    ("HI", 3.0),
    ("N2", 3.0),
    ("O2", 6.0),
    ("CO", 3.0),
    ("NaCl", 3.0),
    ("LiF", 3.0),
    # Small polyatomics
    ("H2O", 3.0),
    ("NH3", 5.0),
    ("CH4", 5.0),
    ("CO2", 3.0),
    ("H2CO", 5.0),
    ("methanol", 3.0),
    # Aromatics
    ("benzene", 3.0),
    ("naphthalene", 3.0),
    ("pyridine", 3.0),
    # Halides
    ("CH3Cl", 5.0),
    ("CF4", 5.0),
    # Larger
    ("ethanol", 3.0),
    ("acetic_acid", 3.0),
    ("acetone", 3.0),
    ("DMF", 5.0),
])
def test_molecule_precision(name, max_err):
    if name not in ALL_MOLS:
        pytest.skip(f"{name} not in database")
    err, result = _percent_err(name)
    assert err < max_err, f"{name}: {err:.2f}% exceeds {max_err}%"


# ── Structural invariants ──

def test_result_has_required_fields():
    """TransferResult must have D_at and E_spectral."""
    topo = build_topology("O")  # H2O
    r = compute_D_at_transfer(topo)
    assert hasattr(r, 'D_at')
    assert hasattr(r, 'E_spectral')
    assert isinstance(r.D_at, float)
    assert r.D_at > 0


def test_diatomic_no_spectral():
    """Diatomic E_spectral should be 0 (no multi-centre)."""
    topo = build_topology("[H][H]")
    r = compute_D_at_transfer(topo)
    assert r.E_spectral == 0.0


def test_d_at_positive_for_all():
    """D_at must be positive for every molecule in the database."""
    for name, d in ALL_MOLS.items():
        topo = build_topology(d['smiles'])
        r = compute_D_at_transfer(topo)
        assert r.D_at > 0, f"{name}: D_at={r.D_at} is not positive"


def test_bit_exact_hash():
    """Bit-exact regression guard for the 4-face refactor."""
    import hashlib
    results = []
    for name, d in sorted(ALL_MOLS.items()):
        topo = build_topology(d['smiles'])
        r = compute_D_at_transfer(topo)
        results.append(f"{name}|{r.D_at:.10f}")
    h = hashlib.sha256("\n".join(results).encode()).hexdigest()[:16]
    assert h == "dc3e31340439e929", f"Hash mismatch: {h} != dc3e31340439e929"
