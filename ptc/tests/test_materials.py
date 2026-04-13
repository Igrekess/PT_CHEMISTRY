"""
test_materials.py -- Benchmark Module 4 (band gaps & materials properties).

Runs all 6 benchmark materials and prints a comparison table.
0 adjustable parameters -- everything derived from s = 1/2.
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ptc.materials import (
    band_gap_pt,
    classify_material,
    dielectric_constant_pt,
    analyze_material,
    MaterialResult,
)

# ── Benchmark targets ──
BENCHMARKS = [
    # (label, Z_A, Z_B, structure, E_gap_exp, classification_exp, eps_exp)
    ('Diamond (C)',  6,  None, 'diamond',     5.47, 'insulator',     5.7),
    ('Si',          14,  None, 'diamond',     1.12, 'semiconductor', 11.7),
    ('Ge',          32,  None, 'diamond',     0.66, 'semiconductor', 16.0),
    ('GaAs',        31,    33, 'zincblende',  1.42, 'semiconductor', 12.9),
    ('NaCl',        11,    17, 'rocksalt',    8.50, 'insulator',      5.9),
    ('MgO',         12,     8, 'rocksalt',    7.80, 'insulator',      9.8),
]


def test_band_gaps():
    """Test that all band gaps are within 10% of experiment."""
    for label, Z_A, Z_B, struct, gap_exp, _, _ in BENCHMARKS:
        gap_pt = band_gap_pt(Z_A, Z_B, struct)
        err = abs(gap_pt - gap_exp) / gap_exp
        assert err < 0.10, (
            f"{label}: PT gap = {gap_pt:.3f} eV, exp = {gap_exp:.2f} eV, "
            f"error = {err*100:.1f}% (>10%)"
        )


def test_classification():
    """Test that all materials are correctly classified."""
    for label, Z_A, Z_B, struct, gap_exp, class_exp, _ in BENCHMARKS:
        gap_pt = band_gap_pt(Z_A, Z_B, struct)
        cls = classify_material(gap_pt)
        assert cls == class_exp, (
            f"{label}: classified as '{cls}', expected '{class_exp}' "
            f"(gap = {gap_pt:.3f} eV)"
        )


def test_dielectric_constants():
    """Test that dielectric constants are within 50% of experiment."""
    for label, Z_A, Z_B, struct, _, _, eps_exp in BENCHMARKS:
        eps_pt = dielectric_constant_pt(Z_A, Z_B, struct)
        err = abs(eps_pt - eps_exp) / eps_exp
        assert err < 0.50, (
            f"{label}: PT eps = {eps_pt:.1f}, exp = {eps_exp:.1f}, "
            f"error = {err*100:.0f}% (>50%)"
        )


def test_analyze_material():
    """Test the analyze_material convenience function."""
    result = analyze_material(14)  # Si
    assert isinstance(result, MaterialResult)
    assert 0.5 < result.E_gap < 2.0
    assert result.classification == 'semiconductor'
    assert result.epsilon > 1.0
    assert result.E_h > 0.0
    assert result.C_ionic == 0.0  # elemental


def test_metals():
    """Test that metals are classified correctly (gap ~ 0 for metals)."""
    # Sodium metal: E_gap should be ~0 in a crystal with BCC structure
    # Na is a simple metal with s-band crossing the Fermi level
    cls = classify_material(0.0)
    assert cls == 'metal'
    cls = classify_material(0.001)
    assert cls == 'metal'  # below s*kT at 300K


def print_benchmark_table():
    """Print a formatted comparison table for all 6 benchmarks."""
    print()
    print("=" * 95)
    print("  PT BAND GAPS AND MATERIALS PROPERTIES -- Module 4")
    print("  Input: s = 1/2 | Parameters: 0 | Derived from: CRT bifurcation P4")
    print("=" * 95)
    print()
    print(f"{'Material':>12s}  {'E_gap(PT)':>9s}  {'E_gap(exp)':>10s}  {'err':>6s}"
          f"  {'Class(PT)':>14s}  {'eps(PT)':>7s}  {'eps(exp)':>8s}  {'E_h':>6s}  {'C':>6s}")
    print("-" * 95)

    errors = []
    for label, Z_A, Z_B, struct, gap_exp, class_exp, eps_exp in BENCHMARKS:
        r = analyze_material(Z_A, Z_B, struct)
        err = (r.E_gap - gap_exp) / gap_exp * 100
        errors.append(abs(err))

        eps_err_str = f"{(r.epsilon - eps_exp)/eps_exp*100:+.0f}%"
        print(f"{label:>12s}  {r.E_gap:9.3f}  {gap_exp:10.2f}  {err:+5.1f}%"
              f"  {r.classification:>14s}  {r.epsilon:7.1f}  {eps_exp:8.1f}  "
              f"{r.E_h:6.3f}  {r.C_ionic:6.3f}")

    mae = sum(errors) / len(errors)
    max_err = max(errors)
    print("-" * 95)
    print(f"{'MAE':>12s}  {mae:9.1f}%    {'Max':>10s}  {max_err:5.1f}%")
    print()

    # Classification check
    all_correct = True
    for label, Z_A, Z_B, struct, gap_exp, class_exp, eps_exp in BENCHMARKS:
        r = analyze_material(Z_A, Z_B, struct)
        if r.classification != class_exp:
            print(f"  MISMATCH: {label} classified as {r.classification}, "
                  f"expected {class_exp}")
            all_correct = False
    if all_correct:
        print("  Classification: 6/6 correct")

    print()
    print("  Key PT constants used:")
    from ptc.constants import S3, C3, C5, D3, S5, S_HALF, P1, RY, D_FULL
    print(f"    sin^2_3 = {S3:.6f}   cos^2_3 = {C3:.6f}")
    print(f"    cos^2_5 = {C5:.6f}   delta_3 = {D3:.6f}")
    print(f"    C3*C5   = {C3*C5:.6f}   S3+D3   = {S3+D3:.6f}")
    print(f"    S3+S5   = {S3+S5:.6f}   Ry/P1   = {RY/P1:.4f} eV (Shannon cap)")
    print(f"    C_Penn  = P1^2+P1*C3+D3 = {P1**2+P1*C3+D3:.4f}")
    print(f"    Penn exp = s+delta_3 = {S_HALF+D3:.4f}")
    print()


if __name__ == '__main__':
    print_benchmark_table()

    # Run assertions
    test_band_gaps()
    test_classification()
    test_dielectric_constants()
    test_analyze_material()
    test_metals()
    print("  ALL TESTS PASSED")
    print()
