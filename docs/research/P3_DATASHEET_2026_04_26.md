<!-- PT predictive datasheet, 2026-04-26 -->
<!-- engine: ptc/transfer_matrix.py post-C9c, ptc/nics.py, ptc/signature.py -->

# Predicted experimental signature — P₃

**SMILES**: `[P]1[P][P]1`   **Formula**: P3   **Atoms**: 3

## Geometry
- P-P: r_e = **2.367 Å**
- P-P: r_e = **2.367 Å**
- P-P: r_e = **2.367 Å**
- Ring radius (centroid → vertex): **1.393 Å**

## Energetics
- Total atomization D_at = **13.225 eV**  (4.408 eV/atom)
- Lowest fragmentation barrier: **10.576 eV**

**Fragmentation channels (ΔE in eV; lowest = activation energy):**
  - ΔE = **+13.23** eV  →  [P] + [P] + [P]  (total atomization (4 atoms))
  - ΔE = **+10.58** eV  →  [P][P] + [P]  (ring → dimer + atom) ← *lowest*

## Aromaticity
- Class: **σ-aromatic (6e), π-radical (3e)**
- σ-aromatic electrons: **6** (4n+2 with n=1 ✓ aromatic)
- π-aromatic electrons: **3** (odd-electron (radical))
- Total delocalized: **9**
- T³ composition coherence f_coh: **1.000**

**NICS(z) profile (PT Pauling-London, in ppm):**

| z (Å) | NICS (ppm) |
|------:|-----------:|
| 0.0  | **-12.64** |
| 0.5  | **-10.54** |
| 1.0  | **-6.78** |
| 1.5  | **-3.98** |
| 2.0  | **-2.36** |
| 3.0  | **-0.94** |

## Vibrational signature
- Raw PT ω uses k = 2D/r² (harmonic). Calibration via Morse factor exp/PT on reference dimer.

| bond | ω_PT raw (cm⁻¹) | Morse factor | ω calibrated (cm⁻¹) |
|------|----------------:|-------------:|--------------------:|
| P-P | 154 | 5.39 | **831** |
| P-P | 154 | 5.39 | **831** |
| P-P | 154 | 5.39 | **831** |

**Ring-breathing mode (totally symmetric, IR-active in C_nv): ≈ 831 cm⁻¹**

## Electronic structure (Koopmans + cycle-Hückel)
- IE (vertical): **12.37 eV**
- EA (vertical): **0.38 eV**
- HOMO–LUMO gap (cycle-Hückel σ): **0.00 eV**


## Experimental protocol — tailored to this candidate

_Cyclic triphosphorus — homonuclear σ-aromatic gas-phase candidate_

### Synthesis (homonuclear neutral cluster)

- **Source:** Knudsen-cell mass spec from elemental precursor
- **Temperature:** 800-1200 K (cluster sublimation)
- **Detection:** TOF-MS, photoionization threshold
- **Lowest fragmentation barrier:** 10.58 eV

### Spectroscopic targets (PT predictions)

| technique | predicted observable | notes |
|-----------|---------------------|-------|
| IR (matrix Ar 4 K) | breathing mode at **831 cm⁻¹** | totally symmetric, A₁ in C_nv |
| ¹H NMR / ³³S NMR | ring current shift, NICS(0) = **-12.6 ppm** | post-DFT GIAO comparison |
| Anion PES | EA = **0.38 ± 0.3 eV** (Wang/Boldyrev) | Franck-Condon to ω vibration |
| Photoionization | IE = **12.37 ± 0.2 eV** | VUV synchrotron source |

### Falsifiability criteria (PT predictions to test)

1. NICS(0) is **diamagnetic** (negative). If post-DFT GIAO gives positive NICS, ring is antiaromatic — PT prediction wrong.
2. NICS(0) / NICS(1) ratio = 1.87. Ratio < 1.5 indicates π-only (no σ aromaticity); ratio > 1.5 confirms σ-dominant or double-aromatic.
4. Ring breathing IR pic at 831 ± 50 cm⁻¹. Significant deviation indicates ring r_e or D₀ wrong.
