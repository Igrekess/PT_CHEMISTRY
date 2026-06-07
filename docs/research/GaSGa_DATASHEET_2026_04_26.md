<!-- PT predictive datasheet, 2026-04-26 -->
<!-- engine: ptc/transfer_matrix.py post-C9c, ptc/nics.py, ptc/signature.py -->

# Predicted experimental signature — GaSGa

**SMILES**: `[Ga]1[S][Ga]1`   **Formula**: SGa2   **Atoms**: 3

## Geometry
- Ga-S: r_e = **2.775 Å**
- S-Ga: r_e = **2.775 Å**
- Ga-Ga: r_e = **3.197 Å**
- Ring radius (centroid → vertex): **1.740 Å**

## Energetics
- Total atomization D_at = **8.570 eV**  (2.857 eV/atom)
- Lowest fragmentation barrier: **5.699 eV**

**Fragmentation channels (ΔE in eV; lowest = activation energy):**
  - ΔE = **+8.57** eV  →  [Ga] + [S] + [Ga]  (total atomization (4 atoms))
  - ΔE = **+5.70** eV  →  [Ga][S] + [Ga]  (ring → dimer + atom) ← *lowest*

## Aromaticity
- Class: **σ-aromatic (2e), π-antiaromatic (4e)**
- σ-aromatic electrons: **2** (4n+2 with n=0 ✓ aromatic)
- π-aromatic electrons: **4** (4n with n=1 (anti-aromatic))
- Total delocalized: **6**
- T³ composition coherence f_coh: **0.951**

**NICS(z) profile (PT Pauling-London, in ppm):**

| z (Å) | NICS (ppm) |
|------:|-----------:|
| 0.0  | **+2.57** |
| 0.5  | **+2.28** |
| 1.0  | **+1.67** |
| 1.5  | **+1.11** |
| 2.0  | **+0.73** |
| 3.0  | **+0.32** |

## Vibrational signature
- Raw PT ω uses k = 2D/r² (harmonic). Calibration via Morse factor exp/PT on reference dimer.

| bond | ω_PT raw (cm⁻¹) | Morse factor | ω calibrated (cm⁻¹) |
|------|----------------:|-------------:|--------------------:|
| Ga-S | 102 | 3.15 | **323** |
| S-Ga | 102 | 3.15 | **323** |
| Ga-Ga | 36 | 1.00 | **36** |

**Ring-breathing mode (totally symmetric, IR-active in C_nv): ≈ 227 cm⁻¹**

## Electronic structure (Koopmans + cycle-Hückel)
- IE (vertical): **3.52 eV**
- EA (vertical): **0.21 eV**
- HOMO–LUMO gap (cycle-Hückel σ): **0.00 eV**


## Experimental protocol — tailored to this candidate

_Gallium-bridged sulfide triangle — heteronuclear_

### Synthesis (heteronuclear ring)

- **Direct route:** pulsed laser ablation of a binary precursor (e.g. Ga₂S₃ for GaSGa)
- **Source:** 532 nm laser, 10-20 mJ/pulse, supersonic He expansion at 5 bar
- **Detection:** TOF-MS (positive + negative modes)
- **Predicted (vertical) IE:** 3.52 eV — within accessible photoionization range

### Spectroscopic targets (PT predictions)

| technique | predicted observable | notes |
|-----------|---------------------|-------|
| IR (matrix Ar 4 K) | breathing mode at **227 cm⁻¹** | totally symmetric, A₁ in C_nv |
| ¹H NMR / ³³S NMR | ring current shift, NICS(0) = **+2.6 ppm** | post-DFT GIAO comparison |
| Anion PES | EA = **0.21 ± 0.3 eV** (Wang/Boldyrev) | Franck-Condon to ω vibration |
| Photoionization | IE = **3.52 ± 0.2 eV** | VUV synchrotron source |

### Falsifiability criteria (PT predictions to test)

1. NICS(0) is **paramagnetic** (positive). If post-DFT GIAO gives negative NICS, the σ-aromatic cancellation is incomplete.
2. NICS(0) / NICS(1) ratio = 1.53. Ratio < 1.5 indicates π-only (no σ aromaticity); ratio > 1.5 confirms σ-dominant or double-aromatic.
4. Ring breathing IR pic at 227 ± 50 cm⁻¹. Significant deviation indicates ring r_e or D₀ wrong.
