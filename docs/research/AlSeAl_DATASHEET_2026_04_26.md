<!-- PT predictive datasheet, 2026-04-26 -->
<!-- engine: ptc/transfer_matrix.py post-C9c, ptc/nics.py, ptc/signature.py -->

# Predicted experimental signature — AlSeAl

**SMILES**: `[Al]1[Se][Al]1`   **Formula**: Al2Se   **Atoms**: 3

## Geometry
- Al-Se: r_e = **2.700 Å**
- Se-Al: r_e = **2.700 Å**
- Al-Al: r_e = **2.343 Å**
- Ring radius (centroid → vertex): **1.525 Å**

## Energetics
- Total atomization D_at = **9.848 eV**  (3.283 eV/atom)
- Lowest fragmentation barrier: **7.061 eV**

**Fragmentation channels (ΔE in eV; lowest = activation energy):**
  - ΔE = **+9.85** eV  →  [Al] + [Se] + [Al]  (total atomization (4 atoms))
  - ΔE = **+7.06** eV  →  [Al][Se] + [Al]  (ring → dimer + atom) ← *lowest*

## Aromaticity
- Class: **σ-aromatic (2e), π-antiaromatic (4e)**
- σ-aromatic electrons: **2** (4n+2 with n=0 ✓ aromatic)
- π-aromatic electrons: **4** (4n with n=1 (anti-aromatic))
- Total delocalized: **6**
- T³ composition coherence f_coh: **0.898**

**NICS(z) profile (PT Pauling-London, in ppm):**

| z (Å) | NICS (ppm) |
|------:|-----------:|
| 0.0  | **+2.77** |
| 0.5  | **+2.37** |
| 1.0  | **+1.62** |
| 1.5  | **+1.00** |
| 2.0  | **+0.62** |
| 3.0  | **+0.26** |

## Vibrational signature
- Raw PT ω uses k = 2D/r² (harmonic). Calibration via Morse factor exp/PT on reference dimer.

| bond | ω_PT raw (cm⁻¹) | Morse factor | ω calibrated (cm⁻¹) |
|------|----------------:|-------------:|--------------------:|
| Al-Se | 116 | 5.14 | **598** |
| Se-Al | 116 | 5.14 | **598** |
| Al-Al | 98 | 4.14 | **404** |

**Ring-breathing mode (totally symmetric, IR-active in C_nv): ≈ 533 cm⁻¹**

## Electronic structure (Koopmans + cycle-Hückel)
- IE (vertical): **2.88 eV**
- EA (vertical): **0.21 eV**
- HOMO–LUMO gap (cycle-Hückel σ): **0.00 eV**


## Experimental protocol — tailored to this candidate

_Aluminum-bridged selenide triangle — heteronuclear_

### Synthesis (heteronuclear ring)

- **Direct route:** pulsed laser ablation of a binary precursor (e.g. Ga₂S₃ for GaSGa)
- **Source:** 532 nm laser, 10-20 mJ/pulse, supersonic He expansion at 5 bar
- **Detection:** TOF-MS (positive + negative modes)
- **Predicted (vertical) IE:** 2.88 eV — within accessible photoionization range

### Spectroscopic targets (PT predictions)

| technique | predicted observable | notes |
|-----------|---------------------|-------|
| IR (matrix Ar 4 K) | breathing mode at **533 cm⁻¹** | totally symmetric, A₁ in C_nv |
| ¹H NMR / ³³S NMR | ring current shift, NICS(0) = **+2.8 ppm** | post-DFT GIAO comparison |
| Anion PES | EA = **0.21 ± 0.3 eV** (Wang/Boldyrev) | Franck-Condon to ω vibration |
| Photoionization | IE = **2.88 ± 0.2 eV** | VUV synchrotron source |

### Falsifiability criteria (PT predictions to test)

1. NICS(0) is **paramagnetic** (positive). If post-DFT GIAO gives negative NICS, the σ-aromatic cancellation is incomplete.
2. NICS(0) / NICS(1) ratio = 1.71. Ratio < 1.5 indicates π-only (no σ aromaticity); ratio > 1.5 confirms σ-dominant or double-aromatic.
4. Ring breathing IR pic at 533 ± 50 cm⁻¹. Significant deviation indicates ring r_e or D₀ wrong.
