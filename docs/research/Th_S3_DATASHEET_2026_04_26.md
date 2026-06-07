<!-- PT predictive datasheet, 2026-04-26 -->
<!-- engine: ptc/transfer_matrix.py post-C9c, ptc/nics.py, ptc/signature.py -->

# Predicted experimental signature — Th@S₃

**SMILES**: `[Th][S]1[S][S]1`   **Formula**: S3Th   **Atoms**: 4

## Geometry
- Th-S: r_e = **3.456 Å**
- S-S: r_e = **2.497 Å**
- S-S: r_e = **2.438 Å**
- S-S: r_e = **2.497 Å**
- Ring radius (centroid → vertex): **1.501 Å**
- Cap height above ring plane: **3.113 Å**

## Energetics
- Total atomization D_at = **13.840 eV**  (3.460 eV/atom)
- Cap binding energy: **4.646 eV**
- Lowest fragmentation barrier: **4.646 eV**

**Fragmentation channels (ΔE in eV; lowest = activation energy):**
  - ΔE = **+13.84** eV  →  [Th] + [S] + [S] + [S]  (total atomization (4 atoms))
  - ΔE = **+4.65** eV  →  [Th] + [S]1[S][S]1  (cap removal: M @ X_n → M + X_n) ← *lowest*
  - ΔE = **+11.37** eV  →  [Th] + [S][S] + [S]  (full ring fragmentation: M + X₂ + X)

## Aromaticity
- Class: **double aromatic (σ ⊕ π)**
- σ-aromatic electrons: **6** (4n+2 with n=1 ✓ aromatic)
- π-aromatic electrons: **6** (4n+2 with n=1 ✓ aromatic)
- Total delocalized: **12**
- T³ composition coherence f_coh: **1.000**

**NICS(z) profile (PT Pauling-London, in ppm):**

| z (Å) | NICS (ppm) |
|------:|-----------:|
| 0.0  | **-18.77** |
| 0.5  | **-16.03** |
| 1.0  | **-10.82** |
| 1.5  | **-6.65** |
| 2.0  | **-4.06** |
| 3.0  | **-1.68** |

## Vibrational signature
- Raw PT ω uses k = 2D/r² (harmonic). Calibration via Morse factor exp/PT on reference dimer.

| bond | ω_PT raw (cm⁻¹) | Morse factor | ω calibrated (cm⁻¹) |
|------|----------------:|-------------:|--------------------:|
| Th-S | 77 | 3.15 | **243** |
| S-S | 128 | 5.30 | **678** |
| S-S | 127 | 5.30 | **674** |
| S-S | 128 | 5.30 | **678** |

**Ring-breathing mode (totally symmetric, IR-active in C_nv): ≈ 677 cm⁻¹**

## Electronic structure (Koopmans + cycle-Hückel)
- IE (vertical): **6.31 eV**
- EA (vertical): **1.04 eV**
- HOMO–LUMO gap (cycle-Hückel σ): **0.00 eV**


## Experimental protocol — tailored to this candidate

_Thorium-capped sulfur triangle (recommended first synthesis target)_

### Synthesis (actinide cap)

- **Solvent:** liquid NH₃ at −78 °C, or DME/THF at −80 °C
- **Reductant:** alkali metal (Na, K) in 2:1 stoichiometry vs cap precursor
- **Cap precursor:** tris-Cp* actinide chloride [Cp*ₓMCl_y] with M = the cap atom
- **Ring source:** Na₂S, K₂Sₓ, or in-situ-generated S²⁻
- **Counter-ion:** [K(crypt-2.2.2)]⁺ for crystal isolation (Goicoechea protocol)
- **Predicted cap binding:** 4.65 eV — synthesis margin OK at < 200 °C

### Spectroscopic targets (PT predictions)

| technique | predicted observable | notes |
|-----------|---------------------|-------|
| IR (matrix Ar 4 K) | breathing mode at **677 cm⁻¹** | totally symmetric, A₁ in C_nv |
| ¹H NMR / ³³S NMR | ring current shift, NICS(0) = **-18.8 ppm** | post-DFT GIAO comparison |
| Anion PES | EA = **1.04 ± 0.3 eV** (Wang/Boldyrev) | Franck-Condon to ω vibration |
| Photoionization | IE = **6.31 ± 0.2 eV** | VUV synchrotron source |
| X-ray monocrystal | r(M-X), cap height, plane symmetry | low T (< 100 K) |
| Mössbauer (¹⁵¹Eu, ²³³U etc.) | isomer shift signs cap oxidation state | distinguishes M(III) from M(IV) |

### Falsifiability criteria (PT predictions to test)

1. NICS(0) is **diamagnetic** (negative). If post-DFT GIAO gives positive NICS, ring is antiaromatic — PT prediction wrong.
2. NICS(0) / NICS(1) ratio = 1.73. Ratio < 1.5 indicates π-only (no σ aromaticity); ratio > 1.5 confirms σ-dominant or double-aromatic.
3. Cap binding ≥ 3.3 eV by TGA-MS or thermal decomposition. Below this, the f-block back-donation channel R₅₇ is overestimated.
4. Ring breathing IR pic at 677 ± 50 cm⁻¹. Significant deviation indicates ring r_e or D₀ wrong.
