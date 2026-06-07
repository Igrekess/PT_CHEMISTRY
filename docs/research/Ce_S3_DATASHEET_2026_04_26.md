<!-- PT predictive datasheet, 2026-04-26 -->
<!-- engine: ptc/transfer_matrix.py post-C9c, ptc/nics.py, ptc/signature.py -->

# Predicted experimental signature — Ce@S₃

**SMILES**: `[Ce][S]1[S][S]1`   **Formula**: S3Ce   **Atoms**: 4

## Geometry
- Ce-S: r_e = **3.198 Å**
- S-S: r_e = **2.498 Å**
- S-S: r_e = **2.438 Å**
- S-S: r_e = **2.498 Å**
- Ring radius (centroid → vertex): **1.501 Å**
- Cap height above ring plane: **2.823 Å**

## Energetics
- Total atomization D_at = **13.539 eV**  (3.385 eV/atom)
- Cap binding energy: **4.345 eV**
- Lowest fragmentation barrier: **4.345 eV**

**Fragmentation channels (ΔE in eV; lowest = activation energy):**
  - ΔE = **+13.54** eV  →  [Ce] + [S] + [S] + [S]  (total atomization (4 atoms))
  - ΔE = **+4.34** eV  →  [Ce] + [S]1[S][S]1  (cap removal: M @ X_n → M + X_n) ← *lowest*
  - ΔE = **+11.07** eV  →  [Ce] + [S][S] + [S]  (full ring fragmentation: M + X₂ + X)

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
| Ce-S | 82 | 3.15 | **258** |
| S-S | 127 | 5.30 | **676** |
| S-S | 127 | 5.30 | **674** |
| S-S | 127 | 5.30 | **676** |

**Ring-breathing mode (totally symmetric, IR-active in C_nv): ≈ 675 cm⁻¹**

## Electronic structure (Koopmans + cycle-Hückel)
- IE (vertical): **5.53 eV**
- EA (vertical): **1.04 eV**
- HOMO–LUMO gap (cycle-Hückel σ): **0.00 eV**


## Experimental protocol — tailored to this candidate

_Cerium-capped sulfur triangle (lanthanide model, less radioactive)_

### Synthesis (lanthanide cap)

- **Easier than actinide** — no radioactive handling required
- Use the same NH₃(l) protocol as Goicoechea, replacing actinide source
- **Recommended cap source:** [Cp*ₓLnCl] tetrahydrofuran adduct
- **Predicted cap binding:** 4.34 eV

### Spectroscopic targets (PT predictions)

| technique | predicted observable | notes |
|-----------|---------------------|-------|
| IR (matrix Ar 4 K) | breathing mode at **675 cm⁻¹** | totally symmetric, A₁ in C_nv |
| ¹H NMR / ³³S NMR | ring current shift, NICS(0) = **-18.8 ppm** | post-DFT GIAO comparison |
| Anion PES | EA = **1.04 ± 0.3 eV** (Wang/Boldyrev) | Franck-Condon to ω vibration |
| Photoionization | IE = **5.53 ± 0.2 eV** | VUV synchrotron source |
| X-ray monocrystal | r(M-X), cap height, plane symmetry | low T (< 100 K) |
| Mössbauer (¹⁵¹Eu, ²³³U etc.) | isomer shift signs cap oxidation state | distinguishes M(III) from M(IV) |

### Falsifiability criteria (PT predictions to test)

1. NICS(0) is **diamagnetic** (negative). If post-DFT GIAO gives positive NICS, ring is antiaromatic — PT prediction wrong.
2. NICS(0) / NICS(1) ratio = 1.73. Ratio < 1.5 indicates π-only (no σ aromaticity); ratio > 1.5 confirms σ-dominant or double-aromatic.
3. Cap binding ≥ 3.0 eV by TGA-MS or thermal decomposition. Below this, the f-block back-donation channel R₅₇ is overestimated.
4. Ring breathing IR pic at 675 ± 50 cm⁻¹. Significant deviation indicates ring r_e or D₀ wrong.
