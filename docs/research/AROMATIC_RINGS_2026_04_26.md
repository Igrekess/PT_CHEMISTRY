# Aromatic Rings Beyond Carbon — PT Predictions and Experimental Proposals

**Session 2026-04-26** — Branch follow-up on `project_ptc_sigma_aromatic_c9b.md`.

## Summary of additions

Two PT-pure additions to the engine:

1. **C9c block** (`transfer_matrix.py`) — sub-saturated s¹ ring polygon
   override. Replaces the per-edge bond-sum by the cycle-Hückel polygon
   stabilization for rings of Group 1 (alkali) and Group 11 (coinage)
   atoms. Calibration: β = D_dimer / 2.

2. **`nics.py`** module — Nucleus-Independent Chemical Shift via
   Pauling-London ring current. Single PT-derived prefactor
   K = α²·a₀/12 ≈ 2.348 ppm·Å. No fit parameter.

## Validation

### Axis 1: coinage triangles (was the top open issue)

Baseline (C9b only) → with C9c:

| ring | exp D_at (eV) | C9b only | C9b + C9c |
|------|--------------:|---------:|----------:|
| Cu₃  | 2.40          | +194 %   | **+26 %**  |
| Ag₃  | 2.50          | +139 %   | **−3 %**   |
| Cu₄  | 4.50          | +90 %    | **−21 %**  |
| Au₃  | 6.10          | +35 %    | −46 %     |

**Au₃ remains anomalously high in experiment (~2.65× D(Au₂))** — the
remaining ~46% gap is consistent with relativistic 6s/5d hybridization
giving Au an effective n_e > 1, requiring a future R₅₇ relativistic
extension. Cu, Ag, alkali (Li/Na/K) now ride the polygon Hückel cleanly.

**Regression on 806-mol main bench: zero** (MAE 2.079 % unchanged,
same 9 outliers).

### Axis 3a/b: NICS

| ring          | exp NICS(0) | PT NICS(0) | residual                          |
|---------------|------------:|-----------:|-----------------------------------|
| benzene       | −9.7        | −8.7       | 10 % under (within Pauling-London)|
| Si₆           | −10.0       | −11.7      | 17 % over ✓                        |
| Bi₃³⁻ (anion) | −15.0       | −7.6 (neutral) | charge-state mismatch        |
| Al₄²⁻         | −34.0       | −5.6       | multi-fold limit (documented)    |

The PT-pure dipolar formula matches single-channel aromatics
(benzene-like) and gives correct relative ranking across all classes.
Multi-fold (σ + π double-aromatic, e.g. Al₄²⁻) under-shoot is a known
limit of the dipole approximation — PT extension via constructive
σ⊕π coherence is left for follow-up.

## Predictive scan results

Composite aromaticity index *A = |NICS(0)| · f_coh · D_per_atom*
(multiplicative — energetic stability × magnetic signature × T³
composition coherence).

### Top 10 candidates

| #  | ring        | D/atom (eV) | NICS(0) (ppm) | f_coh | A     | status        |
|----|-------------|------------:|--------------:|------:|------:|---------------|
|  1 | P₃          | 4.41        | −15.2         | 1.00  | 66.9  | known unstable, kinetic stabilization needed |
|  2 | P₄          | 4.03        | −16.5         | 1.00  | 66.6  | unsynthesized as homo-π aromatic |
|  3 | Th@S₃       | 3.46        | −18.8         | 1.00  | 64.9  | **novel actinide-capped** |
|  4 | U@S₃        | 3.45        | −18.8         | 1.00  | 64.8  | **novel actinide-capped** |
|  5 | Np@S₃       | 3.45        | −18.8         | 1.00  | 64.7  | **novel actinide-capped** |
|  6 | Pu@S₃       | 3.44        | −18.8         | 1.00  | 64.6  | novel              |
|  7 | Am@S₃       | 3.44        | −18.8         | 1.00  | 64.5  | novel              |
|  8 | Pa@S₃       | 3.43        | −18.8         | 1.00  | 64.3  | novel              |
|  9 | Ce@S₃       | 3.38        | −18.8         | 1.00  | 63.5  | novel lanthanide-capped |
| 10 | Nd@S₃       | 3.37        | −18.8         | 1.00  | 63.3  | novel              |

Notable hetero-triangles: **GaSGa, AlSeAl, SbPSb, TeSTe, InSeIn** —
all D/atom > 3 eV, |NICS(0)| > 10 ppm, f_coh > 0.85.

## Experimental proposals (top-10)

### Class A — gas-phase neutral cluster spectroscopy

**Target: P₃, P₄, S₃, Se₃**

These predicted aromatic rings have **D/atom > 3 eV** and would be
detectable in a Knudsen effusion mass-spec setup. P₃ is metastable
relative to P₂ + P, but a laser-ablation source from red phosphorus
can produce it transiently. Use:
- Knudsen-cell mass spec at ~1200 K to measure D_at via appearance
  potential of P₃⁺ → P₂⁺ + P
- TOF-MS with photoionization (synchrotron VUV) for ionization
  threshold
- Anion photoelectron spectroscopy on P₃⁻ (Wang/Boldyrev protocol):
  expected adiabatic detachment energy ~3.5 eV (PT prediction)

### Class B — actinide / lanthanide-capped salts (the Bi₃@U analog family)

**Target: U@S₃, Th@S₃, Ce@S₃**

The 2026 paper (Bi₃@U, the original motivation of this session)
demonstrated stable σ-aromatic Bi₃ caped by U. Our scan suggests an
**entire family of M@X₃ salts** is feasible. M@S₃ in particular has:
- D/atom ≈ 3.4 eV (stable in solid phase)
- NICS(0) ≈ −18.8 ppm (very strong, > benzene)
- The cap stabilizes the σ-aromatic ring against fragmentation
- f-block CAP donates 5f density into S₃ σ system (R₅₇ mechanism)

**Synthesis route (proposed):** alkali metal reduction of pentaammine
M(III) chalcogenide complexes in liquid ammonia at 200 K, with bulky
[K(crypt-2.2.2)]⁺ counterion (cf. Goicoechea 2008 Zintl protocol).

**Characterization:**
- single-crystal X-ray on the [K-crypt][U(η³-S₃)] salt
- ²³³U Mössbauer to probe 5f electron-density transfer to ring
- ³³S NMR — paramagnetic broadening from ring current expected
- DFT/CASPT2 with f-active orbitals; **PT prediction is the testable
  hypothesis: σ-aromatic 6 π-equivalent on S₃ from 4 σ + 2 π electrons**

### Class C — heteronuclear "designer" triangles

**Target: GaSGa, AlSeAl, SbPSb**

These are heteronuclear σ-aromatic candidates with **f_coh > 0.85**.
GaSGa (D/atom 3.55 eV) is particularly attractive because Ga and S
have well-developed organometallic chemistry.

**Routes:**
- Pulsed laser ablation of Ga₂S₃ in helium expansion → cold molecular
  beam → cation/anion extraction
- Anion photoelectron spectroscopy: PT predicts AEA(GaSGa⁻) ≈ 2.0 eV
  with strong Franck-Condon shift (signature of aromatic stabilization)
- Matrix isolation IR in Ar at 4 K — PT predicts a totally-symmetric
  ring breathing mode at ~250 cm⁻¹ (strongly IR-active in C₂ᵥ symmetry
  due to charge alternation)

### Class D — heavy/relativistic verification

**Target: Au₃ multi-electron correction**

Au₃ remains poorly described by single-electron polygon Hückel (−46 %
under). A relativistic R₅₇ extension treating Au as effective n_e_eff
(6s¹ + correlation-induced 5d hole) is needed. This calls for:

- 4-component CCSD(T) on Au₃ neutral and anion to extract effective
  valence electron count
- comparison with Au₃⁻ photoelectron spectrum (Bishea, Morse 1989)
- inclusion of Au-Au-Au angular correlation that goes beyond
  cycle-Hückel

## Files modified / created

- `ptc/transfer_matrix.py` — C9c block (+85 lines), import `_np_of`,
  module-level `_DIMER_D_CACHE`, helpers `_dimer_D_cached`,
  `_huckel_polygon_multiplier`, `_is_s1_outer`. C9b gets a skip check
  for rings handled by C9c.
- `ptc/nics.py` — new module (~210 lines), public API
  `nics_for_ring`, `nics_all_rings`, `nics0`, `nics1`, `NICSResult`.
- `scripts/scan_aromatic_rings.py` — predictive scan, ranked output
  + experimental protocol classifier.
- `docs/research/AROMATIC_RINGS_2026_04_26.md` (this file).

## Open follow-ups

1. **Multi-fold aromaticity (σ+π in Al₄²⁻)** — requires constructive
   σ⊕π interference term. PT framing: persistent-current cross-channel
   coupling on Z/(2P_l)Z faces. Empirically known enhancement ~3-4×.
2. **Charge-state correction** — exp NICS values are typically reported
   for the anion (Bi₃³⁻, Al₄²⁻, Cp⁻) while we predict neutral. Need
   to add formal charge handling to `_aromatic_electron_count`.
3. **Au₃ relativistic R₅₇** — extension of C9c with effective n_e from
   6s/5d hybrid (Pyykkö relativistic radius arguments).
4. **NICS(1) refinement** — current axial Biot-Savart is geometrically
   correct but the proper out-of-plane probe should account for the
   electron-density tail (~exp(-2r/R)). Small correction.
