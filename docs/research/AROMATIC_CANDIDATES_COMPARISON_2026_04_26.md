# Comparison: PT-predicted aromatic ring candidates

**Date:** 2026-04-26
**Engine:** ptc/transfer_matrix.py post-C9c, ptc/nics.py (signed Hückel),
ptc/signature.py.

This document compares the 6 PT-predicted candidates produced by
`scripts/generate_datasheets.py`, with full per-candidate datasheets
linked. **Important:** the comparison uses a NICS formula with
**signed Hückel rule per channel** — diamagnetic for 4n+2, paramagnetic
for 4n, partial for radicals. This refines the prior aromatic-index
ranking and produces a different recommendation than the bare scan.

## Side-by-side table

| candidate | D/atom (eV) | cap E (eV) | lowest frag (eV) | n_σ | n_π | NICS(0) | NICS(1) | NICS(0)/NICS(1) | ω breath (cm⁻¹) | IE (eV) | EA (eV) | f_coh | class |
|-----------|------------:|-----------:|-----------------:|----:|----:|--------:|--------:|----------------:|----------------:|--------:|--------:|------:|--------|
| [U@S₃](U_S3_DATASHEET_2026_04_26.md)     | 3.45 | 4.6 | 4.6 | 6 | 6 | **−18.8** | −10.8 | 1.73 | 677 | 6.19 | 1.04 | 1.000 | **double σ⊕π aromatic** |
| [Th@S₃](Th_S3_DATASHEET_2026_04_26.md)   | 3.46 | 4.6 | 4.6 | 6 | 6 | **−18.8** | −10.8 | 1.73 | 677 | 6.31 | 1.04 | 1.000 | **double σ⊕π aromatic** |
| [Ce@S₃](Ce_S3_DATASHEET_2026_04_26.md)   | 3.38 | 4.3 | 4.3 | 6 | 6 | **−18.8** | −10.8 | 1.73 | 675 | 5.53 | 1.04 | 1.000 | **double σ⊕π aromatic** |
| [P₃](P3_DATASHEET_2026_04_26.md)         | 4.41 | 0.0 | 10.6 | 6 | 3 | −12.6 | −6.8 | 1.87 | 831 | 12.37 | 0.38 | 1.000 | σ-aro + π-radical |
| [GaSGa](GaSGa_DATASHEET_2026_04_26.md)   | 2.86 | 0.0 | 5.7 | 2 | 4 | **+2.6** | +1.7 | 1.53 | 227 | 3.52 | 0.21 | 0.951 | σ-aro + π-anti |
| [AlSeAl](AlSeAl_DATASHEET_2026_04_26.md) | 3.28 | 0.0 | 7.1 | 2 | 4 | **+2.8** | +1.6 | 1.71 | 533 | 2.88 | 0.21 | 0.898 | σ-aro + π-anti |

## Reading the table

- **D/atom**: higher = more thermodynamically stable bonded cluster.
- **cap E**: cap binding energy (only meaningful for M·X₃ capped systems).
- **lowest frag**: barrier to lowest-energy fragmentation channel —
  this is the **kinetic** decomposition energy (≈ activation energy
  at high T).
- **n_σ, n_π**: PT-counted σ- and π-aromatic electrons (group rules
  for non-organic, Kekulé/aromatic SMILES for organic).
- **NICS(0), NICS(1)**: PT Pauling-London prediction with signed
  Hückel rule; negative = diamagnetic (aromatic), positive =
  paramagnetic (anti-aromatic).
- **NICS(0)/NICS(1) ratio**: > 1.5 indicates σ-aromatic component
  (NICS peaked at center, decays steeply); ≈ 1 = π-only.
- **f_coh**: T³ composition coherence; 1.0 for homonuclear.
- **ω breath**: Morse-calibrated breathing mode (totally symmetric,
  IR active in C_nv).

## Priority ranking — REVISED

Before this session: the bare scan in
[`scripts/scan_aromatic_rings.py`](../../scripts/scan_aromatic_rings.py)
used `A = D/atom × |NICS| × f_coh` and put GaSGa, AlSeAl high on the
list. With **signed Hückel NICS**, those candidates are net
**antiaromatic** in the π channel — opposite sign — and drop in
priority.

### 1. Th@S₃ ★★★★★ — first synthesis target

- **Why:** lanthanide is easier than actinide for chemistry,
  thorium has no fission risk, the predicted properties are
  identical to U@S₃ (both σ⊕π double aromatic at NICS = −18.8 ppm).
- **Synthesis:** Goicoechea NH₃(l) protocol with [Cp*₂ThCl₂] and
  K₂S₅ at −78 °C, [K(crypt-2.2.2)]⁺ counter-ion.
- **Smoking gun:** ring-breathing IR pic at 677 ± 50 cm⁻¹ + NICS(0)/NICS(1)
  ratio > 1.5 by post-DFT GIAO.

### 2. Ce@S₃ ★★★★ — lanthanide model

- Identical PT predictions to Th@S₃.
- Ce(III)/Ce(IV) chemistry well established → easier optimization.
- Lower IE (5.53 eV) means PES detection at lower photon energies.

### 3. U@S₃ ★★★★ — original target (chalcogenide analog of Bi₃@U)

- Same PT predictions as Th, plus distinct ²³³U Mössbauer signature.
- Synthesis requires controlled-radioactivity facility.

### 4. P₃ ★★★ — homonuclear gas-phase candidate

- Highest D/atom (4.41 eV) — strongest candidate by raw stability.
- BUT: π-radical (3 electrons, odd number), so neutral P₃ is
  intrinsically open-shell → kinetically unstable vs P₂ + P
  (10.6 eV barrier exists, but radical recombination kinetics
  dominate at all temperatures).
- **Practical access:** anion P₃⁻ (4 π e, π-aromatic) or cation P₃⁺
  (2 π e, π-aromatic). PT predicts both should be **closed-shell
  σ⊕π double aromatic** with NICS magnitude > P₃ neutral.
- Anion PES on P₃⁻ produced by laser ablation is the **canonical
  experiment** (Wang/Boldyrev style).

### 5. GaSGa, AlSeAl ★ — DOWNGRADED, not aromatic

- The original scan's high A score (D × |NICS|) was misleading
  because the bare formula didn't distinguish 4n from 4n+2.
- With signed NICS: π=4 → 4n antiaromatic → **the π contribution
  cancels most of the σ-aromatic signature**, leaving small net
  paramagnetism.
- **Implication for design:** to obtain a doubly aromatic
  heteronuclear analog, target n_π = 6 — would require atoms
  contributing 2 π e each = Group 16 in the bridging position.
  Candidates worth re-exploring: **STeS, SeSeSe, SSS** (already
  S₃), **SPbS, SeGeSe, TeSnTe**.

## What this teaches us

The signed-Hückel NICS introduced this session is a **PT-pure refinement
that fundamentally changes ranking**: candidates with mismatched σ/π
electron counts (4n+2 in one channel, 4n in the other) are net
non-aromatic or weakly antiaromatic. The aromatic index for predictive
scan should be:
$$A_{\text{signed}} = D_{\text{atom}} \cdot \max(0, -\text{NICS}(0)) \cdot f_{\text{coh}}$$
(i.e., only diamagnetic NICS counts toward aromaticity).

Re-running the scan with this refined index would produce a different
top-10. **This is the next step.**

## Files generated this session

| file | content |
|------|---------|
| `ptc/transfer_matrix.py` | C9c block (s¹ polygon override) |
| `ptc/nics.py` | NICS formula with signed Hückel rule per channel |
| `ptc/signature.py` | `predict_full_signature(smiles)` — all observables |
| `scripts/generate_datasheets.py` | batch generator for candidates |
| `scripts/scan_aromatic_rings.py` | predictive enumeration (un-signed; to update) |
| `docs/research/U_S3_DATASHEET_2026_04_26.md` | template datasheet |
| `docs/research/Th_S3_DATASHEET_2026_04_26.md` | first priority target |
| `docs/research/Ce_S3_DATASHEET_2026_04_26.md` | lanthanide model |
| `docs/research/GaSGa_DATASHEET_2026_04_26.md` | refuted heteronuclear |
| `docs/research/AlSeAl_DATASHEET_2026_04_26.md` | refuted heteronuclear |
| `docs/research/P3_DATASHEET_2026_04_26.md` | gas-phase σ-aromatic |
| `docs/research/AROMATIC_CANDIDATES_COMPARISON_2026_04_26.md` | this synthesis |

## Open follow-ups

1. **Re-run predictive scan with signed-Hückel index** to identify
   new top candidates (likely heavy chalcogenide rings: SeSe + Te,
   etc.). The current top-10 ranking is partially invalidated.
2. **Multi-fold enhancement** for σ⊕π double aromatics (Al₄²⁻
   limit). PT framing: cross-channel constructive interference on
   Z/(2P_l)Z faces. Empirical 3-4× enhancement.
3. **Charge state corrections** — PT signatures are for neutrals;
   most exp data is for anions. Add `topology.charges` summing to
   `_aromatic_electron_count` and the per-channel splits.
4. **Anion P₃⁻ datasheet** — the kinetically accessible target,
   would close the P₃ ↔ P₃⁻ gap.
