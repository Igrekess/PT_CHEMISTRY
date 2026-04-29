# EA precision lock and ab-initio refinement map

Date: 2026-04-29

Status: validation lock for the canonical PTC EA channel.

Related files:

- `ptc/atom.py`
- `ptc/ea_residual_fields.py`
- `ptc/ea_precision_audit.py`
- `ptc/tests/test_ea_precision_audit.py`
- `docs/research/PT_EA_CONTINUOUS_CHANNEL_DERIVATION.md`
- `docs/research/PT_EA_CONTACT_DEPTH_OPERATOR.md`

---

## 1. Canonical score

The canonical `EA_eV` engine now uses the contact-depth capture-channel
correction:

```text
EA_geo
  -> capture-side CPR surface transmission
  -> threshold/core/closure residuals
  -> continuous channel residuals
  -> contact-depth operator.
```

Global benchmark on the 73 positive embedded EA references:

```text
geo                      N=73 MAE=1.367068% MAE_eV=0.010871 max=5.685683%
boundary_continuum       N=73 MAE=1.332454% MAE_eV=0.010596 max=5.685683%
threshold_core_closure   N=73 MAE=1.248114% MAE_eV=0.008934 max=5.685683%
continuous_channel       N=73 MAE=1.125303% MAE_eV=0.007973 max=3.803883%
canonical                N=73 MAE=0.983780% MAE_eV=0.006859 max=3.534951%
```

This locks three claims:

```text
1. the canonical EA channel improves the legacy geometric channel;
2. the improvement is not a fitted amplitude;
3. the largest positive-EA residual is now below 3.54%.
```

---

## 2. Comparison scale

This is an atomic EA benchmark.  It should not be mixed with molecular
redox benchmarks or vertical molecular electron affinities.

Useful scale:

```text
PT canonical atomic EA        0.006859 eV = 0.158 kcal/mol
G4-type molecular EA sets     about 0.03--0.04 eV
many DFT/low-cost EA sets     about 0.2--0.6 eV
ultra-precise atomic ab initio can reach sub-meV on selected light atoms
```

So the correct statement is:

```text
PT is not yet "better than all ab initio atomic work".
PT is exceptional for a zero-fit closed atomic formula across the table.
```

---

## 3. Small-atom residuals

The first two periods are where specialized ab-initio calculations become
most informative.  Current canonical residuals:

```text
H   -0.004979 eV
Li  +0.010446 eV
B   +0.000256 eV
C   -0.016334 eV
O   +0.002563 eV
F   +0.025096 eV
Na  -0.004299 eV
Al  -0.008464 eV
Si  -0.015786 eV
P   +0.013980 eV
S   -0.018651 eV
Cl  +0.004255 eV
```

The residual scale is now tens of meV, not tenths of eV.  This is the
regime where high-precision atomic theory separates:

```text
correlation;
finite nuclear mass and recoil;
scalar relativistic mass-velocity/Darwin corrections;
spin-orbit and fine-structure threshold alignment;
QED/contact corrections.
```

---

## 4. PT reading of the ab-initio ladder

The ab-initio ladder has a direct PT translation:

```text
correlation
  -> polygon capture/ejection plus threshold/core fields

finite nuclear mass / recoil / mass polarization
  -> mass-weighted motion of the capture boundary

scalar relativistic terms
  -> CPR surface transmission and (Z alpha)^2 residual fields

spin-orbit / fine-structure threshold splitting
  -> p-threshold and p-closure alignment

QED/contact terms
  -> canonical contact-depth operator, with future sub-residual refinements
```

This gives the next research program.  We should not add another global
EA correction.  We should derive local precision layers that mirror the
ab-initio hierarchy.

---

## 5. Lock tests

Executable tests now enforce:

```text
canonical N=73 MAE < 0.99%;
canonical MAE_eV < 0.0069 eV;
canonical max error < 3.54%;
canonical == contact_depth_operator;
small-atom residual max < 0.026 eV;
projection amplitudes recover -D5^2, -CROSS_37, +CROSS_57;
unprojected p-center pressure does not recover CROSS_57.
```

The last test is important: the double-d screening projector is required
to recover a PT constant.  Without it, the p-center residual is not a PT
coupling.
