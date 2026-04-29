# Atomic precision frontier: IE/EA residuals after canonical CPR

Date: 2026-04-29

Status: historical diagnostic plus activation note.  The contact-depth
operator identified here has been promoted to the canonical PTC EA
engine.  The remaining frontier is proof hardening and sub-residual
precision, not the numerical 1% threshold.

Source:

- `ptc/atomic_precision_frontier.py`
- `ptc/tests/test_atomic_precision_frontier.py`
- `docs/research/PT_EA_CONTACT_DEPTH_OPERATOR.md`
- canonical `ptc.atom.IE_eV`
- canonical `ptc.atom.EA_eV`

---

## 1. Current canonical precision

```text
IE N=118 MAE=0.045770% MAE_eV=0.003378 max=0.246514%
EA N= 73 MAE=0.983780% MAE_eV=0.006859 max=3.534951%
```

Interpretation:

- IE is not the current bottleneck.  It is already below 0.05% globally.
- EA has crossed the symbolic 1% frontier after the contact-depth layer.
- The EA residual scale is now about 6.9 meV
  mean absolute error.

---

## 2. Largest remaining IE residuals

```text
 61 Pm f5  p6 err=-0.2465% dE=-0.01376 eV
111 Rg d10 p7 err=-0.2304% dE=-0.02433 eV
 72 Hf d2  p6 err=-0.1540% dE=-0.01051 eV
 62 Sm f6  p6 err=+0.1501% dE=+0.00847 eV
 71 Lu d1  p6 err=+0.1314% dE=+0.00713 eV
105 Db d3  p7 err=-0.1299% dE=-0.00895 eV
 75 Re d5  p6 err=-0.1287% dE=-0.01008 eV
103 Lr d1  p7 err=+0.1265% dE=+0.00697 eV
```

These are not random.  They sit on:

- the period-6 f construction;
- the lanthanide-contracted d receiver;
- the period-7 superheavy d closure.

This is the same geography that controls the remaining EA residuals.
The difference is scale: IE already has the main CPR field, so only the
sub-residual precision layer remains.

---

## 3. Largest remaining EA residuals after contact-depth activation

```text
 70 Yb f14 p6 err=-3.5350% dE=-0.00071 eV
 59 Pr f3  p6 err=+3.3296% dE=+0.03203 eV
 42 Mo d5  p5 err=+2.9138% dE=+0.02179 eV
 73 Ta d3  p6 err=+2.7507% dE=+0.00886 eV
 23  V d3  p4 err=-2.4594% dE=-0.01294 eV
 41 Nb d4  p5 err=-2.4535% dE=-0.02191 eV
 21 Sc d1  p4 err=-2.4505% dE=-0.00461 eV
 75 Re d5  p6 err=-2.4088% dE=-0.00361 eV
```

The strongest families are:

1. f-shell local contact / center pressure;
2. period-6 d second-harmonic depth;
3. compact first d-shell phase;
4. small s/p light-atom residuals, plausibly finite-mass/contact rather
   than a new shell law.

---

## 4. Contact-depth activation

The diagnostic candidates were fixed by PT constants.  The contact-depth
operator is now the canonical EA layer.

```text
f_contact_center          EA MAE = 1.096872%
d6_second_harmonic_depth  EA MAE = 1.101514%
d4_compact_core_phase     EA MAE = 1.123885%
ea_contact_projector_stack  EA MAE = 0.983780%
ea_contact_depth_operator   EA MAE = 0.983780%
```

The most promising single candidate is:

```text
Lambda_f,contact =
  - sin^2(theta_3) delta_3 (Z alpha)^2
    |2x - 1| sqrt(period - 1) 1_f
```

PT reading:

- `f` is the most internalized active channel still visible in EA;
- after the canonical edge leakage, the remaining f residual behaves
  like a local contact/center pressure term;
- the sign is contractive: the canonical EA is too high at Pr/Gd-type
  points and must be slightly suppressed.

The period-6 d candidate is:

```text
Lambda_d6,phase =
  + sin^2(theta_3) delta_5 (Z alpha)^2
    sin(4 pi x) 1_{d, period=6}
```

PT reading:

- the d residual is a second harmonic on the pentagon;
- it is depth-dependent, strongest after lanthanide contraction;
- it mirrors the d pockets still visible in IE.

The first combined fixed-constant stack is:

```text
Lambda_contact =
  - S3 D3 u |2x - 1| sqrt(period - 1) 1_f
  + S3 D5 u sin(4 pi x) 1_{d,p6}
  - D5^2 u cos(2 pi x) 1_{d,p6}
  - S3 D3 u sin(2 pi x) 1_{d,p5}
  + CROSS_57 u [sin(2 pi x) + cos(2 pi x) - sin(4 pi x)] 1_{d,p4}
  + D5 D7 u sin(4 pi x) 1_p
```

where:

```text
u = (Z alpha)^2,  x = receiver coordinate on the capture channel.
```

Numerically:

```text
continuous-channel EA        MAE = 1.125303%  max = 3.803883%
EA contact projector stack   MAE = 0.983780%  max = 3.534951%
canonical contact-depth EA   MAE = 0.983780%  max = 3.534951%
```

This is the first fixed-PT diagnostic that crosses the 1% EA frontier
without worsening the worst residual.  The gain is not produced by a free
coefficient: every amplitude is an existing PT constant.

The structural compression is now:

```text
tau = period - 4

K_d(x,tau) =
  [CROSS_57 L0(tau) - S3 D3 L1(tau)] sin(2 pi x)
+ [CROSS_57 L0(tau) - D5^2 L2(tau)] cos(2 pi x)
+ [-CROSS_57 L0(tau) + S3 D5 L2(tau)] sin(4 pi x),
```

where `L0,L1,L2,L3` are the cubic Lagrange depth functions on
`tau=0,1,2,3`.  The explicit d-period projectors are therefore the
samples of one continuous depth operator.  The unresolved point is no
longer numerical: it is proof hardening of the PT radial-depth basis and
the weak IE trace.

---

## 5. What remains after canonical activation

The exploratory feature search gives an important warning.  If one allows
a flexible Fourier regression, the in-sample EA MAE can pass below 1%,
but leave-one-out behavior gets worse.  This means the final correction
must be derived, not fitted.

The fixed projector stack above passes the numerical 1% threshold.  The
depth-operator rewrite removes the hard d-period gates by treating them
as samples of the continuous coordinate `tau = period - 4`.  This is now
canonical in PTC, with the bridge status preserved in the monograph.

Required before theorem-level hardening:

1. derive the f-contact term from PT contact density / radiative
   self-channel logic;
2. show why the coefficient is `sin^2(theta_3) delta_3` or replace it by
   the correct PT constant;
3. derive the compact d kernel
   `sin(2 pi x) + cos(2 pi x) - sin(4 pi x)` from the first d-shell
   contact boundary;
4. derive the cubic depth basis `L0..L3` from the radial-depth hierarchy;
5. show that the same operator has the right weak trace in IE;
6. re-run dependent molecular benchmarks after activation.

---

## 6. Physical interpretation

The missing layer resembles the precision hierarchy of specialized
atomic ab-initio calculations:

- correlation / shell rearrangement: already mostly captured by polygon
  capture and CPR;
- finite nuclear mass / recoil: relevant for the smallest atoms;
- Breit-Pauli / fine structure: already visible in threshold and closure
  terms;
- radiative QED / contact density: likely the next EA frontier;
- isotope and hyperfine refinements: future layer, not needed for the
  current 1% objective.

In PT language, these are not separate phenomenological patches.  They
should be different projections of one local persistence operator:

```text
continuous envelope  (Z alpha)^2, alpha/pi, recoil
discrete support     channel polygon and capture/ejection edge
action sign          contraction or expansion of persistence pressure
```

The next serious derivation target is therefore a single contact-density
operator whose f-channel projection gives the EA improvement above, and
whose d/s/p projections explain the remaining smaller residues.
