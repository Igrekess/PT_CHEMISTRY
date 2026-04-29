# EA contact-depth operator

Date: 2026-04-29

Status: `DER-PHYS + VAL`, canonical in PTC as the current `EA_eV`
precision layer.  The operator is derived as the continuous-depth rewrite
of the fixed PT contact projector stack and validated numerically against
the current PTC atomic EA benchmark.  The bridge statement "EA reads the
capture boundary" remains physical bridge language, not an unconditional
mathematical theorem.

Related files:

- `ptc/atomic_precision_frontier.py`
- `ptc/tests/test_atomic_precision_frontier.py`
- `docs/research/PT_ATOMIC_PRECISION_FRONTIER.md`

---

## 1. Problem

The canonical EA channel already includes:

```text
EA_geo
  -> capture-side CPR boundary continuum
  -> p threshold / p closure / d core residual fields
  -> continuous channel residual
```

It reaches:

```text
EA continuous channel  N=73  MAE=1.125303%  MAE_eV=0.007973  max=3.803883%
```

The remaining residuals are small but structured.  They concentrate on
the same geography that appears in the last IE pockets:

```text
f contact pressure,
d compact/contraction/penetration phases,
p residual second harmonic.
```

The aim is not to add fitted coefficients.  The aim is to find the
continuous PT operator whose sampled projections reproduce the residual
field.

---

## 2. Capture coordinate

EA is a capture observable.  If the neutral atom has filling `n` in an
active shell of capacity `N_l = 2(2l+1)`, the incoming electron reads:

```text
n -> n+1.
```

The receiver coordinate is therefore:

```text
x = (n+1)/N_l.
```

For closed shells the diagnostic reads the virtual entry edge:

```text
x = 1/N_l.
```

This keeps the coordinate inside the channel circle and avoids the
unphysical point `(N_l+1)/N_l`.

The perturbative scalar is the same scalar used by the CPR residuals:

```text
u_Z = (Z alpha)^2.
```

---

## 3. d-depth coordinate

The d channel has four realized radial depths in the table:

```text
period 4 -> tau = 0  compact first d shell
period 5 -> tau = 1  transition d shell
period 6 -> tau = 2  lanthanide-contracted d shell
period 7 -> tau = 3  superheavy d shell
```

Define:

```text
tau = period - 4.
```

The continuous depth basis on these four nodes is the cubic Lagrange
basis:

```text
L0(tau) = - (tau-1)(tau-2)(tau-3) / 6
L1(tau) =   tau(tau-2)(tau-3) / 2
L2(tau) = - tau(tau-1)(tau-3) / 2
L3(tau) =   tau(tau-1)(tau-2) / 6
```

Thus `L_i(j) = delta_ij`.  This is the key point: the former period
projectors are not independent empirical gates.  They are the sampled
values of a continuous depth coordinate.

---

## 4. d contact kernel

The first d shell needs the compact boundary kernel:

```text
sin(2 pi x) + cos(2 pi x) - sin(4 pi x).
```

The period-5 transition depth keeps the first oriented harmonic:

```text
sin(2 pi x).
```

The period-6 contracted depth keeps the second harmonic plus the cosine
compression:

```text
sin(4 pi x),  cos(2 pi x).
```

Using only fixed PT amplitudes, the continuous d kernel is:

```text
K_d(x,tau) =
  [CROSS_57 L0(tau) - S3 D3 L1(tau)] sin(2 pi x)
+ [CROSS_57 L0(tau) - D5^2 L2(tau)] cos(2 pi x)
+ [-CROSS_57 L0(tau) + S3 D5 L2(tau)] sin(4 pi x).
```

The superheavy basis `L3` is zero at this order.  This is not a fitted
zero; it says the remaining period-7 d contact signal is already below
the current EA precision layer and should be tested only after the IE weak
trace is derived.

On the actual d periods this continuous kernel samples to:

```text
tau=0 :  CROSS_57 [sin(2pi x) + cos(2pi x) - sin(4pi x)]
tau=1 : -S3 D3 sin(2pi x)
tau=2 :  S3 D5 sin(4pi x) - D5^2 cos(2pi x)
tau=3 :  0
```

This is exactly the previous fixed projector stack, but now written as
one depth-continuous operator.

---

## 5. Full EA contact-depth operator

The leading contact-depth field is:

```text
Lambda_EA,contact(Z) =
  u_Z [
    1_f K_f(x,period)
  + 1_d K_d(x,tau)
  + 1_p K_p(x)
  ].
```

with:

```text
K_f(x,period) =
  - S3 D3 |2x - 1| sqrt(period - 1)

K_p(x) =
  D5 D7 sin(4 pi x)
```

and `K_d` as above.

Every amplitude is already present in PT:

```text
S3 D3       active triangular contact pressure
S3 D5       triangular/pentagonal contraction coupling
D5^2        pentagonal self-coupling
D5 D7       pentagon/heptagon leakage
CROSS_57    Fisher pentagon/heptagon cross-face
```

No fitted coefficient appears.

---

## 6. Validation

Executable check:

```text
PYTHONPATH=. pytest -q ptc/tests/test_atomic_precision_frontier.py
```

The tests verify:

```text
1. the depth operator samples exactly the same field as the fixed stack;
2. the EA MAE crosses the 1% frontier;
3. the worst residual does not worsen.
```

Numerical result:

```text
continuous-channel EA        MAE = 1.125303%  max = 3.803883%
EA contact projector stack   MAE = 0.983780%  max = 3.534951%
canonical contact-depth EA   MAE = 0.983780%  max = 3.534951%
```

The equality between stack and depth operator is exact on the current
discrete table because the depth basis satisfies `L_i(j)=delta_ij`.

---

## 7. Interpretation

In ordinary atomic language, this operator packages residual
contraction, penetration, and contact-density effects.  In PT language,
the same statement is cleaner:

```text
EA reads the incoming capture edge.
The residual field is local on the receiver polygon.
Relativistic/contact strength enters through u_Z = (Z alpha)^2.
The d variations are samples of one depth coordinate tau.
```

This is why the correction improves both the global MAE and the maximum
error: it is acting where the residual geometry actually lives, not as a
global rescaling.

---

## 8. Hardening conditions

This result is activated in `EA_eV`.  The remaining work is not numerical
activation but proof hardening and propagation across dependent molecular
benchmarks.

Required before theorem-level hardening:

1. derive `K_f` directly from the PT local contact-density operator;
2. strengthen the radial-depth hierarchy from interpolation identity to
   structural lemma;
3. isolate the weak IE trace of the same operator;
4. re-run the molecular benchmarks that consume atomic IE/EA inputs.
