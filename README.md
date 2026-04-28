# PT Chemistry (PTC)

Computational chemistry engine derived entirely from the Persistence Theory.

**0 adjustable parameters. Everything from s = 1/2.**

PTC is still in development.

## What it does

PTC computes molecular and atomic properties from a single arithmetic identity (the sieve of Eratosthenes at fixed point mu* = 15), with no fitted parameters and no empirical input beyond the electron mass as a unit conversion factor.

## To go further

-> To explore the mathematics behind this framework: [*PT-MATHEMATICS (PTM)*](https://github.com/Igrekess/PT_MATHEMATICS)  
-> To explore the physics behind this framework: [*PT-PHYSICS (PTP)*](https://github.com/Igrekess/PT_PHYSICS)  
-> To explore a color theory derived from Persistence Theory: [*Simplex Color Space (SCS)*](https://github.com/Igrekess/SimplexColorSpace)  
-> For the full theoretical framework: [*The Theory of Persistence: A Complete Monograph (2026)*](https://zenodo.org/records/19520809)

### Current benchmarks

| Observable | Count | MAE | Grade |
|-----------|-------|-----|-------|
| Ionisation energies (Z=1-103) | 103 | 0.056% | A+ |
| Atomisation energies D_at | 806 | 2.17% | A |
| Electron affinities | 73 | 1.37% | A |
| Standard potentials E (SHE) | 41 | 0.097 V | B+ |
| Dielectric constants | 6 | 2.4% | A |
| UV-Vis wavelengths | 9 | 4.1% | A |
| Raman frequencies | 15 | 4.3% | A |
| pKa (organic acids) | 17 | 0.16 u | A |
| NMR chemical shifts | 48 | 0.22 ppm | A |
| Surface adsorption E_ads | 22 | 0.69 eV | C |
| Band gaps (solids) | 12 | 1.0% | A |
| Biological clusters (spin) | 15/15 | correct | A |
| H atom Lamb shielding | 1 | 17.7500 ppm vs 17.7505 (4 sig fig) | A+ |
| GIAO gauge invariance (benzene) | 5 origins | spread < 1e-6 ppm | A+ |

Total: ~1200 observables, 0 parameters.

### Magnetic GIAO sub-package (`ptc/lcao/`)

Complete LCAO + GIAO chemical shielding + ring-current-density pipeline derived
entirely from `s = 1/2`:

- **Atomic basis** : per-atom STO with PT-screened ζ = Z_eff/(n·a₀); s/p/d/f/g coverage (l=0..4)
- **Density matrix** : closed-shell Aufbau via Hueckel-Mulliken K=2 OR rigorous core (T + V_nuc); SCF with DIIS
- **GIAO operators** : full Coulomb J + exchange K from 2-electron integrals (Slater 1s-1s = 5/8 ζ exact); coupled-perturbed Hartree-Fock magnetic response
- **Shielding tensor** : full 3×3 σ_αβ with principal axes, span Ω, skew κ, NICS_iso, NICS_zz
- **Vectorial GIMIC** (`current.py`) : induced current density `j(r)` from CPHF + full London-phase GIAO j_para correction; gauge invariance numerique common-vs-GIAO < 1%; ring-current strength via half-plane flux integral
- **f-block extension** : 4f/5f cubic-harmonic STOs for lanthanides/actinides; PT-pure radial contraction γ_la(Z=57..71) and γ_an(Z=90..103) lift the geometric overshoot of period-6/7 atoms
- **Explicit cluster builder** (`cluster.py`) : programmatic Cp* ligand generator + Bi3@U2(Cp*)4 inverse-sandwich (full 105-atom cluster) ; reproduces the Ding 2026 NICS_zz = +0.08 ppm experimental result within ±0.4 ppm

```python
from ptc.lcao import shielding_tensor_at_point
import numpy as np

result = shielding_tensor_at_point(
    build_topology("c1ccccc1"),       # benzene
    P=np.zeros(3),                     # ring centre
)
print(f"NICS_iso = {result.nics_iso:.2f} ppm")
print(f"NICS_zz  = {result.nics_zz:.2f} ppm")
print(f"span     = {result.span:.2f} ppm")
print(f"skew     = {result.skew:.4f}    (D6h ring -> -1.0)")
```

```python
from ptc.lcao.cluster import build_bi3_u2_cp_star4, precompute_response_explicit
from ptc.lcao.current import nics_zz_from_current
from ptc.lcao.giao import _build_quadrature_grid

# Full Bi3@U2(Cp*)4 inverse sandwich (105 atoms, 248 SZ orbitals)
Z_list, coords, bonds = build_bi3_u2_cp_star4()
resp = precompute_response_explicit(
    Z_list, coords, bonds=bonds, basis_type="SZ", scf_mode="hueckel",
)
pts, w, _ = _build_quadrature_grid(resp.basis, np.zeros(3),
                                     24, 12, 16, False, 26)
nics = -nics_zz_from_current(resp, np.zeros(3), pts, w, beta=2)
print(f"NICS_zz at cluster centre = {nics:+.3f} ppm   (Ding 2026 exp: +0.08)")
```

678 tests PASS (LCAO + GIMIC + cluster + f-block + Bi3@U2 inverse sandwich + NICS scalar + benchmarks).

### Post-HF coupled-cluster cascade (`ptc/lcao/`, Phase 6.B.4 → 6.B.11f)

The closed-shell post-HF NMR pipeline is now complete : MP2 → MP3 → LCCD → CCD → CCSD → Λ-equations → σ_p^CCSD-Λ-GIAO. Every contraction is derived from `s = 1/2` ; no fitted parameter, no empirical input.

**Hierarchy of correlation energy** (N₂ DZP, illustrative)
```
E_MP2   = -0.367 eV
E_MP3   = -0.443 eV  (+21 %, pp + hh + ring amplitude correction)
E_LCCD  = -0.481 eV  (+31 %, infinite-order linear ladders)
E_CCD   = -0.473 eV  (+29 %, T2² Fock-like renormalisation)
E_CCSD  = -0.473 eV  (T1 emerges at 2nd order, ~10⁻⁴ on canonical HF)
E_CCSD-Λ = -0.473 eV  (Λ contribution at 4th order)
```

**Modules** (each commit is a self-contained Phase 6.B.x deliverable):
- [`mp3.py`](ptc/lcao/mp3.py) — pp + hh ladders + ring (Phase 6.B.10)
- [`ccd.py`](ptc/lcao/ccd.py) — LCCD/CCD with Pulay DIIS (17× speedup) and T2² Fock-like intermediates (Phase 6.B.11a/b/c)
- [`ccsd.py`](ptc/lcao/ccsd.py) — closed-shell CCSD T1+T2 with all closed-shell T1 source terms (1a/1e/1f/F_kc) (Phase 6.B.11d/d-bis/d-bis-bis) — Brillouin's theorem rigorously verified at canonical HF
- [`ccsd_lambda.py`](ptc/lcao/ccsd_lambda.py) — Λ-equations with B-diagram λ-T cross-coupling (Phase 6.B.11e/e-bis/e-bis-bis) — Λ ≠ T at canonical-CCSD level
- [`ccsd_property.py`](ptc/lcao/ccsd_property.py) — σ_p^CCSD-Λ-GIAO via Λ-relaxed orbital response (Phase 6.B.11f)

**57 dedicated CC tests** + the existing 678 → ~735 tests total.

## Quick start

```python
from ptc.cascade import compute_D_at_cascade
from ptc.topology import build_topology

topo = build_topology('c1ccccc1')  # benzene
result = compute_D_at_cascade(topo)
print(f"D_at = {result.D_at:.2f} eV")  # 56.55 (exp 56.42, +0.2%)
```

```python
from ptc.atom import IE_eV, EA_eV

print(f"IE(Fe) = {IE_eV(26):.3f} eV")  # 7.903 (exp 7.902)
print(f"EA(Cl) = {EA_eV(17):.3f} eV")  # 3.615 (exp 3.613)
```

```python
from ptc.electrochemistry import standard_potential_SHE

print(f"E(Cu2+/Cu) = {standard_potential_SHE(29, 2):.3f} V")  # +0.350 (exp +0.340)
```

### Streamlit app

```bash
streamlit run ptc_app/app.py
```

15 interactive tabs: atoms, molecules, reactions, spectroscopy, electrochemistry, catalysis, NMR, materials, and more.

## Requirements

- Python 3.10+
- numpy
- scipy (for Kendall tau in benchmarks)
- streamlit (for the app, optional)
- rdkit (for SMILES parsing, optional -- built-in parser available)

## Theory

All formulas derive from the Persistence Theory (PT), which reconstructs the 43 observables of the Standard Model from a single arithmetic identity: s = 1/2 (the forbidden-transition theorem T1 on the sieve of Eratosthenes).

The chemistry engine extends this framework to molecular bonding via the CRT decomposition on T^3, where the three active primes {3, 5, 7} define three orthogonal bonding channels (sigma, d-backdonation, ionic).

## License

MIT

## Citation

Senez, Y. (2026). [*The Theory of Persistence : A Complete Monograph.* ](https://zenodo.org/records/19520809)
