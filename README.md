# PT Chemistry (PTC)

Computational chemistry engine derived entirely from the Persistence Theory.

**0 adjustable parameters. Everything from s = 1/2.**

## What it does

PTC computes molecular and atomic properties from a single arithmetic identity (the sieve of Eratosthenes at fixed point mu* = 15), with no fitted parameters and no empirical input beyond the electron mass as a unit conversion factor.

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

Total: ~1200 observables, 0 parameters.

## Architecture

The D_at engine uses a bifurcated cascade on T^3 = Z/3Z x Z/5Z x Z/7Z, reflecting the two phases of T4 spectral convergence (Mertens molecular):

```
cascade.py              -- unified entry point
  |
  +-- Phase 1 (n <= 7)  -- Perron eigenvalue lambda_0(T^4)
  |   transfer_matrix.py    dense molecular graphs
  |   + LP->sigma, VSEPR, formal charges
  |
  +-- Phase 2 (n >= 8)  -- per-bond cascade P0->P1->P2->P3
      cascade_phase2.py     sparse graphs, 17 NLO corrections
      cooperative, Dicke, T^3 perturbative, LP->pi,
      ring strain, halogen pi-drain, etc.
```

The bifurcation at n = P_3 = 7 is structural (T4, D08): NLO corrections converge only when the molecular graph has enough second-shell neighbors for Mertens summation.

## PT mechanisms discovered (17)

1. sin^2_3 LFER (pKa)
2. Cascade P_1 fold-back (dielectric)
3. Kirkwood P_3 (dipole correlation)
4. Inclusion-exclusion P_1 U P_2 (UV n->pi*)
5. Double-pass LP (UV sigma->sigma*)
6. Isolated pi bifurcation (UV pi->pi*)
7. s-band bifurcation (E, d-block)
8. CFSE metallic (E, d-block)
9. Relativistic 5d (E, Pt/Au/Hg)
10. Hybrid EP+D_KL (activation energy)
11. Hexane P_0-S_3 (non-polar dielectric)
12. LP->sigma back-donation (NH3, k=0 reference on Z/6Z)
13. LP->pi cross-channel (heterocycles, P(P_1 U P_2) = 0.371)
14. Blyholder-PT 3 channels (catalysis: sigma + pi_back - Pauli)
15. d10 monovalent enhancement (E Cu+)
16. d5s1 3d exchange-locking (E Cr)
17. LP-LP Pauli repulsion (F2 bond length)

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

See: *Persistence Theory: A Complete Monograph* (2026).

## License

MIT

## Citation

```
SENEZ, Y. (2026). Persistence Theory: From Prime Gaps to the Standard Model.
```
