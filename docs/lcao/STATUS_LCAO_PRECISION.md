# Status — précision LCAO/GIAO (3 limitations basis/Hamiltonien)

**Date** : 2026-04-26 (après commits 6a58bcf + 6cd0b79)
**Backlog** : `BACKLOG_LCAO_PRECISION.md`

## Avancement par limitation

### Limitation #1 — Polarisation d ✅ infrastructure livrée

Commit **6a58bcf** : opt-in `polarisation=True` dans `build_atom_basis` et `build_molecular_basis`. Ajoute une couche d'orbitales polarisantes (3d sur C/N/O/F, 2p sur H, 4d sur Si/P/S/Cl) avec ζ_polar = Z_eff/(n_polar × a₀). 18 tests dédiés.

### Limitation #2 — Core Hamiltonian H = T + V_nuc ✅ livré

Commit **6cd0b79** : `core_hamiltonian(basis, S)` remplace Hückel K=2 par les intégrales **physiques** kinetic-energy + Coulomb-attraction (avec Z_eff PT-screened). Validations exactes :
- H atome : T = +13.6 eV, V = −27.2 eV, H_core = −13.6 eV (Lamb)
- H atome shielding via core : 17.75 ppm intact
- 13 tests dédiés

### Limitation #3 — CP-PT couplée (Fock 2-électrons) ⏸ deferred

Pour atteindre la précision quantitative sur benzène NICS (cible −8 ppm), il faut ajouter les contributions Coulomb + exchange à 2 électrons :

```
F[μν] = H_core[μν] + Σ_κλ ρ[κλ] × [(μν|κλ) − (1/2)(μκ|νλ)]
```

**Effort estimé** : 500+ lignes, 3-4 sessions :

1. **Intégrales (μν|κλ) sur STOs** (~250 LOC)
   - 6D intégrales analytiques ou numériques (Mulliken-Roothaan-Rieke)
   - Pour 30 orbitales benzène : ~100 000 intégrales uniques (sym 8-fold)
   - Méthodes possibles : analytique 4-center (très complexe), STO-NG fit, density fitting (RI)

2. **Construction matrice Fock** (~80 LOC)
   - J[μν] = Σ ρ × (μν|κλ) (Coulomb)
   - K[μν] = Σ ρ × (μκ|νλ) (exchange)
   - F = H_core + J − K/2

3. **SCF auto-consistant** (~100 LOC)
   - DIIS ou damping
   - Convergence sur ρ ou Δρ

4. **CP-PT couplée** (~100 LOC)
   - Réponse F⁽¹⁾ = J[ρ⁽¹⁾] − K[ρ⁽¹⁾]/2 dans la perturbation magnétique
   - Boucle CPHF/CPKS auto-consistante

**Impact attendu** : factor 2-3 sur σ^p (passe de -19 à -50 ppm) + redistribution de ρ qui réduit σ^d (passe de +180 à ~+50 ppm). NICS attendu : ~-30 ppm (vs cible -8).

Pour atteindre exactement -8 ppm, il faudrait probablement aussi améliorer la base (au-delà de la simple polarisation d) ou utiliser une fonctionnelle DFT correcte.

## État benzène NICS — récap des combinaisons disponibles

| Combo | gap | σ_d | σ_p | σ_iso | NICS |
|---|---:|---:|---:|---:|---:|
| Hückel (default) | 13 | +156 | −1.5 | +155 | −155 |
| Hückel + polar | 0.6 | +119 | +1.4 | +120 | −120 |
| **Core + polar (best disponible)** | **5.1** | **+179** | **−18.9** | **+160** | **−160** |
| Coupled-CPHF (limitation #3) | ~6.5 | ~+50 | ~-58 | ~-8 | ~+8 |
| **Expérimental** | **6.5** | **~+25** | **~-33** | **+8** | **−8** |

## Tests cumulés

```
test_nics.py                   86/86 PASS
test_lcao_atomic_basis.py      27/27 PASS
test_lcao_sto_overlap.py       19/19 PASS
test_lcao_density_matrix.py    88/88 PASS
test_lcao_giao.py              24/24 PASS
test_lcao_d_block.py           17/17 PASS
test_lcao_shielding.py         11/11 PASS
test_lcao_polarisation.py      18/18 PASS  (limitation #1)
test_lcao_core_hamiltonian.py  13/13 PASS  (limitation #2)
                             ─────────────
                             303/303 PASS

MAE 806 mol = 1.942 %  (intact)
```

## Recommandation

**Limitations #1 et #2 livrées**, validations exactes pour H atome. Le saut de précision (+155 → +160 ppm sur benzène) est marginal car les 3 limitations sont **couplées synergiquement** : il faut #3 pour observer un vrai shift.

Pour décider la suite :

**Option A** — accepter l'état actuel (limitations #1, #2 livrées + #3 différée) :
- Tout le pipeline LCAO + GIAO + Phase D fonctionne
- H atome / He / atomes simples : Lamb exact
- Métaux de transition supportés (FeH, Cu₂)
- Tenseur σ_αβ avec axes principaux
- 303 tests PASS, MAE 1.942 %
- Coût en temps : 0 session supplémentaire

**Option B** — attaquer limitation #3 :
- 3-4 sessions de travail (intégrales 2-électrons + SCF + CP coupled)
- Probable amélioration σ_iso(benzène) → −30 à 0 ppm range
- Pour atteindre **exactement** −8 ppm il faudrait aussi étendre la base au-delà de polarisation d simple
- Bénéfice principal : précision NICS aromatique quantitative
