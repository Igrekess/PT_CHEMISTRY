# Backlog — précision LCAO/GIAO post-roadmap

**Contexte** : la roadmap `PROMPT_PT_LCAO_GIAO.md` est complète à 100 % pour s/p/d (272/272 tests, MAE 1.942 % intacte). Cependant **benzène σ_iso = +156 ppm vs exp +8 ppm**, ce qui révèle 3 limitations basis-set + Hamiltonien orthogonales à la roadmap GIAO :

## Trois limitations identifiées

### 1. Base STO valence-only → besoin de fonctions de polarisation d sur C/N/O

**Symptôme** : la base PT-LCAO actuelle pour C, N, O contient seulement {2s, 2p_x, 2p_y, 2p_z} (4 orbitales). Les fonctions d polarisantes (3d sur C, etc.) sont nécessaires pour décrire correctement la déformation des nuages électroniques sous champ magnétique B.

**Impact estimé** : ~30-50 % de l'écart sur NICS aromatique.

**Solution** : ajouter des fonctions polarisantes 3d sur C, N, O, F (et 4d sur Si, P, S, Cl). Choisir ζ via PT (pas via fit). Une approche PT-pure : ζ_polar = γ_3 × Z_eff (déduction du gradient anomal).

**LOC estimé** : ~100-150 lignes.
**Effort** : 1 session.

### 2. Hückel K=2 → besoin du H_eff dérivé T³ depuis transfer_matrix.py

**Symptôme** : la formule actuelle `H_ij = K × (H_ii + H_jj)/2 × S_ij` avec K=2 (entier PT-pur) reste empirique. Les eigenvalues MO résultantes sont mal calibrées : HOMO-LUMO gap de benzène ~25 eV vs réel ~6.5 eV → suppression de σ^p par facteur 4.

**Impact estimé** : ~30-50 % de l'écart sur NICS.

**Solution** : remplacer le Hückel par le Hamiltonien T³ (Z/3 × Z/5 × Z/7) qui est **dérivé** de la théorie de la persistance dans `ptc/transfer_matrix.py`. Le couplage inter-atomique via T³ donne des éléments off-diagonaux PT-natifs sans paramètre ajusté.

**LOC estimé** : ~250 lignes.
**Effort** : 2 sessions (extraction de la matrice T³ + intégration en remplacement de hueckel_hamiltonian).

### 3. CP-PT découplée → besoin de CP-PT couplée (Fock matrix)

**Symptôme** : la formule actuelle ignore les termes de Coulomb/exchange à 2 électrons dans la réponse à la perturbation magnétique :

σ^p_αβ = (4 α²) Σ_ai L_β_ai M_α_ia / (ε_a − ε_i)

Pas de boucle d'auto-consistance (uncoupled). En CP-PT couplée (Fock complet), une boucle SCF supplémentaire mélange les contributions Coulomb/exchange dans les MOs perturbés.

**Impact estimé** : ~10-30 % de l'écart sur NICS.

**Solution** : implémenter la matrice de Fock à 2 électrons (intégrales (μν|κλ) sur les STOs PT-pures) puis CP-PT couplée auto-consistante.

**LOC estimé** : ~200 lignes (matrice de Fock) + ~150 (CP-PT couplée) = ~350.
**Effort** : 1-2 sessions.

## Ordre de travail recommandé

1. **Basis extension** (limitation #1) — modulaire, peut être commit dans ses fonctions standalone
2. **Hamiltonien T³** (limitation #2) — dépend uniquement de la base + transfer_matrix.py existant
3. **CP-PT couplée** (limitation #3) — dépend des MOs corrects de l'étape 2

Chaque étape est validée par : MAE 806 mol intacte, 272 tests existants intacts, **et** une amélioration de NICS_iso(benzène) vers la cible expérimentale +8 ppm.

## Cible de validation finale

```
benzène centre :
  σ_iso actuel  = +156 ppm    (NICS = -156)
  σ_iso cible   = +8 ± 2 ppm  (NICS = -8 ± 2, exp aromatic)

S₃ centre :
  σ_iso cible   = +20 ppm     (NICS_zz ~ -10 ppm aromatic)

Cp⁻ (cyclopentadienyl) :
  NICS_iso cible = -13 ppm

Al₄²⁻ (PT roadmap original target) :
  NICS_iso cible = -34 ppm   ← pourrait nécessiter aussi Phase A f-block
```

## État de référence à préserver

- 272 tests LCAO PASS
- MAE 806 mol = 1.942 %
- 86/86 NICS scalar tests PASS (chantier 4f)
- H atome σ_iso = 17.75 ppm (Lamb exact)
- Gauge invariance < 1e-6 ppm

## Plan de fichiers

```
ptc/lcao/atomic_basis.py          ← extension polarisation (limitation #1)
ptc/lcao/density_matrix.py        ← H_eff T³ (limitation #2)
ptc/lcao/fock.py                  ← NOUVEAU : 2-electron integrals + Fock
ptc/lcao/cp_pt.py                 ← NOUVEAU : CP-PT couplée (limitation #3)
ptc/lcao/shielding.py             ← intégration via paramagnetic_shielding_tensor
ptc/tests/test_lcao_polarisation.py  ← NOUVEAU
ptc/tests/test_lcao_t3_hamiltonian.py ← NOUVEAU
ptc/tests/test_lcao_cp_coupled.py     ← NOUVEAU
```
