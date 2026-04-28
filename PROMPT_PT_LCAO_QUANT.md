# PROMPT_PT_LCAO_QUANT — Chantier 3 : contracted polarisation PT-pure (à la Dunning)

## État actuel — Phase 6.B.4 → 6.B.7 livrées (10+ commits, 174/174 tests PASS)

Le pipeline σ_p^MP2-GIAO Stanton-Gauss est complet, vectorisé, validé contre la
littérature 1996, et progressivement raffiné :

| Phase | Livrable | Commit |
|---|---|---|
| 6.B.4   | Z-vector closure (Lagrangien analytique FD-validé)        | `358197b` |
| 6.B.4b  | Full Stanton-Gauss σ_p coupled (HF/LO/full retournés)     | `776cadc` |
| 6.B.4c  | Validation DZP+ multi-system (4 molécules)                | `5977ff9` |
| 6.B.4d  | Vectorisation BLAS mo_eri_block (×2)                      | `ed62d67` |
| 6.B.4e  | scipy.cdist (×3-4 sur grid distances)                     | `0c44bcd` |
| 6.B.4f  | TZ2P basis (DZ + d-polar + f-polar)                       | `89fa257` |
| 6.B.5   | Chemical shifts pipeline + scalar_relativistic flag γ_rel | `2190eb6` |
| 6.B.6   | pt-shielding zeta_method propagé + Cowan-Griffin module   | `7da6ff4` |
| 6.B.7a  | Treutler-Ahlrichs M4 log grid (Cl 1s converge à n=18)     | `0e6cfc0` |
| 6.B.7b  | TZ cumulative cascade (cond ×200 mieux)                   | `688edc5` |

### Stack progression N₂ σ_p^HF (validation Stanton-Gauss 1996 cc-pVDZ -440 ppm)

| Configuration | σ_p^HF (ppm) | % de cc-pVDZ |
|---|---|---|
| Initial (pt zeta, valence-only)        | -114 | 26 % |
| + zeta_method='pt-shielding'           | -137 | 31 % |
| + include_core=True                     | -157 | 36 % |
| + TZ cumulative cascade                 | -165 | 38 % |
| **Cible Dunning cc-pVDZ**              | **-440** | **100 %** |

Reste 62 % d'écart. La trajectoire est claire : à chaque levier débloqué, on gagne
5-15 points. Le prochain levier physique est la **contraction des polarisations**.

---

## Mission Chantier 3 — Contracted polarisation PT-pure

### Le problème actuel

Notre PT-DZP a 2 zetas par valence-shell (avec ratio cumulatif 0.808 → 0.563),
plus 1 polarisation (l_val+1) avec un seul zeta. Pour résoudre la queue
intermédiaire d'une orbitale 2p de carbone, on a 2 fonctions ; Dunning cc-pVDZ
en a 4-6 primitives **contractées** en 1-2 fonctions de base, ce qui :
- augmente la flexibilité radiale (les 4-6 primitives couvrent plus d'échelles)
- garde le nombre de DoF moléculaire raisonnable (1-2 fonctions de base par contracted)
- évite la dépendance linéaire (les coefficients de contraction figent une combinaison stable)

### Ce qu'il faut implémenter

Une **fonction de base contractée** :

```
χ_contracted(r) = Σ_i c_i × g_i(r ; ζ_i)
```

où `g_i` sont des STOs primitifs avec ζ_i différents et `c_i` les coefficients
de contraction. La fonction de base est UNE seule entité du point de vue
moléculaire (un seul indice μ, une seule colonne dans la matrice MO), mais elle
intègre N primitives.

### Voie PT-pure pour les coefficients c_i

Dunning utilise des coefficients dérivés du SCF atomique (pour minimiser
l'énergie atomique en présence de la base étendue). En PT, on a deux options :

**Option A — Coefficients depuis l'atom PT spectral**
Notre `ptc/atom.py` calcule l'énergie atomique IE(Z) avec MAE 0.057% via le
cascade PT depuis s=1/2. Pour chaque shell (n, l), il existe une enveloppe
radiale ψ_n_l(r) résultant du cascade. On pourrait extraire ses coefficients
de Fourier sur la base de primitives STO γ_3^k z_base (k = 0, 1, 2, 3) :

```
c_i = ⟨ψ_n_l_PT | g_i⟩
```

avec ψ_n_l_PT calculé via `atom.py` (potentiel PT-screening + diagonalisation
radiale 1D). Cette extraction se fait UNE fois par atome+shell, stockée
comme constante PT.

**Option B — Coefficients analytiques de la cascade**
Plus PT-natural : les coefficients SUIVENT la cascade spectrale.
Pour 4 primitives à ζ_k = z_base × γ_3^k :

```
c_k = γ_3^(k(k-1)/2) × N_norm
```

Ou un autre arrangement où l'enveloppe contractée reproduit Σ_k γ_3^k de
manière "PT-spectrale". À tester.

**Option C — Régression Slater-équivalente (référence)**
Pour chaque shell (n, l) de chaque atome, fit les c_i de manière à ce que
χ_contracted(r) = N × r^(n-1) × e^(-ζ_Slater × r) (la STO Slater de
référence). Pas PT-natural mais utile comme garde-fou.

### Architecture minimale

1. **Nouveau type `PTContractedOrbital`** dans `ptc/lcao/atomic_basis.py` :
   ```python
   @dataclass
   class PTContractedOrbital:
       Z: int
       n: int
       l: int
       m: int
       primitives: list[tuple[float, float]]   # [(ζ_i, c_i), ...]
       occ: float
   ```

2. **Évaluation et gradients** dans `ptc/lcao/giao.py` :
   - `evaluate_sto_contracted(orb, points, atom_pos)` :
     somme pondérée des évaluations primitives.
   - `evaluate_sto_gradient_contracted(orb, points, atom_pos)` :
     même chose pour le gradient.
   - Reuse maximal du code STO existant (les primitives sont des STO standards).

3. **Builder** dans `atomic_basis.py` :
   - Nouveau `basis_type='pVDZ-PT'` ou `'CDZP'` (contracted DZP) :
     - Valence : 4 primitives par (n, l) à ζ_k = z_base × γ_3^k, contractées avec coefficients PT.
     - Polarisation : 2 primitives à ζ_polar1 et ζ_polar2 = γ_3 × ζ_polar1, contractées en 1 fonction.
   - Fallback : si options Option A ou B ne convergent pas, utiliser DZP non-contracté.

4. **Adaptation downstream** :
   - `overlap_matrix`, `kinetic_matrix`, `nuclear_attraction_total`,
     `coulomb_J_matrix`, `exchange_K_matrix` : aucun changement nécessaire si
     l'orbital contractée est traitée comme une "STO élargie".
   - `evaluate_sto_*` doit dispatcher sur le type d'orbital (single zeta vs contracted).

### Critères de succès Chantier 3

1. **Conditioning** : cond(S) reste sous 1e5 sur N₂ et benzène DZP.
2. **σ_p^HF magnitude** sur N₂ doit franchir 50 % de cc-pVDZ
   (i.e., |σ_p^HF| > 220 ppm). Sinon, la flexibilité radiale est encore trop
   limitée et il faut passer à QZP / pVDZ-PT à 6 primitives.
3. **MP2 correction** Δ% sur N₂ doit augmenter (actuellement -3.6 %, S-G donne +18 %).
   Si on franchit -10 %, on est à mi-chemin de Stanton-Gauss.
4. **Tests** : ≥ 12 nouveaux tests sur la contraction (shape, normalisation, gradient
   cohérent, équivalence avec STO single-zeta dans le cas N=1, intégrales
   matricielles cohérentes avec versions analytiques).
5. **Performance** : pas de régression sur les benchmarks existants.

### Décisions à prendre en début de session

1. **Option PT-pure pour les coefficients** : A (depuis atom.py spectral),
   B (analytique cascade), ou C (fit Slater-équivalent) ? Recommandation :
   commencer par C pour valider la machinerie, puis remplacer par A.

2. **Nombre de primitives par shell** : 3 ou 4 ? Plus = plus flexible mais
   plus de compute. Recommandation : 4 primitives (z_base × γ_3^k, k=0..3) pour
   couvrir les régimes core/inner-valence/valence/diffuse.

3. **Polarisation contractée ou découplée** : la 2ème shell de polarisation
   (l_val+2 = f) doit-elle être contractée avec la 1ère (l_val+1 = d) ?
   Recommandation : non, garder f découplé (1 primitive) pour Phase 1 ; passer
   à p+f contracté en Phase 2.

4. **Naming** : `'pVDZ-PT'` (Pople-Hehre style) ou `'CDZP'` (Contracted DZP) ?
   Recommandation : `'pVDZ-PT'` pour clarté avec littérature.

### Plan d'implémentation (3 étapes, ~1-2 jours)

**Étape 1 — Infrastructure orbital contractée (4-6 h)**
- `PTContractedOrbital` dataclass.
- `evaluate_sto_contracted` + `evaluate_sto_gradient_contracted` (boucle sur primitives).
- Tests : norme, gradient FD, équivalence STO single-zeta pour N=1.

**Étape 2 — Builder pVDZ-PT (4 h)**
- Coefficients option B (cascade analytique) en première itération.
- `basis_type='pVDZ-PT'` ajouté à `_BASIS_TYPES` et `build_atom_basis`.
- Tests sur conditioning N₂ + H₂O.

**Étape 3 — Validation N₂ benchmark + pivot vers option A si nécessaire (4 h)**
- Re-run la validation N₂ DZP Becke avec `basis_type='pVDZ-PT'`.
- Si σ_p^HF ne franchit pas 50 % de cc-pVDZ, switch vers option A
  (extraire coefficients depuis ψ_PT_atom).
- Memory + commit Phase 6.B.8.

### Limites résiduelles à NE PAS attaquer en cette session

- **CCSD au-delà de MP2** : reste hors scope. Continuation à la suite de
  Phase 6.B.8 si la basis est satisfaisante.
- **Implémentation analytique du STO kinetic** : suffit pour les chantiers
  actuels grâce au log grid.
- **GPU** : si toute la stack atteint 50 % de cc-pVDZ, performance n'est plus
  le bottleneck. Différé.

### Mémoires à mettre à jour en fin de session

- `project_pt_lcao_quant.md` : ajouter section Phase 6.B.8 (Contracted polar PT).
- `MEMORY.md` : entry one-line.
- Si σ_p^HF franchit 50 % cc-pVDZ : reformuler la conclusion comme
  "PT-NMR validé quantitativement vs Stanton-Gauss à mieux que 50 %, prêt pour
  rédaction d'article PT_NMR_VALIDATION".

---

## Annexes — Référence pour le sprint

### A. Comparaison Dunning cc-pVDZ vs notre PT-DZP

| Atom | Dunning cc-pVDZ | PT-DZP (current) |
|---|---|---|
| H | (4s, 1p) → (2s, 1p) [3 primitives s contractées] | (2s, 1p) — 1 primitive each |
| C | (9s4p1d) → (3s2p1d) | (4s2p1d) — 1 primitive each |

Le facteur ~3 entre primitives Dunning et nos primitives est ce qui manque.

### B. Formules clés

Coefficient cascade PT (option B) :
```
c_k = γ_3^k × γ_5^k(k-1)/2   pour k = 0, 1, 2, 3
ζ_k = z_base × γ_3^k
```

Normalisation après contraction :
```
Σ_ij c_i c_j S_ij(ζ_i, ζ_j) = 1
```

où S_ij est l'overlap entre primitives.

### C. Tests de référence

- ψ_C_2p_PT(r) doit reproduire ⟨r⟩_C_2p ≈ 1.59 Bohr (expérimental).
- σ_p(C dans CH4)/σ_p(C dans CO) ratio expérimental = -184/-2.3 ≈ +80
  (ratio adimensionné, validable même avec σ_p sous-évalué en absolu).
