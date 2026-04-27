# PROMPT_PT_LCAO_QUANT — Mission : Z-vector MP2-GIAO (Phase 6.B.4)

## Statut hérité (livré 2026-04-27, 284 tests PASS, 0 régression)

PT_LCAO+GIMIC est une stack PT-pure end-to-end. Le pipeline reproduit le
NICS_zz expérimental de Bi₃@U₂(Cp*)₄ (Ding 2026, +0.08 ppm) à 0.39 ppm
près sur 105 atomes / 248 orbitales SZ — bien dans le critère cible
±2 ppm.

Composants déjà en place (ne pas refaire) :

| Module | Capacité |
|---|---|
| `lcao/relativistic.py` | γ_la (4f, Z=57..71), γ_an (5f, Z=90..103), γ_rel Dirac. PT-pur depuis γ₅, γ₇. |
| `lcao/giao.py` | STO + gradient analytique pour l=0..3, l=4 via FD. GIAO L matrices per-pair midpoint. |
| `lcao/atomic_basis.py` | Cubic harmonics réelles l=3 (7 fonctions), l=4 (9 fonctions). DZP étendu d-block (f-polar) et f-block (g-polar). `(n-1)d` shell opt-in. |
| `lcao/density_matrix.py` | f-block accepté sans gate. Hueckel-Mulliken K=2. |
| `lcao/fock.py` | HF SCF avec DIIS, CPHF coupled response, level_shift pour clusters frontier-degenerate. |
| `lcao/current.py` | j_para CPHF + j_dia (common, ipsocentric, full GIAO London-phase TERM2). NICS via Biot-Savart. **`nics_profile_cluster(z)` profil le long d'un axe.** |
| `lcao/cluster.py` | `build_explicit_cluster`, **`build_inverse_sandwich(X, n, M, ligands)` générique**, `build_bi3_u2_cp_star4`, `precompute_response_explicit`, **`precompute_response_mp2_explicit`** (HF SCF + MP2 + relax + CPHF). |
| **`lcao/mp2.py`** | **MP2 amplitudes + énergie + 1-RDM blocs occ/vir + pont MO→AO + `mp2_relax_orbitals` (one-shot Fock rebuild via ρ_MP2). Captures occupation-shift (d_occ, d_vir) seulement.** |
| `data/inverse_sandwich_xray.py` | Référence X-ray famille (Ding 2026 + Arnold 2019 + Liddle 2020 + Long 2018 + 2 PT predictions). |
| `ptc_app/components/aromaticity_panel.py` | Panel UI : 3 presets cluster, scan combinatoire 7 classes (`Inverse sandwich X_n@M_2`), profil NICS(z) bouton, expander X-ray. |

---

## Mission de cette session — Z-vector pour MP2-GIAO complet

Le pipeline MP2 actuel intègre l'effet d'**occupation-shift** (blocs
`d_occ` et `d_vir` de la 1-RDM MP2) dans CPHF via `mp2_relax_orbitals`.
La contribution restante — la **rotation orbitale** capturée par le
**Z-vector** — est la *vraie* limite résiduelle pour MP2-GIAO.

Sans Z-vector, la 1-RDM MP2 manque le bloc off-diagonal occ-virt :

```
D_MP2 = D_HF + d_occ ⊕ d_vir + 2·Z       (le 2·Z manque actuellement)
```

Le Z-vector est l'amplitude `Z_ai` qui résout

```
A·Z = -L            (équation Z-vector, Stanton-Gauss 1992)
```

où :
- `A` est le hessien orbital (le MÊME que dans CPHF — réutilisable)
- `L_ai` est le Lagrangien MP2 (gradient de E_MP2 par rapport à la
  rotation orbitale virtuelle-occupée)

Une fois Z connu, la 1-RDM MP2 *relaxée* est complète, et la formule
classique de shielding la prend en argument :

```
σ_p^MP2 = -2 Σ_β Tr[ M_β · (D_HF + d_occ ⊕ d_vir + 2·Z) ]
```

C'est l'achèvement de Phase 6.B.4 (Stanton-Gauss MP2-GIAO).

---

## Lecture obligatoire avant de coder

1. `ptc/lcao/mp2.py` — particulièrement `mp2_density_correction`,
   `mp2_density_correction_AO`, `mp2_relax_orbitals`. La structure
   d_occ/d_vir est déjà construite ; Z s'ajoute en bloc off-diagonal.
2. `ptc/lcao/fock.py::coupled_cphf_response` — le solveur CPHF
   itératif. La structure du Z-vector est rigoureusement la même
   (orbital Hessian, K^(1) iteration). Réutiliser cette infrastructure.
3. `ptc/lcao/__init__.py` — re-exports MP2 ; ajouter ce qu'il faut.
4. `ptc/tests/test_lcao_mp2.py` — 12 tests existants pour MP2 + 2 pour
   MP2-CPHF. Étendre pour Z-vector.
5. Référence : Stanton & Gauss, J. Chem. Phys. 97 (1992) 6602 ; ou
   Helgaker & Jørgensen, *Molecular Electronic-Structure Theory* §13.7.

---

## Plan d'attaque

### Étape 1 — Lagrangien MP2

Construire `L_ai` (RHS du Z-vector) à partir des amplitudes MP2 et des
ERIs. Formule fermée pour MP2 closed-shell :

```
L_ai = -2 Σ_jbk t_ij^bk [2(ja|bk) - (jb|ak)]
       + 2 Σ_jbc t_ij^bc [2(ab|jc) - (ac|jb)]
       + ... (occ-vir mixing)
```

Plus exactement (Helgaker §13.7.3) :

```
L_ai = X_ai^o - X_ai^v
```

où X^o et X^v sont les contributions "orbital relaxation" sur les
blocs occupé et virtuel.

Implémentation dans `mp2.py` :

```python
def mp2_lagrangian(basis, c, n_occ, mp2_result, eri_full=None):
    """MP2 Lagrangian L_ai (n_virt, n_occ) for the Z-vector RHS."""
```

Tests : ≥ 4 tests sur H₂, LiH, BH (formes, signes, magnitude).

### Étape 2 — Solveur Z-vector

L'équation `A·Z = -L` se résout itérativement avec exactement la même
infrastructure que CPHF (orbital Hessian construit via le K^(1)) :

```
Z^(k+1) = Z^(k) - α · [A·Z^(k) + L]
```

avec damping et tolérance comme CPHF.

Réutiliser `coupled_cphf_response` en factorisant la boucle interne :

```python
def solve_z_vector(basis, mo_eigvals, mo_coeffs, n_occ, lagrangian,
                    *, max_iter=30, tol=1e-5, damping=0.5,
                    level_shift=0.0, ...):
    """Solve A·Z = -L via the same iterative scheme as CPHF.

    Returns Z (n_virt, n_occ).
    """
```

Tests : convergence sur H₂ → Z ≈ 0 (no virtual mixing for s-only),
LiH → Z fini, BH → Z fini avec norme < |t-amplitude|.

### Étape 3 — 1-RDM MP2 relaxée complète

La 1-RDM MP2 *relaxée* combine d_occ, d_vir, Z dans une seule
matrice MO :

```
D_MP2_MO[i, j]   = δ_ij ·2 + d_occ[i, j]      (occ-occ block)
D_MP2_MO[a, b]   = d_vir[a, b]                (vir-vir block)
D_MP2_MO[i, a]   = Z[a, i]                    (occ-vir block)
D_MP2_MO[a, i]   = Z[a, i]                    (vir-occ block, sym)
```

AO transform :

```python
def mp2_density_relaxed_AO(c, n_occ, mp2_result, z_vector):
    """Full MP2-relaxed 1-RDM in AO basis (with Z-vector contribution)."""
```

Tests : symétrie, trace = nombre d'électrons, comparaison à
`mp2_density_correction_AO` (même résultat moins le bloc Z).

### Étape 4 — σ_p MP2-GIAO complet

Wrapper haut-niveau `mp2_paramagnetic_shielding(...)` qui :

1. HF SCF (ou Hueckel pour SZ)
2. MP2 amplitudes
3. Lagrangien
4. Z-vector
5. 1-RDM MP2 relaxée
6. σ_p via la formule standard avec D_MP2_relaxed

Validation :
- **Benzène SZ** : NICS_zz HF vs NICS_zz MP2-GIAO. La littérature donne
  ~5-10 % de correction MP2 sur NICS de benzène.
- **N₂ SZ** : σ_p à mid-bond. Stanton & Gauss 1996 donnent une
  correction MP2 spécifique reproductible.
- **Bi₃@U₂(Cp*)₄** : si tractable, comparer NICS_zz HF=+0.474 ppm vs
  MP2-GIAO. Cible : rester dans la fenêtre exp ±0.5 ppm.

### Étape 5 — Intégration current.py

`current_density_at_points` n'a pas besoin de modification : si
`_ResponseData.U_imag` est construit à partir d'une CPHF avec le
Z-vector intégré, `j_para` capte automatiquement la contribution.

Vérifier que `precompute_response_mp2_explicit` accepte un mode
`use_z_vector=True` qui inclut l'étape Z-vector dans la pipeline.

### Étape 6 — Tests + benchmark + UI

- ≥ 25 nouveaux tests (Lagrangien, Z-vector, 1-RDM relaxée,
  σ_p MP2-GIAO end-to-end)
- 0 régression sur les 284 existants
- UI : option "MP2-CPHF (Z-vector)" dans le panel cluster
- Mise à jour du tableau de comparaison NICS_zz pour Bi₃@U₂

---

## Décisions à prendre en début de session

1. **Convention ERI** : continuer avec (ia|jb) chemist, ou basculer
   vers (ij||ab) physicist pour la formulation Helgaker plus directe ?
2. **Solveur Z-vector** : factoriser depuis CPHF (réutilisation max) ou
   solveur séparé (clarté max) ?
3. **Validation expérimentale** : reproduire un cas littérature
   (Stanton-Gauss 1996 N₂, ou Wilson 1996 benzène) ou se contenter
   de signe + ordre de grandeur ?
4. **Performance** : MP2 + Z-vector + CPHF sur Bi₃@U₂(Cp*)₄ va prendre
   plusieurs minutes. Viser une démonstration sur benzène SZ d'abord
   (1-2 min), puis un test ultime Bi₃@U₂ stripped (éventuellement) ?

Recommandation par défaut :
- (1) ERI chemist (compat. existant)
- (2) Factorisation : extraire `_iterative_orbital_response_solver`
  depuis CPHF, le réutiliser pour Z-vector
- (3) Reproduire N₂ SZ comme cas-test propre
- (4) Benzène SZ d'abord, Bi₃@U₂ stripped en stretch

---

## Critères de succès

1. **Lagrangien implémenté** : `mp2_lagrangian` retourne (n_virt, n_occ),
   passe ≥ 4 tests de forme/signe.
2. **Z-vector convergent** : `solve_z_vector` converge < tol=1e-4 en
   < 30 itérations sur H₂, LiH, BH, N₂. Pour H₂ s-only → Z = 0.
3. **1-RDM MP2 relaxée** : trace = N_électrons à 1e-4 près, symétrique,
   diffère de la 1-RDM "MP2 sans relaxation" par le bloc Z.
4. **σ_p MP2-GIAO** : pipeline end-to-end produit un nombre fini, signe
   et ordre de grandeur correct vs HF.
5. **Validation littérature** : N₂ ou benzène reproduit la correction
   MP2-NICS dans une marge documentée.
6. **Tests** : ≥ 25 nouveaux tests, 0 régression sur 284.
7. **UI** : option MP2-CPHF Z-vector dans le panel cluster, comparaison
   HF / MP2 / MP2-Z affichée.

## Limites résiduelles à NE PAS attaquer en cette session

- **Gradient g-orbital analytique** (l=4) : FD reste suffisant pour
  DZP+ ; ce n'est pas le bottleneck NICS. Continuation séparée.
- **f-block Hueckel + d-shell** : conflit irréductible. Utiliser HF
  pour ces cas, ce qu'on fait déjà via `precompute_response_mp2_explicit`.
- **HF SCF parallèle sur cluster 105+ atomes** : Bi₃@U₂(Cp*)₄ HF prend
  >10 min. Parallélisation = session dédiée infrastructure.
- **Coupled-cluster** (CCSD, CCSD(T)) : très loin du scope. C'est la
  *suivante* limite après Z-vector si on veut pousser plus loin.

## Mémoires à mettre à jour en fin de session

- `project_pt_lcao_quant.md` : ajouter section Z-vector / MP2-GIAO
- `MEMORY.md` : entry one-line
- Si N₂ ou benzène reproduit la correction MP2 littérature : reformuler
  conclusion PT_AROMATICITY pour annoncer "MP2-GIAO complet implémenté".
