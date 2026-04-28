# PROMPT_NEXT_SESSION_V3 — NMR benchmark V3 + Becke probe-on-nucleus fix

**État au démarrage** : post Phase 6.B.11f (cascade complète HF → MP2 →
MP3 → LCCD → CCD → CCSD → Λ → σ_p^CCSD-Λ-GIAO) + benchmark V1 + V2.

## Contexte hérité

* **57 tests CC PASS** (mp3.py 12 + ccd.py 19 + ccsd.py 10 + ccsd_lambda
  9 + ccsd_property 7) — pipeline post-HF structurellement vérifié.
* **Bench V1+V2 livré** : 15 entrées (H₂, HF, N₂, CO, F₂, H₂O, NH₃, CH₄,
  HCN), σ_iso ref CCSD(T)/qz2p, MAE 177 ppm avec pVDZ-PT.
* **Best individual** : H₂ ¹H +1.4 ppm, CH₄ ¹³C +11 ppm — Lamb-grade.

## Objectif V3

**MAE σ_iso < 30 ppm** — publishable comme premier framework PT-pure
post-HF NMR ab initio. Trois leviers identifiés ; fix de l'un suffit
pour passer la barre.

## Bug critique #1 — Becke probe-on-nucleus (priorité MAX)

### Symptôme
Avec `use_becke=True` + probe = position d'un atome (offset 0 ou 0.01 Å)
sur HF/¹⁹F : σ_iso = 5.4×10⁶ ppm au lieu de +418 ppm (réf). Sur H₂
isolé ça marche.

### Diagnostic
Dans `ptc/lcao/giao.py::nuclear_attraction_matrix(use_becke=True)` :
* La condition `probe_on_atom = atom_dists.min() < 1e-8` ne s'active
  qu'au tout-premier ordre. Avec probe = nucleus exactement, oui — mais
  alors la grille atomique au noyau probé contient des points avec r→0,
  où 1/|r-K| × ψ²(r) × weight peut exploser.
* Avec probe = nucleus + 0.01 Å, le probe sub-grid s'ajoute → weight
  double-counting avec le sub-grid atomique adjacent.

### Fix proposé
Deux options (à implémenter ensemble pour robustesse) :

**Option A — Treatment analytique pour probe sur atome** :
Quand probe = position d'un atome A :
1. Pour les paires d'orbitales (i, j) toutes deux centrées sur A :
   utiliser la valeur ANALYTIQUE de ⟨φ_i^A | 1/|r-A| | φ_j^A⟩ — dispo
   en closed-form pour STO via formules de Roothaan-Bagus.
2. Pour les paires hybrides (centrées sur A et B≠A) ou (B,C avec B,C≠A)
   utiliser la quadrature Becke standard (qui est OK loin de probe).

**Option B — Corrected probe-sub-grid quand probe ≈ atom** :
Si probe est à < 0.5 Å d'un atome, désactiver le probe sub-grid et
augmenter la résolution du sub-grid atomique près du noyau probé
(n_radial localement plus grand, log scale).

### Implémentation suggérée

```python
# In nuclear_attraction_matrix(use_becke=True):
atom_dists = np.linalg.norm(basis.coords - P[None, :], axis=-1)
on_atom_idx = np.argmin(atom_dists)
probe_on_atom = atom_dists[on_atom_idx] < 1e-8

if probe_on_atom:
    # Option A: split into "same-center" + "other-center" parts
    M = np.zeros((n, n))
    for i, orb_i in enumerate(basis.orbitals):
        for j, orb_j in enumerate(basis.orbitals):
            if (basis.atom_index[i] == on_atom_idx and
                basis.atom_index[j] == on_atom_idx):
                # Both orbitals centred on the probe nucleus —
                # use analytic same-centre 1/r matrix element
                M[i, j] = analytic_one_over_r_same_centre(orb_i, orb_j)
            else:
                # Use regular Becke quadrature with probe-sub-grid
                M[i, j] = becke_off_probe_partial(orb_i, orb_j, P)
    return M
```

L'analytique same-centre `analytic_one_over_r_same_centre(orb_i, orb_j)`
pour deux STO ψ_i = N_i r^{n_i-1} exp(-ζ_i r) Y_{l_i m_i} et same pour j
au même centre :
* Si l_i ≠ l_j ou m_i ≠ m_j : intégrale = 0 (orthogonalité angulaire)
* Si l_i = l_j et m_i = m_j : intégrale radiale
  ∫_0^∞ r^{n_i+n_j-2} exp(-(ζ_i+ζ_j)r) × 1/r × r² dr × N_i N_j
  = N_i N_j (n_i+n_j-1)! / (ζ_i+ζ_j)^{n_i+n_j}

C'est trivial. À implémenter dans `_analytic_one_over_r_same_centre()`.

### Impact attendu
* Enlève le 100-200 ppm de biais probe-offset sur les hétéroatomes
  lourds → MAE 177 → 30-50 ppm.

### Effort estimé
* 2-3 heures pour l'analytique same-centre + dispatch
* 1 heure pour les tests (H₂, HF, N₂, F₂)
* 30 min pour relancer V2 bench → V3 numbers

## Bug #2 — Uncoupled CPHF singularité (priorité moyenne)

### Symptôme
Sur **N₂/pVDZ-PT level HF** dans le bench V2 : σ_iso^HF = 4.8×10¹³ ppm.
Tous les autres niveaux (MP2, CCSD-Λ) marchent (~+353 ppm).

### Diagnostic
Le pipeline HF utilise `paramagnetic_shielding_iso_coupled` (uncoupled
CPHF) qui divise par (ε_a − ε_i). Sur pVDZ-PT/N₂ il y a une paire
quasi-dégénérée occ-virt → division near-singular → blow-up.

MP2 et CCSD-Λ relaxent les orbitales via `mp2_relax_orbitals` →
re-diagonalisation lifte la dégénérescence → marche.

### Fix proposé
Dans `paramagnetic_shielding_iso_coupled` ou la version coupled :
ajouter un epsilon-shift ou un cutoff sur `(ε_a − ε_i) > tol_gap`
(typiquement 1e-3 Hartree). Skip ou regularize les paires presque-
dégénérées.

### Effort estimé
* 1 heure pour le fix + test ciblé sur N₂/pVDZ-PT.

## Levier #3 — NSD-100 reference set (priorité optionnelle)

### Objectif
Étendre le set de 15 entrées à ~80-100 (Standard NMR Database 100,
Auer-Gauss 2003 21 mol full set, ou Flaig-Bremm-Ochsenfeld 2014).

### Effort
* ~1 jour pour la table de références
* ~3-4 jours de compute (sur DZP/pVDZ-PT, cascade complète)

## Plan de session V3

**Phase 1 (1-2h)** : implémenter le fix Becke probe-on-nucleus (Option
A : analytic same-centre 1/r) dans `ptc/lcao/giao.py`. Tester sur H₂
et HF avec `--use-becke true --probe-offset 0.0`.

**Phase 2 (30 min)** : fixer la singularité CPHF (epsilon-shift dans
paramagnetic_shielding_iso_coupled).

**Phase 3 (1h)** : relancer le bench V2 avec les nouveaux fixes
(`--basis pVDZ-PT --use-becke true --probe-offset 0.0`). Comparer
MAE V2 (177 ppm) → V3 (cible ~30-50 ppm).

**Phase 4 (1-2h)** : si Phase 3 réussit, étendre le set à 21+ molécules
(Auer-Gauss complet) et relancer pour stats robustes.

**Phase 5 (30 min)** : update report Markdown + commit privé + sync
public + commit public.

**Total estimé : ~5-7 heures session focalisée.**

## Critères de succès V3

| Critère | V2 actuel | V3 cible |
|---|---|---|
| MAE σ_iso (15 mol) | 177 ppm | < 30 ppm |
| MAE ¹H | 50 ppm | < 5 ppm |
| MAE ¹³C | 148 ppm | < 30 ppm |
| MAE ¹⁵N/¹⁷O/¹⁹F | 250-330 ppm | < 50 ppm |
| Becke probe-on-nucleus | ❌ explose | ✅ stable |
| HF level on pVDZ-PT/N₂ | ❌ singularité | ✅ régularisé |
| H₂ ¹H | +1.4 ppm | < 1 ppm |
| Tests existants | 57 PASS | 57 PASS (no regression) |

## Files de référence

* `ptc/lcao/giao.py::nuclear_attraction_matrix` — à patcher (fix #1)
* `ptc/lcao/giao.py::paramagnetic_shielding_iso_coupled` — à patcher
  (fix #2)
* `scripts/benchmark_nmr_systematic.py` — bench driver (bouger
  `--use-becke true --probe-offset 0.0` quand fix #1 prêt)
* `scripts/output/NMR_BENCHMARK_REPORT.md` — rapport V1/V2 à étendre
  V3.

## Si publication article PT_NMR_VALIDATION

Après V3 (MAE < 30 ppm), le pipeline est prêt pour un article :

* **Titre proposé** : "Post-HF NMR shielding from a single mathematical
  input : the persistence-theory ab initio framework."
* **Cibles** : J. Chem. Phys., PCCP, JCTC.
* **Argument central** : premier framework qui dérive **toute la cascade
  post-HF** (HF, MP2, MP3, CCSD-Λ, GIAO σ_p) d'un input unique (s = 1/2),
  avec MAE comparable aux Pople-style basis sans aucun paramètre ajusté.
* **Estimation effort article** : ~1 semaine de rédaction post-V3.

---

**Bonne session V3 !** Tout est en place, le diagnostic est précis, les
fixes sont identifiés. La barre des 30 ppm est franchissable en une
session focalisée.
