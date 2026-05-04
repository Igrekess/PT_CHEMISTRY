# Brief — Page web "Benchmark PTC"

> Document de handover pour une session future. Tout ce qu'il faut pour
> construire une page web propre qui affiche les benchmarks PTC vs DFT.
> Date du benchmark : 2026-05-03.

---

## 1. Objectif de la page

Présenter de manière **honnête et publiable** que la Théorie de la Persistance
(PTC) bat ou égale B3LYP/6-31G\* et B3LYP/def2-TZVP sur 860 molécules ATcT,
**sans aucun paramètre ajusté**.

Audience : public technique (chimistes, physiciens, reviewers). Ton : factuel,
sans esbroufe, avec les caveats explicites.

---

## 2. Headline numbers à afficher gros en haut de page

```
PTC vs B3LYP/6-31G*    →   MAE 0.60 eV vs 0.93 eV    (gain −35 %)
PTC vs B3LYP/def2-TZVP →   MAE 0.49 eV vs 0.67 eV    (sur 536 mol n≤12)
PTC win rate           →   70.9 % (vs 6-31G*) / 65.9 % (vs def2-TZVP)
Paramètres ajustés     →   0  vs  3 (B3LYP)
```

Encadré "PTC en chiffres" :
- **0** paramètre ajusté
- **1** input fondamental (s = 1/2)
- **860** molécules ATcT testées
- **0.60 eV** MAE global
- **70.9 %** de victoires en H2H vs B3LYP/6-31G\*
- **65.9 %** de victoires en H2H vs B3LYP/def2-TZVP (n≤12, 536 mol)
- **× 3.1** précision chimique vs B3LYP/6-31G\*

---

## 3. Sources de données (chemins relatifs depuis PUBLIC/PT_CHEMISTRY/)

### Comparaison principale 2-voies (860 mol)
- `benchmarkb3lyp/260503_PTC_vs_B3LYP_dossier.md` — **synthèse complète à
  reprendre comme contenu source** (texte + tableaux + pitch)
- `benchmarkb3lyp/260503_benchmark_B3LYP_860.json` — données B3LYP per-mol
- `benchmarkb3lyp/260503_comparison_PTC_vs_B3LYP_860.json` — paires appariées
- `benchmarkb3lyp/260503_comparison_metrics.json` — toutes les métriques
  agrégées (global + per-cat + per-size + histogramme + top-20)
- `benchmarkb3lyp/260503_comparison_per_molecule.csv` — **tableau plat 860
  lignes** (idéal pour charts JS via PapaParse / d3-fetch)

### Comparaison 3-voies (536 mol n≤12, 3 bins)
- `benchmark_b3lyp_def2tzvp/comparison_3way_n12.json` — master 3-voies
- `benchmark_b3lyp_def2tzvp/comparison_3way_n12_summary.csv` — **12 lignes**
  (4 bins × 3 méthodes) idéal pour bar charts
- `benchmark_b3lyp_def2tzvp/comparison_3way_n12_per_molecule.csv` — 536 lignes
- `benchmark_b3lyp_def2tzvp/comparison_3way_n12_winrates.csv` — win rates
  par bin pour bar chart empilé

### Source unique consommée par le panel Streamlit
- `ptc_app/dft_comparison_data.json` — agrège tout (1000 mol + bloc def2-TZVP)
  ; format prêt à servir si la page web tourne sur le même dataset

---

## 4. Structure de page recommandée

```
┌────────────────────────────────────────────────────────────┐
│  Hero : titre + 4 metric cards en gras                      │
│  "PTC bat B3LYP à zéro paramètre, sur 860 molécules ATcT."  │
├────────────────────────────────────────────────────────────┤
│  Pitch (100 mots)                                           │
├────────────────────────────────────────────────────────────┤
│  Section 1 — Comparaison principale (PTC vs B3LYP/6-31G*)   │
│    • Tableau métriques (10 lignes)                          │
│    • Graph 1 : Histogramme erreurs signées superposées      │
│    • Graph 2 : Scatter D_calc vs D_exp (2 séries)           │
│    • Graph 3 : MAE par bin de taille (bar chart groupé)     │
│    • Graph 4 : Win rate PTC par bin (bar chart simple)      │
├────────────────────────────────────────────────────────────┤
│  Section 2 — Test 3-voies contre def2-TZVP                  │
│    • Caption + warning "L'argument 6-31G* low-cost réfuté"  │
│    • Tableau 3-voies × 3-bins (10 lignes)                   │
│    • Graph 5 : Bar chart MAE par bin × 3 méthodes           │
│    • Graph 6 : Bar chart 3-way win rate empilé              │
│    • Encart surprise : "def2-TZVP régresse sur Bin C"       │
├────────────────────────────────────────────────────────────┤
│  Section 3 — Distribution des erreurs                        │
│    • Tableau seuils <1/<2/<5/<10 kcal/mol                   │
│    • CDF cumulative des |err|                               │
├────────────────────────────────────────────────────────────┤
│  Section 4 — Top-20 molécules                                │
│    • Top 20 où B3LYP rate (CO₂, N₂O, thiophènes...)         │
│    • Top 20 où PTC rate (N-hétérocycles, siloxanes)         │
├────────────────────────────────────────────────────────────┤
│  Section 5 — Caveats et méthodologie                         │
│    • Screening rules (9 fails + 8 outliers exclus)          │
│    • Géométries identiques (RDKit MMFF + tables diatomiques)│
│    • ZPE Scott-Radom 0.9806                                 │
│    • def2-TZVP : single-point sur géom 6-31G* + ZPE recyclée│
├────────────────────────────────────────────────────────────┤
│  Section 6 — Reproductibilité                                │
│    • Liens GitHub vers code MIT                             │
│    • Commandes pour régénérer le benchmark                  │
│    • Téléchargement direct des CSV                          │
└────────────────────────────────────────────────────────────┘
```

---

## 5. Détail des 6 graphiques essentiels

| # | Type | Données | Insight |
|---|---|---|---|
| 1 | Histogramme superposé erreurs signées | `260503_comparison_per_molecule.csv` colonnes `err_b3lyp_eV` & `err_ptc_eV` | Biais B3LYP centre vers −0.9 eV, PTC neutre |
| 2 | Scatter D_calc vs D_exp | idem, axes `D_exp_eV` × {`D_b3lyp_eV`, `D_ptc_eV`} + ligne y=x | Outliers visibles, dispersion |
| 3 | Bar chart groupé MAE par bin n_atoms | `260503_comparison_metrics.json` → `per_size_bin` | Dégradation B3LYP avec taille |
| 4 | Bar chart MAE par bin × 3 méthodes | `comparison_3way_n12_summary.csv` | **def2-TZVP régresse sur Bin C** |
| 5 | Stacked bar 3-way win rate par bin | `comparison_3way_n12_winrates.csv` | Composition des wins |
| 6 | CDF des \|err\| | dérivé du CSV per-mol | % mol sous chaque seuil |

---

## 6. Texte source réutilisable

### Pitch (100 mots) — directement copiable

> Sur un benchmark de 860 molécules issues des Active Thermochemical Tables
> (ATcT 0K), la Théorie de la Persistance (PTC) atteint une erreur absolue
> moyenne de **0.60 eV** sur l'énergie d'atomisation, contre **0.93 eV** pour
> B3LYP/6-31G\* — un gain de 35 % sans **un seul paramètre ajusté**, là où
> B3LYP en utilise trois calibrés sur G2. PTC bat aussi B3LYP/def2-TZVP
> (la base "moderne", n=536 mol) avec 66 % de victoires en tête-à-tête.
> **Surprise** : sur les molécules de 9-12 atomes, def2-TZVP est *pire* que
> 6-31G\* — défaut intrinsèque de la fonctionnelle B3LYP que PTC ne partage
> pas. Tous les calculs PTC sont dérivés analytiquement depuis l'unique
> input s = 1/2.

### Corps complet (~400 mots)
Voir section "Texte web" (8) du dossier `260503_PTC_vs_B3LYP_dossier.md`.

### Encart surprise Bin C (à mettre en warning visuel)

> **def2-TZVP régresse derrière 6-31G\* sur les molécules moyennes**.
> Sur les 167 molécules de 9-12 atomes, B3LYP/def2-TZVP atteint MAE = 0.92 eV
> contre 0.82 eV pour 6-31G\*. Phénomène DFT connu : annulation d'erreur
> entre BSIE (basis set incompleteness) et erreur intrinsèque de la
> fonctionnelle dans 6-31G\*, qui disparaît avec def2-TZVP et expose le
> défaut de B3LYP. PTC reste meilleure (0.78 eV) et **non biaisée** (biais
> signé −0.06 eV vs −0.92 eV pour def2-TZVP) — démontrant que le défaut
> B3LYP est **structurel à la fonctionnelle**, pas guérissable par
> meilleure base.

---

## 7. Stack technique suggérée (à confirmer en session future)

Si le site est statique :
- **Astro / 11ty / Next.js** pour le contenu MD
- **Plotly.js** ou **Chart.js** pour les graphiques (lit le CSV directement)
- Charger les CSV via `fetch` + PapaParse côté client
- Téléchargement direct des CSV depuis la page

Si Streamlit (déjà existant dans `ptc_app/`) :
- Le panel `dft_panel.py` est déjà fonctionnel et synchronisé
- Pour une version web "pure" séparée, repartir des CSV ci-dessus

Si Observable / D3 :
- CSV → DataFrame en JS via Arquero
- Graphiques composables avec Observable Plot

---

## 8. Caveats à inscrire honnêtement dans la page

1. **6-31G\* est la base low-cost** de B3LYP. Comparaison contre def2-TZVP
   incluse pour répondre à cette objection.
2. **123 molécules à éléments lourds** (Ag, Au, transition metals, I) ne
   peuvent pas être calculées en B3LYP/6-31G\* — exclues du benchmark côté
   B3LYP. PTC les calcule sans problème (présent dans les 1000 mol totales).
3. **9 échecs SCF/Hessian B3LYP** (NO₂, ClF₃, B₂H₆, fullerène...) exclus.
4. **8 outliers pathologiques B3LYP** (\|err\| > 5 eV : S₂F₁₀ −114 eV,
   PCl₅, SF₆×2, SF₄, HBO₂, phenanthrene, SO₃) exclus.
5. **Géométries** : RDKit MMFF (3+ atomes) ou table diatomique fixe.
   Identiques entre 6-31G\* et def2-TZVP, donc comparaison de base
   "apples-to-apples".
6. **ZPE** : harmoniques B3LYP/6-31G\* × 0.9806 (Scott & Radom 1996).
   Pour def2-TZVP, ZPE recyclée du 6-31G\* (différence < 2 % entre bases,
   négligeable devant l'erreur D₀).
7. **Aucune comparaison contre G4 / W1 / CCSD(T)** — méthodes composites
   à ~0.04 eV de précision mais avec ~10 paramètres ajustés. À mentionner
   pour situer PTC dans le paysage.
8. **Régressions PTC connues** : N-hétérocycles aromatiques (pyrrole,
   indazole, pyridazine, triazine), siloxanes. Chantiers identifiés.
   À mentionner ouvertement.

---

## 9. Régénération des données

Si nouvelles données B3LYP ou PTC arrivent :
```bash
cd PUBLIC/PT_CHEMISTRY/  # ou PT_PROJECTS/PTC/
python scripts/regenerate_dft_comparison_data.py
```

Si nouveau run def2-TZVP :
```bash
cd benchmark_b3lyp_def2tzvp/
python build_input.py --max-atoms 16    # par exemple
python run_def2tzvp.py --input input_n16.json --out results_def2tzvp_n16.json
# puis adapter scripts/regenerate_dft_comparison_data.py pour pointer vers
# le nouveau comparison_3way_n16.json
```

---

## 10. Liens externes à inclure

- **GitHub MIT** : https://github.com/Igrekess/PT_CHEMISTRY (vrai username : Igrekess)
- **ATcT (référence expérimentale)** : https://atct.anl.gov/
- **PySCF (générateur B3LYP)** : https://pyscf.org/
- **Scott & Radom 1996 (ZPE 0.9806)** : J. Phys. Chem. 100, 16502

---

## 11. Checklist de validation avant publication

- [ ] Toutes les figures sont générées à partir des CSV/JSON canoniques
- [ ] Les chiffres affichés correspondent à `260503_PTC_vs_B3LYP_dossier.md`
- [ ] Caveats explicites (basis set, ZPE, screening, régressions PTC)
- [ ] Lien direct pour télécharger les CSV depuis la page
- [ ] Mention claire "0 paramètre ajusté côté PTC" — c'est l'argument central
- [ ] Reproductibilité : lien GitHub + commande de régénération
- [ ] Pas de cherry-picking : les régressions PTC sont mentionnées
- [ ] Comparaison contre def2-TZVP incluse (sinon la page n'est pas crédible
      face à un reviewer)

---

## 12. Pour démarrer la session future

Prompt de départ :

> Construis une page web statique pour afficher les benchmarks PTC.
> Spec dans `benchmarkb3lyp/BRIEF_PAGE_WEB_BENCHMARK.md`.
> Synthèse longue dans `benchmarkb3lyp/260503_PTC_vs_B3LYP_dossier.md`.
> Données canoniques dans :
>   - `benchmarkb3lyp/260503_comparison_per_molecule.csv` (860 lignes)
>   - `benchmark_b3lyp_def2tzvp/comparison_3way_n12_summary.csv` (12 lignes)
>   - `benchmark_b3lyp_def2tzvp/comparison_3way_n12_per_molecule.csv` (536 lignes)
>
> Stack : [à choisir — Astro / Next / pur HTML+Plotly]
> Tone : factuel, scientifique, mention explicite "0 paramètre ajusté".
