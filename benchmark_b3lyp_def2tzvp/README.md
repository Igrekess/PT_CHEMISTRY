# B3LYP/def2-TZVP single-point benchmark — small ATcT subsets

> **Stratégie 3 bins par taille** pour comparaison équitable PTC vs B3LYP/6-31G\* vs B3LYP/def2-TZVP :
>
> | Bin | n_atoms | n_mol | Statut |
> |---|:---:|---:|:---:|
> | **A — petites** | 2-4 | 180 | ✓ run terminé (4.5 min) |
> | **B — moyennes** | 5-8 | 189 | input prêt, run en cours |
> | **C — grandes** | 9-12 | 167 | input prêt |
>
> **Total def2-TZVP : 536 mol**. Couverture exhaustive dans chaque bin (pas
> de sampling). Au-delà de n=12, le coût explose (n=16 ≈ 3.5 min/mol) →
> hors scope, on garde la 2-way `PTC vs 6-31G*` sur les 860 séparément.

Calcule l'énergie d'atomisation D₀ au niveau **B3LYP/def2-TZVP** pour
les 180 molécules les plus petites du benchmark ATcT (≤ 4 atomes lourds + H),
afin de comparer équitablement avec :

- B3LYP/6-31G* (référence basique, déjà calculée)
- PTC (Théorie de la Persistance, 0 paramètre ajusté)

## Pourquoi ce sous-ensemble

Les 180 mol n_atoms ≤ 4 sont **précisément le régime où la qualité de la
base atomique compte le plus** pour la DFT. Le 6-31G\* y a sa pire MAE
(0.79 eV) tandis que sur l'ensemble des 860, le 6-31G* cumule 0.93 eV.
Le but du test : **vérifier si def2-TZVP comble le gap avec PTC** (qui a
0.23 eV de MAE sur ce même sous-ensemble).

## Arborescence

```
benchmark_b3lyp_def2tzvp/
├── README.md                       # ce fichier
├── requirements.txt                # pyscf>=2.3, numpy>=1.21
├── build_input.py               # builder paramétrable (--max-atoms N)
│
├── input_n4.json                   # 180 mol n≤4   (couvre Bin A)
├── input_n8.json                   # 369 mol n≤8   (couvre Bins A+B)
├── input_n12.json                  # 536 mol n≤12  (couvre Bins A+B+C)
│
├── run_def2tzvp.py                 # SCRIPT PRINCIPAL — accepte n'importe quel input
│
├── results_def2tzvp_n4.json        # ✓ déjà calculé (4.5 min)
├── results_def2tzvp_n8.json        # à calculer (~25-40 min)
└── results_def2tzvp_n12.json       # à calculer (~1.5-2 h)
```

## Méthodologie

| Aspect | Choix |
|---|---|
| Fonctionnel | B3LYP (Becke 1993, LYP) |
| Base | def2-TZVP (Weigend & Ahlrichs 2005) |
| Géométrie | **Frozen** depuis le run B3LYP/6-31G* (table diatomique + RDKit MMFF) |
| ZPE | **Recyclée** depuis B3LYP/6-31G* (Scott-Radom 0.9806 déjà appliqué) |
| SCF | conv_tol=1e-9, max_cycle=200, level_shift=0.1 (mol) / 0.2 (atomes), Newton fallback |
| Référence atomique | atomes isolés UKS au même niveau B3LYP/def2-TZVP |

**Énergie d'atomisation** :
```
D_at = Σ E_atome - E_molécule - ZPE
```
avec ZPE déjà scalée et recyclée (suffisant car les ZPE varient
< 2 % entre 6-31G* et def2-TZVP, négligeable devant l'erreur D₀).

## Lancer le benchmark

### Sur la machine cible
```bash
cd benchmark_b3lyp_def2tzvp
pip install -r requirements.txt

# Bin A — n≤4 (180 mol, ~5 min) — DÉJÀ FAIT
python run_def2tzvp.py --input input_n4.json --out results_def2tzvp_n4.json

# Bins A+B — n≤8 (369 mol, ~25-40 min) — EN COURS
python run_def2tzvp.py --input input_n8.json --out results_def2tzvp_n8.json

# Bins A+B+C — n≤12 (536 mol, ~1.5-2 h) — APRÈS LE n8
python run_def2tzvp.py --input input_n12.json --out results_def2tzvp_n12.json

# Reprise après interruption
python run_def2tzvp.py --input input_n12.json --out results_def2tzvp_n12.json --resume

# Smoke test rapide
python run_def2tzvp.py --input input_n12.json --out /tmp/smoke.json --limit 5
```

Le script **checkpoint après chaque molécule**. Ctrl-C à tout moment, reprise
par `--resume`.

> **Note** : `input_n12.json` contient les 369 mol de `input_n8.json` plus
> 167 mol additionnelles n=9-12. Si tu lances `input_n12.json` après avoir
> déjà fait `input_n8.json`, tu peux gagner du temps en pré-remplissant le
> résultat avec `--resume` pointant sur le n8 (à condition que les noms
> matchent — les molécules n≤8 sont strictement les mêmes).

### Coûts mesurés / estimés

Mesuré sur les 3 premières mol n≥6 lors du smoke test :
- oxazole (n=8) : 13.0 s
- Acetyl_chloride (n=7) : 7.1 s
- Methanediol (n=6) : 2.8 s

| Bin | n_atoms | n_mol | s/mol moyen | Coût total |
|---|:---:|---:|---:|---:|
| A | 2-4 | 180 | 1.5 s | **4.5 min ✓ mesuré** |
| B | 5-8 | 189 | ~8 s | ~25 min |
| C | 9-12 | 167 | ~40 s | ~1.5-2 h |

**Total Bin A+B+C (input_n12.json) : ~2-2.5 h** sur un CPU moderne avec
BLAS multi-threadé.

### Étendre au-delà de n≤12 (déconseillé)
```bash
python build_input.py --max-atoms 16
python run_def2tzvp.py --input input_n16.json --out results_def2tzvp_n16.json
```
Coût qui croît rapidement (SCF def2-TZVP en N⁴) — prévoir plusieurs heures
voire une nuit pour n=12-20.

## Format de sortie

`results_def2tzvp_n4.json` contient :

```json
{
  "method": "B3LYP/def2-tzvp",
  "run_config_version": 1,
  "zpe_source": "recycled_from_b3lyp_631gs (Scott-Radom 0.9806 applied)",
  "geometry_source": "frozen from B3LYP/6-31G* run (input_n4.json)",
  "elapsed_s": 12345.6,
  "n_done": 180,
  "results": [
    {
      "name": "N2",
      "smiles": "N#N",
      "category": "Organique",
      "n_atoms": 2,
      "atoms": ["N", "N"],
      "charge": 0,
      "spin": 0,
      "geometry_source": "json_diatomic_re",
      "D_exp": 9.759,
      "E_mol_eV": -2967.78,
      "E_atoms_eV": -2956.45,
      "ZPE_eV": 0.1533,
      "ZPE_source": "recycled_from_b3lyp_631gs",
      "D_elec_eV": 11.33,
      "D_def2tzvp_eV": 9.81,
      "err_eV": 0.05,
      "rel_err_pct": 0.51,
      "status": "ok",
      "time_s": 18.3
    },
    ...
  ]
}
```

## Comparaison avec le benchmark 6-31G*

Une fois `results_def2tzvp_n4.json` produit, on peut comparer
directement avec :

- `../benchmarkb3lyp/260503_benchmark_B3LYP_860.json` (run 6-31G* screened)
- `../benchmarkb3lyp/ptc_fresh_2026-05-01.json` (PTC fresh)

Le sous-ensemble n≤4 contient 180 molécules. MAE attendues :
- B3LYP/6-31G* : 0.79 eV (mesurée)
- PTC : 0.23 eV (mesurée)
- B3LYP/def2-TZVP : **inconnue, à mesurer**

Hypothèse littérature : def2-TZVP devrait améliorer 6-31G* d'un
facteur 1.5-2× sur petites molécules → MAE attendue ~0.3-0.5 eV.

## Reproduire la pré-extraction

Si tu modifies les listes d'exclusion ou le seuil n_atoms :

```bash
cd benchmark_b3lyp_def2tzvp
python build_input.py                  # n≤4
python build_input.py --max-atoms 8    # n≤8
python build_input.py --max-atoms 12   # n≤12
```
(nécessite RDKit + accès à `../benchmarkb3lyp/`)

## Provenance des géométries

Toutes les coordonnées sont **identiques** à celles utilisées pour le run
6-31G\*, ce qui garantit que la différence observée vient uniquement de la
base. Sources :
- `geometry_overrides.json` : 12 mol diatomiques avec spin override manuel
- Table diatomique + rayons covalents RDKit : la majorité des diatomiques
- RDKit MMFF embed (randomSeed=42) : pour 3+ atomes
