# PTC — Persistence Theory Chemistry

Calculateur de chimie quantique basé sur la Théorie de la Persistance (PT).
806 molécules, MAE 1.94%, 0 paramètre ajusté. Tout depuis s = 1/2.

PT-LCAO+GIMIC sub-package : tenseur de blindage RMN + densité de courant
induit (f-block γ_la/γ_an, GIMIC PT-pur, l=3/l=4 cubic harmonics, explicit
cluster builder, Bi3@U2(Cp*)4 inverse sandwich) ;
**NICS_zz Bi3@U2(Cp*)4 = +0.47 ppm vs exp +0.08 (Ding 2026, écart 0.39 ppm).

## Mission

PTC est le **calculateur de chimie le plus physiquement fiable et PT-pur possible**.
Il calcule TOUTES les observables moléculaires depuis les premiers principes PT :

- **Atomiques** : IE (énergie d'ionisation), EA (affinité électronique) pour Z=1-118
- **Liaison** : D₀ (énergie de dissociation), r_e (distance d'équilibre), ω_e (fréquence vibrationnelle)
- **Géométrie** : angles de liaison θ_A, θ_B (depuis la fraction de face × angle solide)
- **Moléculaire** : D_at (énergie d'atomisation totale), spectre de screening (eigenvalues T_P1)
- **Décomposition** : contributions σ, π, ionique par liaison ; screening per-face (P₁, P₂, P₃)
- **Aromaticité** : NICS scalaire (Pauling-London PT), classification σ/π
- **Magnétique GIAO** (sous-package `ptc/lcao/`) : tenseur de blindage chimique σ_αβ(P) à n'importe quel point 3D, avec Hartree-Fock SCF (DIIS) et CPHF couplée pour aromatique. **H atome σ_iso = 17.75 ppm** vs Lamb α²/3 exact ; benzène D6h capturé en tenseur axial oblate.

Chaque observable est DÉRIVÉE, pas ajustée. La MAE est un **indicateur** de la qualité de la physique, pas un **objectif** à optimiser. Ne jamais patcher un score.

## Discipline

À chaque correction, dans cet ordre strict :

1. **Comprendre** — identifier le terme PT manquant. Si tu ne peux pas le nommer en termes de polygone, face, ou mode de Fourier, tu n'as pas compris.
2. **Implémenter** — chaque constante vient de s = 1/2. Aucun paramètre ajusté, aucun fit, aucun seuil arbitraire.
3. **Vérifier** — benchmark 700 mol ET QM9 5K. Les deux doivent aller dans le même sens. Si le benchmark s'améliore mais le QM9 régresse, c'est un overfit déguisé.

---

## Fondements PT — Théorèmes et Principes

### La chaîne causale (1 input, 0 paramètre)

```
s=1/2 → Geom(q) → GFT → Crible → Mertens → sin² → mu*=15
                                                         |
                                                    BIFURCATION
                                                   /           \
                                             q_stat            q_therm
                                             alpha_EM          métrique
                                             leptons           quarks, G
```

### L'unique input : s = 1/2

**T1 (Transitions Interdites)** : P(1→1 mod 3) = P(2→2 mod 3) = 0 pour p > 3.
Preuve : 3 premiers consécutifs couvrent {0,1,2} mod 3. L'involution {1↔2} force n₁=n₂, donc **s = 1/2**.
Conséquence : alpha(3) = s² = 1/4. C'est le SEUL input.

### Théorèmes fondateurs

| Code | Nom | Énoncé |
|------|-----|--------|
| T1 | Transitions Interdites | T[1][1] = T[2][2] = 0 mod 3 → s = 1/2 |
| L0 | Max-Entropie | Distribution géométrique unique : q = 1-2/μ |
| T2 | Conservation | Δ₀ = 0 ⟺ α = 1/4 = s² |
| GFT | Gap Fundamental | log₂(m) = D_KL + H (identité algébrique exacte) |
| T3 | GFT = Ruelle | ‖π_stat − MME‖ < 3.5×10⁻⁷ bits |
| T4 | Formule maîtresse | f(p) = [1+α(p-4+2T₀₀)] / [(p-1)α] |
| T5 | Convergence | ε(k) ~ 0.899 × ∏(1-1/p) → 0 via Mertens |
| T6 | Holonomie | sin²(θ_p) = δ_p(2-δ_p), δ_p = (1-q^p)/p |
| T7 | Point fixe | μ* = 3+5+7 = 15, unique auto-cohérent |

### Les deux q (bifurcation)

```
q_stat  = 1 - 2/μ   = 13/15   (couplage, vertex, leptons, α_EM)
q_therm = exp(-1/μ)  = 0.9355  (géométrie, propagateur, quarks, CKM)
```

### Trois quantités DISTINCTES (ne pas confondre !)

| Quantité | Formule | Nature | Usage |
|----------|---------|--------|-------|
| sin²(θ_p, q_stat) | δ_p(2-δ_p), q=13/15 | Vertex (couplage) | α_EM = ∏ sin² |
| sin²(θ_p, q_therm) | δ_p(2-δ_p), q=e^{-1/15} | Propagateur | α_s, CKM |
| γ_p | -d(ln sin²)/d(ln μ) | Dimension anomale | PMNS, sin²θ_W |

Valeurs à μ*=15 :
```
        sin²(q_stat)   sin²(q_therm)   γ_p     Actif
p=3     0.2192          0.1172          0.808   OUI
p=5     0.1940          0.1102          0.696   OUI
p=7     0.1726          0.1037          0.595   OUI
p=11    0.1390          0.0849          0.426   non (ghost)
```

### 7 Principes architecturaux

1. **CASCADE SÉQUENTIELLE** : p=3 → p=5 → p=7. Chaque filtre ne voit que les survivants du précédent.
2. **CAP SHANNON** : D_KL ≤ 1 par canal. sin²_p < 1 pour tout p.
3. **PYTHAGORE = CRT ORTHOGONAL** : canaux sur cercles CRT différents → Pythagore. E² = E_cov² + E_ion².
4. **BIFURCATION = 3 RÉGIMES** : transfert complet (confinement), transmission partielle (QED), fluctuations (VP).
5. **ROTATION HOLONOMIQUE** : IE se projette sur Ry via cos²₃ × (Ry - IE). Berry phase.
6. **RÉSOMMATION GÉOMÉTRIQUE** : G_Fisher/cos²₃ = 4/0.781 = 5.12 (running coupling).
7. **ANTI-DOUBLE-COMPTAGE (GFT)** : D_KL + H = constante. Ne pas compter deux fois la même énergie.

### Application à la chimie moléculaire

**L'atome PT** = polygones concentriques sur le simplexe T₃ :
- Z/(2P₀)Z = s-block (2 positions)
- Z/(2P₁)Z = p-block (6 positions) ← face hexagonale
- Z/(2P₂)Z = d-block (10 positions) ← face pentagonale
- Z/(2P₃)Z = f-block (14 positions) ← face heptagonale

**La molécule PT** = graphe sur T³ = Z/3Z × Z/5Z × Z/7Z :
- Chaque atome Z a coordonnées (Z%3, Z%5, Z%7) sur le tore
- Screening sur face k = sin²(2πr_k/P_k), zéro si r_k = 0 (invisible)
- Énergie per-bond = DFT sur polygone Z/(2P₁)Z + 3 faces CRT
- Cross-vertex = T³ perturbatif (D₃ coupling)
- Cycles = holonomie T³ (frustration de composition)

**Principes clés pour PTC** :
- D₀ = Ry/P₁ × exp(-S) pour chaque liaison (Shannon cap × screening)
- S = S_polygon (DFT) + S_cross-vertex (T³) + S_spectral (eigenvalues)
- Covalent ⊥ ionique : D₀ = √(D_cov² + D_ion²) via Pythagore (Principe 3)
- LP screene sur Z/(2P₁)Z ; π screene sur Z/(2P₂)Z
- Ring holonomy = Σ_edges sin²(2π·Δr_k/P_k) per face (T³ frustration)
- Parseval sur Z/PZ : moyenne sin²(2πr/P) = s = 1/2 (normalisation)

### Constantes PT utilisées dans PTC

Toutes dérivées de s = 1/2 :
```
S_p = sin²(θ_p, q_stat)     — couplage per-face
D_p = S_p × (1 - S_p)       — dispersion per-face (Fisher parabole)
C_p = cos²(θ_p, q_stat)     — complément
P₁ = 3, P₂ = 5, P₃ = 7     — premiers actifs
Ry = 13.606 eV               — Rydberg (Shannon cap atomique)
D_FULL ≈ 1.005               — profondeur totale du crible
S_HALF = 0.5 = s             — le paramètre fondamental
BETA_GHOST                    — contribution des primes fantômes (p≥11)
```

---

## Observables calculées par PTC

### Niveau atomique (atom.py, Z=1-118)

| Observable | Formule PT | Unité | Précision |
|-----------|-----------|-------|-----------|
| IE (ionisation) | Ry × (Z·exp(-S)/n)² × M/(M+mₑ) | eV | 0.057% (Z≤103) |
| EA (affinité) | Même pipeline, orbital vacancy | eV | 1.37% (73 éléments) |

Le screening S = S_core + S_polygon × D + S_rel est entièrement PT :
- S_polygon = dim1 (propagateur) + dim0 (échange) + dim2 (cercle discret)
- Les cercles Z/(2P_l)Z donnent la structure en couches (s, p, d, f)

### Niveau liaison (bond.py, BondResult)

| Observable | Formule PT | Unité |
|-----------|-----------|-------|
| D₀ | Ry/P₁ × exp(-S_bond) × D_FULL | eV |
| v_sigma | Composante σ (face P₁) | eV |
| v_pi | Composante π (face P₂) = σ × s | eV |
| v_ionic | Composante ionique (face P₃) | eV |
| r_e | Distance d'équilibre (métrique Bianchi I) | Å |
| ω_e | Fréquence vibrationnelle harmonique | cm⁻¹ |
| θ_A, θ_B | Angles de liaison aux vertex | degrés |

La distance d'équilibre r_e émerge de la métrique Bianchi I :
- r_e = f(per_A, per_B, bo, coordination, LP)
- Pas de table de rayons covalents — tout depuis les périodes et la coordination

Les angles θ émergent de la fraction de face sur le polygone Z/(2P₁)Z :
- face_fraction = (z + lp_proj) / (2P₁) → θ = f(face_fraction) × solid_angle
- VSEPR n'est PAS utilisé — les angles sont PT-dérivés

### Niveau moléculaire (transfer_matrix.py, TransferResult)

| Observable | Formule PT | Unité |
|-----------|-----------|-------|
| D_at | Σ D₀(bonds) + E_spectral | eV |
| D_at_P1 | Contribution face hexagonale (σ+π) | eV |
| D_at_P2 | Contribution face pentagonale (d-back) | eV |
| D_at_P3 | Contribution face heptagonale (ionique) | eV |
| E_spectral | Correction multi-centre (eigenvalues T_P1) | eV |
| spectrum_P1 | Valeurs propres de la matrice de transfert | array |
| mechanism | Mécanisme poly activé (si applicable) | string |

### Niveau aromaticité (nics.py, NICSResult)

| Observable | Formule PT | Unité |
|-----------|-----------|-------|
| NICS(z) | −α²·a₀·n_eff·f_coh·R²/(R²+z²)^{3/2}/12 ×10⁶ | ppm |
| n_σ, n_π | classification de groupe (G1/11=1σ, G13=1π, G14=2σ, G15=3, G16=4) | int |
| Hückel signe | +1 si 4n+2 (aromatic), −1 si 4n (anti), +0.5 si radical | sign |
| f_coh | T³ Fourier coherence ring composition | [0,1] |
| Cycle classification | "double aromatic σ⊕π" / "π-aromatic" / "antiaromatic" / etc | str |

API publique sur Molecule :
- `mol.aromaticity()` → NICSResult (NICS₀, NICS₁, R, n_e, f_coh, classe)
- `mol.nics(z=0)` → float ppm at probe z
- `mol.nics_profile(zs=...)` → list[(z, NICS)]
- `mol.signature(cap_idx=None)` → FullSignature complet (datasheet)

### Niveau signature expérimentale (signature.py, FullSignature)

Pour chaque candidat (typiquement aromatic ring novel), fiche expérimentale prédictive :
- Géométrie (bond lengths, ring radius, cap height)
- Énergétique (D_at, cap binding, **3+ canaux de fragmentation chiffrés**)
- σ/π split + classe (Hückel rule per channel)
- Profil NICS(z) z = 0..3 Å
- Vibrationnel **Morse-calibré** (ω_PT × exp/PT factor du dimère de référence)
- IE/EA estimés (Koopmans + cycle-Hückel)
- Critères de falsifiabilité numériques

Génère une datasheet markdown via `format_datasheet(sig)`.

### Philosophie : physique AVANT score

PTC n'est PAS un interpolateur paramétrique. C'est un calculateur ab initio PT.
Chaque nombre vient d'une chaîne de dérivation traçable jusqu'à s = 1/2 :

```
s = 1/2 → q = 1-2/μ → sin²(θ_p) → screening S → D₀ = cap × exp(-S)
```

Quand une observable est mauvaise, chercher le TERME PT MANQUANT dans le
pipeline, pas un coefficient à ajuster. Le diagnostic correct est :
"quelle face du simplexe T³ manque ?" — pas "quel nombre tuner ?".

---

## Architecture du moteur

```
SMILES or formula → Topology (auto-detect, PT solver for formulas)
  → atom_data (IE/EA from atom.py, 0 params, Z=1-118)
  → DIM 1: per-bond energy (_compute_bond_Fp, 3 faces P₁+P₂+P₃)
  → NLO: vertex coupling + T³ perturbative screening [Z mod {3,5,7}]
  → Poly mechanisms: ~42 named (sigma_crowding = T³-rewritten)
  → Spectral: T_P1 matrix eigenvalue corrections
  → Ring T³ holonomy: max(f_homo, f_T3_frustration)
  → Assembly: D_at = Σ bond_energies + E_spectral
```

## Key files

- `ptc/transfer_matrix.py` — main engine T_mol (~7000L), compute_D_at_transfer()
- `ptc/atom.py` — atomic IE/EA calculator, IE_eV(Z), EA_eV(Z)
- `ptc/bond.py` — BondResult, BondGeometry, r_equilibrium(), omega_e()
- `ptc/topology.py` — SMILES parser + PT solver dispatch, build_topology()
- `ptc/topology_solver.py` — variational sieve, solve_topology(formula)
- `ptc/constants.py` — PT constants (S3, D3, P1, Ry, etc.)
- `ptc/periodic.py` — period(), l_of(), element properties
- `ptc/api.py` — public Molecule class + nics()/aromaticity()/signature()
- `ptc/nics.py` — Pauling-London PT NICS, signed Hückel rule per channel
- `ptc/signature.py` — FullSignature for experimental datasheets
- `ptc/data/molecules.py` — primary benchmark (673 mol)
- `ptc/data/molecules_extended.py` — extended NIST molecules (+27)
- `ptc/data/molecules_atct.py` — ATcT 0K reference data
- `ptc/lcao/atomic_basis.py` — STO basis (s/p/d/f/g, l=0..4)
- `ptc/lcao/density_matrix.py` — Hueckel + overlap
- `ptc/lcao/fock.py` — HF SCF with DIIS, CPHF coupled response
- `ptc/lcao/giao.py` — STO eval/gradient + GIAO operators
- `ptc/lcao/shielding.py` — full sigma_alpha,beta tensor pipeline
- `ptc/lcao/current.py` — induced current density (CPHF + full GIAO j_para),
  bond strength via half-plane flux, NICS via Biot-Savart
- `ptc/lcao/cluster.py` — explicit-cluster builder, Cp* ligand generator,
  build_bi3_u2_cp_star4 (full inverse sandwich)
- `ptc/lcao/relativistic.py` — gamma_la/gamma_an PT-pure radial contraction
  (lifts L5 verrou for Z=57..103)
- `ptc/tests/test_transfer_matrix.py` — 263 tests
- `ptc/tests/test_nics.py` — 32 tests aromaticity
- `ptc/tests/test_lcao_*.py` + `test_geometry_fblock.py` + `test_bi3_u2_cluster.py`
  — 678 PASS total (LCAO + GIMIC + cluster + Bi3@U2 inverse sandwich)

## Commands

```bash
python -m pytest ptc/tests/ -x -q          # full suite
python -m pytest ptc/tests/test_bi3_u2_cluster.py -v  # test ultime Bi3@U2
```

## Code Exploration Policy

Use `cymbal` CLI for code navigation — prefer it over Read, Grep, Glob, or Bash for code exploration.
- **New to a repo?**: `cymbal structure` — entry points, hotspots, central packages. Start here.
- **To understand a symbol**: `cymbal investigate <symbol>` — returns source, callers, impact, or members based on what the symbol is.
- **To understand multiple symbols**: `cymbal investigate Foo Bar Baz` — batch mode, one invocation.
- **To trace an execution path**: `cymbal trace <symbol>` — follows the call graph downward (what does X call, what do those call).
- **To assess change risk**: `cymbal impact <symbol>` — follows the call graph upward (what breaks if X changes).
- Before reading a file: `cymbal outline <file>` or `cymbal show <file:L1-L2>`
- Before searching: `cymbal search <query>` (symbols) or `cymbal search <query> --text` (grep)
- Before exploring structure: `cymbal ls` (tree) or `cymbal ls --stats` (overview)
- To disambiguate: `cymbal show path/to/file.go:SymbolName` or `cymbal investigate file.go:Symbol`
- First run: `cymbal index .` to build the initial index (<1s). After that, queries auto-refresh — no manual reindexing needed.
- All commands support `--json` for structured output.
