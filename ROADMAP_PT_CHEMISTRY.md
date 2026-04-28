# PT Chemistry — Roadmap vers une suite QM ab initio PT-pure complète

État au 2026-04-27 (post Phase 6.B.7).

## Couverture actuelle

| Moteur | Domaine | État |
|---|---|---|
| **PTC** (`ptc/transfer_matrix.py`, `ptc/atom.py`, `ptc/bond.py`) | Atomisation, IE, EA, géométries, ω_e diatomiques, angles, screening | Production : 700 mol, MAE 1.7 %, 0 paramètre |
| **PT_LCAO** (`ptc/lcao/*`) | Aromaticité (NICS), shielding RMN σ_d + σ_p^MP2-GIAO Stanton-Gauss | Pipeline complet, 174 tests PASS, validé S-G 1996 (38 % cc-pVDZ N₂) |

## Roadmap par leverage décroissant

Tri par **(impact chimique × accessibilité d'implémentation)** ; chaque entrée donne effort estimé, motivation PT-pure et critère de validation.

---

### Tier 1 — Spectroscopies essentielles (utilisées tous les jours)

#### 🟢 IR / Raman vibrationnel polyatomique
**Manque** : matrice Hessienne ∂²E/∂x∂y pour molécules polyatomiques. On a ω_e diatomique seul.

- **Code** : Hessian par finite-differences (sur PTC) + diagonalisation pondérée massique → modes normaux + IR intensities (∂μ/∂Q).
- **PT-pur** : les fréquences héritent γ_3, γ_5, γ_7 ; potentiel pour signature PT vérifiable.
- **Effort** : 3-5 jours
- **Validation** : H₂O, CO₂, NH₃, CH₄ vs NIST (~1 % MAE expected).
- **Recommandation** : **next priority** — réutilise tout PTC, gros impact, peu de code.

#### 🟡 UV-Vis / spectres d'absorption électronique (CIS / TDHF)
**Manque** : excitations HOMO→LUMO. On a tous les ingrédients post-MP2 dans `ptc/lcao/mp2.py`.

- **Code** : `cis_excitations.py` réutilise les MO eigvals + ERIs.
- **PT-pur** : énergies d'excitation = γ_p × Δε_HF avec corrections PT.
- **Effort** : 5-7 jours
- **Validation** : H₂CO π→π*, butadiène, benzène (états singulet S₁, S₂ vs exp).

#### 🟡 EPR g-tensor + couplages hyperfins A
**Manque** : pour radicaux (3 % des molécules organiques utiles). Réutilise CPHF.

- **PT-pur** : g_e + δg vient de la même physique q_stat/q_therm que α_EM. Direct lien PT.
- **Effort** : 4-6 jours
- **Validation** : NO•, CH₃•, O₂•⁻ (tabulated δg, A vs ESR experimental).

---

### Tier 2 — Réactivité (chimie organique théorique)

#### 🟡 Transition states + barrières d'activation
**Manque** : optimisation géométrique générale + saddle point search.

- **Code** : BFGS / RFO pour minima, P-RFO ou dimer method pour TS.
- **PT-pur** : la PES suit les ratios γ_p ; les TS héritent les bifurcations PT (q_stat ↔ q_therm).
- **Effort** : 7-10 jours
- **Validation** : SN2 H⁻+CH₃Br, Diels-Alder, [3,3]-sigmatropic.
- **Pourquoi crucial** : c'est ce qui transforme le projet de "calculateur structurel" à "calculateur prédictif de réactions" — gros leverage publication.

#### 🟡 Énergies libres + thermochimie finie T
**Manque** : ΔG, ΔH, ΔS à T finie (au-delà ZPE déjà couvert par ω_e).

- **Code** : partition function harmonique + rotor + translation + ZPE.
- **PT-pur** : la bifurcation PT q_stat/q_therm est exactement la bifurcation thermodynamique → liaison directe.
- **Effort** : 2-3 jours
- **Validation** : ΔH_combustion CH₄, ΔG dissociation H₂, équilibre tautomerique.

---

### Tier 3 — Solvation + grande échelle

#### 🟢 Solvation implicite (PCM / COSMO)
**Manque** : terme V_sol dans H_core. Indispensable pour comparer NMR / spectres en solution à l'exp.

- **Code** : surface moléculaire + cavité + Poisson-Boltzmann implicite.
- **PT-pur** : ε_solvent dérivé de PT (ε_H₂O = 78.75 déjà calculé Phase 6.B.6 mémoire).
- **Effort** : 4-6 jours
- **Validation** : ΔG_hydration, NMR shifts en solution.

#### 🔴 Périodique (cristaux, surfaces)
**Manque** : Bloch states, intégration BZ. Couvre matériaux + catalyse hétérogène.

- **PT-pur** : T³ = Z/3Z × Z/5Z × Z/7Z est déjà la structure naturelle → mapping direct sur réseaux cristallins via primes actifs.
- **Effort** : 2-3 semaines
- **Validation** : Si bandgap, NaCl lattice constant, surface Pt(111).

---

### Tier 4 — Précision quantitative

#### 🟡 CCSD, CCSD(T)
**Manque** : T1+T2 amplitudes itératives au-delà MP2. La référence "gold standard".

- **Code** : ~3000 lignes (boucle CC, équations couplées T1, T2).
- **Effort** : 2 semaines
- **Validation** : reproduire Stanton-Gauss 1996 N₂ NMR à 100 % (vs notre 38 % actuel).
- **Pourquoi crucial post-Phase 6.B.8** : si la base contractée pVDZ-PT atteint 50-70 % de cc-pVDZ, alors CCSD ferme le reste.

#### 🔴 DFT hybride au-delà du PT-GGA actuel
**Manque** : fonctionnels hybrides (B3LYP-PT), corrections dispersion D3.

- **Code** : extension du `ptchem_v10_1` hybride.
- **Effort** : 1-2 semaines
- **Validation** : vs B3LYP-D3 sur QM9 5K.

---

### Tier 5 — Dynamique

#### 🟡 Born-Oppenheimer MD
**Manque** : simulation temporelle des trajectoires moléculaires.

- **Code** : intégrateur Verlet + gradients analytiques de l'énergie.
- **Effort** : 1 semaine
- **Validation** : vibrations H₂O, conformations butane, paths réactionnels.

---

## Suggéré pour les prochaines sessions

**Phase 6.C — Hessien + IR (3-5 jours)**
Le levier numéro un en termes de ROI :
- Effort minimal (réutilise PTC)
- Couvre IR/Raman → ~70 % du workflow chimie analytique
- Signature PT prédictive (les fréquences héritent γ_p)
- Préparation pour MD et thermochimie finie T

**Phase 6.D — Thermochimie finie T (2-3 jours)**
Suite directe : avec Hessien on a tout pour construire la partition function vibrationnelle.
- ΔG/ΔH/ΔS à T finie
- Lien direct PT q_stat/q_therm = thermodynamique

**Phase 6.E — Transition states + barrières (7-10 jours)**
Le passage à la vraie chimie computationnelle prédictive :
- Optimisation géométrique RFO pour minima
- P-RFO ou dimer pour TS
- IRC (intrinsic reaction coordinate)
- Eyring → constantes de vitesse

**Phase 6.F — Solvation PCM (4-6 jours)**
Pour comparer à l'expérience en solution.

**Phase 6.G — CCSD** (post Phase 6.B.8)
Si la base contractée pVDZ-PT donne 50 % cc-pVDZ, CCSD ferme le reste.

## Vision à long terme

Avec les Phases 6.C → 6.G, on a une **suite QM ab initio PT-pure complète** couvrant :
- ✓ Énergies (PTC + LCAO)
- ✓ Géométries (PTC)
- ✓ NMR (PT_LCAO)
- ⏭ IR / Raman
- ⏭ Thermochimie finie T
- ⏭ Réactivité (TS, IRC, k_rate)
- ⏭ Solvation
- ⏭ Précision CCSD

→ Comparable au workflow d'un chimiste utilisant Gaussian / ORCA / NWChem, mais
**dérivé entièrement de s = 1/2 et zéro paramètre**. Le projet PT_CHEMISTRY
devient une suite de chimie quantique publiable indépendamment.

## Référencement

Cette roadmap sert de référence pour les prompts session-suivantes
(`PROMPT_PT_LCAO_QUANT.md`, futurs `PROMPT_PT_VIBRATIONAL.md`,
`PROMPT_PT_REACTIVITY.md`, etc.). Mise à jour à chaque Phase complétée.
