# PTC — Roadmap : PT-LCAO + GIAO (mini quantum chemistry PT-natif)

**But** : étendre PTC d'un modèle scalaire avec corrections (Pauling-London + Lamb + σ_p simplifié) à un **mini package de chimie quantique PT-natif** capable de calculer le **tenseur de blindage σ complet à n'importe quel point 3D**, via :
1. Construction d'une **matrice densité PT moléculaire** ρ_PT depuis les occupations de couches atomiques (déjà disponibles dans `atom.py`) + couplage inter-atomique T³ (déjà dans `transfer_matrix.py`)
2. Application de la **perturbation magnétique standard** (potentiel vecteur A, opérateur Zeeman, GIAO) sur ρ_PT
3. Évaluation du tenseur σ_αβ(P) à un point arbitraire 3D

**Ordre de grandeur** : 1 500–3 000 lignes nettes, 4–8 semaines focalisées (sessions multiples).

**Trigger** : fermer le 25 % restant du gap Al₄²⁻ NICS_iso (post-Chantier 4h livre −25.54 vs exp −34) sans paramètre ajusté, et fournir un cadre généraliste pour tous les observables magnétiques (chemical shifts, J-couplings éventuels, susceptibilités).

---

## 0. Préalables obligatoires en début de session

```bash
cd "/Volumes/PT-YS-0326/LA THEORIE DE LA PERSITANCE/PT_PROJECTS/PTC"
find . -name "*.pyc" -delete && find . -path '*__pycache__*' -delete
PYTHONPATH=. python3 -m pytest ptc/tests/test_nics.py -q
PYTHONPATH=. python3 -c "
from ptc.tests.test_transfer_matrix import ALL_MOLS
from ptc.topology import build_topology
from ptc.transfer_matrix import compute_D_at_transfer
import statistics
errs=[]
for n,d in ALL_MOLS.items():
    try:
        r = compute_D_at_transfer(build_topology(d['smiles']))
        errs.append(abs((r.D_at-d['D_at'])/d['D_at']*100))
    except: pass
print(f'MAE={statistics.mean(errs):.3f}%')
"
```

**État de référence figé (post-Chantiers 1–4h, 2026-04-26)** :
- MAE 806 mol = **1.942 %** (à préserver)
- 7 outliers identiques (CH3NH2, thiophene, SO3mol, 1H-tetrazole, Dichloromethylene, Chloroacetylene, Disilanyl)
- `pytest test_nics.py` : **86/86 PASS**
- API publique : `mol.nics(z)`, `mol.nics_zz(z)`, `mol.nics_tensor(z)`, `mol.nics_with_environment(z)`, `mol.nics_biot_savart(z, force_redistribute)`, `mol.nics_paramagnetic(z, force_redistribute, enable_sigma_p)`

**Anti-régression universelle** : avant tout commit modifiant `nics.py`, `signature.py`, `screening_bond_v4.py`, `transfer_matrix.py`, ou `bond.py` :
1. MAE 806 mol DOIT rester ≤ 1.942 %
2. 86/86 tests `test_nics.py` DOIT rester PASS
3. Si nouvelle base orbitale est ajoutée dans `nics_lcao.py` (module dédié — voir §3), elle ne doit PAS impacter les calculs de D_at

---

## 1. Mission

Construire le pipeline :

```
PT atomic shells (atom.py)  ─┐
                             │
T³ inter-atomic coupling ────┼──► PT-LCAO basis  ──► ρ_PT (density matrix)
(transfer_matrix.py)         │           │
                             │           ▼
                             └──► overlap S_AB, kinetic T, e–e screening
                                         │
                                         ▼
                               GIAO perturbation (B-field, vector potential A)
                                         │
                                         ▼
                               σ_αβ(P) tensor at arbitrary point P ∈ ℝ³
                                         │
                                         ▼
                               NICS_iso, NICS_zz, susceptibility, J-couplings
```

Tout dérivé du **seul input s = 1/2** via les primitives PT existantes. **Aucun paramètre ajusté**, aucun fit, aucune base gaussienne empirique.

---

## 2. Principes PT-pure non-négociables

### 2.1 Pas de bases ajustées
La base orbitale **DOIT** être dérivée des écrantages PT déjà calculés dans `atom.py::IE_eV(Z)`. Spécifiquement :
- Pour atome Z, orbitale n,ℓ a forme radiale paramétrée par Z_eff = Z·exp(−S_polygon(Z, n, ℓ)) (Slater-like, mais Z_eff PT-purement dérivé)
- Forme angulaire : harmonique sphérique Y_ℓm standard (orthonormale, exacte, pas un fit)

### 2.2 Couplage inter-atomique = T³ already there
L'overlap S_AB et l'intégrale de transfert t_AB entre orbitales de deux atomes A, B **DOIT** réutiliser le formalisme T³ Z/3Z × Z/5Z × Z/7Z déjà encodé dans `transfer_matrix.py`. Pas de recalcul Slater-Roothaan via `pyscf` ou similaire.

### 2.3 Densité matricielle = occupations × LCAO coefficients
ρ_PT_AB = Σ_k n_k · c_kA* · c_kB où :
- n_k : occupation de MO k via L0 max-entropie + cycle-Hückel sur le polygone moléculaire
- c_kA : coefficient LCAO de l'orbitale atomique de A dans MO k, dérivé via diagonalisation T³

### 2.4 GIAO = London 1937 PT-purement implémenté
L'**origine de jauge** se déplace avec chaque orbitale atomique pour annuler la dépendance non-physique. Préfacteurs PT-purs (α, a₀, K_NICS = α²a₀/12 déjà présents).

### 2.5 Préfacteurs dimensionless = combinaisons de S_3, S_5, S_7, s, q_stat, q_therm
Aucun nouveau coefficient numérique sans dérivation depuis ces constantes. Si le calcul exige un facteur 0.42 quelconque, le remplacer par cos²(θ_5, q_stat) ou équivalent PT-pure, et tracer la dérivation.

---

## 3. Architecture proposée (module-par-module)

### Phase A — Base atomique PT-LCAO (~400 lignes, 1 sem.)

**Nouveau module** : `ptc/lcao/atomic_basis.py`

```python
@dataclass
class PTAtomicOrbital:
    Z: int           # atomic number
    n: int           # principal quantum number
    l: int           # angular momentum
    m: int           # magnetic quantum number (m_l)
    Z_eff: float     # PT screening from atom.py
    occ: float       # occupation in neutral / charged ground state

@dataclass
class PTAtomBasis:
    Z: int
    orbitals: List[PTAtomicOrbital]
    # full Z=1..118 ground state via PT shell filling

def build_atom_basis(Z: int, charge: int = 0) -> PTAtomBasis:
    """Build PT atomic basis from screening calculations in atom.py.
    Reuses IE_eV(Z) infrastructure for Z_eff per shell."""

def overlap_atomic(orb_A: PTAtomicOrbital, orb_B: PTAtomicOrbital,
                   r_AB: np.ndarray) -> complex:
    """⟨φ_A | φ_B⟩ = analytic Slater-type-orbital overlap with PT Z_eff.
    For STO 1s/2p/3d in standard form, closed-form expressions exist
    (Mulliken-Roothaan-Rieke 1949 series). Use those, NO numerical
    integration."""
```

**Validation Phase A** :
- Self-overlap = 1.000 ± 1e-9 pour 50 atomes-test
- H₂ overlap σ pour r_AB = 0.74 Å : ~0.75 (analytique Slater)
- Pas de régression sur D_at 806 mol (Phase A est LECTURE-SEULE des données atomiques, pas de modif moteur)

### Phase B — Matrice densité moléculaire ρ_PT (~600 lignes, 2 sem.)

**Nouveau module** : `ptc/lcao/density_matrix.py`

Construction LCAO via T³ inter-atomique :

```python
@dataclass
class PTMolecularBasis:
    atoms: List[PTAtomBasis]     # per-atom basis sets
    coords: np.ndarray           # 3D coordinates (Å), from r_equilibrium
    n_orbitals: int              # total basis size

def build_molecular_basis(topology: Topology) -> PTMolecularBasis:
    """Generate 3D coordinates and assemble full basis."""

def overlap_matrix(basis: PTMolecularBasis) -> np.ndarray:
    """S[i,j] = ⟨φ_i | φ_j⟩ over all basis pairs, using overlap_atomic."""

def kinetic_matrix(basis: PTMolecularBasis) -> np.ndarray:
    """T[i,j] = ⟨φ_i | -∇²/2 | φ_j⟩ analytic (STO-STO)."""

def density_matrix_PT(topology: Topology,
                      basis: PTMolecularBasis) -> np.ndarray:
    """Construct ρ_PT[i,j] from PT shell occupations + T³ coupling.

    PT-pure procedure :
    1. Assemble Hückel-like effective Hamiltonian H_eff[i,j] from
       T³ couplings between atoms (reuse transfer_matrix infrastructure)
    2. Diagonalize H_eff with overlap S → MO eigenvectors c_k
    3. Apply L0 max-entropie + cycle-Hückel to determine MO occupations
       n_k (already partially in nics.py for ring systems)
    4. Build ρ[i,j] = Σ_k n_k · c_ki* · c_kj
    """
```

**Validation Phase B** :
- ρ_PT idempotence à demi-occupation : ρ S ρ = ρ pour électrons appariés (test pour H₂, He, Ne)
- Trace(ρ S) = N_électrons ± 1e-6 pour 20 molécules-test
- Charges Mulliken cohérentes avec topology (anion → e- supplémentaires sur l'atome chargé)

### Phase C — Perturbation magnétique GIAO (~800 lignes, 2 sem.)

**Nouveau module** : `ptc/lcao/giao.py`

```python
@dataclass
class GIAOOperators:
    L_x: np.ndarray   # ⟨φ_i | L_x | φ_j⟩
    L_y: np.ndarray
    L_z: np.ndarray
    A_int: np.ndarray  # vector potential integrals

def london_phase_factor(orb: PTAtomicOrbital,
                         R_orb: np.ndarray,
                         B: np.ndarray) -> complex:
    """exp(i e/(2c) (B × R_orb) · r) — origin-shifting per orbital.
    PT-pure constants α, a₀."""

def angular_momentum_matrices(basis: PTMolecularBasis) -> GIAOOperators:
    """L_α matrix elements with London origin-shifts. Closed-form
    STO-STO + spherical harmonics products."""

def perturbed_density(rho: np.ndarray, S: np.ndarray, ops: GIAOOperators,
                      B: np.ndarray) -> np.ndarray:
    """First-order ρ⁽¹⁾ from Coupled-Perturbed PT (CPPT) :
       ρ⁽¹⁾ = Σ_{ai} (⟨a|H⁽¹⁾|i⟩ / (ε_a - ε_i)) (|a⟩⟨i| − |i⟩⟨a|)
    where H⁽¹⁾ = Σ_α B_α L_α (Zeeman-like in atomic units).
    Uses MO eigenvalues from Phase B diagonalization."""
```

**Validation Phase C** :
- Hermiticity numérique de L_α : L_α† = L_α à 1e-10
- Origine-jauge invariance : translation de basis_origin doit laisser σ inchangé
- Test analytique : H₂ atom isotropic σ ≈ 17.8 ppm exact (Lamb)
- Benzène σ_iso(0) = +8 ± 1 ppm via GIAO ≈ valeur PT scalaire actuelle

### Phase D — Tenseur σ et observables magnétiques (~500 lignes, 1 sem.)

**Nouveau module** : `ptc/lcao/shielding.py`

```python
@dataclass
class GIAOShieldingTensor:
    sigma: np.ndarray            # 3×3 tensor
    sigma_iso: float
    sigma_zz: float
    eigenvals: np.ndarray        # principal axes
    eigenvecs: np.ndarray
    span: float                  # σ_max - σ_min
    skew: float                  # asymmetry parameter

def shielding_tensor_at_point(P: np.ndarray,
                                basis: PTMolecularBasis,
                                rho: np.ndarray,
                                ops: GIAOOperators,
                                rho_pert: List[np.ndarray]) -> GIAOShieldingTensor:
    """σ_αβ(P) at arbitrary 3D point P. Returns full diagonal+off-diag tensor.
    Derivation:
       σ_αβ(P) = -∂²E/∂μ_α∂B_β |_{B,μ→0}
              = Tr(ρ · ∇_α 1/|r-P| · L_β / r³ + h.c.) - linear-response correction
    """

def nics_iso_giao(mol, ring_index=0, z=0.0) -> float:
    """High-level NICS via full GIAO pipeline. Single-shot test against
    existing scalar NICS values for benzene, S₃, etc."""
```

**Validation Phase D — milestones critiques** :

| Système | Cible exp | Cible GIAO | Pass |
|---|---:|---:|---|
| benzène NICS_iso | ~−8 ppm | ±2 ppm | requis |
| benzène NICS_zz | ~−14.5 ppm | ±3 ppm | requis |
| Cp⁻ NICS_iso | −13 | ±3 ppm | requis |
| **Al₄²⁻ NICS_iso** | **−34** | **±5 ppm** (cible principale) | **requis** |
| Bi₃³⁻ NICS_iso | −15 | ±3 ppm | recommandé |
| Bi₃@U₂ NICS | +0.08 | ±2 ppm (mais nécessite cap explicite) | bonus |

Le succès Phase D = **Al₄²⁻ ∈ [−39, −29] sans toucher d'autre paramètre**. Si la valeur sort, c'est que la base PT-LCAO ou la diagonalisation T³ a un bug à diagnostiquer (mais pas à fitter).

---

## 4. Anti-patterns à éviter

1. **NE PAS importer pyscf, psi4, openmm, ORCA, Gaussian, ou tout autre code de chimie quantique externe.** PTC reste self-contained.
2. **NE PAS utiliser des bases gaussiennes (cc-pVDZ, def2-SVP, etc.).** STO PT-Z_eff uniquement.
3. **NE PAS introduire de paramètres ajustés.** Tout coefficient doit être traçable jusqu'à s = 1/2.
4. **NE PAS faire de SCF Hartree-Fock complet.** Le PT-LCAO repose sur la diagonalisation directe d'un Hamiltonien Hückel-T³ déjà construit, pas sur une procédure auto-consistante. La densité PT vient des occupations PT, pas du minimum variationnel.
5. **NE PAS modifier les chantiers 1–4h existants.** Ce module est ADDITIF. L'API publique existante (`mol.nics()`, `mol.nics_zz()`, `mol.nics_tensor()`, `mol.nics_paramagnetic()`) doit rester intacte.
6. **NE PAS prétendre Phase X complète sans validation phase précédente.** Pipeline strictement séquentiel A → B → C → D.

---

## 5. Livrables par phase

Chaque phase produit :
1. **Code modifié** dans `ptc/lcao/{atomic_basis, density_matrix, giao, shielding}.py`
2. **Tests pytest** dans `ptc/tests/test_lcao_*.py` (additifs, ne touchent pas les 86 existants)
3. **Note de session** dans `docs/superpowers/plans/2026_<date>_lcao_phase_{A,B,C,D}.md`
4. **Commit dédié** : `feat(lcao): Phase X — <résultat chiffré>`
5. **Mise à jour de `PROMPT_PT_LCAO_GIAO.md`** : retirer la phase de la liste, mettre à jour l'état actuel

À l'achèvement Phase D :
- Mise à jour `PROMPT_NEXT_SESSION.md`
- Addendum à `PT_ARTICLES/PT_AROMATICITY/PT_AROMATICITY.tex` : section "Quantitative aromatic shielding from PT-LCAO + GIAO" couvrant Al₄²⁻ résolu et tableau prédictif sur 20+ systèmes

---

## 6. Quick-reference

```bash
# Setup
cd "/Volumes/PT-YS-0326/LA THEORIE DE LA PERSITANCE/PT_PROJECTS/PTC"
find . -name "*.pyc" -delete && find . -path '*__pycache__*' -delete

# Validation pipeline (à chaque commit)
PYTHONPATH=. python3 -m pytest ptc/tests/test_nics.py ptc/tests/test_lcao_*.py -q
PYTHONPATH=. python3 -m pytest ptc/tests/test_transfer_matrix.py -q
PYTHONPATH=. python3 -c "
from ptc.tests.test_transfer_matrix import ALL_MOLS
from ptc.topology import build_topology
from ptc.transfer_matrix import compute_D_at_transfer
import statistics
errs=[]
for n,d in ALL_MOLS.items():
    try:
        r = compute_D_at_transfer(build_topology(d['smiles']))
        errs.append(abs((r.D_at-d['D_at'])/d['D_at']*100))
    except: pass
print(f'MAE={statistics.mean(errs):.3f}%  N={len(errs)}')
"

# Test cible Al₄²⁻ post-Phase D
PYTHONPATH=. python3 -c "
from ptc.api import Molecule
print('NICS_iso Al4²⁻ via GIAO :', Molecule('[Al-]1[Al][Al][Al-]1').nics_giao(z=0).sigma_iso)
"
```

---

## 7. Pointeurs vers code existant à réutiliser (pas réimplémenter)

- `ptc/atom.py::IE_eV(Z)` : screening atomique, donne Z_eff par couche
- `ptc/atom.py::EA_eV(Z)` : affinités, occupations anioniques
- `ptc/transfer_matrix.py::T_mol` : Hamiltonien T³ moléculaire
- `ptc/bond.py::r_equilibrium(...)` : géométrie 3D des bonds (avec γ_rel pour Z≥72)
- `ptc/nics.py::_atomic_local_shielding(...)` : Lamb classique (sera étendu via GIAO)
- `ptc/nics.py::_max_aromatic_redistribute(...)` : occupations multi-fold (utile pour ρ_PT)
- `ptc/constants.py` : ALPHA_PHYS, A_BOHR, P1=3, P2=5, P3=7, S3, S5, S7, GAMMA_3, GAMMA_5, S_HALF (= s = 1/2)
- `ptc/periodic.py` : period(), l_of(), ns_config(), _np_of() — métadonnées atomiques

---

## 8. État de réussite

Le projet est complet quand :

1. ✅ MAE 806 mol toujours = 1.942 % (intacte)
2. ✅ 86/86 tests `test_nics.py` toujours PASS
3. ✅ Nouveau test : `test_lcao_*.py` ≥ 30 tests, tous PASS
4. ✅ **Al₄²⁻ NICS_iso ∈ [−39, −29] via `mol.nics_giao()`** (cible principale)
5. ✅ Benzène/Cp⁻/S₃ continuent à donner les valeurs validées (Chantier 4e)
6. ✅ Documentation : `PT_AROMATICITY.tex` addendum publishable
7. ✅ Aucun fit, aucun paramètre ajusté, traçabilité complète à s = 1/2

Si réussi, PTC devient le **seul calculateur de chimie quantique entièrement dérivé d'un input scalaire** s = 1/2, avec capacité NMR shielding rigoureuse.
