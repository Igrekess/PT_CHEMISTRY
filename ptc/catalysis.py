"""
PTC Catalysis — Catalyse derivee de la Theorie de la Persistance.

Ported from ptchem/core/catalysis.py (642 lines).
All PT-derived formulas preserved exactly.

Deux types de catalyse, TOUS derives de s = 1/2 :

1. CATALYSE HETEROGENE (surfaces metalliques)
   Le catalyseur cree un etat intermediaire adsorbe qui reduit le D_KL
   entre reactif et produit. Le principe de Sabatier emerge naturellement :
   - Adsorption trop faible -> pas de catalyse
   - Adsorption trop forte -> empoisonnement
   - Optimum : E_ads ~ S_HALF x Ea_uncat

   E_ads = S3 x D_bond(M, X) x f_surface(z_coord, d_band)

2. CATALYSE ENZYMATIQUE (biochimique)
   L'enzyme stabilise l'etat de transition par complementarite geometrique
   et electronique. En PT : l'enzyme cree un paysage D_KL local ou le TS
   a un D_KL minimal (= maximum de persistance locale).

   dG_cat = dG_uncat - E_stabilization
   E_stab = S3 x Sum_interactions D_i / P1

Tout depuis s = 1/2, sauf :

DERIVE DE PT (s = 1/2) :
  - Energies d'adsorption (sin^2_3 x D_bond x f_coord x f_d_band)
  - Barriere catalysee (Sabatier parabolique, Marcus-PT)
  - Interactions enzyme-TS (H-bond, ionique, hydrophobe, covalent, metal, pi, vdW)
  - d-band center (depuis IE et remplissage d)
  - Score Sabatier et position volcano

DONNEES EXPERIMENTALES (non derivees) :
  - Structures cristallines des metaux (bcc/fcc/hcp) — partiellement derivable
    de PT (d-filling < 5 -> bcc, > 7 -> fcc/hcp) mais le choix fcc vs hcp
    necessite un calcul d'energie de cohesion au 2e ordre non encore implemente.
  - Facettes de surface (111, 110) — liees a la structure cristalline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ptc.atom import Atom
from ptc.constants import (
    S_HALF, P1, P2, P3,
    S3, S5, S7,
    C3, C5,
    GAMMA_3, GAMMA_5,
    RY, A_BOHR, COULOMB_EV_A, AEM,
)
from ptc.periodic import block_of, n_fill, period


# ============================================================
# METAL SURFACE DATA (d-band center, coordination)
# ============================================================
# d-band center relative to Fermi level (eV)
# Source: Hammer-Norskov d-band model, confirmed by DFT
# In PT: epsilon_d proportional to -IE x C3^(n_d/10) (filling decay)
#
# These are DERIVED from atomic IE and d-filling, not fitted.

def _n_d_in_period(Z: int) -> int:
    """Return the number of d-electrons in the current period.

    For d-block elements (l=2), returns n_fill directly.
    """
    if block_of(Z) != 'd':
        return 0
    return n_fill(Z)


def _d_band_center(Z: int) -> float:
    """PT-derived d-band center relative to Fermi level (eV).

    En PT, le centre de la bande d depend du remplissage :
    epsilon_d = -Ry x S3 x (1 + n_d x C3 / P2)

    Empty d (n_d~0) -> epsilon_d ~ -Ry x S3 ~ -3.0 (near Fermi)
    Half d (n_d=5)  -> epsilon_d ~ -Ry x S3 x 1.4 ~ -4.2 (intermediate)
    Full d (n_d=10) -> epsilon_d ~ -Ry x S3 x 1.8 ~ -5.4 (deep below Fermi)

    Plus IE haute -> bande plus profonde (periode 4 vs 5 vs 6).
    """
    if block_of(Z) != 'd':
        return 0.0

    n_d = _n_d_in_period(Z)
    n_d = min(n_d, 10)

    per = period(Z)

    # PT d-band center: Ry × S₃ × n_d / (P₁ × P₂)
    # The d-band center measures the average energy of d-states
    # relative to the Fermi level.  In PT, this is the throughput
    # of d-filling through the P₁⊗P₂ cross-channel:
    #   ε_d = -Ry × S₃ × n_d / (P₁ × P₂) × period_correction
    # Empty d → ε_d = 0 (at Fermi).  Full d → deep below Fermi.
    # Period correction: heavier TM have deeper bands (relativistic
    # contraction of s-orbital raises d-band energy denominator).
    period_factor = 1.0 + S3 * (per - 4)  # 4=1.0, 5d=1.22, 6d=1.44

    epsilon_d = -RY * S3 * n_d / (P1 * P2) * period_factor

    return epsilon_d


# Bulk coordination numbers for common crystal structures
_BULK_COORD = {
    'fcc': 12,  # Cu, Ag, Au, Pt, Pd, Ni, Al
    'bcc': 8,   # Fe, W, Mo, Cr, V
    'hcp': 12,  # Ti, Co, Zn, Ru, Os
}

# Surface coordination for common facets
_SURFACE_COORD = {
    'fcc_111': 9,
    'fcc_100': 8,
    'fcc_110': 7,
    'bcc_110': 6,
    'bcc_100': 4,
    'step': 5,     # step edge
    'kink': 3,     # kink site
}

# Metal properties: (Z, crystal, common_surface)
_METALS = {
    # Period 4 (3d): Sc-Zn
    'Sc': (21, 'hcp', 'fcc_111'),
    'Ti': (22, 'hcp', 'fcc_111'),
    'V':  (23, 'bcc', 'bcc_110'),
    'Cr': (24, 'bcc', 'bcc_110'),
    'Mn': (25, 'bcc', 'bcc_110'),
    'Fe': (26, 'bcc', 'bcc_110'),
    'Co': (27, 'hcp', 'fcc_111'),
    'Ni': (28, 'fcc', 'fcc_111'),
    'Cu': (29, 'fcc', 'fcc_111'),
    'Zn': (30, 'hcp', 'fcc_111'),
    # Period 5 (4d): Y-Cd
    'Y':  (39, 'hcp', 'fcc_111'),
    'Zr': (40, 'hcp', 'fcc_111'),
    'Nb': (41, 'bcc', 'bcc_110'),
    'Mo': (42, 'bcc', 'bcc_110'),
    'Tc': (43, 'hcp', 'fcc_111'),
    'Ru': (44, 'hcp', 'fcc_111'),
    'Rh': (45, 'fcc', 'fcc_111'),
    'Pd': (46, 'fcc', 'fcc_111'),
    'Ag': (47, 'fcc', 'fcc_111'),
    'Cd': (48, 'hcp', 'fcc_111'),
    # Period 6 (5d): Lu-Hg
    'Lu': (71, 'hcp', 'fcc_111'),
    'Hf': (72, 'hcp', 'fcc_111'),
    'Ta': (73, 'bcc', 'bcc_110'),
    'W':  (74, 'bcc', 'bcc_110'),
    'Re': (75, 'hcp', 'fcc_111'),
    'Os': (76, 'hcp', 'fcc_111'),
    'Ir': (77, 'fcc', 'fcc_111'),
    'Pt': (78, 'fcc', 'fcc_111'),
    'Au': (79, 'fcc', 'fcc_111'),
    'Hg': (80, 'rhomb', 'fcc_111'),
}

SUPPORTED_METALS = sorted(_METALS.keys())


# ============================================================
# HETEROGENEOUS CATALYSIS
# ============================================================

@dataclass
class AdsorptionResult:
    """Result of a surface adsorption computation."""
    metal: str
    adsorbate: str
    E_ads: float          # energie d'adsorption (eV, negatif = exothermique)
    site: str             # site d'adsorption (atop, bridge, hollow)
    d_band_center: float  # epsilon_d du metal (eV)
    z_surface: int        # coordination du site


@dataclass
class CatalysisResult:
    """Result of a heterogeneous catalysis computation."""
    reaction: str
    metal: str
    Ea_uncatalyzed: float    # barriere sans catalyseur (eV)
    Ea_catalyzed: float      # catalyzed barrier (eV)
    Ea_cat_kJ: float         # barriere catalysee (kJ/mol)
    rate_enhancement: float  # k_cat / k_uncat a 298K
    E_ads_reactant: float    # adsorption du reactif (eV)
    E_ads_TS: float          # stabilisation du TS (eV)
    E_ads_product: float     # adsorption du produit (eV)
    sabatier_score: float    # score Sabatier [0, 1]
    volcano_position: str    # 'weak', 'optimal', 'strong'


def _parse_formula_simple(formula: str) -> dict:
    """Parse a simple formula like 'CO', 'H2', 'O2', 'N2', 'H', 'O', 'N'.

    Returns dict {symbol: count}.
    """
    import re
    tokens = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    result = {}
    for sym, cnt in tokens:
        if not sym:
            continue
        result[sym] = result.get(sym, 0) + (int(cnt) if cnt else 1)
    return result


def adsorption_energy(metal_sym: str, adsorbate_formula: str,
                      site: str = 'auto') -> AdsorptionResult:
    """Compute adsorption energy of a molecule on a metal surface.

    PT model:
        E_ads = S3 x D_bond(M, X) x f_coord x f_d_band

    where:
    - D_bond(M, X) = bond energy between metal atom and adsorbate's binding atom
    - f_coord = z_surface / z_bulk (undercoordination enhances adsorption)
    - f_d_band = (1 + epsilon_d / Ry) (d-band position modulates coupling)

    Parameters
    ----------
    metal_sym : str
        Metal symbol (e.g., 'Pt', 'Fe', 'Ni')
    adsorbate_formula : str
        Adsorbate formula (e.g., 'CO', 'H2', 'O2', 'N2', 'H', 'O', 'N')
    site : str
        Adsorption site ('atop', 'bridge', 'hollow', 'auto')
    """
    if metal_sym not in _METALS:
        raise ValueError(f"Metal inconnu : {metal_sym}")

    Z_metal, crystal, default_surface = _METALS[metal_sym]
    a_metal = Atom(Z_metal)

    # d-band center
    eps_d = _d_band_center(Z_metal)

    # Surface coordination
    z_surface = _SURFACE_COORD.get(default_surface, 9)
    z_bulk = _BULK_COORD.get(crystal, 12)

    # -- Identify binding atom in adsorbate --
    _Z_FROM_SYM = {
        'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'S': 16, 'Cl': 17,
    }

    parsed = _parse_formula_simple(adsorbate_formula)
    # Binding atom: the most electronegative non-H atom, or H if atomic
    binding_Z = 1  # default H
    max_chi = 0
    for sym in parsed:
        Z = _Z_FROM_SYM.get(sym, 1)
        a = Atom(Z)
        if a.chi_mulliken > max_chi and Z != 1:
            max_chi = a.chi_mulliken
            binding_Z = Z
    if len(parsed) == 1 and list(parsed.keys())[0] in _Z_FROM_SYM:
        binding_Z = _Z_FROM_SYM[list(parsed.keys())[0]]

    a_bind = Atom(binding_Z)

    # -- Adsorption site --
    if site == 'auto':
        # H prefers hollow/bridge; CO prefers atop; O prefers hollow
        if binding_Z == 1:
            site = 'hollow'
        elif binding_Z == 6:
            site = 'atop'
        else:
            site = 'bridge'

    # Site coordination multiplier
    site_mult = {'atop': 1.0, 'bridge': 1.0 + S3,
                 'hollow': 1.0 + S3 + S5}
    f_site = site_mult.get(site, 1.0)

    # -- Blyholder-PT 3-channel model (σ + π_back − Pauli) --
    # The adsorption bond is a 3-channel CRT process (Principe 3):
    #
    # Canal P₁ (σ donation): adsorbate LP donates σ density to metal.
    #   Molecular: E_σ = Ry × S₃ / P₁ (cap Shannon of P₁ channel)
    #   Atomic:    E_σ = IE_X × S₃ (full single-screening bond)
    #
    # Canal P₂ (π back-donation): metal d-band back-donates into π*.
    #   E_π = |ε_d| × S₅ × f_d(n_d)
    #   f_d = n_d × (2P₂ − n_d) / P₂²  (Friedel parabola, max at d⁵)
    #
    # Pauli repulsion from excess d-filling (n_d > P₂):
    #   E_P = |ε_d| × S₅ × max(0, n_d − P₂) / P₂
    #   Only fires for more-than-half-filled d-band (antibonding occupation).
    #   Uses |ε_d| (not IE_M): the repulsion comes from d-band overlap,
    #   not total ionization.
    #
    # E_ads = −(E_σ + E_π − E_P) × f_site × f_coord
    IE_M = a_metal.IE
    IE_X = a_bind.IE
    n_d = _n_d_in_period(Z_metal)

    # -- Molecular vs atomic adsorption --
    is_molecular = sum(parsed.values()) > 1

    # -- d-band filling function --
    # g(n_d) captures bonding/antibonding balance:
    #   d⁰-d⁵: antibonding states empty → strong bonding
    #   d⁵-d¹⁰: antibonding fills → bonding weakens
    #   g = 1 − (n_d − P₂)² / (P₂² + P₁²) keeps g > 0 for all n_d
    g_d = 1.0 - (n_d - P2) ** 2 / (P2 ** 2 + P1 ** 2)

    # Undercoordination: surface atoms bond more strongly
    f_coord = 1.0 + S3 * (1.0 - z_surface / z_bulk)

    if is_molecular:
        # ── MOLECULAR ADSORBATES (CO, H₂, O₂, N₂) ──
        # Canal P₁ (σ): capped by Ry × S₃ / P₁ (Shannon limit)
        E_sigma = RY * S3 / P1
        # Canal P₂ (π back-donation): Friedel parabola
        f_d_friedel = n_d * (2.0 * P2 - n_d) / (P2 ** 2)
        E_pi_back = abs(eps_d) * S5 * f_d_friedel
        # Total: σ + π, modulated by d-band filling
        E_ads = -(E_sigma + E_pi_back) * g_d * f_site * f_coord
    else:
        # ── ATOMIC ADSORBATES (H, O, N, C) ──
        # Atomic adsorption = FULL surface bond on Z/(2P₁)Z,
        # attenuated by surface coordination sharing.
        # D_MX = √(IE_M × IE_X) × S₃ / P₁ × g(n_d)
        # The √ (geometric mean) comes from PT Principe 5 (rotation
        # holonomique): the bond energy projects onto the Ry reference
        # via the geometric mean of the two vertices' IE.
        # The /P₁ accounts for the bond being ONE of P₁ modes on Z/(2P₁)Z.
        D_MX = math.sqrt(IE_M * IE_X) * S3 / P1 * g_d
        E_ads = -D_MX * f_site * f_coord

    return AdsorptionResult(
        metal=metal_sym,
        adsorbate=adsorbate_formula,
        E_ads=E_ads,
        site=site,
        d_band_center=eps_d,
        z_surface=z_surface,
    )


def catalyzed_barrier(reaction_eq: str, metal_sym: str) -> CatalysisResult:
    """Compute the catalyzed activation barrier for a reaction on a metal.

    PT Sabatier model:
        Ea_cat = Ea_uncat x (1 - f_sabatier)

    where f_sabatier = 4 x x x (1-x) is a parabola maximized at x = S_HALF,
    and x = |E_ads| / Ea_uncat is the adsorption strength relative to barrier.

    The Sabatier principle emerges naturally:
    - x ~ 0: weak adsorption -> f ~ 0 -> no catalysis
    - x ~ S_HALF: optimal -> f = 1 -> maximum catalysis
    - x ~ 1: strong adsorption -> f ~ 0 -> poisoning

    Parameters
    ----------
    reaction_eq : str
        Reaction equation (format: "A + B >> C + D")
    metal_sym : str
        Metal catalyst symbol
    """
    from ptc.reactions import compute_reaction

    # Uncatalyzed barrier via compute_reaction
    rxn = compute_reaction(reaction_eq)
    Ea_uncat = rxn.Ea_eV

    if Ea_uncat < 0.01:
        # No barrier to catalyze
        return CatalysisResult(
            reaction=reaction_eq, metal=metal_sym,
            Ea_uncatalyzed=Ea_uncat, Ea_catalyzed=0.0,
            Ea_cat_kJ=0.0, rate_enhancement=1.0,
            E_ads_reactant=0.0, E_ads_TS=0.0, E_ads_product=0.0,
            sabatier_score=0.0, volcano_position='optimal')

    # -- Adsorption of key ATOMIC species --
    # For heterogeneous catalysis, the rate-determining step is
    # dissociative adsorption. The descriptor is the ATOMIC adsorption
    # energy, not molecular.
    # Simple heuristic: pick the heaviest non-H atom from the reactants
    _Z_FROM_SYM_CAT = {
        'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'S': 16, 'Cl': 17,
    }
    _SYM_FROM_Z = {v: k for k, v in _Z_FROM_SYM_CAT.items()}

    # Find key atom from reaction string (heaviest non-H)
    key_atom_Z = 1
    for sym, Z in _Z_FROM_SYM_CAT.items():
        if sym in reaction_eq and Z > key_atom_Z and sym != 'H':
            key_atom_Z = Z

    # Compute ATOMIC adsorption energy
    key_sym = _SYM_FROM_Z.get(key_atom_Z, 'H')
    try:
        ads_reactant = adsorption_energy(metal_sym, key_sym)
        E_ads_R = ads_reactant.E_ads  # negative (strong for atoms)
    except Exception:
        E_ads_R = -1.0  # default moderate adsorption

    # -- Sabatier score --
    # x = |E_ads| / Ea_uncat (relative adsorption strength)
    x = min(abs(E_ads_R) / max(Ea_uncat, 0.01), 2.0)

    # Sabatier parabola: f = 4x(1-x), maximum at x = 0.5 = S_HALF
    # This is the fraction of barrier reduction
    f_sabatier = max(0.0, 4.0 * x * (1.0 - x))

    # Volcano position
    if x < S3:
        volcano_pos = 'weak'
    elif x > 1.0 - S3:
        volcano_pos = 'strong'
    else:
        volcano_pos = 'optimal'

    # -- Catalyzed barrier --
    # Ea_cat = Ea_uncat x (1 - f_sabatier x sin^2_3)
    # The S3 factor limits the maximum catalytic reduction:
    # even the best catalyst can't eliminate more than S3 ~ 22% of barrier
    # per elementary step. Multiple steps (cascade) can do more.
    #
    # CORRECTION: for industrial catalysts (Pt, Pd, Ru), the reduction
    # is much larger. The multi-step surface mechanism provides this:
    # - Dissociative adsorption breaks bonds on the surface (low barrier)
    # - Surface diffusion brings fragments together
    # - Recombination forms product bonds
    #
    # Total reduction = 1 - (1 - S3)^n_steps where n_steps = P1 (typ.)
    max_reduction = 1.0 - C3 ** P1  # ~ 98% for P1=3 steps

    Ea_cat = Ea_uncat * (1.0 - f_sabatier * max_reduction)
    Ea_cat = max(Ea_cat, S3 * S3 * Ea_uncat)  # minimum barrier

    # TS stabilization: how much the surface stabilizes the TS
    E_ads_TS = -(Ea_uncat - Ea_cat)  # negative = stabilization

    # Product adsorption (for poisoning assessment)
    E_ads_P = E_ads_R * C3  # products adsorb weaker (less unsaturated)

    # Rate enhancement: k_cat/k_uncat = exp((Ea_uncat - Ea_cat) / kT)
    kT = 8.617e-5 * 298.15  # eV at 298K
    delta_Ea = Ea_uncat - Ea_cat
    rate_enh = math.exp(min(delta_Ea / kT, 500))  # cap to avoid overflow

    return CatalysisResult(
        reaction=reaction_eq,
        metal=metal_sym,
        Ea_uncatalyzed=Ea_uncat,
        Ea_catalyzed=Ea_cat,
        Ea_cat_kJ=Ea_cat * 96.485,
        rate_enhancement=rate_enh,
        E_ads_reactant=E_ads_R,
        E_ads_TS=E_ads_TS,
        E_ads_product=E_ads_P,
        sabatier_score=f_sabatier,
        volcano_position=volcano_pos,
    )


def volcano_plot(reaction_eq: str,
                 metals: list[str] | None = None) -> list[CatalysisResult]:
    """Generate a Sabatier volcano plot for a reaction across metals.

    Returns catalysis results sorted by rate enhancement (best first).
    """
    if metals is None:
        metals = list(_METALS.keys())

    results = []
    for metal in metals:
        try:
            cat = catalyzed_barrier(reaction_eq, metal)
            results.append(cat)
        except Exception:
            pass

    results.sort(key=lambda c: -c.rate_enhancement)
    return results


def print_volcano(results: list[CatalysisResult]):
    """Display a volcano plot as a table."""
    if not results:
        print("  Aucun resultat.")
        return

    rxn = results[0].reaction
    print(f"\n{'=' * 75}")
    print(f"  Volcano plot : {rxn}")
    print(f"{'=' * 75}")
    print(f"  {'Metal':>5s}  {'E_ads':>7s}  {'Ea_cat':>7s}  {'Ea_kJ':>7s}  {'k_cat/k':>10s}"
          f"  {'Score':>5s}  {'Position':>8s}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*10}  {'-'*5}  {'-'*8}")

    for c in results:
        k_str = f"{c.rate_enhancement:.1e}" if c.rate_enhancement > 1e3 else f"{c.rate_enhancement:.1f}"
        print(f"  {c.metal:>5s}  {c.E_ads_reactant:+7.3f}  {c.Ea_catalyzed:7.4f}"
              f"  {c.Ea_cat_kJ:7.1f}  {k_str:>10s}"
              f"  {c.sabatier_score:5.3f}  {c.volcano_position:>8s}")

    print(f"{'-' * 75}")
    best = results[0]
    print(f"  Meilleur catalyseur : {best.metal}"
          f" (Ea = {best.Ea_cat_kJ:.1f} kJ/mol,"
          f" x{best.rate_enhancement:.1e} acceleration)")


# ============================================================
# ENZYMATIC CATALYSIS
# ============================================================

@dataclass
class EnzymeResult:
    """Resultat d'un calcul de catalyse enzymatique."""
    reaction: str
    enzyme_type: str            # type d'enzyme
    Ea_uncatalyzed: float       # barriere sans enzyme (eV)
    Ea_enzymatic: float         # barriere enzymatique (eV)
    Ea_enz_kJ: float            # (kJ/mol)
    rate_enhancement: float     # k_cat / k_uncat
    E_stabilization: float      # stabilisation TS (eV)
    n_contacts: int             # nombre de contacts enzyme-TS
    E_per_contact: float        # energy per contact (eV)


# -- Enzyme-TS interaction types (PT-derived energies) --
# Each interaction stabilizes the TS by a PT-derived amount.
# All from s = 1/2.

_INTERACTION_ENERGIES = {
    # Type: (E_per_contact in eV, description)
    'H_bond': (S3 * RY / (P1 * P2),
               "H-bond (N-H...O, O-H...N): S3 x Ry/(P1 x P2)"),
    'ionic': (S3 * COULOMB_EV_A / (P1 * A_BOHR),
              "Salt bridge (COO-...NH3+): S3 x Coulomb/(P1 x a_B)"),
    'hydrophobic': (S3 * S5 * RY / P1,
                    "Hydrophobic contact: S3 x S5 x Ry/P1"),
    'covalent_cat': (S3 * RY / P1,
                     "Covalent catalysis (Ser-O-substrate): S3 x Ry/P1"),
    'metal_ion': (S3 * S3 * RY,
                  "Metal ion stabilization (Zn2+, Mg2+): S3^2 x Ry"),
    'pi_stack': (S3 * S5 * S7 * RY,
                 "pi-stacking (Phe, Trp, Tyr): S3 x S5 x S7 x Ry"),
    'vdw': (S3 * S5 * S7 * RY / P1,
            "van der Waals: S3 x S5 x S7 x Ry/P1"),
}


# -- Standard enzyme classes and their typical interaction profiles --
_ENZYME_PROFILES = {
    'serine_protease': {
        # Chymotrypsin, trypsin, subtilisin
        # Catalytic triad: Ser-His-Asp + oxyanion hole
        'contacts': {
            'H_bond': 4,        # oxyanion hole (2) + His-substrate (2)
            'covalent_cat': 1,  # Ser-O-acyl intermediate
            'hydrophobic': 2,   # specificity pocket
            'ionic': 1,         # Asp-His relay
        },
        'description': "Serine protease (triade catalytique Ser-His-Asp)",
    },
    'metalloenzyme': {
        # Carbonic anhydrase, carboxypeptidase
        'contacts': {
            'metal_ion': 1,     # Zn2+ or Mg2+
            'H_bond': 3,        # substrate positioning
            'ionic': 1,         # charge stabilization
            'hydrophobic': 1,   # pocket
        },
        'description': "Metalloenzyme (Zn2+ ou Mg2+ au site actif)",
    },
    'lysozyme': {
        # Lysozyme: acid-base catalysis
        'contacts': {
            'H_bond': 3,        # Glu35, Asp52
            'ionic': 2,         # oxocarbenium stabilization
            'hydrophobic': 2,   # sugar binding
            'vdw': 3,           # extensive surface contact
        },
        'description': "Lysozyme (catalyse acide-base, Glu35/Asp52)",
    },
    'kinase': {
        # Protein kinase: phosphoryl transfer
        'contacts': {
            'metal_ion': 2,     # Mg2+ (2 ions)
            'H_bond': 4,        # DFG motif + substrate
            'ionic': 2,         # phosphate charges
            'hydrophobic': 1,   # ATP adenine
        },
        'description': "Kinase (transfert de phosphoryle, 2 Mg2+)",
    },
    'cytochrome_P450': {
        # Monooxygenase: Fe-heme + O2 activation
        'contacts': {
            'metal_ion': 1,     # Fe-heme (oxo-ferryl Fe(IV)=O)
            'covalent_cat': 1,  # O insertion into C-H
            'H_bond': 3,        # substrate positioning
            'hydrophobic': 4,   # large hydrophobic pocket
        },
        'description': "Cytochrome P450 (Fe-heme, hydroxylation)",
    },
    'aldolase': {
        # Aldolase: Schiff base mechanism
        'contacts': {
            'covalent_cat': 1,  # Schiff base (Lys-substrate)
            'H_bond': 3,        # enamine stabilization
            'ionic': 1,         # phosphate group
            'hydrophobic': 1,
        },
        'description': "Aldolase (base de Schiff, condensation aldolique)",
    },
    'cholinesterase': {
        # Acetylcholinesterase: fastest enzyme (diffusion-limited)
        'contacts': {
            'covalent_cat': 1,  # Ser-acyl
            'H_bond': 4,        # oxyanion hole + catalytic triad
            'ionic': 2,         # quaternary amine recognition
            'hydrophobic': 3,   # aromatic gorge (Trp, Phe)
            'pi_stack': 2,      # cation-pi interaction
        },
        'description': "Cholinesterase (enzyme la plus rapide, limite par diffusion)",
    },
    'nitrogenase': {
        # FeMo-cofactor: N2 fixation
        'contacts': {
            'metal_ion': 2,     # Fe7Mo cluster (2 cubanes)
            'H_bond': 4,        # proton relay chain
            'ionic': 2,         # charge compensation
            'hydrophobic': 2,   # N2 binding pocket
            'covalent_cat': 1,  # N2 -> 2NH3 multi-electron
        },
        'description': "Nitrogenase (FeMo-cofacteur, fixation N2)",
    },
    'DNA_polymerase': {
        # DNA replication: nucleotide incorporation
        'contacts': {
            'metal_ion': 2,     # Mg2+ (catalytic + structural)
            'H_bond': 6,        # Watson-Crick base pairing + backbone
            'ionic': 2,         # phosphodiester
            'hydrophobic': 2,   # base stacking
            'pi_stack': 2,      # nucleobase stacking
        },
        'description': "ADN polymerase (replication, 2 Mg2+, appariement WC)",
    },
    'ATP_synthase': {
        # Rotary motor: proton gradient -> ATP
        'contacts': {
            'metal_ion': 1,     # Mg2+-ATP
            'H_bond': 5,        # proton channel + substrate binding
            'ionic': 3,         # phosphate charges
            'hydrophobic': 2,   # rotor-stator interface
        },
        'description': "ATP synthase (moteur rotatif, gradient de protons)",
    },
    'proteasome': {
        # 20S proteasome: Thr-nucleophile, protein degradation
        'contacts': {
            'covalent_cat': 1,  # Thr1 N-terminal nucleophile
            'H_bond': 5,        # beta-sheet substrate recognition
            'hydrophobic': 4,   # channel lining
            'vdw': 4,           # extensive surface
        },
        'description': "Proteasome (degradation proteique, Thr nucleophile)",
    },
    'lipase': {
        # Lipase: ester hydrolysis at lipid-water interface
        'contacts': {
            'covalent_cat': 1,  # Ser nucleophile (Ser-His-Asp triad)
            'H_bond': 3,        # oxyanion hole
            'hydrophobic': 5,   # lid domain, acyl chain binding
            'vdw': 3,           # fatty acid positioning
        },
        'description': "Lipase (hydrolyse d'esters, interface lipide-eau)",
    },
    'carbonic_anhydrase': {
        # Fastest metalloenzyme: CO2 + H2O -> HCO3- + H+
        'contacts': {
            'metal_ion': 1,     # Zn2+ (tetrahedral)
            'H_bond': 4,        # proton shuttle (His64)
            'ionic': 1,         # Zn-OH nucleophile
            'hydrophobic': 2,   # CO2 binding
        },
        'description': "Anhydrase carbonique (Zn2+, CO2 + H2O -> HCO3-)",
    },
    'rubisco': {
        # Most abundant enzyme: CO2 fixation in photosynthesis
        'contacts': {
            'metal_ion': 1,     # Mg2+
            'H_bond': 5,        # enediol intermediate stabilization
            'ionic': 2,         # carbamate + phosphate
            'covalent_cat': 1,  # carboxylation
            'hydrophobic': 1,
        },
        'description': "RuBisCO (fixation CO2, photosynthese, enzyme la plus abondante)",
    },
    'generic': {
        # Generic enzyme with moderate contacts
        'contacts': {
            'H_bond': 3,
            'hydrophobic': 2,
            'ionic': 1,
        },
        'description': "Enzyme generique (profil moyen)",
    },
}

ENZYME_TYPES = list(_ENZYME_PROFILES.keys())


def enzyme_catalysis(reaction_eq: str, enzyme_type: str = 'generic',
                     custom_contacts: dict | None = None) -> EnzymeResult:
    """Compute enzymatic rate enhancement using PT TS-stabilization model.

    The enzyme stabilizes the TS by providing complementary interactions.
    Each interaction contributes a PT-derived energy, and the total
    stabilization lowers the activation barrier.

    E_stab = Sum_types n_contacts x E_per_contact
    Ea_enz = Ea_uncat - E_stab (but >= S3^2 x Ea_uncat as minimum)

    Parameters
    ----------
    reaction_eq : str
        Chemical reaction equation
    enzyme_type : str
        One of: 'serine_protease', 'metalloenzyme', 'lysozyme',
                'kinase', 'generic'
    custom_contacts : dict, optional
        Override contact profile: {'H_bond': 4, 'ionic': 2, ...}
    """
    from ptc.reactions import compute_reaction

    # Uncatalyzed barrier
    rxn = compute_reaction(reaction_eq)
    Ea_uncat = rxn.Ea_eV

    # Get contact profile
    if custom_contacts:
        contacts = custom_contacts
    else:
        profile = _ENZYME_PROFILES.get(enzyme_type, _ENZYME_PROFILES['generic'])
        contacts = profile['contacts']

    # -- Compute TS stabilization --
    E_stab = 0.0
    n_total = 0
    for interaction_type, n_contacts in contacts.items():
        if interaction_type in _INTERACTION_ENERGIES:
            E_per, _ = _INTERACTION_ENERGIES[interaction_type]
            E_stab += n_contacts * E_per
            n_total += n_contacts

    # -- Cooperative enhancement --
    # Multiple contacts cooperate: each additional contact beyond P1
    # enhances the others by S5 (entropic cooperativity).
    if n_total > P1:
        coop = 1.0 + (n_total - P1) * S5
        E_stab *= coop

    # -- Enzymatic barrier --
    Ea_enz = Ea_uncat - E_stab
    Ea_enz = max(Ea_enz, S3 * S3 * Ea_uncat)  # minimum barrier

    # Rate enhancement
    kT = 8.617e-5 * 298.15
    delta_Ea = Ea_uncat - Ea_enz
    rate_enh = math.exp(min(delta_Ea / kT, 500))

    E_per_contact = E_stab / max(n_total, 1)

    return EnzymeResult(
        reaction=reaction_eq,
        enzyme_type=enzyme_type,
        Ea_uncatalyzed=Ea_uncat,
        Ea_enzymatic=Ea_enz,
        Ea_enz_kJ=Ea_enz * 96.485,
        rate_enhancement=rate_enh,
        E_stabilization=E_stab,
        n_contacts=n_total,
        E_per_contact=E_per_contact,
    )


def print_enzyme(e: EnzymeResult):
    """Display enzyme catalysis result."""
    desc = _ENZYME_PROFILES.get(e.enzyme_type, {}).get('description', e.enzyme_type)
    print(f"\n{'=' * 65}")
    print(f"  {e.reaction}")
    print(f"  Enzyme : {desc}")
    print(f"{'=' * 65}")
    print(f"  Ea (non catalyse) : {e.Ea_uncatalyzed:.4f} eV"
          f" = {e.Ea_uncatalyzed * 96.485:.1f} kJ/mol")
    print(f"  Ea (enzymatique)  : {e.Ea_enzymatic:.4f} eV"
          f" = {e.Ea_enz_kJ:.1f} kJ/mol")
    print(f"  Stabilisation TS  : {e.E_stabilization:.4f} eV"
          f" ({e.n_contacts} contacts, {e.E_per_contact:.4f} eV chacun)")
    print(f"  Acceleration      : x{e.rate_enhancement:.1e}")
    print(f"{'-' * 65}")


# ============================================================
# EXPERIMENTAL ADSORPTION BENCHMARK (April 2026)
# ============================================================

def benchmark_adsorption() -> dict:
    """Benchmark E_ads against experimental TPD/calorimetry data.

    Returns dict with mae, n_tests, rows (per-pair details), kendall_tau.
    """
    from ptc.data.experimental_adsorption import EXP_ADSORPTION

    rows = []
    for (metal, ads), exp_eV in sorted(EXP_ADSORPTION.items()):
        try:
            r = adsorption_energy(metal, ads)
            err = abs(r.E_ads - exp_eV)
            rows.append({
                'metal': metal,
                'adsorbate': ads,
                'E_ads_PT': r.E_ads,
                'E_ads_exp': exp_eV,
                'error_eV': err,
            })
        except Exception:
            pass

    if not rows:
        return {'mae': float('inf'), 'n_tests': 0, 'rows': []}

    mae = sum(r['error_eV'] for r in rows) / len(rows)

    # Kendall tau (ranking correlation)
    import itertools
    pts = [r['E_ads_PT'] for r in rows]
    exps = [r['E_ads_exp'] for r in rows]
    conc = sum(1 for i, j in itertools.combinations(range(len(pts)), 2)
               if (pts[i] - pts[j]) * (exps[i] - exps[j]) > 0)
    disc = sum(1 for i, j in itertools.combinations(range(len(pts)), 2)
               if (pts[i] - pts[j]) * (exps[i] - exps[j]) < 0)
    n = len(pts)
    tau = (conc - disc) / (n * (n - 1) / 2) if n > 1 else 0

    return {
        'mae': mae,
        'n_tests': len(rows),
        'rows': rows,
        'kendall_tau': tau,
        'n_pass': sum(1 for r in rows if r['error_eV'] < 0.5),
    }


# ============================================================
# INTERACTION ENERGY TABLE
# ============================================================

def print_interactions():
    """Display all PT-derived interaction energies."""
    print(f"\n{'=' * 70}")
    print(f"  Energies d'interaction PT-derivees (s = 1/2)")
    print(f"{'=' * 70}")
    for name, (E, desc) in _INTERACTION_ENERGIES.items():
        print(f"  {name:15s}  {E:.4f} eV  ({E*96.485:6.1f} kJ/mol)  {desc}")
    print(f"{'-' * 70}")
