"""
reaction_predictor.py — Predict reaction products from reactants only.

Migrated from modeleV10/pt_reactions.py (Level C predictor).
Uses PTC cascade + topology_solver instead of modeleV10 engine.

Algorithm:
  1. Extract all atoms from reactants
  2. Enumerate product partitions (known molecules + splits)
  3. Evaluate each partition via D_at (cascade)
  4. Best partition = most negative Delta_G
  5. Ea via Marcus-PT: (lambda + DG)^2 / (4*lambda)
  6. Kinetic blocking check (Ea > Ry * sin^2_3 * s)

0 adjustable parameters. All from s = 1/2.

April 2026 -- Theorie de la Persistance
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ptc.constants import S_HALF, S3, RY, P1
from ptc.data.experimental import SYMBOLS

_SYMBOL_TO_Z = {sym: Z for Z, sym in SYMBOLS.items()}

# ====================================================================
# KNOWN STABLE MOLECULES (signature -> SMILES for PTC)
# ====================================================================

_KNOWN_MOLECULES: Dict[Tuple[int, ...], Tuple[str, str]] = {
    # signature (sorted Z tuple) -> (SMILES, name)
    # Diatomics
    (1, 1): ("[H][H]", "H2"),
    (7, 7): ("N#N", "N2"),
    (8, 8): ("O=O", "O2"),
    (9, 9): ("FF", "F2"),
    (17, 17): ("ClCl", "Cl2"),
    (35, 35): ("BrBr", "Br2"),
    (1, 9): ("[H]F", "HF"),
    (1, 17): ("[H]Cl", "HCl"),
    (1, 35): ("[H]Br", "HBr"),
    (6, 8): ("[C-]#[O+]", "CO"),
    (7, 8): ("[N]=O", "NO"),
    # Triatomics
    (1, 1, 8): ("[H]O[H]", "H2O"),
    (1, 1, 16): ("[H]S[H]", "H2S"),
    (6, 8, 8): ("O=C=O", "CO2"),
    (8, 8, 16): ("O=S=O", "SO2"),
    (1, 7, 8): ("O=N", "HNO"),
    # 4-atom
    (1, 1, 1, 7): ("[H]N([H])[H]", "NH3"),
    (1, 1, 6, 8): ("C=O", "CH2O"),
    (1, 1, 8, 8): ("OO", "H2O2"),
    # 5-atom
    (1, 1, 1, 1, 6): ("C", "CH4"),
    (1, 1, 1, 1, 14): ("[SiH4]", "SiH4"),
    (1, 1, 1, 15): ("[H]P([H])[H]", "PH3"),
    # Halides
    (7, 9, 9, 9): ("FN(F)F", "NF3"),
    (6, 9, 9, 9, 9): ("FC(F)(F)F", "CF4"),
    (1, 1, 1, 6, 17): ("CCl", "CH3Cl"),
    # Hydrocarbons
    (1, 1, 1, 1, 1, 1, 6, 6): ("CC", "C2H6"),
    (1, 1, 1, 1, 6, 6): ("C=C", "C2H4"),
    (1, 1, 6, 6): ("C#C", "C2H2"),
    (1, 6, 7): ("C#N", "HCN"),
    # Salts / ionic
    (11, 17): ("[Na]Cl", "NaCl"),
    (11, 9): ("[Na]F", "NaF"),
    (11, 8, 1): ("[Na]O[H]", "NaOH"),
    # Alcohols / organics
    (1, 1, 1, 1, 6, 8): ("CO", "CH3OH"),
}


# ====================================================================
# FORMULA PARSER
# ====================================================================

def _parse_formula(formula: str) -> List[int]:
    """Parse formula like 'CH4', 'H2O', 'C2H6' into list of Z."""
    z_list = []
    tokens = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    for sym, count in tokens:
        if sym not in _SYMBOL_TO_Z:
            continue
        Z = _SYMBOL_TO_Z[sym]
        n = int(count) if count else 1
        z_list.extend([Z] * n)
    return z_list


def _parse_smiles_atoms(smiles: str) -> List[int]:
    """Extract atom list from SMILES via PTC topology."""
    try:
        from ptc.topology import build_topology
        topo = build_topology(smiles)
        return list(topo.Z_list)
    except Exception:
        return _parse_formula(smiles)


# ====================================================================
# PRODUCT ENUMERATION
# ====================================================================

def _atom_sig(z_list: List[int]) -> Tuple[int, ...]:
    return tuple(sorted(z_list))


def _enumerate_partitions(atom_pool: List[int],
                          max_products: int = 4) -> List[List[Tuple[int, ...]]]:
    """Enumerate possible product partitions from an atom pool.

    Strategy:
      1. Try combinations of known stable molecules
      2. Binary splits for leftovers
      3. "No reaction" = original pool
    """
    pool_counter = Counter(atom_pool)
    partitions: List[List[Tuple[int, ...]]] = []

    # Build known molecule signatures sorted by size (largest first)
    known_sigs = []
    for sig in _KNOWN_MOLECULES:
        known_sigs.append((Counter(sig), sig))
    known_sigs.sort(key=lambda x: -sum(x[0].values()))

    # Recursive search for partitions using known molecules
    _search(pool_counter, known_sigs, [], partitions, max_products, depth=5)

    # "No reaction" partition = all atoms as one molecule
    partitions.append([_atom_sig(atom_pool)])

    # Deduplicate
    unique = []
    seen = set()
    for p in partitions:
        key = tuple(sorted(p))
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique[:20]


def _search(remaining: Counter, known_sigs, current, results,
            max_products: int, depth: int):
    """Recursive partition search."""
    if depth <= 0 or len(current) >= max_products:
        if current and sum(remaining.values()) == 0:
            results.append(list(current))
        return

    if sum(remaining.values()) == 0:
        if current:
            results.append(list(current))
        return

    for sig_counter, sig_tuple in known_sigs:
        if all(remaining.get(z, 0) >= sig_counter[z] for z in sig_counter):
            new_rem = remaining.copy()
            for z, c in sig_counter.items():
                new_rem[z] -= c
                if new_rem[z] == 0:
                    del new_rem[z]
            _search(new_rem, known_sigs, current + [sig_tuple],
                    results, max_products, depth - 1)


# ====================================================================
# EVALUATOR
# ====================================================================

def _compute_D_at(smiles: str) -> float:
    """Compute D_at for a molecule via PTC cascade."""
    try:
        from ptc.cascade import compute_D_at_cascade
        from ptc.topology import build_topology
        topo = build_topology(smiles)
        r = compute_D_at_cascade(topo)
        return r.D_at
    except Exception:
        return 0.0


def _eval_partition(partition: List[Tuple[int, ...]]) -> Tuple[float, List[dict]]:
    """Evaluate a partition: compute total D_at of all product molecules."""
    total_E = 0.0
    details = []

    for sig in partition:
        if len(sig) < 2:
            # Isolated atom: D_at = 0
            sym = SYMBOLS.get(sig[0], '?') if sig else '?'
            details.append({'formula': sym, 'smiles': '', 'D_at': 0.0})
            continue

        known = _KNOWN_MOLECULES.get(sig)
        if known:
            smiles, name = known
            D_at = _compute_D_at(smiles)
            details.append({'formula': name, 'smiles': smiles, 'D_at': D_at})
            total_E += D_at
        else:
            # Unknown molecule: try topology solver
            try:
                from ptc.topology_solver import solve_topology
                from ptc.cascade import compute_D_at_cascade
                z_list = list(sig)
                topo = solve_topology(_formula_from_zlist(z_list))
                r = compute_D_at_cascade(topo)
                details.append({
                    'formula': _formula_from_zlist(z_list),
                    'smiles': '',
                    'D_at': r.D_at,
                })
                total_E += r.D_at
            except Exception:
                details.append({
                    'formula': _formula_from_zlist(list(sig)),
                    'smiles': '',
                    'D_at': 0.0,
                })

    return total_E, details


def _formula_from_zlist(z_list: List[int]) -> str:
    """Generate formula from Z list (Hill order)."""
    counts = Counter(z_list)
    parts = []
    for Z in [6, 1]:
        if Z in counts:
            sym = SYMBOLS.get(Z, '?')
            parts.append(sym + (str(counts[Z]) if counts[Z] > 1 else ''))
            del counts[Z]
    for Z in sorted(counts, key=lambda z: SYMBOLS.get(z, '')):
        sym = SYMBOLS.get(Z, '?')
        parts.append(sym + (str(counts[Z]) if counts[Z] > 1 else ''))
    return ''.join(parts)


# ====================================================================
# RESULT
# ====================================================================

@dataclass
class PredictionResult:
    """Result of reaction prediction."""
    reacts: bool                    # True if a reaction occurs
    reactant_D_at: float            # total D_at of reactants (eV)
    product_D_at: float             # total D_at of best products (eV)
    delta_G: float                  # eV (< 0 = exothermic)
    delta_G_kJ: float               # kJ/mol
    Ea: float                       # activation energy (eV)
    Ea_kJ: float                    # kJ/mol
    kinetically_blocked: bool       # True if Ea too high
    reactant_info: List[dict]       # per-reactant details
    product_info: List[dict]        # per-product details
    alternatives: List[dict] = field(default_factory=list)


# ====================================================================
# MAIN PREDICTOR
# ====================================================================

_EV_TO_KJ = 96.485


def _find_best_stoichiometry(unique_smiles: List[str],
                             max_coeff: int = 4) -> List[str]:
    """Find the minimal stoichiometric ratio that gives a favorable reaction.

    For 2 reactants A, B: tries a:b sorted by total molecules (smallest first).
    Picks the SMALLEST ratio where delta_G per molecule is most negative.
    """
    if len(unique_smiles) < 2:
        return unique_smiles

    A, B = unique_smiles[0], unique_smiles[1]
    rest = unique_smiles[2:]

    # Generate ratios sorted by total count (smallest stoichiometry first)
    ratios = []
    for a in range(1, max_coeff + 1):
        for b in range(1, max_coeff + 1):
            ratios.append((a + b, a, b))
    ratios.sort()  # smallest total first

    best_dG_per_mol = 1e10  # start with worst
    best_list = unique_smiles
    found_any_reaction = False

    for _, a, b in ratios:
        candidate = [A] * a + [B] * b + rest
        try:
            result = _predict_single(candidate)
            dG_per_mol = result.delta_G / (a + b)

            # Accept the best non-identity partition (even endothermic).
            # Compare by atom signature (not SMILES string, since
            # "H2" and "[H][H]" are different strings but same molecule).
            reactant_sigs = {_atom_sig(_parse_smiles_atoms(A)),
                             _atom_sig(_parse_smiles_atoms(B))}
            product_sigs = [_atom_sig(_parse_smiles_atoms(p['smiles']))
                            for p in result.product_info
                            if p.get('smiles')]
            has_real_products = (
                product_sigs and
                not all(ps in reactant_sigs for ps in product_sigs)
            )

            if has_real_products and dG_per_mol < best_dG_per_mol:
                best_dG_per_mol = dG_per_mol
                best_list = candidate
                found_any_reaction = True
        except Exception:
            continue

    return best_list


def _predict_single(reactant_smiles: List[str]) -> PredictionResult:
    """Core prediction for a fixed set of reactant molecules."""
    all_atoms = []
    reactant_info = []
    E_reactants = 0.0

    for smi in reactant_smiles:
        atoms = _parse_smiles_atoms(smi)
        all_atoms.extend(atoms)
        D_at = _compute_D_at(smi)
        E_reactants += D_at
        reactant_info.append({
            'smiles': smi,
            'formula': _formula_from_zlist(atoms),
            'D_at': D_at,
        })

    partitions = _enumerate_partitions(all_atoms)

    evaluated = []
    reactant_sig = _atom_sig(all_atoms)

    for partition in partitions:
        E_products, prod_details = _eval_partition(partition)
        delta_G = -(E_products - E_reactants)

        lam = max((r['D_at'] for r in reactant_info), default=1.0)
        if lam > 0.1:
            Ea = (lam + delta_G) ** 2 / (4.0 * lam)
            Ea = max(Ea, lam * S3 * S_HALF)
        else:
            Ea = abs(delta_G)

        # "No reaction" = single blob OR same set of molecules as reactants
        product_sigs = tuple(sorted(partition))
        reactant_mol_sigs = tuple(sorted(
            _atom_sig(_parse_smiles_atoms(ri['smiles']))
            for ri in reactant_info
        ))
        is_no_reaction = (
            (len(partition) == 1 and partition[0] == reactant_sig)
            or product_sigs == reactant_mol_sigs
            or abs(delta_G) < 1e-6  # numerically zero = no change
        )

        evaluated.append({
            'products': prod_details,
            'E_products': E_products,
            'delta_G': delta_G,
            'Ea': Ea,
            'is_no_reaction': is_no_reaction,
        })

    evaluated.sort(key=lambda x: x['delta_G'])

    # Best exothermic reaction
    exothermic = [e for e in evaluated if not e['is_no_reaction'] and e['delta_G'] < -0.01]
    # Best non-identity reaction (may be endothermic)
    non_identity = [e for e in evaluated if not e['is_no_reaction']]

    if exothermic:
        best = exothermic[0]
    elif non_identity:
        best = non_identity[0]  # best endothermic (still useful for catalysis)
    else:
        best = evaluated[0]  # no reaction at all

    # Kinetic blocking check
    EA_BLOCK = RY * S3 * S_HALF  # ~1.49 eV
    blocked = best['Ea'] > EA_BLOCK and abs(best['delta_G']) < best['Ea']

    # Determine if reaction occurs spontaneously
    reacts = best['delta_G'] < -0.01 and not best.get('is_no_reaction', False)

    return PredictionResult(
        reacts=reacts,
        reactant_D_at=E_reactants,
        product_D_at=best['E_products'],
        delta_G=best['delta_G'],
        delta_G_kJ=best['delta_G'] * _EV_TO_KJ,
        Ea=best['Ea'],
        Ea_kJ=best['Ea'] * _EV_TO_KJ,
        kinetically_blocked=blocked,
        reactant_info=reactant_info,
        product_info=best['products'],
        alternatives=[{
            'products': e['products'],
            'delta_G': e['delta_G'],
            'delta_G_kJ': e['delta_G'] * _EV_TO_KJ,
        } for e in evaluated[:5]],
    )


def predict_reaction(reactant_smiles: List[str]) -> PredictionResult:
    """Predict reaction products from reactant SMILES.

    Automatically finds the optimal stoichiometric ratio when only
    distinct molecules are given (e.g. ["[H][H]", "O=O"] -> 2:1 -> 2 H₂O).

    For pre-expanded lists (e.g. ["[H][H]", "[H][H]", "O=O"]),
    uses the given stoichiometry directly.

    Args:
        reactant_smiles: list of SMILES (may contain duplicates for stoichiometry)

    Returns:
        PredictionResult with predicted products, stoichiometry, thermochemistry.
    """
    unique = list(dict.fromkeys(reactant_smiles))  # preserve order, dedup

    # If all entries are unique (user gave 1 of each), search stoichiometry
    if len(unique) == len(reactant_smiles) and len(unique) >= 2:
        best_list = _find_best_stoichiometry(unique, max_coeff=4)
        return _predict_single(best_list)

    # Otherwise use the given list directly (user specified stoichiometry)
    return _predict_single(reactant_smiles)
