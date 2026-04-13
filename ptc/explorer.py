"""
ptc/explorer.py — Chemical space explorer.

Enumerate ALL molecular formulas satisfying user constraints,
solve the best topology for each, rank by target property.

PTC computes in milliseconds -> can test thousands of combinations.

Usage:
    from ptc.explorer import explore

    results = explore(
        elements=['C', 'H', 'O'],
        max_atoms=6,
        target='max_D_at',
    )
    for r in results[:10]:
        print(r['formula'], r['D_at'], r['smiles'])

April 2026 -- Theorie de la Persistance
"""
from __future__ import annotations

import time
from itertools import product
from typing import Dict, List, Optional

from ptc.data.experimental import SYMBOLS

# Reverse lookup
_SYM_TO_Z = {sym: z for z, sym in SYMBOLS.items()}


def _generate_formulas(elements: List[str], max_atoms: int,
                       min_atoms: int = 2) -> List[Dict[str, int]]:
    """Generate all molecular formulas with given elements and atom count.

    Returns list of dicts like {'C': 2, 'H': 6, 'O': 1}.
    """
    n_el = len(elements)
    formulas = []

    # Generate all combinations of atom counts
    for total in range(min_atoms, max_atoms + 1):
        # Enumerate partitions of 'total' atoms into n_el bins (at least 1 each)
        def _partition(remaining, idx, current):
            if idx == n_el - 1:
                current[elements[idx]] = remaining
                if remaining >= 0:
                    formulas.append(dict(current))
                return
            for count in range(0, remaining + 1):
                current[elements[idx]] = count
                _partition(remaining - count, idx + 1, current)

        _partition(total, 0, {})

    # Filter: must have at least one heavy atom (not just H)
    valid = []
    for f in formulas:
        n_heavy = sum(v for k, v in f.items() if k != 'H')
        n_total = sum(f.values())
        if n_heavy >= 1 and n_total >= min_atoms:
            # Remove zero-count elements
            clean = {k: v for k, v in f.items() if v > 0}
            if clean:
                valid.append(clean)
    return valid


def _formula_string(comp: Dict[str, int]) -> str:
    """Convert composition dict to formula string (Hill notation)."""
    parts = []
    # Hill: C first, H second, then alphabetical
    order = []
    if 'C' in comp:
        order.append('C')
    if 'H' in comp:
        order.append('H')
    for sym in sorted(comp.keys()):
        if sym not in order:
            order.append(sym)
    for sym in order:
        n = comp[sym]
        if n == 1:
            parts.append(sym)
        elif n > 1:
            parts.append(f"{sym}{n}")
    return ''.join(parts)


def explore(elements: List[str],
            max_atoms: int = 6,
            min_atoms: int = 2,
            target: str = 'max_D_at',
            max_formulas: int = 200,
            timeout_s: float = 30.0) -> List[dict]:
    """Explore chemical space: enumerate formulas, solve topologies, rank.

    Parameters
    ----------
    elements : list of str
        Element symbols (e.g. ['C', 'H', 'O'])
    max_atoms : int
        Maximum number of atoms
    min_atoms : int
        Minimum number of atoms
    target : str
        'max_D_at', 'min_D_at', or 'max_D_at_per_atom'
    max_formulas : int
        Maximum number of formulas to test
    timeout_s : float
        Timeout in seconds

    Returns
    -------
    List of dicts with keys: formula, D_at, D_at_per_atom, n_atoms,
                              topology, smiles, time_ms
    """
    from ptc.topology_solver import solve_topology_all, parse_formula
    from ptc.molecule import compute_D_at

    # Validate elements
    for sym in elements:
        if sym not in _SYM_TO_Z:
            raise ValueError(f"Unknown element: {sym}")

    # Generate formulas
    formulas = _generate_formulas(elements, max_atoms, min_atoms)
    if len(formulas) > max_formulas:
        formulas = formulas[:max_formulas]

    results = []
    t_start = time.time()

    for comp in formulas:
        if time.time() - t_start > timeout_s:
            break

        formula_str = _formula_string(comp)
        t0 = time.time()

        try:
            isomers = solve_topology_all(formula_str)
            if not isomers:
                continue
            best = isomers[0]
            res = compute_D_at(best)
            dt = (time.time() - t0) * 1000

            n_atoms = sum(comp.values())
            results.append({
                'formula': formula_str,
                'D_at': res.D_at,
                'D_at_per_atom': res.D_at / max(n_atoms, 1),
                'n_atoms': n_atoms,
                'n_bonds': len(best.bonds),
                'n_isomers': len(isomers),
                'topology': best,
                'time_ms': dt,
            })
        except Exception:
            continue

    # Sort by target
    if target == 'max_D_at':
        results.sort(key=lambda r: -r['D_at'])
    elif target == 'min_D_at':
        results.sort(key=lambda r: r['D_at'])
    elif target == 'max_D_at_per_atom':
        results.sort(key=lambda r: -r['D_at_per_atom'])

    return results
