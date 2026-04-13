"""
PTCS v3.0 — Parser SMILES minimal → MolGraph.

Features:
  - Atomes organiques implicites : C, N, O, S, P, B
  - Atomes entre crochets : [Si], [Fe], [Na], charges [NH4+]
  - Liaisons simples, doubles (=), triples (#)
  - Branches: CC(C)C (isobutane), CC(=O)O (acetic acid)
  - Rings: C1CC1 (cyclopropane), c1ccccc1 (benzene)
  - Aromatiques minuscules : c, n, o, s → bo=1.5 dans les cycles

Falls back to _infer_bonds if no SMILES detected.

All from s = 1/2. 0 adjustable parameters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ============================================================
# SMILES SYMBOL TABLE
# ============================================================

# Organic subset (implicit H allowed)
_ORGANIC = {'B', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I'}

# Aromatic organic subset
_AROMATIC_ORGANIC = {'b', 'c', 'n', 'o', 'p', 's'}

# Symbol → Z lookup (common elements)
_Z_FROM_SYM = {}
_SYM = ["", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
        "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
        "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
        "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
        "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
        "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
        "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
        "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po",
        "At", "Rn"]
for _z, _s in enumerate(_SYM):
    if _s:
        _Z_FROM_SYM[_s] = _z


# ============================================================
# SMILES ATOM
# ============================================================

@dataclass
class SmilesAtom:
    """Atom parsed from SMILES."""
    symbol: str
    Z: int
    aromatic: bool = False
    charge: int = 0
    hcount: int = -1   # -1 = implicit, ≥0 = explicit
    idx: int = 0       # index in atom list


# ============================================================
# SMILES BOND
# ============================================================

@dataclass
class SmilesBond:
    """Bond parsed from SMILES."""
    idx_a: int
    idx_b: int
    order: float   # 1, 2, 3, or 1.5 (aromatic)


# ============================================================
# SMILES PARSER
# ============================================================

def is_smiles(s: str) -> bool:
    """Heuristic: is this SMILES (vs a molecular formula)?

    SMILES si contient : =, #, crochets, minuscules aromatic,
    ou pattern de branchement SMILES.

    Molecular formula if: only [A-Z][a-z]?[0-9]* repeated,
    with AT LEAST one digit serving as element counter,
    or grouping parentheses (CH3)3N.

    Digit-free strings of organic atoms (C, CC, CO, CCl...)
    are treated as SMILES.
    """
    # Unambiguous SMILES characters
    if any(c in s for c in '=#[]'):
        return True

    # Aromatic lowercase atoms
    if re.search(r'[cnops]', s):
        return True

    # SMILES branching: letter followed by ( followed by letter or bond
    # e.g., CC(C)C, CC(=O)O
    if re.search(r'[A-Za-z]\([A-Za-z=#]', s):
        return True

    # Formula check: must match formula pattern AND have at least one
    # element count >= 2 (e.g., H2O, CH4, C2H6).
    # Without such counts, strings like C, CC, CO, CCl are SMILES.
    # Ring closures (C1CC1) use digit 1 which never appears as a
    # meaningful element count in standard formulas.
    _FORMULA_RE = re.compile(
        r'(?:\([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*\)\d+|[A-Z][a-z]?\d*)+')
    if re.fullmatch(_FORMULA_RE, s):
        # Must contain at least one element count >= 2
        if re.search(r'[A-Z][a-z]?[2-9]\d*(?!\d)', s):
            return False

    # Default: treat as SMILES
    return True


def parse_smiles(smiles: str) -> tuple[list[SmilesAtom], list[SmilesBond]]:
    """Parse SMILES string into atoms and bonds.

    Returns (atoms, bonds).
    """
    atoms: list[SmilesAtom] = []
    bonds: list[SmilesBond] = []
    ring_opens: dict[int, tuple[int, float]] = {}  # ring_num → (atom_idx, bond_order)
    branch_stack: list[int] = []  # stack of atom indices
    prev_atom: int = -1
    next_bond_order: float = 1.0

    i = 0
    n = len(smiles)

    while i < n:
        c = smiles[i]

        # ── Bond order specifiers ──
        if c == '=':
            next_bond_order = 2.0
            i += 1
            continue
        elif c == '#':
            next_bond_order = 3.0
            i += 1
            continue
        elif c == '-':
            next_bond_order = 1.0
            i += 1
            continue
        elif c == ':':
            next_bond_order = 1.5
            i += 1
            continue

        # ── Branch open ──
        elif c == '(':
            branch_stack.append(prev_atom)
            i += 1
            continue

        # ── Branch close ──
        elif c == ')':
            if branch_stack:
                prev_atom = branch_stack.pop()
            i += 1
            continue

        # ── Ring closure digit ──
        elif c == '%':
            # Two-digit ring number: %12
            if i + 2 < n and smiles[i + 1].isdigit() and smiles[i + 2].isdigit():
                ring_num = int(smiles[i + 1:i + 3])
                i += 3
            else:
                i += 1
                continue
            _handle_ring(ring_num, prev_atom, next_bond_order,
                         ring_opens, bonds, atoms)
            next_bond_order = 1.0
            continue

        elif c.isdigit():
            ring_num = int(c)
            i += 1
            _handle_ring(ring_num, prev_atom, next_bond_order,
                         ring_opens, bonds, atoms)
            next_bond_order = 1.0
            continue

        # ── Bracket atom [Fe], [NH4+], [Si] ──
        elif c == '[':
            j = smiles.index(']', i)
            bracket = smiles[i + 1:j]
            sa = _parse_bracket_atom(bracket)
            sa.idx = len(atoms)
            atoms.append(sa)
            if prev_atom >= 0:
                bo = next_bond_order
                if atoms[prev_atom].aromatic and sa.aromatic:
                    bo = 1.5
                bonds.append(SmilesBond(prev_atom, sa.idx, bo))
            prev_atom = sa.idx
            next_bond_order = 1.0
            i = j + 1
            continue

        # ── Aromatic atom ──
        elif c in _AROMATIC_ORGANIC:
            sym = c.upper()
            Z = _Z_FROM_SYM.get(sym, 0)
            sa = SmilesAtom(symbol=sym, Z=Z, aromatic=True, idx=len(atoms))
            atoms.append(sa)
            if prev_atom >= 0:
                bo = 1.5 if atoms[prev_atom].aromatic else next_bond_order
                bonds.append(SmilesBond(prev_atom, sa.idx, bo))
            prev_atom = sa.idx
            next_bond_order = 1.0
            i += 1
            continue

        # ── Organic atom (uppercase) ──
        elif c.isupper():
            # Two-letter symbols: Cl, Br, Si, etc.
            sym = c
            if i + 1 < n and smiles[i + 1].islower() and smiles[i + 1] not in 'cnops':
                sym += smiles[i + 1]
                i += 1
            Z = _Z_FROM_SYM.get(sym, 0)
            if Z == 0:
                # Try single letter
                sym = c
                Z = _Z_FROM_SYM.get(sym, 0)
            sa = SmilesAtom(symbol=sym, Z=Z, idx=len(atoms))
            atoms.append(sa)
            if prev_atom >= 0:
                bonds.append(SmilesBond(prev_atom, sa.idx, next_bond_order))
            prev_atom = sa.idx
            next_bond_order = 1.0
            i += 1
            continue

        # ── Whitespace or unknown → skip ──
        else:
            i += 1
            continue

    # ── Add implicit hydrogens ──
    _add_implicit_hydrogens(atoms, bonds)

    return atoms, bonds


def _handle_ring(ring_num: int, prev_atom: int, bond_order: float,
                 ring_opens: dict, bonds: list, atoms: list):
    """Handle ring open/close."""
    if ring_num in ring_opens:
        # Close ring
        open_atom, open_bo = ring_opens.pop(ring_num)
        bo = max(bond_order, open_bo)
        # Aromatic ring closure
        if atoms[open_atom].aromatic and atoms[prev_atom].aromatic:
            bo = 1.5
        bonds.append(SmilesBond(open_atom, prev_atom, bo))
    else:
        # Open ring
        ring_opens[ring_num] = (prev_atom, bond_order)


def _parse_bracket_atom(bracket: str) -> SmilesAtom:
    """Parse bracket content: Fe, NH4+, Si, etc."""
    aromatic = False
    charge = 0
    hcount = 0

    # Check for aromatic lowercase
    if bracket and bracket[0].islower():
        aromatic = True
        bracket = bracket[0].upper() + bracket[1:]

    # Extract symbol (1 or 2 letters)
    m = re.match(r'^([A-Z][a-z]?)', bracket)
    if not m:
        return SmilesAtom(symbol='C', Z=6)
    sym = m.group(1)
    rest = bracket[len(sym):]

    Z = _Z_FROM_SYM.get(sym, 0)

    # Parse H count
    hm = re.search(r'H(\d*)', rest)
    if hm:
        hcount = int(hm.group(1)) if hm.group(1) else 1

    # Parse charge
    cm = re.search(r'(\+\+?|\-\-?|\+\d|\-\d)', rest)
    if cm:
        cs = cm.group(1)
        if cs == '+':
            charge = 1
        elif cs == '++':
            charge = 2
        elif cs == '-':
            charge = -1
        elif cs == '--':
            charge = -2
        elif cs.startswith('+'):
            charge = int(cs[1:])
        elif cs.startswith('-'):
            charge = -int(cs[1:])

    return SmilesAtom(symbol=sym, Z=Z, aromatic=aromatic,
                      charge=charge, hcount=hcount)


def _add_implicit_hydrogens(atoms: list[SmilesAtom], bonds: list[SmilesBond]):
    """Add implicit hydrogens based on valence rules.

    Organic subset default valences:
      B: 3, C: 4, N: 3 or 5, O: 2, P: 3 or 5, S: 2 or 4 or 6
      F: 1, Cl: 1, Br: 1, I: 1
    """
    DEFAULT_VALENCE = {
        'B': [3], 'C': [4], 'N': [3, 5], 'O': [2],
        'P': [3, 5], 'S': [2, 4, 6],
        'F': [1], 'Cl': [1], 'Br': [1], 'I': [1],
    }

    # Compute current bond orders per atom
    atom_bo = [0.0] * len(atoms)
    for b in bonds:
        atom_bo[b.idx_a] += b.order
        atom_bo[b.idx_b] += b.order

    # Compute degree (number of bonds) per atom for aromatic H count
    atom_degree = [0] * len(atoms)
    for b in bonds:
        atom_degree[b.idx_a] += 1
        atom_degree[b.idx_b] += 1

    for i, sa in enumerate(atoms):
        if sa.hcount >= 0:
            # Explicit H count from bracket
            n_h = sa.hcount
        elif sa.symbol in DEFAULT_VALENCE:
            # Implicit H from valence
            # For aromatic atoms: use degree (not bo sum) + 1 for pi electron
            if sa.aromatic:
                current_bo = atom_degree[i] + 1  # degree + 1 pi electron
            else:
                current_bo = atom_bo[i]
            valences = DEFAULT_VALENCE[sa.symbol]
            # Find smallest valid valence
            n_h = 0
            for v in valences:
                diff = v - int(round(current_bo))
                if diff >= 0:
                    n_h = diff
                    break
        else:
            # Unknown element or bracket atom without H spec → no implicit H
            continue

        # Add H atoms
        for _ in range(n_h):
            h_idx = len(atoms)
            atoms.append(SmilesAtom(symbol='H', Z=1, idx=h_idx))
            bonds.append(SmilesBond(i, h_idx, 1.0))
            atom_bo[i] += 1.0
            atom_bo.append(1.0)
