"""
topology.py — Molecular topology from SMILES or formula.

Bridge between input string and the molecular engine.
Topology is a frozen graph: atoms + bonds + derived properties
(coordination, lone pairs, vacancies, rings).

March 2026 — Persistence Theory
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

from ptc.smiles_parser import parse_smiles, is_smiles, SmilesAtom, SmilesBond


# ╔════════════════════════════════════════════════════════════════════╗
# ║  VALENCE RULES                                                   ║
# ╚════════════════════════════════════════════════════════════════════╝

# Standard valence for common elements (max bonding electrons)
_VALENCE = {
    1: 1, 2: 0,                                    # H, He
    3: 1, 4: 2, 5: 3, 6: 4, 7: 3, 8: 2, 9: 1, 10: 0,  # Li-Ne
    11: 1, 12: 2, 13: 3, 14: 4, 15: 3, 16: 2, 17: 1, 18: 0,  # Na-Ar
    19: 1, 20: 2, 26: 2, 29: 2, 30: 2,             # K, Ca, Fe, Cu, Zn
    35: 1, 53: 1,                                    # Br, I
}

# Total valence electrons
_VALENCE_ELECTRONS = {
    1: 1, 2: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7,
    14: 4, 15: 5, 16: 6, 17: 7, 35: 7, 53: 7,
}


def max_valence(Z: int) -> int:
    """Maximum number of bonds for element Z."""
    return _VALENCE.get(Z, min(Z, 4))


def valence_electrons(Z: int) -> int:
    """Number of valence electrons for element Z."""
    return _VALENCE_ELECTRONS.get(Z, max_valence(Z))


# ╔════════════════════════════════════════════════════════════════════╗
# ║  TOPOLOGY DATACLASS                                              ║
# ╚════════════════════════════════════════════════════════════════════╝

@dataclass
class Topology:
    """Molecular topology — the bridge between SMILES and engine.

    Immutable after finalize(). All properties derived from Z_list + bonds.
    """
    Z_list: List[int]                              # atomic numbers
    bonds: List[Tuple[int, int, float]]            # (i, j, bond_order)
    source: str = "smiles"

    # Derived (set by finalize)
    z_count: List[int] = field(default_factory=list)      # coordination number
    sum_bo: List[float] = field(default_factory=list)     # sum of bond orders
    lp: List[int] = field(default_factory=list)           # lone pairs
    vacancy: List[int] = field(default_factory=list)      # orbital vacancies
    charges: List[int] = field(default_factory=list)      # formal charges from SMILES
    rings: List[List[int]] = field(default_factory=list)  # ring systems

    # Vertex classification [D13 spin foam: simplex class on T³]
    vertex_class: List[str] = field(default_factory=list)
    ring_atoms: set = field(default_factory=set)    # atoms in any ring
    ring_bonds: set = field(default_factory=set)    # bonds in any ring

    # Simplicial filtration [PT: p→simplex dimension]
    # angles[center] = list of (bi_1, bi_2) bond-index pairs at vertex center
    # vertex_bonds[k] = list of bond indices touching vertex k
    angles: Dict = field(default_factory=dict)
    vertex_bonds: Dict = field(default_factory=dict)

    def finalize(self) -> "Topology":
        """Compute derived properties from Z_list and bonds."""
        n = len(self.Z_list)
        self.z_count = [0] * n
        self.sum_bo = [0.0] * n

        for i, j, bo in self.bonds:
            self.z_count[i] += 1
            self.z_count[j] += 1
            self.sum_bo[i] += bo
            self.sum_bo[j] += bo

        # Lone pairs = (valence_electrons - sum_bo) / 2
        self.lp = [0] * n
        self.vacancy = [0] * n
        for k in range(n):
            Z = self.Z_list[k]
            ve = valence_electrons(Z)
            used = round(self.sum_bo[k])
            remaining = ve - used
            self.lp[k] = max(0, remaining // 2)
            self.vacancy[k] = max(0, max_valence(Z) - used)

        # ── SIMPLICIAL FILTRATION [PT: p → simplex dimension] ──
        # vertex_bonds[k] = list of bond indices touching vertex k
        # Built FIRST so ring_bonds and vertex_class can use it.
        from collections import defaultdict
        vb: Dict[int, list] = defaultdict(list)
        for bi, (ii, jj, _) in enumerate(self.bonds):
            vb[ii].append(bi)
            vb[jj].append(bi)
        self.vertex_bonds = dict(vb)

        # Edge index for O(1) bond lookup by (i,j) pair
        edge_index: Dict[Tuple[int, int], int] = {}
        for bi, (ii, jj, _) in enumerate(self.bonds):
            edge_index[(ii, jj)] = bi
            edge_index[(jj, ii)] = bi

        # Ring detection (simple DFS)
        self.rings = _detect_rings(n, self.bonds)

        # ── RING ATOMS AND BONDS ──
        self.ring_atoms = set()
        self.ring_bonds = set()
        for ring in self.rings:
            for atom in ring:
                self.ring_atoms.add(atom)
            for idx in range(len(ring)):
                a, b = ring[idx], ring[(idx + 1) % len(ring)]
                bi = edge_index.get((a, b))
                if bi is not None:
                    self.ring_bonds.add(bi)

        # ── VERTEX CLASSIFICATION [D13 spin foam] ──
        # Each vertex is classified by its topological role:
        #   terminal:    z = 1 (leaf of the graph)
        #   star_center: z ≥ 2, ALL neighbors are terminal (pure star)
        #   chain:       z = 2, at least one non-terminal neighbor
        #   branch:      z ≥ 3, at least one non-terminal neighbor
        #   ring:        belongs to a ring (overrides chain/branch)
        self.vertex_class = ['terminal'] * n
        for k in range(n):
            z = self.z_count[k]
            if z <= 1:
                continue
            if k in self.ring_atoms:
                self.vertex_class[k] = 'ring'
                continue
            has_non_terminal_nb = False
            for bi in self.vertex_bonds.get(k, []):
                ii, jj, _ = self.bonds[bi]
                nb = jj if ii == k else ii
                if self.z_count[nb] > 1:
                    has_non_terminal_nb = True
                    break
            if not has_non_terminal_nb:
                self.vertex_class[k] = 'star_center'
            elif z == 2:
                self.vertex_class[k] = 'chain'
            else:
                self.vertex_class[k] = 'branch'

        # angles[center] = [(bi_1, bi_2), ...] all pairs of bonds at vertex
        # These are the dim-2 simplices (triangles through center)
        self.angles = {}
        for center, bis in self.vertex_bonds.items():
            if len(bis) >= 2:
                pairs = []
                for a in range(len(bis)):
                    for b in range(a + 1, len(bis)):
                        pairs.append((bis[a], bis[b]))
                self.angles[center] = pairs

        return self

    @property
    def is_diatomic(self) -> bool:
        """True if molecule has exactly 2 atoms (1-simplex: edge only)."""
        return len(self.Z_list) == 2

    @classmethod
    def from_atoms(cls, Z_list: List[int]) -> "Topology":
        """Create empty topology from atom list (for solver).
        Sorts heavy atoms by IE descending (sieve order), H last.
        """
        from ptc.atom import IE_eV
        heavy = [(Z, IE_eV(Z)) for Z in Z_list if Z > 1]
        hydrogens = [Z for Z in Z_list if Z == 1]
        heavy.sort(key=lambda x: -x[1])
        sorted_Z = [Z for Z, _ in heavy] + hydrogens
        return cls(Z_list=sorted_Z, bonds=[], source="solver")

    @property
    def n_atoms(self) -> int:
        return len(self.Z_list)

    @property
    def n_bonds(self) -> int:
        return len(self.bonds)

    @property
    def heavy_atoms(self) -> List[int]:
        """Indices of non-hydrogen atoms."""
        return [i for i, Z in enumerate(self.Z_list) if Z > 1]

    @property
    def formula(self) -> str:
        """Molecular formula in Hill notation."""
        from collections import Counter
        from ptc.data.experimental import SYMBOLS
        counts = Counter(self.Z_list)
        parts = []
        # Hill: C first, H second, then alphabetical
        for Z in [6, 1]:
            if Z in counts:
                sym = SYMBOLS[Z]
                parts.append(sym + (str(counts[Z]) if counts[Z] > 1 else ''))
                del counts[Z]
        for Z in sorted(counts, key=lambda z: SYMBOLS.get(z, '')):
            sym = SYMBOLS.get(Z, '?')
            parts.append(sym + (str(counts[Z]) if counts[Z] > 1 else ''))
        return ''.join(parts)


def _detect_rings(n: int, bonds: List[Tuple[int, int, float]]) -> List[List[int]]:
    """Ring detection via DFS back-edges (iterative to avoid stack overflow)."""
    adj = [[] for _ in range(n)]
    for i, j, bo in bonds:
        adj[i].append(j)
        adj[j].append(i)

    visited = [False] * n
    parent = [-1] * n
    rings = []

    for start in range(n):
        if visited[start]:
            continue
        # Iterative DFS
        stack = [(start, -1)]
        while stack:
            u, p = stack.pop()
            if visited[u]:
                continue
            visited[u] = True
            parent[u] = p
            for v in adj[u]:
                if v == p:
                    # Skip parent edge (only once for multi-edges)
                    p = -2  # consume the parent skip
                    continue
                if visited[v]:
                    # Back-edge: trace ring via parent chain
                    ring = [v]
                    cur = u
                    steps = 0
                    while cur != v and cur >= 0 and steps < n:
                        ring.append(cur)
                        cur = parent[cur]
                        steps += 1
                    if cur == v and len(ring) >= 3:
                        rings.append(ring)
                else:
                    stack.append((v, u))
    return rings


# ╔════════════════════════════════════════════════════════════════════╗
# ║  TOPOLOGY BUILDERS                                               ║
# ╚════════════════════════════════════════════════════════════════════╝

def topology_from_smiles(smiles: str) -> Topology:
    """Build Topology from SMILES string."""
    atoms, bonds = parse_smiles(smiles)
    Z_list = [a.Z for a in atoms]
    bond_list = [(b.idx_a, b.idx_b, b.order) for b in bonds]
    charges = [getattr(a, 'charge', 0) for a in atoms]
    t = Topology(Z_list=Z_list, bonds=bond_list, source="smiles")
    t.charges = charges
    return t.finalize()


def topology_from_formula(formula: str) -> Topology:
    """Build Topology from formula via PT variational sieve solver.

    The solver enumerates backbone topologies and selects the isomer
    with maximum D_at (persistence principle). No SMILES needed.
    Falls back to the SMILES parser's valence heuristic if the solver
    is unavailable or fails.
    """
    try:
        from ptc.topology_solver import solve_topology
        return solve_topology(formula)
    except Exception:
        # Fallback: SMILES parser's formula mode
        atoms, bonds = parse_smiles(formula)
        Z_list = [a.Z for a in atoms]
        bond_list = [(b.idx_a, b.idx_b, b.order) for b in bonds]
        return Topology(Z_list=Z_list, bonds=bond_list, source="formula").finalize()


_SUBSCRIPT_MAP = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')

import re as _re

# Zero/O confusion: "H20" → "H2O", "N20" → "N2O", "C2H60" → "C2H6O"
# Only replaces 0 that follows a digit AND is at end-of-string or before another digit.
# Does NOT touch "C10H22" (0 followed by H, not digit/end).
_ZERO_O_RE = _re.compile(r'(?<=[0-9])0(?=[0-9]|$)')


def build_topology(input_str: str) -> Topology:
    """Dispatch: SMILES or formula → Topology.

    Detects input type automatically:
      - SMILES: 'CC(=O)O', 'c1ccccc1', '[H]F'  → SMILES parser
      - Formula: 'H2O', 'CH4', 'C2H6O'         → PT variational solver (default)

    Auto-corrections:
      - Unicode subscripts: C₂₁H₃₀O₂ → C21H30O2
      - Zero/O typo: H20 → H2O, N20 → N2O
    """
    input_str = input_str.strip().translate(_SUBSCRIPT_MAP)
    # Fix zero/O confusion in formulas (not SMILES)
    if not is_smiles(input_str):
        input_str = _ZERO_O_RE.sub('O', input_str)
    if is_smiles(input_str):
        return topology_from_smiles(input_str)
    return topology_from_formula(input_str)
