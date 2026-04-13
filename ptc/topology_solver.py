"""
topology_solver.py — Variational sieve topology solver.

Given a molecular formula, finds ALL viable bond topologies (isomers)
ranked by D_at using multi-backbone enumeration + pipeline:
  1. BACKBONE: heuristic heavy-atom connectivity (chain/star variants)
  2. RING CLOSURE: close cycles ≥ 3 atoms
  3. AROMATIC: detect conjugated 5/6-rings → bo=1.5 [T6 holonomy]
  4. FORMAL CHARGES: detect dative bonds (CO) → [C-]#[O+]
  5. STRENGTHEN: increase bond orders where D_at improves
  6. H FILL: attach H to heavy atoms
  7. LOCAL OPTIMIZE: bo change + bond swaps

Multi-isomer: permute backbone connectivity, collect distinct
topologies ranked by D_at. The crible tests ALL configurations.

March 2026 — Persistence Theory
"""
from __future__ import annotations

import math
import re
from itertools import permutations
from typing import Dict, List, Optional, Tuple

from ptc.data.experimental import SYMBOLS
from ptc.topology import Topology, max_valence

# Reverse lookup: symbol → Z
_SYMBOL_TO_Z = {sym: z for z, sym in SYMBOLS.items()}

# Expanded valence for solver (period 3+ d-orbital expansion)
_SOLVER_MAX_BO = {
    1: 1, 2: 0,
    3: 1, 4: 2, 5: 3, 6: 4, 7: 4, 8: 2, 9: 1, 10: 0,  # Li-Ne (O=2: std valence)
    11: 1, 12: 2, 13: 3, 14: 6, 15: 5, 16: 6, 17: 1, 18: 0,  # Na-Ar
    19: 1, 20: 2, 26: 6, 29: 4, 30: 2,
    35: 1, 53: 1,
}

# sp2-compatible elements for aromaticity
_SP2_AROMATIC = {5, 6, 7, 8, 16}  # B, C, N, O, S


def _solver_max_valence(Z: int) -> int:
    return _SOLVER_MAX_BO.get(Z, max_valence(Z))


def parse_formula(formula: str) -> List[int]:
    """Parse molecular formula to list of atomic numbers."""
    tokens = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    Z_list = []
    for symbol, count_str in tokens:
        if not symbol:
            continue
        Z = _SYMBOL_TO_Z.get(symbol)
        if Z is None:
            raise ValueError(f"Unknown element: {symbol}")
        count = int(count_str) if count_str else 1
        Z_list.extend([Z] * count)
    return Z_list


# ╔════════════════════════════════════════════════════════════════════╗
# ║  INTERNAL HELPERS                                                 ║
# ╚════════════════════════════════════════════════════════════════════╝

def _sum_bo_at(bonds, idx):
    return sum(bo for i, j, bo in bonds if i == idx or j == idx)

def _remaining_valence(Z_list, bonds, idx):
    return max(0, _solver_max_valence(Z_list[idx]) - int(round(_sum_bo_at(bonds, idx))))

def _find_bond(bonds, i, j):
    for k, (a, b, bo) in enumerate(bonds):
        if (a == i and b == j) or (a == j and b == i):
            return k, bo
    return None, 0.0

def _evaluate_d_at(Z_list, bonds):
    if not bonds:
        return 0.0
    from ptc.topology import Topology
    from ptc.molecule import compute_D_at
    topo = Topology(Z_list=list(Z_list), bonds=list(bonds), source="solver").finalize()
    try:
        return compute_D_at(topo).D_at
    except Exception:
        return -1e10

def _effective_bo_used(bonds, idx):
    """Bond-order used, treating aromatic bo=1.5 as 1.0 for valence capacity.
    Aromatic bonds occupy 1 σ-slot each (the 0.5 π is delocalized)."""
    return sum(1.0 if bo == 1.5 else bo for i, j, bo in bonds if i == idx or j == idx)

def _quick_h_fill(Z_sorted, backbone, heavy_idx, h_idx):
    """Tentative H fill onto heavy atoms with most remaining standard valence."""
    bonds = list(backbone)
    remaining = {}
    for hv in heavy_idx:
        remaining[hv] = max_valence(Z_sorted[hv]) - int(round(_effective_bo_used(bonds, hv)))
    for h in h_idx:
        best_hv = None
        best_rem = 0
        for hv in heavy_idx:
            r = remaining.get(hv, 0)
            if r > best_rem:
                best_rem = r
                best_hv = hv
        if best_hv is not None and best_rem > 0:
            bonds.append((min(h, best_hv), max(h, best_hv), 1.0))
            remaining[best_hv] -= 1
    return bonds

def _path_length(bonds_list, a, b, n_atoms):
    """Shortest path length between a and b via BFS. -1 if disconnected."""
    adj = [[] for _ in range(n_atoms)]
    for bi, bj, _ in bonds_list:
        adj[bi].append(bj)
        adj[bj].append(bi)
    visited = {a: 0}
    queue = [a]
    while queue:
        cur = queue.pop(0)
        if cur == b:
            return visited[b]
        for nb in adj[cur]:
            if nb not in visited:
                visited[nb] = visited[cur] + 1
                queue.append(nb)
    return -1

def _find_ring_path(bonds_list, start, end, n_atoms):
    """Find ring path from start to end through existing bonds (excl. last bond)."""
    adj = [[] for _ in range(n_atoms)]
    for bi, bj, _ in bonds_list[:-1]:
        adj[bi].append(bj)
        adj[bj].append(bi)
    parent = {start: None}
    queue = [start]
    while queue:
        cur = queue.pop(0)
        if cur == end:
            path = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]
        for nb in adj[cur]:
            if nb not in parent:
                parent[nb] = cur
                queue.append(nb)
    return []

def _bond_signature(Z_list, bonds):
    """Connectivity-aware signature for deduplication.
    Encodes each atom's neighbor set (Z, bo) — distinguishes
    1,1-DFE (F₂C=CH₂) from 1,2-DFE (FHC=CHF)."""
    adj = {}
    for i, j, bo in bonds:
        adj.setdefault(i, []).append((Z_list[j], bo))
        adj.setdefault(j, []).append((Z_list[i], bo))
    atom_sigs = []
    for k in range(len(Z_list)):
        neighbors = tuple(sorted(adj.get(k, [])))
        atom_sigs.append((Z_list[k], neighbors))
    return tuple(sorted(atom_sigs))


# ╔════════════════════════════════════════════════════════════════════╗
# ║  BACKBONE GENERATORS                                              ║
# ╚════════════════════════════════════════════════════════════════════╝

def _build_chain_backbone(Z_sorted, heavy_idx, chain_order):
    """Build chain backbone with specific atom ordering."""
    bonds = []
    if len(chain_order) <= 1:
        return bonds
    terminals = [i for i in chain_order if _solver_max_valence(Z_sorted[i]) <= 1]
    non_terminals = [i for i in chain_order if _solver_max_valence(Z_sorted[i]) > 1]

    if not non_terminals:
        if len(terminals) >= 2:
            bonds.append((terminals[0], terminals[1], 1.0))
        return bonds

    # Star detection: solver_max for CAPACITY, standard mv for PRIORITY
    # SF₆: S solver_max=6, std=2. Star because sole non-terminal.
    # Thiophene: S solver_max=6, std=2 < C std=4. NOT star (C has higher std).
    mvs = [(i, _solver_max_valence(Z_sorted[i])) for i in non_terminals]
    mvs.sort(key=lambda x: -x[1])
    top_mv = mvs[0][1]
    top_std = max(max_valence(Z_sorted[i]) for i in non_terminals)
    # Star center: highest solver_max AND highest standard valence
    centers = [i for i, mv in mvs
               if mv == top_mv and max_valence(Z_sorted[i]) >= top_std]
    others_nt = [i for i in non_terminals if i not in centers]

    if len(centers) == 1 and top_mv >= len(others_nt) + len(terminals) and top_mv > 2:
        center = centers[0]
        for other in others_nt + terminals:
            rem = _solver_max_valence(Z_sorted[center]) - int(round(_sum_bo_at(bonds, center)))
            if rem > 0:
                bonds.append((min(center, other), max(center, other), 1.0))
    else:
        # Chain: connect non-terminals in the given ORDER
        for k in range(1, len(non_terminals)):
            i, j = non_terminals[k - 1], non_terminals[k]
            rem_i = _solver_max_valence(Z_sorted[i]) - int(round(_sum_bo_at(bonds, i)))
            rem_j = _solver_max_valence(Z_sorted[j]) - int(round(_sum_bo_at(bonds, j)))
            if rem_i > 0 and rem_j > 0:
                bonds.append((min(i, j), max(i, j), 1.0))
        # Attach terminals
        for t in terminals:
            best_nt, best_rem = None, 0
            for nt in non_terminals:
                rem = _solver_max_valence(Z_sorted[nt]) - int(round(_sum_bo_at(bonds, nt)))
                if rem > best_rem:
                    best_rem = rem
                    best_nt = nt
            if best_nt is not None and best_rem > 0:
                bonds.append((min(t, best_nt), max(t, best_nt), 1.0))

    return bonds


def _generate_backbones(Z_sorted, heavy_idx):
    """Generate distinct backbone variants by permuting chain order."""
    if len(heavy_idx) <= 1:
        return [_build_chain_backbone(Z_sorted, heavy_idx, heavy_idx)]

    non_terminals = [i for i in heavy_idx if _solver_max_valence(Z_sorted[i]) > 1]
    terminals = [i for i in heavy_idx if _solver_max_valence(Z_sorted[i]) <= 1]

    if len(non_terminals) <= 1:
        return [_build_chain_backbone(Z_sorted, heavy_idx, heavy_idx)]

    # Generate unique permutations of non-terminal atoms
    # (terminals always attach to the nearest center, their order doesn't matter)
    n_nt = len(non_terminals)

    # For small n: try all permutations. For large n: sample.
    if n_nt <= 6:
        # All permutations (max 720)
        perms = set(permutations(non_terminals))
    else:
        # Sample: default + reversed + each atom as first + random swaps
        perms = set()
        perms.add(tuple(non_terminals))
        perms.add(tuple(reversed(non_terminals)))
        for i in range(n_nt):
            rotated = non_terminals[i:] + non_terminals[:i]
            perms.add(tuple(rotated))

    backbones = []
    seen_sigs = set()
    for perm in perms:
        chain_order = list(perm) + terminals
        bb = _build_chain_backbone(Z_sorted, heavy_idx, chain_order)
        sig = _bond_signature(Z_sorted, bb)
        if sig not in seen_sigs:
            seen_sigs.add(sig)
            backbones.append(bb)

    # Also try CLUSTERED TERMINALS: same-element terminals on one atom
    # (covers 1,1-DFE: both F on same C instead of one each)
    if len(terminals) >= 2:
        from collections import Counter
        term_elements = Counter(Z_sorted[t] for t in terminals)
        for Z_term, count in term_elements.items():
            if count < 2:
                continue
            same_terms = [t for t in terminals if Z_sorted[t] == Z_term]
            for center_nt in non_terminals:
                cap = max_valence(Z_sorted[center_nt])
                if cap < count + 1:  # need room for cluster + chain bond
                    continue
                # Build: chain of non-terminals, cluster terminals on center_nt
                cluster_bonds = list(bb_default) if 'bb_default' in dir() else []
                if not cluster_bonds:
                    # Build chain first
                    sorted_nt_cl = sorted(non_terminals, key=lambda x: -max_valence(Z_sorted[x]))
                    for kk in range(1, len(sorted_nt_cl)):
                        ii, jj = sorted_nt_cl[kk-1], sorted_nt_cl[kk]
                        cluster_bonds.append((min(ii,jj), max(ii,jj), 1.0))
                # Remove default terminal attachments, re-attach all same_terms to center
                cluster_bonds_clean = [(a,b,bo) for a,b,bo in cluster_bonds
                                       if a not in same_terms and b not in same_terms]
                for t in same_terms:
                    cluster_bonds_clean.append((min(t, center_nt), max(t, center_nt), 1.0))
                # Attach other terminals normally
                for t in terminals:
                    if t in same_terms:
                        continue
                    best_nt2, best_rem2 = None, 0
                    for nt in non_terminals:
                        rem = _solver_max_valence(Z_sorted[nt]) - int(round(_sum_bo_at(cluster_bonds_clean, nt)))
                        if rem > best_rem2:
                            best_rem2 = rem
                            best_nt2 = nt
                    if best_nt2 is not None and best_rem2 > 0:
                        cluster_bonds_clean.append((min(t, best_nt2), max(t, best_nt2), 1.0))
                sig = _bond_signature(Z_sorted, cluster_bonds_clean)
                if sig not in seen_sigs:
                    seen_sigs.add(sig)
                    backbones.append(cluster_bonds_clean)

    # Also try STAR at each non-terminal (covers nitromethane N-hub)
    # Each atom tries being the hub with all others attached directly.
    for center in non_terminals:
        cap = _solver_max_valence(Z_sorted[center])
        others = [i for i in heavy_idx if i != center]
        if cap < len(others):
            continue  # can't hold all others
        star_bonds = []
        for other in others:
            rem = cap - int(round(_sum_bo_at(star_bonds, center)))
            if rem > 0:
                star_bonds.append((min(center, other), max(center, other), 1.0))
        if star_bonds:
            sig = _bond_signature(Z_sorted, star_bonds)
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                backbones.append(star_bonds)

    return backbones if backbones else [_build_chain_backbone(Z_sorted, heavy_idx, heavy_idx)]


# ╔════════════════════════════════════════════════════════════════════╗
# ║  FULL PIPELINE (per backbone)                                     ║
# ╚════════════════════════════════════════════════════════════════════╝

def _run_pipeline(Z_sorted, bonds, heavy_idx, h_idx, allow_small_rings=True):
    """Run ring closure → aromatic → strengthen → H fill → optimize."""
    n = len(Z_sorted)

    # ── RING CLOSURE ──
    min_ring = 3 if allow_small_rings else 5
    best_ring = None
    best_ring_d_at = _evaluate_d_at(
        Z_sorted, _quick_h_fill(Z_sorted, bonds, heavy_idx, h_idx))
    for i in heavy_idx:
        if _sum_bo_at(bonds, i) >= max_valence(Z_sorted[i]):
            continue
        for j in heavy_idx:
            if j <= i:
                continue
            idx_e, _ = _find_bond(bonds, i, j)
            if idx_e is not None:
                continue
            if _sum_bo_at(bonds, j) >= max_valence(Z_sorted[j]):
                continue
            path_len = _path_length(bonds, i, j, n)
            if path_len < 0:
                continue
            ring_size = path_len + 1
            if ring_size < min_ring:
                continue
            # Block same-element ring closure for small rings
            # (prevents O-O in NO₂ triangle, N-N parasitic rings)
            if ring_size < 5 and Z_sorted[i] == Z_sorted[j]:
                continue
            candidate = bonds + [(i, j, 1.0)]
            filled = _quick_h_fill(Z_sorted, candidate, heavy_idx, h_idx)
            n_h_ok = sum(1 for bi, bj, _ in filled
                         if Z_sorted[bi] == 1 or Z_sorted[bj] == 1)
            if n_h_ok < len(h_idx):
                continue
            d_at = _evaluate_d_at(Z_sorted, filled)
            if d_at > best_ring_d_at + 1e-6:
                best_ring_d_at = d_at
                best_ring = candidate
    if best_ring is not None:
        bonds = best_ring

    # ── AROMATIC DETECTION [T6 holonomy on Z/NZ] ──
    if best_ring is not None:
        ring_close_i, ring_close_j = best_ring[-1][0], best_ring[-1][1]
        ring_path = _find_ring_path(bonds, ring_close_i, ring_close_j, n)
        ring_size = len(ring_path)

        if ring_size in (5, 6):
            ring_set = set(ring_path)
            all_sp2 = all(Z_sorted[a] in _SP2_AROMATIC for a in ring_path)
            if all_sp2:
                # Hückel 4n+2 check [T6 holonomy: aromaticity requires
                # 4n+2 π electrons in the conjugated ring system]
                # C: 1 π electron (p-orbital)
                # N in 5-ring (pyrrole-type): 2 π (LP donated to ring)
                # N in 6-ring (pyridine-type): 1 π (LP in plane)
                # O, S: 2 π (LP donated to ring)
                pi_count = 0
                for a in ring_path:
                    Za = Z_sorted[a]
                    if Za == 6:  # C
                        pi_count += 1
                    elif Za == 7:  # N
                        pi_count += 2 if ring_size == 5 else 1
                    elif Za in (8, 16):  # O, S
                        pi_count += 2
                    else:
                        pi_count += 1
                if pi_count % 4 != 2:  # not 4n+2
                    all_sp2 = False  # block aromatization

            if all_sp2:
                can_aro = True
                for a in ring_path:
                    ring_bo = sum(bo for bi, bj, bo in bonds
                                 if (bi == a or bj == a)
                                 and bi in ring_set and bj in ring_set)
                    total_used = _sum_bo_at(bonds, a)
                    new_total = total_used - ring_bo + 2 * 1.5
                    # Tolerance +0.5 for C/N (standard), extended for
                    # heteroatoms O/S in 5-rings (LP contributes π)
                    tol = 1.5 if (ring_size == 5 and Z_sorted[a] in (8, 16)) else 0.5
                    if new_total > max_valence(Z_sorted[a]) + tol:
                        can_aro = False
                        break
                if can_aro:
                    new_bonds = []
                    for bi, bj, bo in bonds:
                        if bi in ring_set and bj in ring_set:
                            new_bonds.append((bi, bj, 1.5))
                        else:
                            new_bonds.append((bi, bj, bo))
                    filled_aro = _quick_h_fill(Z_sorted, new_bonds, heavy_idx, h_idx)
                    n_h_aro = sum(1 for bi, bj, _ in filled_aro
                                 if Z_sorted[bi] == 1 or Z_sorted[bj] == 1)
                    if n_h_aro >= len(h_idx):
                        d_aro = _evaluate_d_at(Z_sorted, filled_aro)
                        d_cur = _evaluate_d_at(
                            Z_sorted, _quick_h_fill(Z_sorted, bonds, heavy_idx, h_idx))
                        if d_aro > d_cur - 0.5:
                            bonds = new_bonds

    # ── STRENGTHEN ──
    changed = True
    while changed:
        changed = False
        best = None
        best_d_at = _evaluate_d_at(
            Z_sorted, _quick_h_fill(Z_sorted, bonds, heavy_idx, h_idx))
        for k, (i, j, bo) in enumerate(bonds):
            if bo >= 3.0 or bo == 1.5:  # don't strengthen aromatic bonds
                continue
            if _remaining_valence(Z_sorted, bonds, i) <= 0:
                continue
            if _remaining_valence(Z_sorted, bonds, j) <= 0:
                continue
            if Z_sorted[i] == Z_sorted[j]:
                if _sum_bo_at(bonds, i) >= max_valence(Z_sorted[i]):
                    continue
            candidate = list(bonds)
            candidate[k] = (i, j, bo + 1.0)
            filled = _quick_h_fill(Z_sorted, candidate, heavy_idx, h_idx)
            n_h_b = sum(1 for bi, bj, _ in filled
                        if Z_sorted[bi] == 1 or Z_sorted[bj] == 1)
            if n_h_b < len(h_idx):
                continue
            d_at = _evaluate_d_at(Z_sorted, filled)
            if d_at > best_d_at + 1e-6:
                best_d_at = d_at
                best = candidate
        if best is not None:
            bonds = best
            changed = True

    # ── DATIVE STRENGTHEN (H-free heteronuclear diatomics) ──
    # Gate: even electrons only (odd = radical, no dative triple)
    total_e = sum(Z_sorted[i] for i in heavy_idx)
    if not h_idx and len(heavy_idx) == 2 and total_e % 2 == 0:
        changed = True
        while changed:
            changed = False
            best = None
            best_d_at = _evaluate_d_at(Z_sorted, bonds)
            for k, (i, j, bo) in enumerate(bonds):
                if bo >= 3.0:
                    continue
                if Z_sorted[i] == Z_sorted[j]:
                    continue
                rem_i = _remaining_valence(Z_sorted, bonds, i)
                rem_j = _remaining_valence(Z_sorted, bonds, j)
                if rem_i <= 0 and rem_j <= 0:
                    continue
                candidate = list(bonds)
                candidate[k] = (i, j, bo + 1.0)
                d_at = _evaluate_d_at(Z_sorted, candidate)
                if d_at > best_d_at + 1e-6:
                    best_d_at = d_at
                    best = candidate
            if best is not None:
                bonds = best
                changed = True

    # ── FORMAL CHARGES for dative bonds ──
    # Only for DIATOMIC dative bonds (CO, NO): when one atom exceeds
    # standard valence via dative bonding, assign formal charges.
    # NOT for polyatomic hypervalent (SF₆, PF₅): expanded octet, no charges.
    charges = [0] * n
    if len(heavy_idx) == 2 and not h_idx:
        for i, j, bo in bonds:
            std_i = max_valence(Z_sorted[i])
            std_j = max_valence(Z_sorted[j])
            total_i = _sum_bo_at(bonds, i)
            total_j = _sum_bo_at(bonds, j)
            if total_i > std_i and total_j <= std_j:
                charges[j] -= 1
                charges[i] += 1
            elif total_j > std_j and total_i <= std_i:
                charges[i] -= 1
                charges[j] += 1

    # ── H FILL ──
    for h in h_idx:
        best = None
        best_d_at = _evaluate_d_at(Z_sorted, bonds)
        for hv in heavy_idx:
            if _effective_bo_used(bonds, hv) >= max_valence(Z_sorted[hv]):
                continue
            pair = (min(h, hv), max(h, hv), 1.0)
            candidate = bonds + [pair]
            d_at = _evaluate_d_at(Z_sorted, candidate)
            if d_at > best_d_at + 1e-6:
                best_d_at = d_at
                best = candidate
        if best is not None:
            bonds = best

    return bonds, charges


def _local_optimize(Z_list, bonds):
    """Local search: bo changes + remove+strengthen. Max 2 passes."""
    current_d_at = _evaluate_d_at(Z_list, bonds)
    for _pass in range(2):
        improved = False
        for k in range(len(bonds)):
            if k >= len(bonds):
                break
            i, j, bo = bonds[k]
            if bo == 1.5:
                continue  # protect aromatic bonds
            for delta in [+1.0, -1.0]:
                new_bo = bo + delta
                if new_bo < 0.5 or new_bo > 3.0:
                    continue
                if delta > 0:
                    if _remaining_valence(Z_list, bonds, i) <= 0:
                        continue
                    if _remaining_valence(Z_list, bonds, j) <= 0:
                        continue
                    if Z_list[i] == Z_list[j]:
                        if _sum_bo_at(bonds, i) >= max_valence(Z_list[i]):
                            continue
                candidate = list(bonds)
                candidate[k] = (i, j, new_bo)
                d_at = _evaluate_d_at(Z_list, candidate)
                if d_at > current_d_at + 1e-6:
                    bonds = candidate
                    current_d_at = d_at
                    improved = True
                    break
            else:
                # Don't remove H bonds (never sacrifice hydrogen atoms)
                if Z_list[i] == 1 or Z_list[j] == 1:
                    continue
                reduced = [b for idx2, b in enumerate(bonds) if idx2 != k]
                for m, (mi, mj, mbo) in enumerate(reduced):
                    if mbo >= 3.0 or mbo == 1.5:
                        continue  # protect aromatic
                    if _remaining_valence(Z_list, reduced, mi) <= 0:
                        continue
                    if _remaining_valence(Z_list, reduced, mj) <= 0:
                        continue
                    # Same-element guard (prevents N-N over-strengthen)
                    if Z_list[mi] == Z_list[mj]:
                        if _sum_bo_at(reduced, mi) >= max_valence(Z_list[mi]):
                            continue
                    candidate = list(reduced)
                    candidate[m] = (mi, mj, mbo + 1.0)
                    d_at = _evaluate_d_at(Z_list, candidate)
                    if d_at > current_d_at + 1e-6:
                        bonds = candidate
                        current_d_at = d_at
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return bonds


# ╔════════════════════════════════════════════════════════════════════╗
# ║  MAIN ENTRY POINTS                                               ║
# ╚════════════════════════════════════════════════════════════════════╝

def solve_topology_all(formula: str) -> List[Topology]:
    """Find ALL viable isomers for a formula, ranked by D_at (highest first).

    Usage:
        isomers = solve_topology_all("C2H6O")
        # isomers[0] = best (ethanol), isomers[1] = second (DME), ...
    """
    Z_raw = parse_formula(formula)
    from ptc.atom import IE_eV
    heavy = [(Z, IE_eV(Z)) for Z in Z_raw if Z > 1]
    hydrogens = [Z for Z in Z_raw if Z == 1]
    heavy.sort(key=lambda x: -x[1])
    Z_sorted = [Z for Z, _ in heavy] + hydrogens
    n = len(Z_sorted)
    heavy_idx = [i for i in range(n) if Z_sorted[i] > 1]
    h_idx = [i for i in range(n) if Z_sorted[i] == 1]

    # Special case: H₂
    if not heavy_idx and len(h_idx) >= 2:
        topo = Topology(Z_list=Z_sorted, bonds=[(h_idx[0], h_idx[1], 1.0)],
                        source="solver").finalize()
        return [topo]

    # Generate backbone variants
    backbones = _generate_backbones(Z_sorted, heavy_idx)

    # For each backbone: run pipeline with AND without ring closure,
    # and with both min_ring=3 and min_ring=5
    seen_sigs = set()
    results = []  # (D_at, Topology)

    for bb in backbones:
        for allow_small in [False, True]:
            bonds, charges = _run_pipeline(
                Z_sorted, list(bb), heavy_idx, h_idx,
                allow_small_rings=allow_small)
            bonds = _local_optimize(Z_sorted, bonds)

            sig = _bond_signature(Z_sorted, bonds)
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)

            topo = Topology(Z_list=list(Z_sorted), bonds=list(bonds),
                            source="solver")
            if any(c != 0 for c in charges):
                topo.charges = list(charges)
            topo = topo.finalize()

            from ptc.molecule import compute_D_at
            try:
                D = compute_D_at(topo).D_at
                results.append((D, topo))
            except Exception:
                pass

    results.sort(key=lambda x: -x[0])
    return [topo for _, topo in results]


def solve_topology(formula: str) -> Topology:
    """Find the BEST topology for a formula (highest D_at)."""
    isomers = solve_topology_all(formula)
    if not isomers:
        raise ValueError(f"Could not solve topology for {formula}")
    return isomers[0]
