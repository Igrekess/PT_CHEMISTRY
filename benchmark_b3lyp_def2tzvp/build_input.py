#!/usr/bin/env python3
"""
Build input_n{N}.json for the standalone B3LYP/def2-TZVP single-point benchmark.

This script runs on the machine that has access to:
  - the original results_b3lyp.json (for recycled ZPE + spin/charge)
  - the original geometry_overrides.json
  - RDKit (for MMFF embedding of multi-atom molecules)

It writes a fully self-contained input_n{N}.json containing, for each
molecule with n_atoms <= N, status=ok, non pathological):

    - name, smiles, category, D_exp           (metadata)
    - atoms                                   (element symbols)
    - coords_angstrom                         (frozen 3D coords)
    - charge, spin                            (PySCF spin = N_alpha - N_beta)
    - zpe_eV_recycled                         (B3LYP/6-31G* ZPE * 0.9806)
    - geometry_source                         (provenance string)

Usage:
    python build_input_n4.py                    # default: n_atoms <= 4 -> input_n4.json
    python build_input_n4.py --max-atoms 8      # extends to n<=8 -> input_n8.json
    python build_input_n4.py --max-atoms 12     # n<=12 -> input_n12.json

The downstream run_def2tzvp.py then needs only PySCF + NumPy.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ----- locations -----------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parent  # PTC/
B3LYP_DIR = REPO / "benchmarkb3lyp"
RESULTS_PATH   = B3LYP_DIR / "results_b3lyp.json"
OVERRIDES_PATH = B3LYP_DIR / "geometry_overrides.json"

# ----- exclusion lists (must match the screened 860 file) ------------------
FAIL_NAMES = {
    "BrF3", "ClO2_rad", "HBF4_frag", "NO2", "ClF3",
    "Na2F2", "o_Benzyne", "B2H6", "Buckminsterfullerene",
}
PATHO_PAIRS = {
    ("S2F10",  -114.51460532591501),
    ("PCl5",   -40.39757672735492),
    ("SF6",    -29.503521489072092),
    ("SF6",    -29.314521489072092),
    ("HBO2",   -9.287049320674095),
    ("phenanthrene", -7.32752354263738),
    ("SO3mol", -6.884079530764565),
    ("SF4",    -5.190584799158868),
}

# ----- diatomic geometry helpers (mirror run_b3lyp_atct.py) ----------------
DIATOMIC_BOND_LENGTHS_ANG = {
    ("B", "B"): 1.590, ("Cl", "Cl"): 1.988, ("Cu", "Cl"): 2.051,
    ("F", "H"): 0.917, ("F", "Li"): 1.564, ("H", "H"): 0.741,
    ("N", "H"): 1.036, ("N", "N"): 1.098, ("N", "O"): 1.151,
    ("O", "P"): 1.476, ("O", "Si"): 1.510, ("P", "S"): 1.900,
    ("S", "Si"): 1.929,
}


def smiles_element_tokens(smiles: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(smiles):
        ch = smiles[i]
        if ch == "[":
            j = smiles.find("]", i + 1)
            if j < 0:
                break
            content = smiles[i + 1:j]
            m = re.match(r"\d*([A-Z][a-z]?|[bcnops])", content)
            if m:
                sym = m.group(1)
                tokens.append(sym.capitalize() if sym.islower() else sym)
            i = j + 1
            continue
        if smiles.startswith("Cl", i) or smiles.startswith("Br", i):
            tokens.append(smiles[i:i + 2])
            i += 2
            continue
        if ch in "BCNOFPSIH":
            tokens.append(ch)
        elif ch in "bcnops":
            tokens.append(ch.upper())
        i += 1
    return tokens


def coords_for_diatomic(bond_length: float):
    half = bond_length / 2.0
    return [(0.0, 0.0, -half), (0.0, 0.0, half)]


def covalent_bond_length(atoms: list[str], bond_order: float | None) -> float:
    from rdkit import Chem
    pt = Chem.GetPeriodicTable()
    length = pt.GetRcovalent(atoms[0]) + pt.GetRcovalent(atoms[1])
    if bond_order is not None:
        if bond_order >= 2.5:
            length *= 0.78
        elif bond_order >= 1.5:
            length *= 0.87
    return float(length)


def diatomic_geometry_from_smiles(smiles: str):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    try:
        mol = Chem.MolFromSmiles(smiles)
    finally:
        RDLogger.EnableLog("rdApp.*")
    if mol is None:
        atoms = smiles_element_tokens(smiles)
        if len(atoms) != 2:
            return None
        bo = 3 if "#" in smiles else (2 if "=" in smiles else 1)
        pair = tuple(sorted(atoms))
        bl = DIATOMIC_BOND_LENGTHS_ANG.get(pair, covalent_bond_length(atoms, bo))
        return atoms, coords_for_diatomic(bl), bl
    mol = Chem.AddHs(mol)
    if mol.GetNumAtoms() != 2:
        return None
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    pair = tuple(sorted(atoms))
    bond = mol.GetBondBetweenAtoms(0, 1)
    bo = bond.GetBondTypeAsDouble() if bond is not None else None
    bl = DIATOMIC_BOND_LENGTHS_ANG.get(pair, covalent_bond_length(atoms, bo))
    return atoms, coords_for_diatomic(bl), bl


def smiles_to_xyz(smiles: str):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit could not parse: {smiles}")
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            if AllChem.EmbedMolecule(mol, randomSeed=1) != 0:
                raise ValueError("3D embed failed")
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            pass
    finally:
        RDLogger.EnableLog("rdApp.*")
    conf = mol.GetConformer()
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = [tuple(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    return atoms, coords


# ----- main ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-atoms", type=int, default=4,
                    help="Upper bound on n_atoms (heavy + H). Default: 4. "
                         "Output file is input_n{N}.json")
    args = ap.parse_args()
    n_max = int(args.max_atoms)
    out_path = HERE / f"input_n{n_max}.json"

    if not RESULTS_PATH.exists():
        sys.exit(f"missing {RESULTS_PATH}")
    if not OVERRIDES_PATH.exists():
        sys.exit(f"missing {OVERRIDES_PATH}")
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    with open(OVERRIDES_PATH) as f:
        overrides_data = json.load(f)
    overrides = {(o["name"], o["smiles"]): o for o in overrides_data["overrides"]}

    rows_in = [r for r in results["results"] if r.get("n_atoms", 999) <= n_max]

    out_rows = []
    skipped = []
    for r in rows_in:
        if r.get("status") != "ok":
            skipped.append((r["name"], "status_not_ok"))
            continue
        if r["name"] in FAIL_NAMES:
            skipped.append((r["name"], "in_fail_list"))
            continue
        if (r["name"], r.get("err_eV")) in PATHO_PAIRS:
            skipped.append((r["name"], "pathological"))
            continue

        # Recover atoms + coords
        ovr = overrides.get((r["name"], r["smiles"]))
        bond_length = r.get("bond_length_angstrom")
        if ovr and ovr.get("bond_length_angstrom"):
            bond_length = ovr["bond_length_angstrom"]
        try:
            geom_src = r.get("geometry_source", "")
            if bond_length is not None:
                # diatomic
                atoms = smiles_element_tokens(r["smiles"])
                if len(atoms) != 2:
                    diatomic = diatomic_geometry_from_smiles(r["smiles"])
                    if diatomic is None:
                        skipped.append((r["name"], "diatomic_resolve_fail"))
                        continue
                    atoms = diatomic[0]
                coords = coords_for_diatomic(bond_length)
                source = geom_src or "diatomic_table_or_covalent_radii"
            else:
                # try diatomic detection first
                diatomic = diatomic_geometry_from_smiles(r["smiles"])
                if diatomic is not None:
                    atoms, coords, bl_resolved = diatomic
                    bond_length = bl_resolved
                    source = geom_src or "diatomic_table_or_covalent_radii"
                else:
                    atoms, coords = smiles_to_xyz(r["smiles"])
                    source = geom_src or "rdkit_mmff"
        except Exception as e:
            skipped.append((r["name"], f"geom_error: {e}"))
            continue

        if len(atoms) != r["n_atoms"]:
            skipped.append((r["name"], f"atom_count_mismatch {len(atoms)}!={r['n_atoms']}"))
            continue

        out_rows.append({
            "name": r["name"],
            "smiles": r["smiles"],
            "category": r.get("category", "?"),
            "n_atoms": r["n_atoms"],
            "atoms": atoms,
            "coords_angstrom": [list(map(float, c)) for c in coords],
            "bond_length_angstrom": bond_length if bond_length is not None else None,
            "charge": int(r.get("charge", 0)),
            "spin": int(r.get("spin", 0)),
            "zpe_eV_recycled": float(r["ZPE_eV"]),
            "D_exp": float(r["D_exp"]),
            "geometry_source": source,
            "ref_b3lyp_631gs": {
                "D_b3lyp_eV":  r["D_b3lyp_eV"],
                "D_elec_eV":   r["D_elec_eV"],
                "err_eV":      r.get("err_eV"),
                "rel_err_pct": r.get("rel_err_pct"),
            },
        })

    payload = {
        "description": (
            f"Frozen input for B3LYP/def2-TZVP single-point benchmark on the "
            f"screened ATcT subset with n_atoms <= {n_max}. ZPE is recycled "
            f"from B3LYP/6-31G* (Scott-Radom 0.9806 already applied). "
            f"Geometries are identical to the original 6-31G* run."
        ),
        "ZPE_NOTE": (
            "zpe_eV_recycled is already scaled by 0.9806 (Scott & Radom 1996); "
            "use it directly: D_at = E_atoms - E_mol - zpe_eV_recycled"
        ),
        "n_molecules": len(out_rows),
        "n_skipped": len(skipped),
        "skipped": [{"name": n, "reason": why} for (n, why) in skipped],
        "exclusion_rules": {
            "fail_names":   sorted(FAIL_NAMES),
            "patho_pairs":  [(n, e) for (n, e) in sorted(PATHO_PAIRS)],
            "n_atoms_max":  n_max,
        },
        "molecules": out_rows,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"Wrote {out_path}: {len(out_rows)} molecules ({len(skipped)} skipped)")
    if skipped:
        print("Skipped:")
        for n, why in skipped:
            print(f"  {n}  -  {why}")


if __name__ == "__main__":
    main()
