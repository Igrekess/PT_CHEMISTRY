#!/usr/bin/env python3
"""
Reclassify the 1000 benchmark molecules into Organique / Inorganique / d-block
using a robust chemistry criterion.

Audit found ~288/1000 mislabeled categories in benchmark_1000_verified.json
(e.g. quinoline as "Inorganique", B2 as "Organique"). This rebuilds the
"category" field from the SMILES with these rules :

    1. d-block   : contains at least one transition-metal atom
                   (Sc-Zn, Y-Cd, La-Hg, Ac-Cn).
    2. Organique : NOT d-block AND has at least one explicit/implicit C-H
                   bond (i.e. a carbon bonded to a hydrogen).
    3. Inorganique : everything else.

This is the standard IUPAC-flavoured convention for thermochemistry
benchmarks (D_at on ATcT/CCCBDB).

Edge cases handled :
  - CO, CO2, HCN-without-C-H : no C-H bond → Inorganique
  - methane, ethane, large hydrocarbons : C-H present → Organique
  - diborane B2H6 : no C → Inorganique
  - DMSO (C-H present) : Organique
  - ferrocene-like / CoCl2 / Fe carbonyls : d-block (transition metal)

Usage :
    python scripts/fix_categories.py            # apply, write file
    python scripts/fix_categories.py --dry-run  # preview
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VERIFIED = REPO / "ptc" / "data" / "benchmark_1000_verified.json"

# Transition metals (d-block) by symbol
TRANSITION_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    # Lanthanides + actinides (f-block, but binned with d-block here)
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am",
}


def classify(smiles: str) -> str:
    """Return 'd-block' / 'Organique' / 'Inorganique' for a SMILES.

    Rule:
      1. d-block   : contains a transition-metal / lanthanide / actinide atom.
      2. Organique : contains at least one carbon atom bonded to something
                     other than oxygen alone (i.e., C-H, C-C, C-N, C-S,
                     C-halogen, etc.). Catches urea (C-N), HCN (C-H),
                     phosgene (C-Cl), formamide (C-N), etc.
                     Excludes CO, CO2, carbonates (C bonded only to O).
      3. Inorganique : everything else.
    """
    from rdkit import Chem
    from rdkit.RDLogger import DisableLog, EnableLog
    DisableLog("rdApp.*")
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Inorganique"
        elements = {a.GetSymbol() for a in mol.GetAtoms()}
        if elements & TRANSITION_METALS:
            return "d-block"
        mol_h = Chem.AddHs(mol)
        for atom in mol_h.GetAtoms():
            if atom.GetSymbol() != "C":
                continue
            for nb in atom.GetNeighbors():
                if nb.GetSymbol() != "O":
                    return "Organique"
        return "Inorganique"
    finally:
        EnableLog("rdApp.*")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(VERIFIED))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.input)
    with open(in_path) as f:
        mols = json.load(f)
    print(f"Loaded {len(mols)} molecules.")

    changes: list[tuple[str, str, str]] = []
    new_dist: dict[str, int] = {}
    for m in mols:
        old = m.get("category", "?")
        new = classify(m["smiles"])
        new_dist[new] = new_dist.get(new, 0) + 1
        if old != new:
            changes.append((m["name"], old, new))
        m["category_old"] = old  # preserve for audit
        m["category"] = new

    print(f"\n=== Category changes: {len(changes)} ===")
    print(f"Old distribution → New distribution:")
    old_dist: dict[str, int] = {}
    for m in mols:
        old_dist[m["category_old"]] = old_dist.get(m["category_old"], 0) + 1
    for cat in ("Organique", "Inorganique", "d-block"):
        oc = old_dist.get(cat, 0); nc = new_dist.get(cat, 0)
        delta = nc - oc
        sign = "+" if delta >= 0 else ""
        print(f"  {cat:14s}: {oc:>4d} → {nc:>4d}  ({sign}{delta})")

    # Show samples per change type
    by_change: dict[tuple[str, str], list[str]] = {}
    for nm, old, new in changes:
        by_change.setdefault((old, new), []).append(nm)
    print("\nChange breakdown :")
    for (old, new), names in sorted(by_change.items(), key=lambda x: -len(x[1])):
        sample = ", ".join(names[:6])
        more = f" (+{len(names)-6} more)" if len(names) > 6 else ""
        print(f"  {old:14s} → {new:14s} : {len(names):>3d}  e.g. {sample}{more}")

    # Drop the audit field before writing (don't pollute the file)
    for m in mols:
        m.pop("category_old", None)

    if args.dry_run:
        print("\n(dry run — no file modified)")
        return

    # Backup
    backup = in_path.with_name(
        in_path.stem + f".pre_categoryfix_{date.today():%Y-%m-%d}" + in_path.suffix
    )
    if not backup.exists():
        shutil.copy2(in_path, backup)
        print(f"\nBackup: {backup.name}")

    with open(in_path, "w") as f:
        json.dump(mols, f, indent=1)
    print(f"Wrote {in_path}")


if __name__ == "__main__":
    main()
