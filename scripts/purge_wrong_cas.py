#!/usr/bin/env python3
"""
Validate CAS numbers in benchmark_1000_verified.json by cross-checking
against PubChem.

The CAS enrichment via PubChem (scripts/enrich_cas_via_pubchem.py) is fast
but imperfect : ambiguous names like "CH" (methylidyne radical, [C][H])
were silently resolved to the common compound ("CH" → CH4 methane,
CAS 74-82-8). The audit step revealed dozens of such cases.

This script :
  1. For every mol with a CAS, queries PubChem for the molecular formula
     associated with that CAS.
  2. Computes the molecular formula of our stored SMILES via RDKit.
  3. If formulas don't match → the CAS does not refer to our compound.
     Clear cas, url_cccbdb, url_nist_webbook.
  4. Otherwise → CAS verified.

Outputs :
  - Updated benchmark_1000_verified.json (in place; backup written)
  - A list of cleared CAS for traceability

Rate limit : ~3 req/s. Total runtime for ~940 mol : ~5 minutes.

Usage :
    python scripts/purge_wrong_cas.py            # apply
    python scripts/purge_wrong_cas.py --dry-run  # preview only
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

REPO     = Path(__file__).resolve().parents[1]
VERIFIED = REPO / "ptc" / "data" / "benchmark_1000_verified.json"

USER_AGENT = "PTC-benchmark-cas-purge/1.0"
PUBCHEM_FORMULA = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
    "{cas}/property/MolecularFormula/JSON"
)


def pubchem_formula(cas: str, timeout: float = 8.0) -> str | None:
    """Return PubChem's reported molecular formula for a CAS, or None on failure."""
    url = PUBCHEM_FORMULA.format(cas=urllib.parse.quote(cas))
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        props = data.get("PropertyTable", {}).get("Properties", [])
        if props:
            return props[0].get("MolecularFormula")
    except Exception:
        pass
    return None


def smiles_to_formula(smi: str) -> str | None:
    """Compute molecular formula from SMILES via RDKit (with implicit H expanded)."""
    try:
        from rdkit import Chem
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula
        from rdkit.RDLogger import DisableLog, EnableLog
        DisableLog("rdApp.*")
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None
            return CalcMolFormula(mol)
        finally:
            EnableLog("rdApp.*")
    except Exception:
        return None


def parse_formula(f: str | None) -> dict[str, int] | None:
    """Parse a chemical formula string into {element: count} dict.

    Handles Hill order ('CH4'), reverse-Hill ('HCu' vs 'CuH'), charges,
    hydrate dots ('C2H6.HCl'), bracketed elements. Returns None if
    parse fails.
    """
    if not f:
        return None
    import re
    # Strip charge suffixes
    f = re.sub(r"[\+\-]", "", f).replace(".", "")
    # Match (Element)(optional digits) tokens
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", f)
    counts: dict[str, int] = {}
    for elem, num in tokens:
        if not elem:
            continue
        n = int(num) if num else 1
        counts[elem] = counts.get(elem, 0) + n
    return counts if counts else None


def formulas_equal(a: str | None, b: str | None) -> bool:
    """True iff two formula strings represent the same atom counts."""
    pa, pb = parse_formula(a), parse_formula(b)
    if pa is None or pb is None:
        return False
    return pa == pb


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(VERIFIED))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rate", type=float, default=0.35,
                    help="Seconds between PubChem requests (default 0.35 = ~3 req/s)")
    ap.add_argument("--limit", type=int, default=None, help="Process only N rows (debug)")
    args = ap.parse_args()

    in_path = Path(args.input)
    with open(in_path) as f:
        mols = json.load(f)

    todo = [m for m in mols if m.get("cas")]
    if args.limit:
        todo = todo[: args.limit]
    print(f"Mol with CAS to validate: {len(todo)}")

    # Backup before any change
    if not args.dry_run:
        backup = in_path.with_name(
            in_path.stem + f".pre_caspurge_{date.today():%Y-%m-%d}" + in_path.suffix
        )
        if not backup.exists():
            shutil.copy2(in_path, backup)
            print(f"Backup: {backup.name}")

    cleared: list[tuple[str, str, str, str | None]] = []
    verified = 0
    pubchem_no_data = 0
    smiles_parse_fail = 0
    for i, m in enumerate(todo, 1):
        cas = m["cas"]
        our_formula = smiles_to_formula(m["smiles"])
        if not our_formula:
            smiles_parse_fail += 1
            tag = "?"
            reason = "smiles_parse"
        else:
            pc_formula = pubchem_formula(cas)
            if not pc_formula:
                pubchem_no_data += 1
                tag = "—"
                reason = "pubchem_no_data"
            elif formulas_equal(our_formula, pc_formula):
                verified += 1
                tag = "✓"
                reason = "match"
            else:
                # Wrong CAS — clear
                if not args.dry_run:
                    m["cas"] = None
                    m["url_cccbdb"] = None
                    m["url_nist_webbook"] = None
                cleared.append((m["name"], cas, our_formula, pc_formula))
                tag = "❌"
                reason = f"formula_mismatch (ours={our_formula} pubchem={pc_formula})"

        if i % 50 == 0 or tag == "❌":
            print(f"  [{i:>4d}/{len(todo)}] {tag} {m['name'][:25]:25s} {cas:13s} {reason}",
                  flush=True)
        time.sleep(args.rate)

    print()
    print(f"Verified  : {verified}")
    print(f"Cleared   : {len(cleared)}  (CAS pointed to wrong compound)")
    print(f"No PubChem data : {pubchem_no_data}")
    print(f"SMILES parse fail : {smiles_parse_fail}")

    if cleared and not args.dry_run:
        with open(in_path, "w") as f:
            json.dump(mols, f, indent=1)
        print(f"\nWrote {in_path.relative_to(REPO)}")

        # Save the list of cleared
        log = in_path.with_name("cas_cleared_2026-05-03.tsv")
        with open(log, "w") as f:
            f.write("name\twrong_cas\tour_formula\tpubchem_formula\n")
            for n, c, of, pf in cleared:
                f.write(f"{n}\t{c}\t{of}\t{pf or ''}\n")
        print(f"Cleared CAS list  : {log.name}")
    elif args.dry_run:
        print("\n(dry run — no file modified)")


if __name__ == "__main__":
    main()
