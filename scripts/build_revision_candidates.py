#!/usr/bin/env python3
"""
Build a CSV of molecules whose D_exp diverges 2-15% from CCCBDB-derived value.
These are candidates for modernisation against ATcT v1.130.

ATcT website is Cloudflare-protected so cannot be auto-scraped. This script
produces a structured CSV with all metadata + direct ATcT/CCCBDB URLs so the
user can verify each candidate in a browser, then apply updates manually.

Excludes :
  - mol with TM-diatomic pattern (HH spectroscopic D₀ canonical, keep ours)
  - mol with diff > 15 % (large divergences usually means CCCBDB is wrong)
  - mol with diff < 2 % (no need to revise)

Output : benchmarkb3lyp/revision_candidates_2026-05-04.csv
"""
from __future__ import annotations

import csv
import json
import urllib.parse
from pathlib import Path

REPO    = Path(__file__).resolve().parents[1]
AUDIT   = REPO / "benchmarkb3lyp" / "audit_dexp_2026-05-03.json"
VERIFIED = REPO / "ptc" / "data" / "benchmark_1000_verified.json"
OUT     = REPO / "benchmarkb3lyp" / "revision_candidates_2026-05-04.csv"

# Transition metals — we exclude their diatomics from candidates (HH spectroscopic
# D₀ canonical for these; CCCBDB has incoherent values for them).
TM = {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
      "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
      "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
      "La", "Ce"}


def is_tm_diatomic(smiles: str) -> bool:
    """Detect mol with at most ~3 atoms involving a transition metal."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        elements = {a.GetSymbol() for a in mol.GetAtoms()}
        return bool(elements & TM) and mol.GetNumAtoms() <= 3
    except Exception:
        return False


def main():
    with open(AUDIT) as f:
        audit = json.load(f)
    with open(VERIFIED) as f:
        verified = json.load(f)
    v_idx = {(m["name"], m["smiles"]): m for m in verified}

    candidates = []
    for r in audit["results"]:
        if r["status"] != "fail_>2%":
            continue
        diff = r.get("best_diff_pct")
        if diff is None:
            continue
        a = abs(diff)
        if a >= 15 or a < 2:
            continue
        if is_tm_diatomic(r["smiles"]):
            continue
        m = v_idx.get((r["name"], r["smiles"]))
        if not m:
            continue
        cas = m.get("cas") or ""
        cas_url = (
            f"https://atct.anl.gov/Thermochemical%20Data/version%201.130/"
            f"index.php"  # site is Cloudflare-gated, use as entry point
        )
        cccbdb_url = m.get("url_cccbdb") or ""
        nist_url = (
            f"https://webbook.nist.gov/cgi/cbook.cgi?ID={urllib.parse.quote(cas)}&Units=SI"
            if cas else
            f"https://webbook.nist.gov/cgi/cbook.cgi?Name={urllib.parse.quote(m['name'])}&Units=SI"
        )

        # Pull D_at_calc from audit (in eV)
        if r.get("best_convention") == "0K":
            d_calc = r.get("D_at_0K_eV")
            hfg = r.get("hfg_0k_kjmol")
        else:
            d_calc = r.get("D_at_298K_eV")
            hfg = r.get("hfg_298k_kjmol")

        candidates.append({
            "name":        r["name"],
            "smiles":      r["smiles"],
            "cas":         cas,
            "current_D_exp_eV":  m["D_exp"],
            "current_source":    m.get("source", ""),
            "CCCBDB_Hfg_kjmol":  hfg,
            "CCCBDB_convention": r.get("best_convention"),
            "CCCBDB_D_at_eV":    d_calc,
            "diff_pct":          round(diff, 2),
            "ATcT_url":          cas_url,
            "NIST_url":          nist_url,
            "CCCBDB_url":        cccbdb_url,
        })

    # Sort by abs(diff_pct) descending so largest drift first
    candidates.sort(key=lambda x: -abs(x["diff_pct"]))

    with open(OUT, "w", newline="") as f:
        if candidates:
            w = csv.DictWriter(f, fieldnames=list(candidates[0].keys()))
            w.writeheader()
            w.writerows(candidates)

    print(f"Wrote {OUT.relative_to(REPO)}")
    print(f"  {len(candidates)} candidates (2-15 % drift, excluding TM diatomics)")
    print()
    print("Distribution by source :")
    from collections import Counter
    src_count = Counter(c["current_source"] for c in candidates)
    for s, n in sorted(src_count.items(), key=lambda x: -x[1]):
        print(f"  {s:20s} : {n}")
    print()
    print("Top 10 by |drift| :")
    for c in candidates[:10]:
        print(f"  {c['name']:30s} {c['current_source']:12s} "
              f"current={c['current_D_exp_eV']:>7.3f}  "
              f"CCCBDB={c['CCCBDB_D_at_eV']:>7.3f}  "
              f"Δ={c['diff_pct']:>+6.2f}%")


if __name__ == "__main__":
    main()
