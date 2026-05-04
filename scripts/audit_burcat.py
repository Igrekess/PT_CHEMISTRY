#!/usr/bin/env python3
"""
Cross-validation audit of D_exp against Burcat's Third Millennium Database.

Burcat is the de-facto reference for gas-phase thermochemistry in combustion
and atmospheric chemistry. ~3000 species with NASA polynomials + ATcT-derived
ΔfH°(298K) values when available.

Advantages over CCCBDB :
  - Local flat-file → no rate limit, no Cloudflare, no silent defaulting
  - Comments include HF298 + uncertainties + reference (ATcT/JANAF/IVTAN/CODATA)
  - Updated regularly by Goos/Burcat/Ruscic (last 2023)

Pipeline :
  1. Scan BURCAT.THR.txt for blocks tagged with a CAS number on its own line
  2. Extract HF298 from the comment text (kJ/mol, gas phase)
  3. For each mol in our verified file with a CAS, check Burcat
  4. Compute D_at(298K) = Σ Hf_atom_298K - HF298_burcat
  5. Compare to stored D_exp
  6. Output per-mol report + summary

Usage :
    python scripts/audit_burcat.py
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

REPO   = Path(__file__).resolve().parents[1]
BURCAT = REPO / "data" / "BURCAT.THR.txt"
VERIFIED = REPO / "ptc" / "data" / "benchmark_1000_verified.json"
AUDIT_PRIOR = REPO / "benchmarkb3lyp" / "audit_dexp_2026-05-03.json"
OUT_JSON = REPO / "benchmarkb3lyp" / "audit_burcat_2026-05-04.json"
OUT_CSV  = REPO / "benchmarkb3lyp" / "audit_burcat_2026-05-04.csv"

EV_PER_KJMOL = 1.0 / 96.485332

# Atom Hf(298.15K) standard, kJ/mol — same table as CCCBDB audit
ATOM_HF_298K = {
    "H":  217.998, "He":   0.000,
    "Li": 159.300, "Be": 324.000, "B":  565.000,
    "C":  716.680, "N":  472.680, "O":  249.180, "F":   79.380, "Ne":  0.000,
    "Na": 107.500, "Mg": 147.100, "Al": 330.000, "Si": 450.000,
    "P":  316.500, "S":  277.170, "Cl": 121.301, "Ar":   0.000,
    "K":   89.000, "Ca": 177.800,
    "Sc": 377.800, "Ti": 473.000, "V":  514.200, "Cr": 396.600, "Mn": 280.700,
    "Fe": 416.300, "Co": 424.700, "Ni": 429.700, "Cu": 337.400, "Zn": 130.400,
    "As": 302.500, "Se": 235.400, "Br": 111.870,
    "Rb":  80.880, "Sr": 159.000,
    "Y":  424.700, "Zr": 608.800, "Nb": 725.900, "Mo": 658.100, "Tc": 678.000,
    "Ru": 643.000, "Rh": 556.900, "Pd": 378.200, "Ag": 284.900, "Cd": 111.800,
    "I":  106.840,
    "Cs":  76.500, "Ba": 180.000,
    "La": 431.000, "Ce": 420.000, "Hf": 619.200, "Ta": 781.500, "W":  848.100,
    "Pt": 565.300, "Au": 366.100, "Hg":  61.380,
}


def parse_burcat(path: Path) -> dict[str, list[dict]]:
    """Return {CAS : [{'hf298_kjmol', 'description', 'phase'}]}, gas-phase only.

    Strategy : iterate over NASA polynomial blocks (4-line groups identified by
    a final " 1\n" / " 2\n" / " 3\n" / " 4\n" pattern). For each block :
      - read phase letter at column 44 of the first line ('G'/'S'/'L'/'C'/'I')
      - walk backwards through preceding comments to find the most recent
        'HF298=value' AND the most recent CAS (XXX-XX-X line).

    Only blocks with phase 'G' (gas) are kept. Comments inside braces
    {...} are excluded so the "primary" HF298 is picked.
    """
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # Find NASA polynomial blocks: 4 consecutive lines ending in " 1", " 2", " 3", " 4"
    # The first line of the block has the species name + phase letter at col 44
    block_starts: list[int] = []
    for i in range(len(lines) - 3):
        l1, l2, l3, l4 = lines[i], lines[i+1], lines[i+2], lines[i+3]
        if (len(l1) >= 80 and l1.rstrip().endswith("1")
            and l2.rstrip().endswith("2")
            and l3.rstrip().endswith("3")
            and l4.rstrip().endswith("4")):
            block_starts.append(i)

    # CAS line : either "X-XX-X" alone, or first CAS in "X-XX-X or Y-YY-Y"
    cas_re   = re.compile(r"^\s*(\d{2,7}-\d{2}-\d)(?:\s+or\s+\d|\s*$)")
    hf298_re = re.compile(r"HF298\s*=\s*(-?\d+\.?\d*)\s*(?:\+/?-)?\s*\d*\.?\d*\s*(kJ|kcal)?",
                          re.IGNORECASE)
    brace_re = re.compile(r"\{[^}]*\}")

    out: dict[str, list[dict]] = {}
    for bs in block_starts:
        l1 = lines[bs]
        phase = l1[44:45] if len(l1) > 44 else ''
        if phase != 'G':
            continue
        name = l1[:18].strip()

        # Walk backwards until we hit a blank line OR a previous polynomial
        # block end (line ending in trailing " 4"). The comments+CAS for
        # *this* species sit between those boundaries.
        cas, hf298, hf298_unit = None, None, None
        for k in range(bs - 1, max(-1, bs - 30), -1):
            line = lines[k]
            stripped = line.strip()
            if not stripped:
                break  # boundary : blank line ends the comment block
            # Boundary : previous polynomial block last line (ends in " 4")
            if line.rstrip().endswith(" 4") and len(line) > 70:
                break

            no_braces = brace_re.sub("", line)
            if hf298 is None:
                m = hf298_re.search(no_braces)
                if m:
                    try:
                        hf298 = float(m.group(1))
                        hf298_unit = (m.group(2) or "kJ").lower()
                    except ValueError:
                        pass
            if cas is None:
                m = cas_re.match(line)
                if m:
                    cas = m.group(1)
            # Don't break early — keep walking to make sure we don't pick
            # a hf298 from a subsequent line (already handled by direction)

        if cas and hf298 is not None:
            # Convert kcal → kJ if needed (1 kcal = 4.184 kJ)
            if hf298_unit == "kcal":
                hf298 = hf298 * 4.184
            out.setdefault(cas, []).append({
                "hf298_kjmol": hf298,
                "description": name,
                "phase":       phase,
            })
    return out


def count_atoms(smi: str) -> dict[str, int] | None:
    try:
        from rdkit import Chem
        from rdkit.RDLogger import DisableLog, EnableLog
        DisableLog("rdApp.*")
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None
            mol = Chem.AddHs(mol)
            c = {}
            for a in mol.GetAtoms():
                c[a.GetSymbol()] = c.get(a.GetSymbol(), 0) + 1
            return c
        finally:
            EnableLog("rdApp.*")
    except Exception:
        return None


def classify(diff_pct):
    if diff_pct is None:
        return "n/a"
    a = abs(diff_pct)
    if a < 1:   return "ok_<1%"
    if a < 2:   return "warn_1-2%"
    return "fail_>2%"


def main():
    print(f"Parsing {BURCAT.name}...")
    burcat_idx = parse_burcat(BURCAT)
    print(f"  {len(burcat_idx)} CAS found in Burcat (gas-phase blocks only)")
    print(f"  Total gas-phase HF298 entries (incl. polymorphs): "
          f"{sum(len(v) for v in burcat_idx.values())}")

    with open(VERIFIED) as f:
        verified = json.load(f)
    print(f"\nLoaded {len(verified)} mol from verified dataset")

    # Cross-audit
    results: list[dict] = []
    counts = Counter()
    for m in verified:
        cas = m.get("cas")
        out = {
            "name":          m["name"],
            "smiles":        m["smiles"],
            "cas":           cas,
            "source":        m.get("source"),
            "D_exp":         m["D_exp"],
            "burcat_hf298":  None,
            "burcat_desc":   None,
            "D_at_burcat":   None,
            "diff_pct":      None,
            "status":        "n/a",
            "reason":        "",
        }
        if not cas:
            out["reason"] = "no_cas"
            counts["n/a"] += 1
            results.append(out)
            continue
        bcat = burcat_idx.get(cas)
        if not bcat:
            out["reason"] = "not_in_burcat"
            counts["n/a"] += 1
            results.append(out)
            continue
        # All entries are already gas-phase (parser filtered)
        b = bcat[0]  # take first variant if multiple
        out["burcat_hf298"] = b["hf298_kjmol"]
        out["burcat_desc"]  = b["description"]

        elem = count_atoms(m["smiles"])
        if not elem:
            out["reason"] = "smiles_parse_failed"
            counts["n/a"] += 1
            results.append(out)
            continue
        missing = [e for e in elem if e not in ATOM_HF_298K]
        if missing:
            out["reason"] = f"atom_hf_missing:{','.join(missing)}"
            counts["n/a"] += 1
            results.append(out)
            continue

        sum_atoms = sum(ATOM_HF_298K[e] * c for e, c in elem.items())
        d_at_kj = sum_atoms - b["hf298_kjmol"]
        d_at_ev = d_at_kj * EV_PER_KJMOL
        diff = (d_at_ev - m["D_exp"]) / m["D_exp"] * 100.0

        out["D_at_burcat"] = round(d_at_ev, 4)
        out["diff_pct"]    = round(diff, 3)
        out["status"]      = classify(diff)
        counts[out["status"]] += 1
        results.append(out)

    # Save
    with open(OUT_JSON, "w") as f:
        json.dump({
            "n_total":  len(results),
            "counts":   dict(counts),
            "results":  results,
        }, f, indent=1)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "name", "smiles", "cas", "source", "D_exp_eV",
            "burcat_hf298_kjmol", "D_at_burcat_eV", "diff_pct",
            "status", "reason", "burcat_desc",
        ])
        for r in results:
            w.writerow([
                r["name"], r["smiles"], r.get("cas") or "",
                r.get("source") or "", r["D_exp"],
                r.get("burcat_hf298") or "",
                r.get("D_at_burcat") or "",
                r.get("diff_pct") or "",
                r["status"], r.get("reason") or "",
                r.get("burcat_desc") or "",
            ])

    print(f"\n=== Burcat audit summary ===")
    n = len(results)
    for k in ("ok_<1%", "warn_1-2%", "fail_>2%", "n/a"):
        v = counts.get(k, 0)
        print(f"  {k:12s} : {v:>4d}  ({100*v/n:.1f} %)")
    print(f"\nWrote {OUT_JSON.name}  +  {OUT_CSV.name}")

    # Top mismatches
    fails = [r for r in results if r["status"] == "fail_>2%"]
    fails.sort(key=lambda x: -abs(x["diff_pct"] or 0))
    print(f"\nTop 15 worst mismatches (>2 %):")
    for r in fails[:15]:
        print(f"  {r['name']:30s} stored={r['D_exp']:>7.3f}  "
              f"Burcat={r['D_at_burcat']:>7.3f}  Δ={r['diff_pct']:+7.2f}%  "
              f"({r.get('source')})")

    # Cross-comparison with prior CCCBDB audit
    if AUDIT_PRIOR.exists():
        with open(AUDIT_PRIOR) as f:
            cccbdb_audit = json.load(f)
        cccbdb_idx = {(r["name"], r["smiles"]): r for r in cccbdb_audit["results"]}
        print(f"\n=== Cross-source agreement (CCCBDB v3 ∩ Burcat) ===")
        agree, disagree, only_burcat, only_cccbdb = 0, 0, 0, 0
        for r in results:
            key = (r["name"], r["smiles"])
            cb = cccbdb_idx.get(key)
            if not cb:
                continue
            cb_ok = cb["status"] in ("ok_<1%", "warn_1-2%")
            bu_ok = r["status"] in ("ok_<1%", "warn_1-2%")
            cb_na = cb["status"] == "n/a"
            bu_na = r["status"] == "n/a"
            if cb_ok and bu_ok:
                agree += 1
            elif cb_na and not bu_na:
                only_burcat += 1
            elif not cb_na and bu_na:
                only_cccbdb += 1
            elif not cb_na and not bu_na:
                disagree += 1
        print(f"  Both sources agree (<2 %)        : {agree}")
        print(f"  Only Burcat verifiable           : {only_burcat}")
        print(f"  Only CCCBDB verifiable           : {only_cccbdb}")
        print(f"  Both verifiable, disagree (>2 %) : {disagree}")


if __name__ == "__main__":
    main()
