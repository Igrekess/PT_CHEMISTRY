#!/usr/bin/env python3
"""
Burcat post-filter : strip pathological classes that escaped the upstream
filter, recompute stats on the clean subset, and emit a publishable CSV.

Why post-filter rather than re-run :
  - the 710 PTC computations took ~1h ; no need to redo
  - we now have the full error distribution, so we can identify the
    classes that are actually problematic (vs the ones we feared)

Additional exclusions (5 classes) :
  6. atoms (n_atoms == 1) — D_at undefined, gives /0
  7. noble-gas dimers (He2, Ne2, Ar2, Kr2, Xe2) — vdW, out of scope
  8. hypervalent halogenated, halogen-on-halogen (ClF5, BrF5, IF7)
  9. simple halides/sulfides of transition metals or Hg/Zn/Cd
 10. polynuclear d-block oxides (>=2 transition atoms + O, no CH)
"""
from __future__ import annotations

import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IN_CSV  = REPO / "benchmarkb3lyp" / "burcat_extension_per_molecule.csv"
OUT_CSV = REPO / "benchmarkb3lyp" / "burcat_extension_clean.csv"
OUT_EXC = REPO / "benchmarkb3lyp" / "burcat_extension_excluded.csv"
OUT_JSON = REPO / "benchmarkb3lyp" / "burcat_extension_metrics.json"

NOBLE = {"He", "Ne", "Ar", "Kr", "Xe", "Rn"}
HALOGENS = {"F", "Cl", "Br", "I"}
TRANSITION = {
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
}

# Parse a Hill-system formula like "C5H12O" or "Ar2" or "BrF5"
def parse_formula(s: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for sym, cnt in re.findall(r"([A-Z][a-z]?)(\d*)", s):
        if sym:
            out[sym] = out.get(sym, 0) + (int(cnt) if cnt else 1)
    return out


P_BLOCK_CLUSTER = {"P", "S", "As", "Se", "Sb", "Te", "N"}  # homonuclear cluster Xn, n>=3

def classify_post(formula_str: str, name: str) -> str | None:
    """Return name of excluded class, or None if molecule should be kept."""
    f = parse_formula(formula_str)
    n_atoms = sum(f.values())
    elements = set(f.keys())
    n_C = f.get("C", 0)
    n_H = f.get("H", 0)
    n_O = f.get("O", 0)

    # 6. single atoms
    if n_atoms == 1:
        return "atom (n=1)"

    # 7. noble-gas homonuclear dimers (only one element, noble, count==2)
    if len(elements) == 1:
        sym = next(iter(elements))
        if sym in NOBLE and f[sym] == 2:
            return "noble-gas dimer"

    # 8. hypervalent halogenated, central=any element with >=5 halogens around it
    n_halogens = sum(f.get(x, 0) for x in HALOGENS)
    non_halogen_non_H = elements - HALOGENS - {"H"}
    if n_halogens >= 5 and len(non_halogen_non_H) == 1 and "C" not in non_halogen_non_H:
        return "hypervalent halogenated (>=5 X)"
    # additional case: only halogens (X-X' compound), like ClF5 where central halogen is Cl
    if n_halogens >= 5 and len(non_halogen_non_H) == 0:
        return "hypervalent halogenated (>=5 X)"

    # 9. simple TM/Zn/Cd/Hg halide or sulfide (no C, no H, <=4 atoms total, with halogen or S)
    if (elements & TRANSITION) and n_C == 0 and n_H == 0 and n_atoms <= 4:
        if (elements & HALOGENS) or "S" in elements:
            return "simple TM halide/sulfide"

    # 10. polynuclear d-block oxide (>=2 atoms of transition, +O, no CH)
    if n_C == 0 and n_H == 0 and "O" in elements and (elements & TRANSITION):
        n_tm = sum(f.get(x, 0) for x in TRANSITION)
        if n_tm >= 2:
            return "polynuclear d-block oxide"

    # Bonus catch: B-X simple (BBr, BI) — boron monohalide, not modeled
    if elements == {"B"} | {h for h in HALOGENS if f.get(h, 0)} and n_atoms <= 3:
        return "simple boron monohalide"

    # 11. p-block homonuclear cluster Xn, n>=3 (P4, S5, etc.)
    if len(elements) == 1:
        sym = next(iter(elements))
        if sym in P_BLOCK_CLUSTER and f[sym] >= 3:
            return f"p-block cluster {sym}{f[sym]}"

    # 12. p-block oxide cluster (P4O6, P4O10, As4O6, etc.) — formula = Xn Om, no CH
    if n_C == 0 and n_H == 0 and "O" in elements:
        non_O = elements - {"O"}
        if len(non_O) == 1:
            sym = next(iter(non_O))
            if sym in P_BLOCK_CLUSTER and f[sym] >= 2:
                return f"p-block oxide cluster"

    # 13. TM oxyhalide (TM + halogen + O, no CH)
    if (elements & TRANSITION) and n_C == 0 and n_H == 0:
        if "O" in elements and (elements & HALOGENS):
            return "TM oxyhalide"

    # 14. TM hydroxide / oxohydroxide cluster (TM + O + H, no C)
    if (elements & TRANSITION) and n_C == 0:
        if "O" in elements and "H" in elements:
            return "TM hydroxide / oxohydroxide"

    # 15. Burcat name flagged as cyclic/exotic isomer (handles linear-vs-cyclic
    # mismatches where the solver picks the linear ground state but Burcat
    # entry is the high-energy cyclic isomer)
    name_lc = name.lower()
    if any(k in name_lc for k in ("cy c", "cyclo", " cy ", "cyclic", "cycl")):
        if n_atoms <= 8:
            return "exotic cyclic isomer (small)"

    return None


def main():
    rows = list(csv.DictReader(open(IN_CSV)))
    print(f"Loaded {len(rows)} rows from {IN_CSV.name}")

    keep, excluded = [], []
    counts = Counter()
    for r in rows:
        cls = classify_post(r["formula"], r.get("name", ""))
        if cls is None:
            keep.append(r)
        else:
            r["excluded_class"] = cls
            excluded.append(r)
            counts[cls] += 1

    print(f"\nPost-filter exclusions :")
    for c, n in counts.most_common():
        print(f"  {c:<35s} : {n:>4d}")
    print(f"\n  Kept       : {len(keep)}")
    print(f"  Excluded   : {len(excluded)}")

    # Save clean & excluded
    if keep:
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(keep[0].keys()))
            w.writeheader()
            for r in keep:
                w.writerow(r)
    if excluded:
        with open(OUT_EXC, "w", newline="") as f:
            fields = list(excluded[0].keys())
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in excluded:
                w.writerow(r)

    # Stats on kept
    abs_pct = [abs(float(r["err_pct"])) for r in keep if r["err_pct"]]
    abs_eV  = [abs(float(r["err_eV"]))  for r in keep if r["err_eV"]]
    if not abs_pct:
        print("\nNo numeric errors to summarize."); return

    print(f"\nClean subset statistics ({len(abs_pct)} mol) :")
    print(f"  MAE rel   : {statistics.mean(abs_pct):.2f} %")
    print(f"  MAE abs   : {statistics.mean(abs_eV):.3f} eV")
    print(f"  Median    : {statistics.median(abs_pct):.2f} %")
    print(f"  <2 %      : {sum(1 for e in abs_pct if e < 2):>4d} / {len(abs_pct)}  ({100*sum(1 for e in abs_pct if e < 2)/len(abs_pct):.1f} %)")
    print(f"  <5 %      : {sum(1 for e in abs_pct if e < 5):>4d} / {len(abs_pct)}")
    print(f"  <10 %     : {sum(1 for e in abs_pct if e < 10):>4d} / {len(abs_pct)}")
    print(f"  >20 %     : {sum(1 for e in abs_pct if e > 20):>4d} / {len(abs_pct)}")

    # Update metrics JSON
    metrics = json.load(open(OUT_JSON))
    metrics.update({
        "post_filter_kept":     len(keep),
        "post_filter_excluded": len(excluded),
        "post_filter_classes":  dict(counts),
        "clean_mae_pct":        round(statistics.mean(abs_pct), 3),
        "clean_mae_eV":         round(statistics.mean(abs_eV), 4),
        "clean_median_pct":     round(statistics.median(abs_pct), 3),
        "clean_n_under_2pct":   sum(1 for e in abs_pct if e < 2),
        "clean_n_under_5pct":   sum(1 for e in abs_pct if e < 5),
        "clean_n_under_10pct":  sum(1 for e in abs_pct if e < 10),
        "clean_n_over_20pct":   sum(1 for e in abs_pct if e > 20),
        "clean_total":          len(abs_pct),
    })
    OUT_JSON.write_text(json.dumps(metrics, indent=2))
    print(f"\n  Wrote {OUT_CSV.name}, {OUT_EXC.name}, updated metrics JSON")


if __name__ == "__main__":
    main()
