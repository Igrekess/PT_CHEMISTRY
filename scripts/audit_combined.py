#!/usr/bin/env python3
"""
Combine CCCBDB v3 and Burcat audits into a unified per-mol cross-validation
status. Two independent sources catch different molecules ; agreement
between them is the strongest evidence.

Combined status :
  ok_consensus    : both CCCBDB and Burcat give <2 % diff (highest confidence)
  ok_one_source   : at least one source gives <2 %, the other is n/a
  conflict        : both verifiable, disagree (>2 % in either) — needs review
  one_source_fail : one verifies, other flags >2 % — borderline
  n/a             : neither source has the molecule

Output : audit_combined_2026-05-04.json + .csv
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

REPO     = Path(__file__).resolve().parents[1]
CCCBDB_J = REPO / "benchmarkb3lyp" / "audit_dexp_2026-05-03.json"
BURCAT_J = REPO / "benchmarkb3lyp" / "audit_burcat_2026-05-04.json"
OUT_J    = REPO / "benchmarkb3lyp" / "audit_combined_2026-05-04.json"
OUT_CSV  = REPO / "benchmarkb3lyp" / "audit_combined_2026-05-04.csv"


def main():
    with open(CCCBDB_J) as f:
        cc = json.load(f)
    with open(BURCAT_J) as f:
        bu = json.load(f)
    cc_idx = {(r["name"], r["smiles"]): r for r in cc["results"]}
    bu_idx = {(r["name"], r["smiles"]): r for r in bu["results"]}

    # Union of all keys
    all_keys = sorted(set(cc_idx.keys()) | set(bu_idx.keys()))
    print(f"CCCBDB v3 entries : {len(cc_idx)}")
    print(f"Burcat entries    : {len(bu_idx)}")
    print(f"Union             : {len(all_keys)}")

    out: list[dict] = []
    counts = Counter()
    for key in all_keys:
        cr = cc_idx.get(key)
        br = bu_idx.get(key)
        cc_status = (cr or {}).get("status", "n/a")
        bu_status = (br or {}).get("status", "n/a")
        cc_diff = (cr or {}).get("best_diff_pct")
        bu_diff = (br or {}).get("diff_pct")

        cc_ok = cc_status in ("ok_<1%", "warn_1-2%")
        bu_ok = bu_status in ("ok_<1%", "warn_1-2%")
        cc_fail = cc_status == "fail_>2%"
        bu_fail = bu_status == "fail_>2%"
        cc_na = cc_status == "n/a"
        bu_na = bu_status == "n/a"

        if cc_ok and bu_ok:
            combined = "ok_consensus"        # 🟢🟢
        elif (cc_ok and bu_na) or (bu_ok and cc_na):
            combined = "ok_one_source"       # 🟢⚪
        elif cc_ok and bu_fail:
            combined = "one_source_fail"     # 🟢🟠
        elif bu_ok and cc_fail:
            combined = "one_source_fail"
        elif cc_fail and bu_fail:
            combined = "conflict"            # 🔴🔴
        elif (cc_fail and bu_na) or (bu_fail and cc_na):
            combined = "single_fail"         # 🟠⚪
        else:
            combined = "n/a"                 # ⚪⚪
        counts[combined] += 1

        # Pick best diff (smaller absolute) when both available
        best_diff = None
        if cc_diff is not None and bu_diff is not None:
            best_diff = min(cc_diff, bu_diff, key=abs)
        elif cc_diff is not None:
            best_diff = cc_diff
        elif bu_diff is not None:
            best_diff = bu_diff

        out.append({
            "name":            (cr or br)["name"],
            "smiles":          (cr or br)["smiles"],
            "cas":             (cr or br).get("cas"),
            "source":          (cr or br).get("source"),
            "D_exp":           (cr or br)["D_exp"],
            "cccbdb_status":   cc_status,
            "cccbdb_diff_pct": cc_diff,
            "burcat_status":   bu_status,
            "burcat_diff_pct": bu_diff,
            "combined_status": combined,
            "best_diff_pct":   best_diff,
        })

    # Save
    OUT_J.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_J, "w") as f:
        json.dump({"counts": dict(counts), "n_total": len(out), "results": out}, f, indent=1)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "name", "smiles", "cas", "source", "D_exp_eV",
            "cccbdb_status", "cccbdb_diff_pct",
            "burcat_status", "burcat_diff_pct",
            "combined_status", "best_diff_pct",
        ])
        for r in out:
            w.writerow([
                r["name"], r["smiles"], r.get("cas") or "",
                r.get("source") or "", r["D_exp"],
                r["cccbdb_status"], r["cccbdb_diff_pct"] if r["cccbdb_diff_pct"] is not None else "",
                r["burcat_status"], r["burcat_diff_pct"] if r["burcat_diff_pct"] is not None else "",
                r["combined_status"],
                r["best_diff_pct"] if r["best_diff_pct"] is not None else "",
            ])

    print()
    print("=== Combined audit summary ===")
    n = len(out)
    legend = {
        "ok_consensus":     "🟢 both sources agree <2 %",
        "ok_one_source":    "🟢 one source <2 %, other n/a",
        "one_source_fail":  "🟡 one <2 %, the other >2 %",
        "single_fail":      "🟠 one source >2 %, other n/a",
        "conflict":         "🔴 both sources >2 %",
        "n/a":              "⚪ neither source has data",
    }
    for k in ("ok_consensus", "ok_one_source", "one_source_fail",
              "single_fail", "conflict", "n/a"):
        v = counts.get(k, 0)
        print(f"  {k:18s} : {v:>4d}  ({100*v/n:>5.1f} %)  {legend[k]}")

    # High-confidence count (ok_consensus + ok_one_source)
    hc = counts.get("ok_consensus", 0) + counts.get("ok_one_source", 0)
    print(f"\n  High confidence (any source <2 %) : {hc}/{n} ({100*hc/n:.1f} %)")

    print(f"\nWrote {OUT_J.name}  +  {OUT_CSV.name}")


if __name__ == "__main__":
    main()
