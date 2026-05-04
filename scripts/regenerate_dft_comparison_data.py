#!/usr/bin/env python3
"""
Regenerate ptc_app/dft_comparison_data.json from the canonical sources :

  - benchmarkb3lyp/results_b3lyp.json      (B3LYP/6-31G*, ZPE-corrected D_at)
  - benchmarkb3lyp/ptc_fresh_2026-05-01.json (PTC results)

Fixes the previous bug where dft_panel.py was showing :

  D_b3lyp = E_atoms - E_mol  (sans ZPE !)

instead of :

  D_b3lyp = E_atoms - E_mol - ZPE  (avec ZPE Scott-Radom 0.9806)

The previous panel inflated B3LYP MAE artificially (5.90 % au lieu de
4.16 % réels). It also kept the 8 pathological outliers (S2F10, PCl5,
SF6, ...) which dominate the bar charts.

Screening rules (matched to 260503_benchmark_B3LYP_860.json) :
  - 9 SCF/Hessian failures        →  D_b3lyp = None
  - 8 catastrophic |err| > 5 eV   →  D_b3lyp = None
  - 123 mol with elements unsupported by 6-31G* → never had B3LYP →  D_b3lyp = None

Net: 860 mol have a usable D_b3lyp, 140 do not (vs 828/172 in the buggy
panel data).

Usage :
    python scripts/regenerate_dft_comparison_data.py
        # writes ptc_app/dft_comparison_data.json
        # backs up the old file as dft_comparison_data.backup_<date>.json
"""
from __future__ import annotations

import json
import statistics
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # PTC/
RESULTS_B3LYP   = REPO / "benchmarkb3lyp" / "results_b3lyp.json"
PTC_FRESH       = REPO / "benchmarkb3lyp" / "ptc_fresh_2026-05-01.json"
BENCHMARK_LIST  = REPO / "ptc" / "data" / "benchmark_1000_verified.json"
BENCHMARK_LIST_FALLBACK = REPO / "ptc" / "data" / "benchmark_1000.json"
DEF2TZVP_3WAY   = REPO / "benchmark_b3lyp_def2tzvp" / "comparison_3way_n12.json"
DEF2TZVP_INPUT  = REPO / "benchmark_b3lyp_def2tzvp" / "input_n12.json"
PTC_TIMING_OUT  = REPO / "benchmark_b3lyp_def2tzvp" / "ptc_timing_n12.json"
AUDIT_REPORT    = REPO / "benchmarkb3lyp" / "audit_dexp_2026-05-03.json"
AUDIT_COMBINED  = REPO / "benchmarkb3lyp" / "audit_combined_2026-05-04.json"
OUT_PATH        = REPO / "ptc_app" / "dft_comparison_data.json"

# Source URL fields carried through verbatim to the panel data file.
# url_primary/url_pubchem are present for all 1000 mol; url_cccbdb/
# url_nist_webbook for the 146 mol with high-precision provenance.
SOURCE_FIELDS = (
    "source", "source_origin", "cas", "unc_eV",
    "url_primary", "url_pubchem", "url_nist_webbook",
    "url_cccbdb", "url_source_master",
)

# Match the 260503_benchmark_B3LYP_860.json screening
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


def _key(row: dict) -> tuple[str, str]:
    """Composite key (name, smiles) — avoids case-collisions like CoCl2/COCl2."""
    return (row.get("name", ""), row.get("smiles", ""))


def _measure_ptc_timing(input_path: Path, out_path: Path) -> dict | None:
    """Time PTC on the exact same 536 mol of the def2-TZVP n12 set.

    Runs ~1-2 seconds total. Returns per-bin and overall timings, also
    written to ``out_path`` for reuse. Skips silently if PTC modules
    cannot be imported.
    """
    if not input_path.exists():
        return None
    # Make the PTC package importable when running this script from anywhere
    import sys
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    try:
        from ptc.topology import build_topology
        from ptc.cascade_v5 import compute_D_at_cascade
    except Exception as e:
        print(f"[timing] PTC modules unavailable, keeping existing PTC time: {e}")
        return None

    import time
    from collections import defaultdict
    with open(input_path) as f:
        mols = json.load(f)["molecules"]

    times_per_bin: dict[str, list[float]] = defaultdict(list)
    fail = 0
    t0 = time.perf_counter()
    for m in mols:
        t1 = time.perf_counter()
        try:
            compute_D_at_cascade(build_topology(m["smiles"]))
        except Exception:
            fail += 1
        dt = time.perf_counter() - t1
        nat = m.get("n_atoms", 0)
        bn = "A (n=2-4)" if nat <= 4 else ("B (n=5-8)" if nat <= 8 else "C (n=9-12)")
        times_per_bin[bn].append(dt)
    total_s = time.perf_counter() - t0

    out = {
        "n_total": len(mols),
        "total_s": total_s,
        "avg_ms_per_mol": total_s / len(mols) * 1000,
        "failures": fail,
        "per_bin": {
            bn: {
                "n":              len(ts),
                "total_s":        sum(ts),
                "avg_ms_per_mol": sum(ts) / len(ts) * 1000,
                "max_ms":         max(ts) * 1000,
            }
            for bn, ts in times_per_bin.items()
        },
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[timing] PTC on 536 mol : {total_s:.3f} s "
          f"({out['avg_ms_per_mol']:.2f} ms/mol)")
    return out


def _count_atoms(m: dict, b3_row: dict | None = None) -> int | None:
    """Best-effort heavy+H atom count.

    Strategy:
      1. Use n_atoms from B3LYP result if available (already AddHs'd)
      2. Use RDKit AddHs when possible
      3. Fallback: SMILES token count (under-counts implicit H)

    Result is used to pick the right NIST WebBook Mask:
      - n_atoms == 2 -> Mask=400 (Constants of diatomic molecules, has D0)
      - n_atoms >  2 -> Mask=1   (Thermochemistry, has Hf -> derives D_at)
    """
    if b3_row and b3_row.get("n_atoms"):
        return int(b3_row["n_atoms"])
    smi = m.get("smiles", "")
    try:
        from rdkit import Chem
        from rdkit.RDLogger import DisableLog, EnableLog
        DisableLog("rdApp.*")
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mol = Chem.AddHs(mol)
                return mol.GetNumAtoms()
        finally:
            EnableLog("rdApp.*")
    except Exception:
        pass
    import re
    return len(re.findall(r"\[[^\]]+\]|Cl|Br|[BCNOFPSI]", smi))


def main() -> None:
    with open(RESULTS_B3LYP) as f:
        b3lyp_raw = json.load(f)
    with open(PTC_FRESH) as f:
        ptc_raw = json.load(f)
    bench_path = BENCHMARK_LIST if BENCHMARK_LIST.exists() else BENCHMARK_LIST_FALLBACK
    with open(bench_path) as f:
        benchmark = json.load(f)
    print(f"Using benchmark list: {bench_path.name}")

    # Load reliability audit if present (cross-check vs CCCBDB)
    audit_idx: dict[tuple[str, str], dict] = {}
    if AUDIT_REPORT.exists():
        with open(AUDIT_REPORT) as f:
            audit = json.load(f)
        for r in audit.get("results", []):
            audit_idx[(r["name"], r["smiles"])] = r
        print(f"Loaded reliability audit for {len(audit_idx)} mol")

    # Load combined CCCBDB + Burcat audit if present
    combined_idx: dict[tuple[str, str], dict] = {}
    if AUDIT_COMBINED.exists():
        with open(AUDIT_COMBINED) as f:
            comb = json.load(f)
        for r in comb.get("results", []):
            combined_idx[(r["name"], r["smiles"])] = r
        print(f"Loaded combined CCCBDB+Burcat audit for {len(combined_idx)} mol")

    # Composite key (name, smiles) — case-sensitive on name, prevents the
    # CoCl2/COCl2 collision (Cobalt(II) chloride vs phosgene).
    b3lyp_idx: dict[tuple[str, str], dict] = {_key(r): r for r in b3lyp_raw["results"]}
    ptc_idx:   dict[tuple[str, str], dict] = {_key(r): r for r in ptc_raw["results"]}

    rows: list[dict] = []
    n_b3lyp_ok = 0
    n_b3lyp_screened = 0
    n_b3lyp_unsupported = 0
    n_b3lyp_scf_fail = 0
    n_ptc_only = 0
    n_b3lyp_outlier = 0

    for m in benchmark:
        key = _key(m)
        ptc = ptc_idx.get(key)
        b3 = b3lyp_idx.get(key)

        if ptc is None:
            # PTC missing — skip (shouldn't happen on the 1000 set)
            continue

        # Decide on B3LYP value (with ZPE) or None
        D_b3lyp = None
        err_b3lyp = None
        time_b3lyp = None

        if b3 is None:
            # element-excluded a priori (Ag, I, transition metals...)
            n_b3lyp_unsupported += 1
        elif b3.get("status") != "ok":
            n_b3lyp_scf_fail += 1
        elif b3["name"] in FAIL_NAMES:
            # legacy listed fail (already covered by status check)
            n_b3lyp_scf_fail += 1
        elif (b3["name"], b3.get("err_eV")) in PATHO_PAIRS:
            n_b3lyp_outlier += 1
        else:
            # use ZPE-corrected D_at (this is the FIX)
            D_b3lyp = float(b3["D_b3lyp_eV"])
            err_b3lyp = (D_b3lyp - m["D_exp"]) / m["D_exp"] * 100.0
            time_b3lyp = b3.get("time_s")
            n_b3lyp_ok += 1

        if D_b3lyp is None:
            n_b3lyp_screened += 1

        row = {
            "name":     m["name"],
            "smiles":   m["smiles"],
            "D_exp":    m["D_exp"],
            "category": m["category"],
            "n_atoms":  _count_atoms(m, b3),
            "D_ptc":    ptc["d_pt"],
            "err_ptc":  (ptc["d_pt"] - ptc["d_exp"]) / ptc["d_exp"] * 100.0,
            "D_b3lyp":  D_b3lyp,
            "err_b3lyp": err_b3lyp,
            "time_b3lyp": time_b3lyp,
        }
        # Carry source/URL provenance fields verbatim if present
        for fld in SOURCE_FIELDS:
            if fld in m:
                row[fld] = m[fld]
        # Reliability badge from CCCBDB audit (legacy single-source)
        ar = audit_idx.get((m["name"], m["smiles"]))
        if ar:
            row["audit_status"]      = ar.get("status")
            row["audit_diff_pct"]    = ar.get("best_diff_pct")
            row["audit_convention"]  = ar.get("best_convention")
            row["audit_reason"]      = ar.get("reason") or ""
        else:
            row["audit_status"]      = "n/a"
            row["audit_diff_pct"]    = None
            row["audit_convention"]  = None
            row["audit_reason"]      = "no_audit"

        # Combined CCCBDB + Burcat audit
        cr = combined_idx.get((m["name"], m["smiles"]))
        if cr:
            row["combined_status"]     = cr.get("combined_status")  # ok_consensus / ok_one_source / one_source_fail / single_fail / conflict / n/a
            row["combined_best_diff"]  = cr.get("best_diff_pct")
            row["cccbdb_status"]       = cr.get("cccbdb_status")
            row["cccbdb_diff_pct"]     = cr.get("cccbdb_diff_pct")
            row["burcat_status"]       = cr.get("burcat_status")
            row["burcat_diff_pct"]     = cr.get("burcat_diff_pct")
        rows.append(row)

    # Stats
    ptc_errs_all = [abs(r["err_ptc"]) for r in rows if r["err_ptc"] is not None]
    overlap = [r for r in rows if r["D_b3lyp"] is not None]
    ptc_errs_overlap = [abs(r["err_ptc"]) for r in overlap]
    b3lyp_errs_overlap = [abs(r["err_b3lyp"]) for r in overlap]

    # ── Load def2-TZVP 3-way × 3-bin comparison if available ────────────
    def2tzvp_block = None
    if DEF2TZVP_3WAY.exists():
        with open(DEF2TZVP_3WAY) as f:
            tz3 = json.load(f)

        # Measure PTC timing inline on the same 536 mol (~1-2 s total)
        ptc_timing = _measure_ptc_timing(DEF2TZVP_INPUT, PTC_TIMING_OUT)

        # Pull per-molecule def2-TZVP runtimes from the n12 results
        n12_results_path = DEF2TZVP_3WAY.parent / "results_def2tzvp_n12.json"
        runtimes_per_bin: dict = {}
        runtime_overall = None
        if n12_results_path.exists():
            with open(n12_results_path) as f:
                n12_raw = json.load(f)
            runtime_overall = {
                "total_s": n12_raw.get("elapsed_s"),
                "n_done":  n12_raw.get("n_done"),
                "per_mol_avg_s": (
                    n12_raw["elapsed_s"] / n12_raw["n_done"]
                    if n12_raw.get("n_done") else None
                ),
            }
            bins_runtime = {"A (n=2-4)": [], "B (n=5-8)": [], "C (n=9-12)": []}
            for r in n12_raw["results"]:
                t = r.get("time_s")
                if t is None:
                    continue
                nat = r.get("n_atoms", 0)
                if nat <= 4:
                    bins_runtime["A (n=2-4)"].append(t)
                elif nat <= 8:
                    bins_runtime["B (n=5-8)"].append(t)
                else:
                    bins_runtime["C (n=9-12)"].append(t)
            for bn, ts in bins_runtime.items():
                if ts:
                    bin_data = {
                        "n":              len(ts),
                        "total_s":        sum(ts),
                        "per_mol_avg_s":  sum(ts) / len(ts),
                        "per_mol_max_s":  max(ts),
                    }
                    # Pair PTC timing for the same bin (if measured)
                    if ptc_timing and bn in ptc_timing.get("per_bin", {}):
                        ptc_avg_ms = ptc_timing["per_bin"][bn]["avg_ms_per_mol"]
                        bin_data["ptc_per_mol_ms"] = ptc_avg_ms
                        bin_data["speedup"] = (
                            bin_data["per_mol_avg_s"] * 1000 / ptc_avg_ms
                        )
                    runtimes_per_bin[bn] = bin_data

        # Overall PTC timing — use freshly measured if available, else fall
        # back to the legacy 2.97 ms constant for backward compat.
        ptc_per_mol_ms = (
            ptc_timing["avg_ms_per_mol"] if ptc_timing else 2.97
        )
        ptc_total_s = ptc_timing["total_s"] if ptc_timing else None

        # Reference timings on a stable hardware (Apple M1 Max), so the
        # numbers shown in the panel don't drift with whichever machine
        # regenerates the data. Source: user-supplied measurements,
        # 2026-05-03, MacBook Pro Apple M1 Max.
        reference_m1_max = {
            "platform":     "MacBook Pro Apple M1 Max",
            "date":         "2026-05-03",
            "per_bin": {
                "A+B (n≤8)": {
                    "n":                369,
                    "def2tzvp_total_s": 1853.4,
                    "def2tzvp_per_mol_s": 1853.4 / 369,
                    "ptc_total_s":      0.870,
                    "ptc_per_mol_ms":   0.870 / 369 * 1000,
                    "speedup":          (1853.4 / 369) / (0.870 / 369),  # = 2130×
                },
                "C (n=9-12)": {
                    "n":                167,
                    "def2tzvp_total_s": 5117.4,
                    "def2tzvp_per_mol_s": 5117.4 / 167,
                    "ptc_total_s":      0.311,
                    "ptc_per_mol_ms":   0.311 / 167 * 1000,
                    "speedup":          (5117.4 / 167) / (0.311 / 167),  # = 16455×
                },
                "Total (n≤12)": {
                    "n":                536,
                    "def2tzvp_total_s": 6970.8,
                    "def2tzvp_per_mol_s": 6970.8 / 536,
                    "ptc_total_s":      1.182,
                    "ptc_per_mol_ms":   1.182 / 536 * 1000,
                    "speedup":          (6970.8 / 536) / (1.182 / 536),  # = 5897×
                },
            },
            "note": (
                "Measurements on a Apple M1 Max — kept here as a stable "
                "reference. The inline PTC timing recomputed at regen time "
                "may differ slightly depending on the machine running the "
                "regen script."
            ),
        }

        def2tzvp_block = {
            "n_total":          tz3["n_total"],
            "bins":             {},
            "molecules":        [],
            "method":           "B3LYP/def2-TZVP",
            "geometry_source":  "frozen from B3LYP/6-31G* run",
            "zpe_source":       "recycled from B3LYP/6-31G* (Scott-Radom 0.9806)",
            "note":             "3-bin segmented comparison (A: n=2-4, B: n=5-8, C: n=9-12)",
            "runtimes": {
                "overall":           runtime_overall,
                "per_bin":           runtimes_per_bin,
                "ptc_per_mol_ms":    ptc_per_mol_ms,
                "ptc_total_s":       ptc_total_s,
                "ptc_measured_inline": ptc_timing is not None,
                "reference_m1_max":  reference_m1_max,
                "ptc_runtime_note":  (
                    "PTC time mesured inline on the exact same 536 mol "
                    "(typically 1-2 seconds total). def2-TZVP timings are "
                    "single-point only (no Hessian since ZPE is recycled "
                    "from the 6-31G* run); a Hessien complet at def2-TZVP "
                    "would add 5-20× more compute on top of these numbers. "
                    "The reference_m1_max block fixes a stable baseline "
                    "independent of the machine running this regen script."
                ),
            },
        }
        # Per-bin metrics
        for bin_name, bd in tz3["bins"].items():
            n = bd["n"]
            def2tzvp_block["bins"][bin_name] = {
                "n":     n,
                "PTC":   {
                    "mae_pct":    bd["methods"]["PTC"]["mae_pct"],
                    "mae_eV":     bd["methods"]["PTC"]["mae_eV"],
                    "median_eV":  bd["methods"]["PTC"]["median_eV"],
                    "within_1kcal":  bd["methods"]["PTC"]["within_1kcal"],
                    "mean_signed": bd["methods"]["PTC"]["mean_signed"],
                    "pct_negative": bd["methods"]["PTC"]["pct_negative"],
                    "max_eV":      bd["methods"]["PTC"]["max_eV"],
                },
                "B3LYP_def2-TZVP": {
                    "mae_pct":    bd["methods"]["B3LYP/def2-TZVP"]["mae_pct"],
                    "mae_eV":     bd["methods"]["B3LYP/def2-TZVP"]["mae_eV"],
                    "median_eV":  bd["methods"]["B3LYP/def2-TZVP"]["median_eV"],
                    "within_1kcal":  bd["methods"]["B3LYP/def2-TZVP"]["within_1kcal"],
                    "mean_signed": bd["methods"]["B3LYP/def2-TZVP"]["mean_signed"],
                    "pct_negative": bd["methods"]["B3LYP/def2-TZVP"]["pct_negative"],
                    "max_eV":      bd["methods"]["B3LYP/def2-TZVP"]["max_eV"],
                },
                "B3LYP_6-31Gs": {
                    "mae_pct":    bd["methods"]["B3LYP/6-31G*"]["mae_pct"],
                    "mae_eV":     bd["methods"]["B3LYP/6-31G*"]["mae_eV"],
                    "median_eV":  bd["methods"]["B3LYP/6-31G*"]["median_eV"],
                    "within_1kcal":  bd["methods"]["B3LYP/6-31G*"]["within_1kcal"],
                    "mean_signed": bd["methods"]["B3LYP/6-31G*"]["mean_signed"],
                    "pct_negative": bd["methods"]["B3LYP/6-31G*"]["pct_negative"],
                    "max_eV":      bd["methods"]["B3LYP/6-31G*"]["max_eV"],
                },
                "head_to_head": {
                    "PTC_vs_def2TZVP_pct":   100*bd["head_to_head"]["PTC_vs_def2TZVP"]/n,
                    "PTC_vs_6-31Gs_pct":     100*bd["head_to_head"]["PTC_vs_6-31Gs"]/n,
                    "def2TZVP_vs_6-31Gs_pct": 100*bd["head_to_head"]["def2TZVP_vs_6-31Gs"]/n,
                },
                "three_way_best_pct": {
                    "PTC":       100*bd["three_way_best"]["PTC"]/n,
                    "def2-TZVP": 100*bd["three_way_best"]["def2-TZVP"]/n,
                    "6-31Gs":    100*bd["three_way_best"]["6-31Gs"]/n,
                },
            }
        # Per-molecule (just the essentials, for table)
        for tr in tz3["triples"]:
            def2tzvp_block["molecules"].append({
                "name":      tr["name"],
                "smiles":    tr["smiles"],
                "category":  tr["category"],
                "n_atoms":   tr["n_atoms"],
                "D_exp":     tr["D_exp"],
                "D_ptc":     tr["D_ptc_eV"],
                "D_b3lyp_6-31Gs":   tr["D_b6_eV"],
                "D_b3lyp_def2TZVP": tr["D_tz_eV"],
                "err_ptc_eV":            tr["err_ptc_eV"],
                "err_b3lyp_6-31Gs_eV":   tr["err_b6_eV"],
                "err_b3lyp_def2TZVP_eV": tr["err_tz_eV"],
                "err_ptc_pct":            tr["pct_ptc"],
                "err_b3lyp_6-31Gs_pct":   tr["pct_b6"],
                "err_b3lyp_def2TZVP_pct": tr["pct_tz"],
            })

    payload = {
        "all_ptc": rows,
        "n_total": len(rows),
        "n_with_b3lyp": len(overlap),
        "n_fail_b3lyp": n_b3lyp_screened,
        "n_b3lyp_unsupported_basis": n_b3lyp_unsupported,
        "n_b3lyp_scf_fail":          n_b3lyp_scf_fail,
        "n_b3lyp_pathological":      n_b3lyp_outlier,
        "ptc_mae_all":         statistics.mean(ptc_errs_all) if ptc_errs_all else 0.0,
        "ptc_median_all":      statistics.median(ptc_errs_all) if ptc_errs_all else 0.0,
        "ptc_mae_overlap":     statistics.mean(ptc_errs_overlap) if ptc_errs_overlap else 0.0,
        "b3lyp_mae_overlap":   statistics.mean(b3lyp_errs_overlap) if b3lyp_errs_overlap else 0.0,
        "b3lyp_median_overlap": statistics.median(b3lyp_errs_overlap) if b3lyp_errs_overlap else 0.0,
        "ptc_time_ms": 2.97,
        "b3lyp_avg_time_s": statistics.mean(
            r["time_b3lyp"] for r in rows if r["time_b3lyp"] is not None
        ),
        "basis": "6-31g*",
        "zpe_scaled": 0.9806,
        "dataset": "benchmark_1000.json (fixed, same as Benchmark tab)",
        "screening_rules": (
            "B3LYP results filtered: 9 SCF/Hessian failures + 8 catastrophic "
            "|err|>5 eV outliers (S2F10, PCl5, SF6 x2, SF4, HBO2, phenanthrene, "
            "SO3mol) excluded from D_b3lyp. Unsupported elements (Ag, Au, "
            "transition metals, I) never ran B3LYP/6-31G*."
        ),
        "fix_note": (
            "Regenerated 2026-05-03: previous panel data used D_elec (no ZPE) "
            "instead of D_at = D_elec - ZPE_scaled (Scott-Radom 0.9806). "
            "This inflated B3LYP MAE by ~1.1 percentage points. Also fixed a "
            "name collision bug (CoCl2/COCl2 case-insensitive match) that "
            "matched cobalt(II) chloride against phosgene's B3LYP value. Both fixed."
        ),
        "def2tzvp_3way": def2tzvp_block,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=1)

    print(f"Wrote {OUT_PATH}")
    print(f"\n=== Summary ===")
    print(f"  Total molecules in benchmark : {len(rows)}")
    print(f"  PTC computed                 : {len(ptc_errs_all)} (MAE {payload['ptc_mae_all']:.3f}%)")
    print(f"  B3LYP overlap (head-to-head) : {len(overlap)}")
    print(f"  B3LYP unsupported basis      : {n_b3lyp_unsupported}")
    print(f"  B3LYP SCF/Hessian failures   : {n_b3lyp_scf_fail}")
    print(f"  B3LYP pathological outliers  : {n_b3lyp_outlier}")
    print(f"\n=== Head-to-head MAE (n={len(overlap)}) ===")
    print(f"  PTC   MAE : {payload['ptc_mae_overlap']:.3f} %")
    print(f"  B3LYP MAE : {payload['b3lyp_mae_overlap']:.3f} %")
    ptc_wins = sum(1 for r in overlap if abs(r["err_ptc"]) < abs(r["err_b3lyp"]))
    print(f"  PTC wins  : {ptc_wins}/{len(overlap)} ({100*ptc_wins/len(overlap):.1f} %)")


if __name__ == "__main__":
    main()
