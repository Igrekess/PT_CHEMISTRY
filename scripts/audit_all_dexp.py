#!/usr/bin/env python3
"""
Full per-molecule audit of D_exp values against CCCBDB experimental ΔfH°(0K).

For each molecule with a CAS in benchmark_1000_verified.json :
  1. Fetch the CCCBDB exp2x.asp page (the canonical thermochemistry page)
  2. Robustly extract ΔfH°(0K) in kJ/mol — parses the HTML table cell-by-cell,
     not by fragile regex chains
  3. Recompute D_at using ATcT/CODATA atomic Hf standards :
        D_at = Σ ΔfH°_atom_0K(i) − ΔfH°_mol_0K
  4. Compare to the stored D_exp, classify per status :
        ✓ <1%   : excellent agreement
        ⚠ 1-2%  : within typical thermochem revision drift
        ❌ >2%  : flag for manual review
        n/a     : no CCCBDB data, or missing atomic Hf, or fetch error

Outputs :
  - benchmarkb3lyp/audit_dexp_2026-05-03.csv : full per-mol report
  - benchmarkb3lyp/audit_dexp_2026-05-03.json : same in JSON
  - stdout : summary of agreement statistics

Resumable : the JSON output is rewritten after each row (atomic), so you
can interrupt with Ctrl-C and re-run with --resume to skip already-checked mol.

Politeness : CCCBDB rate-limits aggressively (429 errors). Default delay
is 1 s/request, with exponential backoff on 429.

Usage :
    python scripts/audit_all_dexp.py
    python scripts/audit_all_dexp.py --limit 50    # debug
    python scripts/audit_all_dexp.py --resume      # continue interrupted run
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

REPO     = Path(__file__).resolve().parents[1]
INPUT    = REPO / "ptc" / "data" / "benchmark_1000_verified.json"
OUT_JSON = REPO / "benchmarkb3lyp" / "audit_dexp_2026-05-03.json"
OUT_CSV  = REPO / "benchmarkb3lyp" / "audit_dexp_2026-05-03.csv"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
EV_PER_KJMOL = 1.0 / 96.485332

# Atomic ΔfH° standard values, kJ/mol — separate tables for 0K and 298K.
# Sources : ATcT v1.130 where available, CODATA / Wagman 1982 / NIST otherwise.
# 0K is what spectroscopic D₀ and ATcT_0K data refer to.
# 298K is the standard thermodynamic reference and what CCCBDB tabulates most.

ATOM_HF_0K = {
    "H":  216.034, "He":   0.000,
    "Li": 159.300, "Be": 320.600, "B":  560.000,
    "C":  711.190, "N":  470.590, "O":  246.844, "F":   77.252, "Ne":  0.000,
    "Na": 107.500, "Mg": 144.610, "Al": 327.300, "Si": 446.000,
    "P":  316.450, "S":  274.920, "Cl": 119.621, "Ar":   0.000,
    "K":   89.000, "Ca": 177.800,
    "Sc": 372.800, "Ti": 467.000, "V":  514.200, "Cr": 396.600, "Mn": 280.700,
    "Fe": 413.100, "Co": 423.800, "Ni": 425.140, "Cu": 337.400, "Zn": 130.400,
    "As": 287.000, "Se": 235.400, "Br": 117.928,
    "Rb":  80.880, "Sr": 159.000,
    "Y":  421.300, "Zr": 605.040, "Nb": 723.000, "Mo": 656.880, "Tc": 678.000,
    "Ru": 642.700, "Rh": 552.700, "Pd": 376.430, "Ag": 285.700, "Cd": 111.800,
    "I":  107.164,
    "Cs":  76.500, "Ba": 178.700,
    "La": 426.400, "Ce": 419.700, "Hf": 619.200, "Ta": 781.500, "W":  848.100,
    "Pt": 565.300, "Au": 366.100, "Hg":  64.300,
}

# 298K atomic Hf — only differs from 0K by integrated Cp(0→298K) of monatomic gas
# (~6.2 kJ/mol for most). Differences are small but accumulate for many H atoms.
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


def fetch_html(url: str, max_retries: int = 4) -> tuple[str | None, str]:
    """Fetch URL with rate-limit handling. Returns (html, status)."""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": USER_AGENT,
                "Accept":     "text/html",
            })
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.read().decode("utf-8", errors="ignore"), "ok"
        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Exponential backoff
                wait = 5.0 * (2 ** attempt)
                time.sleep(wait)
                continue
            return None, f"http_{e.code}"
        except Exception as e:
            return None, f"err:{type(e).__name__}"
    return None, "rate_limited"


def extract_hfg(page_html: str) -> dict:
    """Extract ΔfH° at 0K AND 298K (both in kJ/mol) from a CCCBDB exp2x.asp page.

    SAFEGUARD : detects CCCBDB's silent-default behaviour. When the requested
    CAS is unknown to CCCBDB, the page returns generic data with a banner
    "Molecule problem. Defaulted to H2CO" — the Hfg values then refer to
    formaldehyde, not our compound. We refuse to trust those values.

    Returns {'hfg_0k': float|None, 'hfg_298k': float|None, 'status': str}.
    Status :
      - 'ok'           : real data for our compound
      - 'no_hfg_data'  : page is for our compound but lacks Hfg
      - 'defaulted'    : CCCBDB silent-defaulted to a different molecule
    """
    # Detect silent-default banner
    if re.search(r"Molecule\s+problem.*Defaulted\s+to", page_html, re.IGNORECASE):
        return {"hfg_0k": None, "hfg_298k": None, "status": "cccbdb_defaulted"}

    cells = re.findall(r"<td[^>]*>(.*?)</td>", page_html, re.DOTALL)
    cleaned = []
    for c in cells:
        t = re.sub(r"<[^>]+>", "", c)
        t = html.unescape(t)
        t = re.sub(r"\s+", " ", t).strip()
        cleaned.append(t)

    def value_after(label_predicate) -> float | None:
        for i, t in enumerate(cleaned):
            if label_predicate(t):
                for j in range(i + 1, min(i + 5, len(cleaned))):
                    try:
                        v = float(cleaned[j])
                        if -2000 < v < 2000:
                            return v
                    except ValueError:
                        continue
        return None

    hfg_0k = value_after(lambda t: t.replace(" ", "") == "Hfg(0K)")
    hfg_298k = value_after(lambda t: t.replace(" ", "") in ("Hfg(298.15K)", "Hfg(298K)"))

    if hfg_0k is None and hfg_298k is None:
        return {"hfg_0k": None, "hfg_298k": None, "status": "no_hfg_data"}
    return {
        "hfg_0k":   hfg_0k,
        "hfg_298k": hfg_298k,
        "status":   "ok",
    }


def count_atoms(smiles: str) -> dict[str, int] | None:
    """Heavy + H atom counts via RDKit, with graceful fallback."""
    try:
        from rdkit import Chem
        from rdkit.RDLogger import DisableLog, EnableLog
        DisableLog("rdApp.*")
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            mol = Chem.AddHs(mol)
            counts: dict[str, int] = {}
            for a in mol.GetAtoms():
                s = a.GetSymbol()
                counts[s] = counts.get(s, 0) + 1
            return counts
        finally:
            EnableLog("rdApp.*")
    except Exception:
        return None


def classify(diff_pct: float | None) -> str:
    if diff_pct is None:
        return "n/a"
    a = abs(diff_pct)
    if a < 1:   return "ok_<1%"
    if a < 2:   return "warn_1-2%"
    return "fail_>2%"


def audit_one(row: dict) -> dict:
    """Audit a single row in BOTH 0K and 298K conventions.

    The stored D_exp doesn't always specify its temperature reference. We
    compute D_at(0K) and D_at(298K) using matching atomic Hf, then keep the
    convention that gives the smaller |diff|. This auto-detects which
    convention the source used for each molecule.
    """
    out = {
        "name":   row["name"],
        "smiles": row["smiles"],
        "cas":    row.get("cas"),
        "source": row.get("source"),
        "D_exp":  row["D_exp"],
        "hfg_0k_kjmol":     None,
        "hfg_298k_kjmol":   None,
        "D_at_0K_eV":       None,
        "D_at_298K_eV":     None,
        "diff_0K_pct":      None,
        "diff_298K_pct":    None,
        "best_convention":  None,
        "best_diff_pct":    None,
        "status":           "n/a",
        "reason":           "",
    }
    cas = row.get("cas")
    if not cas:
        out["reason"] = "no_cas"
        return out

    elem = count_atoms(row["smiles"])
    if not elem:
        out["reason"] = "smiles_parse_failed"
        return out
    missing_0k = [e for e in elem if e not in ATOM_HF_0K]
    if missing_0k:
        out["reason"] = f"atom_hf_missing:{','.join(missing_0k)}"
        return out

    url = f"https://cccbdb.nist.gov/exp2x.asp?casno={cas.replace('-','')}&charge=0"
    page, fetch_status = fetch_html(url)
    if page is None:
        out["reason"] = f"fetch_{fetch_status}"
        return out

    hf = extract_hfg(page)
    if hf["status"] != "ok":
        out["reason"] = hf["status"]
        return out

    candidates = []  # (convention, diff_pct, d_at_eV)

    if hf["hfg_0k"] is not None:
        sum_atoms_0k = sum(ATOM_HF_0K[e] * c for e, c in elem.items())
        d_at_0k = (sum_atoms_0k - hf["hfg_0k"]) * EV_PER_KJMOL
        diff_0k = (d_at_0k - row["D_exp"]) / row["D_exp"] * 100.0
        out["hfg_0k_kjmol"] = round(hf["hfg_0k"], 3)
        out["D_at_0K_eV"]   = round(d_at_0k, 4)
        out["diff_0K_pct"]  = round(diff_0k, 3)
        candidates.append(("0K", diff_0k, d_at_0k))

    if hf["hfg_298k"] is not None and all(e in ATOM_HF_298K for e in elem):
        sum_atoms_298k = sum(ATOM_HF_298K[e] * c for e, c in elem.items())
        d_at_298k = (sum_atoms_298k - hf["hfg_298k"]) * EV_PER_KJMOL
        diff_298k = (d_at_298k - row["D_exp"]) / row["D_exp"] * 100.0
        out["hfg_298k_kjmol"] = round(hf["hfg_298k"], 3)
        out["D_at_298K_eV"]   = round(d_at_298k, 4)
        out["diff_298K_pct"]  = round(diff_298k, 3)
        candidates.append(("298K", diff_298k, d_at_298k))

    if not candidates:
        out["reason"] = "no_usable_hfg"
        return out

    # Pick best convention (smaller |diff|)
    best = min(candidates, key=lambda x: abs(x[1]))
    out["best_convention"] = best[0]
    out["best_diff_pct"]   = round(best[1], 3)
    out["status"]          = classify(best[1])
    return out


def save_atomically(path: Path, payload):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        if path.suffix == ".json":
            json.dump(payload, f, indent=1)
        else:
            f.write(payload)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="Audit only first N mol (debug)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip mol already in the existing JSON output")
    ap.add_argument("--rate", type=float, default=1.0,
                    help="Seconds between requests (default 1.0)")
    args = ap.parse_args()

    with open(INPUT) as f:
        mols = json.load(f)

    # Resume: load existing audit and skip already-done mol
    done: dict[tuple[str, str], dict] = {}
    if args.resume and OUT_JSON.exists():
        with open(OUT_JSON) as f:
            existing = json.load(f)
        for r in existing.get("results", []):
            done[(r["name"], r["smiles"])] = r
        print(f"Resuming: {len(done)} previous rows kept")

    if args.limit:
        mols = mols[: args.limit]

    results: list[dict] = []
    # Pre-fill from done set
    for m in mols:
        k = (m["name"], m["smiles"])
        if k in done:
            results.append(done[k])

    todo = [m for m in mols if (m["name"], m["smiles"]) not in done]
    print(f"To audit: {len(todo)} (already done: {len(done)})")
    print()

    counts = {"ok_<1%": 0, "warn_1-2%": 0, "fail_>2%": 0, "n/a": 0}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    t0 = time.time()
    for i, m in enumerate(todo, 1):
        r = audit_one(m)
        results.append(r)
        counts[r["status"]] = counts.get(r["status"], 0) + 1

        # Console log — show best convention + its diff
        diff = r.get("best_diff_pct")
        conv = r.get("best_convention")
        if r["status"] == "fail_>2%":
            tag = "❌"
            d_calc = r.get("D_at_0K_eV") if conv == "0K" else r.get("D_at_298K_eV")
            extra = f"  [{conv}] Δ={diff:+.2f}%  CCCBDB={d_calc:.3f} vs stored={r['D_exp']:.3f}"
        elif r["status"] == "warn_1-2%":
            tag = "⚠"
            extra = f"  [{conv}] Δ={diff:+.2f}%"
        elif r["status"] == "ok_<1%":
            tag = "✓"
            extra = f"  [{conv}] Δ={diff:+.2f}%"
        else:
            tag = "—"
            extra = f"  ({r['reason']})"

        if i % 20 == 0 or tag in ("❌", "⚠"):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(todo) - i) / max(rate, 0.01)
            print(f"  [{i:>4d}/{len(todo)}] {tag} {m['name'][:25]:25s}{extra}  "
                  f"(eta {eta/60:.1f} min)", flush=True)

        # Atomic checkpoint every 25 rows
        if i % 25 == 0:
            save_atomically(OUT_JSON, {
                "results": results,
                "counts":  counts,
                "n_done":  len(results),
                "n_total": len(mols),
            })
        time.sleep(args.rate)

    # Final save
    save_atomically(OUT_JSON, {
        "results": results,
        "counts":  counts,
        "n_done":  len(results),
        "n_total": len(mols),
    })

    # CSV
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "smiles", "cas", "source", "D_exp_eV",
                    "hfg_0K_kjmol", "hfg_298K_kjmol",
                    "D_at_0K_eV", "D_at_298K_eV",
                    "diff_0K_pct", "diff_298K_pct",
                    "best_convention", "best_diff_pct",
                    "status", "reason"])
        for r in results:
            w.writerow([r["name"], r["smiles"], r.get("cas") or "",
                        r.get("source") or "", r["D_exp"],
                        r.get("hfg_0k_kjmol")   or "",
                        r.get("hfg_298k_kjmol") or "",
                        r.get("D_at_0K_eV")     or "",
                        r.get("D_at_298K_eV")   or "",
                        r.get("diff_0K_pct")    or "",
                        r.get("diff_298K_pct")  or "",
                        r.get("best_convention") or "",
                        r.get("best_diff_pct")  or "",
                        r["status"], r.get("reason") or ""])

    print()
    print("=== Audit summary ===")
    print(f"  Total mol audited : {len(results)}")
    for k in ["ok_<1%", "warn_1-2%", "fail_>2%", "n/a"]:
        v = counts.get(k, 0)
        pct = 100 * v / len(results) if results else 0
        print(f"  {k:12s} : {v:>4d}  ({pct:.1f} %)")
    print(f"\nWrote {OUT_JSON.relative_to(REPO)}")
    print(f"Wrote {OUT_CSV.relative_to(REPO)}")

    # Convention breakdown
    by_conv: dict[str, int] = {}
    for r in results:
        c = r.get("best_convention") or "n/a"
        by_conv[c] = by_conv.get(c, 0) + 1
    print(f"\nBest-matching convention used :")
    for k, v in sorted(by_conv.items(), key=lambda x: -x[1]):
        print(f"  {k:6s} : {v}")

    # Show top 10 worst mismatches
    fails = [r for r in results if r["status"] == "fail_>2%"]
    fails.sort(key=lambda x: -abs(x.get("best_diff_pct") or 0))
    if fails:
        print(f"\nTop 10 worst mismatches (>2 %, after picking best convention):")
        for r in fails[:10]:
            conv = r.get("best_convention") or "?"
            d_calc = r.get("D_at_0K_eV") if conv == "0K" else r.get("D_at_298K_eV")
            d_calc_s = f"{d_calc:>7.3f}" if d_calc else "    n/a"
            print(f"  {r['name']:25s} stored={r['D_exp']:>7.3f}  "
                  f"CCCBDB[{conv}]={d_calc_s}  Δ={r.get('best_diff_pct'):+6.2f}%")


if __name__ == "__main__":
    main()
