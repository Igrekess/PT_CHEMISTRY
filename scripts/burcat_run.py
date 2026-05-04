#!/usr/bin/env python3
"""
Burcat extension benchmark — full run.

Inputs  : burcat_candidates_v2.json (940 NEW closed-shell candidates)
Filters : structural rules excluding classes where PTC has no dedicated
          model yet (transparently declared on the website).
Output  : benchmarkb3lyp/burcat_extension_per_molecule.csv
          benchmarkb3lyp/burcat_extension_metrics.json

Excluded classes (declared) :
  1. Inorganic alkali / alkaline-earth salts (Li,Na,K,Rb,Cs,Be,Mg,Ca,Sr,Ba & no C)
  2. Hypervalent halogenated (>=5 halogens on a single central atom)
  3. p-block homonuclear clusters Xn with n>=3, X in {N, As, Sb}
     (P4 kept — PTC has a phosphorus tetrahedron model)
  4. Small ionic oxides without C or H (<=4 heavy atoms, has O, no CH)
  5. Main-group organometallics (Hg, Pb, Sn, Tl, Bi with >=1 C)
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

# Force unbuffered stdout so we can `tail -f` the log in real time
sys.stdout.reconfigure(line_buffering=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from audit_burcat import ATOM_HF_298K
from ptc.topology_solver import solve_topology
from ptc.transfer_matrix import compute_D_at_transfer

CANDIDATES = REPO / "benchmarkb3lyp" / "burcat_candidates_v2.json"
OUT_CSV    = REPO / "benchmarkb3lyp" / "burcat_extension_per_molecule.csv"
OUT_JSON   = REPO / "benchmarkb3lyp" / "burcat_extension_metrics.json"
OUT_FAIL   = REPO / "benchmarkb3lyp" / "burcat_extension_failures.csv"

CSV_FIELDS = ["cas", "name", "formula", "n_atoms", "hf298_kjmol",
              "d_exp_eV", "d_ptc_eV", "err_eV", "err_pct", "dt_sec"]


def load_done(path: Path) -> tuple[dict[str, dict], set[str]]:
    """Return (existing_rows_by_cas, set_of_done_cas) from a prior run's CSV."""
    if not path.exists():
        return {}, set()
    rows: dict[str, dict] = {}
    done: set[str] = set()
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r["cas"]] = r
            done.add(r["cas"])
    return rows, done


def load_fails(path: Path) -> set[str]:
    """Return set of CAS that previously failed (so we don't retry forever)."""
    if not path.exists():
        return set()
    out: set[str] = set()
    with open(path) as f:
        for r in csv.DictReader(f):
            out.add(r["cas"])
    return out

EV_PER_KJMOL = 1.0 / 96.485332

ALKALI_ALKEARTH = {"Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba"}
HALOGENS        = {"F", "Cl", "Br", "I"}
P_BLOCK_HOMO    = {"N", "As", "Sb"}     # P4 explicitly kept
MAIN_GROUP_HEAVY = {"Hg", "Pb", "Sn", "Tl", "Bi"}


def filter_class(formula: dict[str, int]) -> str | None:
    """Return the name of an excluded class if the formula matches it, else None."""
    elements = set(formula.keys())
    n_C = formula.get("C", 0)
    n_H = formula.get("H", 0)
    heavy_atoms = sum(c for s, c in formula.items() if s != "H")
    n_heavy_kinds = len(elements - {"H"})

    # Rule 1: alkali/alkaline-earth salts without carbon
    if elements & ALKALI_ALKEARTH and n_C == 0:
        return "alkali/alk-earth salt (no C)"

    # Rule 2: hypervalent halogenated — >=5 halogens on a non-C, non-H, non-halogen central atom.
    # Approximation: if there's exactly one non-H, non-halogen atom and >=5 halogens.
    central = elements - HALOGENS - {"H"}
    n_halogens = sum(formula.get(x, 0) for x in HALOGENS)
    if len(central) == 1 and n_halogens >= 5 and "C" not in central:
        return "hypervalent halogenated (>=5 X)"

    # Rule 3: p-block homonuclear cluster Xn (n>=3) with X in N/As/Sb
    if len(formula) == 1:
        sym = next(iter(formula))
        cnt = formula[sym]
        if sym in P_BLOCK_HOMO and cnt >= 3:
            return f"p-block homonuclear cluster {sym}{cnt}"

    # Rule 4: small ionic oxide without CH
    if "O" in elements and n_C == 0 and n_H == 0 and heavy_atoms <= 4:
        return "small ionic oxide (no C, no H)"

    # Rule 5: main-group heavy organometallic
    if elements & MAIN_GROUP_HEAVY and n_C >= 1:
        return "main-group heavy organometallic"

    return None


def main():
    cand = json.load(open(CANDIDATES))["results"]
    print(f"Loaded {len(cand)} NEW Burcat candidates (post electron-parity filter)")

    # ── Apply structural class filter ─────────────────────────────
    excluded: dict[str, list[dict]] = {}
    kept: list[dict] = []
    for r in cand:
        cls = filter_class(r["formula"])
        if cls:
            excluded.setdefault(cls, []).append(r)
        else:
            kept.append(r)
    print(f"\nStructural class filter :")
    for cls, items in sorted(excluded.items(), key=lambda kv: -len(kv[1])):
        print(f"  {cls:<45s} : {len(items):>4d}  (e.g. {items[0]['formula_str']} '{items[0]['name'][:20]}')")
    n_kept = len(kept)
    print(f"\n  Kept after class filter : {n_kept}")

    # ── Apply size cap: skip very large molecules in this pass ────
    SIZE_CAP = 30  # max atoms in formula. Larger ones go in a "macro" pass.
    macro = [r for r in kept if sum(r["formula"].values()) > SIZE_CAP]
    small = [r for r in kept if sum(r["formula"].values()) <= SIZE_CAP]
    print(f"\nSize cap (n_atoms <= {SIZE_CAP}) :")
    print(f"  small molecules : {len(small)}")
    print(f"  macro (>{SIZE_CAP} atoms, deferred) : {len(macro)}")
    if macro[:3]:
        print(f"  e.g. {[r['formula_str'] for r in macro[:3]]}")

    # ── Resume from prior run if CSV exists ───────────────────────
    prior_rows, done_cas = load_done(OUT_CSV)
    prior_fails = load_fails(OUT_FAIL)
    print(f"\nResume :")
    print(f"  rows already in CSV  : {len(done_cas)}")
    print(f"  CAS already failed   : {len(prior_fails)}")

    todo = [r for r in small if r["cas"] not in done_cas and r["cas"] not in prior_fails]
    print(f"  remaining to process : {len(todo)}  (out of {len(small)} small)")

    # ── Open CSV for append (write header if new file) ────────────
    csv_is_new = not OUT_CSV.exists()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    csv_f = open(OUT_CSV, "a", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
    if csv_is_new:
        csv_w.writeheader()
        csv_f.flush()

    fail_is_new = not OUT_FAIL.exists()
    fail_f = open(OUT_FAIL, "a", newline="")
    fail_w = csv.DictWriter(fail_f, fieldnames=["cas", "formula", "err"])
    if fail_is_new:
        fail_w.writeheader()
        fail_f.flush()

    # ── Run PTC on each kept candidate ────────────────────────────
    print(f"\nRunning PTC on {len(todo)} new survivors…")
    rows = list(prior_rows.values())  # keep prior rows for stats
    skipped_atomhf = 0
    failures = []
    slow = []
    t_global = time.time()
    for i, r in enumerate(todo, 1):
        formula = r["formula"]
        # D_at_exp(298K) from Burcat
        if any(s not in ATOM_HF_298K for s in formula):
            skipped_atomhf += 1
            continue
        d_atom_kj = sum(ATOM_HF_298K[s] * c for s, c in formula.items())
        d_at_exp_kj = d_atom_kj - r["hf298_kjmol"]
        d_at_exp_ev = d_at_exp_kj * EV_PER_KJMOL
        # PTC D_at
        t0 = time.time()
        try:
            topo = solve_topology(r["formula_str"])
            res  = compute_D_at_transfer(topo)
            d_at_ptc = float(res.D_at if hasattr(res, "D_at") else res)
            if d_at_ptc != d_at_ptc:        # NaN
                raise ValueError("NaN D_at")
        except Exception as e:
            dt = time.time() - t0
            print(f"  [{i:>4d}/{len(todo)}] FAIL    {r['formula_str']:<14s} ({dt:5.2f}s)  {str(e)[:60]}")
            failures.append({"cas": r["cas"], "formula": r["formula_str"], "err": str(e)[:80]})
            fail_w.writerow({"cas": r["cas"], "formula": r["formula_str"], "err": str(e)[:80]})
            fail_f.flush()
            continue
        dt = time.time() - t0
        if dt > 3:
            slow.append((r["formula_str"], dt))
            print(f"  [{i:>4d}/{len(todo)}] SLOW    {r['formula_str']:<14s} ({dt:5.2f}s)")
        elif i % 25 == 0 or i == 1:
            elapsed = time.time() - t_global
            rate = i / elapsed if elapsed else 0
            eta_s = (len(todo) - i) / rate if rate else 0
            print(f"  [{i:>4d}/{len(todo)}] ok      {r['formula_str']:<14s}  rate={rate:5.1f}/s  ETA={eta_s/60:5.1f} min")

        err_pct = 100 * (d_at_ptc - d_at_exp_ev) / d_at_exp_ev if d_at_exp_ev else None
        new_row = {
            "cas":         r["cas"],
            "name":        r["name"],
            "formula":     r["formula_str"],
            "n_atoms":     sum(formula.values()),
            "hf298_kjmol": r["hf298_kjmol"],
            "d_exp_eV":    round(d_at_exp_ev, 4),
            "d_ptc_eV":    round(d_at_ptc, 4),
            "err_eV":      round(d_at_ptc - d_at_exp_ev, 4),
            "err_pct":     round(err_pct, 3) if err_pct is not None else None,
            "dt_sec":      round(dt, 3),
        }
        rows.append(new_row)
        # Persist immediately (resume-safe)
        csv_w.writerow(new_row)
        csv_f.flush()
    csv_f.close()
    fail_f.close()

    print(f"\nResults :")
    print(f"  PTC computed   : {len(rows)}")
    print(f"  no atom Hf     : {skipped_atomhf}")
    print(f"  PTC failures   : {len(failures)}")
    if failures[:5]:
        print(f"  failure sample :")
        for f in failures[:5]:
            print(f"    {f['cas']:>14s}  {f['formula']:<14s}  {f['err']}")

    # ── Stats (coerce strings from prior CSV rows) ────────────────
    def _fnum(x):
        try:
            return float(x) if x is not None and x != "" else None
        except (TypeError, ValueError):
            return None
    abs_errs = [abs(v) for v in (_fnum(r["err_pct"]) for r in rows) if v is not None]
    abs_eV   = [abs(v) for v in (_fnum(r["err_eV"])  for r in rows) if v is not None]
    if abs_errs:
        print(f"\n  MAE  (relative) : {statistics.mean(abs_errs):.2f} %")
        print(f"  MAE  (absolute) : {statistics.mean(abs_eV):.3f} eV")
        print(f"  Med  (relative) : {statistics.median(abs_errs):.2f} %")
        print(f"  <2 %    : {sum(1 for e in abs_errs if e < 2):>4d} / {len(abs_errs)}")
        print(f"  <5 %    : {sum(1 for e in abs_errs if e < 5):>4d} / {len(abs_errs)}")
        print(f"  <10 %   : {sum(1 for e in abs_errs if e < 10):>4d} / {len(abs_errs)}")
        print(f"  >20 %   : {sum(1 for e in abs_errs if e > 20):>4d} / {len(abs_errs)}")
    print(f"\n  CSV (incremental) : {OUT_CSV.relative_to(REPO)}  ({len(rows)} rows)")
    if (OUT_FAIL).exists() and OUT_FAIL.stat().st_size > 0:
        print(f"  Failures CSV     : {OUT_FAIL.relative_to(REPO)}")

    metrics = {
        "n_candidates_pre_class_filter": len(cand),
        "n_kept_after_class_filter":     n_kept,
        "n_excluded_by_class":           sum(len(v) for v in excluded.values()),
        "excluded_classes":              {k: len(v) for k, v in excluded.items()},
        "n_skipped_no_atom_hf":          skipped_atomhf,
        "n_ptc_failures":                len(failures),
        "n_computed":                    len(rows),
        "mae_pct":     round(statistics.mean(abs_errs), 3) if abs_errs else None,
        "mae_eV":      round(statistics.mean(abs_eV), 4)    if abs_eV   else None,
        "median_pct":  round(statistics.median(abs_errs), 3) if abs_errs else None,
        "n_under_2pct":  sum(1 for e in abs_errs if e < 2),
        "n_under_5pct":  sum(1 for e in abs_errs if e < 5),
        "n_under_10pct": sum(1 for e in abs_errs if e < 10),
        "n_over_20pct":  sum(1 for e in abs_errs if e > 20),
    }
    OUT_JSON.write_text(json.dumps(metrics, indent=2))
    print(f"  Wrote {OUT_JSON.relative_to(REPO)}")


if __name__ == "__main__":
    main()
