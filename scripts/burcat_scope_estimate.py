#!/usr/bin/env python3
"""
Minimal scope: estimate the exploitable Burcat yield for a PTC benchmark.

No network calls, no enrichment — just :
  1. Parse Burcat for all gas-phase species with CAS + HF298
  2. Filter out ions / radicals / excited states by name patterns
  3. Dedup vs the existing ATcT 994-mol verified dataset by CAS
  4. Report a yield breakdown

The goal is to know how many *new* candidates Burcat brings before we
spend any time on the full pipeline (formula -> topology -> PTC run).
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from audit_burcat import parse_burcat  # already extracts {cas: [{name, hf298_kjmol, ...}]}

BURCAT   = REPO / "data" / "BURCAT.THR.txt"
VERIFIED = REPO / "ptc" / "data" / "benchmark_1000_verified.json"

# Patterns that betray an ion / radical / excited state in the Burcat name.
# Burcat is messy — names like "OH", "CH3" can be either the radical or the
# stoichiometric stable molecule (e.g. CH3 alone IS the methyl radical).
ION_PATTERNS = [
    # + or - immediately attached to a chemical token (no preceding letter
    # check needed — Burcat tokens are short). Catches "FH2+", "ALF2-",
    # "BrH2+ Bromonium", "C3+ linear", "Si4H9- see hf2017".
    re.compile(r"(?<=\S)[+\-](?=\s|$)"),
    re.compile(r"\bCATION\b", re.I),
    re.compile(r"\bANION\b", re.I),
    re.compile(r"\bIONIZED?\b", re.I),
    re.compile(r"FLUORONIUM|BROMONIUM|CHLORONIUM|IODONIUM|HYDRONIUM", re.I),
]
RADICAL_PATTERNS = [
    re.compile(r"\bRAD(?:ICAL)?\b", re.I),
    re.compile(r"\.\.\."),              # "..." sometimes used for radicals
]
EXCITED_PATTERNS = [
    re.compile(r"\bEX(?:CITED)?\b", re.I),
    re.compile(r"\(EX\)"),
    re.compile(r"\bMETASTABLE\b", re.I),
    re.compile(r"\bSINGLET\b", re.I),     # explicit singlet/triplet flag
    re.compile(r"\bTRIPLET\b", re.I),
    re.compile(r"\*"),                  # asterisk in name = excited (CH3*, etc)
]
# Condensed phases marked in the name itself (Burcat's phase letter is
# already filtered to G upstream, but some entries still slip through).
CONDENSED_PATTERNS = [
    re.compile(r"\(cr\)", re.I),
    re.compile(r"\(s\)$", re.I),
    re.compile(r"\(l\)$", re.I),
    re.compile(r"\(aq\)", re.I),
]
# Tritium / deuterium-only species: PTC has no isotope handling
ISOTOPE_PATTERNS = [
    re.compile(r"\bTRITIUM\b", re.I),
    re.compile(r"\bDEUTERATED\b", re.I),
    re.compile(r"^T\b"),                 # leading "T" as tritium symbol
    re.compile(r"\bD2\b"),               # D2 = deuterium
]
WARNING_PATTERN = re.compile(r"WARNING", re.I)


def classify(name: str) -> str:
    n = name.strip()
    for p in ION_PATTERNS:
        if p.search(n):
            return "ion"
    for p in RADICAL_PATTERNS:
        if p.search(n):
            return "radical"
    for p in EXCITED_PATTERNS:
        if p.search(n):
            return "excited"
    for p in CONDENSED_PATTERNS:
        if p.search(n):
            return "condensed"
    for p in ISOTOPE_PATTERNS:
        if p.search(n):
            return "isotope"
    if WARNING_PATTERN.search(n):
        return "warning"  # keep but flag
    return "ok"


def main():
    print("=" * 70)
    print("Burcat scope estimate (minimal, no network)")
    print("=" * 70)

    # ── 1. Parse Burcat ────────────────────────────────────────────
    burcat = parse_burcat(BURCAT)
    n_cas = len(burcat)
    n_entries = sum(len(v) for v in burcat.values())
    print(f"\n[1] Burcat gas-phase entries with CAS+HF298 :")
    print(f"    unique CAS : {n_cas}")
    print(f"    total rows : {n_entries}  (some CAS have multiple entries)")

    # Pick the entry with the freshest HF298 per CAS (just take first for now)
    burcat_flat = {cas: entries[0] for cas, entries in burcat.items()}

    # ── 2. Classify by name pattern ────────────────────────────────
    counts = Counter()
    keep: dict[str, dict] = {}
    samples: dict[str, list[str]] = {k: [] for k in ("ion", "radical", "excited", "condensed", "isotope", "warning")}
    for cas, e in burcat_flat.items():
        name = e.get("description", "")
        cat = classify(name)
        counts[cat] += 1
        if cat in samples and len(samples[cat]) < 5:
            samples[cat].append(f"{name}  [CAS {cas}]")
        if cat in ("ok", "warning"):
            keep[cas] = e

    print(f"\n[2] Name-based classification of the {n_cas} unique CAS :")
    for cat in ("ok", "warning", "ion", "radical", "excited", "condensed", "isotope"):
        v = counts.get(cat, 0)
        pct = 100 * v / n_cas if n_cas else 0
        print(f"    {cat:10s} : {v:>5d}  ({pct:>5.1f} %)")

    print(f"\n    Examples of excluded species :")
    for cat in ("ion", "radical", "excited", "condensed", "isotope"):
        if samples[cat]:
            print(f"      {cat}:")
            for s in samples[cat]:
                print(f"        {s}")

    # ── 3. Dedup vs verified ATcT ──────────────────────────────────
    with open(VERIFIED) as f:
        verified = json.load(f)
    rows = verified["rows"] if isinstance(verified, dict) and "rows" in verified else verified
    verified_cas = {r["cas"] for r in rows if r.get("cas")}
    print(f"\n[3] ATcT verified.json :")
    print(f"    total entries : {len(rows)}")
    print(f"    with CAS      : {len(verified_cas)}")

    # Intersection (post name filter)
    cas_kept = set(keep)
    overlap = cas_kept & verified_cas
    new_in_burcat = cas_kept - verified_cas
    burcat_only_no_cas_in_verified = verified_cas - cas_kept

    print(f"\n[4] Overlap analysis (post-filter, by CAS) :")
    print(f"    Burcat kept (ok+warning) : {len(cas_kept)}")
    print(f"    Already in ATcT verified : {len(overlap)}")
    print(f"    NEW from Burcat          : {len(new_in_burcat)}  <-- candidates for extension")
    print(f"    In ATcT but not in Burcat: {len(burcat_only_no_cas_in_verified)}")

    # ── 4. Save the candidate list (for the next pipeline step) ────
    out = REPO / "benchmarkb3lyp" / "burcat_candidates_new.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cand = []
    for cas in sorted(new_in_burcat):
        e = keep[cas]
        cand.append({
            "cas": cas,
            "name": e.get("description"),
            "hf298_kjmol": e.get("hf298_kjmol"),
            "warning": classify(e.get("description", "")) == "warning",
        })
    out.write_text(json.dumps({"n": len(cand), "results": cand}, indent=1))
    print(f"\n[5] Wrote candidate list -> {out.relative_to(REPO)}")
    print(f"    ({len(cand)} new species, sorted by CAS)")

    print("\n" + "=" * 70)
    print(f"BOTTOM LINE :")
    print(f"  ATcT current     : {len(rows)} mol")
    print(f"  Burcat NEW (CAS) : {len(new_in_burcat)} mol")
    print(f"  Combined upper   : {len(rows) + len(new_in_burcat)} mol")
    print(f"  (closed-shell + topology yield will reduce this further)")
    print("=" * 70)


if __name__ == "__main__":
    main()
