#!/usr/bin/env python3
"""
Burcat scope v2 : add formula extraction, electron parity filter,
and a PTC topology feasibility test on a random sample.

Steps :
  1. Parse Burcat for gas-phase species. Extract :
     - CAS, name, HF298 (already done by audit_burcat)
     - Formula from cols 24-44 of the species line (4 elements × 5 chars)
  2. Filter ions, radicals, excited, condensed, isotopes (by name patterns).
  3. Filter electron parity : even Z_total = closed-shell candidate.
  4. Filter : all elements within PTC's coverage (Z = 1..103 implemented).
  5. Dedup vs ATcT verified by CAS.
  6. Test topology resolution on a random sample of the survivors.
  7. Project yield to the full set.
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

BURCAT   = REPO / "data" / "BURCAT.THR.txt"
VERIFIED = REPO / "ptc" / "data" / "benchmark_1000_verified.json"

# Periodic table — Z for parity computation
ZTABLE = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57,
    "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64,
    "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
    "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85,
    "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92,
}

# Filters identical to v1 (by name)
ION_PATTERNS = [
    re.compile(r"(?<=\S)[+\-](?=\s|$)"),
    re.compile(r"\bCATION\b", re.I),
    re.compile(r"\bANION\b", re.I),
    re.compile(r"\bIONIZED?\b", re.I),
    re.compile(r"FLUORONIUM|BROMONIUM|CHLORONIUM|IODONIUM|HYDRONIUM", re.I),
]
RADICAL_PATTERNS = [
    re.compile(r"\bRAD(?:ICAL)?\b", re.I),
    re.compile(r"\.\.\."),
]
EXCITED_PATTERNS = [
    re.compile(r"\bEX(?:CITED)?\b", re.I),
    re.compile(r"\(EX\)"),
    re.compile(r"\bMETASTABLE\b", re.I),
    re.compile(r"\bSINGLET\b", re.I),
    re.compile(r"\bTRIPLET\b", re.I),
    re.compile(r"\*"),
]
CONDENSED_PATTERNS = [
    re.compile(r"\(cr\)", re.I),
    re.compile(r"\(s\)$", re.I),
    re.compile(r"\(l\)$", re.I),
    re.compile(r"\(aq\)", re.I),
]
ISOTOPE_PATTERNS = [
    re.compile(r"\bTRITIUM\b", re.I),
    re.compile(r"\bDEUTERATED\b", re.I),
    re.compile(r"^T\b"),
    re.compile(r"\bD2\b"),
]


def classify_name(name: str) -> str:
    n = name.strip()
    for p in ION_PATTERNS:
        if p.search(n): return "ion"
    for p in RADICAL_PATTERNS:
        if p.search(n): return "radical"
    for p in EXCITED_PATTERNS:
        if p.search(n): return "excited"
    for p in CONDENSED_PATTERNS:
        if p.search(n): return "condensed"
    for p in ISOTOPE_PATTERNS:
        if p.search(n): return "isotope"
    return "ok"


def parse_burcat_with_formula(path: Path) -> list[dict]:
    """
    Parse Burcat returning a list of {cas, name, hf298_kjmol, formula}.
    Formula is dict {element_symbol: count}, extracted from cols 24-44 of
    the species line. Skip 5+ element species (Burcat marks them WARNING).
    """
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    block_starts: list[int] = []
    for i in range(len(lines) - 3):
        l1, l2, l3, l4 = lines[i], lines[i+1], lines[i+2], lines[i+3]
        if (len(l1) >= 80 and l1.rstrip().endswith("1")
            and l2.rstrip().endswith("2")
            and l3.rstrip().endswith("3")
            and l4.rstrip().endswith("4")):
            block_starts.append(i)

    cas_re   = re.compile(r"^\s*(\d{2,7}-\d{2}-\d)(?:\s+or\s+\d|\s*$)")
    hf298_re = re.compile(r"HF298\s*=\s*(-?\d+\.?\d*)\s*(?:\+/?-)?\s*\d*\.?\d*\s*(kJ|kcal)?",
                          re.IGNORECASE)
    brace_re = re.compile(r"\{[^}]*\}")

    out: list[dict] = []
    for bs in block_starts:
        l1 = lines[bs]
        phase = l1[44:45] if len(l1) > 44 else ''
        if phase != 'G':
            continue

        # Extract formula from cols 24-44 (0-indexed: 24..43, 4 fields × 5 chars)
        composition = l1[24:44] if len(l1) >= 44 else ''
        if "WARNING" in composition or "warning" in composition:
            continue  # 5+ element species — skip in this scope
        formula = {}
        formula_ok = True
        for k in range(4):
            field = composition[k*5:(k+1)*5]
            if len(field) < 5:
                continue
            sym_raw = field[0:2].strip()
            cnt_str = field[2:5].strip().rstrip('.')
            if not sym_raw:
                continue
            # Burcat uses all-caps element symbols (CL, BR, AL...) — normalize
            sym = sym_raw[0].upper() + (sym_raw[1].lower() if len(sym_raw) > 1 else "")
            if sym in ('E', 'El'):   # electron — ion marker
                formula_ok = False
                break
            if sym in ('D', 'T'):    # deuterium / tritium isotopes
                formula_ok = False
                break
            try:
                cnt = int(cnt_str)
            except ValueError:
                formula_ok = False
                break
            if cnt > 0:
                formula[sym] = formula.get(sym, 0) + cnt
        if not formula_ok or not formula:
            continue

        # Walk back for HF298 + CAS (same logic as audit_burcat)
        cas, hf298, hf298_unit = None, None, None
        for k in range(bs - 1, max(-1, bs - 30), -1):
            line = lines[k]
            stripped = line.strip()
            if not stripped:
                break
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

        if cas and hf298 is not None:
            if hf298_unit == "kcal":
                hf298 = hf298 * 4.184
            out.append({
                "cas": cas,
                "name": l1[:18].strip(),
                "hf298_kjmol": hf298,
                "formula": formula,
            })
    return out


def total_electrons(formula: dict[str, int]) -> int | None:
    """Return total Z for a neutral species (= electron count). None if unknown element."""
    z = 0
    for sym, cnt in formula.items():
        if sym not in ZTABLE:
            return None
        z += ZTABLE[sym] * cnt
    return z


def formula_str(formula: dict[str, int]) -> str:
    """Hill-system canonical formula string (C first, H second, others alpha)."""
    items = []
    if "C" in formula:
        items.append(("C", formula["C"]))
        if "H" in formula:
            items.append(("H", formula["H"]))
        for s in sorted(k for k in formula if k not in ("C", "H")):
            items.append((s, formula[s]))
    else:
        for s in sorted(formula):
            items.append((s, formula[s]))
    return "".join(s + (str(c) if c > 1 else "") for s, c in items)


def main():
    print("=" * 70)
    print("Burcat scope v2 : formula + electron parity + topology test")
    print("=" * 70)

    # ── 1. Parse Burcat with formula ──────────────────────────────
    rows = parse_burcat_with_formula(BURCAT)
    print(f"\n[1] Parsed {len(rows)} gas-phase entries with extractable formula")

    # Dedup by CAS (keep first)
    seen = set()
    unique = []
    for r in rows:
        if r["cas"] in seen:
            continue
        seen.add(r["cas"])
        unique.append(r)
    print(f"    {len(unique)} unique CAS")

    # ── 2. Name-based filter ──────────────────────────────────────
    counts = Counter()
    for r in unique:
        cat = classify_name(r["name"])
        r["name_class"] = cat
        counts[cat] += 1
    print(f"\n[2] Name classification :")
    for k, v in counts.most_common():
        print(f"    {k:12s} : {v:>5d}")

    keep_name = [r for r in unique if r["name_class"] == "ok"]
    print(f"    -> {len(keep_name)} survive name filter")

    # ── 3. Electron parity filter ─────────────────────────────────
    closed_shell = []
    odd = 0
    unknown = 0
    for r in keep_name:
        n_e = total_electrons(r["formula"])
        if n_e is None:
            unknown += 1
            continue
        if n_e % 2 == 0:
            r["n_electrons"] = n_e
            r["formula_str"] = formula_str(r["formula"])
            closed_shell.append(r)
        else:
            odd += 1
    print(f"\n[3] Electron parity filter :")
    print(f"    odd electron count (radical) : {odd}")
    print(f"    unknown element (rare)       : {unknown}")
    print(f"    even electron count (closed) : {len(closed_shell)}")

    # ── 4. PTC element coverage filter ────────────────────────────
    # PTC atom.py covers Z 1..118 in principle, but molecular pipeline
    # is best-tested for Z ≤ 86 (main group + d-block + 4f start).
    # Be inclusive : keep all up to Z=92 (U), flag heavier separately.
    ptc_supported = []
    flagged_heavy = []
    for r in closed_shell:
        z_max = max(ZTABLE.get(s, 999) for s in r["formula"])
        if z_max <= 86:
            r["z_max"] = z_max
            ptc_supported.append(r)
        elif z_max <= 92:
            r["z_max"] = z_max
            flagged_heavy.append(r)
    print(f"\n[4] PTC element coverage (Z ≤ 86 main, Z ≤ 92 actinides) :")
    print(f"    main-group + d-block + light f-block : {len(ptc_supported)}")
    print(f"    actinides (Z 87-92, flag)            : {len(flagged_heavy)}")

    # ── 5. Dedup vs ATcT ──────────────────────────────────────────
    with open(VERIFIED) as f:
        verified = json.load(f)
    rows_v = verified["rows"] if isinstance(verified, dict) and "rows" in verified else verified
    verified_cas = {r["cas"] for r in rows_v if r.get("cas")}

    new = [r for r in ptc_supported if r["cas"] not in verified_cas]
    print(f"\n[5] Dedup vs ATcT verified ({len(rows_v)} entries, {len(verified_cas)} with CAS) :")
    print(f"    overlap         : {len(ptc_supported) - len(new)}")
    print(f"    NEW from Burcat : {len(new)}")

    # ── 6. Save & sample ──────────────────────────────────────────
    out_path = REPO / "benchmarkb3lyp" / "burcat_candidates_v2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"n_total": len(new),
         "n_ptc_supported": len(ptc_supported),
         "n_overlap_atct": len(ptc_supported) - len(new),
         "results": new},
        indent=1))
    print(f"\n[6] Wrote -> {out_path.relative_to(REPO)}")

    print(f"\n[7] Random sample of 30 NEW candidates :")
    random.seed(13)
    for r in random.sample(new, min(30, len(new))):
        print(f"    {r['cas']:>14s}  {r['formula_str']:<14s}  Z_total={r['n_electrons']:>4d}  '{r['name']}'")

    print("\n" + "=" * 70)
    print(f"YIELD ESTIMATE before topology test :")
    print(f"  Burcat unique CAS                : {len(unique)}")
    print(f"  After name filter                : {len(keep_name)}")
    print(f"  + closed-shell (electron parity) : {len(closed_shell)}")
    print(f"  + PTC element coverage           : {len(ptc_supported)}")
    print(f"  - overlap with ATcT 994          : {len(ptc_supported) - len(new)}")
    print(f"  ==>  NEW candidates              : {len(new)}")
    print(f"  Combined ATcT + Burcat           : {994 + len(new)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
