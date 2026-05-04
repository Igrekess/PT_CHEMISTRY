#!/usr/bin/env python3
"""
Enrich benchmark_1000_verified.json with CAS numbers + CCCBDB URLs by
querying PubChem PUG REST API.

Why: only 146/1000 molecules have a CAS number in the verified file, which
means only those get a direct CCCBDB link to the experimental atomization
energy. The other 854 fall back to a NIST WebBook search-by-name page that
shows heats of formation, NOT the D_at value the user wants to verify.

PubChem can resolve most chemical names → CAS via its synonyms list :
    https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/synonyms/JSON

This script :
  1. Reads benchmark_1000_verified.json
  2. For every mol with cas == null, queries PubChem (with rate limiting)
  3. Filters synonyms for CAS pattern XXXXXX-XX-X
  4. Writes back the file with updated cas + url_cccbdb + url_nist_webbook
  5. Saves a backup before overwriting

Rate limit: PubChem allows ~5 req/s; we use ~3 req/s to be safe.
Total runtime for 854 lookups: ~5 minutes.

Usage :
    python scripts/enrich_cas_via_pubchem.py            # full run
    python scripts/enrich_cas_via_pubchem.py --dry-run  # preview only
    python scripts/enrich_cas_via_pubchem.py --limit 20 # debug
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VERIFIED = REPO / "ptc" / "data" / "benchmark_1000_verified.json"

CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")

USER_AGENT   = "PTC-benchmark-cas-enrichment/1.0"
SYNS_BY_SMILES = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{q}/synonyms/JSON"
SYNS_BY_NAME   = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{q}/synonyms/JSON"


def _http_json(url: str, timeout: float = 10.0):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _extract_synonyms(data) -> list[str]:
    if not data:
        return []
    infos = data.get("InformationList", {}).get("Information", [])
    if not infos:
        return []
    return infos[0].get("Synonym", []) or []


def fetch_synonyms_smiles(smiles: str) -> list[str]:
    """Return synonyms for the compound matching this exact SMILES."""
    return _extract_synonyms(_http_json(SYNS_BY_SMILES.format(q=urllib.parse.quote(smiles))))


def fetch_synonyms_name(name: str) -> list[str]:
    return _extract_synonyms(_http_json(SYNS_BY_NAME.format(q=urllib.parse.quote(name))))


def resolve_cas(name: str, smiles: str) -> tuple[str | None, str]:
    """Return (cas, method_used). Tries SMILES first (unambiguous), then name variants.

    Returns (None, "failed") if no match.
    """
    # 1. SMILES lookup — unambiguous by construction
    syns = fetch_synonyms_smiles(smiles)
    cas = first_cas_in_synonyms(syns)
    if cas:
        return cas, "smiles"

    # 2. Name as-is
    syns = fetch_synonyms_name(name)
    cas = first_cas_in_synonyms(syns)
    if cas:
        return cas, "name"

    # 3. Name with underscores -> spaces (PubChem doesn't recognise "foo_bar")
    if "_" in name:
        normalised = name.replace("_", " ")
        syns = fetch_synonyms_name(normalised)
        cas = first_cas_in_synonyms(syns)
        if cas:
            return cas, "name_normalised_space"

    # 4. Name with underscores -> hyphens (some chemicals use hyphens, e.g. "n-propyl")
    if "_" in name:
        normalised = name.replace("_", "-")
        syns = fetch_synonyms_name(normalised)
        cas = first_cas_in_synonyms(syns)
        if cas:
            return cas, "name_normalised_hyphen"

    return None, "failed"


def first_cas_in_synonyms(syns: list[str]) -> str | None:
    """Return the first CAS-formatted synonym, or None."""
    for s in syns:
        s = s.strip()
        if CAS_RE.match(s):
            return s
    return None


def cccbdb_url(cas: str) -> str:
    """CCCBDB exp atomization energy URL — direct entry for D_at."""
    cas_no_dashes = cas.replace("-", "")
    return f"https://cccbdb.nist.gov/exp2x.asp?casno={cas_no_dashes}&charge=0"


def nist_url_by_cas(cas: str) -> str:
    return f"https://webbook.nist.gov/cgi/cbook.cgi?ID={urllib.parse.quote(cas)}&Units=SI"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(VERIFIED),
                    help="Path to benchmark_1000_verified.json")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N rows that lack CAS (debug)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Don't write the file back; just print what would change")
    ap.add_argument("--rate", type=float, default=0.3,
                    help="Seconds to wait between PubChem requests (default 0.3 = ~3 req/s)")
    ap.add_argument("--retry", type=int, default=2,
                    help="Number of retries on transient failures (default 2)")
    args = ap.parse_args()

    in_path = Path(args.input)
    with open(in_path) as f:
        mols = json.load(f)

    todo = [m for m in mols if not m.get("cas")]
    print(f"Loaded {len(mols)} molecules. {len(todo)} need CAS lookup.")
    if args.limit:
        todo = todo[: args.limit]
        print(f"Limited to first {len(todo)} (debug)")

    # Backup before any change
    if not args.dry_run:
        backup = in_path.with_name(
            in_path.stem + f".backup_{date.today():%Y-%m-%d}" + in_path.suffix
        )
        if not backup.exists():
            shutil.copy2(in_path, backup)
            print(f"Backup written: {backup}")

    enriched = 0
    failed: list[tuple[str, str]] = []
    by_method: dict[str, int] = {}
    for i, m in enumerate(todo, 1):
        name = m["name"]
        smiles = m.get("smiles", "")
        cas, method = resolve_cas(name, smiles)
        by_method[method] = by_method.get(method, 0) + 1
        if cas:
            m["cas"]              = cas
            m["url_cccbdb"]       = cccbdb_url(cas)
            m["url_nist_webbook"] = nist_url_by_cas(cas)
            enriched += 1
            tag = "✓"
        else:
            failed.append((name, smiles))
            tag = "✗"
        if i % 25 == 0 or i == len(todo) or tag == "✗":
            print(f"  [{i:>4d}/{len(todo)}] {tag} {name:30s} {cas or '(no CAS found)'}  via {method}", flush=True)
        time.sleep(args.rate)
    print()
    print("Resolution methods used:")
    for k, v in sorted(by_method.items(), key=lambda x: -x[1]):
        print(f"  {k:25s}: {v}")

    print()
    print(f"Enriched: {enriched}/{len(todo)}")
    print(f"Failed:   {len(failed)}")

    if not args.dry_run:
        with open(in_path, "w") as f:
            json.dump(mols, f, indent=1)
        print(f"Wrote {in_path}")
    else:
        print("(dry run — no file modified)")

    # Save the failed names so the user can curate them
    if failed:
        failed_path = in_path.with_name("cas_lookup_failed.txt")
        with open(failed_path, "w") as f:
            for name, smi in failed:
                f.write(f"{name}\t{smi}\n")
        print(f"Failed lookups saved to {failed_path}")


if __name__ == "__main__":
    main()
