#!/bin/bash
# atct_lookup.sh — open ATcT in browser for one molecule, prepare for screenshot.
#
# Usage:
#   ./atct_lookup.sh "phenanthrene"
#   ./atct_lookup.sh "diallyl sulfide"
#
# Workflow:
#   1. Script opens ATcT v1.130 in default browser
#   2. You pass the Cloudflare challenge (one click) — cookie persists for ~1h
#   3. Use ATcT's search box to find the molecule
#   4. Take a screenshot of the species page (Cmd+Shift+4 on macOS)
#   5. Tell Claude the screenshot path; he reads it and extracts D_at

set -euo pipefail

NAME="${1:-}"
if [ -z "$NAME" ]; then
    echo "Usage: $0 <molecule_name>"
    exit 1
fi

URL="https://atct.anl.gov/Thermochemical%20Data/version%201.130/index.php"

echo "═══════════════════════════════════════════════"
echo "  ATcT v1.130 lookup — $NAME"
echo "═══════════════════════════════════════════════"
echo
echo "Opening browser : $URL"
echo
open "$URL"
echo "Workflow :"
echo "  1. Pass the Cloudflare challenge (one click)"
echo "  2. Search ATcT for: $NAME"
echo "  3. Open the species page (click on the name in the table)"
echo "  4. Cmd+Shift+4, drag to capture the species data section"
echo "     → screenshot saved on Desktop as 'Screenshot YYYY-MM-DD.png'"
echo "     → OR Cmd+Shift+Ctrl+4 to copy to clipboard, then save"
echo
echo "When the screenshot is ready, paste its path to Claude."
echo "Suggested location : /tmp/atct_$(echo "$NAME" | tr -dc '[:alnum:]')_$(date +%Y%m%d_%H%M%S).png"
echo
echo "Or simply drag-and-drop the image into the Claude conversation."
