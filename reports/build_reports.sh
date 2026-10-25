#!/bin/sh
# Build the report PDFs from reports/src/*.md (macOS: pandoc + Google Chrome).
# Usage: sh reports/build_reports.sh [name ...]   e.g. sh reports/build_reports.sh milestone01_proposal
set -e
cd "$(dirname "$0")"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
mkdir -p ../tmp/reports
names="$*"
[ -n "$names" ] || names=$(cd src && ls *.md | sed 's/\.md$//')
for name in $names; do
  pandoc "src/$name.md" --standalone --embed-resources --resource-path=src \
    --css src/style.css -o "../tmp/reports/$name.html"
  "$CHROME" --headless --disable-gpu --no-pdf-header-footer \
    --print-to-pdf="$PWD/$name.pdf" "file://$PWD/../tmp/reports/$name.html" 2>/dev/null
  echo "Built reports/$name.pdf"
done
