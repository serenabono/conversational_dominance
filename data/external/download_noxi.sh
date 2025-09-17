#!/usr/bin/env bash
# download_share_zip.sh
# Usage: ./download_share_zip.sh

URL="https://cloud.hcai.eu/public.php/dav/files/of4F7SDcwRpNGcp/?accept=zip"
OUTFILE="NoXi_dataset.zip"

echo "→ Downloading $URL"
curl -L --fail -o "$OUTFILE" "$URL"

echo "→ Extracting..."
unzip -q "$OUTFILE" -d NoXi_dataset

echo "✅ Done. Files extracted into ./NoXi_dataset"