#!/usr/bin/env bash
# Resumable fetch of the RapidOCR wheel (the pypi CDN drops this connection).
set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p wheels

URL="https://files.pythonhosted.org/packages/59/07/203065283bba0c5e9c10af3602dc25ab3e59ee54dc2d1d22c8faeabf5ed8/rapidocr-3.10.0-py3-none-any.whl"
OUT="wheels/rapidocr-3.10.0-py3-none-any.whl"
TOTAL=27288134

for attempt in $(seq 1 300); do
  have=0
  [ -f "$OUT" ] && have=$(stat -c%s "$OUT")
  if [ "$have" -ge "$TOTAL" ]; then
    echo "[rapidocr] complete ($have bytes, $attempt attempts)"
    exit 0
  fi
  echo "[rapidocr] attempt=$attempt have=$have/$TOTAL"
  curl -sL -C - --max-time 600 --retry 3 --retry-all-errors \
    --speed-limit 1 --speed-time 45 -o "$OUT" "$URL"
done
echo "[rapidocr] gave up"
exit 1
