#!/usr/bin/env bash
# Resumable download of the perception models.
# The link here is throttled and drops connections, so we resume in a loop
# until the local file matches the server's Content-Range total.
set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p models

DET_URL="https://github.com/coff33ninja/go-mcp-computer-use/releases/latest/download/gpa_gui_detector.onnx"
DET_OUT="models/gpa_gui_detector.onnx"
CLS_OUT="models/mobilenetv3_small.onnx"
CLS_URL="https://github.com/coff33ninja/go-mcp-computer-use/releases/latest/download/mobilenetv3_small.onnx"

total_of() {
  curl -sIL -r 0-0 "$1" 2>/dev/null | tr -d '\r' | awk 'tolower($1)=="content-range:"{split($2,a,"/"); print a[2]}' | tail -1
}

fetch() {
  local url="$1" out="$2" total have code
  # A missing asset must never be saved as a model: without this guard a
  # 404 body ("Not Found", 9 bytes) landed in models/mobilenetv3_small.onnx
  # and sat there looking like a downloaded classifier. HEAD first -- if
  # the asset does not exist, say so and fail loudly.
  code=$(curl -sIL -o /dev/null -w '%{http_code}' --max-time 30 "$url")
  if [ "$code" != "200" ] && [ "$code" != "206" ]; then
    echo "[fetch] $out: asset missing upstream (HTTP $code); not saving" >&2
    rm -f "$out"
    return 2
  fi
  total=$(total_of "$url")
  echo "[fetch] $out total=${total:-unknown}"
  for attempt in $(seq 1 200); do
    have=0
    [ -f "$out" ] && have=$(stat -c%s "$out")
    if [ -n "$total" ] && [ "$have" -ge "$total" ] && [ "$have" -gt 1000000 ]; then
      echo "[fetch] $out complete ($have bytes, $attempt attempts)"
      return 0
    fi
    echo "[fetch] $out attempt=$attempt have=$have/$total"
    # -f: a later 404 must not truncate the file or write an error page;
    # models are megabytes, so anything smaller than 1 MB is junk either way.
    curl -sfL -C - --max-time 900 --retry 3 --retry-all-errors \
      --speed-limit 1 --speed-time 60 -o "$out" "$url" || rm -f "$out"
  done
  return 1
}

fetch "$DET_URL" "$DET_OUT"
fetch "$CLS_URL" "$CLS_OUT"
echo "[fetch] done"
