#!/bin/bash
# Parallel ranged download (6 connections) with resume + size verification.
# usage: fetch_parallel.sh <url> <outfile> [connections]
set -euo pipefail
URL="$1"; OUT="$2"; N="${3:-6}"
SIZE=$(curl -sSIL --max-time 60 "$URL" | awk 'tolower($1)=="content-length:"{v=$2} END{print v}' | tr -d '\r')
[ -n "$SIZE" ] && [ "$SIZE" -gt 0 ] || { echo "cannot determine size"; exit 1; }
echo "size=$SIZE conn=$N"
TMP="${OUT}.parts"; mkdir -p "$TMP"
CHUNK=$(( (SIZE + N - 1) / N ))
for i in $(seq 0 $((N-1))); do
  START=$(( i * CHUNK )); END=$(( START + CHUNK - 1 )); [ "$END" -ge "$SIZE" ] && END=$((SIZE-1))
  echo "$i $START $END"
done > "$TMP/ranges.txt"
cat "$TMP/ranges.txt" | xargs -P "$N" -n 3 bash -c '
  i=$0; s=$1; e=$2
  f="'"$TMP"'/part_$i"
  want=$(( e - s + 1 ))
  have=$(stat -f%z "$f" 2>/dev/null || echo 0)
  if [ "$have" -eq "$want" ]; then echo "part $i cached"; exit 0; fi
  for try in 1 2 3 4 5; do
    curl -sSL --fail --retry 3 --retry-delay 2 -r "$s-$e" -o "$f" "'"$URL"'" || true
    have=$(stat -f%z "$f" 2>/dev/null || echo 0)
    [ "$have" -eq "$want" ] && { echo "part $i ok ($want B)"; exit 0; }
    echo "part $i retry $try have=$have want=$want" >&2; sleep 2
  done
  exit 1
'
cat "$TMP"/part_* > "$OUT.tmp"
GOT=$(stat -f%z "$OUT.tmp")
[ "$GOT" -eq "$SIZE" ] || { echo "size mismatch $GOT != $SIZE"; exit 1; }
mv "$OUT.tmp" "$OUT"; rm -rf "$TMP"
echo "done -> $OUT ($GOT bytes)"
