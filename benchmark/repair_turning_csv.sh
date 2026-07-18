#!/bin/sh
f=benchmark/results/sweet_spot_turning_ssv.csv
awk -F, 'NF==11 && $1 != "s" {printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,0,%s,%s\n",$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11; next} {print}' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
echo repaired
