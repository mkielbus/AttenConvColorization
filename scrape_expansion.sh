#!/usr/bin/env bash
# Phase 2a base scrape for the dataset expansion (see docs/plan_walki_z_przeuczeniem.md).
#
# Targets are PROPORTIONAL to the v2 train category counts, summing to the +40,849 images that
# analysis/phase2_gap_analysis.py showed are needed to bring every colour version present in
# val/test up to n_train >= 10. The gap is diffuse (98/98 clusters, top-10 clusters only 16.5%
# of it), so per-cluster query targeting was measured to buy almost nothing over a proportional
# scrape -- the gap-driven part happens at ADMISSION time in 2b/2c instead.
#
# Resumable: each category re-reads its manifest and skips record IDs already seen.
set -u

QUERY="queries/color.txt"
OUT_ROOT="${OUT_ROOT:-data/scraped}"
WORKERS="${WORKERS:-16}"
PYTHON="${PYTHON:-./venv/bin/python}"

# category:target -- proportional to v2 train (photography 8371, industrial 6073, sport 5784,
# nature 3705, fashion 1616, art 1336, migration 574, ww1 534 of 27,993).
TARGETS=(
  "photography:12216"
  "industrial:8863"
  "sport:8441"
  "nature:5407"
  "fashion:2358"
  "art:1950"
  "migration:838"
  "ww1:779"
)

mkdir -p "${OUT_ROOT}/logs"
for entry in "${TARGETS[@]}"; do
  category="${entry%%:*}"
  target="${entry##*:}"
  echo "=== ${category}: target ${target} ($(date +%H:%M:%S)) ==="
  "${PYTHON}" europeana_image_scraper.py "${QUERY}" "${category}" "${target}" \
    --theme "${category}" \
    --workers "${WORKERS}" \
    --out_dir "${OUT_ROOT}/${category}" \
    --manifest "${OUT_ROOT}/manifest_${category}.csv" \
    2>&1 | tee "${OUT_ROOT}/logs/${category}.log"
done
echo "=== scrape finished $(date) ==="
