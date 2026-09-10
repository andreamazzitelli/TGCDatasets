#!/usr/bin/env bash
# Creates the empty folder structure Create ML expects for both classifiers,
# plus a manifest CSV for tracking image provenance/licensing.
# See docs/taxonomy.md for what each label means and
# docs/createml_guide.md for how these folders get used in Create ML.
set -euo pipefail

cd "$(dirname "$0")/.."

TCG_LABELS=(
  pokemon
  magic_the_gathering
  yugioh
  lorcana
  one_piece
  digimon
  dragon_ball_super
  flesh_and_blood
  star_wars_unlimited
  weiss_schwarz
  other_tcg
  not_a_card
)

GRADING_LABELS=(
  raw
  psa
  bgs
  cgc
  sgc
  other_graded
)

for split in train test; do
  for label in "${TCG_LABELS[@]}"; do
    mkdir -p "data/tcg_identifier/$split/$label"
  done
  for label in "${GRADING_LABELS[@]}"; do
    mkdir -p "data/grading_status/$split/$label"
  done
done

MANIFEST="data/manifest.csv"
if [[ ! -f "$MANIFEST" ]]; then
  echo "filename,model,split,label,tcg,source,license_note,date_added,notes" > "$MANIFEST"
  echo "Created $MANIFEST"
fi

echo "Dataset folders ready under data/tcg_identifier/ and data/grading_status/"
echo "(train/ and test/, one subfolder per label). Drop images in, log"
echo "non-self-photographed sources in $MANIFEST."
