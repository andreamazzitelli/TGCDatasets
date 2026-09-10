# Download scripts

Each `download_*.py` pulls card images from one source straight into the
right `data/.../train/<label>/` folder (see `docs/taxonomy.md`) and logs
every file to `data/manifest.csv`. Run `../scaffold_dataset.sh` first so the
folders exist.

```bash
cd scripts/download
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

python3 download_magic.py --limit 5      # smoke test first, see note below
python3 download_magic.py --limit 300    # then a real run
```

Every script takes `--limit` (images to fetch, default 200-360) and `--seed`
(for a reproducible random sample). Re-running skips files already on disk,
so it's safe to bump `--limit` and re-run to top up a class. `--limit` counts
*new* files written this run, not a target total.

## Status — populated against live APIs 2026-09-10

Every scripted class below was run for real and now holds **300 train + 60
test images** (`../split_test_set.py` did the balancing + hold-out; see
below). Only eBay (graded cards) is outstanding, pending developer
credentials.

| Script | Source | Status |
|---|---|---|
| `download_magic.py` | Scryfall bulk data | ✅ Scryfall now serves bulk data as gzipped JSONL; the script handles `jsonl_download_uri` + gunzip. |
| `download_yugioh.py` | YGOPRODeck | ✅ |
| `download_pokemon.py` | Pokémon TCG API | ✅ Keyless rate limit is easy to trip and the API had a transient outage — the script now retries pages with backoff. Set `POKEMONTCG_API_KEY` for headroom. |
| `download_digimon.py` | digimoncard.io | ✅ The search response carries no image URL; the script derives it from the card `id` (`images.digimoncard.io/images/cards/<id>.jpg`). |
| `download_swu.py` | swu-db.com | ✅ |
| `download_fab.py` | the-fab-cube/flesh-and-blood-cards (GitHub) | ✅ Served as WebP — see conversion note below. |
| `download_lorcana.py` | Lorcast | ✅ Served as AVIF — see conversion note below. |
| `download_one_piece.py` | optcgapi.com | ✅ De-duplicates alt-art reprints by `card_set_id`. |
| `download_dbs.py` | dbs-cardgame.com (official) | ✅ Dragon Ball Super — Fusion World. No API; scrapes the per-product card list. Served as WebP — see conversion note. |
| `download_weiss.py` | CCondeluci/WeissSchwarz-ENG-DB (GitHub) | ✅ Weiss Schwarz (EN). Community JSON referencing en.ws-tcg.com images. |
| `download_other_tcg.py` | NetrunnerDB + api.gatcg.com + MarvelCDB | ✅ Catch-all class — even mix of Android: Netrunner, Grand Archive, Marvel Champions. Add sources in `SOURCES`. |
| `download_not_a_card.py` | Lorem Picsum (Unsplash) | ✅ Negative class — real-world non-card photos. Supplement with your own phone photos of non-card subjects for more realism. |
| `download_ebay_raw.py` | eBay Browse API | ⏳ Not run — needs your own developer credentials. Real seller photos of raw cards; adds photographic variety per identifier class (see below). |
| `download_ebay_graded.py` | eBay Browse API | ⏳ Not run — needs your own developer credentials (see below). Fills the `grading_status/` grader classes. |

### Image format conversion (already applied)

Lorcana (AVIF), Flesh and Blood (WebP) and Dragon Ball Super (WebP) don't
reliably ingest into Create ML. Those classes have been batch-converted to
JPEG in place and `data/manifest.csv` filenames updated. If you re-run those
scripts to top up, convert the new files the same way (run from the repo
root):

```bash
find data \( -name '*.avif' -o -name '*.webp' \) | while read f; do
  sips -s format jpeg "$f" --out "${f%.*}.jpg" && rm "$f"
done
```

## Balancing + train/test split — `../split_test_set.py`

The download scripts only write to `train/`. After downloading, run this once
to cap over-collected classes and carve out a reproducible test set:

```bash
python3 scripts/split_test_set.py --cap 360 --test-per-class 60 --seed 42
```

It deletes any train surplus above `--cap`, moves `--test-per-class` images
per class from `train/` to `test/` (seeded, so it's reproducible), and
rewrites `data/manifest.csv` — adding a `split` column, dropping rows whose
file is gone, de-duplicating. Re-running with the same numbers is a no-op
(files already in `test/` count toward the quota). `--dry-run` previews it.

## Still empty — need manual sourcing or your own photos

| Class (model) | Why |
|---|---|
| `raw` + all of `grading_status/` (`psa`/`bgs`/`cgc`/`sgc`/`other_graded`) | Graded slabs come from eBay (below); `raw` needs your own photos of ungraded cards. |

Everything under `data/tcg_identifier/` is now scripted.

## eBay — real seller photos (`download_ebay_*.py`)

Both scripts use eBay's sanctioned **Browse API** (active listings only) via
`_ebay.py`. One-time setup:

1. Register a free app at https://developer.ebay.com/my/keys → Client ID + Secret.
2. `export EBAY_CLIENT_ID=... EBAY_CLIENT_SECRET=...`

### `download_ebay_raw.py` — photographic variety for the identifier model

The API card scans are clean studio shots on plain backgrounds. This pulls
**real seller photos of raw (ungraded) cards** per game — actual lighting,
glare, rotation, held-in-hand — so the identifier model generalises to a
phone camera. Add these *alongside* the scans, not instead of them; ~100–150
per class is plenty.

```bash
python3 download_ebay_raw.py --game all --limit 120      # every game
python3 download_ebay_raw.py --game pokemon --limit 150  # one game
python3 download_ebay_raw.py --game mtg --limit 120 --rich   # + back/angle photos
```

Graded/slab listings and obvious multi-card lots are filtered out; sub-300px
thumbnails are dropped. Still do a quick Finder Gallery-view pass per folder
before training (watermarks, stock graphics, mis-titled lots). Re-run
`../split_test_set.py` afterwards.

### `download_ebay_graded.py` — the grading_status model

```bash
python3 download_ebay_graded.py --grader psa --limit 150
python3 download_ebay_graded.py --grader bgs --limit 150
python3 download_ebay_graded.py --grader cgc --limit 150
python3 download_ebay_graded.py --grader sgc --limit 150
```

The `raw` (ungraded) class for `grading_status/` isn't scripted — fill it
with your own photos of ungraded cards (see `docs/data_collection.md`); a
listing photo of a raw card behind cellophane doesn't match a card held in
hand.

## After downloading

Run `../split_test_set.py` (above), then point Create ML at
`data/tcg_identifier/train` + `data/tcg_identifier/test` as described in
`docs/createml_guide.md`. Check class balance — `for d in
data/tcg_identifier/train/*/; do echo "$(ls "$d" | wc -l) $d"; done` — and
top up any thin classes by re-running the relevant script with a higher
`--limit`, then re-running the split.
