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

Every script takes `--limit` (images to fetch, default 200-300) and `--seed`
(for a reproducible random sample). Re-running skips files already on disk,
so it's safe to bump `--limit` and re-run to top up a class.

## Confidence level — please smoke-test before a big run

This session's network access is proxied and several of the source domains
below were unreachable from it, so these endpoints/field names couldn't be
verified against a live response — only against each source's published
docs. **Run every script with `--limit 5` first.** If a script errors or
downloads nothing, it prints the raw JSON keys it actually saw — paste that
back and the field names can be corrected in a minute.

| Script | Source | Confidence |
|---|---|---|
| `download_magic.py` | Scryfall bulk data | High — well-established, stable API |
| `download_yugioh.py` | YGOPRODeck | High |
| `download_pokemon.py` | Pokémon TCG API | High (you said you already have a source — this is here for completeness) |
| `download_digimon.py` | digimoncard.io | Best-effort, self-diagnosing |
| `download_swu.py` | swu-db.com | Best-effort, self-diagnosing |
| `download_fab.py` | the-fab-cube/flesh-and-blood-cards (GitHub) | Best-effort, self-diagnosing |
| `download_lorcana.py` | Lorcast | Best-effort, self-diagnosing; also note the AVIF format caveat in the script header |
| `download_ebay_graded.py` | eBay Browse API | High (well-documented API), but needs your own developer credentials |

Not scripted:
- **One Piece** — you already have a source.
- **Dragon Ball Super (Fusion World)** and **Weiss Schwarz** — no reliable
  bulk API was found for either (see `docs/data_collection.md`); these two
  likely need manual sourcing or your own photos.

## Graded cards (`download_ebay_graded.py`)

1. Register a free app at https://developer.ebay.com/my/keys to get a
   Client ID and Client Secret.
2. `export EBAY_CLIENT_ID=... EBAY_CLIENT_SECRET=...`
3. Run once per grader label:
   ```bash
   python3 download_ebay_graded.py --grader psa --limit 150
   python3 download_ebay_graded.py --grader bgs --limit 150
   python3 download_ebay_graded.py --grader cgc --limit 150
   python3 download_ebay_graded.py --grader sgc --limit 150
   ```
4. The `raw` (ungraded) class isn't scripted — fill it with your own photos
   of ungraded cards (see `docs/data_collection.md`); a clean digital card
   scan doesn't look like a real photographed raw card, so scraping one in
   isn't useful training data for this class.

## After downloading

Point Create ML at `data/tcg_identifier/train` and `data/grading_status/train`
as described in `docs/createml_guide.md`. Check class balance before
training — `ls -1 data/tcg_identifier/train/*/ | wc -l` per folder, or just
look at Finder — and top up any thin classes by re-running the relevant
script with a higher `--limit`.
