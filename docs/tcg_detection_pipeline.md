# TCG detection pipeline — offline tooling

This doc extends this repo's scope to cover the offline (Python/macOS-side)
tooling for **on-device TCG identification** in the iOS app
(`TCGPortfolioTracker`): detecting *which trading card game* a photographed
card belongs to, before the app queries the backend.

Source brief: `/Users/andrea/TCGFull/docs/TCG_DETECTION_BRIEF_PROD.md`.
Cross-repo map (read this first if you haven't): `/Users/andrea/TCGFull/docs/tcg-detection-plan.md`
— it explains why this repo, rather than a new `tools/` tree, hosts the
brief's §4 "Offline tooling" and §5 "Evaluation harness", and records that
recon (M0) found this repo already has a working `tcg_identifier` classifier
effort that becomes **signal B** of the new pipeline, not a parallel build.

This doc owns: brief §0 (existing-assets audit), §4 (offline tooling), §5
(evaluation harness), and the tooling half of §6 milestones M0/M2. Everything
else — signal fusion, the decision policy, on-device rectification, OCR,
backend calls, UI — lives in `TCGPortfolioTracker/docs/tcg-detection.md` and
is out of scope here (see "Out of scope" below).

## 1. Scope extension: what's reused vs. what's new

The new pipeline needs three signals (brief §1): **(A)** on-device embedding
similarity against a small per-TCG prototype index, **(B)** a Create ML image
classifier, **(C)** targeted regional OCR (config + rules, no image data —
that tooling is iOS-side and out of scope here). This repo already builds
signal B; this doc adds the tooling for signal A and the shared pieces both
signals depend on.

**Reused as-is, no changes:**

- `data/manifest.csv` and its header (`filename,model,split,label,tcg,source,
  license_note,date_added,notes`) — the provenance ledger convention every
  new tool below writes to.
- `scripts/scaffold_dataset.sh` — folder scaffolding pattern (`data/<model>/
  <split>/<label>/`); the new artifacts (index, augmented copies) follow the
  same `data/<name>/...` shape rather than inventing a different layout.
- `scripts/download/*.py` and `scripts/download/common.py` (`append_manifest`,
  `download_image`, `make_session`, `sleep_polite`) — the per-source download
  scripts for `tcg_identifier` classes. These are the same reference images
  the embedding index (signal A) and the classifier's `other`-class-adjacent
  positive classes (signal B) are built from. No new scraping is proposed by
  this doc; per the brief and CLAUDE.md, nobody runs these against a live
  site without the user's explicit ask each time, and any script found to
  ignore rate limits or a source's terms gets flagged here (see §7), not
  silently rewritten.
- `scripts/split_test_set.py` — the cap/hold-out mechanics (deterministic
  seed, manifest reconciliation) are reused conceptually, but see §2: its
  current split is **by image**, which the new pipeline's eval set must not
  be (brief ground rule: split by set/card id, never by image).
- `docs/taxonomy.md`'s `tcg_identifier` label list — the 10 core TCGs +
  `other_tcg` + `not_a_card` become the TCG registry the new signals train
  and score against. Signal A's prototype index and signal B's classifier
  classes must stay in lockstep with this list (adding a TCG here is still
  "one new folder" per `docs/createml_guide.md` §6, extended by one new
  prototype-cluster set per §3.2 below).

**New work this repo takes on for this feature:**

- A **by-set** train/index/eval split, on top of (not replacing)
  `split_test_set.py`'s by-image cap/hold-out (§2).
- A manifest summary report script (§2).
- Four new offline tools: `augment`, `build_index`, `convert_embedder`,
  `train_classifier` extension (§3).
- An evaluation harness reporting per-signal (not fused) accuracy (§4).
- A version manifest for built artifacts feeding delivery infrastructure (§5).

## 2. Manifest extension: splits by set, and a summary report

### What the manifest already gives you, and what it doesn't

`data/manifest.csv`'s `source` column already encodes a per-source item
identifier (e.g. `digimoncard.io:BT1-073`, where `BT1` is the set code), but
there is **no dedicated `set` or `card_id` column**, and the format of that
identifier is source-specific (Scryfall's `source` is a Scryfall UUID with no
visible set code; TCGdex's embeds a set prefix; eBay listings carry no card
identity at all, only a listing id). `split_test_set.py`'s cap/hold-out is
**by image**, chosen with `rng.sample` over the whole class — it does not
know about sets, so as it stands it can and does let two images of the same
card (or the same set) land in both `train/` and `test/`. That's fine for the
existing Create ML test-set convention (Create ML's own accuracy number is
informative regardless), but it directly violates the new pipeline's ground
rule: eval must measure generalization to *unseen* cards, so train/index/eval
must be split by set (or card id), never by image, and reprints/languages/
near-identical arts of the same card must not straddle splits.

**What's missing and needs adding**, in a new `scripts/build_detection_split.py`
(name chosen to avoid colliding with the existing `split_test_set.py`, which
stays as-is for the Create ML `test/` convention; this is a separate,
additive split used only for the new pipeline's train/index/eval, layered on
top of the same `data/manifest.csv`):

- Add a `set` column to the manifest (append-only migration — existing rows
  get `set=""` and are backfilled where recoverable). Populate it going
  forward by having each `download_*.py` script pass the set/card identifier
  it already has in hand at request time (every script already knows this —
  e.g. `download_pokemon.py` iterates `sets`, `download_magic.py`'s Scryfall
  cards carry a `set` field) through to `append_manifest`, which needs a new
  optional `set: str = ""` parameter threaded into `MANIFEST_HEADER`. This is
  a small, additive change to `scripts/download/common.py` and each
  `download_*.py` call site — not a rewrite of how they fetch or name files.
- `scripts/build_detection_split.py`: given the (now set-annotated) manifest,
  assign each **set** (not image) to `train`, `index`, or `eval`, with a
  fixed seed, so every image of that set follows its set's assignment. Sets
  with no recoverable identifier (eBay listings, self-photographed images
  logged without a set) are treated as their own singleton "set" so they
  never silently merge into one bucket — flag rather than guess. Write a new
  `detection_split` column to the manifest (distinct from the existing
  `split` column, which keeps meaning "which `data/<model>/<split>/` folder
  the file is physically in" for Create ML) so the two split concepts don't
  collide.
- `scripts/report_manifest.py`: a read-only summary report over
  `data/manifest.csv` plus the files actually on disk, covering exactly what
  brief §0 asks M0 to audit: images per TCG/language/set, class imbalance
  flags (ratio to the largest class), resolution range (needs Pillow, same
  dependency `download_ebay_raw.py` already uses per `docs/data_collection.md`),
  file format breakdown, corrupted/unreadable files, exact duplicates
  (content hash) and near-duplicates (perceptual hash — new dependency, e.g.
  `imagehash`), and a catalog-scan-vs-real-photo split. That last label isn't
  currently tracked anywhere; the pragmatic source for it is `source`:
  everything from `download_ebay_raw.py` / `download_ebay_graded.py` is a
  real photo, everything else here today is a clean catalog scan. Encode this
  as a `photo_kind` value (`catalog_scan` | `real_photo`) derived from
  `source`'s prefix at report time — new manifest column only if this needs
  to be queried outside the report, which isn't clear yet, so start
  derived-only and promote to a column if a later tool needs to filter on it
  directly.

## 3. New offline tools

All new scripts live under `scripts/`, follow the existing scripts' style
(argparse, `--seed`/`--dry-run` where relevant, writing back into
`data/manifest.csv` via `common.py` helpers), and are Python, matching every
existing script in this repo (no new Swift here — Swift-side conversion of
outputs, if any, is `TCGPortfolioTracker`'s concern, not this repo's).

### 3.1 `scripts/augment.py` — camera-realism simulation

Takes catalog-scan images (identified via the `photo_kind` derivation in §2)
assigned to `train`/`index` by the detection split, and produces augmented
copies under `data/tcg_identifier_augmented/<label>/` (kept separate from
`data/tcg_identifier/`, gitignored the same way, logged to the manifest with
`source` referencing the original filename plus an `augment:` tag so
provenance stays traceable back to the original download). Transforms per
brief §4: perspective warp, blur, glare/specular highlight overlay, color
jitter, sleeve overlay (a semi-transparent plastic-sleeve texture + edge
glare), JPEG re-encode at a lossy quality to introduce artifacts, and partial
shadow. Library choice: Pillow (already a dependency here) covers jitter/blur/
JPEG re-encode directly; perspective warp and shadow are affine/mask
operations Pillow also supports; glare/sleeve overlay are compositing a
synthetic overlay image, no new heavy dependency needed. This is explicitly
**not** the same as Create ML's built-in training-time augmentation panel
(`docs/createml_guide.md` §2) — that panel's Crop/Rotate/Blur/Expose/Noise
run inside Create ML on the *scan* distribution; this script exists because
neither Create ML's augmentation nor the raw scans simulate sleeve glare,
perspective, or JPEG artifacts, which is exactly the gap brief §4 calls out
("catalog scans alone will not generalize to camera input").

### 3.2 `scripts/build_index.py` — on-device prototype index (signal A)

Embeds reference images (index-split, per §2) with the DINOv2 checkpoint
(`facebook/dinov2-small`, 384-dim) and clusters into a few dozen prototype
vectors per TCG, per the plan's decision to reuse TCG_ETL's checkpoint rather
than choose a separate embedder. Two embedding sources, in preference order:

1. **Reuse TCG_ETL's exported embeddings** where the reference image is
   already a catalogued card TCG_ETL has ingested — per
   `TCG_ETL/docs/embedding-export-for-tcg-detection.md` (written in parallel
   by another agent; if it doesn't exist yet or looks incomplete when you
   read this, treat the export format there as authoritative once it lands,
   and don't assume this doc's description of it beyond "384-dim vectors from
   `facebook/dinov2-small`, in pgvector" is final).
2. **Embed fresh** for anything not in that export (e.g. `not_a_card`,
   `other_tcg`, eBay real-photo references) — using the same checkpoint
   loaded locally via `transformers`/`torch` (new dependencies for this repo;
   TCG_ETL already has this exact dependency pair per its own
   `transform/embeddings.py`, so this is not a new integration pattern, just
   a second consumer of the same checkpoint).

Clustering: k-means (or similar) per TCG down to a configurable prototype
count (brief: "a few dozen"), optionally split further per language/frame
family where that's known to help (mirrors signal A's spec in the brief).
Output: a compact binary — float16 matrix of prototype vectors + a JSON
sidecar mapping row → (TCG, optional language/family label) — written to
`data/index/` (new, gitignored like the rest of `data/`; only the built
artifact ships to the app, not this repo's scripts or dataset, per the
brief's ground rules and `README.md`'s existing "why images aren't committed"
rationale extended to built artifacts).

### 3.3 `scripts/convert_embedder.py` — coremltools conversion

Converts the `facebook/dinov2-small` checkpoint to a `.mlmodel` (or
`.mlpackage`) via `coremltools`, so the app can run the *same* embedding
space on-device that `build_index.py` used to build the prototype index —
this is what makes signal A's index and the app's runtime embedder
consistent. Depends on whatever checkpoint reference
`TCG_ETL/docs/embedding-export-for-tcg-detection.md` pins (exact HF revision/
weights hash) so a re-conversion is reproducible and traceable to the same
version the index was built from; if that doc isn't available yet, pin
`facebook/dinov2-small` at whatever revision is currently used and record it
in the version manifest (§5) rather than leaving it implicit.

### 3.4 `train_classifier` — extending the existing effort (signal B)

Not a new script from scratch: extends what's already partway built here
(`docs/createml_guide.md`'s workflow, `data/tcg_identifier/`). Two concrete
additions:

- **`other` class.** `docs/taxonomy.md` already has `not_a_card` (hands,
  tables, packaging, blurry/empty frames) and `other_tcg` (supported-games
  catch-all for TCGs without a dedicated class yet). The brief's signal B
  spec calls for a single `other` class meaning "not a card / unsupported" —
  clarify whether that's `not_a_card` renamed/repurposed, or a third class
  added alongside the existing two ambiguous overlap between `other_tcg`
  ("a TCG we just haven't split out yet") and the brief's `other` ("not
  supported, don't force a guess") is worth resolving explicitly rather than
  silently merging; flagged as an open item in §6.
- **Real camera photos in training, not just scans.** The brief requires
  training on augmented references *and* real camera photos, not catalog
  scans alone. `download_ebay_raw.py` already exists for exactly this (real
  seller photos of raw cards, `docs/data_collection.md` "sourcing images"
  §2) but per `scripts/download/README.md`'s status table it hasn't been run
  yet (needs the user's own eBay developer credentials) — this is an
  existing gap, not new work invented here, and is called out again as an
  owner item in §7.
- **Calibration.** Create ML's own export doesn't include temperature
  scaling; add a small post-training calibration step (fit one scalar
  temperature on the eval set's held-out-by-set logits) and store it
  alongside the `.mlmodel` in the version manifest (§5) so the app can apply
  it without re-deriving it.

Training itself stays the Create ML app workflow already documented in
`docs/createml_guide.md` — this doc doesn't propose scripting around Create
ML's GUI (there's no supported CLI for it), only the data preparation
(augmented + real photos in `train/`) and post-export calibration step that
happen around it.

## 4. Evaluation harness

A new `scripts/eval_harness.py`, command-line, evaluating the **offline/
Python-side signals only** — the embedding index (A) and classifier (B), each
against the same held-out-by-set eval split from §2. It does not attempt
signal C (OCR — Swift/Vision-only, no Python equivalent) or fusion (needs the
iOS-side decision policy, which doesn't exist in this repo). Fused,
policy-level accuracy is evaluated in `TCGPortfolioTracker/docs/tcg-detection.md`,
not here — this harness's job is to give that later evaluation clean,
trustworthy per-signal numbers to fuse, not to duplicate its job.

Reports, mirroring brief §5's list minus the fusion/OCR-dependent parts:

- Per-signal top-1 accuracy (embedding-index nearest-prototype match;
  classifier softmax argmax) on the eval split.
- Confusion matrix per signal, with `other`/`not_a_card`/`other_tcg`
  confusions shown explicitly rather than folded into an aggregate.
- Ambiguity/unknown-rate **proxies**: for the embedding signal, rate where
  the top-2 nearest-prototype TCGs are within a configurable margin; for the
  classifier, rate where the calibrated top-2 softmax gap is below a
  threshold. These are proxies for what the brief calls the fused
  `.ambiguous`/`.unknown` rate — real ambiguity/unknown rates depend on the
  iOS fusion+decision policy and can only be measured there.
- Accuracy broken out by: foil, full-art/textless, sleeved, low-light,
  language, and TCG — each reported separately, never hidden in an average,
  per the brief's explicit "handle look-alike TCGs explicitly ... do not hide
  them in an average" rule. Foil/full-art/sleeved/low-light are not currently
  tracked as manifest columns (today's manifest has no such field); this
  harness needs that metadata on the eval set specifically, which likely
  means hand-labeling the (small, curated) eval set rather than extending the
  whole manifest schema for it — proposed as the minimal-footprint approach,
  open to revision once the real eval photos (§6 open item) are in hand.
- Latency is out of scope for this harness (brief's ms targets are on-device
  Core ML inference numbers, measured in `TCGPortfolioTracker`, not a Python
  proxy that wouldn't be representative).

**Targets — not committed, to be confirmed by the user.** The brief proposes
97% top-1 on clean photos, 90% on hard photos, and ≤10% ambiguous rate (brief
§5, explicitly flagged there as "I will confirm"). This doc reports them
here only as the harness's placeholder thresholds for a pass/fail summary
line — they are not adopted as targets by this doc and must not be presented
to the user as agreed numbers.

## 5. Versioning and delivery

Every build of the offline artifacts (`data/index/*.bin` + `labels.json` from
`build_index.py`, the `.mlmodel` from `convert_embedder.py`, the classifier
`.mlmodel` from §3.4, and `tcg_rules.json` if/when iOS adds it — out of scope
here but the version manifest format should have room for it) gets one
version manifest, produced by a small `scripts/write_version_manifest.py`
(or as a final step folded into each build script — TBD at implementation
time, not decided here) recording:

- `version` (semantic or date-based — TBD, not decided here)
- `build_date`
- `dataset_manifest_hash` — a content hash of `data/manifest.csv` (or the
  relevant `detection_split` subset) at build time, so an artifact is always
  traceable back to the exact dataset state that produced it
- `eval_results` — the eval harness's (§4) summary output for this build
- `min_app_version` — the oldest `TCGPortfolioTracker` version this artifact
  is compatible with

This version manifest is what feeds the delivery mechanism described in
`TCGInfrastructure/docs/model-bundle-delivery.md` (plain text path — that
repo's doc, not linked from here, owns hosting/serving these artifacts for
OTA delivery to the app; this repo only produces the artifacts and their
version manifest, it does not host or serve them).

## 6. Out of scope for this repo

- No app code (Swift, iOS project files) — that's `TCGPortfolioTracker`.
- No fusion or decision-policy logic (weights, margin thresholds, session
  prior, multi-frame voting) — that's the iOS-side pipeline per brief §2–§3,
  documented in `TCGPortfolioTracker/docs/tcg-detection.md`.
- No backend calls or backend contract changes — the backend is read-only and
  this repo never calls it (root `CLAUDE.md` rule); confirmed separately in
  `TCGBackend/docs/tcg-detection-integration.md` that no endpoint changes are
  needed.
- No live scraping without the user's explicit go-ahead, each time — this
  applies to every `download_*.py` script listed in §7 below, same as it
  already applies to this repo's existing effort.

## 7. Open items for the user

Per the brief's own "owner" list and this repo's specific gaps found above —
not resolved by this doc, listed so they're visible:

- **Confirmed TCG/language list to support first.** `docs/taxonomy.md`'s 10
  core TCGs + `other_tcg`/`not_a_card` are this repo's current list; the
  brief asks the user to confirm scope (which TCGs, which languages) before
  this tooling commits to a fixed prototype/classifier class set.
- **Real test photos for the eval set.** The brief requires a real-camera-
  photo eval set never used for training/index building. This repo's only
  real-photo source today is `download_ebay_raw.py` /
  `download_ebay_graded.py`, both **not yet run** (need the user's own eBay
  developer credentials per `scripts/download/README.md`). Until these run
  (with the user's explicit go-ahead) or the user supplies their own photos,
  there is no real-photo eval set, only catalog scans.
- **Data-source terms/licensing review**, per source, so training on that art
  and shipping derived embeddings/prototypes is confirmed allowed. Sources
  currently in `scripts/download/`, for this review: Pokémon TCG data via
  TCGdex (`download_pokemon.py`), Scryfall bulk data (`download_magic.py`),
  YGOPRODeck (`download_yugioh.py`), digimoncard.io (`download_digimon.py`),
  swu-db.com (`download_swu.py`), the-fab-cube/flesh-and-blood-cards on
  GitHub (`download_fab.py`), Lorcast (`download_lorcana.py`), optcgapi.com
  (`download_one_piece.py`), dbs-cardgame.com official site
  (`download_dbs.py`), CCondeluci/WeissSchwarz-ENG-DB on GitHub
  (`download_weiss.py`), NetrunnerDB + api.gatcg.com + MarvelCDB
  (`download_other_tcg.py`), Lorem Picsum/Unsplash (`download_not_a_card.py`),
  and eBay Browse API (`download_ebay_raw.py`, `download_ebay_graded.py`).
  `docs/data_collection.md` already flags per-source terms as unverified by
  this repo ("confirm it yourself against the source's current terms"); the
  new pipeline raises the stakes (shipping derived embeddings/prototypes, not
  just training a private model) so this review matters more than before,
  not less.
- **Confirmed accuracy/latency targets** — §4's 97%/90%/≤10% figures are the
  brief's proposal only, explicitly unconfirmed.
