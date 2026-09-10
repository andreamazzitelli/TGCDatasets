# Data collection

## Volume guidance

Create ML's image classifier will *train* on as few as ~10 images/class,
but accuracy at that size is unreliable. Rough targets:

| Model | Minimum viable | Recommended | Notes |
|---|---|---|---|
| TCG Identifier | 60–80 / class | 200–300 / class | `not_a_card` and `other_tcg` should be at least as large as your average core class, ideally larger — they're catching everything else |
| Grading Status | 80–100 / class | 300–400 / class | `raw` is easy to over-collect; make sure grader classes (especially `sgc`, `other_graded`) aren't starved relative to `psa`/`bgs` |

Keep classes roughly balanced. Create ML doesn't automatically reweight for
class imbalance, so a class with 5x the images of another will bias
predictions toward it.

## Shot guidelines

- **Full card in frame**, reasonably tight crop — not a wide photo of a desk
  with a tiny card in the middle. Real-world usage (someone photographing
  one card) should match your training distribution.
- **Vary background, lighting, and angle** within each class. If every
  `pokemon` photo is on the same white mat under the same lamp, the model
  partially learns "that background" rather than "Pokémon card." Aim for a
  genuine mix: different rooms/surfaces, natural and artificial light, slight
  rotation, some glare/reflection (cards and slabs are glossy — the model
  needs to see that).
- **Both orientations are fine to mix in** (front face is what carries the
  TCG's visual identity — logos, card frame style, text — so weight toward
  fronts, but a few backs/off-angle shots improve robustness).
- **Resolution:** source images at least ~600px on the short side. Create ML
  downsamples internally; there's no benefit to huge files, but don't feed it
  heavily compressed thumbnails either.
- **For grading labels specifically:** get the label/cert area sharp and
  legible — that's the actual signal (holder shape, label color, font,
  barcode placement). A slab shot so blurry the label is unreadable teaches
  the model the wrong thing.
- **Caution with augmentation later:** grading labels contain text and have
  a fixed left/right layout, so when you get to Create ML's augmentation
  options (see `createml_guide.md`), avoid horizontal flip for the grading
  model — it would create label layouts that don't exist in reality.

## Sourcing images

In order of preference:

1. **Your own photographs.** Best option — no licensing question, and the
   lighting/background/angle variety you naturally get from real shooting
   conditions is exactly what generalizes well. If you have access to a
   personal collection or a local card shop willing to let you photograph
   inventory, prioritize this for as many classes as you can cover.
2. **Public card databases/APIs**, for classes you can't source enough real
   photos for. These typically provide clean, front-facing scans — good for
   bootstrapping a class but low in the "real photo" variety described
   above, so don't rely on them exclusively. Per TCG:

   | TCG | Source | Notes |
   |---|---|---|
   | Pokémon | [Pokémon TCG API](https://pokemontcg.io) | |
   | Magic: The Gathering | [Scryfall API](https://scryfall.com/docs/api) | Bulk data + high-res images, best-documented of the set |
   | Yu-Gi-Oh! | [YGOPRODeck API](https://ygoprodeck.com/api-guide/) | |
   | Digimon | [digimoncard.io public API](https://digimoncard.io/api-documentation) | Includes images; rate-limited to 15 req/10s |
   | Star Wars: Unlimited | [swu-db.com API](https://www.swu-db.com/api) | Card search (JSON/CSV) + image redirection |
   | Flesh and Blood | [the-fab-cube/flesh-and-blood-cards](https://github.com/the-fab-cube/flesh-and-blood-cards) | Open-source JSON/CSV with direct image URLs — easiest bulk source of this group |
   | Disney Lorcana | [Lorcast API](https://lorcast.com/docs/api) or [lorcana-api.com](https://lorcana-api.com/) | Lorcast serves images off its own CDN, no stated rate limit |
   | One Piece | [optcgapi.com](https://optcgapi.com/) | |
   | Multi-game shortcut | [apitcg.com](https://www.apitcg.com/) | One API spanning One Piece, Pokémon, Digimon, Magic, Gundam, Dragon Ball Fusion, Union Arena — useful to fill several gaps with one integration; verify current terms yourself before relying on it |
   | Dragon Ball Super (Fusion World) | [official card list](https://www.dbs-cardgame.com/fw/en/cardlist/) | No solid public API found as of this writing — likely more manual sourcing or your own photos |
   | Weiss Schwarz | [WeissSchwarz-ENG-DB](https://github.com/CCondeluci/WeissSchwarz-ENG-DB) | Community dataset pointing at official `en.ws-tcg.com` images — weakest tooling of the group, same caveat as DBS |

   **Check the current terms of each source before use.** Most of these
   allow non-commercial/personal and research use of images via their API,
   but redistribution or commercial-product use often needs separate
   permission from the card publisher (Pokémon, Wizards of the Coast,
   Konami, Bandai, Bushiroad, etc.). This repo doesn't grant or verify that
   clearance — confirm it yourself against the source's current terms,
   especially if the resulting model or app will be distributed or sold.

3. **Graded slab images (PSA/BGS/CGC/SGC).** None of the four grading
   companies expose a bulk "browse all graded cards" API — this is the
   weak link in sourcing compared to the raw-card side above:
   - PSA's [public API](https://www.psacard.com/publicapi/documentation) only
     looks up **one cert number at a time**, and only has images for cards
     graded **after October 2021** — it's a validator for certs you already
     have, not a bulk browsing source. Its ToS restricts scraping; don't
     try to work around that at volume.
   - BGS, CGC, and SGC don't have a known public bulk image API at all.
   - **Most practical bulk source: the [eBay Browse API](https://developer.ebay.com/api-docs/buy/browse/overview.html)**
     (official developer registration, not scraping). Search terms like
     `"PSA 10"`, `"BGS 9.5"`, `"CGC 9"`, `"SGC 10"` across active listings —
     sellers photograph their actual slabs, across every TCG, which is
     exactly the variety this model needs. It only covers active listings
     (sold/completed history needs eBay's restricted Marketplace Insights
     API), but active listings alone are enough volume for training.
   - **Your own or a local card shop's graded inventory**, if you have
     access to any — even 50–100 photographed slabs across the four
     companies, varied angle/lighting, goes a long way, since the model is
     learning slab/label design, not the specific card inside.

## Cleaning up eBay listing photos

Listing photos are real-world photos, not clean scans, so they carry more
noise than the TCG API sources: desks, hands, other cards in frame, seller
watermarks, playmats, low-res thumbnails on old listings.

**Background variety itself is not the problem** — the shot guidelines
above already say to *want* varied backgrounds so the model doesn't key on
"that one white mat." The actual risk is a **background that correlates
with the label instead of the card**: if every `psa` photo happens to sit on
a clean desk (typical eBay staging) while every `raw` photo is on your own
kitchen table, the model can learn "kitchen table = raw" instead of
"no slab = raw," and will misfire the moment you photograph a raw card on a
desk. The fix is balance, not blank backgrounds: make sure each grading
class sees roughly the same *mix* of settings the others do, not a
uniquely clean or uniquely messy one.

Two things are still worth removing outright, and `download_ebay_graded.py`
now does the cheap part automatically:

- **Title filtering** — the script skips listings whose title suggests a
  multi-card lot, an empty holder, or non-card merch (`lot of`, `bundle`,
  `playmat`, `reprint`, `proxy`, etc. — see `TITLE_EXCLUDE` in the script).
- **Minimum resolution** — downloaded images below 300px on the short side
  are dropped (some old listings only have thumbnail-sized photos). Needs
  `Pillow` (in `requirements.txt`); the script just skips this check with a
  warning if Pillow isn't installed.

What the script *can't* catch, so do a quick manual pass before training:
heavy seller watermarks across the image, a listing photo that's actually a
stock graphic instead of the real slab, or a lot photo whose title didn't
match the filter. On macOS this is fastest natively — open the class folder
in Finder, switch to Gallery view (⌘4), and flick through with the arrow
keys, hitting Space for Quick Look and Delete to drop anything bad. For a
few hundred images this takes a couple of minutes; no extra tooling needed.
Don't over-cull, though — a slightly messy real photo is exactly the kind
of variety the model needs to generalize to a phone camera later.

## Tracking provenance: `data/manifest.csv`

`scripts/scaffold_dataset.sh` creates `data/manifest.csv` with this header:

```
filename,model,label,tcg,source,license_note,date_added,notes
```

Log every non-self-photographed image here (one row per file) so you can
answer "where did this come from and am I allowed to use it" later, and so
you can prune/re-source a class without losing track of what's already
covered. Self-photographed images are optional to log but still useful for
tracking which cards/sets you've already covered.
