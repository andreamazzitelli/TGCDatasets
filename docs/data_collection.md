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
   above, so don't rely on them exclusively:
   - Pokémon: the Pokémon TCG API (pokemontcg.io)
   - Magic: The Gathering: Scryfall's bulk data / image API
   - Yu-Gi-Oh!: YGOPRODeck API
   - Others: check each game's community API/database before scraping a
     retailer site directly.

   **Check the current terms of each source before use.** Most of these
   allow non-commercial/personal and research use of images via their API,
   but redistribution or commercial-product use often needs separate
   permission from the card publisher (Pokémon, Wizards of the Coast,
   Konami, etc.) or the grading companies (for slab imagery). This repo
   doesn't grant or verify that clearance — confirm it yourself against the
   source's current terms, especially if the resulting model or app will be
   distributed or sold.
3. **Grading company sample/lookup pages** (PSA cert verification, Beckett
   population report images, etc.) can supplement slab photos for the
   Grading Status model, subject to the same terms-of-use caveat above.

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
