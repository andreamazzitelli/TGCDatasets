#!/usr/bin/env python3
"""Download real-world photos of raw (ungraded) cards from active eBay
listings into data/tcg_identifier/train/<game>/.

The API sources (Scryfall, YGOPRODeck, ...) give clean studio scans on plain
backgrounds. This adds the other half of the distribution: seller photos —
real lighting, glare, slight rotation/perspective, held in hand or on a
desk. The identifier model needs to see that variety to generalise to a
phone camera. Feed these in alongside the scans, not instead of them.

Requires a free eBay developer app (Client ID + Client Secret):
  https://developer.ebay.com/my/keys
  export EBAY_CLIENT_ID=...  EBAY_CLIENT_SECRET=...

Usage:
  python3 download_ebay_raw.py --game pokemon --limit 120
  python3 download_ebay_raw.py --game all --limit 120        # every game
  python3 download_ebay_raw.py --game mtg --limit 120 --rich # + back/angle photos

Listings are noisier than scans: lots, sealed product, watermarks, binders
in frame. This filters obvious non-single-card titles, excludes graded/slab
results, and drops sub-300px thumbnails — but do a quick Finder Gallery-view
pass over each folder before training (see docs/data_collection.md).
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ebay import (
    CCG_SINGLES_CATEGORY,
    auth_session,
    get_access_token,
    listing_image_urls,
    missing_creds_message,
    passes_resolution,
    search_listings,
    title_is_clean,
)
from common import (
    REPO_ROOT,
    append_manifest,
    download_image,
    make_session,
    suffix_from_url,
)

OUT_ROOT = REPO_ROOT / "data" / "tcg_identifier" / "train"

# label (folder name in the taxonomy) -> eBay free-text search query.
# Each query is scoped to raw singles; graded results are filtered out below.
GAME_QUERIES = {
    "pokemon": "pokemon card",
    "magic_the_gathering": "mtg magic the gathering card",
    "yugioh": "yugioh card",
    "digimon": "digimon card game single",
    "one_piece": "one piece card game single",
    "lorcana": "disney lorcana card",
    "star_wars_unlimited": "star wars unlimited card",
    "flesh_and_blood": "flesh and blood tcg card",
    "dragon_ball_super": "dragon ball super fusion world card",
    "weiss_schwarz": "weiss schwarz card",
}
GAME_ALIASES = {"mtg": "magic_the_gathering", "ygo": "yugioh", "op": "one_piece",
                "swu": "star_wars_unlimited", "fab": "flesh_and_blood",
                "dbs": "dragon_ball_super", "ws": "weiss_schwarz"}

# eBay results for "<game> card" still surface slabs — keep this class raw.
GRADED_TERMS = ("psa", "bgs", "beckett", "cgc", "sgc", "graded", "slab", "gem mint 10",
                "gem mt 10", "tag grade")


def looks_raw(title: str) -> bool:
    lowered = title.lower()
    return not any(t in lowered for t in GRADED_TERMS)


def run_game(session, label: str, limit: int, seed: int, rich: bool, use_category: bool) -> int:
    query = GAME_QUERIES[label]
    out_dir = OUT_ROOT / label
    print(f"\n=== {label}: '{query}' ===")
    items = search_listings(
        session, query, want=limit * 4,
        category_ids=CCG_SINGLES_CATEGORY if use_category else None,
    )
    before = len(items)
    items = [i for i in items if title_is_clean(i.get("title", "")) and looks_raw(i.get("title", ""))]
    # one photo set per listing id
    seen_ids: set[str] = set()
    deduped = []
    for i in items:
        iid = str(i.get("itemId", ""))
        if iid and iid not in seen_ids:
            seen_ids.add(iid)
            deduped.append(i)
    print(f"  {before} listings -> {len(deduped)} after filtering")

    random.seed(seed)
    random.shuffle(deduped)

    saved = 0
    for item in deduped:
        if saved >= limit:
            break
        item_id = str(item.get("itemId", saved)).replace("|", "_").replace("v1_", "")
        for n, url in enumerate(listing_image_urls(item, rich=rich, session=session)):
            if saved >= limit:
                break
            dest = out_dir / f"ebay_{item_id}_{n}{suffix_from_url(url)}"
            if not download_image(session, url, dest):
                continue
            if not passes_resolution(dest):
                dest.unlink(missing_ok=True)
                continue
            append_manifest(
                filename=dest.name,
                model="tcg_identifier",
                label=label,
                tcg=label,
                source=f"ebay:{item.get('itemId', '')}",
                license_note="eBay Browse API listing photo — personal/research dataset use; verify eBay API terms before redistribution",
                notes=item.get("title", ""),
            )
            saved += 1
            if saved % 25 == 0:
                print(f"  {saved}/{limit} saved")
    print(f"  done: {saved} images -> {out_dir}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", required=True,
                        help="taxonomy label, an alias (mtg/ygo/op/swu/fab/dbs/ws), or 'all'")
    parser.add_argument("--limit", type=int, default=120, help="images per game (default: 120)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rich", action="store_true",
                        help="also fetch each listing's extra photos (back/angles) — 1 more API call per listing")
    parser.add_argument("--no-category", action="store_true",
                        help="don't restrict to eBay's CCG-singles category (broader, noisier)")
    parser.add_argument("--client-id", default=os.environ.get("EBAY_CLIENT_ID"))
    parser.add_argument("--client-secret", default=os.environ.get("EBAY_CLIENT_SECRET"))
    parser.add_argument("--marketplace", default="EBAY_US")
    args = parser.parse_args()

    if not args.client_id or not args.client_secret:
        sys.exit(missing_creds_message())

    game = args.game.lower()
    game = GAME_ALIASES.get(game, game)
    if game == "all":
        labels = list(GAME_QUERIES)
    elif game in GAME_QUERIES:
        labels = [game]
    else:
        sys.exit(f"Unknown game '{args.game}'. Choose from: {', '.join(GAME_QUERIES)}, or 'all'.")

    session = make_session()
    print("Requesting eBay OAuth token ...")
    token = get_access_token(session, args.client_id, args.client_secret)
    auth_session(session, token, args.marketplace)

    total = 0
    for label in labels:
        total += run_game(session, label, args.limit, args.seed, args.rich, not args.no_category)
    print(f"\nAll done. {total} images across {len(labels)} game(s).")
    print("Next: convert any non-JPEG/PNG, then re-run scripts/split_test_set.py.")


if __name__ == "__main__":
    main()
