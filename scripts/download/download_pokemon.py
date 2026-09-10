#!/usr/bin/env python3
"""Download Pokémon TCG card images from the Pokémon TCG API into
data/tcg_identifier/train/pokemon/.

Included for completeness/consistency with the rest of the pipeline — you
mentioned you already have a Pokémon image source, so you may not need this
one.

Docs: https://docs.pokemontcg.io/
An API key is optional but recommended (raises the rate limit a lot); get
one free at https://dev.pokemontcg.io/ and pass it via --api-key or the
POKEMONTCG_API_KEY env var.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, append_manifest, download_image, make_session, sleep_polite, suffix_from_url

LABEL = "pokemon"
OUT_DIR = REPO_ROOT / "data" / "tcg_identifier" / "train" / LABEL
API_URL = "https://api.pokemontcg.io/v2/cards"
PAGE_SIZE = 250


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api-key", default=os.environ.get("POKEMONTCG_API_KEY"))
    args = parser.parse_args()

    session = make_session()
    if args.api_key:
        session.headers["X-Api-Key"] = args.api_key

    cards: list[dict] = []
    page = 1
    # Over-fetch a few pages then randomly sample, rather than paging
    # through the entire card set every run.
    pages_needed = max(1, (args.limit * 3) // PAGE_SIZE + 1)
    while page <= pages_needed:
        print(f"Fetching page {page} ...")
        batch: list[dict] = []
        # The keyless rate limit is easy to trip; retry a page a few times
        # with backoff before giving up (the API returns an HTML error body
        # on a 429, which shows up here as a JSON decode error).
        for attempt in range(5):
            try:
                r = session.get(API_URL, params={"page": page, "pageSize": PAGE_SIZE}, timeout=30)
                r.raise_for_status()
                batch = r.json().get("data", [])
                break
            except Exception as exc:
                wait = 5 * (attempt + 1)
                print(f"  ! page {page} attempt {attempt + 1} failed ({exc}); retrying in {wait}s", file=sys.stderr)
                sleep_polite(wait)
        if not batch:
            break
        cards.extend(batch)
        page += 1
        sleep_polite(1.0)

    print(f"Got {len(cards)} cards to sample from.")
    random.seed(args.seed)
    random.shuffle(cards)

    saved = 0
    for card in cards:
        if saved >= args.limit:
            break
        images = card.get("images") or {}
        url = images.get("large") or images.get("small")
        if not url:
            continue
        dest = OUT_DIR / f"{card['id']}{suffix_from_url(url)}"
        if download_image(session, url, dest):
            append_manifest(
                filename=dest.name,
                model="tcg_identifier",
                label=LABEL,
                tcg="pokemon",
                source=f"pokemontcg.io:{card.get('id')}",
                license_note="Pokémon TCG API — verify docs.pokemontcg.io terms before redistribution",
                notes=card.get("name", ""),
            )
            saved += 1
            if saved % 25 == 0:
                print(f"  {saved}/{args.limit} saved")

    print(f"Done. Saved {saved} images to {OUT_DIR}")


if __name__ == "__main__":
    main()
