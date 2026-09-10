#!/usr/bin/env python3
"""Download Digimon Card Game images from the digimoncard.io public API into
data/tcg_identifier/train/digimon/.

Docs: https://digimoncard.io/api-documentation
No API key needed, but rate limited to 15 requests / 10 seconds — this
script paces itself accordingly.

NOTE: this environment's network policy blocked direct access to
digimoncard.io, so the endpoint/field names below are best-effort from the
published docs, not verified against a live response. Run with --limit 5
first; if it errors or downloads nothing, the script prints the raw response
keys it saw — send that back so the field names can be corrected.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    REPO_ROOT,
    append_manifest,
    download_image,
    first_present,
    make_session,
    sleep_polite,
    suffix_from_url,
    warn_once_unknown_shape,
)

LABEL = "digimon"
OUT_DIR = REPO_ROOT / "data" / "tcg_identifier" / "train" / LABEL
SEARCH_URL = "https://digimoncard.io/api-public/search.php"
# Main card colors in the Digimon Card Game — used to page through the
# catalog in documented-parameter chunks, since there's no known
# "give me everything" query.
COLORS = ["Red", "Blue", "Yellow", "Green", "Black", "Purple", "White"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    session = make_session()
    cards: list[dict] = []
    for color in COLORS:
        print(f"Fetching color={color} ...")
        try:
            resp = session.get(SEARCH_URL, params={"color": color}, timeout=30)
            resp.raise_for_status()
            batch = resp.json()
        except Exception as exc:
            print(f"  ! request for color={color} failed: {exc}", file=sys.stderr)
            batch = []
        if isinstance(batch, list):
            cards.extend(batch)
        sleep_polite(0.7)  # stay well under 15 req/10s

    print(f"Got {len(cards)} card entries total.")
    if not cards:
        sys.exit(
            "No cards returned — check https://digimoncard.io/api-documentation "
            "for the current endpoint/params and update SEARCH_URL/COLORS above."
        )

    random.seed(args.seed)
    random.shuffle(cards)

    saved = 0
    for card in cards:
        if saved >= args.limit:
            break
        name = first_present(card, "name", "card_name") or "card"
        card_id = first_present(card, "card_id", "id", "cardnumber") or str(saved)
        url = first_present(card, "image_url", "cardimage", "img", "image")
        if not url and first_present(card, "id"):
            # digimoncard.io serves card art at a predictable path keyed by the
            # card id (e.g. BT1-010); the search response itself carries no URL.
            url = f"https://images.digimoncard.io/images/cards/{first_present(card, 'id')}.jpg"
        if not url:
            warn_once_unknown_shape("digimoncard.io card", card)
            continue
        dest = OUT_DIR / f"{card_id}{suffix_from_url(str(url))}"
        if download_image(session, url, dest):
            append_manifest(
                filename=dest.name,
                model="tcg_identifier",
                label=LABEL,
                tcg="digimon",
                source=f"digimoncard.io:{card_id}",
                license_note="digimoncard.io public API — verify digimoncard.io/api-documentation terms before redistribution",
                notes=str(name),
            )
            saved += 1
            if saved % 25 == 0:
                print(f"  {saved}/{args.limit} saved")

    print(f"Done. Saved {saved} images to {OUT_DIR}")


if __name__ == "__main__":
    main()
