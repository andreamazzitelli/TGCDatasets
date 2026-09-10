#!/usr/bin/env python3
"""Download Star Wars: Unlimited card images from the swu-db.com API into
data/tcg_identifier/train/star_wars_unlimited/.

Docs: https://www.swu-db.com/api
No API key needed.

NOTE: this environment's network policy blocked direct access to swu-db.com,
so the endpoint/field names below are best-effort from the published docs
(confirmed: GET https://api.swu-db.com/cards/search?q=... with q syntax like
"c=3" for cost-equals-3), not verified against a live response. Run with
--limit 5 first; if it errors or downloads nothing, the script prints the
raw response keys it saw — send that back so the field names can be
corrected.
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

LABEL = "star_wars_unlimited"
OUT_DIR = REPO_ROOT / "data" / "tcg_identifier" / "train" / LABEL
SEARCH_URL = "https://api.swu-db.com/cards/search"


def image_url_for(card: dict) -> str | None:
    direct = first_present(card, "Image", "FrontArt", "image", "front_art", "Artwork")
    if direct:
        return direct
    set_code = first_present(card, "Set", "set")
    number = first_present(card, "Number", "number", "CardNumber")
    if set_code and number:
        # The docs mention an "image redirection" endpoint for card artwork
        # shaped like this — used as a fallback when no direct image field
        # is present in the search response.
        return f"https://api.swu-db.com/cards/{set_code}/{number}/image"
    return None


def extract_batch(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("Data", "data", "cards", "Cards", "results"):
            if isinstance(payload.get(key), list):
                return payload[key]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    session = make_session()
    cards: list[dict] = []
    for cost in range(10):  # Cost 0-9 covers the large majority of printed cards
        print(f"Fetching cost={cost} ...")
        try:
            resp = session.get(SEARCH_URL, params={"q": f"c={cost}", "format": "json"}, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            print(f"  ! request for cost={cost} failed: {exc}", file=sys.stderr)
            payload = {}
        cards.extend(extract_batch(payload))
        sleep_polite(0.3)

    print(f"Got {len(cards)} card entries total.")
    if not cards:
        sys.exit("No cards returned — check https://www.swu-db.com/api for the current query syntax.")

    random.seed(args.seed)
    random.shuffle(cards)

    saved = 0
    for card in cards:
        if saved >= args.limit:
            break
        url = image_url_for(card)
        if not url:
            warn_once_unknown_shape("swu-db.com card", card)
            continue
        name = first_present(card, "Name", "name") or "card"
        set_code = first_present(card, "Set", "set") or "swu"
        number = first_present(card, "Number", "number") or str(saved)
        dest = OUT_DIR / f"{set_code}_{number}{suffix_from_url(str(url))}"
        if download_image(session, url, dest):
            append_manifest(
                filename=dest.name,
                model="tcg_identifier",
                label=LABEL,
                tcg="star_wars_unlimited",
                source=f"swu-db.com:{set_code}_{number}",
                license_note="swu-db.com community API — verify swu-db.com/api terms before redistribution",
                notes=str(name),
            )
            saved += 1
            if saved % 25 == 0:
                print(f"  {saved}/{args.limit} saved")

    print(f"Done. Saved {saved} images to {OUT_DIR}")


if __name__ == "__main__":
    main()
