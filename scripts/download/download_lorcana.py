#!/usr/bin/env python3
"""Download Disney Lorcana card images from the Lorcast API into
data/tcg_identifier/train/lorcana/.

Docs: https://lorcast.com/docs/api
No API key needed. Lorcast asks for 50-100ms between requests — this script
paces itself accordingly.

NOTE: card images from Lorcast are served in AVIF format. If Create ML
doesn't accept those directly, batch-convert first, e.g. on macOS:
  for f in data/tcg_identifier/train/lorcana/*.avif; do
    sips -s format jpeg "$f" --out "${f%.avif}.jpg" && rm "$f"
  done

This environment's network policy blocked direct access to lorcast.com, so
the endpoint/field names below are best-effort from the published docs, not
verified against a live response. Run with --limit 5 first; if it errors or
downloads nothing, the script prints the raw response keys it saw — send
that back so the field names can be corrected.
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

LABEL = "lorcana"
OUT_DIR = REPO_ROOT / "data" / "tcg_identifier" / "train" / LABEL
SETS_URL = "https://api.lorcast.com/v0/sets"


def extract_list(payload) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("results", "data"):
            if isinstance(payload.get(key), list):
                return payload[key]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    session = make_session()
    print(f"Fetching set list from {SETS_URL} ...")
    sets = extract_list(session.get(SETS_URL, timeout=30).json())
    if not sets:
        sys.exit("No sets returned — check https://lorcast.com/docs/api/sets for the current endpoint shape.")

    cards: list[dict] = []
    for s in sets:
        code = first_present(s, "code", "id")
        if not code:
            continue
        set_cards_url = f"https://api.lorcast.com/v0/sets/{code}/cards"
        print(f"Fetching cards for set {code} ...")
        try:
            resp = session.get(set_cards_url, timeout=30)
            resp.raise_for_status()
            batch = extract_list(resp.json())
        except Exception as exc:
            print(f"  ! request for set {code} failed: {exc}", file=sys.stderr)
            batch = []
        # Tag each card with the set code from this per-set fetch (the cards
        # get flattened into one list below, so this is the only place the
        # association is still available).
        for c in batch:
            if isinstance(c, dict):
                c["_set_code"] = code
        cards.extend(batch)
        sleep_polite(0.1)

    print(f"Got {len(cards)} card entries total.")
    if not cards:
        sys.exit("No cards returned — check https://lorcast.com/docs/api for the current endpoint shape.")

    random.seed(args.seed)
    random.shuffle(cards)

    saved = 0
    for card in cards:
        if saved >= args.limit:
            break
        image_uris = card.get("image_uris") if isinstance(card, dict) else None
        url = None
        if isinstance(image_uris, dict):
            digital = image_uris.get("digital")
            if isinstance(digital, dict):
                url = digital.get("normal") or digital.get("large")
            if not url:
                url = image_uris.get("normal") or image_uris.get("large")
        if not url:
            warn_once_unknown_shape("Lorcast card", card)
            continue
        name = first_present(card, "name") or "card"
        card_id = first_present(card, "id") or str(saved)
        dest = OUT_DIR / f"{card_id}{suffix_from_url(str(url), default='.avif')}"
        if download_image(session, url, dest):
            append_manifest(
                filename=dest.name,
                model="tcg_identifier",
                label=LABEL,
                tcg="lorcana",
                source=f"lorcast.com:{card_id}",
                license_note="Lorcast API — verify lorcast.com/docs/api terms before redistribution",
                notes=str(name),
                set=str(card.get("_set_code", "")) if isinstance(card, dict) else "",
            )
            saved += 1
            if saved % 25 == 0:
                print(f"  {saved}/{args.limit} saved")

    print(f"Done. Saved {saved} images to {OUT_DIR}")


if __name__ == "__main__":
    main()
