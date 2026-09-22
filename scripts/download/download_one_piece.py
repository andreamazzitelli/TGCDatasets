#!/usr/bin/env python3
"""Download One Piece Card Game images from optcgapi.com into
data/tcg_identifier/train/one_piece/.

Docs: https://optcgapi.com/  (no API key needed)
  GET /api/allSets/            -> [{set_name, set_id}, ...]
  GET /api/sets/<set_id>/      -> [{card_name, card_set_id, card_image, ...}, ...]

Card art is served straight off optcgapi.com's CDN via the `card_image`
field. Cards are de-duplicated by `card_set_id` so alternate-art reprints of
the same card don't flood one class.
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
    split_id_prefix,
    suffix_from_url,
    warn_once_unknown_shape,
)

LABEL = "one_piece"
OUT_DIR = REPO_ROOT / "data" / "tcg_identifier" / "train" / LABEL
ALL_SETS_URL = "https://optcgapi.com/api/allSets/"
SET_URL = "https://optcgapi.com/api/sets/{set_id}/"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    session = make_session()

    print(f"Fetching set list from {ALL_SETS_URL} ...")
    try:
        sets = session.get(ALL_SETS_URL, timeout=30).json()
    except Exception as exc:
        sys.exit(f"Could not fetch set list: {exc}")
    set_ids = [first_present(s, "set_id", "id") for s in sets if isinstance(s, dict)]
    set_ids = [s for s in set_ids if s]
    print(f"Got {len(set_ids)} sets.")

    cards: list[dict] = []
    for set_id in set_ids:
        print(f"Fetching set {set_id} ...")
        try:
            resp = session.get(SET_URL.format(set_id=set_id), timeout=30)
            resp.raise_for_status()
            batch = resp.json()
        except Exception as exc:
            print(f"  ! request for set {set_id} failed: {exc}", file=sys.stderr)
            batch = []
        if isinstance(batch, list):
            cards.extend(batch)
        sleep_polite(0.3)

    print(f"Got {len(cards)} card entries total.")
    if not cards:
        sys.exit("No cards returned — check https://optcgapi.com/ for the current endpoint shape.")

    # De-duplicate by card id, keeping the first printing seen.
    unique: dict[str, dict] = {}
    for card in cards:
        if not isinstance(card, dict):
            continue
        cid = first_present(card, "card_set_id", "card_image_id", "card_id") or str(len(unique))
        unique.setdefault(cid, card)
    cards = list(unique.values())

    random.seed(args.seed)
    random.shuffle(cards)

    saved = 0
    for card in cards:
        if saved >= args.limit:
            break
        url = first_present(card, "card_image", "image", "image_url")
        if not url:
            warn_once_unknown_shape("optcgapi.com card", card)
            continue
        cid = first_present(card, "card_set_id", "card_image_id", "card_id") or str(saved)
        name = first_present(card, "card_name", "name") or "card"
        # card_set_id is formatted "<set>-<number>" (e.g. "OP01-001"); the
        # prefix before the hyphen is the set code.
        set_code = split_id_prefix(str(cid))
        dest = OUT_DIR / f"{cid}{suffix_from_url(str(url))}"
        if download_image(session, str(url), dest):
            append_manifest(
                filename=dest.name,
                model="tcg_identifier",
                label=LABEL,
                tcg="one_piece",
                source=f"optcgapi.com:{cid}",
                license_note="optcgapi.com — verify optcgapi.com terms before redistribution",
                notes=str(name),
                set=set_code,
            )
            saved += 1
            if saved % 25 == 0:
                print(f"  {saved}/{args.limit} saved")

    print(f"Done. Saved {saved} images to {OUT_DIR}")


if __name__ == "__main__":
    main()
