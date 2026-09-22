#!/usr/bin/env python3
"""Download Yu-Gi-Oh! card images from the YGOPRODeck API into
data/tcg_identifier/train/yugioh/.

Docs: https://ygoprodeck.com/api-guide/
No API key needed.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, append_manifest, download_image, make_session, suffix_from_url

LABEL = "yugioh"
OUT_DIR = REPO_ROOT / "data" / "tcg_identifier" / "train" / LABEL
API_URL = "https://db.ygoprodeck.com/api/v7/cardinfo.php"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    session = make_session()
    print(f"Fetching full card list from {API_URL} ...")
    resp = session.get(API_URL, timeout=60).json()
    cards = resp.get("data", [])
    print(f"Got {len(cards)} cards.")

    random.seed(args.seed)
    random.shuffle(cards)

    saved = 0
    for card in cards:
        if saved >= args.limit:
            break
        images = card.get("card_images") or []
        if not images:
            continue
        url = images[0].get("image_url")
        if not url:
            continue
        # The card art is generic (not per-printing), but `card_sets` lists
        # every set this card was printed in — take the first as a
        # deterministic, real (if arbitrary-among-reprints) set identifier.
        card_sets = card.get("card_sets") or []
        set_code = card_sets[0].get("set_code", "") if card_sets else ""
        dest = OUT_DIR / f"{card['id']}{suffix_from_url(url)}"
        if download_image(session, url, dest):
            append_manifest(
                filename=dest.name,
                model="tcg_identifier",
                label=LABEL,
                tcg="yugioh",
                source=f"ygoprodeck:{card.get('id')}",
                license_note="YGOPRODeck API — verify ygoprodeck.com/api-guide terms before redistribution",
                notes=card.get("name", ""),
                set=set_code,
            )
            saved += 1
            if saved % 25 == 0:
                print(f"  {saved}/{args.limit} saved")

    print(f"Done. Saved {saved} images to {OUT_DIR}")


if __name__ == "__main__":
    main()
