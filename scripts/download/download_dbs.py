#!/usr/bin/env python3
"""Download Dragon Ball Super Card Game — Fusion World images from the official
card database (dbs-cardgame.com) into
data/tcg_identifier/train/dragon_ball_super/.

There's no public API. The official card list is server-rendered per product
once you pass `?search=true&category[0]=<product_code>`, and card art sits at
a predictable CDN path. This script:

  1. scrapes the product dropdown for its ~28 product codes,
  2. pulls each product's card-list page and extracts the image filenames,
  3. de-duplicates alternate-art / parallel printings down to one per card,
  4. downloads the base art.

Images are served as WebP — run the AVIF/WebP -> JPEG conversion in
scripts/download/README.md afterwards (or scripts/split_test_set.py handles
the manifest either way).
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    REPO_ROOT,
    append_manifest,
    download_image,
    make_session,
    sleep_polite,
)

LABEL = "dragon_ball_super"
OUT_DIR = REPO_ROOT / "data" / "tcg_identifier" / "train" / LABEL
CARDLIST_URL = "https://www.dbs-cardgame.com/fw/en/cardlist/"
IMG_BASE = "https://www.dbs-cardgame.com/fw/images/cards/card/en/"

PRODUCT_RE = re.compile(r'data-val="(\d{6})"')
IMG_RE = re.compile(r'images/cards/card/en/([A-Z0-9]+-\d+[A-Za-z0-9_]*\.webp)')
CARD_NO_RE = re.compile(r'^([A-Z0-9]+-\d+)')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=360)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    session = make_session()

    print(f"Fetching product list from {CARDLIST_URL} ...")
    index_html = session.get(CARDLIST_URL, timeout=30).text
    products = sorted(set(PRODUCT_RE.findall(index_html)))
    if not products:
        sys.exit("No product codes found — the cardlist page markup may have changed.")
    print(f"Got {len(products)} products.")

    # card_no -> image filename (first, i.e. base, printing wins)
    cards: dict[str, str] = {}
    for code in products:
        print(f"Fetching product {code} ...")
        try:
            resp = session.get(CARDLIST_URL, params={"search": "true", "category[0]": code}, timeout=30)
            resp.raise_for_status()
        except Exception as exc:
            print(f"  ! product {code} failed: {exc}", file=sys.stderr)
            continue
        for fname in IMG_RE.findall(resp.text):
            if "noimage" in fname:
                continue
            m = CARD_NO_RE.match(fname)
            if not m:
                continue
            card_no = m.group(1)
            # Prefer the plain "<card_no>.webp" over "_f", "_p1" parallels.
            if card_no not in cards or (fname == f"{card_no}.webp"):
                cards.setdefault(card_no, fname)
                if fname == f"{card_no}.webp":
                    cards[card_no] = fname
        sleep_polite(0.4)

    print(f"Got {len(cards)} unique cards.")
    if not cards:
        sys.exit("No card images found.")

    items = list(cards.items())
    random.seed(args.seed)
    random.shuffle(items)

    saved = 0
    for card_no, fname in items:
        if saved >= args.limit:
            break
        dest = OUT_DIR / f"{card_no}.webp"
        if download_image(session, IMG_BASE + fname, dest):
            append_manifest(
                filename=dest.name,
                model="tcg_identifier",
                label=LABEL,
                tcg="dragon_ball_super",
                source=f"dbs-cardgame.com:{card_no}",
                license_note="dbs-cardgame.com official card list — verify site terms before redistribution",
                notes=card_no,
            )
            saved += 1
            if saved % 25 == 0:
                print(f"  {saved}/{args.limit} saved")
        sleep_polite(0.15)

    print(f"Done. Saved {saved} images to {OUT_DIR}")


if __name__ == "__main__":
    main()
