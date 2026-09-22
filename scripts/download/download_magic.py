#!/usr/bin/env python3
"""Download Magic: The Gathering card images from Scryfall's bulk data into
data/tcg_identifier/train/magic_the_gathering/.

Docs: https://scryfall.com/docs/api/bulk-data
No API key needed. This makes two HTTP requests total (the bulk-data index,
then the bulk file itself), which is Scryfall's recommended way to get many
cards rather than paging through /cards one at a time.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, append_manifest, download_image, make_session, suffix_from_url

LABEL = "magic_the_gathering"
OUT_DIR = REPO_ROOT / "data" / "tcg_identifier" / "train" / LABEL
BULK_INDEX_URL = "https://api.scryfall.com/bulk-data/default_cards"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=300, help="max images to download (default: 300)")
    parser.add_argument("--seed", type=int, default=42, help="random sample seed, for reproducibility")
    args = parser.parse_args()

    session = make_session()

    print(f"Fetching bulk data index from {BULK_INDEX_URL} ...")
    index = session.get(BULK_INDEX_URL, timeout=30).json()
    download_uri = index.get("download_uri") or index.get("jsonl_download_uri")
    if not download_uri:
        sys.exit(f"Unexpected response from Scryfall, no download_uri field: {index}")

    print(f"Downloading card list from {download_uri} (large file, may take a minute) ...")
    raw = session.get(download_uri, timeout=180).content
    if download_uri.endswith(".gz"):
        raw = gzip.decompress(raw)
    text = raw.decode("utf-8")
    if download_uri.endswith(".jsonl") or download_uri.endswith(".jsonl.gz"):
        cards = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        cards = json.loads(text)
    print(f"Got {len(cards)} card entries.")

    random.seed(args.seed)
    random.shuffle(cards)

    saved = 0
    for card in cards:
        if saved >= args.limit:
            break
        if card.get("digital"):
            continue  # skip Arena-only digital cards, not physical cards
        image_uris = card.get("image_uris")
        if not image_uris and card.get("card_faces"):
            image_uris = card["card_faces"][0].get("image_uris")
        if not image_uris:
            continue
        url = image_uris.get("normal") or image_uris.get("large")
        if not url:
            continue
        dest = OUT_DIR / f"{card['id']}{suffix_from_url(url)}"
        if download_image(session, url, dest):
            append_manifest(
                filename=dest.name,
                model="tcg_identifier",
                label=LABEL,
                tcg="magic_the_gathering",
                source=f"scryfall:{card.get('id')}",
                license_note="Scryfall API — verify scryfall.com/docs/api/images before redistribution",
                notes=card.get("name", ""),
                set=card.get("set", ""),
            )
            saved += 1
            if saved % 25 == 0:
                print(f"  {saved}/{args.limit} saved")

    print(f"Done. Saved {saved} images to {OUT_DIR}")


if __name__ == "__main__":
    main()
