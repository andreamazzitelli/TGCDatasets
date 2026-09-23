#!/usr/bin/env python3
"""Download Flesh and Blood card images from the open-source the-fab-cube
card dataset into data/tcg_identifier/train/flesh_and_blood/.

Data source: https://github.com/the-fab-cube/flesh-and-blood-cards
(open-source JSON/CSV card data maintained by the community — check the
repo's LICENSE for current terms before redistributing anything built from
it).

NOTE: this environment's network policy blocked direct access to GitHub raw
content, so the file path/field names below are best-effort from the repo's
documented layout, not verified against a live file. Run with --limit 5
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
    suffix_from_url,
    warn_once_unknown_shape,
)

LABEL = "flesh_and_blood"
OUT_DIR = REPO_ROOT / "data" / "tcg_identifier" / "train" / LABEL
CARD_JSON_URL = "https://raw.githubusercontent.com/the-fab-cube/flesh-and-blood-cards/develop/json/english/card.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    session = make_session()
    print(f"Fetching {CARD_JSON_URL} ...")
    resp = session.get(CARD_JSON_URL, timeout=60)
    if resp.status_code != 200:
        sys.exit(
            f"Got HTTP {resp.status_code} fetching card data. The file path may have moved — "
            "check https://github.com/the-fab-cube/flesh-and-blood-cards for the current json/ layout "
            "and update CARD_JSON_URL above."
        )
    cards = resp.json()
    print(f"Got {len(cards)} card entries.")

    random.seed(args.seed)
    random.shuffle(cards)

    saved = 0
    for card in cards:
        if saved >= args.limit:
            break
        # Only set `set_code` when the image came from a specific printing
        # (the `printings` loop below) — the top-level `card.image` fallback
        # is generic card art with no set attached, so leave it "" there.
        set_code = ""
        url = first_present(card, "image", "image_url")
        if not url:
            printings = card.get("printings")
            if isinstance(printings, list):
                for p in printings:
                    url = first_present(p, "image", "image_url")
                    if url:
                        set_code = str(first_present(p, "set_id", "set", "printing_id") or "")
                        break
        if not url:
            warn_once_unknown_shape("flesh-and-blood-cards entry", card)
            continue
        name = first_present(card, "name") or "card"
        card_id = first_present(card, "unique_id", "id", "card_identifier") or str(saved)
        dest = OUT_DIR / f"{card_id}{suffix_from_url(str(url))}"
        if download_image(session, url, dest):
            append_manifest(
                filename=dest.name,
                model="tcg_identifier",
                label=LABEL,
                tcg="flesh_and_blood",
                source=f"the-fab-cube/flesh-and-blood-cards:{card_id}",
                license_note="Open-source community dataset — check repo LICENSE before redistribution",
                notes=str(name),
                set=set_code,
            )
            saved += 1
            if saved % 25 == 0:
                print(f"  {saved}/{args.limit} saved")

    print(f"Done. Saved {saved} images to {OUT_DIR}")


if __name__ == "__main__":
    main()
