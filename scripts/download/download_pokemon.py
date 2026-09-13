#!/usr/bin/env python3
"""Download Pokémon TCG card images from TCGdex into
data/tcg_identifier/train/pokemon/.

Docs: https://tcgdex.dev/rest. TCGdex's single-shot "list every card" endpoint
(~2MB) reliably truncates mid-stream, so this instead lists sets (small,
~35KB) and pages through a random sample of them, one set at a time - the
same traversal shape the TCG_ETL Pokémon adapter uses. Images are served as
"<image>/<quality>.<ext>" (see https://tcgdex.dev/assets).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, append_manifest, download_image, make_session, sleep_polite

LABEL = "pokemon"
OUT_DIR = REPO_ROOT / "data" / "tcg_identifier" / "train" / LABEL
SETS_URL = "https://api.tcgdex.net/v2/en/sets"
SET_URL = "https://api.tcgdex.net/v2/en/sets/{id}"
IMAGE_SUFFIX = "/high.png"


def fetch_json(session, url: str, retries: int = 4):
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            wait = 4 * (attempt + 1)
            print(f"  ! fetch {url} attempt {attempt + 1} failed ({exc}); retrying in {wait}s", file=sys.stderr)
            sleep_polite(wait)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    session = make_session()

    print(f"Fetching set list from {SETS_URL} ...")
    sets = fetch_json(session, SETS_URL) or []
    print(f"Got {len(sets)} sets.")

    random.seed(args.seed)
    random.shuffle(sets)

    saved = 0
    for set_brief in sets:
        if saved >= args.limit:
            break
        set_id = set_brief.get("id")
        if not set_id:
            continue
        full = fetch_json(session, SET_URL.format(id=set_id))
        sleep_polite(0.2)
        if not full:
            continue
        cards = [c for c in full.get("cards", []) if c.get("image")]
        random.shuffle(cards)
        for card in cards:
            if saved >= args.limit:
                break
            url = card["image"] + IMAGE_SUFFIX
            dest = OUT_DIR / f"{card['id']}.png"
            if download_image(session, url, dest):
                append_manifest(
                    filename=dest.name,
                    model="tcg_identifier",
                    label=LABEL,
                    tcg="pokemon",
                    source=f"tcgdex.net:{card.get('id')}",
                    license_note="TCGdex — verify tcgdex.dev terms before redistribution",
                    notes=card.get("name", ""),
                )
                saved += 1
                if saved % 25 == 0:
                    print(f"  {saved}/{args.limit} saved")
            sleep_polite(0.1)

    print(f"Done. Saved {saved} images to {OUT_DIR}")


if __name__ == "__main__":
    main()
