"""Populate data/tcg_identifier/train/other_tcg/ with cards from trading card
games that don't have their own class yet.

`other_tcg` is the identifier model's catch-all so a supported-but-unlabelled
game doesn't get forced into a wrong core label (see docs/taxonomy.md). It
should therefore be visually *diverse* — this script pulls a roughly even mix
from three unrelated games with clean public image sources:

  - Android: Netrunner   (NetrunnerDB API)
  - Grand Archive         (api.gatcg.com)
  - Marvel Champions LCG  (MarvelCDB API)

Add more sources to SOURCES below to broaden it further.
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
    make_session,
    sleep_polite,
    suffix_from_url,
)

LABEL = "other_tcg"
OUT_DIR = REPO_ROOT / "data" / "tcg_identifier" / "train" / LABEL


def fetch_netrunner(session) -> list[tuple[str, str, str]]:
    """-> list of (game, card_id, image_url)."""
    r = session.get("https://netrunnerdb.com/api/2.0/public/cards", timeout=45)
    r.raise_for_status()
    payload = r.json()
    tmpl = payload.get("imageUrlTemplate", "https://card-images.netrunnerdb.com/v2/large/{code}.jpg")
    out = []
    for c in payload.get("data", []):
        code = c.get("code")
        if code:
            out.append(("netrunner", code, tmpl.replace("{code}", code)))
    return out


def fetch_grand_archive(session) -> list[tuple[str, str, str]]:
    out, page = [], 1
    while True:
        r = session.get("https://api.gatcg.com/cards/search", params={"page": page}, timeout=30)
        r.raise_for_status()
        body = r.json()
        for c in body.get("data", []):
            for ed in c.get("editions", []):
                img = ed.get("image")
                if img:
                    out.append(("grand_archive", ed.get("slug") or ed.get("card_id"), "https://api.gatcg.com" + img))
                    break
        if not body.get("has_more") or page >= body.get("total_pages", page):
            break
        page += 1
        sleep_polite(0.3)
    return out


def fetch_marvel(session) -> list[tuple[str, str, str]]:
    r = session.get("https://marvelcdb.com/api/public/cards/", timeout=45)
    r.raise_for_status()
    out = []
    for c in r.json():
        src = c.get("imagesrc")
        if src:
            out.append(("marvel_champions", c.get("code"), "https://marvelcdb.com" + src))
    return out


SOURCES = {
    "netrunner": fetch_netrunner,
    "grand_archive": fetch_grand_archive,
    "marvel_champions": fetch_marvel,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=360, help="total images across all sources")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    session = make_session()
    rng = random.Random(args.seed)
    per_source = -(-args.limit // len(SOURCES))  # ceil

    saved_total = 0
    for game, fetch in SOURCES.items():
        print(f"Fetching {game} ...")
        try:
            pool = fetch(session)
        except Exception as exc:
            print(f"  ! {game} source failed: {exc}", file=sys.stderr)
            continue
        rng.shuffle(pool)
        print(f"  {len(pool)} candidates")
        saved = 0
        for _, card_id, url in pool:
            if saved >= per_source or saved_total >= args.limit:
                break
            cid = str(card_id).replace("/", "_")
            dest = OUT_DIR / f"{game}_{cid}{suffix_from_url(url)}"
            if download_image(session, url, dest):
                # No `set` threaded here: this class mixes three unrelated
                # games with unverified, inconsistent response shapes (see
                # module docstring), and `other_tcg` is a catch-all label,
                # not a per-TCG one — `cid` (per-card id) already carries
                # what real identity there is, in `source`.
                append_manifest(
                    filename=dest.name,
                    model="tcg_identifier",
                    label=LABEL,
                    tcg=game,
                    source=f"{game}:{cid}",
                    license_note=f"{game} public card API — verify each source's terms before redistribution",
                    notes=game,
                )
                saved += 1
                saved_total += 1
                if saved % 25 == 0:
                    print(f"  {game}: {saved}/{per_source}")
            sleep_polite(0.1)
        print(f"  {game}: saved {saved}")

    print(f"Done. Saved {saved_total} images to {OUT_DIR}")


if __name__ == "__main__":
    main()
