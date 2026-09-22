"""Download Weiss Schwarz (English) card images into
data/tcg_identifier/train/weiss_schwarz/.

Source: the community dataset github.com/CCondeluci/WeissSchwarz-ENG-DB,
which stores one JSON file per set under DB/, each a list of card objects
carrying a direct `image` URL on en.ws-tcg.com plus a `code`.

No API key. We list the repo tree via the GitHub API, sample sets, then pull
cards from those sets' raw JSON.
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

LABEL = "weiss_schwarz"
OUT_DIR = REPO_ROOT / "data" / "tcg_identifier" / "train" / LABEL
TREE_URL = "https://api.github.com/repos/CCondeluci/WeissSchwarz-ENG-DB/git/trees/master?recursive=1"
RAW_BASE = "https://raw.githubusercontent.com/CCondeluci/WeissSchwarz-ENG-DB/master/"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=360)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    session = make_session()
    random.seed(args.seed)

    print(f"Listing set files from {TREE_URL} ...")
    try:
        tree = session.get(TREE_URL, timeout=30).json().get("tree", [])
    except Exception as exc:
        sys.exit(f"Could not list repo tree: {exc}")
    set_paths = [t["path"] for t in tree if t.get("path", "").startswith("DB/") and t["path"].endswith(".json")]
    if not set_paths:
        sys.exit("No DB/*.json set files found — the repo layout may have changed.")
    random.shuffle(set_paths)
    print(f"Got {len(set_paths)} set files.")

    cards: list[dict] = []
    for path in set_paths:
        if len(cards) >= args.limit * 3:  # over-collect, then sample
            break
        try:
            resp = session.get(RAW_BASE + path, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"  ! {path} failed: {exc}", file=sys.stderr)
            continue
        if isinstance(data, list):
            # Tag each card with its source set file (e.g. "DB/S25.json" ->
            # "S25") before flattening into one list — the per-set JSON file
            # this repo's set boundary is the cleanest available identifier,
            # since the exact `code`/`cardcode` field format isn't verified
            # against a live response (see module docstring).
            set_code = Path(path).stem
            for c in data:
                if isinstance(c, dict):
                    c["_set_code"] = set_code
            cards.extend(data)
        sleep_polite(0.2)

    print(f"Collected {len(cards)} card entries.")
    if not cards:
        sys.exit("No cards collected.")

    random.shuffle(cards)

    saved = 0
    for card in cards:
        if saved >= args.limit:
            break
        if not isinstance(card, dict):
            continue
        url = first_present(card, "image", "image_url", "img")
        if not url:
            warn_once_unknown_shape("WeissSchwarz-ENG-DB card", card)
            continue
        code = (first_present(card, "code", "cardcode", "id") or str(saved)).replace("/", "_").replace(" ", "")
        name = first_present(card, "name", "card_name") or "card"
        dest = OUT_DIR / f"{code}{suffix_from_url(str(url))}"
        if download_image(session, str(url), dest):
            append_manifest(
                filename=dest.name,
                model="tcg_identifier",
                label=LABEL,
                tcg="weiss_schwarz",
                source=f"WeissSchwarz-ENG-DB:{code}",
                license_note="Community dataset (CCondeluci/WeissSchwarz-ENG-DB) referencing en.ws-tcg.com — verify terms before redistribution",
                notes=str(name),
                set=str(card.get("_set_code", "")),
            )
            saved += 1
            if saved % 25 == 0:
                print(f"  {saved}/{args.limit} saved")
        sleep_polite(0.1)

    print(f"Done. Saved {saved} images to {OUT_DIR}")


if __name__ == "__main__":
    main()
