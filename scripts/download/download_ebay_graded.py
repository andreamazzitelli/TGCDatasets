#!/usr/bin/env python3
"""Download graded-card slab photos from active eBay listings via the
official eBay Browse API into data/grading_status/train/<grader>/.

Requires a free eBay developer account: register an app at
https://developer.ebay.com/my/keys, then set:
  EBAY_CLIENT_ID=...
  EBAY_CLIENT_SECRET=...
as env vars (or pass --client-id/--client-secret).

Uses eBay's sanctioned Browse API — active listings only. Sold/completed
history needs eBay's restricted Marketplace Insights API, which isn't
generally available; active listings alone are plenty of volume for
training. See docs/data_collection.md for why this is the recommended
source for graded-card images (none of PSA/BGS/CGC/SGC expose a bulk image
catalog of their own).

Note: this fills the grader-label classes (psa/bgs/cgc/sgc/other_graded).
The `raw` class is better filled with your own photos of ungraded cards —
see docs/data_collection.md — since real held-card photos look different
from a clean listing photo of a raw card.

Listing photos are messier than API card scans: multi-card lots, seller
watermarks, playmats/binders in frame, tiny thumbnails on old listings. This
script filters listing titles that are obviously not a single graded card
and drops images below a minimum resolution, but it can't catch everything
— do a quick manual pass (Finder Gallery view + Quick Look) over each
downloaded folder before training. See docs/data_collection.md for the full
"eBay background/quality" guidance, including why background variety itself
is *not* the problem to fix (it's spurious correlation across classes that
is).
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ebay import (
    auth_session,
    get_access_token,
    missing_creds_message,
    passes_resolution,
    search_listings,
    title_is_clean,
)
from common import (
    REPO_ROOT,
    append_manifest,
    download_image,
    make_session,
    suffix_from_url,
)

OUT_ROOT = REPO_ROOT / "data" / "grading_status" / "train"

GRADER_QUERIES = {
    "psa": "PSA graded card",
    "bgs": "BGS Beckett graded card",
    "cgc": "CGC graded card",
    "sgc": "SGC graded card",
    "other_graded": "TAG Arena Club graded card",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grader", choices=sorted(GRADER_QUERIES), required=True)
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--client-id", default=os.environ.get("EBAY_CLIENT_ID"))
    parser.add_argument("--client-secret", default=os.environ.get("EBAY_CLIENT_SECRET"))
    parser.add_argument("--marketplace", default="EBAY_US")
    parser.add_argument("--query", default=None, help="override the default search query for this grader")
    args = parser.parse_args()

    if not args.client_id or not args.client_secret:
        sys.exit(missing_creds_message())

    session = make_session()
    print("Requesting eBay OAuth token ...")
    token = get_access_token(session, args.client_id, args.client_secret)
    auth_session(session, token, args.marketplace)

    out_dir = OUT_ROOT / args.grader
    query = args.query or GRADER_QUERIES[args.grader]

    items = search_listings(session, query, want=args.limit * 2)
    before = len(items)
    items = [i for i in items if title_is_clean(i.get("title", ""))]
    print(f"Got {before} listings for '{query}', {len(items)} after title filtering.")
    random.seed(args.seed)
    random.shuffle(items)

    saved = 0
    for item in items:
        if saved >= args.limit:
            break
        image = item.get("image") or {}
        url = image.get("imageUrl")
        if not url:
            continue
        item_id = str(item.get("itemId", saved)).replace("|", "_")
        dest = out_dir / f"{item_id}{suffix_from_url(url)}"
        if download_image(session, url, dest):
            if not passes_resolution(dest):
                dest.unlink(missing_ok=True)
                continue
            append_manifest(
                filename=dest.name,
                model="grading_status",
                label=args.grader,
                tcg="mixed",
                source=f"ebay:{item.get('itemId', '')}",
                license_note="eBay Browse API listing photo — personal/research dataset use; verify eBay API terms before redistribution",
                notes=item.get("title", ""),
            )
            saved += 1
            if saved % 25 == 0:
                print(f"  {saved}/{args.limit} saved")

    print(f"Done. Saved {saved} images to {out_dir}")


if __name__ == "__main__":
    main()
