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
"""
from __future__ import annotations

import argparse
import base64
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, append_manifest, download_image, make_session, suffix_from_url

OUT_ROOT = REPO_ROOT / "data" / "grading_status" / "train"
TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"
SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"

GRADER_QUERIES = {
    "psa": "PSA graded card",
    "bgs": "BGS Beckett graded card",
    "cgc": "CGC graded card",
    "sgc": "SGC graded card",
    "other_graded": "TAG Arena Club graded card",
}


def get_access_token(session, client_id: str, client_secret: str) -> str:
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    resp = session.post(
        TOKEN_URL,
        headers={
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


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
        sys.exit(
            "Missing eBay API credentials. Register a free app at "
            "https://developer.ebay.com/my/keys and pass --client-id/--client-secret "
            "or set EBAY_CLIENT_ID/EBAY_CLIENT_SECRET."
        )

    session = make_session()
    print("Requesting eBay OAuth token ...")
    token = get_access_token(session, args.client_id, args.client_secret)
    session.headers["Authorization"] = f"Bearer {token}"
    session.headers["X-EBAY-C-MARKETPLACE-ID"] = args.marketplace

    out_dir = OUT_ROOT / args.grader
    query = args.query or GRADER_QUERIES[args.grader]

    items: list[dict] = []
    offset = 0
    page_size = 50
    while len(items) < args.limit * 2 and offset < 1000:
        print(f"Searching '{query}' offset={offset} ...")
        resp = session.get(SEARCH_URL, params={"q": query, "limit": page_size, "offset": offset}, timeout=30)
        if resp.status_code != 200:
            print(f"  ! search failed: HTTP {resp.status_code} {resp.text[:300]}", file=sys.stderr)
            break
        batch = resp.json().get("itemSummaries", [])
        if not batch:
            break
        items.extend(batch)
        offset += page_size

    print(f"Got {len(items)} listings for '{query}'.")
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
