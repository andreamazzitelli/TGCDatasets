"""Shared helpers for the eBay Browse API download scripts
(download_ebay_graded.py, download_ebay_raw.py).

eBay's Browse API is the sanctioned way to read active listings. It needs a
free developer app (https://developer.ebay.com/my/keys) — a Client ID and
Client Secret — and uses the client-credentials OAuth grant (an application
token, no end-user login flow). Active listings only; that's plenty of
volume for a training set.
"""
from __future__ import annotations

import base64
import sys
import time

TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"
SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
ITEM_URL = "https://api.ebay.com/buy/browse/v1/item/{item_id}"
OAUTH_SCOPE = "https://api.ebay.com/oauth/api_scope"

# eBay leaf category "CCG Individual Cards" — keeps results to single cards
# rather than sealed product, playmats, supplies, etc.
CCG_SINGLES_CATEGORY = "183454"

MIN_SHORT_SIDE = 300  # px — below this an eBay photo is barely more than a thumbnail

# Title substrings that usually mean "not one clean single-card photo".
TITLE_EXCLUDE = [
    "lot of", " lot", "bundle", "job lot", "wholesale", "huge lot", "x100", "x50",
    "bulk", "collection", "playset", "complete set", "sealed", "booster box",
    "booster pack", "blister", "tin", "empty case", "empty slab", "no card",
    "case only", "display case", "playmat", "binder", "box only", "sleeve",
    "toploader", "reprint", "proxy", "custom", "orica", "fake", "repro",
]


def missing_creds_message() -> str:
    return (
        "Missing eBay API credentials. Register a free app at "
        "https://developer.ebay.com/my/keys and pass --client-id/--client-secret "
        "or set EBAY_CLIENT_ID / EBAY_CLIENT_SECRET."
    )


def title_is_clean(title: str) -> bool:
    lowered = f" {title.lower()} "
    return not any(bad in lowered for bad in TITLE_EXCLUDE)


def get_access_token(session, client_id: str, client_secret: str) -> str:
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    resp = session.post(
        TOKEN_URL,
        headers={
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={"grant_type": "client_credentials", "scope": OAUTH_SCOPE},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def auth_session(session, token: str, marketplace: str = "EBAY_US"):
    session.headers["Authorization"] = f"Bearer {token}"
    session.headers["X-EBAY-C-MARKETPLACE-ID"] = marketplace
    return session


def search_listings(session, query: str, *, want: int, category_ids: str | None = None,
                    extra_filter: str | None = None, page_size: int = 50) -> list[dict]:
    """Page through item_summary/search until `want` listings are collected
    (or eBay runs out / hits its 10k offset ceiling)."""
    items: list[dict] = []
    offset = 0
    while len(items) < want and offset < 10000:
        params = {"q": query, "limit": page_size, "offset": offset}
        if category_ids:
            params["category_ids"] = category_ids
        if extra_filter:
            params["filter"] = extra_filter
        print(f"  search '{query}' offset={offset} ...")
        resp = session.get(SEARCH_URL, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"  ! search failed: HTTP {resp.status_code} {resp.text[:300]}", file=sys.stderr)
            break
        batch = resp.json().get("itemSummaries", []) or []
        if not batch:
            break
        items.extend(batch)
        offset += page_size
        time.sleep(0.2)
    return items


def listing_image_urls(item: dict, *, rich: bool = False, session=None) -> list[str]:
    """Image URLs for a listing. With rich=True, also fetch the item detail
    for its additional photos (back of card, angles) — one extra API call."""
    urls: list[str] = []
    img = (item.get("image") or {}).get("imageUrl")
    if img:
        urls.append(img)
    if rich and session is not None:
        item_id = item.get("itemId")
        if item_id:
            try:
                resp = session.get(ITEM_URL.format(item_id=item_id), timeout=30)
                if resp.status_code == 200:
                    d = resp.json()
                    primary = (d.get("image") or {}).get("imageUrl")
                    if primary:
                        urls.append(primary)
                    for extra in d.get("additionalImages", []) or []:
                        if extra.get("imageUrl"):
                            urls.append(extra["imageUrl"])
                time.sleep(0.15)
            except Exception as exc:  # noqa: BLE001 - best effort
                print(f"  ! item detail for {item_id} failed: {exc}", file=sys.stderr)
    # de-dupe, keep order
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def passes_resolution(path, min_short_side: int = MIN_SHORT_SIDE) -> bool:
    """True if the image at `path` is at least `min_short_side` on its short
    edge. Returns True (can't tell) if Pillow isn't installed."""
    try:
        from PIL import Image
    except ImportError:
        return True
    try:
        with Image.open(path) as im:
            return min(im.size) >= min_short_side
    except Exception:  # noqa: BLE001
        return False
