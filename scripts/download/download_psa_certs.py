#!/usr/bin/env python3
"""Fetch PSA-graded slab photos for cert numbers YOU supply into
data/grading_status/train/psa/.

This is the official PSA Public API (https://www.psacard.com/publicapi),
used the way it's meant to be used: resolving specific certs you already
have (yours, or a checklist you have real cert numbers for) — not browsing
or enumerating. PSA's terms restrict scraping/bulk discovery; this script
never guesses or walks a range of cert numbers, it only ever looks up
exactly what you pass it.

  GET https://api.psacard.com/publicapi/cert/GetByCertNumber/{cert}
  GET https://api.psacard.com/publicapi/cert/GetImagesByCertNumber/{cert}

Images only exist for certs graded after October 2021 — older certs will
come back with no photo.

QUOTA: an unauthenticated probe against this API returned a "100 calls per
day" quota error, so treat that as the ceiling regardless of whether you
have a token — this script defaults to the image-only call (1 call/cert) and
prints a running call count. Register a token at psacard.com/publicapi and
pass --token/PSA_API_TOKEN if you have one; it's sent as a Bearer header
either way, but even the anonymous quota is that tight, so keep runs small.

Usage:
  python3 download_psa_certs.py --certs 12345678,23456789
  python3 download_psa_certs.py --certs-file my_certs.txt   # one cert per line
  python3 download_psa_certs.py --certs 12345678 --with-metadata  # +1 call/cert, for notes/grade in the manifest
"""
from __future__ import annotations

import argparse
import os
import sys
import time
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

OUT_DIR = REPO_ROOT / "data" / "grading_status" / "train" / "psa"
API_BASE = "https://api.psacard.com/publicapi"
DAILY_QUOTA_HINT = 100  # observed anonymous quota; a token may raise this, but assume not


def parse_certs(args) -> list[str]:
    certs: list[str] = []
    if args.certs:
        certs.extend(c.strip() for c in args.certs.split(",") if c.strip())
    if args.certs_file:
        for line in Path(args.certs_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                certs.append(line)
    # de-dupe, keep order, must be numeric-ish
    seen: set[str] = set()
    out = []
    for c in certs:
        if c not in seen and c.isdigit():
            seen.add(c)
            out.append(c)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--certs", help="comma-separated PSA cert numbers")
    parser.add_argument("--certs-file", help="path to a file with one cert number per line")
    parser.add_argument("--token", default=os.environ.get("PSA_API_TOKEN"), help="PSA public API bearer token (optional)")
    parser.add_argument("--with-metadata", action="store_true",
                        help="also call GetByCertNumber for grade/card details (+1 API call per cert)")
    parser.add_argument("--max-calls", type=int, default=DAILY_QUOTA_HINT,
                        help=f"local safety cap on API calls this run (default: {DAILY_QUOTA_HINT}, the observed daily quota)")
    args = parser.parse_args()

    certs = parse_certs(args)
    if not certs:
        sys.exit("No cert numbers given. Pass --certs 12345678,23456789 or --certs-file <path>.")

    calls_needed = len(certs) * (2 if args.with_metadata else 1)
    print(f"{len(certs)} cert(s) -> up to {calls_needed} API call(s) (cap: {args.max_calls}).")
    if calls_needed > args.max_calls:
        sys.exit(
            f"That would need {calls_needed} calls, above --max-calls {args.max_calls}. "
            "Split into smaller batches (this API's quota is tight) or raise --max-calls if you're sure."
        )

    session = make_session()
    if args.token:
        session.headers["Authorization"] = f"Bearer {args.token}"

    calls_used = 0
    saved = 0
    for cert in certs:
        notes = ""
        if args.with_metadata:
            calls_used += 1
            try:
                r = session.get(f"{API_BASE}/cert/GetByCertNumber/{cert}", timeout=30)
                if r.status_code == 429:
                    print(f"  ! quota hit on cert {cert} metadata call — stopping.", file=sys.stderr)
                    break
                r.raise_for_status()
                spec = (r.json() or {}).get("PSACert", r.json())
                notes = " ".join(
                    str(v) for v in (
                        first_present(spec, "Subject", "CardName"),
                        first_present(spec, "Brand"),
                        first_present(spec, "CardGrade", "Grade"),
                    ) if v
                ) or ""
            except Exception as exc:  # noqa: BLE001 - best effort, keep going without metadata
                print(f"  ! metadata lookup for cert {cert} failed: {exc}", file=sys.stderr)
            time.sleep(0.5)

        calls_used += 1
        try:
            r = session.get(f"{API_BASE}/cert/GetImagesByCertNumber/{cert}", timeout=30)
            if r.status_code == 429:
                print(f"  ! quota hit on cert {cert} image call — stopping.", file=sys.stderr)
                break
            r.raise_for_status()
            payload = r.json()
        except Exception as exc:  # noqa: BLE001 - one bad cert shouldn't kill the run
            print(f"  ! image lookup for cert {cert} failed: {exc}", file=sys.stderr)
            time.sleep(0.5)
            continue

        images = payload if isinstance(payload, list) else payload.get("Images") if isinstance(payload, dict) else None
        if not images:
            print(f"  cert {cert}: no images (likely graded before Oct 2021, or invalid cert)")
            time.sleep(0.5)
            continue

        for n, img in enumerate(images):
            if not isinstance(img, dict):
                continue
            url = first_present(img, "ImageURL", "ImageUrl", "Url")
            if not url:
                warn_once_unknown_shape("PSA GetImagesByCertNumber entry", img)
                continue
            dest = OUT_DIR / f"psa_{cert}_{n}{suffix_from_url(str(url))}"
            if download_image(session, str(url), dest):
                append_manifest(
                    filename=dest.name,
                    model="grading_status",
                    label="psa",
                    tcg="mixed",
                    source=f"psa-publicapi:{cert}",
                    license_note="PSA Public API (psacard.com/publicapi) — official cert lookup; verify terms before redistribution",
                    notes=notes,
                )
                saved += 1
        time.sleep(0.5)

    print(f"Done. {saved} images saved to {OUT_DIR}. API calls used: {calls_used}.")


if __name__ == "__main__":
    main()
