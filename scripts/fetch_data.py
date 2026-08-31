#!/usr/bin/env python
"""Fetch raw external datasets described by a YAML manifest (Figshare collections).

A Figshare *collection* does not expose files directly: it contains *articles*, and
each article contains *files*. This script walks collection -> articles -> files via
the public Figshare v2 API (no auth needed for public data), downloads each file, and
verifies it against the md5 Figshare supplies. Files already present with a matching
checksum are skipped, so the fetch is resumable.

It writes a lock file next to the manifest (``<manifest>.lock.yaml``) recording exactly
what was fetched -- article ids, filenames, sizes, md5s -- so a teammate reproduces the
same bytes rather than "whatever is current."

Usage:
    uv run python scripts/fetch_data.py data/external/kinematic-emg.yaml
    uv run python scripts/fetch_data.py data/external/kinematic-emg.yaml --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import structlog
import yaml

logger = structlog.get_logger(__name__)

FIGSHARE_API = "https://api.figshare.com/v2"
PAGE_SIZE = 100


def _get_json(url: str, retries: int = 3, backoff: float = 1.5) -> dict | list:
    """GET a JSON endpoint with simple exponential-backoff retry."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as err:
            last_err = err
            wait = backoff ** attempt
            logger.warning("api_retry", url=url, attempt=attempt + 1, wait=round(wait, 1))
            time.sleep(wait)
    raise RuntimeError(f"API request failed after {retries} tries: {url}") from last_err


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming md5 of a file (matches Figshare's supplied_md5)."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def list_collection_articles(collection_id: int) -> list[dict]:
    """Return all article summaries in a collection, following pagination."""
    articles: list[dict] = []
    page = 1
    while True:
        url = f"{FIGSHARE_API}/collections/{collection_id}/articles?page={page}&page_size={PAGE_SIZE}"
        batch = _get_json(url)
        if not batch:
            break
        articles.extend(batch)
        page += 1
    logger.info("collection_enumerated", collection_id=collection_id, n_articles=len(articles))
    return articles


def article_files(article_id: int) -> list[dict]:
    """Return the file dicts for one article (name, download_url, supplied_md5, size)."""
    detail = _get_json(f"{FIGSHARE_API}/articles/{article_id}")
    return detail.get("files", [])


def download(url: str, dest: Path, expected_md5: str | None) -> None:
    """Stream a download to dest and verify md5 (if Figshare supplied one)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
        while block := resp.read(1 << 20):
            out.write(block)
    if expected_md5:
        got = _md5(tmp)
        if got != expected_md5:
            tmp.unlink(missing_ok=True)
            raise ValueError(f"md5 mismatch for {dest.name}: expected {expected_md5}, got {got}")
    tmp.replace(dest)


def fetch(manifest_path: Path, dry_run: bool = False) -> int:
    """Fetch everything described by a manifest; write a provenance lock."""
    manifest = yaml.safe_load(manifest_path.read_text())
    collection_id = int(manifest["collection_id"])
    dest_dir = Path(manifest["dest_dir"])

    fetched: list[dict] = []
    for article in list_collection_articles(collection_id):
        for f in article_files(article["id"]):
            dest = dest_dir / f["name"]
            expected = f.get("supplied_md5") or f.get("computed_md5")

            if dest.exists() and expected and _md5(dest) == expected:
                logger.info("skip_present", file=f["name"])
            elif dry_run:
                logger.info("would_fetch", file=f["name"], size=f.get("size"), article=article["id"])
            else:
                logger.info("downloading", file=f["name"], size=f.get("size"))
                download(f["download_url"], dest, expected)

            fetched.append({
                "name": f["name"],
                "article_id": article["id"],
                "size": f.get("size"),
                "md5": expected,
            })

    if not dry_run:
        lock_path = manifest_path.with_suffix(".lock.yaml")
        lock_path.write_text(yaml.safe_dump({
            "collection_id": collection_id,
            "doi": manifest.get("doi"),
            "n_files": len(fetched),
            "files": sorted(fetched, key=lambda x: x["name"]),
        }, sort_keys=False))
        logger.info("lock_written", path=str(lock_path), n_files=len(fetched))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch external datasets by manifest.")
    parser.add_argument("manifest", type=Path, help="Path to the dataset manifest YAML.")
    parser.add_argument("--dry-run", action="store_true", help="List files without downloading.")
    args = parser.parse_args()
    return fetch(args.manifest, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())

