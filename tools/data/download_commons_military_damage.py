"""Download a provenance-rich review set of damaged military vehicle images.

The output is candidate source imagery, not a labeled or split training set.
Every file must be reviewed for component visibility, damage level, duplicates,
and license suitability before it is admitted to a prepared CRATER dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "CRATER-research-dataset-collector/1.0 (Wikimedia Commons API)"

# Search labels are deliberately conservative. They are candidate labels only;
# the visible crop, rather than the search phrase, determines the final label.
SEARCHES = (
    ("destroyed Humvee", "severe_visible_damage", "windshield;wheel;hull;comms_system;mounted_gun"),
    ("damaged Humvee", "possible_damage", "windshield;wheel;hull;comms_system;mounted_gun"),
    ("burned Humvee", "severe_visible_damage", "windshield;wheel;hull;comms_system;mounted_gun"),
    ("destroyed tank", "severe_visible_damage", "track;hull;mounted_gun;comms_system"),
    ("damaged tank", "possible_damage", "track;hull;mounted_gun;comms_system"),
    ("battle damaged tank", "possible_damage", "track;hull;mounted_gun;comms_system"),
    ("destroyed armored personnel carrier", "severe_visible_damage", "wheel;track;hull;mounted_gun;comms_system"),
    ("damaged armored personnel carrier", "possible_damage", "wheel;track;hull;mounted_gun;comms_system"),
    ("destroyed military truck", "severe_visible_damage", "windshield;wheel;hull;comms_system;mounted_gun"),
    ("damaged military truck", "possible_damage", "windshield;wheel;hull;comms_system;mounted_gun"),
    ("burning military vehicle", "severe_visible_damage", "windshield;wheel;track;hull;comms_system;mounted_gun"),
    ("damaged military communications vehicle", "possible_damage", "wheel;track;hull;comms_system"),
    ("destroyed self-propelled anti-aircraft gun", "severe_visible_damage", "track;hull;comms_system;mounted_gun"),
)

FIELDS = (
    "local_file",
    "sha256",
    "bytes",
    "download_status",
    "width",
    "height",
    "commons_page_id",
    "commons_title",
    "commons_page_url",
    "download_url",
    "original_url",
    "mime",
    "license_short_name",
    "license_url",
    "artist",
    "credit",
    "description",
    "search_query",
    "candidate_damage_level",
    "candidate_components",
    "source_asset_id",
    "group_id",
    "split",
    "review_status",
    "review_notes",
)


def clean_markup(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def api_get(params: dict[str, str | int]) -> dict:
    query = urllib.parse.urlencode({"format": "json", "formatversion": 2, **params})
    request = urllib.request.Request(f"{API_URL}?{query}", headers={"User-Agent": USER_AGENT})
    with open_with_retry(request, timeout=60) as response:
        return json.load(response)


def open_with_retry(request: urllib.request.Request, timeout: int, attempts: int = 6):
    for attempt in range(attempts):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == attempts - 1:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** (attempt + 1)
            time.sleep(delay)
    raise RuntimeError("unreachable")


def search(query: str, limit: int, thumbnail_width: int) -> list[dict]:
    payload = api_get(
        {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 6,
            "gsrlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|sha1|extmetadata",
            "iiurlwidth": thumbnail_width,
        }
    )
    return payload.get("query", {}).get("pages", [])


def metadata_value(info: dict, key: str) -> str:
    return info.get("extmetadata", {}).get(key, {}).get("value", "")


def safe_stem(title: str) -> str:
    stem = re.sub(r"^File:", "", title, flags=re.IGNORECASE)
    stem = re.sub(r"\.(jpe?g|png|webp)$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return stem[:100] or "commons_image"


def download(url: str, destination: Path, fallback_url: str) -> tuple[str, int, str]:
    digest = hashlib.sha256()
    total = 0
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        response = open_with_retry(request, timeout=120, attempts=2)
        used_url = url
    except urllib.error.HTTPError as error:
        if error.code != 429 or fallback_url == url:
            raise
        request = urllib.request.Request(fallback_url, headers={"User-Agent": USER_AGENT})
        response = open_with_retry(request, timeout=120, attempts=4)
        used_url = fallback_url
    with response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total, used_url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/military/damage/source/wikimedia_commons_candidates"),
    )
    parser.add_argument("--per-query", type=int, default=12)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--thumbnail-width", type=int, default=1024)
    parser.add_argument(
        "--catalog-only",
        action="store_true",
        help="Write source/license records without downloading missing media.",
    )
    parser.add_argument("--delay", type=float, default=0.1)
    args = parser.parse_args()

    image_dir = args.output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.csv"
    raw_path = args.output / "api_records.json"

    records_by_page: dict[int, tuple[dict, str, str, str]] = {}
    raw_pages: list[dict] = []
    for query, damage, components in SEARCHES:
        for page in search(query, args.per_query, args.thumbnail_width):
            raw_pages.append({"search_query": query, "page": page})
            records_by_page.setdefault(page["pageid"], (page, query, damage, components))
        time.sleep(args.delay)

    raw_path.write_text(json.dumps(raw_pages, indent=2, ensure_ascii=False), encoding="utf-8")
    row_count = 0
    downloaded_count = 0
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        for page_id, (page, query, damage, components) in list(records_by_page.items())[: args.max_images]:
            infos = page.get("imageinfo", [])
            if not infos:
                continue
            info = infos[0]
            mime = info.get("mime", "")
            if mime not in {"image/jpeg", "image/png", "image/webp"}:
                continue
            if info.get("width", 0) < 400 or info.get("height", 0) < 300:
                continue

            download_url = info.get("thumburl") or info.get("url")
            suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[mime]
            filename = f"commons_{page_id}_{safe_stem(page['title'])}{suffix}"
            destination = image_dir / filename
            if destination.exists():
                content = destination.read_bytes()
                sha256, byte_count = hashlib.sha256(content).hexdigest(), len(content)
                local_file = f"images/{filename}"
                download_status = "downloaded"
                downloaded_count += 1
            elif args.catalog_only:
                sha256, byte_count = "", ""
                local_file = ""
                download_status = "url_only"
            else:
                sha256, byte_count, download_url = download(download_url, destination, info.get("url", download_url))
                local_file = f"images/{filename}"
                download_status = "downloaded"
                downloaded_count += 1
                time.sleep(args.delay)

            writer.writerow(
                {
                "local_file": local_file,
                "sha256": sha256,
                "bytes": byte_count,
                "download_status": download_status,
                "width": info.get("thumbwidth") or info.get("width", ""),
                "height": info.get("thumbheight") or info.get("height", ""),
                "commons_page_id": page_id,
                "commons_title": page["title"],
                "commons_page_url": info.get("descriptionurl", ""),
                "download_url": download_url,
                "original_url": info.get("url", ""),
                "mime": mime,
                "license_short_name": clean_markup(metadata_value(info, "LicenseShortName")),
                "license_url": clean_markup(metadata_value(info, "LicenseUrl")),
                "artist": clean_markup(metadata_value(info, "Artist")),
                "credit": clean_markup(metadata_value(info, "Credit")),
                "description": clean_markup(metadata_value(info, "ImageDescription")),
                "search_query": query,
                "candidate_damage_level": damage,
                "candidate_components": components,
                "source_asset_id": f"commons:{page_id}",
                "group_id": "",
                "split": "",
                "review_status": "pending",
                "review_notes": "Search-derived candidates; verify visible damage and component scope.",
                }
            )
            output.flush()
            row_count += 1
            print(f"[{row_count}] {filename}", flush=True)

    print(f"Cataloged {row_count} candidates; {downloaded_count} are present in {image_dir}")
    print(f"Wrote provenance manifest to {manifest_path}")


if __name__ == "__main__":
    main()
