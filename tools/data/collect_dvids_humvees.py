#!/usr/bin/env python3
"""Discover, review, and import DVIDS Humvee photos for CRATER.

The DVIDS website search is used for discovery. Original photos are downloaded
through DVIDS' signed-in "Download Photo" control, then imported from the
browser download directory after visual review. This avoids storing credentials
or session cookies in the repository.
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import re
import shutil
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFont


DATASET = Path(__file__).resolve().parents[2] / "datasets/military/components/dvids_humvee_unlabeled"
MANIFEST = DATASET / "MANIFEST.csv"
SEARCH_URL = "https://www.dvidshub.net/search/?view=grid&sort=publishdate&q={query}&type=image&page={page}"
PAGE_URL = "https://www.dvidshub.net/image/{photo_id}"
USER_AGENT = "CRATER-DVIDS-curator/1.0"
FIELDS = [
    "filename", "dvids_photo_id", "title", "date_taken", "location",
    "original_resolution", "photographer_or_credit", "page_status",
    "source_url", "curation_note",
]


def get_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


class SearchResultsParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[dict[str, str]] = []
        self.current: dict[str, str] | None = None
        self.article_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "article" and "image__cell" in (a.get("class") or "").split():
            self.current = {"photo_id": "", "title": "", "alt": "", "thumbnail_url": ""}
            self.article_depth = 1
            return
        if self.current is None:
            return
        if tag == "article":
            self.article_depth += 1
        if a.get("id", "").startswith("images_"):
            self.current["photo_id"] = a["id"].split("_", 1)[1]
            self.current["title"] = a.get("title", "") or self.current["title"]
        if tag == "img":
            self.current["alt"] = a.get("alt", "")
            self.current["thumbnail_url"] = a.get("src", "")

    def handle_endtag(self, tag: str) -> None:
        if self.current is None or tag != "article":
            return
        self.article_depth -= 1
        if self.article_depth <= 0:
            if not self.current["photo_id"]:
                match = re.search(r"/photos/\d+/(\d+)/", self.current["thumbnail_url"])
                if match:
                    self.current["photo_id"] = match.group(1)
            if self.current["photo_id"]:
                if not self.current["title"]:
                    self.current["title"] = self.current["alt"]
                self.rows.append(self.current)
            self.current = None


class PageTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text_parts: list[str] = []
        self.title = ""
        self.credit = ""
        self._in_h1 = False
        self._in_h3 = False
        self._h1_parts: list[str] = []
        self._h3_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "h1":
            self._in_h1 = True
            self._h1_parts = []
        elif tag == "h3":
            self._in_h3 = True
            self._h3_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "h1" and self._in_h1:
            self.title = " ".join(self._h1_parts).strip()
            self._in_h1 = False
        elif tag == "h3" and self._in_h3:
            value = " ".join(self._h3_parts).strip()
            if value.startswith("Photo by"):
                self.credit = value
            self._in_h3 = False

    def handle_data(self, data: str) -> None:
        value = data.strip()
        if not value:
            return
        self.text_parts.append(value)
        if self._in_h1:
            self._h1_parts.append(value)
        if self._in_h3:
            self._h3_parts.append(value)


def parse_search_page(query: str, page: int) -> list[dict[str, str]]:
    url = SEARCH_URL.format(query=quote_plus(query), page=page)
    parser = SearchResultsParser()
    parser.feed(get_text(url))
    for row in parser.rows:
        row["search_query"] = query
        row["source_url"] = PAGE_URL.format(photo_id=row["photo_id"])
    return parser.rows


def discover(args: argparse.Namespace) -> None:
    all_rows: dict[str, dict[str, str]] = {}
    for query in args.query:
        for page in range(1, args.pages + 1):
            rows = parse_search_page(query, page)
            if not rows:
                break
            for row in rows:
                prior = all_rows.setdefault(row["photo_id"], row)
                prior["search_query"] = "; ".join(dict.fromkeys(
                    (prior.get("search_query", "") + "; " + query).split("; ")
                ))
            print(f"{query}: page {page}, {len(rows)} images ({len(all_rows)} unique)")
            time.sleep(args.delay)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["photo_id", "title", "alt", "search_query", "thumbnail_url", "source_url"]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in all_rows.values():
            writer.writerow({key: row.get(key, "") for key in columns})
    print(f"Wrote {len(all_rows)} candidates to {args.output}")


def make_contact_sheets(args: argparse.Namespace) -> None:
    with args.candidates.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    columns, rows_per_sheet = args.columns, args.rows
    tile_w, tile_h = args.tile_width, args.tile_height
    per_sheet = columns * rows_per_sheet
    session_rows: list[tuple[dict[str, str], Image.Image]] = []
    for index, row in enumerate(rows):
        url = row.get("thumbnail_url", "")
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=30) as response:
                image = Image.open(io.BytesIO(response.read())).convert("RGB")
            session_rows.append((row, image))
        except Exception as error:
            print(f"Skipping preview {row.get('photo_id')}: {error}", file=sys.stderr)
        if len(session_rows) and (len(session_rows) % per_sheet == 0 or index == len(rows) - 1):
            batch_start = len(session_rows) - per_sheet if len(session_rows) % per_sheet == 0 else (len(session_rows) // per_sheet) * per_sheet
            batch = session_rows[batch_start:]
            if not batch:
                continue
            sheet_no = (batch_start // per_sheet) + 1
            sheet = Image.new("RGB", (columns * tile_w, rows_per_sheet * tile_h), "white")
            draw = ImageDraw.Draw(sheet)
            for tile, (row, image) in enumerate(batch):
                x, y = (tile % columns) * tile_w, (tile // columns) * tile_h
                image.thumbnail((tile_w - 8, tile_h - 40))
                sheet.paste(image, (x + (tile_w - image.width) // 2, y + 2))
                draw.text((x + 4, y + tile_h - 34), row.get("photo_id", ""), fill="black", font=font)
                label = row.get("title", "").encode("ascii", "ignore").decode()[:30]
                draw.text((x + 4, y + tile_h - 20), label, fill="black", font=font)
            path = args.output_dir / f"dvids_contact_{sheet_no:03}.jpg"
            sheet.save(path, quality=90)
            print(path)


def get_page_metadata(photo_id: str) -> dict[str, str]:
    url = PAGE_URL.format(photo_id=photo_id)
    page = PageTextParser()
    page.feed(get_text(url))
    text = " ".join(page.text_parts)

    def capture(pattern: str) -> str:
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    status = "PUBLIC DOMAIN" if re.search(r"\bPUBLIC DOMAIN\b", text, re.IGNORECASE) else "UNVERIFIED"
    return {
        "title": page.title,
        "date_taken": capture(r"Date Taken:\s*([0-9.]+)"),
        "location": capture(r"Location:\s*(.*?)(?:\s+Web Views:|\s+Downloads:|\s+Date Taken:|\s+Date Posted:|$)"),
        "original_resolution": capture(r"Resolution:\s*([0-9]+x[0-9]+)"),
        "photographer_or_credit": page.credit,
        "page_status": status,
        "source_url": url,
    }


def finalize(args: argparse.Namespace) -> None:
    approved = [line.strip() for line in args.approved_ids.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    candidates: dict[str, dict[str, str]] = {}
    with args.candidates.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            candidates[row["photo_id"]] = row

    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        existing_rows = list(csv.DictReader(stream))
    existing_ids = {row["dvids_photo_id"] for row in existing_rows}

    verified_metadata: dict[str, dict[str, str]] = {}
    if args.verified_metadata:
        metadata_rows = json.loads(args.verified_metadata.read_text(encoding="utf-8"))
        verified_metadata = {str(row["id"]): row for row in metadata_rows}

    staged: list[tuple[dict[str, str], Path, str]] = []
    for photo_id in approved:
        if photo_id in existing_ids:
            print(f"Already in manifest, skipping {photo_id}")
            continue
        if photo_id not in candidates:
            raise SystemExit(f"Approved ID {photo_id} is missing from {args.candidates}")
        source = next((args.downloads / f"{photo_id}{suffix}" for suffix in (".jpg", ".jpeg", ".JPG", ".JPEG") if (args.downloads / f"{photo_id}{suffix}").is_file()), None)
        if source is None:
            raise SystemExit(f"Missing signed-in DVIDS download for {photo_id} in {args.downloads}")
        with Image.open(source) as image:
            image.verify()
            width, height = image.size
            if image.format != "JPEG":
                raise SystemExit(f"Expected a JPEG for {photo_id}, got {image.format}")
        if args.verified_metadata and photo_id not in verified_metadata:
            raise SystemExit(f"Approved ID {photo_id} is missing from verified metadata {args.verified_metadata}")
        meta = verified_metadata.get(photo_id) or get_page_metadata(photo_id)
        # The verified browser export uses concise field names; accept those as
        # well as the manifest field names to keep browser collection simple.
        meta = {
            **meta,
            "date_taken": meta.get("date_taken", meta.get("date", "")),
            "original_resolution": meta.get("original_resolution", meta.get("resolution", "")),
            "photographer_or_credit": meta.get("photographer_or_credit", meta.get("credit", "")),
            "source_url": meta.get("source_url", PAGE_URL.format(photo_id=photo_id)),
        }
        if meta["page_status"] != "PUBLIC DOMAIN":
            raise SystemExit(f"DVIDS page {photo_id} is not labeled PUBLIC DOMAIN; refusing to import it")
        search = candidates[photo_id]
        note = "Visually screened for Humvee/component prominence and usefulness to vehicle-part detection. "
        note += f"Search caption: {html.unescape(search.get('alt', '')).strip()}"
        if meta["original_resolution"] and meta["original_resolution"] != f"{width}x{height}":
            note += f" Imported file is {width}x{height}; DVIDS reports original resolution {meta['original_resolution']}."
        row = {
            **{key: meta.get(key, "") for key in (
                "date_taken", "location", "original_resolution",
                "photographer_or_credit", "page_status", "source_url",
            )},
            "title": meta.get("title", ""),
            "filename": f"dvids_{photo_id}.jpg",
            "dvids_photo_id": photo_id,
            "curation_note": note,
        }
        staged.append((row, source, f"{width}x{height}"))

    images_dir = DATASET / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    new_rows = []
    for row, source, _downloaded_resolution in staged:
        dest = images_dir / row["filename"]
        if dest.exists():
            raise SystemExit(f"Refusing to overwrite existing image: {dest}")
        shutil.copy2(source, dest)
        new_rows.append(row)

    with MANIFEST.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        for row in new_rows:
            writer.writerow(row)
    print(f"Imported {len(new_rows)} approved images into {DATASET}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    discover_parser = commands.add_parser("discover", help="Search DVIDS and save candidate metadata")
    discover_parser.add_argument("--query", action="append", required=True, help="Repeat for multiple search queries")
    discover_parser.add_argument("--pages", type=int, default=4)
    discover_parser.add_argument("--delay", type=float, default=0.2)
    discover_parser.add_argument("--output", type=Path, required=True)
    discover_parser.set_defaults(func=discover)

    contact_parser = commands.add_parser("contact-sheets", help="Download visible search previews for manual screening")
    contact_parser.add_argument("--candidates", type=Path, required=True)
    contact_parser.add_argument("--output-dir", type=Path, required=True)
    contact_parser.add_argument("--columns", type=int, default=5)
    contact_parser.add_argument("--rows", type=int, default=6)
    contact_parser.add_argument("--tile-width", type=int, default=220)
    contact_parser.add_argument("--tile-height", type=int, default=154)
    contact_parser.set_defaults(func=make_contact_sheets)

    finalize_parser = commands.add_parser("finalize", help="Import approved browser downloads into the dataset")
    finalize_parser.add_argument("--candidates", type=Path, required=True)
    finalize_parser.add_argument("--approved-ids", type=Path, required=True, help="One DVIDS photo ID per line")
    finalize_parser.add_argument("--downloads", type=Path, default=Path.home() / "Downloads")
    finalize_parser.add_argument(
        "--verified-metadata", type=Path,
        help="JSON exported from browser-verified DVIDS pages (avoids anonymous page checks)",
    )
    finalize_parser.set_defaults(func=finalize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
