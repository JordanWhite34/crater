"""Prepare the fixed 16-image Commons pilot used by the damage notebook."""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

from damage_crop_annotation import sha256_file
from download_commons_military_damage import download, safe_stem


SELECTION = (
    (8080687, "none", "Intact wheeled Humvee; windshield, wheels, and body visible."),
    (41650037, "none", "Intact weapon-equipped Humvee for mission-equipment negatives."),
    (72383140, "none", "Intact external communications vehicle."),
    (15039774, "none", "Intact tracked recovery vehicle."),
    (12104401, "none", "Intact military truck with windshield and wheels."),
    (17479467, "moderate", "Gunshot damage to a Humvee."),
    (23011135, "moderate", "Mine/suspension damage to a Humvee."),
    (861502, "moderate", "Localized M113 damage."),
    (10482322, "moderate", "APC damage from armor-piercing rounds."),
    (9973769, "moderate", "Damaged tracked tank."),
    (9906971, "moderate", "Damaged wheeled tanker truck."),
    (10628551, "moderate", "Damaged military truck towing a field gun."),
    (41464040, "severe", "Humvee destroyed in an attack."),
    (155220944, "severe", "Heavily battle-damaged armored Humvee."),
    (744327, "severe", "Destroyed tracked PT-76."),
    (10613225, "severe", "Destroyed tracked Iraqi tank."),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("datasets/military/damage/source/wikimedia_commons_candidates/curated_manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/military/damage/source/commons_labeling_subset_v1"),
    )
    parser.add_argument("--delay", type=float, default=3.0)
    args = parser.parse_args()

    with args.catalog.open(newline="", encoding="utf-8-sig") as source:
        catalog_rows = {int(row["commons_page_id"]): row for row in csv.DictReader(source)}
    missing_ids = [page_id for page_id, _, _ in SELECTION if page_id not in catalog_rows]
    if missing_ids:
        raise ValueError(f"Selected Commons page IDs missing from catalog: {missing_ids}")

    image_dir = args.output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = args.output / "selection_manifest.csv"
    extra_fields = (
        "subset_order",
        "selection_damage_band",
        "selection_reason",
        "subset_local_file",
        "subset_sha256",
        "subset_bytes",
        "subset_download_url",
        "subset_status",
        "subset_error",
    )
    catalog_fields = list(next(iter(catalog_rows.values())).keys())
    failures: list[str] = []

    with output_manifest.open("w", newline="", encoding="utf-8-sig") as destination:
        writer = csv.DictWriter(destination, fieldnames=extra_fields + tuple(catalog_fields))
        writer.writeheader()
        for subset_order, (page_id, damage_band, reason) in enumerate(SELECTION, start=1):
            row = catalog_rows[page_id]
            mime = row["mime"]
            suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[mime]
            filename = f"commons_{page_id}_{safe_stem(row['commons_title'])}{suffix}"
            output_path = image_dir / filename
            used_url = row["download_url"] or row["original_url"]
            error_text = ""
            try:
                if output_path.exists():
                    file_hash = sha256_file(output_path)
                    byte_count = output_path.stat().st_size
                else:
                    source_local = (args.catalog.parent / row["local_file"]).resolve() if row["local_file"] else None
                    if source_local is not None and source_local.is_file():
                        shutil.copy2(source_local, output_path)
                        file_hash = sha256_file(output_path)
                        byte_count = output_path.stat().st_size
                        used_url = "copied_from_curated_cache"
                    else:
                        file_hash, byte_count, used_url = download(
                            used_url,
                            output_path,
                            row["original_url"] or used_url,
                        )
                        time.sleep(args.delay)
                status = "ready"
            except Exception as error:  # Preserve all successful downloads and provenance on partial failure.
                status = "failed"
                error_text = f"{type(error).__name__}: {error}"
                file_hash, byte_count = "", ""
                failures.append(f"{page_id}: {error_text}")

            writer.writerow(
                {
                    "subset_order": subset_order,
                    "selection_damage_band": damage_band,
                    "selection_reason": reason,
                    "subset_local_file": f"images/{filename}" if status == "ready" else "",
                    "subset_sha256": file_hash,
                    "subset_bytes": byte_count,
                    "subset_download_url": used_url,
                    "subset_status": status,
                    "subset_error": error_text,
                    **row,
                }
            )
            destination.flush()
            print(f"[{subset_order:02d}/{len(SELECTION)}] {page_id} {damage_band}: {status}", flush=True)

    print(f"Wrote {output_manifest}")
    if failures:
        raise RuntimeError("Subset preparation had failures:\n- " + "\n- ".join(failures))


if __name__ == "__main__":
    main()
