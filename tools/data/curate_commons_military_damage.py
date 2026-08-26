"""Apply metadata review decisions to the Commons military-damage catalog."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


# These pages are demonstrable search-engine false positives: the metadata is
# about damaged buildings/people, a civilian vehicle, a toy, or a non-vehicle
# use of "tank"/"communications". They remain in the raw manifest for audit.
REJECT_REASONS = {
    22991945: "Weapons-cache photo; metadata does not describe a damaged vehicle.",
    58030376: "Portrait/event photo about an injured veteran; no damaged vehicle described.",
    122394295: "Speaker/event photo; no damaged vehicle described.",
    122394297: "Speaker/event photo; no damaged vehicle described.",
    122394299: "Speaker/event photo; no damaged vehicle described.",
    122394301: "Speaker/event photo; no damaged vehicle described.",
    122394421: "Speaker/event photo; no damaged vehicle described.",
    16138908: "Industrial oil tank, not a military vehicle.",
    27188650: "Rail tank car, not a military vehicle.",
    114318152: "Rail tank wagons, not military vehicles.",
    119124431: "Building damaged by a tank; target vehicle damage is not described.",
    95684632: "Toy truck, not a real military vehicle.",
    17645257: "Wood-gas generator caused a lexical 'burning' match; vehicle is not burning.",
    43693998: "Human casualty image; no target vehicle described.",
    112313791: "Civilian car in a riot, not a military vehicle.",
    136535445: "Civilian passenger car, not a target military vehicle.",
    154576887: "Generic fire-service training vehicles, not military vehicles.",
    71691464: "Civilian solar rickshaw; 'communications' refers to a ministry.",
    71691469: "Civilian solar rickshaw; 'communications' refers to a ministry.",
    71895900: "Government event; no target military vehicle.",
    72078129: "Civilian electric vehicle; 'communications' refers to a ministry.",
    72323843: "Government event; no target military vehicle.",
    108864268: "War memorial, not a military vehicle.",
}

# Metadata explicitly describes an intact vehicle or shows it operating in a
# damaged environment. These are useful negative examples, not damage labels.
NO_VISIBLE_DAMAGE_IDS = {
    8080687,
    23031407,
    23031458,
    23032024,
    23032230,
    23075829,
    41520374,
    1677646,
    8084443,
    26358373,
    35758596,
    39970761,
    41208423,
    41632643,
    41650037,
    51074379,
    53638537,
    15039774,
    15039805,
    23295446,
    25272463,
    25272518,
    39641889,
    165962135,
    12104401,
    21363185,
    179777388,
    179777389,
    115717505,
    35269977,
    72383140,
    72567096,
}

GROUPS = {
    **{page_id: "commons_humvee_breaching_bumper_series" for page_id in (23031407, 23031458, 23032024, 23032230)},
    **{page_id: "commons_humvee_tow_training_2015_series" for page_id in (41632643, 41650037, 51074379, 53638537)},
    **{page_id: "commons_smsgt_del_toro_event_series" for page_id in (122394295, 122394297, 122394299, 122394301, 122394421)},
    **{page_id: "commons_matilda_tobruk_series" for page_id in (15071166, 25732913)},
    **{page_id: "commons_m113_mine_pair" for page_id in (120369563, 120369996)},
    **{page_id: "commons_long_binh_damage_series" for page_id in (161544064, 161544227, 161544431, 161548866, 161549268, 161549389, 161549669, 161549864, 161550361)},
    **{page_id: "commons_iraqi_ifa_w50_pair" for page_id in (9937003, 9937053)},
    **{page_id: "commons_iraqi_tanker_pair" for page_id in (9906971, 9907029)},
    **{page_id: "commons_werkamba_truck_pair" for page_id in (142113321, 142113324)},
    **{page_id: "commons_chernogolovka_museum_series" for page_id in (179777388, 179777389)},
    **{page_id: "commons_panjshir_vehicle_uploads" for page_id in (115180999, 115199367)},
}

COMPONENTS_BY_QUERY = {
    "destroyed Humvee": "windshield;wheel;hull",
    "damaged Humvee": "windshield;wheel;hull",
    "burned Humvee": "windshield;wheel;hull",
    "destroyed tank": "track;hull;mounted_gun",
    "damaged tank": "track;hull;mounted_gun",
    "battle damaged tank": "track;hull;mounted_gun",
    "destroyed armored personnel carrier": "wheel;track;hull;mounted_gun",
    "damaged armored personnel carrier": "wheel;track;hull;mounted_gun",
    "destroyed military truck": "windshield;wheel;hull",
    "damaged military truck": "windshield;wheel;hull",
    "burning military vehicle": "windshield;wheel;track;hull",
    "damaged military communications vehicle": "windshield;wheel;track;hull;comms_system",
    "destroyed self-propelled anti-aircraft gun": "track;hull;comms_system;mounted_gun",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/military/damage/source/wikimedia_commons_candidates/manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/military/damage/source/wikimedia_commons_candidates/curated_manifest.csv"),
    )
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
        fields = list(rows[0]) if rows else []

    curated = []
    for row in rows:
        page_id = int(row["commons_page_id"])
        if page_id in REJECT_REASONS:
            continue
        row["review_status"] = "metadata_match_pending_visual_review"
        row["review_notes"] = (
            "Metadata indicates a relevant military vehicle. Verify component visibility, damage level, "
            "near-duplicates, and artifacts before training."
        )
        if page_id in NO_VISIBLE_DAMAGE_IDS:
            row["candidate_damage_level"] = "no_visible_damage"
            row["review_notes"] = (
                "Metadata indicates an intact military vehicle or damage only in the surrounding scene; "
                "verify no damage cue is visible on each eventual component crop."
            )
        if page_id == 155220944:
            row["candidate_damage_level"] = "severe_visible_damage"
        row["candidate_components"] = COMPONENTS_BY_QUERY[row["search_query"]]
        row["group_id"] = GROUPS.get(page_id, row["source_asset_id"])
        curated.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(curated)

    print(f"Kept {len(curated)} of {len(rows)} records after metadata review")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
