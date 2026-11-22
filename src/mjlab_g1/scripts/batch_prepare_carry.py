"""Convert audited SUGAR CarryBox clips and rank by MuJoCo reference contact.

Geometric contact is not dynamic feasibility: scores flag penetrations that
must be corrected by refiner/collision-aware retargeting before policy training.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path

from convert_sugar_carry import convert
from probe_carry_contact import run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--g1-xml", type=Path, required=True)
    p.add_argument("--box-mesh", type=Path, required=True)
    args = p.parse_args()
    candidates = json.loads(args.audit.read_text(encoding="utf-8"))["candidate_clips"]
    records = []
    for count, name in enumerate(candidates, 1):
        target = args.output / ("carry_place_sugar_" + name.split("_")[-1] + ".npz")
        try:
            with redirect_stdout(io.StringIO()):
                info = convert(args.source / name, target, args.g1_xml)
            probe = run(target, args.g1_xml, args.box_mesh)
            records.append({"clip": name, "status": "converted", "frames": info["frames"],
                            "grasp": info["grasp_frame"], "release": info["release_frame"],
                            "fk_max_mm": info["max_g1_fk_error_mm"],
                            "fk_p95_mm": info["p95_g1_fk_error_mm"],
                            "both_hand_fraction": probe["kinematic_both_hand_fraction_during_source_contact"],
                            "either_hand_fraction": probe["kinematic_either_hand_fraction_during_source_contact"],
                            "penetration_p95_m": probe["hand_contact_penetration_p95_m"],
                            "penetration_max_m": probe["hand_contact_penetration_max_m"],
                            "floor_last_0p3s_fraction": probe["kinematic_floor_contact_last_0p3s_fraction"],
                            "non_hand_contacts": probe["non_hand_box_contacts"], "error": ""})
        except (ValueError, OSError, RuntimeError) as e:
            target.unlink(missing_ok=True)
            records.append({"clip": name, "status": "rejected", "error": str(e)})
        if count % 20 == 0 or count == len(candidates):
            print(f"Audited {count}/{len(candidates)} clips; "
                  f"converted {sum(r['status'] == 'converted' for r in records)}", flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    fields = ("clip", "status", "frames", "grasp", "release", "fk_max_mm",
              "fk_p95_mm", "both_hand_fraction", "either_hand_fraction",
              "penetration_p95_m", "penetration_max_m", "floor_last_0p3s_fraction",
              "non_hand_contacts", "error")
    with args.report.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        writer.writerows(records)
    valid = [r for r in records if r["status"] == "converted"
             and r["penetration_p95_m"] is not None]
    valid.sort(key=lambda r: (r["penetration_p95_m"], -r["both_hand_fraction"]))
    print("Least penetrating references:")
    for r in valid[:10]:
        print({k: r[k] for k in ("clip", "frames", "both_hand_fraction",
                                 "penetration_p95_m", "penetration_max_m")})
    print(f"Rejected: {[r['clip'] for r in records if r['status'] == 'rejected']}")


if __name__ == "__main__":
    main()
