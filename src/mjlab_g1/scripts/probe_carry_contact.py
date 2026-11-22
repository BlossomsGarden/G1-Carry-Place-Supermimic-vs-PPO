"""Check SUGAR reference geometry/contact against the *MuJoCo* G1 and box.

The kinematic replay does not exert forces. This probe counts potential
collisions and explicitly compares them with SUGAR's provided contact labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from render_carry_reference import build_model


def run(motion: Path, g1_xml: Path, box_mesh: Path) -> dict:
    with np.load(motion) as z:
        data = {name: z[name] for name in z.files}
    model = build_model(g1_xml, box_mesh, data["target_pos_w"])
    mj = mujoco.MjData(model)
    box_geom = model.geom("box_collision").id
    floor_geom = model.geom("floor").id
    length = len(data["joint_pos"])
    left = np.zeros(length, dtype=bool)
    right = np.zeros(length, dtype=bool)
    floor = np.zeros(length, dtype=bool)
    left_foot = np.zeros(length, dtype=bool)
    right_foot = np.zeros(length, dtype=bool)
    non_hand_contacts = np.zeros(length, dtype=int)
    min_contact_dist = np.full(length, np.nan)
    sample_contacts = {}
    hand_penetrations = []
    for k in range(length):
        mj.qpos[:3] = data["body_pos_w"][k, 0]
        mj.qpos[3:7] = data["body_quat_w"][k, 0]
        mj.qpos[7:36] = data["joint_pos"][k]
        mj.qpos[36:39] = data["box_pos_w"][k]
        mj.qpos[39:43] = data["box_quat_w"][k]
        mujoco.mj_forward(model, mj)
        contacts = []
        for contact in mj.contact[:mj.ncon]:
            pair = (int(contact.geom1), int(contact.geom2))
            if floor_geom in pair and box_geom not in pair:
                other = pair[0] if pair[1] == floor_geom else pair[1]
                geom_name = model.geom(other).name
                left_foot[k] |= geom_name.startswith("left_foot")
                right_foot[k] |= geom_name.startswith("right_foot")
            if box_geom not in pair:
                continue
            other = pair[0] if pair[1] == box_geom else pair[1]
            if other == floor_geom:
                floor[k] = True
                continue
            body = model.body(model.geom_bodyid[other]).name
            is_hand = "wrist" in body or "hand" in body
            if body.startswith("left_") and is_hand:
                left[k] = True
                hand_penetrations.append(max(0.0, -float(contact.dist)))
            elif body.startswith("right_") and is_hand:
                right[k] = True
                hand_penetrations.append(max(0.0, -float(contact.dist)))
            else:
                non_hand_contacts[k] += 1
            contacts.append((body, round(float(contact.dist), 4)))
        if contacts:
            min_contact_dist[k] = min(d for _, d in contacts)
        if k in (0, int(data["grasp_frame"]), int(data["grasp_frame"]) + 30,
                 length // 2, int(data["release_frame"]) - 5, length - 1):
            sample_contacts[str(k)] = contacts[:12]
    grasp = int(data["grasp_frame"])
    release = int(data["release_frame"])
    both = left & right
    annotated = np.asarray(data["contact_label"], dtype=bool)
    result = {
        "source_clip": str(data["source_clip"].item()),
        "frames": length,
        "grasp_frame": grasp,
        "release_frame": release,
        "source_annotated_contact_frames": int(annotated.sum()),
        "kinematic_left_hand_box_contact_frames": int(left.sum()),
        "kinematic_right_hand_box_contact_frames": int(right.sum()),
        "kinematic_both_hand_box_contact_frames": int(both.sum()),
        "kinematic_both_hand_fraction_during_source_contact": round(float(both[grasp:release].mean()), 3),
        "kinematic_either_hand_fraction_during_source_contact": round(float((left | right)[grasp:release].mean()), 3),
        "kinematic_floor_contact_before_grasp_fraction": round(float(floor[:grasp].mean()), 3),
        "kinematic_floor_contact_after_release_fraction": round(float(floor[release:].mean()), 3),
        "kinematic_floor_contact_last_0p3s_fraction": round(float(floor[-15:].mean()), 3),
        "kinematic_any_foot_floor_fraction_before_grasp": round(
            float((left_foot | right_foot)[:grasp].mean()), 3),
        "reference_pelvis_height_start_m": round(float(data["body_pos_w"][0, 0, 2]), 4),
        "reference_pelvis_height_at_grasp_m": round(float(data["body_pos_w"][grasp, 0, 2]), 4),
        "kinematic_any_foot_floor_fraction_during_carry": round(
            float((left_foot | right_foot)[grasp:release].mean()), 3),
        "kinematic_both_foot_floor_fraction_during_carry": round(
            float((left_foot & right_foot)[grasp:release].mean()), 3),
        "hand_contact_penetration_p95_m": round(float(np.quantile(hand_penetrations, 0.95)), 4) if hand_penetrations else None,
        "hand_contact_penetration_max_m": round(float(max(hand_penetrations)), 4) if hand_penetrations else None,
        "non_hand_box_contacts": int(non_hand_contacts.sum()),
        "sample_contacts": sample_contacts,
        "scope": "Kinematic geometric contacts only, not a physically simulated rollout or policy success",
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--g1-xml", type=Path, required=True)
    p.add_argument("--box-mesh", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = run(args.motion, args.g1_xml, args.box_mesh)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
