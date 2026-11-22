"""Audit all SUGAR CarryBox clips before converting them to mjlab references.

The source contact label is an annotation, not a measured MuJoCo contact.
All times below use the actual joint array length, clipped to the object array.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import pickle

import numpy as np


def audit(folder: Path) -> dict:
    with np.load(folder / "robot_50hz.npz") as robot:
        fps = float(np.asarray(robot["fps"]).item())
        joints = np.asarray(robot["joint_pos"])
        root = np.asarray(robot["body_pos_w"])[:, 0]
    with (folder / "obj_motion_global_50hz.pkl").open("rb") as file:
        obj = pickle.load(file)
    pos = np.asarray(obj["obj_trans"])
    annotation = np.load(folder / "contact_labels_50hz.npy").astype(bool)
    length = min(len(joints), len(root), len(pos), len(annotation))
    if fps != 50 or joints.shape[1] != 29 or length < 100:
        raise ValueError(f"Unexpected source clip {folder.name}: fps={fps}, joints={joints.shape}")
    pos, root, annotation = pos[:length], root[:length], annotation[:length]
    edges = np.diff(np.pad(annotation.astype(int), (1, 1)))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    segments = [(int(a), int(b)) for a, b in zip(starts, stops)]
    if segments:
        start, release = max(segments, key=lambda x: x[1] - x[0])
    else:
        start = release = -1
    floor_z = float(np.median(np.r_[pos[:max(10, length // 12), 2],
                                      pos[-max(10, length // 12):, 2]]))
    box_v = np.linalg.norm(np.diff(pos, axis=0), axis=1) * fps
    root_v = np.linalg.norm(np.diff(root, axis=0), axis=1) * fps
    # A reference is suitable for the intended task only if it has an annotated
    # grasp, displacement, a lowered box, and a distinct no-contact hold at end.
    displacement = float(np.linalg.norm(pos[-1, :2] - pos[0, :2]))
    final_z_err = float(abs(pos[-1, 2] - np.median(pos[:20, 2])))
    qualifies = bool(
        start >= 5 and release < length - 20 and release - start >= 40
        and displacement >= 0.5 and pos[:, 2].max() - floor_z >= 0.15
        and final_z_err <= 0.08 and np.max(box_v) < 3.0
        and np.max(root_v) < 3.0
    )
    return {
        "clip": folder.name,
        "frames": length,
        "duration_s": round(length / fps, 2),
        "robot_frames": len(joints),
        "object_frames": len(obj["obj_trans"]),
        "contact_frames": len(np.load(folder / "contact_labels_50hz.npy")),
        "grasp_frame": start,
        "release_frame": release,
        "annotated_contact_segments": len(segments),
        "object_scale": float(obj["obj_scale"]),
        "object_floor_z_m": round(float(np.median(pos[:20, 2])), 4),
        "object_max_z_m": round(float(pos[:, 2].max()), 4),
        "object_final_z_m": round(float(pos[-1, 2]), 4),
        "object_displacement_m": round(displacement, 3),
        "robot_displacement_m": round(float(np.linalg.norm(root[-1, :2] - root[0, :2])), 3),
        "max_box_speed_mps": round(float(box_v.max()), 3),
        "max_root_speed_mps": round(float(root_v.max()), 3),
        "end_0p4s_box_speed_mps": round(float(box_v[-20:].max()), 3),
        "qualifies_geometrically": qualifies,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--summary", required=True, type=Path)
    args = p.parse_args()
    rows = [audit(d) for d in sorted(args.input.glob("data_*"))]
    if not rows:
        raise ValueError(f"No CarryBox clips under {args.input}")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source": "SUGAR official processed data.zip, CarryBox only",
        "clips": len(rows),
        "source_robot_frames": sum(r["robot_frames"] for r in rows),
        "aligned_frames": sum(r["frames"] for r in rows),
        "fps": 50,
        "qualifying_clips": sum(r["qualifies_geometrically"] for r in rows),
        "candidate_clips": [r["clip"] for r in rows if r["qualifies_geometrically"]],
        "limitations": "Geometric check only; contact and dynamic feasibility require MuJoCo replay.",
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Selected examples:")
    for row in rows[:3]:
        print(row)


if __name__ == "__main__":
    main()
