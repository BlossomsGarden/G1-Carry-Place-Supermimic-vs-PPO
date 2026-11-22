"""Extract one audited physical episode for rendering from an all-world NPZ.

No state is synthesized or replaced by the demonstration. Metadata is checked
against the original eval_track.py physical-score JSON before writing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout-all-npz", required=True, type=Path)
    p.add_argument("--physical-json", required=True, type=Path)
    p.add_argument("--episode-index", required=True, type=int)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    report = json.loads(args.physical_json.read_text(encoding="utf-8"))
    with np.load(args.rollout_all_npz, allow_pickle=False) as source:
        index = args.episode_index
        n = int(source["state"].shape[1])
        if not 0 <= index < n or report["episodes"] != n:
            raise ValueError("Selected episode is not in the scored physical batch")
        ckpt = str(source["checkpoint"].item())
        seed = int(source["seed"])
        randomized = bool(source["randomized"])
        motion_file = Path(str(source["motion_file"].item()))
        if (report["checkpoint"] != ckpt or report["seed"] != seed
                or (report["condition"] == "randomized") != randomized
                or report["motion_file"] != str(motion_file)):
            raise ValueError("Checkpoint, seed, randomization or motion do not match physical score")
        episode = report["per_episode"][index]
        stop = int(source["episode_steps"][index]) + 1
        if stop != episode["episode_steps"] + 1:
            raise ValueError("Episode length does not match physical score")
        selected = {name: source[name][:stop, index]
                    for name in ("state", "left_hand_contact", "right_hand_contact", "floor_contact")}
    with np.load(motion_file, allow_pickle=False) as reference:
        target_pos = reference["target_pos_w"].copy()
        target_quat = reference["target_quat_wxyz"].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **selected,
                        both_hands_contact=selected["left_hand_contact"] & selected["right_hand_contact"],
                        episode_complete=episode["completed"],
                        terminated=episode["terminated"],
                        episode_success=episode["success"],
                        episode_index=index, seed=seed, randomized=randomized,
                        checkpoint=ckpt, target_pos_w=target_pos,
                        target_quat_wxyz=target_quat, motion_file=str(motion_file))
    print(f"Wrote episode {index} physical states {stop} frames; strict success={episode['success']}")


if __name__ == "__main__":
    main()
