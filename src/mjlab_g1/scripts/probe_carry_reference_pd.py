"""Try an ideal reference joint-position servo in a real CarryPlace rollout.

This diagnostic bypasses the learned policy and sees the future reference;
failure is informative, but a successful oracle is not policy success.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mjlab_g1.envs.smtrack_env import SMTrackEnv
from mjlab_g1.scripts.eval_track import _carry_contacts
from mjlab_g1.tasks.smtrack import make_env_cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--motion", type=Path, default=None)
    args = p.parse_args()
    cfg = make_env_cfg("G1-SMTrack-CarryPlace", play=True, motion_file=args.motion)
    cfg.scene.num_envs = 1
    cfg.auto_reset = False
    cfg.seed = 0
    env = SMTrackEnv(cfg, device=args.device)
    env.reset()
    term = env.action_manager.get_term("joint_pos")
    cmd = env.motion_command
    sample = []
    with torch.no_grad():
        for t in range(cmd.num_frames - 1):
            k = env.get_ref_index()
            desired = cmd.motion.joint_pos[k][:, term.target_ids]
            action = (desired - term.offset) / term.scale
            _, _, failed, clipped, _ = env.step(action)
            two_hands, on_floor = _carry_contacts(env)
            state = env.get_state()[0]
            sample.append([float(state[73]), float(state[2]),
                           bool(two_hands[0]), bool(on_floor[0])])
            if bool(failed[0]) or bool(clipped[0]):
                break
    with np.load(env.motion_file) as reference:
        gr, rel = int(reference["grasp_frame"]), int(reference["release_frame"])
    report = {"requested_steps": cmd.num_frames - 1, "actual_steps": len(sample),
              "grasp_reference_frame": gr, "release_reference_frame": rel,
              "reached_grasp_frame": len(sample) > gr,
              "reached_release_frame": len(sample) > rel,
              "box_max_height_m": max(x[0] for x in sample),
              "pelvis_min_height_m": min(x[1] for x in sample),
              "real_sim_both_hand_contact_steps": sum(x[2] for x in sample),
              "terminated_by": {name: bool(env.termination_manager.get_term(name)[0])
                                for name in env.termination_manager.active_terms},
              "reference_pelvis_z_at_end_m": float(cmd.motion.body_pos_w[
                  min(len(sample), cmd.num_frames - 1), 0, 2]),
              "last_box_xyz": env.get_state()[0, 71:74].detach().cpu().tolist(),
              "scope": "oracle joint position targets in physics; not trained-policy success"}
    print(json.dumps(report, indent=2))
    env.close()


if __name__ == "__main__":
    main()
