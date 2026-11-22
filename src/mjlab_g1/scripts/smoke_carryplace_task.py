"""Create the CarryPlace MJWarp task and step a zero-action diagnostic rollout.

This is a task wiring check only; no learned policy or successful manipulation is claimed.
"""
from __future__ import annotations

import argparse
import json

import torch

from mjlab_g1.envs.smtrack_env import SMTrackEnv
from mjlab_g1.tasks.smtrack import make_env_cfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    cfg = make_env_cfg("G1-SMTrack-CarryPlace", play=True)
    cfg.scene.num_envs = 1
    cfg.auto_reset = False
    cfg.seed = 0
    env = SMTrackEnv(cfg, device=args.device)
    env.reset()  # construction allocates buffers; reset actually positions robot and box
    initial = env.get_state()[0, 71:74].detach().cpu().tolist()
    positions = []
    term = trunc = False
    with torch.no_grad():
        for _ in range(args.steps):
            _, _, terminated, truncated, _ = env.step(
                torch.zeros_like(env.action_manager.action)
            )
            positions.append(env.get_state()[0, 71:74].detach().cpu().tolist())
            term = bool(terminated[0].item())
            trunc = bool(truncated[0].item())
            if term or trunc:
                break
    print(json.dumps({
        "task": "G1-SMTrack-CarryPlace",
        "motion": env.motion_file,
        "device": args.device,
        "num_envs": env.num_envs,
        "state_dim": env.state_dim,
        "actions": list(env.action_manager.action.shape),
        "initial_box_xyz": initial,
        "first_box_xyz": positions[0] if positions else None,
        "last_box_xyz": positions[-1] if positions else None,
        "steps": len(positions),
        "terminated": term, "truncated": trunc,
        "scope": "Zero action task smoke test; not policy success or physics replay",
    }, indent=2))
    env.close()


if __name__ == "__main__":
    main()
