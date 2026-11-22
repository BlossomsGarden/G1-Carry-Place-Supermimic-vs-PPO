"""Repeat an identical physical evaluation and locate the first divergence.

Run on the original Linux server while the training process continues. The
evaluation lock prevents an active checkpoint monitor from overwriting its
shared output names before they are copied into the probe directory.
"""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--episodes", type=int, default=64)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    shared = args.repo / "runs/evaluations" / f"supermimic_{args.seed}_n{args.episodes}"
    env = os.environ.copy()
    env.update(CARRY_ENV_SH="/data4/DATA/wlh/offer/ROS2/env.sh",
               CARRY_PYTHON="/data/wlh/miniconda3/envs/wan/bin/python",
               CARRY_TMPDIR="/data4/DATA/wlh/offer/tmp",
               CARRY_XDG_CACHE_HOME="/data4/DATA/wlh/offer/tmp/xdg-cache",
               CARRY_INDUCTOR_CACHE_DIR="/data4/DATA/wlh/offer/tmp/torchinductor")
    copied = []
    with (args.repo / "runs/evaluations/monitor_eval.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for run in ("a", "b"):
            log = args.output / f"repeat_{run}.log"
            with log.open("w") as stream:
                subprocess.run(["bash", "scripts/eval_checkpoints.sh", "supermimic",
                                str(args.gpu), str(args.seed), str(args.episodes),
                                str(args.checkpoint)], cwd=args.repo, env=env,
                               stdout=stream, stderr=subprocess.STDOUT, timeout=1200, check=True)
            saved = args.output / f"repeat_{run}"
            for ext in (".npz", "_metrics.json", ".json"):
                shutil.copy2(shared.with_name(shared.name + ext),
                             saved.with_name(saved.name + ext))
            copied.append(saved)

    traces = []
    for path in copied:
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as file:
            traces.append({name: file[name] for name in ("xi", "state", "action", "episode_steps")})
    a, b = traces
    dist = abs(a["state"] - b["state"])
    diverged = (dist > 1e-6).any(axis=-1)
    first = [int(np.flatnonzero(diverged[:, i])[0]) if diverged[:, i].any() else None
             for i in range(args.episodes)]
    scores = [json.loads(path.with_name(path.name + "_metrics.json").read_text())
              for path in copied]
    passed = [[bool(x["terminal_box_near_goal"]) for x in s["per_episode"]] for s in scores]
    result = {"checkpoint": str(args.checkpoint), "gpu": args.gpu, "seed": args.seed,
              "episodes": args.episodes,
              "initial_dr_identical": bool(np.array_equal(a["xi"], b["xi"])),
              "initial_state_max_abs": float(dist[0].max()),
              "first_action_max_abs": float(abs(a["action"][0] - b["action"][0]).max()),
              "first_divergent_frame_per_episode": first,
              "total_distinct_final_success_labels": sum(x != y for x, y in zip(*passed)),
              "terminal_counts": [sum(x) for x in passed],
              "max_state_difference": float(dist.max())}
    path = args.output / "repeatability.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k != "first_divergent_frame_per_episode"}, indent=2))


if __name__ == "__main__":
    main()
