"""Apply the same sparse terminal-box metric to archived physical evaluations.

This does not rerun MuJoCo: each source JSON already records the final box
position error for every independently simulated episode. Source hashes and
per-episode decisions are exported so the result can be audited.

Usage: python scripts/rescore_terminal_placement.py
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

if __package__:
    from .score_carry_object_outcome import TERMINAL_GOAL_RADIUS_M
else:
    from score_carry_object_outcome import TERMINAL_GOAL_RADIUS_M

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "docs" / "metrics"
OUTPUT = METRICS / "terminal_goal_3d_12cm_rescore.json"
SEEDS = (2027, 314159, 161803)
SM_RUNS = (
    ("sm4096_h32_ori04_s0", 1000),
    ("sm4096_h16_ori04_s0", 800),
    ("sm4096_h32_ori12_s0", 1000),
    ("sm4096_h16_ori12_s0", 800),
)
FLAGS = ("terminal_box_near_goal", "intermittent_carry",
         "strict_success_after_floor_guard", "completed")


def rescore_file(filename: str) -> dict:
    path = METRICS / filename
    source_bytes = path.read_bytes()
    source = json.loads(source_bytes)
    if source["episodes"] != 64 or len(source["per_episode"]) != 64:
        raise ValueError(f"Expected exactly 64 physical episodes: {filename}")
    episodes = []
    for index, row in enumerate(source["per_episode"]):
        distance = math.hypot(row["target_xy_error_m"],
                              row["target_height_error_m"])
        if not math.isfinite(distance):
            raise ValueError(f"Non-finite terminal distance: {filename}:{index}")
        episodes.append({
            "index": index,
            "terminal_distance_3d_m": distance,
            "terminal_box_near_goal": distance <= TERMINAL_GOAL_RADIUS_M,
            "intermittent_carry": bool(row["intermittent_carry"]),
            "strict_success_after_floor_guard": bool(row["strict_success_after_floor_guard"]),
            "completed": bool(row["completed"]),
            "target_pose": bool(row["target_pose"]),
            "released_landed": bool(row["released_landed"]),
            "settled": bool(row["settled"]),
        })
    for metric in FLAGS[1:3]:
        observed = sum(e[metric] for e in episodes) / len(episodes)
        if not math.isclose(observed, source["metrics"][metric + "_rate"], abs_tol=1e-12):
            raise ValueError(f"Archived aggregate disagrees: {filename}:{metric}")
    return {
        "source": str(path.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "checkpoint": source["checkpoint"],
        "seed": source["seed"],
        "counts": {metric: sum(e[metric] for e in episodes) for metric in FLAGS},
        "near_without_intermediate_carry": sum(
            e["terminal_box_near_goal"] and not e["intermittent_carry"] for e in episodes),
        "near_but_early_termination": sum(
            e["terminal_box_near_goal"] and not e["completed"] for e in episodes),
        "episodes": episodes,
    }


def build_report() -> dict:
    groups = {}
    for name, iteration in SM_RUNS:
        stem = name.removesuffix("_s0")
        groups[name] = {
            "learner": "supermimic", "iteration": iteration,
            "by_seed": {str(seed): rescore_file(
                f"{stem}_iter{iteration}_seed{seed}_n64_geometry.json") for seed in SEEDS},
        }
    groups["ppo3999"] = {
        "learner": "ppo", "iteration": 3999,
        "by_seed": {str(seed): rescore_file(
            f"ppo3999_seed{seed}_n64_geometry.json") for seed in SEEDS[:2]},
    }
    for group in groups.values():
        rows = list(group["by_seed"].values())
        group["episodes"] = 64 * len(rows)
        group["counts"] = {metric: sum(row["counts"][metric] for row in rows)
                           for metric in FLAGS}
        group["near_without_intermediate_carry"] = sum(
            row["near_without_intermediate_carry"] for row in rows)
        group["near_but_early_termination"] = sum(
            row["near_but_early_termination"] for row in rows)
    return {
        "definition": "Terminal box centre Euclidean 3D distance to the reference target centre <= 0.12 m; binary at final simulated frame only; no orientation, contact, lift, stability, completion or release requirement",
        "radius_m": TERMINAL_GOAL_RADIUS_M,
        "evaluation": "Retrospective re-score of archived MuJoCo physical episodes; no new policy rollout",
        "selection_note": "These four archived SuperMimic checkpoint iterations were originally selected with 16 development episodes using the earlier metric. Ranking them on the held-out endpoint is descriptive, not an unbiased checkpoint-selection protocol.",
        "groups": groups,
    }


def main() -> None:
    report = build_report()
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, group in report["groups"].items():
        counts = group["counts"]
        seeds = "/".join(str(row["counts"]["terminal_box_near_goal"])
                         for row in group["by_seed"].values())
        print(f"{name}: terminal={seeds} ({counts['terminal_box_near_goal']}/{group['episodes']}), "
              f"intermittent={counts['intermittent_carry']}/{group['episodes']}, "
              f"continuous={counts['strict_success_after_floor_guard']}/{group['episodes']}, "
              f"near without intermittent={group['near_without_intermediate_carry']}")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
