"""Validate per-episode scores before assembling the three training curves.

Archive the remote output folders under docs/metrics/learning_curve/{ppo,single,four}.
This program recounts every 64-episode file rather than trusting the monitors.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def check_pair(folder, iteration, expected, *, baseline=False, suffix=""):
    results = []
    all_episodes = []
    for seed in (2027, 314159):
        stem = f"iter{iteration}_seed{seed}"
        stem += "_gpu3" if baseline else suffix
        path = folder / (stem + "_metrics.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        episodes = payload["per_episode"]
        if (len(episodes) != 64 or payload["seed"] != seed or
                payload["episodes"] != 64 or
                Path(payload["checkpoint"]).name != f"model_{iteration}.pt"):
            raise ValueError(f"Wrong seed or episode count: {path}")
        all_episodes.extend(episodes)
        results.append({"seed": seed,
                        "terminal": sum(bool(x["terminal_box_near_goal"]) for x in episodes),
                        "intermittent": sum(bool(x["intermittent_carry"]) for x in episodes),
                        "strict_both_hands": sum(bool(x["strict_success_after_floor_guard"]) for x in episodes),
                        "source": path.as_posix(),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    total = sum(r["terminal"] for r in results)
    if expected != total:
        raise ValueError(f"Source JSON mismatch at iteration {iteration}: {expected} != {total}")
    stages = ("lifted_15cm", "airborne_transport_50cm", "both_touched_near_lift",
              "target_pose", "released_landed", "settled", "upright", "completed")
    return {"iteration": iteration, "terminal_128": total,
            "intermittent_128": sum(r["intermittent"] for r in results),
            "strict_128": sum(r["strict_both_hands"] for r in results),
            "stage_128": {stage: sum(bool(x[stage]) for x in all_episodes) for stage in stages},
            "median_goal_distance_3d_m": round(statistics.median(
                float(x["target_distance_3d_m"]) for x in all_episodes), 4),
            "seeds": results}


def checked_points(folder, monitor):
    output = []
    for measured in monitor["results"]:
        iteration = measured["iteration"]
        point = check_pair(folder, iteration, measured["terminal_128"])
        if measured.get("confirm_terminal_128") is not None:
            repeat = check_pair(folder, iteration, measured["confirm_terminal_128"],
                                suffix="_confirm")
            point["same_gpu_confirm_128"] = repeat["terminal_128"]
            point["confirmation_sources"] = repeat["seeds"]
            third = folder / f"iter{iteration}_seed161803_heldout_metrics.json"
            payload = json.loads(third.read_text(encoding="utf-8"))
            episode = payload["per_episode"]
            if (len(episode) != 64 or payload["seed"] != 161803 or
                    payload["episodes"] != 64 or
                    Path(payload["checkpoint"]).name != f"model_{iteration}.pt"):
                raise ValueError(f"Third seed, checkpoint or episode count is wrong: {third}")
            third_count = sum(bool(x["terminal_box_near_goal"]) for x in episode)
            if third_count != measured["heldout"]["terminal"]:
                raise ValueError(f"Third seed count mismatch: {third}")
            point["third_seed_64"] = third_count
            point["third_seed_source"] = {"path": third.as_posix(),
                                          "sha256": hashlib.sha256(third.read_bytes()).hexdigest()}
        output.append(point)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True,
                        help="Containing ppo/, single/, four/ subdirectories")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.archive
    ppo = json.loads((base / "ppo/curve.json").read_text(encoding="utf-8"))
    single = json.loads((base / "single/monitor_state.json").read_text(encoding="utf-8"))
    four = json.loads((base / "four/monitor_state.json").read_text(encoding="utf-8"))

    ppo_points = [check_pair(base / "ppo", p["iteration"], p["terminal_128"])
                  for p in ppo["points"]]
    baseline_expected = sum(
        sum(bool(x["terminal_box_near_goal"]) for x in json.loads(
            (base / "single" / f"iter1000_seed{seed}_gpu3_metrics.json").read_text())["per_episode"])
        for seed in (2027, 314159))
    single_points = [check_pair(base / "single", 1000, baseline_expected, baseline=True)]
    single_points += checked_points(base / "single", single)
    four_points = checked_points(base / "four", four)
    if not (ppo_points and single_points):
        raise ValueError("PPO and the single-GPU baseline need observed points")
    if not four_points and four["status"] != "running":
        raise ValueError("A completed four-GPU run cannot have an empty curve")
    data = {"metric": "terminal box center <=0.12 m from target center (3D, last frame)",
            "denominator": 128, "evaluation_seeds": [2027, 314159],
            "episodes_per_seed": 64,
            "warning": "Repeated checkpoint selection uses the same seeds; 128/128 is sample accuracy, not population certainty.",
            "series": [
                {"id": "ppo_4gpu", "label": "PPO · 四卡，8192 环境 · 从零训练",
                 "gpu": "historical four-GPU training; curve evaluated on cuda:2", "points": ppo_points},
                {"id": "supermimic_1gpu", "label": "SuperMimic · 单卡，4096 环境 · 第 1000 轮续训",
                 "gpu": "cuda:6 training; curve evaluated on cuda:3", "points": single_points},
                {"id": "supermimic_4gpu", "label": "SuperMimic · 四卡，8192 环境 · 从零训练",
                 "gpu": "cuda:1,4,5,7 training; curve evaluated on cuda:2", "points": four_points,
                 "status": four["status"]},
            ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("counts:", [(s["id"], [(p["iteration"], p["terminal_128"])
                              for p in s["points"]]) for s in data["series"]])


if __name__ == "__main__":
    main()
