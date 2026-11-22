"""Summarize measured SuperMimic run costs from archived server logs.

Training GPU memory is whole-device nvidia-smi memory sampled every two seconds.
The recorded GPU-hours include all configured training cards but exclude replay
evaluation on separate cards; wall time includes initialization and checkpoint IO.
"""

import argparse
import datetime as dt
import json
import re
from pathlib import Path


LINE = re.compile(r"\[SuperMimic\] it (\d+) \| env_steps (\d+).*?\| (\d+)s(?:\s|$)")


def analyze(folder):
    metadata_path = next(folder.glob("*_run_metadata.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    slug = metadata["name"]
    devices = metadata.get("gpu_list", [metadata.get("gpu")])
    logfile = folder / (slug + ".log")
    observations = {}
    for line in logfile.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LINE.search(line)
        if match:
            iteration, steps, seconds = map(int, match.groups())
            observations[iteration] = {"iteration": iteration, "env_steps": steps,
                                       "train_loop_elapsed_s": seconds}
    history = sorted(observations.values(), key=lambda x: x["iteration"])
    # Only rank zero logs. With DDP, the physical transition count is the sum
    # across identical-size ranks, not the rank-zero count alone.
    for row in history:
        row["rank0_env_steps"] = row.pop("env_steps")
        row["global_env_steps"] = row["rank0_env_steps"] * len(devices)
    sample_path = folder / (slug + "_gpu_samples.csv")
    peaks = {str(i): 0 for i in devices}
    count = 0
    timestamps = []
    for line in sample_path.read_text(encoding="utf-8").splitlines():
        try:
            timestamp, body = line.split(",", 1)
            cells = [tuple(int(value.strip()) for value in gpu.split(","))
                     for gpu in body.split(";")]
            if any(len(cell) != 4 for cell in cells):
                continue
            timestamps.append(int(timestamp))
            count += 1
            for index, used_mib, _total_mib, _utilization in cells:
                if str(index) in peaks:
                    peaks[str(index)] = max(peaks[str(index)], used_mib)
        except ValueError:
            continue
    if not count:
        raise ValueError(f"No valid GPU samples: {sample_path}")
    start = dt.datetime.fromisoformat((folder / (slug + "_start_utc.txt")).read_text().strip().replace("Z", "+00:00"))
    end_path = folder / (slug + "_end_utc.txt")
    end = dt.datetime.fromisoformat(end_path.read_text().strip().replace("Z", "+00:00")) if end_path.exists() else None
    stop_path = folder / (slug + "_controlled_stop.json")
    controlled_stop = json.loads(stop_path.read_text(encoding="utf-8")) if stop_path.exists() else None
    if controlled_stop:
        if not controlled_stop["reason"].startswith("confirmed target;"):
            raise ValueError(f"Unrecognized controlled stop: {stop_path}")
        # The original shell was signaled before it could write an exit code.
        # This observation is an upper bound on its actual termination time.
        end = dt.datetime.fromisoformat(controlled_stop["exit_confirmed_utc"])
    elapsed = round((end - start).total_seconds() if end else max(timestamps) - start.timestamp(), 1)
    exit_path = folder / (slug + "_exit_code.txt")
    status = json.loads((folder / "monitor_state.json").read_text(encoding="utf-8"))
    if controlled_stop and not status["status"].startswith("target corroborated"):
        raise ValueError("Controlled stop without corroborated target")
    best = max(status.get("results", []), key=lambda x: x["terminal_128"], default=None)
    original_baseline = None
    if folder.name == "single":
        files = [folder / f"iter1000_seed{seed}_gpu3_metrics.json"
                 for seed in (2027, 314159)]
        if all(path.is_file() for path in files):
            episodes = [episode for path in files for episode in
                        json.loads(path.read_text(encoding="utf-8"))["per_episode"]]
            if len(episodes) != 128:
                raise ValueError("Expected 128 original checkpoint evaluation episodes")
            original_baseline = {"iteration": 1000,
                                 "terminal_128": sum(bool(x["terminal_box_near_goal"]) for x in episodes),
                                 "intermittent_128": sum(bool(x["intermittent_carry"]) for x in episodes),
                                 "strict_128": sum(bool(x["strict_success_after_floor_guard"]) for x in episodes),
                                 "eval_gpu": 3}
    speed = None
    if len(history) >= 2:
        last = history[-1]
        earlier = next((x for x in reversed(history[:-1])
                        if last["iteration"] - x["iteration"] >= 100), None)
        if earlier is not None:
            seconds = last["train_loop_elapsed_s"] - earlier["train_loop_elapsed_s"]
            if seconds > 0:
                speed = round((last["global_env_steps"] - earlier["global_env_steps"]) / seconds, 1)
    return {"run_name": slug, "training_gpus": devices,
            "global_num_envs": metadata.get("global_num_envs", metadata.get("num_envs")),
            "num_training_gpus": len(devices), "start_utc": start.isoformat(),
            "end_utc": end.isoformat() if end else None, "wall_elapsed_s": elapsed,
            "gpu_hours_to_date": round(elapsed * len(devices) / 3600, 3),
            "sample_count": count, "gpu_peak_used_mib": peaks,
            "last_log": history[-1] if history else None,
            "steady_real_env_steps_per_s_last_100plus_iters": speed,
            "training_exit_code": int(exit_path.read_text().strip()) if exit_path.exists() else None,
            "training_terminal": exit_path.exists() or controlled_stop is not None,
            "wall_elapsed_is_upper_bound": controlled_stop is not None,
            "controlled_stop": controlled_stop,
            "monitor_status": status.get("status"),
            "original_checkpoint_baseline": original_baseline,
            "best_continuation_checkpoint": {k: best[k] for k in ("iteration", "terminal_128",
                                                                 "intermittent_128", "strict_128")} if best else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = {name: analyze(args.archive / name) for name in ("single", "four")}
    data = {"status": "final" if all(x["training_terminal"] for x in results.values()) else "in_progress",
            "measurement": "Whole-device nvidia-smi memory used, sampled approximately every 2 s",
            "runs": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
