"""Watch a SuperMimic run and physically evaluate each saved checkpoint.

Run this on the original training server, separately from the training process.
Each 64-episode physical rollout uses the same GPU, seeds, and scoring script.
"""

import argparse
import datetime as dt
import fcntl
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, data):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-pid", type=int, required=True)
    parser.add_argument("--eval-gpu", type=int, default=7)
    parser.add_argument("--first", type=int, default=1100)
    parser.add_argument("--last", type=int, default=3000)
    parser.add_argument("--interval", type=int, default=100)
    parser.add_argument("--original-iteration", type=int, default=1000)
    parser.add_argument("--threshold-total", type=int, default=127)
    args = parser.parse_args()
    if not 1 <= args.threshold_total <= 128:
        parser.error("threshold-total must be from 1 through 128")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "monitor_state.json"
    state = {"start_utc": utc_now(), "original_iteration": args.original_iteration,
             "training_pid": args.train_pid, "eval_gpu": args.eval_gpu,
             "threshold": f"{args.threshold_total}/128 on 2027+314159; repeat on same GPU",
             "results": [], "status": "running"}
    atomic_json(state_path, state)
    env = os.environ.copy()
    env.update(CARRY_ENV_SH="/data4/DATA/wlh/offer/ROS2/env.sh",
               CARRY_PYTHON="/data/wlh/miniconda3/envs/wan/bin/python",
               CARRY_TMPDIR="/data4/DATA/wlh/offer/tmp",
               CARRY_XDG_CACHE_HOME="/data4/DATA/wlh/offer/tmp/xdg-cache",
               CARRY_INDUCTOR_CACHE_DIR="/data4/DATA/wlh/offer/tmp/torchinductor")

    def evaluate(iteration, seed, checkpoint, suffix=""):
        prefix = f"iter{iteration}_seed{seed}{suffix}"
        raw = args.repo / "runs" / "evaluations" / f"supermimic_{seed}_n64"
        target = args.output_dir / prefix
        log = args.output_dir / (prefix + ".log")
        command = ["bash", "scripts/eval_checkpoints.sh", "supermimic",
                   str(args.eval_gpu), str(seed), "64", str(checkpoint)]
        started = time.monotonic()
        # The project's evaluator uses fixed output names per seed. Serialize
        # through archival so simultaneous single/4-GPU monitors cannot race.
        lock_path = args.repo / "runs" / "evaluations" / "monitor_eval.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with log.open("w") as output:
                result = subprocess.run(command, cwd=args.repo, env=env,
                                        stdout=output, stderr=subprocess.STDOUT, timeout=1200)
            if result.returncode:
                raise RuntimeError(f"Evaluation {prefix} failed: code={result.returncode}; log={log}")
            shutil.copyfile(raw.with_name(raw.name + ".json"),
                            target.with_name(target.name + "_eval.json"))
            metrics_path = raw.with_name(raw.name + "_metrics.json")
            destination = target.with_name(target.name + "_metrics.json")
            shutil.copyfile(metrics_path, destination)
        payload = json.loads(destination.read_text())
        episodes = payload["per_episode"]
        if len(episodes) != 64 or Path(payload["checkpoint"]).resolve() != checkpoint.resolve():
            raise RuntimeError(f"Wrong episodes/checkpoint in {destination}")
        result = {"seed": seed, "terminal": sum(bool(x["terminal_box_near_goal"]) for x in episodes),
                  "intermittent": sum(bool(x["intermittent_carry"]) for x in episodes),
                  "strict_both_hands": sum(bool(x["strict_success_after_floor_guard"]) for x in episodes),
                  "seconds": round(time.monotonic() - started, 2),
                  "metrics_json": str(destination)}
        print(json.dumps({"iteration": iteration, "tag": suffix, **result}), flush=True)
        return result

    try:
        for iteration in range(args.first, args.last + 1, args.interval):
            ckpt = args.checkpoint_dir / f"model_{iteration}.pt"
            while True:
                if ckpt.is_file() and ckpt.stat().st_size > 200_000_000:
                    size = ckpt.stat().st_size
                    time.sleep(12)
                    if ckpt.stat().st_size == size:
                        break
                if not Path(f"/proc/{args.train_pid}").exists():
                    state["status"] = f"training exited before checkpoint {iteration}"
                    atomic_json(state_path, state)
                    return
                time.sleep(18)
            results = [evaluate(iteration, seed, ckpt) for seed in (2027, 314159)]
            total = sum(r["terminal"] for r in results)
            row = {"iteration": iteration, "new_iterations": iteration - args.original_iteration,
                   "checkpoint": str(ckpt), "utc": utc_now(), "results": results,
                   "terminal_128": total, "intermittent_128": sum(r["intermittent"] for r in results),
                   "strict_128": sum(r["strict_both_hands"] for r in results)}
            state["results"].append(row)
            atomic_json(state_path, state)
            if total >= args.threshold_total:
                confirm = [evaluate(iteration, seed, ckpt, "_confirm") for seed in (2027, 314159)]
                heldout = evaluate(iteration, 161803, ckpt, "_heldout")
                row["confirm"] = confirm
                row["confirm_terminal_128"] = sum(r["terminal"] for r in confirm)
                row["heldout"] = heldout
                atomic_json(state_path, state)
                if row["confirm_terminal_128"] >= args.threshold_total:
                    state["status"] = f"target corroborated at iteration {iteration}"
                    state["stop_utc"] = utc_now()
                    atomic_json(state_path, state)
                    if Path(f"/proc/{args.train_pid}").exists():
                        os.killpg(args.train_pid, signal.SIGINT)
                    return
        state["status"] = f"budget complete without corroborated {args.threshold_total}/128"
        state["stop_utc"] = utc_now()
        atomic_json(state_path, state)
    except Exception as error:
        state["status"] = "monitor error"
        state["error"] = str(error)
        atomic_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
