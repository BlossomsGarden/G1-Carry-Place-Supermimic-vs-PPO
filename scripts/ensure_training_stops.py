"""Finish a corroborated run if SIGINT leaves training workers alive.

Run separately on the training host with a unique run name and original PID.
All matched processes must belong to that run; no action occurs for a failed or
unfinished evaluation. The stop marker records an *upper bound* on wall time,
never an invented launcher exit code.
"""

import argparse
import datetime as dt
import json
import os
import signal
import time
from pathlib import Path


def now():
    return dt.datetime.now(dt.timezone.utc)


def is_alive(pid):
    try:
        return "State:\tZ" not in Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return False


def trainer_pids(run):
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            command = (path / "cmdline").read_bytes().replace(b"\0", b" ")
            if b"train_track.py" in command and run.encode() in command:
                yield int(path.name)
        except (OSError, ProcessLookupError):
            continue


def signal_matching_groups(run, launcher_pid, signo):
    groups = set()
    original = Path(f"/proc/{launcher_pid}/cmdline")
    if is_alive(launcher_pid):
        cmdline = original.read_bytes().replace(b"\0", b" ")
        if b"run_carry_experiment.sh" not in cmdline or run.encode() not in cmdline:
            raise RuntimeError("Launcher PID belongs to another run")
        if os.getpgid(launcher_pid) != launcher_pid:
            raise RuntimeError("Launcher process-group ID changed")
        groups.add(launcher_pid)
    for pid in trainer_pids(run):
        try:
            groups.add(os.getpgid(pid))
        except ProcessLookupError:
            pass
    for pgid in groups:
        try:
            os.killpg(pgid, signo)
        except ProcessLookupError:
            pass
    print("sent", signal.Signals(signo).name, "to verified groups", sorted(groups), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--train-pid", required=True, type=int)
    parser.add_argument("--runs-root", required=True, type=Path)
    args = parser.parse_args()
    if not args.run.startswith("sm4g8192_") or args.train_pid <= 1:
        parser.error("Only the configured four-GPU SuperMimic run may be guarded")
    root = args.runs_root
    monitor = root / (args.run + "_evaluations/monitor_state.json")
    exit_code = root / (args.run + "_exit_code.txt")
    marker = root / (args.run + "_controlled_stop.json")
    while True:
        if marker.exists() or exit_code.exists():
            return
        if monitor.exists():
            state = json.loads(monitor.read_text())
            if state["status"].startswith("target corroborated"):
                break
            if state["status"] != "running":
                print("monitor ended without corroborated target:", state["status"], flush=True)
                return
        time.sleep(20)
    stop_requested = dt.datetime.fromisoformat(state["stop_utc"])
    # Allow the monitor's SIGINT to finish cleanly before escalation.
    time.sleep(max(0, 120 - (now() - stop_requested).total_seconds()))
    if exit_code.exists():
        return
    if is_alive(args.train_pid) or list(trainer_pids(args.run)):
        signal_matching_groups(args.run, args.train_pid, signal.SIGTERM)
        term_sent = now().isoformat()
    else:
        term_sent = None
    for _ in range(12):
        if not is_alive(args.train_pid) and not list(trainer_pids(args.run)):
            break
        time.sleep(10)
    else:
        signal_matching_groups(args.run, args.train_pid, signal.SIGKILL)
        time.sleep(10)
    if is_alive(args.train_pid) or list(trainer_pids(args.run)):
        raise RuntimeError("Trainer survived SIGTERM/SIGKILL; no stop marker recorded")
    if exit_code.exists():
        return
    payload = {
        "reason": ("confirmed target; SIGTERM after SIGINT did not exit"
                   if term_sent else "confirmed target; SIGINT stopped launcher without exit status"),
        "train_pid": args.train_pid,
        "last_checkpoint_iteration": state["results"][-1]["iteration"],
        "monitor_requested_sigint_utc": state["stop_utc"],
        "sigterm_sent_utc": term_sent,
        "exit_confirmed_utc": now().isoformat(),
        "exit_confirmed_is_upper_bound": True,
        "launcher_exit_code_unavailable": True,
    }
    pending = marker.with_suffix(".pending")
    pending.write_text(json.dumps(payload, indent=2) + "\n")
    pending.replace(marker)
    print("recorded controlled stop", marker, flush=True)


if __name__ == "__main__":
    main()
