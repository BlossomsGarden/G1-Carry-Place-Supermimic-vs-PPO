"""Score terminal goal location and physical handling diagnostics for CarryPlace.

Usage: python scripts/score_carry_object_outcome.py --rollout TRACE.npz
       --eval-json SCORE.json --motion REFERENCE.npz --output SECONDARY.json

The alternate test tolerates intermittent hand contact but still requires an
actually airborne box to travel, arrive, be released onto the floor and settle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mjlab_g1.tasks.smtrack.carryplace_eval import evaluate_carry_place

# The CarryPlace ground is z=0.  The largest vertex radius of the unscaled
# sugar_small_box_convex.obj collision mesh is 0.29545 m.  A box whose centre
# is above 0.33 m cannot touch that plane, regardless of its orientation.
# Warp sometimes reports a stale box/plane contact while the box is elevated.
IMPOSSIBLE_FLOOR_CONTACT_ABOVE_Z_M = 0.33
TERMINAL_GOAL_RADIUS_M = 0.12


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def score_episode(
    state: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    floor: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    complete: bool,
) -> dict:
    """Score measured MuJoCo states/contacts; dimensions [time, 84]."""
    state = np.asarray(state)
    left, right, floor = (np.asarray(a, dtype=bool) for a in (left, right, floor))
    if state.ndim != 2 or state.shape[1] != 84 or len(state) < 2:
        raise ValueError("expected 2+ physical frames with 84 state values")
    if not all(len(a) == len(state) for a in (left, right, floor)):
        raise ValueError("state and contacts must have identical frame counts")
    if not np.isfinite(state).all():
        raise ValueError("physical state contains NaNs/infinities")
    pos, quat = state[:, 71:74], state[:, 74:78]
    impossible_floor = floor & (pos[:, 2] > IMPOSSIBLE_FLOOR_CONTACT_ABOVE_Z_M)
    plausible_floor = floor & ~impossible_floor
    box_v, box_w = state[:, 78:81], state[:, 81:84]
    z0 = pos[0, 2]
    peak_lift = float(pos[:, 2].max() - z0)
    lift_frames = np.flatnonzero(pos[:, 2] >= z0 + 0.15)
    first_lift = int(lift_frames[0]) if len(lift_frames) else len(state)
    # A 5 cm clearance (with no floor collision) gives a contiguous transport
    # interval. The score is NET XY displacement, not path length or final XY.
    airborne = (pos[:, 2] > z0 + 0.05) & ~plausible_floor
    flights = [
        (float(np.linalg.norm(pos[b - 1, :2] - pos[a, :2])), a, b)
        for a, b in _segments(airborne) if b - a >= 10
    ]
    flight_m, start, end = max(flights, default=(0.0, 0, 0))
    any_hand = left | right
    hand_duty = float(any_hand[start:end].mean()) if end > start else 0.0
    both_duty = float((left & right)[start:end].mean()) if end > start else 0.0
    # The interaction check is an additional diagnostic, not a strict grip.
    near_lift = slice(max(0, first_lift - 15), min(len(state), first_lift + 10))
    both_touched_near_lift = bool(
        first_lift < len(state) and left[near_lift].any() and right[near_lift].any()
    )
    target_xy_error = float(np.linalg.norm(pos[-1, :2] - target_pos[:2]))
    target_height_error = float(abs(pos[-1, 2] - target_pos[2]))
    target_distance_3d = float(np.hypot(target_xy_error, target_height_error))
    orientation_error = float(2 * np.arccos(np.clip(abs(np.dot(quat[-1], target_quat)), 0, 1)))
    final_speed = float(np.linalg.norm(box_v[-15:], axis=1).max())
    final_spin = float(np.linalg.norm(box_w[-15:], axis=1).max())
    _, x, y, _ = state[-1, 3:7]
    final_tilt = float(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
    stages = {
        "lifted_15cm": bool(peak_lift >= 0.15),
        "airborne_transport_50cm": bool(flight_m >= 0.5),
        "target_pose": bool(target_xy_error <= 0.12 and target_height_error <= 0.06
                            and orientation_error <= 0.35),
        "released_landed": bool(not any_hand[-15:].any() and plausible_floor[-15:].sum() >= 13),
        "settled": bool(final_speed <= 0.12 and final_spin <= 0.50),
        "upright": bool(state[-1, 2] >= 0.60 and final_tilt <= 0.61),
        "completed": bool(complete),
    }
    outcome = all(stages.values())
    interaction = bool(outcome and both_touched_near_lift and hand_duty >= 0.20)
    # Also repeat the original continuous-grip test with only the physically
    # impossible floor flags removed.  The original raw verdict stays intact.
    corrected_strict = False
    if len(state) >= 50:
        corrected_strict = bool(evaluate_carry_place(
            box_pos=pos, box_quat_wxyz=quat, box_lin_vel=box_v,
            box_ang_vel=box_w, pelvis_pos=state[:, :3],
            pelvis_quat_wxyz=state[:, 3:7],
            both_hands_contact=left & right, floor_contact=plausible_floor,
            target_pos=target_pos, target_quat_wxyz=target_quat,
            episode_complete=complete)["success"])
    return {
        **stages,
        "terminal_box_near_goal": bool(target_distance_3d <= TERMINAL_GOAL_RADIUS_M),
        "object_outcome": bool(outcome),
        "intermittent_carry": interaction,
        "strict_success_after_floor_guard": corrected_strict,
        "impossible_floor_contact_frames": int(impossible_floor.sum()),
        "both_touched_near_lift": both_touched_near_lift,
        "any_hand_airborne_duty": hand_duty,
        "both_hands_airborne_duty": both_duty,
        "airborne_transport_m": float(flight_m),
        "airborne_start_frame": int(start) if end > start else None,
        "airborne_end_frame": int(end - 1) if end > start else None,
        "peak_lift_m": peak_lift,
        "target_xy_error_m": target_xy_error,
        "target_height_error_m": target_height_error,
        "target_distance_3d_m": target_distance_3d,
        "target_orientation_error_rad": orientation_error,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rollout", type=Path, required=True)
    p.add_argument("--motion", type=Path, help="required for --rollout-all-npz exports")
    p.add_argument("--eval-json", type=Path, help="required for --rollout-all-npz exports")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    with np.load(args.rollout) as trace:
        state = trace["state"]
        if state.ndim == 2:
            states = state[:, None, :]
            contacts = {key: trace[key][:, None] for key in
                        ("left_hand_contact", "right_hand_contact", "floor_contact")}
            target_pos = trace["target_pos_w"]
            target_quat = trace["target_quat_wxyz"]
            complete = [bool(trace["episode_complete"])]
            strict = [bool(trace["episode_success"])]
            counts = [len(state) - 1]
        else:
            if args.motion is None or args.eval_json is None:
                p.error("batch rollout needs --motion and --eval-json")
            with np.load(args.motion) as motion:
                target_pos = motion["target_pos_w"]
                target_quat = motion["target_quat_wxyz"]
            evaluation = json.loads(args.eval_json.read_text(encoding="utf-8"))
            counts = trace["episode_steps"].tolist()
            rows = evaluation["per_episode"]
            if len(rows) != state.shape[1] or len(counts) != len(rows):
                raise ValueError("batch rollout does not align with evaluation JSON")
            complete = [r["completed"] for r in rows]
            strict = [r["success"] for r in rows]
            contacts = {key: trace[key] for key in
                        ("left_hand_contact", "right_hand_contact", "floor_contact")}
            states = state
        scores = [score_episode(states[:steps + 1, i],
                                contacts["left_hand_contact"][:steps + 1, i],
                                contacts["right_hand_contact"][:steps + 1, i],
                                contacts["floor_contact"][:steps + 1, i],
                                target_pos, target_quat, complete[i])
                  for i, steps in enumerate(counts)]
        for row, is_strict in zip(scores, strict):
            row["strict_success_existing_evaluator"] = bool(is_strict)
        metrics = {key + "_rate": sum(row[key] for row in scores) / len(scores)
                   for key in ("terminal_box_near_goal", "object_outcome", "intermittent_carry",
                               "strict_success_existing_evaluator",
                               "strict_success_after_floor_guard")}
        report = {"rollout": str(args.rollout), "checkpoint": str(trace["checkpoint"].item()),
                  "seed": int(trace["seed"].item()), "episodes": len(scores),
                  "thresholds": {"terminal_goal_3d_m": TERMINAL_GOAL_RADIUS_M,
                                 "lift_m": 0.15, "airborne_clearance_m": 0.05,
                                 "airborne_net_xy_m": 0.5, "minimum_flight_frames": 10,
                                 "contact_duty_for_interaction": 0.20,
                                 "target_xy_m": 0.12, "target_height_m": 0.06,
                                 "target_orientation_rad": 0.35, "settle_frames": 15,
                                 "floor_contact_ignored_above_box_center_z_m":
                                 IMPOSSIBLE_FLOOR_CONTACT_ABOVE_Z_M},
                  "metrics": metrics, "per_episode": scores}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": report["checkpoint"], "seed": report["seed"],
                      "episodes": len(scores), **metrics}))


if __name__ == "__main__":
    main()
