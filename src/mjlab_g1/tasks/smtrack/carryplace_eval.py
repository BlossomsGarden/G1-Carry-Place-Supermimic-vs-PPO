"""Episode-level CarryPlace success checks, separate from tracking rewards.

The final verdict requires real simulator contacts and velocities. Kinematic
reference playback must never be reported as a successful trained policy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CarryPlaceThresholds:
    min_transport_m: float = 0.5
    min_lift_m: float = 0.15
    target_xy_tolerance_m: float = 0.12
    final_height_tolerance_m: float = 0.06
    final_orientation_tolerance_rad: float = 0.35
    max_final_box_speed_mps: float = 0.12
    max_final_box_spin_radps: float = 0.50
    min_final_pelvis_height_m: float = 0.60
    max_final_pelvis_tilt_rad: float = 0.61
    min_grasp_frames: int = 5  # 0.10 s of uninterrupted two-hand contact
    min_carry_frames: int = 10  # 0.20 s lifted with uninterrupted contact
    settle_frames: int = 15  # 0.30 s at 50 Hz


def stopped_in_fallen_pose(terminated: bool, pelvis_height_m: float,
                           pelvis_tilt_rad: float) -> bool:
    """Operational fall proxy at episode end, distinct from box tracking failure.

    The demonstration leans ~45 degrees to reach the box; count a severe tilt
    only above 60 degrees, or a pelvis below 0.55 m. This is a terminal-pose
    classification, not an explanation of which event caused the failure.
    """
    return bool(terminated and (pelvis_height_m < 0.55 or pelvis_tilt_rad > 1.05))


def evaluate_carry_place(
    box_pos: np.ndarray,
    box_quat_wxyz: np.ndarray,
    box_lin_vel: np.ndarray,
    box_ang_vel: np.ndarray,
    pelvis_pos: np.ndarray,
    pelvis_quat_wxyz: np.ndarray,
    both_hands_contact: np.ndarray,
    floor_contact: np.ndarray,
    target_pos: np.ndarray,
    target_quat_wxyz: np.ndarray,
    thresholds: CarryPlaceThresholds = CarryPlaceThresholds(),
    episode_complete: bool = True,
) -> dict[str, float | bool]:
    """Check one 50 Hz physical rollout; contacts must come from MuJoCo ncon.

    A phase cannot pass just because a reference pose happens to put the box at
    the goal: lift and transport must actually occur before release and settle.
    """
    pos = np.asarray(box_pos)
    quat = np.asarray(box_quat_wxyz)
    vel = np.asarray(box_lin_vel)
    spin = np.asarray(box_ang_vel)
    pelvis = np.asarray(pelvis_pos)
    pq = np.asarray(pelvis_quat_wxyz)
    hands = np.asarray(both_hands_contact, dtype=bool)
    floor = np.asarray(floor_contact, dtype=bool)
    length = len(pos)
    n = thresholds.settle_frames
    if length < max(50, n * 3) or any(len(v) != length for v in
                                      (quat, vel, spin, pelvis, pq, hands, floor)):
        raise ValueError("Need matching 50 Hz arrays for a full CarryPlace episode")
    if not all(np.isfinite(a).all() for a in (pos, quat, vel, spin, pelvis, pq)):
        raise ValueError("Non-finite physical rollout")
    target = np.asarray(target_pos)
    target_q = np.asarray(target_quat_wxyz)
    displacement = float(np.linalg.norm(pos[-1, :2] - pos[0, :2]))
    max_lift = float(np.max(pos[:, 2]) - pos[0, 2])
    target_error = float(np.linalg.norm(pos[-1, :2] - target[:2]))
    height_error = float(abs(pos[-1, 2] - target[2]))
    angle_error = float(2 * np.arccos(np.clip(abs(np.dot(quat[-1], target_q)), 0, 1)))
    last_speed = float(np.max(np.linalg.norm(vel[-n:], axis=1)))
    last_spin = float(np.max(np.linalg.norm(spin[-n:], axis=1)))
    w, x, y, z = pq[-1]
    gravity_tilt = float(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
    # Transport must happen *while* both hands hold the elevated box. Otherwise
    # sliding it across the floor and briefly lifting it would be a false positive.
    # All contacts come from the simulated rollout, never the SUGAR annotations.
    airborne = pos[:, 2] > pos[0, 2] + thresholds.min_lift_m
    first_lift = int(np.flatnonzero(airborne)[0]) if np.any(airborne) else length
    def segments(mask: np.ndarray) -> list[tuple[int, int]]:
        edges = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
        return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))

    longest_prelift_grip = int(max((end - start for start, end in
                                   segments(hands[:first_lift])), default=0))
    grip_before_lift = longest_prelift_grip >= thresholds.min_grasp_frames
    first_both = np.flatnonzero(hands)
    carried_m = max((float(np.linalg.norm(pos[end - 1, :2] - pos[start, :2]))
                     for start, end in segments(airborne & hands & ~floor)
                     if end - start >= thresholds.min_carry_frames), default=0.0)
    released = bool(not np.any(hands[-n:]))
    landed = bool(np.count_nonzero(floor[-n:]) >= n - 2)
    # Sequential stage gates: a box stationary on the original floor must not
    # receive credit for "settled" when the robot never grasped it.
    grasp_ok = bool(grip_before_lift)
    carry_ok = bool(grasp_ok and carried_m >= thresholds.min_transport_m
                    and displacement >= thresholds.min_transport_m)
    place_ok = bool(carry_ok and target_error <= thresholds.target_xy_tolerance_m
                    and height_error <= thresholds.final_height_tolerance_m
                    and angle_error <= thresholds.final_orientation_tolerance_rad)
    released_ok = bool(place_ok and released and landed)
    passed = {
        "grasp": grasp_ok,
        "carry": carry_ok,
        "place": place_ok,
        "released": released_ok,
        "settled": released_ok and last_speed <= thresholds.max_final_box_speed_mps
                   and last_spin <= thresholds.max_final_box_spin_radps,
        "upright": pelvis[-1, 2] >= thresholds.min_final_pelvis_height_m
                   and gravity_tilt <= thresholds.max_final_pelvis_tilt_rad,
        "completed": bool(episode_complete),
    }
    return {**{key: bool(value) for key, value in passed.items()},
            "success": bool(all(passed.values())),
            "transport_m": displacement, "carried_m": carried_m,
            "max_lift_m": max_lift,
            "first_lift_frame": first_lift if first_lift < length else None,
            "first_bilateral_contact_frame": int(first_both[0]) if len(first_both) else None,
            "longest_prelift_bilateral_contact_frames": longest_prelift_grip,
            "target_error_m": target_error, "height_error_m": height_error,
            "orientation_error_rad": angle_error, "final_speed_mps": last_speed,
            "final_spin_radps": last_spin, "final_pelvis_tilt_rad": gravity_tilt}
