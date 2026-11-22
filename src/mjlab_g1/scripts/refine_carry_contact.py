"""Retarget SUGAR's arm joints around its box; retain the original raw clip.

Kinematic signed-distance IK is a necessary geometry check, not a guarantee
that the trajectory can be executed by the robot under physical simulation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from render_carry_reference import build_model


ARM_JOINTS = tuple(f"{side}_{name}_joint" for side in ("left", "right") for name in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
    "wrist_roll", "wrist_pitch", "wrist_yaw"))


def refine(source: Path, output: Path, g1_xml: Path, box_mesh: Path,
           stride: int = 1) -> dict:
    with np.load(source) as file:
        z = {key: file[key] for key in file.files}
    m = build_model(g1_xml, box_mesh, z["target_pos_w"])
    d = mujoco.MjData(m)
    arms = []
    for side in ("left", "right"):
        ids = np.array([int(m.joint(f"{side}_{name}_joint").qposadr[0]) for name in
                        ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                         "wrist_roll", "wrist_pitch", "wrist_yaw")], dtype=int)
        arm_gids = (m.geom(f"{side}_hand_collision").id,
                    m.geom(f"{side}_wrist_collision").id)
        lower = m.jnt_range[[m.joint(f"{side}_{name}_joint").id for name in
                             ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                              "wrist_roll", "wrist_pitch", "wrist_yaw")], 0]
        upper = m.jnt_range[[m.joint(f"{side}_{name}_joint").id for name in
                             ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                              "wrist_roll", "wrist_pitch", "wrist_yaw")], 1]
        arms.append((ids, arm_gids, lower, upper))
    box_gid = m.geom("box_collision").id
    floor_gid = m.geom("floor").id
    foot_gids = [i for i in range(m.ngeom)
                 if m.geom(i).name.startswith(("left_foot", "right_foot"))
                 and "collision" in m.geom(i).name]
    if not foot_gids:
        raise ValueError("Missing foot collisions in G1 XML")
    root = np.asarray(z["body_pos_w"][:, 0], dtype=np.float64).copy()
    gaps = np.empty(len(root), dtype=np.float64)
    foot_fromto = np.empty(6, dtype=np.float64)
    for k in range(len(root)):
        d.qpos[:3] = root[k]
        d.qpos[3:7] = z["body_quat_w"][k, 0]
        d.qpos[7:36] = z["joint_pos"][k]
        d.qpos[36:39] = z["box_pos_w"][k]
        d.qpos[39:43] = z["box_quat_w"][k]
        mujoco.mj_forward(m, d)
        gaps[k] = min(mujoco.mj_geomDistance(m, d, gid, floor_gid, 10, foot_fromto)
                      for gid in foot_gids)
    # The source has a 6.5 mm median approach foot clearance. Reposition the
    # pelvis vertically by at most 2 cm, preserving its XY walk trajectory.
    # A smooth correction avoids frame-to-frame impulses from switching feet.
    foot_shift = np.clip(savgol_filter(gaps + .001, 11, 2), -.02, .02)
    root[:, 2] -= foot_shift
    out = np.asarray(z["joint_pos"], dtype=np.float64).copy()
    gr, rel = int(z["grasp_frame"]), int(z["release_frame"])
    # Only move arms during the labelled contact interval. Fade into/out of the
    # grasp to avoid hard joint discontinuities at the contact phase boundary.
    anchor = max(gr - 12, 0)
    end = min(rel + 12, len(out) - 1)
    selected = list(range(gr, rel, stride))
    if selected[-1] != rel - 1:
        selected.append(rel - 1)
    corrections = np.zeros((len(selected), 14), dtype=np.float64)
    last = [None, None]
    samples = {}
    fromto = np.empty(6, dtype=np.float64)
    for index, k in enumerate(selected):
        d.qpos[:3] = root[k]
        d.qpos[3:7] = z["body_quat_w"][k, 0]
        d.qpos[7:36] = z["joint_pos"][k]
        d.qpos[36:39] = z["box_pos_w"][k]
        d.qpos[39:43] = z["box_quat_w"][k]
        for arm_ix, (ids, (hand, wrist), low, high) in enumerate(arms):
            original = d.qpos[ids].copy()
            # The wrist collision must not enter the box; hand contact should
            # have millimetric overlap so MuJoCo reports a genuine contact.
            def residual(q: np.ndarray) -> np.ndarray:
                d.qpos[ids] = q
                mujoco.mj_forward(m, d)
                h = float(mujoco.mj_geomDistance(m, d, hand, box_gid, 10, fromto))
                w = float(mujoco.mj_geomDistance(m, d, wrist, box_gid, 10, fromto))
                previous = last[arm_ix] if last[arm_ix] is not None else original
                return np.r_[80 * (h + 0.001), 60 * min(w + 0.002, 0.0),
                             0.24 * (q - original), 0.04 * (q - previous)]

            start = np.clip(original if last[arm_ix] is None else
                            0.6 * original + 0.4 * last[arm_ix], low + 1e-5, high - 1e-5)
            fit = least_squares(residual, start, bounds=(low + 1e-5, high - 1e-5),
                                max_nfev=35, ftol=5e-4, xtol=5e-4, gtol=5e-4)
            d.qpos[ids] = fit.x
            corrections[index, arm_ix * 7:(arm_ix + 1) * 7] = fit.x - original
            last[arm_ix] = fit.x
            if k in (gr, gr + 30, (gr + rel) // 2, rel - 1):
                samples[f"{k}_{arm_ix}"] = {"hand_distance_m": round(float(
                    mujoco.mj_geomDistance(m, d, hand, box_gid, 10, fromto)), 4),
                    "wrist_distance_m": round(float(
                        mujoco.mj_geomDistance(m, d, wrist, box_gid, 10, fromto)), 4),
                    "delta_joint_rad": np.round(fit.x - original, 3).tolist()}
    # Interpolate sparse IK corrections, then smoothly ramp at grasp and release.
    all_frames = np.arange(anchor, end + 1)
    for arm_ix, (ids, _, _, _) in enumerate(arms):
        for j, qid in enumerate(ids):
            delta = np.interp(all_frames, selected, corrections[:, arm_ix * 7 + j])
            fade = np.minimum(np.clip((all_frames - anchor) / max(1, gr - anchor), 0, 1),
                              np.clip((end - all_frames) / max(1, end - rel + 1), 0, 1))
            out[anchor:end + 1, qid - 7] += delta * fade
    dt = 1 / float(np.asarray(z["fps"]).ravel()[0])
    qpos = np.concatenate([root, z["body_quat_w"][:, 0], out], axis=1)
    robot_model = mujoco.MjModel.from_xml_path(str(g1_xml))
    qvel = np.zeros((len(out), robot_model.nv), dtype=np.float64)
    for k in range(len(out)):
        lo, hi = max(0, k - 1), min(len(out) - 1, k + 1)
        mujoco.mj_differentiatePos(robot_model, qvel[k], (hi - lo) * dt, qpos[lo], qpos[hi])
    body_p = np.empty_like(z["body_pos_w"])
    body_q = np.empty_like(z["body_quat_w"])
    body_v = np.empty_like(z["body_lin_vel_w"])
    body_w = np.empty_like(z["body_ang_vel_w"])
    v = np.empty(6)
    for k in range(len(out)):
        d.qpos[:36] = qpos[k]
        d.qpos[36:39] = z["box_pos_w"][k]
        d.qpos[39:43] = z["box_quat_w"][k]
        d.qvel[:robot_model.nv] = qvel[k]
        mujoco.mj_forward(m, d)
        body_p[k] = d.xpos[1:31]
        body_q[k] = d.xquat[1:31]
        for bid in range(1, 31):
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, v, 0)
            body_w[k, bid - 1] = v[:3]
            body_v[k, bid - 1] = v[3:]
    joint_change = float(np.abs(out - z["joint_pos"]).max())
    z.update(joint_pos=out.astype(np.float32), joint_vel=qvel[:, 6:35].astype(np.float32),
             body_pos_w=body_p, body_quat_w=body_q,
             body_lin_vel_w=body_v, body_ang_vel_w=body_w,
             contact_refiner=np.asarray("signed_distance_arm_ik_v1"),
             source_sha256=np.asarray(hashlib.sha256(source.read_bytes()).hexdigest()))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **z)
    result = {"source": str(source), "output": str(output), "frames": len(out),
              "optimized_frames": len(selected), "stride": stride,
              "source_foot_gap_approach_p50_m": round(float(np.median(gaps[:gr])), 4),
              "pelvis_z_correction_abs_p95_m": round(float(np.quantile(np.abs(foot_shift), .95)), 4),
              "max_joint_change_rad": round(joint_change, 5),
              "peak_joint_velocity_radps": round(float(np.abs(qvel[:, 6:35]).max()), 3),
              "samples": samples, "scope": "kinematic IK only; no physical feasibility claim"}
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--g1-xml", type=Path, required=True)
    p.add_argument("--box-mesh", type=Path, required=True)
    p.add_argument("--stride", type=int, default=1)
    args = p.parse_args()
    print(json.dumps(refine(args.source, args.output, args.g1_xml, args.box_mesh,
                            args.stride), indent=2))
