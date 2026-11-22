"""Convert one SUGAR CarryBox 50 Hz clip to mjlab's G1 tracking reference.

The SUGAR NPZ stores its joints in Isaac Lab's breadth-first articulation
order, NOT Unitree's SDK / mjlab's 29-joint order. Both are the same G1
kinematic tree: matching the fourteen published body references verifies the
reordering before this script writes any output. No motion is stitched.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

SOURCE_JOINTS = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)
TRACKED_SOURCE_BODIES = {
    "pelvis": 0, "left_hip_roll_link": 5, "left_knee_link": 11,
    "left_ankle_roll_link": 21, "right_hip_roll_link": 6,
    "right_knee_link": 12, "right_ankle_roll_link": 22, "torso_link": 10,
    "left_shoulder_roll_link": 19, "left_elbow_link": 25,
    "left_wrist_yaw_link": 31, "right_shoulder_roll_link": 20,
    "right_elbow_link": 26, "right_wrist_yaw_link": 32,
}
BOX_SHAPE = "sugar_small_box_convex_v1"


def phase_frames(labels: np.ndarray) -> tuple[int, int]:
    edges = np.diff(np.pad(labels.astype(np.int32), (1, 1)))
    segments = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    if not segments:
        raise ValueError("No SUGAR object-contact segment")
    start, release = max(segments, key=lambda ab: ab[1] - ab[0])
    if not (5 <= start and start + 40 <= release < len(labels) - 20):
        raise ValueError(f"Incomplete grasp/carry/release phase: {(start, release, len(labels))}")
    return int(start), int(release)


def convert(source: Path, output: Path, g1_xml: Path | None = None) -> dict:
    with np.load(source / "robot_50hz.npz") as file:
        robot = {name: file[name] for name in file.files}
    with (source / "obj_motion_global_50hz.pkl").open("rb") as file:
        obj = pickle.load(file)
    labels = np.load(source / "contact_labels_50hz.npy").astype(bool)
    fps = float(np.asarray(robot["fps"]).item())
    length = len(robot["joint_pos"])
    if fps != 50 or robot["joint_pos"].shape != (length, 29):
        raise ValueError(f"Expected 50 Hz, 29 joints: {source}")
    if len(labels) < length or len(obj["obj_trans"]) < length:
        raise ValueError(f"Object/contacts shorter than robot: {source}")
    labels = labels[:length]
    grasp, release = phase_frames(labels)
    if g1_xml is None:
        import mjlab
        g1_xml = Path(mjlab.__file__).parent / "asset_zoo/robots/unitree_g1/xmls/g1.xml"
    model_path = g1_xml
    model = mujoco.MjModel.from_xml_path(str(model_path))
    mj = mujoco.MjData(model)
    if model.nq != 36 or model.nbody != 31:
        raise ValueError("Installed mjlab G1 asset differs from the verified 29 DoF model")
    target_joints = [model.joint(i).name for i in range(model.njnt)
                     if model.jnt_qposadr[i] >= 7]
    permutation = [SOURCE_JOINTS.index(name) for name in target_joints]
    joints = np.asarray(robot["joint_pos"][:, permutation], dtype=np.float64)
    joint_vel = np.asarray(robot["joint_vel"][:, permutation], dtype=np.float64)
    source_root = robot["body_pos_w"][:, 0]
    source_quat = robot["body_quat_w"][:, 0]
    all_qpos = np.concatenate((source_root, source_quat, joints), axis=1)
    dt = 1.0 / fps
    root_vel = np.empty((length, 6), dtype=np.float64)
    for k in range(length):
        low, high = max(k - 1, 0), min(k + 1, length - 1)
        vel = np.zeros(model.nv, dtype=np.float64)
        mujoco.mj_differentiatePos(model, vel, (high - low) * dt,
                                  all_qpos[low], all_qpos[high])
        root_vel[k] = vel[:6]
    body_pos = np.empty((length, 30, 3), dtype=np.float32)
    body_quat = np.empty((length, 30, 4), dtype=np.float32)
    body_lin_vel = np.empty_like(body_pos)
    body_ang_vel = np.empty_like(body_pos)
    max_source_error = 0.0
    max_error_at = (-1, "")
    fk_errors = []
    velocity = np.empty(6, dtype=np.float64)
    for k in range(length):
        mj.qpos[:] = all_qpos[k]
        mj.qvel[:6] = root_vel[k]
        mj.qvel[6:] = joint_vel[k]
        mujoco.mj_forward(model, mj)
        body_pos[k] = mj.xpos[1:]
        body_quat[k] = mj.xquat[1:]
        for name, source_index in TRACKED_SOURCE_BODIES.items():
            error = np.linalg.norm(mj.xpos[model.body(name).id] -
                                   robot["body_pos_w"][k, source_index])
            fk_errors.append(float(error))
            if error > max_source_error:
                max_source_error = float(error)
                max_error_at = (k, name)
        for body_id in range(1, model.nbody):
            mujoco.mj_objectVelocity(model, mj, mujoco.mjtObj.mjOBJ_BODY,
                                     body_id, velocity, 0)
            body_ang_vel[k, body_id - 1] = velocity[:3]
            body_lin_vel[k, body_id - 1] = velocity[3:]
    p95_source_error = float(np.quantile(fk_errors, 0.95))
    # SUGAR's exported per-body buffers are not exactly synchronized with
    # its joint/root state (an ankle can differ by ~1 cm on a moving frame).
    # Recompute all bodies from qpos, and reject substantially incompatible clips.
    if max_source_error > 0.020 or p95_source_error > 0.006:
        raise ValueError(f"G1 FK max error {max_source_error:.4f} m at {max_error_at}; "
                         f"p95 {p95_source_error:.4f} m; "
                         f"count > 2mm {np.count_nonzero(np.asarray(fk_errors) > 0.002)}")
    rotation = Rotation.from_matrix(np.asarray(obj["obj_rot"][:length]))
    q_xyzw = rotation.as_quat()
    q_wxyz = q_xyzw[:, [3, 0, 1, 2]]
    for k in range(1, length):
        if np.dot(q_wxyz[k - 1], q_wxyz[k]) < 0:
            q_wxyz[k] *= -1
    box_pos = np.asarray(obj["obj_trans"][:length])
    box_vel = np.asarray(obj["obj_lin_vel"][:length])
    box_ang_vel = np.asarray(obj["obj_ang_vel"][:length])
    for name, value in (("box position", box_pos), ("box velocity", box_vel)):
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite {name} in {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, fps=np.array([50], dtype=np.int32),
        joint_pos=joints.astype(np.float32), joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos, body_quat_w=body_quat,
        body_lin_vel_w=body_lin_vel, body_ang_vel_w=body_ang_vel,
        box_pos_w=box_pos.astype(np.float32), box_quat_w=q_wxyz.astype(np.float32),
        box_lin_vel_w=box_vel.astype(np.float32),
        box_ang_vel_w=box_ang_vel.astype(np.float32),
        contact_label=labels,
        grasp_frame=np.array(grasp, dtype=np.int64),
        release_frame=np.array(release, dtype=np.int64),
        target_pos_w=np.median(box_pos[-20:], axis=0).astype(np.float32),
        target_quat_wxyz=q_wxyz[-1].astype(np.float32),
        box_shape=np.array(BOX_SHAPE),
        source_clip=np.array(source.name),
    )
    info = dict(source=source.name, frames=length, grasp_frame=grasp,
                release_frame=release, max_g1_fk_error_mm=round(max_source_error * 1000, 3),
                p95_g1_fk_error_mm=round(p95_source_error * 1000, 3),
                start_box=box_pos[0].round(3).tolist(),
                final_box=box_pos[-1].round(3).tolist())
    print(info, "->", output)
    return info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--g1-xml", type=Path, help="Explicit mjlab v1.4 G1 XML (for offline preprocessing)")
    args = p.parse_args()
    convert(args.source, args.output, args.g1_xml)


if __name__ == "__main__":
    main()
