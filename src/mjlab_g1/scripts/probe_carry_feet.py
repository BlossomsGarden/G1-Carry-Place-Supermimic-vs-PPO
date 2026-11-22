"""Measure minimum G1 foot clearance from the MuJoCo ground in a reference clip."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from render_carry_reference import build_model


def run(path: Path, xml: Path, mesh: Path) -> dict:
    with np.load(path) as f:
        z = {k: f[k] for k in f.files}
    model = build_model(xml, mesh, z["target_pos_w"])
    data = mujoco.MjData(model)
    floor = model.geom("floor").id
    feet = [i for i in range(model.ngeom) if model.geom(i).name.startswith(
        ("left_foot", "right_foot")) and "collision" in model.geom(i).name]
    if not feet:
        raise ValueError("No foot collision geoms in G1 model")
    gap = np.empty(len(z["joint_pos"]), dtype=float)
    fromto = np.empty(6)
    for k in range(len(gap)):
        data.qpos[:3] = z["body_pos_w"][k, 0]
        data.qpos[3:7] = z["body_quat_w"][k, 0]
        data.qpos[7:36] = z["joint_pos"][k]
        data.qpos[36:39] = z["box_pos_w"][k]
        data.qpos[39:43] = z["box_quat_w"][k]
        mujoco.mj_forward(model, data)
        gap[k] = min(mujoco.mj_geomDistance(model, data, gid, floor, 10, fromto)
                     for gid in feet)
    gr, rel = int(z["grasp_frame"]), int(z["release_frame"])
    stages = {"approach": gap[:gr], "carry": gap[gr:rel], "release": gap[rel:]}
    return {"foot_geoms": [model.geom(i).name for i in feet],
            "gap_by_stage": {name: {"p05_m": float(np.quantile(a, .05)),
                                    "p50_m": float(np.median(a)),
                                    "p95_m": float(np.quantile(a, .95)),
                                    "contact_fraction": float(np.mean(a <= .005))}
                             for name, a in stages.items()},
            "selected_gap_m": {str(k): round(float(gap[k]), 4)
                               for k in (0, 15, 31, 50, 100, 130, 151, 152, 250, 350, 459, 499)}}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--motion", required=True, type=Path)
    p.add_argument("--g1-xml", required=True, type=Path)
    p.add_argument("--box-mesh", required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(run(args.motion, args.g1_xml, args.box_mesh), indent=2))
