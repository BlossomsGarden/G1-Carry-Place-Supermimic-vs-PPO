"""Inspect per-hand contact and signed distances in a kinematic SUGAR clip."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from render_carry_reference import build_model


def diagnose(motion: Path, g1_xml: Path, box_mesh: Path) -> dict:
    with np.load(motion) as file:
        z = {name: file[name] for name in file.files}
    m = build_model(g1_xml, box_mesh, z["target_pos_w"])
    d = mujoco.MjData(m)
    box = m.geom("box_collision").id
    hands = {side: [gid for gid in range(m.ngeom)
                    if m.body(m.geom_bodyid[gid]).name.startswith(side + "_")
                    and ("hand" in m.body(m.geom_bodyid[gid]).name
                         or "wrist" in m.body(m.geom_bodyid[gid]).name)]
             for side in ("left", "right")}
    cases = [0, 130, 152, 182, 250, 350, 450, 459, len(z["joint_pos"]) - 1]
    frames = {}
    for k in cases:
        d.qpos[:3] = z["body_pos_w"][k, 0]
        d.qpos[3:7] = z["body_quat_w"][k, 0]
        d.qpos[7:36] = z["joint_pos"][k]
        d.qpos[36:39] = z["box_pos_w"][k]
        d.qpos[39:43] = z["box_quat_w"][k]
        mujoco.mj_forward(m, d)
        frame = {"box_xyz": d.qpos[36:39].round(4).tolist(),
                 "wrist_sites": {side: [[m.geom(gid).name,
                                         d.geom_xpos[gid].round(3).tolist(),
                                         round(float(mujoco.mj_geomDistance(m, d, gid, box, 1, np.empty(6))), 4)]
                                        for gid in gids]
                                 for side, gids in hands.items()},
                 "penetration_contacts": [[m.geom(int(c.geom1)).name,
                                           m.geom(int(c.geom2)).name,
                                           round(float(c.dist), 4)]
                                          for c in d.contact[:d.ncon]
                                          if box in (int(c.geom1), int(c.geom2))]}
        frames[str(k)] = frame
    return {"hands": {s: [m.geom(i).name for i in g] for s, g in hands.items()},
            "box_mass_kg": float(m.body_mass[m.body("box").id]), "frames": frames}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--g1-xml", type=Path, required=True)
    p.add_argument("--box-mesh", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(diagnose(args.motion, args.g1_xml, args.box_mesh), indent=2))
