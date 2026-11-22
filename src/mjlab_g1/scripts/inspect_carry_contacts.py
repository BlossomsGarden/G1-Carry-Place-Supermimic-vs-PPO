"""Inspect available MuJoCo Warp contact fields for CarryPlace evaluation."""
import json

import torch
import warp as wp

from mjlab_g1.envs.smtrack_env import SMTrackEnv
from mjlab_g1.tasks.smtrack import make_env_cfg


def info(value):
    result = {"type": str(type(value)), "shape": str(getattr(value, "shape", None))}
    try:
        array = wp.to_torch(value)
        result["torch_shape"] = list(array.shape)
        result["first"] = array.reshape(-1)[:8].detach().cpu().tolist()
    except Exception as err:
        result["error"] = str(err)[:180]
    return result


cfg = make_env_cfg("G1-SMTrack-CarryPlace", play=True)
cfg.scene.num_envs = 2
cfg.auto_reset = False
env = SMTrackEnv(cfg, device="cuda:0")
env.reset()
env.step(torch.zeros_like(env.action_manager.action))
sim = env.sim
data = sim.wp_data
fields = {}
for name in dir(data):
    if name in ("ncon", "nworld"):
        value = getattr(data, name)
        fields[name] = info(value) if hasattr(value, "shape") else str(type(value))
if hasattr(data, "contact"):
    fields["contact_members"] = {name: info(getattr(data.contact, name))
                                  for name in dir(data.contact)
                                  if name in ("geom", "dist", "worldid", "dim", "type")}
print(json.dumps({"fields": fields, "box_gid": env._box_gid,
                  "floor_gids": [[i, sim.mj_model.geom(i).name]
                                 for i in range(sim.mj_model.ngeom)
                                 if sim.mj_model.geom_bodyid[i] == 0],
                  "hand_gids": [[i, sim.mj_model.geom(i).name]
                                 for i in range(sim.mj_model.ngeom)
                                 if "hand_collision" in (sim.mj_model.geom(i).name or "")],
                  "nconmax": sim.mj_model.nconmax, "num_envs": env.num_envs}, indent=2))
env.close()
