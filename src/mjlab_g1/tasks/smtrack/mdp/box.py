"""Box entity for the lift task (brief §3).

One free body with one box geom. The size matches `make_box_reference.py`, which writes
the same half extents into the clip as `box_half_extents` (checked by the command).
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg

BOX_HALF_EXTENTS = (0.15, 0.16, 0.18)  # keep in sync with make_box_reference.py
BOX_MASS = 1.5
BOX_FRICTION = 1.0
BOX_BODY_NAME = "box"
BOX_GEOM_NAME = "box_collision"
CARRY_BOX_MASS = 0.5  # SUGAR SMALLBOX_CFG; the lift baseline remains 1.5 kg.
CARRY_BOX_SHAPE = "sugar_small_box_convex_v1"
CARRY_BOX_MESH = Path(__file__).resolve().parents[5] / "assets/sugar_small_box_convex.obj"


def get_box_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  spec.modelname = "smtrack_box"
  body = spec.worldbody.add_body(name=BOX_BODY_NAME)
  body.add_freejoint(name="box_freejoint")
  geom = body.add_geom(
    name=BOX_GEOM_NAME,
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=list(BOX_HALF_EXTENTS),
    mass=BOX_MASS,
  )
  geom.condim = 3
  geom.friction = [BOX_FRICTION, 0.005, 0.0001]
  # Priority 1 makes box contacts use the box's own condim/friction. The G1 hand
  # geoms are condim 1 (frictionless) in mjlab's FULL_COLLISION, and with equal
  # priority MuJoCo takes max(friction) = 1.0, which would cancel the friction DR.
  geom.priority = 1
  geom.rgba = [0.85, 0.55, 0.25, 1.0]
  return spec


def get_box_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(pos=(0.45, 0.0, BOX_HALF_EXTENTS[2])),
    spec_fn=get_box_spec,
  )


def get_carry_box_spec() -> mujoco.MjSpec:
  """A dynamic convex hull of the SUGAR small-box collision mesh, with real contacts."""
  if not CARRY_BOX_MESH.is_file():
    raise FileNotFoundError(f"Missing converted SUGAR collision mesh {CARRY_BOX_MESH}")
  spec = mujoco.MjSpec()
  spec.modelname = "smtrack_sugar_small_box"
  spec.meshdir = str(CARRY_BOX_MESH.parent)
  spec.add_mesh(name="sugar_box_mesh", file=CARRY_BOX_MESH.name)
  body = spec.worldbody.add_body(name=BOX_BODY_NAME)
  body.add_freejoint(name="box_freejoint")
  geom = body.add_geom(
    name=BOX_GEOM_NAME, type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="sugar_box_mesh", mass=CARRY_BOX_MASS,
  )
  geom.condim = 3
  geom.friction = [BOX_FRICTION, 0.005, 0.0001]
  geom.priority = 1
  geom.rgba = [0.85, 0.55, 0.25, 1.0]
  return spec


def get_carry_box_cfg() -> EntityCfg:
  return EntityCfg(
    # The command reset uses the selected clip pose; keep the static default aligned.
    init_state=EntityCfg.InitialStateCfg(pos=(-1.058, -0.276, 0.208)),
    spec_fn=get_carry_box_spec,
  )
