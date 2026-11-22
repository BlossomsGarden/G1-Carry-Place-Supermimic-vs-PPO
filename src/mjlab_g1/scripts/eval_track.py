"""Evaluate a PPO or SuperMimic checkpoint on an SMTrack task (brief §5) -> JSON.

  PYTHONPATH=src python src/mjlab_g1/scripts/eval_track.py --task G1-SMTrack-Walk \
    --learner ppo --checkpoint logs/smtrack/.../model_29999.pt [--episodes 64] [--output m.json]

Nominal condition (protocol §2.2): play cfg = every episode from frame 0, no DR (foot
friction 1.0, xi nominal), no observation noise, no reset noise, deterministic actions.
Episodes run in parallel, one per env, and end at the first termination or the clip end.
Nominal episodes are identical, so their success is one rollout, not a rate. --randomize keeps
the frame-0 start and deterministic actions but restores the training cfg's DR, observation
noise and reset noise, so each episode differs (robustness condition).

Metrics (per episode, then mean over episodes):
- success: reached the clip end without a termination; fall_rate: terminated.
- norm_return: undiscounted reward sum / ((F-1) x max per-step reward); steps after an
  episode ends add 0. norm_return_tracking uses only the 6 tracking kernels.
- body_pos_err_{mean,max}_m (global, 14 tracked bodies) and anchor_rel_body_pos_err_*
  (R_anchor^T (p_b - p_anchor) vs the same for the reference), joint_pos_err_rad (mean |dq|),
  peak_joint_speed (max |qd|, with the reference's peak for scale).
- lift: box_success (no termination and box z >= ref z - 0.1 at the clip end),
  box_pos_err_m, peak_box_vz, box_tilt_end_deg, box_rise_5_to_55cm_s.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
import warp as wp

from mjlab.utils.lab_api.math import quat_apply_inverse

TRACKING_TERMS = (
  "motion_global_root_pos",
  "motion_global_root_ori",
  "motion_body_pos",
  "motion_body_ori",
  "motion_body_lin_vel",
  "motion_body_ang_vel",
)
BOX_SUCCESS_TOL = 0.1
RISE_LOW, RISE_HIGH = 0.05, 0.55
# The CarryPlace collision mesh has maximum vertex radius 0.29545 m and the
# plane is at z=0; a box centre over 0.33 m cannot touch the plane.  Guard the
# Warp contact buffer against stale box/plane pairs during high lifts.
CARRY_BOX_FLOOR_PLAUSIBLE_Z_M = 0.33


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  from mjlab_g1.tasks.smtrack import list_tasks

  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("--task", choices=list_tasks(), required=True)
  p.add_argument("--learner", choices=("ppo", "supermimic"), required=True)
  p.add_argument("--checkpoint", required=True, type=Path)
  p.add_argument("--episodes", type=int, default=64)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--motion-file", type=Path, default=None,
                 help="reference override; default is the contact-refined CarryPlace clip")
  p.add_argument("--rollout-npz", type=Path, default=None,
                 help="save one selected episode's real simulated states and contacts")
  p.add_argument("--rollout-all-npz", type=Path, default=None,
                 help="save all physical transitions, executed actions and DR for model diagnostics")
  p.add_argument("--rollout-index", type=int, default=0,
                 help="zero-based episode index in the parallel evaluation (default: 0)")
  p.add_argument("--device", default=None)
  p.add_argument("--output", type=Path, default=None, help="JSON path (default: stdout only)")
  p.add_argument("--randomize", action="store_true",
                 help="training DR + obs noise + reset noise (still frame 0, deterministic actions)")
  p.add_argument("--box-mass-kg", type=float, default=None,
                 help="CarryPlace-only stress test: fix actual box mass; default uses task DR")
  p.add_argument("--agent-cfg", type=Path, default=None,
                 help="agent.json (default: <checkpoint dir>/params/agent.json)")
  p.add_argument("--sm-clean-obs", action="store_true",
                 help="with --randomize, give SuperMimic clean observations (the pre-2026-10-23 "
                      "behaviour; PPO always gets the env's obs corruption)")
  return p.parse_args(argv)


def obs_noise(args: argparse.Namespace) -> bool:
  """Whether the actor sees sensor noise. PPO reads the env's corrupted actor_hist (on under
  --randomize); SuperMimic's inference policy builds its own frames from the state, so it needs
  the same noise switched on explicitly (identical half-widths, features.OBS_NOISE)."""
  if args.learner == "supermimic" and args.sm_clean_obs:
    return False
  return bool(args.randomize)


def load_agent_cfg(args: argparse.Namespace) -> dict:
  path = args.agent_cfg or args.checkpoint.parent / "params" / "agent.json"
  if path.exists():
    with open(path) as f:
      return json.load(f)
  print(f"[eval_track] {path} not found; using the default {args.learner} cfg")
  if args.learner == "ppo":
    from mjlab_g1.tasks.smtrack import make_ppo_cfg

    return dataclasses.asdict(make_ppo_cfg(args.task))
  from mjlab_g1.rl.supermimic.config import SuperMimicRunnerCfg

  return dataclasses.asdict(SuperMimicRunnerCfg())


def make_policy(args: argparse.Namespace, vec_env: Any, device: str) -> Any:
  from mjlab_g1.scripts.train_track import make_runner

  agent = load_agent_cfg(args)
  if args.learner == "supermimic":
    # A one-clip deterministic rollout gains nothing from torch.compile, and compiling on
    # the CPU (inductor + C++ toolchain) would dominate the eval time.
    agent.setdefault("memory", {})["compile"] = False
  runner = make_runner(args.learner, vec_env, agent, None, device)
  runner.load(str(args.checkpoint), map_location=device)
  policy = runner.get_inference_policy(device)
  if args.learner == "supermimic":
    policy.obs_noise = obs_noise(args)
  return policy


def max_step_reward(env: Any, terms: tuple[str, ...] | None = None) -> float:
  """Sum of positive weights x dt: every exp kernel at 1, penalties at 0."""
  rm = env.reward_manager
  total = sum(
    rm.get_term_cfg(n).weight
    for n in rm.active_terms
    if rm.get_term_cfg(n).weight > 0 and (terms is None or n in terms)
  )
  return total * env.step_dt


class EpisodeStats:
  """Per-env accumulators over the valid (pre-done) steps of the single episode."""

  def __init__(self, n: int, device: str):
    z = lambda: torch.zeros(n, device=device)  # noqa: E731
    self.ret, self.ret_track, self.steps = z(), z(), z()
    self.body_err_sum, self.anchor_err_sum, self.joint_err_sum = z(), z(), z()
    self.body_err_max, self.anchor_err_max, self.qd_max = z(), z(), z()
    self.box_err_sum, self.box_vz_max = z(), z()
    self.t_low = torch.full((n,), math.nan, device=device)
    self.t_high = torch.full((n,), math.nan, device=device)

  @staticmethod
  def add(acc: torch.Tensor, x: torch.Tensor, m: torch.Tensor) -> None:
    acc += torch.where(m, x, torch.zeros_like(x))

  @staticmethod
  def peak(acc: torch.Tensor, x: torch.Tensor, m: torch.Tensor) -> None:
    acc.copy_(torch.where(m, torch.maximum(acc, x), acc))


def tracking_errors(env: Any, k: torch.Tensor) -> dict[str, torch.Tensor]:
  """Errors of the post-step robot vs reference frame k (the frame this step was scored on)."""
  cmd = env.motion_command
  robot = env.scene["robot"]
  m = cmd.motion
  ref_pos = m.body_pos_w[k] + env.scene.env_origins[:, None]
  ref_quat_a = m.body_quat_w[k, cmd.motion_anchor_body_index]
  pos = cmd.robot_body_pos_w
  a = cmd.motion_anchor_body_index
  glob = torch.norm(pos - ref_pos, dim=-1)  # [N,14]
  rel = quat_apply_inverse(cmd.robot_anchor_quat_w[:, None].expand_as(cmd.robot_body_quat_w),
                           pos - pos[:, a : a + 1])
  rel_ref = quat_apply_inverse(ref_quat_a[:, None].expand_as(cmd.robot_body_quat_w),
                               ref_pos - ref_pos[:, a : a + 1])
  anchor = torch.norm(rel - rel_ref, dim=-1)
  joint = (robot.data.joint_pos - m.joint_pos[k]).abs().mean(-1)
  return {
    "body_mean": glob.mean(-1), "body_max": glob.max(-1).values,
    "anchor_mean": anchor.mean(-1), "anchor_max": anchor.max(-1).values,
    "joint": joint, "qd_max": robot.data.joint_vel.abs().max(-1).values,
  }


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
  if args.task == "G1-SMTrack-CarryPlace":
    return evaluate_carryplace(args)
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab_g1.envs.smtrack_env import SMTrackEnv
  from mjlab_g1.tasks.smtrack import make_env_cfg

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  cfg = make_env_cfg(args.task, play=not args.randomize, motion_file=args.motion_file)
  if args.randomize:
    cfg.commands["motion"].sampling_mode = "start"
  cfg.scene.num_envs = args.episodes
  cfg.seed = args.seed
  # Finished envs are frozen (stepped, ignored), never reset: a partial reset() would
  # advance every other env's reference frame (see smtrack_env.py).
  cfg.auto_reset = False
  env = SMTrackEnv(cfg, device=device)
  vec_env = RslRlVecEnvWrapper(env)
  policy = make_policy(args, vec_env, device)

  n, cmd = env.num_envs, env.motion_command
  num_steps = cmd.num_frames - 1  # rewarded frames 1..F-1 from a frame-0 start
  r_max, r_max_track = max_step_reward(env), max_step_reward(env, TRACKING_TERMS)
  track_ids = [env.reward_term_names.index(t) for t in TRACKING_TERMS]
  st = EpisodeStats(n, device)
  active = torch.ones(n, dtype=torch.bool, device=device)
  success = torch.zeros(n, dtype=torch.bool, device=device)
  fell = torch.zeros(n, dtype=torch.bool, device=device)
  box_ok = torch.zeros(n, dtype=torch.bool, device=device)
  box_tilt_end = torch.full((n,), math.nan, device=device)
  box_z0 = env.get_state()[:, 73].clone() if env.has_box else None
  dt = env.step_dt

  obs = vec_env.get_observations()
  for t in range(num_steps + 1):
    k = env.get_ref_index()
    obs, rew, _, extras = vec_env.step(policy(obs).to(device))
    term = env.reset_terminated.clone()
    trunc = extras["time_outs"].to(torch.bool)
    m = active
    st.add(st.ret, rew, m)
    st.add(st.ret_track, env.get_reward_terms()[:, track_ids].sum(-1), m)
    st.add(st.steps, torch.ones_like(rew), m)
    e = tracking_errors(env, k)
    st.add(st.body_err_sum, e["body_mean"], m)
    st.add(st.anchor_err_sum, e["anchor_mean"], m)
    st.add(st.joint_err_sum, e["joint"], m)
    st.peak(st.body_err_max, e["body_max"], m)
    st.peak(st.anchor_err_max, e["anchor_max"], m)
    st.peak(st.qd_max, e["qd_max"], m)
    if env.has_box:
      s = env.get_state()
      box_z = s[:, 73]
      st.add(st.box_err_sum, torch.norm(s[:, 71:74] - cmd.box_ref_pos[k], dim=-1), m)
      st.peak(st.box_vz_max, s[:, 80].abs(), m)
      rise = box_z - box_z0
      tt = (t + 1) * dt
      st.t_low = torch.where(m & st.t_low.isnan() & (rise >= RISE_LOW), tt, st.t_low)
      st.t_high = torch.where(m & st.t_high.isnan() & (rise >= RISE_HIGH), tt, st.t_high)

    done = active & (term | trunc)
    success |= done & trunc & ~term
    fell |= done & term
    if env.has_box and done.any():
      s = env.get_state()
      ok = s[:, 73] >= cmd.box_ref_pos[k, 2] - BOX_SUCCESS_TOL
      box_ok |= done & trunc & ~term & ok
      # Tilt = angle between the box z axis and world z.
      up = quat_apply_inverse(s[:, 74:78], torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3))
      tilt = torch.rad2deg(torch.acos(up[:, 2].clamp(-1.0, 1.0)))
      box_tilt_end = torch.where(done, tilt, box_tilt_end)
    active = active & ~done
    env._manual_reset_pending.zero_()  # freeze finished envs instead of resetting them
    if not active.any():
      break

  steps = st.steps.clamp(min=1)
  per_ep = {
    "success": success.float(),
    "fall": fell.float(),
    "norm_return": st.ret / (num_steps * r_max),
    "norm_return_tracking": st.ret_track / (num_steps * r_max_track),
    "episode_steps": st.steps,
    "body_pos_err_mean_m": st.body_err_sum / steps,
    "body_pos_err_max_m": st.body_err_max,
    "anchor_rel_body_pos_err_mean_m": st.anchor_err_sum / steps,
    "anchor_rel_body_pos_err_max_m": st.anchor_err_max,
    "joint_pos_err_rad": st.joint_err_sum / steps,
    "peak_joint_speed": st.qd_max,
  }
  if env.has_box:
    per_ep |= {
      "box_success": box_ok.float(),
      "box_pos_err_m": st.box_err_sum / steps,
      "peak_box_vz": st.box_vz_max,
      "box_tilt_end_deg": box_tilt_end,
      "box_rise_5_to_55cm_s": st.t_high - st.t_low,
    }
  summary = {key: _nanmean(v) for key, v in per_ep.items()}
  summary["fall_rate"] = summary.pop("fall")
  summary["success_rate"] = summary.pop("success")
  summary["ref_peak_joint_speed"] = float(cmd.motion.joint_vel.abs().max())
  vec_env.close()
  return {
    "task": args.task,
    "learner": args.learner,
    "checkpoint": str(args.checkpoint),
    "condition": "randomized" if args.randomize else "nominal",
    "obs_noise": obs_noise(args),
    "episodes": n,
    "clip_steps": num_steps,
    "max_step_reward": r_max,
    "metrics": summary,
    "per_episode": {key: v.tolist() for key, v in per_ep.items()},
  }


def _carry_contact_parts(
  env: Any, box_center_z: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return actual left-hand, right-hand and floor contacts in each world."""
  model = env.sim.mj_model
  contact = env.sim.wp_data.contact
  geoms = wp.to_torch(contact.geom).long()
  worlds = wp.to_torch(contact.worldid).long()
  active = (wp.to_torch(contact.type) > 0) & (wp.to_torch(contact.dist) <= 0.005)
  active &= (worlds >= 0) & (worlds < env.num_envs)
  box = env._box_gid
  hand_ids = {side: [i for i in range(model.ngeom)
                     if (model.geom(i).name or "").endswith(side + "_hand_collision")]
              for side in ("left", "right")}
  floor_ids = [i for i in range(model.ngeom)
               if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE]
  if any(len(v) != 1 for v in hand_ids.values()) or not floor_ids:
    raise RuntimeError(f"Could not identify hand and floor geoms: {hand_ids}, {floor_ids}")
  paired_box = (geoms[:, 0] == box) | (geoms[:, 1] == box)
  contacts = []
  for gid in (hand_ids["left"][0], hand_ids["right"][0]):
    by_env = torch.zeros(env.num_envs, device=worlds.device, dtype=torch.bool)
    matching = active & paired_box & ((geoms[:, 0] == gid) | (geoms[:, 1] == gid))
    by_env[worlds[matching]] = True
    contacts.append(by_env)
  floor = torch.zeros_like(contacts[0])
  for gid in floor_ids:
    matching = active & paired_box & ((geoms[:, 0] == gid) | (geoms[:, 1] == gid))
    floor[worlds[matching]] = True
  if box_center_z is None:
    box_center_z = env.get_state()[:, 73]
  floor &= box_center_z <= CARRY_BOX_FLOOR_PLAUSIBLE_Z_M
  return contacts[0], contacts[1], floor


def _carry_contacts(env: Any) -> tuple[torch.Tensor, torch.Tensor]:
  """Preserve the two-hand contact interface used by physical replay probes."""
  left, right, floor = _carry_contact_parts(env)
  return left & right, floor


@torch.no_grad()
def evaluate_carryplace(args: argparse.Namespace) -> dict[str, Any]:
  """Score only real simulator rollouts, never SUGAR's reference contact labels."""
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab_g1.envs.smtrack_env import SMTrackEnv
  from mjlab_g1.tasks.smtrack import make_env_cfg
  from mjlab_g1.tasks.smtrack.carryplace_eval import evaluate_carry_place, stopped_in_fallen_pose
  from mjlab_g1.tasks.smtrack.mdp.box import CARRY_BOX_MASS

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  cfg = make_env_cfg(args.task, play=not args.randomize, motion_file=args.motion_file)
  mass_scale = None
  if args.box_mass_kg is not None:
    if not math.isfinite(args.box_mass_kg) or args.box_mass_kg <= 0:
      raise ValueError("--box-mass-kg must be a finite positive number")
    mass_scale = args.box_mass_kg / CARRY_BOX_MASS
    cfg.events["box_mass"].params["scale_range"] = (mass_scale, mass_scale)
  if args.randomize:
    cfg.commands["motion"].sampling_mode = "start"
  cfg.scene.num_envs = args.episodes
  cfg.seed = args.seed
  cfg.auto_reset = False
  env = SMTrackEnv(cfg, device=device)
  # The wrapper performs the first reset; reset-mode mass DR has not run before it.
  vec_env = RslRlVecEnvWrapper(env)
  if mass_scale is not None:
    measured_scale = env.get_xi()[:, -2]
    if not torch.allclose(measured_scale, torch.full_like(measured_scale, mass_scale),
                          rtol=1e-4, atol=1e-4):
      raise RuntimeError(
        f"actual simulator box mass differs from --box-mass-kg: target={args.box_mass_kg} kg, "
        f"target_scale={mass_scale}, observed_scales={measured_scale.detach().cpu().tolist()}"
      )
  policy = make_policy(args, vec_env, device)
  with np.load(env.motion_file) as ref:
    target_pos = ref["target_pos_w"].copy()
    target_quat = ref["target_quat_wxyz"].copy()
  n = env.num_envs
  clip_steps = env.motion_command.num_frames - 1
  initial = env.get_state().clone()
  samples = [initial]
  actions = [] if args.rollout_all_npz is not None else None
  xi = env.get_xi().detach().cpu().numpy() if actions is not None else None
  left_contacts = [torch.zeros(n, device=device, dtype=torch.bool)]
  right_contacts = [torch.zeros_like(left_contacts[0])]
  floor_contacts = [torch.zeros_like(left_contacts[0])]
  active = torch.ones(n, device=device, dtype=torch.bool)
  term_all = torch.zeros_like(active)
  complete = torch.zeros_like(active)
  steps = torch.zeros(n, device=device, dtype=torch.long)
  term_names = list(env.termination_manager.active_terms)
  terminal_causes = {name: torch.zeros_like(active) for name in term_names}
  obs = vec_env.get_observations()
  for t in range(clip_steps + 1):
    action = policy(obs).to(device)
    if actions is not None:
      actions.append(action.clone())
    obs, _, _, extras = vec_env.step(action)
    samples.append(env.get_state().clone())
    left, right, floor = _carry_contact_parts(env, samples[-1][:, 73])
    left_contacts.append(left)
    right_contacts.append(right)
    floor_contacts.append(floor)
    ended = env.reset_terminated.clone()
    truncated = extras["time_outs"].to(torch.bool)
    for name in term_names:
      terminal_causes[name] |= active & env.termination_manager.get_term(name)
    steps += active.long()
    term_all |= active & ended
    complete |= active & truncated & ~ended
    active &= ~(ended | truncated)
    env._manual_reset_pending.zero_()
    if not active.any():
      break
  state = torch.stack(samples).detach().cpu().numpy()
  left_hands = torch.stack(left_contacts).detach().cpu().numpy()
  right_hands = torch.stack(right_contacts).detach().cpu().numpy()
  hands = left_hands & right_hands
  floor = torch.stack(floor_contacts).detach().cpu().numpy()
  counts = steps.detach().cpu().numpy()
  completed = complete.detach().cpu().numpy()
  terminated = term_all.detach().cpu().numpy()
  causes = {key: val.detach().cpu().numpy() for key, val in terminal_causes.items()}
  per_episode = []
  for i in range(n):
    stop = int(counts[i]) + 1  # the initial state plus each physically simulated step
    traj = state[:stop, i]
    # A fall before 50 frames has no physically completed 50 Hz episode. Padding
    # enables partial phase diagnostics; completed=False always forbids success.
    pad = max(0, 50 - stop)
    padded = np.pad(traj, ((0, pad), (0, 0)), mode="edge")
    hand = np.pad(hands[:stop, i], (0, pad), constant_values=False)
    on_floor = np.pad(floor[:stop, i], (0, pad), constant_values=False)
    score = evaluate_carry_place(
      box_pos=padded[:, 71:74], box_quat_wxyz=padded[:, 74:78],
      box_lin_vel=padded[:, 78:81], box_ang_vel=padded[:, 81:84],
      pelvis_pos=padded[:, :3], pelvis_quat_wxyz=padded[:, 3:7],
      both_hands_contact=hand, floor_contact=on_floor,
      target_pos=target_pos, target_quat_wxyz=target_quat,
      episode_complete=bool(completed[i]),
    )
    per_episode.append({"episode_steps": int(counts[i]), "terminated": bool(terminated[i]),
                        "terminal_causes": [key for key in term_names if causes[key][i]],
                        "final_pelvis_height_m": float(traj[-1, 2]),
                        "final_box_xyz_m": traj[-1, 71:74].tolist(),
                        "first_left_contact_frame": next((k for k, hit in
                                                          enumerate(left_hands[:stop, i]) if hit), None),
                        "first_right_contact_frame": next((k for k, hit in
                                                           enumerate(right_hands[:stop, i]) if hit), None),
                        "left_hand_contact_frames": int(left_hands[:stop, i].sum()),
                        "right_hand_contact_frames": int(right_hands[:stop, i].sum()),
                        "both_hands_contact_frames": int(hands[:stop, i].sum()),
                        **score})
  if args.rollout_npz is not None:
    if not 0 <= args.rollout_index < n:
      raise ValueError(f"--rollout-index must be in [0, {n}), got {args.rollout_index}")
    args.rollout_npz.parent.mkdir(parents=True, exist_ok=True)
    i = args.rollout_index
    stop = int(counts[i]) + 1
    np.savez_compressed(args.rollout_npz, state=state[:stop, i],
                        both_hands_contact=hands[:stop, i], floor_contact=floor[:stop, i],
                        left_hand_contact=left_hands[:stop, i],
                        right_hand_contact=right_hands[:stop, i],
                        episode_complete=bool(completed[i]), terminated=bool(terminated[i]),
                        episode_success=bool(per_episode[i]["success"]),
                        episode_index=i, seed=args.seed, randomized=bool(args.randomize),
                        checkpoint=str(args.checkpoint),
                        target_pos_w=target_pos, target_quat_wxyz=target_quat,
                        motion_file=np.asarray(env.motion_file))
  if args.rollout_all_npz is not None:
    args.rollout_all_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.rollout_all_npz, state=state,
                        action=torch.stack(actions).detach().cpu().numpy(),
                        xi=xi, episode_steps=counts, left_hand_contact=left_hands,
                        right_hand_contact=right_hands, floor_contact=floor,
                        seed=args.seed, randomized=bool(args.randomize),
                        checkpoint=str(args.checkpoint), motion_file=np.asarray(env.motion_file))
  keys = ("grasp", "carry", "place", "released", "settled", "upright",
          "completed", "success", "transport_m", "carried_m", "max_lift_m",
          "target_error_m", "episode_steps")
  metrics = {k + ("_rate" if isinstance(per_episode[0][k], (bool, np.bool_)) else "_mean"):
             float(np.mean([e[k] for e in per_episode])) for k in keys}
  # Box-position failure may end an upright robot. Earlier evaluations called
  # every non-clip termination a fall; retain the event under its true name.
  metrics["non_clip_termination_rate"] = float(np.mean(terminated))
  metrics["fall_rate"] = float(np.mean([
      stopped_in_fallen_pose(e["terminated"], e["final_pelvis_height_m"],
                             e["final_pelvis_tilt_rad"]) for e in per_episode]))
  for name, values in causes.items():
    metrics["termination_" + name + "_rate"] = float(np.mean(values))
  for name in ("left_hand_contact_frames", "right_hand_contact_frames", "both_hands_contact_frames"):
    metrics[name + "_mean"] = float(np.mean([e[name] for e in per_episode]))
    metrics[name.replace("_frames", "_ever_rate")] = float(np.mean([e[name] > 0 for e in per_episode]))
  vec_env.close()
  result = {"task": args.task, "learner": args.learner,
          "checkpoint": str(args.checkpoint), "motion_file": str(cfg.commands["motion"].motion_file),
          "condition": "randomized" if args.randomize else "nominal",
          "obs_noise": obs_noise(args), "seed": args.seed, "episodes": n,
          "clip_steps": clip_steps, "metric_version": 2,
          "metrics": metrics, "per_episode": per_episode,
          "scope": "Real MuJoCo Warp policy rollout contacts, not kinematic labels"}
  if mass_scale is not None:
    result["box_mass_kg"] = float(args.box_mass_kg)
    result["measured_box_mass_scale"] = float(measured_scale.mean())
  return result


def _nanmean(x: torch.Tensor) -> float | None:
  x = x.float()
  ok = ~x.isnan()
  return float(x[ok].mean()) if ok.any() else None


def main(argv: list[str] | None = None) -> dict[str, Any]:
  args = parse_args(argv)
  if args.box_mass_kg is not None and args.task != "G1-SMTrack-CarryPlace":
    raise ValueError("--box-mass-kg is only supported for G1-SMTrack-CarryPlace")
  result = evaluate(args)
  text = json.dumps(result, indent=2)
  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
  print(json.dumps({"task": result["task"], "learner": result["learner"], **result["metrics"]}, indent=2))
  return result


if __name__ == "__main__":
  os.environ.setdefault("MUJOCO_GL", "disable")
  main()
