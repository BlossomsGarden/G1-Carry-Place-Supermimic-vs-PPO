"""Train PPO or SuperMimic on an SMTrack task (brief §5).

  PYTHONPATH=src python src/mjlab_g1/scripts/train_track.py G1-SMTrack-Walk --learner ppo
  PYTHONPATH=src python src/mjlab_g1/scripts/train_track.py G1-SMTrack-Lift --learner supermimic \
    --num-envs 4096 --max-iterations 100000 --seed 0 --run-name s0
  ... --smoke --device cpu      # CPU plumbing check: 8 envs, small nets, 3 iterations

Both learners build the same env cfg (only num_envs differs) and log to
logs/smtrack/<task>/<learner>/<date>_<run>/ with params/{env,agent}.{yaml,json}.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from mjlab.utils.os import dump_yaml

LEARNERS = ("ppo", "supermimic")
DEFAULT_NUM_ENVS = {"ppo": 8192, "supermimic": 4096}  # paper PPO-Sim / Table A.5
SMOKE_NUM_ENVS = 8
SMOKE_ITERS = 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  from mjlab_g1.tasks.smtrack import list_tasks

  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("task", choices=list_tasks())
  p.add_argument("--learner", choices=LEARNERS, required=True)
  p.add_argument("--num-envs", type=int, default=None)
  p.add_argument("--motion-file", type=Path, default=None,
                 help="override the task's reference clip for both learners")
  p.add_argument("--max-iterations", type=int, default=None)
  p.add_argument("--lr-cosine-iters", type=int, default=None,
                 help="SuperMimic: keep original LR horizon when limiting experiment budget")
  p.add_argument("--resume-checkpoint", type=Path, default=None,
                 help="resume learner weights/optimizers; SuperMimic replay is repopulated")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--run-name", default="")
  p.add_argument("--device", default=None, help="default: cuda:0 if available, else cpu")
  p.add_argument("--log-root", default="logs/smtrack")
  p.add_argument("--smoke", action="store_true", help="8 envs, widths/16, 3 iterations")
  p.add_argument("--save-interval", type=int, default=None, help="checkpoint every N iterations")
  p.add_argument("--imag-microbatches", type=int, default=None,
                 help="accumulate the same imagined batch in N smaller chunks (reduces peak GPU memory)")
  p.add_argument("--imagination-horizon", type=int, default=None,
                 help="SuperMimic auxiliary study: replace the 32-step imagined rollout length")
  p.add_argument(
    "--start-uniform-mix",
    type=float,
    default=None,
    help="uniform weight of the adaptive start-frame sampler (paper: 0.5; default: env cfg)",
  )
  p.add_argument("--start-only", action="store_true",
                 help="auxiliary curriculum: reset every episode at frame 0 of the selected clip")
  p.add_argument("--box-ori-std", type=float, default=None,
                 help="auxiliary reward study: positive orientation error scale in radians")
  p.add_argument("--box-mass-kg", type=float, default=None,
                 help="CarryPlace-only auxiliary study: fix actual box mass during training")
  p.add_argument(
    "--compile",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="torch.compile the networks (default on; --no-compile for eager)",
  )
  return p.parse_args(argv)


def _shrink(dims: tuple[int, ...] | list[int]) -> tuple[int, ...]:
  return tuple(max(4, d // 16) for d in dims)


def build_agent_cfg(args: argparse.Namespace) -> Any:
  from mjlab_g1.tasks.smtrack import make_ppo_cfg

  if args.learner == "ppo":
    if args.imagination_horizon is not None:
      raise ValueError("--imagination-horizon is only supported for SuperMimic")
    cfg = make_ppo_cfg(args.task)
    if args.smoke:
      cfg.actor.hidden_dims = _shrink(cfg.actor.hidden_dims)
      cfg.critic.hidden_dims = _shrink(cfg.critic.hidden_dims)
      cfg.max_iterations = SMOKE_ITERS
      cfg.save_interval = 1
  else:
    from mjlab_g1.rl.supermimic.config import SuperMimicRunnerCfg

    cfg = SuperMimicRunnerCfg.smoke() if args.smoke else SuperMimicRunnerCfg()
    cfg.experiment_name = f"smtrack_{args.task}_supermimic"
  if args.learner == "supermimic":
    cfg.memory.compile = args.compile
    if args.lr_cosine_iters is not None:
      if args.lr_cosine_iters < 1:
        raise ValueError("--lr-cosine-iters must be positive")
      cfg.lr_cosine_iters = args.lr_cosine_iters
    if args.imag_microbatches is not None:
      if args.imag_microbatches < 1:
        raise ValueError("--imag-microbatches must be positive")
      cfg.memory.imag_microbatches = args.imag_microbatches
    if args.imagination_horizon is not None:
      if args.imagination_horizon < 1:
        raise ValueError("--imagination-horizon must be positive")
      cfg.policy.horizon = args.imagination_horizon
  if args.max_iterations is not None:
    cfg.max_iterations = args.max_iterations
  if args.save_interval is not None:
    cfg.save_interval = args.save_interval
  cfg.seed = args.seed
  cfg.run_name = args.run_name
  cfg.logger = "tensorboard"
  return cfg


def make_log_dir(args: argparse.Namespace) -> Path:
  name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.run_name:
    name += f"_{args.run_name}"
  return Path(args.log_root).resolve() / args.task / args.learner / name


def dump_params(log_dir: Path, env_cfg: Any, agent_cfg: Any) -> None:
  """yaml for people, json for eval_track.py (tuples come back as lists either way)."""
  env_d, agent_d = dataclasses.asdict(env_cfg), dataclasses.asdict(agent_cfg)
  dump_yaml(log_dir / "params" / "env.yaml", env_d)
  dump_yaml(log_dir / "params" / "agent.yaml", agent_d)
  with open(log_dir / "params" / "agent.json", "w") as f:
    json.dump(agent_d, f, indent=2, default=str)


def make_runner(learner: str, env: Any, agent_cfg: dict, log_dir: str | None, device: str) -> Any:
  """Runner for either learner; `agent_cfg` is a dict (runners mutate it in place)."""
  if learner == "ppo":
    from mjlab.rl import MjlabOnPolicyRunner

    # mjlab's runner strips the cnn/rnn keys the vendored MLPModel rejects.
    return MjlabOnPolicyRunner(env, agent_cfg, log_dir, device)
  from mjlab_g1.rl.supermimic.runner import SuperMimicRunner

  return SuperMimicRunner(env, agent_cfg, log_dir, device)


def main(argv: list[str] | None = None) -> Path:
  args = parse_args(argv)
  world = int(os.environ.get("WORLD_SIZE", "1"))
  rank = int(os.environ.get("RANK", "0"))
  local_rank = int(os.environ.get("LOCAL_RANK", "0"))
  if world > 1:
    if not torch.cuda.is_available():
      raise RuntimeError("distributed training requires CUDA")
    torch.cuda.set_device(local_rank)
    if args.learner == "supermimic":
      # rsl_rl's PPO runner initializes its own process group.
      dist.init_process_group(backend="nccl")
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.utils.torch import configure_torch_backends
  from mjlab_g1.envs.smtrack_env import SMTrackEnv
  from mjlab_g1.tasks.smtrack import make_env_cfg

  device = f"cuda:{local_rank}" if world > 1 else (args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
  configure_torch_backends()
  if args.compile and not device.startswith("cuda"):
    # Inductor's CPU backend needs a C++ toolchain; CPU runs are plumbing checks anyway.
    print("[train_track] --compile ignored on CPU", flush=True)
    args.compile = False

  env_cfg = make_env_cfg(args.task, motion_file=args.motion_file)
  if args.box_mass_kg is not None:
    from mjlab_g1.tasks.smtrack.mdp.box import CARRY_BOX_MASS

    if args.task != "G1-SMTrack-CarryPlace" or args.learner != "ppo":
      raise ValueError("--box-mass-kg is only supported for CarryPlace PPO fine-tuning")
    if not math.isfinite(args.box_mass_kg) or args.box_mass_kg <= 0:
      raise ValueError("--box-mass-kg must be a finite positive number")
    scale = args.box_mass_kg / CARRY_BOX_MASS
    env_cfg.events["box_mass"].params["scale_range"] = (scale, scale)
  if args.smoke:
    num_envs = SMOKE_NUM_ENVS
  else:
    num_envs = args.num_envs or DEFAULT_NUM_ENVS[args.learner]
  if num_envs % world:
    raise ValueError(f"--num-envs ({num_envs}) must be divisible by WORLD_SIZE ({world})")
  env_cfg.scene.num_envs = num_envs // world
  env_cfg.seed = args.seed + rank
  if args.start_uniform_mix is not None:
    env_cfg.commands["motion"].adaptive_uniform_mix = args.start_uniform_mix
  if args.start_only:
    env_cfg.commands["motion"].sampling_mode = "start"
  if args.box_ori_std is not None:
    if args.box_ori_std <= 0 or "box_ori" not in env_cfg.rewards:
      raise ValueError("--box-ori-std requires a positive scale and a box task")
    env_cfg.rewards["box_ori"].params["std"] = args.box_ori_std
  if args.learner == "ppo":
    # The clip end is the end of the task, not a time limit: with no `time_outs` in
    # extras rsl_rl's PPO does not bootstrap there, matching SuperMimic's clip_end="stop".
    env_cfg.is_finite_horizon = True
  agent_cfg = build_agent_cfg(args)
  if world > 1 and args.learner == "supermimic":
    if agent_cfg.wm.batch_windows % world or agent_cfg.policy.num_imagined % world:
      raise ValueError("world-model batch and imagined batch must divide by WORLD_SIZE")
    agent_cfg.wm.batch_windows //= world
    agent_cfg.policy.num_imagined //= world

  log_dir = make_log_dir(args) if rank == 0 else None
  if world > 1 and args.learner == "supermimic":
    shared = [str(log_dir) if rank == 0 else ""]
    dist.broadcast_object_list(shared, src=0)
    log_dir = Path(shared[0])
  if rank == 0:
    dump_params(log_dir, env_cfg, agent_cfg)
    if world > 1:
      distributed_info = {
        "world_size": world, "global_num_envs": num_envs,
        "envs_per_rank": num_envs // world,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
      }
      if args.learner == "supermimic":
        distributed_info.update({
          "imagined_per_rank": agent_cfg.policy.num_imagined,
          "wm_batch_windows_per_rank": agent_cfg.wm.batch_windows,
        })
      with open(log_dir / "params" / "distributed.json", "w") as f:
        json.dump(distributed_info, f, indent=2)
    print(f"[train_track] {args.task} {args.learner} N={num_envs} world={world} "
          f"iters={agent_cfg.max_iterations} seed={args.seed} -> {log_dir}", flush=True)

  env = SMTrackEnv(env_cfg, device=device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if args.box_mass_kg is not None:
    observed = env.get_xi()[:, -2]
    expected = torch.full_like(observed, args.box_mass_kg / CARRY_BOX_MASS)
    if not torch.allclose(observed, expected, rtol=1e-4, atol=1e-4):
      raise RuntimeError(f"box mass override did not reach simulation: {observed[:4]}")
    if rank == 0:
      print(f"[train_track] verified simulator box mass={args.box_mass_kg:g} kg", flush=True)
  agent_dict = dataclasses.asdict(agent_cfg)
  if args.learner == "ppo" and args.compile:
    from mjlab_g1.tasks.smtrack.rl_cfgs import TORCH_COMPILE_MODE

    agent_dict["torch_compile_mode"] = TORCH_COMPILE_MODE
  runner = make_runner(args.learner, vec_env, agent_dict, str(log_dir) if rank == 0 else None, device)
  if args.resume_checkpoint is not None:
    runner.load(str(args.resume_checkpoint), map_location=device)
    if rank == 0:
      detail = ("SuperMimic replay repopulates" if args.learner == "supermimic"
                else f"PPO adaptive lr={runner.alg.learning_rate:.8g}")
      print(f"[train_track] resumed from {args.resume_checkpoint} at "
            f"iteration {runner.current_learning_iteration}; {detail}", flush=True)
  if world > 1 and args.learner == "supermimic":
    from mjlab_g1.rl.supermimic.distributed import broadcast_model

    for module in (runner.alg.wm, runner.alg.actor, runner.alg.critic):
      broadcast_model(module)
  # Start frames come from the adaptive sampler; random episode clocks would only shorten
  # the first episodes (the clip-end truncation is keyed on the reference frame).
  remaining = max(0, agent_cfg.max_iterations - runner.current_learning_iteration)
  runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=False)
  if world > 1:
    with torch.no_grad():
      modules = ((runner.alg.wm, runner.alg.actor, runner.alg.critic)
                 if args.learner == "supermimic" else (runner.alg.actor, runner.alg.critic))
      checks = torch.stack([
        torch.stack([p.detach().double().sum() for p in module.parameters()]).sum()
        for module in modules
      ] + [
        torch.stack([p.detach().double().square().sum() for p in module.parameters()]).sum()
        for module in modules
      ] + [torch.tensor(torch.cuda.max_memory_allocated(device) / 2**20, device=device)])
      gathered = [torch.empty_like(checks) for _ in range(world)]
      dist.all_gather(gathered, checks)
      rank_stats = torch.stack(gathered).cpu()
      drift = (rank_stats[:, :-1] - rank_stats[0, :-1]).abs().max().item()
    if rank == 0:
      with open(log_dir / "params" / "distributed_validation.json", "w") as f:
        json.dump({"max_model_checksum_difference": drift,
                   "pytorch_peak_allocated_mib_by_rank": rank_stats[:, -1].tolist()}, f, indent=2)
    if drift > 1e-6:
      raise RuntimeError(f"distributed model parameters diverged between ranks: {drift}")
  vec_env.close()
  if world > 1:
    dist.barrier()
    dist.destroy_process_group()
  return log_dir


if __name__ == "__main__":
  os.environ.setdefault("MUJOCO_GL", "disable")
  main()
