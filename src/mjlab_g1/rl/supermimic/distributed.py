"""Synchronous data parallel helpers for the SuperMimic learner.

Each process owns its own simulator and replay. All ranks share model parameters,
running normalization statistics and averaged gradients before each Adam step.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn


def world_size() -> int:
  return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


@torch.no_grad()
def batch_moments(x: torch.Tensor, valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Globally aggregate sample count, mean and M2 of a feature batch."""
  x = x.reshape(-1, x.shape[-1]).to(torch.float64)
  n = valid.to(torch.float64).sum() if valid is not None else x.new_tensor(x.shape[0])
  sums = x.sum(0)
  sums_sq = x.square().sum(0)
  if world_size() > 1:
    packed = torch.cat((n.reshape(1), sums, sums_sq))
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    n, sums, sums_sq = packed[0], packed[1:1 + x.shape[-1]], packed[1 + x.shape[-1]:]
  mean = sums / n.clamp(min=1)
  m2 = (sums_sq - sums.square() / n.clamp(min=1)).clamp(min=0)
  return n, mean, m2


@torch.no_grad()
def broadcast_model(module: nn.Module) -> None:
  if world_size() > 1:
    for param in module.parameters():
      dist.broadcast(param.data, src=0)
    for buf in module.buffers():
      dist.broadcast(buf, src=0)


@torch.no_grad()
def average_gradients(module: nn.Module) -> None:
  """Reduce before norm clipping, so every rank takes the same optimizer step."""
  if world_size() == 1:
    return
  params = [p for p in module.parameters() if p.requires_grad]
  if not params:
    return
  flat = torch.cat([p.grad.reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1) for p in params])
  dist.all_reduce(flat, op=dist.ReduceOp.SUM)
  flat.div_(world_size())
  offset = 0
  for p in params:
    part = flat[offset:offset + p.numel()].view_as(p)
    if p.grad is None:
      p.grad = part.clone()
    else:
      p.grad.copy_(part)
    offset += p.numel()
