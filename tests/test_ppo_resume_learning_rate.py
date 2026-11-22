"""A resumed adaptive PPO run must preserve its checkpoint's actual learning rate."""
from __future__ import annotations

import unittest

import torch
from rsl_rl.algorithms.ppo import PPO


class PpoResumeLearningRateTest(unittest.TestCase):
    def test_resume_restores_optimizer_and_kl_scheduler_learning_rates(self) -> None:
        def make_ppo() -> PPO:
            return PPO(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1),
                       storage=object(), device="cpu", learning_rate=1e-3)

        original = make_ppo()
        for group in original.optimizer.param_groups:
            group["lr"] = 1e-5
        state = original.save()

        resumed = make_ppo()
        self.assertEqual(resumed.learning_rate, 1e-3)
        resumed.load(state, None, strict=True)
        self.assertAlmostEqual(resumed.learning_rate, 1e-5)
        self.assertAlmostEqual(resumed.optimizer.param_groups[0]["lr"], 1e-5)


if __name__ == "__main__":
    unittest.main()
