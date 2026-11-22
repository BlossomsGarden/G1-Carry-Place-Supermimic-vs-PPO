"""Checkpoint glob and development-only best selection must see real names."""
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
LEGACY = (ROOT / "scripts/finalize_carry_training.py",
          ROOT / "scripts/monitor_carry_aux.py")
if not all(path.is_file() for path in LEGACY):
    raise unittest.SkipTest("Legacy curriculum monitors are not shipped in this repository")

from scripts import finalize_carry_training as finalizer
from scripts import monitor_carry_aux as auxiliary


class CheckpointMonitorTest(unittest.TestCase):
    def test_checkpoint_names_are_found(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root / "tests") as temporary:
            folder = Path(temporary) / "logs/smtrack/G1-SMTrack-CarryPlace/ppo"
            run = folder / "2026-11-22_01-51-11_cp_ppo_grasp145_350_startonly_s0"
            run.mkdir(parents=True)
            (run / "model_1200.pt").touch()
            (run / "model_invalid.pt").touch()
            old_aux, old_final = auxiliary.PROJECT, finalizer.PROJECT
            try:
                auxiliary.PROJECT = finalizer.PROJECT = Path(temporary)
                rows = list(auxiliary.candidates())
                self.assertEqual([(item[1], item[2]) for item in rows],
                                 [("cp_ppo_grasp145_350_startonly_s0", 1200)])
                self.assertEqual(list(finalizer.checkpoints(
                    "ppo", "cp_ppo_grasp145_350_startonly_s0")), [1200])
            finally:
                auxiliary.PROJECT, finalizer.PROJECT = old_aux, old_final

    def test_best_uses_development_score(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root / "tests") as temporary:
            evals = Path(temporary) / "evaluations"
            evals.mkdir()
            name = "cp_ppo_v2_s0_4g"
            for iteration, successes in ((800, 0.125), (1000, 0.0)):
                metrics = {"success_rate": successes, "carry_rate": 0.5,
                           "grasp_rate": 0.5, "max_lift_m_mean": 0.5,
                           "episode_steps_mean": 499.0}
                (evals / f"{name}_iter{iteration:04d}_randomized16.json").write_text(
                    json.dumps({"metrics": metrics}), encoding="utf-8")
            previous = finalizer.ROOT
            try:
                finalizer.ROOT = Path(temporary)
                chosen, _ = finalizer.select_best({800: Path("a"), 1000: Path("b")}, name)
                self.assertEqual(chosen, 800)
            finally:
                finalizer.ROOT = previous


if __name__ == "__main__":
    unittest.main()
