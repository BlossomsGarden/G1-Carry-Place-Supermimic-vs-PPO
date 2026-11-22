"""Check resumed learner accounting without needing a GPU or mjlab imports."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/summarize_carry_runs.py"
if not SOURCE.is_file():
    raise unittest.SkipTest("Legacy CarryPlace runs analyzer is not shipped in this repository")
SPEC = importlib.util.spec_from_file_location("carry_run_summary", SOURCE)
assert SPEC and SPEC.loader
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


class SuperMimicLogAccountingTest(unittest.TestCase):
    def test_four_rank_resume_does_not_recount_loaded_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            # The checkpoint stores 4,161,536 transitions per old rank. The new
            # four-rank run replenishes 4,096*32 and takes 50*4,096*2.5 steps.
            record = {"iteration": 850, "wall_time": 458.0,
                      "perf/prefill_s": 26.0,
                      "env_steps": 4 * 4_161_536 + 4096 * 32 + 50 * 4096 * 2.5}
            (run / "metrics.jsonl").write_text(json.dumps(record), encoding="utf-8")
            stats = SUMMARY.supermimic_metrics(
                run, resume_iteration=800,
                preloaded_env_steps=4 * 4_161_536, prefill_envs=4096)
            self.assertEqual(stats["recent_interval_s_per_iter"], 8.64)
            self.assertEqual(stats["recent_interval_env_steps_per_s"], 1185)

    def test_first_baseline_interval_excludes_prefill(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            record = {"iteration": 50, "wall_time": 466.0,
                      "perf/prefill_s": 26.0,
                      "env_steps": 4096 * 32 + 50 * 4096 * 2.5}
            (run / "metrics.jsonl").write_text(json.dumps(record), encoding="utf-8")
            stats = SUMMARY.supermimic_metrics(run)
            self.assertEqual(stats["recent_interval_s_per_iter"], 8.8)
            self.assertEqual(stats["recent_interval_env_steps_per_s"], 1164)


if __name__ == "__main__":
    unittest.main()
