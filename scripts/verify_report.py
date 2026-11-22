"""Check published CarryPlace numbers against archived JSON and training logs.

No simulator or CUDA needed: python scripts/verify_report.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs" / "metrics"
SEEDS = (2027, 314159, 161803)
EXPECTED = {
    "sm4096_h32_ori04_s0": (1000, (51, 51, 51), 13789),
    "sm4096_h16_ori04_s0": (800, (32, 34, 36), 6873),
    "sm4096_h32_ori12_s0": (1000, (58, 52, 58), 13503),
    "sm4096_h16_ori12_s0": (800, (22, 35, 36), 6986),
}


def read_json(name: str) -> dict:
    return json.loads((ARCHIVE / name).read_text(encoding="utf-8"))


def main() -> None:
    runs = read_json("sm4096_fourway_summary.json")["runs"]
    corrected = read_json("sm4096_floor_guard_summary.json")
    print("run | checkpoint | selected_s | total_s | env_step/s | peak_MiB | intermittent /192 | strict /192")
    for name, (expected_iter, expected_counts, expected_time) in EXPECTED.items():
        source = runs[name]
        fixed = corrected["runs"][name]
        iteration = fixed["best_iteration_from_corrected_dev"]
        assert iteration == expected_iter
        log = (ARCHIVE / f"{name}.log").read_text(encoding="utf-8")
        match = re.search(rf"\[SuperMimic\] it {iteration} \|[^\n]*\| (\d+)s", log)
        assert match is not None
        selected_seconds = int(match.group(1))
        assert selected_seconds == expected_time
        counts, strict = [], []
        for seed in SEEDS:
            suffix = f"iter{iteration}_seed{seed}_n64"
            report = fixed["holdout"][str(seed)][suffix]
            path = ARCHIVE / f"{name.removesuffix('_s0')}_{suffix}_geometry.json"
            scores = read_json(path.name)["per_episode"]
            assert len(scores) == 64
            count = sum(bool(row["intermittent_carry"]) for row in scores)
            strict_count = sum(bool(row["strict_success_after_floor_guard"]) for row in scores)
            assert abs(count / 64 - report["metrics"]["intermittent_carry_rate"]) < 1e-12
            counts.append(count)
            strict.append(strict_count)
        assert tuple(counts) == expected_counts
        assert sum(strict) == 0
        t = source["training"]
        assert t["last_iteration"] == 1000 and t["env_steps"] == 10371072
        print(f"{name} | {iteration} | {selected_seconds} | {t['wall_s']:g} | "
              f"{t['steady_real_env_steps_s']:.0f} | {t['gpu_peak_used_mib_2s_samples']} | "
              f"{'/'.join(map(str, counts))} ({sum(counts)}/192) | {sum(strict)}/192")

    old, new, intermittent = [], [], []
    for seed in SEEDS[:2]:
        original = read_json(f"ppo3999_heldout_seed{seed}_64.json")
        assert original["episodes"] == 64
        old.append(sum(bool(row["success"]) for row in original["per_episode"]))
        metrics = corrected["historical"][f"ppo-seed{seed}"][f"iter3999_seed{seed}_n64"]["metrics"]
        new.append(round(metrics["strict_success_after_floor_guard_rate"] * 64))
        intermittent.append(round(metrics["intermittent_carry_rate"] * 64))
    assert old == [22, 29] and new == [22, 31] and intermittent == [64, 63]
    baseline = read_json("summary_post_sm.json")["PPO"]
    assert baseline["training"]["last_total_env_steps"] == 786432000
    print(f"PPO model_3999 | original strict: {old} ({sum(old)}/128) | "
          f"floor-guard strict: {new} ({sum(new)}/128) | "
          f"intermittent: {intermittent} ({sum(intermittent)}/128) | "
          f"wall_s={baseline['elapsed_seconds']:g} | gpu_hours={baseline['gpu_hours_so_far']:g}")
    print("Archive verification passed. Training and GPU inference were not rerun.")


if __name__ == "__main__":
    main()
