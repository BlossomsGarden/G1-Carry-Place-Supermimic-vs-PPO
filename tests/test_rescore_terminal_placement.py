"""Archived physical episodes use the same position-only decision as live scoring."""

from scripts.rescore_terminal_placement import build_report


def test_archived_endpoints_and_intermediate_scores_are_reproducible():
    groups = build_report()["groups"]
    assert len(groups) == 5
    assert [groups[name]["counts"]["terminal_box_near_goal"] for name in (
        "sm4096_h32_ori04_s0", "sm4096_h16_ori04_s0",
        "sm4096_h32_ori12_s0", "sm4096_h16_ori12_s0", "ppo3999"
    )] == [178, 175, 188, 131, 127]
    best = groups["sm4096_h32_ori12_s0"]
    assert [row["counts"]["terminal_box_near_goal"]
            for row in best["by_seed"].values()] == [64, 60, 64]
    assert best["counts"]["intermittent_carry"] == 168
    assert best["counts"]["strict_success_after_floor_guard"] == 0
    assert best["near_without_intermediate_carry"] == 20
    assert all(group["near_but_early_termination"] == 0 for group in groups.values())
