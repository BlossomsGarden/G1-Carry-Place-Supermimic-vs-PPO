"""Phase gates for 50 Hz simulated G1 CarryPlace rollouts."""
import sys
import json
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mjlab_g1.tasks.smtrack.carryplace_eval import evaluate_carry_place, stopped_in_fallen_pose


def episode() -> dict:
    n = 120
    box = np.zeros((n, 3))
    box[:, 2] = 0.2
    box[30:80, 2] = 0.45
    box[35:80, 0] = np.linspace(0.0, 0.8, 45)
    box[80:, 0] = 0.8
    hands = np.zeros(n, dtype=bool)
    hands[20:80] = True
    floor = np.ones(n, dtype=bool)
    floor[30:80] = False
    identity = np.zeros((n, 4))
    identity[:, 0] = 1.0
    pelvis = np.zeros_like(box)
    pelvis[:, 2] = 0.8
    return dict(box_pos=box, box_quat_wxyz=identity,
                box_lin_vel=np.gradient(box, 0.02, axis=0),
                box_ang_vel=np.zeros_like(box), pelvis_pos=pelvis,
                pelvis_quat_wxyz=identity.copy(),
                both_hands_contact=hands, floor_contact=floor,
                target_pos=np.array([0.8, 0.0, 0.2]),
                target_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]))


class CarryPlaceSuccessTest(unittest.TestCase):
    def test_box_termination_is_not_a_physical_fall(self):
        self.assertFalse(stopped_in_fallen_pose(True, 0.77, 0.78))
        self.assertTrue(stopped_in_fallen_pose(True, 0.52, 0.10))
        self.assertTrue(stopped_in_fallen_pose(True, 0.77, 1.10))
        self.assertFalse(stopped_in_fallen_pose(False, 0.52, 1.10))

    def test_good_sequence(self):
        score = evaluate_carry_place(**episode())
        self.assertTrue(score["success"])
        self.assertGreaterEqual(score["carried_m"], 0.5)
        self.assertEqual(score["first_lift_frame"], 30)
        self.assertEqual(score["first_bilateral_contact_frame"], 20)
        self.assertEqual(score["longest_prelift_bilateral_contact_frames"], 10)
        json.dumps(score)  # every per-episode diagnostic must fit the JSON report

    def test_late_bilateral_contact_still_fails_strict_grasp(self):
        e = episode()
        e["both_hands_contact"][:35] = False
        score = evaluate_carry_place(**e)
        self.assertEqual(score["first_lift_frame"], 30)
        self.assertEqual(score["first_bilateral_contact_frame"], 35)
        self.assertEqual(score["longest_prelift_bilateral_contact_frames"], 0)
        self.assertGreaterEqual(score["carried_m"], 0.5)
        self.assertFalse(score["grasp"])
        self.assertFalse(score["success"])

    def test_floor_drag_is_not_carry(self):
        e = episode()
        e["box_pos"][20:30, 0] = np.linspace(0, 0.8, 10)
        e["box_pos"][30:, 0] = 0.8
        e["box_lin_vel"] = np.gradient(e["box_pos"], 0.02, axis=0)
        score = evaluate_carry_place(**e)
        self.assertFalse(score["carry"])
        self.assertFalse(score["success"])

    def test_not_released(self):
        e = episode()
        e["both_hands_contact"][-15:] = True
        self.assertFalse(evaluate_carry_place(**e)["released"])

    def test_wrong_target_or_no_floor_contact(self):
        e = episode()
        e["target_pos"][0] = 1.3
        e["floor_contact"][-15:] = False
        score = evaluate_carry_place(**e)
        self.assertFalse(score["place"])
        self.assertFalse(score["released"])

    def test_unsettled_box(self):
        e = episode()
        e["box_lin_vel"][-15:, 0] = 0.3
        self.assertFalse(evaluate_carry_place(**e)["settled"])

    def test_early_termination_cannot_count_as_success(self):
        e = episode()
        score = evaluate_carry_place(**e, episode_complete=False)
        self.assertFalse(score["completed"])
        self.assertFalse(score["success"])

    def test_unmoved_box_does_not_count_as_settled_stage(self):
        e = episode()
        e["box_pos"][:, 0] = 0.0
        e["box_pos"][:, 2] = 0.2
        e["box_lin_vel"][:] = 0
        e["both_hands_contact"][:] = False
        e["floor_contact"][:] = True
        score = evaluate_carry_place(**e)
        self.assertFalse(score["grasp"])
        self.assertFalse(score["settled"])

    def test_ground_supported_transport_is_not_carry(self):
        e = episode()
        e["floor_contact"][:] = True
        score = evaluate_carry_place(**e)
        self.assertFalse(score["carry"])


if __name__ == "__main__":
    unittest.main()
