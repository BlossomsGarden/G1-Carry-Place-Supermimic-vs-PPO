"""Regression tests for the second, physically grounded CarryPlace score."""

from pathlib import Path

import numpy as np

from scripts.score_carry_object_outcome import score_episode


TRACE = (Path(__file__).resolve().parents[1]
         / 'assets/sm4096_h32_ori12_iter1000_seed161803_episode0_physical.npz')


def _load():
    with np.load(TRACE) as trace:
        return {key: trace[key].copy() for key in trace.files}


def _score(trace, steps=None, floor=None):
    n = steps or len(trace['state'])
    return score_episode(trace['state'][:n], trace['left_hand_contact'][:n],
                         trace['right_hand_contact'][:n],
                         trace['floor_contact'][:n] if floor is None else floor[:n],
                         trace['target_pos_w'], trace['target_quat_wxyz'],
                         bool(trace['episode_complete']))


def test_real_lift_transport_release_succeeds_without_continuous_grip():
    record = _score(_load())
    assert record['terminal_box_near_goal']
    assert record['target_distance_3d_m'] <= 0.12
    assert record['object_outcome'] and record['intermittent_carry']
    assert record['airborne_transport_m'] > 1.4
    assert record['both_hands_airborne_duty'] < 1.0


def test_impossible_floor_flags_do_not_erase_visible_airborne_carry():
    trace = _load()
    record = _score(trace, floor=np.ones_like(trace['floor_contact'], dtype=bool))
    assert record['object_outcome']
    assert record['intermittent_carry']
    assert record['impossible_floor_contact_frames'] > 0


def test_floor_drag_cannot_count_as_airborne_carry():
    trace = _load()
    trace['state'][:, 73] = trace['state'][0, 73]
    record = _score(trace, floor=np.ones_like(trace['floor_contact'], dtype=bool))
    assert not record['airborne_transport_50cm']
    assert not record['object_outcome']
    assert record['impossible_floor_contact_frames'] == 0


def test_real_raised_box_with_spurious_floor_contact_is_successful():
    record = _score(_load())
    assert record['impossible_floor_contact_frames'] >= 200
    assert record['airborne_transport_m'] > 1.4
    assert record['object_outcome'] and record['intermittent_carry']
    assert not record['strict_success_after_floor_guard']


def test_short_episode_returns_failure_diagnostic_instead_of_crashing():
    trace = _load()
    record = _score(trace, steps=21)
    assert not record['object_outcome']
    assert not record['intermittent_carry']


def test_terminal_goal_is_a_position_only_binary_outcome():
    trace = _load()
    trace['state'][-1, 71:74] = trace['target_pos_w']
    trace['left_hand_contact'][:] = False
    trace['right_hand_contact'][:] = False
    trace['episode_complete'] = np.array(False)
    score = _score(trace)
    assert score['terminal_box_near_goal']
    assert score['target_distance_3d_m'] == 0.0
    assert not score['intermittent_carry']
    assert not score['strict_success_after_floor_guard']
    assert not score['completed']

    trace['state'][-1, 71:74] = trace['target_pos_w'] + [0.08, 0.08, 0.08]
    score = _score(trace)
    assert not score['terminal_box_near_goal']
    assert score['target_distance_3d_m'] > 0.12
