
import math
import pytest


from src.markov_core import prob_p1_wins_game_from_points

TOL = 1e-12


def approx(a, b, tol=TOL):
    return abs(a - b) < tol


def test_extreme_probabilities():
    assert approx(prob_p1_wins_game_from_points(1.0, 0, 0), 1.0)
    assert approx(prob_p1_wins_game_from_points(1.0, 3, 3), 1.0)  
    assert approx(prob_p1_wins_game_from_points(1.0, 0, 3), 1.0)

    assert approx(prob_p1_wins_game_from_points(0.0, 0, 0), 0.0)
    assert approx(prob_p1_wins_game_from_points(0.0, 3, 3), 0.0)
    assert approx(prob_p1_wins_game_from_points(0.0, 3, 0), 0.0)


def test_terminal_states():
    assert approx(prob_p1_wins_game_from_points(0.37, 4, 0), 1.0)
    assert approx(prob_p1_wins_game_from_points(0.37, 0, 4), 0.0)
    assert approx(prob_p1_wins_game_from_points(0.37, 5, 3), 1.0)
    assert approx(prob_p1_wins_game_from_points(0.37, 3, 5), 0.0)


def test_deuce_closed_form():
    for p in [0.2, 0.35, 0.5, 0.62, 0.8]:
        q = 1.0 - p
        expected = (p * p) / (p * p + q * q)
        got = prob_p1_wins_game_from_points(p, 3, 3)
        assert abs(got - expected) < 1e-12


def test_advantage_relations():
    for p in [0.25, 0.5, 0.7]:
        q = 1.0 - p
        deuce = prob_p1_wins_game_from_points(p, 3, 3)

        adv_p1 = prob_p1_wins_game_from_points(p, 4, 3)
        expected_adv_p1 = p * 1.0 + q * deuce
        assert abs(adv_p1 - expected_adv_p1) < 1e-12

        adv_p2 = prob_p1_wins_game_from_points(p, 3, 4)
        expected_adv_p2 = p * deuce + q * 0.0
        assert abs(adv_p2 - expected_adv_p2) < 1e-12


def test_monotonicity_in_p():
    states = [(0, 0), (1, 0), (0, 1), (3, 2), (2, 3), (3, 3), (4, 3), (3, 4)]
    ps = [0.1, 0.3, 0.5, 0.7, 0.9]

    for a, b in states:
        prev = None
        for p in ps:
            val = prob_p1_wins_game_from_points(p, a, b)
            if prev is not None:
                assert val >= prev - 1e-12, f"Monotonicity failed at state {(a,b)}"
            prev = val


def test_bellman_consistency_local():
    def is_terminal(a, b):
        return (a >= 4 or b >= 4) and abs(a - b) >= 2

    for p in [0.3, 0.5, 0.7]:
        for a in range(0, 6):
            for b in range(0, 6):
                if is_terminal(a, b):
                    continue
                W = prob_p1_wins_game_from_points
                lhs = W(p, a, b)
                rhs = p * W(p, a + 1, b) + (1.0 - p) * W(p, a, b + 1)
                assert abs(lhs - rhs) < 1e-10, (p, a, b, lhs, rhs)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        prob_p1_wins_game_from_points(-0.1, 0, 0)
    with pytest.raises(ValueError):
        prob_p1_wins_game_from_points(1.1, 0, 0)
    with pytest.raises(ValueError):
        prob_p1_wins_game_from_points(0.5, -1, 0)
    with pytest.raises(ValueError):
        prob_p1_wins_game_from_points(0.5, 0, -2)
