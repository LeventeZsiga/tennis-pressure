# tests/test_win_prob_from_state.py

import math
import pytest

from src.markov_from_state import win_prob_from_state
from src.model_state import ModelState
from src.markov_core import match_win_prob_best_of_3


TOL = 1e-8


def approx(a, b, tol=TOL):
    return abs(a - b) < tol


# -------------------------------------------------
# 1) Start of match should equal Phase 1 model
# -------------------------------------------------

def test_start_of_match_matches_phase1():
    p1, p2 = 0.62, 0.60

    state = ModelState(
        sets1=0, sets2=0,
        games1=0, games2=0,
        pts1=0, pts2=0,
        server=1,
        in_tb=False
    )

    w_state = win_prob_from_state(state, p1, p2)
    w_phase1 = match_win_prob_best_of_3(p1, p2, first_server=1)

    assert approx(w_state, w_phase1)


# -------------------------------------------------
# 2) Match already finished
# -------------------------------------------------

def test_match_already_won():
    state = ModelState(
        sets1=2, sets2=0,
        games1=0, games2=0,
        pts1=0, pts2=0,
        server=1,
        in_tb=False
    )
    assert win_prob_from_state(state, 0.6, 0.6) == 1.0


def test_match_already_lost():
    state = ModelState(
        sets1=0, sets2=2,
        games1=0, games2=0,
        pts1=0, pts2=0,
        server=1,
        in_tb=False
    )
    assert win_prob_from_state(state, 0.6, 0.6) == 0.0


# -------------------------------------------------
# 3) Symmetry check (identical serve strength)
# -------------------------------------------------

def test_symmetry_equal_players():
    state = ModelState(
        sets1=0, sets2=0,
        games1=0, games2=0,
        pts1=0, pts2=0,
        server=1,
        in_tb=False
    )

    w = win_prob_from_state(state, 0.6, 0.6)
    assert approx(w, 0.5)


# -------------------------------------------------
# 4) Near-certain win / loss states
# -------------------------------------------------

def test_almost_certain_win():
    # P1 up 1 set, 5-0, 40-0 on serve
    state = ModelState(
        sets1=1, sets2=0,
        games1=5, games2=0,
        pts1=3, pts2=0,
        server=1,
        in_tb=False
    )

    w = win_prob_from_state(state, 0.62, 0.60)
    assert w > 0.95


def test_almost_certain_loss():
    # P1 down 1 set, 0-5, 0-40 with opponent serving
    state = ModelState(
        sets1=0, sets2=1,
        games1=0, games2=5,
        pts1=0, pts2=3,
        server=2,
        in_tb=False
    )

    w = win_prob_from_state(state, 0.62, 0.60)
    assert w < 0.05


# -------------------------------------------------
# 5) Tiebreak sanity check
# -------------------------------------------------

def test_tiebreak_mid_state_sanity():
    # 6-6 in games, 3-3 in tiebreak, P1 serving
    state = ModelState(
        sets1=0, sets2=0,
        games1=6, games2=6,
        pts1=0, pts2=0,      # ignored in TB
        server=1,
        in_tb=True,
        tb1=3,
        tb2=3
    )

    w = win_prob_from_state(state, 0.62, 0.60)

    # should be between 0 and 1 and not explode
    assert 0.0 < w < 1.0