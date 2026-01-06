from __future__ import annotations

from .state_space import State 
from .state_rules import (
    game_winner,
    is_tiebreak_start,
    set_winner_non_tb,
    tiebreak_winner,
)

def next_state(s: State, point_winner: int) -> State:

    if point_winner not in (1, 2):
        raise ValueError(f"point_winner must be 1 or 2, got {point_winner}")


    p1_pts = s.p1_points + (1 if point_winner == 1 else 0)
    p2_pts = s.p2_points + (1 if point_winner == 2 else 0)

    if s.tiebreak == 1:
        tb_win = tiebreak_winner(p1_pts, p2_pts)
        if tb_win is None:
            return s._replace(p1_points = p1_pts, p2_points = p2_pts)
        
        if tb_win == 1:
            p1_sets = s.p1_sets + 1
            p2_sets = s.p2_sets

        else:
            p1_sets = s.p1_sets
            p2_sets = s.p2_sets + 1

        is_match_over = (p1_sets == 2) or (p2_sets == 2)
        next_set_no = s.set_no if is_match_over else (s.set_no + 1)

        return State(
            p1_sets = p1_sets,
            p2_sets = p2_sets,
            set_no = next_set_no,
            p1_games = 0,
            p2_games = 0,
            p1_points = 0,
            p2_points = 0,
            is_p1_server = 1 - s.is_p1_server,
            tiebreak = 0,
        )
    
    gwin = game_winner(p1_pts, p2_pts)
    if gwin is None:
        return s._replace(p1_points = p1_pts, p2_points = p2_pts)
    
    if gwin == 1:
        p1_games = s.p1_games + 1
        p2_games = s.p2_games
    else:
        p1_games = s.p1_games
        p2_games = s.p2_games + 1

    
    swin = set_winner_non_tb(p1_games, p2_games)
    if swin is not None:
        if swin == 1:
            p1_sets = s.p1_sets + 1
            p2_sets = s.p2_sets
        else:
            p1_sets = s.p1_sets
            p2_sets = s.p2_sets + 1

        is_match_over = (p1_sets == 2) or (p2_sets == 2)
        next_set_no = s.set_no if is_match_over else (s.set_no + 1)

        return State(
            p1_sets = p1_sets,
            p2_sets = p2_sets,
            set_no = next_set_no,
            p1_games = 0,
            p2_games = 0,
            p1_points = 0,
            p2_points = 0,
            is_p1_server = 1 - s.is_p1_server,
            tiebreak = 0,
        )
    tb = 1 if is_tiebreak_start(p1_games, p2_games) else 0

    return State(
        p1_sets = s.p1_sets,
        p2_sets = s.p2_sets,
        set_no = s.set_no,
        p1_games = p1_games,
        p2_games = p2_games,
        p1_points = 0,
        p2_points = 0,
        is_p1_server = 1 - s.is_p1_server,
        tiebreak = tb,
    )
    