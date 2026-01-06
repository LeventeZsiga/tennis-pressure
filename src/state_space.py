from __future__ import annotations
from typing import NamedTuple

class State(NamedTuple):
    p1_sets: int
    p2_sets: int
    set_no: int
    p1_games: int
    p2_games: int
    p1_points: int 
    p2_points: int
    is_p1_server: int
    tiebreak: int

def row_to_state(row) -> State:
    return State(
        p1_sets=int(row["P1SetsWon_upto_before"]),
        p2_sets=int(row["P2SetsWon_upto_before"]),
        set_no=int(row["SetNo"]),
        p1_games=int(row["P1GamesWon_before"]),
        p2_games=int(row["P2GamesWon_before"]),
        p1_points=int(row["p1_pts_game_before"]),
        p2_points=int(row["p2_pts_game_before"]),
        is_p1_server=int(row["is_p1_server"]),
        tiebreak=int(row["tiebreak"]),
    )

def is_terminal(row) -> bool:
    return bool(row["is_match_end"])