from __future__ import annotations
from typing import Optional

def game_winner(p1_pts: int, p2_pts: int) -> Optional[int]:
    if p1_pts >= 4 and (p1_pts - p2_pts) >= 2:
        return 1
    if p2_pts >= 4 and (p2_pts - p1_pts) >= 2:
        return 2
    return None

def is_tiebreak_start(p1_games: int, p2_games: int) -> bool:
    return p1_games == 6 and p2_games == 6

def set_winner_non_tb(p1_games: int, p2_games: int) -> Optional[int]:
    if p1_games >= 6 and (p1_games - p2_games) >= 2:
        return 1
    if p2_games >= 6 and (p2_games - p1_games) >= 2:
        return 2
    return None

def tiebreak_winner(p1_tb: int, p2_tb: int) -> Optional[int]:
    if p1_tb >= 7 and (p1_tb - p2_tb) >= 2:
        return 1
    if p2_tb >= 7 and (p2_tb - p1_tb) >= 2:
        return 2
    return None