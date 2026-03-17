from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache

@dataclass(frozen=True)
class ModelState:

    sets1: int
    sets2: int
    games1: int
    games2: int
    pts1: int
    pts2: int
    server: int
    in_tb: bool = False
    tb1: int = 0
    tb2: int = 0

def _p1_point_win_prob_from_state(p_srv1: float, p_srv2: float, server: int) -> float:
    if server == 1:
        return p_srv1
    elif server == 2:
        return 1.0 - p_srv2
    else:
        raise ValueError(f"Invalid server: {server}")

def _tb_start_server_from_current(server_now: int, points_played: int) -> int:
    if points_played == 0:
        return server_now
    block = (points_played - 1) // 2
    
    if block % 2 == 0:
        return 3 - server_now
    else:        
        return server_now
    
def _set_is_over(g1: int, g2: int) -> bool:
    if g1 >= 6 or g2 >= 6:
        if abs(g1 - g2) >= 2 and (g1 == 6 or g2 == 6 or g1 == 7 or g2 == 7):
            return True
        if (g1 == 7 and g2 == 5) or (g1 == 5 and g2 == 7):
            return True
        if (g1 == 7 and g2 == 6) or (g1 == 6 and g2 == 7):
            return True
    return False