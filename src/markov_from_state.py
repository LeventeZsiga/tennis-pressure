from __future__ import annotations
from functools import lru_cache

from .model_state import (
    ModelState,
    _p1_point_win_prob_from_state,
    _tb_start_server_from_current,
    _set_is_over,
)

from .markov_core import (
    hold_prob_from_point_prob,
    tb_win_prob,
    tb_win_prob_from_state,
    prob_p1_wins_game_from_points,
)

def win_prob_from_state(state: ModelState, p_srv1: float, p_srv2: float) -> float:

    if state.server not in (1, 2):
        raise ValueError("Invalid server value: {}".format(state.server))
    if state.sets1 < 0 or state.sets2 < 0 or state.sets1 > 2 or state.sets2 > 2:
        raise ValueError("Invalid sets values: sets1={}, sets2={}".format(state.sets1, state.sets2))
    if not (0.0 <= p_srv1 <= 1.0 and 0.0 <= p_srv2 <= 1.0):
        raise ValueError("Invalid point win probabilities: p_srv1={}, p_srv2={}".format(p_srv1, p_srv2))
    
    hold1 = hold_prob_from_point_prob(p_srv1)
    hold2 = hold_prob_from_point_prob(p_srv2)

    def p1_wins_game_when_server_is(server: int) -> float:
       return hold1 if server == 1 else (1.0 - hold2)
    
    @lru_cache(maxsize=None)
    def set_outcome_dist_from_games(g1: int, g2: int, next_server: int) -> dict:
        if(g1 >= 6 or g2 >= 6) and abs(g1 - g2) >= 2:
            winner = 1 if g1 > g2 else 2
            return {(winner, next_server): 1.0}
        
        if g1 == 6 and g2 == 6:
            p_tb = tb_win_prob(p_srv1, p_srv2, next_server)
            ns = 3 - next_server
            return {(1, ns): p_tb, (2, ns): 1.0 - p_tb}
        
        p = p1_wins_game_when_server_is(next_server)
        dist = {}

        for k, v in set_outcome_dist_from_games(g1 + 1, g2, 3 - next_server).items():
            dist[k] = dist.get(k, 0.0) + p * v

        for k, v in set_outcome_dist_from_games(g1, g2 + 1, 3 - next_server).items():
            dist[k] = dist.get(k, 0.0) + (1.0 - p) * v

        return dist
    
    @lru_cache(maxsize=None)
    def match_win_from_set_state(s1: int, s2: int, g1: int, g2: int, next_server: int) -> float:

        if s1 >= 2:
            return 1.0
        if s2 >= 2:
            return 0.0

        if _set_is_over(g1, g2):
            if g1 > g2:
                return match_win_from_set_state(s1 + 1, s2, 0, 0, next_server)
            else:
                return match_win_from_set_state(s1, s2 + 1, 0, 0, next_server)

        dist = set_outcome_dist_from_games(g1, g2, next_server)
        out = 0.0
        for (winner, next_set_first_server), prob in dist.items():
            if winner == 1:
                out += prob * match_win_from_set_state(s1 + 1, s2, 0, 0, next_set_first_server)
            else:
                out += prob * match_win_from_set_state(s1, s2 + 1, 0, 0, next_set_first_server)

        return out
    
    if state.in_tb:
        i = state.tb1 + state.tb2
        tb_start = _tb_start_server_from_current(state.server, i)
        p_tb_win = tb_win_prob_from_state(p_srv1, p_srv2, tb_start, state.tb1, state.tb2)
        next_set_first_server = 3 - state.server

        win_path = match_win_from_set_state(state.sets1 + 1, state.sets2, 0, 0, next_set_first_server)
        lose_path = match_win_from_set_state(state.sets1, state.sets2 + 1, 0, 0, next_set_first_server)

        return p_tb_win * win_path + (1.0 - p_tb_win) * lose_path
    
    p_point = _p1_point_win_prob_from_state(p_srv1, p_srv2, state.server)
    p_game_win = prob_p1_wins_game_from_points(p_point, state.pts1, state.pts2)
    
    g1_win, g2_win = state.games1 + 1, state.games2
    next_server_after_game = 3 - state.server

    if _set_is_over(g1_win, g2_win):
        if g1_win > g2_win:
            win_state_prob = match_win_from_set_state(state.sets1 + 1, state.sets2, 0, 0, next_server_after_game)
        else:
            win_state_prob = match_win_from_set_state(state.sets1, state.sets2 + 1, 0, 0, next_server_after_game)
    else:
        win_state_prob = match_win_from_set_state(state.sets1, state.sets2, g1_win, g2_win, next_server_after_game)

    g1_lose, g2_lose = state.games1, state.games2 + 1

    if _set_is_over(g1_lose, g2_lose):
        if g1_lose > g2_lose:
            lose_state_prob = match_win_from_set_state(state.sets1 + 1, state.sets2, 0, 0, next_server_after_game)
        else:
            lose_state_prob = match_win_from_set_state(state.sets1, state.sets2 + 1, 0, 0, next_server_after_game)
    else:
        lose_state_prob = match_win_from_set_state(state.sets1, state.sets2, g1_lose, g2_lose, next_server_after_game)

    return p_game_win * win_state_prob + (1.0 - p_game_win) * lose_state_prob