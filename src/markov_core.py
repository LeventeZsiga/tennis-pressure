from __future__ import annotations
from functools import lru_cache
from math import comb
import numpy as np


def prob_p1_wins_game_from_points(p_point_win_p1: float, pts1: int, pts2: int) -> float:
    p = float(p_point_win_p1)
    if not (0.0 <= p <= 1.0):
        raise ValueError("p_point_win_p1 must be in [0,1]") 
    if pts1 < 0 or pts2 < 0:
        raise ValueError("Points must be non-negative.")
    
    if(pts1 >= 4 or pts2 >= 4) and abs(pts1 - pts2) >= 2:
        return 1.0 if pts1 > pts2 else 0.0
    
    q = 1.0 - p

    if pts1 >= 3 and pts2 >= 3:
        if pts1 == pts2:
            return (p * p) / (p * p + q * q)
        if pts1 == pts2 + 1:
            deuce = (p * p) / (p * p + q * q)
            return p * 1.0 + q * deuce
        if pts2 == pts1 + 1:
            deuce = (p * p) / (p * p + q * q)
            return p * deuce + q * 0.0
        
        return 1.0 if pts1 > pts2 else 0.0
    
    @lru_cache(maxsize=None)
    def F(a: int, b: int) -> float:
        if (a >= 4 or b >= 4) and abs(a - b) >= 2:
            return 1.0 if a > b else 0.0
        
        if a >= 3 and b >= 3:
            return prob_p1_wins_game_from_points(p, a, b)
        return p * F(a + 1, b) + (1.0 - p) * F(a, b + 1)
    return F(int(pts1), int(pts2))

        

def hold_prob_from_point_prob(p: float) -> float:
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    q = 1.0 - p

    win_pre = 0.0
    for x in range(0, 3):
        win_pre += comb(3 + x, x) * (p ** 4) * (q ** x)

    reach_deuce = comb(6, 3) * (p ** 3) * (q ** 3)

    win_from_deuce = (p * p) / (p * p + q * q)

    return win_pre + reach_deuce * win_from_deuce




def tb_server(start_server: int, i: int) -> int:
    if i == 0:
        return start_server
    block = (i - 1) // 2
    return (3 - start_server) if (block % 2 == 0) else start_server


def p1_point_win_prob(p_srv1: float, p_srv2: float, server: int) -> float:
    return p_srv1 if server == 1 else (1.0 - p_srv2)


def tb_win_prob_deuce_region(p_srv1: float, p_srv2: float, start_server: int, i_mod4_start: int) -> float:
    diffs = [-1, 0, 1]
    phases = [0, 1, 2, 3]
    idx = {(d, ph): k for k, (d, ph) in enumerate((d, ph) for d in diffs for ph in phases)}
    n = len(idx)  # 12

    A = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)

    for (d, ph), k in idx.items():
        A[k, k] = 1.0

        
        i_example = 12 + ph
        srv = tb_server(start_server, i_example)
        p = p1_point_win_prob(p_srv1, p_srv2, srv)

        d_w = d + 1
        ph_n = (ph + 1) % 4
        if d_w == 2:
            b[k] += p * 1.0
        else:
            A[k, idx[(d_w, ph_n)]] -= p

        d_l = d - 1
        if d_l == -2:
            pass
        else:
            A[k, idx[(d_l, ph_n)]] -= (1.0 - p)

    V = np.linalg.solve(A, b)
    return float(V[idx[(0, i_mod4_start)]])


@lru_cache(maxsize=None)
def tb_win_prob(p_srv1: float, p_srv2: float, start_server: int) -> float:
    @lru_cache(maxsize=None)
    def F(a: int, b: int) -> float:
        if (a >= 7 or b >= 7) and abs(a - b) >= 2:
            return 1.0 if a > b else 0.0

        if a >= 6 and b >= 6:
            i_mod4 = (a + b) % 4
            diff = a - b
            if diff == 0:
                return tb_win_prob_deuce_region(p_srv1, p_srv2, start_server, i_mod4)
            if diff == 1:
                i_example = 12 + i_mod4
                srv = tb_server(start_server, i_example)
                p = p1_point_win_prob(p_srv1, p_srv2, srv)
                return p * 1.0 + (1.0 - p) * tb_win_prob_deuce_region(p_srv1, p_srv2, start_server, (i_mod4 + 1) % 4)
            if diff == -1:
                i_example = 12 + i_mod4
                srv = tb_server(start_server, i_example)
                p = p1_point_win_prob(p_srv1, p_srv2, srv)
                return p * tb_win_prob_deuce_region(p_srv1, p_srv2, start_server, (i_mod4 + 1) % 4) + (1.0 - p) * 0.0

        i = a + b
        srv = tb_server(start_server, i)
        p = p1_point_win_prob(p_srv1, p_srv2, srv)
        return p * F(a + 1, b) + (1.0 - p) * F(a, b + 1)

    return F(0, 0)

def tb_win_prob_from_state(p_srv1: float, p_srv2: float, start_server: int, a0: int, b0: int) -> float:
    a0 = int(a0)
    b0 = int(b0)
    if a0 < 0 or b0 < 0:
        raise ValueError("Tiebreak points must be non-negative.")
    if start_server not in (1, 2):
        raise ValueError("start_server must be 1 or 2.")

    @lru_cache(maxsize=None)
    def F(a: int, b: int) -> float:
        if (a >= 7 or b >= 7) and abs(a - b) >= 2:
            return 1.0 if a > b else 0.0

        if a >= 6 and b >= 6:
            i_mod4 = (a + b) % 4
            diff = a - b

            if diff == 0:
                return tb_win_prob_deuce_region(p_srv1, p_srv2, start_server, i_mod4)

            if diff == 1:
                i_example = 12 + i_mod4
                srv = tb_server(start_server, i_example)
                p = p1_point_win_prob(p_srv1, p_srv2, srv)
                return p * 1.0 + (1.0 - p) * tb_win_prob_deuce_region(
                    p_srv1, p_srv2, start_server, (i_mod4 + 1) % 4
                )

            if diff == -1:
                i_example = 12 + i_mod4
                srv = tb_server(start_server, i_example)
                p = p1_point_win_prob(p_srv1, p_srv2, srv)
                return p * tb_win_prob_deuce_region(
                    p_srv1, p_srv2, start_server, (i_mod4 + 1) % 4
                ) + (1.0 - p) * 0.0

        i = a + b
        srv = tb_server(start_server, i)
        p = p1_point_win_prob(p_srv1, p_srv2, srv)
        return p * F(a + 1, b) + (1.0 - p) * F(a, b + 1)

    return F(a0, b0)

def set_outcome_distribution(p_srv1: float, p_srv2: float, first_server: int):
    hold1 = hold_prob_from_point_prob(p_srv1)
    hold2 = hold_prob_from_point_prob(p_srv2)

    def p1_wins_game(game_server: int) -> float:
        return hold1 if game_server == 1 else (1.0 - hold2)

    @lru_cache(maxsize=None)
    def S(g1: int, g2: int, server: int):
        if (g1 >= 6 or g2 >= 6) and abs(g1 - g2) >= 2:
            winner = 1 if g1 > g2 else 2
            return {(winner, server): 1.0}

        if g1 == 6 and g2 == 6:
            p_tb = tb_win_prob(p_srv1, p_srv2, start_server=server)
            next_server = 3 - server
            return {(1, next_server): p_tb, (2, next_server): 1.0 - p_tb}

        p = p1_wins_game(server)
        dist = {}

        for k, v in S(g1 + 1, g2, 3 - server).items():
            dist[k] = dist.get(k, 0.0) + p * v

        for k, v in S(g1, g2 + 1, 3 - server).items():
            dist[k] = dist.get(k, 0.0) + (1.0 - p) * v

        return dist

    return S(0, 0, first_server)


def match_win_prob_best_of_3(p_srv1: float, p_srv2: float, first_server: int | None = None) -> float:
    if not (0.0 <= p_srv1 <= 1.0 and 0.0 <= p_srv2 <= 1.0):
        raise ValueError("Probabilities must be in [0,1].")
    if first_server is not None and first_server not in (1, 2):
        raise ValueError("first_server must be None, 1, or 2")

    @lru_cache(maxsize=None)
    def M(s1: int, s2: int, next_set_first_server: int) -> float:
        if s1 == 2:
            return 1.0
        if s2 == 2:
            return 0.0

        dist = set_outcome_distribution(p_srv1, p_srv2, next_set_first_server)
        out = 0.0
        for (winner, next_server), prob in dist.items():
            if winner == 1:
                out += prob * M(s1 + 1, s2, next_server)
            else:
                out += prob * M(s1, s2 + 1, next_server)
        return out

    def solve(fs: int) -> float:
        return M(0, 0, fs)

    if first_server is None:
        return 0.5 * solve(1) + 0.5 * solve(2)
    return solve(first_server)


