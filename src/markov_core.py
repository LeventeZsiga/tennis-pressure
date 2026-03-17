from __future__ import annotations
from functools import lru_cache
from math import comb
import numpy as np


def prob_p1_wins_game_from_points(p_point_win_p1: float, pts1: int, pts2: int) -> float:
    """P(P1 wins game) given p=P(P1 wins point on serve) and current score (pts1, pts2)."""
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

        


# 1) Point -> game (hold prob)

def hold_prob_from_point_prob(p: float) -> float:
    """P(server holds a standard advantage game) given p=P(win point on serve)."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    q = 1.0 - p

    # win before deuce (4-x with x<=2)
    win_pre = 0.0
    for x in range(0, 3):
        win_pre += comb(3 + x, x) * (p ** 4) * (q ** x)

    # reach deuce (3-3)
    reach_deuce = comb(6, 3) * (p ** 3) * (q ** 3)

    # from deuce, win prob = p^2 / (p^2 + q^2)
    win_from_deuce = (p * p) / (p * p + q * q)

    return win_pre + reach_deuce * win_from_deuce



# 2) Tiebreak serving pattern

def tb_server(start_server: int, i: int) -> int:
    """
    Standard TB serving:
      i=0: start_server
      i=1,2: other
      i=3,4: start
      i=5,6: other ...
    """
    if i == 0:
        return start_server
    block = (i - 1) // 2
    return (3 - start_server) if (block % 2 == 0) else start_server


def p1_point_win_prob(p_srv1: float, p_srv2: float, server: int) -> float:
    """P(P1 wins a point) given who serves."""
    return p_srv1 if server == 1 else (1.0 - p_srv2)


# ---------------------------------------------------------
# 3) Tiebreak win prob without infinite recursion
#    - exact DP until reaching 6-6 region
#    - from (>=6,>=6) solve finite 12-state linear system
# ---------------------------------------------------------

def tb_win_prob_deuce_region(p_srv1: float, p_srv2: float, start_server: int, i_mod4_start: int) -> float:
    """
    Compute P1 win probability from tiebreak 'deuce region' where both >=6.
    State is (diff, phase) where diff in {-1,0,1} and phase = i mod 4.
    Absorb when diff hits +2 (P1 wins) or -2 (P1 loses).
    """
    # Map transient states to indices
    diffs = [-1, 0, 1]
    phases = [0, 1, 2, 3]
    idx = {(d, ph): k for k, (d, ph) in enumerate((d, ph) for d in diffs for ph in phases)}
    n = len(idx)  # 12

    A = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)

    for (d, ph), k in idx.items():
        # equation: V = p*V(next_win) + (1-p)*V(next_lose)
        A[k, k] = 1.0

        # determine server for this point from phase (need actual i%4 only)
        # because tb_server depends on i, and from i>=1 pattern repeats with period 4
        # but safest is: just compute using any i with same mod4 that is >=12.
        i_example = 12 + ph  # 12 is 6-6, then add phase
        srv = tb_server(start_server, i_example)
        p = p1_point_win_prob(p_srv1, p_srv2, srv)

        # if P1 wins point -> diff+1, phase+1
        d_w = d + 1
        ph_n = (ph + 1) % 4
        if d_w == 2:
            # absorb win
            b[k] += p * 1.0
        else:
            A[k, idx[(d_w, ph_n)]] -= p

        # if P1 loses point -> diff-1
        d_l = d - 1
        if d_l == -2:
            # absorb lose -> contributes 0
            pass
        else:
            A[k, idx[(d_l, ph_n)]] -= (1.0 - p)

    V = np.linalg.solve(A, b)
    return float(V[idx[(0, i_mod4_start)]])


@lru_cache(maxsize=None)
def tb_win_prob(p_srv1: float, p_srv2: float, start_server: int) -> float:
    """
    Full P(P1 wins tiebreak) with win-by-2, using:
      - exact DP for scores where not both >=6
      - finite linear-system solution in the (>=6,>=6) region
    """
    @lru_cache(maxsize=None)
    def F(a: int, b: int) -> float:
        # terminal
        if (a >= 7 or b >= 7) and abs(a - b) >= 2:
            return 1.0 if a > b else 0.0

        # enter deuce-region solver when both >=6
        if a >= 6 and b >= 6:
            i_mod4 = (a + b) % 4
            diff = a - b
            if diff == 0:
                return tb_win_prob_deuce_region(p_srv1, p_srv2, start_server, i_mod4)
            if diff == 1:
                # from +1: next point win => win, lose => back to diff 0 with phase+1
                i_example = 12 + i_mod4
                srv = tb_server(start_server, i_example)
                p = p1_point_win_prob(p_srv1, p_srv2, srv)
                return p * 1.0 + (1.0 - p) * tb_win_prob_deuce_region(p_srv1, p_srv2, start_server, (i_mod4 + 1) % 4)
            if diff == -1:
                i_example = 12 + i_mod4
                srv = tb_server(start_server, i_example)
                p = p1_point_win_prob(p_srv1, p_srv2, srv)
                return p * tb_win_prob_deuce_region(p_srv1, p_srv2, start_server, (i_mod4 + 1) % 4) + (1.0 - p) * 0.0

        # normal step (this part cannot go deep; it reaches >=6 quickly)
        i = a + b
        srv = tb_server(start_server, i)
        p = p1_point_win_prob(p_srv1, p_srv2, srv)
        return p * F(a + 1, b) + (1.0 - p) * F(a, b + 1)

    return F(0, 0)

def tb_win_prob_from_state(p_srv1: float, p_srv2: float, start_server: int, a0: int, b0: int) -> float:
    """
    P(P1 wins tiebreak) from an intermediate tiebreak score (a0, b0),
    where a0=points won by P1 in TB, b0=points won by P2 in TB.

    Uses:
      - terminal condition if already decided
      - deuce-region linear system when both >=6
      - otherwise finite DP forward
    """
    a0 = int(a0)
    b0 = int(b0)
    if a0 < 0 or b0 < 0:
        raise ValueError("Tiebreak points must be non-negative.")
    if start_server not in (1, 2):
        raise ValueError("start_server must be 1 or 2.")

    @lru_cache(maxsize=None)
    def F(a: int, b: int) -> float:
        # terminal
        if (a >= 7 or b >= 7) and abs(a - b) >= 2:
            return 1.0 if a > b else 0.0

        # deuce region (both >= 6)
        if a >= 6 and b >= 6:
            i_mod4 = (a + b) % 4
            diff = a - b

            if diff == 0:
                return tb_win_prob_deuce_region(p_srv1, p_srv2, start_server, i_mod4)

            if diff == 1:
                # from +1: next win => win TB, next loss => back to diff 0 with phase+1
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

        # normal step (finite until reaching >=6,>=6)
        i = a + b
        srv = tb_server(start_server, i)
        p = p1_point_win_prob(p_srv1, p_srv2, srv)
        return p * F(a + 1, b) + (1.0 - p) * F(a, b + 1)

    return F(a0, b0)
# ---------------------------------------------------------
# 4) Set outcome distribution (winner + next server)
#    We need next_server for match-level exactness.
# ---------------------------------------------------------

def set_outcome_distribution(p_srv1: float, p_srv2: float, first_server: int):
    """
    Returns a dict:
      { (winner, next_server_after_set): probability }
    where winner is 1 for P1, 2 for P2.

    Exact game-level model:
      - hold probs computed from point probs
      - games alternate serve
      - TB at 6-6 (TB counts as a game for serve alternation)
    """
    hold1 = hold_prob_from_point_prob(p_srv1)  # P1 holds on serve
    hold2 = hold_prob_from_point_prob(p_srv2)  # P2 holds on serve

    def p1_wins_game(game_server: int) -> float:
        return hold1 if game_server == 1 else (1.0 - hold2)

    @lru_cache(maxsize=None)
    def S(g1: int, g2: int, server: int):
        # terminal by 2 games at >=6
        if (g1 >= 6 or g2 >= 6) and abs(g1 - g2) >= 2:
            # set ends after this last game already happened, so 'server' is the next server
            winner = 1 if g1 > g2 else 2
            return {(winner, server): 1.0}

        # tiebreak at 6-6
        if g1 == 6 and g2 == 6:
            # TB start server is 'server' (who is due next)
            p_tb = tb_win_prob(p_srv1, p_srv2, start_server=server)
            # After TB, server toggles once (TB treated as a game)
            next_server = 3 - server
            return {(1, next_server): p_tb, (2, next_server): 1.0 - p_tb}

        p = p1_wins_game(server)
        dist = {}

        # P1 wins game -> (g1+1, g2), next server toggles
        for k, v in S(g1 + 1, g2, 3 - server).items():
            dist[k] = dist.get(k, 0.0) + p * v

        # P1 loses game
        for k, v in S(g1, g2 + 1, 3 - server).items():
            dist[k] = dist.get(k, 0.0) + (1.0 - p) * v

        return dist

    return S(0, 0, first_server)


# -----------------------------
# 5) Match win prob best-of-3
# -----------------------------

def match_win_prob_best_of_3(p_srv1: float, p_srv2: float, first_server: int | None = None) -> float:
    """
    Standalone Phase 1 model:
      - input: point win prob on serve for each player
      - output: P1 match win probability (best-of-3)
      - exact propagation of server across sets using set_outcome_distribution
    """
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


