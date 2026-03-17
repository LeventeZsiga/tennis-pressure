from __future__ import annotations
from pathlib import Path
from dataclasses import replace
import pandas as pd

from src.model_state import ModelState
from src.markov_from_state import win_prob_from_state


ROOT = Path.cwd()
DATA_PROCESSED = ROOT / "data_processed"
DATA_PATH = DATA_PROCESSED / "05_final_filtered_adv_matches.parquet"

N_POINTS = 250

def game_over(a: int, b: int) -> bool:
    return (a >= 4 or b >= 4) and abs(a - b) >= 2

def tiebreak_over(a: int, b: int) -> bool:
    return (a >= 7 or b >= 7) and abs(a - b) >= 2

def set_over(g1: int, g2: int) -> bool:
    if (g1 >= 6 or g2 >= 6) and abs(g1 - g2) >= 2:
        return True
    if (g1, g2) in ((7, 6), (6, 7)):
        return True
    return False

def tb_server(tb_start_server: int, tb_point_index: int) -> int:
    if tb_point_index == 0:
        return tb_start_server
    block = (tb_point_index - 1) // 2
    return (3 - tb_start_server) if (block % 2 == 0) else tb_start_server

def tb_start_server_from_current(server_now: int, points_played: int) -> int:
    if points_played == 0:
        return server_now
    block = (points_played - 1) // 2
    if block % 2 == 0:
        return 3 - server_now
    return server_now


def extract_state(row: pd.Series) -> ModelState:
    in_tb = bool(row["tiebreak"])
    pw = int(row["PointWinner"])  # 1 or 2

    sets1 = int(row["P1SetsWon_upto"])
    sets2 = int(row["P2SetsWon_upto"])
    if bool(row["is_set_end"]):
        if pw == 1:
            sets1 -= 1
        elif pw == 2:
            sets2 -= 1

    games1 = int(row["P1GamesWon"])
    games2 = int(row["P2GamesWon"])
    if bool(row["is_game_end"]):
        if pw == 1:
            games1 -= 1
        elif pw == 2:
            games2 -= 1

    pts1 = int(row["p1_pts_game_before"])
    pts2 = int(row["p2_pts_game_before"])

    tb1 = int(row["p1_tb_points"]) if in_tb else 0
    tb2 = int(row["p2_tb_points"]) if in_tb else 0
    if in_tb:
        if pw == 1:
            tb1 -= 1
        elif pw == 2:
            tb2 -= 1
        tb1 = max(tb1, 0)
        tb2 = max(tb2, 0)

    return ModelState(
        sets1=sets1,
        sets2=sets2,
        games1=games1,
        games2=games2,
        pts1=pts1,
        pts2=pts2,
        server=int(row["server"]),
        in_tb=in_tb,
        tb1=tb1,
        tb2=tb2,
    )


def fmt_state(s: ModelState) -> str:
    if s.in_tb:
        return (f"sets={s.sets1}-{s.sets2} "
                f"games={s.games1}-{s.games2} "
                f"TB={s.tb1}-{s.tb2} "
                f"server={'P1' if s.server==1 else 'P2'}")
    return (f"sets={s.sets1}-{s.sets2} "
            f"games={s.games1}-{s.games2} "
            f"pts={s.pts1}-{s.pts2} "
            f"server={'P1' if s.server==1 else 'P2'}")


def apply_point(state: ModelState, p1_wins_point: bool) -> ModelState:
    if state.sets1 == 2 or state.sets2 == 2:
        raise ValueError("Match is already over")
    
    if state.in_tb:
        tb1, tb2 = state.tb1, state.tb2
        if p1_wins_point:
            tb1 += 1
        else:
            tb2 += 1

        i_before = state.tb1 + state.tb2
        tb_start = tb_start_server_from_current(state.server, i_before)

        if tiebreak_over(tb1, tb2):
            if tb1 > tb2:
                sets1, sets2 = state.sets1 + 1, state.sets2
            else:
                sets1, sets2 = state.sets1, state.sets2 + 1
            
            last_point_server = tb_server(tb_start, i_before)
            next_server = 3 - last_point_server

            return ModelState(
                sets1=sets1,
                sets2=sets2,
                games1=0,
                games2=0,
                pts1=0,
                pts2=0,
                server=next_server,
                in_tb=False,
                tb1=0,
                tb2=0,
            )
        i_after = i_before + 1
        next_server = tb_server(tb_start, i_after)
        return replace(state, tb1=tb1, tb2=tb2, server=next_server)
    
    pts1, pts2 = state.pts1, state.pts2
    if p1_wins_point:
        pts1 += 1
    else:
        pts2 += 1

    if not game_over(pts1, pts2):
        return replace(state, pts1=pts1, pts2=pts2)
    
    g1, g2 = state.games1, state.games2
    if pts1 > pts2:
        g1 += 1
    else:
        g2 += 1

    next_server = 3 - state.server

    if set_over(g1, g2):
        if g1 > g2:
            return ModelState(state.sets1 + 1, state.sets2, 0, 0, 0, 0, next_server, False, 0, 0)
        else:
            return ModelState(state.sets1, state.sets2 + 1, 0, 0, 0, 0, next_server, False, 0, 0)
        
    if g1 == 6 and g2 == 6:
        return ModelState(state.sets1, state.sets2, g1, g2, 0, 0, next_server, True, 0, 0)
    
    return ModelState(state.sets1, state.sets2, g1, g2, 0, 0, next_server, False, 0, 0)
        
def compute_match_serve_probs(m):
    p1_serves = m["is_p1_server"] == 1
    p2_serves = ~p1_serves

    valid = m["PointServer"].isin([1, 2])
    m_valid = m[valid]

    p1_serves = m_valid["is_p1_server"] == 1
    p2_serves = ~p1_serves

    p_srv1 = m_valid.loc[p1_serves, "server_won_point"].mean()
    p_srv2 = m_valid.loc[p2_serves, "server_won_point"].mean()

    return float(p_srv1), float(p_srv2)


def main():

    df = pd.read_parquet(DATA_PATH)

    match_id = "2024-wimbledon-2602"
    m = df[df["match_id"] == match_id].copy()
    m = m[m["PointServer"].isin([1, 2])].copy()
    m = m[(m["P1SetsWon_upto"] < 2) & (m["P2SetsWon_upto"] < 2)].copy() 

    m = m.sort_values(["SetNo", "GameNo", "PointNumber"]).reset_index(drop=True)

    p_srv1, p_srv2 = compute_match_serve_probs(m)

    m_small = m.head(N_POINTS)

    print("=" * 96)
    print(f"Match {match_id} | p_srv1={p_srv1:.3f}  p_srv2={p_srv2:.3f} | showing first {len(m_small)} points")
    print("=" * 96)    

    for i, row in enumerate(m_small.itertuples(index=False),start=1):
        row = row._asdict()
        s = extract_state(pd.Series(row))

        W_before = win_prob_from_state(s, p_srv1, p_srv2)

        s_win = apply_point(s, True)
        s_lose = apply_point(s, False)

        W_win = win_prob_from_state(s_win, p_srv1, p_srv2)
        W_lose = win_prob_from_state(s_lose, p_srv1, p_srv2)

        pressure = W_win - W_lose
        if W_win + 1e-12 < W_lose:
            raise ValueError(
                f"W_win < W_lose at match {match_id}, "
                f"Set {r['SetNo']} Game {r['GameNo']} Point {r['PointNumber']} "
                f"state={s} W_win={W_win:.6f} W_lose={W_lose:.6f}"
            )
        
        print(f"\nPoint {i} | SetNo={row['SetNo']} GameNo={row['GameNo']} PointNumber={row['PointNumber']}")
        print(f"  state:   {fmt_state(s)}")
        print(f"  W_before: {W_before:.6f}")
        print(f"  if P1 wins  -> {fmt_state(s_win)} | W_win={W_win:.6f}")
        print(f"  if P1 loses -> {fmt_state(s_lose)} | W_lose={W_lose:.6f}")
        print(f"  pressure (W_win - W_lose): {pressure:.6f}")


if __name__ == "__main__":
    main()