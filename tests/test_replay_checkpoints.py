from __future__ import annotations

import pandas as pd

from src.state_space import row_to_state
from src.transitions import next_state
from src.state_space import State

DATA_PATH = "data_processed/04_final_filtered_dataset.parquet"

def load_df() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values(
        ["match_id", "SetNo", "GameNo", "PointNumber"]
    ).reset_index(drop = True)
    return df

def replay_match_and_check_scorekeeping(df: pd.DataFrame, match_id: str) -> tuple[int, int]:
    m = df[df["match_id"] == match_id].copy()
    if len(m) == 0:
        return(0, 0)
    
    first = m.iloc[0]
    s = State(
        p1_sets=0,
        p2_sets=0,
        set_no=1,
        p1_games=0,
        p2_games=0,
        p1_points=0,
        p2_points=0,
        is_p1_server=int(first["is_p1_server"]),
        tiebreak=0,
    )


    n_checks = 0
    n_bad = 0

    for i, row in m.iterrows():
        pw = row["PointWinner"]
        if pd.isna(pw) or int(pw) not in (1, 2):
            continue
        s = next_state(s, int(pw))

        if row["is_game_end"] and not row["is_set_end"]:
            n_checks += 1
            if (
                s.p1_games != int(row["P1GamesWon"])
                or s.p2_games != int(row["P2GamesWon"])
            ):
                n_bad += 1
                print("\nGame mismatch in match:", match_id, "at index", i)
                print("Predicted games:", s.p1_games, s.p2_games)
                print("Actual games:   ", row["P1GamesWon"], row["P2GamesWon"])

        if row["is_set_end"]:
            n_checks += 1
            if(
                s.p1_sets != int(row["P1SetsWon_upto"])
                or s.p2_sets != int(row["P2SetsWon_upto"])
            ):
                n_bad += 1
                print("\nSet mismatch in match:", match_id, "at index", i)
                print("Predicted sets:", s.p1_sets, s.p2_sets)
                print("Actual sets:   ", row["P1SetsWon_upto"], row["P2SetsWon_upto"])

        if row["is_match_end"]:
            n_checks += 1
            if not (s.p1_sets == 2 or s.p2_sets == 2):
                n_bad += 1
                print("\nMatch end mismatch in match:", match_id)
                print("Final predicted sets:", s.p1_sets, s.p2_sets)

    return n_checks, n_bad


def main():
    df = load_df()
    match_ids = df["match_id"].drop_duplicates().head(10).tolist()

    total_checks = 0
    total_bad = 0

    for mid in match_ids:
        n_checks, n_bad = replay_match_and_check_scorekeeping(df, mid)
        total_checks += n_checks
        total_bad += n_bad

    print("\n=== Scorekeeping replay test ===")
    print("Matches tested:", len(match_ids))
    print("Checkpoints checked:", total_checks)
    print("Mismatches:", total_bad)

    if total_bad > 0:
        raise SystemExit("FAIL: scorekeeping mismatches found")
    print("PASS: scorekeeping matches dataset")

if __name__ == "__main__":
    main()