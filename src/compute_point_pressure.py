import pandas as pd
from pathlib import Path

from src.manual_state_extraction import (
    compute_match_serve_probs,
    extract_state,
    apply_point,
)

from src.markov_from_state import win_prob_from_state

ROOT = Path.cwd()
DATA_PROCESSED = ROOT / "data_processed"
INPUT_PATH = DATA_PROCESSED / "05_final_filtered_adv_matches.parquet"
OUTPUT_PATH = DATA_PROCESSED / "06_point_pressure.parquet"

def compute_pressure_for_match(m: pd.DataFrame) -> pd.DataFrame:
    m = m.sort_values(["SetNo", "GameNo", "PointNumber"], kind="mergesort")
    m = m[m["PointServer"].isin([1, 2])].copy()

    p_srv1, p_srv2 = compute_match_serve_probs(m)

    win_probs = []
    pressure_raws = []
    pressure_swings = []

    for row in m.itertuples(index=False):
        row_series = pd.Series(row._asdict())

        state = extract_state(row_series)

        W_before = win_prob_from_state(state, p_srv1, p_srv2)

        s_win = apply_point(state, True)
        s_lose = apply_point(state, False)

        W_win = win_prob_from_state(s_win, p_srv1, p_srv2)
        W_lose = win_prob_from_state(s_lose, p_srv1, p_srv2)

        pressure_raw = W_win - W_lose
        pressure_swing = pressure_raw / 2.0

        win_probs.append(W_before)
        pressure_raws.append(pressure_raw)
        pressure_swings.append(pressure_swing)

    m["win_prob_p1"] = win_probs
    m["pressure_raw"] = pressure_raws
    m["pressure_swing"] = pressure_swings

    return m


def main():
    df = pd.read_parquet(INPUT_PATH)
    match_ids = df["match_id"].unique()
    all_dfs = []

    for match_id in match_ids:
        print(f"Processing match {match_id}...")
        m = df[df["match_id"] == match_id].copy()
        m_pressure = compute_pressure_for_match(m)
        all_dfs.append(m_pressure)

    result_df = pd.concat(all_dfs, ignore_index=True)
    result_df.to_parquet(OUTPUT_PATH)
    print(f"Saved point pressure data to {OUTPUT_PATH}")


if __name__ == "__main__":
   main()