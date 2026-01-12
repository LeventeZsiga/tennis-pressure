from __future__ import annotations
import pandas as pd
import numpy as np

DATA_PATH = "data_processed/05_final_match_serve_probs.parquet"

def load_df() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values(["match_id", "SetNo", "GameNo", "PointNumber"]).reset_index(drop=True)
    return df

def test_probabilities_bounded():
    df = load_df()
    assert df["p_p1_wins_point"].between(0.0, 1.0).all(), \
    "Probabilities are not bounded between 0 and 1"

def test_cumulative_counts_consistent():
    df = load_df()
    
    for mid, m in df.groupby("match_id"):
        for col in [
            "p1_serving_upto_before",
            "p2_serving_upto_before",
            "p1_srv_won_upto_before",
            "p2_srv_won_upto_before"
        ]: 
            diffs = m[col].diff().fillna(0)
            assert (diffs >= 0).all(), f"Cumulative counts in {col} decrease in match_id {mid}"

def test_small_sample_manual_recompute_counts():
    df = load_df()

    for mid, m in df.groupby("match_id"):
        m = m.reset_index(drop=True)
        p1_serves = 0
        p2_serves = 0
        p1_wins = 0
        p2_wins = 0 

        for i, row in m.iterrows():
            assert row["p1_serving_upto_before"] == p1_serves
            assert row["p2_serving_upto_before"] == p2_serves
            assert row["p1_srv_won_upto_before"] == p1_wins
            assert row["p2_srv_won_upto_before"] == p2_wins

            if row["is_p1_server"] == 1:
                p1_serves += 1
                if row["server_won_point"] == 1:
                    p1_wins += 1
            else:
                p2_serves += 1
                if row["server_won_point"] == 1:
                    p2_wins += 1

        break

def test_probability_matches_counts():
    df = load_df()
    eps = 1e-9
    for _, row in df.iterrows():
        if row["is_p1_server"] == 1:
            p = (row["p1_srv_won_upto_before"] + 1.0) / (row["p1_serving_upto_before"] + 2.0)
        else:
            p_p2_wins = (row["p2_srv_won_upto_before"] + 1.0) / (row["p2_serving_upto_before"] + 2.0)
            p = 1.0 - p_p2_wins

        assert abs(float(row["p_p1_wins_point"]) - float(p)) < eps


def main():
    print("Running serve probability feature tests...")

    test_probabilities_bounded()
    print("✓ probabilities bounded")

    test_cumulative_counts_consistent()
    print("✓ cumulative counts consistent")

    test_small_sample_manual_recompute_counts()
    print("✓ manual recomputation matches")

    test_probability_matches_counts()
    print("✓ probabilities match counts")

    print("\nPASS: all serve probability feature tests passed")


if __name__ == "__main__":
    main()