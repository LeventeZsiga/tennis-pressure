# src/serve_probs.py
from __future__ import annotations

import numpy as np
import pandas as pd

IN_PATH = "data_processed/04_final_filtered_dataset.parquet"
OUT_PATH = "data_processed/05_final_match_serve_probs.parquet"

DROP_COLS = [
    "p1_serving",
    "p2_serving",
    "p1_srv_points_won",
    "p2_srv_points_won",
]


def add_match_level_serve_features(df: pd.DataFrame, a: float = 1.0, b: float = 1.0) -> pd.DataFrame:

    df = df.sort_values(["match_id", "SetNo", "GameNo", "PointNumber"]).copy()
    g = df.groupby("match_id", sort=False)

    is_p1_server = df["is_p1_server"].astype(int)
    is_p2_server = (1 - is_p1_server).astype(int)

    p1_srv_won_point = ((df["is_p1_server"] == 1) & (df["server_won_point"] == 1)).astype(int)
    p2_srv_won_point = ((df["is_p1_server"] == 0) & (df["server_won_point"] == 1)).astype(int)

    df["p1_serving_upto_before"] = g["is_p1_server"].transform(
        lambda s: (s == 1).cumsum().shift(1).fillna(0)
    ).astype(int)

    df["p2_serving_upto_before"] = g["is_p1_server"].transform(
        lambda s: (s == 0).cumsum().shift(1).fillna(0)
    ).astype(int)

    df["_p1_srv_won_point"] = p1_srv_won_point
    df["_p2_srv_won_point"] = p2_srv_won_point

    df["p1_srv_won_upto_before"] = g["_p1_srv_won_point"].transform(
        lambda s: s.cumsum().shift(1).fillna(0)
    ).astype(int)

    df["p2_srv_won_upto_before"] = g["_p2_srv_won_point"].transform(
        lambda s: s.cumsum().shift(1).fillna(0)
    ).astype(int)

    p1_on_serve = (df["p1_srv_won_upto_before"] + a) / (df["p1_serving_upto_before"] + a + b)
    p2_on_serve = (df["p2_srv_won_upto_before"] + a) / (df["p2_serving_upto_before"] + a + b)

    df["p_p1_wins_point"] = np.where(df["is_p1_server"] == 1, p1_on_serve, 1.0 - p2_on_serve).astype(float)

    df.drop(columns=["_p1_srv_won_point", "_p2_srv_won_point"], inplace=True)

    return df


def main() -> None:
    df = pd.read_parquet(IN_PATH)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    df = add_match_level_serve_features(df, a=1.0, b=1.0)

    cols = [
        "p1_serving_upto_before",
        "p1_srv_won_upto_before",
        "p2_serving_upto_before",
        "p2_srv_won_upto_before",
        "p_p1_wins_point",
    ]

    print("Saved columns added:", [c for c in cols if c in df.columns])
    print("p_p1_wins_point min/max:", float(df["p_p1_wins_point"].min()), float(df["p_p1_wins_point"].max()))

    p1_mean = float(df.loc[df["is_p1_server"] == 1, "p_p1_wins_point"].mean())
    p2_mean = float((1.0 - df.loc[df["is_p1_server"] == 0, "p_p1_wins_point"]).mean())
    print("Mean P(P1 wins | P1 serves):", p1_mean)
    print("Mean P(P2 wins | P2 serves):", p2_mean)

    df.to_parquet(OUT_PATH, index=False)
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()
