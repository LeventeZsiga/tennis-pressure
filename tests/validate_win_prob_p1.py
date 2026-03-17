import pandas as pd
from pathlib import Path

ROOT = Path.cwd()
DATA_PROCESSED = ROOT / "data_processed"
INPUT_PATH = DATA_PROCESSED / "06_point_pressure.parquet"

df = pd.read_parquet(INPUT_PATH)

print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print("Unique matches:", df["match_id"].nunique())


assert df["win_prob_p1"].between(0,1).all()
print("Win probability bounds OK")

assert df["win_prob_p1"].notna().all()
print("No missing win probabilities")

starts = df[
    (df["SetNo"] == 1) &
    (df["GameNo"] == 1) &
    (df["PointNumber"] == 1)
]

print(starts["win_prob_p1"].describe())


ends = df[df["is_match_end"] == True]

print(ends["win_prob_p1"].describe())


ends = df[df["is_match_end"] == True]

print(ends["win_prob_p1"].describe())


sample_match = df[df["match_id"] == df["match_id"].iloc[0]]
print(sample_match[["SetNo","GameNo","PointNumber","win_prob_p1"]].head(20))