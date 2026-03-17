import pandas as pd
from pathlib import Path

ROOT = Path.cwd()
DATA_PROCESSED = ROOT / "data_processed"
INPUT_PATH = DATA_PROCESSED / "06_point_pressure.parquet"

df = pd.read_parquet(INPUT_PATH)

assert df["win_prob_p1"].between(0,1).all()
print("Win probability bounds OK")

assert df["win_prob_p1"].notna().all()
print("No missing win probabilities")

assert df["pressure_raw"].notna().all()
assert df["pressure_swing"].notna().all()
print("No missing pressure values")

assert (df["pressure_raw"] >= 0).all()
assert (df["pressure_swing"] >= 0).all()
print("Pressure values non-negative")

assert (df["pressure_raw"] <= 1).all()
print("Pressure_raw within bounds")

print("\nWin probability distribution")
print(df["win_prob_p1"].describe())

print("\nPressure_raw distribution")
print(df["pressure_raw"].describe())

print("\nPressure_swing distribution")
print(df["pressure_swing"].describe())

print("\nTop 10 highest pressure points:")
print(
    df.sort_values("pressure_raw", ascending=False)[
        ["match_id","SetNo","GameNo","PointNumber","pressure_raw"]
    ].head(10)
)