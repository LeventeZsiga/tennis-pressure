from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

# Project imports (must be runnable from project root with `python -m ...`)
from src.model_state import ModelState
from src.markov_from_state import win_prob_from_state
from src.manual_state_extraction import apply_point, compute_match_serve_probs, extract_state


@dataclass
class ValidationResult:
    match_id: str
    ok: bool
    error_type: str = ""
    error_msg: str = ""
    context: str = ""


def validate_match_pressure(df: pd.DataFrame, match_id: str, eps: float = 1e-12) -> ValidationResult:
    try:
        m = df[df["match_id"] == match_id].copy()
        if m.empty:
            return ValidationResult(match_id=match_id, ok=False, error_type="EmptyMatch", error_msg="match_id not found")

        m = m[m["PointServer"].isin([1, 2])].copy()
        if m.empty:
            return ValidationResult(match_id=match_id, ok=False, error_type="NoRealPoints", error_msg="no PointServer in {1,2}")

        m = m.sort_values(["SetNo", "GameNo", "PointNumber"]).reset_index(drop=True)

        p_srv1, p_srv2 = compute_match_serve_probs(m)

        for row in m.itertuples(index=False):
            rowd: Dict[str, Any] = row._asdict()
            s: ModelState = extract_state(pd.Series(rowd))

            if s.sets1 >= 2 or s.sets2 >= 2:
                continue

            W_before = win_prob_from_state(s, p_srv1, p_srv2)

            s_win = apply_point(s, True)
            s_lose = apply_point(s, False)

            W_win = win_prob_from_state(s_win, p_srv1, p_srv2)
            W_lose = win_prob_from_state(s_lose, p_srv1, p_srv2)

           
            if not (0.0 - eps <= W_before <= 1.0 + eps):
                return ValidationResult(
                    match_id=match_id,
                    ok=False,
                    error_type="ProbOutOfBounds",
                    error_msg=f"W_before={W_before}",
                    context=f"SetNo={rowd.get('SetNo')} GameNo={rowd.get('GameNo')} PointNumber={rowd.get('PointNumber')} state={s}",
                )
            if not (0.0 - eps <= W_win <= 1.0 + eps):
                return ValidationResult(
                    match_id=match_id,
                    ok=False,
                    error_type="ProbOutOfBounds",
                    error_msg=f"W_win={W_win}",
                    context=f"SetNo={rowd.get('SetNo')} GameNo={rowd.get('GameNo')} PointNumber={rowd.get('PointNumber')} state={s}",
                )
            if not (0.0 - eps <= W_lose <= 1.0 + eps):
                return ValidationResult(
                    match_id=match_id,
                    ok=False,
                    error_type="ProbOutOfBounds",
                    error_msg=f"W_lose={W_lose}",
                    context=f"SetNo={rowd.get('SetNo')} GameNo={rowd.get('GameNo')} PointNumber={rowd.get('PointNumber')} state={s}",
                )

            if W_win + eps < W_lose:
                return ValidationResult(
                    match_id=match_id,
                    ok=False,
                    error_type="MonotonicityViolation",
                    error_msg=f"W_win={W_win:.12f} < W_lose={W_lose:.12f}",
                    context=f"SetNo={rowd.get('SetNo')} GameNo={rowd.get('GameNo')} PointNumber={rowd.get('PointNumber')} state={s}",
                )

        return ValidationResult(match_id=match_id, ok=True)

    except RecursionError as e:
        return ValidationResult(match_id=match_id, ok=False, error_type="RecursionError", error_msg=str(e))
    except Exception as e:
        return ValidationResult(match_id=match_id, ok=False, error_type=type(e).__name__, error_msg=str(e))


def validate_random_matches(df: pd.DataFrame, n: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    match_ids = df["match_id"].dropna().unique().tolist()
    if not match_ids:
        raise ValueError("No match_ids found in dataset.")

    n = min(n, len(match_ids))
    sample_ids = rng.sample(match_ids, n)

    results: List[Dict[str, Any]] = []
    for mid in sample_ids:
        res = validate_match_pressure(df, mid)
        results.append(
            {
                "match_id": res.match_id,
                "ok": res.ok,
                "error_type": res.error_type,
                "error_msg": res.error_msg,
                "context": res.context,
            }
        )

    return pd.DataFrame(results).sort_values(["ok", "match_id"]).reset_index(drop=True)


def _default_data_path() -> Path:
    root = Path.cwd()
    return root / "data_processed" / "05_final_filtered_adv_matches.parquet"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default=str(_default_data_path()),
        help="Path to parquet dataset (default: data_processed/05_final_filtered_adv_matches.parquet)",
    )
    parser.add_argument("--n", type=int, default=50, help="Number of random matches to validate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for match sampling")
    parser.add_argument("--out", type=str, default="", help="Optional output CSV path for results")
    args = parser.parse_args()

    df = pd.read_parquet(args.data)

    results = validate_random_matches(df, n=args.n, seed=args.seed)

    print(results["ok"].value_counts(dropna=False))
    failed = results[results["ok"] == False]
    if not failed.empty:
        print("\nFailures (up to 20):")
        print(failed.head(20).to_string(index=False))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_path, index=False)
        print("\nSaved results to:", out_path)


if __name__ == "__main__":
    main()


def test_random_matches_termination_smoke():
    df = pd.read_parquet(_default_data_path())
    results = validate_random_matches(df, n=10, seed=123)
    assert results["ok"].all(), f"Some matches failed:\n{results[results['ok'] == False]}"