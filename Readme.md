Point-level pressure modelling in WTA Grand Slam matches

This repository contains data preprocessing and feature engineering code for point-level analysis of psychological pressure in women’s Grand Slam tennis matches.

Data scope:

Tour: WTA
Tournaments: Grand Slams (Australian Open, French Open, Wimbledon, US Open)
Rounds: Quarterfinals and later (QF+)
Format: Best-of-3 sets
Data level: Point-by-point
Years: 2011-2024
Tournaments with missing WTA data (e.g. Wimbledon 2016) are excluded.

tennis-pressure/
│
├── data_raw/
│   ├── matches_raw/              # Raw match-level files per tournament
│   ├── points_raw/               # Raw point-level files per tournament
│   └── matches_points_merged/    # Merged raw match + point-level data
│
├── data_processed/
│   ├── 01_score_state_features.parquet
│   │   # Point-level dataset with score/state features added
│   ├── 02_serve_state_features.parquet
│   │   # Point-level dataset with serve-related features added
│   ├── 03_tiebreak_pressure_flags.parquet
│   │   # Flags identifying pressure-relevant tiebreak situations
│   ├── 04_final_filtered_dataset.parquet
│   │   # Fully cleaned and filtered point-level dataset
│   └── 05_final_match_serve_probs.parquet
│       # Match-level serve probability features (used by the Markov model
│
├── notebooks/
│   ├── preprocessing/
│   │   ├── 01_split_mbm_pbp.ipynb
│   │   ├── 02_inspect_columns.ipynb
│   │   ├── 03_merge_matches_points.ipynb
│   │   ├── 04_score_state_features.ipynb
│   │   ├── 05_serve_state_features.ipynb
│   │   ├── 06_tiebreak_pressure_flags.ipynb
│   │   └── 07_filter_incomplete_matches.ipynb
│   │       # Step-by-step preprocessing and feature construction
│   │
│   └── validation/
│       └── 01_green_light.ipynb
│           # Final sanity checks and validation of processed datasets
│
├── src/
│   ├── state_space.py             # State definition for the Markov chain
│   ├── state_rules.py             # Tennis scoring rules (games, sets, tiebreaks)
│   ├── transitions.py             # State transition logic given a point winner
│   ├── serve_probs.py             # Match-level serve probability estimation
│   └── pressure.py                # Pressure score computation logic
│
├── tests/
│   ├── test_replay_checkpoints.py
│   │   # Validates scorekeeping by replaying matches from point winners
│   └── test_serve_probs.py
│       # Validates serve probability feature construction
│
├── .gitignore
├── README.md
└── outputs/

1. Load raw match-level and point-level data
2. Merge on match_id
3. Filter to WTA QF+ matches
4. Add score state features (point, game, set, match state)
5. Add serve state features
6. (Upcoming) Add pressure / critical point features
7. Save intermediate datasets to data_processed
8. 


- Raw data files are not modified in place
- Each processing step saves a new dataset
- Notebooks are intended to be run in order


Status
------
State space defined, state transition logic validated, serve probabilities recalculated.

