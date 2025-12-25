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
│   └── matches_points_merged/    # Merged raw match + point data
│
├── data_processed/
│   ├── 01_score_state_features.parquet
│   ├── 02_serve_state_features.parquet
│   └── (later) 03_pressure_features.parquet
│
├── notebooks/
│   ├── 00_data_validation.ipynb
│   ├── 01_merge_and_filter.ipynb
│   ├── 02_score_state_features.ipynb
│   ├── 03_serve_state_features.ipynb
│   └── 04_pressure_features.ipynb
│
├── .gitignore
└── README.md


1. Load raw match-level and point-level data
2. Merge on match_id
3. Filter to WTA QF+ matches
4. Add score state features (point, game, set, match state)
5. Add serve state features
6. (Upcoming) Add pressure / critical point features
7. Save intermediate datasets to data_processed
8. This shows structure without drowning in details.


- Raw data files are not modified in place
- Each processing step saves a new dataset
- Notebooks are intended to be run in order


Status
------
Data prepocessing and validation complete.
