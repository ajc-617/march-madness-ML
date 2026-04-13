import pandas as pd
import numpy as np

def feature_engineering():

    team_stats_df = pd.read_csv("data/processed/barttorvik_team_stats.csv")
    results_df = pd.read_csv("data/processed/sports_ref_team_results.csv")
    #accounting for covid season (2020 is dropped because no tournament but there aree stats for it)
    seasons = pd.concat([results_df["season"], team_stats_df["season"]]).drop_duplicates()
    print(type(seasons))
    team_stats_df = team_stats_df[team_stats_df["season"].isin(seasons)]
    results_df = results_df[results_df["season"].isin(seasons)]
    # Build spelling → TeamID lookup (all keys lowercased)
    spellings = pd.read_csv("MTeamSpellings.csv")
    name_to_id = dict(zip(spellings["TeamNameSpelling"].str.lower(), spellings["TeamID"]))
    for season in seasons:
        mm_teams = results_df[results_df["season"] == int(season)][["team_winner", "team_loser"]]
        #stack winner and loser columns on top of each other, then gets unique values
        mm_teams = list(mm_teams.stack().drop_duplicates())
        teams_from_stats = list(team_stats_df[team_stats_df["season"] == int(season)]["team"].unique())
        #for all teams from results
        for team in teams_from_stats:
            #if lowercase team name is not a key in name_to_id, print the team name and season
            if team.lower() not in name_to_id:
                print(f"no match for {team} in season {int(season)}")
    #Need to convert all team names to lowercase because that's what name_to_id uses as keys         
    team_stats_df["team_id"] = team_stats_df["team"].str.lower().map(name_to_id)
    results_df["team_winner_id"] = results_df["team_winner"].str.lower().map(name_to_id)
    results_df["team_loser_id"] = results_df["team_loser"].str.lower().map(name_to_id)

    # All numeric stat columns from barttorvik (unk_* kept since values are non-zero)
    STAT_COLS = [
        "adjoe", "adjde", "barthag", "wins", "games_played",
        "efg_o", "efg_d", "ftr", "ftrd", "tor", "tord",
        "orb", "drb", "adj_t", "twop_o", "twop_d",
        "threep_o", "threep_d", "unk_3", "unk_4",
        "threer_o", "threer_d", "wab", "unk_11", "adj_em"
    ]

    #making two dataframes so we can merge onto results_df to form matchups dataframe
    winner_stats = (
        team_stats_df[["team_id", "season"] + STAT_COLS]
        .rename(columns={c: f"w_{c}" for c in STAT_COLS})
    )
    loser_stats = (
        team_stats_df[["team_id", "season"] + STAT_COLS]
        .rename(columns={c: f"l_{c}" for c in STAT_COLS})
    )


    #dropping team id because we don't need it because we already have the winner and loser ids
    matchups = (
        results_df
        .merge(winner_stats, left_on=["team_winner_id", "season"], right_on=["team_id", "season"], how="inner")
        .drop(columns=["team_id"])
        .merge(loser_stats, left_on=["team_loser_id", "season"], right_on=["team_id", "season"], how="inner")
        .drop(columns=["team_id"])
    )

    print(f"Games after stat merge: {len(matchups)} (dropped {len(results_df) - len(matchups)} with missing stats)")

    # consistent team ordering: team_1 = lower seed number (better seeded)
    # at inference time we don't know the winner, so we order by seed instead such that t1 is always first
    # margin = t1_score - t2_score  -> negative means upset
    is_t1_winner = matchups["seed_winner"] <= matchups["seed_loser"]

    #if t1 (lower seed) is the winner, then t1_seed is the seed of the winner, otherwise it is the seed of the loser
    #same thing for t2 and for two score columns
    #TODO maybe simplify this? Seems like there should be an easier way to do this than have four different lines of code
    matchups["t1_seed"]  = np.where(is_t1_winner, matchups["seed_winner"],  matchups["seed_loser"])
    matchups["t2_seed"]  = np.where(is_t1_winner, matchups["seed_loser"],   matchups["seed_winner"])
    matchups["t1_score"] = np.where(is_t1_winner, matchups["score_winner"], matchups["score_loser"])
    matchups["t2_score"] = np.where(is_t1_winner, matchups["score_loser"],  matchups["score_winner"])

    for col in STAT_COLS:
        matchups[f"t1_{col}"] = np.where(is_t1_winner, matchups[f"w_{col}"], matchups[f"l_{col}"])
        matchups[f"t2_{col}"] = np.where(is_t1_winner, matchups[f"l_{col}"], matchups[f"w_{col}"])

    # ---LABELS ---
    #total points is the sum of the two scores
    matchups["total_points"]   = matchups["t1_score"] + matchups["t2_score"]
    #Therefore winning margin is (lower seed score) - (higher seed score)
    matchups["winning_margin"] = matchups["t1_score"] - matchups["t2_score"]

    # seed_diff: positive means t1 is favored (e.g. seed 1 vs seed 16 → diff = 15)
    matchups["seed_diff"] = matchups["t2_seed"] - matchups["t1_seed"]


    # ── Feature matrix and label vectors ──
    FEATURE_COLS = (
        ["t1_seed", "t2_seed"]
        + [f"t1_{c}" for c in STAT_COLS]
        + [f"t2_{c}" for c in STAT_COLS]
    )
    LABEL_COLS = ["total_points", "winning_margin"]

    # ── Train / val / test split by season (no shuffle — prevents leakage) ──
    # 2020 had no tournament (COVID), so effective seasons: 2010-2019, 2021-2025
    train_seasons = list(range(2010, 2022))   # 10 tournament years
    val_seasons   = [2022, 2023]
    test_seasons  = [2024, 2025]

    train_df = matchups[matchups["season"].isin(train_seasons)]
    val_df   = matchups[matchups["season"].isin(val_seasons)]
    test_df  = matchups[matchups["season"].isin(test_seasons)]

    X_train, y_train = train_df[FEATURE_COLS].values, train_df[LABEL_COLS].values
    X_val,   y_val   = val_df[FEATURE_COLS].values,   val_df[LABEL_COLS].values
    X_test,  y_test  = test_df[FEATURE_COLS].values,  test_df[LABEL_COLS].values

    print(f"Features: {len(FEATURE_COLS)}")
    print(f"Train: {X_train.shape}  ({sorted(train_df['season'].unique())})")
    print(f"Val:   {X_val.shape}    ({sorted(val_df['season'].unique())})")
    print(f"Test:  {X_test.shape}   ({sorted(test_df['season'].unique())})")

    return X_train, y_train, X_val, y_val, X_test, y_test
