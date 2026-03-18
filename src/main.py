# ─────────────────────────────────────────────
# 6. MAIN — RUN THE FULL PIPELINE
# ─────────────────────────────────────────────

from data.data_prep import scrape_all_years


if __name__ == "__main__":

    # ── Step 1: Scrape Barttorvik ──
    print("=== Scraping Barttorvik ===")
    team_stats = scrape_all_years(start=2010, end=2024)
    team_stats.to_csv("barttorvik_team_stats.csv", index=False)
    print(f"Saved {len(team_stats)} rows to barttorvik_team_stats.csv\n")

    # ── Step 2: Load Kaggle tournament data ──
    # Download from: https://www.kaggle.com/competitions/march-machine-learning-mania-2024/data
    print("=== Loading Kaggle tournament data ===")
    try:
        tourney_results = pd.read_csv("MNCAATourneyCompactResults.csv")
        seeds           = pd.read_csv("MNCAATourneySeeds.csv")
        teams_lookup    = pd.read_csv("MTeams.csv")

        # Filter to years we have Barttorvik data for
        tourney_results = tourney_results[tourney_results["Season"] >= 2010]

        # ── Step 3: Build matchup dataset ──
        print("=== Building matchup dataset ===")
        # NOTE: Team names between Barttorvik and Kaggle often differ slightly
        # (e.g. "UConn" vs "Connecticut"). You'll need a crosswalk CSV to align them.
        # A community-maintained one: https://github.com/elitasson/ncaa-name-crosswalk
        matchups = build_matchup_dataset(team_stats, tourney_results, seeds, teams_lookup)
        matchups.to_csv("tournament_matchups.csv", index=False)
        print(f"Saved {len(matchups)} matchup rows to tournament_matchups.csv\n")

        # ── Step 4: Train/val split by year (never shuffle across years) ──
        train_df = matchups[matchups["season"] < 2022]
        val_df   = matchups[matchups["season"] >= 2022]

        X_train = train_df[FEATURE_COLS].fillna(0)
        y_train = train_df["label"]
        X_val   = val_df[FEATURE_COLS].fillna(0)
        y_val   = val_df["label"]

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
        val_data   = lgb.Dataset(X_val,   label=y_val,   reference=train_data)

        # ── Step 5: Train the model ──
        print("=== Training LightGBM ===")
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=25),
        ]

        model = lgb.train(
            LGBM_PARAMS,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )

        # ── Step 6: Evaluate ──
        val_preds  = model.predict(X_val)
        val_labels = (val_preds >= 0.5).astype(int)

        print(f"\nVal Log Loss: {log_loss(y_val, val_preds):.4f}")
        print(f"Val Accuracy: {accuracy_score(y_val, val_labels):.3f}")
        print(f"Baseline (always pick lower seed): "
              f"{accuracy_score(y_val, (val_df['seed_diff'] < 0).astype(int)):.3f}")

        # ── Step 7: Feature importance ──
        print("\n=== Feature Importance (gain) ===")
        importance = pd.Series(
            model.feature_importance(importance_type="gain"),
            index=FEATURE_COLS
        ).sort_values(ascending=False)
        print(importance.to_string())

        # ── Step 8: Save ──
        model.save_model("tourney_model.lgbm")
        joblib.dump(importance, "feature_importance.pkl")
        print("\nModel saved to tourney_model.lgbm")

    except FileNotFoundError as e:
        print(f"\nKaggle files not found: {e}")
        print("Download from https://www.kaggle.com/competitions/march-machine-learning-mania-2024/data")
        print("Then re-run this script.")