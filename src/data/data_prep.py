"""
NCAA Tournament Bracket Prediction - Data Collection & Model Starter
Sources: Barttorvik (team stats) + Sports Reference (tournament results)

Steps:
  1. Scrape team season stats from Barttorvik for each year
  2. Scrape tournament results from Sports Reference
  3. Join into matchup-level dataset and save to CSV
  4. Load into a PyTorch Dataset for training
"""

import json
import requests
import pandas as pd
import time
from io import StringIO


def _fetch_json_with_browser(url: str):
    """
    Fetch a JSON endpoint that sits behind a JS bot-challenge page.
    The challenge is a simple form POST with js_test_submitted=1 plus
    whatever cookies the server set on the initial GET.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    })
    # Step 1: GET to receive the challenge cookie
    session.get(url, timeout=15)
    # Step 2: POST the JS-test form to pass the challenge
    resp = session.post(url, data={"js_test_submitted": "1"}, timeout=15)
    resp.raise_for_status()
    return resp.json()

# ── Optional: suppress SSL warnings if you hit certificate issues ──
import warnings
warnings.filterwarnings("ignore")


def scrape_barttorvik(year: int) -> pd.DataFrame:
    """
    Pull end-of-season team stats from Barttorvik for a given year.
    Returns a DataFrame with one row per team.
    'year' = the year the tournament was played (e.g. 2023 for 2022-23 season)
    """
    url = f"https://barttorvik.com/trank.php?year={year}&json=1"
    raw = _fetch_json_with_browser(url)

    # with open(f"test_{year}.json", "w") as file:
    #     json.dump(raw, file)
    #     exit(0)
    # Barttorvik JSON: list of lists. Column order can shift year-to-year,
    # so we grab the headers from a known stable endpoint when possible.
    # These columns are stable across recent years:

    new_columns = [
        "team", #UC Santa Barbara
        "adjoe", #104.3
        "adjde", #106.7
        "barthag", #0.4345
        "record", #22-10
        "wins", #22
        "games_played", #32
        "efg_o", #50.3
        "efg_d", #49.7
        "ftr", #39.5
        "ftrd", #32.4
        "tor", #16.8
        "tord", #16.6
        "orb", #33.1
        "drb", #26.2
        "adj_t", #64.2597
        "twop_o", #49.4
        "twop_d", #50.8
        "threep_o", #34.8
        "threep_d", #31.8,
        "unk_1", 
        "unk_2", 
        "unk_3", #50.6
        "unk_4", #46.2
        "threer_o", #32.7
        "threer_d", #35.5
        "adj_t_2", #64.2597
        "unk_5",
        "unk_6",
        "unk_7",
        "season", #2019
        "unk_8",
        "unk_9",
        "unk_10",
        "wab", #-5.84772
        "unk_11", #71.8
        "unk_12" #null
    ]

    #deep to drop record because it's a string also adj_t_2 because it's a duplicate
    rows = [row for row in raw]
    print(len(rows))
    
    df = pd.DataFrame(rows, columns=new_columns)
    # Derived feature: efficiency margin (single best tournament predictor)
    df["adj_em"] = pd.to_numeric(df["adjoe"], errors="coerce") - pd.to_numeric(df["adjde"], errors="coerce")
    return df


def scrape_all_years(start: int = 2010, end: int = 2025):
    """Scrape Barttorvik for a range of years and concatenate."""
    frames = []
    for year in range(start, end + 1):
        print(f"Fetching {year}...\n")
        try:
            df = scrape_barttorvik(year)
            frames.append(df)
        except Exception as e:
            print(f"FAILED WITH EXCEPTION ({e})")
          # to avoid throttling esrver too much
        time.sleep(1.5)
    return pd.concat(frames, ignore_index=True).drop(["record"], axis=1)

def clean_barttorvik_columns(df): 

    # len(df["unk_12"][df["unk_12"].notna()])
    # len(df["unk_10"][df["unk_10"].str.len() > 0])
    # len(df["unk_9"][df["unk_9"].str.len() > 0])
    # len(df["unk_8"][df["unk_8"].str.len() > 0])
    # len(df["unk_7"][df["unk_7"].str.len() > 0])
    # len(df["unk_6"][df["unk_6"].str.len() > 0])
    # len(df["unk_5"][df["unk_5"].str.len() > 0])
    #print(len(df["unk_1"][df["unk_1"] != 0]))
    #print(len(df["unk_2"][df["unk_2"] != 0]))

    #dropping a bunch of unknown columns from the JSON return
    df = df.drop(["unk_1", "unk_2", "unk_5", "unk_6", "unk_7", "unk_8", "unk_9", "unk_10", "unk_12"], axis=1)
    #dropping duplicate adjusted tempo column as well
    df = df.drop(["adj_t_2"], axis=1)
    return df




# ─────────────────────────────────────────────
# 2. SCRAPE TOURNAMENT RESULTS (Sports Reference)
# ─────────────────────────────────────────────

def scrape_tourney_results(year: int) -> pd.DataFrame:
    """
    Pull NCAA tournament game results from Sports Reference for a given year.
    Returns DataFrame with columns: year, round, team_winner, team_loser,
    score_winner, score_loser, seed_winner, seed_loser
    """
    import requests
    url = f"https://www.sports-reference.com/cbb/postseason/{year}-ncaa.html"
    headers = {"User-Agent": "Mozilla/5.0"}

    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    # Sports Reference uses <div id="bracket"> with nested tables
    # pd.read_html parses all tables on the page
    tables = pd.read_html(StringIO(resp.text))

    # The bracket data isn't in a clean table format on this page.
    # Instead, parse game divs. For a simpler approach, use their
    # game log / schedule pages which ARE clean tables:
    schedule_url = (
        f"https://www.sports-reference.com/cbb/postseason/men/{year}-ncaa.html"
    )
    resp2 = requests.get(schedule_url, headers=headers, timeout=15)

    # Fallback: return empty with schema so pipeline doesn't break
    # In practice you may need to refine this parser per year's HTML structure.
    # See note below about using a pre-built dataset instead.
    return pd.DataFrame(columns=[
        "year", "round", "seed_winner", "team_winner",
        "score_winner", "score_loser", "seed_loser", "team_loser"
    ])


# ─────────────────────────────────────────────
# NOTE: For tournament results, the easiest path is a pre-built dataset.
# Download the Kaggle "March Machine Learning Mania" dataset:
#   https://www.kaggle.com/competitions/march-machine-learning-mania-2024/data
# It has clean CSVs for every tournament game since 1985.
# Files you want:
#   MNCAATourneyCompactResults.csv  ← game outcomes (winner, loser, score)
#   MNCAATourneySeeds.csv           ← seeds per team per year
#   MTeams.csv                      ← team ID → team name mapping
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# 3. BUILD MATCHUP DATASET
# ─────────────────────────────────────────────

def build_matchup_dataset(
    team_stats: pd.DataFrame,
    tourney_results: pd.DataFrame,
    seeds: pd.DataFrame,
    teams_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join team stats with tournament matchups to create one row per game.
    Features are expressed as Team A minus Team B (matchup differentials).
    Label: 1 if Team A won, 0 if Team B won.

    Expects Kaggle-format DataFrames:
      tourney_results: Season, WTeamID, LTeamID, WScore, LScore
      seeds:           Season, Seed (e.g. 'W01'), TeamID
      teams_lookup:    TeamID, TeamName
    """
    # Parse seed number from string like 'W01' → 1
    seeds = seeds.copy()
    seeds["seed_num"] = seeds["Seed"].str.extract(r"(\d+)").astype(int)

    rows = []
    for _, game in tourney_results.iterrows():
        season = game["Season"]

        w_id = game["WTeamID"]
        l_id = game["LTeamID"]

        # Look up team names
        w_name = teams_lookup.loc[teams_lookup["TeamID"] == w_id, "TeamName"].values
        l_name = teams_lookup.loc[teams_lookup["TeamID"] == l_id, "TeamName"].values
        if len(w_name) == 0 or len(l_name) == 0:
            continue
        w_name, l_name = w_name[0], l_name[0]

        # Get seeds
        w_seed = seeds.loc[(seeds["Season"] == season) & (seeds["TeamID"] == w_id), "seed_num"].values
        l_seed = seeds.loc[(seeds["Season"] == season) & (seeds["TeamID"] == l_id), "seed_num"].values
        w_seed = w_seed[0] if len(w_seed) else None
        l_seed = l_seed[0] if len(l_seed) else None

        # Get team stats from Barttorvik (match on team name + year)
        # Note: names may not match exactly — you'll need a name crosswalk.
        # See the name_crosswalk note below.
        w_stats = team_stats[(team_stats["year"] == season) & (team_stats["team"] == w_name)]
        l_stats = team_stats[(team_stats["year"] == season) & (team_stats["team"] == l_name)]

        if w_stats.empty or l_stats.empty:
            continue

        w = w_stats.iloc[0]
        l = l_stats.iloc[0]

        stat_cols = ["adj_em", "adjoe", "adjde", "barthag", "efg_o", "efg_d",
                     "tor", "tord", "orb", "drb", "ftr", "adj_t", "wab"]

        # Build feature row as winner - loser differentials
        feature_row = {"season": season, "label": 1}
        for col in stat_cols:
            try:
                feature_row[f"diff_{col}"] = float(w[col]) - float(l[col])
            except (ValueError, TypeError):
                feature_row[f"diff_{col}"] = None

        feature_row["seed_diff"] = (w_seed or 0) - (l_seed or 0)
        feature_row["w_seed"] = w_seed
        feature_row["l_seed"] = l_seed
        feature_row["w_team"] = w_name
        feature_row["l_team"] = l_name

        rows.append(feature_row)

        # Also add the flipped version (loser perspective) with label=0
        # This doubles your training data and removes ordering bias
        flipped = {k: v for k, v in feature_row.items()}
        for col in stat_cols:
            key = f"diff_{col}"
            if flipped[key] is not None:
                flipped[key] = -flipped[key]
        flipped["seed_diff"] = -flipped["seed_diff"]
        flipped["w_team"], flipped["l_team"] = l_name, w_name
        flipped["w_seed"], flipped["l_seed"] = l_seed, w_seed
        flipped["label"] = 0
        rows.append(flipped)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = scrape_all_years()
    df = clean_barttorvik_columns(df)


    #TODO DOUBLE CHECK THESE COLUMNS TO MAKE SURE THEY'RE RIGHT FOR ALL YEARS
    #     new_columns = [
    #     "team", #UC Santa Barbara
    #     "adjoe", #104.3
    #     "adjde", #106.7
    #     "barthag", #0.4345
    #     "record", #22-10
    #     "wins", #22
    #     "games_played", #32
    #     "efg_o", #50.3
    #     "efg_d", #49.7
    #     "ftr", #39.5
    #     "ftrd", #32.4
    #     "tor", #16.8
    #     "tord", #16.6
    #     "orb", #33.1
    #     "drb", #26.2
    #     "adj_t", #64.2597
    #     "twop_o", #49.4
    #     "twop_d", #50.8
    #     "threep_o", #34.8
    #     "threep_d", #31.8,
    #     "unk_1", 0
    #     "unk_2", 0
    #     "unk_3", #50.6
    #     "unk_4", #46.2
    #     "threer_o", #32.7
    #     "threer_d", #35.5
    #     "adj_t_2", #64.2597
    #     "unk_5",
    #     "unk_6",
    #     "unk_7",
    #     "season", #2019
    #     "unk_8",
    #     "unk_9",
    #     "unk_10",
    #     "wab", #-5.84772
    #     "unk_11", #71.8
    #     "unk_12" #null
    # ]