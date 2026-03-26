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
from bs4 import BeautifulSoup
# ── Optional: suppress SSL warnings if you hit certificate issues ──
import warnings
warnings.filterwarnings("ignore")


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

def _scrape_barttorvik(year: int) -> pd.DataFrame:
    """
    Pull end-of-season team stats from Barttorvik for a given year.
    Returns a DataFrame with one row per team.
    'year' = the year the tournament was played (e.g. 2023 for 2022-23 season)
    """
    url = f"https://barttorvik.com/trank.php?year={year}&json=1"
    raw = _fetch_json_with_browser(url)

    # Colums for barttorvik data: If it has unk in it I don't know what it is.
    #unk_3, unk_4, and unk_11 are kept because they're non-zero values even though I don't know what they are
    #Because still might be relevant for training

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
    df = pd.DataFrame(rows, columns=new_columns)
    #saving raw dataframe to folder
    df.to_csv(f"../../data/raw/bartorvik/{year}.csv")
    # Derived feature: efficiency margin (single best tournament predictor)
    df["adj_em"] = pd.to_numeric(df["adjoe"], errors="coerce") - pd.to_numeric(df["adjde"], errors="coerce")
    return df

def _clean_barttorvik_columns(df) -> pd.DataFrame: 
    """drops unknown columns from barttorvik dataset that are also falsy, that being an empty string or 0"""
    #dropping a bunch of unknown columns from the JSON return that have all falsy values
    df = df.drop(["unk_1", "unk_2", "unk_5", "unk_6", "unk_7", "unk_8", "unk_9", "unk_10", "unk_12"], axis=1)
    #dropping duplicate adjusted tempo column as well, and record because we already have games and wins
    df = df.drop(["adj_t_2", "record"], axis=1)
    return df

def scrape_all_years(start: int = 2010, end: int = 2025):
    """Scrape Barttorvik for a range of years and concatenate."""
    frames = []
    for year in range(start, end + 1):
        print(f"Fetching {year}...")
        try:
            df = _scrape_barttorvik(year)
            frames.append(df)
        except Exception as e:
            print(f"FAILED WITH EXCEPTION ({e})")
          # to avoid throttling server too much
        time.sleep(1.5)
    cleaned_df = _clean_barttorvik_columns(pd.concat(frames, ignore_index=True))
    cleaned_df.to_csv("../../data/processed/team_stats.csv", index=False)
    return cleaned_df


# ─────────────────────────────────────────────
# 2. SCRAPE TOURNAMENT RESULTS (Sports Reference)
# ─────────────────────────────────────────────

def _parse_team(div):
    """Takes in a div for either the winning team or losing team of the matchup, returns tuple in form of (seed, team name, score of team)"""
    seed  = div.find("span").get_text(strip=True) if div.find("span") else None
    links = div.find_all("a")
    team  = links[0].get_text(strip=True) if links else None
    score = links[1].get_text(strip=True) if len(links) > 1 else None
    return (int(seed) if seed and seed.isdigit() else None,
            team,
            int(score) if score and score.isdigit() else None)

def scrape_tourney_results(year: int) -> pd.DataFrame:
    """
    Pull NCAA tournament game results from Sports Reference for a given year.
    Returns a DataFrame with columns: year, region, seed_winner, team_winner,
    score_winner, score_loser, seed_loser, team_loser.

    Note: First Four play-in games are not included — only R64 through Championship.
    """
    url = f"https://www.sports-reference.com/cbb/postseason/men/{year}-ncaa.html"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    brackets_div = soup.find("div", id="brackets")

    records = []
    #national region contains final 4 for that year
    regions = ["east", "midwest", "south", "west", "national"]

    for region in regions:
        region_div = brackets_div.find("div", id=region)
        if not region_div:
            continue

        bracket = region_div.find("div", id="bracket")
        if not bracket:
            continue

        for round_div in bracket.find_all("div", class_="round"):
            for game_div in round_div.find_all("div", recursive=False):
                team_divs = game_div.find_all("div", recursive=False)
                if len(team_divs) < 2:
                    continue

                team_a, team_b = team_divs[0], team_divs[1]
                winner_div, loser_div = (team_a, team_b) if "winner" in team_a.get("class", []) else (team_b, team_a)


                seed_w, team_w, score_w = _parse_team(winner_div)
                seed_l, team_l, score_l = _parse_team(loser_div)

                records.append({
                    "year":         year,
                    "region":       region,
                    "seed_winner":  seed_w,
                    "team_winner":  team_w,
                    "score_winner": score_w,
                    "score_loser":  score_l,
                    "seed_loser":   seed_l,
                    "team_loser":   team_l,
                })

    return pd.DataFrame(records)