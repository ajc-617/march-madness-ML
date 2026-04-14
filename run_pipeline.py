import os
from src.data.data_prep import scrape_all_years, scrape_tourney_results
from src.features.feature_eng import feature_engineering
import pandas as pd


if __name__ == "__main__":

    print("=== Scraping Barttorvik ===")
    #stores data in data/processed/barttorvik_team_stats.csv
    scrape_all_years()
    #assert that barttorvik team stats file exists
    assert os.path.isfile("data/processed/barttorvik_team_stats.csv"), "File data/processed/barttorvik_team_stats.csv not found"
    #stores data
    print("=== Scraping Sports Reference Results ===")
    scrape_tourney_results()
    #assert that results file exists
    assert os.path.isfile("data/processed/sports_ref_team_results.csv"), "File data/processed/sports_ref_team_results.csv not found"

    #splits = feature_engineering()