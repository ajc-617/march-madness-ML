from bs4 import BeautifulSoup
from main import page_paths
import requests

base_url = "https://www.ncaa.com/stats/basketball-men/d1/current/team"

team_stats_dictionary = {}

first_run = True
num_teams = 0
num_iter = 0

for cur_key in page_paths:
    if num_iter == 12:
        print("jere")
    new_url = base_url + "/" + page_paths[cur_key]
    for x in range (1, 9):
        actual_new_url = new_url + "/" + "p" + str(x)
        cur_page = requests.get(actual_new_url)
        doc = BeautifulSoup(cur_page.text, "html.parser")
        rows = doc.find_all("tr")
        rows = rows[1:len(rows)]
        for cur_row in rows:
            entries = cur_row.find_all("td")
            if cur_key != 'Fouls/Game' and cur_key != 'OPP RPG':
                cur_stat = entries[len(entries) - 1].string
            else:
                cur_stat = entries[len(entries) - 2].string
            cur_team_name = cur_row.find("a").string
            if first_run:
                team_stats_dictionary[cur_team_name] = {cur_key : cur_stat}
                continue
            team_stats_dictionary[cur_team_name][cur_key] = cur_stat
    first_run = False
    num_iter += 1
            
print("done")