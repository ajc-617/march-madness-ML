from bs4 import BeautifulSoup

import requests
import json

def fetch_stats():
    base_url = "https://www.ncaa.com/stats/basketball-men/d1/current/team"
    
    team_stats_dictionary = {}

    f = open('paths.json')

    page_paths = json.load(f)

    first_run = True

    for cur_key in page_paths:
        new_url = base_url + "/" + page_paths[cur_key]
        for x in range (1, 9):
            actual_new_url = new_url + "/" + "p" + str(x)
            cur_page = requests.get(actual_new_url)
            doc = BeautifulSoup(cur_page.text, "html.parser")
            rows = doc.find_all("tr")
            rows = rows[1:len(rows)]
            for cur_row in rows:
                entries = cur_row.find_all("td")
                if cur_key != 'Fouls/Game' and cur_key != 'OPP RPG': #for some reasoon Fouls/Game and OPP RPG are in the second to last column of theur respective pages
                    cur_stat = entries[len(entries) - 1].string
                else:
                    cur_stat = entries[len(entries) - 2].string
                cur_team_name = cur_row.find("a").string
                if first_run:
                    team_stats_dictionary[cur_team_name] = {cur_key : cur_stat}
                    continue
                team_stats_dictionary[cur_team_name][cur_key] = cur_stat
        first_run = False

    return team_stats_dictionary
            

if __name__ == "__main__":
    output_dict = fetch_stats()
    json_object = json.dumps(fetch_stats(), indent = 4)
    print(json_object)
    with open("stats.json", "w") as outfile:
        json.dump(output_dict, outfile)