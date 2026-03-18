page_paths = {'Assist/Turnover Ratio': '474', 'Assists/Game': '216', 'Bench Points/Game': '1284', 'BKPG': '214', 'Effective FG Pct': '1288', 'Fastbreak PTS': '1285', 'FG Pct': '148', 'FG Pct Defense': '149', \
              'Fouls/Game': '286', 'FT Attempts/Game': '638', 'FT Pct': '150', 'FT Made/game': '633', 'OPP RPG': '151', 'DRPG/Game': '859', 'ORPG/Game': '857', 'RPG': '932', 'OPP PPG': '146', \
              'PPG': '145', 'STPG': '215', '3FGA/Game': '625', '3FG Pct': '152', 'OPP 3FG Pct': '518', '3PG': '153', 'TO Margin': '519', 'OPP TO/Game': '931', 'TO/Game': '217', 'Win Pct': '168'}

string_paths = str(page_paths)
new_str = ""

for cur_char in range(0, len(string_paths)):
    if string_paths[cur_char] == '\'':
         new_str += '\"'
    else:
        new_str += string_paths[cur_char]

print(new_str)