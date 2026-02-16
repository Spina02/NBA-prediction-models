import pandas as pd
import numpy as np
import kagglehub
from pathlib import Path

# ------------------------------- Initialization -------------------------------

print(">>> Downloading dataset...")
path = kagglehub.dataset_download("nathanlauga/nba-games")

out_dir = Path(__file__).resolve().parent.parent.parent / "data"
out_dir.mkdir(parents=True, exist_ok=True)

# Load data
print(">>> Loading CSVs...")
games = pd.read_csv(path + "/games.csv")
games_details = pd.read_csv(path + "/games_details.csv", low_memory=False)
players = pd.read_csv(path + "/players.csv")
teams = pd.read_csv(path + "/teams.csv")

dataframes = {}

# -------------------- Teams & Players Lookup Separation --------------------

# Teams Lookup
teams_lookup = teams[['TEAM_ID', 'ABBREVIATION', 'NICKNAME', 'CITY', 'HEADCOACH']].drop_duplicates()
# Players Lookup
players_lookup = players[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'SEASON']].drop_duplicates()

dataframes['teams_lookup'] = teams_lookup
dataframes['players_lookup'] = players_lookup

# ---------------------------- Home / Away Separation ---------------------------
print(">>> Processing Team Logs...")

cols = ['GAME_DATE_EST', 'GAME_ID', 'SEASON', 
        'HOME_TEAM_ID', 'PTS_home', 'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home', 'HOME_TEAM_WINS',
        'VISITOR_TEAM_ID', 'PTS_away', 'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away']

# Drop duplicate rows in source data (e.g. 2020 COVID bubble duplicates in Kaggle dataset)
games = games[cols].drop_duplicates(subset=['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID']).sort_values(by=['GAME_DATE_EST'])

# Home Perspective
home_df = games.copy()
home_df['TEAM_ID'] = home_df['HOME_TEAM_ID']
home_df['OPP_TEAM_ID'] = home_df['VISITOR_TEAM_ID']
home_df['IS_HOME'] = 1
home_df['WL'] = home_df['HOME_TEAM_WINS']
rename_home = {c: c.replace('_home', '') for c in cols if '_home' in c}
home_df = home_df.rename(columns=rename_home).drop(columns=[c for c in cols if '_away' in c])

# Away Perspective
away_df = games.copy()
away_df['TEAM_ID'] = away_df['VISITOR_TEAM_ID']
away_df['OPP_TEAM_ID'] = away_df['HOME_TEAM_ID']
away_df['IS_HOME'] = 0
away_df['WL'] = 1 - away_df['HOME_TEAM_WINS'] # Invert for Visitor
rename_away = {c: c.replace('_away', '') for c in cols if '_away' in c}
away_df = away_df.rename(columns=rename_away).drop(columns=[c for c in cols if '_home' in c])

# Stack
games_log = pd.concat([home_df, away_df], axis=0).sort_values(['TEAM_ID', 'GAME_DATE_EST'])

# -------------------------------- Team Rest & Rolling Stats --------------------------------
print(">>> Calculating Rolling Stats (Input Features) & Rest Days...")

# Ensure datetime for rest calculation
games_log['GAME_DATE_EST'] = pd.to_datetime(games_log['GAME_DATE_EST'])

# Team-level rest days
games_log['REST_DAYS_TEAM'] = (
    games_log
    .groupby('TEAM_ID')['GAME_DATE_EST']
    .diff()
    .dt.days
)
rest_median = games_log['REST_DAYS_TEAM'].median()
games_log['REST_DAYS_TEAM'] = games_log['REST_DAYS_TEAM'].fillna(rest_median)
games_log['B2B_TEAM'] = (games_log['REST_DAYS_TEAM'] == 1).astype(int)

stat_cols = ['PTS', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST', 'REB', 'WL']

# Shift(1) is CRITICAL to avoid data leakage
rolled_stats_l5 = games_log.groupby(['TEAM_ID', 'SEASON'])[stat_cols].apply(
    lambda x: x.shift(1).rolling(window=5, min_periods=2).mean()
)
rolled_stats_l5.columns = [f'{col}_L5' for col in stat_cols]

rolled_stats_l10 = games_log.groupby(['TEAM_ID', 'SEASON'])[stat_cols].apply(
    lambda x: x.shift(1).rolling(window=10, min_periods=3).mean()
)
rolled_stats_l10.columns = [f'{col}_L10' for col in stat_cols]

# Win streak (pre-game): consecutive wins immediately before current game
def _win_streak_pre_game(wl: pd.Series) -> pd.Series:
    wl_prev = wl.shift(1).fillna(0).astype(int)
    out = np.zeros(len(wl_prev), dtype=int)
    streak = 0
    for i, v in enumerate(wl_prev.values):
        if v == 1:
            streak += 1
        else:
            streak = 0
        out[i] = streak
    return pd.Series(out, index=wl.index)

games_log['WIN_STREAK'] = games_log.groupby(['TEAM_ID', 'SEASON'])['WL'].transform(_win_streak_pre_game)

# Align indexes before concatenation to avoid InvalidIndexError
games_log = games_log.reset_index(drop=True)
rolled_stats_l5 = rolled_stats_l5.reset_index(drop=True)
rolled_stats_l10 = rolled_stats_l10.reset_index(drop=True)

nba_ml_games = pd.concat([games_log, rolled_stats_l5, rolled_stats_l10], axis=1).dropna()
dataframes['games'] = nba_ml_games

# ------------------------------- Matchup Data (Individual + Differentials) ----------------------------------

print(">>> Creating Matchup Data (Individual Stats + Differentials)...")

df = nba_ml_games.copy()

# 1. Split Home and Away
df_home = df[df['IS_HOME'] == 1].copy()
df_away = df[df['IS_HOME'] == 0].copy()

# 2. Define columns to merge and calculate differences for
# We want rolling stats (L5 + L10) plus rest/back-to-back and win streak
stat_cols = [c for c in df.columns if ('_L5' in c or '_L10' in c)] + ['REST_DAYS_TEAM', 'B2B_TEAM', 'WIN_STREAK']

# 3. Rename Away columns to avoid collision during merge
away_rename = {col: f"{col}_OPP" for col in stat_cols}
away_rename['TEAM_ID'] = 'TEAM_ID_OPP'
df_away = df_away.rename(columns=away_rename)

# 4. Merge on GAME_ID (keeping both home and away stats)
nba_ml_matchups = pd.merge(
    df_home, 
    df_away[['GAME_ID'] + list(away_rename.values())], 
    on='GAME_ID', 
    how='inner'
)

# 5. Calculate Differentials (Home - Away) for all stat columns
diff_features = []
for col in stat_cols:
    diff_col = f"{col}_DIFF"
    nba_ml_matchups[diff_col] = nba_ml_matchups[col] - nba_ml_matchups[f"{col}_OPP"]
    diff_features.append(diff_col)

# 6. Select final columns: Info + Individual Home Stats + Opponent Stats + Differentials
# IS_HOME is omitted: it's always 1 in the matchup (every row is home team POV)
info_cols = ['GAME_DATE_EST', 'GAME_ID', 'SEASON', 'HOME_TEAM_ID', 'OPP_TEAM_ID', 'WL']
home_stat_cols = stat_cols  # Individual home team stats
opp_stat_cols = [f"{col}_OPP" for col in stat_cols]  # Opponent stats

# Final dataset has: info + home stats + opponent stats + differentials
final_matchup_cols = info_cols + home_stat_cols + opp_stat_cols + diff_features

nba_ml_matchups = nba_ml_matchups[final_matchup_cols]

# Add to dataframes dictionary to be saved
dataframes['matchups'] = nba_ml_matchups

# ------------------------------- Player Impact ----------------------------------
print(">>> Calculating Player Impact (Target Generation)...")

details = games_details.drop(columns=['NICKNAME', 'COMMENT', 'TEAM_CITY'])

# 1. Parse Minutes
def parse_minutes(min_str):
    if pd.isna(min_str): return 0.0
    if isinstance(min_str, (int, float)): return float(min_str)
    try:
        if ':' in min_str:
            m, s = min_str.split(':')
            return float(m) + float(s)/60
        return float(min_str)
    except:
        return 0.0

details['MIN_FLT'] = details['MIN'].apply(parse_minutes)

# 2. Fill NaNs
stat_cols_player = ['FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PF', 'PTS', 'PLUS_MINUS']
details[stat_cols_player] = details[stat_cols_player].fillna(0)

# 3. Add Game Context for Calculations
details = details.merge(games[['GAME_ID', 'GAME_DATE_EST', 'SEASON']], on='GAME_ID', how='left')

# ----------------------- PACE CALCULATION  -----------------------
# 1. Aggregation: Sum stats per Team per Game to get Team Totals
team_game_stats = details.groupby(['GAME_ID', 'TEAM_ID'])[['FGA', 'TO', 'FTA', 'OREB', 'MIN_FLT']].sum().reset_index()

# 2. Calculate Team Possessions (Basic Formula)
# Formula: 0.96 * (FGA + TOV + 0.44 * FTA - ORB)
team_game_stats['POSS'] = 0.96 * (team_game_stats['FGA'] + team_game_stats['TO'] + 0.44 * team_game_stats['FTA'] - team_game_stats['OREB'])

# 3. Calculate Opponent Possessions (The Step We Missed)
# By grouping by GAME_ID and summing POSS, we get (Team_POSS + Opp_POSS)
team_game_stats['GAME_POSS_TOTAL'] = team_game_stats.groupby('GAME_ID')['POSS'].transform('sum')

# Pace = 48 * ( (Tm Poss + Opp Poss) / (2 * (Tm MP / 5)) )
# Tm MP is 'MIN_FLT' (approx 240 mins in regulation)
team_minutes = team_game_stats['MIN_FLT'].replace(0, 240) 

team_game_stats['PACE'] = 48 * (team_game_stats['GAME_POSS_TOTAL'] / (2 * (team_minutes / 5)))

# 5. Merge Pace back to details
details = details.merge(team_game_stats[['GAME_ID', 'TEAM_ID', 'PACE']], on=['GAME_ID', 'TEAM_ID'], how='left')
details = details.rename(columns={'PACE': 'team_Pace'})

# ----------------------- SEASON AGGREGATES -----------------------

season_stats = details.groupby('SEASON').agg({
    'AST': 'mean', 'FGM': 'mean', 'FTM': 'mean', 'PTS': 'mean',
    'FGA': 'mean', 'OREB': 'mean', 'TO': 'mean', 'FTA': 'mean', 
    'REB': 'mean', 'PF': 'mean', 'team_Pace': 'mean'
}).rename(columns={
    'AST': 'lg_AST', 'FGM': 'lg_FG', 'FTM': 'lg_FT', 'PTS': 'lg_PTS',
    'FGA': 'lg_FGA', 'OREB': 'lg_ORB', 'TO': 'lg_TOV', 'FTA': 'lg_FTA',
    'REB': 'lg_TRB', 'PF': 'lg_PF', 'team_Pace': 'lg_Pace'
})

# Calculate PER constants
season_stats['factor'] = (2 / 3) - (0.5 * (season_stats['lg_AST'] / season_stats['lg_FG'])) / (2 * (season_stats['lg_FG'] / season_stats['lg_FT']))
season_stats['VOP']    = season_stats['lg_PTS'] / (season_stats['lg_FGA'] - season_stats['lg_ORB'] + season_stats['lg_TOV'] + 0.44 * season_stats['lg_FTA'])
season_stats['DRB_pct'] = (season_stats['lg_TRB'] - season_stats['lg_ORB']) / season_stats['lg_TRB']

details = details.merge(season_stats, on='SEASON', how='left')

# ----------------------- PER CALCULATION -----------------------
# Historical Team Context
details = details.sort_values(['TEAM_ID', 'GAME_DATE_EST'])
details['team_AST'] = details.groupby('TEAM_ID')['AST'].transform(lambda x: x.shift(1).expanding().mean())
details['team_FG']  = details.groupby('TEAM_ID')['FGM'].transform(lambda x: x.shift(1).expanding().mean())
details['team_AST'] = details['team_AST'].fillna(details['lg_AST']) 
details['team_FG']  = details['team_FG'].fillna(details['lg_FG'])

details['inv_MP'] = 1 / details['MIN_FLT']
details.loc[details['MIN_FLT'] == 0, 'inv_MP'] = 0

pace_adjustment = details['lg_Pace'] / details['team_Pace']

# uPER Formula
details['uPER'] = details['inv_MP'] * (
    details['FG3M'] 
    + (2/3) * details['AST'] 
    + (2 - details['factor'] * (details['team_AST'] / details['team_FG'])) * details['FGM'] 
    + (details['FTM'] * 0.5 * (1 + (1 - (details['team_AST'] / details['team_FG'])) + (2/3) * (details['team_AST'] / details['team_FG']))) 
    - details['VOP'] * details['TO']
    - details['VOP'] * details['DRB_pct'] * (details['FGA'] - details['FGM'])
    - details['VOP'] * 0.44 * (0.44 + (0.56 * details['DRB_pct'])) * (details['FTA'] - details['FTM'])
    + details['VOP'] * (1 - details['DRB_pct']) * (details['REB'] - details['OREB']) 
    + details['VOP'] * details['DRB_pct'] * details['OREB'] 
    + details['VOP'] * details['STL'] 
    + details['VOP'] * details['DRB_pct'] * details['BLK'] 
    - details['PF'] * ((details['lg_FT'] / details['lg_PF']) - 0.44 * (details['lg_FTA'] / details['lg_PF']) * details['VOP'])
)

details['aPER'] = details['uPER'] * pace_adjustment
details['lg_aPER'] = details.groupby('SEASON')['aPER'].transform('mean')
details['PER'] = details['aPER'] * (15 / details['lg_aPER'])

# ----------------------- GAME SCORE (For Robust MoM) -----------------------
details['GAME_SCORE'] = (
    details['PTS'] + 0.4 * details['FGM'] - 0.7 * details['FGA'] - 
    0.4 * (details['FTA'] - details['FTM']) + 0.7 * details['OREB'] + 
    0.3 * details['DREB'] + details['STL'] + 0.7 * details['AST'] + 
    0.7 * details['BLK'] - 0.4 * details['PF'] - details['TO']
)

# Clean up
details = details.drop(columns=['uPER', 'aPER', 'lg_aPER', 'inv_MP'])

# ----------------------- Player & Team Rest Features -----------------------

# Team-level rest: merge from games_log (per TEAM_ID & GAME_ID)
team_rest_cols = ['GAME_ID', 'TEAM_ID', 'REST_DAYS_TEAM', 'B2B_TEAM']
team_rest = games_log[team_rest_cols].drop_duplicates()
details = details.merge(team_rest, on=['GAME_ID', 'TEAM_ID'], how='left')

# Player-level rest: days since previous game for each player
details['GAME_DATE_EST'] = pd.to_datetime(details['GAME_DATE_EST'])
details = details.sort_values(['PLAYER_ID', 'GAME_DATE_EST'])
details['REST_DAYS_PLAYER'] = (
    details
    .groupby('PLAYER_ID')['GAME_DATE_EST']
    .diff()
    .dt.days
)
player_rest_median = details['REST_DAYS_PLAYER'].median()
details['REST_DAYS_PLAYER'] = details['REST_DAYS_PLAYER'].fillna(player_rest_median)
details['B2B_PLAYER'] = (details['REST_DAYS_PLAYER'] == 1).astype(int)

# Define Target: Man of the Match based on GAME_SCORE (Volume + Efficiency)
# Using GAME_SCORE is safer than PER for single-game ranking
details['GAME_RANK'] = details.groupby('GAME_ID')['GAME_SCORE'].rank(method='min', ascending=False)
details['IS_MOM'] = (details['GAME_RANK'] == 1).astype(int)

# Drop PER helper columns that are not directly useful as features
per_helper_cols = [
    'lg_AST', 'lg_FG', 'lg_FT', 'lg_PTS',
    'lg_FGA', 'lg_ORB', 'lg_TOV', 'lg_FTA',
    'lg_TRB', 'lg_PF', 'lg_Pace',
    'factor', 'VOP', 'DRB_pct'
]
details = details.drop(columns=[c for c in per_helper_cols if c in details.columns])

dataframes['details'] = details

# ------------------------------------- Save -------------------------------------
print(">>> Saving to Disk...")

for name, df in dataframes.items():
    df.to_csv(out_dir / f"nba_ml_{name}.csv", index=False)
    print(f"Saved: {out_dir / f'nba_ml_{name}.csv'} ({len(df)} rows)")

sample_size = 100
heads_dir = out_dir / "heads"
heads_dir.mkdir(parents=True, exist_ok=True)

for name, df in dataframes.items():
    df_head = df.sample(n=min(sample_size, len(df)), random_state=42)
    df_head.to_csv(heads_dir / f"{name}_head.csv", index=False)
    print(f"Saved: {heads_dir / f'{name}_head.csv'}")