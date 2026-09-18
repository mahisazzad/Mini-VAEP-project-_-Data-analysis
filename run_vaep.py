print("--- INITIALIZING VAEP PIPELINE ---")

import sys
import os
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. NUMPY COMPATIBILITY PATCH
# ==============================================================================
import numpy as np
np_patches = {
    "float_": np.float64,
    "string_": np.bytes_,
    "int_": np.int64,
    "bool_": np.bool_,
    "object_": np.object_,
    "complex_": np.complex128,
}
for attr, replacement in np_patches.items():
    if not hasattr(np, attr):
        setattr(np, attr, replacement)

import pandas as pd
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import requests
from torch.utils.data import DataLoader, TensorDataset

import socceraction.spadl as spadl
import socceraction.vaep.features as fs
import socceraction.vaep.labels as lab
import socceraction.vaep.formula as vaep_formula

# Handle Socceraction StatsBomb Loader Imports
try:
    from socceraction.data.statsbomb import StatsBombLoader
    USE_SBLoader = True
except ImportError:
    USE_SBLoader = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Setup complete! Running on device: {device}")

# Cache Paths
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "statsbomb_data"))
os.makedirs(os.path.join(CACHE_DIR, "events"), exist_ok=True)
os.makedirs(os.path.join(CACHE_DIR, "matches"), exist_ok=True)
os.makedirs(os.path.join(CACHE_DIR, "lineups"), exist_ok=True)

RAW_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

# ==============================================================================
# DATA RETRIEVAL ENGINE
# ==============================================================================
def download_and_cache(endpoint, local_rel_path):
    cache_path = os.path.join(CACHE_DIR, local_rel_path)
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data:
                    return True
        except Exception:
            pass
        os.remove(cache_path)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    url = f"{RAW_BASE}/{endpoint}"
    try:
        res = requests.get(url, headers=HEADERS, timeout=15)
        if res.status_code == 200 and len(res.text) > 10:
            parsed = json.loads(res.text)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(parsed, f)
            return True
    except Exception:
        pass
    return False

def parse_events_file(game_id, home_team_id, sb_loader=None):
    if USE_SBLoader and sb_loader is not None:
        try:
            events_df = sb_loader.events(game_id)
            actions = spadl.statsbomb.convert_to_actions(events_df, home_team_id)
            actions['game_id'] = game_id
            return actions, None
        except Exception as e:
            return None, f"Game {game_id}: {str(e)}"

    cache_path = os.path.join(CACHE_DIR, f"events/{game_id}.json")
    if not os.path.exists(cache_path):
        return None, f"File missing: {cache_path}"
        
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            raw_events = json.load(f)
            
        if not raw_events:
            return None, f"Empty JSON: {game_id}"

        actions = spadl.statsbomb.convert_to_actions(raw_events, home_team_id)
        actions['game_id'] = game_id
        return actions, None
    except Exception as e:
        return None, f"Game {game_id}: {str(e)}"

def process_matches(games_df):
    spadl_actions = []
    errors = []
    
    download_tasks = [(f"events/{g['game_id']}.json", f"events/{g['game_id']}.json") for _, g in games_df.iterrows()]

    print("\nChecking and downloading event files...")
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(download_and_cache, ep, lp) for ep, lp in download_tasks]
        for _ in tqdm.tqdm(as_completed(futures), total=len(download_tasks), desc="Downloading Events"):
            pass

    games_list = games_df.to_dict('records')
    sb_loader = StatsBombLoader(getter="local", root=CACHE_DIR) if USE_SBLoader else None
    
    print("Converting events to SPADL actions...")
    for g in tqdm.tqdm(games_list, desc="Parsing SPADL"):
        res, err = parse_events_file(g['game_id'], g['home_team_id'], sb_loader)
        if res is not None:
            spadl_actions.append(res)
        else:
            errors.append(err)

    if not spadl_actions:
        print("\n--- SAMPLE PARSING ERRORS ---")
        for err in list(set(errors))[:5]:
            print(f"- {err}")
        raise ValueError("Failed to extract valid SPADL actions across matches.")

    print(f"Successfully converted {len(spadl_actions)} / {len(games_df)} matches into SPADL format.")
    df_actions = pd.concat(spadl_actions, ignore_index=True)

    # Attach string names required by labels module
    if hasattr(spadl, 'add_names'):
        df_actions = spadl.add_names(df_actions)
    else:
        type_df = pd.DataFrame(spadl.config.actiontypes, columns=['type_id', 'type_name'])
        result_df = pd.DataFrame(spadl.config.results, columns=['result_id', 'result_name'])
        bodypart_df = pd.DataFrame(spadl.config.bodyparts, columns=['bodypart_id', 'bodypart_name'])
        
        if 'type_name' not in df_actions.columns:
            df_actions = df_actions.merge(type_df, on='type_id', how='left')
        if 'result_name' not in df_actions.columns:
            df_actions = df_actions.merge(result_df, on='result_id', how='left')
        if 'bodypart_name' not in df_actions.columns:
            df_actions = df_actions.merge(bodypart_df, on='bodypart_id', how='left')

    print("Building gamestates and extracting features...")
    gamestates = fs.gamestates(df_actions, nb_prev_actions=2)

    feature_fns = [
        fs.actiontype_onehot,
        fs.bodypart_onehot,
        fs.result_onehot,
        fs.startlocation,
        fs.endlocation,
        fs.movement,
        fs.space_delta,
    ]

    X = pd.concat([fn(gamestates) for fn in feature_fns], axis=1)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')

    print("Generating labels...")
    Y_scores = lab.scores(df_actions, nr_actions=10).values.flatten().astype('float32')
    Y_concedes = lab.concedes(df_actions, nr_actions=10).values.flatten().astype('float32')

    return df_actions, X, Y_scores, Y_concedes

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
if __name__ == "__main__":
    
    # --- PHASE 1: LOAD & PROCESS TRAIN DATASET ---
    print("\nLoading Training Matches...")
    train_tournaments = [
        (43, 3),   # World Cup 2018
        (55, 43),  # Euro 2020
        (16, 4),   # Champions League 2018/19
        (11, 1),   # La Liga 2017/18
        (2, 44),   # Premier League 2003/04
    ]

    train_games_list = []
    for comp_id, season_id in train_tournaments:
        rel_path = f"matches/{comp_id}/{season_id}.json"
        if download_and_cache(rel_path, rel_path):
            with open(os.path.join(CACHE_DIR, rel_path), 'r', encoding='utf-8') as f:
                matches_raw = json.load(f)
                for m in matches_raw:
                    train_games_list.append({
                        'game_id': m['match_id'],
                        'home_team_id': m['home_team']['home_team_id']
                    })

    train_games = pd.DataFrame(train_games_list).drop_duplicates(subset=['game_id'])
    print(f"Total training matches registered: {len(train_games)}")
    
    _, X_train, Y_scores_train, Y_concedes_train = process_matches(train_games)

    # --- PHASE 2: PYTORCH MODEL TRAINING ---
    print("\nTraining PyTorch Deep Learning Models (P_scores & P_concedes)...")

    class VAEPNeuralNet(nn.Module):
        def __init__(self, input_dim):
            super(VAEPNeuralNet, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x)

    # FIX: Ensure exact [N, 1] shape alignment for BCELoss
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_scores_tensor = torch.tensor(Y_scores_train, dtype=torch.float32).reshape(-1, 1)
    y_concedes_tensor = torch.tensor(Y_concedes_train, dtype=torch.float32).reshape(-1, 1)

    train_loader_scores = DataLoader(TensorDataset(X_train_tensor, y_scores_tensor), batch_size=512, shuffle=True)
    train_loader_concedes = DataLoader(TensorDataset(X_train_tensor, y_concedes_tensor), batch_size=512, shuffle=True)

    def train_nn(train_loader, input_dim, name="Model"):
        model = VAEPNeuralNet(input_dim).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        model.train()
        for epoch in range(8):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        print(f"[{name}] Training completed.")
        return model

    model_scores = train_nn(train_loader_scores, X_train.shape[1], name="P_scores")
    model_concedes = train_nn(train_loader_concedes, X_train.shape[1], name="P_concedes")

    # --- PHASE 3: EVALUATION ON TEST DATASET ---
    print("\nLoading Test Matches (La Liga 2018/19)...")
    rel_test_path = "matches/11/4.json"
    download_and_cache(rel_test_path, rel_test_path)
    
    with open(os.path.join(CACHE_DIR, rel_test_path), 'r', encoding='utf-8') as f:
        test_matches_raw = json.load(f)

    test_games_list = [{
        'game_id': m['match_id'],
        'home_team_id': m['home_team']['home_team_id']
    } for m in test_matches_raw]

    test_games = pd.DataFrame(test_games_list).drop_duplicates(subset=['game_id'])
    print(f"Total test matches registered: {len(test_games)}")

    df_actions_test, X_test, _, _ = process_matches(test_games)

    print("\nPredicting probabilities and calculating VAEP values...")
    model_scores.eval()
    model_concedes.eval()

    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
        df_actions_test['P_scores'] = model_scores(X_test_tensor).cpu().numpy().flatten()
        df_actions_test['P_concedes'] = model_concedes(X_test_tensor).cpu().numpy().flatten()

    df_vaep = vaep_formula.value(df_actions_test, df_actions_test['P_scores'], df_actions_test['P_concedes'])
    df_test_full = pd.concat([df_actions_test, df_vaep], axis=1)

    # Attach Player Names from Lineup Files
    print("Mapping player metadata...")
    players_dict = {}
    for game_id in test_games['game_id'].unique():
        lineup_path = f"lineups/{game_id}.json"
        if download_and_cache(lineup_path, lineup_path):
            try:
                with open(os.path.join(CACHE_DIR, lineup_path), 'r', encoding='utf-8') as f:
                    lineup_raw = json.load(f)
                for team in lineup_raw:
                    for player in team.get('lineup', []):
                        players_dict[player['player_id']] = player['player_name']
            except Exception:
                pass

    players_df = pd.DataFrame(list(players_dict.items()), columns=['player_id', 'player_name'])
    df_test_full = df_test_full.merge(players_df, on='player_id', how='left')

   # --- PHASE 4: DETAILED ANALYTICS (SCORES / CONCEDES & PER 90 METRICS) ---
    print("\nCalculating player playing time for Per 90 normalization...")

    # Identify exact score/concedes column names returned by vaep_formula
    score_col = 'offensive_value' if 'offensive_value' in df_test_full.columns else ('scores_value' if 'scores_value' in df_test_full.columns else 'scores')
    concede_col = 'defensive_value' if 'defensive_value' in df_test_full.columns else ('concedes_value' if 'concedes_value' in df_test_full.columns else 'concedes')

    # Calculate actual minutes played per player across test games
    player_minutes = {}
    
    for game_id in test_games['game_id'].unique():
        event_path = f"events/{game_id}.json"
        full_event_path = os.path.join(CACHE_DIR, event_path)
        if os.path.exists(full_event_path):
            with open(full_event_path, 'r', encoding='utf-8') as f:
                events_data = json.load(f)
                
            last_event = events_data[-1] if events_data else {}
            match_mins = last_event.get('minute', 90) + (last_event.get('second', 0) / 60.0)

            player_starts = {}
            for ev in events_data:
                p_id = ev.get('player', {}).get('id')
                ev_type = ev.get('type', {}).get('name')
                
                if ev_type == 'Starting XI':
                    for p in ev.get('tactics', {}).get('lineup', []):
                        player_starts[p['player']['id']] = 0.0

                elif ev_type == 'Substitution':
                    sub_out_id = p_id
                    sub_in_id = ev.get('substitution', {}).get('replacement', {}).get('id')
                    sub_time = ev.get('minute', 0) + (ev.get('second', 0) / 60.0)

                    if sub_out_id in player_starts:
                        played = sub_time - player_starts.pop(sub_out_id)
                        player_minutes[sub_out_id] = player_minutes.get(sub_out_id, 0.0) + played
                    
                    if sub_in_id:
                        player_starts[sub_in_id] = sub_time

            for p_id, start_t in player_starts.items():
                player_minutes[p_id] = player_minutes.get(p_id, 0.0) + (match_mins - start_t)

    df_mins = pd.DataFrame(list(player_minutes.items()), columns=['player_id', 'minutes_played'])

    # Aggregate VAEP scores & concedes robustly
    player_stats = (
        df_test_full.groupby(['player_id', 'player_name'])
        .agg(
            total_vaep=('vaep_value', 'sum'),
            offensive_vaep=(score_col, 'sum'),
            defensive_vaep=(concede_col, 'sum'),
            total_actions=('game_id', 'count')
        )
        .reset_index()
    )

    player_stats = player_stats.merge(df_mins, on='player_id', how='left')
    player_stats['minutes_played'] = player_stats['minutes_played'].fillna(0)

    # Filter out low sample size (min 450 minutes)
    MIN_MINUTES = 450
    df_filtered = player_stats[player_stats['minutes_played'] >= MIN_MINUTES].copy()

    # Calculate Per 90 metrics
    df_filtered['vaep_p90'] = (df_filtered['total_vaep'] / df_filtered['minutes_played']) * 90
    df_filtered['offense_p90'] = (df_filtered['offensive_vaep'] / df_filtered['minutes_played']) * 90
    df_filtered['defense_p90'] = (df_filtered['defensive_vaep'] / df_filtered['minutes_played']) * 90

    print("\n" + "="*85)
    print(" TOP 10 PLAYERS BY VAEP PER 90 MINUTES (LA LIGA 2018/19 - MIN 450 MINS)")
    print("="*85 + "\n")

    top_p90 = (
        df_filtered.sort_values(by='vaep_p90', ascending=False)
        .head(10)[
            ['player_name', 'minutes_played', 'total_vaep', 'vaep_p90', 'offense_p90', 'defense_p90']
        ]
    )

    top_p90.columns = ['Player Name', 'Mins', 'Total VAEP', 'VAEP/90', 'Offense/90', 'Defense/90']
    print(top_p90.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


 # --- REMAINING RESULTS REQUIREMENTS ---

# Ensure we reference correct column names returned by vaep_formula
off_col = 'offensive_value' if 'offensive_value' in df_test_full.columns else 'scores'
def_col = 'defensive_value' if 'defensive_value' in df_test_full.columns else 'concedes'

# 1. Top 10 Individual Actions (Non-Penalty)
df_non_penalty = df_test_full[df_test_full['type_name'] != 'shot_penalty'].copy()
top_10_actions = (
    df_non_penalty.sort_values(by='vaep_value', ascending=False)
    .head(10)[['player_name', 'type_name', 'result_name', 'vaep_value', off_col, def_col]]
)

print("\n" + "="*75)
print(" TOP 10 INDIVIDUAL ACTIONS (NON-PENALTY)")
print("="*75)
print(top_10_actions.to_string(index=False))


# 2. Top 10 Defensive Actions
# Sort by highest defensive contribution (defensive_value or lower P_concedes contribution)
top_10_defensive = (
    df_test_full.sort_values(by=def_col, ascending=False)
    .head(10)[['player_name', 'type_name', 'result_name', def_col, 'vaep_value']]
)

print("\n" + "="*75)
print(" TOP 10 DEFENSIVE ACTIONS")
print("="*75)
print(top_10_defensive.to_string(index=False))


# 3. Traditional Stats Comparison (G, A, Tackle, Interception vs VAEP)
df_test_full['is_goal'] = (df_test_full['type_name'] == 'shot') & (df_test_full['result_name'] == 'success')
df_test_full['is_tackle'] = df_test_full['type_name'].str.contains('tackle', case=False, na=False) & (df_test_full['result_name'] == 'success')
df_test_full['is_interception'] = df_test_full['type_name'].str.contains('interception', case=False, na=False) & (df_test_full['result_name'] == 'success')

trad_stats = (
    df_test_full.groupby(['player_id', 'player_name'])
    .agg(
        Total_VAEP=('vaep_value', 'sum'),
        Offensive_VAEP=(off_col, 'sum'),
        Defensive_VAEP=(def_col, 'sum'),
        Goals=('is_goal', 'sum'),
        Tackles=('is_tackle', 'sum'),
        Interceptions=('is_interception', 'sum')
    )
    .reset_index()
    .sort_values(by='Total_VAEP', ascending=False)
    .head(10)
)

print("\n" + "="*85)
print(" VAEP VS TRADITIONAL STATS COMPARISON (TOP 10 PLAYERS)")
print("="*85)
print(trad_stats.to_string(index=False, float_format=lambda x: f"{x:.3f}"))