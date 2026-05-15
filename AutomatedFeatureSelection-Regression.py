import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import os

os.makedirs('relevance', exist_ok=True)
os.makedirs('datasetRoot', exist_ok=True)

# ============================================================
# Funzione campionamento
# ============================================================
def sample_dataset(df, max_rows, random_state=42, dataset_name='dataset'):
    if max_rows is None or len(df) <= max_rows:
        print(f"  [INFO] {dataset_name}: {len(df)} righe, nessun campionamento")
        return df
    df_sampled = df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
    print(f"  [INFO] {dataset_name}: campionato {max_rows} righe da {len(df)} originali")
    return df_sampled

# ============================================================
# Funzione generica (tutti i dataset tranne Superconductivity)
# ============================================================
def select_features(df, target_col, n_top=3, n_middle=3, n_low=3,
                    random_state=42, dataset_name='dataset'):

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # rimuovi feature con varianza quasi zero
    low_var_cols = [col for col in X.columns if X[col].std() < 1e-6]
    if low_var_cols:
        print(f"  [WARN] Rimosse {len(low_var_cols)} feature a varianza zero: {low_var_cols}")
        X = X.drop(columns=low_var_cols)

    # solo numeriche
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_num = X[numeric_cols]

    X_train, _, y_train, _ = train_test_split(
        X_num, y, test_size=0.2, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    rf = RandomForestRegressor(
        n_estimators=200, max_features='sqrt',
        min_samples_leaf=5, random_state=random_state, n_jobs=-1
    )
    rf.fit(X_train_s, y_train)

    importance_df = pd.DataFrame({
        'feature':    numeric_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    importance_df.to_csv(
        f'relevance/{dataset_name}_feature_importance.csv', index=False
    )

    n_features = len(importance_df)
    mid_start  = (n_features // 2) - (n_middle // 2)

    top_features    = importance_df.head(n_top)['feature'].tolist()
    middle_features = importance_df.iloc[mid_start:mid_start+n_middle]['feature'].tolist()
    low_features    = importance_df.tail(n_low)['feature'].tolist()

    print(f"  TOP    {n_top}: {top_features}")
    print(f"  MIDDLE {n_middle}: {middle_features}")
    print(f"  LOW    {n_low}: {low_features}")

    return {
        'top':          top_features,
        'middle':       middle_features,
        'low':          low_features,
        'all_selected': top_features + middle_features + low_features,
    }

# ============================================================
# Funzione specializzata per Superconductivity
# ============================================================
def select_features_superconductivity(df, target_col='critical_temp',
                                       n_top=3, n_middle=3, n_low=3,
                                       random_state=42):
    properties = [
        'atomic_mass', 'fie', 'atomic_radius', 'Density',
        'ElectronAffinity', 'FusionHeat', 'ThermalConductivity', 'Valence'
    ]
    standalone = ['number_of_elements']

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    rf = RandomForestRegressor(
        n_estimators=200, max_features='sqrt',
        min_samples_leaf=5, random_state=random_state, n_jobs=-1
    )
    rf.fit(X_train_s, y_train)

    importance_df = pd.DataFrame({
        'feature':    X.columns.tolist(),
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    importance_df.to_csv(
        'relevance/superconductivity_feature_importance.csv', index=False
    )

    # best feature per ogni proprietà
    best_per_property = {}
    for prop in properties:
        prop_features = importance_df[
            importance_df['feature'].str.contains(prop, regex=False)
        ]
        if not prop_features.empty:
            best = prop_features.iloc[0]
            best_per_property[prop] = {
                'feature':    best['feature'],
                'importance': float(best['importance'])
            }
    for feat in standalone:
        row = importance_df[importance_df['feature'] == feat]
        if not row.empty:
            best_per_property[feat] = {
                'feature':    feat,
                'importance': float(row.iloc[0]['importance'])
            }

    # ordina per importance
    ranked = sorted(
        best_per_property.items(),
        key=lambda x: x[1]['importance'],
        reverse=True
    )

    print("\n  Ranking proprietà (best feature per gruppo):")
    for i, (prop, info) in enumerate(ranked):
        print(f"    {i+1:2d}. {prop:25s} → {info['feature']:35s} imp={info['importance']:.4f}")

    n_props  = len(ranked)
    mid_start = (n_props // 2) - (n_middle // 2)

    top_features    = [p[1]['feature'] for p in ranked[:n_top]]
    middle_features = [p[1]['feature'] for p in ranked[mid_start:mid_start+n_middle]]
    low_features    = [p[1]['feature'] for p in ranked[-n_low:]]

    print(f"\n  TOP    {n_top}: {top_features}")
    print(f"  MIDDLE {n_middle}: {middle_features}")
    print(f"  LOW    {n_low}: {low_features}")

    return {
        'top':          top_features,
        'middle':       middle_features,
        'low':          low_features,
        'all_selected': top_features + middle_features + low_features,
    }

# ============================================================
# Configurazione dataset
# ============================================================
datasets_config = {
    'california_housing': {
        'path':     'datasetRoot/california_housing.csv',
        'target':   'MedHouseVal',
        'max_rows': 5000,
    },
    'concrete_strength': {
        'path':     'datasetRoot/concrete_strength.csv',
        'target':   None,  # verrà rilevato automaticamente (ultima colonna)
        'max_rows': None,
    },
    'bike_sharing': {
        'path':     'datasetRoot/bike_sharing.csv',
        'target':   'cnt',
        'max_rows': 5000,
    },
    'superconductivity': {
        'path':     'datasetRoot/superconductivity.csv',
        'target':   'critical_temp',
        'max_rows': 5000,
    },
}

# ============================================================
# Esecuzione
# ============================================================
all_features = {}

for name, cfg in datasets_config.items():
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")

    if not os.path.exists(cfg['path']):
        print(f"  [SKIP] File non trovato: {cfg['path']}")
        continue

    df = pd.read_csv(cfg['path'])
    print(f"  Dimensioni originali: {df.shape}")

    # rileva target se non specificato (usa ultima colonna)
    target_col = cfg['target'] if cfg['target'] else df.columns[-1]
    print(f"  Target: '{target_col}'")

    # campionamento
    df_work = sample_dataset(df, cfg['max_rows'], dataset_name=name)

    # salva versione campionata se diversa dall'originale
    if cfg['max_rows'] is not None and len(df) > cfg['max_rows']:
        sampled_path = cfg['path'].replace('.csv', '_sampled.csv')
        df_work.to_csv(sampled_path, index=False)
        print(f"  Campionato salvato in: {sampled_path}")

    # selezione feature
    if name == 'superconductivity':
        features = select_features_superconductivity(
            df_work, target_col=target_col
        )
    else:
        features = select_features(
            df_work, target_col=target_col, dataset_name=name
        )

    all_features[name] = {
        'path':         cfg['path'],
        'sampled_path': cfg['path'].replace('.csv', '_sampled.csv') if cfg['max_rows'] else cfg['path'],
        'target':       target_col,
        'max_rows':     cfg['max_rows'],
        'top':          features['top'],
        'middle':       features['middle'],
        'low':          features['low'],
        'all_selected': features['all_selected'],
    }
# ============================================================
# Superconductivity: crea versione ridotta (righe + colonne)
# ============================================================
df_super = pd.read_csv('datasetRoot/superconductivity.csv')

# campionamento righe
df_super_sampled = df_super.sample(n=5000, random_state=42).reset_index(drop=True)

# recupera le feature selezionate dal summary
with open('relevance/feature_selection_summary.json', 'r') as f:
    summary = json.load(f)

selected_features = summary['superconductivity']['all_selected']
target_col = summary['superconductivity']['target']

# tieni solo le 9 feature selezionate + target
cols_to_keep = selected_features + [target_col]
df_super_reduced = df_super_sampled[cols_to_keep]

# salva
reduced_path = 'datasetRoot/superconductivity_sampled.csv'
df_super_reduced.to_csv(reduced_path, index=False)

print(f"\nSuperconductivity ridotto:")
print(f"  Righe:   {len(df_super_reduced)}")
print(f"  Colonne: {list(df_super_reduced.columns)}")
# ============================================================
# Riepilogo finale
# ============================================================
print(f"\n{'='*60}")
print("RIEPILOGO FEATURE SELEZIONATE")
print(f"{'='*60}")
for name, info in all_features.items():
    print(f"\n{name}  (target: {info['target']})")
    print(f"  top:    {info['top']}")
    print(f"  middle: {info['middle']}")
    print(f"  low:    {info['low']}")

# salva JSON riepilogativo
summary_path = 'relevance/feature_selection_summary.json'
with open(summary_path, 'w') as f:
    json.dump(all_features, f, indent=4)
print(f"\nRiepilogo salvato in '{summary_path}'")