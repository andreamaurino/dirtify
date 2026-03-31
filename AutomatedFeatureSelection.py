import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# ============================================================
# 1) Laplacian Score
# ============================================================
from sklearn.metrics import pairwise_distances

def laplacian_score(X, k_neighbors=5, max_samples=2000, random_state=42):

    n_samples, n_features = X.shape

    if n_samples > max_samples:
        print(f"  Campiono {max_samples} righe per Laplacian Score")
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n_samples, max_samples, replace=False)
        X = X[idx]
        n_samples = max_samples

    print(f"  Calcolo matrice distanze ({n_samples}x{n_samples})...", flush=True)
    
    # implementazione manuale — evita kneighbors_graph e threadpoolctl
    dist_matrix = pairwise_distances(X, metric='euclidean')
    
    W = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        neighbors = np.argsort(dist_matrix[i])[1:k_neighbors+1]
        W[i, neighbors] = 1
        W[neighbors, i] = 1

    D = W.sum(axis=1)
    D_mat = np.diag(D)
    L = D_mat - W

    scores = np.zeros(n_features)
    for i in range(n_features):
        f = X[:, i]
        f_mean = (D * f).sum() / D.sum()
        f_centered = f - f_mean
        numerator   = f_centered @ L @ f_centered
        denominator = f_centered @ D_mat @ f_centered
        scores[i] = numerator / denominator if denominator > 1e-10 else np.inf

    return scores
# ============================================================
# 2) Correlation filter
# ============================================================
def correlation_filter(X_df, selected_features, threshold=0.85):
    """
    Tra feature altamente correlate, tieni quella con 
    Laplacian Score più basso (più rilevante).
    
    X_df:              DataFrame con tutte le feature
    selected_features: lista ordinata per Laplacian Score (asc)
    threshold:         soglia di correlazione per rimozione
    """
    corr_matrix = X_df[selected_features].corr().abs()
    to_keep = []
    to_drop = set()

    for feat in selected_features:  # già ordinata per score
        if feat in to_drop:
            continue
        to_keep.append(feat)
        # marca come ridondanti le feature correlate
        correlated = corr_matrix[feat][
            (corr_matrix[feat] > threshold) & 
            (corr_matrix[feat].index != feat)
        ].index.tolist()
        to_drop.update(correlated)

    return to_keep

def stratified_sample(df, target_col, sample_size=10000, random_state=42):
    """
    Campione stratificato che preserva le proporzioni delle classi.
    """
    n_total = len(df)
    frac = sample_size / n_total

    if frac >= 1.0:
        print(f"Dataset già piccolo ({n_total} righe) — nessun campionamento")
        return df

    print(f"Campionamento stratificato: {n_total} → {sample_size} righe "
          f"({frac*100:.1f}%)")

    sample, _ = train_test_split(
        df,
        train_size=sample_size,
        stratify=df[target_col],
        random_state=random_state
    )

    # verifica proporzioni
    print("\nProporzioni classi nel campione vs originale:")
    orig_props   = df[target_col].value_counts(normalize=True)
    sample_props = sample[target_col].value_counts(normalize=True)
    check = pd.DataFrame({
        'originale': orig_props,
        'campione':  sample_props
    }).round(3)
    print(check)

    return sample.reset_index(drop=True)
# ============================================================
# 3) Selezione stratificata
# ============================================================
def stratified_feature_selection(X_df, laplacian_scores, feature_names,
                                  n_total=12, 
                                  top_frac=0.50, 
                                  mid_frac=0.30, 
                                  bot_frac=0.20,
                                  corr_threshold=0.85,
                                  random_state=42):
    """
    Seleziona feature da tre strati del ranking Laplacian Score:
      - Top (score basso)  → certamente rilevanti
      - Mid                → esplorazione
      - Bottom (score alto)→ ricerca sorprese
    
    n_total:      numero target di feature finali
    top/mid/bot_frac: proporzione da ogni strato
    """
    np.random.seed(random_state)

    n_features = len(feature_names)
    
    # ranking per Laplacian Score (score basso = più rilevante)
    ranking = np.argsort(laplacian_scores)
    ranked_features = [feature_names[i] for i in ranking]

    # dimensioni degli strati
    n_top = int(n_features * top_frac)
    n_mid = int(n_features * mid_frac)
    # bottom = tutto il resto

    top_pool = ranked_features[:n_top]
    mid_pool = ranked_features[n_top:n_top + n_mid]
    bot_pool = ranked_features[n_top + n_mid:]

    # quante feature da ogni strato
    n_from_top = max(1, int(n_total * top_frac))
    n_from_mid = max(1, int(n_total * mid_frac))
    n_from_bot = max(1, n_total - n_from_top - n_from_mid)

    print(f"\nPool disponibili:")
    print(f"  Top    ({top_frac*100:.0f}%): {len(top_pool)} feature → seleziono {n_from_top}")
    print(f"  Middle ({mid_frac*100:.0f}%): {len(mid_pool)} feature → seleziono {n_from_mid}")
    print(f"  Bottom ({bot_frac*100:.0f}%): {len(bot_pool)} feature → seleziono {n_from_bot}")

    # selezione: top = migliori, mid e bot = campionamento random
    selected_top = top_pool[:n_from_top]
    selected_mid = list(np.random.choice(mid_pool, 
                                          min(n_from_mid, len(mid_pool)), 
                                          replace=False)) if mid_pool else []
    selected_bot = list(np.random.choice(bot_pool, 
                                          min(n_from_bot, len(bot_pool)), 
                                          replace=False)) if bot_pool else []

    # unisci mantenendo ordine per score
    all_selected = selected_top + selected_mid + selected_bot

    # applica correlation filter (priorità alle feature con score migliore)
    all_selected_ordered = sorted(all_selected, 
                                   key=lambda f: laplacian_scores[feature_names.index(f)])
    final_features = correlation_filter(X_df, all_selected_ordered, corr_threshold)

    return final_features, {
        'top': selected_top,
        'mid': selected_mid,
        'bot': selected_bot,
        'after_corr_filter': final_features
    }


# ============================================================
# 4) Report e plot
# ============================================================
def plot_feature_selection(feature_names, laplacian_scores, 
                            final_features, strata):
    """
    Visualizza il ranking Laplacian Score con evidenziati
    gli strati e le feature selezionate.
    """
    ranking = np.argsort(laplacian_scores)
    ranked_names  = [feature_names[i] for i in ranking]
    ranked_scores = [laplacian_scores[i] for i in ranking]

    colors = []
    for f in ranked_names:
        if f in strata['top'] and f in final_features:
            colors.append('steelblue')    # top selezionata
        elif f in strata['mid'] and f in final_features:
            colors.append('orange')       # mid selezionata
        elif f in strata['bot'] and f in final_features:
            colors.append('red')          # bottom selezionata — sorpresa
        else:
            colors.append('lightgray')    # non selezionata

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(ranked_names)), ranked_scores, color=colors)
    ax.set_xticks(range(len(ranked_names)))
    ax.set_xticklabels(ranked_names, rotation=90, fontsize=8)
    ax.set_ylabel('Laplacian Score (lower = more relevant)')
    ax.set_title('Feature Selection — Laplacian Score + Stratified Sampling')

    # legend
    from matplotlib.patches import Patch
    legend = [
        Patch(color='steelblue', label='Top selected'),
        Patch(color='orange',    label='Mid selected'),
        Patch(color='red',       label='Bottom selected (surprise)'),
        Patch(color='lightgray', label='Not selected'),
    ]
    ax.legend(handles=legend, loc='upper left')
    plt.tight_layout()
    plt.savefig('feature_selection.png', dpi=150, bbox_inches='tight')
    plt.show()

def recommended_sample_size(n_features, n_clusters, min_per_cluster=500):
    """
    Regola empirica: almeno 500 campioni per cluster
    e almeno 100 campioni per feature.
    """
    by_cluster  = n_clusters * min_per_cluster
    by_features = n_features * 100
    recommended = max(by_cluster, by_features)
    
    print(f"Dimensione campione consigliata:")
    print(f"  Per cluster  ({n_clusters} × {min_per_cluster}): {by_cluster}")
    print(f"  Per feature  ({n_features} × 100):              {by_features}")
    print(f"  Consigliata:                                     {recommended}")
    
    return recommended

# ============================================================
# 5) MAIN
# ============================================================
def select_features(csv_path,
                    target_col="covert_type",
                    n_total=16,
                    k_neighbors=5,
                    corr_threshold=0.85,
                    top_frac=0.50,
                    mid_frac=0.30,
                    bot_frac=0.20,
                    random_state=42,
                    max_samples=10000,   # soglia per campionamento
                    plot=True):

    df = pd.read_csv(csv_path)

    # rimuovi colonne completamente NaN (virgola trailing nel CSV)
    cols_before = df.shape[1]
    df = df.dropna(axis=1, how='all')
    cols_after = df.shape[1]
    if cols_before != cols_after:
        print(f"  Rimosse {cols_before - cols_after} colonne vuote")

    # strip spazi dai nomi colonna
    df.columns = df.columns.str.strip()

    feature_cols = df.columns[:-1].tolist()

    print(f"  Colonne totali: {df.shape[1]}")
    print(f"  Target: '{target_col}'")
    print(f"  Valori unici target: {df[target_col].nunique()}")
    print(f"  NaN nel target: {df[target_col].isna().sum()}")
 
    # -- campionamento stratificato se necessario -------------
    n_clusters = df[target_col].nunique()
    print(f"  Cluster:  {n_clusters}")

    sample_size = recommended_sample_size(len(feature_cols), n_clusters)
    sample_size = min(sample_size, max_samples)  # cap a max_samples

    if len(df) > sample_size:
        df_work = stratified_sample(df, target_col, sample_size, random_state)
    else:
        df_work = df.copy()

    # -- salva campione per Dirtify ---------------------------
    sample_path = csv_path.replace('.csv', f'_sample{len(df_work)}.csv')
    df_work.to_csv(sample_path, index=False)
    print(f"\nCampione salvato in: {sample_path}")
    print("→ Usa questo file per gli esperimenti ESP invece del dataset completo")

    # -- feature selection sul campione -----------------------
    X_df = df_work[feature_cols].copy()

    # variance filter
    var = X_df.var()
    zero_var = var[var < 1e-10].index.tolist()
    if zero_var:
        print(f"\nRimosse {len(zero_var)} feature a varianza zero")
        X_df = X_df.drop(columns=zero_var)
        feature_cols = X_df.columns.tolist()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    # Laplacian Score sul campione
    print(f"\nCalcolo Laplacian Score su {len(df_work)} campioni...")
    scores = laplacian_score(X_scaled, k_neighbors=k_neighbors)

    # selezione stratificata
    final_features, strata = stratified_feature_selection(
        X_df, scores, feature_cols,
        n_total=n_total,
        top_frac=top_frac,
        mid_frac=mid_frac,
        bot_frac=bot_frac,
        corr_threshold=corr_threshold,
        random_state=random_state
    )

    # -- report -----------------------------------------------
    print(f"\n{'='*60}")
    print(f"FEATURE SELEZIONATE ({len(final_features)}):")
    print(f"{'='*60}")
    
    report = []
    for f in final_features:
        idx   = feature_cols.index(f)
        score = scores[idx]
        rank  = int(np.argsort(scores).tolist().index(idx)) + 1
        
        if f in strata['top']:
            strato = 'TOP'
        elif f in strata['mid']:
            strato = 'MID'
        else:
            strato = 'BOT '  # sorpresa potenziale
            
        report.append({
            'feature':        f,
            'laplacian_score': round(score, 4),
            'rank':           f"{rank}/{len(feature_cols)}",
            'strato':         strato
        })
    
    report_df = pd.DataFrame(report)
    print(report_df.to_string(index=False))
    print(f"{'='*60}")
     # -- salva campione RIDOTTO con solo feature selezionate ------
    # (eseguito dopo aver calcolato final_features)
    cols_to_keep = final_features + [target_col]
    df_reduced = df_work[cols_to_keep]
    
    reduced_path = csv_path.replace(
        '.csv', 
        f'_sample{len(df_work)}_selected{len(final_features)}feat.csv'
    )
    df_reduced.to_csv(reduced_path, index=False)
    
    print(f"Campione ridotto salvato in:  {reduced_path}")
    print(f"  Feature: {df_work.shape[1]-1} → {len(final_features)}")
    print(f"  Righe:   {len(df_work)}")
    print(f"→ Usa questo file per gli esperimenti ESP")
    # -- salva lista feature per Dirtify ----------------------
    output_path = csv_path.replace('.csv', '_selected_features.txt')
    with open(output_path, 'w') as f:
        f.write('\n'.join(final_features))
    print(f"\nFeature salvate in: {output_path}")

    # -- plot --------------------------------------------------
    if plot:
        plot_feature_selection(feature_cols, scores, final_features, strata)

    return final_features, report_df


# ============================================================
# Esempio di utilizzo
# ============================================================
if __name__ == "__main__":
    
    features, report = select_features(
        csv_path='./datasetRoot/optdigit.csv',
        target_col="class",
        n_total=12,          # feature totali da selezionare
        k_neighbors=5,       # vicini per Laplacian Score
        corr_threshold=0.95, # soglia correlazione
        top_frac=0.50,       # 50% dalle più rilevanti
        mid_frac=0.30,       # 30% dalla fascia media
        bot_frac=0.20,       # 20% dalle meno rilevanti (sorprese)
        random_state=42,
        plot=True
    )
    
    print(f"\nFeature da usare in Dirtify:")
    print(features)

