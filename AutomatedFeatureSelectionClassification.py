import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# 1) Supervised Feature Scoring
#    Per classificazione: score ALTO = più rilevante
#    (inverso rispetto al Laplacian Score)
# ============================================================

def mutual_info_score(X, y, random_state=42):
    """
    Mutual Information tra ogni feature e il target.
    Vantaggio: cattura relazioni non lineari, robusto.
    Score alto = feature più rilevante.
    """
    scores = mutual_info_classif(X, y, random_state=random_state)
    return scores


def anova_f_score(X, y):
    """
    ANOVA F-test: misura separazione tra classi.
    Vantaggio: veloce, interpretabile.
    Score alto = feature più rilevante.
    Attenzione: assume relazioni lineari.
    """
    f_scores, _ = f_classif(X, y)
    f_scores = np.nan_to_num(f_scores, nan=0.0)
    return f_scores


def combined_score(X, y, weight_mi=0.6, weight_f=0.4, random_state=42):
    """
    Combina MI e ANOVA F-score (normalizzati) con pesi configurabili.
    Robusto sia su relazioni lineari che non lineari.
    """
    mi   = mutual_info_score(X, y, random_state)
    f    = anova_f_score(X, y)

    # normalizza in [0,1]
    def norm(v):
        r = v.max() - v.min()
        return (v - v.min()) / r if r > 1e-10 else np.zeros_like(v)

    return weight_mi * norm(mi) + weight_f * norm(f)


# ============================================================
# 2) Correlation filter
# ============================================================

def correlation_filter(X_df, selected_features, scores_dict, threshold=0.85):
    """
    Tra feature altamente correlate, mantieni quella con score più alto.
    
    scores_dict: {feature_name: score} — score alto = più rilevante
    """
    # ordina per score decrescente (migliori prima)
    ordered = sorted(selected_features, key=lambda f: scores_dict[f], reverse=True)

    corr_matrix = X_df[ordered].corr().abs()
    to_keep = []
    to_drop = set()

    for feat in ordered:
        if feat in to_drop:
            continue
        to_keep.append(feat)
        correlated = corr_matrix[feat][
            (corr_matrix[feat] > threshold) &
            (corr_matrix[feat].index != feat)
        ].index.tolist()
        to_drop.update(correlated)

    return to_keep


# ============================================================
# 3) Stratified sample
# ============================================================

def stratified_sample(df, target_col, sample_size=10000, random_state=42):
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
# 4) Stratified feature selection
# ============================================================

def stratified_feature_selection(X_df, scores, feature_names,
                                  n_total=12,
                                  top_frac=0.50,
                                  mid_frac=0.30,
                                  bot_frac=0.20,
                                  corr_threshold=0.85,
                                  random_state=42):
    """
    Seleziona feature da tre strati del ranking (score ALTO = migliore):
      - Top (score alto)   → certamente rilevanti
      - Mid                → esplorazione
      - Bottom (score basso) → ricerca sorprese
    """
    np.random.seed(random_state)

    n_features = len(feature_names)

    # ranking decrescente: le migliori per prime
    ranking = np.argsort(scores)[::-1]
    ranked_features = [feature_names[i] for i in ranking]

    # strati
    n_top = int(n_features * top_frac)
    n_mid = int(n_features * mid_frac)

    top_pool = ranked_features[:n_top]
    mid_pool = ranked_features[n_top:n_top + n_mid]
    bot_pool = ranked_features[n_top + n_mid:]

    n_from_top = max(1, int(n_total * top_frac))
    n_from_mid = max(1, int(n_total * mid_frac))
    n_from_bot = max(1, n_total - n_from_top - n_from_mid)

    print(f"\nPool disponibili:")
    print(f"  Top    ({top_frac*100:.0f}%): {len(top_pool)} feature → seleziono {n_from_top}")
    print(f"  Middle ({mid_frac*100:.0f}%): {len(mid_pool)} feature → seleziono {n_from_mid}")
    print(f"  Bottom ({bot_frac*100:.0f}%): {len(bot_pool)} feature → seleziono {n_from_bot}")

    selected_top = top_pool[:n_from_top]
    selected_mid = list(np.random.choice(mid_pool,
                                          min(n_from_mid, len(mid_pool)),
                                          replace=False)) if mid_pool else []
    selected_bot = list(np.random.choice(bot_pool,
                                          min(n_from_bot, len(bot_pool)),
                                          replace=False)) if bot_pool else []

    all_selected = selected_top + selected_mid + selected_bot
    scores_dict  = {feature_names[i]: scores[i] for i in range(n_features)}

    final_features = correlation_filter(X_df, all_selected, scores_dict, corr_threshold)

    return final_features, {
        'top': selected_top,
        'mid': selected_mid,
        'bot': selected_bot,
        'after_corr_filter': final_features
    }


# ============================================================
# 5) Plot
# ============================================================

def plot_feature_selection(feature_names, scores, final_features, strata,
                            score_label='Feature Score (higher = more relevant)'):
    # ranking decrescente
    ranking = np.argsort(scores)[::-1]
    ranked_names  = [feature_names[i] for i in ranking]
    ranked_scores = [scores[i] for i in ranking]

    colors = []
    for f in ranked_names:
        if f in strata['top'] and f in final_features:
            colors.append('steelblue')
        elif f in strata['mid'] and f in final_features:
            colors.append('orange')
        elif f in strata['bot'] and f in final_features:
            colors.append('red')
        else:
            colors.append('lightgray')

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(ranked_names)), ranked_scores, color=colors)
    ax.set_xticks(range(len(ranked_names)))
    ax.set_xticklabels(ranked_names, rotation=90, fontsize=8)
    ax.set_ylabel(score_label)
    ax.set_title('Feature Selection — Supervised Score + Stratified Sampling')

    legend = [
        Patch(color='steelblue', label='Top selected'),
        Patch(color='orange',    label='Mid selected'),
        Patch(color='red',       label='Bottom selected'),
        Patch(color='lightgray', label='Not selected'),
    ]
    ax.legend(handles=legend, loc='upper right')
    plt.tight_layout()
    plt.savefig('feature_selection_classification.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 6) Dimensione campione consigliata
# ============================================================

def recommended_sample_size(n_features, n_classes, min_per_class=500):
    by_class    = n_classes * min_per_class
    by_features = n_features * 100
    recommended = max(by_class, by_features)

    print(f"Dimensione campione consigliata:")
    print(f"  Per classe   ({n_classes} × {min_per_class}): {by_class}")
    print(f"  Per feature  ({n_features} × 100):           {by_features}")
    print(f"  Consigliata:                                  {recommended}")

    return recommended


# ============================================================
# 7) MAIN
# ============================================================

def select_features(csv_path,
                    target_col="NSP",
                    n_total=9,
                    scoring='combined',       # 'mi' | 'anova' | 'combined'
                    weight_mi=0.6,            # usato solo con scoring='combined'
                    weight_f=0.4,
                    corr_threshold=0.80,
                    top_frac=0.50,
                    mid_frac=0.30,
                    bot_frac=0.20,
                    random_state=42,
                    max_samples=10000,
                    plot=True):
    """
    Pipeline di feature selection per classificazione.

    Parameters
    ----------
    scoring : str
        'mi'       → Mutual Information (non lineare, più lento)
        'anova'    → ANOVA F-score (lineare, veloce)
        'combined' → combinazione pesata MI + ANOVA (default)
    """
    df = pd.read_csv(csv_path)

    # pulizia colonne
    df = df.dropna(axis=1, how='all')
    df.columns = df.columns.str.strip()

    # gestisci target categorico
    if df[target_col].dtype == object:
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])
        print(f"  Target codificato: {list(le.classes_)}")

    feature_cols = [c for c in df.columns if c != target_col]

    print(f"  Colonne totali: {df.shape[1]}")
    print(f"  Feature:        {len(feature_cols)}")
    print(f"  Target: '{target_col}'")
    print(f"  Classi: {sorted(df[target_col].unique())}")
    print(f"  NaN nel target: {df[target_col].isna().sum()}")

    n_classes   = df[target_col].nunique()
    sample_size = recommended_sample_size(len(feature_cols), n_classes)
    sample_size = min(sample_size, max_samples)

    # campionamento stratificato
    if len(df) > sample_size:
        df_work = stratified_sample(df, target_col, sample_size, random_state)
    else:
        df_work = df.copy()

    # salva campione completo
    sample_path = csv_path.replace('.csv', f'_sample{len(df_work)}.csv')
    df_work.to_csv(sample_path, index=False)
    print(f"\nCampione salvato in: {sample_path}")

    # prepara X, y
    X_df = df_work[feature_cols].copy()
    y    = df_work[target_col].values

    # variance filter
    var = X_df.var()
    zero_var = var[var < 1e-10].index.tolist()
    if zero_var:
        print(f"\nRimosse {len(zero_var)} feature a varianza zero: {zero_var}")
        X_df = X_df.drop(columns=zero_var)
        feature_cols = X_df.columns.tolist()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    # scoring supervisionato
    print(f"\nCalcolo feature scores (metodo: {scoring}) su {len(df_work)} campioni...")

    if scoring == 'mi':
        scores = mutual_info_score(X_scaled, y, random_state)
        score_label = 'Mutual Information (higher = more relevant)'
    elif scoring == 'anova':
        scores = anova_f_score(X_scaled, y)
        score_label = 'ANOVA F-score (higher = more relevant)'
    else:  # combined
        scores = combined_score(X_scaled, y, weight_mi, weight_f, random_state)
        score_label = f'Combined Score (MI×{weight_mi} + ANOVA×{weight_f}, higher = more relevant)'

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

    # report
    print(f"\n{'='*60}")
    print(f"FEATURE SELEZIONATE ({len(final_features)}):")
    print(f"{'='*60}")

    report = []
    scores_dict = {feature_cols[i]: scores[i] for i in range(len(feature_cols))}
    ranking_desc = np.argsort(scores)[::-1].tolist()

    for f in final_features:
        idx   = feature_cols.index(f)
        score = scores[idx]
        rank  = ranking_desc.index(idx) + 1

        if f in strata['top']:
            strato = 'TOP'
        elif f in strata['mid']:
            strato = 'MID'
        else:
            strato = 'BOT'

        report.append({
            'feature': f,
            'score':   round(score, 4),
            'rank':    f"{rank}/{len(feature_cols)}",
            'strato':  strato
        })

    report_df = pd.DataFrame(report)
    print(report_df.to_string(index=False))
    print(f"{'='*60}")

    # salva campione ridotto
    cols_to_keep = final_features + [target_col]
    df_reduced   = df_work[cols_to_keep]
    reduced_path = csv_path.replace(
        '.csv',
        f'_sample{len(df_work)}_selected{len(final_features)}feat.csv'
    )
    df_reduced.to_csv(reduced_path, index=False)
    print(f"\nCampione ridotto salvato in: {reduced_path}")
    print(f"  Feature: {len(feature_cols)} → {len(final_features)}")
    print(f"  Righe:   {len(df_work)}")
    print(f"→ Usa questo file per gli esperimenti ESP (classificazione)")

    # salva lista feature
    output_path = csv_path.replace('.csv', '_selected_features.txt')
    with open(output_path, 'w') as fh:
        fh.write('\n'.join(final_features))
    print(f"Feature salvate in: {output_path}")

    if plot:
        plot_feature_selection(feature_cols, scores, final_features, strata,
                                score_label)

    return final_features, report_df


# ============================================================
# Esempio di utilizzo
# ============================================================
if __name__ == "__main__":

    features, report = select_features(
        csv_path='./datasetRoot/shuttle.csv',
        target_col="class",          # target binario/multiclasse
        n_total=9,
        scoring='combined',            # 'mi' | 'anova' | 'combined'
        weight_mi=0.6,
        weight_f=0.4,
        corr_threshold=0.85,
        top_frac=0.50,
        mid_frac=0.30,
        bot_frac=0.20,
        random_state=42,
        max_samples=10000,
        plot=True
    )

    print(f"\nFeature da usare in Dirtify (classificazione):")
    print(features)