"""
ESP Curve Shape Clustering
==========================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from collections import defaultdict


# ──────────────────────────────────────────────────────────────────
# 0. Utility: lunghezza attesa y_matrix
# ──────────────────────────────────────────────────────────────────

def _expected_n_levels(scenarios: list) -> int:
    """Restituisce la lunghezza y più comune tra gli scenari."""
    lengths = []
    for s in scenarios:
        ym = np.array(s['y_matrix'])
        if ym.ndim == 2:
            lengths.append(ym.shape[1])
        elif ym.ndim == 1:
            lengths.append(len(ym))
    if not lengths:
        return 5
    return max(set(lengths), key=lengths.count)


def _safe_mean_curve(y_matrix_raw, n_levels: int) -> np.ndarray:
    """
    Calcola la curva media da y_matrix, troncando o paddando
    a n_levels colonne per garantire shape uniforme.
    """
    ym = np.array(y_matrix_raw)
    if ym.ndim == 1:
        ym = ym.reshape(1, -1)
    # tronca o padda ogni riga
    rows = []
    for row in ym:
        if len(row) >= n_levels:
            rows.append(row[:n_levels])
        else:
            pad = np.full(n_levels - len(row), np.nan)
            rows.append(np.concatenate([row, pad]))
    ym_fixed = np.array(rows, dtype=float)
    return np.nanmean(ym_fixed, axis=0)


def _filter_scenarios(scenarios: list) -> list:
    """
    Rimuove scenari con y_matrix non rettangolare o con meno di 2 run.
    """
    n_levels = _expected_n_levels(scenarios)
    clean = []
    for s in scenarios:
        ym = np.array(s['y_matrix'])
        if ym.ndim != 2:
            print(f"[SKIP] {s['scenario_name']}: y_matrix ndim={ym.ndim}")
            continue
        if ym.shape[0] < 2:
            print(f"[SKIP] {s['scenario_name']}: solo {ym.shape[0]} run")
            continue
        clean.append(s)
    if len(clean) < len(scenarios):
        print(f"[INFO] _filter_scenarios: {len(scenarios)-len(clean)} "
              f"scenari rimossi, {len(clean)} rimasti")
    return clean


# ──────────────────────────────────────────────────────────────────
# 1. Estrazione features dalla curva
# ──────────────────────────────────────────────────────────────────

def extract_curve_features(scenario: dict,
                            n_levels: int = None) -> np.ndarray:
    """
    Dalla y_matrix (n_runs x n_levels) estrae:
      - curva media AMI normalizzata z-score (shape-invariant)
      - slope tra livelli consecutivi (n_levels-1 valori)
    """
    n = n_levels if n_levels is not None else 5
    mean_curve = _safe_mean_curve(scenario['y_matrix'], n)
    slopes     = np.diff(mean_curve)

    std = mean_curve.std()
    mean_curve_norm = (
        (mean_curve - mean_curve.mean()) / std
        if std > 1e-8
        else mean_curve - mean_curve.mean()
    )
    return np.concatenate([mean_curve_norm, slopes])


# ──────────────────────────────────────────────────────────────────
# 2. Scelta automatica di k con silhouette
# ──────────────────────────────────────────────────────────────────

def _find_best_k(feature_matrix: np.ndarray,
                 k_range: range = range(2, 8),
                 random_state: int = 42) -> int:
    best_k, best_score = k_range.start, -1
    for k in k_range:
        if k >= len(feature_matrix):
            break
        km     = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(feature_matrix)
        score  = silhouette_score(feature_matrix, labels)
        print(f"  k={k}  silhouette={score:.4f}")
        if score > best_score:
            best_score, best_k = score, k
    print(f"  → miglior k: {best_k}  (silhouette={best_score:.4f})")
    return best_k


# ──────────────────────────────────────────────────────────────────
# 3. Clustering principale
# ──────────────────────────────────────────────────────────────────

def cluster_esp_curves(significant_scenarios: list,
                       n_clusters: int = None,
                       auto_k: bool = True,
                       k_range: range = range(2, 7),
                       random_state: int = 42) -> dict:

    significant_scenarios = _filter_scenarios(significant_scenarios)
    if len(significant_scenarios) < 4:
        print(f"[WARN] cluster_esp_curves: solo {len(significant_scenarios)} "
              f"scenari validi, clustering saltato")
        return None

    n_levels = _expected_n_levels(significant_scenarios)
    if n_levels != 5:
        print(f"[INFO] n_levels atteso: {n_levels}")

    raw_features = [extract_curve_features(s, n_levels)
                    for s in significant_scenarios]

    lengths = set(len(f) for f in raw_features)
    if len(lengths) > 1:
        min_len = min(lengths)
        print(f"[WARN] feature lengths diverse {lengths}, tronco a {min_len}")
        raw_features = [f[:min_len] for f in raw_features]

    features = np.array(raw_features, dtype=float)

    # --- FIX NaN: sostituisce NaN con la media della colonna ---
    if np.any(np.isnan(features)):
        n_nan = int(np.sum(np.isnan(features)))
        print(f"[WARN] features contiene {n_nan} NaN, imputo con media colonna")
        col_means = np.nanmean(features, axis=0)
        # se una colonna è tutta NaN la sostituisce con 0
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        nan_mask = np.isnan(features)
        features[nan_mask] = np.take(col_means,
                                     np.where(nan_mask)[1])

    scaler          = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # verifica finale: nessun NaN o inf dopo scaling
    if np.any(~np.isfinite(features_scaled)):
        print("[WARN] features_scaled contiene NaN/inf dopo scaling, "
              "rimpiazzo con 0")
        features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0,
                                        neginf=0.0)

    if auto_k or n_clusters is None:
        print("Ricerca k ottimale:")
        k = _find_best_k(features_scaled, k_range, random_state)
    else:
        k = n_clusters

    km     = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    labels = km.fit_predict(features_scaled)
    sil    = silhouette_score(features_scaled, labels)

    x_points = significant_scenarios[0]['x_points']

    mean_curves = {}
    for c in range(k):
        idx = [i for i, l in enumerate(labels) if l == c]
        cluster_curves = np.array([
            _safe_mean_curve(significant_scenarios[i]['y_matrix'], n_levels)
            for i in idx
        ], dtype=float)
        # imputa NaN anche nelle curve medie
        cluster_curves = np.where(np.isnan(cluster_curves),
                                  np.nanmean(cluster_curves, axis=0,
                                             keepdims=True),
                                  cluster_curves)
        mean_curves[c] = np.nanmean(cluster_curves, axis=0)

    cluster_info = defaultdict(list)
    for i, (s, label) in enumerate(zip(significant_scenarios, labels)):
        cluster_info[int(label)].append({
            'scenario_name': s['scenario_name'],
            'modelName':     s['model'],
            'errorType':     s['error'],
            'feature':       s['features'],
            'AEPC_mean':     s['AEPC_mean'],
        })

    total = sum(len(v) for v in cluster_info.values())
    assert total == len(significant_scenarios), \
        f"Bug: {total} != {len(significant_scenarios)}"

    return {
        'scenarios':    significant_scenarios,
        'labels':       labels,
        'mean_curves':  mean_curves,
        'cluster_info': dict(cluster_info),
        'k':            k,
        'silhouette':   sil,
        'x_points':     x_points,
        'n_levels':     n_levels,
    }

# ──────────────────────────────────────────────────────────────────
# 4. Visualizzazione
# ──────────────────────────────────────────────────────────────────

def _draw_cluster(ax, c, results, color, sign_label=""):
    x          = results['x_points']
    n_levels   = results.get('n_levels', 5)
    info_list  = results['cluster_info'][c]
    mean_curve = results['mean_curves'][c]
    scenarios  = results['scenarios']
    labels     = results['labels']
    # Aggiungi questo dizionario all'inizio del file o dentro _draw_cluster
    MODEL_ABBREV = {
        'Gaussian Mixture Model': 'GM',
        'Hierarchical Clustering': 'HC',
        'K-Means': 'KM',
        'HDBSCAN': 'HD',
        'BIRCH': 'BI',

        "LogisticRegression": 'LR',
        "K-Nearest Neighbors": 'KNN',
        "NaiveBayes": 'NB',
        "SupportVectorMachine": 'SVM',
        "DecisionTreeClassifier": 'DT',
        "RadialBasis Function SVM": 'RBF-SVM',
        "Xgboost": 'XGB', 
        "Multi-Layer Perceptron": 'MLP', 
        "AdaBoost": 'AB', 
        "LinearDiscriminantAnalysis": 'LDA',
        "SGDClassifier": 'SGD', 
        "RandomForest": 'RF',
        "RidgeClassifier": 'Ridge',
        "Extra Trees": 'ET',
        "GaussianNB": 'GNB',
        "QuadraticDiscriminantAnalysis": 'QDA',
        "SDGClassifier": 'SDG',
        "ExtraTreesClassifier":"ET",
        "RandomForestClassifier":"RF",
        "AdaBoostClassifier":"ADA",
        "KNeighborsClassifier":"KNN",
        "XGBClassifier":"XGB",
        "MLPClassifier":"MLP",def _Draw_cluster_originale(cluster_id, cluster_data):
            """
            Stampa i modelli e gli errori di un cluster, ciascuno su una singola riga.
            """
            print(f"Cluster {cluster_id}:")
            
            # Converte tutti gli elementi in stringhe e li unisce con una virgola
            modelli_str = ", ".join(map(str, cluster_data['models']))
            print(f"  Modelli: {modelli_str}")
            
            errori_str = ", ".join(map(str, cluster_data['errors']))
            print(f"  Errori:  {errori_str}")
            print("-" * 30)
        
    }

    ERROR_ABBREV = {
        'duplicate': 'Dup',
        'missing': 'Mis',
        'outlier': 'Out',
        'noise': 'Noi',
        'label': 'Label',
    }
    for i, l in enumerate(labels):
        if l != c:
            continue
        # --- FIX: usa _safe_mean_curve per il plot delle singole curve ---
        ym = np.array(scenarios[i]['y_matrix'])
        for row in ym:
            row_fixed = row[:n_levels] if len(row) >= n_levels else row
            if len(row_fixed) == len(x):
                ax.plot(x, row_fixed, color=color,
                        alpha=0.05, linewidth=0.6)
        mean_row = _safe_mean_curve(scenarios[i]['y_matrix'], n_levels)
        if len(mean_row) == len(x):
            ax.plot(x, mean_row, color=color, alpha=0.35, linewidth=1.0)

    if len(mean_curve) == len(x):
        ax.plot(x, mean_curve, color=color, linewidth=2.5, zorder=10)
        ax.axhline(mean_curve[0], color='gray', linestyle='--',
                   linewidth=0.8, alpha=0.5)

    models = [MODEL_ABBREV.get(i['modelName'], i['modelName']) 
          for i in info_list]
    errors = [ERROR_ABBREV.get(i['errorType'], i['errorType']) 
            for i in info_list]
    mc = {m: models.count(m) for m in set(models)}
    ec = {e: errors.count(e) for e in set(errors)}
    
    prefix   = f"{sign_label} — " if sign_label else ""
    subtitle = (
        f"n={len(info_list)}\n"
        f"{', '.join(f'{m}({n})' for m,n in sorted(mc.items()))}\n"
        f"{', '.join(f'{e}({n})' for e,n in sorted(ec.items()))}"
    )
    ax.set_title(f"{prefix}Cluster {c}\n{subtitle}", 
                 fontsize=24)
    ax.set_xlabel("Error rate (%)", fontsize=10)
    ax.tick_params(axis='both', labelsize=10)
    ax.text(0.98, 0.02, f"sil={results['silhouette']:.3f}",
            transform=ax.transAxes, fontsize=9,
            ha='right', va='bottom', color='black')



def plot_clusters_full(results: dict,
                       output_path: str = "esp_clusters.png",
                       figsize_per_cluster: tuple = (4, 3.5),
                       metric: str = "AMI"):
    if results is None:
        print("[WARN] plot_clusters_full: results è None, skip")
        return
    k      = results['k']
    colors = cm.tab10(np.linspace(0, 0.9, k))
    fig, axes = plt.subplots(1, k,
                             figsize=(figsize_per_cluster[0] * k,
                                      figsize_per_cluster[1]),
                             sharey=True)
    if k == 1:
        axes = [axes]

    for c, ax in enumerate(axes):
        _draw_cluster(ax, c, results, colors[c])
        if c == 0:
            ax.set_ylabel(metric)

    fig.suptitle(
        f"ESP Curve Shape Clustering  "
        f"(k={k}, silhouette={results['silhouette']:.3f})",
        fontsize=10, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Salvato: {output_path}")


# ──────────────────────────────────────────────────────────────────
# 5. Summary testuale
# ──────────────────────────────────────────────────────────────────

def _shape_label(slopes: np.ndarray) -> str:
    if all(s < 0 for s in slopes):
        return "monotona decrescente"
    if all(s > 0 for s in slopes):
        return "monotona crescente"
    if slopes[0] < -0.001 and max(slopes[1:]) < 0.005:
        return "drop iniziale poi stabile"
    if slopes[0] > 0 and slopes[-1] < 0:
        return "sale poi scende"
    if slopes[0] > 0 and min(slopes) > -0.005:
        return "crescente con plateau"
    return "forma complessa"


def print_cluster_summary(results: dict):
    if results is None:
        print("[WARN] print_cluster_summary: results è None")
        return
    print(f"\n{'='*60}")
    print(f"ESP CURVE CLUSTERING — k={results['k']}, "
          f"silhouette={results['silhouette']:.4f}")
    print(f"Total scenari: {len(results['scenarios'])}")
    print(f"{'='*60}")

    for c in range(results['k']):
        info_list  = results['cluster_info'][c]
        mean_curve = results['mean_curves'][c]
        slopes     = np.diff(mean_curve)
        models = [i['modelName'] for i in info_list]
        errors = [i['errorType'] for i in info_list]
        aecp   = [i['AEPC_mean'] for i in info_list]

        print(f"\nCluster {c}  (n={len(info_list)})")
        print(f"  Curva media : {np.round(mean_curve, 4)}")
        print(f"  Slope       : {np.round(slopes, 4)}")
        print(f"  AEPC medio  : {np.mean(aecp):.2f}%")
        print(f"  Modelli     : { {m: models.count(m) for m in set(models)} }")
        print(f"  Errori      : { {e: errors.count(e) for e in set(errors)} }")
        print(f"  Forma       : {_shape_label(slopes)}")


# ──────────────────────────────────────────────────────────────────
# 6. Clustering separato per segno AEPC
# ──────────────────────────────────────────────────────────────────

def cluster_by_sign(significant_scenarios: list,
                    auto_k: bool = True,
                    k_range_pos: range = range(2, 5),
                    k_range_neg: range = range(2, 6),
                    random_state: int = 42) -> dict:

    pos = [s for s in significant_scenarios if s['AEPC_mean'] > 0]
    neg = [s for s in significant_scenarios if s['AEPC_mean'] <= 0]

    print(f"Scenari positivi (AEPC>0) : {len(pos)}")
    print(f"Scenari negativi (AEPC<=0): {len(neg)}")

    res_pos, res_neg = None, None

    if len(pos) >= 4:
        print("\n--- Clustering POSITIVI ---")
        res_pos = cluster_esp_curves(
            pos, auto_k=auto_k, k_range=k_range_pos,
            random_state=random_state
        )
    else:
        print(f"Troppo pochi scenari positivi ({len(pos)}) per clusterizzare.")

    if len(neg) >= 4:
        print("\n--- Clustering NEGATIVI ---")
        res_neg = cluster_esp_curves(
            neg, auto_k=auto_k, k_range=k_range_neg,
            random_state=random_state
        )
    else:
        print(f"Troppo pochi scenari negativi ({len(neg)}) per clusterizzare.")

    return {
        'positive':   res_pos,
        'negative':   res_neg,
        'n_positive': len(pos),
        'n_negative': len(neg),
    }


def plot_clusters_by_sign(results_by_sign: dict,
                           output_path: str = "esp_clusters_by_sign.png",
                           metric: str = "AMI"):
    res_pos = results_by_sign['positive']
    res_neg = results_by_sign['negative']
    k_pos   = res_pos['k'] if res_pos else 0
    k_neg   = res_neg['k'] if res_neg else 0
    n_cols  = max(k_pos, k_neg, 1)

    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 9),
                             sharey='row')
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    colors_pos = cm.Greens(np.linspace(0.45, 0.85, k_pos)) if k_pos else []
    colors_neg = cm.Reds(np.linspace(0.45, 0.85, k_neg))   if k_neg else []

    def _fill_row(res, axes_row, colors, sign_label):
        if res is None:
            for ax in axes_row:
                ax.axis('off')
            return
        for c in range(res['k']):
            _draw_cluster(axes_row[c], c, res, colors[c], sign_label)
            if c == 0:
                axes_row[c].set_ylabel(metric)
        for c in range(res['k'], n_cols):
            axes_row[c].axis('off')

    _fill_row(res_pos, axes[0], colors_pos, "AEPC > 0")
    _fill_row(res_neg, axes[1], colors_neg, "AEPC ≤ 0")

    n_pos = results_by_sign['n_positive']
    n_neg = results_by_sign['n_negative']
    #fig.suptitle(
    #    f"ESP Curve Shape Clustering by Sign\n"
    #    f"Positive (n={n_pos})  |  Negative (n={n_neg})",
    #    fontsize=11, fontweight='bold'
    #)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Salvato: {output_path}")


def print_summary_by_sign(results_by_sign: dict):
    for sign, key in [("POSITIVI (AEPC>0)", 'positive'),
                      ("NEGATIVI (AEPC<=0)", 'negative')]:
        res = results_by_sign[key]
        if res is None:
            print(f"\n{sign}: nessun cluster disponibile")
            continue
        print(f"\n{'='*60}")
        print(f"SCENARI {sign}  —  k={res['k']}, "
              f"silhouette={res['silhouette']:.4f}")
        print(f"{'='*60}")
        for c in range(res['k']):
            info_list  = res['cluster_info'][c]
            mean_curve = res['mean_curves'][c]
            slopes     = np.diff(mean_curve)
            models = [i['modelName'] for i in info_list]
            errors = [i['errorType'] for i in info_list]
            aecp   = [i['AEPC_mean'] for i in info_list]
            print(f"\n  Cluster {c}  (n={len(info_list)})")
            print(f"  Curva media : {np.round(mean_curve, 4)}")
            print(f"  Slope       : {np.round(slopes, 4)}")
            print(f"  AEPC medio  : {np.mean(aecp):.2f}%")
            print(f"  Modelli     : { {m: models.count(m) for m in set(models)} }")
            print(f"  Errori      : { {e: errors.count(e) for e in set(errors)} }")
            print(f"  Forma       : {_shape_label(slopes)}")