"""
ESP Curve Shape Clustering
==========================
Raggruppa gli scenari ESP in base alla forma della curva AMI vs. error rate.

Struttura attesa per ogni scenario nella lista significant_scenarios:
    {
        'scenario_name': str,
        'model':         str,
        'error':         str,
        'features':      str,
        'x_points':      list[float],   # es. [0, 20, 40, 60, 80]
        'y_matrix':      list[list],    # shape (n_runs, n_levels)
        'AECP_mean':     float,
        ...
    }

Uso:
    from esp_curve_clustering import (
        cluster_esp_curves, plot_clusters_full, print_cluster_summary,
        cluster_by_sign, plot_clusters_by_sign, print_summary_by_sign
    )

    # Clustering globale
    results = cluster_esp_curves(significant_scenarios, auto_k=True)
    print_cluster_summary(results)
    plot_clusters_full(results, output_path="esp_clusters.png")

    # Clustering separato per segno AEPC
    results_sign = cluster_by_sign(significant_scenarios)
    print_summary_by_sign(results_sign)
    plot_clusters_by_sign(results_sign, output_path="esp_clusters_by_sign.png")
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from collections import defaultdict


# ──────────────────────────────────────────────────────────────────
# 1. Estrazione features dalla curva
# ──────────────────────────────────────────────────────────────────

def extract_curve_features(scenario: dict) -> np.ndarray:
    """
    Dalla y_matrix (n_runs x n_levels) estrae:
      - curva media AMI normalizzata z-score (shape-invariant)
      - slope tra livelli consecutivi (4 valori)
    Restituisce un vettore 1D (9 valori) che rappresenta la forma.
    """
    y_matrix   = np.array(scenario['y_matrix'])
    mean_curve = y_matrix.mean(axis=0)
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
    """
    Clusterizza gli scenari ESP in base alla forma della curva AMI.

    Parametri
    ----------
    significant_scenarios : list di dict (struttura descritta sopra)
    n_clusters            : int  — numero cluster fisso (ignora auto_k)
    auto_k                : bool — cerca k ottimale con silhouette
    k_range               : range da esplorare se auto_k=True
    random_state          : int

    Ritorna dict con:
        scenarios    : la lista originale
        labels       : array cluster label per ogni scenario
        mean_curves  : {cluster_id: curva AMI media}
        cluster_info : {cluster_id: [info scenario]}
        k, silhouette, x_points
    """
    features        = np.array([extract_curve_features(s)
                                 for s in significant_scenarios])
    scaler          = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    if auto_k or n_clusters is None:
        print("Ricerca k ottimale:")
        k = _find_best_k(features_scaled, k_range, random_state)
    else:
        k = n_clusters

    km     = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    labels = km.fit_predict(features_scaled)
    sil    = silhouette_score(features_scaled, labels)

    x_points = significant_scenarios[0]['x_points']

    # Curva media per cluster (scala originale)
    mean_curves = {}
    for c in range(k):
        idx = [i for i, l in enumerate(labels) if l == c]
        cluster_curves = np.array([
            np.array(significant_scenarios[i]['y_matrix']).mean(axis=0)
            for i in idx
        ])
        mean_curves[c] = cluster_curves.mean(axis=0)

    # Info per cluster
    cluster_info = defaultdict(list)
    for i, (s, label) in enumerate(zip(significant_scenarios, labels)):
        cluster_info[int(label)].append({
            'scenario_name': s['scenario_name'],
            'modelName':     s['model'],
            'errorType':     s['error'],
            'feature':       s['features'],
            'AECP_mean':     s['AECP_mean'],
        })

    # Verifica somma
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
    }


# ──────────────────────────────────────────────────────────────────
# 4. Visualizzazione
# ──────────────────────────────────────────────────────────────────

def _draw_cluster(ax, c, results, color, sign_label=""):
    """Disegna un singolo pannello cluster."""
    x          = results['x_points']
    info_list  = results['cluster_info'][c]
    mean_curve = results['mean_curves'][c]
    scenarios  = results['scenarios']
    labels     = results['labels']

    for i, l in enumerate(labels):
        if l != c:
            continue
        y = np.array(scenarios[i]['y_matrix'])
        for row in y:
            ax.plot(x, row, color=color, alpha=0.05, linewidth=0.6)
        ax.plot(x, y.mean(axis=0), color=color, alpha=0.35, linewidth=1.0)

    ax.plot(x, mean_curve, color=color, linewidth=2.5, zorder=10)
    ax.axhline(mean_curve[0], color='gray', linestyle='--',
               linewidth=0.8, alpha=0.5)

    models = [i['modelName'] for i in info_list]
    errors = [i['errorType'] for i in info_list]
    mc = {m: models.count(m) for m in set(models)}
    ec = {e: errors.count(e) for e in set(errors)}

    prefix = f"{sign_label} — " if sign_label else ""
    subtitle = (
        f"n={len(info_list)}\n"
        f"{', '.join(f'{m}({n})' for m,n in sorted(mc.items()))}\n"
        f"{', '.join(f'{e}({n})' for e,n in sorted(ec.items()))}"
    )
    ax.set_title(f"{prefix}Cluster {c}\n{subtitle}", fontsize=7.5)
    ax.set_xlabel("Error rate (%)")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    ax.text(0.98, 0.02, f"sil={results['silhouette']:.3f}",
            transform=ax.transAxes, fontsize=7,
            ha='right', va='bottom', color='gray')


def plot_clusters_full(results: dict,
                       output_path: str = "esp_clusters.png",
                       figsize_per_cluster: tuple = (4, 3.5)):
    """Plot con tutte le curve individuali per ogni cluster."""
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
            ax.set_ylabel("AMI")

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
        aecp   = [i['AECP_mean'] for i in info_list]

        print(f"\nCluster {c}  (n={len(info_list)})")
        print(f"  Curva media : {np.round(mean_curve, 4)}")
        print(f"  Slope       : {np.round(slopes, 4)}")
        print(f"  AECP medio  : {np.mean(aecp):.2f}%")
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
    """
    Separa gli scenari in positivi (AECP>0) e negativi (AECP<=0),
    poi clusterizza ciascun gruppo per forma indipendentemente.
    """
    pos = [s for s in significant_scenarios if s['AECP_mean'] > 0]
    neg = [s for s in significant_scenarios if s['AECP_mean'] <= 0]

    print(f"Scenari positivi (AECP>0) : {len(pos)}")
    print(f"Scenari negativi (AECP<=0): {len(neg)}")

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
                           output_path: str = "esp_clusters_by_sign.png"):
    """
    Plot su due righe:
      riga superiore (verde) = cluster AEPC positivi
      riga inferiore (rossa) = cluster AEPC negativi
    """
    res_pos = results_by_sign['positive']
    res_neg = results_by_sign['negative']
    k_pos   = res_pos['k'] if res_pos else 0
    k_neg   = res_neg['k'] if res_neg else 0
    n_cols  = max(k_pos, k_neg, 1)

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 7),
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
                axes_row[c].set_ylabel("AMI")
        for c in range(res['k'], n_cols):
            axes_row[c].axis('off')

    _fill_row(res_pos, axes[0], colors_pos, "AEPC > 0")
    _fill_row(res_neg, axes[1], colors_neg, "AEPC ≤ 0")

    n_pos = results_by_sign['n_positive']
    n_neg = results_by_sign['n_negative']
    fig.suptitle(
        f"ESP Curve Shape Clustering by Sign\n"
        f"Positive (n={n_pos})  |  Negative (n={n_neg})",
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
            aecp   = [i['AECP_mean'] for i in info_list]
            print(f"\n  Cluster {c}  (n={len(info_list)})")
            print(f"  Curva media : {np.round(mean_curve, 4)}")
            print(f"  Slope       : {np.round(slopes, 4)}")
            print(f"  AECP medio  : {np.mean(aecp):.2f}%")
            print(f"  Modelli     : { {m: models.count(m) for m in set(models)} }")
            print(f"  Errori      : { {e: errors.count(e) for e in set(errors)} }")
            print(f"  Forma       : {_shape_label(slopes)}")