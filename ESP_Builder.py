import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import epc_calcolus
import re

# ============================================================
# 1) Estrazione curve + AECP (robustness_index) e salvataggio JSON
# ============================================================
def analyze_robustness(df, metric, epc_df, dataset_name):
    """Estrae le informazioni di robustezza e le salva in un file JSON."""
    df = df.copy()
    df[metric] = pd.to_numeric(df[metric], errors='coerce')

    baseline_df = df[df['percentage'] == 0].copy()
    test_df = df[df['percentage'] > 0].copy()

    curve_data_list = []

    for (run_id, feat, err), group_data in test_df.groupby(['experiment_run', 'feature', 'errorType']):
        for model in group_data['modelName'].unique():

            b_val_series = baseline_df[baseline_df['modelName'] == model][metric].dropna()
            if b_val_series.empty:
                continue
            b_val = float(b_val_series.values[0])

            current_test = group_data[group_data['modelName'] == model].copy().dropna(subset=[metric])
            current_test = current_test.groupby('percentage')[metric].mean().reset_index()

            combined = pd.concat([
                pd.DataFrame({'percentage': [0.0], metric: [b_val]}),
                current_test[['percentage', metric]]
            ], ignore_index=True).sort_values('percentage')

            x = combined['percentage'].values * 100.0
            y = combined[metric].values.astype(float)
            baseline = float(y[0])

            # AECP in percentuale (soglia operativa: |AECP| > 5.0)
            area_baseline_total = baseline * (x.max() - x.min()) if x.max() != x.min() else 1.0
            area_degradation = np.trapz(y - baseline, x)
            robustness_index = (area_degradation / area_baseline_total) * 100.0

            # EPC
            try:
                myepc = epc_df[
                    (epc_df['feature'] == feat) &
                    (epc_df['errorType'] == err) &
                    (epc_df['modelName'] == model) &
                    (epc_df['experiment_run'] == run_id)
                ]['epc'].iloc[0]
                myepc = float(myepc) if isinstance(myepc, (int, float, np.floating)) else 0.0
            except Exception:
                myepc = 0.0

            curve_entry = {
                'experiment_run': int(run_id),
                'dataset': dataset_name,
                'feature': feat,
                'errorType': err,
                'modelName': model,
                'metric': metric,
                'x': x.tolist(),
                'y': y.tolist(),
                'baseline': baseline,
                'robustness_index': float(robustness_index),
                'epc': float(myepc),
            }
            curve_data_list.append(curve_entry)

    os.makedirs('./CurveData', exist_ok=True)
    json_filename = f"./CurveData/{dataset_name.replace('.csv', '')}_curve.json"
    with open(json_filename, 'w') as f:
        json.dump(curve_data_list, f, indent=4)

    print(f"Dati salvati in: {json_filename}")
    return json_filename


# ============================================================
# 2) Statistiche
# ============================================================
def get_confidence_interval_wilcoxon(data, confidence=0.95):
    """
    CI t-Student per la media + p-value Wilcoxon signed-rank vs 0.
    Non parametrico: non assume normalità della distribuzione.
    Ritorna: mean, low, high, p_val
    """
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) < 2:
        return np.nan, np.nan, np.nan, np.nan

    mean = float(np.mean(data))
    sem = stats.sem(data)

    # p-val Wilcoxon signed-rank vs ipotesi nulla: mediana = 0
    try:
        _, p_val = wilcoxon(data)
    except Exception:
        # fallback se tutti i valori sono uguali
        p_val = 1.0

    if sem == 0 or np.isnan(sem):
        return mean, mean, mean, float(p_val)

    low, high = stats.t.interval(confidence, len(data) - 1, loc=mean, scale=sem)
    return float(mean), float(low), float(high), float(p_val)


def get_confidence_interval(data, confidence=0.95):
    """
    CI t-Student per media + p-value t-test vs 0.
    Usato per le slope dei segmenti (non per il filtraggio scenari).
    Ritorna: mean, low, high, p_val
    """
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return np.nan, np.nan, np.nan, np.nan

    mean = float(np.mean(data))
    sem = stats.sem(data)
    _, p_val = stats.ttest_1samp(data, 0)

    if sem == 0 or np.isnan(sem) or len(data) < 2:
        return mean, mean, mean, float(p_val)

    low, high = stats.t.interval(confidence, len(data) - 1, loc=mean, scale=sem)
    return float(mean), float(low), float(high), float(p_val)


def pointwise_mean_ci(y_matrix, confidence=0.95):
    """
    CI punto-per-punto della media: per ogni colonna calcola mean ± h
    usando t-Student.
    """
    y_matrix = np.asarray(y_matrix, dtype=float)
    n = y_matrix.shape[0]

    mean = np.mean(y_matrix, axis=0)
    sem = stats.sem(y_matrix, axis=0, nan_policy="omit")

    h = np.zeros_like(mean)
    ok = (sem > 0) & ~np.isnan(sem) & (n >= 2)
    if np.any(ok):
        tcrit = stats.t.ppf((1 + confidence) / 2., n - 1)
        h[ok] = tcrit * sem[ok]

    low = mean - h
    high = mean + h
    return mean, low, high


# ============================================================
# 3) Lettura JSON e costruzione scenari
# ============================================================
def extract_vectors_from_json(json_path):
    if not os.path.exists(json_path):
        print(f"Errore: file non trovato: {json_path}")
        return {}

    with open(json_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    group_cols = ['dataset', 'feature', 'errorType', 'modelName', 'metric']
    scenarios = {}

    for name, group in df.groupby(group_cols):
        scenario_key = f"{name[3]} | {name[2]} | {name[1]}"  # model | error | feature

        AECP_vector = group['robustness_index'].values.astype(float)
        epc_vector = group['epc'].values.astype(float)

        y_matrix = np.array(group['y'].tolist(), dtype=float)
        x_points = np.array(group['x'].iloc[0], dtype=float)

        scenarios[scenario_key] = {
            'info': {
                'dataset': name[0],
                'feature': name[1],
                'errorType': name[2],
                'modelName': name[3],
                'metric': name[4]
            },
            'x_points': x_points,
            'AECP_vector': AECP_vector,
            'epc_vector': epc_vector,
            'y_matrix': y_matrix,
            'n_runs': len(group)
        }

    return scenarios



# ============================================================
# 4) Plot ibrido
# ============================================================
def plot_hybrid_robustness(
    scenario_name,
    x_points,
    y_matrix,
    ax,
    AECP,
    EPC,
    dataset_name,
    performance_metric,
    confidence=0.95,
    target_span=0.30,
    pad=0.05,
    y_clip=(0.0, 1.05),
    y_mode="mean",
    slope_fmt="{:+.3f}",
    flat_eps=1e-12,
    text_fontsize=13,
    text_offset_frac=0.03
):
    x_points = np.asarray(x_points, dtype=float)
    y_matrix = np.asarray(y_matrix, dtype=float)

    mean_y, ci_y_low, ci_y_high = pointwise_mean_ci(y_matrix, confidence=confidence)
    baseline_y = float(mean_y[0])

    ax.plot(
        x_points,
        np.full_like(x_points, baseline_y),
        linestyle="--", color="black", linewidth=1.2, alpha=0.7, zorder=1
    )

    # aree: verde se sopra baseline, rosa se sotto
    ax.fill_between(
        x_points, baseline_y, mean_y,
        where=(mean_y >= baseline_y),
        interpolate=True, color="lightgreen", alpha=0.25, zorder=0
    )
    ax.fill_between(
        x_points, baseline_y, mean_y,
        where=(mean_y < baseline_y),
        interpolate=True, color="lightpink", alpha=0.25, zorder=0
    )

    ax.plot(
        x_points, mean_y,
        color="lightgray", linestyle="--", linewidth=1, zorder=2
    )

    # breakpoints da cambio segno slope della curva media
    mean_slopes = np.diff(mean_y) / np.diff(x_points)

    sign_eps = 1e-6
    def sgn(m):
        if abs(m) <= sign_eps:
            return 0
        return 1 if m > 0 else -1

    seg_sign = np.array([sgn(m) for m in mean_slopes], dtype=int)
    break_idx = [i for i in range(1, len(seg_sign)) if seg_sign[i] != seg_sign[i-1]]

    for i in break_idx:
        ax.axvline(
            x=x_points[i], color="black", linestyle="--",
            linewidth=1.4, alpha=0.6, zorder=10
        )

    n_seg = len(x_points) - 1
    seg_cat = ["skip"] * n_seg

    for i in range(n_seg):
        dx = x_points[i+1] - x_points[i]
        if dx == 0:
            continue
        slopes = (y_matrix[:, i+1] - y_matrix[:, i]) / dx
        m, low, high, _ = get_confidence_interval(slopes, confidence=confidence)
        is_sig = (high < 0) or (low > 0)
        if not is_sig:
            seg_cat[i] = "ns"
        else:
            if abs(m) <= flat_eps:
                seg_cat[i] = "flat"
            elif m < 0:
                seg_cat[i] = "down"
            else:
                seg_cat[i] = "up"

    groups = []
    i = 0
    while i < n_seg:
        if seg_cat[i] == "skip":
            i += 1
            continue
        cat = seg_cat[i]
        j = i
        while j + 1 < n_seg and seg_cat[j + 1] == cat:
            if cat == "ns" and seg_sign[j + 1] != seg_sign[j]:
                break
            j += 1
        groups.append((i, j, cat))
        i = j + 1

    if y_mode == "all":
        y_all = y_matrix.flatten()
    else:
        y_all = mean_y

    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    span = y_max - y_min

    if span < target_span:
        center = (y_max + y_min) / 2
        y_lower = center - target_span / 2
        y_upper = center + target_span / 2
    else:
        y_lower = y_min - pad
        y_upper = y_max + pad

    if y_clip is not None:
        y_lower = max(y_clip[0], y_lower)
        y_upper = min(y_clip[1], y_upper)

    if y_upper <= y_lower:
        y_upper = y_lower + 1e-6

    ax.set_ylim(y_lower, y_upper)
    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    y_offset = text_offset_frac * y_span

    for (seg_start, seg_end, cat) in groups:
        idx_start = seg_start
        idx_end = seg_end + 1

        if cat == "ns":
            xs = x_points[idx_start:idx_end+1]
            lo = ci_y_low[idx_start:idx_end+1]
            hi = ci_y_high[idx_start:idx_end+1]
            ax.fill_between(xs, lo, hi, interpolate=True, alpha=0.20, zorder=2)

        x0, x1 = x_points[idx_start], x_points[idx_end]
        dx_group = x1 - x0
        if dx_group == 0:
            continue

        group_slopes = (y_matrix[:, idx_end] - y_matrix[:, idx_start]) / dx_group
        gm, glow, ghigh, _ = get_confidence_interval(group_slopes, confidence=confidence)
        group_sig = (ghigh < 0) or (glow > 0)

        if cat == "ns":
            seg_color, seg_lw, text_color = "gray", 1.5, "black"
        elif cat == "down":
            seg_color, seg_lw, text_color = "red", 3.5, "red"
        elif cat == "up":
            seg_color, seg_lw, text_color = "green", 3.5, "green"
        else:
            seg_color, seg_lw, text_color = "gray", 3.0, "black"

        for k in range(idx_start, idx_end):
            ax.plot(
                [x_points[k], x_points[k+1]],
                [mean_y[k], mean_y[k+1]],
                color=seg_color, linewidth=seg_lw,
                solid_capstyle="round",
                zorder=4 if cat != "ns" else 3
            )

        x_v = x_points[idx_end]
        if not np.isclose(x_v, x_points[0]):
            ax.axvline(
                x=x_v, color="gray", linestyle=(0, (3, 3)),
                linewidth=0.9, alpha=0.25, zorder=1
            )

        x_mid = 0.5 * (x0 + x1)
        y_mid = 0.5 * (mean_y[idx_start] + mean_y[idx_end])
        direction = -1 if gm < 0 else 1

        ax.text(
            x_mid, y_mid + direction * y_offset,
            slope_fmt.format(gm * 100),
            ha="center", va="center", fontsize=text_fontsize,
            fontweight="bold" if (group_sig and cat != "ns") else "normal",
            color=text_color,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="none", alpha=0.8),
            zorder=5
        )

    ax.set_title(
        f"{dataset_name}\nSignificance Analysis: {scenario_name}\n"
        f"AECP: {AECP:.2f}% - EPC: {EPC:.2f}"
    )
    ax.set_xlabel("Error Percentage (%)")
    ax.set_ylabel(performance_metric + " Value")

    ax.set_title(
        f"{dataset_name}\nSignificance Analysis: {scenario_name}\n"
        f"AEPC: {AECP:.2f}% - EPC: {EPC:.2f}",
        fontsize=13
    )
    ax.set_xlabel("Error Percentage (%)", fontsize=12)
    ax.set_ylabel(performance_metric + " Value", fontsize=12)
    ax.tick_params(axis='both', labelsize=11)

    ax.grid(True, alpha=0.3)


# ============================================================
# 5) MAIN: load -> compute -> BY-FDR -> filter -> report -> plot
# ============================================================
if __name__ == "__main__":

    # --- settings ---
    #dataset_name = 'online_shoppers_intentionCorrelatedFeatures.csv'
    #dataset_name = 'online_shoppers_intention.csv'
    
    #dataset_name = 'online_shoppers_intentionOneFeatures.csv'
    dataset_name = 'iris.csv'
    metric = "AMI"
    # AECP in percentuale: soglia pratica 5% come da paper
    AECP_ABS_THRESHOLD = 0.0

    filename = 'experiments_' + dataset_name
    percorso_file = './experiments/' + filename

    df_caricato = pd.read_csv(percorso_file)

    # EPC + JSON curve
    clean_name = dataset_name.replace('.csv', '')
    file_path = f"./CurveData/{clean_name}_curve.json"

    if os.path.exists(file_path):
        json_path = file_path
    else:
        epc_df = epc_calcolus.start(dataset_name, filename, metric)
        json_path = analyze_robustness(df_caricato, metric, epc_df, dataset_name)

    results_dict = extract_vectors_from_json(json_path)

    # -------------------------------------------------------
    # FASE 1: calcola statistiche per TUTTI gli scenari
    # usando Wilcoxon signed-rank (non parametrico, vs mediana=0)
    # -------------------------------------------------------
    all_stats = []

    for scenario_name, data in results_dict.items():
        AECP_vec = np.asarray(data['AECP_vector'], dtype=float)
        epc_vec = np.asarray(data['epc_vector'], dtype=float)
        epc_vec = epc_vec[~np.isnan(epc_vec)]

        if "Labels" in scenario_name:
            print(f"Scenario: {scenario_name}  EPC values: {epc_vec}")

        if len(AECP_vec) < 2:
            continue

        AECP_mean, AECP_low, AECP_high, AECP_p = get_confidence_interval_wilcoxon(AECP_vec)
        EPC_mean, EPC_low, EPC_high, EPC_p = get_confidence_interval_wilcoxon(epc_vec)

        all_stats.append({
            'scenario_name': scenario_name,
            'data': data,
            'AECP_mean': AECP_mean,
            'AECP_low': AECP_low,
            'AECP_high': AECP_high,
            'AECP_p': AECP_p,
            'EPC_mean': EPC_mean,
            'EPC_low': EPC_low,
            'EPC_high': EPC_high,
            'EPC_p': EPC_p,
        })

    print(f"\nScenari totali valutati: {len(all_stats)}")

    # -------------------------------------------------------
    # FASE 2: BY-FDR su p-value AECP
    # Controlla FDR sotto dipendenza arbitraria tra test
    # (più conservativo di BH ma corretto per strutture di dipendenza
    # ignote, come nel nostro caso: stessi dataset e modelli condivisi)
    # -------------------------------------------------------
    AECP_pvals = np.array([s['AECP_p'] for s in all_stats])
    AECP_pvals = np.nan_to_num(AECP_pvals, nan=1.0)

    reject_fdr, pvals_corrected, _, _ = multipletests(
        AECP_pvals, alpha=0.05, method='fdr_by'
    )

    n_sig_fdr = int(np.sum(reject_fdr))
    print(f"Scenari significativi dopo BY-FDR (alpha=0.05): {n_sig_fdr}")

    # -------------------------------------------------------
    # FASE 3: filtro pratico su |AECP_mean| > soglia
    # -------------------------------------------------------
    rows = []
    significant_scenarios = []

    for i, s in enumerate(all_stats):
        # criterio 1: significativo dopo BY-FDR su AECP
        if not reject_fdr[i]:
            continue

        # criterio 2: rilevanza pratica |AECP_mean| > 5%
        if abs(s['AECP_mean']) <= AECP_ABS_THRESHOLD:
            continue

        rows.append({
            "Scenario": s['scenario_name'],
            "n_runs": s['data']['n_runs'],
            "AECP Mean (%)": f"{s['AECP_mean']:.4f}",
            "AECP CI 95%": f"[{s['AECP_low']:.4f}, {s['AECP_high']:.4f}]",
            "AECP p raw": f"{s['AECP_p']:.4g}",
            "AECP p BY": f"{pvals_corrected[i]:.4g}",
            "EPC Mean": f"{s['EPC_mean']:.4f}",
            "EPC CI 95%": f"[{s['EPC_low']:.4f}, {s['EPC_high']:.4f}]",
        })

        significant_scenarios.append({
            "scenario_name": s['scenario_name'],
            "x_points": s['data']['x_points'],
            "y_matrix": s['data']['y_matrix'],
            "AECP_mean": s['AECP_mean'],
            "AECP_ci": (s['AECP_low'], s['AECP_high']),
            "AECP_p_raw": s['AECP_p'],
            "AECP_p_by": pvals_corrected[i],
            "EPC_mean": s['EPC_mean'],
            "EPC_ci": (s['EPC_low'], s['EPC_high']),
        })

    # -------------------------------------------------------
    # Report
    # -------------------------------------------------------
    if not rows:
        print("Nessuno scenario significativo trovato.")
    else:
        df_sig = pd.DataFrame(rows).sort_values(["Scenario"]).reset_index(drop=True)
        print("\n" + "=" * 140)
        print(f"{'SCENARI SIGNIFICATIVI (BY-FDR + soglia pratica |AECP| > 5%)':^140}")
        print("=" * 140)
        print(df_sig.to_string(index=False))
        print("=" * 140)
        print(f"Scenari significativi finali: {len(significant_scenarios)}")

        # Plot

    for sc in significant_scenarios:
        fig = plt.figure(figsize=(14, 7))
        ax = plt.gca()

        plot_hybrid_robustness(
            scenario_name=sc["scenario_name"],
            x_points=sc["x_points"],
            y_matrix=sc["y_matrix"],
            ax=ax,
            AECP=sc["AECP_mean"],
            EPC=sc["EPC_mean"],
            dataset_name=dataset_name,
            performance_metric=metric,
            text_fontsize=13
        )

        txt = (
            f"AECP mean {sc['AECP_mean']:.3f}%  CI95% [{sc['AECP_ci'][0]:.3f}, {sc['AECP_ci'][1]:.3f}]\n"
            f"AECP p-raw {sc['AECP_p_raw']:.4g}  p-BY {sc['AECP_p_by']:.4g}\n"
            f"EPC  mean {sc['EPC_mean']:.3f}   CI95% [{sc['EPC_ci'][0]:.3f}, {sc['EPC_ci'][1]:.3f}]"
        )
        ax.text(
            0.01, 0.01, txt,
            transform=ax.transAxes,
            ha="left", va="bottom", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                    alpha=0.85, edgecolor="none")
        )

        plt.tight_layout()
        
        # sanitizza il nome file rimuovendo caratteri non validi
        safe_name = re.sub(r'[\\/:*?"<>|]', '_', sc['scenario_name'])
        safe_name = safe_name.replace(' ', '_')
        #plt.savefig(f"{safe_name}.png", dpi=300, bbox_inches='tight') per salvare le immagini
        plt.show()
        plt.close()