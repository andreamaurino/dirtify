
#from multiprocessing.reduction import duplicate
import socket
from pucktrick.noisy import *
from pucktrick.labels import *
from pucktrick.duplicated import *
from pucktrick.missing import *
from pucktrick.outliers import *
import cvxopt as opt
from cvxopt import blas, solvers
import category_encoders
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd
import duckdb
from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.metrics import matthews_corrcoef
import json
import os
import csv
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from strategy import RunStrategy
from sklearn.impute import SimpleImputer
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score
from sklearn.cluster import (
    KMeans,
    HDBSCAN,
    AgglomerativeClustering,
    SpectralClustering,
    MeanShift,
    AffinityPropagation,
    
)
import pycaret.classification as pycaret_clf
import pycaret.regression as pycaret_reg
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from hdbscan import HDBSCAN


from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances_argmin

CLUSTERING_MODELS = {
    "K-Means":                  KMeans(),
    "HDBSCAN":                  HDBSCAN(),
    "Hierarchical Clustering":  AgglomerativeClustering(),
    "Gaussian Mixture Model":   GaussianMixture(),
    "BIRCH":                    Birch(),
}

def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2cor = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    return np.sqrt(phi2cor / min((kcorr - 1), (rcorr - 1)))

def calculate_feature_target_correlation_after_error(modified_data, target_variable, error_type, percentage, feature, featureType):
    correlations = []
    
    # Pre-elaborazione dei dati: rimuovere NaN e infiniti solo nelle colonne di interesse
    modified_data = modified_data[[feature, target_variable]].dropna()
    modified_data = modified_data[~modified_data.isin([np.inf, -np.inf]).any(axis=1)]

    # Funzione per aggiungere una correlazione se valida
    def add_correlation(correlation_type, corr_value):
        if not np.isnan(corr_value):  
            correlations.append({
                "Feature": feature,
                "Error Type": error_type,
                "Percentage": percentage,
                "Correlation": corr_value,
                "Correlation Type": correlation_type
            })
        else:
            print(f"Correlation value not valid for  {feature}. Ignored.")

    # Selezione del tipo di correlazione basato su featureType
    try:
        if featureType in ["continous", "discrete"]:
            corr, _ = pointbiserialr(modified_data[feature], modified_data[target_variable])
            add_correlation("Point-Biserial", corr)

        elif featureType in ["categoricalString", "categoricalInt"]:
            contingency_table = pd.crosstab(modified_data[feature], modified_data[target_variable])
            cramer_v_value = cramers_v(contingency_table.values)
            add_correlation("Cramér's V", cramer_v_value)

        elif featureType == "binary":
            if len(modified_data[feature].unique()) == 2:  # Verifica se è binaria
                corr = matthews_corrcoef(modified_data[feature], modified_data[target_variable])
                add_correlation("Matthews Correlation Coefficient", corr)
            else:
                print(f"{feature} variable is not binary. Unique values: {modified_data[feature].unique()}")

    except Exception as e:
        print(f"Error in correlation calculus for  {feature}: {e}")

    # Creazione del DataFrame e salvataggio in CSV
    if correlations:
        results_df = pd.DataFrame(correlations)
        results_filename = 'correlation_results.csv'
        if os.path.exists(results_filename):
            # Append se il file esiste già
            results_df.to_csv(results_filename, mode='a', header=False, index=False)
        else:
            # Crea un nuovo file con intestazione
            results_df.to_csv(results_filename, index=False)
        print(f"correlations saved in {results_filename}.")
    else:
        print("No correlation found.")

def calculate_permutation_importance(modified_data, target_variable, error_type, percentage):
    # Rimuovere NaN e infiniti
    modified_data = modified_data.dropna()
    modified_data = modified_data[~modified_data.isin([np.inf, -np.inf]).any(axis=1)]
    
    # Separare feature e target
    X = modified_data.drop(columns=[target_variable])
    y = modified_data[target_variable]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modello (puoi scegliere un altro)
    model = LinearRegression(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calcolo della permutation importance
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

    # Creazione DataFrame con i risultati
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": perm_importance.importances_mean,
        "Error Type": error_type,
        "Percentage": percentage
    })

    # Ordinare per importanza
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    
    # Salvare il risultato
    results_filename = 'feature_importance_results.csv'
    if os.path.exists(results_filename):
        importance_df.to_csv(results_filename, mode='a', header=False, index=False)
    else:
        importance_df.to_csv(results_filename, index=False)

    print(f"Feature importance saved in {results_filename}.")
    return importance_df


def save_results_to_csv(results, output_file="synthetic_data_analysis_results.csv"):
    """
    Save results into one csv file.
    """
    if not results:
        print("Error: 'results' is empty.")  # Debug
        return

    if not isinstance(results, list) or not all(isinstance(r, dict) for r in results):
        print("Error: 'results' is not a list.")  # Debug
        return

    write_header = not os.path.exists(output_file)
    
    with open(output_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        
        if write_header:
            writer.writerow([
                "datasetName",
                "errorType",
                "percentage",
                "feature",
                "modelName",
                "Accuracy",
                "Auc",
                "Recall",
                "Precision",
                "F1",
            ])
        
        for result in results:
            writer.writerow([
                result.get("datasetName", ""),
                result.get("errorType", ""),
                result.get("percentage", ""),
                result.get("feature", ""),
                result.get("modelName", ""),
                result.get("Accuracy", ""),
                result.get("Auc", ""),
                result.get("Recall", ""),
                result.get("Precision", ""),
                result.get("F1", ""),
            ])

def _classification_metrics(stg: RunStrategy):
        stg.target_variable = stg.target_variable[0] if isinstance(stg.target_variable, list) else stg.target_variable
        s = pycaret_clf.setup(stg.train_df, target=stg.target_variable, session_id=123)
        models = pycaret_clf.compare_models(include=stg.models, n_select=20)
        if not isinstance(models, list):
            models = [models]
        for m in models:
            predictions = pycaret_clf.predict_model(m, data=stg.test_df)
            y_true = predictions[stg.target_variable]
            y_pred = predictions['prediction_label']
            model_name = m.__class__.__name__
            accuracy = round(accuracy_score(y_true, y_pred), 4)
            precision = round(precision_score(y_true, y_pred), 4)
            recall = round(recall_score(y_true, y_pred), 4)
            f1 = round(f1_score(y_true, y_pred), 4)
            auc = None
            if "prediction_score" in predictions.columns:
                auc = round(roc_auc_score(y_true, predictions['prediction_score']), 4) 
                stg.con.execute("""
                    INSERT INTO experiments 
                    VALUES (?,?, ?, ?, ?, ?, ?, ?, ?, ?, ?,NULL, NULL, NULL)
                    """, [
                    stg.run,
                    stg.dataset_name,
                    str(stg.EType),  # o json.dumps(EType)
                    stg.percentage,
                    stg.feature,
                    model_name,
                    accuracy,
                    auc,
                    recall,
                    precision,
                    f1
                    ])
            else:
                stg.con.execute("""
                    INSERT INTO experiments 
                    VALUES (?,?, ?, ?, ?, ?, ?, NULL, ?, ?, ?,NULL, NULL, NULL)
                        """, [
                    stg.run,
                    stg.dataset_name,
                    str(stg.EType),  # o json.dumps(EType)
                    stg.percentage,
                    stg.feature,
                    model_name,
                    accuracy,
                    recall,
                    precision,
                    f1
                    ])

def _regression_metrics(stg: RunStrategy):

    stg.target_variable = stg.target_variable[0] if isinstance(stg.target_variable, list) else stg.target_variable
    s = pycaret_reg.setup(stg.train_df, target=stg.target_variable, session_id=123)
    models = pycaret_reg.compare_models(include=stg.models, n_select=20)
    
    if not isinstance(models, list):
        models = [models]

    for m in models:
        predictions = pycaret_reg.predict_model(m, data=stg.test_df)
        y_true = predictions[stg.target_variable]
        y_pred = predictions['prediction_label']
        model_name = m.__class__.__name__

        rmse = float(round(np.sqrt(mean_squared_error(y_true, y_pred)), 4))
        mae  = float(round(mean_absolute_error(y_true, y_pred), 4))
        r2   = float(round(r2_score(y_true, y_pred), 4))
        mse  = float(round(mean_squared_error(y_true, y_pred), 4))

        stg.con.execute("""
            INSERT INTO experiments
            VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, ?, ?, ?, ?)
            """, [
            stg.run,
            stg.dataset_name,
            str(stg.EType),
            stg.percentage,
            stg.feature,
            model_name,
            rmse,
            mae,
            r2,
            mse
        ])


def _fresh_model(model_name: str, n_clusters: int):
    return {
        "K-Means":                  KMeans(n_clusters=n_clusters, random_state=42),
        "Gaussian Mixture Model":   GaussianMixture(n_components=n_clusters, random_state=42),
        "Hierarchical Clustering":  AgglomerativeClustering(n_clusters=n_clusters),
        "HDBSCAN":                  HDBSCAN(prediction_data=True),
        "BIRCH":                    Birch(n_clusters=n_clusters)
    }[model_name]

def get_n_clusters(stg, train_features: pd.DataFrame) -> int:
    if stg.n_clusters is not None:
        return stg.n_clusters
    
    best_k = 2
    best_score = -1
    
    for k in range(2, 11):  # cerca tra 2 e 10
        labels = KMeans(n_clusters=k, random_state=42).fit_predict(train_features)
        score = silhouette_score(train_features, labels)
        if score > best_score:
            best_score = score
            best_k = k
    
    stg.n_clusters = best_k  # salva in stg per i run successivi
    return best_k

def _clustering_metrics(stg: RunStrategy):
    
    has_target = (
        stg.target_variable is not None and
        stg.target_variable in stg.train_df.columns
    )

    if stg.noisy_df is not None and len(stg.noisy_df) > 0:
        train_source = stg.noisy_df
    else:
        train_source = stg.train_df

    if has_target:
        train_features = train_source.drop(columns=[stg.target_variable])
        test_features  = stg.test_df.drop(columns=[stg.target_variable])
    else:
        train_features = train_source
        test_features  = stg.test_df

    imputer = SimpleImputer(strategy='mean')
    train_features = pd.DataFrame(
        imputer.fit_transform(train_features),
        columns=train_features.columns
    )
    test_features = pd.DataFrame(
        imputer.transform(test_features),
        columns=test_features.columns
    )

    assert not train_features.isnull().any().any(), f"NaN in train: {train_features.isnull().sum()}"
    assert not test_features.isnull().any().any(), f"NaN in test: {test_features.isnull().sum()}"

    n_clusters = get_n_clusters(stg, train_features)

    if has_target:
        y_true = stg.test_df[stg.target_variable]

    for model_name in stg.models:
        if model_name not in CLUSTERING_MODELS:
            print(f"Modello {model_name} non riconosciuto, skippato")
            continue

        model = _fresh_model(model_name, n_clusters)
        labels = None

        try:
            if type(model).__name__ == 'HDBSCAN':
                model.fit(train_features)
                train_labels = model.labels_
                unique_labels = np.unique(train_labels[train_labels != -1])
                if len(unique_labels) < 2:
                    print(f"HDBSCAN: meno di 2 cluster trovati, skip")
                    continue
                centroids = np.array([
                    train_features.values[train_labels == i].mean(axis=0)
                    for i in unique_labels
                ])
                labels = pairwise_distances_argmin(test_features.values, centroids)

            elif hasattr(model, 'predict'):
                # K-Means, GMM, BIRCH
                model.fit(train_features)
                labels = model.predict(test_features)

            else:
                # Hierarchical Clustering (AgglomerativeClustering)
                model.fit(train_features)
                if model.labels_ is None:
                    print(f"Warning {model_name}: labels_ è None, skip")
                    continue
                centroids = np.array([
                    train_features.values[model.labels_ == i].mean(axis=0)
                    for i in range(n_clusters)
                ])
                labels = pairwise_distances_argmin(test_features.values, centroids)

            if labels is None:
                print(f"Warning {model_name}: labels non calcolate, skip")
                continue

            k = len(np.unique(labels[labels != -1]))

            silhouette = None
            ami        = None

            if k > 1:
                silhouette = round(silhouette_score(test_features, labels), 4)

            if len(y_true) == len(labels):
                ami = round(adjusted_mutual_info_score(y_true, labels), 4)
            else:
                print(f"Warning {model_name}: dimensioni diverse, AMI skippato")

            print(f"Model: {model_name}, Silhouette: {silhouette}, AMI: {ami}, Clusters: {k}")
            stg.con.execute("""
                INSERT INTO experiments 
                VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, ?, ?, ?)
            """, [
                stg.run, stg.dataset_name, str(stg.EType),
                stg.percentage, stg.feature, model_name,
                ami, silhouette, k
            ])

        except Exception as e:
            print(f"Errore con {model_name}: {e}")
            continue

def performanceAnalysis(stg: RunStrategy): #, con, dataset_name, train_df, test_df, target_column, model_touse, EType="NULL", feature="NULL", percentage=0):
    if stg.task=="Classification":
        _classification_metrics(stg)
    elif stg.task=="Regression":
        _regression_metrics(stg)
    elif stg.task=="Clustering":
        _clustering_metrics(stg)
    else :
        print(f"Task {stg.task} not supported.")




def AnalyzeValues(stg:RunStrategy):   
    stg.noisy_df = stg.train_df.copy()
    local_noisy_df=stg.noisy_df.copy()
    train_df=stg.train_df.copy()
    if stg.EType == "label":
          stg.feature = stg.target_variable
    elif stg.EType != "duplicated":
        stg.feature = stg.strategy.get("affected_features")
    else: #duplicated
        stg.feature = stg.target_variable
    step = stg.strategy.get("percentage")
    stg.percentage=step
    while stg.percentage < 1:
            all_results = []
            stg.strategy["mode"]="extended"
            stg.percentage=round(stg.percentage, 1)
            match   stg.EType:
                case "noise":
                    error,local_noisy_df = noise(local_noisy_df, stg.strategy,train_df)
                case "missing":
                    error,local_noisy_df = missing(local_noisy_df, stg.strategy,train_df)
                case "outlier":
                    error,local_noisy_df = outlier(local_noisy_df, stg.strategy,train_df)
                case "label":           
                    error,local_noisy_df = labels(local_noisy_df, stg.strategy,train_df)
                case "duplicate": 
                    error,local_noisy_df = duplicate(local_noisy_df, stg.strategy,train_df)
            stg.noisy_df=local_noisy_df.copy()
            results = performanceAnalysis(stg)
            all_results.append(results)
            stg.percentage += step
            stg.strategy["percentage"]=stg.percentage
    stg.strategy["percentage"]=step
    return all_results
        