
#from multiprocessing.reduction import duplicate
import socket
from pucktrick.noisy import *
from pucktrick.labels import *
from pucktrick.duplicated import *
from pucktrick.missing import *
from pucktrick.outliers import *
from pycaret.classification import *
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
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering,
    SpectralClustering,
    MeanShift,
    AffinityPropagation,
    OPTICS
)
from sklearn.mixture import GaussianMixture

CLUSTERING_MODELS = {
    "K-Means":                  KMeans(),
    "DBSCAN":                   DBSCAN(),
    "Hierarchical Clustering":  AgglomerativeClustering(),
    "Gaussian Mixture Model":   GaussianMixture(),
    "Spectral Clustering":      SpectralClustering(),
    "Agglomerative Clustering": AgglomerativeClustering(),
    "Mean Shift":               MeanShift(),
    "Affinity Propagation":     AffinityPropagation(),
    "OPTICS":                   OPTICS(),
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
        
        s = setup(stg.train_df, target=stg.target_variable, session_id=123)
        models = compare_models(include=stg.models, n_select=20)
        if not isinstance(models, list):
            models = [models]
        for m in models:
            predictions = predict_model(m, data=stg.test_df)
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
        print(f"Task {stg.task} not supported.")

def _fresh_model(model_name: str, n_clusters: int):
    return {
        "K-Means":                  KMeans(n_clusters=n_clusters, random_state=42),
        "DBSCAN":                   DBSCAN(),                            
        "Hierarchical Clustering":  AgglomerativeClustering(n_clusters=n_clusters),
        "Gaussian Mixture Model":   GaussianMixture(n_components=n_clusters,random_state=42),
        "Spectral Clustering":      SpectralClustering(n_clusters=n_clusters, random_state=42),
        "Agglomerative Clustering": AgglomerativeClustering(n_clusters=n_clusters),
        "Mean Shift":               MeanShift(),                         
        "Affinity Propagation":     AffinityPropagation(random_state=42),               
        "OPTICS":                   OPTICS(),                            
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
    if has_target:
        train_features = stg.train_df.drop(columns=[stg.target_variable])
        test_features  = stg.test_df.drop(columns=[stg.target_variable])
    else:
        train_features = stg.train_df
        test_features  = stg.test_df

    n_clusters = get_n_clusters(stg, train_features)

    for model_name in stg.models:
        model = _fresh_model(model_name, n_clusters)
        if model_name not in CLUSTERING_MODELS:
            print(f"Modello {model_name} non riconosciuto, skippato")
            continue
        try:
            if hasattr(model, 'predict'):
                model.fit(train_features)
                labels = model.predict(test_features)
            else:
                labels = model.fit_predict(test_features)

            # ← y_true calcolato QUI, dopo labels, e dipende da has_target
            if has_target:
                y_true = stg.test_df[stg.target_variable]  # etichette reali, sempre
            else:
                if model_name not in stg.ground_truth_labels:
                    stg.ground_truth_labels[model_name] = pd.Series(labels, name="cluster_label")
                y_true = stg.ground_truth_labels[model_name]

            k = len(np.unique(labels[labels != -1]))

            silhouette = None
            ami        = None

            if k > 1:
                silhouette = round(silhouette_score(test_features, labels), 4)

            if len(y_true) == len(labels):
                #da modificare in AEPC = AUC(ami_noisy - ami_clean) / (2 * max_error)
                #poi si deve calcolare il maxerror ch è la percentale massima di errore e.g. 0.8
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
    if stg.task=="classification":
        _classification_metrics(stg)
    elif stg.task=="Regression":
        _regression_metrics(stg)
    elif stg.task=="Clustering":
        _clustering_metrics(stg)
    else :
        print(f"Task {stg.task} not supported.")

#labels
def AnalyzeWrongLabels(stg:RunStrategy):#run,con,dataset_name,target_variable,strategy,train_df,test_df, model_touse):
  stg.EType="Labels"
  stg.target_variable = stg.strategy.get("affected_features")
  stg.feature = stg.target_variable
  step = stg.strategy.get("percentage")
  stg.percentage=step
  stg.noisy_df=stg.train_df.copy()
  while stg.percentage<1:
    print("***********************************")
    print("*********** RUN:"+str(stg.run))
    print("***********************************")
    print ("labels error")
    print("target:", stg.target_variable)
    print ("step:"+str(round(stg.percentage,1)))
    stg.strategy["mode"]="extended"
    error,stg.noisy_df=labels(stg.noisy_df,stg.strategy,stg.train_df)
    all_results = []
    stg.percentage=round(stg.percentage, 1)
    results = performanceAnalysis(stg)#, con,dataset_name, noisy_df,test_df, target_variable, model_touse, errorType, target_variable, round(stg.percentage,1))
    all_results.append(results)
    
    stg.percentage+=step
  return all_results

#duplicated
def AnalyzeDuplicatedValues(stg:RunStrategy):#run,con,dataset_name,target_variable,strategy,train_df,test_df, model_touse):
  stg.EType="Duplicated"
  step = stg.strategy.get("percentage")
  stg.percentage=step
  stg.noisy_df=stg.train_df.copy()
  while stg.percentage<1:
    print("***********************************")
    print("*********** RUN:"+str(stg.run))
    print("***********************************")

    print ("Duplicate error")
    print ("step:"+str(round(stg.percentage,1)))
    stg.strategy["mode"]="extended"
    error,stg.noisy_df=duplicate(stg.noisy_df,stg.strategy,stg.train_df)
    all_results = []
    stg.percentage=round(stg.percentage, 1)
    results = performanceAnalysis(stg)#, con,dataset_name, noisy_df,test_df, target_variable, model_touse, errorType, target_variable, round(stg.percentage,1))
    all_results.append(results)
    stg.percentage+=step
  return all_results

#Noise
def AnalyzeNoiseValues(stg:RunStrategy):#run,con,dataset_name, strategy, train_df, test_df, target_variable, model_touse):
    
    stg.EType = "Noisy"
    stg.noisy_df = stg.train_df.copy()
    columns = stg.strategy.get("affected_features")
    step = stg.strategy.get("percentage")
    stg.percentage=step
    while stg.percentage < 1:
            all_results = []
            print("Noise error")
            #print(f"Feature: {columns[i]}")
            print("***********************************")
            print("*********** RUN:"+str(stg.run))
            print("***********************************")
            print("feature: " + ", ".join(columns))
            print(f"Step: {round(stg.percentage, 1)}")
            stg.strategy["mode"]="extended"
            error,stg.noisy_df = noise(stg.noisy_df, stg.strategy,stg.train_df)
            stg.percentage=round(stg.percentage, 1)
            results = performanceAnalysis(stg)#, con, dataset_name, noisy_df, test_df, target_variable, model_touse, stg.EType, columns, )
            all_results.append(results)

            stg.percentage += step

    return all_results

#missing
def AnalyzeMissingValues(stg:RunStrategy):#run,con,dataset_name,strategy,train_df,test_df, target_variable, model_touse):
  stg.EType="Missing"
  columns = stg.strategy.get("affected_features")
  step = stg.strategy.get("percentage")
  stg.noisy_df=stg.train_df.copy()
  stg.percentage=step
  while stg.percentage<1:
      print("***********************************")
      print("*********** RUN:"+str(stg.run))
      print("***********************************")

      print ("Missing error")
  #    print ("feature: "+columns[i])
      print("feature: " + ", ".join(columns))
      print ("step:"+str(round(stg.percentage,1)))
      stg.strategy["mode"]="extended"
      error,stg.noisy_df=missing(stg.noisy_df,stg.strategy,stg.train_df)
      all_results = []
      stg.percentage=round(stg.percentage, 1)
      results = performanceAnalysis(stg)#, con,dataset_name, noisy_df,test_df, target_variable, model_touse, EType, columns,  round(stg.percentage,1))
      all_results.append(results)
      stg.percentage+=step
  return all_results

#outlier
def AnalyzeOutlierValues(stg:RunStrategy):#run,con,dataset_name, strategy,train_df,test_df, target_variable, model_touse):
  stg.EType="Outlier"
  columns = stg.strategy.get("affected_features")
  step = stg.strategy.get("percentage")
  stg.noisy_df=stg.train_df.copy()
  stg.percentage=step   
  while stg.percentage<1:
      print("***********************************")
      print("*********** RUN:"+str(stg.run))
      print("***********************************")

      print("Outlier error")
      print("feature: " + ", ".join(columns))
      print ("step:"+str(round(stg.percentage,1)))
      stg.strategy["mode"]="extended"
      error,stg.noisy_df=outlier(stg.noisy_df,stg.strategy,stg.train_df)      
      all_results = []
      stg.percentage=round(stg.percentage, 1)
      results = performanceAnalysis(stg)#, con,dataset_name, noisy_df,test_df, target_variable, model_touse, EType, columns,  round(stg.percentage,1))
      all_results.append(results)
      stg.percentage+=step
  return all_results



def AnalyzeValues(stg:RunStrategy):   
    local_train_df = stg.train_df.copy()
    local_noise_df = stg.train_df.copy()
    stg.noisy_df = stg.train_df.copy()
    if stg.EType == "label":
        #  stg.target_variable = stg.strategy.get("affected_features")
        #  stg.target_variable = stg.target_variable[0] if isinstance(stg.target_variable, list) else stg.target_variable
          stg.feature = stg.target_variable
    elif stg.EType != "duplicated":
        stg.feature = stg.strategy.get("affected_features")
    else: #duplicated
        stg.feature = stg.target_variable
    step = stg.strategy.get("percentage")
    stg.percentage=step
    while stg.percentage < 1:
            all_results = []
            print("***********************************")
            print("*********** RUN:"+str(stg.run))
            print("***********************************")
            print(f"{stg.EType} error")
            if stg.EType == "label":
                print("target:", stg.target_variable)   
            elif stg.EType != "duplicated":
                print("feature: " + ", ".join(stg.feature))
            print(f"Step: {round(stg.percentage, 1)}")
            stg.strategy["mode"]="extended"
            match   stg.EType:
                case "noise":
                    error,stg.noisy_df = noise(stg.noisy_df, stg.strategy,stg.train_df)
                case "missing":
                    error,stg.noisy_df = missing(stg.noisy_df, stg.strategy,stg.train_df)
                case "outlier":
                    error,stg.noisy_df = outlier(stg.noisy_df, stg.strategy,stg.train_df)
                case "label":           
                    error,stg.noisy_df = labels(stg.noisy_df, stg.strategy,stg.train_df)
                case "duplicated": 
                    error,stg.noisy_df = duplicate(stg.noisy_df, stg.strategy,stg.train_df)

            stg.percentage=round(stg.percentage, 1)
            results = performanceAnalysis(stg)
            all_results.append(results)

            stg.percentage += step
    stg.train_df=local_train_df.copy()
    stg.noisy_df=local_noise_df.copy()
    return all_results
        