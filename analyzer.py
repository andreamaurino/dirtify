
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


#common functions
def performanceAnalysis(run, con, dataset_name, train_df, test_df, target_column, model_touse, EType="NULL", feature="NULL", percentage=0):
    included_models = model_touse
    s = setup(train_df, target=target_column, session_id=123)

    models = compare_models(include=included_models, n_select=20)
    if not isinstance(models, list):
        models = [models]

    for m in models:
        predictions = predict_model(m, data=test_df)
        y_true = predictions[target_column]
        y_pred = predictions['prediction_label']
        
        model_name = m.__class__.__name__
        accuracy = round(accuracy_score(y_true, y_pred), 4)
        precision = round(precision_score(y_true, y_pred), 4)
        recall = round(recall_score(y_true, y_pred), 4)
        f1 = round(f1_score(y_true, y_pred), 4)
        print("y_tue "+str(y_true.isna().sum()))
        auc = None
        
     #Passiamo il risultato come lista per salvarlo singolarmente
        if "prediction_score" in predictions.columns:
            auc = round(roc_auc_score(y_true, predictions['prediction_score']), 4) 
            
            con.execute("""
    INSERT INTO experiments 
    VALUES (?,?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", [
    run,
    dataset_name,
    str(EType),  # o json.dumps(EType)
    percentage,
    feature,
    model_name,
    accuracy,
    auc,
    recall,
    precision,
    f1
])
        else:
            con.execute("""
    INSERT INTO experiments 
    VALUES (?,?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
""", [
    run,
    dataset_name,
    str(EType),  # o json.dumps(EType)
    percentage,
    feature,
    model_name,
    accuracy,
    recall,
    precision,
    f1
])
            
#Noise
def AnalyzeNoiseValues(run,con,dataset_name, strategy, train_df, test_df, target_variable, model_touse):
    
    EType = "Noisy"
    noisy_df = train_df.copy()
    all_results = []

    columns = strategy.get("affected_features")
    step = strategy.get("percentage")
    percentage=step
    #for i in range(len(columns)): 
    while percentage < 1:
            print("Noise error")
            #print(f"Feature: {columns[i]}")
            print("***********************************")
            print("*********** RUN:"+str(run))
            print("***********************************")
            print("feature: " + ", ".join(columns))
            print(f"Step: {round(percentage, 1)}")
            strategy["mode"]="new"
            error,noisy_df = noise(noisy_df, strategy,train_df)
            results = performanceAnalysis(run,con,dataset_name, noisy_df, test_df, target_variable, model_touse, EType, columns, round(percentage, 1))
            all_results.append(results)

            percentage += step

    return all_results

#labels
def AnalyzeWrongLabels(run,con,dataset_name,target_variable,strategy,train_df,test_df, model_touse, output_file="synthetic_data_analysis_results.csv"):
  errorType="Labels"
  target_variable = strategy.get("affected_features")
  step = strategy.get("percentage")
  percentage=step
  noisy_df=train_df.copy()
  while percentage<1:
    print("***********************************")
    print("*********** RUN:"+str(run))
    print("***********************************")
    print ("labels error")
    print("target:", target_variable)
    print ("step:"+str(round(percentage,1)))
    strategy["mode"]="extended"
    error,noisy_df=labels(noisy_df,strategy,train_df)
    all_results = []
    results = performanceAnalysis(run, con,dataset_name, noisy_df,test_df, target_variable, model_touse, errorType, target_variable, round(percentage,1))
    all_results.append(results)
    
    percentage+=step
  return all_results

#duplicated
def AnalyzeDuplicatedValues(run,con,dataset_name,target_variable,strategy,train_df,test_df, model_touse):
  errorType="Duplicated"
  step = strategy.get("percentage")
  percentage=step
  noisy_df=train_df.copy()
  while percentage<1:
    print("***********************************")
    print("*********** RUN:"+str(run))
    print("***********************************")

    print ("Duplicate error")
    print ("step:"+str(round(percentage,1)))
    strategy["mode"]="new"
    error,noisy_df=duplicate(noisy_df,strategy,train_df)
    all_results = []
    results = performanceAnalysis(run, con,dataset_name, noisy_df,test_df, target_variable, model_touse, errorType, target_variable, round(percentage,1))
    all_results.append(results)
    percentage+=step
  return all_results


#missing
def AnalyzeMissingValues(run, con,dataset_name,strategy,train_df,test_df, target_variable, model_touse):
  EType="Missing"
  columns = strategy.get("affected_features")
  step = strategy.get("percentage")
  noisy_df=train_df.copy()
  #for i in range(len(columns)):
  percentage=step
  while percentage<1:
      print("***********************************")
      print("*********** RUN:"+str(run))
      print("***********************************")

      print ("Missing error")
  #    print ("feature: "+columns[i])
      print("feature: " + ", ".join(columns))
      print ("step:"+str(round(percentage,1)))
      strategy["mode"]="new"
      error,noisy_df=missing(noisy_df,strategy,train_df)
      all_results = []
      results = performanceAnalysis(run, con,dataset_name,noisy_df,test_df, target_variable, model_touse, EType, columns,  round(percentage,1))
      all_results.append(results)
      percentage+=step
  return all_results

#outlier
def AnalyzeOutlierValues(run,con,dataset_name, strategy,train_df,test_df, target_variable, model_touse):
  EType="Outlier"
  columns = strategy.get("affected_features")
  step = strategy.get("percentage")
  noisy_df=train_df.copy()
  #for i in range(len(columns)):
  percentage=step
  while percentage<1:
      print("***********************************")
      print("*********** RUN:"+str(run))
      print("***********************************")

      print("Outlier error")
#      print ("feature: "+columns[i])
      print("feature: " + ", ".join(columns))
      print ("step:"+str(round(percentage,1)))
      strategy["mode"]="new"
      error,noisy_df=outlier(noisy_df,strategy,train_df)      
      all_results = []
      results = performanceAnalysis(run, con,dataset_name,noisy_df,test_df, target_variable, model_touse, EType, columns,  round(percentage,1))
      all_results.append(results)
      percentage+=step
  return all_results



        

   
