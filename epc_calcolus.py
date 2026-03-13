import numpy as np
import pandas as pd
from pycaret.classification import *
import cvxopt as opt
from cvxopt import blas, solvers
import category_encoders
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import duckdb
from pucktrick.noisy import *
from pucktrick.labels import *
import analyzer
import matplotlib.pyplot as plt
import os
#import img2pdf
#from pathlib import Path

def start(dataset_name, filename,performance_metric="Accuracy"):
   # Setup percorsi e connessione
    filename = f"experiments_{dataset_name}"
    experiment_path = os.path.join('experiments', filename)
    
    con = duckdb.connect(f"{dataset_name}.db")
    
    # 1. Caricamento dati originali
    con.execute("DROP TABLE IF EXISTS experiments")
    con.execute(f"CREATE TABLE experiments AS SELECT * FROM read_csv_auto('{experiment_path}')")
    
    # 2. Creazione tabelle EPC
    con.execute("DROP TABLE IF EXISTS epc")
    con.execute("""
        CREATE TABLE epc (
            experiment_run INTEGER, 
            datasetName VARCHAR, 
            errorType VARCHAR, 
            feature VARCHAR, 
            modelName VARCHAR, 
            epc DOUBLE
        )
    """)

    # 3. Estrazione liste di lavoro (Scenario unico: Run + Modello + Errore + Feature)
    # Escludiamo le righe di Baseline (errorType NULL) per iterare solo sugli "stress test"
    scenarios = con.execute("""
        SELECT DISTINCT experiment_run, modelName, errorType, feature 
        FROM experiments 
        WHERE errorType != 'NULL' AND feature != 'NULL' AND modelName != 'NULL'
    """).fetch_df()

    print(f"Inizio calcolo EPC per {len(scenarios)} scenari...")

    #performance_metric = 'F1' # Puoi parametrizzarla se necessario

    for _, row in scenarios.iterrows():
        run = int(row['experiment_run'])
        model = row['modelName']
        etype = row['errorType']
        feat = row['feature']

        # A. Recupero Baseline (percentage 0) per questa specifica RUN e MODELLO
        # Nota: La baseline non ha feature associata (è NULL nell'esperimento originale)
        baseline_query = """
            SELECT percentage, {} as perf 
            FROM experiments 
            WHERE experiment_run = ? AND modelName = ? AND errorType = 'NULL'
        """.format(performance_metric)
        
        baseline_df = con.execute(baseline_query, [run, model]).fetch_df()

        # B. Recupero Dati di degrado (20, 40, 60, 80) per lo scenario corrente
        test_query = """
            SELECT percentage, {} as perf 
            FROM experiments 
            WHERE experiment_run = ? AND modelName = ? AND errorType = ? AND feature = ?
        """.format(performance_metric)
        
        test_df = con.execute(test_query, [run, model, etype, feat]).fetch_df()

        if baseline_df.empty or test_df.empty:
            continue

        # C. Unione e ordinamento per creare la curva
        full_curve = pd.concat([baseline_df, test_df]).sort_values('percentage')
        
        # Pulizia duplicati (nel caso ci fossero più righe per la stessa percentuale)
        full_curve = full_curve.groupby('percentage')['perf'].mean().reset_index()

        # D. Calcolo EPC (Correlazione di Pearson inversa)
        x = full_curve['percentage'].values
        y = full_curve['perf'].values
        
        # Gestione varianza zero (se y è costante, corrcoef restituisce NaN)
        if np.var(y) < 1e-9:
            epc_val = 0.0
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                corr = np.corrcoef(x, y)[0, 1]
                epc_val = -1 * corr if not np.isnan(corr) else 0.0
        # E. Inserimento nel Database
        con.execute("""
            INSERT INTO epc VALUES (?, ?, ?, ?, ?, ?)
        """, [run, dataset_name, etype, feat, model, round(float(epc_val), 4)])
    # Exporting and sorting the results
    epc_df = con.sql('select * from epc').fetchdf()
    epc_df.to_csv(f'./epc_at/epc_{dataset_name}.csv', index=False) 
    # Filtra e restituisci i risultati dalla tabella corretta
    #epc_df = con.sql('''
    #    select experiment_run,datasetName, errorType, feature, modelName, percentage, epc_at 
    #    from epc_at e
    #    where e.percentage = (select max(percentage) from epc_at where experiment_run=e.experiment_run and datasetName=e.datasetName and errorType=e.errorType and feature=e.feature and modelName=e.modelName)
    #    order by epc_at asc
    #    ''').to_df()

# Salva il risultato ordinato nel file CSV
    epc_df.to_csv(f'./epc_at/epc_at_sorted_{dataset_name}.csv', index=False)

    return epc_df


  

def visualize_feature(df,dataset_name,file_name):
 con = duckdb.connect(dataset_name+".db")
 con.sql('drop table if exists experiments')
 con.sql('CREATE TABLE experiments AS SELECT * FROM read_csv_auto(\"'+file_name+'\")')
 con.sql('drop table if exists epc')
 con.sql('CREATE TABLE epc AS SELECT * FROM read_csv_auto(\"./epc_at/epc_'+dataset_name+'\")')
 con.sql('drop table if exists epc_at')
 con.sql('CREATE TABLE epc_at AS SELECT * FROM read_csv_auto(\"./epc_at/epc_at_'+dataset_name+'\")') 
 feature=""
 errorType=""
 model=""
 for item,row in df.iterrows():
  feature=row["feature"]
  errorType=row["errorType"]
  model=row["modelName"]
  baseline_df=con.sql('select percentage,accuracy,auc,precision,recall, f1  from experiments where datasetName=\''+dataset_name+'\'  and errorType=\'NULL\' and modelName=\''+model+'\'' ).to_df()
  base_df=con.sql('select percentage,accuracy,auc,precision,recall, f1 from experiments where datasetName=\''+dataset_name+'\' and errorType=\''+errorType+'\' and modelName=\''+model+'\' and feature=\''+feature+'\'' ).to_df()
  base_df= pd.concat([baseline_df, base_df], ignore_index=True)
  plt.plot(base_df['percentage'], base_df['Accuracy'], label='Accuracy', color='blue')
  #plt.plot(base_df['percentage'], base_df['Precision'], label='Precision', color='red')
  #plt.plot(base_df['percentage'], base_df['Recall'], label='Recall', color='green')
  #plt.plot(base_df['percentage'], base_df['F1'], label='F1', color='brown')
  #plt.plot(base_df['percentage'], base_df['Auc'], label='Auc', color='orange')
  plt.xlabel('Percentage')
  plt.ylabel('Values')
  plt.title('Trend of performance for feature '+feature+' related to model '+model+' for error '+errorType)
  #plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
  #plt.savefig(directory+'/'+model+'_'+errorType+'_'+feature+'_'+dataset_name+'.jpg')
  plt.show()




def visualize_epc(df,dataset_name,file_name):
 con = duckdb.connect(dataset_name+".db")
 con.sql('drop table if exists experiments')
 con.sql('CREATE TABLE experiments AS SELECT * FROM read_csv_auto(\"'+file_name+'\")')
 for item,row in df.iterrows():
  feature=row["feature"]
  errorType=row["errorType"]
  model=row["modelName"]
  base_df=duckdb.sql('select percentage,epc_at as epc from epc_at where datasetName=\''+dataset_name+'\' and errorType=\''+errorType+'\' and modelName=\''+model+'\' and feature=\''+feature+'\'' ).to_df()
  plt.plot(base_df['percentage'], base_df['epc'], label='Accuracy', color='blue')
  plt.xlabel('Percentage')
  plt.ylabel('EPC')
  plt.title('Trend of EPC for feature '+feature+' related to model '+model+' for error '+errorType)
  plt.show()
