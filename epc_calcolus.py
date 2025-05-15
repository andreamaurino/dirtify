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

'''
# trovare il modo di caricare la  il db experiments
def start(dataset_name,filename):

  filename = f"experiments_{dataset_name}"
  experiment_path = os.path.join('experiments', filename)
  
  duckdb.sql('drop table if exists experiments')
  duckdb.sql('CREATE TABLE experiments AS SELECT * FROM read_csv_auto(\"'+experiment_path+'\")')
  duckdb.sql('drop table if exists epc')
  duckdb.sql('drop table if exists epc_at')
  duckdb.sql('CREATE TABLE epc (datasetName VARCHAR, errorType VARCHAR, feature VARCHAR, modelName VARCHAR, epc DOUBLE)')
  duckdb.sql('CREATE TABLE epc_at (datasetName VARCHAR, errorType VARCHAR, feature VARCHAR, modelName VARCHAR, percentage DOUBLE, epc_at DOUBLE)')
  #epc calculus
  ##feature_df=pd.DataFrame({'feature':['Warehouse_block']})
  epsilon=0.001
  feature_df=duckdb.sql('select distinct feature from experiments where datasetName=\''+dataset_name+'\' and feature !=\'NULL\'').to_df()

  ##feature_df=pd.DataFrame({'feature':['Warehouse_block','Mode_of_Shipment', 'Product_importance','Gender']})
  feature=""
  errorType=""
  model=""
  performance=""
  final_df=pd.DataFrame(columns=['errorType', 'feature', 'model', 'performance', 'EPC'])
  performance_df=pd.DataFrame({'performance':['accuracy']}) ## ,'auc','precision','recall','f1'
  for value3 in performance_df['performance']:
    performance=value3
  for value in feature_df['feature']:
   feature=value
   errorType_df=duckdb.sql('select distinct errorType from experiments where datasetName=\''+dataset_name+'\' and feature=\''+feature+'\' and errorType !=\'NULL\'').to_df()
   model_df=duckdb.sql('select distinct modelName from experiments where datasetName=\''+dataset_name+'\' and feature=\''+feature+'\' and modelName !=\'NULL\'').to_df()
   for value2 in model_df['modelName']:
    model=value2
    for value1 in errorType_df['errorType']:
      errorType=value1
      baseline_df=duckdb.sql('select percentage,'+ performance+' as performance  from experiments where datasetName=\''+dataset_name+'\'  and errorType=\'NULL\' and modelName=\''+model+'\'' ).to_df()
      base_df=duckdb.sql('select percentage, '+ performance+' as performance from experiments where datasetName=\''+dataset_name+'\' and errorType=\''+errorType+'\' and modelName=\''+model+'\' and feature=\''+feature+'\'' ).to_df()
      base_df= pd.concat([baseline_df, base_df], ignore_index=True)
      epc_df = base_df.copy()
      epc_df['percentage'] = base_df['percentage' ]
      #epc_df.loc[0, 'performance'] = 1
      epc_txt=  None
      for i in range(1, len(base_df)):
            epc_df.loc[i, 'performance'] = (base_df.loc[i, 'performance'] )#/ base_df.loc[i-1, 'performance'])
            x=epc_df['percentage'].iloc[:i+1].to_numpy()
            y=epc_df['performance'].iloc[:i+1].to_numpy()
            #epc=epc_df['percentage'].corr(epc_df['performance'], method='pearson')
            epc = -1* np.corrcoef(x,y)[0, 1]
            percentage=epc_df.loc[i,'percentage']
            if  np.isnan(epc):
              epc_txt= "0"
            else:
              var_performance= epc_df['performance'].var()
              if var_performance<=epsilon:
                epc_txt= "0"
              else:
                epc_txt=str(round(epc,4))
            duckdb.sql('INSERT INTO epc_at values (\''+dataset_name +'\',\''+errorType+'\',\''+feature+'\',\''+model+'\','+str(percentage)+','+epc_txt+')')
      #epc_final= epc_df['percentage'].corr(epc_df['performance'], method='pearson')
      epc_final = -1*np.corrcoef(x,y)[0, 1]
      if  np.isnan(epc_final):
              epc_txt= "0"
      else:
              var_performance= epc_df['performance'].var()
              if var_performance<=epsilon:
                epc_txt= "0"
              else:
                epc_txt=str(str(round(epc_final,4)))

      duckdb.sql('INSERT INTO epc values (\''+dataset_name +'\',\''+errorType+'\',\''+feature+'\',\''+model+'\','+epc_txt+')')

  epc_df=duckdb.sql('select * from epc').to_df()
  epc_df.to_csv('epc_'+dataset_name, index=False)
  epc_at_df=duckdb.sql('select * from epc_at').to_df()
  epc_at_df.to_csv('epc_at_'+dataset_name, index=False)
  epc_df=duckdb.sql('select * from epc where epc>=0.7 order by epc ').to_df()
  return epc_df

'''
def start(dataset_name, filename):
    filename = f"experiments_{dataset_name}"
    experiment_path = os.path.join('experiments', filename)
    con = duckdb.connect(dataset_name+".db")
    con.sql('drop table if exists experiments')
    con.sql('CREATE TABLE experiments AS SELECT * FROM read_csv_auto("'+experiment_path+'")')
    con.sql('drop table if exists epc')
    con.sql('drop table if exists epc_at')
    con.sql('CREATE TABLE epc (datasetName VARCHAR, errorType VARCHAR, feature VARCHAR, modelName VARCHAR, epc DOUBLE)')
    con.sql('CREATE TABLE epc_at (datasetName VARCHAR, errorType VARCHAR, feature VARCHAR, modelName VARCHAR, percentage DOUBLE, epc_at DOUBLE)')

    epsilon = 0.001
    feature_df = con.sql('select distinct feature from experiments where datasetName=\''+dataset_name+'\' and feature !=\'NULL\'').to_df()
    performance_df = pd.DataFrame({'performance': ['Accuracy']})  # Add other metrics if needed

    for performance in performance_df['performance']:
        for feature in feature_df['feature']:
            errorType_df = con.sql('select distinct errorType from experiments where datasetName=\''+dataset_name+'\' and feature=\''+feature+'\' and errorType !=\'NULL\'').to_df()
            model_df = con.sql('select distinct modelName from experiments where datasetName=\''+dataset_name+'\' and feature=\''+feature+'\' and modelName !=\'NULL\'').to_df()

            for model in model_df['modelName']:
                for errorType in errorType_df['errorType']:
                    baseline_df = con.sql('select percentage, '+ performance +' as performance from experiments where datasetName=\''+dataset_name+'\' and errorType=\'NULL\' and modelName=\''+model+'\'').to_df()
                    base_df = con.sql('select percentage, '+ performance +' as performance from experiments where datasetName=\''+dataset_name+'\' and errorType=\''+errorType+'\' and modelName=\''+model+'\' and feature=\''+feature+'\'').to_df()
                    base_df = pd.concat([baseline_df, base_df], ignore_index=True)

                    epc_df = base_df.copy()
                    epc_df['percentage'] = base_df['percentage']
                    
                    for i in range(1, len(base_df)):
                        epc_df.loc[i, 'performance'] = base_df.loc[i, 'performance']
                        x = epc_df['percentage'].iloc[:i+1].to_numpy()
                        y = epc_df['performance'].iloc[:i+1].to_numpy()

                        epc = -1 * np.corrcoef(x, y)[0, 1]
                        percentage = epc_df.loc[i, 'percentage']
                        epc_txt = "0" if np.isnan(epc) or epc_df['performance'].var() <= epsilon else str(round(epc, 4))

                        con.sql('INSERT INTO epc_at values (\''+dataset_name+'\',\''+errorType+'\',\''+feature+'\',\''+model+'\','+str(percentage)+','+epc_txt+')')

                    epc_final = -1 * np.corrcoef(epc_df['percentage'].to_numpy(), epc_df['performance'].to_numpy())[0, 1]
                    epc_txt = "0" if np.isnan(epc_final) or epc_df['performance'].var() <= epsilon else str(round(epc_final, 4))

                    con.sql('INSERT INTO epc values (\''+dataset_name+'\',\''+errorType+'\',\''+feature+'\',\''+model+'\','+epc_txt+')')

    # Exporting and sorting the results
    epc_at_df = con.sql('select * from epc_at order by percentage asc').to_df()
    epc_at_df.to_csv(f'epc_at_{dataset_name}.csv', index=False)

    epc_df = con.sql('select * from epc').to_df()
    epc_df.to_csv(f'epc_{dataset_name}.csv', index=False)

    # Filtra e restituisci i risultati dalla tabella corretta
    epc_df = con.sql('''
        select datasetName, errorType, feature, modelName, percentage, epc_at 
        from epc_at 
        where epc_at >= 0.7 and epc_at != 1.0
        order by epc_at asc
        ''').to_df()

# Salva il risultato ordinato nel file CSV
    epc_df.to_csv(f'epc_at_sorted_{dataset_name}.csv', index=False)

    return epc_df


  
"""*Visualize chart for experiments*"""
def visualize_feature(df,dataset_name,file_name):
 con = duckdb.connect(dataset_name+".db")
 con.sql('drop table if exists experiments')
 con.sql('CREATE TABLE experiments AS SELECT * FROM read_csv_auto(\"'+file_name+'\")')
 con.sql('drop table if exists epc')
 con.sql('CREATE TABLE epc AS SELECT * FROM read_csv_auto(\"epc_'+dataset_name+'\")')
 con.sql('drop table if exists epc_at')
 con.sql('CREATE TABLE epc_at AS SELECT * FROM read_csv_auto(\"epc_at_'+dataset_name+'\")') 
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

"""*Visualize chart for epc*"""


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
#      table_df=duckdb.sql('select * from epc').to_df()
#      immagini = [str(file) for file in Path(directory).glob("*.jpg")]
# with open("Analysis.pdf", "wb") as f:
#    f.write(img2pdf.convert(immagini))
