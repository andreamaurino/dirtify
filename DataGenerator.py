from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np

#adult = fetch_openml("adult", version=2, as_frame=True)
#X = adult.data
#y = adult.target

#print(X.shape, y.shape)
#print(X.head())
#X["income"] = y
#X.to_csv("adult_clean.csv", index=False)

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
#optical_recognition_of_handwritten_digits = fetch_ucirepo(id=80) 
  
# data (as pandas dataframes) 
#X = optical_recognition_of_handwritten_digits.data.features 
#y = optical_recognition_of_handwritten_digits.data.targets 
#df = pd.concat([X, y], axis=1)
# metadata 
#print(optical_recognition_of_handwritten_digits.metadata) 
  
# variable information 
#print(optical_recognition_of_handwritten_digits.variables) 
#df.to_csv("optdigit.csv", index=False)
df=pd.read_csv("./datasetRoot/satimage_sample3600.csv")
# controlla la distribuzione delle classi
print(df['class'].value_counts())
print(df.shape)
#print(df[feature_cols].describe())
#print(df[feature_cols].std())
# se std è vicino a 0 per alcune feature → problema
df_exp = pd.read_csv('experiments/experiments_satimage_sample3600.csv')
# quanto cambia il dataset sporcando una feature al 8
# nel CSV risultante, ora devono essere diversi
feat1_ami = df_exp[
    (df_exp['feature'] == '[A13attr]') & 
    (df_exp['modelName'] == 'K-Means') &
    (df_exp['percentage'] == 0.2)
]['AMI'].values

feat2_ami = df_exp[
    (df_exp['feature'] == '[F30attr]') & 
    (df_exp['modelName'] == 'K-Means') &
    (df_exp['percentage'] == 0.2)
]['AMI'].values

print(feat1_ami)
print(feat2_ami)
print("Identici?", np.allclose(feat1_ami, feat2_ami))