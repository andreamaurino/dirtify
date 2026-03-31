from sklearn.datasets import fetch_openml
import pandas as pd

#adult = fetch_openml("adult", version=2, as_frame=True)
#X = adult.data
#y = adult.target

#print(X.shape, y.shape)
#print(X.head())
#X["income"] = y
#X.to_csv("adult_clean.csv", index=False)

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
optical_recognition_of_handwritten_digits = fetch_ucirepo(id=80) 
  
# data (as pandas dataframes) 
X = optical_recognition_of_handwritten_digits.data.features 
y = optical_recognition_of_handwritten_digits.data.targets 
df = pd.concat([X, y], axis=1)
# metadata 
print(optical_recognition_of_handwritten_digits.metadata) 
  
# variable information 
print(optical_recognition_of_handwritten_digits.variables) 
df.to_csv("optdigit.csv", index=False)