from sklearn.datasets import fetch_openml
import pandas as pd

adult = fetch_openml("adult", version=2, as_frame=True)
X = adult.data
y = adult.target

print(X.shape, y.shape)
print(X.head())
X["income"] = y
X.to_csv("adult_clean.csv", index=False)
