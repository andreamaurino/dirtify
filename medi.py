import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from pucktrick.labels import *
from pucktrick.missing import *
from pucktrick.outliers import *
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.impute import SimpleImputer

def create_flip_distribution(num_classes, keep_prob=0.5):
    
    flip_prob_adjacent = (1 - keep_prob) / 2  # (1 - 0.5) / 2 = 0.25
    
    distribution = {}
    for src_cls in range(num_classes):
        flip_targets = {str(src_cls): round(keep_prob, 4)}
        
        next_cls = (src_cls + 1) % num_classes
        flip_targets[str(next_cls)] = round(flip_prob_adjacent, 4)
        
        prev_cls = (src_cls - 1) % num_classes
        flip_targets[str(prev_cls)] = round(flip_prob_adjacent, 4)
        
        distribution[str(src_cls)] = flip_targets
    
    return distribution

def create_model(model_name, random_state):
    if model_name == "RF":
        return RandomForestClassifier(n_estimators=100, random_state=random_state)
    elif model_name == "LinearSVM":
        return LinearSVC(random_state=random_state, dual="auto", max_iter=10000)
    else:
        return KNeighborsClassifier(n_neighbors=5)

# Main code


DATASETS = ["winequality-red.csv","pendigit-multi.csv"]
MODELS = {
    "RF": RandomForestClassifier(n_estimators=100, random_state=42),
    "LinearSVM": LinearSVC(random_state=42, max_iter=5000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
}

ERROR_TYPES = ["NNAR","NCAR","NAR"]
INTENSITIES = [0.10, 0.20, 0.30, 0.5]
FEATURE_COMBINATIONS = [
    ["label"],
    ["feature"],
]
results = []

for dataset_name in DATASETS:
    df = pd.read_csv(f"datasetroot/{dataset_name}")

    if dataset_name=="winequality-red.csv":
        X, y = df.drop("quality", axis=1), df["quality"]
        target="quality"
        num_classes=6
        features_for_similarity = X.columns.tolist()  
    else:
        X, y = df.drop("class", axis=1), df["class"]
        target="class"
        num_classes=10
        features_for_similarity = X.columns.tolist()  
    df_temp = X.copy()
    df_temp[target] = y
    
    correlations = df_temp.corr()[target].drop(target).abs()  # Absolute correlation with the target
    feature = correlations.idxmax()  # most correlated feature
    top_correlation = correlations.max()
    
    for trial in range(20):  # 20 runs for each dataset
        print("************************************************************")
        print("running trial", trial, "for dataset", dataset_name)
        print("************************************************************")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=trial, stratify=y
        )
        X_train_temp = X_train.copy()
        X_train_temp[target] = y_train  

        for error_type in ERROR_TYPES:
                 for intensity in INTENSITIES:
                    # corrupt ony the training set
                    if error_type == "NCAR":
                        strategy = {
                            "affected_features": [target],
                            "selection_criteria": "all",
                            "percentage": intensity,
                            "mode": "new",
                            "perturbate_data": {
                                "sampling": "random",
                                "distribution": "random",
                                "noise_model": "NCAR"
                            }
                        }
                    elif error_type == "NAR":
                        flip_dist = create_flip_distribution(num_classes, keep_prob=0.2)
                        strategy = {
                            "affected_features": [target],
                            "selection_criteria": "all",
                            "percentage": intensity,
                            "mode": "new",
                            "perturbate_data": {
                                "sampling": "random",
                                "distribution": "random",
                                "noise_model": "NAR",
                                "param": {
                                    "flip_distribution": flip_dist
                                }
                            }
                        }
                    elif error_type=="NNAR":
                        strategy = {
                            "affected_features": [target],
                            "selection_criteria": "all",
                            "percentage": intensity,
                            "mode": "new",
                            "perturbate_data": {
                                "sampling": "random",
                                "distribution": "random",
                                "noise_model": "NNAR",
                                "param": {
                                    "features_for_similarity": features_for_similarity
                                }
                            }
                        }
                    
                    error,X_train_corrupted=labels(X_train_temp, strategy)
                   
                    composed_strategy={ "affected_features": [feature],
                            "selection_criteria": "all",
                            "percentage": 1,
                            "mode": "composed",
                            "perturbate_data": {
                                "sampling": "random",
                                "distribution": "random",
                            }
                        }
                    error,X_train_corrupted_composed=outlier(X_train_corrupted, strategy,X_train_temp)
                                       
                    
                    y_train_corrupted = X_train_corrupted[target]
                    y_train_corrupted_composed = X_train_corrupted_composed[target]
                    X_train_corrupted = X_train_corrupted.drop(target, axis=1, errors='ignore') 
                    X_train_corrupted_composed = X_train_corrupted_composed.drop(target, axis=1, errors='ignore')

                    for model_name, model in MODELS.items():
                        model = create_model(model_name, trial*5)
                        
                        model.fit(X_train_corrupted, y_train_corrupted)
                        y_pred = model.predict(X_test)
                        
                        acc = accuracy_score(y_test, y_pred)
                        f1_macro = f1_score(y_test, y_pred, average="macro")
                        f1_weighted = f1_score(y_test, y_pred, average="weighted")

                        
                        results.append({
                            "dataset": dataset_name,
                            "trial": trial,
                            "error_type": error_type,
                            "intensity": intensity,
                            "features": "target",
                            "model": model_name,
                            "accuracy": acc,
                            "f1_macro": f1_macro,
                            "f1_weighted": f1_weighted
                        })

                        model = create_model(model_name, trial*5)
                        model.fit(X_train_corrupted_composed, y_train_corrupted_composed)
                        y_pred = model.predict(X_test)
                        
                        acc = accuracy_score(y_test, y_pred)
                        f1_macro = f1_score(y_test, y_pred, average="macro")
                        f1_weighted = f1_score(y_test, y_pred, average="weighted")

                        
                        results.append({
                            "dataset": dataset_name,
                            "trial": trial,
                            "error_type": error_type,
                            "intensity": intensity,
                            "features": feature,
                            "model": model_name,
                            "accuracy": acc,
                            "f1_macro": f1_macro,
                            "f1_weighted": f1_weighted
                        })

results_df = pd.DataFrame(results)
results_df.to_csv("results.csv", index=False)