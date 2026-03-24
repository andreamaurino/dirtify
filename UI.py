import pandas as pd
import duckdb
from analyzer import *
import json
import os
from scipy.stats import chi2_contingency, pointbiserialr,spearmanr, pearsonr
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from strategy import RunStrategy
from pandas.api.types import (
    is_string_dtype, is_object_dtype, is_categorical_dtype,
    is_integer_dtype, is_float_dtype, is_bool_dtype,
    is_datetime64_any_dtype, is_numeric_dtype
)
import pandas as pd
import csv

def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2cor = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    return np.sqrt(phi2cor / min((kcorr - 1), (rcorr - 1)))

def calculate_correlations(data, target_variable, continuous_variables, discrete_variables, categorical_string_variables, categorical_int_variables, binary_variables):
    correlations = {}
    processed_features = set()

    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    def add_correlation(var, correlation_type, corr_value):
        if not np.isnan(corr_value):  
            correlations[var] = {"type": correlation_type, "correlation": corr_value}
        else:
            print(f"correlation valued not considere for {var}. Ignored.")

    # Continua 
    for var in continuous_variables:
        if var in data.columns and var not in processed_features:
            try:
                corr, _ = pointbiserialr(data[var], data[target_variable])
                add_correlation(var, "Point-biserial", corr)
                processed_features.add(var)
            except Exception as e:
                print(f"Error in calculating the Point-biserial for {var}: {e}")

    # Discrete
    for var in discrete_variables:
        if var in data.columns and var not in processed_features:
            try:
                corr, _ = pointbiserialr(data[var], data[target_variable])
                add_correlation(var, "Point-biserial", corr)
                processed_features.add(var)
            except Exception as e:
                print(f"Error in calculus of the Point-biserial correlation for {var}: {e}")

    # Categorical (stringhe e interi)
    for var in categorical_string_variables + categorical_int_variables:
        if var in data.columns and var not in processed_features:
            try:
                contingency_table = pd.crosstab(data[var], data[target_variable])
                cramer_v_value = cramers_v(contingency_table.values)
                add_correlation(var, "Cramér's V", cramer_v_value)
                processed_features.add(var)
            except Exception as e:
                print(f" Errors in Cramér's V calculation for {var}: {e}")

  
    # Binarie
    for var in binary_variables:
        if var in data.columns and var not in processed_features:
            try:
                if len(data[var].unique()) == 2:  # Verifica se è veramente binaria
                    corr = matthews_corrcoef(data[var], data[target_variable])
                    add_correlation(var, "Matthews Correlation Coefficient", corr)
                    processed_features.add(var)
                else:
                    print(f"Variabile {var} non è binaria. Valori unici: {data[var].unique()}")
            except Exception as e:
                print(f"Errore nel calcolo del Matthews Correlation Coefficient per {var}: {e}")

    # Soglie di correlazione
    point_biserial_threshold = 0.5
    cramer_v_threshold = 0.5
    matthews_threshold = 0.5

    # Lista per le correlazioni rilevanti
    relevant_correlations = []

    for var, corr_info in correlations.items():
        if corr_info["type"] == "Point-biserial" and abs(corr_info["correlation"]) >= point_biserial_threshold:
            relevant_correlations.append({"Feature": var, "Type": corr_info["type"], "Correlation": corr_info["correlation"]})
        elif corr_info["type"] == "Cramér's V" and abs(corr_info["correlation"]) >= cramer_v_threshold:
            relevant_correlations.append({"Feature": var, "Type": corr_info["type"], "Correlation": corr_info["correlation"]})
        elif corr_info["type"] == "Matthews Correlation Coefficient" and abs(corr_info["correlation"]) >= matthews_threshold:
            relevant_correlations.append({"Feature": var, "Type": corr_info["type"], "Correlation": corr_info["correlation"]})

    # Stampa le correlazioni più rilevanti
    print("Most significant correlations with the target variable:")
    for corr in relevant_correlations:
        print(f"Feature: {corr['Feature']}, Tipo: {corr['Type']}, Correlazione: {corr['Correlation']}")

    return correlations




def calculate_feature_correlations(data, continuous_variables, discrete_variables, categorical_string_variables, categorical_int_variables, binary_variables, output_csv_path):
    correlations = []

    label_encoders = {}
    for var in categorical_string_variables:
        if var in data.columns:
            le = LabelEncoder()
            data[var] = le.fit_transform(data[var].astype(str))
            label_encoders[var] = le

    all_variables = continuous_variables + discrete_variables + categorical_string_variables + categorical_int_variables + binary_variables
    variable_pairs = list(combinations(all_variables, 2))

    for var1, var2 in variable_pairs:
        if var1 not in data.columns or var2 not in data.columns:
            continue

        correlation_info = {"Feature1": var1, "Feature2": var2, "Type": None, "Correlation": None}
        
        try:
            if var1 in continuous_variables and var2 in continuous_variables:
                corr, _ = spearmanr(data[var1], data[var2])
                correlation_info.update({"Type": "Spearman", "Correlation": corr})

            elif var1 in discrete_variables and var2 in discrete_variables:
                corr, _ = spearmanr(data[var1], data[var2])
                correlation_info.update({"Type": "Spearman", "Correlation": corr})

            elif var1 in categorical_string_variables + categorical_int_variables and var2 in categorical_string_variables + categorical_int_variables:
                contingency_table = pd.crosstab(data[var1], data[var2])
                cramer_v_value = cramers_v(contingency_table.values)
                correlation_info.update({"Type": "Cramér's V", "Correlation": cramer_v_value})

            elif (var1 in continuous_variables or var1 in discrete_variables) and (var2 in categorical_string_variables + categorical_int_variables):
                f_statistic = data.groupby(var2)[var1].mean().var()
                correlation_info.update({"Type": "ANOVA", "Correlation": f_statistic})

            elif (var2 in continuous_variables or var2 in discrete_variables) and (var1 in categorical_string_variables + categorical_int_variables):
                f_statistic = data.groupby(var1)[var2].mean().var()
                correlation_info.update({"Type": "ANOVA", "Correlation": f_statistic})

            elif (var1 in continuous_variables and var2 in discrete_variables) or (var1 in discrete_variables and var2 in continuous_variables):
                corr, _ = spearmanr(data[var1], data[var2])
                correlation_info.update({"Type": "Spearman", "Correlation": corr})

            elif var1 in binary_variables and var2 in binary_variables:
                corr, _ = pearsonr(data[var1], data[var2])
                correlation_info.update({"Type": "Phi Coefficient", "Correlation": corr})

            elif (var1 in binary_variables and var2 in continuous_variables + discrete_variables) or \
                 (var2 in binary_variables and var1 in continuous_variables + discrete_variables):
                corr, _ = spearmanr(data[var1], data[var2])
                correlation_info.update({"Type": "Spearman", "Correlation": corr})

            elif (var1 in binary_variables and var2 in categorical_string_variables + categorical_int_variables) or \
                 (var2 in binary_variables and var1 in categorical_string_variables + categorical_int_variables):
                contingency_table = pd.crosstab(data[var1], data[var2])
                cramer_v_value = cramers_v(contingency_table.values)
                correlation_info.update({"Type": "Cramér's V", "Correlation": cramer_v_value})

        except Exception as e:
            print(f"Errore calcolando correlazione tra {var1} e {var2}: {e}")

        correlations.append(correlation_info)

    df = pd.DataFrame(correlations)
    df.to_csv(output_csv_path, index=False)
    print(f"Correlations saved to {output_csv_path}")

    print("\n--- Correlation Results ---")
    for row in correlations:
        print(f"Feature1: {row['Feature1']}, Feature2: {row['Feature2']}, Type: {row['Type']}, Correlation: {row['Correlation']}")
            
def remove_datetime_columns(df):
    datetime_cols = []
    for col in df.columns:
        try:
            pd.to_datetime(df[col], format='%d-%m-%Y %H:%M', errors='raise')
            datetime_cols.append(col) 
        except (ValueError, TypeError):
            continue  

    df = df.drop(columns=datetime_cols)
    return df

def remove_non_unique_columns(df):
    unique_counts = df.nunique()
    non_unique_cols = unique_counts[unique_counts <= 1].index  
    print(f"Not unique coloumns founds: {non_unique_cols.tolist()}")  
    return df.drop(columns=non_unique_cols)  

def featureImportance(dataset_name, train_df, test_df, target_variable):
    X_train = remove_datetime_columns(train_df.drop(columns=[target_variable]))
    X_train = remove_non_unique_columns(X_train)

    X_test = remove_datetime_columns(test_df.drop(columns=[target_variable]))
    X_test = remove_non_unique_columns(X_test)

    y_train = train_df[target_variable]
    y_test = test_df[target_variable]

    categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_columns)
        ],
        remainder='passthrough' 
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    # OHE
    feature_names = preprocessor.get_feature_names_out()

    # standardizzo
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    # Logistic Regression L1 con 5-fold cross-validation
    model_lr = LogisticRegressionCV(
        penalty='l1',
        solver='saga',
        cv=5,
        max_iter=1000,
        random_state=0,
        n_jobs=-1,
        scoring='accuracy',
        refit=True
    )

    model_lr.fit(X_train_scaled, y_train)
    coefficients = model_lr.coef_.flatten()

    # filtra le prime 10 feature con valori maggiori
    importance_df_lr = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    importance_df_lr = importance_df_lr[importance_df_lr['Coefficient'].abs() > 0.01] 
    top_10_positive = importance_df_lr.sort_values(by='Coefficient', ascending=False).head(10)
    top_10_negative = importance_df_lr.sort_values(by='Coefficient', ascending=True).head(10)
    importance_df_lr_filtered = pd.concat([top_10_positive, top_10_negative])

    importance_df_lr_filtered.to_csv(f'relevance/l1_logistic_regression_importance.csv', index=False)
    print(f"Feature importance saved in 'relevance/l1_logistic_regression_importance.csv'")

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df_lr_filtered['Feature'], importance_df_lr_filtered['Coefficient'], color='skyblue')
    plt.xlabel('Coefficient')
    plt.title('Top 10 Feature Coefficients from L1 Logistic Regression')
    plt.axvline(0, color='gray', linestyle='--')  
    plt.savefig('relevance/l1_logistic_regression_importance_plot.png')
    plt.close()

    # Random Forest
    param_grid = {
        'max_features': [3, 5, 9, 11],
        'min_samples_leaf': [1, 5, 20]
    }
    
    rf_model = RandomForestClassifier(random_state=0, n_jobs=-1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search.fit(X_train_scaled, y_train) 
    best_rf_model = grid_search.best_estimator_
    feature_importances = best_rf_model.feature_importances_

    # filtra le prime 10 feature con valori maggiori
    importance_df_rf = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df_rf = importance_df_rf.sort_values(by='Importance', ascending=False).head(10)

    importance_df_rf.to_csv(f'relevance/random_forest_importance.csv', index=False)
    print(f"Feature importance saved in 'relevance/random_forest_importance.csv'")

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df_rf['Feature'], importance_df_rf['Importance'], color='lightgreen')
    plt.xlabel('Importance (Gini Index)')
    plt.title('Top 10 Feature Importance from Random Forest')
    plt.savefig('relevance/random_forest_importance_plot.png')
    plt.close()

    return importance_df_lr_filtered, importance_df_rf

tutte_le_correlazioni = {}

def calcola_correlazione(dataframe, target_variable, error_type, step, feature_type):
    correlations = dataframe.corr()[target_variable]
    correlations = correlations.dropna() 
    key = f"{error_type}_step{step}_feature{feature_type}"
    tutte_le_correlazioni[key] = correlations
    print(f"error correlation calculated: {error_type}, step: {step}, feature_type: {feature_type}")

def calculate_correlations_with_mutual_info(data, target_variable, discrete_variables, continuous_variables, binary_variables, categorical_string_variables, categorical_int_variables):
    # TARGET
    if data[target_variable].dtype == 'object':
        le = LabelEncoder()
        data[target_variable] = le.fit_transform(data[target_variable])

    categorical_variables = categorical_int_variables + categorical_string_variables

    # VAR CATEGORICHE
    for var in categorical_variables:
        if data[var].dtype == 'object':
            le = LabelEncoder()
            data[var] = le.fit_transform(data[var])

    # Combinazione di variabili continue e categoriali
    all_features = discrete_variables + continuous_variables + binary_variables  + categorical_string_variables + categorical_int_variables

    X = data[all_features]
    y = data[target_variable]

    # MUTUAL INFO
    try:
        mi_scores = mutual_info_classif(X, y, random_state=0)
        mi_dict = {var: score for var, score in zip(all_features, mi_scores)}

        threshold = 0
        relevant_features = {var: score for var, score in mi_dict.items() if score > threshold}

        print("Mutual information based feature relevance:")
        for feature, score in relevant_features.items():
            print(f"Feature: {feature}, Score: {score}")

        return relevant_features
    except Exception as e:
        print(f"Error in calculating mutual information : {e}")
        return {}
    



def featureImportancePermutation(dataset_name, train_df, test_df, target_variable):
    X_train = remove_datetime_columns(train_df.drop(columns=[target_variable]))
    X_train = remove_non_unique_columns(X_train)

    X_test = remove_datetime_columns(test_df.drop(columns=[target_variable]))
    X_test = remove_non_unique_columns(X_test)

    y_train = train_df[target_variable]
    y_test = test_df[target_variable]

    categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_columns)
        ],
        remainder='passthrough'  
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    # Standardizzazione
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    # Logistic Regression con L1
    model_lr = LogisticRegressionCV(
        penalty='l1',
        solver='saga',
        cv=5,
        max_iter=1000,
        random_state=0,
        n_jobs=-1,
        scoring='accuracy'
    )
    model_lr.fit(X_train_scaled, y_train)

    # Random Forest con GridSearch
    param_grid = {'max_features': [3, 5, 9, 11], 'min_samples_leaf': [1, 5, 20]}
    rf_model = RandomForestClassifier(random_state=0, n_jobs=-1)
    grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    best_rf_model = grid_search.best_estimator_

    # Permutation Importance per Logistic Regression
    perm_lr = permutation_importance(model_lr, X_test_scaled, y_test, scoring='accuracy', n_repeats=10, random_state=0, n_jobs=-1)
    importance_df_lr = pd.DataFrame({'Feature': feature_names, 'Importance': perm_lr.importances_mean})
    importance_df_lr = importance_df_lr.sort_values(by='Importance', ascending=False).head(10)

    # Permutation Importance per Random Forest
    perm_rf = permutation_importance(best_rf_model, X_test_scaled, y_test, scoring='accuracy', n_repeats=10, random_state=0, n_jobs=-1)
    importance_df_rf = pd.DataFrame({'Feature': feature_names, 'Importance': perm_rf.importances_mean})
    importance_df_rf = importance_df_rf.sort_values(by='Importance', ascending=False).head(10)

    importance_df_lr.to_csv(f'relevance/permutation_importance_lr.csv', index=False)
    importance_df_rf.to_csv(f'relevance/permutation_importance_rf.csv', index=False)
    
    print("Permutation Importance saved in 'relevance/'")

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df_lr['Feature'], importance_df_lr['Importance'], color='skyblue')
    plt.xlabel('Permutation Importance')
    plt.title('Top 10 Feature Importance - Logistic Regression')
    plt.savefig('relevance/permutation_importance_lr_plot.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df_rf['Feature'], importance_df_rf['Importance'], color='lightgreen')
    plt.xlabel('Permutation Importance')
    plt.title('Top 10 Feature Importance - Random Forest')
    plt.savefig('relevance/permutation_importance_rf_plot.png')
    plt.close()

    return importance_df_lr, importance_df_rf


def stability_analysis(model, X_train, X_test, y_train, y_test, n_repeats=10):
    perm = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=42, n_jobs=-1)

    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Mean Importance': perm.importances_mean,
        'Std Importance': perm.importances_std
    })

    importance_df = importance_df.sort_values(by='Mean Importance', ascending=False)
    importance_df['Stability'] = importance_df['Std Importance'] / importance_df['Mean Importance']

    print(importance_df)

    return importance_df

  
def nunique(s: pd.Series) -> int:
        return s.nunique(dropna=True)

def is_integer_like_float(s: pd.Series) -> bool:
        """True se tutti i valori non-NaN sono numeri e senza parte frazionaria."""
        sn = s.dropna()
        if sn.empty:
            return False
        # deve essere numerico
        if not is_numeric_dtype(sn):
            return False
        # controlla parte frazionaria ~ 0
        return np.isclose(sn % 1, 0).all()

def correlation_analysis(train_df, target_variable,data):
 discrete_variables=[] 
 categorical_string_variables=[] 
 date_variables=[] 
 continuous_variables=[] 
 categorical_int_variables=[] 
 binary_variables=[]
 max_categorical_card = 20
 for feature in train_df.columns:
        s = train_df[feature]
        nuniq = nunique(s)

        # 1) Datetime
        if is_datetime64_any_dtype(s):
            date_variables.append(feature)
            continue

        # 2) Boolean (inclusi pandas BooleanDtype)
        if is_bool_dtype(s):
            binary_variables.append(feature)
            continue

        # 3) Numeriche
        if is_integer_dtype(s):
            # Integers puri (anche Int64 pandas)
            vals = set(s.dropna().unique().tolist())
            if  vals.issubset({0, 1}):
                binary_variables.append(feature)
            elif nuniq <= max_categorical_card:
                categorical_int_variables.append(feature)
            else:
                discrete_variables.append(feature)
            continue

        if is_float_dtype(s):
            # Float che in realtà sono interi (es. NaN presenti hanno forzato a float)
            if is_integer_like_float(s):
                vals = set(pd.Series(s.dropna().astype(np.int64)).unique().tolist())
                if  vals.issubset({0, 1}):
                    binary_variables.append(feature)
                elif nuniq <= max_categorical_card:
                    categorical_int_variables.append(feature)  # interi con bassa cardinalità
                else:
                    discrete_variables.append(feature)        # interi con alta cardinalità
            else:
                continuous_variables.append(feature)
            continue

        # 4) Categorical dtype esplicito
        if is_categorical_dtype(s):
            # decidi se trattare come stringhe categoriche
            if nuniq <= max_categorical_card:
                categorical_string_variables.append(feature)
            else:
                # categoria ad alta cardinalità -> in genere la si tratta come testo
                categorical_string_variables.append(feature)
            continue

        # 5) Stringhe / object (spesso stringhe/miste)
        if is_string_dtype(s) or is_object_dtype(s):
            # se è interamente stringa (ignorando NaN), valuta cardinalità
            if nuniq <= max_categorical_card:
                categorical_string_variables.append(feature)
            else:
                # testo ad alta cardinalità -> lasciamo tra le stringhe categoriche “larghe”
                categorical_string_variables.append(feature)
            continue

        # 6) Fallback: prova a inferire
        if is_numeric_dtype(s):
            # safety net per casi rari
            if is_integer_like_float(s):
                if nuniq <= max_categorical_card:
                    categorical_int_variables.append(feature)
                else:
                    discrete_variables.append(feature)
            else:
                continuous_variables.append(feature)
        else:
            raise ValueError(f"Unsupported column type for feature '{feature}' (dtype: {s.dtype}).")

  



######################################
#
#
#calcolo della correlazione e rilevanza
#
#
######################################
 correlations = {
    "Continuous": {},
    "Discrete": {},
    "CategoricalString": {},
    "CategoricalInt": {},
    "Binary": {}
} 
 
 processed_features = set()

 correlations = []

 if continuous_variables:
    continuous_results = calculate_correlations(train_df, target_variable, continuous_variables, [], [], [], [])
    for feature, result in continuous_results.items():
        if feature not in processed_features:
            correlations.append({"Feature": feature, "Type": "Continuous", "Correlation": result["correlation"], "Type of Correlation": result["type"]})
            processed_features.add(feature)

 if discrete_variables:
    discrete_results = calculate_correlations(train_df, target_variable, [], discrete_variables, [], [], [])
    for feature, result in discrete_results.items():
        if feature not in processed_features:  
            correlations.append({"Feature": feature, "Type": "Discrete", "Correlation": result["correlation"], "Type of Correlation": result["type"]})
            processed_features.add(feature)

 if categorical_string_variables:
    categorical_string_results = calculate_correlations(train_df, target_variable, [], [], categorical_string_variables, [], [])
    for feature, result in categorical_string_results.items():
        if feature not in processed_features: 
            correlations.append({"Feature": feature, "Type": "CategoricalString", "Correlation": result["correlation"], "Type of Correlation": result["type"]})
            processed_features.add(feature)

 if categorical_int_variables:
    categorical_int_results = calculate_correlations(train_df, target_variable, [], [], [], categorical_int_variables, [])
    for feature, result in categorical_int_results.items():
        if feature not in processed_features: 
            correlations.append({"Feature": feature, "Type": "CategoricalInt", "Correlation": result["correlation"], "Type of Correlation": result["type"]})
            processed_features.add(feature)

 if binary_variables:
    binary_results = calculate_correlations(train_df, target_variable, [], [], [], [], binary_variables)
    for feature, result in binary_results.items():
        if feature not in processed_features:  
            correlations.append({"Feature": feature, "Type": "Binary", "Correlation": result["correlation"], "Type of Correlation": result["type"]})
            processed_features.add(feature)

 correlation_df = pd.DataFrame(correlations)

 results_dir = 'relevance'
 os.makedirs(results_dir, exist_ok=True)
 correlation_df.to_csv(os.path.join(results_dir, 'correlation_results.csv'), index=False)
 print("Risultati salvati in 'correlation_results.csv'") 


 plt.figure(figsize=(10, 6))
 sns.barplot(x='Feature', y='Correlation', hue='Type of Correlation', data=correlation_df)
 plt.title('Grouped Bar Chart of Feature Correlations with Target')
 plt.xticks(rotation=45, ha='right')
 plt.tight_layout()
 bar_chart_path = os.path.join(results_dir, 'correlation_grouped_bar_chart.png')
 plt.savefig(bar_chart_path)
 print(f"Grouped Bar Chart  saved in {bar_chart_path}")


 output_csv_path = "feature_correlations.csv"

# CORRELAZIONI TRA FEATURE
 correlations = calculate_feature_correlations(
        data=data,
        continuous_variables=continuous_variables,
        discrete_variables=discrete_variables,
        categorical_string_variables=categorical_string_variables,
        categorical_int_variables=categorical_int_variables,
        binary_variables = binary_variables,
        output_csv_path=output_csv_path
    )
 


 mi_scores = calculate_correlations_with_mutual_info(data, target_variable, discrete_variables, continuous_variables, binary_variables, categorical_string_variables, categorical_int_variables)

 # IMPORTANCE
 #featureImportance(dataset_name, train_df, test_df, target_variable)


def start(json_name,directory=""):
 print("start experiments")
 n_runs = 19 
 ## read json 
 filepath=""
 if directory=="":
    file_path = json_name
 else:
    file_path = os.path.join(directory, json_name
                             )
 with open(file_path, 'r') as file:
    Jsondata = json.load(file)

 dataset_name = os.path.join('datasetRoot', Jsondata['datasetName'])
 print("Dataset name:", Jsondata['datasetName'])
 con = duckdb.connect(dataset_name+".db")
 con.sql('drop table if exists experiments')
 con.sql('CREATE TABLE experiments (experiment_run INTEGER, datasetName VARCHAR, errorType VARCHAR, percentage DOUBLE, feature VARCHAR, modelName VARCHAR, Accuracy DOUBLE, Auc DOUBLE, Recall DOUBLE, Precision DOUBLE, F1 DOUBLE,AMI DOUBLE NULL, SILHOUETTE DOUBLE NULL, K DOUBLE NULL)')

 data = pd.read_csv(dataset_name, sep=',') #encoding per anonymized loan
 target_variable = Jsondata.get('targetVariable')
 data = data.dropna()
 if not(target_variable is None or  target_variable == ''):
     data = data.dropna(subset=[target_variable])
 
 dataset_name = Jsondata['datasetName']
 
 train_df=data
 train_df.index = range(0, len(train_df))
 try:
    test_name=Jsondata['testset']
 except KeyError:
     print('no testset detected')
     test_name=""

#start experiments
 for run in range(n_runs):
    print("***********************************")
    print("*********** RUN:"+str(run))
    print("***********************************")

    if test_name == "":
            train_df, test_df = train_test_split(data, test_size=0.2)

    else:
            test_name_path = os.path.join('datasetRoot', test_name)  
            test_df = pd.read_csv(test_name_path, sep=',')
            print('Test set detected')
    train_df.index = range(0, len(train_df)) 
    short_models=Jsondata['machineLearningModels']
    included_models=short_models
    task=Jsondata['task']
    data = data.dropna()
    if not( target_variable is  None or target_variable == ''):
         data = data.dropna(subset=[target_variable])

    #correlation_analysis(train_df, target_variable,data)


    ##########################
    #
    #
    #start experimental anaysis
    #
    #
    ###########################
    stg=RunStrategy(
        run=run,
        con=con,
        task=task,
        dataset_name=dataset_name,
        target_variable=target_variable,
        train_df=train_df,
        test_df=test_df,
        n_clusters=Jsondata.get('Clusters'),
        strategy=None,        
    )

    
    for document in Jsondata['Experiments']:
        stg.EType = document["Errortype"]
        if document['Errortype']=="standard":
            try:
                stg.models=document['machineLearningModels']
            except KeyError:
                stg.models=included_models
                performanceAnalysis(stg) #run,con,dataset_name,train_df,test_df, target_variable,models)
        else:
            stg.strategy=document["Strategy"]
            try: 
                stg.models=document['machineLearningModels']
            except KeyError:
                stg.models=included_models
            AnalyzeValues(stg)
        experiments_dir = 'experiments'
        os.makedirs(experiments_dir, exist_ok=True)

        experiments_df = con.sql('SELECT * FROM experiments').to_df()
        experiments_df.insert(0, ',', range(len(experiments_df)))
        experiments_filename = f'experiments_{os.path.basename(dataset_name)}'  
        experiments_df.to_csv(os.path.join(experiments_dir, experiments_filename), index=False)
        print(f" file {experiments_filename} saved in the 'experiments' folder.")

 experiments_dir = 'experiments'
 os.makedirs(experiments_dir, exist_ok=True)
 experiments_df = con.sql('SELECT * FROM experiments').to_df()
 experiments_df.insert(0, ',', range(len(experiments_df)))
 dataset_file_name = os.path.basename(dataset_name)  
 experiments_filename = f'experiments_{dataset_file_name}' 
 experiments_df.to_csv(os.path.join(experiments_dir, experiments_filename), index=False)
 print(f"That's all folks! file {experiments_filename} saved in the 'experiments' folder.")



