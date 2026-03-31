from dataclasses import dataclass, field
import pandas as pd
import duckdb

@dataclass
class RunStrategy:
    run: int
    con: duckdb.DuckDBPyConnection
    dataset_name: str
    strategy: dict
    train_df: pd.DataFrame
    noisy_df: pd.DataFrame
    test_df: pd.DataFrame
    target_variable: str
    models: list[str] = field(default_factory=list) 
    EType: str = "NULL"
    feature: str = "NULL" 
    percentage: int = 0     
    task: str = "classification"
    ground_truth_labels: dict = field(default_factory=dict)  
    n_clusters: int = None 
  