"""
UI Layer for the Data Analysis Tool

This module represents the user interaction layer.
It defines how the user loads datasets, configures the analysis,
and triggers the backend computation.

The implementation is intentionally lightweight and focuses
on modeling the interaction flow rather than rendering a full GUI.
"""

from typing import Optional


class DataAnalysisUI:
    def __init__(self):
        self.dataset_path: Optional[str] = None
        self.testset_path: Optional[str] = None
        self.target_variable: Optional[str] = None
        self.analysis_strategy: Optional[str] = None

    def open_dataset(self, path: str):
        """User selects a dataset file"""
        self.dataset_path = path
        print(f"Dataset selected: {path}")

    def open_testset(self, path: str):
        """User selects an optional test dataset"""
        self.testset_path = path
        print(f"Test set selected: {path}")

    def select_target_variable(self, target: str):
        """User selects the target variable"""
        self.target_variable = target
        print(f"Target variable selected: {target}")

    def select_analysis_strategy(self, strategy: str):
        """User selects the analysis strategy"""
        self.analysis_strategy = strategy
        print(f"Analysis strategy selected: {strategy}")

    def run_analysis(self, config_file: str):
        """
        Triggers the backend analysis process.
        This method delegates the computation to the analysis engine.
        """
        print("Starting analysis with configuration:")
        print(f"- Dataset: {self.dataset_path}")
        print(f"- Testset: {self.testset_path}")
        print(f"- Target: {self.target_variable}")
        print(f"- Strategy: {self.analysis_strategy}")

        # Example integration point with backend
        # start(config_file)

    def show_results(self):
        """Displays analysis results (tables and plots)"""
        print("Displaying analysis results")

    def save_project(self):
        """Saves project configuration and results"""
        print("Project saved successfully")



if __name__ == "__main__":
    """
    Example usage demonstrating the interaction flow of the UI layer.

    This block is provided for illustrative purposes only.
    It shows how a user would interact with the system through the UI,
    without requiring a real graphical interface.
    """

    ui = DataAnalysisUI()
    ui.open_dataset("datasetRoot/example.csv")
    ui.select_target_variable("target")
    ui.select_analysis_strategy("custom_rules")
    ui.run_analysis("json/example_config.json")    