"""
UI layer per il Data Analysis Tool.
Questo modulo rappresenta il livello di interazione con l'utente.
Modella il flusso di utilizzo dell'applicazione e prepara
una configurazione compatibile con il backend Dirtify.
"""

from typing import Optional
import os
import json


class DataAnalysisUI:
    def __init__(self):
        self.dataset_path: Optional[str] = None
        self.testset_path: Optional[str] = None
        self.target_variable: Optional[str] = None
        self.analysis_strategy: Optional[str] = None

    def open_dataset(self, path: str):
        self.dataset_path = path
        print(f"Dataset selezionato: {path}")

    def open_testset(self, path: str):
        self.testset_path = path
        print(f"Test set selezionato: {path}")

    def select_target_variable(self, target: str):
        self.target_variable = target
        print(f"Variabile target selezionata: {target}")

    def select_analysis_strategy(self, strategy: str):
        self.analysis_strategy = strategy
        print(f"Strategia di analisi selezionata: {strategy}")

    def generate_dirtify_config(self, output_path: str = "json/ui_generated_config.json"):
        """
        Genera un file JSON compatibile con il backend Dirtify.
        Questa è una prima integrazione tra la UI e il motore di analisi.
        """

        if not self.dataset_path:
            raise ValueError("Dataset non selezionato")

        if not self.target_variable:
            raise ValueError("Variabile target non selezionata")

        dataset_name = os.path.basename(self.dataset_path)
        testset_name = os.path.basename(self.testset_path) if self.testset_path else ""

        dirtify_config = {
            "datasetName": dataset_name,
            "testset": testset_name,
            "targetVariable": self.target_variable,
            "machineLearningModels": ["lr"],
            "isBinary": "Yes",
            "Experiments": [
                {
                    "Errortype": "standard"
                }
            ]
        }

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w") as file:
            json.dump(dirtify_config, file, indent=4)

        print(f"Configurazione Dirtify generata: {output_path}")
        return output_path

    def run_analysis(self, config_file: str = "json/ui_generated_config.json"):
        """
        Avvio dell'analisi.
        Per ora genera una configurazione compatibile con Dirtify.
        Il collegamento diretto con UI.start(config_file) verrà aggiunto dopo.
        """

        print("Preparazione analisi con la seguente configurazione:")
        print(f"- Dataset: {self.dataset_path}")
        print(f"- Testset: {self.testset_path}")
        print(f"- Target: {self.target_variable}")
        print(f"- Strategia: {self.analysis_strategy}")

        generated_config = self.generate_dirtify_config(config_file)

        print("File di configurazione pronto per il backend Dirtify:")
        print(generated_config)

        return generated_config

    def show_results(self):
        print("Visualizzazione dei risultati dell'analisi")

    def save_project(self):
        """Salvataggio della configurazione e dei risultati"""
        print("Progetto salvato correttamente")


if __name__ == "__main__":
    ui = DataAnalysisUI()
    ui.open_dataset("datasetRoot/adultdata.csv")
    ui.open_testset("datasetRoot/adultTest.csv")
    ui.select_target_variable("income2")
    ui.select_analysis_strategy("standard_analysis")
    ui.run_analysis("json/ui_generated_config.json")