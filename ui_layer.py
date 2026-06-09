"""
UI layer per il Data Analysis Tool.

Questo modulo rappresenta il livello di interazione con l'utente.
Modella il flusso di utilizzo dell'applicazione senza implementare
una vera interfaccia grafica.
"""

from typing import Optional


class DataAnalysisUI:
    def __init__(self):
        # Stato corrente dell'interazione utente
        self.dataset_path: Optional[str] = None
        self.testset_path: Optional[str] = None
        self.target_variable: Optional[str] = None
        self.analysis_strategy: Optional[str] = None


    def open_dataset(self, path: str):
        """Selezione del dataset da parte dell'utente"""
        self.dataset_path = path
        print(f"Dataset selezionato: {path}")


    def open_testset(self, path: str):
        """Selezione opzionale del test set"""
        self.testset_path = path
        print(f"Test set selezionato: {path}")


    def select_target_variable(self, target: str):
        """Scelta della variabile target"""
        self.target_variable = target
        print(f"Variabile target selezionata: {target}")


    def select_analysis_strategy(self, strategy: str):
        """Scelta della strategia di analisi"""
        self.analysis_strategy = strategy
        print(f"Strategia di analisi selezionata: {strategy}")

        

    def run_analysis(self, config_file: str):
        """
        Avvio dell'analisi.
        Questo metodo rappresenta il punto di integrazione
        con il backend di analisi.
        """
        print("Avvio analisi con la seguente configurazione:")
        print(f"- Dataset: {self.dataset_path}")
        print(f"- Testset: {self.testset_path}")
        print(f"- Target: {self.target_variable}")
        print(f"- Strategia: {self.analysis_strategy}")

        # Punto di integrazione con il backend
        # start(config_file)

    def show_results(self):
        """Visualizzazione dei risultati (tabelle e grafici)"""
        print("Visualizzazione dei risultati dell'analisi")

    def save_project(self):
        """Salvataggio della configurazione e dei risultati"""
        print("Progetto salvato correttamente")


if __name__ == "__main__":
    """
    Esempio di utilizzo del livello UI.
    Serve solo a mostrare il flusso di interazione dell'utente.
    """

    ui = DataAnalysisUI()
    ui.open_dataset("datasetRoot/example.csv")
    ui.select_target_variable("target")
    ui.select_analysis_strategy("custom_rules")
    ui.run_analysis("json/example_config.json")