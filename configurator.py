import sys
import csv
import json
import os 
import pandas as pd
from PyQt5.QtWidgets import QApplication,  QTableWidget,  QTableWidgetItem, QWidget, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QLabel, QRadioButton, QButtonGroup, QScrollArea, QLabel, QLineEdit,  QHBoxLayout, QComboBox
from PyQt5.QtGui import QDoubleValidator
import UI
##########################################
#
# dopo la scelta del dataset e eventuale test set, l'utente deve selezionare i modelli ML
# in quella schermata devono apparire i pulsanti di 
# back, save, save & run
# 1) experimental one feature alone
# 2) experimental correlated features
# 3) custom experimental rule  
#
# 1) l'utente deve decidere ml, lo step gli errori a checkbox, eventuali selection_criteria default all
# 2) l'utente deve decidere ml, lo step gli errori a checkbox, eventuali selection_criteria default all e le soglie min max di correlation. 
# 3) l'utente deve inserire afected features a checkbox, selection criteria, step e modelli
# 
# in futuro il dataset modificato potrebbe essere usato per altri errori  
#
#########################################
class CSVColumnTypeSelector(QWidget):
    def __init__(self, grouped_columns={}, parent=None):
        super(CSVColumnTypeSelector, self).__init__(parent)
        self.grouped_columns = grouped_columns
        if len(grouped_columns) > 1:
            if len(grouped_columns["Experiments"]) > 1:
                grouped_columns["Experiments"] = [
                    Experiments for Experiments in grouped_columns["Experiments"]
                    if not (Experiments["Errortype"] == "duplicate")
                ]
        self.init_ui()
        self.column_types = {}  # Dizionario per memorizzare i tipi selezionati
        self.column_types2 = {}  # Dizionario per memorizzare i ML

    def init_ui(self):
        self.setWindowTitle("START ANALYSIS")
        self.setGeometry(100, 100, 1000, 1000)

        # Layout principale
        self.main_layout = QVBoxLayout()

        # Bottone per aprire il dialogo
        self.open_button = QPushButton('Select Dataset', self)
        self.open_button.clicked.connect(self.open_file_dialog)
        self.main_layout.addWidget(self.open_button)

        # Bottone per aprire il dialogo
        self.open_button1 = QPushButton('Select Testset', self)
        self.open_button1.clicked.connect(self.open_file_dialog2)
        self.main_layout.addWidget(self.open_button1)

        # Crea un QScrollArea
        self.scroll_areadf = QScrollArea(self)
        self.scroll_areadf.setWidgetResizable(True)
        self.scroll_areadf.setEnabled(False)
        self.main_layout.addWidget(self.scroll_areadf)

        # ScrollArea per ospitare le colonne e i combo box
      #  self.scroll_area = QScrollArea(self)
      #  self.scroll_area_widget = QWidget()
      #  self.scroll_area_layout = QVBoxLayout(self.scroll_area_widget)

      #  self.scroll_area.setWidgetResizable(True)
      #  self.scroll_area.setWidget(self.scroll_area_widget)
      #  self.main_layout.addWidget(self.scroll_area)

        # ScrollArea per ospitare i modelli di ML e i combo box
        self.scroll_area2 = QScrollArea(self)
        self.scroll_area_widget2 = QWidget()
        self.scroll_area_layout2 = QVBoxLayout(self.scroll_area_widget2)

        self.scroll_area2.setWidgetResizable(True)
        self.scroll_area2.setWidget(self.scroll_area_widget2)
        self.main_layout.addWidget(self.scroll_area2)
        # Label e target"
        self.target_label = QLabel("Target:", self)
        self.target_label.setStyleSheet("color: #ffffff;")  # Testo bianco
        self.target_input = QLineEdit(self)
        self.target_input.setText("")
        self.target_input.setStyleSheet("color: #ffffff; background-color: #444444;")  # Testo bianco e sfondo scuro

        # Aggiungi label e input in un layout orizzontale
        self.target_layout = QHBoxLayout()
        self.target_layout.addWidget(self.target_label)
        self.target_layout.addWidget(self.target_input)
        self.main_layout.addLayout(self.target_layout)

        # Bottone per salvare i risultati come JSON (spostato alla fine)
        self.button = QPushButton("Insert Experimental Strategy", self)
        self.button.clicked.connect(self.open_standard)
        self.button.setEnabled(False)  # Disabilitato finché non viene caricato un file
        self.main_layout.addWidget(self.button)

        self.setLayout(self.main_layout)

        self.setStyleSheet("""
            QWidget {
                background-color: #333; /* Grigio scuro */
                color: #FFF; /* Testo bianco */
                font-family: Arial, sans-serif;
            }
            QPushButton {
                background-color: #0088CC; /* Azzurro */
                color: white;
                border-radius: 5px;
                padding: 10px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #006699; /* Azzurro scuro al passaggio del mouse */
            }
            QLabel {
                margin: 10px 0;
            }
        """)

    def open_file_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_path, _ = QFileDialog.getOpenFileName(self, "Seleziona File CSV", "datasetRoot",  # Imposta 'datasetRoot' come directory di partenza
                                                    "File CSV (*.csv);;Tutti i file (*)",
                                                    options=options)

        if file_path:
            self.csv_file_name = file_path  # Usa il percorso completo 
            self.testset_name = ""
            self.display_columns(file_path)
            self.button.setEnabled(True)  
            self.display_columns2()
            self.scroll_areadf.setEnabled(True)

            df = pd.read_csv(self.csv_file_name, sep=",", encoding='iso-8859-1')  
            # Crea un QTableWidget per visualizzare il DataFrame
            table_widget = QTableWidget(self)
            table_widget.setRowCount(15)
            table_widget.setColumnCount(len(df.columns))
            table_widget.setHorizontalHeaderLabels(df.columns)

            for i in range(15):
                for j in range(len(df.columns)):
                    table_widget.setItem(i, j, QTableWidgetItem(str(df.iat[i, j])))

            self.scroll_areadf.setWidget(table_widget)

    def open_file_dialog2(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_path, _ = QFileDialog.getOpenFileName(self, "Seleziona Testset", "",
                                                    "File CSV (*.csv);;Tutti i file (*)",
                                                    options=options)

        if file_path:
            self.testset_name = os.path.basename(file_path) 
            self.display_columns(file_path)

    def display_columns(self, file_path):
        # Pulisce il layout precedente e resetta le selezioni
        #self.clear_layout(self.scroll_area_layout)
        #self.column_types.clear()

        # Legge la prima riga del file CSV
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            headers = next(reader)


    def display_columns2(self):
        # Pulisce il layout precedente e resetta le selezioni
        self.clear_layout(self.scroll_area_layout2)
        self.column_types2.clear()
        # Crea un set di radiobuttons per ogni modello di ML
        ML = ["Logistic Regression", 
              "K-Nearest Neighbors", 
              "Naive Bayes", 
              "Support Vector Machine", 
              "Decision Tree", 
              "Radial Basis Function SVM",
              "Gaussian Process Classifier",
              "Multi-Layer Perceptron",
              "Ridge Classifier",
              "Quadratic Discriminant Analysis",
              "AdaBoost",
              "Linear Discriminant Analysis",
              "Dummy Classifier",
              "Random Forest",
              "Extra Trees",
              "Gradient Boosting Classifier",
              "LightGBM",
              "CatBoost"
              ]

        for model in ML:
            self.add_column_radiobuttons2(model)

    def add_column_radiobuttons2(self, column_name2):
        # Layout orizzontale per ogni colonna
        column_layout2 = QHBoxLayout()

        # Etichetta con il nome della colonna
        label2 = QLabel(column_name2, self)
        column_layout2.addWidget(label2)

        # Gruppo di radiobuttons
        button_group2 = QButtonGroup(self)

        # Opzioni dei radiobuttons
        options = ["Yes","No"]
        for option2 in options:
            radiobutton2 = QRadioButton(option2, self)
            button_group2.addButton(radiobutton2)
            column_layout2.addWidget(radiobutton2)

            # Collega ogni radiobutton a una funzione che aggiorna il dizionario delle selezioni
            radiobutton2.toggled.connect(lambda checked, col=column_name2, opt=option2: self.update_column_type2(col, opt, checked))

        # Imposta il radiobutton "Discrete" come predefinito
        button_group2.buttons()[0].setChecked(True)

        # Memorizza la selezione predefinita
        self.column_types2[column_name2] = "Yes"

        # Aggiunge il layout al layout principale
        self.scroll_area_layout2.addLayout(column_layout2)
    
    def update_column_type2(self, column_name, option, checked):
        if checked:  # Aggiorna solo se il radiobutton è selezionato
            self.column_types2[column_name] = option

    def open_standard(self):
            # Crea un dizionario per raggruppare le colonne per tipo
            grouped_columns = {
                "datasetName": self.csv_file_name,
                "target":self.target_input.text(),
                "machineLearningModels":[],
                "Experiments":[]
            }
            if self.testset_name!="":
                grouped_columns["testset"]=self.testset_name
                # Popola il dizionario ml raggruppando le colonne per tipo
            for column2, col_type2 in self.column_types2.items():
                if col_type2 == "Yes":
                    grouped_columns["machineLearningModels"].append(column2)
            addedDocument={"Errortype":"standard"}
            grouped_columns["Experiments"].append(addedDocument)
            print(grouped_columns)
            self.close()  # Chiude la finestra principale
            self.standard_window = MissingWindow(grouped_columns)  # Apre la nuova finestra
            self.standard_window.show()

    def show_final_window(self):
       self.final_window.show()

    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget() is not None:
                    child.widget().deleteLater()
                elif child.layout() is not None:
                    self.clear_layout(child.layout())


# Classe per la finestra "one feature"
class MissingWindow(QWidget):
    def __init__(self, grouped_columns, parent=None):
        super(MissingWindow, self).__init__(parent)
        self.grouped_columns = grouped_columns
        if len(grouped_columns["Experiments"]) > 1:
            grouped_columns["Experiments"] = [
                exp for exp in grouped_columns["Experiments"]
                if exp.get("ErrorStrategy") != "one-feature"
            ]    
        self.setWindowTitle("One Features at time")
        self.setGeometry(100, 100, 1000, 1000)
        self.setStyleSheet("background-color: #2c2c2c; color: #ffffff;")

        self.column_types3 = {}
        self.column_types2 = {}

        self.layout = QVBoxLayout()
        
        # Header Label
        self.label = QLabel("Choose ML Model", self)
        self.label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.layout.addWidget(self.label)

        # ScrollArea for ML Models
        self.scroll_area3 = QScrollArea(self)
        self.scroll_area_widget3 = QWidget()
        self.scroll_area_layout3 = QVBoxLayout(self.scroll_area_widget3)
        self.display_columns3()
        self.scroll_area3.setWidgetResizable(True)
        self.scroll_area3.setWidget(self.scroll_area_widget3)
        self.layout.addWidget(self.scroll_area3)
        # Step Input
        self.step_label = QLabel("Step:", self)
        self.step_input = QLineEdit(self)
        self.step_input.setText("0.2")
        self.step_input.setStyleSheet("background-color: #3c3c3c; color: #ffffff;")
        
        self.step_layout = QHBoxLayout()
        self.step_layout.addWidget(self.step_label)
        self.step_layout.addWidget(self.step_input)
        self.layout.addLayout(self.step_layout)
 # selection Input
        self.selection_label = QLabel("Selection criteria:", self)
        self.selection_input = QLineEdit(self)
        self.selection_input.setText("all")
        self.selection_input.setStyleSheet("background-color: #3c3c3c; color: #ffffff;")
        
        self.selection_layout = QHBoxLayout()
        self.selection_layout.addWidget(self.selection_label)
        self.selection_layout.addWidget(self.selection_input)
        self.layout.addLayout(self.selection_layout)

     
        # ScrollArea for Columns
        self.scroll_area2 = QScrollArea(self)
        self.scroll_area_widget2 = QWidget()
        self.scroll_area_layout2 = QVBoxLayout(self.scroll_area_widget2)
        self.scroll_area2.setWidgetResizable(True)
        self.scroll_area2.setWidget(self.scroll_area_widget2)
        self.layout.addWidget(self.scroll_area2)
        self.display_columns2()

        # Navigation 
        button_layout = QHBoxLayout()  # Crea un layout orizzontale per i pulsanti

        self.prev_button = QPushButton("Back", self)
        self.prev_button.setStyleSheet("background-color: #0088CC; color: #ffffff;")
        self.prev_button.clicked.connect(self.open_prev_window)
        button_layout.addWidget(self.prev_button)  # Aggiungi il pulsante "Back" al layout orizzontale

        self.skip_button = QPushButton("Skip", self)
        self.skip_button.setStyleSheet("background-color: #0088CC; color: #ffffff;")
        self.skip_button.clicked.connect(self.open_skip_window)
        button_layout.addWidget(self.skip_button)  # Aggiungi il pulsante "Skip" al layout orizzontale

        self.next_button = QPushButton("Next", self)
        self.next_button.setStyleSheet("background-color: #0088CC; color: #ffffff;")
        self.next_button.clicked.connect(self.open_next_window)
        button_layout.addWidget(self.next_button)  # Aggiungi il pulsante "Next" al layout orizzontale

        self.layout.addLayout(button_layout)  # Aggiungi il layout orizzontale al layout principale

        self.setLayout(self.layout)

    def open_prev_window(self):
        self.close()
        self.next_window = CSVColumnTypeSelector(self.grouped_columns)
        self.next_window.show()

    def open_skip_window(self):
        print(self.grouped_columns)
        self.close()
        self.next_window = DuplicateWindow(self.grouped_columns)
        self.next_window.show()

    def open_next_window(self):
        newML = []
        addedDocument = {"ErrorStrategy": "one-feature"}
        addedDocument["Step"] = float(self.step_input.text())
        addedDocument["Selection_criteria"] = self.selection_input.text()
        for column3, col_type3 in self.column_types3.items():
            if col_type3 == "Yes":
                newML.append(column3)
        if newML != self.grouped_columns["machineLearningModels"]:
            addedDocument["machineLearningModels"] = newML
        newColumns = []
        for column2, col_type2 in self.column_types2.items():
            if col_type2 == "Yes":
                newColumns.append(column2)
        self.grouped_columns["Experiments"].append(addedDocument)
        print(self.grouped_columns)
        self.close()
        self.next_window = DuplicateWindow(self.grouped_columns)
        self.next_window.show()


    def display_columns2(self):
        self.clear_layout2(self.scroll_area_layout2)

    def add_column_radiobuttons2(self, column_name2):
        column_layout2 = QHBoxLayout()
        label2 = QLabel(column_name2, self)
        column_layout2.addWidget(label2)

        button_group2 = QButtonGroup(self)
        options = ["Yes", "No"]
        for option2 in options:
            radiobutton2 = QRadioButton(option2, self)
            button_group2.addButton(radiobutton2)
            column_layout2.addWidget(radiobutton2)
            radiobutton2.toggled.connect(lambda checked, col=column_name2, opt=option2: self.update_column_type2(col, opt, checked))

        button_group2.buttons()[0].setChecked(True)
        self.column_types2[column_name2] = "Yes"
        self.scroll_area_layout2.addLayout(column_layout2)

    def update_column_type2(self, column_name, option, checked):
        if checked:
            self.column_types2[column_name] = option

    def display_columns3(self):
        self.clear_layout3(self.scroll_area_layout3)
        self.column_types3.clear()
        ML = self.grouped_columns["machineLearningModels"]
        for model in ML:
            self.add_column_radiobuttons3(model)

    def add_column_radiobuttons3(self, column_name3):
        column_layout3 = QHBoxLayout()
        label3 = QLabel(column_name3, self)
        column_layout3.addWidget(label3)

        button_group3 = QButtonGroup(self)
        options = ["Yes", "No"]
        for option3 in options:
            radiobutton3 = QRadioButton(option3, self)
            button_group3.addButton(radiobutton3)
            column_layout3.addWidget(radiobutton3)
            radiobutton3.toggled.connect(lambda checked, col=column_name3, opt=option3: self.update_column_type3(col, opt, checked))

        button_group3.buttons()[0].setChecked(True)
        self.column_types3[column_name3] = "Yes"
        self.scroll_area_layout3.addLayout(column_layout3)

    def update_column_type3(self, column_name, option, checked):
        if checked:
            self.column_types3[column_name] = option

    def clear_layout3(self, layout):
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget() is not None:
                    child.widget().deleteLater()
                elif child.layout() is not None:
                    self.clear_layout(child.layout())

    def clear_layout2(self, layout):
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget() is not None:
                    child.widget().deleteLater()
                elif child.layout() is not None:
                    self.clear_layout(child.layout())

# Classe per la finestra corelate
class DuplicateWindow(QWidget):

    def __init__(self, grouped_columns, parent=None):
        super(DuplicateWindow, self).__init__(parent)
        self.grouped_columns = grouped_columns
        if len(grouped_columns["Experiments"]) > 1:
            grouped_columns["Experiments"] = [
                exp for exp in grouped_columns["Experiments"]
                if exp.get("ErrorStrategy") != "correlated-features"
            ]    
        self.setWindowTitle("Correlated Features")
        self.setGeometry(100, 100, 1000, 1000)

        # Imposta il colore di sfondo
        self.setStyleSheet("background-color: #2e2e2e; color: #ffffff;")  # Sfondo grigio scuro

        self.column_types3 = {}  # Dizionario per memorizzare i ML

        self.layout = QVBoxLayout()
        self.label = QLabel("Choose ML", self)
        self.label.setStyleSheet("color: #ffffff;")  # Imposta il testo in bianco
        self.layout.addWidget(self.label)

        # ScrollArea per ospitare i modelli di ML e i radiobuttons
        self.scroll_area3 = QScrollArea(self)
        self.scroll_area_widget3 = QWidget()
        self.scroll_area_layout3 = QVBoxLayout(self.scroll_area_widget3)
        self.display_columns3()

        self.scroll_area3.setWidgetResizable(True)
        self.scroll_area3.setWidget(self.scroll_area_widget3)
        self.layout.addWidget(self.scroll_area3)

        # Label e input per "Step"
        self.step_label = QLabel("Step:", self)
        self.step_label.setStyleSheet("color: #ffffff;")  # Testo bianco
        self.step_input = QLineEdit(self)
        self.step_input.setText("0.2")

        # Aggiungi label e input in un layout orizzontale
        self.step_layout = QHBoxLayout()
        self.step_layout.addWidget(self.step_label)
        self.step_layout.addWidget(self.step_input)
        self.layout.addLayout(self.step_layout)


        # selection Input
        self.selection_label = QLabel("Selection criteria:", self)
        self.selection_input = QLineEdit(self)
        self.selection_input.setText("all")
        self.selection_input.setStyleSheet("background-color: #3c3c3c; color: #ffffff;")

        # Aggiungi label e input in un layout orizzontale
        self.selection_layout = QHBoxLayout()
        self.selection_layout.addWidget(self.selection_label)
        self.selection_layout.addWidget(self.selection_input)
        self.layout.addLayout(self.selection_layout)

        # selection Input
        self.min_label = QLabel("Min:", self)
        self.min_input = QLineEdit(self)
        self.min_input.setText("0.6")
        self.min_input.setStyleSheet("background-color: #3c3c3c; color: #ffffff;")

        # Aggiungi label e input in un layout orizzontale
        self.min_layout = QHBoxLayout()
        self.min_layout.addWidget(self.min_label)
        self.min_layout.addWidget(self.min_input)
        self.layout.addLayout(self.min_layout)


        # selection Input
        self.max_label = QLabel("Max:", self)
        self.max_input = QLineEdit(self)
        self.max_input.setText("1.0")
        self.max_input.setStyleSheet("background-color: #3c3c3c; color: #ffffff;")
        # Aggiungi label e input in un layout orizzontale
        self.max_layout = QHBoxLayout()
        self.max_layout.addWidget(self.max_label)
        self.max_layout.addWidget(self.max_input)
        self.layout.addLayout(self.max_layout)

        
        # Layout orizzontale per i pulsanti di navigazione
        button_layout = QHBoxLayout()  # Crea un layout orizzontale per i pulsanti

        self.prev_button = QPushButton("Back", self)
        self.prev_button.setStyleSheet("background-color: #0088CC; color: #ffffff")
        self.prev_button.clicked.connect(self.open_prev_window)
        button_layout.addWidget(self.prev_button)  # Aggiungi il pulsante "Back" al layout orizzontale

        self.skip_button = QPushButton("Skip", self)
        self.skip_button.setStyleSheet("background-color: #0088CC; color: #ffffff;")
        self.skip_button.clicked.connect(self.open_skip_window)
        button_layout.addWidget(self.skip_button)  # Aggiungi il pulsante "Skip" al layout orizzontale

        self.next_button = QPushButton("Next", self)
        self.next_button.setStyleSheet("background-color: #0088CC; color: #000000;")
        experiments = self.grouped_columns.get("Experiments", [])
        if len(experiments)==2:
           self.next_button.setEnabled(False)
        else:
            self.next_button.setEnabled(True)
            self.next_button.setStyleSheet("background-color: #0088CC; color: #ffffff;")

        self.next_button.clicked.connect(self.open_next_window)
        button_layout.addWidget(self.next_button)  # Aggiungi il pulsante "Next" al layout orizzontale

        # Aggiungi il layout dei pulsanti orizzontale al layout principale
        self.layout.addLayout(button_layout)

        self.setLayout(self.layout)

    def open_prev_window(self):
        self.close()  # Chiude la finestra "Duplicate Analysis"
        self.next_window = MissingWindow(self.grouped_columns)  # Apre la nuova finestra
        self.next_window.show()

    def open_skip_window(self):
        self.close()  # Chiude la finestra "Duplicate Analysis"
        self.next_window = labelWindow(self.grouped_columns)  # Apre la nuova finestra
        self.next_window.show()

    def open_next_window(self):
        newML = []
        addedDocument = {"ErrorStrategy": "Correlated-features"}
        addedDocument["Step"] = float(self.step_input.text())
        addedDocument["Selection_criteria"] = self.selection_input.text()
        addedDocument["Min"] = float(self.min_input.text())
        addedDocument["Max"] = float(self.max_input.text())
        for column3, col_type3 in self.column_types3.items():
            if col_type3 == "Yes":
                newML.append(column3)
        if newML != self.grouped_columns["machineLearningModels"]:
            addedDocument["machineLearningModels"] = newML
        self.grouped_columns["Experiments"].append(addedDocument)
        print(self.grouped_columns)
        self.close()
        self.next_window = labelWindow(self.grouped_columns)  # Apre la nuova finestra
        self.next_window.show()

    def display_columns3(self):
        # Pulisce il layout precedente e resetta le selezioni
        self.clear_layout3(self.scroll_area_layout3)
        self.column_types3.clear()
        # Crea un set di radiobuttons per ogni colonna
        ML = self.grouped_columns["machineLearningModels"]
        for model in ML:
            self.add_column_radiobuttons3(model)

    def add_column_radiobuttons3(self, column_name3):
        # Layout orizzontale per ogni colonna
        column_layout3 = QHBoxLayout()

        # Etichetta con il nome della colonna
        label3 = QLabel(column_name3, self)
        column_layout3.addWidget(label3)

        # Gruppo di radiobuttons
        button_group3 = QButtonGroup(self)

        # Opzioni dei radiobuttons
        options = ["Yes", "No"]
        for option3 in options:
            radiobutton3 = QRadioButton(option3, self)
            button_group3.addButton(radiobutton3)
            column_layout3.addWidget(radiobutton3)

            # Collega ogni radiobutton a una funzione che aggiorna il dizionario delle selezioni
            radiobutton3.toggled.connect(lambda checked, col=column_name3, opt=option3: self.update_column_type3(col, opt, checked))

        # Imposta il radiobutton "Yes" come predefinito
        button_group3.buttons()[0].setChecked(True)

        # Memorizza la selezione predefinita
        self.column_types3[column_name3] = "Yes"

        # Aggiunge il layout al layout principale
        self.scroll_area_layout3.addLayout(column_layout3)

    def update_column_type3(self, column_name, option, checked):
        if checked:  # Aggiorna solo se il radiobutton è selezionato
            self.column_types3[column_name] = option

    def clear_layout3(self, layout):
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget() is not None:
                    child.widget().deleteLater()
                elif child.layout() is not None:
                    self.clear_layout3(child.layout())

# Classe per la finestra custom role
class labelWindow(QWidget):

    def __init__(self, grouped_columns, parent=None):
        super(labelWindow, self).__init__(parent)
        self.grouped_columns = grouped_columns
        if len(grouped_columns["Experiments"]) > 1:
            grouped_columns["Experiments"] = [
                exp for exp in grouped_columns["Experiments"]
                if exp.get("ErrorStrategy") != "custom-role"
            ]    

        self.setWindowTitle("Custome role")
        self.setGeometry(100, 100, 1000, 1000)

        # Imposta il colore di sfondo
        self.setStyleSheet("background-color: #2e2e2e;")  # Sfondo grigio scuro

        self.column_types3 = {}  # Dizionario per memorizzare i ML
        self.column_types2 = {}  # Dictionary for features
        self.layout = QVBoxLayout()

        # Label per il titolo
        self.label = QLabel("Choose ML", self)
        self.label.setStyleSheet("color: #ffffff;")  # Testo bianco
        self.layout.addWidget(self.label)

        # ScrollArea per ospitare i modelli di ML e i radiobuttons
        self.scroll_area3 = QScrollArea(self)
        self.scroll_area_widget3 = QWidget()
        self.scroll_area_layout3 = QVBoxLayout(self.scroll_area_widget3)
        self.display_columns3()

        self.scroll_area3.setWidgetResizable(True)
        self.scroll_area3.setWidget(self.scroll_area_widget3)
        self.layout.addWidget(self.scroll_area3)

        # Label e input per "Step"
        self.step_label = QLabel("Step:", self)
        self.step_label.setStyleSheet("color: #ffffff;")  # Testo bianco
        self.step_input = QLineEdit(self)
        self.step_input.setText("0.2")
        self.step_input.setStyleSheet("color: #ffffff; background-color: #444444;")  # Testo bianco e sfondo scuro

        # Aggiungi label e input in un layout orizzontale
        self.step_layout = QHBoxLayout()
        self.step_layout.addWidget(self.step_label)
        self.step_layout.addWidget(self.step_input)
        self.layout.addLayout(self.step_layout)

        self.label_features = QLabel("Choose Features", self)
        self.label_features.setStyleSheet("color: #ffffff;")  
        self.layout.addWidget(self.label_features)

        # ScrollArea for integer features
        self.scroll_area2 = QScrollArea(self)
        self.scroll_area_widget2 = QWidget()
        self.scroll_area_layout2 = QVBoxLayout(self.scroll_area_widget2)

        self.scroll_area2.setWidgetResizable(True)
        self.scroll_area2.setWidget(self.scroll_area_widget2)
        self.layout.addWidget(self.scroll_area2)

        self.display_columns2() 

  # selection Input
        self.selection_label = QLabel("Selection criteria:", self)
        self.selection_label.setStyleSheet("color: #ffffff;")  # Testo bianco
        self.selection_input = QLineEdit(self)
        self.selection_input.setText("all")
        self.selection_input.setStyleSheet("background-color: #3c3c3c; color: #ffffff;")

        # Aggiungi label e input in un layout orizzontale
        self.selection_layout = QHBoxLayout()
        self.selection_layout.addWidget(self.selection_label)
        self.selection_layout.addWidget(self.selection_input)
        self.layout.addLayout(self.selection_layout)

  # Label e input per "distribution"
        self.distribution_label = QLabel("Distribution:", self)
        self.distribution_label.setStyleSheet("color: #ffffff;")  # Testo bianco
        self.distribution_input = QLineEdit(self)
        self.distribution_input.setText("random")
        self.distribution_input.setStyleSheet("color: #ffffff; background-color: #444444;")  # Testo bianco e sfondo scuro

        # Aggiungi label e input in un layout orizzontale
        self.distribution_layout = QHBoxLayout()
        self.distribution_layout.addWidget(self.distribution_label)
        self.distribution_layout.addWidget(self.distribution_input)
        self.layout.addLayout(self.distribution_layout)
   # Label e input per "param"
        self.param_label = QLabel("Parameter:", self)
        self.param_label.setStyleSheet("color: #ffffff;")  # Testo bianco
        self.param_input = QLineEdit(self)
        self.param_input.setText("")
        self.param_input.setStyleSheet("color: #ffffff; background-color: #444444;")  # Testo bianco e sfondo scuro

        # Aggiungi label e input in un layout orizzontale
        self.param_layout = QHBoxLayout()
        self.param_layout.addWidget(self.param_label)
        self.param_layout.addWidget(self.param_input)
        self.layout.addLayout(self.param_layout)

  # Label e input per "value"
        self.value_label = QLabel("Parameter value:", self)
        self.value_label.setStyleSheet("color: #ffffff;")  # Testo bianco
        self.value_input = QLineEdit(self)
        self.value_input.setText("")
        self.value_input.setStyleSheet("color: #ffffff; background-color: #444444;")  # Testo bianco e sfondo scuro

        # Aggiungi label e input in un layout orizzontale
        self.value_layout = QHBoxLayout()
        self.value_layout.addWidget(self.value_label)
        self.value_layout.addWidget(self.value_input)
        self.layout.addLayout(self.value_layout)

        # Crea un layout orizzontale per i pulsanti
        button_layout = QHBoxLayout()

        
        self.next_button = QPushButton("Save", self)
        self.next_button.setStyleSheet("background-color: #0088CC; color: #000000;")
        self.next_button.clicked.connect(self.open_next_window)
        experiments = self.grouped_columns.get("Experiments", [])
        if len(experiments)==2:
           self.next_button.setEnabled(False)
        else:
            self.next_button.setEnabled(True)
            self.next_button.setStyleSheet("background-color: #0088CC; color: #ffffff;")

        
        self.layout.addWidget(self.next_button)

        self.run_button = QPushButton("Save and Run...", self)
        self.run_button.setStyleSheet("background-color: #0088CC; color: #000000;")
        self.run_button.clicked.connect(self.run)
        experiments = self.grouped_columns.get("Experiments", [])
        if len(experiments)==2:
           self.run_button.setEnabled(False)
        else:
            self.run_button.setEnabled(True)
            self.run_button.setStyleSheet("background-color: #0088CC; color: #ffffff;")
       
        self.layout.addWidget(self.run_button)

        self.skip_button = QPushButton("Skip and Save", self)
        self.skip_button.setStyleSheet("background-color: #0088CC; color: #000000;")
        experiments = self.grouped_columns.get("Experiments", [])
        if len(experiments)==1:
           self.skip_button.setEnabled(False)
        else:
            self.skip_button.setEnabled(True)
            self.skip_button.setStyleSheet("background-color: #0088CC; color: #ffffff;")
       
        self.skip_button.clicked.connect(self.open_skip_window)
        self.layout.addWidget(self.skip_button)
        
        self.skip_button2 = QPushButton("Skip, Save and Run", self)
        self.skip_button2.setStyleSheet("background-color: #0088CC; color: #000000;")
        if len(experiments)==1:
           self.skip_button2.setEnabled(False)
        else:
            self.skip_button2.setEnabled(True)
            self.skip_button2.setStyleSheet("background-color: #0088CC; color: #ffffff;")

        self.skip_button2.clicked.connect(self.skip_run)
        self.layout.addWidget(self.skip_button2)

        self.prev_button = QPushButton("Back", self)
        self.prev_button.setStyleSheet("background-color: #0088CC; color: #ffffff;")
        self.prev_button.clicked.connect(self.open_prev_window)
        self.layout.addWidget(self.prev_button)

        self.layout.addLayout(button_layout)

        self.setLayout(self.layout)

        # Aggiungi il layout dei pulsanti al layout principale
        self.layout.addLayout(button_layout)


        self.setLayout(self.layout)

    def open_prev_window(self):
        self.close()  
        self.next_window = DuplicateWindow(self.grouped_columns)  # Apre la nuova finestra
        self.next_window.show()

   
    def display_columns2(self):
        self.clear_layout2(self.scroll_area_layout2)
        self.column_types2.clear()


        column_df=pd.read_csv(self.grouped_columns["datasetName"])
        columns=column_df.columns.tolist()
        for column in columns:
            self.add_column_radiobuttons2(column)

    def add_column_radiobuttons2(self, column_name2):
        # Layout orizzontale per ogni colonna
        column_layout2 = QHBoxLayout()
        # Etichetta con il nome della colonna
        label2 = QLabel(column_name2, self)
        label2.setStyleSheet("color: #ffffff;")  # Testo bianco
        column_layout2.addWidget(label2)

        # Gruppo di radiobuttons
        button_group2 = QButtonGroup(self)

        # Opzioni dei radiobuttons
        options = ["Yes","No"]
        for option2 in options:
            radiobutton2 = QRadioButton(option2, self)
            radiobutton2.setStyleSheet("color: #ffffff;")  # Testo bianco per i radiobuttons
            button_group2.addButton(radiobutton2)
            column_layout2.addWidget(radiobutton2)

            # Collega ogni radiobutton a una funzione che aggiorna il dizionario delle selezioni
            radiobutton2.toggled.connect(lambda checked, col=column_name2, opt=option2: self.update_column_type2(col, opt, checked))

        button_group2.buttons()[1].setChecked(True)

        # Memorizza la selezione predefinita
        self.column_types2[column_name2] = "No"

        # Aggiunge il layout al layout principale
        self.scroll_area_layout2.addLayout(column_layout2)
    
    def update_column_type2(self, column_name, option, checked):
        if checked:  # Aggiorna solo se il radiobutton è selezionato
            self.column_types2[column_name] = option

    def display_columns3(self):
        # Pulisce il layout precedente e resetta le selezioni
        self.clear_layout3(self.scroll_area_layout3)
        self.column_types3.clear()
        # Crea un set di radiobuttons per ogni colonna
        ML = self.grouped_columns["machineLearningModels"]
        for model in ML:
            self.add_column_radiobuttons3(model)

    def add_column_radiobuttons3(self, column_name3):
        # Layout orizzontale per ogni colonna
        column_layout3 = QHBoxLayout()

        # Etichetta con il nome della colonna
        label3 = QLabel(column_name3, self)
        label3.setStyleSheet("color: #ffffff;")  # Testo bianco
        column_layout3.addWidget(label3)

        # Gruppo di radiobuttons
        button_group3 = QButtonGroup(self)

        # Opzioni dei radiobuttons
        options = ["Yes", "No"]
        for option3 in options:
            radiobutton3 = QRadioButton(option3, self)
            radiobutton3.setStyleSheet("color: #ffffff;")  # Testo bianco per i radiobuttons
            button_group3.addButton(radiobutton3)
            column_layout3.addWidget(radiobutton3)

            # Collega ogni radiobutton a una funzione che aggiorna il dizionario delle selezioni
            radiobutton3.toggled.connect(lambda checked, col=column_name3, opt=option3: self.update_column_type3(col, opt, checked))

        # Imposta il radiobutton "Yes" come predefinito
        button_group3.buttons()[0].setChecked(True)

        # Memorizza la selezione predefinita
        self.column_types3[column_name3] = "Yes"

        # Aggiunge il layout al layout principale
        self.scroll_area_layout3.addLayout(column_layout3)

    def update_column_type3(self, column_name, option, checked):
        if checked:  # Aggiorna solo se il radiobutton è selezionato
            self.column_types3[column_name] = option

    def clear_layout3(self, layout):
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget() is not None:
                    child.widget().deleteLater()
                elif child.layout() is not None:
                    self.clear_layout(child.layout())

    def open_next_window(self):
        newML = []
        newFeat=[]
        addedDocument = {"ErrorStrategy": "custom-role"}
        addedDocument["Step"] = float(self.step_input.text())
        addedDocument["affected_features"]=self.affected_input.text()
        addedDocument["selection_criteria"]=self.selection_input.text()
        addedDocument["distribution"]=self.distribution_input.text()
        if self.param_input.text()=="" or self.param_input.text() is None:
            addedDocument["param"]=None
        else:
            addedDocument["param"]=self.param_input.text()
        if self.value_input.text()=="" or self.value_input.text() is None:
            addedDocument["value"]=None
        else:
            addedDocument["value"]=self.value_input.text()
        for column2, col_type2 in self.column_types2.items():
            if col_type2 == "Yes":
                newFeat.append(column2)
        addedDocument["affected_features"] = newFeat
        self.grouped_columns["Experiments"].append(addedDocument)

        for column3, col_type3 in self.column_types3.items():
            if col_type3 == "Yes":
                newML.append(column3)
        if newML != self.grouped_columns["machineLearningModels"]:
            addedDocument["machineLearningModels"] = newML
        self.grouped_columns["Experiments"].append(addedDocument)
        
        print(self.grouped_columns)
        self.close()  # Chiude la finestra "Duplicate Error"
    
    def run(self):
        
        newML = []
        newFeat=[]
        addedDocument = {"ErrorStrategy": "custom-role"}
        addedDocument["Step"] = float(self.step_input.text())
        addedDocument["affected_features"]=self.affected_input.text()
        addedDocument["selection_criteria"]=self.selection_input.text()
        addedDocument["distribution"]=self.distribution_input.text()
        if self.param_input.text()=="" or self.param_input.text() is None:
            addedDocument["param"]=None
        else:
            addedDocument["param"]=self.param_input.text()
        if self.value_input.text()=="" or self.value_input.text() is None:
            addedDocument["value"]=None
        else:
            addedDocument["value"]=self.value_input.text()
        for column2, col_type2 in self.column_types2.items():
            if col_type2 == "Yes":
                newFeat.append(column2)
        addedDocument["affected_features"] = newFeat
        self.grouped_columns["Experiments"].append(addedDocument)

        for column3, col_type3 in self.column_types3.items():
            if col_type3 == "Yes":
                newML.append(column3)
        if newML != self.grouped_columns["machineLearningModels"]:
            addedDocument["machineLearningModels"] = newML
        self.grouped_columns["Experiments"].append(addedDocument)
        
        print(self.grouped_columns)

        self.file_name=self.save(self.grouped_columns)
        UI.start(self.file_name)
        self.close()

    def open_prev_window(self):
        self.close()  
        self.next_window = DuplicateWindow(self.grouped_columns)  
        self.next_window.show()

    def open_skip_window(self):
        
        self.save(self.grouped_columns)
        self.close() 

    def skip_run(self):
        self.file_name=self.save(self.grouped_columns)
        UI.start(self.file_name)
        self.close() 

    def save(self, grouped_columns):
        model_map = {
            "Logistic Regression": "lr",
            "K-Nearest Neighbors": "knn",
            "Naive Bayes": "nb",
            "Support Vector Machine": "svm",
            "Decision Tree": "dt",
            "Radial Basis Function SVM": "rbfsvm",
            "Gaussian Process Classifier": "gpc",
            "Multi-Layer Perceptron": "mlp",
            "Ridge Classifier": "ridge",
            "Quadratic Discriminant Analysis": "qda",
            "AdaBoost": "ada",
            "Linear Discriminant Analysis": "lda",
            "Dummy Classifier": "dummy",
            "Random Forest": "rf",
            "Extra Trees": "et",
            "Gradient Boosting Classifier": "gbc",
            "LightGBM": "lightgbm",
            "CatBoost": "catboost"  
        }

        grouped_columns["machineLearningModels"] = [
            model_map.get(model, model) for model in grouped_columns["machineLearningModels"]
        ]
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save JSON File", "", "JSON Files (*.json);;All Files (*)", options=options)
        
        try:
                with open(file_name, 'w') as json_file:
                    json.dump(grouped_columns, json_file, indent=4)
                print(f'Successfully saved JSON to {self.file_name}')
        except Exception as e:
                print(f'Error saving JSON file: {e}')
        return file_name
  
    def clear_layout2(self, layout):
         if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget() is not None:
                    child.widget().deleteLater()
                elif child.layout() is not None:
                    self.clear_layout(child.layout())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CSVColumnTypeSelector()
    ex.show()
    sys.exit(app.exec_())

    