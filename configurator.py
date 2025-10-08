import sys
import csv
import json
import os 
import pandas as pd
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication,  QTableWidget,  QTableWidgetItem, QWidget, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QLabel, QRadioButton, QButtonGroup, QScrollArea, QLabel, QLineEdit,  QHBoxLayout, QComboBox
from PyQt5.QtGui import QDoubleValidator
import UI
from typing import List, Dict, Any
import itertools


class InitWindow(QWidget):
    def __init__(self, grouped_columns={}, parent=None):
        super(InitWindow, self).__init__(parent)
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
        
        self.target_label.setFont(QFont("Arial", 12))
        self.target_label.setStyleSheet("color: #ffffff;background-color: #000000;")  # Testo bianco
        self.target_label.setStyleSheet("color: #ffffff; background-color: #000000;")  # Testo bianco
        self.target_input = QLineEdit(self)
        self.target_input.setText(self.tmp_df.columns[-1] if hasattr(self, 'tmp_df') else "")
        self.target_input.setStyleSheet("color: #ffffff; background-color: #000000;")  # Testo bianco e sfondo scuro

        # Aggiungi label e input in un layout orizzontale
        self.target_layout = QHBoxLayout()
        self.target_layout.addWidget(self.target_label)
        self.target_layout.addWidget(self.target_input)
        self.main_layout.addLayout(self.target_layout)

        # Bottone per salvare i risultati come JSON (spostato alla fine)
        self.button = QPushButton("Create Error Strategy", self)
        self.button.clicked.connect(self.open_standard)
        self.button.setEnabled(False)  # Disabilitato finché non viene caricato un file
        self.main_layout.addWidget(self.button)

        self.setLayout(self.main_layout)

        self.setStyleSheet("""
        QWidget {
            background-color: #333333;
            color: #FFFFFF;
            font-family: Arial, sans-serif;
        }

        /* ScrollArea: evita “isole” bianche */
        QScrollArea, QScrollArea > QWidget, QScrollArea > QWidget > QWidget#qt_scrollarea_viewport {
            background-color: transparent;
        }

        /* Tabelle */
        QTableView, QTableWidget {
            background-color: #2b2b2b;
            color: #eaeaea;
            gridline-color: #555555;
            selection-background-color: #006699;
            selection-color: #ffffff;
            alternate-background-color: #242424;
        }

        /* Intestazioni delle tabelle */
        QHeaderView::section {
            background-color: #444444;
            color: #ffffff;
            padding: 6px;
            border: 1px solid #555555;
            font-weight: 600;
        }

        /* Angolo in alto a sinistra della tabella */
        QTableCornerButton::section {
            background-color: #444444;
            border: 1px solid #555555;
        }

        QPushButton {
            background-color: #0088CC;
            color: white;
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
        }
     QPushButton:hover { background-color: #006699; }
        QLabel { margin: 10px 0; }

        /* Campi input coerenti con tema scuro */
        QLineEdit {
            background-color: #000000;
            color: #ffffff;
            border: 1px solid #555555;
            padding: 6px;
            border-radius: 4px;
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
            self.tmp_df = pd.read_csv(file_path, sep=",", encoding='iso-8859-1')

    def display_columns(self, file_path):
        # Pulisce il layout precedente e resetta le selezioni
        #self.clear_layout(self.scroll_area_layout)
        #self.column_types.clear()

        # Legge la prima riga del file CSV
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            headers = next(reader)


    def display_columns2(self):
        self.tmp_df = pd.read_csv(self.csv_file_name, sep=",", encoding='iso-8859-1')
        self.target_input.setText(self.tmp_df.columns[-1] if hasattr(self, 'tmp_df') else "")
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
        label2.setFont(QFont("Arial", 12))
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
                "targetVariable":self.target_input.text(),
                "features":[],
                "machineLearningModels":[],
                "Experiments":[]
            }
            if self.testset_name!="":
                grouped_columns["testset"]=self.testset_name
                # Popola il dizionario ml raggruppando le colonne per tipo
            for column2, col_type2 in self.column_types2.items():
                if col_type2 == "Yes":
                    grouped_columns["machineLearningModels"].append(column2)
            self.tmp_df = pd.read_csv(self.csv_file_name, sep=",", encoding='iso-8859-1')
            grouped_columns["features"] = list(self.tmp_df.columns)
            addedDocument={"Errortype":"standard"}
            grouped_columns["Experiments"].append(addedDocument)
         #   print(grouped_columns)
            self.close()  # Chiude la finestra principale
            self.standard_window = OneFeature(grouped_columns)  # Apre la nuova finestra
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
class OneFeature(QWidget):
    def __init__(self, grouped_columns, parent=None):
        super(OneFeature, self).__init__(parent)
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
        self.column_types31 = {}
        self.column_types2 = {}

        self.layout = QVBoxLayout()
        # features
        self.label11 = QLabel("Choose Features", self)
        self.label11.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.layout.addWidget(self.label11)

        # ScrollArea for ML Models
        self.scroll_area31 = QScrollArea(self)
        self.scroll_area_widget31 = QWidget()
        self.scroll_area_layout31 = QVBoxLayout(self.scroll_area_widget31)
        self.display_columns31()
        self.scroll_area31.setWidgetResizable(True)
        self.scroll_area31.setWidget(self.scroll_area_widget31)
        self.layout.addWidget(self.scroll_area31)

        # Header Label
        self.label12 = QLabel("Choose ML Model", self)
        self.label12.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.layout.addWidget(self.label12)

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
        self.next_window = InitWindow(self.grouped_columns)
        self.next_window.show()

    def open_skip_window(self):
      #  print(self.grouped_columns)
        self.close()
        self.next_window = CorrelatedFeature(self.grouped_columns)
        self.next_window.show()

    def open_next_window(self):
        newML = []
        features = []
        addedDocument = {"ErrorStrategy": "one-feature"}
        addedDocument["Step"] = float(self.step_input.text())
        addedDocument["Selection_criteria"] = self.selection_input.text()
        addedDocument["distribution"]=self.distribution_input.text()
        if self.param_input.text()=="" or self.param_input.text() is None:
            addedDocument["param"]=None
        else:
            addedDocument["param"]=self.param_input.text()
        if self.value_input.text()=="" or self.value_input.text() is None:
            addedDocument["value"]=None
        else:
            addedDocument["value"]=self.value_input.text()
        for column3, col_type3 in self.column_types3.items():
            if col_type3 == "Yes":
                newML.append(column3)
        for column31, col_type31 in self.column_types31.items():
            if col_type31 == "Yes":
                features.append(column31)
        if features != self.grouped_columns["features"]:        
            self.grouped_columns["features"] = features   
        if newML != self.grouped_columns["machineLearningModels"]:
            addedDocument["machineLearningModels"] = newML
        newColumns = []
        for column2, col_type2 in self.column_types2.items():
            if col_type2 == "Yes":
                newColumns.append(column2)
        self.grouped_columns["Experiments"].append(addedDocument)
       # print(self.grouped_columns)
        self.close()
        self.next_window = CorrelatedFeature(self.grouped_columns)
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






    def display_columns31(self):
        self.clear_layout31(self.scroll_area_layout31)
        self.column_types31.clear()
        features = self.grouped_columns["features"]
        for feat in features:
            self.add_column_radiobuttons31(feat)

    def add_column_radiobuttons31(self, column_name31):
        column_layout31 = QHBoxLayout()
        label31 = QLabel(column_name31, self)
        column_layout31.addWidget(label31)
        button_group31 = QButtonGroup(self)
        options = ["Yes", "No"]
        for option31 in options:
            radiobutton31 = QRadioButton(option31, self)
            button_group31.addButton(radiobutton31)
            column_layout31.addWidget(radiobutton31)
            radiobutton31.toggled.connect(lambda checked, col=column_name31, opt=option31: self.update_column_type31(col, opt, checked))

        button_group31.buttons()[0].setChecked(True)
        self.column_types31[column_name31] = "Yes"
        self.scroll_area_layout31.addLayout(column_layout31)

    def update_column_type31(self, column_name, option, checked):
        if checked:
            self.column_types31[column_name] = option

    def clear_layout31(self, layout):
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
class CorrelatedFeature(QWidget):

    def __init__(self, grouped_columns, parent=None):
        super(CorrelatedFeature, self).__init__(parent)
        self.grouped_columns = grouped_columns
        if len(grouped_columns["Experiments"]) > 1:
            grouped_columns["Experiments"] = [
                exp for exp in grouped_columns["Experiments"]
                if exp.get("ErrorStrategy") != "Correlated-features"
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
        self.next_window = OneFeature(self.grouped_columns)  # Apre la nuova finestra
        self.next_window.show()

    def open_skip_window(self):
        self.close()  # Chiude la finestra "Duplicate Analysis"
        self.next_window = CustomRole(self.grouped_columns)  # Apre la nuova finestra
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
        self.next_window = CustomRole(self.grouped_columns)  # Apre la nuova finestra
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
class CustomRole(QWidget):

    def __init__(self, grouped_columns, parent=None):
        super(CustomRole, self).__init__(parent)
        self.grouped_columns = grouped_columns
        if len(grouped_columns["Experiments"]) > 1:
            grouped_columns["Experiments"] = [
                exp for exp in grouped_columns["Experiments"]
                if exp.get("ErrorStrategy") != "custom-role"
            ]    
        self.ErrorType="Label"
        self.setWindowTitle("Custome role")
        self.setGeometry(100, 100, 1000, 1000)

        # Imposta il colore di sfondo
        self.setStyleSheet("background-color: #2e2e2e;")  # Sfondo grigio scuro

        self.column_types3 = {}  # Dizionario per memorizzare i ML
        self.column_types2 = {}  # Dictionary for features
        self.layout = QVBoxLayout()

        self.label1 = QLabel("Error Type:")
        self.label1.setStyleSheet("color: #ffffff;")  # Testo bianco
        self.layout.addWidget(self.label1)
        
        # ComboBox (scelta a tendina)
        self.combo = QComboBox()
        self.combo.addItems(["Label","Missing","Noise","Outlier"])
        # Segnale per catturare il cambiamento
        self.combo.currentIndexChanged.connect(self.selection_changed)
        self.layout.addWidget(self.combo)
        self.combo.setStyleSheet("""
            QComboBox {
                color: white;         /* testo della scelta */
            }
            QComboBox QAbstractItemView {
                color: white;         /* testo delle voci */
                selection-background-color: #555555; /* sfondo della voce selezionata */
            }
        """)
        
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

        self.next_rule = QPushButton("Add new rule", self)
        self.next_rule.setStyleSheet("background-color: #0088CC; color: #000000;")
        self.next_rule.clicked.connect(self.open_same_window)
        experiments = self.grouped_columns.get("Experiments", [])
        if len(experiments)==2:
            if self.grouped_columns["Experiments"][1]["ErrorStrategy"] == "one-feature": #or ....:
                self.next_rule.setEnabled(False)
        else:
            self.next_rule.setEnabled(True)
            self.next_rule.setStyleSheet("background-color: #0088CC; color: #ffffff;")

        
        self.layout.addWidget(self.next_rule)

        self.next_button = QPushButton("Save", self)
        self.next_button.setStyleSheet("background-color: #0088CC; color: #000000;")
        self.next_button.clicked.connect(self.open_next_window)
        if len(experiments)==2:
            if self.grouped_columns["Experiments"][1]["ErrorStrategy"] == "one-feature": #or ....:
                self.next_button.setEnabled(False)
        else:
            self.next_button.setEnabled(True)
            self.next_button.setStyleSheet("background-color: #0088CC; color: #ffffff;")

        self.next_button.clicked.connect(self.open_next_window)
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
        self.run_button.clicked.connect(self.run)  
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

    def selection_changed(self, index):
        self.ErrorType=self.combo.currentText()


    def open_prev_window(self):
        self.close()  
        self.next_window = CorrelatedFeature(self.grouped_columns)  # Apre la nuova finestra
        self.next_window.show()
   
    def display_columns2(self):
        self.clear_layout2(self.scroll_area_layout2)
        self.column_types2.clear()
        column_df=pd.read_csv(self.grouped_columns["datasetName"])
        columns=column_df.columns.tolist()
        for column in columns:
            self.add_column_radiobuttons2(column)
    
    def clear_layout2(self, layout):
         if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget() is not None:
                    child.widget().deleteLater()
                elif child.layout() is not None:
                    self.clear_layout(child.layout())


class NoiseDiscreteWindow(QWidget):
    
    def __init__(self, grouped_columns, parent=None):
        super(NoiseDiscreteWindow, self).__init__(parent)
        self.grouped_columns = grouped_columns
        if len(grouped_columns["Experiments"]) > 1:
            grouped_columns["Experiments"] = [
                Experiments for Experiments in grouped_columns["Experiments"] 
                if not (Experiments["Errortype"] == "noise" and Experiments["FeatureType"] == "discrete")
            ]
    
        self.setWindowTitle("Noise Discrete Analysis")
        self.setGeometry(100, 100, 1000, 1000)

        # Imposta il colore di sfondo della finestra
        self.setStyleSheet("background-color: #2b2b2b; color: #ffffff;")  # Grigio scuro

        self.column_types3 = {}  # Dizionario per memorizzare i modelli ML
        self.column_types2 = {}  # Dizionario per memorizzare le features

        self.layout = QVBoxLayout()

        # Etichetta "Choose ML"
        self.label = QLabel("Choose ML", self)
        self.label.setStyleSheet("color: white;")  # Testo bianco
        self.layout.addWidget(self.label)

        # ScrollArea per i modelli ML
        self.scroll_area3 = QScrollArea(self)
        self.scroll_area_widget3 = QWidget()
        self.scroll_area_layout3 = QVBoxLayout(self.scroll_area_widget3)
        self.display_columns3()

        self.scroll_area3.setWidgetResizable(True)
        self.scroll_area3.setWidget(self.scroll_area_widget3)
        self.layout.addWidget(self.scroll_area3)

        # Label e input per "Step"
        self.step_label = QLabel("Step:", self)
        self.step_label.setStyleSheet("color: white;")  # Testo bianco
        self.step_input = QLineEdit(self)
        self.step_input.setText("0.2")

        self.step_layout = QHBoxLayout()
        self.step_layout.addWidget(self.step_label)
        self.step_layout.addWidget(self.step_input)
        self.layout.addLayout(self.step_layout)

        # Etichetta "Choose Discrete Features"
        self.label = QLabel("Choose Discrete Features", self)
        self.label.setStyleSheet("color: white;")  # Testo bianco
        self.layout.addWidget(self.label)

        # ScrollArea per le features
        self.scroll_area2 = QScrollArea(self)
        self.scroll_area_widget2 = QWidget()
        self.scroll_area_layout2 = QVBoxLayout(self.scroll_area_widget2)

        self.scroll_area2.setWidgetResizable(True)
        self.scroll_area2.setWidget(self.scroll_area_widget2)
        self.layout.addWidget(self.scroll_area2)

        self.display_columns2()

        # Layout per i bottoni
        button_layout = QHBoxLayout()

        # Pulsante "Back"
        self.prev_button = QPushButton("Back", self)
        self.prev_button.setStyleSheet("background-color: #0088CC; color: #000000;")  # Azzurro con testo bianco
        self.prev_button.clicked.connect(self.open_prev_window)
        button_layout.addWidget(self.prev_button)

        # Pulsante "Skip"
        self.skip_button = QPushButton("Skip", self)
        self.skip_button.setStyleSheet("background-color: #0088CC; color: #000000;")  # Azzurro con testo bianco
        self.skip_button.clicked.connect(self.open_skip_window)
        button_layout.addWidget(self.skip_button)

        # Pulsante "Next"
        self.next_button = QPushButton("Next", self)
        self.next_button.setStyleSheet("background-color: #0088CC; color: #000000;")  # Azzurro con testo bianco
        self.next_button.clicked.connect(self.open_next_window)
        if len(self.grouped_columns["discreteFeatures"]) == 0:
            self.next_button.setEnabled(False)
        button_layout.addWidget(self.next_button)

        # Aggiungi il layout dei pulsanti al layout principale
        self.layout.addLayout(button_layout)

        self.setLayout(self.layout)

    def open_prev_window(self):
        self.close()  # Chiude la finestra "missing Error"
        self.next_window = OutlierCatIntWindow(self.grouped_columns)  # Apre la nuova finestra
        self.next_window.show()

    def open_skip_window(self):
        self.close()  # Chiude la finestra "Duplicate Error"
        self.next_window = NoiseContinousWindow(self.grouped_columns)  # Apre la nuova finestra
        self.next_window.show()

    def open_next_window(self):
        newML=[]
        newColumn=[]
        addedDocument={"Errortype":"noise"}
        addedDocument["FeatureType"]="discrete"
        addedDocument["Step"]= float(self.step_input.text())
        for column2, col_type2 in self.column_types2.items():
                if col_type2 == "Yes":
                    newColumn.append(column2)
        if newColumn!=self.grouped_columns["discreteFeatures"]:
            addedDocument["FeatureArray"]=newColumn
        for column3, col_type3 in self.column_types3.items():
                if col_type3 == "Yes":
                    newML.append(column3)
        if newML!=self.grouped_columns["machineLearningModels"]:
            addedDocument["machineLearningModels"]=newML
        self.grouped_columns["Experiments"].append(addedDocument)
        print(self.grouped_columns["Experiments"])
        self.close()  # Chiude la finestra "Duplicate Error"
        self.next_window = NoiseContinousWindow(self.grouped_columns)  # Apre la nuova finestra
        self.next_window.show()

    def display_columns2(self):
        # Pulisce il layout precedente e resetta le selezioni
        self.clear_layout2(self.scroll_area_layout2)
        self.column_types2.clear()
        # Crea un set di radiobuttons per ogni colonna
        columns=self.grouped_columns["discreteFeatures"]
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


    def open_same_window(self):
        newML = []
        newFeat=[]
        addedDocument = {"ErrorStrategy": "custom-role"}
        addedDocument["ErrorType"]=self.ErrorType
        addedDocument["Step"] = float(self.step_input.text())
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
        self.prepare(self.grouped_columns)
        print(self.grouped_columns)
        self.close()
        self.next_window = CustomRole(self.grouped_columns)  
        self.next_window.show()


    def open_next_window(self):
        newML = []
        newFeat=[]
        addedDocument = {"ErrorStrategy": "custom-role"}
        addedDocument["Step"] = float(self.step_input.text())
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
        self.grouped_columns["Experiments"].append(addedDocument)#

        for column3, col_type3 in self.column_types3.items():
           if col_type3 == "Yes":
                newML.append(column3)
        if newML != self.grouped_columns["machineLearningModels"]:
            addedDocument["machineLearningModels"] = newML
        self.grouped_columns["Experiments"].append(addedDocument)
        
        self.close()  # Chiude la finestra "Duplicate Error"
        self.prepare(self.grouped_columns)
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog  # <— workaround
        file_name, _ = QFileDialog.getSaveFileName(
                self, "Save JSON File", "", "JSON Files (*.json);;All Files (*)", options=options
            )
        try:
                with open(file_name, 'w') as json_file:
                    json.dump(self.grouped_columns, json_file, indent=4)
                print(f'Successfully saved JSON to {file_name}')
        except Exception as e:
                print(f'Error saving JSON file: {e}')
        return file_name
  

    def run(self):
        file_name=self.open_next_window()
        UI.start(file_name)
        self.close()

    def open_prev_window(self):
        self.close()  
        self.next_window = CorrelatedFeature(self.grouped_columns)  
        self.next_window.show()

    def open_skip_window(self):
        
        self.prepare(self.grouped_columns)
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog  # <— workaround
        file_name, dialog= QFileDialog.getSaveFileName(
                self, "Save JSON File", "", "JSON Files (*.json);;All Files (*)", options=options
            )
        dialog.setStyleSheet("""
            QWidget { 
                color: white; 
                background-color: #222222; 
            }
            QLineEdit {
                color: white;
                background-color: #333333;
            }
            QListView, QTreeView {
                color: white;
                background-color: #333333;
            }
            QPushButton {
                color: white;
                background-color: #444444;
            }
            QPushButton::hover {
                background-color: #555555;
            }
        """)
        try:
                with open(file_name, 'w') as json_file:
                    json.dump(self.grouped_columns, json_file, indent=4)
                print(f'Successfully saved JSON to {file_name}')
        except Exception as e:
                print(f'Error saving JSON file: {e}')
        self.close() 

    def skip_run(self):
        self.prepare(self.grouped_columns)
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog  # <— workaround
        file_name, _ = QFileDialog.getSaveFileName(
                self, "Save JSON File", "", "JSON Files (*.json);;All Files (*)", options=options
            )
        try:
                with open(file_name, 'w') as json_file:
                    json.dump(self.grouped_columns, json_file, indent=4)
                print(f'Successfully saved JSON to {file_name}')
        except Exception as e:
                print(f'Error saving JSON file: {e}')
        UI.start(file_name)
        self.close() 

    def expand_one_feature_strategy(self, config: List[Dict[str, Any]], dataset_columns: List[str]) -> List[Dict[str, Any]]:
        ERRORTYPES = ["duplicate", "labels", "outlier", "noise"]
        one_feature_blocks = []
        for item in config:
            if item.get("ErrorStrategy") == "one-feature":
                one_feature_blocks.append(item)

        if not one_feature_blocks:
            return config

        new_docs: List[Dict[str, Any]] = []
        for block in one_feature_blocks:
            selection_criteria = block.get("Selection_criteria") or block.get("Selection-criteria")
            distribution = block.get("distribution")
            percentage = block.get("Step")

            for etype in ERRORTYPES:
                if etype == "labels":
                    doc = {
                            "Errortype": etype,
                            "Strategy": {
                                "affected_features": self.grouped_columns["targetVariable"],
                                "selection_criteria": selection_criteria,
                                "percentage": percentage,
                            "mode": "new",
                            "perturbate_data": {
                                "distribution": distribution
                                }
                        }
                    }
                    new_docs.append(doc)
                else:
                    for col in dataset_columns:
                        doc = {
                            "Errortype": etype,
                            "Strategy": {
                                "affected_features": [col],
                                "selection_criteria": selection_criteria,
                                "percentage": percentage,
                            "mode": "new",
                            "perturbate_data": {
                                "distribution": distribution
                                }
                            }
                        }
                        new_docs.append(doc)

        return new_docs

    def correlated_feature_groups(self):
        df=pd.read_csv(self.grouped_columns["datasetName"])
        target=self.grouped_columns["targetVariable"]
        min_corr=self.grouped_columns["Experiments"][1]["Min"] 
        max_corr=self.grouped_columns["Experiments"][1]["Max"]    
        original_cols = [c for c in df.columns if c != target]

        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        df_dummies = pd.get_dummies(df[original_cols], drop_first=False)

        dummy_to_orig = {}
        for col in df_dummies.columns:
            base = col.split("_")[0]
            dummy_to_orig[col] = base

        corr = df_dummies.corr().abs()

        pairs = [
            (d1, d2)
            for d1, d2 in itertools.combinations(df_dummies.columns, 2)
            if min_corr <= corr.loc[d1, d2] < max_corr
        ]

        groups_dummy = []
        for d1, d2 in pairs:
            found = False
            for g in groups_dummy:
                if d1 in g or d2 in g:
                    g.update([d1, d2])
                    found = True
                    break
            if not found:
                groups_dummy.append(set([d1, d2]))

        groups_orig = []
        for gd in groups_dummy:
            s = set(dummy_to_orig[d] for d in gd)
            groups_orig.append(s)

        merged = []
        for s in groups_orig:
            found = False
            for m in merged:
                if not s.isdisjoint(m):  # se c’è intersezione
                    m.update(s)
                    found = True
                    break
            if not found:
                merged.append(set(s))
                merged = [g for g in merged if len(g) >= 2]

        return merged

    def prepare(self, grouped_columns):
        ERRORTYPES = ["duplicate", "labels", "outlier", "noise"]
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
        if grouped_columns["Experiments"][1]["ErrorStrategy"] == "one-feature":
            dataset_columns=grouped_columns["features"].copy()
            dataset_columns.remove(grouped_columns["targetVariable"])
            grouped_columns["Experiments"].append(self.expand_one_feature_strategy(grouped_columns["Experiments"], dataset_columns))
            new_doc = [d for d in grouped_columns["Experiments"] if "ErrorStrategy" not in d]
            grouped_columns["Experiments"] = new_doc
        elif grouped_columns["Experiments"][1]["ErrorStrategy"] == "Correlated-features":
            ## calcola la correlazione
            groups=self.correlated_feature_groups()
            print("i gruppi")
            print(groups)
            for i in ERRORTYPES:
                for group in groups:
                    doc = {
                        "Errortype":i,
                        "Strategy": {
                            "affected_features": group,
                            "selection_criteria": self.grouped_columns["Experiments"][1]["Selection_criteria"],
                            "min": self.grouped_columns["Experiments"][1]["Min"],
                            "max": self.grouped_columns["Experiments"][1]["Max"],
                            "percentage": self.grouped_columns["Experiments"][1]["Step"],
                            "mode": "new",
                            "perturbate_data": {
                                "distribution": "random"
                                }
                            }
                        }
                    grouped_columns["Experiments"].append(doc)
            new_doc = [d for d in grouped_columns["Experiments"] if "ErrorStrategy" not in d]
            grouped_columns["Experiments"] = new_doc
        else: 
            
                doc = {
                            "Errortype":self.ErrorType,
                            "Strategy": {
                                "affected_features": self.grouped_columns["Experiments"][1]["affected_features"],
                                "selection_criteria": self.grouped_columns["Experiments"][1]["selection_criteria"],
                                "percentage": self.grouped_columns["Experiments"][1]["Step"],
                            "mode": "new",
                            "target_variable": self.grouped_columns["targetVariable"],
                            "perturbate_data": {
                                "distribution": self.grouped_columns["Experiments"][1]["distribution"]
                                }
                            }
                        }
                grouped_columns["Experiments"].append(doc)
                new_doc = [d for d in grouped_columns["Experiments"] if "ErrorStrategy" not in d]
                grouped_columns["Experiments"] = new_doc
        print(grouped_columns)        
        return grouped_columns
    
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
    ex = InitWindow()
    ex.show()
    sys.exit(app.exec_())

    