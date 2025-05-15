import argparse
import sys
from PyQt5.QtWidgets import QApplication, QFileDialog
import UI


def select_json_file():
    app = QApplication(sys.argv)  
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    file_name, _ = QFileDialog.getOpenFileName(None, "Select configuration file ", "", 
                                                "JSON file (*.json);;all files (*)", 
                                                options=options)
    return file_name

if __name__ == "__main__":
    file_name = select_json_file() 
    if file_name:  
        UI.start(file_name)
    else:
        print("No selected file")
