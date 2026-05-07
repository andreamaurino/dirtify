import argparse
import sys
from PyQt5.QtWidgets import QApplication, QFileDialog
import UI
# forza flush immediato su stdout — fondamentale su HPC con output rediretto su file
sys.stdout.reconfigure(line_buffering=True)
def select_json_file():
    app = QApplication(sys.argv)  
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    file_name, _ = QFileDialog.getOpenFileName(None, "Select configuration file ", "", 
                                                "JSON file (*.json);;all files (*)", 
                                                options=options)
    return file_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?", help="JSON config file")  # nargs="?" lo rende opzionale
    parser.add_argument("multiprocessor",default="No", nargs="?", help="Multiprocessor file")  # nargs="?" lo rende opzionale
    parser.add_argument("worker",default=1, nargs="?", help="worker number")  # nargs="?" lo rende opzionale
    args = parser.parse_args()

    file_name = args.file if args.file else select_json_file()
    multiprocessor = args.multiprocessor
    worker = args.worker
    if file_name:
        UI.start(file_name,multiprocessor,int(worker))
    else:
        print("No selected file")