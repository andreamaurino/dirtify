# dirtify
A suite for studying how the quality of the training set impacts the performance of machine learning tasks.

## Install
To install the suite create a virtual env in your directory by typing this command (use python 3.11 version)

python -m venv env

Then activate the env
source env/bin/activate (linux)
env\Scripts\activate (windows)

Finally  install requirements 
pip install -r requirements.txt

## Execute
To use the suite you have select a dataset in CSV format and store into the datasetRoot directory, then run the  file configurator.py the final json file will be stored in the json directory.
If you already created the configuration file  you can use: python  dirtify.py  and then select your json fil. Results are stored in the experiments folder
For the EPC analysis use python epc_analysis.py datasetname.csv results are stored in the epcResults directory. Experiments and results are stored in the database datasetname.csv.db




