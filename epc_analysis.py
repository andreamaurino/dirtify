import argparse
import epc_calcolus
import os
import sys
if len(sys.argv) != 2:
        print("Usage: python start_analysis.py <datasetname.csv>")
        sys.exit(1)
parser = argparse.ArgumentParser(description='Please give me the dataset name in csv format')

# Aggiungere un argomento per il nome del file
parser.add_argument('dataset_name', nargs='?', default='penguins_binary_classification.csv', type=str, help='filename')

# Parse degli argomenti dalla riga di comando
args = parser.parse_args()
dataset_name=args.dataset_name
# Utilizzare il nome del file passato come argomento
file_name = "experiments_"+dataset_name
epc_df=epc_calcolus.start(dataset_name,file_name)

output_dir = 'epcResults'
os.makedirs(output_dir, exist_ok=True)

output_file_name = os.path.join(output_dir, f"epc_{dataset_name}")
epc_df.to_csv(output_file_name, index=False)

print(epc_df)
#epc_calcolus.visualize_feature(epc_df,dataset_name,file_name)
#epc_calcolus.visualize_epc(epc_df,dataset_name,file_name)
