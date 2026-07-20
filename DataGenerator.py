from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
from pucktrick.labels import *
#adult = fetch_openml("adult", version=2, as_frame=True)
#X = adult.data
#y = adult.target

#print(X.shape, y.shape)
#print(X.head())
#X["income"] = y
#X.to_csv("adult_clean.csv", index=False)

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
#optical_recognition_of_handwritten_digits = fetch_ucirepo(id=80) 
  
# data (as pandas dataframes) 
#X = optical_recognition_of_handwritten_digits.data.features 
#y = optical_recognition_of_handwritten_digits.data.targets 
#df = pd.concat([X, y], axis=1)
# metadata 
#print(optical_recognition_of_handwritten_digits.metadata) 
  
# variable information 
#print(optical_recognition_of_handwritten_digits.variables) 
#df.to_csv("optdigit.csv", index=False)
#df=pd.read_csv("./datasetRoot/satimage_sample3600.csv")
# controlla la distribuzione delle classi
#print(df['class'].value_counts())
#print(df.shape)
#print(df[feature_cols].describe())
#print(df[feature_cols].std())
# se std è vicino a 0 per alcune feature → problema
#df_exp = pd.read_csv('experiments/experiments_satimage_sample3600.csv')
# quanto cambia il dataset sporcando una feature al 8
# nel CSV risultante, ora devono essere diversi
#feat1_ami = df_exp[
#    (df_exp['feature'] == '[A13attr]') & 
#    (df_exp['modelName'] == 'K-Means') &
#    (df_exp['percentage'] == 0.2)
#]['AMI'].values

#feat2_ami = df_exp[
#    (df_exp['feature'] == '[F30attr]') & 
#    (df_exp['modelName'] == 'K-Means') &
#    (df_exp['percentage'] == 0.2)
#]['AMI'].values

#print(feat1_ami)
#print(feat2_ami)
#print("Identici?", np.allclose(feat1_ami, feat2_ami))
from ucimlrepo import fetch_ucirepo

# # Dry Bean
# dry_bean = fetch_ucirepo(id=602)
# df_bean = dry_bean.data.features
# df_bean['target'] = dry_bean.data.targets
# print(df_bean)
# df_bean.to_csv("./datasetRoot/drybean.csv")

# # Pendigits
# pendigits = fetch_ucirepo(id=81)
# df_pen = pendigits.data.features
# df_pen['target'] = pendigits.data.targets
# df_pen.to_csv("./datasetRoot/pendigit.csv")
# print(df_pen)


#dowload data fom NASA 
# """
# download_nasa_ds_prime_prime.py
# ===============================
# Scarica MC2, KC3, MW1 nella versione DS'' (Shepperd et al., 2013)
# dalla fonte più affidabile disponibile, converte in CSV e verifica
# la correttezza (# istanze, # feature, % buggy).

# Fonti utilizzate (in ordine di priorità):
#   1. klainfo/NASADefectDataset (GitHub) → versione DS'' certa
#   2. Zenodo SeaCraft records         → fallback per MW1
#   3. Nota manuale se entrambe falliscono

# Riferimento: Shepperd M., Song Q., Sun Z., Mair C. (2013).
#   "Data Quality: Some Comments on the NASA Software Defect Datasets."
#   IEEE TSE 39(8):1208-1215. DOI: 10.1109/TSE.2013.11

# Versioni attese (DS'' = rimosse duplicate + inconsistenti):
#   MC2  → 127 istanze, 40 feature
#   KC3  → 434 istanze, 40 feature (NASA version)
#   MW1  → 370 istanze, 40 feature (NASA version)

# Uso:
#     pip install requests scipy pandas
#     python download_nasa_ds_prime_prime.py

# Output:
#     ./nasa_ds_pp/MC2_DS_pp.csv
#     ./nasa_ds_pp/KC3_DS_pp.csv
#     ./nasa_ds_pp/MW1_DS_pp.csv
#     ./nasa_ds_pp/summary.csv
# """

# import io
# import os
# import zipfile
# import requests
# import pandas as pd
# from scipy.io import arff

# # ---------------------------------------------------------------------------
# # Configurazione
# # ---------------------------------------------------------------------------

# OUTPUT_DIR = "./nasa_ds_pp"

# # GitHub ZIP dell'intero repo klainfo/NASADefectDataset
# GITHUB_ZIP_URL = (
#     "https://github.com/klainfo/NASADefectDataset/archive/refs/heads/master.zip"
# )

# # Zenodo fallback per MW1 DS'' (unico con file DS'' esplicito su Zenodo)
# ZENODO_MW1_DS_PP = (
#     "https://zenodo.org/records/268490/files/MW1''.arff?download=1"
# )

# # Mapping: nome dataset → path atteso nel repo GitHub (CleanedData/MDP/)
# # I file DS'' nel repo klainfo seguono il pattern NomeDataset''.arff
# GITHUB_PATHS = {
#     "MC2": [
#         "NASADefectDataset-master/CleanedData/MDP/MC2/MC2''.arff",
#         "NASADefectDataset-master/CleanedData/MDP/MC2''.arff",
#         "NASADefectDataset-master/CleanedData/MDP/mc2''.arff",
#     ],
#     "KC3": [
#         "NASADefectDataset-master/CleanedData/MDP/KC3/KC3''.arff",
#         "NASADefectDataset-master/CleanedData/MDP/KC3''.arff",
#         "NASADefectDataset-master/CleanedData/MDP/kc3''.arff",
#     ],
#     "MW1": [
#         "NASADefectDataset-master/CleanedData/MDP/MW1/MW1''.arff",
#         "NASADefectDataset-master/CleanedData/MDP/MW1''.arff",
#         "NASADefectDataset-master/CleanedData/MDP/mw1''.arff",
#     ],
# }

# # Valori attesi per validazione (Shepperd 2013, Table 2 — DS'')
# EXPECTED = {
#     "MC2": {"instances": 127, "features": 40},
#     "KC3": {"instances": 434, "features": 40},
#     "MW1": {"instances": 370, "features": 40},
# }


# # ---------------------------------------------------------------------------
# # Parsing ARFF → DataFrame
# # ---------------------------------------------------------------------------

# def arff_bytes_to_df(raw_bytes: bytes, dataset_name: str) -> pd.DataFrame:
#     """Converte bytes di un file ARFF in DataFrame pandas."""
#     try:
#         text = raw_bytes.decode("utf-8", errors="replace")
#     except Exception:
#         text = raw_bytes.decode("latin-1", errors="replace")

#     # scipy.io.arff vuole un file-like object
#     data, meta = arff.loadarff(io.StringIO(text))
#     df = pd.DataFrame(data)

#     # Decodifica byte strings
#     for col in df.select_dtypes(include=["object"]).columns:
#         df[col] = df[col].apply(
#             lambda x: x.decode("utf-8", errors="replace")
#             if isinstance(x, bytes) else x
#         )
#     print(f"\n  🔍 [{dataset_name}] Colonne disponibili: {df.columns.tolist()}")
#     print(f"  🔍 [{dataset_name}] Ultima colonna '{df.columns[-1]}': {df.iloc[:,-1].value_counts().to_dict()}")
#     print(f"  🔍 [{dataset_name}] Dtype ultima colonna: {df.iloc[:,-1].dtype}")

#     # Normalizza colonna target → 'buggy' (0/1)
#     target_candidates = [
#         "defects", "bug", "bugs", "Bugs", "BugCount", "buggy",
#         "class", "label", "Defects", "number_of_bugs",
#          "Defective", "defective" 
#     ]
#     target_col = None
#     for col in df.columns:
#         if col.lower() in [c.lower() for c in target_candidates]:
#             target_col = col
#             break
#     if target_col is None:
#         target_col = df.columns[-1]
#         print(f"  ⚠️  [{dataset_name}] Target non trovato, uso ultima colonna: '{target_col}'")

#     df = df.rename(columns={target_col: "buggy"})
#     if df["buggy"].dtype == object:
#         df["buggy"] = df["buggy"].apply(
#             lambda x: 1 if str(x).strip().lower() in
#                     ["1", "true", "yes", "y", "buggy", "true."] else 0
#         )
#     else:
#         df["buggy"] = (df["buggy"].astype(float) > 0).astype(int)

#     return df


# # ---------------------------------------------------------------------------
# # Validazione
# # ---------------------------------------------------------------------------

# def validate(df: pd.DataFrame, name: str) -> bool:
#     """Controlla che il dataset abbia le dimensioni attese per DS''."""
#     n = len(df)
#     p = len(df.columns) - 1  # escluso target
#     pct = round(df["buggy"].mean() * 100, 1)
#     exp = EXPECTED.get(name, {})

#     print(f"\n  📊 {name} — {n} istanze | {p} feature | {pct}% buggy")

#     ok = True
#     if exp:
#         if n != exp["instances"]:
#             print(f"  ⚠️  Istanze attese: {exp['instances']}, trovate: {n}")
#             print(f"       → Potrebbe essere DS' invece di DS''")
#             ok = False
#         else:
#             print(f"  ✅  Istanze corrispondono a DS'' ({exp['instances']})")

#         if p != exp["features"]:
#             print(f"  ⚠️  Feature attese: {exp['features']}, trovate: {p}")
#             ok = False
#         else:
#             print(f"  ✅  Feature corrispondono ({exp['features']})")

#     return ok


# # ---------------------------------------------------------------------------
# # Download da GitHub ZIP
# # ---------------------------------------------------------------------------

# def download_github_zip() -> dict[str, bytes] | None:
#     """
#     Scarica il repo klainfo/NASADefectDataset come ZIP e
#     restituisce un dict {dataset_name: arff_bytes}.
#     """
#     print(f"\n📦 Download GitHub ZIP: {GITHUB_ZIP_URL}")
#     print("   (può richiedere qualche secondo...)")

#     try:
#         r = requests.get(GITHUB_ZIP_URL, timeout=120)
#         r.raise_for_status()
#     except Exception as e:
#         print(f"  ❌ Download GitHub ZIP fallito: {e}")
#         return None

#     print(f"  ✅ ZIP scaricato ({len(r.content)/1024/1024:.1f} MB)")

#     results = {}
#     with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
#         # Mostra tutti i file disponibili in CleanedData/
#         cleaned_files = [
#             n for n in zf.namelist()
#             if "CleanedData" in n and n.endswith(".arff")
#         ]
#         print(f"\n  📁 File DS trovati nel ZIP ({len(cleaned_files)}):")
#         for f in cleaned_files:
#             print(f"     {f}")

#         for dataset, path_candidates in GITHUB_PATHS.items():
#             found = False
#             for path in path_candidates:
#                 if path in zf.namelist():
#                     results[dataset] = zf.read(path)
#                     print(f"\n  ✅ {dataset} → {path}")
#                     found = True
#                     break
#             if not found:
#                 # Cerca in modo fuzzy
#                 matches = [
#                     n for n in cleaned_files
#                     if dataset.lower() in n.lower() and "''" in n
#                 ]
#                 if matches:
#                     results[dataset] = zf.read(matches[0])
#                     print(f"\n  ✅ {dataset} → {matches[0]} (fuzzy match)")
#                 else:
#                     print(f"\n  ⚠️  {dataset} DS'' non trovato nel ZIP")

#     return results if results else None


# # ---------------------------------------------------------------------------
# # Fallback: Zenodo diretti
# # ---------------------------------------------------------------------------

# def download_zenodo_fallback(missing: list[str]) -> dict[str, bytes]:
#     """
#     Scarica i dataset mancanti da Zenodo.
#     Per ora supporta MW1 (unico con DS'' esplicito su Zenodo).
#     """
#     zenodo_urls = {
#         "MW1": ZENODO_MW1_DS_PP,
#         # KC3 e MC2 non hanno DS'' esplicito su Zenodo —
#         # in quel caso occorre scaricare manualmente dal repo klainfo
#     }

#     results = {}
#     for name in missing:
#         if name in zenodo_urls:
#             url = zenodo_urls[name]
#             print(f"\n  📥 Zenodo fallback per {name}: {url}")
#             try:
#                 r = requests.get(url, timeout=60)
#                 r.raise_for_status()
#                 results[name] = r.content
#                 print(f"  ✅ {name} scaricato da Zenodo ({len(r.content)/1024:.0f} KB)")
#             except Exception as e:
#                 print(f"  ❌ Zenodo fallback fallito per {name}: {e}")
#         else:
#             print(f"\n  ℹ️  {name} non disponibile via Zenodo con DS'' esplicito.")
#             print(f"      → Scarica manualmente da:")
#             print(f"        https://github.com/klainfo/NASADefectDataset")
#             print(f"        Percorso: CleanedData/MDP/{name}/{name}''.arff")

#     return results


# # ---------------------------------------------------------------------------
# # Main
# # ---------------------------------------------------------------------------

# def main():
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     summary_rows = []

#     print("=" * 60)
#     print("NASA DS'' Downloader — MC2, KC3, MW1")
#     print("Shepperd et al. (2013), IEEE TSE 39(8):1208-1215")
#     print("=" * 60)

#     # Step 1: prova GitHub ZIP
#     github_results = download_github_zip()
#     arff_bytes_map = github_results or {}

#     # Step 2: fallback Zenodo per quelli mancanti
#     missing = [
#         name for name in ["MC2", "KC3", "MW1"]
#         if name not in arff_bytes_map
#     ]
#     if missing:
#         print(f"\n⚠️  Dataset mancanti dopo GitHub: {missing}")
#         print("   Tentativo fallback Zenodo...")
#         zenodo_results = download_zenodo_fallback(missing)
#         arff_bytes_map.update(zenodo_results)

#     # Step 3: converti, valida, salva
#     print("\n" + "=" * 60)
#     print("Conversione ARFF → CSV")
#     print("=" * 60)

#     for name in ["MC2", "KC3", "MW1"]:
#         if name not in arff_bytes_map:
#             print(f"\n❌ {name}: non disponibile — download manuale richiesto")
#             print(f"   Istruzioni:")
#             print(f"   1. Vai su https://github.com/klainfo/NASADefectDataset")
#             print(f"   2. Scarica il file CleanedData/MDP/{name}/{name}''.arff")
#             print(f"   3. Esegui di nuovo lo script con il file in ./manual/{name}.arff")
#             continue

#         print(f"\n→ Elaborazione {name}...")
#         try:
#             df = arff_bytes_to_df(arff_bytes_map[name], name)
#             is_valid = validate(df, name)

#             out_path = os.path.join(OUTPUT_DIR, f"{name}_DS_pp.csv")
#             df.to_csv(out_path, index=False)
#             print(f"  💾 Salvato: {out_path}")

#             summary_rows.append({
#                 "dataset": f"{name}_DS''",
#                 "instances": len(df),
#                 "features": len(df.columns) - 1,
#                 "pct_buggy": round(df["buggy"].mean() * 100, 1),
#                 "pct_missing_any": round(
#                     df.drop(columns=["buggy"]).isnull().any(axis=1).mean() * 100, 1
#                 ),
#                 "ds_pp_validated": is_valid,
#                 "csv_path": out_path,
#                 "doi_source": {
#                     "MC2": "github.com/klainfo/NASADefectDataset",
#                     "KC3": "github.com/klainfo/NASADefectDataset",
#                     "MW1": "10.5281/zenodo.268490",
#                 }.get(name, ""),
#             })

#         except Exception as e:
#             print(f"  ❌ Errore durante elaborazione {name}: {e}")

#     # Step 4: riepilogo
#     if summary_rows:
#         summary_df = pd.DataFrame(summary_rows)
#         summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
#         summary_df.to_csv(summary_path, index=False)

#         print("\n" + "=" * 60)
#         print("RIEPILOGO DATASET")
#         print("=" * 60)
#         print(summary_df[
#             ["dataset", "instances", "features", "pct_buggy",
#              "pct_missing_any", "ds_pp_validated"]
#         ].to_string(index=False))
#         print(f"\n💾 Summary salvato: {summary_path}")

#     # Step 5: note metodologiche
#     print("\n" + "=" * 60)
#     print("NOTE METODOLOGICHE per il paper")
#     print("=" * 60)
#     print("""
# Versione usata: DS'' (double prime) di Shepperd et al. (2013)
#   - DS'  = rimosse istanze duplicate
#   - DS'' = rimosse duplicate + inconsistenti (rimozione più aggressiva)

# Da citare nel paper:
#   Shepperd, M., Song, Q., Sun, Z., & Mair, C. (2013).
#   Data Quality: Some Comments on the NASA Software Defect Datasets.
#   IEEE Transactions on Software Engineering, 39(8), 1208–1215.
#   https://doi.org/10.1109/TSE.2013.11

# Versioni NASA MDP (40 feature, McCabe + Halstead):
#   MC2: 127 istanze DS'' | 161 originali | ~35% buggy
#   KC3: 434 istanze DS'' | 458 originali |  ~9% buggy
#   MW1: 370 istanze DS'' | 403 originali |  ~8% buggy

# ATTENZIONE: La versione PROMISE degli stessi dataset ha meno feature
#   (KC3: 39, MW1: 37-38) e non è equivalente a quella NASA MDP.
#   Indicare sempre "versione NASA MDP DS''" nel paper.
# """)


# # ---------------------------------------------------------------------------
# # Fallback manuale da file locale
# # ---------------------------------------------------------------------------

# def load_manual_arff(path: str, dataset_name: str) -> pd.DataFrame | None:
#     """
#     Carica un file ARFF scaricato manualmente.
#     Usa questa funzione se il download automatico fallisce.

#     Esempio:
#         df = load_manual_arff("./KC3_DS_pp.arff", "KC3")
#         df.to_csv("./nasa_ds_pp/KC3_DS_pp.csv", index=False)
#     """
#     if not os.path.exists(path):
#         print(f"File non trovato: {path}")
#         return None
#     with open(path, "rb") as f:
#         raw = f.read()
#     df = arff_bytes_to_df(raw, dataset_name)
#     validate(df, dataset_name)
    
#     print(df.columns.tolist())        # vedi i nomi colonne
#     print(df.iloc[:, -1].value_counts())  # valori dell'ultima colonna
#     print(df.dtypes)
#     return df


# if __name__ == "__main__":
#     #main()
#     df_raw = pd.read_csv("./nasa_ds_pp/MW1_DS_pp.csv")

#     # Controlla il tipo reale
#     print(df_raw["buggy"].dtype)
#     print(df_raw["buggy"].unique())      # valori distinti presenti
#     print(df_raw["buggy"].value_counts(dropna=False))  # include NaN

# import pandas as pd
# import numpy as np
# from ucimlrepo import fetch_ucirepo
# from sklearn.datasets import fetch_california_housing
# import os

# os.makedirs('datasetRoot', exist_ok=True)

# # ============================================================
# # 1. California Housing (sklearn — più affidabile di UCI)
# # ============================================================
# print("Scaricando California Housing...")
# california = fetch_california_housing(as_frame=True)
# df_california = california.frame  # include già la colonna target 'MedHouseVal'
# df_california.to_csv('datasetRoot/california_housing.csv', index=False)
# print(f"  Salvato: {len(df_california)} righe, {len(df_california.columns)} colonne")
# print(f"  Colonne: {list(df_california.columns)}")

# # ============================================================
# # 2. Concrete Compressive Strength (UCI id=165)
# # ============================================================
# print("Scaricando Concrete Compressive Strength...")
# concrete = fetch_ucirepo(id=165)
# df_concrete = pd.concat([concrete.data.features, concrete.data.targets], axis=1)
# # rinomina target per chiarezza
# df_concrete.columns = [c.strip() for c in df_concrete.columns]
# df_concrete.to_csv('datasetRoot/concrete_strength.csv', index=False)
# print(f"  Salvato: {len(df_concrete)} righe, {len(df_concrete.columns)} colonne")
# print(f"  Colonne: {list(df_concrete.columns)}")

# # ============================================================
# # 3. Student Performance (UCI id=320)
# # ============================================================
# print("Scaricando Student Performance...")
# student = fetch_ucirepo(id=320)
# df_student = pd.concat([student.data.features, student.data.targets], axis=1)
# df_student.columns = [c.strip() for c in df_student.columns]
# # il target è G3 (voto finale), droppiamo G1 e G2 che sono voti intermedi
# # e creerebbero data leakage
# if 'G1' in df_student.columns:
#     df_student = df_student.drop(columns=['G1', 'G2'])
# df_student.to_csv('datasetRoot/student_performance.csv', index=False)
# print(f"  Salvato: {len(df_student)} righe, {len(df_student.columns)} colonne")
# print(f"  Colonne: {list(df_student.columns)}")

# # ============================================================
# # 4. Bike Sharing (UCI id=275)
# # ============================================================
# print("Scaricando Bike Sharing...")
# bike = fetch_ucirepo(id=275)
# df_bike = pd.concat([bike.data.features, bike.data.targets], axis=1)
# df_bike.columns = [c.strip() for c in df_bike.columns]
# # droppiamo 'casual' e 'registered' che sommati danno 'cnt' (target) — data leakage
# if 'casual' in df_bike.columns:
#     df_bike = df_bike.drop(columns=['casual', 'registered'])
# # droppiamo anche 'instant' (indice) e 'dteday' (data stringa)
# for col in ['instant', 'dteday']:
#     if col in df_bike.columns:
#         df_bike = df_bike.drop(columns=[col])
# df_bike.to_csv('datasetRoot/bike_sharing.csv', index=False)
# print(f"  Salvato: {len(df_bike)} righe, {len(df_bike.columns)} colonne")
# print(f"  Colonne: {list(df_bike.columns)}")

# # ============================================================
# # 5. Superconductivity (UCI id=464)
# # ============================================================
# print("Scaricando Superconductivity...")
# supercon = fetch_ucirepo(id=464)
# df_supercon = pd.concat([supercon.data.features, supercon.data.targets], axis=1)
# df_supercon.columns = [c.strip() for c in df_supercon.columns]
# df_supercon.to_csv('datasetRoot/superconductivity.csv', index=False)
# print(f"  Salvato: {len(df_supercon)} righe, {len(df_supercon.columns)} colonne")
# print(f"  Colonne: {list(df_supercon.columns)}")

# # ============================================================
# # Riepilogo finale
# # ============================================================
# print("\n=== RIEPILOGO ===")
# datasets = {
#     'california_housing.csv':  'MedHouseVal',
#     'concrete_strength.csv':   'Concrete compressive strength',
#     'student_performance.csv': 'G3',
#     'bike_sharing.csv':        'cnt',
#     'superconductivity.csv':   'critical_temp',
# }
# for fname, target in datasets.items():
#     path = f'datasetRoot/{fname}'
#     if os.path.exists(path):
#         df = pd.read_csv(path)
#         print(f"  {fname}")
#         print(f"    righe: {len(df)}, colonne: {len(df.columns)}, target: '{target}'")
#         print(f"    target range: [{df[target].min():.2f}, {df[target].max():.2f}]")
#         print(f"    missing: {df.isnull().sum().sum()}")

import pandas as pd
import os

base_dir = "./experiments"

for x in range(20):
    filename = f"checkpoint_bike_sharing_sampled.csv_run{x}.csv"
    filepath = os.path.join(base_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"[SKIP] File non trovato: {filename}")
        continue
    
    df = pd.read_csv(filepath)
    original_rows = len(df)
    
    # Rimuove righe che contengono "noise-shift" in qualsiasi colonna
    mask = ~df.apply(lambda row: row.astype(str).str.contains("noise-shift").any(), axis=1)
    df_filtered = df[mask]
    
    removed = original_rows - len(df_filtered)
    df_filtered.to_csv(filepath, index=False)
    
    print(f"[OK] {filename}: {removed} righe rimosse, {len(df_filtered)} rimaste")

print("\nDone.")