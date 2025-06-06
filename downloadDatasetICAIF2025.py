from ucimlrepo import fetch_ucirepo 
import pandas as pd 
import yfinance as yf
from datetime import datetime

# fetch dataset 
#statlog_german_credit_data = fetch_ucirepo(id=144) 
#X = statlog_german_credit_data.data.features 
#y = statlog_german_credit_data.data.targets 

#df=X
#df2=y
#df['Target'] = df2
#print(df)

#df.to_csv("./datasetRoot/german_credit_data.csv")

#statlog_australian_credit_approval = fetch_ucirepo(id=143) 

#X = statlog_australian_credit_approval.data.features 
#y = statlog_australian_credit_approval.data.targets 

#df=X
#df2=y
#df['Target'] = df2
#print(df)
#df.to_csv("./datasetRoot/australian_credit_data.csv")

#######
# Yhaoo finance
#######

import pandas as pd
import yfinance as yf

import pandas as pd
import yfinance as yf

# Step 1: Ottieni i ticker
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
tickers_df = pd.read_html(url)[0]
tickers = tickers_df['Symbol'].tolist()
tickers = [t.replace('.', '-') for t in tickers]

# Step 2: Imposta le date
start_date = "2025-05-01"
end_date = "2025-05-31"

# Step 3: Scarica i dati
#print("Downloading data...")
#all_data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
#print("Download completed.")

# Step 4: Riorganizza i dati in formato long
#long_data = []
#for ticker in tickers:
#    try:
#        df = all_data[ticker].copy()
#        if df.empty:
#            print(f"Nessun dato per {ticker}")
#            continue
#        df['Date'] = df.index
#        df['Ticker'] = ticker
#        long_data.append(df)
#    except Exception as e:
#        print(f"Errore per {ticker}: {e}")

# Step 5: Verifica e concatena
#if not long_data:
#    raise ValueError("Nessun dato valido scaricato per alcun ticker.")

#result_df = pd.concat(long_data)
#result_df.reset_index(drop=True, inplace=True)
#result_df = result_df[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close',  'Volume']]

# Step 6: Salva su CSV
#result_df.to_csv("./datasetRoot/sp500_may2025.csv", index=False)
 #print("Salvato CSV con", len(result_df), "righe.")
result_df=pd.read_csv("./datasetRoot/sp500_may2025.csv")

# Funzione per calcolare il target  

def compute_target(df):
    df = df.sort_values('Date')
    df['Next_Open'] = df['Open'].shift(-1)
    df['target'] = (df['Close'] > df['Next_Open']).astype(int)
    df.iloc[-1, df.columns.get_loc('target')] = 0  # ultima riga del ticker non ha giorno successivo
    return df.drop(columns=['Next_Open'])

# Applica la funzione a ciascun ticker
result_df = result_df.groupby('Ticker', group_keys=False).apply(compute_target)

# Ordina e salva
result_df = result_df.sort_values(['Ticker', 'Date'])
result_df.to_csv("./datasetRoot/sp500_may2025_with_target.csv", index=False)

print(result_df.head(10))