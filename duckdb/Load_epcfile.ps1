# Directory dei CSV
$csvDirectory = "C:\Users\douni\OneDrive\Desktop\epc\epcResults"

# Connessione a DuckDB 
$duckdbCLI = "C:\Users\douni\OneDrive\Desktop\epc\duckdb\duckdb.exe"
$databaseFile = "epc"
#cancello tutte le tabelle
    $insertQuery = "SELECT 'DROP TABLE ' || table_name || ';' AS drop_command FROM information_schema.tables WHERE table_schema = 'main';"
    & $duckdbCLI $databaseFile -c $insertQuery

#carico i file epc
    $insertQuery = "CREATE OR REPLACE TABLE EPCDATASET (DATASETNAME VARCHAR(255) primary key);"
    & $duckdbCLI $databaseFile -c $insertQuery

# Itera sui file CSV
Get-ChildItem -Path $csvDirectory -Filter *.csv | ForEach-Object {
		$filePath = $_.FullName
		$fileName = $_.BaseName.Replace("-", "_")  # Sostituisci "-" con "_"
		$query = "CREATE OR REPLACE TABLE $fileName AS SELECT * FROM read_csv_auto('$filePath');"
		& $duckdbCLI $databaseFile -c $query
		$insertQuery = "INSERT INTO EPCDATASET VALUES ('$fileName');"
		& $duckdbCLI $databaseFile -c $insertQuery
		Write-Host "Created table: $fileName"
	
}

#carico i file degli esperimenti
$csvDirectory = "C:\Users\douni\OneDrive\Desktop\epc\experiments"

    $insertQuery = "CREATE OR REPLACE TABLE EXPDATASET (DATASETNAME VARCHAR(255) primary key);"
    & $duckdbCLI $databaseFile -c $insertQuery

# Itera sui file CSV
Get-ChildItem -Path $csvDirectory -Filter *.csv | ForEach-Object {
    if ($_.BaseName -ne "correlation_results.csv") {
	
		$filePath = $_.FullName
		$fileName = $_.BaseName.Replace("-", "_")  # Sostituisci "-" con "_"
		$query = "CREATE OR REPLACE TABLE $fileName AS SELECT * FROM read_csv_auto('$filePath');"
		& $duckdbCLI $databaseFile -c $query
		$insertQuery = "INSERT INTO EXPDATASET VALUES ('$fileName');"
		& $duckdbCLI $databaseFile -c $insertQuery
		Write-Host "Created table: $fileName"
	}
}

