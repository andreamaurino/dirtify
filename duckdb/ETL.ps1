
# Connessione a DuckDB 
$duckdbCLI = "C:\Users\douni\OneDrive\Desktop\epc\duckdb\duckdb.exe"
$databaseFile = "epc"

# Percorso del file SQL
$sqlFile = "C:\Users\douni\OneDrive\Desktop\epc\duckdb\create_database_epc.txt"

# Verifica se il file esiste
if (Test-Path $sqlFile) {# Leggi i comandi SQL dal file
    $sqlCommands = Get-Content $sqlFile -Raw  # Leggi l'intero file come stringa

    # Esegui i comandi SQL
    & $duckdbCLI $databaseFile -c $sqlCommands
    Write-Host "Executed SQL commands from $sqlFile"
} else {
    Write-Host "SQL file not found: $sqlFile"
}
###################################
#
#creare le righe di modperf
#
#################
$insertQuery = "truncate MODPERF;"
& $duckdbCLI $databaseFile -c $insertQuery
Write-Host "create dfm table rows"
$insertQuery ="CREATE OR REPLACE TABLE sql_commands AS 
SELECT 
    'INSERT INTO MODPERF (DATASETNAME, ACC, ERRTYPE, ERRPERC, FEATNAME, MODELNAME, AUC, REC, PREC, F1) ' ||
    'SELECT  datasetName, Accuracy, errorType, percentage, feature, modelName, Auc, Recall, Precision, F1 ' ||
    'FROM ' || datasetName || '  ;' AS command
     FROM EXPDATASET ;
"
& $duckdbCLI $databaseFile -c $insertQuery
# Esporta i comandi in un file CSV
$outputCsv = "C:\Users\douni\OneDrive\Desktop\epc\duckdb\commands.csv"
$exportQuery = "COPY (SELECT command FROM sql_commands) TO '$outputCsv' WITH (HEADER, DELIMITER ',');"
& $duckdbCLI $databaseFile -c $exportQuery
# Leggi i comandi dal file CSV
$commands = Import-Csv $outputCsv

# Esegui ogni comando SQL
foreach ($row in $commands) {
	# Ottieni il valore massimo dell'ID da MODPERF
    $command = $row.command.Trim()  # Rimuovi spazi extra
	
	& $duckdbCLI $databaseFile -c $command
    Write-Host "Executed"
}
#setto a zero gli auc mancanti
$insertQuery ="update modperf set auc=0 where auc is null ;"
& $duckdbCLI $databaseFile -c $insertQuery
#########################################
#
#aggiungo l'epc
#
#########################################
Write-Host "****************** Update epc"

$insertQuery ="CREATE OR REPLACE TABLE sql_commands AS 
SELECT 
    'update MODPERF as M  set EPC= e.epc_at  from ' || datasetName ||
    ' as e where  M.ERRTYPE=e.errorType and M.MODELNAME=e.modelName and M.FEATNAME=e.feature and M.ERRPERC=e.percentage;' AS command
     FROM EPCDATASET ;
"
& $duckdbCLI $databaseFile -c $insertQuery
# Esporta i comandi in un file CSV
$outputCsv = "C:\Users\douni\OneDrive\Desktop\epc\duckdb\commands1.csv"
$exportQuery = "COPY (SELECT command FROM sql_commands) TO '$outputCsv' WITH (HEADER, DELIMITER ',');"
& $duckdbCLI $databaseFile -c $exportQuery
Write-Host "Exported commands to $outputCsv"
# Leggi i comandi dal file CSV
$commands = Import-Csv $outputCsv

# Esegui ogni comando SQL
foreach ($row in $commands) {
    $command = $row.command.Trim()  # Rimuovi spazi extra	
	& $duckdbCLI $databaseFile -c $command
    Write-Host "Executed"
}
Write-Host "***************** update modelname"
#####################################
#
#aggiungo i modeltype
#
#####################################
$insertQuery ="update MODPERF set MODELNAME=M.MODELNAME, MODELTYPE=M.MODELTYPE from MODELS M where MODPERF.MODELNAME=M.MODELLOCALNAME  ;
"
& $duckdbCLI $databaseFile -c $insertQuery


######################################
#
#assegno il tipo di features
#
######################################
$insertQuery ="truncate features;truncate jsondataset"
& $duckdbCLI $databaseFile -c $insertQuery

# carico i file json della directory di configurazione in enne tabelle
#carico i file degli esperimenti
$csvDirectory = "C:\Users\douni\OneDrive\Desktop\epc\json"

$insertQuery = "CREATE OR REPLACE TABLE JSONDATASET (DATASETNAME VARCHAR(255) primary key);"
& $duckdbCLI $databaseFile -c $insertQuery

# Itera sui file json
Write-Host "***************** update feature type"

Get-ChildItem -Path $csvDirectory -Filter *.json| ForEach-Object {
    
		$filePath = $_.FullName
		$fileName = $_.BaseName.Replace("-", "_")+"json"  # Sostituisci "-" con "_"
		$query = "CREATE OR REPLACE TABLE $fileName AS SELECT *  FROM read_json_auto('$filePath');"
		& $duckdbCLI $databaseFile -c $query
		
		$insertQuery = "INSERT INTO JSONDATASET VALUES ('$fileName');"
		& $duckdbCLI $databaseFile -c $insertQuery
		Write-Host "$fileName"
		# per ogni tabella faccio il join di modperf con la tabella appena creata così

		$query="insert into features 
			 SELECT datasetName, UNNEST(discreteFeatures) AS feature, 'discrete' FROM $fileName;"			 
		& $duckdbCLI $databaseFile -c $query
			 
		$query="insert into features 
			select datasetname, unnest(continousFeatures) as feature, 'continue'  from $fileName;"			 
		& $duckdbCLI $databaseFile -c $query
		
		$query="insert into features 
			select datasetname, unnest(categoricalFeaturesString) as feature, 'categorical String' from $fileName;"			 
		& $duckdbCLI $databaseFile -c $query
			 
		$query="insert into features 
			select datasetname, unnest(categoricalFeaturesInteger) as feature, 'categorical Integer' from $fileName;"			 
		& $duckdbCLI $databaseFile -c $query
			 
		$query="insert into features 
			select datasetname, unnest(binaryFeatures) as feature, 'binary'  from $fileName;"			 
		& $duckdbCLI $databaseFile -c $query
			 
		$query="insert into features 
			select datasetname, targetVariable as feature, 'target' as feattype from $fileName;"
		& $duckdbCLI $databaseFile -c $query

}
$query="update MODPERF as M set feattype=f.type from features as f where M.datasetName=f.datasetName and M.featname=f.feature;
			"			 
		& $duckdbCLI $databaseFile -c $query
		
############################################################
#
#per ogni dataset,modello,tipo di errore, feature assegno i valori iniziali di performance
#
############################################################
# Esporta i comandi in un file CSV
$outputCsv = "C:\Users\douni\OneDrive\Desktop\epc\duckdb\data_to_add_commands.csv"
$exportQuery = "COPY (select distinct m1.datasetname, m1.modelname, m1.acc,m1.prec,m1.rec,m1.f1,m1.auc, m2.featname,m2.feattype,m2.errtype,
m2.modeltype from modperf m1 join modperf m2 on (m1.datasetname=m2.datasetname and m1.modelname=m2.modelname) where m1.errperc=0.0 )
TO '$outputCsv' WITH (HEADER, DELIMITER ',');"
& $duckdbCLI $databaseFile -c $exportQuery
# Leggi i comandi dal file CSV
$commands = Import-Csv $outputCsv

# Importa il contenuto del file CSV
$data = Import-Csv $outputCsv

# Itera su ogni riga del CSV
# Itera su ogni riga del CSV
$outputtable = "C:\Users\douni\OneDrive\Desktop\epc\duckdb\tabella.csv"
# Dati da salvare (inizialmente vuoto)
$datacsv = @()

foreach ($row in $data) {
    # Assegna i valori alle variabili
    $datasetname = $row.datasetname
    $modelname = $row.modelname
	$acc = $row.acc
    $prec = $row.prec
    $rec = $row.rec
    $f1 = $row.f1
	$auc=$row.auc
	$featname = $row.featname
	$feattype = $row.feattype
	$errtype=$row.errtype
	$modeltype=$row.modeltype
    $datacsv += [PSCustomObject]@{
			acc         = $acc
			prec        = $prec
			rec         = $rec
			epc_at		= $null
			f1          = $f1
			auc			= $auc
			datasetname = $datasetname
			errtype		=$errtype
			errperc		= 0.0
			feattype	=$feattype
			featname	=$featname
			modelname   = $modelname
			modeltype   = $modeltype
				
			}			
}
		# Esporta i dati in CSV
		$datacsv | Export-Csv -Path $outputtable -NoTypeInformation -Force 		
		# Rimuovi le virgolette dal CSV
		(Get-Content $outputtable) -replace '"', '' | Set-Content -Path $outputtable

$insertQuery ="insert into MODPERF select * from read_csv_auto('$outputtable');"
		& $duckdbCLI $databaseFile -c $insertQuery
		
