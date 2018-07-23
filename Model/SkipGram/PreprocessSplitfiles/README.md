# Preprocess of ukWaC data for Model

* create listFiles.txt from the location where splitFiles (smaller chunks) are stored.
* Set up the 'folder' name to store the output files for each spiltFile
* Attend the values set in preprocess.py before executing (like threshold window size, eg:'d')
* If GPU available, execute GridCommand.sh, else BashCommand.sh

## next step:

* Based on window size filter triples after combining all small chunks (Execute createDatasets.py)

