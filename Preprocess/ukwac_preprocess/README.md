
# Steps for preprocessing:


*  gzip  -k -d ukwac_dep_parsed_.gz
*  mv ukwac_dep_parsed_ ukwac_file
*  cut --delimiter=$'\t' -f1,3 ukwac_file > ukwac_words_.txt
*  python3 extractSentences_V2.py
*  Split the big file into smaller chunks
	* split <bigFile> splitFilePrefix_ --number=l/1000 -e 
*  If decompressed file is created replacing .gz fil then execute 
	* gzip -9 ukwac_file_
*  else
	* rm ukwac_file
* Remove </text> ..... lines ans blank lines
   	* sed 's/<\/text>//g' ukwac_file_
	* sed '/^ *$/d' ukwac_file_	
	
