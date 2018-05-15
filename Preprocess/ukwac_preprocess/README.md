
Steps for preprocessing:


0) gzip  -k -d ukwac_dep_parsed_.gz
1) mv ukwac_dep_parsed_ ukwac_file
2) cut --delimiter=$'\t' -f1,3 ukwac_file > ukwac_words_.txt
3) python3 extractSentences_V2.py
4) Split the big file into smaller chunks
	#split <bigFile> splitFilePrefix_ --number=l/1000 -e 
5) If decompressed file is created replacing .gz fil then execute 
	 gzip -9 ukwac_file_
   else
	rm ukwac_file
6) Remove </text> ..... lines ans blank lines
   	sed 's/<\/text>//g' ukwac_file_
	sed '/^ *$/d' ukwac_file_	
	
