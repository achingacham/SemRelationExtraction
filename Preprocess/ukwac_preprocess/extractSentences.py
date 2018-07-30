
import numpy as np
import re


'''
Inputs :

        no : refernce number of input file
        
Output :
        
        Outfile file with sentences per line. 
        POS tags of each word follows the word in every sentence.
        
'''

no='1'
outputFile_2 = open("output_ukwac_"+no,"w")
    
with open("ukwac_words_"+no+".txt","r",encoding="ISO-8859-1") as inputFile:
    inputWords = inputFile.readlines()
   
    k = 0
    sentence  = ""
    for words in inputWords[1:]:
        if re.match('<s>',words):
        
            print("New sentence : ",k)
            
            #outputFile_2.write(sentence.strip()+"\n")
            sentence = ""
            k += 1
        else:
            cleaned = re.sub('\n'," ",words)
            sentence += re.sub('</s>','\n',cleaned)
            
            if re.match('</s>',words):
                outputFile_2.write(sentence.strip()+"\n")

        
inputFile.close()
outputFile_2.close()

