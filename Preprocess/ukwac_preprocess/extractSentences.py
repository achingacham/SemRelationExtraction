
# coding: utf-8

# In[2]:


import numpy as np
import re


# In[4]:


'''
#Other method

with open("column1_ukwac","r") as inputFile:
    inputWords = inputFile.readlines()
    print(inputWords)
    #print([re.sub('\n$',' ',words) for words in inputWords])
    cleanWords = "".join([re.sub('\n$',' ',words) for words in inputWords[2:]])
    cleanWords = re.sub(" <s> ","\n",cleanWords)
    with open("output_ukwac","w") as outputFile:
        outputFile.write(re.sub(" </s>","",cleanWords))
'''


# In[17]:


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

