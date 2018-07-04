
# coding: utf-8

# In[3]:


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


# In[6]:


no='1'

with open("ukwac_words_"+no,"r") as inputFile:
    inputWords = inputFile.readlines()
    sentence_dict = dict()
    for words in inputWords[1:]:
        if re.match('<s>',words):
            
            k = len(sentence_dict)
            print("New sentence : ",k+1)
            sentence_dict[k+1] = ""
        else:
            cleaned = re.sub('\n'," ",words)
            #sentence_dict[k+1] += cleaned
            sentence_dict[k+1] += re.sub('</s> ','\n',cleaned)
     
    with open("output_ukwac_"+no,"w") as outputFile_2:
        [outputFile_2.write("\n"+sentence.strip()) for sentence in sentence_dict.values()]

